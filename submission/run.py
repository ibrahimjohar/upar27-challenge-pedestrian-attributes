"""UPAR 2027 Track 1 submission.

model.pt is a TorchScript module traced with batch size 64. It takes uint8 images
of shape (B, 256, 128, 3) and returns 40 probabilities per image; normalisation,
flip averaging and the per-attribute offsets are part of the module.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 64
H, W = 256, 128

_MODEL = None
_DEVICE = None
_ATTRS: list[str] = []
_POOL = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))


def load_model() -> None:
    global _MODEL, _DEVICE, _ATTRS
    with open(os.path.join(HERE, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    _ATTRS = meta["attributes"]
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(os.path.join(HERE, "model.pt"), map_location=_DEVICE)
    #weights are saved in fp16
    model = model.half() if _DEVICE.type == "cuda" else model.float()
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(BATCH_SIZE, H, W, 3, dtype=torch.uint8, device=_DEVICE))
    _MODEL = model


def _load_image(path: str) -> np.ndarray:
    #same preprocessing as load_rgb_resized in train.py
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.uint8)


def _infer(arrays: list[np.ndarray]) -> np.ndarray:
    n = len(arrays)
    x = np.stack(arrays)
    if n < BATCH_SIZE:
        x = np.concatenate([x, np.repeat(x[-1:], BATCH_SIZE - n, axis=0)])
    with torch.inference_mode():
        p = _MODEL(torch.from_numpy(x).to(_DEVICE, non_blocking=True)).float().cpu().numpy()
    return p[:n]


def predict_batch(samples: list[dict[str, Any]]) -> list[list[float]]:
    if _MODEL is None:
        load_model()
    arrays = list(_POOL.map(_load_image, [s["image_path"] for s in samples]))
    probs = np.concatenate([_infer(arrays[i:i + BATCH_SIZE]) for i in range(0, len(arrays), BATCH_SIZE)])
    names = samples[0].get("attribute_names") or _ATTRS
    if list(names) != _ATTRS:
        col = {a: j for j, a in enumerate(_ATTRS)}
        probs = probs[:, [col[a] for a in names]]
    probs = np.clip(np.nan_to_num(probs, nan=0.0), 0.0, 1.0)
    return probs.astype(float).tolist()


def predict_image(sample: dict[str, Any]) -> list[float]:
    return predict_batch([sample])[0]
