"""UPAR 2027 Track 1 code submission.

The ingestion program imports this file and calls load_model() once, then
predict_batch(samples) (preferred over predict_image when defined).

Ships: model.pt  - TorchScript, takes uint8 [64, 256, 128, 3] RGB, returns
                   40 probabilities per image (normalisation, flip-TTA and the
                   tuned per-attribute offsets are inside the graph)
       meta.json - attribute order and settings
Only torch, numpy and Pillow are used (all present in the evaluation image).
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
BATCH_SIZE = 64          # the TorchScript graph was traced at exactly this batch size
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
    # weights are stored in fp16 to keep the zip small; fp16 on GPU, fp32 on CPU
    model = model.half() if _DEVICE.type == "cuda" else model.float()
    model.eval()
    with torch.inference_mode():  # warm-up
        model(torch.zeros(BATCH_SIZE, H, W, 3, dtype=torch.uint8, device=_DEVICE))
    _MODEL = model


def _load_image(path: str) -> np.ndarray:
    # identical to train.load_rgb_resized
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.uint8)


def _infer(arrays: list[np.ndarray]) -> np.ndarray:
    n = len(arrays)
    x = np.stack(arrays)
    if n < BATCH_SIZE:  # pad the last batch to the traced size
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
    if list(names) != _ATTRS:  # reorder by name if the platform ever uses another order
        col = {a: j for j, a in enumerate(_ATTRS)}
        probs = probs[:, [col[a] for a in names]]
    probs = np.clip(np.nan_to_num(probs, nan=0.0), 0.0, 1.0)
    return probs.astype(float).tolist()


def predict_image(sample: dict[str, Any]) -> list[float]:
    return predict_batch([sample])[0]
