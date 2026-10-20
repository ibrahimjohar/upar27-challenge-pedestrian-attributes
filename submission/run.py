"""UPAR 2027 Track 1 submission.

model.pt is a TorchScript module traced with batch size 64. It takes uint8 images
of shape (B, 256, 128, 3) and returns 40 probabilities per image; normalisation,
flip averaging and the per-attribute offsets are part of the module.
"""
from __future__ import annotations

import json
import os
import sys
import time
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
_SEEN = 0
_T0 = None


def _log(msg: str) -> None:
    print(f"[run.py] {msg}", flush=True)
    sys.stdout.flush()


def load_model() -> None:
    global _MODEL, _DEVICE, _ATTRS
    t0 = time.time()
    with open(os.path.join(HERE, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    _ATTRS = meta["attributes"]
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"torch {torch.__version__}, device {_DEVICE}")
    model = torch.jit.load(os.path.join(HERE, "model.pt"), map_location=_DEVICE)
    #weights are saved in fp16
    model = model.half() if _DEVICE.type == "cuda" else model.float()
    if meta.get("int8"):
        _fill_int8(model, meta["int8"])
    model.eval()
    _MODEL = model
    _log(f"model loaded in {time.time() - t0:.1f}s")


def _fill_int8(model, names: list[str]) -> None:
    #the large weight matrices are stored as int8 with one scale per row
    data = np.load(os.path.join(HERE, "weights_int8.npz"))
    sd = model.state_dict()
    with torch.no_grad():
        for i, name in enumerate(names):
            w = torch.from_numpy(data[f"q{i}"]).float() * torch.from_numpy(data[f"s{i}"])[:, None]
            sd[name].copy_(w.reshape(sd[name].shape))
    _log(f"filled {len(names)} int8 weight matrices")


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
        p = _MODEL(torch.from_numpy(x).to(_DEVICE)).float().cpu().numpy()
    return p[:n]


def predict_batch(samples: list[dict[str, Any]]) -> list[list[float]]:
    global _SEEN, _T0
    if _MODEL is None:
        load_model()
    if _T0 is None:
        _T0 = time.time()
    arrays = [_load_image(s["image_path"]) for s in samples]
    probs = np.concatenate([_infer(arrays[i:i + BATCH_SIZE]) for i in range(0, len(arrays), BATCH_SIZE)])
    names = samples[0].get("attribute_names") or _ATTRS
    if list(names) != _ATTRS:
        col = {a: j for j, a in enumerate(_ATTRS)}
        probs = probs[:, [col[a] for a in names]]
    probs = np.clip(np.nan_to_num(probs, nan=0.0), 0.0, 1.0)

    before = _SEEN
    _SEEN += len(samples)
    if _SEEN // 2000 > before // 2000 or before == 0:
        _log(f"{_SEEN} images, {_SEEN / max(time.time() - _T0, 1e-6):.0f} images/s")
    return probs.astype(float).tolist()


def predict_image(sample: dict[str, Any]) -> list[float]:
    return predict_batch([sample])[0]
