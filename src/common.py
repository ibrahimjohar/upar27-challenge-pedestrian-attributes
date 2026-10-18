"""Attribute list, UPAR evaluation metrics and threshold calibration.

The metric follows get_pedestrian_metrics from the UPAR 2024 challenge code, which
the 2027 evaluation page says is reused. Instance F1 is computed from the mean
instance precision and recall, Challenge_Avg is the harmonic mean of mA and
Instance F1, and each dataset is scored separately before averaging.

With exclude_absent=True an attribute that has no positives in a dataset is
left out for that dataset (e.g. Market1501 has no pink upper-body labels).
"""
from __future__ import annotations

import csv
from typing import Iterable

import numpy as np

ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun", "Accessory-Hat",
]
NUM_ATTR = len(ATTRIBUTES)
EPS = 1e-20
SCALAR_KEYS = ["Challenge_Avg", "mA", "Instance_F1", "Label_F1",
               "Instance_Acc", "Instance_Prec", "Instance_Rec"]


def read_gt(csv_path: str) -> tuple[list[str], np.ndarray]:
    """Returns the image paths and an (N, 40) uint8 label array."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip().lstrip("#").strip() for h in rows[0]]
    if header[1:] != ATTRIBUTES:
        raise ValueError(f"unexpected header in {csv_path}")
    paths = [r[0] for r in rows[1:]]
    labels = np.asarray([[int(v) for v in r[1:]] for r in rows[1:]], dtype=np.int8)
    if labels.shape[1] != NUM_ATTR or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"labels in {csv_path} are not all 0/1")
    return paths, labels.astype(np.uint8)


def dataset_of(path: str) -> str:
    return path.replace("\\", "/").split("/")[0]


def pedestrian_metrics(gt: np.ndarray, pred: np.ndarray, index=None) -> dict:
    """pred is a boolean (N, 40) array; index restricts the attributes that are scored."""
    gt = np.asarray(gt)
    pred = np.asarray(pred).astype(bool)
    if index is not None:
        gt, pred = gt[:, index], pred[:, index]
    g1, g0 = gt == 1, gt == 0

    #label based
    gt_pos = g1.sum(0).astype(float)
    gt_neg = g0.sum(0).astype(float)
    tp = (g1 & pred).sum(0).astype(float)
    tn = (g0 & ~pred).sum(0).astype(float)
    fp = (g0 & pred).sum(0).astype(float)
    pos_rec = tp / (gt_pos + EPS)
    neg_rec = tn / (gt_neg + EPS)
    label_ma = (pos_rec + neg_rec) / 2
    label_prec = tp / (tp + fp + EPS)
    label_f1 = 2 * label_prec * pos_rec / (label_prec + pos_rec + EPS)

    #instance based
    i_gt = g1.sum(1).astype(float)
    i_pred = pred.sum(1).astype(float)
    inter = (g1 & pred).sum(1).astype(float)
    union = (g1 | pred).sum(1).astype(float)
    i_acc = np.mean(inter / (union + EPS))
    i_prec = np.mean(inter / (i_pred + EPS))
    i_rec = np.mean(inter / (i_gt + EPS))
    i_f1 = 2 * i_prec * i_rec / (i_prec + i_rec + EPS)

    ma = float(np.mean(label_ma))
    return {
        "mA": ma,
        "Label_F1": float(np.mean(label_f1)),
        "Instance_Acc": float(i_acc),
        "Instance_Prec": float(i_prec),
        "Instance_Rec": float(i_rec),
        "Instance_F1": float(i_f1),
        "Challenge_Avg": float(2 * ma * i_f1 / (ma + i_f1 + EPS)),
        "label_ma": label_ma,
        "label_f1": label_f1,
        "pos_recall": pos_rec,
        "neg_recall": neg_rec,
    }


def valid_index(gt: np.ndarray) -> np.ndarray:
    return np.where(gt.sum(0) > 0)[0]


def leaderboard_metrics(gt: np.ndarray, pred: np.ndarray, ds: Iterable[str],
                        exclude_absent: bool = True) -> dict:
    """Scores each dataset separately and averages the results."""
    ds = np.asarray(list(ds))
    per = {}
    for name in sorted(set(ds)):
        m = ds == name
        idx = valid_index(gt[m]) if exclude_absent else None
        per[name] = pedestrian_metrics(gt[m], pred[m], idx)
        per[name]["n"] = int(m.sum())
    out = {k: float(np.mean([per[d][k] for d in per])) for k in SCALAR_KEYS}
    out["Challenge_Avg_std"] = float(np.std([per[d]["Challenge_Avg"] for d in per]))
    out["per_dataset"] = {d: {k: per[d][k] for k in SCALAR_KEYS + ["n"]} for d in per}
    return out


def calibrate_offsets(logits: np.ndarray, gt: np.ndarray, ds, grid=None,
                      passes: int = 2, exclude_absent: bool = True) -> tuple[np.ndarray, list]:
    """Coordinate ascent on per-attribute logit offsets to maximise Challenge_Avg.

    The evaluation thresholds probabilities at 0.5, so adding an offset to an
    attribute's logit is equivalent to moving its decision threshold.
    """
    if grid is None:
        grid = np.round(np.arange(-1.5, 1.5001, 0.125), 4)
    ds = np.asarray(list(ds))
    off = np.zeros(logits.shape[1])

    def score(o):
        return leaderboard_metrics(gt, (logits + o) > 0, ds, exclude_absent)["Challenge_Avg"]

    best = score(off)
    log = [("start", best)]
    for p in range(passes):
        for j in range(logits.shape[1]):
            best_val = off[j]
            for g in grid:
                o = off.copy()
                o[j] = g
                s = score(o)
                if s > best + 1e-7:
                    best, best_val = s, g
            off[j] = best_val
        log.append((f"pass{p + 1}", best))
    return off, log
