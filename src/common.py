"""Shared pieces: attribute order, the official UPAR metric, and threshold calibration.

numpy only (no torch), so it can be imported anywhere.

Metric = the WACV 2024 UPAR evaluation (get_pedestrian_metrics), which the 2027
Codabench Evaluation page says it reproduces exactly:
  * threshold 0.5 on probabilities
  * label-based mA with eps = 1e-20
  * instance precision / recall averaged over images, then
    Instance_F1 = 2PR/(P+R) of those AVERAGES (not the mean of per-image F1)
  * Challenge_Avg = harmonic mean of mA and Instance_F1
  * every dataset scored on its own, leaderboard value = mean over datasets
"Attributes whose ground truth is marked invalid in a dataset are excluded."
The public labels contain no invalid markers, so `exclude_absent=True` treats an
attribute with zero positives in a dataset (e.g. UpperBody-Color-Pink in
Market1501, which Market does not annotate) as invalid there. Both variants are
reported; model selection uses exclude_absent=True.
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
NUM_ATTR = len(ATTRIBUTES)  # 40
DATASETS = ["Market1501", "PA100k", "PETA"]
EPS = 1e-20


def read_gt(csv_path: str) -> tuple[list[str], np.ndarray]:
    """Read an official gt.csv -> (relative image paths, uint8 labels [N,40]).

    Checks that the header lists the 40 attributes in the expected order.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip().lstrip("#").strip() for h in rows[0]]
    if header[1:] != ATTRIBUTES:
        raise ValueError(f"Unexpected attribute header in {csv_path}: {header[1:5]}...")
    paths = [r[0] for r in rows[1:]]
    labels = np.asarray([[int(v) for v in r[1:]] for r in rows[1:]], dtype=np.int8)
    if labels.shape[1] != NUM_ATTR or not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be 0/1 with 40 columns (no invalid markers expected).")
    return paths, labels.astype(np.uint8)


def dataset_of(path: str) -> str:
    return path.replace("\\", "/").split("/")[0]


def pedestrian_metrics(gt: np.ndarray, pred: np.ndarray, index=None) -> dict:
    """Exact port of the UPAR 2024 get_pedestrian_metrics on BINARY predictions."""
    gt = np.asarray(gt)
    pred = np.asarray(pred).astype(bool)
    if index is not None:
        gt, pred = gt[:, index], pred[:, index]
    g1, g0 = gt == 1, gt == 0
    # label-based
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
    # instance-based
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


SCALAR_KEYS = ["Challenge_Avg", "mA", "Instance_F1", "Label_F1",
               "Instance_Acc", "Instance_Prec", "Instance_Rec"]


def valid_index(gt: np.ndarray) -> np.ndarray:
    """Attributes with at least one positive in this subset (proxy for 'valid')."""
    return np.where(gt.sum(0) > 0)[0]


def leaderboard_metrics(gt: np.ndarray, pred: np.ndarray, ds: Iterable[str],
                        exclude_absent: bool = True) -> dict:
    """Score each dataset separately and average, like the leaderboard."""
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_offsets(logits: np.ndarray, gt: np.ndarray, ds, grid=None,
                      passes: int = 2, exclude_absent: bool = True) -> tuple[np.ndarray, list]:
    """Per-attribute logit offsets maximising the leaderboard Challenge_Avg.

    The grader thresholds probabilities at 0.5 (logit 0). Adding an offset b_j to
    attribute j's logit moves its effective threshold, so the tuned cut-off still
    lands exactly at 0.5 for the grader. Coordinate ascent over a small grid.
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
            cand_best, cand_val = best, off[j]
            for g in grid:
                o = off.copy(); o[j] = g
                s = score(o)
                if s > cand_best + 1e-7:
                    cand_best, cand_val = s, g
            off[j] = cand_val
            best = cand_best
        log.append((f"pass{p + 1}", best))
    return off, log
