"""Scores single models and logit-averaged ensembles on the validation set.

usage: python tools/eval_ensemble.py clip=val_clip.npz dinov2=val_dinov2.npz camaug=val_camaug.npz

Each npz is the val_logits.npz written by train.py. Offsets are scored two ways:
tuned and scored on the full val set (optimistic), and 2-fold (tuned on one half,
scored on the other), which is the fairer number for comparing configurations.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import EPS, dataset_of, leaderboard_metrics, valid_index


def load(spec):
    name, path = spec.split("=", 1)
    d = np.load(path, allow_pickle=True)
    return name, d["logits"].astype(np.float64), d["gt"].astype(np.int64), [str(p) for p in d["paths"]]


def fast_calibrate(lg, gt, ds, grid=None, passes=2):
    """Same coordinate ascent as common.calibrate_offsets, but only the changed column is rescored."""
    if grid is None:
        grid = np.round(np.arange(-1.5, 1.5001, 0.125), 4)
    n_attr = lg.shape[1]
    parts = []
    for d in sorted(set(ds)):
        m = ds == d
        g = gt[m] == 1
        v = np.zeros(n_attr, dtype=bool)
        v[valid_index(gt[m])] = True
        pred = (lg[m] > 0) & v
        gpos, gneg = g.sum(0).astype(float), (~g).sum(0).astype(float)
        lma = ((g & pred).sum(0) / (gpos + EPS) + (~g & ~pred).sum(0) / (gneg + EPS)) / 2
        parts.append({"lg": lg[m], "g": g, "v": v, "pred": pred, "gpos": gpos, "gneg": gneg, "lma": lma,
                      "inter": (g & pred).sum(1).astype(float), "ipred": pred.sum(1).astype(float),
                      "igt": (g & v).sum(1).astype(float)})

    def part_score(p, j=None, col=None):
        inter, ipred, lma = p["inter"], p["ipred"], p["lma"]
        if j is not None and p["v"][j]:
            old = p["pred"][:, j]
            gj = p["g"][:, j]
            inter = inter - (old & gj) + (col & gj)
            ipred = ipred - old + col
            lma = lma.copy()
            lma[j] = ((gj & col).sum() / (p["gpos"][j] + EPS) + (~gj & ~col).sum() / (p["gneg"][j] + EPS)) / 2
        ma = lma[p["v"]].mean()
        prec, rec = np.mean(inter / (ipred + EPS)), np.mean(inter / (p["igt"] + EPS))
        f1 = 2 * prec * rec / (prec + rec + EPS)
        return 2 * ma * f1 / (ma + f1 + EPS), inter, ipred, lma

    off = np.zeros(n_attr)
    best = np.mean([part_score(p)[0] for p in parts])
    for _ in range(passes):
        for j in range(n_attr):
            best_val = off[j]
            for gv in grid:
                s = np.mean([part_score(p, j, (p["lg"][:, j] + gv) > 0)[0] for p in parts])
                if s > best + 1e-7:
                    best, best_val = s, gv
            off[j] = best_val
            for p in parts:
                if p["v"][j]:
                    col = (p["lg"][:, j] + off[j]) > 0
                    _, p["inter"], p["ipred"], p["lma"] = part_score(p, j, col)
                    p["pred"][:, j] = col
    return off, best


def score(lg, gt, ds):
    m = leaderboard_metrics(gt, lg > 0, ds)
    return m["Challenge_Avg"], m["mA"], m["Instance_F1"]


def two_fold(lg, gt, ds, seed=0):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(gt), dtype=int)
    for d in set(ds):
        idx = np.flatnonzero(ds == d)
        rng.shuffle(idx)
        fold[idx[: len(idx) // 2]] = 1
    pred = np.zeros_like(gt, dtype=bool)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        off, _ = fast_calibrate(lg[tr], gt[tr], ds[tr])
        pred[te] = (lg[te] + off) > 0
    m = leaderboard_metrics(gt, pred, ds)
    return m["Challenge_Avg"]


def main():
    models = [load(a) for a in sys.argv[1:] if "=" in a]
    paths = models[0][3]
    for name, _, gt, p in models[1:]:
        assert p == paths, f"{name}: validation order differs"
    gt = models[0][2]
    ds = np.asarray([dataset_of(p) for p in paths])
    lg = {name: x for name, x, _, _ in models}
    names = list(lg)

    configs = [((n,), (1.0,)) for n in names]
    for a, b in itertools.combinations(names, 2):
        for w in (0.25, 0.5, 0.75):
            configs.append(((a, b), (w, 1 - w)))
    if len(names) == 3:
        configs.append((tuple(names), (1 / 3, 1 / 3, 1 / 3)))

    rows = []
    for members, weights in configs:
        z = sum(w * lg[n] for n, w in zip(members, weights))
        avg, ma, f1 = score(z, gt, ds)
        off, _ = fast_calibrate(z, gt, ds)
        avg_off, ma_off, f1_off = score(z + off, gt, ds)
        cv = two_fold(z, gt, ds)
        label = " + ".join(f"{w:.2f} {n}" for n, w in zip(members, weights))
        rows.append({"config": label, "no_offsets": round(avg, 4), "mA": round(ma, 4), "inst_f1": round(f1, 4),
                     "offsets_in_sample": round(avg_off, 4), "offsets_2fold": round(cv, 4)})
        print(f"{label:45s} plain {avg:.4f}  offsets {avg_off:.4f}  2-fold {cv:.4f}", flush=True)

    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "ensemble_val.json"
    json.dump(rows, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
