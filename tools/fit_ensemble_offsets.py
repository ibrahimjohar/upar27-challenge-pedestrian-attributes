"""Fits the per-attribute logit offsets of a logit-averaged ensemble on the validation set.

usage: python tools/fit_ensemble_offsets.py run_clip_vitb16/val_logits.npz run_dinov2_vitb14/val_logits.npz \
           --shift 0.5 --out offsets_shift050.json

The submitted model uses equal weights, offsets fitted on the averaged logits, and +0.5 on every attribute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
from common import ATTRIBUTES, dataset_of, leaderboard_metrics
from eval_ensemble import fast_calibrate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits", nargs="+", help="val_logits.npz of each member")
    ap.add_argument("--weights", type=float, nargs="*", default=None)
    ap.add_argument("--shift", type=float, default=0.0, help="added to every offset after fitting")
    ap.add_argument("--out", default="offsets.json")
    args = ap.parse_args()

    ws = args.weights or [1.0 / len(args.logits)] * len(args.logits)
    z, gt, paths = 0.0, None, None
    for p, w in zip(args.logits, ws):
        d = np.load(p, allow_pickle=True)
        if paths is None:
            gt, paths = d["gt"].astype(np.int64), [str(x) for x in d["paths"]]
        else:
            #every member must be scored on the same images in the same order
            assert [str(x) for x in d["paths"]] == paths, f"{p}: validation order differs"
        z = z + w * d["logits"].astype(np.float64)
    ds = np.asarray([dataset_of(p) for p in paths])

    off, _ = fast_calibrate(z, gt, ds)
    off = off + args.shift
    m = leaderboard_metrics(gt, z + off > 0, ds)
    out = {"attributes": ATTRIBUTES, "offsets": [round(float(v), 4) for v in off],
           "members": args.logits, "weights": ws, "shift": args.shift,
           "val_Challenge_Avg": round(m["Challenge_Avg"], 4)}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"val Challenge_Avg {m['Challenge_Avg']:.4f}  mA {m['mA']:.4f}  inst_F1 {m['Instance_F1']:.4f}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
