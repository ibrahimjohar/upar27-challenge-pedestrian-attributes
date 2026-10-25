"""Runs a submission zip (or its extracted folder) locally, the way the challenge server does.

usage: python tools/predict.py --submission upar_track1_submission.zip --data-root /data \
           --list /data/annotations/task1/val/gt.csv --out predictions.csv

--list is a csv whose first column holds image paths relative to --data-root (the header row is skipped).
If it also holds the 40 labels, the challenge metrics are printed. --folder predicts every image
under a directory instead.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import tempfile
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from common import ATTRIBUTES, dataset_of, leaderboard_metrics, read_gt

EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_run(submission):
    if submission.endswith(".zip"):
        tmp = tempfile.mkdtemp(prefix="upar_sub_")
        zipfile.ZipFile(submission).extractall(tmp)
        submission = tmp
    spec = importlib.util.spec_from_file_location("submission_run", os.path.join(submission, "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    run.load_model()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="upar_track1_submission.zip or the extracted folder")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--list", default=None, help="csv with image paths in the first column")
    ap.add_argument("--folder", default=None, help="predict every image under this folder")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    gt = None
    if args.list:
        try:
            paths, gt = read_gt(args.list)
        except ValueError:
            #no labels, only image paths
            with open(args.list, newline="", encoding="utf-8") as fh:
                paths = [r[0] for r in list(csv.reader(fh))[1:] if r]
    elif args.folder:
        paths = sorted(os.path.relpath(os.path.join(d, f), args.data_root or args.folder)
                       for d, _, fs in os.walk(args.folder) for f in fs if f.lower().endswith(EXT))
        args.data_root = args.data_root or args.folder
    else:
        ap.error("give --list or --folder")

    run = load_run(args.submission)
    t0 = time.time()
    probs = []
    for i in range(0, len(paths), args.batch):
        samples = [{"image_path": os.path.join(args.data_root, p), "attribute_names": ATTRIBUTES}
                   for p in paths[i:i + args.batch]]
        probs += run.predict_batch(samples)
    probs = np.asarray(probs)
    print(f"{len(paths)} images in {time.time() - t0:.0f}s")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_path"] + ATTRIBUTES)
        for p, row in zip(paths, probs):
            w.writerow([p] + [f"{v:.4f}" for v in row])
    print("saved", args.out)

    if gt is not None:
        ds = np.asarray([dataset_of(p) for p in paths])
        m = leaderboard_metrics(gt, probs > 0.5, ds)
        print(f"Challenge_Avg {m['Challenge_Avg']:.4f}  mA {m['mA']:.4f}  inst_F1 {m['Instance_F1']:.4f}")


if __name__ == "__main__":
    main()
