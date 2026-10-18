"""Turn a trained checkpoint into a Codabench-ready zip, then test that zip like the grader would.

Run it in an environment with torch==2.4.1 (the grader's version) - 03_export_and_package.ipynb does this.

    python src/export_submission.py --ckpt run_clip/best.pth --offsets run_clip/offsets.json \
        --data-root /tmp/data --out-dir /tmp/build --zip upar_track1_submission.zip

Steps: rebuild model (no download) -> wrap with uint8 input, normalisation, flip-TTA,
offsets and sigmoid -> trace at batch 64 -> store fp16 -> write run.py/meta.json/metadata.yaml
-> zip the folder CONTENTS -> import the zipped run.py in a fresh namespace and compare its
outputs with the eager model on real validation images (batch 64, 1 and an odd size).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn

from common import ATTRIBUTES, read_gt
from train import H, W, build_model, load_rgb_resized

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(os.path.dirname(HERE), "submission")  # repo/submission: run.py + metadata.yaml
BATCH = 64


class Deploy(nn.Module):
    def __init__(self, net, mean, std, offsets, flip: bool):
        super().__init__()
        self.net, self.flip = net, flip
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32).view(1, -1))

    def forward(self, x):  # x: uint8 [B, H, W, 3]
        x = x.permute(0, 3, 1, 2).type_as(self.mean) / 255.0
        x = (x - self.mean) / self.std
        lg = self.net(x)
        if self.flip:
            lg = (lg + self.net(torch.flip(x, dims=[3]))) / 2
        return torch.sigmoid((lg + self.offsets).float())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pth (train-only run) or final.pth (train+val run)")
    ap.add_argument("--offsets", default=None, help="offsets.json from a train-only run (omit -> all zeros)")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--fp32-weights", action="store_true", help="store fp32 instead of fp16 (2x larger zip)")
    ap.add_argument("--data-root", default=None, help="for the smoke test on real val images")
    ap.add_argument("--ann-root", default=None)
    ap.add_argument("--out-dir", default="build")
    ap.add_argument("--zip", default="upar_track1_submission.zip")
    args = ap.parse_args()

    if not torch.__version__.startswith("2.4.1"):
        print(f"WARNING: torch {torch.__version__}; the grader uses 2.4.1. Export with 2.4.1 to be safe.")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert cfg["attributes"] == ATTRIBUTES, "checkpoint attribute order differs"
    net, _, _ = build_model(cfg["backbone"], pretrained=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    offsets = np.zeros(len(ATTRIBUTES))
    if args.offsets:
        od = json.load(open(args.offsets))
        assert od["attributes"] == ATTRIBUTES
        offsets = np.asarray(od["offsets"], dtype=np.float32)
    deploy = Deploy(net, cfg["mean"], cfg["std"], offsets, not args.no_flip).eval()

    # ---- trace (fixed batch 64, CPU, fp32) and store weights in fp16
    example = torch.randint(0, 256, (BATCH, H, W, 3), dtype=torch.uint8)
    with torch.no_grad():
        traced = torch.jit.trace(copy.deepcopy(deploy), example, check_trace=False)
        ref = deploy(example)
        got = traced(example)
    print(f"trace check (fp32): max |diff| = {(ref - got).abs().max().item():.2e}")

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    to_save = traced if args.fp32_weights else traced.half()
    to_save.save(os.path.join(args.out_dir, "model.pt"))
    shutil.copy(os.path.join(TEMPLATE, "run.py"), args.out_dir)
    shutil.copy(os.path.join(TEMPLATE, "metadata.yaml"), args.out_dir)
    meta = {"attributes": ATTRIBUTES, "backbone": cfg["backbone"], "input": [H, W],
            "flip_tta": not args.no_flip, "offsets": offsets.tolist(),
            "trained_on_val": bool(cfg.get("train_on_val")), "checkpoint_epoch": ck.get("epoch"),
            "torch_export_version": torch.__version__}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1)

    # ---- zip the CONTENTS (run.py at archive root)
    if os.path.exists(args.zip):
        os.remove(args.zip)
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(args.out_dir)):
            zf.write(os.path.join(args.out_dir, f), arcname=f)
    names = zipfile.ZipFile(args.zip).namelist()
    assert "run.py" in names and "metadata.yaml" in names and "model.pt" in names, names
    print(f"zip: {args.zip}  {os.path.getsize(args.zip) / 1e6:.1f} MB  files={names}  sha256={sha256(args.zip)[:16]}")

    # ---- smoke test: unzip to a fresh folder and import run.py like the ingestion program
    test_dir = args.out_dir + "_ziptest"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    zipfile.ZipFile(args.zip).extractall(test_dir)
    spec = importlib.util.spec_from_file_location("submission_run", os.path.join(test_dir, "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    t0 = time.time()
    run.load_model()
    print(f"load_model: {time.time() - t0:.1f}s on {run._DEVICE}")

    if args.data_root:
        ann = args.ann_root or os.path.join(args.data_root, "annotations")
        paths, _ = read_gt(os.path.join(ann, "task1", "val", "gt.csv"))
        rng = np.random.default_rng(0)
        pick = [paths[i] for i in rng.choice(len(paths), 64 + 1 + 37, replace=False)]
        samples = [{"index": i, "image": p, "image_path": os.path.join(args.data_root, p),
                    "attribute_names": ATTRIBUTES} for i, p in enumerate(pick)]
        out = run.predict_batch(samples[:64]) + run.predict_batch(samples[64:65]) + run.predict_batch(samples[65:])
        single = run.predict_image(samples[0])
        assert len(out) == len(samples) and all(len(r) == 40 for r in out)
        arr = np.asarray(out)
        assert np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1
        x = torch.from_numpy(np.stack([np.asarray(load_rgb_resized(s["image_path"])) for s in samples]))
        with torch.no_grad():
            eager = torch.cat([deploy(x[i:i + 64]) for i in range(0, len(x), 64)]).numpy()
        diff = np.abs(arr - eager).max()
        flips = int(((arr > 0.5) != (eager > 0.5)).sum())
        print(f"run.py vs eager fp32: max |prob diff| = {diff:.4f}, decision flips = {flips}/{arr.size}, "
              f"predict_image consistent = {np.allclose(single, out[0], atol=1e-3)}")
        # throughput
        big = samples * 5
        t0 = time.time()
        for i in range(0, len(big), 64):
            run.predict_batch(big[i:i + 64])
        dt = (time.time() - t0) / len(big)
        print(f"throughput here: {1 / dt:.0f} images/s ({run._DEVICE}); the grader's A10 is faster than a T4")
        if diff > 0.05:
            print("WARNING: large difference between run.py and eager model - investigate before uploading")
    report = {"zip": os.path.basename(args.zip), "zip_mb": round(os.path.getsize(args.zip) / 1e6, 1),
              "sha256": sha256(args.zip), "ckpt": args.ckpt, "offsets": args.offsets, **meta}
    report.pop("attributes", None)
    json.dump(report, open(os.path.splitext(args.zip)[0] + "_manifest.json", "w"), indent=1)
    print("OK - upload", args.zip, "under My Submissions")


if __name__ == "__main__":
    main()
