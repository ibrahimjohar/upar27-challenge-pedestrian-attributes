"""Exports a trained checkpoint to TorchScript and builds the Codabench zip.

usage: python src/export_submission.py --ckpt runs/clip/best.pth --offsets runs/clip/offsets.json \
           --data-root /tmp/data --zip upar_track1_submission.zip

Several --ckpt paths make an ensemble: logits are averaged with --weights (equal by default),
each model with its own normalisation, before the offsets are added.

Run it with torch 2.4.1 to match the evaluation container. After zipping, the
run.py inside the zip is loaded the same way the ingestion program loads it and
its outputs are compared with the eager model on a few validation images.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn

from common import ATTRIBUTES, read_gt
from train import H, W, build_model, load_rgb_resized

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(os.path.dirname(HERE), "submission")
BATCH = 64


class Deploy(nn.Module):
    #uint8 (B, H, W, 3) in, probabilities out
    def __init__(self, nets, means, stds, weights, offsets, flip: bool):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.flip = flip
        self.weights = [float(w) for w in weights]
        self.register_buffer("mean", torch.tensor(means, dtype=torch.float32).view(-1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(stds, dtype=torch.float32).view(-1, 1, 3, 1, 1))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32).view(1, -1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).type_as(self.mean) / 255.0
        lg = None
        for k, net in enumerate(self.nets):
            xk = (x - self.mean[k]) / self.std[k]
            out = net(xk)
            if self.flip:
                out = (out + net(torch.flip(xk, dims=[3]))) / 2
            out = out.float() * self.weights[k]
            lg = out if lg is None else lg + out
        return torch.sigmoid(lg + self.offsets)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, nargs="+", help="best.pth or final.pth, several for an ensemble")
    ap.add_argument("--weights", type=float, nargs="*", default=None)
    ap.add_argument("--offsets", default=None, help="offsets.json from a train-only run")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--fp32-weights", action="store_true")
    ap.add_argument("--data-root", default=None, help="used to test the zip on validation images")
    ap.add_argument("--ann-root", default=None)
    ap.add_argument("--out-dir", default="build")
    ap.add_argument("--zip", default="upar_track1_submission.zip")
    args = ap.parse_args()

    if not torch.__version__.startswith("2.4.1"):
        print(f"warning: exporting with torch {torch.__version__}, the evaluation uses 2.4.1")

    nets, means, stds, cfgs, epochs = [], [], [], [], []
    for path in args.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        assert cfg["attributes"] == ATTRIBUTES, f"attribute order in {path} does not match"
        net, _, _ = build_model(cfg["backbone"], pretrained=False)
        net.load_state_dict(ck["state_dict"])
        nets.append(net.eval())
        means.append(cfg["mean"])
        stds.append(cfg["std"])
        cfgs.append(cfg)
        epochs.append(ck.get("epoch"))
    weights = args.weights or [1.0 / len(nets)] * len(nets)
    assert len(weights) == len(nets), "one weight per checkpoint"
    offsets = np.zeros(len(ATTRIBUTES))
    if args.offsets:
        od = json.load(open(args.offsets))
        assert od["attributes"] == ATTRIBUTES
        offsets = np.asarray(od["offsets"], dtype=np.float32)
    deploy = Deploy(nets, means, stds, weights, offsets, not args.no_flip).eval()

    #trace in fp32 on cpu, weights are stored in fp16 afterwards
    example = torch.randint(0, 256, (BATCH, H, W, 3), dtype=torch.uint8)
    with torch.no_grad():
        traced = torch.jit.trace(copy.deepcopy(deploy), example, check_trace=False)
        diff = (deploy(example) - traced(example)).abs().max().item()
    print(f"trace max abs diff: {diff:.2e}")

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    to_save = traced if args.fp32_weights else traced.half()
    to_save.save(os.path.join(args.out_dir, "model.pt"))
    shutil.copy(os.path.join(TEMPLATE, "run.py"), args.out_dir)
    shutil.copy(os.path.join(TEMPLATE, "metadata.yaml"), args.out_dir)
    meta = {"attributes": ATTRIBUTES, "backbone": [c["backbone"] for c in cfgs], "weights": weights,
            "input": [H, W], "flip_tta": not args.no_flip, "offsets": offsets.tolist(),
            "trained_on_val": [bool(c.get("train_on_val")) for c in cfgs], "checkpoint_epoch": epochs,
            "torch_export_version": torch.__version__}
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1)

    #run.py has to sit at the root of the archive
    if os.path.exists(args.zip):
        os.remove(args.zip)
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(args.out_dir)):
            zf.write(os.path.join(args.out_dir, f), arcname=f)
    names = zipfile.ZipFile(args.zip).namelist()
    assert "run.py" in names and "metadata.yaml" in names and "model.pt" in names, names
    zip_mb = os.path.getsize(args.zip) / 1e6
    print(f"{args.zip}: {zip_mb:.1f} MB, files {names}")

    #load the zipped run.py like the ingestion program does
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

        #a full batch, a single image and an odd-sized batch
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
        same_single = np.allclose(single, out[0], atol=1e-3)
        print(f"run.py vs eager: max prob diff {diff:.4f}, flipped decisions {flips}/{arr.size}, "
              f"predict_image matches: {same_single}")

        big = samples * 5
        t0 = time.time()
        for i in range(0, len(big), 64):
            run.predict_batch(big[i:i + 64])
        dt = (time.time() - t0) / len(big)
        print(f"speed: {1 / dt:.0f} images/s on {run._DEVICE}")
        if diff > 0.05:
            print("warning: run.py output differs noticeably from the eager model")

    report = {"zip": os.path.basename(args.zip), "zip_mb": round(zip_mb, 1),
              "sha256": sha256(args.zip), "ckpt": args.ckpt, "offsets": args.offsets, **meta}
    report.pop("attributes", None)
    json.dump(report, open(os.path.splitext(args.zip)[0] + "_manifest.json", "w"), indent=1)
    print(f"done: {args.zip}")


if __name__ == "__main__":
    main()
