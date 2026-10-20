"""Training script for UPAR 2027 Track 1 (pedestrian attribute recognition).

usage: python src/train.py --data-root /tmp/data --backbone clip_vitb16 --epochs 10 --out runs/clip

The full training state is saved to <out>/last.pth after every epoch and every
--ckpt-every-min minutes, and running the same command again resumes from it.
With --time-budget-hours the run saves and stops once the budget is used up.
After the last epoch the best checkpoint is evaluated on the validation set and
the per-attribute threshold offsets are tuned.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

from common import (ATTRIBUTES, NUM_ATTR, SCALAR_KEYS, calibrate_offsets, dataset_of,
                    leaderboard_metrics, pedestrian_metrics, read_gt)

H, W = 256, 128
BACKBONES = {
    "clip_vitb16": ("vit_base_patch16_clip_224.openai", {"img_size": (H, W)}),
    "dinov2_vitb14": ("vit_base_patch14_dinov2.lvd142m", {"img_size": (252, 126)}),
    "vit_in21k": ("vit_base_patch16_224.augreg2_in21k_ft_in1k", {"img_size": (H, W)}),
    "siglip_vitb16": ("vit_base_patch16_siglip_224.webli", {"img_size": (H, W)}),
    "convnext_tiny": ("convnext_tiny.fb_in22k_ft_in1k", {}),
    "convnext_small": ("convnext_small.fb_in22k_ft_in1k", {}),
    "resnet50": ("resnet50.a1_in1k", {}),
}


def load_rgb_resized(path: str) -> Image.Image:
    #submission/run.py uses the same resize
    with Image.open(path) as im:
        return im.convert("RGB").resize((W, H), Image.BILINEAR)


class UPARDataset(Dataset):
    def __init__(self, root, paths, labels, transform):
        self.root = root
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb_resized(os.path.join(self.root, self.paths[i]))
        return self.transform(img), torch.from_numpy(self.labels[i].astype(np.float32))


class CameraDegrade:
    """Low resolution, blur and JPEG artefacts, as seen in surveillance footage."""

    def __init__(self, p_down=0.5, p_blur=0.3, p_jpeg=0.4):
        self.p_down = p_down
        self.p_blur = p_blur
        self.p_jpeg = p_jpeg
        self.blur = T.GaussianBlur(5, sigma=(0.1, 1.5))

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p_down:
            s = random.uniform(0.3, 0.8)
            small = (max(8, int(W * s)), max(16, int(H * s)))
            down = random.choice([Image.BILINEAR, Image.BICUBIC, Image.NEAREST])
            img = img.resize(small, down).resize((W, H), Image.BILINEAR)
        if random.random() < self.p_blur:
            img = self.blur(img)
        if random.random() < self.p_jpeg:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(30, 90))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


def make_transforms(mean, std, train: bool, aug: str = "basic"):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        return T.Compose(norm)
    #no hue/saturation jitter since colours are labels
    steps = [
        T.RandomHorizontalFlip(),
        T.Pad(10, padding_mode="edge"),
        T.RandomCrop((H, W)),
    ]
    if aug == "camera":
        steps.append(CameraDegrade())
    steps += [
        T.ColorJitter(brightness=0.2, contrast=0.2),
        *norm,
        T.RandomErasing(p=0.3, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
    ]
    return T.Compose(steps)


def balanced_weights(paths, power: float):
    #weight of an image = n_dataset ** -power, power=0.5 gives sqrt balancing
    ds = [dataset_of(p) for p in paths]
    counts = {d: ds.count(d) for d in set(ds)}
    return torch.tensor([counts[d] ** (-power) for d in ds], dtype=torch.double)


def build_model(backbone: str, pretrained: bool):
    import timm
    name, kw = BACKBONES.get(backbone, (backbone, {}))
    model = timm.create_model(name, pretrained=pretrained, num_classes=NUM_ATTR, **kw)
    pe = getattr(model, "patch_embed", None)
    if pe is not None and (H % pe.patch_size[0] or W % pe.patch_size[1]):
        #patch 14 does not divide 256x128, the patch conv just skips the last 4 rows and 2 columns
        pe.img_size = None
    cfg = model.pretrained_cfg
    mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    return model, mean, std


class WeightedBCE(nn.Module):
    """Weighted BCE from Li et al. (2015): positives get exp(1 - p), negatives exp(p)."""

    def __init__(self, pos_rate: np.ndarray):
        super().__init__()
        p = torch.tensor(pos_rate, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        self.register_buffer("wp", torch.exp(1 - p))
        self.register_buffer("wn", torch.exp(p))

    def forward(self, logits, y):
        w = y * self.wp + (1 - y) * self.wn
        return F.binary_cross_entropy_with_logits(logits.float(), y, weight=w)


class EMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def predict_logits(model, loader, device, flip_tta=False):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            lg = model(x).float()
            if flip_tta:
                lg = (lg + model(torch.flip(x, dims=[3])).float()) / 2
        out.append(lg.cpu())
    return torch.cat(out).numpy()


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available() and len(s["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(s["cuda"])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="directory containing Market1501/, PA100k/ and PETA/")
    ap.add_argument("--ann-root", default=None, help="defaults to <data-root>/annotations")
    ap.add_argument("--backbone", default="clip_vitb16")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--layer-decay", type=float, default=0.65)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.9995)
    ap.add_argument("--balance-power", type=float, default=0.5)
    ap.add_argument("--aug", default="basic", choices=["basic", "camera"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-on-val", action="store_true", help="train on train+val, no validation")
    ap.add_argument("--limit", type=int, default=0, help="use N random train and val images (debugging)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="auto", help="auto, none or a checkpoint path")
    ap.add_argument("--ckpt-every-min", type=float, default=20.0)
    ap.add_argument("--time-budget-hours", type=float, default=0.0, help="0 = no limit")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    start_time = time.time()
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "PAUSED")):
        os.remove(os.path.join(args.out, "PAUSED"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann = args.ann_root or os.path.join(args.data_root, "annotations")

    tr_paths, tr_y = read_gt(os.path.join(ann, "task1", "train", "gt.csv"))
    va_paths, va_y = read_gt(os.path.join(ann, "task1", "val", "gt.csv"))
    if args.limit:
        rng = np.random.default_rng(0)
        ti = rng.choice(len(tr_paths), min(args.limit, len(tr_paths)), replace=False)
        vi = rng.choice(len(va_paths), min(args.limit, len(va_paths)), replace=False)
        tr_paths, tr_y = [tr_paths[i] for i in ti], tr_y[ti]
        va_paths, va_y = [va_paths[i] for i in vi], va_y[vi]
    if args.train_on_val:
        tr_paths, tr_y = tr_paths + va_paths, np.concatenate([tr_y, va_y])
    va_ds = [dataset_of(p) for p in va_paths]
    print(f"train: {len(tr_paths)}, val: {len(va_paths)}, device: {device}, gpus: {torch.cuda.device_count()}", flush=True)

    model, mean, std = build_model(args.backbone, not args.no_pretrained)
    model.to(device)
    criterion = WeightedBCE(tr_y.mean(0)).to(device)

    train_set = UPARDataset(args.data_root, tr_paths, tr_y, make_transforms(mean, std, True, args.aug))
    val_set = UPARDataset(args.data_root, va_paths, va_y, make_transforms(mean, std, False))
    sampler = WeightedRandomSampler(balanced_weights(tr_paths, args.balance_power), len(tr_paths), replacement=True)
    tl = DataLoader(train_set, args.batch_size, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(val_set, args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    from timm.optim import create_optimizer_v2
    opt = create_optimizer_v2(model, "adamw", lr=args.lr, weight_decay=args.weight_decay,
                              layer_decay=args.layer_decay)
    #LambdaLR ignores timm's lr_scale, so apply it to the base lr
    for g in opt.param_groups:
        g["lr"] = args.lr * g.get("lr_scale", 1.0)
        g["initial_lr"] = g["lr"]
    total_steps = args.epochs * len(tl)
    warmup = int(args.warmup_epochs * len(tl))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, args.ema_decay)
    net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    config = dict(vars(args), mean=mean, std=std, H=H, W=W, attributes=ATTRIBUTES)
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=1)

    last_path = os.path.join(args.out, "last.pth")
    best_path = os.path.join(args.out, "best.pth")
    resume_path = None
    if args.resume == "auto":
        resume_path = last_path if os.path.exists(last_path) else None
    elif args.resume != "none":
        resume_path = args.resume

    epoch_done, steps_done, best, best_ep = 0, 0, -1.0, -1
    ep_loss, ep_n = 0.0, 0
    if resume_path:
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        if st["config"]["backbone"] != args.backbone or st["config"]["train_on_val"] != args.train_on_val:
            sys.exit(f"{resume_path} has a different backbone or train_on_val setting")
        if st["config"]["epochs"] != args.epochs:
            print(f"epochs changed from {st['config']['epochs']} to {args.epochs}")
        model.load_state_dict(st["model"])
        ema.module.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        epoch_done, steps_done = st["epoch_done"], st["steps_done"]
        best, best_ep = st["best"], st["best_ep"]
        ep_loss, ep_n = st.get("ep_loss", 0.0), st.get("ep_n", 0)
        print(f"resuming from {resume_path}: epoch {epoch_done + 1}, step {steps_done}/{len(tl)}, "
              f"best {best:.4f} at epoch {best_ep}", flush=True)

    def save_state(e_done, s_done, loss_sum, n):
        atomic_save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                     "epoch_done": e_done, "steps_done": s_done, "best": best, "best_ep": best_ep,
                     "ep_loss": loss_sum, "ep_n": n, "config": config, "epoch": e_done}, last_path)

    hist_path = os.path.join(args.out, "history.csv")
    new_hist = not os.path.exists(hist_path) or not resume_path
    hist_f = open(hist_path, "w" if new_hist else "a", newline="")
    hist = csv.writer(hist_f)
    if new_hist:
        hist.writerow(["epoch", "train_loss", "minutes"] + [f"val_{k}" for k in SCALAR_KEYS])

    budget_s = args.time_budget_hours * 3600
    last_ckpt = time.time()

    for ep in range(epoch_done + 1, args.epochs + 1):
        net.train()
        t0 = time.time()
        skip = steps_done if ep == epoch_done + 1 else 0
        if skip == 0:
            ep_loss, ep_n = 0.0, 0
        loader_it = iter(tl)
        for it in range(skip, len(tl)):
            x, y = next(loader_it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            ep_loss += loss.item()
            ep_n += 1

            if it % 200 == 0:
                lr = max(g["lr"] for g in opt.param_groups)
                print(f"  ep {ep} it {it}/{len(tl)} loss {ep_loss / ep_n:.4f} lr {lr:.2e}", flush=True)
            if budget_s and time.time() - start_time > budget_s:
                save_state(ep - 1, it + 1, ep_loss, ep_n)
                with open(os.path.join(args.out, "PAUSED"), "w") as fh:
                    fh.write(f"epoch {ep} step {it + 1}/{len(tl)}\n")
                print(f"time budget reached, stopped at epoch {ep} step {it + 1}/{len(tl)}", flush=True)
                hist_f.close()
                return
            if time.time() - last_ckpt > args.ckpt_every_min * 60:
                save_state(ep - 1, it + 1, ep_loss, ep_n)
                last_ckpt = time.time()
        del loader_it

        mins = (time.time() - t0) / 60
        train_loss = ep_loss / max(1, ep_n)
        row = [ep, round(train_loss, 5), round(mins, 2)]
        if not args.train_on_val:
            lg = predict_logits(ema.module, vl, device)
            m = leaderboard_metrics(va_y, lg > 0, va_ds)
            row += [round(m[k], 5) for k in SCALAR_KEYS]
            per_ds = " ".join(f"{d} {v['Challenge_Avg']:.3f}" for d, v in m["per_dataset"].items())
            print(f"ep {ep} loss {train_loss:.4f} ({mins:.1f} min) | val avg {m['Challenge_Avg']:.4f} "
                  f"mA {m['mA']:.4f} inst_f1 {m['Instance_F1']:.4f} | {per_ds}", flush=True)
            if m["Challenge_Avg"] > best:
                best, best_ep = m["Challenge_Avg"], ep
                atomic_save({"state_dict": ema.module.state_dict(), "config": config, "epoch": ep}, best_path)
        else:
            print(f"ep {ep} loss {train_loss:.4f} ({mins:.1f} min)", flush=True)
            atomic_save({"state_dict": ema.module.state_dict(), "config": config, "epoch": ep},
                        os.path.join(args.out, "final.pth"))
        hist.writerow(row)
        hist_f.flush()
        save_state(ep, 0, 0.0, 0)
        last_ckpt = time.time()
    hist_f.close()

    if args.train_on_val:
        with open(os.path.join(args.out, "DONE"), "w") as fh:
            fh.write("train+val\n")
        print("finished (train+val)")
        return

    print(f"best epoch {best_ep}, val Challenge_Avg {best:.4f}", flush=True)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    ema.module.load_state_dict(ck["state_dict"])
    lg_plain = predict_logits(ema.module, vl, device)
    lg_flip = predict_logits(ema.module, vl, device, flip_tta=True)
    offsets, cal_log = calibrate_offsets(lg_flip, va_y, va_ds)
    np.savez_compressed(os.path.join(args.out, "val_logits.npz"), logits=lg_flip, logits_plain=lg_plain,
                        gt=va_y, paths=np.array(va_paths))
    json.dump({"attributes": ATTRIBUTES, "offsets": offsets.tolist(), "log": cal_log},
              open(os.path.join(args.out, "offsets.json"), "w"), indent=1)

    variants = {
        "plain": lg_plain > 0,
        "flip": lg_flip > 0,
        "flip + offsets": (lg_flip + offsets) > 0,
    }
    datasets = sorted(set(va_ds))
    summary = {}
    with open(os.path.join(args.out, "results_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "exclude_absent"] + SCALAR_KEYS + [f"{d}_Challenge_Avg" for d in datasets])
        for name, pred in variants.items():
            for ex in (True, False):
                m = leaderboard_metrics(va_y, pred, va_ds, ex)
                w.writerow([name, ex] + [round(m[k], 4) for k in SCALAR_KEYS]
                           + [round(m["per_dataset"][d]["Challenge_Avg"], 4) for d in datasets])
                if ex:
                    summary[name] = m
                    print(f"{name:16s} avg {m['Challenge_Avg']:.4f}  mA {m['mA']:.4f}  inst_f1 {m['Instance_F1']:.4f}")

    pred = variants["flip + offsets"]
    dsa = np.asarray(va_ds)
    with open(os.path.join(args.out, "per_attribute.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attribute", "dataset", "n_pos", "pos_recall", "neg_recall", "label_mA", "label_F1", "offset"])
        for d in datasets:
            msk = dsa == d
            m = pedestrian_metrics(va_y[msk], pred[msk])
            for j, a in enumerate(ATTRIBUTES):
                w.writerow([a, d, int(va_y[msk, j].sum()), round(m["pos_recall"][j], 4), round(m["neg_recall"][j], 4),
                            round(m["label_ma"][j], 4), round(m["label_f1"][j], 4), offsets[j]])
    json.dump(summary, open(os.path.join(args.out, "results_summary.json"), "w"), indent=1)
    with open(os.path.join(args.out, "DONE"), "w") as fh:
        fh.write(f"best epoch {best_ep}, val Challenge_Avg {best:.4f}\n")
    print(f"results saved to {args.out}")


if __name__ == "__main__":
    main()
