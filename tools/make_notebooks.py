"""Writes notebooks/01-03. Run from the repo root: python tools/make_notebooks.py"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def notebook(cells, gpu):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isInternetEnabled": True}}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 4}


REPO_CFG = '''
GITHUB_REPO = "ibrahimjohar/upar27-challenge-pedestrian-attributes"
BRANCH = "main"
'''

CLONE = '''
#token comes from the Kaggle secret GITHUB_TOKEN, the clone stays in /tmp so it is not saved with the output
import os, subprocess, shutil
TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    print("no GITHUB_TOKEN secret, results will not be pushed")
owner = GITHUB_REPO.split("/")[0]
url = f"https://{owner}:{TOKEN}@github.com/{GITHUB_REPO}.git" if TOKEN else f"https://github.com/{GITHUB_REPO}.git"
CODE = "/tmp/code"
shutil.rmtree(CODE, ignore_errors=True)
r = subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, CODE], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError("git clone failed: " + (r.stderr.replace(TOKEN, "***") if TOKEN else r.stderr))
print(subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%h %s (%cr)"], capture_output=True, text=True).stdout)
'''

FIND_TAR = '''
import glob, time
tars = glob.glob("/kaggle/input/**/upar_data.tar", recursive=True)
assert tars, "add the output of notebook 01 as an input"
if not os.path.exists("/tmp/data/annotations/task1/train/gt.csv"):
    t0 = time.time()
    os.makedirs("/tmp/data", exist_ok=True)
    subprocess.run(["tar", "-xf", tars[0], "-C", "/tmp/data"], check=True)
    print(f"extracted in {time.time() - t0:.0f}s")
!du -sh /tmp/data
'''

#00 data exploration
nb0 = notebook([
    md("""
# 00 Data exploration

Accelerator: None, Internet: On, secret `GITHUB_TOKEN` attached. Input: output of notebook 01.

What is in the Track 1 data, and which of it matters for the model: split sizes per source dataset,
how often each attribute is positive, attributes missing from a dataset, image sizes, and how the three
cameras differ in brightness, contrast and sharpness. Tables and plots go to `results/data_exploration/`.
"""),
    code(REPO_CFG + '''
PUSH_RESULTS = True
SAMPLE_PER_DATASET = 3000   # images per dataset for the size and appearance statistics
'''),
    code(CLONE),
    code(FIND_TAR),
    code('''
import sys, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, f"{CODE}/src")
from common import ATTRIBUTES, read_gt, dataset_of

OUT = "/kaggle/working/data_exploration"
os.makedirs(OUT, exist_ok=True)
DATA = "/tmp/data"
splits = {}
for split in ("train", "val"):
    paths, y = read_gt(f"{DATA}/annotations/task1/{split}/gt.csv")
    splits[split] = (np.asarray(paths), y, np.asarray([dataset_of(p) for p in paths]))
DATASETS = sorted(set(splits["train"][2]))
'''),
    md("""
## 1. Split sizes

Train and validation come from the same three datasets but in very different proportions.
"""),
    code('''
rows = []
for split, (paths, y, ds) in splits.items():
    for d in DATASETS:
        rows.append({"split": split, "dataset": d, "images": int((ds == d).sum())})
sizes = pd.DataFrame(rows).pivot(index="dataset", columns="split", values="images")
sizes["train_share"] = (sizes["train"] / sizes["train"].sum()).round(3)
sizes["val_share"] = (sizes["val"] / sizes["val"].sum()).round(3)
sizes.loc["total"] = [sizes["train"].sum(), sizes["val"].sum(), 1.0, 1.0]
sizes.to_csv(f"{OUT}/split_sizes.csv")
sizes
'''),
    md("""
## 2. How often each attribute is positive

Positive rate per attribute in the training set, per source dataset. 0 means the attribute is never
labelled positive in that dataset.
"""),
    code('''
paths, y, ds = splits["train"]
rate = pd.DataFrame({d: y[ds == d].mean(0) for d in DATASETS}, index=ATTRIBUTES)
rate["all"] = y.mean(0)
rate = rate.sort_values("all")
rate.round(4).to_csv(f"{OUT}/positive_rate_train.csv")

fig, ax = plt.subplots(figsize=(7, 11))
im = ax.imshow(rate[DATASETS + ["all"]].values, aspect="auto", cmap="viridis", vmin=0, vmax=0.5)
ax.set_yticks(range(len(rate)))
ax.set_yticklabels(rate.index, fontsize=8)
ax.set_xticks(range(len(DATASETS) + 1))
ax.set_xticklabels(DATASETS + ["all"])
plt.colorbar(im, ax=ax, label="positive rate (clipped at 0.5)")
ax.set_title("attribute positive rate, train")
plt.tight_layout()
plt.savefig(f"{OUT}/positive_rate_train.png", dpi=150)
plt.show()

print("attributes below 5% positive:", int((rate["all"] < 0.05).sum()))
print("attributes below 1% positive:", int((rate["all"] < 0.01).sum()))
print("rarest:", ", ".join(f"{a} {r:.4f}" for a, r in rate["all"].head(8).items()))
for d in DATASETS:
    print(f"never positive in {d}:", [a for a in ATTRIBUTES if rate.loc[a, d] == 0])
print("attributes with left/right in the name:", [a for a in ATTRIBUTES if "left" in a.lower() or "right" in a.lower()])
print("colour attributes:", sum("Color" in a for a in ATTRIBUTES), "of", len(ATTRIBUTES))
'''),
    md("""
## 3. Train vs validation label distribution

If a dataset's validation split had a different label mix from its training split, the val score would be
harder to trust for model selection.
"""),
    code('''
vp, vy, vds = splits["val"]
fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 4))
gap = {}
for ax, d in zip(axes, DATASETS):
    tr, va = y[ds == d].mean(0), vy[vds == d].mean(0)
    gap[d] = np.abs(tr - va).max()
    ax.scatter(tr, va, s=12)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(d)
    ax.set_xlabel("train positive rate")
axes[0].set_ylabel("val positive rate")
plt.tight_layout()
plt.savefig(f"{OUT}/train_vs_val_rate.png", dpi=150)
plt.show()
print("largest train/val gap in positive rate:", {d: round(float(g), 3) for d, g in gap.items()})
for d in DATASETS:
    diff = vy[vds == d].mean(0) - y[ds == d].mean(0)
    j = int(np.abs(diff).argmax())
    print(f"{d}: biggest shift {ATTRIBUTES[j]} {y[ds == d].mean(0)[j]:.3f} (train) -> {vy[vds == d].mean(0)[j]:.3f} (val)")
'''),
    md("""
## 4. Labels per image and attributes that go together
"""),
    code('''
card = pd.DataFrame({d: pd.Series(y[ds == d].sum(1)).describe()[["mean", "min", "max"]] for d in DATASETS}).T
card.round(2).to_csv(f"{OUT}/labels_per_image.csv")
print(card.round(2))

#phi coefficient between pairs of attributes
yc = y.astype(float)
std = yc.std(0)
keep = std > 0
corr = np.corrcoef(yc[:, keep].T)
names = np.asarray(ATTRIBUTES)[keep]
pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pairs.append((names[i], names[j], corr[i, j]))
pairs = pd.DataFrame(pairs, columns=["a", "b", "phi"]).sort_values("phi", key=abs, ascending=False)
pairs.head(20).round(3).to_csv(f"{OUT}/top_correlations.csv", index=False)
pairs.head(12).round(3)
'''),
    md("""
## 5. Image sizes

Native crop sizes per dataset, on a random sample. Every image is resized to 256x128 (height x width) for the model.
"""),
    code('''
random.seed(0)
sample = {}
for d in DATASETS:
    idx = np.flatnonzero(np.concatenate([splits["train"][2], splits["val"][2]]) == d)
    allp = np.concatenate([splits["train"][0], splits["val"][0]])
    sample[d] = [allp[i] for i in random.sample(list(idx), min(SAMPLE_PER_DATASET, len(idx)))]

size_rows = []
for d, ps in sample.items():
    for p in ps:
        with Image.open(os.path.join(DATA, p)) as im:
            w, h = im.size
        size_rows.append({"dataset": d, "w": w, "h": h})
sz = pd.DataFrame(size_rows)
sz["aspect_h_over_w"] = sz.h / sz.w
summary = sz.groupby("dataset").agg(h_median=("h", "median"), w_median=("w", "median"),
                                    h_min=("h", "min"), h_max=("h", "max"),
                                    aspect_median=("aspect_h_over_w", "median"),
                                    below_256_tall=("h", lambda s: float((s < 256).mean())))
summary.round(3).to_csv(f"{OUT}/image_sizes.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for d in DATASETS:
    s = sz[sz.dataset == d]
    ax[0].hist(s.h, bins=40, alpha=0.5, label=d)
    ax[1].hist(s.aspect_h_over_w, bins=40, alpha=0.5, label=d)
ax[0].set_xlabel("height (px)")
ax[1].set_xlabel("height / width")
ax[1].axvline(2.0, color="k", ls="--", lw=0.8)
ax[0].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/image_sizes.png", dpi=150)
plt.show()
summary.round(3)
'''),
    md("""
## 6. How the cameras differ

Brightness (mean grey level), contrast (std of grey level) and sharpness (variance of the Laplacian,
computed after resizing to 256x128 so the datasets are comparable) on the same sample.
These differences are the domain shift the hidden test set will add to.
"""),
    code('''
def appearance(path):
    with Image.open(os.path.join(DATA, path)) as im:
        g = np.asarray(im.convert("L").resize((128, 256), Image.BILINEAR), dtype=np.float32)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return g.mean(), g.std(), lap.var()

app_rows = []
for d, ps in sample.items():
    for p in ps[:1500]:
        b, c, s = appearance(p)
        app_rows.append({"dataset": d, "brightness": b, "contrast": c, "sharpness": s})
app = pd.DataFrame(app_rows)
app.groupby("dataset").median().round(1).to_csv(f"{OUT}/appearance.csv")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, ["brightness", "contrast", "sharpness"]):
    ax.boxplot([app[app.dataset == d][col] for d in DATASETS], showfliers=False)
    ax.set_xticks(range(1, len(DATASETS) + 1))
    ax.set_xticklabels(DATASETS)
    ax.set_title(col)
axes[2].set_yscale("log")
plt.tight_layout()
plt.savefig(f"{OUT}/appearance.png", dpi=150)
plt.show()
app.groupby("dataset").median().round(1)
'''),
    md("""
## 7. Examples

Six random training images per dataset with a few of their positive attributes. Saved as jpg so it stays
out of the repo (PETA is research-use only).
"""),
    code('''
random.seed(1)
short = lambda a: a.replace("UpperBody-", "U-").replace("LowerBody-", "L-").replace("Accessory-", "").replace("Color-", "")
fig, axes = plt.subplots(len(DATASETS), 6, figsize=(12, 4.2 * len(DATASETS)))
for r, d in enumerate(DATASETS):
    idx = random.sample(list(np.flatnonzero(ds == d)), 6)
    for c, i in enumerate(idx):
        with Image.open(os.path.join(DATA, paths[i])) as im:
            axes[r, c].imshow(im.convert("RGB").resize((128, 256)))
        pos = [short(ATTRIBUTES[j]) for j in np.flatnonzero(y[i]) if "Adult" not in ATTRIBUTES[j]][:5]
        axes[r, c].set_title("\\n".join(pos), fontsize=7)
        axes[r, c].axis("off")
    axes[r, 0].text(-20, 128, d, rotation=90, va="center", ha="right", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/examples.jpg", dpi=110)
plt.show()
'''),
    md("""
# Part B: looking at the images

The numbers above say what is in the labels. This part opens the images themselves: native sizes at true scale,
colour and greyscale, what an average crop looks like, what the model's patches cover, the same attribute in each
dataset, the rare attributes, some classic filters, and what the training augmentations actually produce.
"""),
    code('''
import torch
from PIL import ImageOps, ImageDraw
from train import make_transforms, H, W

def load(p, size=None):
    with Image.open(os.path.join(DATA, p)) as im:
        im = im.convert("RGB")
        return im.resize(size, Image.BILINEAR) if size else im.copy()

def show_row(ax_row, imgs, titles=None, fs=7):
    for k, ax in enumerate(ax_row):
        ax.axis("off")
        if k < len(imgs):
            ax.imshow(imgs[k], cmap="gray" if np.asarray(imgs[k]).ndim == 2 else None)
            if titles:
                ax.set_title(titles[k], fontsize=fs)

allp = np.concatenate([splits["train"][0], splits["val"][0]])
ally = np.concatenate([splits["train"][1], splits["val"][1]])
allds = np.concatenate([splits["train"][2], splits["val"][2]])
A = {a: j for j, a in enumerate(ATTRIBUTES)}
rng = np.random.default_rng(0)

def positives(attr, d=None, n=1):
    m = ally[:, A[attr]] == 1
    if d is not None:
        m &= allds == d
    idx = np.flatnonzero(m)
    return [allp[i] for i in rng.choice(idx, min(n, len(idx)), replace=False)] if len(idx) else []
'''),
    md("""
## 8. Native size, drawn at true scale

One image per dataset at its real pixel size, next to the 256x128 version the model gets.
Market1501 crops are all 128x64, so every one of them is upscaled 2x.
"""),
    code('''
picks = {d: allp[rng.choice(np.flatnonzero(allds == d))] for d in DATASETS}
natives = {d: load(p) for d, p in picks.items()}
canvas_h = max(256, max(im.height for im in natives.values()))
canvas = Image.new("RGB", (sum(im.width + W + 60 for im in natives.values()), canvas_h), "white")
x = 0
draw = ImageDraw.Draw(canvas)
for d, im in natives.items():
    canvas.paste(im, (x, canvas_h - im.height))
    draw.text((x, 2), f"{d} {im.width}x{im.height}", fill="black")
    x += im.width + 20
    canvas.paste(im.resize((W, H), Image.BILINEAR), (x, canvas_h - H))
    draw.text((x, 12), "256x128", fill="black")
    x += W + 40
plt.figure(figsize=(canvas.width / 80, canvas.height / 80))
plt.imshow(canvas)
plt.axis("off")
plt.show()
'''),
    md("""
## 9. Greyscale and night images

Some cameras record in greyscale (night or infrared). On those, colour labels cannot be read from the image.
An image counts as greyscale here when its three channels differ by less than 3 grey levels on average.
"""),
    code('''
def channel_spread(p):
    a = np.asarray(load(p, (64, 128)), dtype=np.float32)
    return float(np.mean(np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])) / 2)

grey = {}
for d in DATASETS:
    spreads = np.asarray([channel_spread(p) for p in sample[d][:1500]])
    grey[d] = (spreads < 3).mean()
print("share of greyscale images:", {d: round(float(v), 3) for d, v in grey.items()})

fig, axes = plt.subplots(1, 8, figsize=(13, 4))
found = [p for p in sample["PA100k"][:3000] if channel_spread(p) < 3][:8]
colour_idx = [j for j, a in enumerate(ATTRIBUTES) if "Color" in a]
lookup = {p: i for i, p in enumerate(allp)}
titles = []
for p in found:
    yy = ally[lookup[p]]
    titles.append("\\n".join(ATTRIBUTES[j].replace("Body-Color-", "-") for j in colour_idx if yy[j])[:60])
show_row(axes, [load(p, (W, H)) for p in found], titles)
plt.suptitle("PA-100K greyscale images and the colour labels they carry", fontsize=10)
plt.tight_layout()
plt.show()
'''),
    md("""
## 10. Colour casts

Mean colour of each image (red vs blue minus green), per dataset. A camera with a colour cast shifts its whole cloud.
"""),
    code('''
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for d in DATASETS:
    m = np.asarray([np.asarray(load(p, (32, 64)), dtype=np.float32).reshape(-1, 3).mean(0) for p in sample[d][:800]])
    ax[0].scatter(m[:, 0] - m[:, 1], m[:, 2] - m[:, 1], s=4, alpha=0.4, label=d)
    for c, col in enumerate("rgb"):
        ax[1].hist(m[:, c], bins=40, histtype="step", color=col, ls=["-", "--", ":"][DATASETS.index(d)])
ax[0].axhline(0, color="k", lw=0.5)
ax[0].axvline(0, color="k", lw=0.5)
ax[0].set_xlabel("mean R - G")
ax[0].set_ylabel("mean B - G")
ax[0].legend(markerscale=4)
ax[1].set_title("per-image mean of each channel (solid Market, dashed PA-100K, dotted PETA)", fontsize=9)
plt.tight_layout()
plt.show()
'''),
    md("""
## 11. Mean and standard deviation image

Average of 1000 resized crops per dataset. A sharp mean silhouette means people are centred and scaled
consistently; the std image shows where the variation is (background, legs, arms).
"""),
    code('''
fig, axes = plt.subplots(2, len(DATASETS), figsize=(7, 7))
for c, d in enumerate(DATASETS):
    stack = np.stack([np.asarray(load(p, (W, H)), dtype=np.float32) for p in sample[d][:1000]])
    axes[0, c].imshow(stack.mean(0).astype(np.uint8))
    axes[0, c].set_title(f"{d} mean", fontsize=9)
    axes[1, c].imshow(stack.std(0).mean(-1), cmap="magma")
    axes[1, c].set_title("std", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()
'''),
    md("""
## 12. What one token sees

CLIP ViT-B/16 cuts the 256x128 input into a 16x8 grid of 16x16 patches; DINOv2 ViT-B/14 uses an 18x9 grid
of 14x14 patches. A head is only a couple of patches, sunglasses or a hat less than one.
"""),
    code('''
fig, axes = plt.subplots(1, 6, figsize=(12, 5))
ex = [p for d in DATASETS for p in positives("Accessory-Hat", d)]
ex += [p for d in DATASETS for p in positives("Accessory-Glasses-Normal", d)]
for ax, p in zip(axes, ex):
    im = load(p, (W, H))
    dr = ImageDraw.Draw(im)
    for gx in range(0, W + 1, 16):
        dr.line([(gx, 0), (gx, H)], fill=(255, 255, 0), width=1)
    for gy in range(0, H + 1, 16):
        dr.line([(0, gy), (W, gy)], fill=(255, 255, 0), width=1)
    ax.imshow(im)
    ax.axis("off")
axes[0].set_title("hat", fontsize=9)
axes[3].set_title("glasses", fontsize=9)
plt.suptitle("16x16 patch grid (CLIP) on hat and glasses examples", fontsize=10)
plt.tight_layout()
plt.show()
'''),
    md("""
## 13. The same attribute in each dataset

Four positives per dataset for a few attributes. Differences here are differences in how each source looks
and how it was labelled, which is what the model has to bridge.
"""),
    code('''
for attr in ["Accessory-Backpack", "Accessory-Hat", "UpperBody-Color-Red", "LowerBody-Type-Skirt&Dress", "Age-Old"]:
    fig, axes = plt.subplots(1, 12, figsize=(14, 3))
    imgs, titles = [], []
    for d in DATASETS:
        for p in positives(attr, d, 4):
            imgs.append(load(p, (W, H)))
            titles.append(d)
    show_row(axes, imgs, titles)
    plt.suptitle(attr, fontsize=10)
    plt.tight_layout()
    plt.show()
'''),
    md("""
## 14. The rarest attributes

These are the attributes with well under 1% positives, which mA weights as much as any other.
"""),
    code('''
for attr in ["LowerBody-Color-Purple", "LowerBody-Color-Orange", "Accessory-Glasses-Sun", "Hair-Length-Bald", "LowerBody-Color-Yellow"]:
    idx = np.flatnonzero(ally[:, A[attr]] == 1)
    fig, axes = plt.subplots(1, 10, figsize=(13, 3))
    pick = rng.choice(idx, min(10, len(idx)), replace=False)
    show_row(axes, [load(allp[i], (W, H)) for i in pick], [allds[i] for i in pick])
    plt.suptitle(f"{attr}: {len(idx)} positives in train+val", fontsize=10)
    plt.tight_layout()
    plt.show()
'''),
    md("""
## 15. Filters: what detail survives

Greyscale, Sobel edge magnitude and Laplacian on one image per dataset, at native size and after resizing to
256x128. Edges are what the early layers pick up, so this shows how much fine structure each camera gives.
"""),
    code('''
def sobel(g):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    pad = np.pad(g, 1, mode="edge")
    gx = sum(kx[i, j] * pad[i:i + g.shape[0], j:j + g.shape[1]] for i in range(3) for j in range(3))
    gy = sum(kx.T[i, j] * pad[i:i + g.shape[0], j:j + g.shape[1]] for i in range(3) for j in range(3))
    return np.hypot(gx, gy)

def laplacian(g):
    pad = np.pad(g, 1, mode="edge")
    return pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * g

fig, axes = plt.subplots(len(DATASETS), 6, figsize=(11, 4.4 * len(DATASETS)))
for r, d in enumerate(DATASETS):
    im = natives[d]
    g = np.asarray(im.convert("L"), dtype=np.float32)
    gr = np.asarray(im.convert("L").resize((W, H), Image.BILINEAR), dtype=np.float32)
    eq = np.asarray(ImageOps.equalize(im.resize((W, H), Image.BILINEAR)))
    panels = [im, sobel(g), np.abs(laplacian(g)), sobel(gr), np.abs(laplacian(gr)), eq]
    names = [f"{d} native {im.width}x{im.height}", "Sobel native", "|Laplacian| native",
             "Sobel 256x128", "|Laplacian| 256x128", "histogram equalised"]
    for c, (pnl, nm) in enumerate(zip(panels, names)):
        axes[r, c].imshow(pnl, cmap="gray" if np.asarray(pnl).ndim == 2 else None)
        axes[r, c].set_title(nm, fontsize=7)
        axes[r, c].axis("off")
plt.tight_layout()
plt.show()

#average log power spectrum per dataset, a blurry source has less energy away from the centre
fig, axes = plt.subplots(1, len(DATASETS), figsize=(9, 4))
for ax, d in zip(axes, DATASETS):
    acc = np.zeros((H, W))
    for p in sample[d][:500]:
        g = np.asarray(load(p, (W, H)).convert("L"), dtype=np.float32)
        acc += np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))))
    ax.imshow(acc / 500, cmap="viridis")
    ax.set_title(f"{d} log spectrum", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
'''),
    md("""
## 16. What the training augmentations produce

The first row is one image through the training transform used for every run (flip, pad and crop, brightness
and contrast jitter, random erasing). The second row adds the camera degradation tried in the camaug run.
The third row shows what hue jitter would do to a red top, which is why it is not used.
"""),
    code('''
mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
undo = lambda t: np.clip(t.permute(1, 2, 0).numpy() * np.array(std) + np.array(mean), 0, 1)
red = positives("UpperBody-Color-Red")[0]
base = load(red, (W, H))
basic = make_transforms(mean, std, True, "basic")
camera = make_transforms(mean, std, True, "camera")
torch.manual_seed(0)
random.seed(0)
fig, axes = plt.subplots(3, 7, figsize=(12, 10))
show_row(axes[0], [base] + [undo(basic(base)) for _ in range(6)], ["original"] + ["basic"] * 6)
show_row(axes[1], [base] + [undo(camera(base)) for _ in range(6)], ["original"] + ["camera"] * 6)
hsv = np.asarray(base.convert("HSV")).copy()
hue = []
for shift in (0, 20, 45, 85, 130, 170, 210):
    h = hsv.copy()
    h[..., 0] = (h[..., 0].astype(int) + shift) % 256
    hue.append(Image.fromarray(h, "HSV").convert("RGB"))
show_row(axes[2], hue, [f"hue +{s}" for s in (0, 20, 45, 85, 130, 170, 210)])
plt.tight_layout()
plt.show()
'''),
    md("""
## 17. Hardest-looking images

Per dataset: the darkest, the blurriest (lowest Laplacian variance) and the most colour-cast crops in the sample.
"""),
    code('''
fig, axes = plt.subplots(len(DATASETS), 9, figsize=(13, 4.2 * len(DATASETS)))
for r, d in enumerate(DATASETS):
    a = app[app.dataset == d].reset_index(drop=True)
    ps = sample[d][:1500]
    dark = a.nsmallest(3, "brightness").index
    blur = a.nsmallest(3, "sharpness").index
    rgb = np.asarray([np.asarray(load(ps[i], (32, 64)), dtype=np.float32).reshape(-1, 3).mean(0) for i in range(400)])
    tint = np.argsort(-(np.abs(rgb[:, 0] - rgb[:, 1]) + np.abs(rgb[:, 2] - rgb[:, 1])))[:3]
    imgs = [load(ps[i], (W, H)) for i in list(dark) + list(blur) + list(tint)]
    show_row(axes[r], imgs, ["dark"] * 3 + ["blurry"] * 3 + ["colour cast"] * 3)
    axes[r, 0].text(-15, H / 2, d, rotation=90, va="center", ha="right", fontsize=10)
plt.tight_layout()
plt.show()
'''),
    md("""
## What this changed in the approach

| Observation | Decision |
|---|---|
| PA-100K is about 81% of the training images, Market1501 and PETA about 10% and 9% | dataset-balanced sampling (weights from the square root of the inverse dataset size), so the two small sources are seen more often without being repeated excessively |
| 21 of 40 attributes are positive in under 5% of training images, 7 in under 1% (purple trousers 0.17%), and items like sunglasses or a hat cover less than one 16x16 patch (sections 12, 14) | weighted BCE (Li et al. 2015) so rare positives count more; mA, half of the score, punishes a model that never predicts them; per-attribute decision offsets tuned on validation |
| five colour attributes are never positive in Market1501, one in PETA | scored per dataset and only on the attributes present there, which is how the challenge metric is computed |
| crops are tall and narrow: Market1501 is always 128x64, PA-100K has a median of 201x74 and PETA 160x74, and 75-100% of crops are shorter than 256 px (sections 5 and 8) | one fixed 256x128 (2:1) input instead of a square one, so most images are upscaled rather than squashed; ViT position embeddings resampled to that 16x8 grid |
| no attribute depends on left vs right | horizontal flip is safe as augmentation and as test-time averaging |
| 24 of 40 attributes are colours, and a hue shift turns a red top into an orange, green or purple one (section 16) | mild brightness/contrast jitter only, no hue/saturation change, no greyscale |
| PA-100K is far sharper than the other two (median Laplacian variance 176 vs 48 and 46), Market1501 is the darkest, PETA has visible colour casts, and some PA-100K images are greyscale night shots that still carry colour labels (sections 6, 9, 10, 15, 17) | the hidden camera will differ again, so we leaned on pretrained backbones that have seen very varied images (CLIP, DINOv2) and tried a camera degradation augmentation (blur, downscaling, JPEG) to imitate the softer sources |
| validation is half Market1501, while training is 81% PA-100K; PA-100K's validation split also has fewer short sleeves (44% vs 59%) | model selection uses the challenge formula (score per source, then averaged) so no single source decides; train and val images never overlap |
| strong pairs such as trousers vs skirt (phi -0.98) and female vs long hair (0.63) | one shared 40-way head on a shared backbone, so these dependencies are learned jointly rather than per attribute |
"""),
    code('''
if PUSH_RESULTS and TOKEN:
    !python {CODE}/tools/push_results.py --repo {CODE} --run-dir {OUT} --name data_exploration --message "data exploration"
!ls -lh {OUT}
'''),
], gpu=False)

#01 download
nb1 = notebook([
    md("""
# 01 Data download

Accelerator: None, Internet: On. Run with Save Version > Save & Run All.

Downloads Market1501, PA-100K and PETA with the official script and saves them as `upar_data.tar`,
which notebooks 02 and 03 use as input. If a Google Drive download fails because of the quota, run it again later.
"""),
    code('''
!git clone -q https://github.com/speckean/UPAR-Challenge-2027.git /tmp/repo
!pip install -q gdown
%cd /tmp/repo
!python download_datasets.py --data-dir /tmp/data
'''),
    code('''
import csv, os, collections
missing = collections.Counter()
total = collections.Counter()
for split in ("train", "val"):
    rows = list(csv.reader(open(f"/tmp/repo/data/annotations/task1/{split}/gt.csv")))[1:]
    for r in rows:
        d = r[0].split("/")[0]
        total[d] += 1
        if not os.path.exists(os.path.join("/tmp/data", r[0])):
            missing[d] += 1
print("images:", dict(total))
print("missing:", dict(missing))
assert sum(missing.values()) == 0, "some images are missing, run the download again"
'''),
    code('''
!cp -r /tmp/repo/data/annotations /tmp/data/annotations
!cd /tmp/data && tar -cf /kaggle/working/upar_data.tar annotations Market1501/bounding_box_train Market1501/bounding_box_test Market1501/query PA100k/release_data PETA/images
!ls -lh /kaggle/working/
'''),
], gpu=False)

#02 train
nb2 = notebook([
    md("""
# 02 Training

Accelerator: GPU T4 x2, Internet: On, secret `GITHUB_TOKEN` attached. Input: output of notebook 01.

Run long jobs with Save Version > Save & Run All. `last.pth` is saved every 20 minutes and after each epoch,
and training stops after `TIME_BUDGET_H` hours (Kaggle ends sessions at 12 h). To continue a stopped run,
add that version of this notebook as an input and run it again.

Result files (csv, json, png, logs) are pushed to `results/<RUN_NAME>/` in the repo. Weights stay in the notebook output.
"""),
    code(REPO_CFG + '''
BACKBONE = "clip_vitb16"    # or dinov2_vitb14 / vit_in21k / siglip_vitb16 / convnext_small / resnet50
EPOCHS = 10
BATCH = 64
LR = 2e-4                   # 1e-4 dinov2, 4e-4 convnext, 1e-3 resnet50
TRAIN_ON_VAL = False        # final model on train+val
AUG = "basic"               # or "camera": adds low-res / blur / JPEG degradation
RUN_NAME = f"run_{BACKBONE}" + ("_camaug" if AUG == "camera" else "") + ("_trainval" if TRAIN_ON_VAL else "")
TIME_BUDGET_H = 11.0
SMOKE_TEST = True           # one epoch on 640 images before the full run
PUSH_RESULTS = True
'''),
    code(CLONE),
    code("!pip install -q timm==1.0.15\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code(FIND_TAR),
    code('''
#copy the checkpoint of a previous version if it was added as input
OUT = f"/kaggle/working/{RUN_NAME}"
os.makedirs(OUT, exist_ok=True)
prev = sorted(glob.glob(f"/kaggle/input/**/{RUN_NAME}/last.pth", recursive=True))
if prev and not os.path.exists(f"{OUT}/last.pth"):
    src = os.path.dirname(prev[-1])
    for f in os.listdir(src):
        if os.path.isfile(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), OUT)
    print("copied checkpoint from", src)
RESUMING = os.path.exists(f"{OUT}/last.pth")
print("resume" if RESUMING else "new run", OUT)
'''),
    code('''
if SMOKE_TEST and not RESUMING:
    !cd {CODE} && python src/train.py --data-root /tmp/data --backbone {BACKBONE} --epochs 1 --limit 640 --batch-size 32 --workers 4 --out /tmp/smoke --resume none
'''),
    code('''
#run through Popen so the exit code is visible: -9 means the process was killed (usually out of RAM)
WORKERS = 4
flags = "--train-on-val" if TRAIN_ON_VAL else ""
#convnext leaks host memory and slows down 20x when the batch is split over two gpus, one gpu works
if BACKBONE.startswith("convnext"):
    flags += " --single-gpu"
cmd = (f"cd {CODE} && python src/train.py --data-root /tmp/data --backbone {BACKBONE} --epochs {EPOCHS} "
       f"--batch-size {BATCH} --lr {LR} --aug {AUG} --workers {WORKERS} --time-budget-hours {TIME_BUDGET_H} --out {OUT} {flags}")
with open(f"{OUT}/train.log", "a") as log:
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    proc.wait()
print("exit code", proc.returncode)
!free -g
'''),
    code('''
import pandas as pd
import matplotlib.pyplot as plt

if os.path.exists(f"{OUT}/DONE"):
    STATUS = "done"
elif os.path.exists(f"{OUT}/PAUSED"):
    STATUS = "paused"
else:
    STATUS = "stopped"
print("status:", STATUS)

h = pd.read_csv(f"{OUT}/history.csv") if os.path.exists(f"{OUT}/history.csv") else pd.DataFrame()
display(h)
if os.path.exists(f"{OUT}/results_summary.csv"):
    display(pd.read_csv(f"{OUT}/results_summary.csv"))
if "val_Challenge_Avg" in h and len(h):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, marker="o")
    ax[0].set_title("train loss")
    ax[0].set_xlabel("epoch")
    for k in ["val_Challenge_Avg", "val_mA", "val_Instance_F1"]:
        ax[1].plot(h.epoch, h[k], marker="o", label=k[4:])
    ax[1].set_title("validation")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/curves.png", dpi=150)
    plt.show()
!ls -lh {OUT}
'''),
    code('''
if PUSH_RESULTS and TOKEN:
    !python {CODE}/tools/push_results.py --repo {CODE} --run-dir {OUT} --name {RUN_NAME} --message "train {RUN_NAME}: {STATUS}"
'''),
], gpu=True)

#03 export
nb3 = notebook([
    md("""
# 03 Export

Accelerator: GPU T4 (or None, the export also works on CPU), Internet: On, secret `GITHUB_TOKEN` attached.
Inputs: notebook 01 (data) and the training notebook(s). For a train+val model the offsets are taken from the train-only run in `results/`.

For an ensemble, list the run folders in `ENSEMBLE` and attach every training notebook that holds one of them
(a notebook can only be attached once, so each member has to come from a different notebook or version).
Its offsets are tuned on the averaged validation logits with `tools/eval_ensemble.py` and stored in `results/`.

Kaggle's Python 3.13 has no torch 2.4.1 build, so the export uses the installed torch; TorchScript files saved this way load in 2.4.1.
The zip is written to the notebook output.
"""),
    code('!pip install -q timm==1.0.15'),
    code(REPO_CFG + '''
PUSH_RESULTS = True
ENSEMBLE = []               # e.g. ["run_clip_vitb16", "run_dinov2_vitb14"]
ENSEMBLE_OFFSETS = "results/ensemble_clip_dinov2/offsets.json"
'''),
    code(CLONE),
    code(FIND_TAR),
    code('''
if ENSEMBLE:
    found = [glob.glob(f"/kaggle/input/**/{r}/best.pth", recursive=True) for r in ENSEMBLE]
    for r, f in zip(ENSEMBLE, found):
        assert f, f"{r}/best.pth not found, attach the notebook version that trained it"
    CKPT = " ".join(f[0] for f in found)
    OFFSETS = f"{CODE}/{ENSEMBLE_OFFSETS}"
    SUB_NAME = "submission_ensemble_" + "_".join(r.replace("run_", "") for r in ENSEMBLE)
    SUB_NAME += os.path.splitext(os.path.basename(ENSEMBLE_OFFSETS))[0].replace("offsets", "")
    print("using", CKPT, OFFSETS)
else:
    cks = sorted(glob.glob("/kaggle/input/**/final.pth", recursive=True)) + sorted(glob.glob("/kaggle/input/**/best.pth", recursive=True))
    #offsets always come from a train-only run, either an attached 02 version or results/ in the repo
    offs = sorted(p for p in glob.glob("/kaggle/input/**/offsets.json", recursive=True) if "trainval" not in p)
    offs += sorted(p for p in glob.glob(f"{CODE}/results/run_*/offsets.json") if "trainval" not in p)
    print("checkpoints:", cks)
    print("offsets:", offs)
    CKPT = cks[0]                       # final.pth if there is one, otherwise best.pth
    OFFSETS = offs[0] if offs else ""
    SUB_NAME = "submission_" + os.path.basename(os.path.dirname(CKPT))
    print("using", CKPT, OFFSETS)
'''),
    code('''
OUTD = f"/kaggle/working/{SUB_NAME}"
os.makedirs(OUTD, exist_ok=True)
off_flag = f"--offsets {OFFSETS}" if OFFSETS else ""
#two fp16 models make a ~320 MB zip, which Codabench did not accept, int8 weights keep it near 160 MB
int8_flag = "--int8" if ENSEMBLE else ""
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
!cd {CODE} && python src/export_submission.py --ckpt {CKPT} {off_flag} {int8_flag} --data-root /tmp/data --out-dir /tmp/build --zip {OUTD}/upar_track1_submission.zip 2>&1 | tee {OUTD}/export.log
'''),
    code('''
if PUSH_RESULTS and TOKEN:
    !python {CODE}/tools/push_results.py --repo {CODE} --run-dir {OUTD} --name {SUB_NAME} --message "export {SUB_NAME}"
!ls -lh {OUTD}
'''),
], gpu=True)

#04 results analysis
nb4 = notebook([
    md("""
# 04 Results analysis

Accelerator: None, Internet: On, secret `GITHUB_TOKEN` attached.
Inputs: notebook 01 (images), the `clip-vitb16-baseline` dataset and `02_train_dinov2` (their `val_logits.npz`),
and `03_export_ensemble` (the final submission zip).

All runs and submissions in one place: what was tried, how it scored on validation and on the hidden test set,
which attributes fail and why, what the ensemble and the decision shift change, the final model's worst mistakes
with their images, and a demo of the submitted zip.
"""),
    code(REPO_CFG + '''
PUSH_RESULTS = True
'''),
    code(CLONE),
    code(FIND_TAR),
    code('''
import sys, json, random, zipfile, importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, f"{CODE}/src")
sys.path.insert(0, f"{CODE}/tools")
from common import ATTRIBUTES, dataset_of, leaderboard_metrics, pedestrian_metrics, valid_index
from eval_ensemble import fast_calibrate, two_fold

RES = f"{CODE}/results"
OUT = "/kaggle/working/results_analysis"
os.makedirs(OUT, exist_ok=True)
DATA = "/tmp/data"

def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f"not found: {pattern}"
    return hits[0]

clip_npz = np.load(find("/kaggle/input/**/run_clip_vitb16/val_logits.npz"), allow_pickle=True)
dino_npz = np.load(find("/kaggle/input/**/run_dinov2_vitb14/val_logits.npz"), allow_pickle=True)
paths = np.asarray([str(p) for p in clip_npz["paths"]])
assert (paths == np.asarray([str(p) for p in dino_npz["paths"]])).all()
gt = clip_npz["gt"].astype(int)
ds = np.asarray([dataset_of(p) for p in paths])
DATASETS = sorted(set(ds))
print(len(paths), "validation images", {d: int((ds == d).sum()) for d in DATASETS})
'''),
    md("""
## 1. Every run and submission

Validation numbers come from each run's `results/` folder. Hidden-test numbers are copied from the Codabench
leaderboard (they cannot be recomputed locally).
"""),
    code('''
runs = {"run_resnet50": "ResNet-50 (ImageNet-1k)", "run_vit_in21k": "ViT-B/16 (ImageNet-21k)",
        "run_dinov2_vitb14": "DINOv2 ViT-B/14", "run_clip_vitb16": "CLIP ViT-B/16",
        "run_clip_vitb16_camaug": "CLIP ViT-B/16 + camera aug", "run_convnext_small": "ConvNeXt-S (ImageNet-22k)"}
rows = []
for r, name in runs.items():
    if not os.path.exists(f"{RES}/{r}/results_summary.csv"):
        continue
    s = pd.read_csv(f"{RES}/{r}/results_summary.csv")
    s = s[s.exclude_absent].set_index("variant")
    h = pd.read_csv(f"{RES}/{r}/history.csv")
    cfg = json.load(open(f"{RES}/{r}/config.json"))
    rows.append({"model": name, "epochs": cfg["epochs"], "lr": cfg["lr"],
                 "best_epoch": int(h.val_Challenge_Avg.idxmax()) + 1, "min/epoch": round(h.minutes.mean(), 1),
                 "val plain": s.loc["plain", "Challenge_Avg"], "val flip": s.loc["flip", "Challenge_Avg"],
                 "val flip+offsets": s.loc["flip + offsets", "Challenge_Avg"],
                 "Market1501": s.loc["flip + offsets", "Market1501_Challenge_Avg"],
                 "PA100k": s.loc["flip + offsets", "PA100k_Challenge_Avg"],
                 "PETA": s.loc["flip + offsets", "PETA_Challenge_Avg"]})
val_table = pd.DataFrame(rows)
val_table.to_csv(f"{OUT}/validation_runs.csv", index=False)
display(val_table)

test_table = pd.DataFrame([
    ["sub02", "CLIP ViT-B/16", 0.7766, 0.7787, 0.5766, 0.7469, 0.8186, 0.7746],
    ["sub03", "CLIP refit on train+val", 0.7679, 0.7656, 0.5770, 0.7501, 0.8041, 0.7702],
    ["sub04", "CLIP + camera aug", 0.7701, 0.7719, 0.5768, 0.7438, 0.8085, 0.7684],
    ["sub05", "DINOv2 ViT-B/14", 0.7733, 0.7727, 0.5887, 0.7553, 0.8068, 0.7738],
    ["sub06", "CLIP + DINOv2 fp16 (318 MB, rejected by the platform)", None, None, None, None, None, None],
    ["sub07", "CLIP + DINOv2, int8 weights", 0.7816, 0.7795, 0.5959, 0.7642, 0.8172, 0.7836],
    ["sub08", "sub07 + shift 0.5 (final)", 0.7833, 0.7873, 0.5907, 0.7401, 0.8385, 0.7794],
    ["#1", "best leaderboard entry", 0.8021, 0.8098, 0.5938, 0.7591, 0.8483, 0.7945],
], columns=["submission", "model", "score", "mA", "label_F1", "inst_prec", "inst_rec", "inst_F1"])
test_table.to_csv(f"{OUT}/test_submissions.csv", index=False)
display(test_table)
'''),
    md("""
## 2. Learning curves
"""),
    code('''
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for r, name in runs.items():
    if not os.path.exists(f"{RES}/{r}/history.csv"):
        continue
    h = pd.read_csv(f"{RES}/{r}/history.csv")
    ax[0].plot(h.epoch, h.train_loss, marker="o", label=name)
    ax[1].plot(h.epoch, h.val_Challenge_Avg, marker="o", label=name)
ax[0].set_title("training loss")
ax[1].set_title("validation Challenge score")
ax[1].set_ylim(0.7, 0.89)
for a in ax:
    a.set_xlabel("epoch")
    a.grid(alpha=0.3)
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/learning_curves.png", dpi=130)
plt.show()
'''),
    md("""
## 3. Per attribute

Label mA per attribute (averaged over the sources that use it) for CLIP, DINOv2 and their ensemble, next to the
training positive rate from notebook 00.
"""),
    code('''
off = np.asarray(json.load(open(f"{RES}/ensemble_clip_dinov2/offsets.json"))["offsets"])
off_shift = np.asarray(json.load(open(f"{RES}/ensemble_clip_dinov2/offsets_shift050.json"))["offsets"])
z_clip, z_dino = clip_npz["logits"].astype(float), dino_npz["logits"].astype(float)
z_ens = 0.5 * z_clip + 0.5 * z_dino
clip_off = np.asarray(json.load(open(f"{RES}/run_clip_vitb16/offsets.json"))["offsets"])
dino_off = np.asarray(json.load(open(f"{RES}/run_dinov2_vitb14/offsets.json"))["offsets"])
preds = {"CLIP": z_clip + clip_off > 0, "DINOv2": z_dino + dino_off > 0, "ensemble": z_ens + off > 0,
         "ensemble + shift": z_ens + off_shift > 0}

def per_attribute_ma(pred):
    acc = {a: [] for a in ATTRIBUTES}
    for d in DATASETS:
        m = ds == d
        idx = valid_index(gt[m])
        r = pedestrian_metrics(gt[m], pred[m], idx)
        for k, j in enumerate(idx):
            acc[ATTRIBUTES[j]].append(r["label_ma"][k])
    return {a: float(np.mean(v)) for a, v in acc.items()}

rate = pd.read_csv(f"{RES}/data_exploration/positive_rate_train.csv", index_col=0)["all"]
attr = pd.DataFrame({k: per_attribute_ma(v) for k, v in preds.items()})
attr["train_pos_rate"] = rate.reindex(attr.index)
attr["dino_minus_clip"] = attr.DINOv2 - attr.CLIP
attr = attr.sort_values("ensemble")
attr.round(4).to_csv(f"{OUT}/per_attribute.csv")
print("weakest attributes for the ensemble:")
display(attr.head(10).round(3))
print("largest CLIP / DINOv2 differences:")
display(attr.sort_values("dino_minus_clip").iloc[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]][["CLIP", "DINOv2", "dino_minus_clip"]].round(3))

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(attr.train_pos_rate, attr.CLIP, label="CLIP", s=22)
ax.scatter(attr.train_pos_rate, attr.DINOv2, label="DINOv2", s=22, marker="s")
ax.scatter(attr.train_pos_rate, attr.ensemble, label="ensemble", s=26, marker="^")
for a in attr.index[:6]:
    ax.annotate(a, (attr.train_pos_rate[a], attr.ensemble[a]), fontsize=7, xytext=(4, -3), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("training positive rate")
ax.set_ylabel("label mA (val)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/per_attribute.png", dpi=130)
plt.show()
'''),
    md("""
## 4. Per dataset and the ensemble

Scores per source dataset for each model, then the ensemble comparison with offsets fitted on one half of
validation and scored on the other (2-fold), which is the fair comparison.
"""),
    code('''
rows = []
for name, p in preds.items():
    m = leaderboard_metrics(gt, p, ds)
    row = {"model": name, "score": m["Challenge_Avg"], "mA": m["mA"], "inst_F1": m["Instance_F1"],
           "precision": m["Instance_Prec"], "recall": m["Instance_Rec"]}
    row.update({d: m["per_dataset"][d]["Challenge_Avg"] for d in DATASETS})
    rows.append(row)
per_ds = pd.DataFrame(rows).round(4)
per_ds.to_csv(f"{OUT}/per_dataset.csv", index=False)
display(per_ds)

ens_rows = []
for name, z in [("CLIP", z_clip), ("DINOv2", z_dino), ("CLIP + DINOv2", z_ens)]:
    ens_rows.append({"model": name, "no offsets": leaderboard_metrics(gt, z > 0, ds)["Challenge_Avg"],
                     "offsets, 2-fold": two_fold(z, gt, ds)})
display(pd.DataFrame(ens_rows).round(4))
'''),
    md("""
## 5. The decision shift: validation against the hidden test set

A constant added to every logit of the ensemble, on top of the fitted offsets.
"""),
    code('''
shift_rows = []
for delta in [0, 0.25, 0.5, 0.75, 1.0]:
    m = leaderboard_metrics(gt, z_ens + off + delta > 0, ds)
    shift_rows.append({"shift": delta, "score": m["Challenge_Avg"], "mA": m["mA"], "inst_F1": m["Instance_F1"],
                       "precision": m["Instance_Prec"], "recall": m["Instance_Rec"]})
shift_val = pd.DataFrame(shift_rows).round(4)
display(shift_val)
test_pts = test_table.set_index("submission").loc[["sub07", "sub08"]]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for k in ["score", "mA", "inst_F1"]:
    ax[0].plot(shift_val["shift"], shift_val[k] - shift_val[k][0], marker="o", label=k)
for k, col in [("score", "score"), ("mA", "mA"), ("inst_F1", "inst_F1")]:
    v = test_pts[col].values
    ax[1].plot([0, 0.5], v - v[0], marker="o", label=k)
ax[0].set_title("validation")
ax[1].set_title("hidden test (sub07 vs sub08)")
for a in ax:
    a.axhline(0, color="grey", lw=0.8)
    a.set_xlabel("shift")
    a.grid(alpha=0.3)
ax[0].set_ylabel("change vs. no shift")
ax[0].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/shift.png", dpi=130)
plt.show()
'''),
    md("""
## 6. Where the final model goes wrong

Predictions of the final rule (ensemble + offsets + shift) on validation. First the attributes it most often
predicts wrongly, then the validation images with the lowest per-image F1 in each dataset, with the attributes it
added (false positives) and missed (false negatives).
"""),
    code('''
final = preds["ensemble + shift"]
fp = (final & (gt == 0)).sum(0)
fn = (~final & (gt == 1)).sum(0)
err = pd.DataFrame({"false_pos": fp, "false_neg": fn, "positives": gt.sum(0)}, index=ATTRIBUTES)
err["fn_share_of_positives"] = (err.false_neg / err.positives.clip(lower=1)).round(3)
print("most false positives:")
display(err.sort_values("false_pos", ascending=False).head(8))
print("most missed positives (relative to how many exist):")
display(err[err.positives >= 50].sort_values("fn_share_of_positives", ascending=False).head(8))
err.to_csv(f"{OUT}/errors_per_attribute.csv")

inter = (final & (gt == 1)).sum(1)
prec = inter / np.maximum(final.sum(1), 1)
rec = inter / np.maximum(gt.sum(1), 1)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
short = lambda a: a.replace("UpperBody-", "U-").replace("LowerBody-", "L-").replace("Accessory-", "").replace("Color-", "")

fig, axes = plt.subplots(len(DATASETS), 6, figsize=(14, 5.2 * len(DATASETS)))
for r, d in enumerate(DATASETS):
    idx = np.flatnonzero(ds == d)
    worst = idx[np.argsort(f1[idx])[:6]]
    for c, i in enumerate(worst):
        with Image.open(os.path.join(DATA, paths[i])) as im:
            axes[r, c].imshow(im.convert("RGB").resize((128, 256)))
        added = [short(ATTRIBUTES[j]) for j in np.flatnonzero(final[i] & (gt[i] == 0))][:4]
        missed = [short(ATTRIBUTES[j]) for j in np.flatnonzero(~final[i] & (gt[i] == 1))][:4]
        axes[r, c].set_title(f"F1 {f1[i]:.2f}\\n+ " + ", ".join(added) + "\\n- " + ", ".join(missed), fontsize=7)
        axes[r, c].axis("off")
    axes[r, 0].text(-20, 128, d, rotation=90, va="center", ha="right", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/worst_examples.jpg", dpi=100)
plt.show()
'''),
    md("""
## 7. Demo: the submitted model

Loads `run.py` and the model from the final submission zip exactly as the challenge server does, and predicts
attributes for a few random validation images. Shown are the attributes predicted as present (probability above
0.5), marked against the ground truth.
"""),
    code('''
zip_path = find("/kaggle/input/**/*shift050*/upar_track1_submission.zip")
demo_dir = "/tmp/final_submission"
shutil.rmtree(demo_dir, ignore_errors=True)
zipfile.ZipFile(zip_path).extractall(demo_dir)
spec = importlib.util.spec_from_file_location("submission_run", f"{demo_dir}/run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
run.load_model()

random.seed(7)
pick = random.sample(range(len(paths)), 6)
samples = [{"index": k, "image": paths[i], "image_path": os.path.join(DATA, paths[i]), "attribute_names": ATTRIBUTES}
           for k, i in enumerate(pick)]
probs = np.asarray(run.predict_batch(samples))

fig, axes = plt.subplots(1, 6, figsize=(16, 6))
for ax, s, p, i in zip(axes, samples, probs, pick):
    with Image.open(s["image_path"]) as im:
        ax.imshow(im.convert("RGB").resize((128, 256)))
    lines = []
    for j in np.argsort(-p):
        if p[j] < 0.5:
            break
        mark = "ok" if gt[i, j] == 1 else "x"
        lines.append(f"{short(ATTRIBUTES[j])} {p[j]:.2f} {mark}")
    missed = [short(ATTRIBUTES[j]) for j in np.flatnonzero((gt[i] == 1) & (p < 0.5))]
    if missed:
        lines.append("missed: " + ", ".join(missed))
    ax.set_title("\\n".join(lines), fontsize=7, loc="left")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/demo.jpg", dpi=100)
plt.show()
'''),
    code('''
if PUSH_RESULTS and TOKEN:
    !python {CODE}/tools/push_results.py --repo {CODE} --run-dir {OUT} --name results_analysis --message "results analysis"
!ls -lh {OUT}
'''),
], gpu=False)

os.makedirs(OUT, exist_ok=True)
for name, nb in [("00_data_exploration.ipynb", nb0), ("01_download_data.ipynb", nb1), ("02_train.ipynb", nb2), ("03_export_and_package.ipynb", nb3), ("04_results_analysis.ipynb", nb4)]:
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", name)