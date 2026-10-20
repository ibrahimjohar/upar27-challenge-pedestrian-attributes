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
## What this changed in the approach

| Observation | Decision |
|---|---|
| PA-100K is about 81% of the training images, Market1501 and PETA about 10% and 9% | dataset-balanced sampling (weights from the square root of the inverse dataset size), so the two small sources are seen more often without being repeated excessively |
| 21 of 40 attributes are positive in under 5% of training images, 7 in under 1% (purple trousers 0.17%) | weighted BCE (Li et al. 2015) so rare positives count more; mA, half of the score, punishes a model that never predicts them |
| five colour attributes are never positive in Market1501, one in PETA | scored per dataset and only on the attributes present there, which is how the challenge metric is computed |
| crops are tall and narrow (about 2:1 or taller) and native sizes differ a lot between sources (section 5) | one fixed 256x128 (2:1) input instead of a square one; ViT position embeddings resampled to that 16x8 grid |
| no attribute depends on left vs right | horizontal flip is safe as augmentation and as test-time averaging |
| 24 of 40 attributes are colours | mild brightness/contrast jitter only, no hue/saturation change, no greyscale |
| brightness, contrast and sharpness differ clearly between the three sources | the hidden camera will differ again, so we leaned on pretrained backbones that have seen very varied images (CLIP, DINOv2) and tried a camera degradation augmentation |
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
flags = "--train-on-val" if TRAIN_ON_VAL else ""
!cd {CODE} && python src/train.py --data-root /tmp/data --backbone {BACKBONE} --epochs {EPOCHS} --batch-size {BATCH} --lr {LR} --aug {AUG} --workers 4 --time-budget-hours {TIME_BUDGET_H} --out {OUT} {flags} 2>&1 | tee -a {OUT}/train.log
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

os.makedirs(OUT, exist_ok=True)
for name, nb in [("00_data_exploration.ipynb", nb0), ("01_download_data.ipynb", nb1), ("02_train.ipynb", nb2), ("03_export_and_package.ipynb", nb3)]:
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", name)
