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
BACKBONE = "clip_vitb16"    # or convnext_tiny / convnext_small
EPOCHS = 10
BATCH = 64
LR = 2e-4                   # 4e-4 for convnext
TRAIN_ON_VAL = False        # final model on train+val
RUN_NAME = f"run_{BACKBONE}" + ("_trainval" if TRAIN_ON_VAL else "")
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
!cd {CODE} && python src/train.py --data-root /tmp/data --backbone {BACKBONE} --epochs {EPOCHS} --batch-size {BATCH} --lr {LR} --workers 4 --time-budget-hours {TIME_BUDGET_H} --out {OUT} {flags} 2>&1 | tee -a {OUT}/train.log
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

Accelerator: GPU T4, Internet: On, secret `GITHUB_TOKEN` attached.
Inputs: notebook 01 (data) and notebook 02 (trained run). For a train+val model, also add the train-only run for its `offsets.json`.

torch 2.4.1 is installed first to match the evaluation container. The zip is written to the notebook output.
"""),
    code('!pip install -q torch==2.4.1 torchvision==0.19.1 timm==1.0.15 "numpy<2"'),
    code(REPO_CFG + '''
PUSH_RESULTS = True
'''),
    code(CLONE),
    code(FIND_TAR),
    code('''
cks = sorted(glob.glob("/kaggle/input/**/final.pth", recursive=True) + glob.glob("/kaggle/input/**/best.pth", recursive=True))
offs = sorted(p for p in glob.glob("/kaggle/input/**/offsets.json", recursive=True) if "trainval" not in p)
print("checkpoints:", cks)
print("offsets:", offs)
CKPT = cks[0]                       # final.pth if there is one, otherwise best.pth
OFFSETS = offs[0] if offs else ""   # offsets always come from a train-only run
SUB_NAME = "submission_" + os.path.basename(os.path.dirname(CKPT))
print("using", CKPT, OFFSETS)
'''),
    code('''
OUTD = f"/kaggle/working/{SUB_NAME}"
os.makedirs(OUTD, exist_ok=True)
off_flag = f"--offsets {OFFSETS}" if OFFSETS else ""
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
!cd {CODE} && python src/export_submission.py --ckpt {CKPT} {off_flag} --data-root /tmp/data --out-dir /tmp/build --zip {OUTD}/upar_track1_submission.zip 2>&1 | tee {OUTD}/export.log
'''),
    code('''
if PUSH_RESULTS and TOKEN:
    !python {CODE}/tools/push_results.py --repo {CODE} --run-dir {OUTD} --name {SUB_NAME} --message "export {SUB_NAME}"
!ls -lh {OUTD}
'''),
], gpu=True)

os.makedirs(OUT, exist_ok=True)
for name, nb in [("01_download_data.ipynb", nb1), ("02_train.ipynb", nb2), ("03_export_and_package.ipynb", nb3)]:
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", name)
