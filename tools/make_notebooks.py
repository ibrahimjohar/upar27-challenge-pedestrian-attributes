"""Generates notebooks/01-03 (run from the repo root: python tools/make_notebooks.py).

The notebooks contain no model code: they clone this GitHub repo on Kaggle, so a fix
pushed to GitHub is picked up by the next run.
"""
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
GITHUB_REPO = "YOUR_GITHUB_USERNAME/upar-track1"   # <- edit once: owner/name of your repo
BRANCH = "main"
'''

CLONE = '''
# Clone the code from GitHub. Private repo: attach the Kaggle secret GITHUB_TOKEN
# (Add-ons -> Secrets). The token is only used inside /tmp/code (never saved to the output).
import os, subprocess, shutil
assert "YOUR_GITHUB_USERNAME" not in GITHUB_REPO, "Set GITHUB_REPO in the first code cell"
TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    print("No GITHUB_TOKEN secret attached: works for a public repo, but results will not be pushed.")
owner = GITHUB_REPO.split("/")[0]
url = f"https://{owner}:{TOKEN}@github.com/{GITHUB_REPO}.git" if TOKEN else f"https://github.com/{GITHUB_REPO}.git"
CODE = "/tmp/code"
shutil.rmtree(CODE, ignore_errors=True)
r = subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, CODE], capture_output=True, text=True)
if r.returncode:
    raise RuntimeError("git clone failed: " + (r.stderr.replace(TOKEN, "***") if TOKEN else r.stderr))
print("code version:", subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%h  %s  (%cr)"],
                                      capture_output=True, text=True).stdout)
'''

FIND_TAR = '''
import glob, time
tars = glob.glob("/kaggle/input/**/upar_data.tar", recursive=True)
assert tars, "Add the output of notebook 01 as input (Add Input -> Your Work -> Notebooks)"
if not os.path.exists("/tmp/data/annotations/task1/train/gt.csv"):
    t0 = time.time()
    os.makedirs("/tmp/data", exist_ok=True)
    subprocess.run(["tar", "-xf", tars[0], "-C", "/tmp/data"], check=True)
    print(f"extracted {tars[0]} in {time.time() - t0:.0f}s")
!du -sh /tmp/data
'''

# ------------------------------------------------------------------ 01 download
nb1 = notebook([
    md("""
# 01 - Download the UPAR 2027 data (run once)
**Settings (right panel):** Accelerator **None**, Internet **On**.
Then **Save Version -> Save & Run All** (runs on Kaggle's servers; you can close the tab).
The output `upar_data.tar` is the input for notebooks 02 and 03.

If Google Drive refuses a download ("too many users"), wait an hour and re-run.
"""),
    code('''
!git clone -q https://github.com/speckean/UPAR-Challenge-2027.git /tmp/repo
!pip install -q gdown
%cd /tmp/repo
!python download_datasets.py --data-dir /tmp/data
'''),
    code('''
# check every annotated image exists
import csv, os, collections
missing = collections.Counter(); total = collections.Counter()
for split in ("train", "val"):
    rows = list(csv.reader(open(f"/tmp/repo/data/annotations/task1/{split}/gt.csv")))[1:]
    for r in rows:
        d = r[0].split("/")[0]; total[d] += 1
        if not os.path.exists(os.path.join("/tmp/data", r[0])):
            missing[d] += 1
print("images per dataset:", dict(total)); print("missing:", dict(missing))
assert sum(missing.values()) == 0, "some images are missing - re-run later (Google Drive quota)"
'''),
    code('''
!cp -r /tmp/repo/data/annotations /tmp/data/annotations
!cd /tmp/data && tar -cf /kaggle/working/upar_data.tar annotations Market1501/bounding_box_train Market1501/bounding_box_test Market1501/query PA100k/release_data PETA/images
!ls -lh /kaggle/working/
'''),
], gpu=False)

# ------------------------------------------------------------------ 02 train
nb2 = notebook([
    md("""
# 02 - Train (UPAR 2027 Track 1), crash-safe
**Settings:** Accelerator **GPU T4 x2**, Internet **On**, Add-ons -> Secrets -> attach **GITHUB_TOKEN**.
**Inputs:** output of notebook 01. To *resume*, also add the output of a previous version of this notebook.

* Long runs: use **Save Version -> Save & Run All**. It runs on Kaggle's servers, so your laptop,
  browser or internet can disconnect without affecting it.
* A full training checkpoint (`last.pth`) is written every 20 min and after every epoch.
* The run pauses itself cleanly after `TIME_BUDGET_H` hours (Kaggle stops sessions at 12 h).
* **To continue a paused or crashed run:** Add Input -> Your Work -> Notebooks -> this notebook
  (the version that stopped), keep the same settings, and Save & Run All again. It finds `last.pth`
  and continues from the saved epoch and step.
* Small result files (history, metrics, tables, curves, logs) are pushed to `results/<RUN_NAME>/`
  in your GitHub repo. Weights stay in this notebook's Output (too big for GitHub).
"""),
    code(REPO_CFG + '''
BACKBONE = "clip_vitb16"    # or "convnext_tiny" (smaller zip), "convnext_small"
EPOCHS = 10
BATCH = 64
LR = 2e-4                   # convnext: 4e-4 is a reasonable start
TRAIN_ON_VAL = False        # True only for the final model (export it with offsets from a train-only run)
RUN_NAME = f"run_{BACKBONE}" + ("_trainval" if TRAIN_ON_VAL else "")
TIME_BUDGET_H = 11.0        # pause cleanly before Kaggle's 12 h limit
SMOKE_TEST = True           # 1 quick epoch on 640 images first (skipped when resuming)
PUSH_RESULTS = True
'''),
    code(CLONE),
    code("!pip install -q timm==1.0.15\n!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    code(FIND_TAR),
    code('''
# Resume: bring in the checkpoint of a previous version of this notebook, if one was added as input
OUT = f"/kaggle/working/{RUN_NAME}"
os.makedirs(OUT, exist_ok=True)
prev = sorted(glob.glob(f"/kaggle/input/**/{RUN_NAME}/last.pth", recursive=True))
if prev and not os.path.exists(f"{OUT}/last.pth"):
    src = os.path.dirname(prev[-1])
    for f in os.listdir(src):
        if os.path.isfile(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), OUT)
    print("will RESUME from", src)
RESUMING = os.path.exists(f"{OUT}/last.pth")
print("resuming" if RESUMING else "fresh run", "->", OUT)
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
import pandas as pd, matplotlib.pyplot as plt
state = "DONE" if os.path.exists(f"{OUT}/DONE") else "PAUSED - add this version as input and run again" if os.path.exists(f"{OUT}/PAUSED") else "STOPPED EARLY - check train.log; re-run with this version as input to resume"
STATUS_WORD = state.split(" ")[0]
print("STATUS:", state)
h = pd.read_csv(f"{OUT}/history.csv") if os.path.exists(f"{OUT}/history.csv") else pd.DataFrame()
display(h)
if os.path.exists(f"{OUT}/results_summary.csv"):
    display(pd.read_csv(f"{OUT}/results_summary.csv"))
if "val_Challenge_Avg" in h and len(h):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss, marker="o"); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    for k in ["val_Challenge_Avg", "val_mA", "val_Instance_F1"]:
        ax[1].plot(h.epoch, h[k], marker="o", label=k[4:])
    ax[1].set_title("validation (mean over datasets)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=150); plt.show()
!ls -lh {OUT}
'''),
    code('''
if PUSH_RESULTS and TOKEN:
    !python {CODE}/tools/push_results.py --repo {CODE} --run-dir {OUT} --name {RUN_NAME} --message "train {RUN_NAME}: {STATUS_WORD}"
else:
    print("not pushing (PUSH_RESULTS off or no GITHUB_TOKEN)")
'''),
], gpu=True)

# ------------------------------------------------------------------ 03 export
nb3 = notebook([
    md("""
# 03 - Export, package and test the Codabench zip
**Settings:** Accelerator **GPU T4**, Internet **On**, Secrets -> attach **GITHUB_TOKEN**.
**Inputs:** outputs of notebook 01 (data, for the test) and notebook 02 (trained run;
for a train+val final model also add the train-only run, which supplies `offsets.json`).
Installs torch 2.4.1 = the grader's version before anything imports torch.
Download `upar_track1_submission.zip` from the Output panel and upload it under **My Submissions**.
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
print("checkpoints:", *cks, sep="\\n  "); print("offsets:", *offs, sep="\\n  ")
CKPT = cks[0]                       # final.pth (train+val) first if present, else best.pth
OFFSETS = offs[0] if offs else ""   # always from a train-only run
SUB_NAME = "submission_" + os.path.basename(os.path.dirname(CKPT))
print("\\nCKPT:", CKPT, "\\nOFFSETS:", OFFSETS)
'''),
    code('''
OUTD = f"/kaggle/working/{SUB_NAME}"
os.makedirs(OUTD, exist_ok=True)
off_flag = f"--offsets {OFFSETS}" if OFFSETS else ""
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
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
    print("wrote notebooks/" + name)
