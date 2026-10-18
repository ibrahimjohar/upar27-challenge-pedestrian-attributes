"""Copy the small result files of a run into <repo>/results/<run_name>/ and push them to GitHub.

Called from the Kaggle notebooks after training / export. The repo must have been cloned
with a token that can write (see notebooks). Weights (*.pth, *.pt, *.zip, *.npz) are never
pushed: GitHub rejects files > 100 MB and they stay in the Kaggle notebook output instead.

    python tools/push_results.py --repo /tmp/code --run-dir /kaggle/working/run_clip_vitb16 \
        --name run_clip_vitb16 --message "train: clip_vitb16 10 ep"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

KEEP_EXT = {".csv", ".json", ".png", ".log", ".txt", ".md", ".yaml"}
MAX_MB = 20
EXTRA_NAMES = {"DONE", "PAUSED"}


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        # never echo the remote URL (it contains the token)
        msg = (r.stderr or r.stdout).replace(git_remote_url(repo), "<remote>")
        raise RuntimeError(f"git {args[0]} failed: {msg.strip()[:500]}")
    return r


def git_remote_url(repo):
    r = subprocess.run(["git", "-C", repo, "remote", "get-url", "origin"], capture_output=True, text=True)
    return r.stdout.strip() or "\0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", required=True, help="folder name under results/")
    ap.add_argument("--message", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dst = os.path.join(args.repo, "results", args.name)
    os.makedirs(dst, exist_ok=True)
    copied = []
    for f in sorted(os.listdir(args.run_dir)):
        p = os.path.join(args.run_dir, f)
        ext = os.path.splitext(f)[1].lower()
        if not os.path.isfile(p) or not (ext in KEEP_EXT or f in EXTRA_NAMES):
            continue
        if os.path.getsize(p) > MAX_MB * 1e6:
            print(f"skip {f} (> {MAX_MB} MB)")
            continue
        shutil.copy2(p, os.path.join(dst, f))
        copied.append(f)
    print(f"copied to results/{args.name}: {copied}")
    if args.dry_run or not copied:
        return

    git(args.repo, "config", "user.name", "kaggle-runner")
    git(args.repo, "config", "user.email", "kaggle-runner@users.noreply.github.com")
    git(args.repo, "add", os.path.join("results", args.name))
    if not git(args.repo, "status", "--porcelain", check=False).stdout.strip():
        print("nothing new to commit")
        return
    git(args.repo, "commit", "-m", args.message or f"results: {args.name}")
    for attempt in range(3):
        r = git(args.repo, "push", "origin", "HEAD", check=False)
        if r.returncode == 0:
            print("pushed to GitHub")
            return
        git(args.repo, "pull", "--rebase", "origin", git(args.repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
            check=False)
    sys.exit("push failed (check the token has 'Contents: Read and write' on this repo)")


if __name__ == "__main__":
    main()
