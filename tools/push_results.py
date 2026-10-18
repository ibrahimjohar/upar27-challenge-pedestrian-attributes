"""Copies the small output files of a run to results/<name>/ in the repo and pushes them.

usage: python tools/push_results.py --repo /tmp/code --run-dir /kaggle/working/run_clip_vitb16 --name run_clip_vitb16

Checkpoints, zips and .npz files are skipped (GitHub rejects files over 100 MB).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

KEEP_EXT = {".csv", ".json", ".png", ".log", ".txt", ".md", ".yaml"}
KEEP_NAMES = {"DONE", "PAUSED"}
MAX_MB = 20


def remote_url(repo):
    r = subprocess.run(["git", "-C", repo, "remote", "get-url", "origin"], capture_output=True, text=True)
    return r.stdout.strip() or "\0"


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        #the remote url contains the token
        msg = (r.stderr or r.stdout).replace(remote_url(repo), "<remote>")
        raise RuntimeError(f"git {args[0]} failed: {msg.strip()[:500]}")
    return r


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
        if not os.path.isfile(p) or not (ext in KEEP_EXT or f in KEEP_NAMES):
            continue
        size_mb = os.path.getsize(p) / 1e6
        if size_mb > MAX_MB:
            print(f"skipping {f} ({size_mb:.0f} MB)")
            continue
        shutil.copy2(p, os.path.join(dst, f))
        copied.append(f)
    print(f"copied {len(copied)} files to results/{args.name}: {', '.join(copied)}")
    if args.dry_run or not copied:
        return

    git(args.repo, "config", "user.name", "kaggle-runner")
    git(args.repo, "config", "user.email", "kaggle-runner@users.noreply.github.com")
    git(args.repo, "add", os.path.join("results", args.name))
    if not git(args.repo, "status", "--porcelain", check=False).stdout.strip():
        print("nothing to commit")
        return
    git(args.repo, "commit", "-m", args.message or f"results: {args.name}")
    branch = git(args.repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    for _ in range(3):
        if git(args.repo, "push", "origin", "HEAD", check=False).returncode == 0:
            print("pushed")
            return
        git(args.repo, "pull", "--rebase", "origin", branch, check=False)
    sys.exit("push failed, check that the token has write access to the repo")


if __name__ == "__main__":
    main()
