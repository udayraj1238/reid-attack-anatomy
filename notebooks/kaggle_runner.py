"""Kaggle Standalone Script Runner: ReID Adversarial Robustness Pipeline.

When run as a Kaggle Script (kernel_type: 'script'), this file executes directly
as the root process in the Kaggle container, guaranteeing that every print/log statement
is streamed LIVE in real time to the Kaggle Logs tab.
"""
import os
import sys
import time
import subprocess
import datetime

# Ensure unbuffered execution
os.environ["PYTHONUNBUFFERED"] = "1"


def log_step(title):
    print(f"\n{'='*70}", flush=True)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {title}", flush=True)
    print(f"{'='*70}\n", flush=True)


def run_cmd(cmd, cwd=None):
    print(f">>> Running: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        print(f"*** Command failed with return code {res.returncode}: {cmd}", flush=True)
        # Note: we continue or exit based on severity
    return res.returncode


def main():
    t_start = time.time()
    log_step("STEP 0: Initializing Environment & Syncing Codebase")

    # 1. Install dependencies
    run_cmd("pip -q install timm gradio")

    # 2. Sync codebase
    repo_dir = "reid-attack-anatomy"
    if not os.path.exists(repo_dir):
        run_cmd(f"git clone https://github.com/udayraj1238/{repo_dir}.git")
    else:
        run_cmd("git pull", cwd=repo_dir)

    # 3. Smoke test
    run_cmd("python -u tests/smoke_test.py", cwd=repo_dir)

    data_dir = "/kaggle/input"

    # Step 1: Train Baseline Victim Models
    log_step("STEP 1A: Training ResNet-50 Baseline (~20-25 min)")
    run_cmd(
        f"python -u -m reidattack.train --data {data_dir} --arch resnet50 --epochs 30 --log_every 20 --skip_if_exists",
        cwd=repo_dir
    )

    log_step("STEP 1B: Training DenseNet-121 Baseline (~25-30 min)")
    run_cmd(
        f"python -u -m reidattack.train --data {data_dir} --arch densenet121 --epochs 30 --log_every 20 --skip_if_exists",
        cwd=repo_dir
    )

    log_step("STEP 1C: Training ViT-Small Baseline (~35-40 min)")
    run_cmd(
        f"python -u -m reidattack.train --data {data_dir} --arch vit_small --epochs 30 --log_every 20 --skip_if_exists",
        cwd=repo_dir
    )

    # Step 2: Attack Zoo & Transfer Matrix
    ckpts = "checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth"
    log_step("STEP 2: Evaluating Attack Zoo & Cross-Architecture Transfer Matrix")
    run_cmd(
        f"python -u scripts/run_attacks.py --data {data_dir} --checkpoints {ckpts} --out-dir results",
        cwd=repo_dir
    )

    # Step 3: Physical Adversarial Patch
    log_step("STEP 3: Training Universal Torso Adversarial Patch with EOT")
    run_cmd(
        f"python -u -m reidattack.patch --data {data_dir} --ckpt checkpoints/resnet50.pth --epochs 5 --out checkpoints/torso_patch.pth",
        cwd=repo_dir
    )

    # Step 4: Adversarial Training Defense
    log_step("STEP 4: Training Adversarially Robust ResNet-50 Defense Model")
    run_cmd(
        f"python -u -m reidattack.adv_train --data {data_dir} --arch resnet50 --eps 8 --epochs 25 --log_every 15 --skip_if_exists",
        cwd=repo_dir
    )

    # Step 5: Verify Gradio Demo App
    log_step("STEP 5: Verifying Gradio Interactive Demo")
    run_cmd("python -u demo/app.py --check", cwd=repo_dir)

    total_mins = (time.time() - t_start) / 60.0
    log_step(f"PIPELINE COMPLETED SUCCESSFULLY in {total_mins:.1f} minutes!")


if __name__ == "__main__":
    main()
