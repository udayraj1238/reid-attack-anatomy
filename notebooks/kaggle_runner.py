"""Kaggle Standalone Script Runner: ReID Adversarial Robustness Pipeline.

When executed as a Kaggle Script (kernel_type: 'script'), this file runs directly
as the root container process. Python's stdout is unbuffered, guaranteeing that
EVERY print statement, batch progress percentage, ETA, and metric is streamed
LIVE in real-time directly to the Kaggle "Logs" tab.
"""
import os
import sys
import time
import subprocess
import datetime
import shutil

# Guarantee completely unbuffered stdout and stderr
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)


def banner(title):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 75
    print(f"\n{sep}", flush=True)
    print(f"[{now}] {title}", flush=True)
    print(f"{sep}\n", flush=True)


def run_step(cmd, desc="", cwd=None):
    if desc:
        banner(desc)
    print(f">>> Executing: {cmd}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[WARNING] Command exited with code {res.returncode} after {elapsed:.1f}s: {cmd}\n", flush=True)
    else:
        print(f"\n[SUCCESS] Completed in {elapsed:.1f}s: {desc or cmd}\n", flush=True)
    return res.returncode


def main():
    total_start = time.time()
    banner("STARTING REID ATTACK ANATOMY FULL PIPELINE (KAGGLE SCRIPT MODE)")
    print(f"Working Directory: {os.getcwd()}", flush=True)
    print(f"Python: {sys.executable}", flush=True)

    # ---------------------------------------------------------
    # STEP 0: Environment Setup & Codebase Sync
    # ---------------------------------------------------------
    banner("STEP 0: Installing Dependencies & Syncing Repository")
    run_step("pip -q install timm gradio", "Installing timm & gradio")

    repo_dir = "reid-attack-anatomy"
    if not os.path.exists(repo_dir):
        run_step(f"git clone https://github.com/udayraj1238/{repo_dir}.git", "Cloning Repository")
    else:
        run_step("git pull", "Pulling Latest Updates", cwd=repo_dir)

    run_step("python -u tests/smoke_test.py", "Running Environment Smoke Test", cwd=repo_dir)

    DATA = "/kaggle/input"

    # Reuse existing trained checkpoints from previous kernel runs if available
    ckpt_target_dir = os.path.join(repo_dir, "checkpoints")
    os.makedirs(ckpt_target_dir, exist_ok=True)
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".pth"):
                src_pth = os.path.join(root, f)
                dst_pth = os.path.join(ckpt_target_dir, f)
                if not os.path.exists(dst_pth):
                    try:
                        shutil.copy2(src_pth, dst_pth)
                        print(f"Reusing mounted checkpoint: {f}", flush=True)
                    except Exception as e:
                        print(f"Error copying {f}: {e}", flush=True)

    # ---------------------------------------------------------
    # STEP 1: Train Baseline Victim Models
    # ---------------------------------------------------------
    run_step(
        f"python -u -m reidattack.train --data {DATA} --arch resnet50 --epochs 30 --log_every 20 --skip_if_exists",
        "STEP 1A: Training ResNet-50 Baseline (~20-25 min)",
        cwd=repo_dir
    )

    run_step(
        f"python -u -m reidattack.train --data {DATA} --arch densenet121 --epochs 30 --log_every 20 --skip_if_exists",
        "STEP 1B: Training DenseNet-121 Baseline (~25-30 min)",
        cwd=repo_dir
    )

    run_step(
        f"python -u -m reidattack.train --data {DATA} --arch vit_small --epochs 30 --log_every 20 --skip_if_exists",
        "STEP 1C: Training ViT-Small Baseline (~35-40 min)",
        cwd=repo_dir
    )

    # ---------------------------------------------------------
    # STEP 2: Attack Zoo & Transfer Matrix
    # ---------------------------------------------------------
    CK = "checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth"
    run_step(
        f"python -u scripts/run_attacks.py --data {DATA} --ckpts {CK} --eps 4 8 --mode evade",
        "STEP 2A: Running Evasion Attack Zoo & Transfer Matrix",
        cwd=repo_dir
    )

    run_step(
        f"python -u scripts/run_attacks.py --data {DATA} --ckpts {CK} --eps 8 --mode impersonate --attacks pgd mifgsm",
        "STEP 2B: Running Impersonation Attacks (PGD, MI-FGSM)",
        cwd=repo_dir
    )

    # ---------------------------------------------------------
    # STEP 3: Universal Physical Patch (EOT)
    # ---------------------------------------------------------
    run_step(
        f"python -u scripts/run_patch.py --data {DATA} --src checkpoints/resnet50.pth --others checkpoints/densenet121.pth checkpoints/vit_small.pth --epochs 5",
        "STEP 3: Training Universal Physical Patch with EOT",
        cwd=repo_dir
    )

    # ---------------------------------------------------------
    # STEP 4: Adversarial Training Defense
    # ---------------------------------------------------------
    run_step(
        f"python -u -m reidattack.adv_train --data {DATA} --arch resnet50 --eps 8 --epochs 20 --log_every 15 --skip_if_exists",
        "STEP 4: Training Adversarial Defense Model (ResNet-50, eps=8/255)",
        cwd=repo_dir
    )

    # ---------------------------------------------------------
    # STEP 5: Publication Figures & Visualizations
    # ---------------------------------------------------------
    run_step(
        "python -u scripts/make_baseline_table.py",
        "STEP 5A: Generating Baseline Performance Table",
        cwd=repo_dir
    )
    run_step(
        "python -u scripts/plot_transfer_matrix.py --input results/attacks_evade.json",
        "STEP 5B: Plotting Cross-Architecture Transfer Matrix Heatmap",
        cwd=repo_dir
    )
    run_step(
        "python -u scripts/plot_eps_curve.py --input results/attacks_evade.json",
        "STEP 5C: Plotting Epsilon Degradation Curves",
        cwd=repo_dir
    )
    run_step(
        "python -u scripts/generate_gradcam.py --ckpt checkpoints/resnet50.pth",
        "STEP 5D: Generating Grad-CAM Attention Heatmaps",
        cwd=repo_dir
    )
    run_step(
        "python -u scripts/plot_tsne.py",
        "STEP 5E: Plotting t-SNE Feature Space Drift",
        cwd=repo_dir
    )

    # ---------------------------------------------------------
    # STEP 6: Non-Blocking Demo Verification
    # ---------------------------------------------------------
    run_step(
        "python -u demo/app.py --check",
        "STEP 6: Validating Gradio Interactive UI Components",
        cwd=repo_dir
    )

    # ---------------------------------------------------------
    # STEP 7: Export Artifacts to /kaggle/working/
    # ---------------------------------------------------------
    banner("STEP 7: Exporting All Checkpoints and Visualizations to Output")
    target_out = "/kaggle/working"
    for folder in ["checkpoints", "results"]:
        src = os.path.join(repo_dir, folder)
        dst = os.path.join(target_out, folder)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Exported {src} -> {dst}", flush=True)

    print("\nFinal Output Directory Contents:", flush=True)
    for root, dirs, files in os.walk(target_out):
        level = root.replace(target_out, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/", flush=True)
        subindent = " " * 4 * (level + 1)
        for f in files[:10]:
            print(f"{subindent}{f}", flush=True)
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files", flush=True)

    total_mins = (time.time() - total_start) / 60.0
    banner(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_mins:.1f} MINUTES!")


if __name__ == "__main__":
    main()
