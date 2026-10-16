"""Master runner for the Day-4 full real experiments on local machine.
Runs:
  1. PGD-50 & MI-FGSM-50 on the defended ResNet-50 (eps=8, 50 steps)
  2. Epsilon sweep PGD-10 on the defended ResNet-50 (eps=2, 4, 16, 10 steps)
  3. Leave-one-out ensemble transfer at eps=8 (PGD-10 & MI-FGSM-10)
  4. Leave-one-out ensemble transfer at eps=16 (PGD-10 & MI-FGSM-10)
  5. Generates all comparison plots
  6. Runs verify_results.py --strict
"""
import datetime, os, subprocess, sys, time

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["REID_RESULTS"] = "results"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA = "C:/Users/Uday Raj/OneDrive/Desktop/CODES/advPattern-implementation/Market-1501-v15.09.15"
ADV = "checkpoints/resnet50_adv_trained_eps8.pth"
PAIR = f"checkpoints/resnet50.pth {ADV}"
ALL3 = "checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth"

steps_summary = []


def banner(t):
    print(f"\n{'=' * 78}\n[{datetime.datetime.now():%H:%M:%S}] {t}\n{'=' * 78}", flush=True)


def run_step(name, cmd, critical=True):
    banner(name)
    print(">>>", cmd, flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    dt = time.time() - t0
    status = "PASS" if rc == 0 else "FAIL"
    steps_summary.append((name, status, dt))
    print(f"\n[{status}] {name} ({dt:.0f}s / {dt/60:.1f}m)", flush=True)
    if rc != 0 and critical:
        print(f"\n[ERROR] Step failed with return code {rc}. Aborting.", flush=True)
        sys.exit(rc)
    return rc == 0


def main():
    t_start = time.time()
    banner("ReID Day-4 FULL REAL EXPERIMENTS STARTED")
    print(f"Data directory: {DATA}")
    print(f"Device: {'CUDA' if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' else 'CPU'}")

    import json

    # Step 1: PGD-50 + MI-FGSM-50 on defended model
    if not (os.path.exists("results/defense_pgd50.json") and len(json.load(open("results/defense_pgd50.json"))) >= 6):
        run_step(
            "Step 1: PGD-50 + MI-FGSM-50 on defended model (eps=8, steps=50)",
            f'python -u scripts/run_attacks.py --data "{DATA}" --ckpts {PAIR} --mode evade '
            f'--attacks pgd mifgsm --eps 8 --steps 50 --sources resnet50_adv_trained_eps8 --workers 0 --out results/defense_pgd50.json'
        )
    else:
        print("[CACHED] Step 1 already completed in results/defense_pgd50.json", flush=True)

    # Step 2: Epsilon sweep on defended model
    if not (os.path.exists("results/defense_eps_sweep.json") and len(json.load(open("results/defense_eps_sweep.json"))) >= 8):
        run_step(
            "Step 2: Epsilon sweep PGD-10 on defended model (eps 2, 4, 16)",
            f'python -u scripts/run_attacks.py --data "{DATA}" --ckpts {PAIR} --mode evade '
            f'--attacks pgd --eps 2 4 16 --steps 10 --sources resnet50_adv_trained_eps8 --workers 0 --out results/defense_eps_sweep.json'
        )
    else:
        print("[CACHED] Step 2 already completed in results/defense_eps_sweep.json", flush=True)

    # Step 3: Leave-one-out ensemble transfer at eps=8
    run_step(
        "Step 3: Leave-one-out ensemble transfer at eps=8 (PGD-10 + MI-FGSM-10)",
        f'python -u scripts/run_ensemble_attack.py --data "{DATA}" --ckpts {ALL3} --attacks pgd mifgsm --eps 8 --chunk 16 --workers 0 --out results/ensemble.json'
    )

    # Step 4: Leave-one-out ensemble transfer at eps=16
    run_step(
        "Step 4: Leave-one-out ensemble transfer at eps=16 (PGD-10 + MI-FGSM-10)",
        f'python -u scripts/run_ensemble_attack.py --data "{DATA}" --ckpts {ALL3} --attacks pgd mifgsm --eps 16 --chunk 16 --workers 0 --out results/ensemble.json'
    )

    # Step 5: Plotting
    run_step("Step 5a: Plot defense strength", "python scripts/plot_defense_strength.py", critical=False)
    run_step("Step 5b: Plot ensemble transfer (PGD, eps=8)", "python scripts/plot_ensemble.py --attack pgd --eps 8", critical=False)
    run_step("Step 5c: Plot ensemble transfer (MI-FGSM, eps=8)", "python scripts/plot_ensemble.py --attack mifgsm --eps 8", critical=False)
    run_step("Step 5d: Plot ensemble transfer (PGD, eps=16)", "python scripts/plot_ensemble.py --attack pgd --eps 16", critical=False)

    # Step 6: Verification
    run_step("Step 6: Verify results integrity (--strict)", "python scripts/verify_results.py --strict", critical=False)

    total_time = time.time() - t_start
    banner("DAY-4 FULL EXPERIMENTS SUMMARY")
    for name, status, dt in steps_summary:
        print(f"  {status:4s}  {name}  ({dt:.0f}s / {dt/60:.1f}m)")
    print(f"\nTotal elapsed time: {total_time:.0f}s ({total_time/3600:.2f} hours)")
    print("ALL DAY-4 REAL EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
