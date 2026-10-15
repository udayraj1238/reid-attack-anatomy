"""Day-4 experiments (strict runner). Needs the repo pushed to GitHub (it clones/pulls it) and the checkpoints dataset attached.

  1. PGD-50 / MI-FGSM-50 white-box attack on the adversarially-trained ResNet-50  (is 29% Rank-1 real?)
  2. epsilon sweep on the defended model (gradient-masking sanity check)
  3. leave-one-out ENSEMBLE transfer (is ViT's ~88% under CNN transfer a real barrier or a weak-surrogate artefact?)
Step 3 runs eps=8 first (comparable to the single-surrogate table), then eps=16.

Edit constants, then:  kaggle kernels push -p notebooks/day4
Recommended order: DRY_RUN = True first (crash test on 160 queries -> results_dryrun/), then DRY_RUN = False.
"""
import datetime, os, shutil, subprocess, sys, time

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ------------------------------ settings ------------------------------
DRY_RUN = True
REPO_URL = "https://github.com/udayraj1238/reid-attack-anatomy.git"
BRANCH = "main"
DATA = "/kaggle/input"
RUN_EPS16_ENSEMBLE = True      # extra ~same cost as the eps=8 ensemble run
# ----------------------------------------------------------------------

REPO = "reid-attack-anatomy"
RESULTS = "results_dryrun" if DRY_RUN else "results"
Q = "--max_query 160" if DRY_RUN else ""
os.environ["REID_RESULTS"] = RESULTS
summary = []


def banner(t):
    print(f"\n{'=' * 78}\n[{datetime.datetime.now():%H:%M:%S}] {t}\n{'=' * 78}", flush=True)


def step(name, cmd, critical=False, cwd=REPO):
    banner(name); print(">>>", cmd, flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True, cwd=cwd if os.path.isdir(cwd) else None).returncode
    dt = time.time() - t0
    summary.append((name, "PASS" if rc == 0 else "FAIL", dt, critical))
    print(f"[{'PASS' if rc == 0 else 'FAIL'}] {name} ({dt:.0f}s)", flush=True)
    if rc != 0 and critical:
        finish(aborted=name)
    return rc == 0


def finish(aborted=None):
    banner("SUMMARY")
    for n, s, dt, c in summary:
        print(f"  {s:4s}  {n}  ({dt:.0f}s){'  [critical]' if c else ''}")
    failed = [n for n, s, _, _ in summary if s == "FAIL"]
    if aborted:
        print(f"\nABORTED at critical step: {aborted}")
    out = "/kaggle/working"
    if os.path.isdir(out):
        src = os.path.join(REPO, RESULTS)
        if os.path.isdir(src):
            dst = os.path.join(out, RESULTS)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            print(f"exported {RESULTS}/ to {out}")
    print("\nRESULT:", "ALL STEPS PASSED" if not failed else f"{len(failed)} STEP(S) FAILED: {failed}")
    sys.exit(1 if failed else 0)


def main():
    banner(f"ReID Day-4 experiments (dry_run={DRY_RUN})")
    step("setup: pip", "pip -q install timm", cwd=".")
    if not os.path.isdir(REPO):
        step("setup: clone repo", f"git clone -b {BRANCH} {REPO_URL} {REPO}", critical=True, cwd=".")
    else:
        step("setup: git pull", f"git pull origin {BRANCH}", cwd=REPO)
    step("smoke test", "python -u tests/smoke_test.py", critical=True)

    ck = os.path.join(REPO, "checkpoints"); os.makedirs(ck, exist_ok=True)
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".pth") and not os.path.exists(os.path.join(ck, f)):
                shutil.copy2(os.path.join(root, f), os.path.join(ck, f)); print("using checkpoint", f, flush=True)
    need = ["resnet50", "densenet121", "vit_small", "resnet50_adv_trained_eps8"]
    missing = [n for n in need if not os.path.exists(os.path.join(ck, f"{n}.pth"))]
    if missing:
        print(f"[FAIL] missing checkpoints: {missing}", flush=True)
        summary.append(("check checkpoints exist", "FAIL", 0, True)); finish(aborted="check checkpoints exist")

    # the real-results files from the main run are NOT on Kaggle; Day-4 plots that compare against them run locally afterwards.
    ADV = "checkpoints/resnet50_adv_trained_eps8.pth"
    PAIR = f"checkpoints/resnet50.pth {ADV}"
    ALL3 = "checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth"
    step("defence: PGD-50 + MI-FGSM-50, white-box on the defended model (eps 8)",
         f"python -u scripts/run_attacks.py --data {DATA} --ckpts {PAIR} --mode evade --attacks pgd mifgsm --eps 8 --steps 50 "
         f"--sources resnet50_adv_trained_eps8 --out {RESULTS}/defense_pgd50.json {Q}")
    step("defence: epsilon sweep PGD-10 on the defended model (eps 2 4 16)",
         f"python -u scripts/run_attacks.py --data {DATA} --ckpts {PAIR} --mode evade --attacks pgd --eps 2 4 16 --steps 10 "
         f"--sources resnet50_adv_trained_eps8 --out {RESULTS}/defense_eps_sweep.json {Q}")
    step("ensemble transfer (leave-one-out) eps 8, PGD-10 + MI-FGSM-10",
         f"python -u scripts/run_ensemble_attack.py --data {DATA} --ckpts {ALL3} --attacks pgd mifgsm --eps 8 {Q}")
    if RUN_EPS16_ENSEMBLE:
        step("ensemble transfer (leave-one-out) eps 16",
             f"python -u scripts/run_ensemble_attack.py --data {DATA} --ckpts {ALL3} --attacks pgd mifgsm --eps 16 {Q}")
    finish()


if __name__ == "__main__":
    main()
