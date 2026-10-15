"""Kaggle script runner (strict). Re-runs everything that was missing, using the already-trained checkpoints.

Differences from the old runner: a failed step is NEVER hidden. Critical steps abort the run, every step is
listed PASS/FAIL at the end, and the process exits non-zero if anything failed.

Edit the constants below, then:  kaggle kernels push -p notebooks
Inputs to attach to the kernel: dataset 'pengcw1/market-1501' and YOUR checkpoints dataset (see GUIDE_5DAY.md).
"""
import datetime, os, shutil, subprocess, sys, time

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ------------------------------ settings ------------------------------
DRY_RUN = True           # True = ~10 min crash-test on 160 queries; writes to results_dryrun/ (never use for the report)
REPO_URL = "https://github.com/udayraj1238/reid-attack-anatomy.git"
BRANCH = "main"
TRAIN_IF_MISSING = True  # train baselines / adversarial model only if their .pth is not found
DATA = "/kaggle/input"
# ----------------------------------------------------------------------

REPO = "reid-attack-anatomy"
RESULTS = "results_dryrun" if DRY_RUN else "results"
Q = "--max_query 160" if DRY_RUN else ""
os.environ["REID_RESULTS"] = RESULTS
summary = []


def banner(t):
    print(f"\n{'=' * 78}\n[{datetime.datetime.now():%H:%M:%S}] {t}\n{'=' * 78}", flush=True)


def step(name, cmd, critical=False, cwd=REPO):
    banner(name)
    print(">>>", cmd, flush=True)
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
    # export whatever real results exist
    out = "/kaggle/working"
    if os.path.isdir(out):
        for folder in (RESULTS, "docs"):
            src = os.path.join(REPO, folder)
            if os.path.isdir(src):
                dst = os.path.join(out, folder)
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
        print(f"exported {RESULTS}/ and docs/ to {out}")
    print("\nRESULT:", "ALL STEPS PASSED" if not failed else f"{len(failed)} STEP(S) FAILED: {failed}")
    sys.exit(1 if failed else 0)


def main():
    banner(f"ReID attack anatomy - strict re-run (dry_run={DRY_RUN})")
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
    if TRAIN_IF_MISSING and not DRY_RUN:
        for arch in ("resnet50", "densenet121", "vit_small"):
            step(f"train {arch} (skipped if checkpoint exists)",
                 f"python -u -m reidattack.train --data {DATA} --arch {arch} --epochs 30 --skip_if_exists")
    base = [f for f in ("resnet50", "densenet121", "vit_small") if os.path.exists(os.path.join(ck, f"{f}.pth"))]
    if len(base) != 3:
        print(f"[FAIL] need resnet50/densenet121/vit_small checkpoints in {ck}, found {base}", flush=True)
        summary.append(("check base checkpoints exist", "FAIL", 0, True))
        finish(aborted="check base checkpoints exist")
    if TRAIN_IF_MISSING and not DRY_RUN:
        step("adv-train resnet50 (skipped if exists)", f"python -u -m reidattack.adv_train --data {DATA} --arch resnet50 --eps 8 --epochs 20 --skip_if_exists")
    CK = "checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth"
    adv = "checkpoints/resnet50_adv_trained_eps8.pth"
    has_adv = os.path.exists(os.path.join(ck, "resnet50_adv_trained_eps8.pth"))

    step("baselines (real eval of every checkpoint)", f"python -u scripts/eval_baselines.py --data {DATA} --ckpts {CK}", critical=True)
    step("evasion: attack zoo + transfer matrix (eps 8, all sources)",
         f"python -u scripts/run_attacks.py --data {DATA} --ckpts {CK} --mode evade --attacks fgsm pgd mifgsm logit_pgd --eps 8 {Q}", critical=True)
    step("evasion: eps sweep (resnet50 as source)",
         f"python -u scripts/run_attacks.py --data {DATA} --ckpts {CK} --mode evade --attacks fgsm pgd mifgsm logit_pgd --eps {'4' if DRY_RUN else '2 4 16'} --sources resnet50 {Q}")
    step("impersonation (pgd, mifgsm)",
         f"python -u scripts/run_attacks.py --data {DATA} --ckpts {CK} --mode impersonate --attacks pgd mifgsm --eps 8 {Q}")
    step("universal patch (EOT)",
         f"python -u scripts/run_patch.py --data {DATA} --src checkpoints/resnet50.pth --others checkpoints/densenet121.pth checkpoints/vit_small.pth --epochs {2 if DRY_RUN else 40}")
    if has_adv:
        step("defence: standard vs adversarially-trained resnet50",
             f"python -u scripts/run_attacks.py --data {DATA} --ckpts checkpoints/resnet50.pth {adv} --mode evade --attacks pgd mifgsm --eps 8 --out {RESULTS}/defense.json {Q}")
    else:
        print("[skip] no adversarially-trained checkpoint found -> no defence results", flush=True)

    # ---- figures: real data only (each script stops with an error if its JSON is missing)
    fig = [
        ("table: baselines", "python -u scripts/make_baseline_table.py"),
        ("fig: transfer PGD rank1", "python -u scripts/plot_transfer_matrix.py --attack pgd --eps 8 --metric rank1"),
        ("fig: transfer PGD mAP", "python -u scripts/plot_transfer_matrix.py --attack pgd --eps 8 --metric mAP"),
        ("fig: transfer MI-FGSM rank1", "python -u scripts/plot_transfer_matrix.py --attack mifgsm --eps 8 --metric rank1"),
        ("fig: transfer Logit-PGD rank1", "python -u scripts/plot_transfer_matrix.py --attack logit_pgd --eps 8 --metric rank1"),
        ("fig: impersonation hit@1", f"python -u scripts/plot_transfer_matrix.py --input {RESULTS}/attacks_impersonate.json --attack pgd --eps 8 --metric impersonation_hit@1"),
        ("fig: eps curve", "python -u scripts/plot_eps_curve.py --model resnet50"),
        ("fig: patch effect", "python -u scripts/plot_patch.py"),
        ("fig: defence", "python -u scripts/plot_defense.py") if has_adv else None,
        ("fig: Grad-CAM (real)", f"python -u scripts/generate_gradcam.py --data {DATA} --ckpt checkpoints/resnet50.pth"),
        ("fig: t-SNE (real)", f"python -u scripts/plot_tsne.py --data {DATA} --ckpt checkpoints/resnet50.pth"),
        ("fig: retrieval before/after (real)", f"python -u scripts/retrieval_figure.py --data {DATA} --ckpt checkpoints/resnet50.pth"),
        ("demo self-check (real model)", f"python -u demo/app.py --data {DATA} --ckpt checkpoints/resnet50.pth --patch {RESULTS}/patch.pth --check"),
    ]
    for item in fig:
        if item:
            step(item[0], item[1])
    step("docs: results summary + facts", "python -u scripts/build_docs.py")
    if not DRY_RUN:                     # dry run evaluates 160 queries, so cross-checks against full baselines would (rightly) disagree
        step("integrity check", "python -u scripts/verify_results.py")
    finish()


if __name__ == "__main__":
    main()
