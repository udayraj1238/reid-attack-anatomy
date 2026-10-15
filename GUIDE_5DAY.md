# Working guide: fix the repo, get REAL results, finish in 5-6 days

## 0. Where you stand (honest)
Real: the three baselines, the adversarially-trained ResNet-50, all `.pth` files (from your Kaggle training logs).
Not real (removed in this zip): every figure in `results/`, the old baseline table, and all numbers in the old slides/script/viva sheet.
Never run: attacks, impersonation, patch evaluation, defence evaluation (the `torch.load` crash). This zip fixes the code so they run, and makes it impossible to draw a figure without real data.

## 1. Put the fix on GitHub (10 min, do first)
```bash
cd reid-attack-anatomy                      # your existing local clone
git checkout -b fix/real-results
git rm -r --ignore-unmatch results docs configs demo PHASE1_REPORT.md PHASE1_REVIEWER_GUIDE.md PLAN.md README.md \
    scripts notebooks reidattack tests
# copy the CONTENTS of the new zip folder over the repo folder (keep your .git), then:
git add -A
git commit -m "Real-results-only pipeline: remove placeholder figures/docs, add strict runner, real Grad-CAM/t-SNE/demo"
git checkout main && git merge fix/real-results && git push origin main
```
Old commits still contain the fake figures; `CHANGELOG.md` says so openly. That is the right way to handle it.

## 2. Kaggle setup (one-time, 15 min)
1. Kaggle -> Datasets -> New Dataset -> upload your `.pth` files (resnet50, densenet121, vit_small, resnet50_adv_trained_eps8; skip `*_adv_latest.pth`) -> name it **reid-checkpoints** (private).
2. Make sure `notebooks/kernel-metadata.json` lists your checkpoints dataset (`<your-username>/reid-checkpoints`) and `pengcw1/market-1501`.
3. **Dry run first**: in `notebooks/kaggle_runner.py` set `DRY_RUN = True` -> `kaggle kernels push -p notebooks`. ~10-15 min, tiny subset, writes to `results_dryrun/`. It must end with `ALL STEPS PASSED`. If a step fails, paste me its log lines and I fix it. Dry-run numbers are NEVER used in the report.
4. Set `DRY_RUN = False`, push again. Full run is roughly 1.5-2.5 h on a T4 (attacks ~45 min, patch ~10 min, figures ~10 min). It saves after every config and resumes if interrupted.
5. Download the output: `results/` (JSON + PNG + MANIFEST) and `docs/RESULTS_SUMMARY.md`; copy into the repo.

## 3. After the real run (30 min)
```bash
python scripts/verify_results.py          # must say OK
python scripts/build_docs.py --fill       # writes docs/SLIDES.md, SPEAKER_SCRIPT.md, VIVA_QA_SHEET.md from real numbers
git add results docs && git commit -m "Add real Phase 1 results" && git push
```
`build_docs --fill` refuses to run if any number has no real source.

### Sanity ranges (my expectation, not a guarantee - investigate if far off)
- White-box PGD-10 at eps=8 on a standard model normally pushes Rank-1 close to 0-10%. If it stays above ~30%, look for a bug (wrong model mode, eps, steps) before believing it.
- Black-box transfer should be weaker than white-box on every row. If not, tell me.
- Random patch should hurt much less than the trained patch. If the trained patch barely beats random, train longer (`--epochs 80`) or enlarge it (`--ph 80 --pw 60`) and report what you tried.
- Adversarially-trained model: clean Rank-1 should drop a bit, robust Rank-1 should rise. A defence evaluated only with PGD-10 can look better than it is: do the Day-4 stronger-attack check.

## 4. Day-by-day (team of ~4; every person must be able to explain every figure)
| Day | Goal | Done when |
|---|---|---|
| 1 | Section 1 + 2 done; dry run passes; full run launched. Meanwhile everyone reads `README.md` + viva Q1-6 | repo on GitHub clean; full run running |
| 2 | Full results in; `verify_results.py` OK; look at every figure and write 1-2 sentences per figure *in your own words* (what the axes mean, what you see, one hypothesis) | `docs/` filled; notes file per person |
| 3 | **Understanding day.** Each member teaches one file in 5 min: A `attacks.py` (FGSM/PGD/MI-FGSM, objectives), B `patch.py` (EOT, TV), C `evaluate.py` + `losses.py` (CMC/mAP, junk rule, triplet), D `models.py` + `adv_train.py`. Mock viva with the sheet | everyone answers Q1-16 without notes |
| 4 | One optional experiment to deepen understanding (pick ONE): (a) stronger attack on the defence: `run_attacks.py --ckpts checkpoints/resnet50.pth checkpoints/resnet50_adv_trained_eps8.pth --attacks pgd --eps 8 --steps 50 --out results/defense_pgd50.json`; (b) patch size sweep (`run_patch.py --ph 32 --pw 24` vs default, use `--out results/patch_small`); (c) PGD steps ablation (run `run_attacks.py --attacks pgd --eps 8 --sources resnet50 --steps N --out results/steps_N.json` once for N = 1, 5, 10, 20) | one new real figure/table you can explain |
| 5 | Demo + slides: `python demo/app.py ...` rehearsal on your laptop (needs Market-1501 + checkpoints, or Kaggle). Turn `docs/SLIDES.md` into your deck (ask me to generate the PPTX once numbers exist). Rehearse twice with a timer | deck + working demo |
| 6 | Buffer, final `verify_results.py`, README results section, tag release `v1.0` | submitted |

## 5. Presenting honestly (the professor grades understanding)
- Say: backbones are ImageNet-pretrained; an "epoch" is one pass over the 751 identities; single seed; patch is digital-only.
- Give explanations as hypotheses unless you tested them (e.g. do not say "ViTs resist transfer because of global attention" unless a result supports and you label it a hypothesis).
- If asked how a number was produced: point to the script, the JSON and `MANIFEST.json`.

## 6. How to work with me
Paste: (1) the final SUMMARY block from the Kaggle log, (2) `docs/RESULTS_SUMMARY.md`, (3) any failing step's last ~40 log lines. I will then fix bugs, write the interpretation text from your real numbers, and build the PPTX. Please do not paste tokens/keys.
