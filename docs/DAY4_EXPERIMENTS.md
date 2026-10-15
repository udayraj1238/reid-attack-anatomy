# Day-4 experiments: commands + what each outcome would mean (written BEFORE seeing results)

Both experiments use the same eps budget as the main tables. Single seed, 3,368 queries: differences under ~1 point (about 30 queries) are noise-level.

## Run order
```bash
# 0) push the new files to GitHub (the Kaggle runner clones from there)
git add scripts/run_ensemble_attack.py scripts/plot_ensemble.py scripts/plot_defense_strength.py notebooks/day4 docs/DAY4_EXPERIMENTS.md
git commit -m "Day-4: PGD-50 defence attack + leave-one-out ensemble transfer" && git push origin main

# 1) crash test on Kaggle: DRY_RUN = True in notebooks/day4/kaggle_day4_runner.py  (about 15 min, writes results_dryrun/, never use for the report)
kaggle kernels push -p notebooks/day4
# 2) real run: set DRY_RUN = False, push again (cost estimate, not measured: PGD-50 ~25-40 min, ensembles ~40-70 min total)
# 3) download results/ from the kernel output, copy ONLY these into your repo's results/:
#      defense_pgd50.json  defense_eps_sweep.json  ensemble.json   and merge the new MANIFEST entries
# 4) local plots + checks (need the main-run JSONs already in results/):
python scripts/plot_defense_strength.py
python scripts/plot_ensemble.py --attack pgd    --eps 8
python scripts/plot_ensemble.py --attack mifgsm --eps 8
python scripts/plot_ensemble.py --attack pgd    --eps 16      # only if the eps=16 ensemble ran; single-surrogate bars exist only where run_attacks covered that eps
python scripts/verify_results.py --strict
```
Important: copy the Kaggle MANIFEST entries (`run_attacks:defense_pgd50.json`, `run_attacks:defense_eps_sweep.json`, `run_ensemble:ensemble.json`) into your repo's `results/MANIFEST.json` rather than overwriting the file.

Run one piece alone (e.g. only one ensemble):
```bash
python scripts/run_ensemble_attack.py --data <market> --ckpts checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth --ensembles resnet50+densenet121 --attacks pgd --eps 8
python scripts/run_attacks.py --data <market> --ckpts checkpoints/resnet50.pth checkpoints/resnet50_adv_trained_eps8.pth --attacks pgd --eps 8 --steps 50 --sources resnet50_adv_trained_eps8 --out results/defense_pgd50.json
```
Both are resumable and save after every configuration.

## Experiment A - is the defence's 29.19% real? (PGD-50, MI-FGSM-50, epsilon sweep)
Reference numbers (PGD-10): defended model clean 74.52, white-box PGD-10 29.19, MI-FGSM-10 25.86.
| Observation | Reasonable reading |
|---|---|
| PGD-50 within a few points of PGD-10 (e.g. 25-29) | robustness against first-order attacks at eps=8 looks genuine; report PGD-50 as the headline number |
| PGD-50 far lower (e.g. < 10) | PGD-10 overstated the defence; report PGD-50 and say so |
| Rank-1 does NOT fall as eps grows, or white-box ends up above black-box | suspect gradient masking; do not claim robustness until investigated |
Caveat to state either way: a defence is only as strong as the strongest attack tried; no adaptive attack beyond these.

## Experiment B - is ViT's ~88% under CNN transfer a real barrier? (leave-one-out ensemble)
Reference (single surrogate, PGD eps=8, Rank-1 on ViT-Small): from ResNet-50 88.15, from DenseNet-121 88.66 (clean 91.75).
| Ensemble resnet50+densenet121 -> ViT (PGD, eps 8) | Reasonable reading |
|---|---|
| drops by more than ~15 points (below ~73) | the single-surrogate gap was partly a weak-surrogate effect; do NOT claim intrinsic ViT resistance |
| stays within ~5 points of 88 | transfer barrier persists with better gradients: consistent with, not proof of, an architecture/training effect |
| in between | partial; say "ensembling helps the attacker by X points" and leave the cause open |
Also read the other two rows (held-out ResNet-50 and DenseNet-121): if the ensemble helps CNNs a lot but ViT little, that asymmetry is the finding. Whatever the result, the cause (attention vs pre-training vs patch tokenization) stays a hypothesis unless you run a controlled test (e.g. train a ViT with a different pre-training or a CNN with the same pre-training).

## What to say in the presentation
Report the number you measured, the reference it is compared with, and one sentence of interpretation chosen from the tables above. Do not describe the outcome as "proving" a mechanism.
