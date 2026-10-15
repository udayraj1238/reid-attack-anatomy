# Anatomy of an Attack on Person Re-Identification (+ universal torso patch)

Deep Learning course project. We study **how adversarial attacks work on person re-identification (ReID)**, why they
**transfer between architectures**, whether a **single universal patch** can hide a person, and what **adversarial training** buys.

## Why ReID attacks differ from classification attacks
A classifier outputs a label. A ReID system outputs an embedding and is judged on the **ranking** of a gallery. So attacks
optimise in embedding space: **evasion** (move away from your own embedding) and **impersonation** (move toward a chosen identity).
We also run a classification-style `logit_pgd` baseline to show the difference.

## What is here
| Path | What |
|---|---|
| `reidattack/` | data (Market-1501 + PK sampler), models (ResNet-50 / DenseNet-121 / ViT-Small, BNNeck), losses, CMC/mAP, **FGSM / PGD / MI-FGSM from scratch**, EOT patch, adversarial training, logging, provenance |
| `scripts/` | `eval_baselines`, `run_attacks` (transfer matrix, eps sweep, impersonation, defence), `run_patch`, figure scripts, `build_docs`, `verify_results` |
| `demo/app.py` | Gradio demo on the real model |
| `notebooks/kaggle_runner.py` | strict one-shot Kaggle runner |
| `docs/*.template.md` | slides, speaker script, viva sheet - numbers are filled from real results |
| `GUIDE_5DAY.md` | step-by-step working guide |

## Results are never typed in
Every number/figure comes from `results/*.json` written by a real run; `results/MANIFEST.json` stores commit + checkpoint hashes;
`python scripts/verify_results.py` checks everything. See `docs/RESULTS_SUMMARY.md` (auto-generated) for the current numbers.

## Quick start
```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # ~1-2 min, no dataset, no results produced
# GPU (Kaggle T4): train, attack, patch, defend, plot -> see GUIDE_5DAY.md and notebooks/kaggle_runner.py
python -m reidattack.train --data <market1501_dir> --arch resnet50 --epochs 30
python scripts/eval_baselines.py --data <market1501_dir> --ckpts checkpoints/*.pth
python scripts/run_attacks.py --data <market1501_dir> --ckpts checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth
python demo/app.py --data <market1501_dir> --ckpt checkpoints/resnet50.pth --patch results/patch.pth
```

## Method notes (say these in the viva)
- Backbones are **ImageNet-pretrained**; loss = label-smoothed ID loss + batch-hard triplet; PK sampler (16 ids x 4 images).
- One training "epoch" = one pass over the 751 identities (4 images each), i.e. ~23% of the images.
- Attacks work on raw [0,1] pixels (normalisation is inside the model); L-inf budget eps/255; PGD step = 2.5*eps/steps, 10 steps, random start.
- Evaluation: Market-1501 protocol with the same-id-same-camera junk rule; Rank-1/5/10 and mAP.

## Limitations
Single seed, one dataset, digital (not printed) patch, 10-step PGD, no adaptive attack against the defence.
