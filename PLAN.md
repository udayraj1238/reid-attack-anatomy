# 10-12 day plan (check off as you go)

| Day | Goal | Output |
|-----|------|--------|
| 1 | Run `smoke_test.py`, set up Kaggle + Market-1501, push repo to GitHub | green smoke test, repo live |
| 2-3 | Train ResNet50 (BoT baseline) -> ~85% Rank-1 / ~65-70% mAP expected; start DenseNet121 | `checkpoints/resnet50.pth` |
| 4 | Train ViT-small; clean-accuracy table for all 3 models | baseline table |
| 5 | Attack zoo: FGSM / PGD / MI-FGSM / logit-PGD, eps 4 & 8, evasion | `results/attacks_evade.json` |
| 6 | Impersonation + transfer matrix heatmap; eps sweep curve | transfer figure |
| 7 | Analysis figures: Grad-CAM before/after, t-SNE embedding drift | "why" figures |
| 8 | Universal patch (EOT) + random-patch control + transfer | `patch.png`, patch results |
| 9 | Defence: PGD adversarial training of ResNet50 (robustness vs accuracy trade-off) | trade-off plot |
| 10 | Gradio demo (upload photo -> paste patch -> ranking collapse) + README results | demo |
| 11 | Slides + speaker script + Q&A sheet | deck |
| 12 | Buffer + 2 full rehearsals | |

Group split suggestion: A = training/baselines, B = attacks/transfer, C = patch/demo, D = analysis/defence/slides.
Everybody must be able to explain every figure: the professor grades understanding, not just results.
