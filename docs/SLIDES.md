<!-- AUTO-GENERATED 2026-10-15 16:01 UTC from results/*.json by scripts/build_docs.py - do not hand-edit numbers -->
# Slides (10) - numbers are filled automatically from results/*.json by `python scripts/build_docs.py --fill`

## 1. Title
Anatomy of an Attack on Person Re-Identification (+ an "invisibility" torso patch)
Team / course / date

## 2. Why this matters
- ReID = find the same person across cameras: the system returns a *ranking* of gallery images, not a label.
- If an attacker can push the true match out of the top-k, surveillance/retrieval fails silently.
- Question we answer: **how do attacks work on embeddings, why do they transfer between architectures, and can a printable patch do it?**

## 3. Victims and clean accuracy (Market-1501, 3,368 queries, 15,913 gallery)
| Model | Rank-1 | mAP |
|---|---|---|
| ResNet-50 (BoT, BNNeck) | 87.0 | 70.6 |
| DenseNet-121 | 77.3 | 56.6 |
| ViT-Small/16 | 91.8 | 79.7 |
Training: ImageNet-pretrained backbones, ID loss + batch-hard triplet loss, PK sampler. Note: one "epoch" = one pass over the 751 identities (4 images each), not over all images.

## 4. Threat model and objectives
- White-box (attacker has weights) vs black-box transfer (attack crafted on a surrogate). L-inf budget eps = 8/255 on raw pixels.
- **Evasion**: minimise cosine(f(x_adv), f(x)).  **Impersonation**: maximise cosine(f(x_adv), centroid of a target identity).
- Baseline for comparison: classification-style attack on the ID-classifier logits (Logit-PGD).

## 5. Algorithms (all implemented from scratch)
FGSM (one step) -> PGD (iterate + project) -> MI-FGSM (PGD + momentum on L1-normalised gradient).

## 6. Results: attacking one model (white-box, eps=8)
- Rank-1 on ResNet-50: clean 87.0 -> FGSM 39.6, PGD-10 0.0, MI-FGSM 0.0.
- Embedding-space attack (PGD) vs classification-style attack (Logit-PGD): 0.0 vs 0.3 Rank-1.
- Figures: eps_degradation_curve.png, retrieval_before_after.png, gradcam_analysis.png, tsne_embedding_drift.png.

## 7. Transfer between architectures (PGD, eps=8)
- Figure: transfer_matrix_pgd_eps8_rank1.png. Mean Rank-1 under black-box transfer: PGD 61.0, MI-FGSM 50.6 (white-box PGD mean: 0.0).
- ResNet-50 -> DenseNet-121: 30.5;  ResNet-50 -> ViT-Small: 88.2.
- Impersonation hit@1 (PGD): white-box 98.6%, transfer 0.4%.
- **Interpretation: write 2 sentences ONLY after looking at the heat-map. Do not claim a cause (e.g. "ViTs are robust because of global attention") unless you tested it.**

## 8. Universal torso patch (EOT)
- One 64x48 patch for everyone; trained with random scale/rotation/shift/brightness/noise (EOT) + total-variation smoothness.
- ResNet-50 Rank-1: clean 87.0, random patch 73.2, trained patch 2.7, trained patch with random transformations 2.7.
- Transfer of the same patch: DenseNet-121 13.6, ViT-Small 74.0.
- Caveat: digital pasting only; printing/camera effects not tested.

## 9. Defence: adversarial training (ResNet-50, PGD, eps=8)
| | clean Rank-1 | under white-box PGD Rank-1 |
|---|---|---|
| standard | 87.0 | 0.0 |
| adversarially trained | 74.5 | 29.2 |

## 10. Takeaways, limitations, future work
- Limitations: single seed, one dataset, "epoch" definition, digital-only patch, attacker knows the gallery model in white-box tests.
- Future: stronger adaptive attacks against the defence, physical printing test, more architectures.
