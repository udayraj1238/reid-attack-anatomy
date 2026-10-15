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
| ResNet-50 (BoT, BNNeck) | {{base_resnet50_r1}} | {{base_resnet50_map}} |
| DenseNet-121 | {{base_densenet121_r1}} | {{base_densenet121_map}} |
| ViT-Small/16 | {{base_vit_small_r1}} | {{base_vit_small_map}} |
Training: ImageNet-pretrained backbones, ID loss + batch-hard triplet loss, PK sampler. Note: one "epoch" = one pass over the 751 identities (4 images each), not over all images.

## 4. Threat model and objectives
- White-box (attacker has weights) vs black-box transfer (attack crafted on a surrogate). L-inf budget eps = 8/255 on raw pixels.
- **Evasion**: minimise cosine(f(x_adv), f(x)).  **Impersonation**: maximise cosine(f(x_adv), centroid of a target identity).
- Baseline for comparison: classification-style attack on the ID-classifier logits (Logit-PGD).

## 5. Algorithms (all implemented from scratch)
FGSM (one step) -> PGD (iterate + project) -> MI-FGSM (PGD + momentum on L1-normalised gradient).

## 6. Results: attacking one model (white-box, eps=8)
- Rank-1 on ResNet-50: clean {{base_resnet50_r1}} -> FGSM {{wb_fgsm_resnet50_r1}}, PGD-10 {{wb_pgd_resnet50_r1}}, MI-FGSM {{wb_mifgsm_resnet50_r1}}.
- Embedding-space attack (PGD) vs classification-style attack (Logit-PGD): {{wb_pgd_resnet50_r1}} vs {{wb_logit_pgd_resnet50_r1}} Rank-1.
- Figures: eps_degradation_curve.png, retrieval_before_after.png, gradcam_analysis.png, tsne_embedding_drift.png.

## 7. Transfer between architectures (PGD, eps=8)
- Figure: transfer_matrix_pgd_eps8_rank1.png. Mean Rank-1 under black-box transfer: PGD {{xfer_pgd_mean_r1}}, MI-FGSM {{xfer_mifgsm_mean_r1}} (white-box PGD mean: {{wb_pgd_mean_r1}}).
- ResNet-50 -> DenseNet-121: {{xfer_pgd_resnet50_to_densenet121_r1}};  ResNet-50 -> ViT-Small: {{xfer_pgd_resnet50_to_vit_small_r1}}.
- Impersonation hit@1 (PGD): white-box {{imp_pgd_wb_mean_hit}}%, transfer {{imp_pgd_xfer_mean_hit}}%.
- **Interpretation: write 2 sentences ONLY after looking at the heat-map. Do not claim a cause (e.g. "ViTs are robust because of global attention") unless you tested it.**

## 8. Universal torso patch (EOT)
- One 64x48 patch for everyone; trained with random scale/rotation/shift/brightness/noise (EOT) + total-variation smoothness.
- ResNet-50 Rank-1: clean {{patch_clean_resnet50_r1}}, random patch {{patch_random_patch_resnet50_r1}}, trained patch {{patch_trained_patch_resnet50_r1}}, trained patch with random transformations {{patch_trained_patch_eot_resnet50_r1}}.
- Transfer of the same patch: DenseNet-121 {{patch_trained_patch_densenet121_r1}}, ViT-Small {{patch_trained_patch_vit_small_r1}}.
- Caveat: digital pasting only; printing/camera effects not tested.

## 9. Defence: adversarial training (ResNet-50, PGD, eps=8)
| | clean Rank-1 | under white-box PGD Rank-1 |
|---|---|---|
| standard | {{def_clean_resnet50_r1}} | {{def_wb_pgd_resnet50_r1}} |
| adversarially trained | {{def_clean_resnet50_adv_trained_eps8_r1}} | {{def_wb_pgd_resnet50_adv_trained_eps8_r1}} |

## 10. Takeaways, limitations, future work
- Limitations: single seed, one dataset, "epoch" definition, digital-only patch, attacker knows the gallery model in white-box tests.
- Future: stronger adaptive attacks against the defence, physical printing test, more architectures.
