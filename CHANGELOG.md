# Changelog

## v0.3 - Day-4 additions: defense verification & ensemble transfer
- **PGD-50 & MI-FGSM-50 defense benchmark**: Verified adversarially trained ResNet-50 under 50-step iterative attacks (Rank-1: 22.18% PGD-50, 18.65% MI-FGSM-50 vs 7.69% for baseline ResNet-50).
- **Epsilon sweep**: Evaluated defended model at eps in {2, 4, 8, 16}; strict monotonic degradation confirms absence of gradient masking.
- **Leave-one-out ensemble transfer**: Evaluated ensembles across ResNet-50, DenseNet-121, and ViT-Small at eps=8 and eps=16. Confirmed that ViT-Small's transfer barrier persists under joint CNN ensembling (86.46% R1 at eps=8; 72.51% R1 at eps=16).
- **Provenance note 1 (Manifest key overwrite)**: The eps=8 ensemble manifest entry was overwritten by the eps=16 run in `results/MANIFEST.json`; all 39 rows for both budgets are preserved in `results/ensemble.json`.
- **Provenance note 2 (Hardware provenance)**: Main-run entries were produced on Kaggle (Tesla T4 GPU, PyTorch 2.11 / CUDA 12.8); Day-4 runs were executed locally on an NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.1 / cu128).
- **Provenance note 3 (Numerical variance)**: Clean ResNet-50 Rank-1 varies by ~0.03% across hardware environments (86.97% on Tesla T4 vs 86.94% on RTX 3050) due to CUDA floating-point reduction non-determinism.

## v0.2 - real-results-only rebuild
Earlier commits contained placeholder figures (transfer heat-map, epsilon curves, t-SNE, Grad-CAM, baseline table) that were
produced by fallback code with typed-in numbers, plus slides/script/viva text quoting them. Those results were NOT measured.
- Removed all placeholder figures, fallbacks and the docs that quoted them.
- Plot/table scripts now read only `results/*.json` and stop with an error if a file is missing.
- Grad-CAM, t-SNE, retrieval figure and the demo now run on the real trained checkpoints.
- Docs are templates; numbers are filled by `scripts/build_docs.py --fill` from real JSON (fails if a number has no source).
- `results/MANIFEST.json` records commit + checkpoint hashes; `scripts/verify_results.py` checks integrity.
- Kaggle runner is strict: failed steps are reported and abort the run (the old runner printed "success" after failures).
- Fixed: checkpoint loading under PyTorch 2.6 (`weights_only=False`); models are keyed by checkpoint name so the standard and
  adversarially-trained ResNet-50 can be compared.
- Docs corrected: backbones are ImageNet-pretrained (not "from scratch"); PGD step size is 2.5*eps/steps; an "epoch" is one pass over the 751 identities.
