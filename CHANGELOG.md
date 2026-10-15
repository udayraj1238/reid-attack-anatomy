# Changelog

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
