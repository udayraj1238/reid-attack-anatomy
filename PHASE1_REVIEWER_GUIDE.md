# Reviewer Guide: Phase 1 ReID Adversarial Robustness Evaluation

Welcome to the **Phase 1** submission package for **Anatomy of an Attack on Person Re-Identification**.

This package contains 100% of the code, scripts, configurations, empirical results, publication figures, documentation, and presentation materials required to evaluate Phase 1.

---

## 1. Quick File Map & Purpose

| Directory / File | Description & Purpose |
|---|---|
| `PHASE1_REPORT.md` | **Full Technical Report**: Comprehensive academic report detailing mathematical formulations, algorithms, architectures, and empirical findings. |
| `README.md` | Main repository overview with setup instructions and project architecture. |
| `PLAN.md` | 12-day milestone tracking and 4-way group task distribution plan. |
| `reidattack/` | **Core PyTorch Package**: Modular library containing model definitions (`models.py`), metric-space loss functions (`losses.py`), Market-1501 dataset loader (`data.py`), CMC & mAP evaluation engine (`evaluate.py`), ReID attack zoo (`attacks.py`), EOT physical patch module (`patch.py`), and safe model I/O (`io.py`). |
| `scripts/` | Standalone execution pipelines for attack zoo evaluation (`run_attacks.py`), patch training (`run_patch.py`), Grad-CAM heatmaps (`generate_gradcam.py`), t-SNE projections (`plot_tsne.py`), and visualization plots. |
| `results/` | **Pre-Generated Publication Figures & Tables**: Baseline accuracy table (`baseline_table.md`), cross-architecture transfer heatmap (`transfer_matrix_evade.png`), perturbation degradation curves (`eps_degradation_curve.png`), Grad-CAM attention drift (`gradcam_analysis.png`), and t-SNE embedding drift (`tsne_embedding_drift.png`). |
| `notebooks/` | Cloud execution runners for Kaggle: `kaggle_runner.py` (unbuffered script runner) and `kaggle_phase1.ipynb` (interactive notebook). |
| `demo/app.py` | **Interactive Gradio Web Application**: Demonstrates digital query perturbation ($\epsilon \in [1, 16]$) and physical universal torso patch application with real-time similarity metrics. |
| `tests/smoke_test.py` | 60-second end-to-end CPU verification test checking loss computation, architecture forward passes, and CMC rank calculation. |
| `docs/` | Examination materials including 15 Technical Viva Q&As (`VIVA_QA_SHEET.md`), full presentation script (`SPEAKER_SCRIPT.md`), and presentation slides (`SLIDES.md`). |

---

## 2. How to Review & Reproduce

### Step 1: Run the 60-Second Smoke Test (No GPU Required)
Verify that all mathematical models, loss calculations, and metric pipelines compile and run cleanly:
```bash
python tests/smoke_test.py
```
*Expected output: All 5 assertion stages pass with green checks in under 60 seconds.*

### Step 2: Launch the Interactive Demo
Experience the physical torso patch and digital adversarial attacks interactively:
```bash
pip install gradio
python demo/app.py
```
Open `http://localhost:7860` in any web browser. You can select clean synthetic or uploaded pedestrian queries, apply FGSM/PGD perturbations, overlay the universal torso patch, and observe the immediate similarity drop.

### Step 3: Inspect Empirical Benchmark Results
All benchmark results and publication-quality figures are pre-rendered in [`results/`](./results/):
- **Baseline Accuracy**: Open `results/baseline_table.md` for exact parameter counts, embedding dimensions, Rank-1, Rank-5, and mAP across all three architectures.
- **Cross-Model Vulnerability**: View `results/transfer_matrix_evade.png` for white-box vs. black-box surrogate transferability.
- **Attention Degradation**: View `results/gradcam_analysis.png` to examine how perturbations displace model focus from pedestrian torso/limbs to background clutter.
- **Latent Drift**: View `results/tsne_embedding_drift.png` to inspect 2D feature manifold collapse under attack.

---

## 3. Pre-Trained Neural Network Checkpoints

To ensure this submission zip remains lightweight, portable (< 2 MB), and easy to email or upload without 500 MB of binary neural network weights, trained `.pth` checkpoints are hosted on Kaggle and GitHub:

- **Kaggle Kernel**: [kaggle.com/code/udayraj1238/reid-attack-anatomy-runner](https://www.kaggle.com/code/udayraj1238/reid-attack-anatomy-runner)
- **GitHub Repository**: [github.com/udayraj1238/reid-attack-anatomy](https://github.com/udayraj1238/reid-attack-anatomy)

### Checkpoints Trained & Verified in Phase 1:
- `resnet50.pth`: ResNet-50 Bag-of-Tricks Baseline (86.97% Rank-1, 70.64% mAP)
- `densenet121.pth`: DenseNet-121 Baseline (77.35% Rank-1, 56.63% mAP)
- `vit_small.pth`: ViT-Small/16 Baseline (91.75% Rank-1, 79.69% mAP)
- `resnet50_adv_trained_eps8.pth`: Adversarially Robust ResNet-50 Defense Model (74.52% Rank-1, 52.28% mAP)
