# Anatomy of a ReID Attack (+ Invisibility-Cloak Patch)

A from-scratch study of **adversarial attacks on person re-identification (ReID)**: *why* they work,
*why* they transfer between architectures, and how a single printable **universal patch** can make a
person vanish from a retrieval system. Deep Learning course project.

## Why ReID attacks are different from classification attacks
A classifier outputs one label; a ReID system outputs an **embedding** and is judged by the **ranking**
of gallery images. So the attack objective changes:

| | Classification | ReID (this repo) |
|---|---|---|
| Target of the attack | class logits | embedding distance / retrieval ranking |
| Untargeted goal | wrong label | **evasion**: true matches fall out of the top-k |
| Targeted goal | chosen label | **impersonation**: rank as a chosen identity |
| Metric | accuracy | Rank-1, mAP (Market-1501 protocol) |

We implement both objectives and a classification-style `logit_pgd` baseline to show the difference.

## What is inside
- `reidattack/` - data loader, 3 victim models (ResNet-50, DenseNet-121, ViT-Small; BNNeck baseline), triplet+ID loss, Market-1501 CMC/mAP evaluation, **FGSM / PGD / MI-FGSM from scratch**, **Universal EOT Patch**, and **Adversarial Training Defence**
- `scripts/`
  - `run_attacks.py` - attack zoo + cross-architecture transfer matrix (evasion & impersonation)
  - `run_patch.py` - trains the torso patch, compares against a random-patch control, tests transfer
  - `plot_transfer_matrix.py` - creates the transfer matrix heatmap
  - `plot_eps_curve.py` - generates epsilon degradation curves
  - `generate_gradcam.py` - creates Grad-CAM saliency disruption figures
  - `plot_tsne.py` - generates 2D t-SNE embedding drift visualization
  - `make_baseline_table.py` - generates the clean baseline comparison table
- `demo/app.py` - interactive **Gradio web demo** (upload/sample image, apply patch/attacks, view ranking collapse)
- `tests/smoke_test.py` - 1-minute sanity check (no dataset needed)
- `notebooks/kaggle_phase1.ipynb` - end-to-end Kaggle runner (Steps 1 to 6)
- `docs/`
  - `SLIDES.md` - Master 15-slide presentation deck covering all 4 team roles
  - `SPEAKER_SCRIPT.md` - 12-minute word-for-word spoken transcript with slide transitions
  - `VIVA_QA_SHEET.md` - 25-question in-depth Viva Voce examination Q&A guide
- `results/` - generated heatmaps, curves, Grad-CAM, t-SNE plots, and baseline table
- `PLAN.md` - 12-day milestone plan & 4-way team task partition

---

## 👥 4-Way Team Work Breakdown
| Member | Focus Area | Key Deliverables |
|---|---|---|
| **Person A** | Baselines & Metric Learning | ResNet-50, DenseNet-121, ViT-Small, PK-Sampler, Triplet+ID loss, Market-1501 evaluation, `results/baseline_table.md` |
| **Person B** | Attacks & Cross-Model Transfer | FGSM, PGD, MI-FGSM, Evasion & Impersonation, Transfer matrix heatmap (`results/transfer_matrix_evade.png`), Epsilon sweep (`results/eps_degradation_curve.png`) |
| **Person C** | Universal Patch & Interactive Demo | Universal Torso Patch with EOT, Random Patch control, `scripts/run_patch.py`, Gradio web demo (`demo/app.py`) |
| **Person D** | Explainability, Defence & Presentation | Grad-CAM (`results/gradcam_analysis.png`), t-SNE drift (`results/tsne_embedding_drift.png`), Adversarial Training (`reidattack/adv_train.py`), `docs/SLIDES.md`, `docs/SPEAKER_SCRIPT.md`, `docs/VIVA_QA_SHEET.md` |

---

## 🚀 Quick start (free Kaggle T4)
```bash
pip install -r requirements.txt
python tests/smoke_test.py                                   # sanity check
python -m reidattack.train --data <market1501_dir> --arch resnet50 --epochs 30
python scripts/run_attacks.py --data <market1501_dir> --ckpts checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth
python scripts/run_patch.py   --data <market1501_dir> --src checkpoints/resnet50.pth --others checkpoints/densenet121.pth
python -m reidattack.adv_train --data <market1501_dir> --arch resnet50 --eps 8 --epochs 20
```

---

## 🎨 Interactive Gradio Demo
```bash
pip install gradio
python demo/app.py
```
Open `http://127.0.0.1:7860` to interactively upload images, toggle attacks/patches, and observe retrieval collapse in real-time.

---

## 📊 Key Findings & Results
- **Clean Baselines:** ResNet-50 (**88.5% Rank-1**, 71.2% mAP), DenseNet-121 (86.2% Rank-1), ViT-Small (**89.8% Rank-1**, 73.5% mAP).
- **White-box Attacks:** Drops ResNet-50 to **12.4% Rank-1** (PGD $\epsilon=8/255$).
- **Transferability:** CNNs transfer strongly to CNNs (45.2% Rank-1), but Vision Transformers exhibit natural resistance (58.7% Rank-1) due to global patch self-attention.
- **Universal Patch:** Drops Rank-1 to **29.7%**, while a Random Patch Control only drops to **81.4%**, proving true adversarial corruption.
- **Adversarial Training Defense:** Boosts robust Rank-1 under PGD attack from **12.4% to 64.8%** (+52.4% gain).

---

## 🎓 Presentation & Viva Voce
- [Presentation Slides (15 Slides)](docs/SLIDES.md)
- [Word-for-Word Speaker Script (12 Minutes)](docs/SPEAKER_SCRIPT.md)
- [Comprehensive Viva Voce Q&A Sheet (25 Questions)](docs/VIVA_QA_SHEET.md)
