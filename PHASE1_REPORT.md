# Phase 1 Final Report: Anatomy of an Attack on Person Re-Identification

**Project**: Deep Learning Capstone / Adversarial Robustness  
**Repository**: [github.com/udayraj1238/reid-attack-anatomy](https://github.com/udayraj1238/reid-attack-anatomy)  
**Dataset**: Market-1501 Benchmark (1,501 identities, 32,668 images, 6 cameras)  
**Hardware & Environment**: Kaggle Cloud (NVIDIA T4 GPU 16GB, PyTorch 2.6+, CUDA 12.4)  
**Status**: **Phase 1 100% Completed & Empirically Verified**

---

## 1. Executive Summary

Person Re-Identification (ReID) is an open-set retrieval problem where surveillance systems match identities across disjoint camera networks. Unlike traditional closed-set image classification where models output fixed category probabilities, ReID systems operate in a continuous **metric embedding space** ($\mathbb{R}^D$).

In **Phase 1**, we designed, implemented, and empirically benchmarked:
1. **Three Diverse Victim Baselines**: ResNet-50 (Bag of Tricks), DenseNet-121, and Vision Transformer (ViT-Small/16) trained from scratch on Market-1501 with mixed ID and batch-hard triplet loss.
2. **ReID-Specific Adversarial Attack Zoo**: Mathematical reformulation of gradient-based attacks (FGSM, PGD, MI-FGSM, Logit-PGD) tailored for pairwise metric spaces under both **Evasion** (identity concealment) and **Impersonation** (targeted identity takeover) objectives.
3. **Physical-World Universal Torso Patch**: EOT-optimized (Expectation Over Transformation) universal printable patch designed to disrupt surveillance matching across unknown viewpoints, scales, and lighting.
4. **Adversarial Training Defense**: Robustified ResNet-50 model trained against on-the-fly PGD perturbations ($\epsilon = 8/255$), quantifying the fundamental clean accuracy vs. robustness trade-off.
5. **Interpretability & Feature Drift Visualizations**: Publication-grade baseline tables, transfer heatmaps, epsilon degradation sweeps, Grad-CAM attention maps, and 2D t-SNE feature space projections.

All models reached strong benchmark convergence on Market-1501, led by **ViT-Small (91.75% Rank-1, 79.69% mAP)** and **ResNet-50 (86.97% Rank-1, 70.64% mAP)**.

---

## 2. Problem Formulation & ReID Metric Space Mathematics

### 2.1 The Open-Set ReID Challenge
In surveillance, query identities $\mathcal{Y}_{\text{test}}$ never overlap with training identities $\mathcal{Y}_{\text{train}}$ ($\mathcal{Y}_{\text{train}} \cap \mathcal{Y}_{\text{test}} = \emptyset$). Consequently, models cannot rely on fixed output classification heads at inference.

Given a query image $\mathbf{x}_q$ and gallery set $\mathcal{G} = \{(\mathbf{g}_i, y_i, c_i)\}_{i=1}^N$, the network extracts $L_2$-normalized feature representations:
$$\mathbf{f}_q = \frac{\phi(\mathbf{x}_q)}{\|\phi(\mathbf{x}_q)\|_2} \in \mathbb{R}^D, \quad \mathbf{f}_g = \frac{\phi(\mathbf{g}_i)}{\|\phi(\mathbf{g}_i)\|_2} \in \mathbb{R}^D$$

Retrieval ranking is determined by pairwise Euclidean distance in the unit hypersphere:
$$d(\mathbf{f}_q, \mathbf{f}_g) = \|\mathbf{f}_q - \mathbf{f}_g\|_2 = \sqrt{2 - 2 \langle \mathbf{f}_q, \mathbf{f}_g \rangle}$$

### 2.2 Combined Loss Formulation
To build a discriminative metric space, models are trained with a joint loss function:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ID}} + \mathcal{L}_{\text{triplet}}$$

1. **Identification Loss (Cross-Entropy with Label Smoothing)**:
   $$\mathcal{L}_{\text{ID}} = - \sum_{k=1}^C q_k \log \frac{e^{\mathbf{w}_k^T \mathbf{f} + b_k}}{\sum_{j=1}^C e^{\mathbf{w}_j^T \mathbf{f} + b_j}}$$
   where $q_k = 1 - \epsilon_{\text{ls}}$ for the ground-truth identity and $\frac{\epsilon_{\text{ls}}}{C}$ otherwise ($\epsilon_{\text{ls}} = 0.1$).

2. **Batch-Hard Triplet Loss with Soft Margin**:
   Each training mini-batch is constructed using a **PK Sampler** ($P = 16$ identities, $K = 4$ images each, $B = 64$). For every anchor $a$, the hardest positive and hardest negative within the batch are mined:
   $$\mathcal{L}_{\text{triplet}} = \sum_{a=1}^B \left[ \max_{p \in \mathcal{P}_a} \|\mathbf{f}_a - \mathbf{f}_p\|_2 - \min_{n \in \mathcal{N}_a} \|\mathbf{f}_a - \mathbf{f}_n\|_2 + \alpha \right]_+$$
   where $\alpha = 0.3$ is the enforcement margin.

### 2.3 Evaluation Protocol & Same-Camera Junk Rule
Evaluation follows the official Market-1501 standard:
- **Cumulative Matching Characteristic (CMC)**: Computes the probability that a true positive identity appears in the top-$k$ retrieved ranks (Rank-1, Rank-5, Rank-10).
- **Mean Average Precision (mAP)**: Evaluates recall across multi-camera captures:
  $$\text{AP} = \sum_{k=1}^N P(k) \cdot \Delta R(k), \quad \text{mAP} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{AP}(q)$$
- **Same-Camera Junk Rule**: Gallery images sharing both identical person ID and identical camera ID ($y_g = y_q \land c_g = c_q$) are marked as junk and removed from the ranking list to prevent artificial inflation from static viewpoint correlation.

---

## 3. Victim Architectures & Implementation

We selected three distinct structural paradigms to assess cross-architecture vulnerability and transferability:

| Architecture | Paradigm | Parameter Count | Embedding Dim ($D$) | Key Architectural Features |
|---|---|---|---|---|
| **ResNet-50 (BoT)** | Deep Residual CNN | 25.6M | 2048 | Last conv stride=1 (preserves 16x8 spatial resolution), GAP, BNNeck |
| **DenseNet-121** | Dense Feature Reuse | 8.0M | 1024 | Direct concatenation across layers, high feature reuse, low parameter footprint |
| **ViT-Small / 16** | Vision Transformer | 22.1M | 384 | 16x16 patch projection, [CLS] token, 12 self-attention heads, global receptive field |

---

## 4. Empirical Benchmark Results (Market-1501)

The models were trained on Kaggle using NVIDIA T4 GPU accelerators. All training metrics were logged with unbuffered batch monitoring and evaluated against the full Market-1501 query (3,368 images) and gallery (15,913 images) test partitions:

```
=============================================================================================
Architecture        Params    Emb Dim   Epochs   Rank-1 (%)   Rank-5 (%)   mAP (%)    Train Time
=============================================================================================
ViT-Small / 16      22.1M     384       30       91.75%       96.67%       79.69%     6.5 mins
ResNet-50 (BoT)     25.6M     2048      30       86.97%       95.40%       70.64%     9.6 mins
DenseNet-121        8.0M      1024      30       77.35%       91.95%       56.63%     7.5 mins
---------------------------------------------------------------------------------------------
Adv-ResNet50 (ε=8)  25.6M     2048      20       74.52%       —            52.28%     40.3 mins
=============================================================================================
```

### Key Empirical Findings:
1. **ViT Superiority**: The Vision Transformer outperformed ResNet-50 by **+4.78% Rank-1** and **+9.05% mAP**, establishing that global self-attention across non-overlapping image patches captures fine-grained pedestrian attributes (e.g. shoes, backpacks, clothing patterns) significantly better than local CNN convolutions.
2. **DenseNet Parameter Efficiency**: DenseNet-121 achieved solid performance (77.35% Rank-1) with only 31% of ResNet-50's parameters (8.0M vs. 25.6M), making it suitable for edge deployment.
3. **Defense Trade-off**: ResNet-50 with PGD adversarial training retained **74.52% clean Rank-1**, illustrating the characteristic trade-off between clean accuracy and perturbed robustness.

---

## 5. Adversarial Attack Zoo Formulation

Unlike classification attacks that maximize cross-entropy on static labels, ReID attacks operate on **relative pairwise distance in continuous representation space**.

```
              Clean Query x
                    │
            ┌───────┴───────┐
            ▼               ▼
     [Evasion Attack]   [Impersonation Attack]
            │               │
            ▼               ▼
   Maximize distance    Minimize distance
    from true self        to target identity
   d(f(x_adv), f(x))    d(f(x_adv), f_target)
            │               │
            ▼               ▼
   Conceal identity     Hijack identity
```

### 5.1 Evasion Objective (Identity Concealment)
Forces the perturbed query embedding $\mathbf{f}(\mathbf{x}_{\text{adv}})$ to become orthogonal or opposing to its original clean representation $\mathbf{f}(\mathbf{x})$:
$$\mathcal{L}_{\text{evade}}(\mathbf{x}_{\text{adv}}) = - \langle \mathbf{f}(\mathbf{x}_{\text{adv}}), \mathbf{f}(\mathbf{x}) \rangle$$
$$\mathbf{x}_{\text{adv}} = \mathbf{x} + \delta, \quad \|\delta\|_\infty \le \epsilon$$

### 5.2 Impersonation Objective (Targeted Hijacking)
Pulls the query embedding toward an attacker-selected target identity prototype centroid $\mathbf{c}_t = \frac{1}{|\mathcal{G}_t|} \sum_{\mathbf{g} \in \mathcal{G}_t} \mathbf{f}(\mathbf{g})$:
$$\mathcal{L}_{\text{impersonate}}(\mathbf{x}_{\text{adv}}) = + \langle \mathbf{f}(\mathbf{x}_{\text{adv}}), \mathbf{c}_t \rangle$$

### 5.3 Implemented Attack Algorithms
1. **Fast Gradient Sign Method (FGSM)**:
   $$\mathbf{x}_{\text{adv}} = \text{clip}_{[\mathbf{0},\mathbf{1}]} \left( \mathbf{x} + \epsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(\mathbf{x})) \right)$$
2. **Projected Gradient Descent (PGD-$K$)**:
   $$\mathbf{x}^{t+1} = \Pi_{\mathbf{x} + \mathcal{S}} \left( \mathbf{x}^t + \alpha \cdot \text{sign}(\nabla_{\mathbf{x}^t} \mathcal{L}(\mathbf{x}^t)) \right), \quad \alpha = \frac{2\epsilon}{K}$$
3. **Momentum Iterative FGSM (MI-FGSM)**:
   $$\mathbf{g}_{t+1} = \mu \cdot \mathbf{g}_t + \frac{\nabla_{\mathbf{x}^t} \mathcal{L}(\mathbf{x}^t)}{\|\nabla_{\mathbf{x}^t} \mathcal{L}(\mathbf{x}^t)\|_1}$$
   $$\mathbf{x}^{t+1} = \Pi_{\mathbf{x} + \mathcal{S}} \left( \mathbf{x}^t + \alpha \cdot \text{sign}(\mathbf{g}_{t+1}) \right)$$
4. **Logit-PGD Baseline**:
   Maximizes classification cross-entropy on the ID classifier head, providing an empirical comparison between classification attacks and true metric-space attacks.

---

## 6. Physical Universal Torso Invisibility Patch (EOT)

For real-world surveillance circumvention, digital pixel-level perturbations are impractical because security cameras sample footage through variable lighting, lens distortions, and physical distance.

### 6.1 Expectation Over Transformation (EOT) Framework
We train a single universal patch $P \in [0, 1]^{3 \times 64 \times 48}$ positioned on the pedestrian torso $(c_x, c_y) = (0.0, -0.05)$:
$$\min_P \mathbb{E}_{\mathbf{x} \sim \mathcal{X}, T \sim \mathcal{T}} \left[ \langle \phi(T(\mathbf{x}, P)), \phi(\mathbf{x}) \rangle + \lambda \mathcal{L}_{\text{TV}}(P) \right]$$

Transformations $T$ sampled during each forward pass:
- **Spatial Rotation**: $\theta \sim \mathcal{U}(-20^\circ, +20^\circ)$
- **Scale Variation**: $s \sim \mathcal{U}(0.8, 1.25)$
- **Torso Translation**: $\Delta x, \Delta y \sim \mathcal{U}(-0.15, +0.15)$
- **Photometric Lighting Jitter**: $b \sim \mathcal{U}(0.8, 1.2)$
- **Sensor Noise**: $\eta \sim \mathcal{N}(0, 0.02^2)$

### 6.2 Total Variation (TV) Regularization
To ensure physical printability without chaotic high-frequency noise that cameras cannot capture:
$$\mathcal{L}_{\text{TV}}(P) = \frac{1}{HW} \sum_{i,j} \left( |P_{i+1, j} - P_{i, j}| + |P_{i, j+1} - P_{i, j}| \right)$$

---

## 7. Artifact Deliverables Summary

All Phase 1 code, models, and figures are organized within the repository structure:

```
reid-attack-anatomy/
├── checkpoints/               # Trained neural network weights (.pth)
│   ├── resnet50.pth           # ResNet-50 baseline (70.64% mAP)
│   ├── densenet121.pth        # DenseNet-121 baseline (56.63% mAP)
│   ├── vit_small.pth          # ViT-Small/16 baseline (79.69% mAP)
│   └── resnet50_adv_trained_eps8.pth # Robust ResNet-50 defense model
├── results/                   # Generated publication figures
│   ├── baseline_table.md      # Markdown baseline table
│   ├── transfer_matrix_evade.png # Cross-model transfer heatmap
│   ├── eps_degradation_curve.png # Epsilon degradation curve
│   ├── gradcam_analysis.png   # Grad-CAM attention degradation
│   └── tsne_embedding_drift.png # 2D t-SNE representation collapse
├── reidattack/                # Core modular PyTorch package
│   ├── models.py              # ResNet-50, DenseNet-121, ViT-Small
│   ├── losses.py              # Batch-hard Triplet + Label-smoothed CE
│   ├── data.py                # Market-1501 loader + PK Sampler
│   ├── evaluate.py            # Rank-1/5/10 & mAP with junk rule
│   ├── attacks.py             # FGSM, PGD, MI-FGSM, Logit-PGD
│   ├── patch.py               # EOT Universal Patch module
│   ├── adv_train.py           # On-the-fly PGD adversarial training
│   ├── io.py                  # PyTorch 2.6 safe model loader
│   └── logger.py              # Persistent unbuffered dual logger
├── scripts/                   # Evaluation & plotting pipelines
│   ├── run_attacks.py         # Cross-architecture transfer matrix
│   ├── run_patch.py           # Patch training & evaluation
│   ├── make_baseline_table.py # Markdown table generation
│   ├── plot_transfer_matrix.py# Heatmap plotting
│   ├── plot_eps_curve.py      # Epsilon degradation plotting
│   ├── generate_gradcam.py    # Grad-CAM feature localization
│   ├── plot_tsne.py           # 2D t-SNE projection
│   └── monitor_kaggle.py      # Live cloud kernel polling tool
├── notebooks/                 # Cloud execution pipelines
│   ├── kaggle_runner.py       # Unbuffered standalone runner
│   └── kaggle_phase1.ipynb    # Kaggle Jupyter notebook runner
├── demo/                      # Interactive user interface
│   └── app.py                 # Gradio ReID attack simulation
├── docs/                      # Examination & Viva preparation
│   ├── VIVA_QA_SHEET.md       # 15 technical interview Q&As
│   ├── SPEAKER_SCRIPT.md      # Presentation script
│   └── SLIDES.md              # 10 presentation slides
└── tests/
    └── smoke_test.py          # 60-second end-to-end sanity test
```

---

## 8. Verification & Reviewer Instructions

Reviewers can verify Phase 1 on any machine using the following self-contained steps:

### Quick Sanity Test (No GPU Required, < 60s)
```bash
python tests/smoke_test.py
```
*Validates batch-hard triplet mining, metric-space loss functions, model instantiation across all 3 architectures, and CMC rank-1 calculation.*

### Interactive UI Demonstration
```bash
python demo/app.py
```
*Launches local Gradio interface demonstrating pedestrian query upload, digital attack simulation ($\epsilon \in [1, 16]$), and physical torso patch overlay.*

### Cloud Replication on Kaggle
1. Upload `notebooks/kaggle_runner.py` or run `kaggle kernels push -p notebooks`.
2. Attach dataset `pengcw1/market-1501` under `/kaggle/input`.
3. Enable GPU (T4 x1 or x2) and Internet: ON.
4. All checkpoints and visualization artifacts will be exported automatically to `/kaggle/working/`.
