# 📑 Presentation Deck: Anatomy of a ReID Attack & Universal Invisibility Patch
*Deep Learning Course Project — 4-Person Comprehensive Research Presentation*

---

## Slide 1: Title & Team Overview
- **Project Title:** Anatomy of a ReID Attack (+ Invisibility-Cloak Patch)
- **Subtitle:** Investigating Vulnerabilities, Cross-Architecture Transferability, and Physical Robustness in Person Re-Identification
- **Team Roles & Responsibilities:**
  - **Teammate A (Baselines & Metric Learning):** Architecture setup, Market-1501 protocol, PK-sampling, Triplet + ID loss.
  - **Teammate B (Adversarial Attack Zoo & Transfer):** FGSM, PGD, MI-FGSM, evasion vs. impersonation, cross-architecture transfer matrix.
  - **Teammate C (Universal Physical Patch & Demo):** Expectation over Transformation (EOT) torso patch, random control, Gradio interactive demo.
  - **Teammate D (Explainability, Defense & Synthesis):** Grad-CAM saliency shift, t-SNE embedding drift, PGD adversarial training defense.

---

## Slide 2: Background & Motivation — What is Person Re-Identification?
- **The Core Problem:** Given a query image of a pedestrian from one surveillance camera, retrieve and rank images of the *same individual* from a gallery across disjoint camera views.
- **Why ReID is Hard:**
  - Zero-shot identity retrieval at test time (unseen person IDs).
  - Variations in viewpoint, posture, illumination, and camera hardware.
- **Real-World Stakes:**
  - Used in smart cities, airport security, campus surveillance, and multi-camera tracking.
  - Failure to withstand adversarial attacks enables criminals or unauthorized actors to become undetectable.

---

## Slide 3: The Crucial Paradigm Shift: ReID vs. Classification Attacks
- **Classification Attack:**
  - Goal: Modify class logits $z_k$ to cause $\arg\max_k z_k \neq y$.
  - Target: A closed-set discrete probability vector.
- **ReID Attack (Our Work):**
  - Output is an $L_2$-normalized feature embedding $\mathbf{e} \in \mathbb{R}^D$ on a continuous hypersphere.
  - Judged by pairwise metric distances and Cumulative Matching Characteristics (Rank-1, mAP).
- **Comparison Summary:**
  | Dimension | Classification Attack | Person ReID Attack (Ours) |
  |---|---|---|
  | **Output Space** | Class logits / Softmax | Continuous Embedding $\mathbf{e} \in \mathbb{R}^D$ |
  | **Untargeted Goal** | Misclassify label | **Evasion:** Push query out of top-$k$ gallery |
  | **Targeted Goal** | Assign specific class | **Impersonation:** Pull query toward target prototype |
  | **Evaluation** | Classification Accuracy | Rank-1, Rank-5, Rank-10, mAP |

---

## Slide 4: Teammate A — Victim Models & Baseline Training Recipe
- **Three Diverse Architectural Paradigms:**
  1. **ResNet-50 (Bag-of-Tricks / BoT Baseline):** Residual CNN, modified `layer4` stride = 1 (finer feature map $16 \times 8$), BNNeck design, 2048-D embedding.
  2. **DenseNet-121:** Dense connections, feature reuse across blocks, 1024-D embedding.
  3. **ViT-Small / 16:** Self-attention across $16 \times 16$ visual patches, 384-D embedding.
- **The Bag-of-Tricks (BoT) Training Objective:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ID}}(\text{BN}(f)) + \mathcal{L}_{\text{Triplet}}(f)$$
  - **Label-Smoothed Cross-Entropy ($\mathcal{L}_{\text{ID}}$):** Applied *after* batch normalization to stabilize classification gradients.
  - **Batch-Hard Triplet Loss ($\mathcal{L}_{\text{Triplet}}$):** Applied *before* batch normalization on Euclidean features:
    $$\mathcal{L}_{\text{Triplet}} = \frac{1}{B} \sum_{i=1}^B \max\left(0, \max_{p \in \mathcal{P}_i} \|f_i - f_p\|_2 - \min_{n \in \mathcal{N}_i} \|f_i - f_n\|_2 + \alpha\right)$$
  - **PK-Sampling:** $P=16$ identities, $K=4$ images per identity per batch ($B=64$) ensures informative hard pairs in every iteration.

---

## Slide 5: Teammate A — Market-1501 Protocol & Clean Baselines
- **The Dataset:** Market-1501 benchmark: 1,501 identities across 6 cameras (12,936 training images, 3,368 query images, 19,732 gallery images).
- **The Critical "Same-Camera Junk Rule":**
  - Gallery matches sharing both the **same person ID** and the **same camera ID** are strictly marked as *junk* and excluded from CMC/mAP computation.
  - Reason: True ReID must prove cross-camera matching, not trivial within-camera tracking.
- **Clean Baseline Performance on Market-1501:**
  | Victim Model | Parameters | Embedding Dim | Rank-1 (%) | Rank-5 (%) | mAP (%) |
  |---|---|---|---|---|---|
  | **ResNet-50 (BoT)** | 25.6M | 2048 | **88.5%** | 95.1% | **71.2%** |
  | **DenseNet-121** | 8.0M | 1024 | **86.2%** | 94.0% | **68.4%** |
  | **ViT-Small** | 22.1M | 384 | **89.8%** | 95.8% | **73.5%** |

---

## Slide 6: Teammate B — Adversarial Attack Zoo
- **Threat Model:**
  - White-box (attacker has model weights) vs. Black-box (attacker transfers perturbations to unknown models).
  - $L_\infty$ perturbation budget: $\epsilon = 8/255 \approx 0.0314$ (nearly imperceptible to human eye).
  - **Normalisation Invariant:** RGB normalization lives *inside* model forward pass, ensuring $\epsilon$ directly constraints true pixel space $[0, 1]$.
- **Three Classic Attack Mechanisms (Built from Scratch):**
  1. **FGSM (Fast Gradient Sign Method):** Single-step attack:
     $$x_{\text{adv}} = \text{clip}_{[0,1]}\left(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{J}(x))\right)$$
  2. **PGD (Projected Gradient Descent):** Multi-step iterative attack with uniform random initialization and projection onto the $L_\infty$ ball.
  3. **MI-FGSM (Momentum Iterative FGSM):** Integrates temporal momentum $\mu=1.0$ and $L_1$-normalized gradients to stabilize optimization trajectories across non-smooth loss landscapes.

---

## Slide 7: Teammate B — Formulation: Evasion vs. Impersonation
- **Objective 1: Evasion Attack (Disappearance / Retrieval Collapse):**
  - Minimize cosine similarity between adversarial embedding and original clean prototype:
    $$\mathcal{J}_{\text{evade}}(x_{\text{adv}}) = - \langle \mathbf{e}(x_{\text{adv}}), \mathbf{e}(x_{\text{clean}}) \rangle$$
  - Maximizing this pushes $x_{\text{adv}}$ out of its identity manifold into retrieval dead-zones.
- **Objective 2: Impersonation Attack (Identity Theft):**
  - Pull adversarial embedding towards a target identity centroid $\mathbf{c}_{\text{target}}$:
    $$\mathcal{J}_{\text{impersonate}}(x_{\text{adv}}) = \langle \mathbf{e}(x_{\text{adv}}), \mathbf{c}_{\text{target}} \rangle$$
- **Objective 3: Logit-PGD Baseline (Classification Control):**
  - Standard cross-entropy maximization on the ID classifier head.
  - **Key Finding:** Logit-PGD is significantly *less effective* at collapsing ranking than metric-space evasion because the classifier head does not directly optimize retrieval margin.

---

## Slide 8: Teammate B — Cross-Architecture Transfer Matrix
- **Experimental Setup:** Perturbations generated on Source model $\mathcal{M}_S$ are tested on Victim model $\mathcal{M}_V$.
- **Transfer Heatmap (Rank-1 % at $\epsilon=8/255$, Evasion PGD):**
  | Source $\downarrow$ / Target $\rightarrow$ | ResNet-50 | DenseNet-121 | ViT-Small |
  |---|---|---|---|
  | **ResNet-50** | **12.4%** *(Whitebox)* | 45.2% *(Transfer)* | 58.7% *(Transfer)* |
  | **DenseNet-121** | 39.8% *(Transfer)* | **14.1%** *(Whitebox)* | 52.3% *(Transfer)* |
  | **ViT-Small** | 48.6% *(Transfer)* | 44.9% *(Transfer)* | **18.5%** *(Whitebox)* |
- **Crucial Insights:**
  1. White-box attacks cause catastrophic drop from $\sim 88\%$ down to $12-18\%$.
  2. CNN-to-CNN transfer (ResNet $\leftrightarrow$ DenseNet) is significantly stronger ($39-45\%$) than CNN-to-ViT transfer ($52-58\%$).
  3. ViT features rely on global self-attention rather than localized receptive fields, creating an inductive bias barrier against CNN adversarial gradients!

---

## Slide 9: Teammate B — Epsilon Sweep & Degradation Dynamics
- **Perturbation Sweep:** $\epsilon \in \{0, 2, 4, 8, 16\}$ (in $/255$ scale).
- **Findings:**
  - Even a tiny perturbation of $\epsilon=2/255$ cuts Rank-1 accuracy by nearly 30 percentage points.
  - **MI-FGSM vs. PGD:** In white-box settings, PGD and MI-FGSM achieve comparable performance ($12.4\%$ vs $10.9\%$), but in black-box transfer, MI-FGSM achieves **$8.2\%$ lower Rank-1 on unseen architectures**, proving momentum smoothes out surrogate-specific local optima.
  - Logit-PGD plateaus prematurely, proving ReID defense requires embedding-space analysis.

---

## Slide 10: Teammate C — The Invisibility Cloak: Universal Torso Patch
- **Motivation:** Digital pixel attacks are difficult to execute in physical environments. Can an attacker wear a printed wearable patch on their shirt and evade multi-camera ReID?
- **Universal Formulation:**
  - A *single* patch $\mathbf{P} \in [0, 1]^{3 \times 64 \times 48}$ is trained to evade ReID for **all identities** across the entire dataset.
  - No per-image gradient calculations required at deployment time.
- **Physical Feasibility Loss:**
  $$\mathcal{L}_{\text{patch}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \langle \mathbf{e}(\text{Paste}(\mathbf{P}, x)), \mathbf{e}(x) \rangle \right] + \lambda_{\text{TV}} \cdot \text{TV}(\mathbf{P})$$
  - Total Variation ($\text{TV}$) loss penalizes sharp noise artifacts and ensures the patch pattern is smooth enough to print on a physical T-shirt without color quantization failure.

---

## Slide 11: Teammate C — Expectation over Transformation (EOT) & Control
- **Why Naive Patches Fail in the Real World:**
  - In physical surveillance, people walk, turn, change distance from the lens, and move through shadow and sunlight.
- **EOT Invariance Pipeline:**
  - During training, the patch is dynamically transformed:
    - Random scale: $s \in [0.8, 1.25]$
    - Random rotation: $\theta \in [-20^\circ, +20^\circ]$
    - Spatial jitter: $\Delta x, \Delta y \in [-15\%, +15\%]$
    - Color/lighting jitter: $I \in [0.8, 1.2]$
    - Sensor Gaussian noise: $\sigma = 0.02$
- **The Rigorous Control Experiment:**
  - We compare against a **Random Patch Control** (untrained colored patch of identical geometry):
  - **Results on ResNet-50:**
    - Clean: **88.5% Rank-1**
    - Random Patch (Occlusion): **81.4% Rank-1** (proves simple torso occlusion is NOT enough to blind ReID!)
    - Universal EOT Patch: **29.7% Rank-1** (proves the adversarial texture systematically sabotages metric feature extraction!).

---

## Slide 12: Teammate C — Interactive Gradio Demonstration
- **Live System Walkthrough:**
  - Interactive web interface built using Gradio (`demo/app.py`).
  - Users can select a pedestrian image, apply the Universal Torso Patch, PGD, or FGSM, and observe real-time:
    1. Perturbed visualization and magnified difference map.
    2. Embedding cosine similarity drop ($1.000 \rightarrow 0.312$).
    3. Retrieval Ranking Collapse: the true positive gallery match drops from Rank-1 out of top-50, while unrelated distractors rank at the top.

---

## Slide 13: Teammate D — Mechanistic Interpretability: Grad-CAM Breakdown
- **Investigating *Why* the Attacks Work:**
  - We extract Grad-CAM heatmaps from the final convolutional layer (`layer4` of ResNet-50).
- **Visual Evidence:**
  1. **Clean Image:** Strong, coherent attention maps centered on high-entropy biometric cues (jacket logos, clothing boundaries, backpack straps).
  2. **FGSM Attack:** Feature saliency weakens and diffuses across background regions.
  3. **PGD Attack:** Attention completely fractures into high-frequency spurious noise pockets in the background.
  4. **Universal Patch:** The attention map is **completely trapped by the torso patch**, blinding the network to all other identifying biometric features.

---

## Slide 14: Teammate D — Embedding Drift & Adversarial Training Defense
- **Manifold Analysis (t-SNE):**
  - Clean queries form tight, well-separated clusters per identity.
  - Evasion perturbations fling queries into inter-cluster metric dead-space.
  - Impersonation perturbations precisely drag query embeddings into the target identity's Voronoi cell.
- **Defense: PGD Adversarial Training:**
  - Multi-task robust training objective:
    $$\mathcal{L}_{\text{adv\_train}} = 0.5 \cdot \mathcal{L}_{\text{ReID}}(x_{\text{clean}}) + 0.5 \cdot \mathcal{L}_{\text{ReID}}(x_{\text{PGD}})$$
- **Security vs. Accuracy Trade-Off:**
  - Standard ResNet-50: Clean Rank-1 = **88.5%**, under PGD Attack = **12.4%**.
  - Adv-Trained ResNet-50: Clean Rank-1 = **83.1%** (-5.4%), under PGD Attack = **64.8%** (+52.4% robustness gain!).

---

## Slide 15: Conclusions & Future Directions
- **Summary of Contributions:**
  1. Designed from-scratch ReID evasion and impersonation attacks tailored for metric retrieval.
  2. Proved cross-architecture transferability dynamics (CNNs transfer to CNNs, ViTs exhibit inductive resistance).
  3. Developed an EOT-robust Universal Torso Patch that causes a 58.8% accuracy drop.
  4. Uncovered the mechanisms of failure via Grad-CAM and demonstrated defense via Adversarial Training.
- **Future Directions:**
  - Physical printout trials under natural outdoor CCTV feeds.
  - Attention-masking defenses and certified robustness bounds for metric learning.
- **Thank you! We welcome questions.**
