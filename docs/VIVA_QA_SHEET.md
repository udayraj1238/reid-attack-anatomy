# 🎓 Master Viva Voce Q&A Preparation Sheet
**Project:** Anatomy of a ReID Attack (+ Invisibility-Cloak Patch)  
**Target:** Final Deep Learning Viva / Oral Defense  
**Structure:** 25 Comprehensive Technical Questions & In-Depth Expert Answers split across all 4 Team Roles.

---

## 🏛️ General & Conceptual Questions (For Any Team Member)

### Q1: Why can't we simply treat Person Re-Identification as a 1000-class image classification problem?
**Ideal Answer:**
> "Image classification operates in a **closed-set** paradigm: all test classes must be explicitly seen during training. In contrast, Person Re-Identification is an **open-set, zero-shot metric learning retrieval problem**. In real-world surveillance (and on the Market-1501 benchmark), the people appearing in the query and gallery at test time are completely different individuals from the training set.  
> If we only trained a classifier with softmax output, the network would only know how to output probabilities for the training identities. ReID requires training a feature extractor that maps arbitrary pedestrian images into a metric embedding space $\mathbb{R}^D$ where $L_2$ distance reflects identity similarity."

### Q2: Why is the Market-1501 "same-camera junk rule" necessary? What happens if you omit it?
**Ideal Answer:**
> "In Market-1501, pedestrian images are captured across 6 different camera views. If a query pedestrian is photographed by Camera 1 at time $t$, and the gallery contains another image of the same pedestrian from Camera 1 at time $t+1$, matching them would simply be trivial background and pose tracking within the same physical camera.  
> ReID is designed to evaluate **cross-camera association**. Therefore, any gallery image that shares both the same person ID and the same camera ID as the query is designated as 'junk'. It is strictly omitted from the retrieved ranking list: it is neither counted as a true positive nor penalized as a false positive. If omitted, your reported Rank-1 and mAP would be artificially inflated by single-camera near-duplicate matches."

### Q3: Why is RGB normalization placed *inside* the model forward pass instead of torchvision dataset transforms?
**Ideal Answer:**
> "If normalization (subtracting ImageNet mean and dividing by standard deviation) is performed in the DataLoader transform, the tensors passed to the attack function are no longer in the range $[0, 1]$. Because the standard deviation for the blue channel is $\approx 0.225$ and the red channel is $\approx 0.229$, an $L_\infty$ perturbation budget of $\epsilon = 8/255$ in normalized space would correspond to unequal, distorted perturbation budgets across the three color channels in real pixel space!  
> By keeping dataset images strictly in raw $[0, 1]$ pixel values and performing normalization as the very first buffer operation inside `model.forward()`, an attack constraint of $\|x_{\text{adv}} - x\|_\infty \le \epsilon$ and clamping to $[0, 1]$ guarantees that $\epsilon$ corresponds exactly to actual physical RGB pixel shifts."

---

## 👤 Questions for Person A (Baselines, Metric Learning & Data Pipeline)

### Q4: Why do we use the BNNeck structure in the Bag-of-Tricks (BoT) baseline?
**Ideal Answer:**
> "In standard metric learning, there is a known tension between Cross-Entropy ID loss and Triplet loss. Triplet loss works best on unconstrained Euclidean feature space $f \in \mathbb{R}^D$ to pull positive pairs together and push negative pairs apart. However, Cross-Entropy loss with a linear classification weight $W$ requires features to be centered and scaled; otherwise, large-norm features dominate the softmax gradients.  
> The **BNNeck** (Batch Normalization Neck) resolves this:
> 1. Triplet loss is computed on the pre-BN features $f$, preserving Euclidean metric distance constraints.
> 2. The features pass through a 1D Batch Normalization layer with no bias term ($f_t = \text{BN}(f)$).
> 3. Classification Cross-Entropy is computed on $f_t$. The BN layer centers the features on a hypersphere, stabilizing softmax gradients without distorting the underlying Euclidean metric space."

### Q5: Explain how the batch-hard triplet loss works and why PK-sampling is mandatory.
**Ideal Answer:**
> "Standard triplet loss selects random $(a, p, n)$ triplets, the vast majority of which become trivial (zero loss) early in training. **Batch-hard triplet loss** (Hermans et al.) dynamically mines the most informative pairs within each mini-batch:
> $$\mathcal{L}_{\text{triplet}} = \frac{1}{B} \sum_{i=1}^B \max\left(0, \max_{p \in \mathcal{P}_i} \|f_i - f_p\|_2 - \min_{n \in \mathcal{N}_i} \|f_i - f_n\|_2 + \alpha\right)$$
> For each anchor $i$, it finds the hardest positive (the furthest image of the same identity) and the hardest negative (the closest image of a different identity).  
> If we used a standard random sampler, a batch of size 64 might only contain 1 or 2 images per person, meaning many anchors would have zero positive pairs in the batch! The **PK-Sampler** guarantees that every batch contains exactly $P$ distinct identities and $K$ images per identity (we use $P=16, K=4, B=64$), ensuring every sample has multiple positive candidates to mine from."

### Q6: What architectural modifications were made to ResNet-50 for ReID?
**Ideal Answer:**
> "Standard ResNet-50 applies a stride of 2 in the first convolution of `layer4`, reducing the spatial resolution of the feature map from $14 \times 7$ down to $7 \times 4$. For fine-grained pedestrian re-identification, a $7 \times 4$ spatial grid loses critical spatial details (such as shoe color or belt patterns).  
> Following the BoT recipe, we set the stride of `layer4[0].conv2` and its residual shortcut downsample to $1 \times 1$. This preserves a larger spatial feature map of $16 \times 8$ before Global Average Pooling (GAP), which consistently yields a 2–3% improvement in Rank-1 retrieval accuracy."

---

## 👤 Questions for Person B (Attack Zoo, Objectives & Transferability)

### Q7: Mathematically compare our ReID Evasion Objective against standard Classification Cross-Entropy attack.
**Ideal Answer:**
> "A classification attack maximizes cross-entropy with respect to the true label $y$:
> $$\max_{x_{\text{adv}}} \mathcal{L}_{\text{CE}}(f(x_{\text{adv}}), y) = - \log \left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)$$
> In ReID, test images have no target classes. Our **Evasion Objective** operates directly on the $L_2$-normalized unit hypersphere $\mathbb{S}^{D-1}$:
> $$\mathcal{J}_{\text{evade}}(x_{\text{adv}}) = - \langle \mathbf{e}(x_{\text{adv}}), \mathbf{e}(x_{\text{clean}}) \rangle = - \frac{\mathbf{e}(x_{\text{adv}})^\top \mathbf{e}(x_{\text{clean}})}{\|\mathbf{e}(x_{\text{adv}})\|_2 \|\mathbf{e}(x_{\text{clean}})\|_2}$$
> Maximizing $\mathcal{J}_{\text{evade}}$ is equivalent to minimizing cosine similarity, which directly drives the perturbed embedding orthogonal or antipodal to its original representation. Our experiments proved that attacking the classification logits (`logit_pgd`) only drops Rank-1 to 52.3%, whereas minimizing cosine similarity (`evade_pgd`) collapses Rank-1 to 12.4%."

### Q8: Why does MI-FGSM achieve superior cross-architecture transferability compared to PGD?
**Ideal Answer:**
> "Standard PGD evaluates the gradient $\nabla_x \mathcal{J}(x_t)$ at each step and takes a step along its sign. However, deep neural network loss surfaces are non-smooth and contain many sharp, surrogate-specific local maxima and oscillations. Multi-step PGD tends to overfit to these local high-frequency peaks of the source model, resulting in poor black-box transferability to a different architecture.  
> **MI-FGSM** incorporates a velocity vector (momentum $\mu=1.0$) on $L_1$-normalized gradients:
> $$g_{t+1} = \mu \cdot g_t + \frac{\nabla_x \mathcal{J}(x_t)}{\|\nabla_x \mathcal{J}(x_t)\|_1}$$
> The momentum acts as a low-pass filter over the optimization path, smoothing out jagged local gradient fluctuations and aligning the update vector along the broader, architecture-invariant adversarial direction."

### Q9: Why was cross-architecture transfer weaker from CNNs (ResNet/DenseNet) to Vision Transformers (ViT)?
**Ideal Answer:**
> "This boils down to **inductive bias and receptive fields**:
> 1. CNNs (ResNet, DenseNet) rely on translation invariance and strictly localized $3 \times 3$ convolutional filters. Adversarial gradients on CNNs tend to exploit high-frequency local textural edges.
> 2. Vision Transformers partition the image into non-overlapping patches ($16 \times 16$) and rely on global multi-head self-attention across the entire visual token sequence from the very first layer.  
> Because ViT forms representations based on global semantic relationships rather than localized convolutional kernel textures, the local high-frequency patterns crafted by CNN surrogates do not disrupt ViT self-attention maps as severely. This creates an architectural barrier against transfer."

---

## 👤 Questions for Person C (Universal Physical Patch & EOT)

### Q10: How does a Universal Adversarial Perturbation (UAP) differ from an input-dependent attack?
**Ideal Answer:**
> "An input-dependent attack (like standard FGSM or PGD) computes $\nabla_x \mathcal{J}(x)$ for a specific image $x$. It requires access to the input image, multiple backward passes, and crafts a bespoke noise tensor $\delta_x$ that only works on that single image.  
> A **Universal** attack solves an empirical risk minimization problem over the entire data distribution:
> $$\min_{\mathbf{P}} \mathbb{E}_{x \sim \mathcal{D}} \left[ \langle \mathbf{e}(\text{Paste}(\mathbf{P}, x)), \mathbf{e}(x) \rangle \right]$$
> Once trained, this single static patch $\mathbf{P}$ can be pasted onto *any* unseen person's photo at test time without computing a single gradient or backward pass. It acts as an offline, universal bypass."

### Q11: What is Expectation over Transformation (EOT), and why is it essential for physical world attacks?
**Ideal Answer:**
> "If you optimize a patch on fixed pixel coordinates, the optimizer creates high-frequency pixel alignments that are extremely brittle. The moment a person takes a step forward (scale change), tilts their torso (rotation), or walks under daylight (illumination shift), the physical projection changes and the naive patch completely fails.  
> **Expectation over Transformation (EOT)** (Athalye et al.) models the physical distribution of transformations $\mathcal{T}$. During each forward pass of training, we sample a composite transformation $t \sim \mathcal{T}$ (scale $s \in [0.8, 1.25]$, rotation $\theta \in [-20^\circ, 20^\circ]$, translation $\Delta x, \Delta y$, color jitter $I$, and sensor noise $\sigma$):
> $$\mathcal{L}_{\text{EOT}} = \mathbb{E}_{x \sim \mathcal{D}, t \sim \mathcal{T}} \left[ \mathcal{L}(\text{Paste}(t(\mathbf{P}), x)) \right]$$
> By backpropagating through differentiable affine bilinear grids, the patch learns spatial patterns that remain adversarial regardless of camera angle or viewpoint."

### Q12: Why did you include a Random Patch Control, and what did it prove?
**Ideal Answer:**
> "Whenever you occlude part of an image with a patch, critics may argue: *'Of course retrieval dropped—you covered up 20% of their body!'*  
> To disprove this and isolate the true adversarial effect, we implemented an untrained **Random Patch Control** of identical physical dimensions ($64 \times 48$).  
> On ResNet-50:
> - Clean retrieval: **88.5% Rank-1**
> - Random Patch (pure physical occlusion): **81.4% Rank-1** (only a 7.1% drop, proving ReID models are resilient to occlusion because they utilize head, lower legs, and context).
> - Universal EOT Patch: **29.7% Rank-1** (a massive 58.8% drop!).  
> This proves conclusively that the performance collapse is driven by active adversarial feature corruption, not occlusion."

---

## 👤 Questions for Person D (Explainability, Defenses & Robustness)

### Q13: What does Grad-CAM reveal about why Person ReID models fail under adversarial attacks?
**Ideal Answer:**
> "Grad-CAM visualizes the weighted combination of forward feature activation maps using the gradients flowing back from the metric score:
> $$L_{\text{Grad-CAM}}^k = \text{ReLU}\left(\sum_k \alpha_k A^k\right), \quad \text{where } \alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial \mathcal{S}}{\partial A_{i,j}^k}$$
> Our analysis revealed three distinct failure modes:
> 1. **Under Clean Conditions:** Attention is sharply concentrated on human body semantics—torso patterns, logos, and leg silhouettes.
> 2. **Under PGD Noise:** The attention completely disperses into background noise artifacts, indicating that the model's high-level filters are hijacked by high-frequency pixel perturbations.
> 3. **Under the Universal Patch:** The entire spatial activation field collapses into the patch rectangle. The patch acts as an 'attention sink', starving the rest of the pedestrian's body of feature representation."

### Q14: How does Adversarial Training work for Person ReID, and why does it hurt clean accuracy?
**Ideal Answer:**
> "In ReID, adversarial training incorporates both clean and PGD-perturbed images in each batch:
> $$\mathcal{L}_{\text{defense}} = 0.5 \cdot \mathcal{L}_{\text{ReID}}(x_{\text{clean}}) + 0.5 \cdot \mathcal{L}_{\text{ReID}}(x_{\text{PGD}})$$
> The model learns to map both $x$ and $x + \delta$ to the same identity subspace.  
> The **clean accuracy penalty** (in our case, clean Rank-1 dropped from 88.5% to 83.1%) is an established theoretical trade-off known as the *odds of robustness vs. accuracy* (Tsipras et al.). Standard training allows the network to exploit fragile, non-robust high-frequency features that happen to correlate with identity in the training set. Adversarial training forces the network to ignore these non-robust features and rely solely on robust, low-frequency geometric patterns, which slightly reduces peak clean discriminability but provides massive resilience (+52.4% under attack)."

### Q15: If you cannot retrain the model, what test-time defenses could you deploy against the physical patch?
**Ideal Answer:**
> "Several inference-time defenses exist:
> 1. **Spatial Local Outlier Detection / Patch Inpainting:** Because adversarial patches have high Total Variation or anomalous local activation gradients, a local spatial anomaly detector can identify the patch bounding box, mask it out, and perform feature extraction on the remaining body parts.
> 2. **Part-Based Pooling (PCB):** Instead of global average pooling over the entire image, divide the feature map into 6 horizontal stripes. If the torso stripe is attacked, the head, upper-leg, and shoe stripes still yield uncontaminated partial identity vectors.
> 3. **Randomized Smoothing or Bit-Depth Reduction:** Preprocessing inputs with random cropping or JPEG compression destroys high-frequency digital PGD perturbations."

---

## 💡 Rapid-Fire Technical Drill

| Question | Short Punchy Answer |
|---|---|
| **What was your perturbation budget?** | $L_\infty$ norm with $\epsilon = 8/255 \approx 0.0314$ in $[0, 1]$ pixel space. |
| **What optimizer and learning rate did you use?** | Adam optimizer with initial LR of $3.5 \times 10^{-4}$, 5-epoch linear warmup, and multistep decay at 50% and 80% of epochs. |
| **What margin did you use in Triplet Loss?** | An Euclidean margin of $\alpha = 0.3$. |
| **How many identities are in Market-1501 training?** | 751 training identities; 750 query/gallery identities across 6 cameras. |
| **How does mAP differ from Rank-1?** | Rank-1 only checks if the single top-ranked image is correct; mAP calculates the area under the Precision-Recall curve across *all* matching gallery instances, reflecting total retrieval quality. |
