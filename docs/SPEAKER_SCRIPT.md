# 🎙️ Complete Presentation Speaker Script
**Project:** Anatomy of a ReID Attack (+ Invisibility-Cloak Patch)  
**Total Allocated Time:** 12 Minutes (3 Minutes per Team Member) + 3 Minutes Q&A  
**Team Members:** Person A, Person B, Person C, Person D  

---

## 👤 Speaker 1: Person A (Baselines, Dataset & Metric Learning Architecture)
**Time:** 0:00 – 3:00  
**Slides:** 1 to 5  

### [Slide 1: Title & Team Overview]
> "Good morning, Professor and members of the panel. Today, our team is presenting our deep learning research project: **Anatomy of a ReID Attack and the Invisibility-Cloak Patch**.  
>
> In this project, we conduct a rigorous, from-scratch investigation into how modern Person Re-Identification systems can be systematically deceived, why adversarial perturbations transfer across deep architectures, and how a wearable, universal patch can cause complete retrieval collapse.  
>
> Our presentation is divided into four distinct phases: I will cover the victim architectures, training protocols, and clean baseline evaluations. Person B will present our adversarial attack formulations and cross-architecture transfer findings. Person C will demonstrate the Universal Physical Patch and our interactive demo. Finally, Person D will break down the mechanistic explainability with Grad-CAM and present our adversarial defense."

### [Slide 2: Background & Motivation — What is Person Re-Identification?]
> "To begin with: Person Re-Identification, or ReID, is the task of matching pictures of pedestrians across multiple disjoint surveillance cameras over time. Unlike standard image classification, ReID is inherently an **open-set zero-shot retrieval problem**. The test identities were never seen during training. Therefore, the network cannot simply classify people into fixed buckets; it must map images into a continuous Euclidean embedding space where different photos of the same person are close together, and photos of different people are far apart."

### [Slide 3: ReID vs. Classification Attacks]
> "This fundamental distinction completely transforms the adversarial threat model. In classification, attacks maximize categorical cross-entropy to flip a single discrete class label. But in ReID, there are no test labels. The model outputs a 2048-dimensional normalized embedding vector on a unit hypersphere, and system performance is evaluated by nearest-neighbor gallery ranking.  
>
> Consequently, our attack objectives must directly target metric space: **Evasion**, where we force true gallery matches out of top ranks, and **Impersonation**, where we pull a query into the cluster of a targeted identity."

### [Slide 4: Victim Models & Baseline Training Recipe]
> "To evaluate adversarial vulnerability and transferability across diverse paradigms, we implemented three distinct victim backbones:
> 1. **ResNet-50** following the standard Bag-of-Tricks (BoT) recipe: we modified the stride of stage 4 from 2 to 1 to preserve spatial resolution at $16 \times 8$, added a Batch Normalization Neck (BNNeck), and extracted 2048-dimensional features.
> 2. **DenseNet-121**, which emphasizes deep feature reuse across dense blocks with 1024-dimensional embeddings.
> 3. **ViT-Small/16**, a Vision Transformer dividing the pedestrian into $16 \times 16$ visual tokens with self-attention.
>
> For training, we used the dual-objective loss: label-smoothed cross-entropy after the BNNeck to learn class boundaries, combined with **batch-hard triplet loss** before the BNNeck to enforce an Euclidean margin of $\alpha=0.3$. To make batch-hard mining effective, we wrote a custom **PK-Sampler** that draws $P=16$ identities and $K=4$ images per identity per batch, guaranteeing that every single anchor has valid hard positive and hard negative pairs."

### [Slide 5: Market-1501 Protocol & Clean Baselines]
> "We trained and benchmarked all models on the Market-1501 dataset. A crucial protocol requirement we strictly enforced in our evaluation code is the **Same-Camera Junk Rule**: whenever an image from the gallery matches both the query's person ID and its camera ID, it is marked as junk and removed before calculating Rank-1 and mean Average Precision (mAP). This guarantees we measure true cross-camera association rather than trivial single-camera tracking.
>
> On clean test data, our ResNet-50 achieved **88.5% Rank-1 and 71.2% mAP**, DenseNet achieved **86.2% Rank-1 and 68.4% mAP**, and ViT-Small achieved **89.8% Rank-1 and 73.5% mAP**.  
>
> With these high-performing clean baselines verified, I will now hand over to **Person B** to explain how we systematically attacked them."

---

## 👤 Speaker 2: Person B (Adversarial Attack Zoo & Cross-Architecture Transfer)
**Time:** 3:00 – 6:00  
**Slides:** 6 to 9  

### [Slide 6: Adversarial Attack Zoo]
> "Thank you, Person A. Once we established our strong victim baselines, my objective was to construct an adversarial attack suite from scratch to determine whether ReID systems are vulnerable to imperceptible perturbations.
>
> We adopted an $L_\infty$ threat budget of $\epsilon = 8/255$, which translates to roughly 3% maximum pixel deviation—virtually invisible to human observers. Notice an architectural detail in our implementation: image normalization happens *inside* the model forward pass. This ensures that our perturbation budget directly constrains raw pixel space $[0, 1]$, making the threat model realistic.
>
> We implemented three foundational gradient-based optimizers:
> - **FGSM**, which takes a single step along the gradient sign.
> - **PGD**, which performs multiple small projected gradient steps with uniform random restarts.
> - **MI-FGSM**, which incorporates a momentum decay factor of $\mu=1.0$ on $L_1$-normalized gradients to prevent the attack from getting stuck in poor surrogate-specific local optima."

### [Slide 7: Formulation: Evasion vs. Impersonation]
> "The heart of our attack suite lies in the objective functions:
> 1. In **Evasion attacks**, the goal is to make a person invisible to the retrieval system. We minimize the cosine similarity between the adversarial embedding and the clean embedding. Maximizing this negative dot-product drives the query into the empty metric space between identity clusters, so none of the true gallery images can rank in the top-$k$.
> 2. In **Impersonation attacks**, the goal is identity theft. We first compute the geometric centroid of a target victim identity across the gallery, and then maximize the cosine similarity between the adversarial query and this target prototype.
> 3. As a baseline control, we also built **Logit-PGD**, which attacks the auxiliary classification head using standard cross-entropy. Interestingly, Logit-PGD proved significantly inferior: it disrupted classification logits, but the metric embedding retained enough geometric structure that true gallery images still ranked near the top. This experimentally proves that attacking ReID requires metric-space objectives."

### [Slide 8: Cross-Architecture Transfer Matrix]
> "Next, we investigated black-box transferability by generating perturbations on one surrogate model and evaluating retrieval degradation on all other models.
>
> Looking at our transfer matrix on Slide 8:
> - On the diagonal, white-box attacks are devastating: ResNet-50 Rank-1 drops from 88.5% down to **12.4%**.
> - Off the diagonal, we observe striking architectural patterns: attacks transfer readily between CNN architectures—perturbations crafted on ResNet-50 degrade DenseNet-121 down to **45.2%** Rank-1.
> - However, transferring from CNNs to the Vision Transformer is markedly harder: ViT-Small retains **58.7%** Rank-1 against ResNet-crafted attacks. Because ViTs rely on global patch self-attention rather than localized convolutional receptive fields, their gradient geometries differ fundamentally, providing natural resistance against CNN-crafted adversarial noise."

### [Slide 9: Epsilon Sweep & Degradation Dynamics]
> "Finally, we performed an epsilon sweep from $\epsilon=0$ to $\epsilon=16/255$. Even at a very subtle $\epsilon=2/255$, Rank-1 accuracy falls by nearly 30%. Furthermore, our MI-FGSM attack consistently outperformed standard PGD in black-box transfer by an average of 8.2% across all architectures, proving that gradient momentum effectively smooths out surrogate architecture artifacts.
>
> Now, digital pixel perturbations are powerful in software, but can an attacker fool a surveillance camera in the physical world? To answer this, I pass the floor to **Person C** to introduce our Universal Invisibility Patch."

---

## 👤 Speaker 3: Person C (Universal Torso Patch & Interactive Demo)
**Time:** 6:00 – 9:00  
**Slides:** 10 to 12  

### [Slide 10: The Invisibility Cloak: Universal Torso Patch]
> "Thank you, Person B. While digital noise works against stored video frames, an adversary walking through an airport cannot hack the camera's sensor bitstream in real time. They need a physical countermeasure—a wearable artifact.
>
> We designed an **Adversarial Invisibility Patch**: a single, printable $64 \times 48$ pattern pasted on the pedestrian's torso.
> 
> What makes our patch unique is that it is **Universal**: we do not optimize a new patch for each person or photo. A single texture is trained across the entire training dataset so that *anyone* wearing it evades multi-camera identification.
> 
> To ensure physical printability, our loss combines cosine similarity minimization with **Total Variation (TV) regularization**. The TV penalty suppresses high-frequency salt-and-pepper noise, resulting in smooth color transitions that print accurately on fabric without printer color clipping."

### [Slide 11: Expectation over Transformation (EOT) & Control Experiment]
> "A major challenge in physical adversarial attacks is environmental variation: in real surveillance, a pedestrian turns, approaches the camera, steps into shadows, or is captured by cameras with varying sensor noise. A naive patch optimized on static pixels completely fails when tilted.
>
> To overcome this, we trained our patch using **Expectation over Transformation (EOT)**. During each optimization step, the patch undergoes randomized spatial and optical augmentations:
> - Random affine scaling between 80% and 125%,
> - In-plane rotation between $-20^\circ$ and $+20^\circ$,
> - Torso translation jitter of $\pm 15\%$,
> - Global illumination jitter and Gaussian sensor noise.
>
> To scientifically prove that the attack is caused by adversarial features rather than simple occlusion, we conducted a **Random Patch Control** experiment using an unoptimized patch of identical dimensions.
> 
> As shown on Slide 11:
> - The Clean ResNet-50 baseline achieves **88.5% Rank-1**.
> - When the pedestrian's torso is occluded by the Random Patch, Rank-1 only dips to **81.4%**, proving that ReID models easily recognize pedestrians using head, legs, and silhouette cues.
> - But when our **Universal EOT Patch** is applied, Rank-1 plummets to **29.7%**! This proves that the trained patch is not merely hiding the torso—it actively injects malicious gradient signals that drown out all genuine biometric cues."

### [Slide 12: Interactive Gradio Demonstration]
> "To bring these findings to life, we developed an interactive web application using Gradio, located in `demo/app.py`.
> 
> In this demo, users can select any pedestrian query, apply either digital PGD attacks or the Universal Torso Patch, and observe the immediate collapse of the retrieval ranking. In clean mode, the top 2 retrieved gallery images are true cross-camera matches with >90% confidence. But the moment the Universal Patch is toggled on, the embedding similarity to the clean identity drops to 0.31, and the true matches fall completely out of the top 50, replaced by unrelated distractor identities.
>
> Now, to understand *why* this dramatic collapse happens inside the neural network and how we can defend against it, I will invite **Person D** to present our analysis and defense."

---

## 👤 Speaker 4: Person D (Mechanistic Interpretability, Defense & Synthesis)
**Time:** 9:00 – 12:00  
**Slides:** 13 to 15  

### [Slide 13: Mechanistic Interpretability: Grad-CAM Breakdown]
> "Thank you, Person C. As deep learning researchers, obtaining high attack success rates is only half the battle; we need to explain *why* these models fail at the feature representation level.
>
> To investigate this, I performed **Grad-CAM saliency analysis** on the final convolutional activations (`layer4`) of ResNet-50.
> 
> The results on Slide 13 reveal the underlying breakdown:
> - On **Clean images**, the network's attention focuses precisely on distinctive biometric regions: the upper torso, jacket logos, bag straps, and footwear.
> - Under **FGSM**, the attention map visibly dilutes and expands into the image periphery.
> - Under **PGD**, the attention map fractures entirely: the network stops attending to the pedestrian and instead fixates on imperceptible high-frequency background noise artifacts.
> - Most strikingly, under the **Universal Torso Patch**, the entire activation energy of the convolutional backbone is **completely monopolized by the patch itself**. The patch acts as an attention sink, completely blinding the network to the individual's face, clothing, and limbs."

### [Slide 14: Embedding Drift & Adversarial Training Defense]
> "We also mapped the metric manifold using **t-SNE dimensionality reduction**. Under clean conditions, gallery and query features form tight, isolated clusters for each person ID. Under evasion attacks, the query vectors are propelled outwards into the inter-cluster void. Under impersonation attacks, the query trajectory moves directly into the target identity's cluster, confirming successful identity theft in latent space.
>
> To defend against these vulnerabilities, I formulated and implemented **Adversarial Training for ReID** (`reidattack/adv_train.py`). Standard classification adversarial training only optimizes cross-entropy on perturbed images. In ReID, we must preserve metric distances.
> 
> Our multi-task defense loss is:
> $$\mathcal{L}_{\text{defense}} = 0.5 \cdot \mathcal{L}_{\text{ReID}}(x_{\text{clean}}) + 0.5 \cdot \mathcal{L}_{\text{ReID}}(x_{\text{adv\_PGD}})$$
>
> The defense results demonstrate a clear security trade-off:
> - The undefended ResNet-50 collapsed from **88.5%** down to **12.4%** under PGD attack.
> - The adversarially trained ResNet-50 suffers a small clean accuracy penalty of 5.4% (achieving 83.1% clean Rank-1), but its robust accuracy under PGD attack surges from **12.4% up to 64.8%**—a massive **+52.4% gain in adversarial resilience**!"

### [Slide 15: Conclusions & Future Directions]
> "To conclude our presentation:
> 1. We designed from-scratch metric-space evasion and impersonation attacks, demonstrating that ReID systems cannot be evaluated using traditional classification metrics.
> 2. We quantified cross-architecture transferability, showing that Vision Transformers provide greater architectural resistance against CNN-crafted perturbations.
> 3. We engineered a printable Universal Torso Patch with EOT that drops Rank-1 retrieval by 58.8% while proving with random controls that the effect is genuinely adversarial, not merely occlusion.
> 4. Through Grad-CAM and t-SNE, we uncovered the attention-sink mechanism of failure and demonstrated that metric-aware adversarial training recovers substantial robustness.
>
> All code, test suites, training scripts, and demos are fully modular and available in our repository. Thank you for your time, and our team is now ready to take your questions."
