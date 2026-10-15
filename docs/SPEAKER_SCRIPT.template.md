# Speaker script (~10 min). Numbers auto-filled. Words in [brackets] are notes to yourself, not to read out.

**Slide 2 (45s).** Person re-identification matches a person across cameras by comparing embeddings and ranking a gallery. Unlike a classifier, there is no fixed label set, so an attack does not need to flip a label - it only needs to change the ranking. We ask three things: how do such attacks work, why do they transfer between models, and can a single printable patch do it.

**Slide 3 (60s).** We trained three victims on Market-1501: ResNet-50 reached {{base_resnet50_r1}} percent Rank-1 and {{base_resnet50_map}} mAP, DenseNet-121 {{base_densenet121_r1}} and {{base_densenet121_map}}, ViT-Small {{base_vit_small_r1}} and {{base_vit_small_map}}. All use ImageNet-pretrained weights, identity loss plus batch-hard triplet loss. [Say clearly: our "epoch" is one pass over the 751 identities.]

**Slide 4 (75s).** Threat model: the attacker may change each pixel by at most 8 out of 255. In evasion we minimise the cosine similarity between the attacked image's embedding and its own clean embedding. In impersonation we maximise similarity to the average embedding of a chosen identity. As a baseline we also attack the classifier logits, the way a classification attack would.

**Slide 5 (60s).** FGSM takes one signed-gradient step. PGD repeats small steps and projects back into the 8/255 ball. MI-FGSM adds momentum to the normalised gradient, which smooths the update direction. All three are written from scratch in `attacks.py`; they differ only in steps, start and momentum.

**Slide 6 (75s).** On ResNet-50, Rank-1 falls from {{base_resnet50_r1}} to {{wb_fgsm_resnet50_r1}} with FGSM, {{wb_pgd_resnet50_r1}} with PGD and {{wb_mifgsm_resnet50_r1}} with MI-FGSM. Attacking the classifier logits instead leaves Rank-1 at {{wb_logit_pgd_resnet50_r1}}. [Compare the two honestly: say which is stronger according to these numbers.] The retrieval figure shows real queries before and after.

**Slide 7 (90s).** The transfer matrix: diagonal is white-box, off-diagonal is black-box. Average Rank-1 under black-box PGD is {{xfer_pgd_mean_r1}} versus {{wb_pgd_mean_r1}} white-box. From ResNet-50, Rank-1 is {{xfer_pgd_resnet50_to_densenet121_r1}} on DenseNet-121 and {{xfer_pgd_resnet50_to_vit_small_r1}} on ViT-Small. [Describe the pattern you see in the heat-map. Offer a hypothesis, label it a hypothesis.]

**Slide 8 (90s).** The patch is one 64 by 48 image optimised over the whole training set with random scale, rotation, shift, brightness and noise, so it is not tied to one image. On ResNet-50, Rank-1 goes from {{patch_clean_resnet50_r1}} clean to {{patch_random_patch_resnet50_r1}} with a random patch of the same size - the control - and {{patch_trained_patch_resnet50_r1}} with our trained patch. With random transformations at test time it is {{patch_trained_patch_eot_resnet50_r1}}. [Mention: digital pasting only, printing not tested.]

**Slide 9 (60s).** Adversarial training with PGD at 8/255: clean Rank-1 {{def_clean_resnet50_r1}} for the standard model versus {{def_clean_resnet50_adv_trained_eps8_r1}} for the robust one; under white-box PGD {{def_wb_pgd_resnet50_r1}} versus {{def_wb_pgd_resnet50_adv_trained_eps8_r1}}.

**Slide 10 (45s).** Limitations: one seed, one dataset, digital-only patch. Everything shown comes from scripts in the repository and a provenance file that records the commit and checkpoint hashes.
