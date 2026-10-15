<!-- AUTO-GENERATED 2026-10-15 16:01 UTC from results/*.json by scripts/build_docs.py - do not hand-edit numbers -->
# Speaker script (~10 min). Numbers auto-filled. Words in [brackets] are notes to yourself, not to read out.

**Slide 2 (45s).** Person re-identification matches a person across cameras by comparing embeddings and ranking a gallery. Unlike a classifier, there is no fixed label set, so an attack does not need to flip a label - it only needs to change the ranking. We ask three things: how do such attacks work, why do they transfer between models, and can a single printable patch do it.

**Slide 3 (60s).** We trained three victims on Market-1501: ResNet-50 reached 87.0 percent Rank-1 and 70.6 mAP, DenseNet-121 77.3 and 56.6, ViT-Small 91.8 and 79.7. All use ImageNet-pretrained weights, identity loss plus batch-hard triplet loss. [Say clearly: our "epoch" is one pass over the 751 identities.]

**Slide 4 (75s).** Threat model: the attacker may change each pixel by at most 8 out of 255. In evasion we minimise the cosine similarity between the attacked image's embedding and its own clean embedding. In impersonation we maximise similarity to the average embedding of a chosen identity. As a baseline we also attack the classifier logits, the way a classification attack would.

**Slide 5 (60s).** FGSM takes one signed-gradient step. PGD repeats small steps and projects back into the 8/255 ball. MI-FGSM adds momentum to the normalised gradient, which smooths the update direction. All three are written from scratch in `attacks.py`; they differ only in steps, start and momentum.

**Slide 6 (75s).** On ResNet-50, Rank-1 falls from 87.0 to 39.6 with FGSM, 0.0 with PGD and 0.0 with MI-FGSM. Attacking the classifier logits instead leaves Rank-1 at 0.3. [Compare the two honestly: say which is stronger according to these numbers.] The retrieval figure shows real queries before and after.

**Slide 7 (90s).** The transfer matrix: diagonal is white-box, off-diagonal is black-box. Average Rank-1 under black-box PGD is 61.0 versus 0.0 white-box. From ResNet-50, Rank-1 is 30.5 on DenseNet-121 and 88.2 on ViT-Small. [Describe the pattern you see in the heat-map. Offer a hypothesis, label it a hypothesis.]

**Slide 8 (90s).** The patch is one 64 by 48 image optimised over the whole training set with random scale, rotation, shift, brightness and noise, so it is not tied to one image. On ResNet-50, Rank-1 goes from 87.0 clean to 73.2 with a random patch of the same size - the control - and 2.7 with our trained patch. With random transformations at test time it is 2.7. [Mention: digital pasting only, printing not tested.]

**Slide 9 (60s).** Adversarial training with PGD at 8/255: clean Rank-1 87.0 for the standard model versus 74.5 for the robust one; under white-box PGD 0.0 versus 29.2.

**Slide 10 (45s).** Limitations: one seed, one dataset, digital-only patch. Everything shown comes from scripts in the repository and a provenance file that records the commit and checkpoint hashes.
