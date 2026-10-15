# Results summary (auto-generated 2026-10-15 16:01 UTC)

Every number below is computed from `results/*.json` written by real runs. See `results/MANIFEST.json` for provenance.

## 1. Clean baselines
| Model | Params (M) | Rank-1 | Rank-5 | Rank-10 | mAP |
|---|---|---|---|---|---|
| resnet50 | 25.1 | 86.97 | 95.4 | 97.36 | 70.64 |
| densenet121 | 7.7 | 77.35 | 91.95 | 95.07 | 56.63 |
| vit_small | 21.9 | 91.75 | 96.67 | 98.01 | 79.69 |

## 2. Evasion attacks

### FGSM eps=8/255 - Rank-1 (%) (rows = source, cols = target; diagonal = white-box)
| source \ target | resnet50 | densenet121 | vit_small |
|---|---|---|---|
| resnet50 | 39.58 | 44.15 | 87.86 |
| densenet121 | 54.69 | 23.01 | 87.62 |
| vit_small | 64.9 | 43.74 | 43.05 |

### PGD eps=8/255 - Rank-1 (%) (rows = source, cols = target; diagonal = white-box)
| source \ target | resnet50 | densenet121 | vit_small |
|---|---|---|---|
| resnet50 | 0.0 | 30.46 | 88.15 |
| densenet121 | 51.84 | 0.0 | 88.66 |
| vit_small | 64.07 | 43.05 | 0.0 |

### MIFGSM eps=8/255 - Rank-1 (%) (rows = source, cols = target; diagonal = white-box)
| source \ target | resnet50 | densenet121 | vit_small |
|---|---|---|---|
| resnet50 | 0.0 | 17.81 | 83.61 |
| densenet121 | 34.47 | 0.0 | 85.24 |
| vit_small | 52.38 | 30.34 | 0.3 |

### LOGIT_PGD eps=8/255 - Rank-1 (%) (rows = source, cols = target; diagonal = white-box)
| source \ target | resnet50 | densenet121 | vit_small |
|---|---|---|---|
| resnet50 | 0.3 | 39.31 | 88.24 |
| densenet121 | 53.18 | 0.06 | 88.66 |
| vit_small | 66.57 | 43.23 | 0.03 |

### White-box vs black-box summary (eps=8, Rank-1 %, lower = stronger attack)
| attack | white-box mean | black-box (transfer) mean |
|---|---|---|
| fgsm | 35.21 | 63.83 |
| pgd | 0.0 | 61.04 |
| mifgsm | 0.1 | 50.64 |
| logit_pgd | 0.13 | 63.2 |

### Epsilon sweep, white-box on resnet50 (Rank-1 %)
| attack | eps=2 | eps=4 | eps=8 | eps=16 |
|---|---|---|---|---|
| fgsm | 74.44 | 62.8 | 39.58 | 4.31 |
| pgd | 0.12 | 0.0 | 0.0 | 0.0 |
| mifgsm | 3.92 | 0.06 | 0.0 | 0.0 |
| logit_pgd | 6.29 | 1.25 | 0.3 | 0.0 |

## 3. Impersonation attacks (higher hit@1 = stronger attack)
| attack | source | target | hit@1 (%) | Rank-1 of true id (%) |
|---|---|---|---|---|
| pgd | resnet50 | resnet50 | 99.29 | 0.03 |
| pgd | resnet50 | densenet121 | 0.77 | 50.36 |
| pgd | resnet50 | vit_small | 0.09 | 88.93 |
| pgd | densenet121 | resnet50 | 0.71 | 65.97 |
| pgd | densenet121 | densenet121 | 98.31 | 0.03 |
| pgd | densenet121 | vit_small | 0.12 | 88.42 |
| pgd | vit_small | resnet50 | 0.39 | 72.15 |
| pgd | vit_small | densenet121 | 0.42 | 53.12 |
| pgd | vit_small | vit_small | 98.13 | 0.0 |
| mifgsm | resnet50 | resnet50 | 99.67 | 0.0 |
| mifgsm | resnet50 | densenet121 | 2.29 | 37.02 |
| mifgsm | resnet50 | vit_small | 0.3 | 84.17 |
| mifgsm | densenet121 | resnet50 | 1.63 | 52.52 |
| mifgsm | densenet121 | densenet121 | 99.61 | 0.03 |
| mifgsm | densenet121 | vit_small | 0.27 | 83.73 |
| mifgsm | vit_small | resnet50 | 1.01 | 62.32 |
| mifgsm | vit_small | densenet121 | 0.77 | 41.98 |
| mifgsm | vit_small | vit_small | 99.58 | 0.0 |

## 4. Universal torso patch (EOT)
| model | role | condition | Rank-1 | mAP |
|---|---|---|---|---|
| resnet50 | source | clean | 86.97 | 70.64 |
| resnet50 | source | random_patch | 73.19 | 53.42 |
| resnet50 | source | trained_patch | 2.67 | 3.29 |
| resnet50 | source | trained_patch_eot | 2.73 | 3.18 |
| densenet121 | transfer | clean | 77.35 | 56.63 |
| densenet121 | transfer | random_patch | 63.27 | 43.08 |
| densenet121 | transfer | trained_patch | 13.63 | 9.76 |
| densenet121 | transfer | trained_patch_eot | 13.87 | 9.21 |
| vit_small | transfer | clean | 91.75 | 79.69 |
| vit_small | transfer | random_patch | 82.39 | 68.64 |
| vit_small | transfer | trained_patch | 74.02 | 60.28 |
| vit_small | transfer | trained_patch_eot | 70.81 | 58.73 |

Patch training: 40 epochs, mean cosine-sim to clean per epoch: [0.3162, 0.2379, 0.2401, 0.2379, 0.2408, 0.2369, 0.2384, 0.2385, 0.2371, 0.2411, 0.2381, 0.2393, 0.2366, 0.2378, 0.238, 0.2362, 0.2397, 0.2379, 0.2371, 0.2382, 0.2385, 0.2371, 0.2375, 0.2385, 0.2391, 0.2371, 0.2389, 0.2368, 0.2396, 0.2373, 0.2374, 0.2381, 0.2403, 0.2376, 0.2396, 0.2386, 0.24, 0.237, 0.2343, 0.2347]

## 5. Adversarial-training defence (eps=8)
| attack | source | target | Rank-1 | mAP |
|---|---|---|---|---|
| clean | - | resnet50 | 86.97 | 70.64 |
| clean | - | resnet50_adv_trained_eps8 | 74.52 | 52.28 |
| pgd | resnet50 | resnet50 | 0.0 | 0.05 |
| pgd | resnet50 | resnet50_adv_trained_eps8 | 72.39 | 49.96 |
| pgd | resnet50_adv_trained_eps8 | resnet50 | 11.61 | 7.61 |
| pgd | resnet50_adv_trained_eps8 | resnet50_adv_trained_eps8 | 29.19 | 20.26 |
| mifgsm | resnet50 | resnet50 | 0.0 | 0.05 |
| mifgsm | resnet50 | resnet50_adv_trained_eps8 | 71.79 | 49.65 |
| mifgsm | resnet50_adv_trained_eps8 | resnet50 | 12.89 | 8.8 |
| mifgsm | resnet50_adv_trained_eps8 | resnet50_adv_trained_eps8 | 25.86 | 17.98 |
