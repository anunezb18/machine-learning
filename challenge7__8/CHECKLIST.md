# CHECKLIST.md — Challenge 7 | Group 8

## 1. Dataset pair and categories
- **Source domain**: SVHN (Street View House Numbers — real photographs of digits)
- **Target domain**: MNIST (handwritten greyscale digits)
- **Task**: 10-class digit recognition (classes 0–9)
- **Few-shot budget**: 50 labelled images per class for training (500 total)

## 2. Pretrained backbone and trainable parameters

| Model variant | Backbone | Strategy | Trainable params |
|---|---|---|---|
| Frozen Backbone | ResNet-50 (ImageNet) | Feature extraction (fc only) | ~2 050 |
| Fine-Tuned | ResNet-50 (ImageNet) | layer3 + layer4 + fc unfrozen | ~14.9 M |
| From Scratch | ResNet-50 (random init) | Full backbone | ~25.6 M |
| Target Fine-Tune | ResNet-50 (Part A init) | layer3 + layer4 + fc on MNIST | ~14.9 M |
| Style-Aug | ResNet-50 (Part A init) | layer3 + layer4 + fc + synth data | ~14.9 M |
| DANN | ResNet-50 (ImageNet) | Feature extractor + class head + domain head | ~15.2 M |

## 3. Accuracy results (mean ± std, 3 seeds: 42 / 123 / 7)

| Model variant | Source (SVHN) acc | Target (MNIST) acc | Δshift |
|---|---|---|---|
| Frozen Backbone | 0.2673 ± 0.0077 | 0.1478 ± 0.0311 | +0.1196 ± 0.0246 |
| Fine-Tuned | 0.7351 ± 0.0173 | 0.3739 ± 0.0701 | +0.3612 ± 0.0560 |
| From Scratch | 0.1764 ± 0.0508 | 0.0956 ± 0.0056 | +0.0807 ± 0.0512 |
| Baseline (no adapt) | 0.7490 ± 0.0057 | 0.4161 ± 0.0164 | +0.3329 ± 0.0221 |
| Target Fine-Tune | 0.3407 ± 0.0067 | 0.9389 ± 0.0064 | −0.5982 ± 0.0124 |
| Style-Aug | 0.7967 ± 0.0073 | 0.5056 ± 0.0107 | +0.2911 ± 0.0026 |
| DANN | 0.6723 ± 0.0062 | 0.6387 ± 0.0091 | +0.0337 ± 0.0029 |

## 4. Domain shift penalty

- **Best Part A model** (Fine-Tuned): Δshift = 0.3612 ± 0.0560
- **Best Part C model** (Target Fine-Tune): Δshift = −0.5982 ± 0.0124 (MNIST acc exceeds SVHN acc after adaptation)
- **Absolute improvement on target domain**: 0.9389 − 0.3739 = **+0.5650** accuracy points

## 5. Neural Style Transfer parameters

- **Content layer**: `relu4_2` (VGG-19 index 21)
- **Style layers**: `relu1_1, relu2_1, relu3_1, relu4_1, relu5_1`
- **α (content weight)**: 1.0
- **β (style weight)**: 1 × 10⁴  → ratio α/β = 10⁻⁴
- **Optimisation steps**: 300 L-BFGS steps (≈ 15 outer iterations × 20 inner)
- **Image size**: 256 × 256 px
- **Visual quality**: The generated images preserve digit shape and stroke structure from SVHN while adopting the high-contrast, thin-stroke monochrome appearance of MNIST handwritten digits. The α/β = 10⁻⁴ ratio provides visible style transfer without distorting digit content beyond recognition.

## 6. Best adaptation strategy

**Target Fine-Tune** achieved the highest target-domain accuracy for this domain pair, reaching **0.9389 ± 0.0064** on MNIST and inverting the domain shift to Δshift = −0.5982. SVHN and MNIST share the same output classes (digits 0–9) but differ drastically in appearance: SVHN digits are photographed in natural scenes with colour, texture, and background clutter, while MNIST digits are clean, centred, grey-scale strokes. The backbone features learned on SVHN capture colour and texture cues that do not generalise to MNIST's sparse pixel patterns. Fine-tuning the last two residual blocks on a small set of 50 labelled MNIST images per class is sufficient to re-calibrate high-level feature representations toward stroke-based patterns, producing a +0.5650 absolute improvement on the target domain at the cost of reduced SVHN accuracy (catastrophic forgetting: 0.7351 → 0.3407). Style-transfer augmentation improves over the baseline (0.5056 vs 0.4161) without requiring target labels, but is limited because SVHN images contain multiple digits and heavy backgrounds that partially survive style transfer. DANN is competitive as an unsupervised method (0.6387), substantially reducing Δshift from +0.3329 to +0.0337 without any target-domain labels, but is outperformed by the supervised fine-tune approach.