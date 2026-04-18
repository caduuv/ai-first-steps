# Module 7 — Final Project: Cervical Cytology Image Synthesis

## 🎯 Project Goal

Build a **Conditional GAN or Diffusion Model** capable of synthesizing realistic cervical cytology images conditioned on medical parameters:

- **Cell type**: Normal, LSIL, HSIL, ASC-US, ASC-H
- **Abnormality level**: Grade of dysplasia
- **Morphological features**: Cell size, nuclear-to-cytoplasmic ratio, staining intensity

---

## 📖 Background: Cervical Cytology

### What is Cervical Cytology?

Cervical cytology (Pap smear) is a screening test where cells are collected from the cervix and examined microscopically. The goal is to detect **precancerous changes** before they progress to cancer.

### The Bethesda Classification System

| Category | Description | Clinical Significance |
|----------|-------------|----------------------|
| **NILM** | Negative for Intraepithelial Lesion or Malignancy | Normal |
| **ASC-US** | Atypical Squamous Cells of Undetermined Significance | Borderline |
| **ASC-H** | Atypical — cannot exclude HSIL | Potential high-grade |
| **LSIL** | Low-grade Squamous Intraepithelial Lesion | Mild dysplasia |
| **HSIL** | High-grade Squamous Intraepithelial Lesion | Moderate/severe dysplasia |
| **SCC** | Squamous Cell Carcinoma | Cancer |

### Key Morphological Features

When pathologists examine cells, they look for:

1. **Nuclear size**: Abnormal cells tend to have enlarged nuclei
2. **Nuclear-to-cytoplasmic (N/C) ratio**: Higher ratio = more abnormal
3. **Nuclear shape**: Irregular contours suggest abnormality
4. **Chromatin pattern**: Hyperchromatic = darkly stained = suspicious
5. **Cell clustering**: Abnormal cells may form characteristic patterns

### Why Generate Synthetic Images?

- **Class imbalance**: Abnormal cells are rare in screening datasets
- **Data augmentation**: More diverse training data improves classifiers
- **Education**: Generate examples for training pathologists
- **Privacy**: Synthetic images don't contain patient-identifiable information
- **Research**: Test hypotheses about morphological features

---

## 🏗️ Proposed Architecture

### Option A: Conditional GAN (cGAN)

```
Conditioning Vector: [cell_type, abnormality_grade, nc_ratio, nuclear_size, ...]
                                    ↓
[Random Noise z] + [Condition] → [Generator (U-Net)] → Synthetic Cell Image
                                                           ↓
[Real Cell Image] + [Condition] → [Discriminator (PatchGAN)] → Real/Fake?
```

**Generator**: U-Net with skip connections for preserving fine details
**Discriminator**: PatchGAN for texture-level discrimination
**Conditioning**: Label embedding + FiLM (Feature-wise Linear Modulation)

### Option B: Conditional Diffusion Model

```
[Clean Image] → [Forward: Add Noise] → [Noisy Image at step t]
                                              ↓
[Noisy Image] + [Timestep t] + [Condition] → [U-Net Denoiser] → [Predicted Noise]
                                              ↓
[Reverse: Remove Noise for T steps] → [Synthetic Image]
```

**Architecture**: U-Net with attention layers
**Conditioning**: Cross-attention or concatenation
**Sampling**: DDPM or DDIM sampler

### Recommendation

Start with **cGAN** (faster to train, simpler to debug), then explore diffusion models if you want state-of-the-art quality.

---

## 📊 Datasets

### Option 1: Simulated Data (Default)

A programmatic generator that creates simple cell-like images for testing:
- Circular cells with varying sizes
- Nuclear/cytoplasm regions
- Staining variation
- Controlled parameters

This allows the project to work **without downloading real medical data**.

### Option 2: SIPaKMeD Dataset

- **Source**: [SIPaKMeD](https://www.cs.uoi.gr/~marina/sipakmed.html)
- **Content**: 4,049 cell images in 5 classes
- **Resolution**: Variable, typically 50-200px per cell
- **License**: Research use

### Option 3: Herlev Dataset

- **Source**: Herlev University Hospital
- **Content**: 917 cell images in 7 classes
- **Resolution**: Standardized
- **License**: Research use with citation

See `data/README.md` for download and preprocessing instructions.

---

## 📋 Implementation Roadmap

### Phase 1: Data Pipeline
- [ ] Implement simulated data generator
- [ ] Build data loading pipeline (PyTorch Dataset)
- [ ] Implement augmentation (rotation, flip, color jitter)
- [ ] Create train/validation/test splits

### Phase 2: Baseline Model
- [ ] Implement conditional generator (FC-based, then convolutional)
- [ ] Implement conditional discriminator
- [ ] Set up training loop with logging
- [ ] Train on simulated data → verify everything works

### Phase 3: Full Model
- [ ] Switch to U-Net generator
- [ ] Implement PatchGAN discriminator
- [ ] Add spectral normalization + gradient penalty
- [ ] Train on real data (if available)

### Phase 4: Evaluation
- [ ] Visual quality inspection
- [ ] FID score computation
- [ ] Per-class generation quality
- [ ] Downstream classifier performance (train on synthetic, test on real)

### Phase 5: Documentation
- [ ] Document architecture decisions
- [ ] Record experiment results
- [ ] Discuss ethical considerations
- [ ] Suggest future improvements

---

## 🔗 References

1. Goodfellow et al. (2014) *Generative Adversarial Networks*
2. Mirza & Osindero (2014) *Conditional Generative Adversarial Nets*
3. Ho et al. (2020) *Denoising Diffusion Probabilistic Models*
4. Isola et al. (2017) *Image-to-Image Translation with cGANs (Pix2Pix)*
5. Brock et al. (2019) *Large Scale GAN Training for High Fidelity Natural Image Synthesis (BigGAN)*

---

## ➡️ Getting Started

1. Review the documentation in this module
2. Start with `data/README.md` for dataset setup
3. Check `models/README.md` for architecture details
4. Follow the implementation roadmap above
