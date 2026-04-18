# Module 6 — Conditional Generation

## 🎯 Learning Objectives

By the end of this module, you will:

1. Understand **conditional generation** — controlling what the model produces
2. Build a **Conditional GAN (cGAN)** that generates specific digits on demand
3. Implement a **Conditional VAE (cVAE)** for label-driven reconstruction
4. Understand **diffusion model** concepts (theory + simple implementation)
5. Design a **full training pipeline** for conditional generative models

---

## 📖 Theory Overview

### Why Conditional Generation?

Unconditional models generate "random" samples. But in practice, we need control:
- Generate a **specific digit** (0-9)
- Generate a cell image with a **specific abnormality type**
- Generate a face with **specific attributes** (age, expression)

### Conditional GAN (cGAN)

The key idea: feed the class label to BOTH the generator and discriminator.

```
[Noise z] + [Label y] → [Generator] → Fake Image
                                          ↓
[Real Image] + [Label y] → [Discriminator] → Real/Fake?
```

The label can be:
- One-hot encoded class
- Embedding vector
- Continuous attribute (age, severity score)

### Conditional VAE (cVAE)

Same as VAE, but the encoder and decoder both receive the label:

```
[Image x] + [Label y] → [Encoder] → μ, σ
                                       ↓
[z ~ N(μ,σ)] + [Label y] → [Decoder] → Reconstruction
```

### Diffusion Models (Preview)

Diffusion models are a newer, powerful approach:
1. **Forward process**: Gradually add noise to an image (destroy it)
2. **Reverse process**: Learn to denoise step by step (recreate it)
3. **Conditioning**: Guide the denoising with class labels or text

---

## 📂 Scripts in This Module

| Script | Description |
|--------|-------------|
| `01_conditional_gan.py` | cGAN conditioned on digit labels (MNIST) |
| `02_conditional_vae.py` | Conditional VAE |
| `03_diffusion_intro.py` | Simple diffusion model concepts |
| `training/train_cgan.py` | Full cGAN training pipeline |
| `training/train_utils.py` | Training utilities |
| `evaluation/evaluate.py` | FID, IS, visual inspection |

---

## ➡️ Next Module

[Module 7 — Final Project](../module_07_final_project/README.md)
