# Module 5 — Generative Models

## 🎯 Learning Objectives

By the end of this module, you will:

1. Understand the difference between **discriminative** and **generative** models
2. Build an **autoencoder** for dimensionality reduction and reconstruction
3. Implement a **Variational Autoencoder (VAE)** with the reparameterization trick
4. Understand **GAN** theory (generator vs discriminator, Nash equilibrium)
5. Build and train a **basic GAN** to generate handwritten digits
6. Diagnose common GAN training issues (mode collapse, instability)

---

## 📖 Theory Overview

### Discriminative vs Generative Models

| | Discriminative | Generative |
|---|---|---|
| **Goal** | Learn P(y\|x) — classify | Learn P(x) — generate |
| **Examples** | CNN classifiers, logistic regression | VAE, GAN, diffusion |
| **Output** | Class label | New data samples |
| **Use case** | "Is this cell abnormal?" | "Generate a new abnormal cell image" |

### Autoencoders

```
Input → [Encoder] → Latent Code (z) → [Decoder] → Reconstruction
         compress      bottleneck       decompress
```

The bottleneck forces the network to learn a **compressed representation** of the data.

### Variational Autoencoders (VAE)

VAEs add a probabilistic twist:
- Encoder outputs **mean and variance** of a distribution, not a fixed point
- Sample z from this distribution (reparameterization trick)
- KL divergence loss regularizes the latent space
- Result: a smooth, continuous latent space you can sample from

### Generative Adversarial Networks (GAN)

Two networks playing a minimax game:
- **Generator**: Creates fake data from random noise
- **Discriminator**: Tries to distinguish real from fake

```
Random Noise (z) → [Generator] → Fake Image
                                      ↓
Real Image ───────────────→ [Discriminator] → Real or Fake?
```

Training converges when the generator fools the discriminator 50% of the time.

---

## 📂 Scripts in This Module

| Script | Description |
|--------|-------------|
| `01_autoencoder.py` | Basic autoencoder on MNIST |
| `02_variational_autoencoder.py` | VAE with reparameterization trick |
| `03_gan_theory.py` | GAN loss functions, Nash equilibrium |
| `04_basic_gan.py` | Basic GAN for MNIST generation |
| `05_gan_training_tips.py` | Mode collapse, training stability |
| `utils.py` | Visualization helpers |

---

## ➡️ Next Module

[Module 6 — Conditional Generation](../module_06_conditional_generation/README.md)
