# Models — Architecture Documentation

## Overview

This document describes the neural network architectures planned for
the cervical cytology image synthesis project.

## Architecture A: Conditional GAN (cGAN)

### Generator: U-Net with FiLM Conditioning

```
Input: [z (128-dim noise)] + [condition embedding]

Encoder Path:
  Conv(3→64, k3s2p1) → BatchNorm → LeakyReLU    # 64→32
  Conv(64→128, k3s2p1) → BatchNorm → LeakyReLU   # 32→16
  Conv(128→256, k3s2p1) → BatchNorm → LeakyReLU  # 16→8
  Conv(256→512, k3s2p1) → BatchNorm → LeakyReLU  # 8→4

Bottleneck:
  [z + condition] → Linear → Reshape → Conv(512, k3s1p1)

Decoder Path (with skip connections):
  ConvT(512→256) + Skip → BatchNorm → ReLU   # 4→8
  ConvT(256→128) + Skip → BatchNorm → ReLU   # 8→16
  ConvT(128→64) + Skip → BatchNorm → ReLU    # 16→32
  ConvT(64→3, k3s2p1) → Tanh                  # 32→64
```

### FiLM Conditioning (Feature-wise Linear Modulation)

Instead of concatenating the condition label, FiLM modulates
feature maps at each layer:

```
γ, β = MLP(condition)
h_new = γ * h + β
```

This allows fine-grained control over generated features at every scale.

### Discriminator: PatchGAN

```
Input: [Image (3, 64, 64)] + [Condition (broadcast to image)]

Conv(3+n_cond→64, k4s2p1) → LeakyReLU           # 64→32
Conv(64→128, k4s2p1) → InstanceNorm → LeakyReLU  # 32→16
Conv(128→256, k4s2p1) → InstanceNorm → LeakyReLU # 16→8
Conv(256→1, k4s1p1)                               # 8→7 (patch scores)
```

PatchGAN outputs a **map of scores** (not a single Real/Fake), encouraging
texture-level realism.

## Architecture B: Conditional Diffusion (DDPM)

### U-Net Denoiser

```
Input: [Noisy Image (3, 64, 64)] + [Timestep Embedding]

Encoder:
  ResBlock(3→64) + Self-Attention   # 64×64
  ↓ Downsample
  ResBlock(64→128) + Self-Attention  # 32×32
  ↓ Downsample
  ResBlock(128→256)                  # 16×16
  ↓ Downsample
  ResBlock(256→512)                  # 8×8

Bottleneck:
  ResBlock(512)
  Cross-Attention(condition)  # Inject class conditioning

Decoder (with skip connections):
  ↑ Upsample + Skip
  ResBlock(512→256)            # 16×16
  ↑ Upsample + Skip
  ResBlock(256→128)            # 32×32
  ↑ Upsample + Skip
  ResBlock(128→64)             # 64×64
  
  Conv(64→3)  # Predicted noise

Output: Predicted noise ε_θ(x_t, t, c)
```

### Timestep Embedding

Sinusoidal position encoding (like in Transformers):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### Conditioning Strategy

**Classifier-Free Guidance**: During training, randomly drop the condition
(replace with null embedding) 10% of the time. At inference:

```
ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
```

Where `w` is the guidance scale (typically 3-7).

## Hyperparameters (Planned)

| Parameter | cGAN Value | Diffusion Value |
|-----------|-----------|----------------|
| Image size | 64×64 | 64×64 |
| Batch size | 32 | 16 |
| Learning rate | 2e-4 | 1e-4 |
| Optimizer | Adam (β1=0.5) | AdamW |
| Epochs | 200 | 100 |
| Latent dim | 128 | N/A |
| Timesteps | N/A | 1000 |
| Guidance scale | N/A | 5.0 |
