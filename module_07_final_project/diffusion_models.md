# Diffusion Models — Comprehensive Documentation

## Overview

Diffusion models (also called Score-Based Generative Models) represent
the current state-of-the-art in image generation. This document covers
the theory, architecture, and practical considerations for applying
diffusion models to cervical cytology image synthesis.

---

## Core Concept

### The Two Processes

**Forward Process (Diffusion)**: Gradually add Gaussian noise to a data sample
over T timesteps until it becomes pure noise.

```
x_0 → x_1 → x_2 → ... → x_T ≈ N(0, I)
    +noise  +noise         pure noise
```

**Reverse Process (Denoising)**: Learn to reverse each noise step, recovering
the original data from pure noise.

```
x_T → x_{T-1} → ... → x_1 → x_0
noise  -noise        -noise  clean image
```

### Mathematical Formulation

**Forward process** (known, fixed):
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

**Closed-form sampling at any timestep**:
```
q(x_t | x_0) = N(x_t; √(ᾱ_t) · x_0, (1-ᾱ_t) · I)

where ᾱ_t = Π_{s=1}^{t} (1 - β_s)
```

This means: to get a noisy version at step t, we don't need to
iterate through all previous steps!

```python
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

**Reverse process** (learned):
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

---

## Noise Schedules

### Linear Schedule (DDPM, 2020)
```python
betas = np.linspace(beta_start, beta_end, T)
# beta_start = 0.0001, beta_end = 0.02, T = 1000
```

### Cosine Schedule (Improved DDPM, 2021)
```python
def cosine_schedule(T, s=0.008):
    steps = np.arange(T + 1) / T
    alpha_bar = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0, 0.999)
```

**Why cosine?** Linear schedule destroys information too quickly at
the beginning. Cosine preserves more structure in early steps.

---

## Architecture: U-Net

The standard architecture for diffusion models is a **U-Net** with:

### 1. Residual Blocks
```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_embed_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
    
    def forward(self, x, t_emb):
        h = self.norm1(F.silu(self.conv1(x)))
        h += self.time_mlp(t_emb)[:, :, None, None]  # Add time info
        h = self.norm2(F.silu(self.conv2(h)))
        return h + x  # Skip connection
```

### 2. Self-Attention Layers
Applied at lower resolutions (16×16 or 8×8) to capture global patterns:
```python
class SelfAttention(nn.Module):
    def __init__(self, channels):
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
```

### 3. Timestep Embedding
Sinusoidal positional encoding (same as Transformers):
```python
def sinusoidal_embed(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)
```

---

## Conditioning Methods

### 1. Concatenation
Simplest: concatenate condition embedding with input or features.
```python
x = torch.cat([noisy_image, condition_map], dim=1)
```

### 2. Cross-Attention
Used in Stable Diffusion. Condition is projected to keys/values:
```python
# In a transformer block:
Q = Linear(image_features)
K = Linear(condition_embedding)
V = Linear(condition_embedding)
attention = softmax(Q @ K.T / sqrt(d)) @ V
```

### 3. Classifier-Free Guidance (CFG)
Train one model for both conditional and unconditional generation:
- During training: randomly drop the condition 10% of the time
- During sampling: combine conditional and unconditional predictions

```python
# At inference:
noise_uncond = model(x_t, t, null_condition)
noise_cond = model(x_t, t, condition)
noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
```

Higher guidance scale → more faithful to condition but less diverse.

---

## Sampling Algorithms

### DDPM Sampling (Slow but High Quality)
```python
x = torch.randn(shape)  # Start from noise

for t in reversed(range(T)):
    noise_pred = model(x, t, condition)
    
    # Compute mean
    alpha = alphas[t]
    alpha_bar = alpha_bars[t]
    beta = betas[t]
    
    mean = (1 / sqrt(alpha)) * (x - (beta / sqrt(1 - alpha_bar)) * noise_pred)
    
    if t > 0:
        noise = torch.randn_like(x)
        x = mean + sqrt(beta) * noise
    else:
        x = mean
```
**Steps**: 1000 (slow!)

### DDIM Sampling (Fast)
Deterministic sampling with fewer steps:
```python
# Can use 50-200 steps instead of 1000
# Same trained model, different sampler
```
**Steps**: 50-200 (20-40× faster)

---

## Application to Cervical Cytology

### Conditioning Variables
- **Cell type** (categorical): Normal, LSIL, HSIL, ASC-US, ASC-H
- **Morphological features** (continuous):
  - Nuclear-to-cytoplasmic ratio: [0.1, 0.9]
  - Nuclear size: [small, medium, large]
  - Chromatin density: [light, moderate, dark]

### Expected Challenges
1. **Small dataset**: Medical image datasets are typically small (100-5000 images)
2. **Class imbalance**: Abnormal cells are rare
3. **Fine details**: Nuclear chromatin patterns require high-resolution generation
4. **Validation**: Requires expert pathologist review

### Mitigation Strategies
1. **Transfer learning**: Pretrain on a larger natural image dataset, fine-tune on cytology
2. **Progressive growing**: Start with low resolution, increase gradually
3. **Augmentation**: Aggressive augmentation of training data
4. **Few-shot techniques**: Adapt few-shot generative methods

---

## Key References

1. **DDPM**: Ho et al. (2020) *Denoising Diffusion Probabilistic Models* — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
2. **Improved DDPM**: Nichol & Dhariwal (2021) — Cosine schedule, learned variance
3. **DDIM**: Song et al. (2020) *Denoising Diffusion Implicit Models* — Faster sampling
4. **Classifier-Free Guidance**: Ho & Salimans (2022) — Standard conditioning approach
5. **Stable Diffusion**: Rombach et al. (2022) *Latent Diffusion Models* — Efficient training in latent space
6. **Medical Imaging**: Kazerouni et al. (2023) *Diffusion Models in Medical Imaging* — Survey
