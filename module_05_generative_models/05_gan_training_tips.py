"""
Module 5 — Script 05: GAN Training Tips
=========================================

Common GAN training problems and their solutions.
We demonstrate mode collapse, training instability,
and effective countermeasures.

Topics:
  - Mode collapse detection and prevention
  - Label smoothing
  - Spectral normalization
  - Feature matching
  - Wasserstein loss (WGAN) concepts

Run interactively with VS Code cells (# %%) or as a script.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## 1. Detecting Mode Collapse

# %%
print("=== GAN Training Tips ===")
print("""
MODE COLLAPSE: The #1 GAN training problem.

Signs of mode collapse:
  1. Generated images all look the same
  2. Low diversity in outputs despite different input noise
  3. Discriminator loss drops to near zero
  4. Generator loss increases or oscillates

Detection strategies:
  - Visual inspection: generate many samples and check diversity
  - Inception Score (IS): measures both quality AND diversity
  - FID (Fréchet Inception Distance): compares feature distributions
""")

# %% [markdown]
# ## 2. Tip 1 — Label Smoothing

# %%
print("\n=== Tip 1: Label Smoothing ===")
print("""
Instead of training D with hard labels (1=real, 0=fake):
  → Use soft labels: real=0.9, fake=0.1 (or smooth randomly)

Why? Prevents D from becoming too confident, which would cause
vanishing gradients for G.
""")

# Example
hard_real = torch.ones(5, 1)
soft_real = torch.FloatTensor(5, 1).uniform_(0.8, 1.0)  # Random in [0.8, 1.0]

hard_fake = torch.zeros(5, 1)
soft_fake = torch.FloatTensor(5, 1).uniform_(0.0, 0.2)  # Random in [0.0, 0.2]

print(f"Hard real labels: {hard_real.squeeze().tolist()}")
print(f"Soft real labels: {soft_real.squeeze().round(decimals=3).tolist()}")
print(f"Hard fake labels: {hard_fake.squeeze().tolist()}")
print(f"Soft fake labels: {soft_fake.squeeze().round(decimals=3).tolist()}")

# %% [markdown]
# ## 3. Tip 2 — Spectral Normalization

# %%
print("\n=== Tip 2: Spectral Normalization ===")
print("""
Spectral normalization constrains the Lipschitz constant of D.
This stabilizes training by preventing D from changing too rapidly.

In PyTorch, it's a one-line addition:
  nn.utils.spectral_norm(nn.Linear(256, 128))
""")

# Demonstration
linear = nn.Linear(256, 128)
sn_linear = nn.utils.spectral_norm(nn.Linear(256, 128))

print(f"Normal Linear weight shape: {linear.weight.shape}")
print(f"SN Linear weight shape: {sn_linear.weight.shape}")
print(f"SN Linear has 'weight_orig' attr: {hasattr(sn_linear, 'weight_orig')}")

class SNDiscriminator(nn.Module):
    """Discriminator with Spectral Normalization."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(784, 512)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.model(x)

print("Spectral Normalized Discriminator created ✓")

# %% [markdown]
# ## 4. Tip 3 — Training Schedule

# %%
print("\n=== Tip 3: Training Schedule ===")
print("""
Common training schedules:

1. EQUAL TRAINING (default):
   Train D once, then G once, per batch.
   
2. D TRAINS MORE:
   Train D k times, then G once.
   Helps when D is weak and can't provide useful gradients.
   Common: k=5 for WGAN.
   
3. CONDITIONAL TRAINING:
   Only train D when its accuracy drops below threshold.
   Prevents D from dominating.

4. TWO TIME-SCALE UPDATE RULE (TTUR):
   Use different learning rates: lr_D > lr_G
   Gives D time to learn before G starts chasing.
""")

# %% [markdown]
# ## 5. Tip 4 — Wasserstein Loss (WGAN)

# %%
print("\n=== Tip 4: Wasserstein Loss ===")
print("""
WGAN replaces BCE loss with Wasserstein (Earth Mover's) distance.

Standard GAN loss:
  D: -[log D(x) + log(1 - D(G(z)))]
  G: -log D(G(z))

WGAN loss (simpler!):
  D (Critic): D(G(z)) - D(x)     (minimize: make real score higher)
  G:          -D(G(z))            (minimize: make fake score higher)

Key differences:
  - D outputs a score, not a probability (no sigmoid!)
  - D is called a "critic" instead of discriminator
  - Must enforce Lipschitz constraint (weight clipping or gradient penalty)
  - More stable training, meaningful loss metric
""")

class WGANCritic(nn.Module):
    """WGAN Critic — No sigmoid at the end!"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),  # Raw score, no sigmoid!
        )
    
    def forward(self, x):
        return self.model(x)

# WGAN-GP (Gradient Penalty) loss
def gradient_penalty(critic, real, fake, device):
    """
    Compute gradient penalty for WGAN-GP.
    
    Enforces ||∇D(αx + (1-α)G(z))||₂ ≈ 1
    This is a softer alternative to weight clipping.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Interpolate between real and fake
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    
    # Critic score
    d_interpolated = critic(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

print("WGAN-GP components defined ✓")

# %% [markdown]
# ## 6. Summary of Best Practices

# %%
print("""
=== GAN Training Best Practices Summary ===

Architecture:
  ✓ Use LeakyReLU in D (not ReLU — avoids dead neurons)
  ✓ Use BatchNorm in G (helps training stability)
  ✓ Use Spectral Normalization in D
  ✓ Avoid MaxPool — use strided convolutions instead

Training:
  ✓ Use Adam optimizer with β₁=0.5, β₂=0.999
  ✓ Use learning rate 1e-4 to 2e-4
  ✓ Apply label smoothing (real=0.9 instead of 1.0)
  ✓ Monitor D accuracy — if it reaches 100%, G can't learn
  ✓ Generate samples at regular intervals to check quality

Loss Functions:
  ✓ Use non-saturating loss for G: max log(D(G(z)))
  ✓ Consider WGAN-GP for more stable training
  ✓ Track FID score for quantitative evaluation

Debugging:
  ✓ If all generated images look the same → mode collapse
  ✓ If D loss → 0 and G loss → ∞ → D is too powerful
  ✓ If both losses oscillate wildly → lr too high
  ✓ If G loss → 0 quickly → D is too weak
""")

print("\n✅ Module 5, Script 05 complete!")
print("\n🎉 Module 5 complete! You can build autoencoders, VAEs, and GANs!")
print("Next: Module 6 — Conditional Generation (control WHAT you generate) →")
