"""
Module 6 — Script 03: Diffusion Models Introduction
=====================================================

A conceptual introduction to diffusion models — the architecture
behind DALL-E 2, Stable Diffusion, and Imagen. We implement a
simplified version to build understanding.

Topics:
  - Forward diffusion process (adding noise)
  - Reverse diffusion process (denoising)
  - Noise schedule
  - Simple denoising network
  - Connection to score matching

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
# ## 1. The Diffusion Process — Intuition

# %%
print("=== Diffusion Models: The Core Idea ===")
print("""
Imagine you have a clear photograph. Diffusion models work in two phases:

FORWARD PROCESS (Destroying the image):
  Step 0: Original image (clear)
  Step 1: Add a tiny bit of noise
  Step 2: Add more noise
  ...
  Step T: Pure random noise (image completely destroyed)

REVERSE PROCESS (Recreating the image):
  Step T: Start from pure noise
  Step T-1: Remove a tiny bit of noise (learned by a neural network!)
  ...
  Step 0: Recovered image

The magic: A neural network learns to REVERSE each tiny noise step.
If it can do this well, we can start from pure noise and generate new images!
""")

# %% [markdown]
# ## 2. Forward Diffusion — Adding Noise

# %%
# Load a sample image
transform = transforms.ToTensor()
mnist = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
original_image = mnist[7][0]  # A digit

print("\n=== Forward Diffusion Process ===")

def add_noise(image: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Add Gaussian noise to an image.
    
    noisy = sqrt(1 - noise_level) * image + sqrt(noise_level) * noise
    
    This weighted sum preserves the signal-to-noise ratio:
      noise_level = 0: original image
      noise_level = 1: pure noise
    """
    noise = torch.randn_like(image)
    return np.sqrt(1 - noise_level) * image + np.sqrt(noise_level) * noise

# Visualize the forward process
noise_levels = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

fig, axes = plt.subplots(1, len(noise_levels), figsize=(16, 2.5))
for i, level in enumerate(noise_levels):
    noisy = add_noise(original_image, level)
    axes[i].imshow(noisy.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[i].set_title(f't={level}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Forward Diffusion: Gradually Adding Noise', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_06_conditional_generation/diffusion_forward.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Forward diffusion visualization saved")

# %% [markdown]
# ## 3. Noise Schedule

# %%
print("\n=== Noise Schedules ===")

T = 1000  # Number of timesteps

# Linear schedule (original DDPM)
beta_start, beta_end = 1e-4, 0.02
betas_linear = np.linspace(beta_start, beta_end, T)
alphas_linear = 1 - betas_linear
alpha_bar_linear = np.cumprod(alphas_linear)

# Cosine schedule (improved DDPM)
s = 0.008
steps = np.arange(T + 1)
f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
alpha_bar_cosine = f / f[0]
alpha_bar_cosine = alpha_bar_cosine[:T]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.plot(betas_linear, color='#3498DB', linewidth=2)
ax1.set_xlabel('Timestep t', fontsize=11)
ax1.set_ylabel('β_t', fontsize=11)
ax1.set_title('Noise Schedule: β_t (how much noise per step)', fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(alpha_bar_linear, label='Linear', color='#E74C3C', linewidth=2)
ax2.plot(alpha_bar_cosine, label='Cosine', color='#3498DB', linewidth=2)
ax2.set_xlabel('Timestep t', fontsize=11)
ax2.set_ylabel('ᾱ_t (cumulative signal retention)', fontsize=11)
ax2.set_title('Signal Retention Over Time', fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_06_conditional_generation/noise_schedules.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Noise schedules saved")

# %% [markdown]
# ## 4. Simple Denoising Network

# %%
class SimpleDenoiser(nn.Module):
    """
    A simple U-Net-like denoiser for MNIST.
    
    Input: noisy image (1, 28, 28) + timestep embedding
    Output: predicted noise (1, 28, 28)
    
    The network learns to predict WHAT NOISE was added,
    so we can subtract it to get the clean image.
    """
    
    def __init__(self, time_embed_dim: int = 32):
        super().__init__()
        
        # Time embedding (sinusoidal, like in transformers)
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 28→14
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 14→7
            nn.ReLU(),
        )
        
        # Middle (with time conditioning)
        self.middle = nn.Sequential(
            nn.Linear(128 * 7 * 7 + time_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 7 * 7),
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 14→28
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
    
    def forward(self, x, t):
        """
        Predict the noise in x at timestep t.
        
        Args:
            x: noisy image (batch, 1, 28, 28)
            t: timestep normalized to [0, 1] (batch, 1)
        """
        # Time embedding
        t_embed = self.time_embed(t)
        
        # Encode
        h = self.encoder(x)
        
        # Flatten and add time info
        h_flat = h.view(h.size(0), -1)
        h_flat = torch.cat([h_flat, t_embed], dim=1)
        h_flat = self.middle(h_flat)
        h = h_flat.view(-1, 128, 7, 7)
        
        # Decode
        noise_pred = self.decoder(h)
        return noise_pred


# %% [markdown]
# ## 5. Training the Denoiser

# %%
print("\n=== Training Simple Denoiser ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)

model = SimpleDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

n_epochs = 10
losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    
    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.size(0)
        
        # Sample random timesteps
        t = torch.rand(batch_size, 1).to(device)
        
        # Add noise at this timestep
        noise = torch.randn_like(images)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_images = torch.sqrt(1 - t_expanded) * images + torch.sqrt(t_expanded) * noise
        
        # Predict noise
        noise_pred = model(noisy_images, t)
        
        # Loss: how well did we predict the noise?
        loss = criterion(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"  Epoch {epoch+1:>2}/{n_epochs}: loss = {avg_loss:.6f}")

# %% [markdown]
# ## 6. Simple Denoising Demonstration

# %%
print("\n=== Denoising Demo ===")
model.eval()

# Take a test image and add noise, then try to denoise
test_img = mnist[0][0].unsqueeze(0).to(device)

# Progressive denoising
fig, axes = plt.subplots(3, 6, figsize=(14, 6))

noise_levels_demo = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

for i, level in enumerate(noise_levels_demo):
    # Add noise
    noise = torch.randn_like(test_img)
    noisy = np.sqrt(1 - level) * test_img + np.sqrt(level) * noise
    
    # Predict noise
    t = torch.tensor([[level]]).to(device)
    with torch.no_grad():
        pred_noise = model(noisy, t)
    
    # Simple single-step denoising (approximate)
    denoised = (noisy - np.sqrt(level) * pred_noise) / max(np.sqrt(1 - level), 0.01)
    denoised = torch.clamp(denoised, 0, 1)
    
    axes[0, i].imshow(test_img.cpu().squeeze(), cmap='gray')
    axes[0, i].set_title('Original', fontsize=9)
    axes[0, i].axis('off')
    
    axes[1, i].imshow(noisy.cpu().squeeze(), cmap='gray')
    axes[1, i].set_title(f'Noisy (t={level})', fontsize=9)
    axes[1, i].axis('off')
    
    axes[2, i].imshow(denoised.cpu().squeeze(), cmap='gray')
    axes[2, i].set_title('Denoised', fontsize=9)
    axes[2, i].axis('off')

plt.suptitle('Single-Step Denoising (Simple Diffusion)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_06_conditional_generation/diffusion_denoising.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Denoising demo saved")

print("""
=== What's Next? ===

This is a SIMPLIFIED diffusion model. Real diffusion models (DDPM, DDIM):
  1. Use hundreds of tiny denoising steps (not just one)
  2. Have more sophisticated architectures (full U-Net with attention)
  3. Support conditioning (class labels, text, other images)
  4. Use advanced samplers for faster generation

Module 7's documentation covers the full diffusion architecture
you'd need for cervical cytology image generation.
""")

print("\n✅ Module 6, Script 03 complete!")
