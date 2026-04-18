"""
Module 5 — Script 02: Variational Autoencoder (VAE)
=====================================================

The VAE fixes the autoencoder's generation problem by enforcing
a structured latent space. Instead of encoding to a point, it
encodes to a distribution, enabling smooth sampling.

Topics:
  - VAE vs autoencoder
  - Reparameterization trick
  - ELBO loss (reconstruction + KL divergence)
  - Smooth latent space
  - Digit morphing in latent space

Run interactively with VS Code cells (# %%) or as a script.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## 1. VAE Architecture

# %%
class VAE(nn.Module):
    """
    Variational Autoencoder.
    
    Key differences from standard autoencoder:
    1. Encoder outputs TWO things: mean (μ) and log-variance (log σ²)
    2. Latent code z is SAMPLED: z = μ + σ * ε, where ε ~ N(0,1)
    3. Loss = Reconstruction Loss + KL Divergence
    
    The KL divergence pushes the latent distribution toward N(0,1),
    creating a smooth, continuous latent space perfect for sampling.
    """
    
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Two separate heads for mean and log-variance
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )
    
    def encode(self, x):
        """Encode input to (mean, log_variance) of latent distribution."""
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε
        
        Why not just sample z ~ N(μ, σ²)?
        Because random sampling is NOT differentiable!
        The trick: sample ε ~ N(0,1), then compute z = μ + σ * ε
        Now the randomness (ε) is external, and z is a deterministic
        function of μ and σ, which ARE differentiable.
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)     # ε ~ N(0, 1)
        return mu + std * eps           # z = μ + σ * ε
    
    def decode(self, z):
        """Decode latent code to reconstructed image."""
        return self.decoder_net(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence
    
    Reconstruction: How well can we reconstruct the input?
    KL Divergence: How close is the latent distribution to N(0,1)?
    
    KL(N(μ, σ²) || N(0, 1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    
    β controls the trade-off:
      β < 1: Better reconstruction, less structured latent space
      β = 1: Standard VAE (ELBO)
      β > 1: More structured latent space, blurrier reconstructions
    """
    # Reconstruction loss (binary cross-entropy per pixel)
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784),
                                         reduction='sum')
    
    # KL divergence (analytical formula for Gaussian)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# %% [markdown]
# ## 2. Train the VAE

# %%
print("=== VAE Training ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Use latent_dim=2 so we can visualize directly (no PCA needed)
model = VAE(latent_dim=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 20
losses_history = {'Total': [], 'Reconstruction': [], 'KL': []}

for epoch in range(n_epochs):
    model.train()
    total, recon_total, kl_total = 0, 0, 0
    
    for images, _ in train_loader:
        images = images.to(device)
        
        recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += loss.item()
        recon_total += recon_loss.item()
        kl_total += kl_loss.item()
    
    n = len(train_dataset)
    losses_history['Total'].append(total / n)
    losses_history['Reconstruction'].append(recon_total / n)
    losses_history['KL'].append(kl_total / n)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:>2}: total={total/n:.2f}, "
              f"recon={recon_total/n:.2f}, kl={kl_total/n:.2f}")

# %% [markdown]
# ## 3. Loss Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.plot(losses_history['Total'], label='Total', color='#E74C3C', linewidth=2)
ax1.plot(losses_history['Reconstruction'], label='Reconstruction', color='#3498DB', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (per sample)')
ax1.set_title('VAE Losses', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(losses_history['KL'], label='KL Divergence', color='#2ECC71', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('KL Loss (per sample)')
ax2.set_title('KL Divergence', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_05_generative_models/vae_losses.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Loss curves saved")

# %% [markdown]
# ## 4. Reconstructions

# %%
model.eval()
test_images, _ = next(iter(test_loader))
test_images = test_images[:10].to(device)

with torch.no_grad():
    recon, _, _ = model(test_images)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('VAE Recon.', fontsize=11, fontweight='bold')
plt.suptitle('VAE Reconstructions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/vae_reconstructions.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Reconstructions saved")

# %% [markdown]
# ## 5. Latent Space Visualization (2D)

# %%
model.eval()
all_mu = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        mu, _ = model.encode(images.to(device))
        all_mu.append(mu.cpu().numpy())
        all_labels.append(labels.numpy())

all_mu = np.concatenate(all_mu)
all_labels = np.concatenate(all_labels)

fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap='tab10',
                      alpha=0.5, s=5)
plt.colorbar(scatter, ax=ax, label='Digit')
ax.set_xlabel('$z_1$', fontsize=12)
ax.set_ylabel('$z_2$', fontsize=12)
ax.set_title('VAE Latent Space (2D) — Smooth and Structured!', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/vae_latent_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Latent space saved")

# %% [markdown]
# ## 6. Generate by Sampling the Latent Space

# %%
model.eval()

# Generate a grid by sampling z uniformly across the latent space
n = 20
z1 = np.linspace(-3, 3, n)
z2 = np.linspace(-3, 3, n)

fig, axes = plt.subplots(n, n, figsize=(12, 12))

for i, z2_val in enumerate(reversed(z2)):
    for j, z1_val in enumerate(z1):
        z = torch.tensor([[z1_val, z2_val]], dtype=torch.float32).to(device)
        with torch.no_grad():
            generated = model.decode(z)
        axes[i, j].imshow(generated.cpu().squeeze(), cmap='gray')
        axes[i, j].axis('off')

plt.suptitle('VAE: Sampling Across the 2D Latent Space\nSmooth transitions between digits!',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/vae_latent_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Latent grid saved")

# %% [markdown]
# ## 7. Digit Morphing

# %%
# Interpolate between two digits in latent space
model.eval()

# Find a 3 and a 7 in the test set
digit_indices = {}
for idx, (_, label) in enumerate(test_dataset):
    if label not in digit_indices:
        digit_indices[label] = idx
    if len(digit_indices) == 10:
        break

img_3 = test_dataset[digit_indices[3]][0].unsqueeze(0).to(device)
img_7 = test_dataset[digit_indices[7]][0].unsqueeze(0).to(device)

with torch.no_grad():
    mu_3, _ = model.encode(img_3)
    mu_7, _ = model.encode(img_7)

# Interpolate
n_steps = 10
fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))

for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
    z = (1 - alpha) * mu_3 + alpha * mu_7
    with torch.no_grad():
        generated = model.decode(z)
    axes[i].imshow(generated.cpu().squeeze(), cmap='gray')
    axes[i].set_title(f'α={alpha:.1f}', fontsize=9)
    axes[i].axis('off')

plt.suptitle('Morphing: 3 → 7 via Latent Space Interpolation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/vae_morphing.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Digit morphing saved")

print("\n✅ Module 5, Script 02 complete!")
print("VAE creates a SMOOTH, STRUCTURED latent space — perfect for generation!")
