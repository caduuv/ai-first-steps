"""
Module 5 — Script 01: Autoencoder
===================================

An autoencoder learns to compress data into a low-dimensional
representation and then reconstruct it. It's the simplest
generative model and the foundation for VAEs.

Topics:
  - Encoder-decoder architecture
  - Bottleneck and latent space
  - Reconstruction loss
  - Latent space visualization
  - Generating new data by sampling the latent space

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
# ## 1. Define the Autoencoder

# %%
class Autoencoder(nn.Module):
    """
    A simple autoencoder for MNIST.
    
    Encoder: 784 → 256 → 64 → latent_dim
    Decoder: latent_dim → 64 → 256 → 784
    
    The bottleneck (latent_dim) forces the network to learn
    a compressed representation. Smaller = more compression.
    """
    
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),  # Output in [0, 1] to match pixel values
            nn.Unflatten(1, (1, 28, 28)),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


# %% [markdown]
# ## 2. Prepare Data and Train

# %%
print("=== Autoencoder Training ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Create model
latent_dim = 16
model = Autoencoder(latent_dim=latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

total_params = sum(p.numel() for p in model.parameters())
print(f"Latent dimension: {latent_dim}")
print(f"Parameters: {total_params:,}")
print(f"Compression ratio: {28*28}/{latent_dim} = {28*28/latent_dim:.0f}x")

# Training loop
n_epochs = 15
train_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for images, _ in train_loader:  # Labels not used!
        images = images.to(device)
        
        # Forward
        reconstructions = model(images)
        loss = criterion(reconstructions, images)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 3 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:>2}/{n_epochs}: loss = {avg_loss:.6f}")

# %% [markdown]
# ## 3. Visualize Reconstructions

# %%
model.eval()
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:10].to(device)

with torch.no_grad():
    reconstructions = model(test_images)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Recon.', fontsize=11, fontweight='bold')
plt.suptitle('Autoencoder Reconstructions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/ae_reconstructions.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Reconstructions saved")

# %% [markdown]
# ## 4. Explore the Latent Space

# %%
# Encode all test images and visualize in 2D
# If latent_dim > 2, we use PCA to project to 2D

model.eval()
all_z = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        z = model.encode(images.to(device))
        all_z.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

all_z = np.concatenate(all_z)
all_labels = np.concatenate(all_labels)

# Project to 2D if needed
if latent_dim > 2:
    from sklearn.decomposition import PCA
    z_2d = PCA(n_components=2).fit_transform(all_z)
else:
    z_2d = all_z

fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=all_labels, cmap='tab10',
                      alpha=0.5, s=5)
plt.colorbar(scatter, ax=ax, label='Digit')
ax.set_xlabel('Latent Dim 1', fontsize=11)
ax.set_ylabel('Latent Dim 2', fontsize=11)
ax.set_title('Autoencoder Latent Space (PCA to 2D)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/ae_latent_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Latent space visualization saved")

# %% [markdown]
# ## 5. Generating New Images

# %%
# Sample random points from the latent space
# Problem: the latent space is NOT structured — sampling may give garbage!
# This is why we need VAEs (next script).

model.eval()
n_samples = 64

# Random points in latent space
z_random = torch.randn(n_samples, latent_dim).to(device)
with torch.no_grad():
    generated = model.decode(z_random)

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        axes[i, j].imshow(generated[idx].cpu().squeeze(), cmap='gray')
        axes[i, j].axis('off')

plt.suptitle('Generated Images (Random Latent Sampling)\nNote: Quality varies — AE latent space is unstructured!',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/ae_generated.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Generated images saved")

# %% [markdown]
# ## 6. Effect of Latent Dimension

# %%
print("\n=== Latent Dimension Comparison ===")

fig, axes = plt.subplots(4, 10, figsize=(15, 6))

dims_to_test = [2, 8, 32, 128]
sample_images = test_images[:10]

for row, dim in enumerate(dims_to_test):
    ae = Autoencoder(latent_dim=dim).to(device)
    opt = optim.Adam(ae.parameters(), lr=1e-3)
    
    # Quick training (5 epochs)
    ae.train()
    for ep in range(5):
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            recons = ae(imgs)
            loss = criterion(recons, imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    ae.eval()
    with torch.no_grad():
        recons = ae(sample_images)
    
    for col in range(10):
        axes[row, col].imshow(recons[col].cpu().squeeze(), cmap='gray')
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(f'dim={dim}', fontsize=10, fontweight='bold', rotation=0, labelpad=40)
    print(f"  dim={dim:>3}: final loss = {loss.item():.6f}")

plt.suptitle('Effect of Latent Dimension on Reconstruction Quality',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/ae_latent_dims.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Latent dimension comparison saved")

print("\n✅ Module 5, Script 01 complete!")
print("Key takeaway: Autoencoders reconstruct well but can't generate reliably.")
print("Next: VAEs fix this with a structured latent space! →")
