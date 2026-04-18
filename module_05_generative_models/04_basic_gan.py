"""
Module 5 — Script 04: Basic GAN
=================================

Build and train a GAN to generate MNIST digits from random noise.
This is your first real generative model!

Topics:
  - Generator architecture
  - Discriminator architecture
  - Alternating training
  - Monitoring GAN training
  - Generated digit samples

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
# ## 1. Define Generator and Discriminator

# %%
class Generator(nn.Module):
    """
    Generator: Random noise (z) → Fake image.
    
    Takes a 100-dimensional noise vector and upscales it
    to a 28×28 image through transposed convolutions.
    """
    
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.model = nn.Sequential(
            # latent_dim → 256
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            # 256 → 512
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            # 512 → 1024
            nn.Linear(1024 if False else 512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            # 1024 → 784 (28×28)
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),  # Output in [-1, 1]
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """
    Discriminator: Image → Real/Fake probability.
    
    A simple classifier that outputs the probability
    that the input image is real (vs generated).
    """
    
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, img):
        return self.model(img)


# Fix the generator (the 512→1024 layer had a typo)
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)


# %% [markdown]
# ## 2. Training Setup

# %%
print("=== Basic GAN Training ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Data: normalize to [-1, 1] to match Tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # [0,1] → [-1,1]
])

dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Models
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers (separate for G and D!)
optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterion = nn.BCELoss()

g_params = sum(p.numel() for p in generator.parameters())
d_params = sum(p.numel() for p in discriminator.parameters())
print(f"Generator parameters: {g_params:,}")
print(f"Discriminator parameters: {d_params:,}")

# Fixed noise for visualizing progress
fixed_noise = torch.randn(64, latent_dim).to(device)

# %% [markdown]
# ## 3. Training Loop

# %%
n_epochs = 30
g_losses = []
d_losses = []
d_real_accs = []
d_fake_accs = []

for epoch in range(n_epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    epoch_d_real_acc = 0
    epoch_d_fake_acc = 0
    n_batches = 0
    
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ====== Train Discriminator ======
        # D should maximize: log(D(x)) + log(1 - D(G(z)))
        
        optimizer_D.zero_grad()
        
        # Real images
        d_real = discriminator(real_images)
        d_real_loss = criterion(d_real, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z).detach()  # Detach: don't update G here
        d_fake = discriminator(fake_images)
        d_fake_loss = criterion(d_fake, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # ====== Train Generator ======
        # G should maximize: log(D(G(z))) (non-saturating loss)
        
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        d_fake = discriminator(fake_images)
        g_loss = criterion(d_fake, real_labels)  # G wants D to think fakes are real
        
        g_loss.backward()
        optimizer_G.step()
        
        # Track metrics
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_d_real_acc += (d_real > 0.5).float().mean().item()
        epoch_d_fake_acc += (d_fake < 0.5).float().mean().item()
        n_batches += 1
    
    # Average metrics
    g_losses.append(epoch_g_loss / n_batches)
    d_losses.append(epoch_d_loss / n_batches)
    d_real_accs.append(epoch_d_real_acc / n_batches)
    d_fake_accs.append(epoch_d_fake_acc / n_batches)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:>2}/{n_epochs}: "
              f"G_loss={g_losses[-1]:.4f}, D_loss={d_losses[-1]:.4f}, "
              f"D(real)={d_real_accs[-1]:.2%}, D(fake)={d_fake_accs[-1]:.2%}")

# %% [markdown]
# ## 4. Visualize Generated Samples

# %%
generator.eval()
with torch.no_grad():
    generated = generator(fixed_noise).cpu()

# Denormalize from [-1,1] to [0,1]
generated = (generated + 1) / 2

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        axes[i, j].imshow(generated[i * 8 + j].squeeze(), cmap='gray')
        axes[i, j].axis('off')

plt.suptitle(f'GAN Generated Digits (Epoch {n_epochs})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('module_05_generative_models/gan_generated.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Generated digits saved")

# %% [markdown]
# ## 5. Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

ax1.plot(g_losses, label='Generator', color='#3498DB', linewidth=1.5)
ax1.plot(d_losses, label='Discriminator', color='#E74C3C', linewidth=1.5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('GAN Training Losses', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(d_real_accs, label='D(real) accuracy', color='#2ECC71', linewidth=1.5)
ax2.plot(d_fake_accs, label='D(fake) accuracy', color='#F39C12', linewidth=1.5)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Discriminator Accuracy', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_05_generative_models/gan_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Training curves saved")

print("\n✅ Module 5, Script 04 complete!")
print("You just trained a GAN to generate handwritten digits from noise!")
