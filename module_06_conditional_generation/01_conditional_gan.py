"""
Module 6 — Script 01: Conditional GAN (cGAN)
==============================================

Generate SPECIFIC digits on demand! This is the key step toward
generating medical images conditioned on diagnoses.

Topics:
  - Label embedding
  - Conditioning G and D on class labels
  - Training a cGAN
  - Generating specific digits
  - Label interpolation

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
# ## 1. Conditional Generator and Discriminator

# %%
class ConditionalGenerator(nn.Module):
    """
    Generator conditioned on class label.
    
    Input: noise z (100-dim) + label embedding (10-dim) = 110-dim
    Output: 28×28 image
    
    The label tells G WHAT to generate.
    """
    
    def __init__(self, latent_dim: int = 100, n_classes: int = 10, embed_dim: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Label embedding: converts integer label to dense vector
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
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
    
    def forward(self, z, labels):
        # Embed the label
        label_embedding = self.label_embed(labels)  # (batch, embed_dim)
        
        # Concatenate noise and label
        x = torch.cat([z, label_embedding], dim=1)  # (batch, latent_dim + embed_dim)
        
        img = self.model(x)
        return img.view(-1, 1, 28, 28)


class ConditionalDiscriminator(nn.Module):
    """
    Discriminator conditioned on class label.
    
    Input: flattened image (784) + label embedding (10) = 794
    Output: probability of being real
    
    D has to learn: "Is this a REAL example of digit Y?"
    """
    
    def __init__(self, n_classes: int = 10, embed_dim: int = 10):
        super().__init__()
        
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        
        self.model = nn.Sequential(
            nn.Linear(28 * 28 + embed_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_embedding = self.label_embed(labels)
        x = torch.cat([img_flat, label_embedding], dim=1)
        return self.model(x)


# %% [markdown]
# ## 2. Training Setup

# %%
print("=== Conditional GAN Training ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

latent_dim = 100
n_classes = 10

generator = ConditionalGenerator(latent_dim, n_classes).to(device)
discriminator = ConditionalDiscriminator(n_classes).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterion = nn.BCELoss()

print(f"G parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"D parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# %% [markdown]
# ## 3. Training Loop

# %%
n_epochs = 30
g_losses = []
d_losses = []

for epoch in range(n_epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    n_batches = 0
    
    for real_images, real_labels in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        
        real_target = torch.ones(batch_size, 1).to(device) * 0.9  # Label smoothing
        fake_target = torch.zeros(batch_size, 1).to(device)
        
        # ====== Train Discriminator ======
        optimizer_D.zero_grad()
        
        # Real images with correct labels
        d_real = discriminator(real_images, real_labels)
        d_real_loss = criterion(d_real, real_target)
        
        # Fake images with random labels
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_labels = torch.randint(0, n_classes, (batch_size,)).to(device)
        fake_images = generator(z, fake_labels).detach()
        d_fake = discriminator(fake_images, fake_labels)
        d_fake_loss = criterion(d_fake, fake_target)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # ====== Train Generator ======
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_labels = torch.randint(0, n_classes, (batch_size,)).to(device)
        gen_images = generator(z, gen_labels)
        d_gen = discriminator(gen_images, gen_labels)
        g_loss = criterion(d_gen, torch.ones(batch_size, 1).to(device))
        
        g_loss.backward()
        optimizer_G.step()
        
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        n_batches += 1
    
    g_losses.append(epoch_g_loss / n_batches)
    d_losses.append(epoch_d_loss / n_batches)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:>2}/{n_epochs}: G_loss={g_losses[-1]:.4f}, D_loss={d_losses[-1]:.4f}")

# %% [markdown]
# ## 4. Generate Specific Digits

# %%
print("\n=== Generating Specific Digits ===")
generator.eval()

fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for digit in range(10):
    z = torch.randn(10, latent_dim).to(device)
    labels = torch.full((10,), digit, dtype=torch.long).to(device)
    
    with torch.no_grad():
        generated = generator(z, labels).cpu()
    
    generated = (generated + 1) / 2  # [-1,1] → [0,1]
    
    for j in range(10):
        axes[digit, j].imshow(generated[j].squeeze(), cmap='gray')
        axes[digit, j].axis('off')
    
    axes[digit, 0].set_ylabel(f'{digit}', fontsize=14, fontweight='bold',
                                rotation=0, labelpad=15)

plt.suptitle('cGAN: Each Row = Specific Digit', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_06_conditional_generation/cgan_specific_digits.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Specific digit generation saved")

# %% [markdown]
# ## 5. Label Interpolation

# %%
# Interpolate between two digit labels in embedding space
generator.eval()

digit_a, digit_b = 3, 8
z_fixed = torch.randn(1, latent_dim).to(device)

# Get embeddings
embed_a = generator.label_embed(torch.tensor([digit_a]).to(device))
embed_b = generator.label_embed(torch.tensor([digit_b]).to(device))

n_steps = 12
fig, axes = plt.subplots(1, n_steps, figsize=(15, 2))

for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
    # Interpolate embeddings
    embed_interp = (1 - alpha) * embed_a + alpha * embed_b
    
    # Bypass label_embed and feed directly
    x = torch.cat([z_fixed, embed_interp], dim=1)
    with torch.no_grad():
        img = generator.model(x).view(1, 1, 28, 28)
    
    img = (img + 1) / 2
    axes[i].imshow(img.cpu().squeeze(), cmap='gray')
    axes[i].set_title(f'α={alpha:.1f}', fontsize=8)
    axes[i].axis('off')

plt.suptitle(f'Label Interpolation: {digit_a} → {digit_b}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_06_conditional_generation/cgan_interpolation.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Label interpolation saved")

# %% [markdown]
# ## 6. Training Curves

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(g_losses, label='Generator', color='#3498DB', linewidth=1.5)
ax.plot(d_losses, label='Discriminator', color='#E74C3C', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('cGAN Training Losses', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module_06_conditional_generation/cgan_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Training curves saved")

print("\n✅ Module 6, Script 01 complete!")
print("You can now generate SPECIFIC digits on demand!")
