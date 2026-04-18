"""
Module 6 — Script 02: Conditional VAE (cVAE)
==============================================

A Conditional VAE combines the structured latent space of VAEs
with the ability to control generation via class labels.

Topics:
  - Conditioning encoder and decoder on labels
  - Label-driven generation
  - Comparing cVAE vs cGAN outputs
  - Disentangled latent space

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
# ## 1. Conditional VAE Architecture

# %%
class ConditionalVAE(nn.Module):
    """
    Conditional VAE: Encode and decode with label information.
    
    Encoder: image (784) + label (one-hot 10) → μ, log_var
    Decoder: z (latent) + label (one-hot 10) → reconstructed image
    """
    
    def __init__(self, latent_dim: int = 16, n_classes: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # Encoder: takes image + one-hot label
        self.encoder = nn.Sequential(
            nn.Linear(784 + n_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: takes z + one-hot label
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )
    
    def encode(self, x, y_onehot):
        """Encode image+label to latent distribution parameters."""
        x_flat = x.view(x.size(0), -1)
        inputs = torch.cat([x_flat, y_onehot], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z, y_onehot):
        """Decode latent+label to image."""
        inputs = torch.cat([z, y_onehot], dim=1)
        return self.decoder(inputs).view(-1, 1, 28, 28)
    
    def forward(self, x, y_onehot):
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y_onehot)
        return recon, mu, logvar


# %% [markdown]
# ## 2. Train the cVAE

# %%
print("=== Conditional VAE Training ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

n_classes = 10
latent_dim = 16
model = ConditionalVAE(latent_dim=latent_dim, n_classes=n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def to_onehot(labels, n_classes):
    """Convert integer labels to one-hot vectors."""
    return F.one_hot(labels, n_classes).float()


n_epochs = 20
losses = {'Total': [], 'Recon': [], 'KL': []}

for epoch in range(n_epochs):
    model.train()
    total_loss, recon_total, kl_total = 0, 0, 0
    
    for images, labels in train_loader:
        images = images.to(device)
        y_onehot = to_onehot(labels, n_classes).to(device)
        
        recon, mu, logvar = model(images, y_onehot)
        
        # Loss
        recon_loss = F.binary_cross_entropy(recon.view(-1, 784), images.view(-1, 784), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        recon_total += recon_loss.item()
        kl_total += kl_loss.item()
    
    n = len(train_dataset)
    losses['Total'].append(total_loss / n)
    losses['Recon'].append(recon_total / n)
    losses['KL'].append(kl_total / n)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:>2}: total={total_loss/n:.2f}, recon={recon_total/n:.2f}, kl={kl_total/n:.2f}")

# %% [markdown]
# ## 3. Generate Specific Digits

# %%
print("\n=== cVAE: Generating Specific Digits ===")
model.eval()

fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for digit in range(10):
    z = torch.randn(10, latent_dim).to(device)
    y_onehot = to_onehot(torch.full((10,), digit, dtype=torch.long), n_classes).to(device)
    
    with torch.no_grad():
        generated = model.decode(z, y_onehot).cpu()
    
    for j in range(10):
        axes[digit, j].imshow(generated[j].squeeze(), cmap='gray')
        axes[digit, j].axis('off')
    axes[digit, 0].set_ylabel(f'{digit}', fontsize=14, fontweight='bold',
                                rotation=0, labelpad=15)

plt.suptitle('cVAE: Each Row = Specific Digit', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_06_conditional_generation/cvae_specific_digits.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 cVAE digit generation saved")

# %% [markdown]
# ## 4. Style Transfer Between Digits

# %%
# Take the "style" (latent code) from one digit and the label from another
model.eval()

fig, axes = plt.subplots(10, 10, figsize=(10, 10))

# Get a fixed style from a digit "3"
style_img = test_dataset[2][0].unsqueeze(0).to(device)
style_label = to_onehot(torch.tensor([test_dataset[2][1]]), n_classes).to(device)
with torch.no_grad():
    style_mu, _ = model.encode(style_img, style_label)

# Apply this style to all digit classes
for digit in range(10):
    y_onehot = to_onehot(torch.full((10,), digit, dtype=torch.long), n_classes).to(device)
    z = style_mu.repeat(10, 1) + torch.randn(10, latent_dim).to(device) * 0.3
    
    with torch.no_grad():
        generated = model.decode(z, y_onehot).cpu()
    
    for j in range(10):
        axes[digit, j].imshow(generated[j].squeeze(), cmap='gray')
        axes[digit, j].axis('off')
    axes[digit, 0].set_ylabel(f'{digit}', fontsize=14, fontweight='bold',
                                rotation=0, labelpad=15)

plt.suptitle('cVAE: Same Style, Different Digits', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_06_conditional_generation/cvae_style_transfer.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Style transfer saved")

# %% [markdown]
# ## 5. Latent Space Visualization

# %%
model.eval()
all_mu = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        y_onehot = to_onehot(labels, n_classes).to(device)
        mu, _ = model.encode(images.to(device), y_onehot)
        all_mu.append(mu.cpu().numpy())
        all_labels.append(labels.numpy())

all_mu = np.concatenate(all_mu)
all_labels = np.concatenate(all_labels)

# PCA to 2D
from sklearn.decomposition import PCA
z_2d = PCA(n_components=2).fit_transform(all_mu)

fig, ax = plt.subplots(figsize=(8, 7))
scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.4, s=5)
plt.colorbar(scatter, ax=ax, label='Digit')
ax.set_xlabel('PC1', fontsize=11)
ax.set_ylabel('PC2', fontsize=11)
ax.set_title('cVAE Latent Space (PCA)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_06_conditional_generation/cvae_latent_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Latent space saved")

print("\n✅ Module 6, Script 02 complete!")
