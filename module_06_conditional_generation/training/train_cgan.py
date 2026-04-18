"""
Module 6 — Full cGAN Training Pipeline
========================================

A production-ready training script for the Conditional GAN,
with logging, checkpointing, and evaluation.

Usage:
    python training/train_cgan.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from training.train_utils import TrainingLogger, save_checkpoint, generate_and_save_grid


class CGANGenerator(nn.Module):
    """Conditional Generator with label embedding."""
    
    def __init__(self, latent_dim=100, n_classes=10, embed_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
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
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    
    def forward(self, z, labels):
        embed = self.label_embed(labels)
        x = torch.cat([z, embed], dim=1)
        return self.model(x).view(-1, 1, 28, 28)


class CGANDiscriminator(nn.Module):
    """Conditional Discriminator with spectral normalization."""
    
    def __init__(self, n_classes=10, embed_dim=10):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(784 + embed_dim, 512)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        embed = self.label_embed(labels)
        x = torch.cat([img_flat, embed], dim=1)
        return self.model(x)


def main():
    print("=" * 55)
    print("  Conditional GAN — Full Training Pipeline")
    print("=" * 55)
    
    # Config
    latent_dim = 100
    n_classes = 10
    n_epochs = 50
    batch_size = 128
    lr = 2e-4
    save_dir = 'cgan_output'
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Models
    G = CGANGenerator(latent_dim, n_classes).to(device)
    D = CGANDiscriminator(n_classes).to(device)
    
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    logger = TrainingLogger(save_dir)
    
    print(f"G params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"D params: {sum(p.numel() for p in D.parameters()):,}")
    print(f"Training for {n_epochs} epochs...\n")
    
    # Training
    for epoch in range(n_epochs):
        G.train()
        D.train()
        g_loss_sum, d_loss_sum = 0, 0
        n_batches = 0
        
        for real_imgs, real_labels in dataloader:
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            
            real_target = torch.ones(bs, 1).to(device) * 0.9
            fake_target = torch.zeros(bs, 1).to(device)
            
            # Train D
            optimizer_D.zero_grad()
            d_real = D(real_imgs, real_labels)
            z = torch.randn(bs, latent_dim).to(device)
            fake_labels = torch.randint(0, n_classes, (bs,)).to(device)
            fake_imgs = G(z, fake_labels).detach()
            d_fake = D(fake_imgs, fake_labels)
            d_loss = criterion(d_real, real_target) + criterion(d_fake, fake_target)
            d_loss.backward()
            optimizer_D.step()
            
            # Train G
            optimizer_G.zero_grad()
            z = torch.randn(bs, latent_dim).to(device)
            gen_labels = torch.randint(0, n_classes, (bs,)).to(device)
            gen_imgs = G(z, gen_labels)
            d_gen = D(gen_imgs, gen_labels)
            g_loss = criterion(d_gen, torch.ones(bs, 1).to(device))
            g_loss.backward()
            optimizer_G.step()
            
            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            n_batches += 1
        
        g_avg = g_loss_sum / n_batches
        d_avg = d_loss_sum / n_batches
        logger.log(epoch, G_Loss=g_avg, D_Loss=d_avg)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{n_epochs}: G={g_avg:.4f}, D={d_avg:.4f}")
            generate_and_save_grid(G, n_classes, latent_dim, device,
                                    os.path.join(save_dir, f'samples_epoch_{epoch+1}.png'))
    
    # Save final model and plots
    save_checkpoint(G, optimizer_G, n_epochs, g_avg, os.path.join(save_dir, 'generator.pth'))
    save_checkpoint(D, optimizer_D, n_epochs, d_avg, os.path.join(save_dir, 'discriminator.pth'))
    logger.plot()
    
    print(f"\n✅ Training complete! Results saved to {save_dir}/")


if __name__ == '__main__':
    main()
