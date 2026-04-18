"""
Module 6 — Evaluation Utilities
=================================

Evaluation metrics and visualization for conditional generative models.
Includes FID (concept), visual inspection, and per-class quality analysis.

Usage:
    python evaluation/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg


def compute_fid_score(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance (FID) between two sets of features.
    
    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^(1/2))
    
    Lower FID = generated images are more similar to real images.
    
    Note: In practice, features come from a pretrained Inception network.
    Here we use raw pixel features as a simplified demonstration.
    """
    mu_r = np.mean(real_features, axis=0)
    mu_f = np.mean(fake_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_f = np.cov(fake_features, rowvar=False)
    
    # Mean difference
    diff = mu_r - mu_f
    
    # Matrix square root
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    
    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


def visual_quality_report(generator, dataloader, n_classes, latent_dim,
                           device, save_dir='evaluation_output'):
    """Generate a comprehensive visual quality report."""
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    
    # 1. Per-class generation grid
    fig, axes = plt.subplots(n_classes, 10, figsize=(12, n_classes * 1.2))
    
    for cls in range(n_classes):
        z = torch.randn(10, latent_dim).to(device)
        labels = torch.full((10,), cls, dtype=torch.long).to(device)
        
        with torch.no_grad():
            images = generator(z, labels).cpu()
        images = (images + 1) / 2
        
        for j in range(10):
            axes[cls, j].imshow(images[j].squeeze(), cmap='gray')
            axes[cls, j].axis('off')
        axes[cls, 0].set_ylabel(f'{cls}', fontsize=10, fontweight='bold',
                                  rotation=0, labelpad=12)
    
    plt.suptitle('Per-Class Generated Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Diversity check: generate multiple samples for same class + noise
    fig, axes = plt.subplots(3, 10, figsize=(12, 3.5))
    z_fixed = torch.randn(1, latent_dim).to(device)
    
    for i in range(10):
        # Same noise, different class
        labels = torch.tensor([i]).to(device)
        with torch.no_grad():
            img = generator(z_fixed, labels).cpu()
        img = (img + 1) / 2
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        axes[0, i].set_title(f'{i}', fontsize=9)
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Same z\nDiff label', fontsize=8, rotation=0, labelpad=40)
    
    for i in range(10):
        # Different noise, same class (digit 5)
        z = torch.randn(1, latent_dim).to(device)
        labels = torch.tensor([5]).to(device)
        with torch.no_grad():
            img = generator(z, labels).cpu()
        img = (img + 1) / 2
        axes[1, i].imshow(img.squeeze(), cmap='gray')
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('Diff z\nSame label', fontsize=8, rotation=0, labelpad=40)
    
    for i in range(10):
        # Random everything
        z = torch.randn(1, latent_dim).to(device)
        labels = torch.randint(0, 10, (1,)).to(device)
        with torch.no_grad():
            img = generator(z, labels).cpu()
        img = (img + 1) / 2
        axes[2, i].imshow(img.squeeze(), cmap='gray')
        axes[2, i].axis('off')
    axes[2, 0].set_ylabel('Random\nboth', fontsize=8, rotation=0, labelpad=40)
    
    plt.suptitle('Diversity Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'diversity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Simplified FID using pixel features
    print("\n=== Simplified FID (Pixel Features) ===")
    real_features = []
    for imgs, _ in dataloader:
        real_features.append(imgs.view(imgs.size(0), -1).numpy())
        if len(real_features) >= 10:
            break
    real_features = np.concatenate(real_features)[:1000]
    
    fake_features = []
    for _ in range(8):
        z = torch.randn(128, latent_dim).to(device)
        labels = torch.randint(0, n_classes, (128,)).to(device)
        with torch.no_grad():
            imgs = generator(z, labels).cpu()
        imgs = (imgs + 1) / 2
        fake_features.append(imgs.view(128, -1).numpy())
    fake_features = np.concatenate(fake_features)[:1000]
    
    fid = compute_fid_score(real_features, fake_features)
    print(f"Simplified FID Score: {fid:.2f}")
    print("(Note: Real FID uses Inception features, not raw pixels)")
    
    print(f"\n📊 Quality report saved to {save_dir}/")


if __name__ == '__main__':
    print("Run this module's evaluation functions from a training script.")
    print("See training/train_cgan.py for usage.")
