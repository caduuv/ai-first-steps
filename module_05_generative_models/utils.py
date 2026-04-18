"""
Module 5 — Shared Utilities
=============================

Helper functions for generative model visualization.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show_reconstructions(model, data_loader, device, n=10, save_path=None):
    """Show original images and their reconstructions."""
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:n].to(device)
    
    with torch.no_grad():
        if hasattr(model, 'encode') and hasattr(model, 'decode'):
            # VAE
            mu, logvar = model.encode(images)
            z = model.reparameterize(mu, logvar)
            recons = model.decode(z)
        else:
            # Standard autoencoder
            recons = model(images)
    
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=11, fontweight='bold')
        
        axes[1, i].imshow(recons[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Recon.', fontsize=11, fontweight='bold')
    
    plt.suptitle('Reconstructions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def show_generated_images(images, n_row=8, title='Generated Images', save_path=None):
    """Display a grid of generated images."""
    n = min(len(images), n_row * n_row)
    n_col = n_row
    
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 1.2, n_row * 1.2))
    for i in range(n_row):
        for j in range(n_col):
            idx = i * n_col + j
            if idx < n:
                axes[i, j].imshow(images[idx].cpu().squeeze(), cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space(model, data_loader, device, save_path=None):
    """Plot the 2D latent space colored by digit class."""
    model.eval()
    zs, labels = [], []
    
    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            if hasattr(model, 'encode'):
                mu, _ = model.encode(images)
                z = mu
            else:
                z = model.encoder(images)
            zs.append(z.cpu().numpy())
            labels.append(lbls.numpy())
    
    zs = np.concatenate(zs, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if zs.shape[1] > 2:
        from sklearn.decomposition import PCA
        zs = PCA(n_components=2).fit_transform(zs)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(zs[:, 0], zs[:, 1], c=labels, cmap='tab10',
                          alpha=0.5, s=5)
    plt.colorbar(scatter, ax=ax, label='Digit Class')
    ax.set_xlabel('Latent Dimension 1', fontsize=11)
    ax.set_ylabel('Latent Dimension 2', fontsize=11)
    ax.set_title('Latent Space Visualization', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_losses(losses_dict, save_path=None):
    """Plot training losses from a dictionary."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    for idx, (name, losses) in enumerate(losses_dict.items()):
        color = colors[idx % len(colors)]
        ax.plot(losses, label=name, color=color, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Losses', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
