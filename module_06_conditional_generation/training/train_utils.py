"""
Module 6 — Training Utilities
===============================

Shared utilities for training conditional generative models.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class TrainingLogger:
    """Logs and visualizes training metrics."""
    
    def __init__(self, save_dir: str = '.'):
        self.metrics = {}
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def log(self, epoch: int, **kwargs):
        """Log metrics for an epoch."""
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def plot(self, filename: str = 'training_curves.png'):
        """Plot all logged metrics."""
        n_metrics = len(self.metrics)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        for idx, (name, values) in enumerate(self.metrics.items()):
            color = colors[idx % len(colors)]
            axes[idx].plot(values, color=color, linewidth=1.5)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(name)
            axes[idx].set_title(name, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save a training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path, device='cpu'):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def generate_and_save_grid(generator, n_classes, latent_dim, device,
                            save_path, n_per_class=10):
    """Generate a grid of images from a conditional generator."""
    generator.eval()
    
    fig, axes = plt.subplots(n_classes, n_per_class,
                              figsize=(n_per_class * 1.2, n_classes * 1.2))
    
    for cls in range(n_classes):
        z = torch.randn(n_per_class, latent_dim).to(device)
        labels = torch.full((n_per_class,), cls, dtype=torch.long).to(device)
        
        with torch.no_grad():
            images = generator(z, labels).cpu()
        
        # Denormalize
        images = (images + 1) / 2
        
        for j in range(n_per_class):
            axes[cls, j].imshow(images[j].squeeze(), cmap='gray')
            axes[cls, j].axis('off')
        axes[cls, 0].set_ylabel(f'{cls}', fontsize=10, fontweight='bold',
                                  rotation=0, labelpad=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
