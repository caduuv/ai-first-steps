"""Evaluation utilities for CIFAR-10 classifier."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data.dataset import get_dataloaders, CLASS_NAMES
from models.cnn import CIFAR10CNN


def evaluate_model(model_path: str = 'best_model.pth'):
    """Load and evaluate a trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    _, test_loader = get_dataloaders(batch_size=256, augment=False)
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('CIFAR-10 Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("📊 Confusion matrix saved")


if __name__ == '__main__':
    evaluate_model()
