"""CNN model architecture for CIFAR-10 classification."""

import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """
    A deeper CNN for CIFAR-10 with BatchNorm and Dropout.
    
    Architecture:
      Conv Block 1: 3→32→32, MaxPool → 16×16
      Conv Block 2: 32→64→64, MaxPool → 8×8
      Conv Block 3: 64→128→128, MaxPool → 4×4
      FC: 128*4*4 → 256 → 10
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(dropout_rate),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
