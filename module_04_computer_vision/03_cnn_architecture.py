"""
Module 4 — Script 03: CNN Architecture
========================================

Build a CNN step by step, understanding each component.
We'll classify MNIST digits with >98% accuracy.

Topics:
  - CNN architecture design
  - Feature map visualization
  - Training a CNN
  - Understanding what each layer learns

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
# ## 1. Define the CNN

# %%
class MNISTClassifier(nn.Module):
    """
    A CNN for MNIST digit classification.
    
    Architecture breakdown:
      Conv1: 1→16 filters, 3×3 → detect edges, simple patterns
      Conv2: 16→32 filters, 3×3 → detect combinations of edges (corners, curves)
      FC1: 32*7*7 → 128 → combine spatial features
      FC2: 128 → 10 → classify into 10 digits
    """
    
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Block 1: 1→16, with pooling: 28×28 → 14×14
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 28, 28) → (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (16, 14, 14)
            
            # Conv Block 2: 16→32, with pooling: 14×14 → 7×7
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (32, 7, 7)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),                    # → (32 * 7 * 7) = 1568
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),              # 10 digit classes
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def get_feature_maps(self, x):
        """Extract feature maps from each conv layer for visualization."""
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x.detach())
        return features


# %% [markdown]
# ## 2. Prepare Data

# %%
print("=== Preparing MNIST Data ===")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batches per epoch: {len(train_loader)}")

# %% [markdown]
# ## 3. Train the CNN

# %%
print("\n=== Training CNN ===")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = MNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Training loop
n_epochs = 5
train_losses = []
test_accs = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            test_total += batch_y.size(0)
            test_correct += predicted.eq(batch_y).sum().item()
    
    test_acc = test_correct / test_total
    test_accs.append(test_acc)
    
    print(f"  Epoch {epoch+1}/{n_epochs}: loss={train_loss:.4f}, "
          f"train_acc={train_acc:.2%}, test_acc={test_acc:.2%}")

# %% [markdown]
# ## 4. Visualize Feature Maps

# %%
print("\n=== Feature Map Visualization ===")

# Get a sample image
sample, label = test_dataset[0]
sample = sample.unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    features = model.get_feature_maps(sample)

fig, axes = plt.subplots(2, 9, figsize=(16, 4))

# Original image
axes[0, 0].imshow(sample.cpu().squeeze(), cmap='gray')
axes[0, 0].set_title(f'Input ({label})', fontweight='bold', fontsize=9)
axes[0, 0].axis('off')

# First conv layer feature maps (16 filters, show 8)
for i in range(8):
    axes[0, i+1].imshow(features[0][0, i].cpu(), cmap='viridis')
    axes[0, i+1].set_title(f'Conv1 f{i+1}', fontsize=8)
    axes[0, i+1].axis('off')

# Second conv layer feature maps (32 filters, show 8)
axes[1, 0].axis('off')
for i in range(8):
    axes[1, i+1].imshow(features[1][0, i].cpu(), cmap='viridis')
    axes[1, i+1].set_title(f'Conv2 f{i+1}', fontsize=8)
    axes[1, i+1].axis('off')

plt.suptitle('What the CNN "Sees" at Each Layer', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_04_computer_vision/feature_maps.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Feature maps saved")

# %% [markdown]
# ## 5. Confusion Matrix and Misclassifications

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get all predictions
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'MNIST Confusion Matrix (Acc: {test_accs[-1]:.2%})', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('module_04_computer_vision/cnn_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Confusion matrix saved")

print(f"\n✅ Module 4, Script 03 complete!")
print(f"Final test accuracy: {test_accs[-1]:.2%}")
