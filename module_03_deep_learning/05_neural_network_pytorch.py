"""
Module 3 — Script 05: Neural Network with PyTorch
===================================================

Now we rewrite our from-scratch network using PyTorch.
You'll see how PyTorch automates everything we did manually:
  - Automatic differentiation (no manual backprop!)
  - Built-in optimizers
  - GPU support
  - Modular layer definitions

Topics:
  - PyTorch tensors vs NumPy arrays
  - nn.Module: defining networks
  - Autograd: automatic differentiation
  - Training loop best practices
  - Comparing with our NumPy implementation

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# %% [markdown]
# ## 1. PyTorch Tensors — NumPy's GPU-Accelerated Cousin

# %%
print("=== PyTorch Tensors ===")

# Creating tensors
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.randn(3, 4)  # Random normal

print(f"Tensor: {t1}")
print(f"Shape: {t2.shape}")
print(f"Dtype: {t2.dtype}")
print(f"Device: {t2.device}")

# NumPy ↔ PyTorch conversion
np_array = np.array([1.0, 2.0, 3.0])
tensor_from_np = torch.from_numpy(np_array)
back_to_np = tensor_from_np.numpy()
print(f"\nNumPy → Tensor → NumPy: {np_array} → {tensor_from_np} → {back_to_np}")

# Automatic differentiation preview
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1
y.backward()              # Compute dy/dx automatically!
print(f"\ny = x² + 3x + 1 at x=2:")
print(f"  y = {y.item():.1f}")
print(f"  dy/dx = {x.grad.item():.1f} (analytical: 2*2 + 3 = 7)")

# %% [markdown]
# ## 2. Defining a Network with nn.Module

# %%
class SimpleNetwork(nn.Module):
    """
    The SAME network we built from scratch, now in PyTorch.
    
    Compare this with our NumPy NeuralNetwork class:
    - No manual weight initialization (nn.Linear does it)
    - No manual forward computation (just chain layers)
    - No manual backward pass (autograd handles it)
    """
    
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Print architecture
        print("Network Architecture:")
        for i, layer in enumerate(self.network):
            print(f"  [{i}] {layer}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Create the network
print("\n=== Network Definition ===")
model = SimpleNetwork(input_size=2, hidden_sizes=[32, 16], output_size=1)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# %% [markdown]
# ## 3. Preparing Data

# %%
print("\n=== Data Preparation ===")

# Generate data
X_np, y_np = make_moons(n_samples=500, noise=0.2, random_state=42)
y_np = y_np.reshape(-1, 1).astype(np.float32)
X_np = X_np.astype(np.float32)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test)

# Create DataLoader for mini-batch training
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Batches per epoch: {len(train_loader)}")

# %% [markdown]
# ## 4. Training Loop — The PyTorch Way

# %%
print("\n=== Training with PyTorch ===")

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Training loop
n_epochs = 200
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(n_epochs):
    model.train()  # Set to training mode
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        # 1. Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # 2. Backward pass (automatic!)
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        
        # 3. Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Track metrics
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Evaluate (no gradient computation needed)
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        train_pred = model(X_train_t)
        test_pred = model(X_test_t)
        test_loss = criterion(test_pred, y_test_t).item()
        test_losses.append(test_loss)
        
        train_acc = ((train_pred > 0.5).float() == y_train_t).float().mean().item()
        test_acc = ((test_pred > 0.5).float() == y_test_t).float().mean().item()
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    if epoch % 40 == 0:
        print(f"  Epoch {epoch:>3}: train_loss={avg_loss:.4f}, test_loss={test_loss:.4f}, "
              f"train_acc={train_acc:.2%}, test_acc={test_acc:.2%}")

print(f"\nFinal → Train: {train_accs[-1]:.1%}, Test: {test_accs[-1]:.1%}")

# %% [markdown]
# ## 5. Visualize Results

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Loss curves
axes[0].plot(train_losses, label='Train', color='#3498DB', linewidth=1.5)
axes[0].plot(test_losses, label='Test', color='#E74C3C', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Test Loss', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(train_accs, label='Train', color='#3498DB', linewidth=1.5)
axes[1].plot(test_accs, label='Test', color='#E74C3C', linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training & Test Accuracy', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Decision boundary
model.eval()
xx = np.linspace(-2, 3, 200)
yy = np.linspace(-1.5, 2, 200)
XX, YY = np.meshgrid(xx, yy)
grid = torch.from_numpy(np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32))

with torch.no_grad():
    Z = model(grid).numpy().reshape(XX.shape)

axes[2].contourf(XX, YY, Z, levels=50, cmap='RdYlBu_r', alpha=0.7)
axes[2].scatter(X_test[y_test.flatten() == 0, 0], X_test[y_test.flatten() == 0, 1],
                c='#3498DB', s=15, alpha=0.7, label='Class 0')
axes[2].scatter(X_test[y_test.flatten() == 1, 0], X_test[y_test.flatten() == 1, 1],
                c='#E74C3C', s=15, alpha=0.7, label='Class 1')
axes[2].set_title(f'Decision Boundary (Test Acc: {test_accs[-1]:.1%})', fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.savefig('module_03_deep_learning/pytorch_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 PyTorch results saved")

# %% [markdown]
# ## 6. Comparison: NumPy vs PyTorch

# %%
print("\n=== NumPy vs PyTorch Comparison ===")
print(f"{'Feature':<25} {'NumPy (from scratch)':<25} {'PyTorch':<25}")
print(f"{'─'*25} {'─'*25} {'─'*25}")
print(f"{'Forward pass':<25} {'Manual matrix ops':<25} {'Automatic':<25}")
print(f"{'Backward pass':<25} {'Manual chain rule':<25} {'Autograd':<25}")
print(f"{'Optimizer':<25} {'Manual SGD':<25} {'Adam, SGD, etc.':<25}")
print(f"{'GPU support':<25} {'No':<25} {'Yes (CUDA)':<25}")
print(f"{'Debugging':<25} {'Print statements':<25} {'TensorBoard, pdb':<25}")
print(f"{'Code lines':<25} {'~100 lines':<25} {'~30 lines':<25}")
print(f"{'Understanding':<25} {'Deep ✓':<25} {'Practical ✓':<25}")

print("\n💡 Both are valuable! NumPy teaches you HOW it works.")
print("   PyTorch lets you focus on WHAT to build.")

print("\n✅ Module 3, Script 05 complete!")
print("\n🎉 Module 3 complete! You can build neural networks from scratch AND with PyTorch.")
print("Next: Module 4 — Computer Vision →")
