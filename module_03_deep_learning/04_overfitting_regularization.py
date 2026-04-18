"""
Module 3 — Script 04: Overfitting & Regularization
====================================================

Understanding overfitting is crucial for building models that
generalize to unseen data. This script demonstrates the problem
and several solutions.

Topics:
  - What is overfitting?
  - L2 regularization (weight decay)
  - Dropout
  - Early stopping
  - Comparing regularized vs unregularized models

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from utils import sigmoid, relu, relu_derivative

# %% [markdown]
# ## 1. Generate Data and Demonstrate Overfitting

# %%
print("=== Overfitting Demonstration ===")
np.random.seed(42)

# Generate a noisy non-linear dataset
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
y = y.reshape(-1, 1).astype(float)

# Split: intentionally small training set to encourage overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %% [markdown]
# ## 2. Network WITHOUT Regularization

# %%
class RegularizedNetwork:
    """Neural network with optional L2 regularization and dropout."""
    
    def __init__(self, layer_sizes, l2_lambda=0.0, dropout_rate=0.0, seed=42):
        np.random.seed(seed)
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.n_layers = len(layer_sizes) - 1
        
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, X, training=True):
        """Forward pass with optional dropout."""
        self.z_cache = []
        self.a_cache = [X]
        self.dropout_masks = []
        
        a = X
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.z_cache.append(z)
            
            if i < self.n_layers - 1:
                a = np.maximum(0, z)  # ReLU
                
                # Dropout (only during training)
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                    a *= mask
                    a /= (1 - self.dropout_rate)  # Inverted dropout scaling
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append(np.ones_like(a))
            else:
                a = sigmoid(z)
            
            self.a_cache.append(a)
        
        return a
    
    def compute_loss(self, y_true, y_pred):
        """Loss with optional L2 penalty."""
        eps = 1e-7
        bce = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        
        # L2 regularization term
        l2_penalty = 0
        if self.l2_lambda > 0:
            for W in self.weights:
                l2_penalty += np.sum(W ** 2)
            l2_penalty *= self.l2_lambda / (2 * len(y_true))
        
        return bce + l2_penalty
    
    def backward(self, X, y, lr=0.1):
        """Backward pass with L2 regularization."""
        n = X.shape[0]
        
        dz = self.a_cache[-1] - y
        
        for i in range(self.n_layers - 1, -1, -1):
            a_prev = self.a_cache[i]
            
            dW = (a_prev.T @ dz) / n
            db = np.sum(dz, axis=0, keepdims=True) / n
            
            # Add L2 gradient: d/dW (λ/2n * ||W||²) = λ/n * W
            if self.l2_lambda > 0:
                dW += (self.l2_lambda / n) * self.weights[i]
            
            if i > 0:
                da = dz @ self.weights[i].T
                # Apply dropout mask to gradient
                if self.dropout_rate > 0 and i - 1 < len(self.dropout_masks):
                    da *= self.dropout_masks[i - 1]
                    da /= (1 - self.dropout_rate)
                dz = da * (self.z_cache[i-1] > 0).astype(float)
            
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db
        
        return self.compute_loss(y, self.a_cache[-1])
    
    def train(self, X_train, y_train, X_val, y_val, n_epochs=2000, lr=0.1,
              early_stopping_patience=0):
        """Train with optional early stopping."""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(n_epochs):
            # Forward + backward on training data
            self.forward(X_train, training=True)
            train_loss = self.backward(X_train, y_train, lr=lr)
            train_losses.append(train_loss)
            
            # Evaluate on validation data (no dropout)
            val_pred = self.forward(X_val, training=False)
            val_loss = self.compute_loss(y_val, val_pred)
            val_losses.append(val_loss)
            
            # Early stopping
            if early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [(W.copy(), b.copy()) for W, b in zip(self.weights, self.biases)]
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"  Early stopping at epoch {epoch} (patience={early_stopping_patience})")
                        # Restore best weights
                        for i, (W, b) in enumerate(best_weights):
                            self.weights[i] = W
                            self.biases[i] = b
                        break
            
            if epoch % 500 == 0:
                train_acc = np.mean((self.forward(X_train, False) > 0.5) == y_train)
                val_acc = np.mean((self.forward(X_val, False) > 0.5) == y_val)
                print(f"  Epoch {epoch:>4}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"train_acc={train_acc:.2%}, val_acc={val_acc:.2%}")
        
        return train_losses, val_losses


# %% [markdown]
# ## 3. Compare: No Regularization vs L2 vs Dropout vs Early Stopping

# %%
configs = [
    ("No Regularization", {"l2_lambda": 0.0, "dropout_rate": 0.0}, {}),
    ("L2 Regularization (λ=0.1)", {"l2_lambda": 0.1, "dropout_rate": 0.0}, {}),
    ("Dropout (rate=0.3)", {"l2_lambda": 0.0, "dropout_rate": 0.3}, {}),
    ("Early Stopping (patience=100)", {"l2_lambda": 0.0, "dropout_rate": 0.0},
     {"early_stopping_patience": 100}),
]

results = {}
for name, net_kwargs, train_kwargs in configs:
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    net = RegularizedNetwork([2, 32, 16, 1], **net_kwargs, seed=42)
    train_losses, val_losses = net.train(X_train, y_train, X_test, y_test,
                                          n_epochs=3000, lr=0.5, **train_kwargs)
    
    # Final accuracy
    train_pred = net.forward(X_train, training=False)
    test_pred = net.forward(X_test, training=False)
    train_acc = np.mean((train_pred > 0.5) == y_train)
    test_acc = np.mean((test_pred > 0.5) == y_test)
    
    results[name] = {
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_acc': train_acc, 'test_acc': test_acc, 'network': net
    }
    print(f"\n  Final → Train: {train_acc:.2%}, Test: {test_acc:.2%}")

# %% [markdown]
# ## 4. Visualize Results

# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for idx, (name, res) in enumerate(results.items()):
    # Loss curves
    ax_loss = axes[0, idx]
    ax_loss.plot(res['train_losses'], label='Train', color='#3498DB', linewidth=1, alpha=0.7)
    ax_loss.plot(res['val_losses'], label='Validation', color='#E74C3C', linewidth=1, alpha=0.7)
    ax_loss.set_title(name.split('(')[0].strip(), fontweight='bold', fontsize=10)
    ax_loss.set_xlabel('Epoch', fontsize=9)
    ax_loss.set_ylabel('Loss', fontsize=9)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_ylim(0, 1)
    
    # Decision boundaries
    ax_db = axes[1, idx]
    xx = np.linspace(-2, 3, 200)
    yy = np.linspace(-1.5, 2, 200)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    Z = res['network'].forward(grid, training=False).reshape(XX.shape)
    
    ax_db.contourf(XX, YY, Z, levels=50, cmap='RdYlBu_r', alpha=0.7)
    ax_db.scatter(X_test[y_test.flatten() == 0, 0], X_test[y_test.flatten() == 0, 1],
                  c='#3498DB', s=10, alpha=0.6)
    ax_db.scatter(X_test[y_test.flatten() == 1, 0], X_test[y_test.flatten() == 1, 1],
                  c='#E74C3C', s=10, alpha=0.6)
    ax_db.set_title(f'Test Acc: {res["test_acc"]:.1%}', fontsize=10)

plt.suptitle('Regularization Comparison', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_03_deep_learning/regularization.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Regularization comparison plot saved")

# Summary table
print("\n=== Summary ===")
print(f"{'Method':<30} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>10}")
print(f"{'─'*30} {'─'*10} {'─'*10} {'─'*10}")
for name, res in results.items():
    gap = res['train_acc'] - res['test_acc']
    print(f"{name.split('(')[0].strip():<30} {res['train_acc']:>10.1%} "
          f"{res['test_acc']:>10.1%} {gap:>10.1%}")

print("\n💡 Key insight: Regularization reduces the gap between train and test accuracy!")
print("   The goal is NOT to maximize training accuracy, but TEST accuracy.")

print("\n✅ Module 3, Script 04 complete!")
