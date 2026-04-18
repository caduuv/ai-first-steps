"""
Module 3 — Script 02: Neural Network from Scratch (NumPy)
==========================================================

We build a complete multi-layer neural network using ONLY NumPy.
No PyTorch, no TensorFlow — just math. This is the most important
script in the course for understanding how neural networks really work.

Topics:
  - Network architecture (input → hidden → output)
  - Forward propagation
  - Backpropagation with chain rule
  - Training loop with mini-batches
  - Solving XOR!

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import sigmoid, sigmoid_derivative, relu, relu_derivative, one_hot_encode, accuracy

# %% [markdown]
# ## 1. Neural Network Architecture

# %%
class NeuralNetwork:
    """
    A from-scratch neural network with arbitrary layers.
    
    Architecture: Input → [Hidden Layer(s) with ReLU] → Output (Sigmoid/Softmax)
    
    This implementation stores intermediate values needed for backpropagation.
    Understanding this class is understanding HOW neural networks learn.
    """
    
    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01,
                 seed: int = 42):
        """
        Initialize the network.
        
        Args:
            layer_sizes: List of neurons per layer, e.g., [2, 8, 4, 1]
                         means 2 inputs, two hidden layers (8 and 4), 1 output.
            learning_rate: Step size for gradient descent.
        """
        np.random.seed(seed)
        self.lr = learning_rate
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize weights and biases using He initialization
        # He init: w ~ N(0, sqrt(2/fan_in)) — works well with ReLU
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
            print(f"  Layer {i+1}: {fan_in} → {fan_out} "
                  f"(W shape: {W.shape}, b shape: {b.shape})")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute predictions.
        
        Stores intermediate values 'z' (pre-activation) and 'a' (post-activation)
        which are needed for backpropagation.
        """
        self.z_cache = []  # Pre-activation values
        self.a_cache = [X]  # Post-activation values (input is a[0])
        
        a = X
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.z_cache.append(z)
            
            if i < self.n_layers - 1:
                # Hidden layers: ReLU activation
                a = relu(z)
            else:
                # Output layer: Sigmoid activation
                a = sigmoid(z)
            
            self.a_cache.append(a)
        
        return a
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Backward pass: compute gradients and update weights.
        
        This is where the chain rule magic happens!
        
        For the output layer (with sigmoid + BCE):
            dz = a - y  (simplified derivative)
        
        For hidden layers:
            dz = (dz_next @ W_next.T) * relu_derivative(z)
        
        Weight gradients:
            dW = a_prev.T @ dz / n
            db = sum(dz) / n
        """
        n = X.shape[0]  # Batch size
        
        # --- Output layer gradient ---
        # For sigmoid + binary cross-entropy, the gradient simplifies to:
        output = self.a_cache[-1]
        dz = output - y  # Shape: (n, output_size)
        
        # --- Backpropagate through layers ---
        for i in range(self.n_layers - 1, -1, -1):
            a_prev = self.a_cache[i]  # Activation from previous layer
            
            # Compute gradients
            dW = (a_prev.T @ dz) / n
            db = np.sum(dz, axis=0, keepdims=True) / n
            
            # Compute dz for previous layer (if not input layer)
            if i > 0:
                dz = (dz @ self.weights[i].T) * relu_derivative(self.z_cache[i-1])
            
            # Update weights
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db
        
        # Compute loss
        eps = 1e-7
        loss = -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 1000,
              verbose: bool = True) -> list[float]:
        """Train the network and return loss history."""
        loss_history = []
        
        for epoch in range(n_epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            loss = self.backward(X, y)
            loss_history.append(loss)
            
            if verbose and (epoch % (n_epochs // 10) == 0):
                predictions = (output > 0.5).astype(int)
                acc = np.mean(predictions == y)
                print(f"  Epoch {epoch:>5}: loss = {loss:.4f}, accuracy = {acc:.2%}")
        
        return loss_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (0 or 1)."""
        return (self.forward(X) > 0.5).astype(int)


# %% [markdown]
# ## 2. Solving XOR!

# %%
print("=== Neural Network: Solving XOR ===")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

# 2 inputs → 8 hidden neurons → 1 output
nn_xor = NeuralNetwork([2, 8, 1], learning_rate=0.5)
print()
loss_history = nn_xor.train(X_xor, y_xor, n_epochs=5000)

print("\nXOR Predictions:")
predictions = nn_xor.forward(X_xor)
for x, y, pred in zip(X_xor, y_xor, predictions):
    print(f"  {x.astype(int)} → {pred[0]:.4f} (expected: {int(y[0])}) "
          f"{'✓' if round(pred[0]) == int(y[0]) else '✗'}")

# %% [markdown]
# ## 3. Visualize XOR Decision Boundary

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Decision boundary
xx = np.linspace(-0.5, 1.5, 200)
yy = np.linspace(-0.5, 1.5, 200)
XX, YY = np.meshgrid(xx, yy)
grid = np.column_stack([XX.ravel(), YY.ravel()])
Z = nn_xor.forward(grid).reshape(XX.shape)

contour = ax1.contourf(XX, YY, Z, levels=50, cmap='RdYlGn', alpha=0.8)
plt.colorbar(contour, ax=ax1, label='Output probability')
ax1.scatter(X_xor[y_xor.flatten() == 0, 0], X_xor[y_xor.flatten() == 0, 1],
            c='red', s=150, edgecolors='black', linewidths=2, marker='o', label='Class 0', zorder=5)
ax1.scatter(X_xor[y_xor.flatten() == 1, 0], X_xor[y_xor.flatten() == 1, 1],
            c='green', s=150, edgecolors='black', linewidths=2, marker='s', label='Class 1', zorder=5)
ax1.set_title('XOR Decision Boundary (Neural Network)', fontweight='bold')
ax1.legend(fontsize=10)

# Loss curve
ax2.plot(loss_history, color='#E74C3C', linewidth=1.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_03_deep_learning/xor_solved.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 XOR solution plot saved")

# %% [markdown]
# ## 4. Multi-Class Classification

# %%
print("\n=== Multi-Class: Spirals Dataset ===")
np.random.seed(42)

# Generate spiral dataset (3 classes)
n_per_class = 100
n_classes = 3
X_spiral = np.zeros((n_per_class * n_classes, 2))
y_spiral = np.zeros(n_per_class * n_classes, dtype=int)

for c in range(n_classes):
    idx = range(n_per_class * c, n_per_class * (c + 1))
    r = np.linspace(0.0, 1, n_per_class)
    t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + np.random.randn(n_per_class) * 0.2
    X_spiral[idx] = np.column_stack([r * np.sin(t), r * np.cos(t)])
    y_spiral[idx] = c

# For multi-class: modify to use softmax output
# (Simplified: train one-vs-all with sigmoid)
# Binary: class 0 vs rest
y_binary = (y_spiral == 0).astype(float).reshape(-1, 1)

nn_spiral = NeuralNetwork([2, 32, 16, 1], learning_rate=0.5)
print()
loss_spiral = nn_spiral.train(X_spiral, y_binary, n_epochs=3000, verbose=True)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Data
colors = ['#E74C3C', '#3498DB', '#2ECC71']
for c in range(n_classes):
    mask = y_spiral == c
    ax1.scatter(X_spiral[mask, 0], X_spiral[mask, 1], c=colors[c], s=15,
                alpha=0.7, label=f'Class {c}')
ax1.set_title('Spiral Dataset (3 Classes)', fontweight='bold')
ax1.legend()
ax1.set_aspect('equal')

# Decision boundary for class 0
xx = np.linspace(-1.5, 1.5, 200)
yy = np.linspace(-1.5, 1.5, 200)
XX, YY = np.meshgrid(xx, yy)
grid = np.column_stack([XX.ravel(), YY.ravel()])
Z = nn_spiral.forward(grid).reshape(XX.shape)

ax2.contourf(XX, YY, Z, levels=50, cmap='RdYlBu_r', alpha=0.7)
for c in range(n_classes):
    mask = y_spiral == c
    ax2.scatter(X_spiral[mask, 0], X_spiral[mask, 1], c=colors[c], s=15,
                alpha=0.7, edgecolors='white', linewidths=0.3)
ax2.set_title('Neural Network Decision Boundary', fontweight='bold')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('module_03_deep_learning/spirals.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Spirals plot saved")

print("\n✅ Module 3, Script 02 complete!")
print("You just built a neural network from SCRATCH. Amazing!")
