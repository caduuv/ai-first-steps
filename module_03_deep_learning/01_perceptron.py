"""
Module 3 — Script 01: The Perceptron
======================================

The perceptron is the simplest neural unit and the building block
of all neural networks. We implement it from scratch and train it
on a simple classification task.

Topics:
  - Perceptron model
  - Step activation function
  - Perceptron learning rule
  - Linear separability
  - Limitations of a single perceptron

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. The Perceptron Model

# %%
class Perceptron:
    """
    A single perceptron (the simplest neural unit).
    
    Model: output = step(w · x + bias)
    
    The perceptron learning rule:
      - If prediction is correct: do nothing
      - If prediction is wrong: nudge weights toward the correct answer
      
    w_new = w_old + lr * (y_true - y_pred) * x
    
    Limitations:
      - Can only learn LINEARLY SEPARABLE patterns
      - Cannot solve XOR (we'll fix this with multi-layer networks!)
    """
    
    def __init__(self, n_features: int, learning_rate: float = 0.1):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.lr = learning_rate
    
    def step(self, z: float) -> int:
        """Step activation: 1 if z >= 0, else 0."""
        return 1 if z >= 0 else 0
    
    def predict(self, x: np.ndarray) -> int:
        """Forward pass: compute weighted sum and apply step function."""
        z = np.dot(self.weights, x) + self.bias
        return self.step(z)
    
    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 100) -> list[float]:
        """
        Train the perceptron using the perceptron learning rule.
        Returns accuracy history.
        """
        accuracy_history = []
        
        for epoch in range(n_epochs):
            correct = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                
                # Update rule
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                
                if prediction == yi:
                    correct += 1
            
            acc = correct / len(X)
            accuracy_history.append(acc)
            
            if epoch % 20 == 0 or acc == 1.0:
                print(f"  Epoch {epoch:>3}: accuracy = {acc:.2%}")
            
            if acc == 1.0:
                print(f"  ✓ Converged at epoch {epoch}!")
                break
        
        return accuracy_history


# %% [markdown]
# ## 2. AND Gate — Linearly Separable

# %%
print("=== Training Perceptron: AND Gate ===")
# AND truth table
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

p_and = Perceptron(n_features=2, learning_rate=0.1)
history_and = p_and.train(X_and, y_and, n_epochs=100)

print(f"\nFinal weights: {p_and.weights.round(4)}, bias: {p_and.bias:.4f}")
print("Predictions:")
for x, y in zip(X_and, y_and):
    pred = p_and.predict(x)
    print(f"  {x} → {pred} (expected: {y}) {'✓' if pred == y else '✗'}")

# %% [markdown]
# ## 3. OR Gate — Linearly Separable

# %%
print("\n=== Training Perceptron: OR Gate ===")
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

p_or = Perceptron(n_features=2, learning_rate=0.1)
history_or = p_or.train(X_or, y_or, n_epochs=100)

print("Predictions:")
for x, y in zip(X_or, y_or):
    pred = p_or.predict(x)
    print(f"  {x} → {pred} (expected: {y}) {'✓' if pred == y else '✗'}")

# %% [markdown]
# ## 4. XOR Gate — NOT Linearly Separable!

# %%
print("\n=== Training Perceptron: XOR Gate ===")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

p_xor = Perceptron(n_features=2, learning_rate=0.1)
history_xor = p_xor.train(X_xor, y_xor, n_epochs=100)

print("Predictions:")
for x, y in zip(X_xor, y_xor):
    pred = p_xor.predict(x)
    print(f"  {x} → {pred} (expected: {y}) {'✓' if pred == y else '✗'}")
print("⚠️ Perceptron CANNOT solve XOR — we need multiple layers!")

# %% [markdown]
# ## 5. Visualize Decision Boundaries

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

datasets = [(X_and, y_and, p_and, 'AND Gate ✓'),
            (X_or, y_or, p_or, 'OR Gate ✓'),
            (X_xor, y_xor, p_xor, 'XOR Gate ✗')]

for ax, (X, y, model, title) in zip(axes, datasets):
    # Plot decision boundary
    xx = np.linspace(-0.5, 1.5, 200)
    yy = np.linspace(-0.5, 1.5, 200)
    XX, YY = np.meshgrid(xx, yy)
    Z = np.array([[model.predict(np.array([x, y])) for x in xx] for y in yy])
    
    ax.contourf(XX, YY, Z, levels=[-0.5, 0.5, 1.5], colors=['#FADBD8', '#D5F5E3'], alpha=0.7)
    ax.contour(XX, YY, Z, levels=[0.5], colors='gray', linewidths=2)
    
    # Plot data points
    for cls, color, marker in [(0, '#E74C3C', 'o'), (1, '#2ECC71', 's')]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                   s=100, edgecolors='black', linewidths=1.5, label=f'Class {cls}', zorder=5)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig('module_03_deep_learning/perceptron_boundaries.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Decision boundaries saved")

# %% [markdown]
# ## 6. A Larger Dataset

# %%
print("\n=== Perceptron on Larger Dataset ===")
np.random.seed(42)

# Generate two linearly separable clusters
n = 100
class0 = np.random.randn(n, 2) + np.array([-1, -1])
class1 = np.random.randn(n, 2) + np.array([2, 2])
X_data = np.vstack([class0, class1])
y_data = np.array([0] * n + [1] * n)

# Shuffle
indices = np.random.permutation(2 * n)
X_data = X_data[indices]
y_data = y_data[indices]

p_large = Perceptron(n_features=2, learning_rate=0.01)
history_large = p_large.train(X_data, y_data, n_epochs=100)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Decision boundary
xx = np.linspace(-4, 5, 200)
yy = np.linspace(-4, 5, 200)
XX, YY = np.meshgrid(xx, yy)
Z = np.array([[p_large.predict(np.array([x, y])) for x in xx] for y in yy])

ax1.contourf(XX, YY, Z, levels=[-0.5, 0.5, 1.5], colors=['#FADBD8', '#D5F5E3'], alpha=0.5)
ax1.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1], c='#E74C3C', alpha=0.6, s=20, label='Class 0')
ax1.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1], c='#2ECC71', alpha=0.6, s=20, label='Class 1')
ax1.set_title('Perceptron on 200 Points', fontweight='bold')
ax1.legend()

# Accuracy curve
ax2.plot(history_large, color='#3498DB', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_03_deep_learning/perceptron_large.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Large dataset plot saved")

print("\n✅ Module 3, Script 01 complete!")
print("Key takeaway: A single perceptron can only solve linearly separable problems.")
print("Next: We'll stack perceptrons into a neural network to solve XOR and more! →")
