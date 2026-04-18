"""
Module 3 — Script 03: Backpropagation Deep Dive
=================================================

A detailed, step-by-step walkthrough of backpropagation.
We derive and implement every gradient by hand, then verify
with numerical gradient checking.

Topics:
  - Chain rule applied to neural networks
  - Gradient flow through layers
  - Computational graph perspective
  - Gradient checking for correctness
  - Vanishing/exploding gradients

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. The Chain Rule — The Heart of Backpropagation

# %%
# Consider a simple computation: L = (sigmoid(w * x + b) - y)²
#
# To update w, we need dL/dw. The chain rule decomposes this:
#
# Let z = w * x + b          (linear transformation)
# Let a = sigmoid(z)          (activation)
# Let L = (a - y)²            (loss)
#
# Then:
#   dL/dw = dL/da * da/dz * dz/dw
#         = 2(a-y) * sigmoid'(z) * x

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

print("=== Chain Rule Example ===")
# Let's compute the gradient step by step
x = 2.0   # Input
y = 1.0   # Target
w = 0.5   # Weight
b = -0.1  # Bias

# Forward pass (compute each step)
z = w * x + b           # z = 0.9
a = sigmoid(z)           # a ≈ 0.7109
L = (a - y) ** 2         # L ≈ 0.0835

print(f"Forward pass:")
print(f"  z = w·x + b = {w}·{x} + {b} = {z:.4f}")
print(f"  a = σ(z) = σ({z:.4f}) = {a:.4f}")
print(f"  L = (a - y)² = ({a:.4f} - {y})² = {L:.4f}")

# Backward pass (chain rule, from output to input)
dL_da = 2 * (a - y)           # ∂L/∂a
da_dz = sigmoid_deriv(z)       # ∂a/∂z = σ'(z)
dz_dw = x                     # ∂z/∂w = x
dz_db = 1                     # ∂z/∂b = 1

dL_dw = dL_da * da_dz * dz_dw  # Chain rule!
dL_db = dL_da * da_dz * dz_db

print(f"\nBackward pass (chain rule):")
print(f"  dL/da = 2(a - y) = {dL_da:.4f}")
print(f"  da/dz = σ'(z) = {da_dz:.4f}")
print(f"  dz/dw = x = {dz_dw:.4f}")
print(f"  dL/dw = {dL_da:.4f} × {da_dz:.4f} × {dz_dw:.4f} = {dL_dw:.4f}")
print(f"  dL/db = {dL_da:.4f} × {da_dz:.4f} × {dz_db:.4f} = {dL_db:.4f}")

# Verify with numerical gradient
h = 1e-5
num_grad_w = ((sigmoid(((w+h) * x + b)) - y)**2 - (sigmoid(((w-h) * x + b)) - y)**2) / (2*h)
print(f"\nNumerical gradient check:")
print(f"  Analytical dL/dw = {dL_dw:.6f}")
print(f"  Numerical  dL/dw = {num_grad_w:.6f}")
print(f"  Difference: {abs(dL_dw - num_grad_w):.2e}")

# %% [markdown]
# ## 2. Multi-Layer Backpropagation

# %%
class TwoLayerNetwork:
    """
    A 2-layer network with explicit, step-by-step backpropagation.
    
    Architecture: Input(2) → Hidden(4, ReLU) → Output(1, Sigmoid)
    
    Every gradient computation is documented.
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        # Layer 1: 2 → 4
        self.W1 = np.random.randn(2, 4) * 0.5
        self.b1 = np.zeros((1, 4))
        # Layer 2: 4 → 1
        self.W2 = np.random.randn(4, 1) * 0.5
        self.b2 = np.zeros((1, 1))
    
    def forward(self, X):
        """Forward pass with caching for backprop."""
        # Layer 1
        self.z1 = X @ self.W1 + self.b1         # (n, 4)
        self.a1 = np.maximum(0, self.z1)          # ReLU → (n, 4)
        
        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2    # (n, 1)
        self.a2 = sigmoid(self.z2)                # Sigmoid → (n, 1)
        
        return self.a2
    
    def backward(self, X, y, lr=0.1):
        """
        Backward pass with detailed documentation.
        """
        n = X.shape[0]
        
        # ====== OUTPUT LAYER ======
        # Loss = BCE = -(1/n) Σ [y·log(a2) + (1-y)·log(1-a2)]
        # For sigmoid + BCE, the combined gradient simplifies to:
        # dL/dz2 = a2 - y
        dz2 = self.a2 - y                        # (n, 1)
        
        # Gradients for W2 and b2:
        # dL/dW2 = a1.T @ dz2 / n
        # dL/db2 = sum(dz2) / n
        dW2 = (self.a1.T @ dz2) / n              # (4, 1)
        db2 = np.sum(dz2, axis=0, keepdims=True) / n  # (1, 1)
        
        # ====== HIDDEN LAYER ======
        # Propagate gradient backward through W2:
        # dL/da1 = dz2 @ W2.T
        da1 = dz2 @ self.W2.T                    # (n, 4)
        
        # Through ReLU activation:
        # dL/dz1 = dL/da1 * relu'(z1)
        # relu'(z) = 1 if z > 0, else 0
        dz1 = da1 * (self.z1 > 0).astype(float)  # (n, 4)
        
        # Gradients for W1 and b1:
        dW1 = (X.T @ dz1) / n                    # (2, 4)
        db1 = np.sum(dz1, axis=0, keepdims=True) / n  # (1, 4)
        
        # ====== UPDATE WEIGHTS ======
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        
        # Compute and return loss
        eps = 1e-7
        loss = -np.mean(y * np.log(self.a2 + eps) + (1 - y) * np.log(1 - self.a2 + eps))
        return loss, {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


# %% [markdown]
# ## 3. Gradient Checking — Verify Correctness

# %%
print("\n=== Gradient Checking ===")

def numerical_gradient_check(network, X, y, param_name, epsilon=1e-5):
    """Compute numerical gradients and compare with analytical."""
    param = getattr(network, param_name)
    num_grad = np.zeros_like(param)
    
    for i in range(param.shape[0]):
        for j in range(param.shape[1]):
            # f(θ + ε)
            param[i, j] += epsilon
            network.forward(X)
            eps_val = 1e-7
            loss_plus = -np.mean(y * np.log(network.a2 + eps_val) +
                                  (1 - y) * np.log(1 - network.a2 + eps_val))
            
            # f(θ - ε)
            param[i, j] -= 2 * epsilon
            network.forward(X)
            loss_minus = -np.mean(y * np.log(network.a2 + eps_val) +
                                   (1 - y) * np.log(1 - network.a2 + eps_val))
            
            # Restore
            param[i, j] += epsilon
            
            num_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return num_grad


# Test on XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

net = TwoLayerNetwork(seed=42)
net.forward(X)
loss, analytical_grads = net.backward(X, y, lr=0.0)  # lr=0 so weights don't change

for param_name in ['W1', 'W2']:
    # Reset forward to current state
    net_check = TwoLayerNetwork(seed=42)
    num_grad = numerical_gradient_check(net_check, X, y, param_name)
    analytical = analytical_grads[f'd{param_name}']
    
    rel_error = np.abs(analytical - num_grad) / (np.abs(analytical) + np.abs(num_grad) + 1e-8)
    max_error = rel_error.max()
    
    print(f"\n{param_name}:")
    print(f"  Max relative error: {max_error:.2e}")
    print(f"  Pass: {max_error < 1e-5} {'✓' if max_error < 1e-5 else '✗'}")

# %% [markdown]
# ## 4. Vanishing and Exploding Gradients

# %%
print("\n=== Vanishing/Exploding Gradients ===")

# Demonstrate gradient flow through deep networks
# With sigmoid activation, gradients multiply by sigmoid_deriv at each layer
# sigmoid_deriv(z) ∈ (0, 0.25], so gradients shrink exponentially!

depths = [1, 5, 10, 20, 50]
gradient_magnitudes_sigmoid = []
gradient_magnitudes_relu = []

for depth in depths:
    # Sigmoid: gradient shrinks by at most 0.25× per layer
    # After d layers: max gradient ≈ 0.25^d
    grad_sigmoid = 0.25 ** depth
    gradient_magnitudes_sigmoid.append(grad_sigmoid)
    
    # ReLU: gradient is 1 (if active) or 0
    # After d layers: gradient = 1 (if all neurons active)
    grad_relu = 1.0 ** depth  # Simplified
    gradient_magnitudes_relu.append(grad_relu)

print(f"{'Depth':>6} {'Sigmoid Grad':>15} {'ReLU Grad':>15}")
print(f"{'─'*6} {'─'*15} {'─'*15}")
for d, gs, gr in zip(depths, gradient_magnitudes_sigmoid, gradient_magnitudes_relu):
    print(f"{d:>6} {gs:>15.2e} {gr:>15.2e}")

print("\n💡 This is why ReLU replaced Sigmoid in deep networks!")
print("   Sigmoid gradients vanish exponentially with depth.")
print("   ReLU preserves gradient magnitude (when active).")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Gradient flow comparison
ax1.semilogy(depths, gradient_magnitudes_sigmoid, 'o-', color='#E74C3C',
             linewidth=2, markersize=8, label='Sigmoid')
ax1.semilogy(depths, gradient_magnitudes_relu, 's-', color='#2ECC71',
             linewidth=2, markersize=8, label='ReLU')
ax1.set_xlabel('Network Depth (layers)', fontsize=11)
ax1.set_ylabel('Maximum Gradient Magnitude', fontsize=11)
ax1.set_title('Gradient Flow vs Depth', fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Activation function derivatives
z = np.linspace(-5, 5, 200)
ax2.plot(z, sigmoid_deriv(z), color='#E74C3C', linewidth=2, label='σ\'(z) (max=0.25)')
relu_d = (z > 0).astype(float)
ax2.plot(z, relu_d, color='#2ECC71', linewidth=2, label='ReLU\'(z) (0 or 1)')
ax2.set_xlabel('z', fontsize=11)
ax2.set_ylabel('Derivative', fontsize=11)
ax2.set_title('Activation Derivatives', fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.1, 1.3)

plt.tight_layout()
plt.savefig('module_03_deep_learning/gradient_flow.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Gradient flow plot saved")

# %% [markdown]
# ## 5. Training the Network on XOR

# %%
print("\n=== Training on XOR ===")
net = TwoLayerNetwork(seed=42)
losses = []

for epoch in range(5000):
    net.forward(X)
    loss, _ = net.backward(X, y, lr=0.5)
    losses.append(loss)
    
    if epoch % 1000 == 0:
        predictions = (net.a2 > 0.5).astype(int)
        acc = np.mean(predictions == y)
        print(f"  Epoch {epoch:>5}: loss = {loss:.4f}, accuracy = {acc:.0%}")

print("\nFinal predictions:")
net.forward(X)
for xi, yi, pred in zip(X, y, net.a2):
    print(f"  {xi.astype(int)} → {pred[0]:.4f} (expected: {int(yi[0])})")

print("\n✅ Module 3, Script 03 complete!")
