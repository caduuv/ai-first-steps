"""
Module 1 — Script 05: Linear Regression from Scratch
=====================================================

This is the capstone of Module 1. We bring together everything we've
learned (linear algebra, loss functions, gradients) to build a complete
linear regression model from scratch — no sklearn, no PyTorch.

Topics:
  - Problem formulation
  - Closed-form (analytical) solution
  - Gradient descent solution
  - Comparison of both approaches
  - Feature scaling importance
  - Multi-feature regression

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Generate Synthetic Data

# %%
np.random.seed(42)

# Generate data: y = 3x + 7 + noise
# This simulates a real-world relationship with some randomness
n_samples = 100
X = np.random.uniform(-5, 5, n_samples)
noise = np.random.normal(0, 2, n_samples)
y = 3 * X + 7 + noise  # True: slope=3, intercept=7

print("=== Synthetic Data ===")
print(f"Generated {n_samples} samples")
print(f"True relationship: y = 3x + 7 + noise")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

# Visualize
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, alpha=0.6, color='#3498DB', s=30, label='Data points')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Linear Regression: Synthetic Data', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module_01_fundamentals/linear_regression_data.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Data plot saved")

# %% [markdown]
# ## 2. Linear Regression: The Model

# %%
# Model: y_pred = w * x + b
# Parameters to learn: w (weight/slope), b (bias/intercept)
# Objective: minimize MSE loss

def predict(X: np.ndarray, w: float, b: float) -> np.ndarray:
    """Linear model prediction: y = wx + b."""
    return w * X + b


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss."""
    return np.mean((y_true - y_pred) ** 2)

# %% [markdown]
# ## 3. Closed-Form Solution (Normal Equation)

# %%
# For linear regression, we can solve for the optimal w and b analytically!
# This is the "Normal Equation": θ = (X^T X)^(-1) X^T y
# Where X has a column of ones (for the bias term).

# Add bias column (column of 1s)
X_with_bias = np.column_stack([X, np.ones(n_samples)])

# Normal equation: θ = (X^T X)^(-1) X^T y
theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

w_closed = theta[0]
b_closed = theta[1]

y_pred_closed = predict(X, w_closed, b_closed)
loss_closed = mse_loss(y, y_pred_closed)

print("\n=== Closed-Form Solution (Normal Equation) ===")
print(f"w = {w_closed:.4f} (true: 3.0)")
print(f"b = {b_closed:.4f} (true: 7.0)")
print(f"MSE = {loss_closed:.4f}")

# %% [markdown]
# ## 4. Gradient Descent Solution

# %%
def compute_gradients(X: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    """
    Compute gradients of MSE loss w.r.t. w and b.
    
    MSE = (1/n) Σ (y_i - (w*x_i + b))²
    
    ∂MSE/∂w = -(2/n) Σ x_i * (y_i - (w*x_i + b))
    ∂MSE/∂b = -(2/n) Σ (y_i - (w*x_i + b))
    """
    n = len(X)
    y_pred = predict(X, w, b)
    error = y - y_pred
    
    dw = -(2 / n) * np.sum(X * error)
    db = -(2 / n) * np.sum(error)
    
    return dw, db


def linear_regression_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.01,
                          n_epochs: int = 1000) -> tuple[float, float, list[float]]:
    """
    Train linear regression using gradient descent.
    
    Returns: (w, b, loss_history)
    """
    # Initialize parameters randomly
    w = np.random.randn() * 0.01
    b = 0.0
    loss_history = []
    
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        loss_history.append(loss)
        
        # Compute gradients
        dw, db = compute_gradients(X, y, w, b)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
        
        # Log progress
        if epoch % 200 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:>4}: loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
    
    return w, b, loss_history


print("\n=== Gradient Descent Solution ===")
w_gd, b_gd, loss_history = linear_regression_gd(X, y, lr=0.01, n_epochs=1000)

print(f"\nFinal: w = {w_gd:.4f}, b = {b_gd:.4f}")
print(f"Final MSE = {loss_history[-1]:.4f}")

# %% [markdown]
# ## 5. Compare Both Solutions

# %%
print("\n=== Comparison ===")
print(f"{'Method':<20} {'w':>8} {'b':>8} {'MSE':>10}")
print(f"{'─'*20} {'─'*8} {'─'*8} {'─'*10}")
print(f"{'True values':<20} {'3.0000':>8} {'7.0000':>8} {'N/A':>10}")
print(f"{'Normal Equation':<20} {w_closed:>8.4f} {b_closed:>8.4f} {loss_closed:>10.4f}")
print(f"{'Gradient Descent':<20} {w_gd:>8.4f} {b_gd:>8.4f} {loss_history[-1]:>10.4f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Plot 1: Data with fitted lines
x_line = np.linspace(X.min(), X.max(), 100)
axes[0].scatter(X, y, alpha=0.4, color='#3498DB', s=20, label='Data')
axes[0].plot(x_line, predict(x_line, w_closed, b_closed), color='#E74C3C',
             linewidth=2, label=f'Normal Eq. (w={w_closed:.2f}, b={b_closed:.2f})')
axes[0].plot(x_line, predict(x_line, w_gd, b_gd), color='#2ECC71', linewidth=2,
             linestyle='--', label=f'GD (w={w_gd:.2f}, b={b_gd:.2f})')
axes[0].set_xlabel('X', fontsize=11)
axes[0].set_ylabel('y', fontsize=11)
axes[0].set_title('Fitted Lines', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Plot 2: Loss over epochs
axes[1].plot(loss_history, color='#E74C3C', linewidth=1.5)
axes[1].axhline(y=loss_closed, color='#3498DB', linestyle='--', alpha=0.7, label='Optimal (Normal Eq.)')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('MSE Loss', fontsize=11)
axes[1].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Plot 3: Residuals
residuals = y - predict(X, w_gd, b_gd)
axes[2].scatter(predict(X, w_gd, b_gd), residuals, alpha=0.5, color='#8E44AD', s=20)
axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[2].set_xlabel('Predicted y', fontsize=11)
axes[2].set_ylabel('Residual (y - ŷ)', fontsize=11)
axes[2].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_01_fundamentals/linear_regression_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Results plot saved")

# %% [markdown]
# ## 6. Multi-Feature Linear Regression

# %%
# Real datasets have multiple features. Let's extend our model.
# y = w1*x1 + w2*x2 + ... + wn*xn + b

print("\n=== Multi-Feature Linear Regression ===")
np.random.seed(42)

# Generate: y = 2*x1 + (-3)*x2 + 5 + noise
n = 200
X_multi = np.random.randn(n, 2)
w_true = np.array([2.0, -3.0])
b_true = 5.0
y_multi = X_multi @ w_true + b_true + np.random.normal(0, 1, n)

# Normal equation
X_multi_bias = np.column_stack([X_multi, np.ones(n)])
theta_multi = np.linalg.inv(X_multi_bias.T @ X_multi_bias) @ X_multi_bias.T @ y_multi

print(f"True weights: {w_true}, bias: {b_true}")
print(f"Found weights: {theta_multi[:2].round(4)}, bias: {theta_multi[2]:.4f}")
print(f"Error in weights: {np.abs(w_true - theta_multi[:2]).round(4)}")

# Gradient descent solution for multi-feature
def multi_linear_regression_gd(X, y, lr=0.01, n_epochs=1000):
    """Multi-feature gradient descent."""
    n_features = X.shape[1]
    w = np.random.randn(n_features) * 0.01
    b = 0.0
    losses = []
    
    for epoch in range(n_epochs):
        y_pred = X @ w + b
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)
        
        # Gradients
        error = y_pred - y
        dw = (2 / len(X)) * (X.T @ error)
        db = (2 / len(X)) * np.sum(error)
        
        w -= lr * dw
        b -= lr * db
    
    return w, b, losses

w_multi_gd, b_multi_gd, losses_multi = multi_linear_regression_gd(X_multi, y_multi, lr=0.01, n_epochs=2000)
print(f"\nGD weights: {w_multi_gd.round(4)}, bias: {b_multi_gd:.4f}")
print(f"GD final loss: {losses_multi[-1]:.4f}")

# %% [markdown]
# ## 7. Feature Scaling Importance

# %%
# When features have very different scales, gradient descent struggles.
# Feature scaling (normalization/standardization) fixes this.

print("\n=== Feature Scaling Demo ===")

# Create data with vastly different scales
X_scaled_test = np.column_stack([
    np.random.normal(1000, 100, 200),   # Feature 1: large scale
    np.random.normal(0.01, 0.001, 200)  # Feature 2: tiny scale
])
w_true_scaled = np.array([0.5, 1000])
y_scaled = X_scaled_test @ w_true_scaled + 3

# Without scaling: gradient descent struggles
w_no_scale, b_no_scale, loss_no_scale = multi_linear_regression_gd(
    X_scaled_test, y_scaled, lr=1e-8, n_epochs=1000)
print(f"Without scaling - Final loss: {loss_no_scale[-1]:.4f}")

# With scaling (z-score normalization)
X_mean = X_scaled_test.mean(axis=0)
X_std = X_scaled_test.std(axis=0)
X_normalized = (X_scaled_test - X_mean) / X_std

w_scaled, b_scaled, loss_scaled = multi_linear_regression_gd(
    X_normalized, y_scaled, lr=0.01, n_epochs=1000)
print(f"With scaling - Final loss: {loss_scaled[-1]:.4f}")

print("\n✅ Module 1, Script 05 complete!")
print("\n🎉 Module 1 complete! You now understand the math behind ML.")
print("Move on to Module 2: Python for Machine Learning →")
