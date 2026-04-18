"""
Module 1 — Script 04: Gradients & Optimization
================================================

This script demystifies gradients and gradient descent — the algorithm
that makes neural networks learn. We start with numerical gradients
to build intuition, then move to analytical gradients.

Topics:
  - What is a gradient?
  - Numerical gradient computation
  - Analytical gradients
  - Gradient descent algorithm
  - Learning rate effects
  - Gradient descent on 2D functions

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. What is a Gradient?

# %%
# The gradient of a function f at point x tells you:
#   - DIRECTION: which way f increases fastest
#   - MAGNITUDE: how steep the increase is
#
# For a function of one variable, the gradient is the derivative: df/dx
# For a function of multiple variables, the gradient is a vector of partial derivatives.
#
# To MINIMIZE a function (= reduce loss), we move OPPOSITE to the gradient.

# Simple example: f(x) = x²
# Derivative: f'(x) = 2x
# At x=3: f'(3) = 6 → function is increasing, take a step to the LEFT

def f(x):
    """Simple quadratic function. Minimum at x=0."""
    return x ** 2

def f_derivative(x):
    """Analytical derivative of f(x) = x²."""
    return 2 * x

print("=== Gradients of f(x) = x² ===")
for x in [-3, -1, 0, 1, 3]:
    print(f"  x={x:+.0f}: f(x)={f(x):.0f}, f'(x)={f_derivative(x):+.0f}")

# %% [markdown]
# ## 2. Numerical Gradient Computation

# %%
def numerical_gradient(func, x: float, h: float = 1e-5) -> float:
    """
    Compute the gradient numerically using the central difference method.
    
    f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
    
    This works for ANY function, even if we can't derive the analytical gradient.
    It's slower but useful for:
      - Checking analytical gradients (gradient checking)
      - Quick prototyping
    """
    return (func(x + h) - func(x - h)) / (2 * h)


print("\n=== Numerical vs Analytical Gradients ===")
for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    numerical = numerical_gradient(f, x)
    analytical = f_derivative(x)
    print(f"  x={x:+.1f}: numerical={numerical:+.6f}, analytical={analytical:+.6f}, "
          f"diff={abs(numerical-analytical):.2e}")

# %% [markdown]
# ## 3. Gradient Descent — The Learning Algorithm

# %%
def gradient_descent_1d(func, grad_func, x_init: float, lr: float = 0.1,
                         n_steps: int = 20) -> list[tuple[float, float]]:
    """
    Gradient descent in 1D.
    
    Algorithm:
      1. Start at some initial x
      2. Compute gradient at x
      3. Update: x = x - lr * gradient
      4. Repeat
    
    The learning rate (lr) controls step size:
      - Too large: overshoots the minimum, may diverge
      - Too small: converges very slowly
      - Just right: converges efficiently
    
    Returns history of (x, f(x)) pairs.
    """
    x = x_init
    history = [(x, func(x))]
    
    for _ in range(n_steps):
        grad = grad_func(x)
        x = x - lr * grad  # The KEY update rule!
        history.append((x, func(x)))
    
    return history


# Run gradient descent on f(x) = x²
history = gradient_descent_1d(f, f_derivative, x_init=4.0, lr=0.1, n_steps=20)

print("\n=== Gradient Descent on f(x) = x² ===")
print(f"Starting at x = 4.0, learning rate = 0.1")
print(f"{'Step':>4} {'x':>10} {'f(x)':>10} {'gradient':>10}")
print(f"{'─'*4} {'─'*10} {'─'*10} {'─'*10}")
for i, (x, fx) in enumerate(history[:10]):
    grad = f_derivative(x)
    print(f"{i:>4} {x:>10.6f} {fx:>10.6f} {grad:>10.6f}")
print(f"  ... (final 10 steps omitted)")
print(f"Final: x = {history[-1][0]:.8f}, f(x) = {history[-1][1]:.8f}")

# %% [markdown]
# ## 4. Learning Rate Effects

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
learning_rates = [0.01, 0.1, 0.8]
titles = ['Too Small (lr=0.01)', 'Just Right (lr=0.1)', 'Too Large (lr=0.8)']
colors = ['#3498DB', '#2ECC71', '#E74C3C']

x_range = np.linspace(-5, 5, 200)

for idx, (lr, title, color) in enumerate(zip(learning_rates, titles, colors)):
    ax = axes[idx]
    history = gradient_descent_1d(f, f_derivative, x_init=4.0, lr=lr, n_steps=20)
    
    # Plot function
    ax.plot(x_range, f(x_range), color='gray', linewidth=1.5, alpha=0.5)
    
    # Plot gradient descent path
    xs = [h[0] for h in history]
    ys = [h[1] for h in history]
    ax.plot(xs, ys, 'o-', color=color, markersize=4, linewidth=1.5, alpha=0.8)
    ax.plot(xs[0], ys[0], 's', color=color, markersize=8, label='Start')
    ax.plot(xs[-1], ys[-1], '*', color=color, markersize=12, label='End')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('f(x)', fontsize=11)
    ax.set_ylim(-1, 20)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of Learning Rate on Gradient Descent', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('module_01_fundamentals/learning_rate_effects.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Learning rate effects plot saved")

# %% [markdown]
# ## 5. Multi-Variable Gradient Descent

# %%
def f_2d(x, y):
    """
    A 2D function with minimum at approximately (1, 1).
    f(x, y) = (x - 1)² + 2(y - 1)²
    """
    return (x - 1)**2 + 2 * (y - 1)**2

def grad_f_2d(x, y):
    """Gradient of f_2d: [∂f/∂x, ∂f/∂y]."""
    df_dx = 2 * (x - 1)
    df_dy = 4 * (y - 1)
    return np.array([df_dx, df_dy])


def numerical_gradient_2d(func, point: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Compute gradient numerically for a multi-variable function."""
    grad = np.zeros_like(point)
    for i in range(len(point)):
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += h
        point_minus[i] -= h
        grad[i] = (func(*point_plus) - func(*point_minus)) / (2 * h)
    return grad


# Gradient descent in 2D
x, y = 4.0, 4.0
lr = 0.1
history_2d = [(x, y, f_2d(x, y))]

for _ in range(50):
    grad = grad_f_2d(x, y)
    x -= lr * grad[0]
    y -= lr * grad[1]
    history_2d.append((x, y, f_2d(x, y)))

print("\n=== 2D Gradient Descent ===")
print(f"Start: ({history_2d[0][0]:.2f}, {history_2d[0][1]:.2f}), loss = {history_2d[0][2]:.4f}")
print(f"End:   ({history_2d[-1][0]:.4f}, {history_2d[-1][1]:.4f}), loss = {history_2d[-1][2]:.8f}")

# Visualize 2D gradient descent
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Contour plot with path
xx = np.linspace(-1, 5, 100)
yy = np.linspace(-1, 5, 100)
X, Y = np.meshgrid(xx, yy)
Z = f_2d(X, Y)

contour = ax1.contourf(X, Y, Z, levels=30, cmap='RdYlBu_r', alpha=0.8)
plt.colorbar(contour, ax=ax1, label='f(x, y)')

path_x = [h[0] for h in history_2d]
path_y = [h[1] for h in history_2d]
ax1.plot(path_x, path_y, 'o-', color='white', markersize=3, linewidth=1.5)
ax1.plot(path_x[0], path_y[0], 's', color='lime', markersize=10, label='Start')
ax1.plot(1, 1, '*', color='yellow', markersize=15, label='Minimum (1,1)')

ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('y', fontsize=11)
ax1.set_title('2D Gradient Descent Path', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)

# Loss over iterations
losses = [h[2] for h in history_2d]
ax2.plot(losses, color='#E74C3C', linewidth=2)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Loss Over Time', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('module_01_fundamentals/gradient_descent_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 2D gradient descent plot saved")

# %% [markdown]
# ## 6. Gradient Checking

# %%
# Always verify your analytical gradients with numerical gradients!
# This is called "gradient checking" and catches bugs in backpropagation.

print("\n=== Gradient Checking ===")
test_points = [np.array([2.0, 3.0]), np.array([-1.0, 0.5]), np.array([0.0, 0.0])]

for point in test_points:
    analytical = grad_f_2d(*point)
    numerical = numerical_gradient_2d(f_2d, point)
    
    # Relative error: a standard way to compare gradients
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    
    print(f"Point {point}:")
    print(f"  Analytical: {analytical}")
    print(f"  Numerical:  {numerical}")
    print(f"  Rel. error: {rel_error}")
    print(f"  Pass: {np.all(rel_error < 1e-5)}")

print("\n✅ Module 1, Script 04 complete!")
