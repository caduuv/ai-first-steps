"""
Module 1 — Script 03: Loss Functions
======================================

Loss functions measure how wrong a model's predictions are.
This script implements the most common loss functions from scratch,
then shows their PyTorch equivalents.

Topics:
  - Mean Squared Error (MSE) for regression
  - Mean Absolute Error (MAE) for regression
  - Binary Cross-Entropy for binary classification
  - Categorical Cross-Entropy for multi-class classification
  - Visualizing loss landscapes

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Mean Squared Error (MSE)

# %%
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error: average of squared differences.
    
    MSE = (1/n) Σ (y_true_i - y_pred_i)²
    
    Properties:
      - Always non-negative (squared terms)
      - Penalizes large errors more than small ones (quadratic)
      - Differentiable everywhere (smooth gradient for optimization)
      - Used for regression tasks
    
    Why squared and not just absolute?
      - Squaring makes the function differentiable at 0
      - It penalizes outliers more, which can be good or bad
    """
    return np.mean((y_true - y_pred) ** 2)


# Example: predicting house prices (in $1000s)
y_true = np.array([200, 300, 400, 250, 350])
y_pred = np.array([210, 290, 420, 240, 330])

loss = mse_loss(y_true, y_pred)
print("=== Mean Squared Error ===")
print(f"True values:  {y_true}")
print(f"Predictions:  {y_pred}")
print(f"Errors:       {y_true - y_pred}")
print(f"Squared:      {(y_true - y_pred)**2}")
print(f"MSE = {loss:.2f}")

# %% [markdown]
# ## 2. Mean Absolute Error (MAE)

# %%
def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error: average of absolute differences.
    
    MAE = (1/n) Σ |y_true_i - y_pred_i|
    
    Properties:
      - Less sensitive to outliers than MSE
      - Not differentiable at 0 (but subgradient exists)
      - More interpretable: "on average, predictions are off by X"
    """
    return np.mean(np.abs(y_true - y_pred))


mae = mae_loss(y_true, y_pred)
print(f"\n=== Mean Absolute Error ===")
print(f"MAE = {mae:.2f}")
print(f"Interpretation: predictions are off by ${mae*1000:.0f} on average")

# %% [markdown]
# ## 3. Comparing MSE and MAE

# %%
# Let's see how they react to an outlier
y_true_outlier = np.array([200, 300, 400, 250, 350])
y_pred_outlier = np.array([210, 290, 420, 240, 500])  # Last prediction is way off!

print("\n=== MSE vs MAE with Outlier ===")
print(f"True:  {y_true_outlier}")
print(f"Pred:  {y_pred_outlier}")
print(f"MSE = {mse_loss(y_true_outlier, y_pred_outlier):.2f}  ← Explodes due to outlier")
print(f"MAE = {mae_loss(y_true_outlier, y_pred_outlier):.2f}  ← More robust")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

errors = np.linspace(-5, 5, 200)

ax1.plot(errors, errors**2, color='#E74C3C', linewidth=2, label='MSE (squared)')
ax1.plot(errors, np.abs(errors), color='#3498DB', linewidth=2, label='MAE (absolute)')
ax1.set_xlabel('Error (y_true - y_pred)', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('MSE vs MAE: Loss per Sample', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Gradient comparison
ax2.plot(errors, 2 * errors, color='#E74C3C', linewidth=2, label='MSE gradient (2·error)')
ax2.plot(errors, np.sign(errors), color='#3498DB', linewidth=2, label='MAE gradient (sign)')
ax2.set_xlabel('Error', fontsize=11)
ax2.set_ylabel('Gradient', fontsize=11)
ax2.set_title('Gradient: How Strongly to Correct', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_01_fundamentals/mse_vs_mae.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 MSE vs MAE plot saved")

# %% [markdown]
# ## 4. Binary Cross-Entropy (BCE)

# %%
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: maps any real number to (0, 1).
    
    σ(z) = 1 / (1 + e^(-z))
    
    Used as the output activation for binary classification:
    the output represents a probability.
    """
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """
    Binary Cross-Entropy Loss.
    
    BCE = -(1/n) Σ [y·log(p) + (1-y)·log(1-p)]
    
    Where y is the true label (0 or 1) and p is the predicted probability.
    
    Properties:
      - Undefined for p=0 or p=1 (hence the eps clipping)
      - Heavily penalizes confident wrong predictions
      - The standard loss for binary classification
    
    Why not MSE for classification?
      - MSE creates a non-convex loss landscape for sigmoid outputs
      - Cross-entropy gives stronger gradients when predictions are very wrong
    """
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Example: binary classification (disease: yes/no)
y_true_binary = np.array([1, 0, 1, 1, 0])  # Actual labels
y_pred_probs = np.array([0.9, 0.1, 0.8, 0.7, 0.3])  # Predicted probabilities

loss_bce = binary_cross_entropy(y_true_binary, y_pred_probs)
print("\n=== Binary Cross-Entropy ===")
print(f"True labels:    {y_true_binary}")
print(f"Predicted probs: {y_pred_probs}")
print(f"BCE Loss = {loss_bce:.4f}")

# Show how loss increases with wrong predictions
print("\nBCE for different predictions when y_true=1:")
for p in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]:
    loss = -np.log(p)
    print(f"  P(y=1) = {p:.2f} → loss = {loss:.4f}")

# %% [markdown]
# ## 5. Categorical Cross-Entropy

# %%
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax: converts raw scores to probabilities for multi-class.
    
    softmax(z_i) = e^(z_i) / Σ e^(z_j)
    
    Properties:
      - All outputs sum to 1
      - All outputs are positive
      - Preserves relative ordering of inputs
    """
    # Subtract max for numerical stability (prevents overflow)
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum()


def categorical_cross_entropy(y_true_onehot: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """
    Categorical Cross-Entropy Loss for multi-class classification.
    
    CCE = -Σ y_true_i · log(y_pred_i)
    
    y_true is one-hot encoded: [0, 0, 1, 0] for class 2
    y_pred is a probability distribution from softmax
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true_onehot * np.log(y_pred))


# Example: classifying cell types (3 classes)
print("\n=== Categorical Cross-Entropy ===")
raw_scores = np.array([2.0, 1.0, 0.5])  # Raw network output (logits)
probs = softmax(raw_scores)
y_true_onehot = np.array([1, 0, 0])  # True class is 0

print(f"Raw scores (logits): {raw_scores}")
print(f"After softmax: {probs.round(4)}")
print(f"True class (one-hot): {y_true_onehot}")
print(f"CCE Loss = {categorical_cross_entropy(y_true_onehot, probs):.4f}")

# Compare: what if the model was wrong?
wrong_probs = softmax(np.array([0.5, 2.0, 1.0]))  # Highest score on wrong class
print(f"\nWrong prediction probs: {wrong_probs.round(4)}")
print(f"CCE Loss (wrong) = {categorical_cross_entropy(y_true_onehot, wrong_probs):.4f} ← Much higher!")

# %% [markdown]
# ## 6. Sigmoid and Softmax Visualization

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Sigmoid
z = np.linspace(-6, 6, 200)
ax1.plot(z, sigmoid(z), color='#E74C3C', linewidth=2.5)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('z (logit)', fontsize=11)
ax1.set_ylabel('σ(z)', fontsize=11)
ax1.set_title('Sigmoid Function', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

# Softmax: show how it creates a probability distribution
categories = ['Normal', 'LSIL', 'HSIL']
scores_sets = [
    [3.0, 1.0, 0.5],    # Confident: Normal
    [1.0, 1.0, 1.0],    # Uncertain: equal
    [0.5, 1.0, 3.0],    # Confident: HSIL
]
colors = ['#2ECC71', '#F39C12', '#E74C3C']
x = np.arange(len(categories))
width = 0.25

for i, scores in enumerate(scores_sets):
    probs = softmax(np.array(scores))
    ax2.bar(x + i * width, probs, width, color=colors[i], alpha=0.8,
            label=f'Scores: {scores}')

ax2.set_xticks(x + width)
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylabel('Probability', fontsize=11)
ax2.set_title('Softmax: Scores → Probabilities', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('module_01_fundamentals/sigmoid_softmax.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Sigmoid/Softmax plot saved")

print("\n✅ Module 1, Script 03 complete!")
