"""
Module 2 — Script 03: Visualization
=====================================

Learn to create effective visualizations for data exploration and
model evaluation. Good plots help you find patterns, catch errors,
and communicate results.

Topics:
  - Matplotlib fundamentals
  - Seaborn statistical plots
  - Common ML visualization patterns
  - Customization and styling

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set global style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})

# %% [markdown]
# ## 1. Matplotlib Fundamentals

# %%
print("=== Matplotlib Fundamentals ===")

# Basic line plot
fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 2 * np.pi, 100)
ax.plot(x, np.sin(x), label='sin(x)', color='#E74C3C', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', color='#3498DB', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Sine and Cosine', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('module_02_python_ml/01_line_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Line plot saved")

# %% [markdown]
# ## 2. Subplots — Multiple Plots in One Figure

# %%
np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Histogram
data_hist = np.random.normal(0, 1, 1000)
axes[0, 0].hist(data_hist, bins=40, color='#9B59B6', edgecolor='white', alpha=0.8)
axes[0, 0].set_title('Histogram', fontweight='bold')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Scatter plot
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.6, c='#E67E22', s=30)
axes[0, 1].set_title('Scatter Plot', fontweight='bold')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')

# Bar plot
categories = ['Normal', 'LSIL', 'HSIL', 'ASC-US']
counts = [450, 120, 80, 50]
colors = ['#2ECC71', '#F39C12', '#E74C3C', '#3498DB']
axes[1, 0].bar(categories, counts, color=colors, edgecolor='white')
axes[1, 0].set_title('Cell Type Distribution', fontweight='bold')
axes[1, 0].set_ylabel('Count')

# Box plot
data_box = [np.random.normal(0, 1, 100),
            np.random.normal(1, 1.5, 100),
            np.random.normal(-0.5, 0.8, 100)]
bp = axes[1, 1].boxplot(data_box, labels=['Group A', 'Group B', 'Group C'],
                         patch_artist=True)
for patch, color in zip(bp['boxes'], ['#3498DB', '#E74C3C', '#2ECC71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1, 1].set_title('Box Plot', fontweight='bold')

plt.suptitle('Matplotlib Subplot Gallery', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_02_python_ml/02_subplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Subplots saved")

# %% [markdown]
# ## 3. Seaborn Statistical Plots

# %%
print("\n=== Seaborn Statistical Plots ===")

# Create a richer dataset
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age': np.concatenate([
        np.random.normal(35, 8, 100),
        np.random.normal(50, 10, 60),
        np.random.normal(55, 12, 40)
    ]),
    'cell_count': np.concatenate([
        np.random.normal(180, 30, 100),
        np.random.normal(100, 25, 60),
        np.random.normal(60, 20, 40)
    ]),
    'abnormality_score': np.concatenate([
        np.random.beta(2, 10, 100),
        np.random.beta(5, 5, 60),
        np.random.beta(8, 3, 40)
    ]),
    'cell_type': ['Normal'] * 100 + ['LSIL'] * 60 + ['HSIL'] * 40
})

# Distribution plot with KDE
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, col in enumerate(['age', 'cell_count', 'abnormality_score']):
    for cell_type in ['Normal', 'LSIL', 'HSIL']:
        subset = df[df['cell_type'] == cell_type][col]
        axes[i].hist(subset, bins=20, alpha=0.5, label=cell_type, density=True)
    axes[i].set_title(f'{col} Distribution', fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].legend(fontsize=9)

plt.tight_layout()
plt.savefig('module_02_python_ml/03_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Distribution plot saved")

# %% [markdown]
# ## 4. Correlation Heatmap

# %%
fig, ax = plt.subplots(figsize=(6, 5))

numeric_cols = ['age', 'cell_count', 'abnormality_score']
corr = df[numeric_cols].corr()

sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, linewidths=1)
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('module_02_python_ml/04_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Correlation heatmap saved")

# %% [markdown]
# ## 5. Pair Plot — Multivariate Relationships

# %%
g = sns.pairplot(df, hue='cell_type', vars=numeric_cols,
                  palette={'Normal': '#2ECC71', 'LSIL': '#F39C12', 'HSIL': '#E74C3C'},
                  plot_kws={'alpha': 0.5, 's': 20},
                  diag_kws={'alpha': 0.5})
g.figure.suptitle('Pair Plot by Cell Type', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('module_02_python_ml/05_pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Pair plot saved")

# %% [markdown]
# ## 6. ML-Specific Visualizations

# %%
# Confusion Matrix
from sklearn.metrics import confusion_matrix

y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 0, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 0, 2, 2])

cm = confusion_matrix(y_true, y_pred)
labels = ['Normal', 'LSIL', 'HSIL']

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('module_02_python_ml/06_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Confusion matrix saved")

# %% [markdown]
# ## 7. Training Loss Curves

# %%
# Simulated training curves (we'll generate real ones later)
epochs = np.arange(1, 101)
train_loss = 2.5 * np.exp(-0.05 * epochs) + 0.3 + np.random.normal(0, 0.02, 100)
val_loss = 2.5 * np.exp(-0.04 * epochs) + 0.45 + np.random.normal(0, 0.03, 100)
# Simulate overfitting after epoch 70
val_loss[70:] += np.linspace(0, 0.3, 30)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Loss curves
ax1.plot(epochs, train_loss, label='Training Loss', color='#3498DB', linewidth=1.5)
ax1.plot(epochs, val_loss, label='Validation Loss', color='#E74C3C', linewidth=1.5)
ax1.axvline(x=70, color='gray', linestyle='--', alpha=0.5, label='Overfitting starts')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss', fontweight='bold')
ax1.legend()

# Accuracy curves (simulated)
train_acc = 1 - train_loss / 3
val_acc = 1 - val_loss / 3
ax2.plot(epochs, train_acc * 100, label='Training Accuracy', color='#3498DB', linewidth=1.5)
ax2.plot(epochs, val_acc * 100, label='Validation Accuracy', color='#E74C3C', linewidth=1.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training vs Validation Accuracy', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig('module_02_python_ml/07_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Training curves saved")

print("\n✅ Module 2, Script 03 complete!")
