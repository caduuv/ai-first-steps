"""
Module 2 — Mini Project: Exploratory Data Analysis
====================================================

A complete EDA workflow on the Breast Cancer Wisconsin dataset.
This ties together NumPy, Pandas, and visualization skills.

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# %% [markdown]
# ## 1. Load and Inspect

# %%
print("=" * 60)
print("  Exploratory Data Analysis: Breast Cancer Wisconsin")
print("=" * 60)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
df['diagnosis_label'] = df['diagnosis'].map({0: 'Malignant', 1: 'Benign'})

print(f"\nDataset shape: {df.shape}")
print(f"Features: {len(data.feature_names)}")
print(f"Classes: {data.target_names}")
print(f"\nClass distribution:")
print(df['diagnosis_label'].value_counts())
print(f"\nFirst 5 rows:")
print(df.head())

# %% [markdown]
# ## 2. Statistical Summary

# %%
print("\n=== Statistical Summary ===")
print(df.describe().round(2))

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# %% [markdown]
# ## 3. Class Distribution

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
counts = df['diagnosis_label'].value_counts()
ax1.bar(counts.index, counts.values, color=['#E74C3C', '#2ECC71'], edgecolor='white')
ax1.set_title('Diagnosis Distribution', fontweight='bold', fontsize=13)
ax1.set_ylabel('Count')

# Pie chart
ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=['#E74C3C', '#2ECC71'], startangle=90,
        explode=(0.05, 0))
ax2.set_title('Diagnosis Proportions', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('module_02_python_ml/mini_project_eda/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Class distribution plot saved")

# %% [markdown]
# ## 4. Feature Distributions by Class

# %%
# Select the most important features (mean values)
key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                'mean smoothness', 'mean compactness']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, feature in enumerate(key_features):
    for label, color in [('Malignant', '#E74C3C'), ('Benign', '#2ECC71')]:
        subset = df[df['diagnosis_label'] == label][feature]
        axes[idx].hist(subset, bins=25, alpha=0.6, color=color, label=label, density=True)
    
    axes[idx].set_title(feature, fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].legend(fontsize=9)

plt.suptitle('Feature Distributions: Malignant vs Benign', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_02_python_ml/mini_project_eda/feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Feature distributions plot saved")

# %% [markdown]
# ## 5. Correlation Analysis

# %%
# Use only the 10 "mean" features for a cleaner heatmap
mean_features = [col for col in df.columns if 'mean' in col]
corr_matrix = df[mean_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation (Mean Features)', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('module_02_python_ml/mini_project_eda/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Correlation matrix saved")

# Highly correlated pairs
print("\n=== Highly Correlated Feature Pairs (|r| > 0.8) ===")
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.8:
            print(f"  {mean_features[i]} ↔ {mean_features[j]}: r = {r:.3f}")

# %% [markdown]
# ## 6. Feature Importance (Simple Univariate)

# %%
# Which features best separate the classes?
# Simple approach: compute the difference in means normalized by pooled std
print("\n=== Feature Discriminative Power ===")

feature_scores = {}
for feature in mean_features:
    malignant = df[df['diagnosis'] == 0][feature]
    benign = df[df['diagnosis'] == 1][feature]
    
    # Cohen's d: effect size measure
    pooled_std = np.sqrt((malignant.std()**2 + benign.std()**2) / 2)
    cohens_d = abs(malignant.mean() - benign.mean()) / pooled_std
    feature_scores[feature] = cohens_d

# Sort by discriminative power
sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

header_d = "Cohen's d"
print(f"{'Feature':<25} {header_d:>10} {'Interpretation':>15}")
print(f"{'─'*25} {'─'*10} {'─'*15}")
for feature, score in sorted_features:
    interpretation = 'Large' if score > 0.8 else 'Medium' if score > 0.5 else 'Small'
    print(f"{feature:<25} {score:>10.3f} {interpretation:>15}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 5))
features_sorted = [f[0].replace('mean ', '') for f in sorted_features]
scores_sorted = [f[1] for f in sorted_features]

colors = ['#E74C3C' if s > 0.8 else '#F39C12' if s > 0.5 else '#3498DB' for s in scores_sorted]
ax.barh(features_sorted[::-1], scores_sorted[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
ax.set_title('Feature Discriminative Power', fontweight='bold', fontsize=13)
ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect threshold')
ax.legend()

plt.tight_layout()
plt.savefig('module_02_python_ml/mini_project_eda/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Feature importance plot saved")

# %% [markdown]
# ## 7. Scatter Plot: Best 2 Features

# %%
best_features = [sorted_features[0][0], sorted_features[1][0]]

fig, ax = plt.subplots(figsize=(8, 6))
for label, color, marker in [('Malignant', '#E74C3C', 'x'), ('Benign', '#2ECC71', 'o')]:
    mask = df['diagnosis_label'] == label
    ax.scatter(df[mask][best_features[0]], df[mask][best_features[1]],
               c=color, marker=marker, alpha=0.6, s=30, label=label)

ax.set_xlabel(best_features[0], fontsize=12)
ax.set_ylabel(best_features[1], fontsize=12)
ax.set_title(f'Top 2 Discriminative Features', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_02_python_ml/mini_project_eda/scatter_top_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Top features scatter plot saved")

print("\n" + "=" * 60)
print("  EDA Summary")
print("=" * 60)
print(f"  📊 Dataset: {df.shape[0]} samples, {len(mean_features)} mean features")
print(f"  📊 Classes: Benign ({(df['diagnosis']==1).sum()}), Malignant ({(df['diagnosis']==0).sum()})")
print(f"  📊 Most discriminative: {sorted_features[0][0]} (d={sorted_features[0][1]:.2f})")
print(f"  📊 Least discriminative: {sorted_features[-1][0]} (d={sorted_features[-1][1]:.2f})")
print(f"  📊 Highly correlated pairs: {sum(1 for f in sorted_features if f[1] > 0.8)}")
print(f"\n✅ Mini-project EDA complete!")
