"""
Module 2 — Script 04: Data Pipelines
======================================

Building robust data pipelines is essential for reproducible ML.
This script covers loading, cleaning, transforming, and splitting data.

Topics:
  - Loading data from various sources
  - Handling missing values strategies
  - Feature engineering
  - Train/validation/test splitting
  - Data preprocessing pipelines
  - Reproducibility with random seeds

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# %% [markdown]
# ## 1. Loading Data

# %%
print("=== Loading Data ===")

# Using sklearn's built-in datasets (no download needed)
wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
y_wine = pd.Series(wine.target, name='target')

print(f"Wine dataset: {X_wine.shape[0]} samples, {X_wine.shape[1]} features")
print(f"Classes: {wine.target_names}")
print(f"Class distribution: {np.bincount(y_wine)}")
print(f"\nFeature names:\n{wine.feature_names}")
print(f"\nFirst 3 rows:\n{X_wine.head(3)}")

# %% [markdown]
# ## 2. Data Inspection Checklist

# %%
def inspect_data(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Comprehensive data inspection function.
    Run this on every new dataset!
    """
    print(f"\n{'='*50}")
    print(f"  Data Inspection: {name}")
    print(f"{'='*50}")
    
    print(f"\n📏 Shape: {df.shape}")
    print(f"\n📊 Data Types:")
    print(df.dtypes.value_counts())
    
    print(f"\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  None! ✅")
    else:
        print(missing[missing > 0])
    
    print(f"\n📈 Numerical Summary:")
    print(df.describe().round(2))
    
    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    print(f"\n🔁 Duplicates: {n_duplicates}")
    
    # Check for constant columns (zero variance)
    constant_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                     if df[col].std() == 0]
    if constant_cols:
        print(f"\n⚠️ Constant columns: {constant_cols}")
    else:
        print(f"\n✅ No constant columns")


inspect_data(X_wine, "Wine Dataset")

# %% [markdown]
# ## 3. Handling Missing Values

# %%
print("\n=== Missing Value Strategies ===")

# Create a dataset with missing values for demonstration
np.random.seed(42)
df_demo = X_wine.copy()

# Introduce missing values randomly (10% missing)
mask = np.random.random(df_demo.shape) < 0.1
df_demo = df_demo.mask(mask)

missing_count = df_demo.isnull().sum()
print(f"Missing values per column:\n{missing_count[missing_count > 0]}")
print(f"Total missing: {df_demo.isnull().sum().sum()} / {df_demo.size} "
      f"({df_demo.isnull().sum().sum()/df_demo.size*100:.1f}%)")

# Strategy 1: Drop rows with missing values
df_dropped = df_demo.dropna()
print(f"\nStrategy 1 - Drop rows: {len(df_demo)} → {len(df_dropped)} rows")

# Strategy 2: Fill with mean (simple imputation)
df_mean_filled = df_demo.fillna(df_demo.mean())
print(f"Strategy 2 - Mean fill: {df_mean_filled.isnull().sum().sum()} missing remaining")

# Strategy 3: Fill with median (robust to outliers)
df_median_filled = df_demo.fillna(df_demo.median())
print(f"Strategy 3 - Median fill: {df_median_filled.isnull().sum().sum()} missing remaining")

# Strategy 4: Forward fill (for time series)
df_ffill = df_demo.ffill()
remaining = df_ffill.isnull().sum().sum()
print(f"Strategy 4 - Forward fill: {remaining} missing remaining")

print("\n💡 Rule of thumb:")
print("  - <5% missing → mean/median imputation")
print("  - 5-20% missing → more sophisticated imputation (KNN, MICE)")
print("  - >50% missing → consider dropping the feature")

# %% [markdown]
# ## 4. Feature Engineering

# %%
print("\n=== Feature Engineering ===")

df_fe = X_wine.copy()
df_fe['target'] = y_wine

# Create interaction features
df_fe['alcohol_x_flavanoids'] = df_fe['alcohol'] * df_fe['flavanoids']

# Create ratio features (domain knowledge driven)
df_fe['color_intensity_per_hue'] = df_fe['color_intensity'] / (df_fe['hue'] + 1e-8)

# Binning continuous variables
df_fe['alcohol_category'] = pd.cut(df_fe['alcohol'],
                                    bins=[0, 12, 13, 15],
                                    labels=['Low', 'Medium', 'High'])

# Log transform (for skewed features)
df_fe['log_proline'] = np.log1p(df_fe['proline'])

print(f"Original features: {X_wine.shape[1]}")
print(f"After feature engineering: {len([c for c in df_fe.columns if c != 'target'])}")
print(f"\nNew features:")
for col in ['alcohol_x_flavanoids', 'color_intensity_per_hue', 'alcohol_category', 'log_proline']:
    print(f"  {col}: {df_fe[col].dtype}")

# %% [markdown]
# ## 5. Train/Validation/Test Split

# %%
print("\n=== Data Splitting ===")

# Why three splits?
# - Training: model learns from this
# - Validation: tune hyperparameters on this (prevents overfitting to test set)
# - Test: final evaluation only (touch this ONCE!)

X = X_wine.values
y = y_wine.values

# Method 1: sklearn train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"Training:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# Check stratification preserved class distribution
print(f"\nClass distribution:")
print(f"  Full:       {np.bincount(y) / len(y)}")
print(f"  Train:      {np.bincount(y_train) / len(y_train)}")
print(f"  Validation: {np.bincount(y_val) / len(y_val)}")
print(f"  Test:       {np.bincount(y_test) / len(y_test)}")

# %% [markdown]
# ## 6. Feature Scaling

# %%
print("\n=== Feature Scaling ===")

# IMPORTANT: Fit scaler on TRAINING data only!
# Then transform validation and test with the same parameters.
# This prevents "data leakage" — information from test set affecting training.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit + transform
X_val_scaled = scaler.transform(X_val)             # Transform only!
X_test_scaled = scaler.transform(X_test)           # Transform only!

print("Before scaling:")
print(f"  Train mean: {X_train[:, 0].mean():.2f}, std: {X_train[:, 0].std():.2f}")
print(f"  Test  mean: {X_test[:, 0].mean():.2f}, std: {X_test[:, 0].std():.2f}")

print("\nAfter scaling:")
print(f"  Train mean: {X_train_scaled[:, 0].mean():.6f}, std: {X_train_scaled[:, 0].std():.6f}")
print(f"  Test  mean: {X_test_scaled[:, 0].mean():.4f}, std: {X_test_scaled[:, 0].std():.4f}")
print("  (Test stats won't be exactly 0/1 — that's expected and correct!)")

# %% [markdown]
# ## 7. Complete Data Pipeline

# %%
def create_ml_pipeline(dataset_loader, test_size: float = 0.2,
                        val_size: float = 0.2, random_state: int = 42):
    """
    End-to-end data pipeline for ML.
    
    Steps:
    1. Load and inspect data
    2. Handle missing values
    3. Split into train/val/test
    4. Scale features (fit on train only)
    """
    print("\n" + "="*50)
    print("  ML Data Pipeline")
    print("="*50)
    
    # 1. Load
    data = dataset_loader()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    print(f"\n[1/4] Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # 2. Clean
    missing = X.isnull().sum().sum()
    if missing > 0:
        X = X.fillna(X.median())
        print(f"[2/4] Filled {missing} missing values with median")
    else:
        print(f"[2/4] No missing values")
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)
    print(f"[3/4] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # 4. Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print(f"[4/4] Scaled with StandardScaler")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'feature_names': data.feature_names,
        'target_names': data.target_names
    }


# Run the pipeline
pipeline_data = create_ml_pipeline(load_breast_cancer)
print(f"\nPipeline complete! Ready for modeling.")
print(f"Features: {len(pipeline_data['feature_names'])}")
print(f"Classes: {pipeline_data['target_names']}")

print("\n✅ Module 2, Script 04 complete!")
