"""
Module 2 — Script 02: Pandas Essentials
=========================================

Pandas is the standard tool for data manipulation in Python.
This script covers the core operations you'll use in every ML project.

Topics:
  - DataFrames: creation, inspection, selection
  - Filtering and boolean indexing
  - GroupBy: split-apply-combine
  - Handling missing data
  - Merging and joining DataFrames
  - Data types and conversions

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import pandas as pd

# %% [markdown]
# ## 1. Creating DataFrames

# %%
print("=== Creating DataFrames ===")

# From a dictionary
data = {
    'patient_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'age': [45, 32, 58, 28, 67, 41, 55, 36],
    'cell_type': ['Normal', 'LSIL', 'HSIL', 'Normal', 'HSIL', 'Normal', 'LSIL', 'Normal'],
    'cell_count': [150, 89, 45, 200, 30, 175, 95, 180],
    'abnormality_score': [0.12, 0.65, 0.89, 0.08, 0.95, 0.15, 0.58, 0.10]
}
df = pd.DataFrame(data)

print(df)
print(f"\nShape: {df.shape} (rows × columns)")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")

# %% [markdown]
# ## 2. Inspecting Data

# %%
print("\n=== Data Inspection ===")

# First/last rows
print("Head (first 3 rows):")
print(df.head(3))

# Statistical summary
print("\nDescribe (numerical columns):")
print(df.describe().round(2))

# Value counts
print("\nCell type distribution:")
print(df['cell_type'].value_counts())

# Info about memory and types
print("\nInfo:")
df.info()

# %% [markdown]
# ## 3. Selecting Data

# %%
print("\n=== Selecting Data ===")

# Single column (returns Series)
ages = df['age']
print(f"Ages: {ages.values}")

# Multiple columns (returns DataFrame)
subset = df[['patient_id', 'cell_type', 'abnormality_score']]
print(f"\nSubset:\n{subset}")

# By position (iloc)
print(f"\nRow 0: {df.iloc[0].values}")
print(f"Rows 1-3, Cols 0-2:\n{df.iloc[1:4, 0:3]}")

# By label (loc)
print(f"\nRow 0, 'cell_type': {df.loc[0, 'cell_type']}")

# %% [markdown]
# ## 4. Filtering with Boolean Indexing

# %%
print("\n=== Filtering ===")

# Single condition
high_risk = df[df['abnormality_score'] > 0.5]
print(f"High risk patients (score > 0.5):\n{high_risk}")

# Multiple conditions (use & for AND, | for OR)
older_abnormal = df[(df['age'] > 40) & (df['cell_type'] != 'Normal')]
print(f"\nOlder patients with abnormalities:\n{older_abnormal}")

# isin for multiple values
lsil_hsil = df[df['cell_type'].isin(['LSIL', 'HSIL'])]
print(f"\nLSIL or HSIL patients:\n{lsil_hsil[['patient_id', 'cell_type']]}")

# %% [markdown]
# ## 5. Creating New Columns

# %%
print("\n=== Creating Columns ===")

# Computed column
df['risk_category'] = pd.cut(df['abnormality_score'],
                              bins=[0, 0.3, 0.7, 1.0],
                              labels=['Low', 'Medium', 'High'])

# Apply function
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 40 else 'Middle' if x < 60 else 'Senior')

print(df[['patient_id', 'age', 'age_group', 'abnormality_score', 'risk_category']])

# %% [markdown]
# ## 6. GroupBy: Split-Apply-Combine

# %%
print("\n=== GroupBy Operations ===")

# Group by cell type and compute statistics
grouped = df.groupby('cell_type').agg({
    'age': ['mean', 'std'],
    'cell_count': 'mean',
    'abnormality_score': ['mean', 'max']
}).round(2)

print(f"Statistics by cell type:\n{grouped}")

# Simple aggregation
print(f"\nMean abnormality score by cell type:")
print(df.groupby('cell_type')['abnormality_score'].mean().round(3))

# Count per group
print(f"\nCount per risk category:")
print(df.groupby('risk_category', observed=True).size())

# %% [markdown]
# ## 7. Handling Missing Data

# %%
print("\n=== Missing Data ===")

# Create data with missing values
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': ['x', 'y', 'z', 'x', 'y']
})
print(f"Data with NaN:\n{df_missing}")

# Check for missing values
print(f"\nMissing counts:\n{df_missing.isnull().sum()}")

# Fill missing values
df_filled = df_missing.copy()
df_filled['A'] = df_filled['A'].fillna(df_filled['A'].mean())  # Fill with mean
df_filled['B'] = df_filled['B'].ffill()  # Forward fill
print(f"\nAfter filling:\n{df_filled}")

# Drop rows with any missing values
df_dropped = df_missing.dropna()
print(f"\nAfter dropping NaN rows:\n{df_dropped}")

# %% [markdown]
# ## 8. Merging DataFrames

# %%
print("\n=== Merging DataFrames ===")

# Patient demographics
demographics = pd.DataFrame({
    'patient_id': [101, 102, 103, 104, 105],
    'gender': ['F', 'F', 'F', 'F', 'F'],
    'smoking': [False, True, False, False, True]
})

# Test results
test_results = pd.DataFrame({
    'patient_id': [101, 102, 103, 106, 107],
    'hpv_positive': [False, True, True, False, True],
    'test_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19']
})

# Inner join (only matching records)
inner_merged = pd.merge(demographics, test_results, on='patient_id', how='inner')
print(f"Inner join:\n{inner_merged}")

# Left join (all demographics, matching test results)
left_merged = pd.merge(demographics, test_results, on='patient_id', how='left')
print(f"\nLeft join:\n{left_merged}")

# Outer join (all records from both)
outer_merged = pd.merge(demographics, test_results, on='patient_id', how='outer')
print(f"\nOuter join:\n{outer_merged}")

# %% [markdown]
# ## 9. Sorting and Ranking

# %%
print("\n=== Sorting & Ranking ===")

# Sort by abnormality score (descending)
sorted_df = df.sort_values('abnormality_score', ascending=False)
print(f"Sorted by risk (highest first):")
print(sorted_df[['patient_id', 'cell_type', 'abnormality_score']].head())

# Rank
df['risk_rank'] = df['abnormality_score'].rank(ascending=False).astype(int)
print(f"\nWith risk ranking:")
print(df[['patient_id', 'abnormality_score', 'risk_rank']].sort_values('risk_rank'))

# %% [markdown]
# ## 10. Practical Pattern: Preparing Data for ML

# %%
print("\n=== ML Data Preparation Pattern ===")

# 1. Select features and target
features = df[['age', 'cell_count', 'abnormality_score']].copy()
target = (df['cell_type'] != 'Normal').astype(int)  # Binary: abnormal or not

# 2. Check for missing values
print(f"Missing values: {features.isnull().sum().sum()}")

# 3. Convert to numpy arrays (what ML models expect)
X = features.values
y = target.values
print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"y shape: {y.shape}, distribution: {np.bincount(y)}")

# 4. Simple train/test split (we'll use sklearn in the next module)
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(0.8 * len(X))
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"\nTrain: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

print("\n✅ Module 2, Script 02 complete!")
