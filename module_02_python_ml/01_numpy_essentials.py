"""
Module 2 — Script 01: NumPy Essentials
========================================

Master the core NumPy operations that power all of machine learning.
Vectorized operations replace slow Python loops with fast C operations.

Topics:
  - Array creation and manipulation
  - Indexing and slicing
  - Broadcasting rules
  - Vectorized operations vs loops
  - Random number generation
  - Practical patterns for ML

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import time

# %% [markdown]
# ## 1. Array Creation

# %%
print("=== Array Creation ===")

# From lists
a = np.array([1, 2, 3, 4, 5])
print(f"From list: {a}, dtype={a.dtype}, shape={a.shape}")

# 2D array (matrix)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"Matrix shape: {M.shape}, ndim={M.ndim}")

# Common initializations
zeros = np.zeros((3, 4))             # All zeros
ones = np.ones((2, 3))               # All ones
identity = np.eye(3)                  # Identity matrix
arange = np.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1.0]

print(f"zeros(3,4) shape: {zeros.shape}")
print(f"arange(0,10,2): {arange}")
print(f"linspace(0,1,5): {linspace}")

# Random arrays (critical for ML: weight initialization)
np.random.seed(42)
uniform = np.random.uniform(0, 1, (3, 3))
normal = np.random.randn(3, 3)       # Standard normal N(0,1)
print(f"\nRandom normal:\n{normal.round(3)}")

# %% [markdown]
# ## 2. Indexing and Slicing

# %%
print("\n=== Indexing & Slicing ===")

data = np.arange(20).reshape(4, 5)
print(f"Original (4×5):\n{data}")

# Basic indexing
print(f"\nRow 0: {data[0]}")
print(f"Element [1,3]: {data[1, 3]}")
print(f"Last row: {data[-1]}")
print(f"Rows 1-2, Cols 2-4:\n{data[1:3, 2:5]}")

# Boolean indexing (used constantly for filtering)
mask = data > 10
print(f"\nElements > 10: {data[mask]}")
print(f"Mask shape: {mask.shape}")

# Fancy indexing
indices = np.array([0, 2, 3])
print(f"Rows [0,2,3]:\n{data[indices]}")

# Important: views vs copies
# Slicing creates a VIEW (shared memory), not a copy!
view = data[0:2]
view[0, 0] = 999
print(f"\nModifying view also modifies original: data[0,0] = {data[0, 0]}")
data[0, 0] = 0  # Reset

# To create a copy:
copy = data[0:2].copy()
copy[0, 0] = 999
print(f"Copy doesn't affect original: data[0,0] = {data[0, 0]}")

# %% [markdown]
# ## 3. Reshaping

# %%
print("\n=== Reshaping ===")

# Reshape: changes the shape without changing data
a = np.arange(12)
print(f"Original: {a} (shape={a.shape})")

reshaped = a.reshape(3, 4)
print(f"Reshaped (3×4):\n{reshaped}")

reshaped_auto = a.reshape(2, -1)  # -1 = infer this dimension
print(f"Reshaped (2×?):\n{reshaped_auto}")

# Flatten: multi-dim → 1D
print(f"Flattened: {reshaped.flatten()}")

# Transpose
print(f"Transposed (4×3):\n{reshaped.T}")

# Adding dimensions (used for broadcasting)
vec = np.array([1, 2, 3])
print(f"\nvec shape: {vec.shape}")
print(f"vec[:, np.newaxis] shape: {vec[:, np.newaxis].shape}")  # Column vector
print(f"vec[np.newaxis, :] shape: {vec[np.newaxis, :].shape}")  # Row vector

# %% [markdown]
# ## 4. Broadcasting — The Most Important NumPy Concept

# %%
print("\n=== Broadcasting ===")

# Broadcasting rules:
# 1. If arrays have different ndims, prepend 1s to the smaller shape
# 2. Dimensions are compatible if they're equal OR one of them is 1
# 3. Arrays are broadcast along size-1 dimensions

# Example 1: Scalar + Array
a = np.array([1, 2, 3])
print(f"[1,2,3] + 10 = {a + 10}")  # 10 is broadcast to [10,10,10]

# Example 2: Matrix + Vector (normalize columns)
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

col_means = M.mean(axis=0)  # Shape: (3,)
print(f"\nMatrix:\n{M}")
print(f"Column means: {col_means}")
print(f"Centered (M - means):\n{M - col_means}")  # (3,3) - (3,) broadcasts!

# Example 3: Outer product via broadcasting
row = np.array([1, 2, 3])[:, np.newaxis]  # (3,1)
col = np.array([10, 20, 30])[np.newaxis, :]  # (1,3)
print(f"\nOuter product via broadcasting:\n{row * col}")

# Example 4: Distance matrix (very common in ML)
# Compute pairwise distances between points
points = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
# Expand dims for broadcasting: (3,1,2) - (1,3,2) → (3,3,2)
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt(np.sum(diff**2, axis=2))
print(f"\nPoints:\n{points}")
print(f"Pairwise distance matrix:\n{distances.round(3)}")

# %% [markdown]
# ## 5. Vectorization: Speed Matters

# %%
print("\n=== Vectorization Performance ===")

# Compare loop vs vectorized dot product
size = 1_000_000
a = np.random.randn(size)
b = np.random.randn(size)

# Python loop
start = time.time()
dot_loop = 0
for i in range(size):
    dot_loop += a[i] * b[i]
time_loop = time.time() - start

# NumPy vectorized
start = time.time()
dot_numpy = np.dot(a, b)
time_numpy = time.time() - start

print(f"Python loop:  {time_loop:.4f}s (result={dot_loop:.4f})")
print(f"NumPy dot:    {time_numpy:.6f}s (result={dot_numpy:.4f})")
print(f"Speedup: {time_loop/time_numpy:.0f}x faster!")

# Rule of thumb: NEVER use Python loops for array math.
# Always find the vectorized NumPy operation.

# %% [markdown]
# ## 6. Common ML Patterns with NumPy

# %%
print("\n=== ML Patterns ===")
np.random.seed(42)

# Pattern 1: Z-score normalization
data = np.random.normal(loc=50, scale=15, size=(100, 3))
normalized = (data - data.mean(axis=0)) / data.std(axis=0)
print(f"Before normalization - means: {data.mean(axis=0).round(2)}")
print(f"After normalization  - means: {normalized.mean(axis=0).round(10)}")
print(f"After normalization  - stds:  {normalized.std(axis=0).round(10)}")

# Pattern 2: One-hot encoding
labels = np.array([0, 2, 1, 0, 3])
n_classes = labels.max() + 1
one_hot = np.eye(n_classes)[labels]
print(f"\nLabels: {labels}")
print(f"One-hot:\n{one_hot}")

# Pattern 3: Batching data
dataset = np.arange(100)
batch_size = 32
n_batches = len(dataset) // batch_size
print(f"\nDataset size: {len(dataset)}, Batch size: {batch_size}")
print(f"Number of full batches: {n_batches}")
for i in range(n_batches):
    batch = dataset[i*batch_size:(i+1)*batch_size]
    print(f"  Batch {i}: indices {batch[0]}-{batch[-1]}")

# Pattern 4: Argmax (getting the predicted class)
logits = np.array([[2.1, 0.5, -1.0],
                    [0.3, 3.2, 1.1],
                    [-0.5, 0.1, 4.0]])
predictions = np.argmax(logits, axis=1)
print(f"\nLogits:\n{logits}")
print(f"Predictions (argmax): {predictions}")

# Pattern 5: Masking (handling missing data)
data_with_nans = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
valid_mask = ~np.isnan(data_with_nans)
clean_mean = data_with_nans[valid_mask].mean()
print(f"\nData with NaN: {data_with_nans}")
print(f"Mean (ignoring NaN): {clean_mean:.2f}")
print(f"NumPy nanmean: {np.nanmean(data_with_nans):.2f}")

print("\n✅ Module 2, Script 01 complete!")
