"""
Module 1 — Script 01: Linear Algebra Foundations
=================================================

This script covers the fundamental linear algebra concepts that underpin
all of machine learning. We start with pure Python implementations to
build intuition, then show the NumPy equivalents for practical use.

Topics:
  - Vectors: creation, addition, scalar multiplication, dot product
  - Matrices: creation, multiplication, transpose
  - Norms and distances
  - Eigenvalues (conceptual introduction)

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np

# %% [markdown]
# ## 1. Vectors — The Building Blocks

# %%
# === PURE PYTHON VECTORS ===
# A vector is simply an ordered list of numbers.
# In ML, each number represents a feature (e.g., pixel intensity, patient age).

def vector_add(v1: list[float], v2: list[float]) -> list[float]:
    """Add two vectors element-wise."""
    assert len(v1) == len(v2), "Vectors must have the same dimension"
    return [a + b for a, b in zip(v1, v2)]


def scalar_multiply(scalar: float, v: list[float]) -> list[float]:
    """Multiply every element of a vector by a scalar."""
    return [scalar * x for x in v]


def dot_product(v1: list[float], v2: list[float]) -> float:
    """
    Compute the dot product of two vectors.
    
    The dot product measures how "aligned" two vectors are:
      - Positive: vectors point in similar directions
      - Zero: vectors are perpendicular (orthogonal)
      - Negative: vectors point in opposite directions
    
    In ML, this is the core operation of a neural network neuron:
      output = dot(weights, inputs) + bias
    """
    assert len(v1) == len(v2), "Vectors must have the same dimension"
    return sum(a * b for a, b in zip(v1, v2))


# Let's test our implementations
v1 = [1.0, 2.0, 3.0]
v2 = [4.0, 5.0, 6.0]

print("=== Pure Python Vector Operations ===")
print(f"v1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {vector_add(v1, v2)}")
print(f"3 * v1 = {scalar_multiply(3, v1)}")
print(f"v1 · v2 = {dot_product(v1, v2)}")  # 1*4 + 2*5 + 3*6 = 32

# %% [markdown]
# ## 2. The Same Operations with NumPy
# NumPy is *vastly* faster for large vectors because it uses optimized C code.

# %%
# === NUMPY VECTORS ===
v1_np = np.array([1.0, 2.0, 3.0])
v2_np = np.array([4.0, 5.0, 6.0])

print("\n=== NumPy Vector Operations ===")
print(f"v1 + v2 = {v1_np + v2_np}")          # Element-wise addition
print(f"3 * v1 = {3 * v1_np}")               # Scalar multiplication
print(f"v1 · v2 = {np.dot(v1_np, v2_np)}")   # Dot product
print(f"v1 * v2 = {v1_np * v2_np}")           # Element-wise multiply (NOT dot product!)

# %% [markdown]
# ## 3. Vector Norms — Measuring Length

# %%
def vector_norm(v: list[float], p: int = 2) -> float:
    """
    Compute the Lp norm of a vector.
    
    L1 norm (Manhattan distance): sum of absolute values
    L2 norm (Euclidean distance): square root of sum of squares
    
    Norms are used everywhere in ML:
      - L2 regularization penalizes large weights using the L2 norm
      - Distance metrics use norms to measure similarity
    """
    return sum(abs(x) ** p for x in v) ** (1 / p)


v = [3.0, 4.0]
print("\n=== Vector Norms ===")
print(f"v = {v}")
print(f"L1 norm = {vector_norm(v, p=1)}")      # |3| + |4| = 7
print(f"L2 norm = {vector_norm(v, p=2)}")      # sqrt(9 + 16) = 5.0

# NumPy equivalent
v_np = np.array(v)
print(f"NumPy L2 norm = {np.linalg.norm(v_np)}")

# %% [markdown]
# ## 4. Matrices — Transformations and Data

# %%
# === PURE PYTHON MATRICES ===
# A matrix is a 2D array of numbers. In ML:
#   - A dataset is a matrix: rows = samples, columns = features
#   - A neural network layer weight is a matrix

def matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Multiply two matrices A (m×n) and B (n×p) to produce C (m×p).
    
    This is THE fundamental operation in neural networks.
    A single forward pass through a layer is: output = A @ x + b
    where A is the weight matrix and x is the input vector.
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    assert n == len(B), f"Incompatible shapes: A is {m}×{n}, B is {len(B)}×{p}"
    
    # Initialize result matrix with zeros
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C


def transpose(A: list[list[float]]) -> list[list[float]]:
    """
    Transpose a matrix: swap rows and columns.
    
    If A is m×n, then A^T is n×m.
    Used constantly in: backpropagation, computing gradients, data reshaping.
    """
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


# Test matrix operations
A = [[1, 2],
     [3, 4],
     [5, 6]]  # 3×2 matrix

B = [[7, 8, 9],
     [10, 11, 12]]  # 2×3 matrix

C = matrix_multiply(A, B)
print("\n=== Pure Python Matrix Operations ===")
print("A (3×2):")
for row in A:
    print(f"  {row}")
print("B (2×3):")
for row in B:
    print(f"  {row}")
print("A × B (3×3):")
for row in C:
    print(f"  {row}")

print("\nTranspose of A:")
for row in transpose(A):
    print(f"  {row}")

# %% [markdown]
# ## 5. NumPy Matrix Operations

# %%
A_np = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=float)  # 3×2

B_np = np.array([[7, 8, 9],
                  [10, 11, 12]], dtype=float)  # 2×3

print("\n=== NumPy Matrix Operations ===")
print(f"A shape: {A_np.shape}")
print(f"B shape: {B_np.shape}")
print(f"A @ B:\n{A_np @ B_np}")          # Matrix multiplication
print(f"A.T:\n{A_np.T}")                 # Transpose
print(f"A * 2:\n{A_np * 2}")             # Element-wise scalar multiply

# %% [markdown]
# ## 6. Identity Matrix and Inverse

# %%
# The identity matrix: multiplying by it changes nothing (like multiplying by 1)
I = np.eye(3)
print("\n=== Identity Matrix ===")
print(f"I (3×3):\n{I}")

# Matrix inverse: A @ A_inv = I (only for square, non-singular matrices)
M = np.array([[2, 1],
               [5, 3]], dtype=float)
M_inv = np.linalg.inv(M)

print(f"\nM:\n{M}")
print(f"M inverse:\n{M_inv}")
print(f"M @ M_inv (should be identity):\n{np.round(M @ M_inv, 10)}")

# %% [markdown]
# ## 7. Eigenvalues — A Conceptual Preview

# %%
# Eigenvalues and eigenvectors decompose a matrix into its fundamental "actions".
# For a matrix A and vector v: A @ v = λ * v
# v is an eigenvector (direction preserved), λ is the eigenvalue (scaling factor).
#
# Why this matters in ML:
#   - PCA (dimensionality reduction) finds eigenvectors of the covariance matrix
#   - Understanding training dynamics (large eigenvalues = fast learning directions)

M = np.array([[4, 2],
               [1, 3]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(M)
print("\n=== Eigenvalues & Eigenvectors ===")
print(f"Matrix M:\n{M}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Verify: M @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    lhs = M @ v
    rhs = lam * v
    print(f"\nEigenvector {i+1}: {v}")
    print(f"  M @ v     = {lhs}")
    print(f"  λ * v     = {rhs}")
    print(f"  Equal? {np.allclose(lhs, rhs)}")

# %% [markdown]
# ## 8. Practical Application: Data as Matrices

# %%
# In ML, your dataset IS a matrix:
#   - Rows = samples (patients, images, sentences)
#   - Columns = features (measurements, pixel values, word counts)

# Example: 4 patients with 3 measurements each
patients = np.array([
    [120, 80, 25],   # Patient 1: systolic BP, diastolic BP, age
    [140, 90, 55],   # Patient 2
    [110, 70, 30],   # Patient 3
    [135, 85, 45],   # Patient 4
], dtype=float)

print("\n=== Data as a Matrix ===")
print(f"Patient data shape: {patients.shape} (4 patients × 3 features)")
print(f"Mean per feature: {patients.mean(axis=0)}")   # Mean across rows
print(f"Std per feature: {patients.std(axis=0)}")      # Std across rows

# Normalization (z-score): crucial preprocessing step
normalized = (patients - patients.mean(axis=0)) / patients.std(axis=0)
print(f"\nNormalized data:\n{np.round(normalized, 3)}")
print(f"Normalized means: {np.round(normalized.mean(axis=0), 10)}")  # Should be ~0
print(f"Normalized stds: {np.round(normalized.std(axis=0), 10)}")    # Should be ~1

print("\n✅ Module 1, Script 01 complete!")

# %%
