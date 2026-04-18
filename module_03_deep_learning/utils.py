"""
Module 3 — Shared Utilities
=============================

Helper functions used across Module 3 scripts.
"""

import numpy as np


# === Activation Functions ===

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation: maps to (0, 1)."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: σ(z) * (1 - σ(z))."""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, z)."""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)


def tanh(z: np.ndarray) -> np.ndarray:
    """Tanh activation: maps to (-1, 1)."""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh²(z)."""
    return 1 - np.tanh(z) ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax: converts scores to probabilities."""
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


# === Loss Functions ===

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                          eps: float = 1e-7) -> float:
    """Binary cross-entropy loss."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                                eps: float = 1e-7) -> float:
    """Categorical cross-entropy loss."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss."""
    return np.mean((y_true - y_pred) ** 2)


# === Data Utilities ===

def one_hot_encode(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoded matrix."""
    return np.eye(n_classes)[labels]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    return np.mean(y_true == y_pred)


def train_test_split_simple(X, y, test_size=0.2, seed=42):
    """Simple train/test split."""
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
