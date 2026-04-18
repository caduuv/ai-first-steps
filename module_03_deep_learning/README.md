# Module 3 — Deep Learning

## 🎯 Learning Objectives

By the end of this module, you will:

1. Understand the **perceptron** — the simplest neural unit
2. Build a **full neural network from scratch** using only NumPy
3. Implement **backpropagation** and understand the chain rule
4. Apply **regularization** techniques to prevent overfitting
5. Rebuild everything in **PyTorch** and understand the framework

---

## 📖 Theory Overview

### From Perceptron to Deep Networks

A **perceptron** is the simplest neural unit:
```
output = activation(w · x + b)
```

Stack many perceptrons in layers → neural network. Stack many layers → **deep** learning.

### The Forward Pass

Data flows through layers, each applying: `output = activation(W @ input + bias)`

Common activations:
- **ReLU**: `max(0, x)` — simple, effective, standard
- **Sigmoid**: `1 / (1 + e^(-x))` — output in (0,1), for probabilities
- **Tanh**: `(e^x - e^(-x)) / (e^x + e^(-x))` — output in (-1,1)

### Backpropagation

The algorithm that makes learning possible:
1. Forward pass: compute predictions
2. Compute loss
3. Backward pass: compute gradients using the **chain rule**
4. Update weights: `w = w - lr * gradient`

The chain rule decomposes complex derivatives:
```
∂L/∂w = ∂L/∂output · ∂output/∂z · ∂z/∂w
```

### Regularization

Techniques to prevent overfitting:
- **L2 regularization** (weight decay): penalizes large weights
- **Dropout**: randomly zeroes neurons during training
- **Early stopping**: stop training when validation loss increases

---

## 📂 Scripts in This Module

| Script | Description |
|--------|-------------|
| `01_perceptron.py` | Single perceptron from scratch |
| `02_neural_network_numpy.py` | Full NN from scratch with NumPy |
| `03_backpropagation.py` | Backprop derivation and implementation |
| `04_overfitting_regularization.py` | Dropout, L2, early stopping demos |
| `05_neural_network_pytorch.py` | Same NN rewritten in PyTorch |
| `utils.py` | Shared helpers |

---

## ➡️ Next Module

[Module 4 — Computer Vision](../module_04_computer_vision/README.md)
