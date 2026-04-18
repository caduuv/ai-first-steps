# Module 1 — Mathematical & Statistical Foundations

## 🎯 Learning Objectives

By the end of this module, you will:

1. Understand **vectors and matrices** as the language of machine learning
2. Work with **probability distributions** and apply Bayes' theorem
3. Implement **loss functions** from scratch and understand their role
4. Compute **gradients** and implement gradient descent
5. Build a **linear regression** model from absolute scratch — no frameworks

---

## 📖 Theory Overview

### Why Math Matters for AI

Every machine learning algorithm is fundamentally a mathematical optimization problem. When a neural network "learns", it is:

1. **Representing** data as vectors and matrices (linear algebra)
2. **Modeling** uncertainty in predictions (probability & statistics)
3. **Measuring** how wrong its predictions are (loss functions)
4. **Improving** by following the steepest path downhill (gradients & optimization)

Understanding these foundations will make everything else in this course click.

---

### Linear Algebra Essentials

**Vectors** represent data points. An image pixel, a patient measurement, a word embedding — all vectors.

**Matrices** represent transformations. A neural network layer is literally a matrix multiplication followed by a nonlinear function.

Key operations:
- **Dot product**: Measures similarity between vectors
- **Matrix multiplication**: Applies transformations
- **Transpose**: Flips rows and columns
- **Determinant & Inverse**: Understanding when systems have solutions

---

### Probability & Statistics

Machine learning makes predictions under **uncertainty**. Key concepts:

- **Probability distributions**: How data is spread (Gaussian, Bernoulli, Uniform)
- **Bayes' theorem**: Updating beliefs with new evidence
- **Expected value & variance**: Summarizing distributions
- **Sampling**: Generating data from distributions (crucial for generative models!)

---

### Loss Functions

A loss function answers: *"How wrong is my model?"*

- **Mean Squared Error (MSE)**: For regression — penalizes large errors quadratically
- **Cross-Entropy**: For classification — measures divergence between predicted and actual distributions
- **Custom losses**: In generative models, we'll design losses that encourage realistic image generation

---

### Gradients & Optimization

The **gradient** of a function points in the direction of steepest increase. To minimize a loss:

1. Compute the gradient of the loss with respect to model parameters
2. Take a small step in the **opposite** direction (gradient descent)
3. Repeat until convergence

This is the beating heart of how neural networks learn.

---

## 📂 Scripts in This Module

| Script | Description |
|--------|-------------|
| `01_linear_algebra.py` | Vectors, matrices, transformations with pure Python & NumPy |
| `02_probability_statistics.py` | Distributions, Bayes' theorem, sampling |
| `03_loss_functions.py` | MSE, cross-entropy, implemented from scratch |
| `04_gradients_optimization.py` | Numerical gradients, gradient descent |
| `05_linear_regression.py` | Full linear regression from scratch |

---

## 🔗 Prerequisites

- Python 3.10+
- NumPy (installed via `requirements.txt`)

---

## ➡️ Next Module

When you're comfortable with these foundations, move to [Module 2 — Python for Machine Learning](../module_02_python_ml/README.md).
