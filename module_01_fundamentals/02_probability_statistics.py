"""
Module 1 — Script 02: Probability & Statistics
================================================

This script covers the probability and statistics concepts essential
for understanding machine learning. Every model deals with uncertainty,
and these tools help us quantify and work with it.

Topics:
  - Probability basics (events, conditional probability)
  - Common distributions (Uniform, Gaussian, Bernoulli)
  - Bayes' theorem
  - Expected value, variance, standard deviation
  - Sampling from distributions
  - Central Limit Theorem

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Probability Basics

# %%
# Probability measures how likely an event is to occur.
# P(event) ranges from 0 (impossible) to 1 (certain).

# Example: Rolling a fair die
print("=== Probability Basics ===")
total_outcomes = 6

# P(rolling a 3)
p_three = 1 / total_outcomes
print(f"P(rolling a 3) = {p_three:.4f}")

# P(rolling even) = P(2) + P(4) + P(6)
p_even = 3 / total_outcomes
print(f"P(rolling even) = {p_even:.4f}")

# P(rolling > 4) = P(5) + P(6)
p_gt_4 = 2 / total_outcomes
print(f"P(rolling > 4) = {p_gt_4:.4f}")

# Complement: P(not A) = 1 - P(A)
print(f"P(rolling <= 4) = {1 - p_gt_4:.4f}")

# %% [markdown]
# ## 2. Conditional Probability

# %%
# P(A|B) = P(A and B) / P(B)
# "Probability of A given that B has occurred"

# Example: Medical test
# P(Disease) = 0.01 (1% prevalence)
# P(Positive | Disease) = 0.95 (sensitivity)
# P(Positive | No Disease) = 0.05 (false positive rate)

print("\n=== Conditional Probability — Medical Test ===")
p_disease = 0.01
p_positive_given_disease = 0.95  # Sensitivity (true positive rate)
p_positive_given_no_disease = 0.05  # False positive rate

# What's P(Positive)?
# P(Positive) = P(Pos|Disease)·P(Disease) + P(Pos|No Disease)·P(No Disease)
p_positive = (p_positive_given_disease * p_disease +
              p_positive_given_no_disease * (1 - p_disease))
print(f"P(Positive test) = {p_positive:.4f}")

# %% [markdown]
# ## 3. Bayes' Theorem

# %%
# Bayes' theorem lets us "flip" conditional probabilities:
# P(A|B) = P(B|A) · P(A) / P(B)
#
# This is foundational in ML:
#   - Bayesian inference: updating beliefs with data
#   - Naive Bayes classifiers
#   - Understanding posterior distributions

# Using our medical test example:
# P(Disease | Positive) = P(Positive | Disease) · P(Disease) / P(Positive)

p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print("\n=== Bayes' Theorem ===")
print(f"P(Disease | Positive test) = {p_disease_given_positive:.4f}")
print(f"  → Only {p_disease_given_positive*100:.1f}% chance of disease despite positive test!")
print(f"  → This is because the disease is rare (1% prevalence)")
print(f"  → Most positive results are false positives")

# Let's also compute P(No Disease | Positive)
p_no_disease_given_positive = 1 - p_disease_given_positive
print(f"P(No Disease | Positive test) = {p_no_disease_given_positive:.4f}")

# %% [markdown]
# ## 4. Common Probability Distributions

# %%
print("\n=== Common Distributions ===")
np.random.seed(42)

# --- Uniform Distribution ---
# All outcomes equally likely. Range [a, b].
uniform_samples = np.random.uniform(low=0, high=1, size=10000)
print(f"Uniform [0,1]: mean={uniform_samples.mean():.4f}, std={uniform_samples.std():.4f}")

# --- Gaussian (Normal) Distribution ---
# Bell curve. Defined by mean (μ) and standard deviation (σ).
# This is THE most important distribution in ML:
#   - Weight initialization
#   - Noise in generative models
#   - Many natural phenomena
gaussian_samples = np.random.normal(loc=0, scale=1, size=10000)  # Standard normal
print(f"Gaussian N(0,1): mean={gaussian_samples.mean():.4f}, std={gaussian_samples.std():.4f}")

# --- Bernoulli Distribution ---
# Binary outcome: success (1) or failure (0) with probability p.
# Used for: binary classification, dropout, coin flips.
p_success = 0.7
bernoulli_samples = np.random.binomial(n=1, p=p_success, size=10000)
print(f"Bernoulli (p=0.7): mean={bernoulli_samples.mean():.4f} (should be ≈ 0.7)")

# %% [markdown]
# ## 5. Visualizing Distributions

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(uniform_samples, bins=50, color='#4A90D9', edgecolor='white', alpha=0.8)
axes[0].set_title('Uniform Distribution [0, 1]', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

axes[1].hist(gaussian_samples, bins=50, color='#E74C3C', edgecolor='white', alpha=0.8)
axes[1].set_title('Gaussian Distribution N(0, 1)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

axes[2].hist(bernoulli_samples, bins=[-0.5, 0.5, 1.5], color='#2ECC71', edgecolor='white', alpha=0.8, rwidth=0.5)
axes[2].set_title('Bernoulli Distribution (p=0.7)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Frequency')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Failure (0)', 'Success (1)'])

plt.tight_layout()
plt.savefig('module_01_fundamentals/distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n📊 Distribution plots saved to module_01_fundamentals/distributions.png")

# %% [markdown]
# ## 6. Expected Value and Variance

# %%
# Expected value E[X]: the "average" outcome, weighted by probability.
# Variance Var[X]: how spread out the distribution is.
# Standard deviation σ = sqrt(Var[X])

print("\n=== Expected Value & Variance ===")

# For discrete distributions, compute manually:
# E[X] = Σ x_i · P(x_i)
# Example: Fair die
die_values = np.array([1, 2, 3, 4, 5, 6])
die_probs = np.array([1/6] * 6)

expected_value = np.sum(die_values * die_probs)
variance = np.sum((die_values - expected_value)**2 * die_probs)
std_dev = np.sqrt(variance)

print(f"Fair die:")
print(f"  E[X] = {expected_value:.4f}")     # Should be 3.5
print(f"  Var[X] = {variance:.4f}")          # Should be ~2.917
print(f"  σ = {std_dev:.4f}")

# For continuous: use sample statistics
print(f"\nGaussian samples (N(0,1)):")
print(f"  Sample mean = {gaussian_samples.mean():.4f}")
print(f"  Sample variance = {gaussian_samples.var():.4f}")
print(f"  Sample std = {gaussian_samples.std():.4f}")

# %% [markdown]
# ## 7. Sampling — Foundation of Generative Models

# %%
# Generating data from distributions is the CORE idea behind generative AI.
# A generative model learns a distribution and then samples from it.

print("\n=== Sampling from Distributions ===")

# Sampling from a multivariate Gaussian (2D)
# This is exactly what a VAE's latent space looks like!
mean = np.array([2.0, 3.0])
covariance = np.array([[1.0, 0.5],    # Correlated variables
                         [0.5, 1.0]])

samples_2d = np.random.multivariate_normal(mean, covariance, size=500)

print(f"2D Gaussian samples shape: {samples_2d.shape}")
print(f"Sample mean: {samples_2d.mean(axis=0).round(3)}")
print(f"Sample covariance:\n{np.cov(samples_2d.T).round(3)}")

# Visualize
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.4, s=10, color='#8E44AD')
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('Samples from 2D Gaussian\n(Preview of Latent Space!)', fontsize=13, fontweight='bold')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module_01_fundamentals/2d_gaussian_sampling.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 2D Gaussian plot saved to module_01_fundamentals/2d_gaussian_sampling.png")

# %% [markdown]
# ## 8. Central Limit Theorem

# %%
# The CLT states: the mean of many independent random variables
# tends toward a Gaussian distribution, regardless of the original
# distribution. This explains why the Gaussian is so prevalent in ML.

print("\n=== Central Limit Theorem ===")

# Take means of uniform samples (clearly not Gaussian)
sample_sizes = [1, 5, 30, 100]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

for idx, n in enumerate(sample_sizes):
    # Take 10000 means, each from n uniform samples
    means = [np.random.uniform(0, 1, n).mean() for _ in range(10000)]
    axes[idx].hist(means, bins=50, color='#3498DB', edgecolor='white', alpha=0.8)
    axes[idx].set_title(f'Mean of {n} samples', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Value')
    if idx == 0:
        axes[idx].set_ylabel('Frequency')

plt.suptitle('Central Limit Theorem: Uniform → Gaussian', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('module_01_fundamentals/central_limit_theorem.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 CLT plot saved to module_01_fundamentals/central_limit_theorem.png")

print("\n✅ Module 1, Script 02 complete!")
