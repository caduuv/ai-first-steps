"""
Module 5 — Script 03: GAN Theory
==================================

Before building a GAN, we need to understand the theory.
This script covers the mathematical framework and intuition.

Topics:
  - Minimax game formulation
  - Generator and discriminator objectives
  - Nash equilibrium
  - Loss functions and their properties
  - Why GANs are hard to train

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. The GAN Game

# %%
print("=== GAN Theory ===")
print("""
GAN = Generative Adversarial Network

Two players in a minimax game:

  GENERATOR (G):  Takes random noise z → produces fake data G(z)
                  Goal: Fool the discriminator
  
  DISCRIMINATOR (D): Takes data x → outputs probability it's real
                     Goal: Correctly classify real vs fake

The game:
  min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
         ↑                          ↑
    D wants to     D wants to reject fakes
    accept reals   G wants D to accept fakes
""")

# %% [markdown]
# ## 2. Understanding the Losses

# %%
# Discriminator loss: Binary cross-entropy
# For real data: maximize log(D(x))         → D should output 1 for real
# For fake data: maximize log(1 - D(G(z)))  → D should output 0 for fake

# Generator loss: minimize log(1 - D(G(z)))
# But in practice: maximize log(D(G(z)))    → non-saturating loss

# Let's visualize the loss landscape

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# D(x) range
d = np.linspace(0.01, 0.99, 200)

# Discriminator loss for REAL data
axes[0].plot(d, np.log(d), color='#2ECC71', linewidth=2.5)
axes[0].set_xlabel('D(x) — Discriminator output for REAL data', fontsize=10)
axes[0].set_ylabel('log(D(x))', fontsize=10)
axes[0].set_title('D Loss (Real): Maximize log(D(x))', fontweight='bold')
axes[0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
axes[0].annotate('D wants D(x)→1', xy=(0.8, -0.5), fontsize=10, color='#2ECC71')
axes[0].grid(True, alpha=0.3)

# Discriminator loss for FAKE data
axes[1].plot(d, np.log(1 - d), color='#E74C3C', linewidth=2.5)
axes[1].set_xlabel('D(G(z)) — Discriminator output for FAKE data', fontsize=10)
axes[1].set_ylabel('log(1 - D(G(z)))', fontsize=10)
axes[1].set_title('D Loss (Fake): Maximize log(1-D(G(z)))', fontweight='bold')
axes[1].axvline(x=0.0, color='gray', linestyle='--', alpha=0.5)
axes[1].annotate('D wants D(G(z))→0', xy=(0.1, -0.5), fontsize=10, color='#E74C3C')
axes[1].grid(True, alpha=0.3)

# Generator loss comparison
g_saturating = np.log(1 - d)       # Original: min log(1 - D(G(z)))
g_non_saturating = -np.log(d)       # Practical: max log(D(G(z)))

axes[2].plot(d, -g_saturating, color='#F39C12', linewidth=2.5, label='Saturating', linestyle='--')
axes[2].plot(d, g_non_saturating, color='#9B59B6', linewidth=2.5, label='Non-saturating')
axes[2].set_xlabel('D(G(z))', fontsize=10)
axes[2].set_ylabel('Generator Loss', fontsize=10)
axes[2].set_title('Generator Loss Comparison', fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].annotate('Non-saturating has\nstronger gradients\nwhen D(G(z))≈0',
                  xy=(0.1, 3), fontsize=9, color='#9B59B6')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(0, 5)

plt.tight_layout()
plt.savefig('module_05_generative_models/gan_losses.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 GAN loss functions saved")

# %% [markdown]
# ## 3. Nash Equilibrium

# %%
print("""
=== Nash Equilibrium ===

Training goal: Neither G nor D can improve by changing strategy alone.

At equilibrium:
  - G generates data indistinguishable from real data
  - D outputs 0.5 for everything (can't tell real from fake)
  - P(generated) = P(real)  — the generator learned the data distribution!

In practice, reaching Nash equilibrium is HARD:
  1. Training oscillates (D gets better, G catches up, D adapts...)
  2. Mode collapse: G finds one output that fools D and sticks with it
  3. Vanishing gradients: If D is too good, G gets no useful signal
""")

# %% [markdown]
# ## 4. Training Dynamics Simulation

# %%
# Simple 1D simulation of GAN dynamics
np.random.seed(42)

# True distribution: N(5, 1)
true_mean = 5.0
true_std = 1.0

# Generator starts with wrong distribution: N(0, 1)
g_mean = 0.0
g_std = 1.0
lr = 0.1

history = {'g_mean': [g_mean], 'g_std': [g_std]}

for step in range(50):
    # Sample from true and generated distributions
    real_samples = np.random.normal(true_mean, true_std, 100)
    fake_samples = np.random.normal(g_mean, g_std, 100)
    
    # Simple gradient: push generator mean toward true mean
    g_mean += lr * (np.mean(real_samples) - np.mean(fake_samples))
    g_std += lr * 0.5 * (np.std(real_samples) - np.std(fake_samples))
    g_std = max(g_std, 0.1)
    
    history['g_mean'].append(g_mean)
    history['g_std'].append(g_std)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Distribution evolution
x = np.linspace(-3, 10, 200)
true_pdf = np.exp(-(x - true_mean)**2 / (2 * true_std**2)) / (true_std * np.sqrt(2 * np.pi))
ax1.plot(x, true_pdf, 'k-', linewidth=2, label='Real Distribution')

steps_to_show = [0, 5, 15, 49]
colors = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']
for step, color in zip(steps_to_show, colors):
    m = history['g_mean'][step]
    s = history['g_std'][step]
    gen_pdf = np.exp(-(x - m)**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi))
    ax1.plot(x, gen_pdf, '--', color=color, linewidth=1.5, label=f'G at step {step}')

ax1.set_xlabel('Value', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Generator Distribution Evolution', fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Mean convergence
ax2.plot(history['g_mean'], color='#3498DB', linewidth=2, label='G mean')
ax2.axhline(y=true_mean, color='red', linestyle='--', label=f'True mean ({true_mean})')
ax2.set_xlabel('Training Step', fontsize=11)
ax2.set_ylabel('Generator Mean', fontsize=11)
ax2.set_title('Generator Mean Convergence', fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module_05_generative_models/gan_dynamics.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 GAN dynamics simulation saved")

# %% [markdown]
# ## 5. Common GAN Problems

# %%
print("""
=== Common GAN Training Problems ===

1. MODE COLLAPSE
   - Generator produces only a few types of outputs
   - Example: Only generates the digit "1" regardless of input noise
   - Cause: G finds a "safe" output that always fools D
   - Solutions: Feature matching, minibatch discrimination, Wasserstein loss

2. TRAINING INSTABILITY
   - Loss oscillates wildly, never converges
   - G and D "chase" each other without reaching equilibrium
   - Solutions: Spectral normalization, gradient penalty, careful lr tuning

3. VANISHING GRADIENTS
   - When D is too good, D(G(z)) ≈ 0 for all fake data
   - log(1 - D(G(z))) ≈ log(1) ≈ 0 → no gradient for G!
   - Solutions: Non-saturating loss, Wasserstein loss, label smoothing

4. EVALUATION DIFFICULTY
   - Unlike classification, there's no single metric for "good generation"
   - Metrics: FID (Fréchet Inception Distance), IS (Inception Score)
   - Visual inspection is still important!
""")

print("\n✅ Module 5, Script 03 complete!")
print("Now that you understand the theory, let's build a real GAN! →")
