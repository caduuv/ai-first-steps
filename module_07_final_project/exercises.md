# Module 7 — Final Project Exercises

## Planning Exercises

### Exercise 7.1: Architecture Selection
Based on the documentation in this module:
1. Compare cGAN vs cVAE vs Diffusion for cervical cytology generation
2. List 3 advantages and 3 disadvantages of each approach
3. Which would you choose for a small dataset (~1000 images)? Why?
4. Which would you choose if you had unlimited compute? Why?

### Exercise 7.2: Conditioning Design
Design the conditioning interface for your model:
1. What cell attributes should be controllable?
2. How would you encode categorical features (cell type)?
3. How would you encode continuous features (N/C ratio)?
4. Should you use one-hot, embeddings, or continuous conditioning?
5. Draw a diagram of how conditions flow through the generator

### Exercise 7.3: Data Strategy
Plan your data pipeline:
1. Which dataset(s) would you use and why?
2. Design the augmentation pipeline for cytology images
3. How would you handle class imbalance?
4. What preprocessing steps are needed (stain normalization, etc.)?
5. How would you create a simulated dataset for development?

---

## Implementation Exercises

### Exercise 7.4: Simulated Data Generator
Implement a simulated cervical cytology dataset:
1. Generate circular "cells" with varying properties
2. Add nucleus with controlled N/C ratio
3. Apply simulated staining (color variation)
4. Create 4 classes with different statistical properties
5. Save as a PyTorch Dataset with labels and feature metadata

### Exercise 7.5: Baseline cGAN
Implement the full cGAN pipeline from this module:
1. Build the conditional generator and discriminator
2. Implement the training loop with WGAN-GP loss
3. Train on your simulated dataset
4. Generate per-class samples and evaluate visually
5. Compute simplified FID scores

### Exercise 7.6: Evaluation Pipeline
Build a comprehensive evaluation pipeline:
1. Generate 500 samples per class
2. Compute FID-like metric using simple features
3. Train a classifier on real data, then on real+synthetic
4. Compare accuracy improvement
5. Create a visual quality report

---

## Research Exercises

### Exercise 7.7: Literature Review
Read and summarize these papers (1 paragraph each):
1. Original GAN paper (Goodfellow et al., 2014)
2. Conditional GAN paper (Mirza & Osindero, 2014)
3. DDPM paper (Ho et al., 2020)
4. One paper on synthetic medical image generation (your choice)

### Exercise 7.8: Ethics Discussion
Write a 1-page response addressing:
1. Should synthetic medical images be clearly watermarked?
2. What safeguards should exist before using synthetic data in clinical AI?
3. How might synthetic cytology images affect pathologist training?
4. What role should regulatory bodies play in synthetic medical data?

---

## Challenge Exercise

### Exercise 7.9: Full Implementation
Implement the complete final project:
1. Set up the full data pipeline (real or simulated data)
2. Implement a conditional generative model (your choice of architecture)
3. Train for sufficient epochs to achieve reasonable quality
4. Evaluate with all metrics described in the evaluation README
5. Write a project report including:
   - Architecture decisions and justification
   - Training details and hyperparameter choices
   - Quantitative and qualitative evaluation results
   - Ethical considerations specific to your implementation
   - Future improvements you would make

This exercise ties together EVERYTHING from the entire course!
