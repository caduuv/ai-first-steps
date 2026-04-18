# Module 6 — Exercises

## Conceptual Questions

### Exercise 6.1: Conditional vs Unconditional
1. Why is conditional generation more useful for medical imaging than unconditional?
2. How does the discriminator's task change when it receives the label? Why is this important?
3. What would happen if you train a cGAN but only condition G (not D) on the label?

### Exercise 6.2: Diffusion vs GAN
Compare diffusion models and GANs across these dimensions:
1. Training stability
2. Generation quality
3. Generation speed
4. Mode coverage (diversity)
5. Controllability (conditional generation)

### Exercise 6.3: FID Score
1. Why can't we just use pixel-level MSE between real and generated images as a quality metric?
2. What does FID actually measure? Why is it considered the standard GAN metric?
3. What are the limitations of FID?

---

## Implementation Exercises

### Exercise 6.4: Improved cGAN
Improve the cGAN from this module:
1. Replace FC layers with convolutional layers (DCGAN-style)
2. Add spectral normalization to both G and D
3. Implement WGAN-GP loss instead of BCE
4. Compare generated quality with the original FC-based cGAN

### Exercise 6.5: Conditional VAE with CNN
Convert the FC-based cVAE to a convolutional architecture:
1. Encoder: Conv2d layers for feature extraction
2. Decoder: ConvTranspose2d layers for upsampling
3. Inject the label at the bottleneck (concatenate with z)
4. Compare reconstruction quality and generation diversity

### Exercise 6.6: Multi-Attribute Conditioning
Extend the cGAN to accept multiple conditions:
1. Digit class (0-9)
2. Thickness (thin/normal/bold) — encode as continuous value
3. Slant (left/straight/right) — encode as continuous value
4. Train on MNIST and see if the model can disentangle these attributes

### Exercise 6.7: Classifier-Guided Generation
Implement a simple classifier-guided approach:
1. Train a digit classifier on MNIST
2. During GAN generation, use the classifier's gradients to push generated images toward a target class
3. Compare with the cGAN approach

---

## Challenge Exercise

### Exercise 6.8: Full Diffusion Model (DDPM)
Implement a complete DDPM (Denoising Diffusion Probabilistic Model):
1. Define the full forward process with β schedule
2. Implement a U-Net with time embeddings and residual connections
3. Train the denoising model on MNIST
4. Implement the reverse sampling process (T=1000 steps)
5. Add class conditioning using classifier-free guidance
6. Generate samples and compute FID

*This is a significant project — refer to the "Denoising Diffusion Probabilistic Models" paper by Ho et al. (2020).*
