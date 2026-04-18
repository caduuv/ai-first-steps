# Module 5 — Exercises

## Conceptual Questions

### Exercise 5.1: Autoencoder vs VAE
1. Why can't a standard autoencoder generate new, realistic images by sampling random latent codes?
2. How does the KL divergence term in the VAE loss fix this problem?
3. What happens if you set β=0 in the VAE loss? What about β→∞?

### Exercise 5.2: GAN Theory
1. Explain the minimax game in your own words. What is each player trying to do?
2. Why does the original GAN loss (min log(1-D(G(z)))) suffer from vanishing gradients?
3. What does it mean when D(G(z)) → 0.5 for all fake inputs?

### Exercise 5.3: Mode Collapse
1. Describe 3 signs that your GAN is experiencing mode collapse
2. How does label smoothing help prevent mode collapse?
3. Why does WGAN-GP tend to be more stable than standard GAN training?

---

## Implementation Exercises

### Exercise 5.4: Convolutional Autoencoder
Replace the fully-connected autoencoder with a convolutional one:
1. Encoder: Conv2d layers with stride=2 for downsampling
2. Decoder: ConvTranspose2d layers for upsampling
3. Compare reconstruction quality with the FC autoencoder
4. How many fewer parameters does the CNN version have?

### Exercise 5.5: β-VAE Exploration
1. Train multiple VAEs with different β values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
2. For each, plot: reconstruction quality, latent space structure, generated samples
3. Find the β that gives the best trade-off between reconstruction and generation
4. What pattern do you notice as β increases?

### Exercise 5.6: DCGAN
Implement a Deep Convolutional GAN (DCGAN):
1. Generator uses ConvTranspose2d layers
2. Discriminator uses Conv2d layers with stride
3. Follow the DCGAN guidelines (no pooling, BatchNorm, etc.)
4. Compare generated image quality with the FC-based GAN

### Exercise 5.7: GAN Training Analysis
Train a GAN on Fashion-MNIST instead of MNIST:
1. Log and plot: G loss, D loss, D(real), D(fake) every epoch
2. Save generated samples every 5 epochs to create a "training progression"
3. Try to identify mode collapse if it occurs
4. Implement one fix (label smoothing, spectral norm, or lr scheduling) and compare

---

## Challenge Exercise

### Exercise 5.8: VAE-GAN Hybrid
Combine VAE and GAN ideas:
1. Use a VAE encoder-decoder architecture
2. Add a discriminator that judges the decoder's output
3. Train with: VAE loss (reconstruction + KL) + GAN loss (adversarial)
4. Compare with pure VAE and pure GAN results
5. Does the hybrid produce sharper images than VAE alone?
