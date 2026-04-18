# Training Pipeline Documentation

## Overview

The training pipeline is designed for reproducibility and experimentation.
All hyperparameters are configured via YAML files.

## Training Loop Pseudocode

### cGAN Training

```python
for epoch in range(n_epochs):
    for real_images, cell_types, features in dataloader:
        # 1. Encode conditions
        condition = encode_condition(cell_types, features)
        
        # 2. Train Discriminator
        real_score = D(real_images, condition)
        
        z = sample_noise(batch_size, latent_dim)
        fake_images = G(z, condition)
        fake_score = D(fake_images.detach(), condition)
        
        d_loss = wasserstein_loss(real_score, fake_score)
        d_loss += gradient_penalty(D, real_images, fake_images, condition)
        update(D, d_loss)
        
        # 3. Train Generator (every k steps)
        if step % k == 0:
            fake_images = G(z, condition)
            fake_score = D(fake_images, condition)
            g_loss = -fake_score.mean()
            update(G, g_loss)
        
        # 4. Log metrics
        log(d_loss, g_loss, d_real_score, d_fake_score)
    
    # 5. Generate samples and compute FID
    if epoch % eval_freq == 0:
        generate_samples(G, save_dir)
        compute_fid(real_data, generated_data)
```

### Diffusion Model Training

```python
for epoch in range(n_epochs):
    for images, cell_types, features in dataloader:
        # 1. Sample random timestep
        t = uniform_sample(0, T)
        
        # 2. Add noise to images
        noise = randn_like(images)
        noisy_images = sqrt(alpha_bar[t]) * images + sqrt(1 - alpha_bar[t]) * noise
        
        # 3. Predict noise
        condition = encode_condition(cell_types, features)
        
        # Classifier-free guidance: drop condition 10% of time
        if random() < 0.1:
            condition = null_condition
        
        noise_pred = denoiser(noisy_images, t, condition)
        
        # 4. Simple MSE loss
        loss = mse(noise_pred, noise)
        update(denoiser, loss)
```

## Training Schedule

1. **Warm-up** (epochs 1-5): Lower learning rate, verify loss decreases
2. **Main training** (epochs 5-150): Full learning rate
3. **Fine-tuning** (epochs 150-200): Reduced learning rate (0.1×)

## Checkpointing

- Save model every 10 epochs
- Save best model based on FID score
- Save optimizer state for resume capability

## Monitoring

Metrics to track:
- Generator loss / discriminator loss (or denoiser loss)
- FID score (every 10 epochs)
- Per-class sample quality (visual inspection)
- Gradient norms (detect instability)
- Learning rate schedule
