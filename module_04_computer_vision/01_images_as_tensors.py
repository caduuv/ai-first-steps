"""
Module 4 — Script 01: Images as Tensors
=========================================

Images are just numbers! This script shows how to load, manipulate,
and understand images as multi-dimensional arrays (tensors).

Topics:
  - Loading images with PIL/Torchvision
  - Color channels (RGB, Grayscale)
  - Pixel value ranges and normalization
  - Basic image operations
  - Visualizing image tensors

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Loading MNIST Digits

# %%
print("=== Images as Tensors: MNIST ===")

# Download MNIST dataset
transform = transforms.ToTensor()
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Look at one image
image, label = mnist[0]
print(f"Image type: {type(image)}")
print(f"Image shape: {image.shape}  → (Channels, Height, Width)")
print(f"Label: {label}")
print(f"Pixel value range: [{image.min():.4f}, {image.max():.4f}]")
print(f"Data type: {image.dtype}")

# Visualize
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    ax = axes[i // 5, i % 5]
    img, lbl = mnist[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {lbl}', fontsize=11)
    ax.axis('off')

plt.suptitle('MNIST Digits — Images are Just Numbers!', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('module_04_computer_vision/mnist_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 MNIST samples saved")

# %% [markdown]
# ## 2. Understanding Pixel Values

# %%
print("\n=== Pixel Values ===")

image, label = mnist[0]
pixel_array = image.squeeze().numpy()  # Remove channel dim → (28, 28)

print(f"Image shape: {pixel_array.shape}")
print(f"A 5×5 patch from the center:")
center = pixel_array[10:15, 10:15]
print(np.round(center, 2))
print(f"\n0.0 = black (background), 1.0 = white (digit)")

# Visualize pixel values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(pixel_array, cmap='gray')
ax1.set_title(f'Digit: {label}', fontsize=13, fontweight='bold')
ax1.axis('off')

# Heatmap of actual values
im = ax2.imshow(pixel_array, cmap='hot', interpolation='nearest')
ax2.set_title('Pixel Intensity Values', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax2, label='Intensity')

plt.tight_layout()
plt.savefig('module_04_computer_vision/pixel_values.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Pixel values visualization saved")

# %% [markdown]
# ## 3. Color Images (CIFAR-10)

# %%
print("\n=== Color Images: CIFAR-10 ===")

cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

image_rgb, label = cifar[0]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Shape: {image_rgb.shape}  → (3 channels, 32×32 pixels)")
print(f"Label: {label} ({class_names[label]})")
print(f"R channel range: [{image_rgb[0].min():.3f}, {image_rgb[0].max():.3f}]")
print(f"G channel range: [{image_rgb[1].min():.3f}, {image_rgb[1].max():.3f}]")
print(f"B channel range: [{image_rgb[2].min():.3f}, {image_rgb[2].max():.3f}]")

# Visualize channels
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

# Full RGB
axes[0].imshow(image_rgb.permute(1, 2, 0))  # CHW → HWC for matplotlib
axes[0].set_title(f'RGB ({class_names[label]})', fontweight='bold')
axes[0].axis('off')

# Individual channels
channel_names = ['Red', 'Green', 'Blue']
cmaps = ['Reds', 'Greens', 'Blues']
for i in range(3):
    axes[i+1].imshow(image_rgb[i], cmap=cmaps[i])
    axes[i+1].set_title(f'{channel_names[i]} Channel', fontweight='bold')
    axes[i+1].axis('off')

plt.suptitle('Color Image = 3 Channels', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('module_04_computer_vision/color_channels.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Color channels visualization saved")

# %% [markdown]
# ## 4. Image Transformations

# %%
print("\n=== Image Transformations ===")

img, label = cifar[42]

# Various transformations
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Original
axes[0, 0].imshow(img.permute(1, 2, 0))
axes[0, 0].set_title('Original', fontweight='bold')

# Horizontal flip
flipped = torch.flip(img, [2])
axes[0, 1].imshow(flipped.permute(1, 2, 0))
axes[0, 1].set_title('Horizontal Flip', fontweight='bold')

# Vertical flip
vflipped = torch.flip(img, [1])
axes[0, 2].imshow(vflipped.permute(1, 2, 0))
axes[0, 2].set_title('Vertical Flip', fontweight='bold')

# Rotate 90°
rotated = torch.rot90(img, 1, [1, 2])
axes[0, 3].imshow(rotated.permute(1, 2, 0))
axes[0, 3].set_title('Rotate 90°', fontweight='bold')

# Brightness increase
bright = torch.clamp(img * 1.5, 0, 1)
axes[1, 0].imshow(bright.permute(1, 2, 0))
axes[1, 0].set_title('Brighter', fontweight='bold')

# Darkness
dark = img * 0.5
axes[1, 1].imshow(dark.permute(1, 2, 0))
axes[1, 1].set_title('Darker', fontweight='bold')

# Grayscale
gray = img.mean(dim=0, keepdim=True).repeat(3, 1, 1)
axes[1, 2].imshow(gray.permute(1, 2, 0))
axes[1, 2].set_title('Grayscale', fontweight='bold')

# Inverted
inverted = 1 - img
axes[1, 3].imshow(inverted.permute(1, 2, 0))
axes[1, 3].set_title('Inverted', fontweight='bold')

for ax in axes.flatten():
    ax.axis('off')

plt.suptitle('Image Transformations', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_04_computer_vision/transformations.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Image transformations saved")

# %% [markdown]
# ## 5. Batch of Images — What the Network Sees

# %%
print("\n=== Batch Dimensions ===")

# In PyTorch, images come in batches: (Batch, Channels, Height, Width)
batch_loader = torch.utils.data.DataLoader(cifar, batch_size=16, shuffle=True)
batch_images, batch_labels = next(iter(batch_loader))

print(f"Batch shape: {batch_images.shape}")
print(f"  Batch size: {batch_images.shape[0]}")
print(f"  Channels:   {batch_images.shape[1]}")
print(f"  Height:     {batch_images.shape[2]}")
print(f"  Width:      {batch_images.shape[3]}")
print(f"Labels: {[class_names[l] for l in batch_labels.tolist()]}")

# Normalization (standard for ImageNet-pretrained models)
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
normalized = (batch_images - mean[None, :, None, None]) / std[None, :, None, None]
print(f"\nNormalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

print("\n✅ Module 4, Script 01 complete!")
