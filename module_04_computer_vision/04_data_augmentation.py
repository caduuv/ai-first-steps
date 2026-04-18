"""
Module 4 — Script 04: Data Augmentation
=========================================

Data augmentation artificially expands the training set by applying
random transformations. Essential for medical imaging where data is scarce.

Topics:
  - Why augmentation works
  - Common augmentation transforms
  - PyTorch transforms pipeline
  - Comparing performance with/without augmentation
  - Medical imaging specific augmentations

Run interactively with VS Code cells (# %%) or as a script.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## 1. Why Data Augmentation?

# %%
print("=== Data Augmentation ===")
print("""
Why augment data?
  1. More data → better generalization
  2. Teaches invariances (a cat rotated is still a cat)
  3. Reduces overfitting
  4. Especially critical in medical imaging with limited samples

Key principle: Apply transformations that preserve the class label.
  ✓ Flipping a cell image → still the same cell type
  ✓ Rotating → still the same cell type
  ✗ Cropping too aggressively → might lose diagnostic features
""")

# %% [markdown]
# ## 2. Common Augmentation Transforms

# %%
# Load sample image
cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=transforms.ToTensor())
sample_img, label = cifar[42]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define various augmentations
augmentations = {
    'Original': transforms.Compose([]),
    'H-Flip': transforms.RandomHorizontalFlip(p=1.0),
    'V-Flip': transforms.RandomVerticalFlip(p=1.0),
    'Rotation': transforms.RandomRotation(30),
    'Color Jitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    'Random Crop': transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
    'Grayscale': transforms.RandomGrayscale(p=1.0),
    'Affine': transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
}

# Show each augmentation applied multiple times
fig, axes = plt.subplots(len(augmentations), 5, figsize=(12, len(augmentations) * 2.2))

pil_img = transforms.ToPILImage()(sample_img)

for row, (name, transform) in enumerate(augmentations.items()):
    for col in range(5):
        if name == 'Original':
            aug_img = pil_img
        else:
            aug_img = transform(pil_img)
        axes[row, col].imshow(aug_img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(name, fontsize=10, fontweight='bold', rotation=0,
                                       labelpad=70, va='center')

plt.suptitle(f'Data Augmentation Gallery ({class_names[label]})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_04_computer_vision/augmentation_gallery.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Augmentation gallery saved")

# %% [markdown]
# ## 3. Building an Augmentation Pipeline

# %%
print("\n=== Augmentation Pipeline ===")

# Training transform: with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# Test transform: NO augmentation (just normalize)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

print("Training pipeline:")
for t in train_transform.transforms:
    print(f"  → {t}")

print("\nTest pipeline:")
for t in test_transform.transforms:
    print(f"  → {t}")

print("\n⚠️ IMPORTANT: Never augment test/validation data!")
print("   Augmentation is a training-time regularization technique.")

# %% [markdown]
# ## 4. Medical Imaging Specific Augmentations

# %%
print("\n=== Medical Imaging Augmentations ===")

# For cytology images, appropriate augmentations include:
medical_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # Cells can appear in any orientation
    transforms.RandomVerticalFlip(),              # Same reason
    transforms.RandomRotation(360),               # Full rotation invariance
    transforms.ColorJitter(
        brightness=0.3,                           # Staining variation
        contrast=0.3,                             # Microscope settings
        saturation=0.3,                           # Stain saturation
        hue=0.1,                                  # Slight color shifts
    ),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),                     # Cell position variation
        scale=(0.9, 1.1),                         # Magnification variation
    ),
    transforms.GaussianBlur(3, sigma=(0.1, 1.0)),  # Focus variation
    transforms.ToTensor(),
])

print("Medical imaging augmentation pipeline:")
for t in medical_augmentation.transforms:
    print(f"  → {t}")

print("\nDomain-specific considerations:")
print("  ✓ Full rotation: cells have no inherent orientation")
print("  ✓ Color jitter: accounts for staining variation")
print("  ✓ Gaussian blur: simulates out-of-focus regions")
print("  ✗ Extreme crops: could remove diagnostic features")
print("  ✗ Horizontal text flip: N/A for cell images but relevant for pathology reports")

# Show medical augmentation on a sample
fig, axes = plt.subplots(2, 5, figsize=(14, 5))
for i in range(10):
    ax = axes[i // 5, i % 5]
    aug_img = medical_augmentation(pil_img)
    # Denormalize for display if normalized, else just show
    if aug_img.max() <= 1.0:
        ax.imshow(aug_img.permute(1, 2, 0).clip(0, 1))
    else:
        ax.imshow(aug_img.permute(1, 2, 0))
    ax.set_title(f'Aug {i+1}', fontsize=9)
    ax.axis('off')

plt.suptitle('Medical-Style Augmentations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_04_computer_vision/medical_augmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Medical augmentation examples saved")

print("\n✅ Module 4, Script 04 complete!")
