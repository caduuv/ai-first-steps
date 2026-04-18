"""
Module 4 — Script 02: Convolution
===================================

The convolution operation is what makes CNNs work. We implement it
from scratch, then show the PyTorch equivalent.

Topics:
  - 2D convolution from scratch
  - Edge detection kernels
  - Multiple filters
  - PyTorch Conv2d
  - Pooling operations

Run interactively with VS Code cells (# %%) or as a script.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Convolution from Scratch

# %%
def convolve2d(image: np.ndarray, kernel: np.ndarray, stride: int = 1,
               padding: int = 0) -> np.ndarray:
    """
    2D convolution operation (from scratch).
    
    The kernel slides over the image, computing element-wise
    multiplication and summation at each position.
    
    Args:
        image: 2D array (H, W)
        kernel: 2D array (kH, kW), must be smaller than image
        stride: step size for sliding the kernel
        padding: zero-padding added to image borders
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    h, w = image.shape
    kh, kw = kernel.shape
    
    # Output dimensions
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region under the kernel
            region = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            # Element-wise multiply and sum
            output[i, j] = np.sum(region * kernel)
    
    return output


# %% [markdown]
# ## 2. Edge Detection Kernels

# %%
print("=== Edge Detection with Convolution ===")

# Load a sample image
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                     transform=transforms.ToTensor())
image = mnist[3][0].squeeze().numpy()  # (28, 28)

# Define edge detection kernels
kernels = {
    'Horizontal Edge': np.array([[-1, -1, -1],
                                   [ 0,  0,  0],
                                   [ 1,  1,  1]], dtype=float),
    
    'Vertical Edge': np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]], dtype=float),
    
    'Sobel X': np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=float),
    
    'Sobel Y': np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=float),
    
    'Sharpen': np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=float),
    
    'Blur (3×3)': np.ones((3, 3), dtype=float) / 9,
}

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

# Original
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original', fontweight='bold')
axes[0, 0].axis('off')

# Apply each kernel
for idx, (name, kernel) in enumerate(kernels.items()):
    row = (idx + 1) // 4
    col = (idx + 1) % 4
    result = convolve2d(image, kernel, padding=1)
    axes[row, col].imshow(result, cmap='gray')
    axes[row, col].set_title(name, fontweight='bold', fontsize=10)
    axes[row, col].axis('off')

# Remove empty subplot if needed
axes[1, 3].axis('off')

plt.suptitle('Convolution with Different Kernels', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('module_04_computer_vision/convolution_kernels.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Convolution kernels visualization saved")

# %% [markdown]
# ## 3. PyTorch Conv2d

# %%
print("\n=== PyTorch Conv2d ===")

# Create a Conv2d layer
conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)

# Input: (batch, channels, height, width)
input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
print(f"Input shape: {input_tensor.shape}")

# Apply convolution
output = conv(input_tensor)
print(f"Output shape: {output.shape}")
print(f"  → 4 feature maps, same spatial size (padding=1)")

# Visualize learned filters (random initialization)
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Input', fontweight='bold')
axes[0].axis('off')

for i in range(4):
    feature_map = output[0, i].detach().numpy()
    axes[i+1].imshow(feature_map, cmap='viridis')
    axes[i+1].set_title(f'Filter {i+1}', fontweight='bold')
    axes[i+1].axis('off')

plt.suptitle('PyTorch Conv2d Output (Random Filters)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_04_computer_vision/pytorch_conv2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 PyTorch Conv2d output saved")

# %% [markdown]
# ## 4. Pooling Operations

# %%
print("\n=== Pooling Operations ===")

# Pooling reduces spatial dimensions while keeping important features
# Max Pooling: takes the maximum value in each window
# Average Pooling: takes the average value

# Create sample feature map
feature_map = torch.randn(1, 1, 8, 8)

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

max_result = max_pool(feature_map)
avg_result = avg_pool(feature_map)

print(f"Input: {feature_map.shape}")
print(f"After MaxPool2d(2): {max_result.shape}")
print(f"After AvgPool2d(2): {avg_result.shape}")

# Visualize pooling on real image
input_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()

pooled_2 = nn.MaxPool2d(2)(input_img)
pooled_4 = nn.MaxPool2d(4)(input_img)
pooled_7 = nn.MaxPool2d(7)(input_img)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, img_t, title in zip(axes,
    [input_img, pooled_2, pooled_4, pooled_7],
    ['Original (28×28)', 'MaxPool2d(2) → 14×14', 'MaxPool2d(4) → 7×7', 'MaxPool2d(7) → 4×4']):
    ax.imshow(img_t.squeeze().numpy(), cmap='gray')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.axis('off')

plt.suptitle('Max Pooling: Progressive Downsampling', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('module_04_computer_vision/pooling.png', dpi=150, bbox_inches='tight')
plt.close()
print("📊 Pooling visualization saved")

# %% [markdown]
# ## 5. Stride and Padding Effects

# %%
print("\n=== Stride and Padding ===")

# Show how different stride and padding affect output size
configs = [
    (1, 0, "stride=1, pad=0"),
    (1, 1, "stride=1, pad=1"),
    (2, 0, "stride=2, pad=0"),
    (2, 1, "stride=2, pad=1"),
]

for stride, padding, desc in configs:
    conv_test = nn.Conv2d(1, 1, kernel_size=3, stride=stride, padding=padding)
    out = conv_test(input_img)
    print(f"  {desc}: {input_img.shape[-2]}×{input_img.shape[-1]} → {out.shape[-2]}×{out.shape[-1]}")

# Output size formula:
# out = floor((input + 2*padding - kernel_size) / stride) + 1
print(f"\nFormula: out = floor((input + 2·pad - kernel) / stride) + 1")

print("\n✅ Module 4, Script 02 complete!")
