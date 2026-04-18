# Module 4 — Computer Vision

## 🎯 Learning Objectives

By the end of this module, you will:

1. Understand images as **tensors** and manipulate them programmatically
2. Implement **convolution** from scratch and understand its role in CNNs
3. Build a **Convolutional Neural Network (CNN)** step by step
4. Apply **data augmentation** to improve model generalization
5. Complete a full **image classification project** with proper architecture

---

## 📖 Theory Overview

### Images as Tensors

An image is a 3D tensor: **(Height × Width × Channels)**
- Grayscale: 1 channel
- RGB: 3 channels (Red, Green, Blue)
- Medical imaging: often more channels (fluorescence, stains)

Pixel values typically range 0–255 (uint8) or 0.0–1.0 (float).

### Convolution

Convolution is the key operation in CNNs. A small **kernel** (filter) slides over the image:
- Detects local patterns (edges, textures, shapes)
- Parameter sharing: same kernel applied everywhere
- Translation invariance: detects patterns regardless of position

### CNN Architecture

```
Input Image
  ↓
[Conv2D + ReLU]    → Detect low-level features (edges)
[MaxPool]          → Reduce spatial dimensions
  ↓
[Conv2D + ReLU]    → Detect mid-level features (textures)
[MaxPool]          → Further reduction
  ↓
[Flatten]          → Reshape to 1D
[Fully Connected]  → Classification
[Softmax]          → Output probabilities
```

### Data Augmentation

Artificially expand the training set by applying random transformations:
- Random rotations, flips, crops
- Color jittering, brightness changes
- Critical for medical imaging (limited data)

---

## 📂 Scripts in This Module

| Script | Description |
|--------|-------------|
| `01_images_as_tensors.py` | Loading, displaying, manipulating images |
| `02_convolution.py` | Manual convolution + PyTorch Conv2d |
| `03_cnn_architecture.py` | Building a CNN step by step |
| `04_data_augmentation.py` | Transforms, augmentation strategies |
| `project_classifier/` | Full CIFAR-10 classification project |

---

## ➡️ Next Module

[Module 5 — Generative Models](../module_05_generative_models/README.md)
