# Data — Cervical Cytology Dataset

## Overview

This directory will contain the cervical cytology image dataset used
for training the conditional generative model.

## Option 1: Simulated Dataset (Default)

A Python script generates simple cell-like images for development and testing:

```python
# Example usage (to be implemented):
from data.simulated_generator import generate_dataset

dataset = generate_dataset(
    n_samples=5000,
    image_size=64,
    cell_types=['Normal', 'LSIL', 'HSIL', 'ASC-US'],
)
```

### Simulated Cell Properties

| Cell Type | Nuclear Size | N/C Ratio | Chromatin | Shape |
|-----------|-------------|-----------|-----------|-------|
| Normal    | Small       | Low       | Even      | Regular |
| LSIL      | Medium      | Medium    | Slightly uneven | Slightly irregular |
| HSIL      | Large       | High      | Hyperchromatic | Irregular |
| ASC-US    | Variable    | Variable  | Variable  | Variable |

## Option 2: SIPaKMeD Dataset

### Download
1. Visit: https://www.cs.uoi.gr/~marina/sipakmed.html
2. Request access and download
3. Extract to `data/sipakmed/`

### Preprocessing Steps
1. Crop individual cells from whole slide images
2. Resize to uniform 64×64 or 128×128
3. Normalize staining (optional: Macenko normalization)
4. Create label mapping: filename → cell type

### Expected Directory Structure
```
data/
├── sipakmed/
│   ├── normal/
│   │   ├── cell_001.png
│   │   └── ...
│   ├── lsil/
│   ├── hsil/
│   ├── asc_us/
│   └── asc_h/
├── train/
├── val/
└── test/
```

## Option 3: Herlev Dataset

### Download
Search for "Herlev Pap smear dataset" in academic databases.

### Cell Classes (7 categories)
1. Superficial squamous epithelial
2. Intermediate squamous epithelial
3. Columnar epithelial
4. Mild squamous non-keratinizing dysplasia
5. Moderate squamous non-keratinizing dysplasia
6. Severe squamous non-keratinizing dysplasia
7. Squamous cell carcinoma in situ intermediate

## Data Pipeline Design

```python
# Planned PyTorch Dataset interface:
class CervicalCytologyDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None):
        # Load images and labels
        # Apply transforms
        pass
    
    def __getitem__(self, idx):
        image = self.images[idx]         # Tensor (C, H, W)
        cell_type = self.cell_types[idx]  # int
        features = self.features[idx]     # dict: nc_ratio, nuclear_size, etc.
        return image, cell_type, features
    
    def __len__(self):
        return len(self.images)
```

## Preprocessing Notes

1. **Stain Normalization**: Cytology images vary greatly in staining intensity. Consider Macenko or Reinhard normalization.
2. **Cell Segmentation**: If using whole-slide images, cells need to be segmented first.
3. **Quality Control**: Remove out-of-focus or overlapping cell images.
4. **Class Balancing**: Use oversampling or augmentation for rare classes (HSIL, ASC-H).
