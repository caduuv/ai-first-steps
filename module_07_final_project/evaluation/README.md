# Evaluation Strategy Documentation

## Overview

Evaluating generative models for medical imaging requires both
quantitative metrics and qualitative assessment by domain experts.

## Quantitative Metrics

### 1. Fréchet Inception Distance (FID)

**What**: Measures the distance between feature distributions of real and generated images.

**How**: Extract features from a pretrained Inception-v3 network, compute mean and covariance of real and generated feature distributions, then compute the Fréchet distance.

**Interpretation**:
- FID = 0: Identical distributions
- FID < 50: Good quality for medical images
- FID < 100: Reasonable quality
- FID > 200: Poor quality

### 2. Inception Score (IS)

**What**: Measures generation quality (sharp, recognizable images) AND diversity.

**Limitation**: Doesn't compare to real data distribution directly. FID is generally preferred.

### 3. Per-Class FID

Compute FID separately for each cell type to ensure balanced quality:
- FID_Normal, FID_LSIL, FID_HSIL, FID_ASC-US

### 4. Downstream Classifier Performance

The ultimate test: are synthetic images useful?

```
Test 1: Train classifier on real data → evaluate accuracy
Test 2: Train classifier on real + synthetic data → evaluate accuracy
Test 3: Train classifier on only synthetic data → evaluate accuracy

If Test 2 > Test 1: synthetic data provides value!
If Test 3 ≈ Test 1: synthetic data is very realistic!
```

## Qualitative Assessment

### Expert Evaluation Protocol

1. Present pathologists with mixed set of real + synthetic images
2. Ask them to classify as real or synthetic
3. Ask them to assess clinical plausibility
4. Collect feedback on specific defects

### Visual Inspection Checklist

- [ ] Nuclear shape and boundary definition
- [ ] Cytoplasm texture and boundaries
- [ ] Chromatin pattern realism
- [ ] Appropriate N/C ratio for labeled class
- [ ] Color consistency with real staining
- [ ] No obvious artifacts (checkerboard, blur, distortion)
- [ ] Diversity within each class
- [ ] Class-specific features are present

### Failure Mode Analysis

Common generated image defects:
1. **Checkerboard artifacts**: From ConvTranspose2d — fix with upsampling + conv
2. **Mode collapse**: All images look the same — increase diversity loss
3. **Unrealistic borders**: Cell edges look artificial — use PatchGAN
4. **Color drift**: Generated colors don't match staining — improve data normalization
5. **Missing features**: Nuclei lack detail — increase model capacity

## Evaluation Pipeline

```python
def evaluate_model(model, real_loader, save_dir):
    # 1. Generate images per class
    generate_per_class_samples(model, n_per_class=500)
    
    # 2. Compute FID
    overall_fid = compute_fid(real_features, fake_features)
    per_class_fid = {cls: compute_fid(real[cls], fake[cls]) for cls in classes}
    
    # 3. Visual quality report
    create_comparison_grid(real_samples, fake_samples)
    
    # 4. Downstream classification
    train_classifier_and_evaluate(synthetic_data, real_test_data)
    
    # 5. Save report
    save_evaluation_report(metrics, visualizations)
```
