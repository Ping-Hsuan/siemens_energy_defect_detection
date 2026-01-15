# Evaluation Metrics for Defect Image Generation

**Project:** Siemens Energy - Bottle Defect Detection using Stable Diffusion  
**Task 7:** Automated Quantitative Evaluation Metrics  
**Date:** January 2026

---

## Overview

This document outlines quantitative metrics for evaluating the quality of synthetically generated defect images from the LoRA-tuned Stable Diffusion model. The evaluation framework combines general image quality metrics with domain-specific defect detection criteria.

---

## 1. General Image Quality Metrics

### 1.1 Fréchet Inception Distance (FID)

**Purpose:** Measures the distance between real and generated image distributions in feature space.

**Definition:**

$$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$$

Where:
- $\mu_r, \mu_g$ = mean of real and generated image features
- $\Sigma_r, \Sigma_g$ = covariance matrices of real and generated features
- Features extracted from InceptionV3 pool3 layer (2048-dim)

**Interpretation:**
- **FID < 30:** Excellent quality (comparable to real images)
- **FID 30-50:** Good quality (suitable for few-shot scenarios)
- **FID 50-100:** Acceptable for data augmentation
- **FID > 100:** Poor quality, significant distribution mismatch

**Limitations:**
FID requires $\ge50$ samples per class for stable covariance matrix estimation. **For small datasets like ours (~20 samples per class), FID will have high variance.** 

**Mitigation:** Generate more synthetic samples per class than the real ones (stabilizes one side of comparison) and compute combined FID on all 83 real vs 200-400 synthetic for better stability.

### 1.2 Inception Score (IS)

**Purpose:** Evaluates image quality and diversity using a pretrained classifier.

**Definition:**

$$\text{IS} = \exp(\mathbb{E}_x[D_{KL}(p(y|x) \| p(y))])$$

Where:
- $p(y|x)$ = conditional class distribution from InceptionV3
- $p(y)$ = marginal class distribution
- $D_{KL}$ = Kullback-Leibler divergence

**Interpretation:**
- **Higher IS = Better:** More confident predictions + diverse samples
- **IS > 3.0:** Good for multi-object datasets (faces, animals, vehicles)
- **IS 1.5-2.5:** Expected for single-object datasets (all bottles)
- **IS < 1.5:** Possible mode collapse or extremely low quality

**Limitations:**
InceptionV3 is trained on ImageNet (not defect-specific) and measures object-level diversity, not defect-level diversity. **For single-object defect datasets like ours (all bottles), IS will be naturally low (1.5-2.5) and is not meaningful.** Since all images are classified as "bottle," there is low diversity in ImageNet space, resulting in low IS scores, which is expected.

---

## 2. Defect-Specific Metrics

### 2.1 Defect Classification Accuracy (DCA)

**Purpose:** Evaluate if synthetic defects are correctly classified by a defect classifier trained on real images. Adapts the manufacturing quality control concept: synthetic images are the "products" being inspected, and the classifier (fine-tuned on real bottles) acts as the "quality inspector."

**Method:**
1. Fine-tune a pre-trained classifier (e.g., ResNet-50 starting from ImageNet weights) on **real** bottle images (83 samples) to learn defect patterns
2. Generate synthetic samples per defect class (50-100 recommended)
3. Run the fine-tuned classifier on synthetic images
4. Measure per-class accuracy

**Definition:**

$$\text{DCA}_c = \frac{\text{Correct predictions for class } c}{\text{Total synthetic samples of class } c}$$

**Why This Matters:**
- If synthetic defects are correctly recognized by the real-trained classifier, they're realistic enough to pass quality control
- High DCA means synthetic samples exhibit authentic defect characteristics learned from real data
- Class-specific DCA reveals which defect types are well-generated vs need improvement
- **Primary metric for defect quality** (unlike IS/FID which measure general image quality)

---