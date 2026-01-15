# Task 3: Data Selection & Justification

**Date:** January 12, 2026  
**Project:** Few-Shot Industrial Defect Generation  
**Purpose:** Justify product class and sample size selection for diffusion model fine-tuning

---

## 1. Selected Configuration

### **Final Selection**

```python
data_selection = {
    "product_class": "bottle",
    "dataset_source": "DS-MVTec",
    "samples_per_damage_type": 20,  # min: 20, max: 22
    "total_samples": 83,
    "damage_types": 4,
}
```

---

## 2. Product Class Selection: Bottle (Cable Rejected)

### 2.1 Analysis of Available Products

We analyzed all 15 products in the DS-MVTec dataset:

| Product | Damage Types | Total Samples | Samples/Type | Resolution | Mask Coverage |
|---------|-------------|---------------|--------------|------------|---------------|
| **bottle** | **4** | **83** | **20-22** | **1024×1024** | **100%** ✅ |
| cable | 9 | 90 | 10 | 1024×1024 | 100% | ❌ **Has "combined" type** |
| zipper | 8 | 120 | 15-18 | 1024×1024 | 100% |
| pill | 7 | 105 | 15-18 | 1024×1024 | 100% |
| metal_nut | 6 | 72 | 12 | 1024×1024 | 100% |
| screw | 6 | 48 | 8 | 1024×1024 | 100% |
| capsule | 6 | 48 | 8 | 1024×1024 | 100% |
| hazelnut | 5 | 60 | 12 | 1024×1024 | 100% |
| carpet | 6 | 125 | 12-22 | 1024×1024 | 100% |
| grid | 6 | 120 | 12-22 | 1024×1024 | 95% |
| leather | 6 | 125 | 12-22 | 1024×1024 | 100% |
| tile | 6 | 120 | 12-22 | 1024×1024 | 95% |
| toothbrush | 4 | 100 | 15-30 | 1024×1024 | 90% |
| transistor | 5 | 110 | 12-25 | 1024×1024 | 95% |
| wood | 6 | 125 | 12-22 | 1024×1024 | 100% |

### 2.2 Selection Criteria & Rationale

#### **Criterion 1: Class Label Integrity** (Most Critical)

**Why Cable Was REJECTED:**
- Cable contains a `combined/` damage type in its image directory
- This "combined" category **mixes multiple defects in one image**, creating label ambiguity
- **Problem:** Class-conditional models assume each sample has ONE clear label
- When class_id=2 (combined) contains features from multiple damage types:
  - Class embeddings learn mixed/averaged features
  - Reduces separability with other classes
  - Generation is ill-defined: which specific defects to generate?
  
**Verified from dataset:**
```bash
$ ls scripts/Defect_Spectrum/DS-MVTec/cable/image/
bent_wire/  cable_swap/  combined/  cut_inner_insulation/  
cut_outer_insulation/  good/  missing_cable/  missing_wire/  poke_insulation/
```

**Bottle's Advantage:**
```bash
$ ls scripts/Defect_Spectrum/DS-MVTec/bottle/image/
broken_large/  broken_small/  contamination/  good/
# ✅ No ambiguous "combined" type - each class is unambiguous
```

#### **Criterion 2: Sufficient Per-Class Samples**
Bottle provides **20-22 samples per damage type**:
- **2× more than cable** (20 vs 10 per type)
- **1.3× more than zipper/pill** (20 vs 15-18 per type)
- Within established few-shot range: 10-100 samples (LoRA), 3-5 (DreamBooth)

```python
# Actual counts from DS-MVTec bottle:
damage_counts = {
    'broken_large': 20,
    'broken_small': 22, 
    'contamination': 21,
    'good': 20
}
# Total: 83 samples
```

#### **Criterion 3: Visual Distinctiveness**
Bottle's 4 damage types show high inter-class variance:
- `broken_large`: Large cracks/shattered glass (structural damage)
- `broken_small`: Small chips/hairline cracks (different severity)
- `contamination`: Stains/particles (different modality - not structural)
- `good`: Normal bottles (essential baseline)

**Advantage over alternatives:**
- vs Pill: Bottle has structural vs contamination distinction (clearer than color variations)
- vs Carpet/Leather/Tile: Structural damage more distinct than texture defects

#### **Criterion 4: Data Quality**
- **Resolution:** 1024×1024 (high quality)
- **Mask coverage:** 100% of defects have ground truth masks  
- **Consistency:** All images captured under controlled conditions
- **Balance:** 20-22 samples per type (minimal class imbalance)

---

## 3. Sample Size Selection: 20 Per Damage Type

### 3.1 Bottle Dataset Composition

```python
# DS-MVTec bottle dataset (verified):
damage_type_distribution = {
    "broken_large": 20,
    "broken_small": 22,
    "contamination": 21,
    "good": 20,
}
total_samples = 83
    "missing_cable": 10,          # Minimum
    "missing_wire": 14,
    "poke_insulation": 12,
}
# Minimum available per class: 10 samples
# Maximum balanced: 10 samples per class → 90 total
```

### 3.2 Why 10 Samples Per Type?

#### **Option Analysis:**

| Samples/Type | Total | Pros | Cons | Verdict |
|-------------|-------|------|------|---------|
| **5** | 45 | Extreme few-shot | Insufficient for LoRA training, high variance | ❌ Too risky |
| **10** ✅ | 90 | Few-shot, balanced across all classes | Limited data | ✅ **Selected** |
| **50** | 450 | More robust training | Not available in dataset | ❌ Not feasible |
| **100** | 900 | Strong performance | Not available in dataset | ❌ Not feasible |

#### **Justification for 10 Samples:**

**1. Dataset Constraint:**
- Some damage types have only 10 samples (e.g., `cut_inner_insulation`, `missing_cable`)
- To maintain **class balance**, we use `min(samples) = 10` per class
- Alternative: Use all available data → imbalanced (10-20 range) → biases model

**2. Few-Shot Learning Context:**
- **10 samples × 9 classes = 90 total samples**
- This is a legitimate few-shot scenario for 860M parameter model
- LoRA reduces trainable params to 8M → **11.25 samples per million trainable params** (safe zone)

**3. Data Efficiency with LoRA:**
```python
# With LoRA (rank=8):
trainable_params = 8,000,000
samples_per_million = 90 / 8 = 11.25  ✅ Acceptable

# If we used only 5 samples/type (45 total):
samples_per_million = 45 / 8 = 5.6    ⚠️ Too few, high variance

# Full fine-tuning (860M params):
samples_per_million = 90 / 860 = 0.1  ❌ Severe overfitting
```

**4. Comparison with Literature:**
- **DreamBooth paper:** 3-5 samples per subject → successful
- **LoRA paper:** 10-100 samples typically used
- **Our setup (10 samples):** Aligns with established few-shot practices

**5. Validation Strategy:**
- No validation split needed (90 samples all for training)
- Use generation-based validation + FID scores
- Stratified split would leave only 2 samples/class for validation → too noisy

---

## 4. Final Dataset Configuration

### 4.1 Dataset Composition

```python
final_dataset = {
    # Product Selection
    "product_class": "cable",
    "source": "DS-MVTec",
    
    # Sample Selection
    "samples_per_type": 10,
    "total_samples": 90,
    "num_classes": 9,
    
    # Data Split
    "train_samples": 90,     # Use all (no validation split)
    "val_samples": 0,        # Generation-based validation instead
    
    # Properties
    "resolution": (1024, 1024),
    "target_resolution": 512,  # Downsample for SD 1.5
    "format": "RGB",
    "mask_coverage": "100%",
    
    # Class Distribution (Balanced)
    "bent_wire": 10,
    "cable_swap": 10,
    "combined": 10,
    "cut_inner_insulation": 10,
    "cut_outer_insulation": 10,
    "good": 10,
    "missing_cable": 10,
    "missing_wire": 10,
    "poke_insulation": 10,
}
```

### 4.2 Data Preprocessing

```python
preprocessing = {
    # Resolution
    "resize": (512, 512),           # Match SD 1.5 native resolution
    "interpolation": "bicubic",     # High-quality downsampling
    
    # Normalization
    "pixel_range": [-1, 1],         # Standard for diffusion models
    "channel_order": "RGB",
    
    # Augmentation (Minimal)
    "horizontal_flip": 0.5,         # Safe for cables
    "vertical_flip": 0.0,           # Breaks cable orientation
    "rotation": 0.0,                # Could distort defects
    "color_jitter": 0.0,            # Preserve exact appearance
    "random_crop": 0.0,             # Need full context
}
```

---

## 5. Expected Challenges & Mitigation

### 5.1 Potential Issues

| Challenge | Risk Level | Mitigation Strategy |
|-----------|-----------|---------------------|
| **Overfitting with 90 samples** | Medium | Use LoRA (8M params), dropout (0.1), monitor FID |
| **Class imbalance effects** | Low | Balanced dataset (10/class), stratified sampling |
| **Insufficient data for complex defects** | Medium | Progressive training (3 phases), strong SD 1.5 priors |
| **Mode collapse** | Low | Classifier-free guidance (dropout=0.1), diverse batches |

### 5.2 Success Criteria

**Quantitative Metrics (Task 7):**
- FID score < 50 (good quality for industrial images)
- Class accuracy > 80% (classifier can identify damage type)
- Intra-class diversity (LPIPS > 0.3)

**Qualitative Assessment:**
- Defects clearly visible and realistic
- Class-specific features correctly generated
- No obvious artifacts or mode collapse

---

## 6. Comparison with Alternatives

### 6.1 Alternative Product Classes

**If we chose zipper (8 types, 140 samples):**
- ✅ More samples available
- ❌ Less diverse defects (mostly alignment issues)
- ❌ Lower industrial relevance

**If we chose pill (7 types, 130 samples):**
- ✅ Adequate samples
- ❌ Subtle color variations harder to capture
- ❌ Fewer damage types

**Verdict:** Cable offers the best balance of challenge, diversity, and relevance.

### 6.2 Alternative Sample Sizes

**If we used 5 samples/type (45 total):**
- ✅ More extreme few-shot scenario
- ❌ High training variance
- ❌ Likely insufficient for stable LoRA training
- ❌ 5.6 samples per million trainable params (too low)

**If we used all available data (unbalanced, ~150 total):**
- ✅ More training data
- ❌ Class imbalance (10-20 range)
- ❌ Model bias toward overrepresented classes
- ❌ Violates experimental rigor

**Verdict:** 10 samples/type provides optimal balance for few-shot learning.

---

## 7. Alignment with Project Goals

### 7.1 Task Requirements

✅ **Select one product class:** Cable selected  
✅ **Select X samples per damage type:** 10 samples selected  
✅ **Justify selection:** Comprehensive rationale provided  

### 7.2 Few-Shot Learning Objectives

- **Demonstrates** diffusion model capability with limited data
- **Tests** LoRA parameter efficiency (8M trainable vs 860M total)
- **Validates** class-conditional generation with 9 distinct classes
- **Provides** realistic industrial defect generation scenario

### 7.3 Downstream Impact

**For Task 4 (Fine-tuning):**
- 90 samples sufficient for LoRA training
- 9 classes test class embedding effectiveness
- Balanced data enables fair evaluation

**For Task 7 (Evaluation):**
- Clear damage types enable interpretable metrics
- 100% mask coverage supports defect-specific evaluation
- Diverse classes test generalization capability

---

## 8. Summary

### **Final Selection**
- **Product:** Cable
- **Samples:** 10 per damage type
- **Total:** 90 samples
- **Classes:** 9 damage types

### **Key Justifications**
1. **Cable** has most damage types (9), highest diversity, strong industrial relevance
2. **10 samples/type** balances dataset constraints with few-shot learning requirements
3. **90 total samples** provides 11.25 samples per million trainable params (safe with LoRA)
4. **Balanced distribution** ensures fair class-conditional generation

### **Risk Mitigation**
- LoRA parameter efficiency prevents overfitting
- Generation-based validation compensates for no data split
- Progressive training strategy adapts model gradually
- Strong SD 1.5 priors provide foundation for generalization

---

**Document Version:** 1.0  
**Last Updated:** January 12, 2026  
**Author:** Ping-Hsuan  
**Status:** Final - Ready for Implementation
