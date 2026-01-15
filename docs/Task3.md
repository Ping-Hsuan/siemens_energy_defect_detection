# Task 3: Data Selection & Justification

This documentation justify why product class `bottle` is picked for diffusion model fine-tuning

---

## 1. Selected Configuration

### **Final Selection**: Bottle with 20 samples per damage type

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

## 2. Justification

We analyzed all 15 products in the DS-MVTec dataset:

| Product | Damage Types | Total Samples | Samples/Type | Resolution | Mask Coverage |
|---------|-------------|---------------|--------------|------------|---------------|
| **bottle** | **4** | **83** | **20-22** | **1024×1024** | **100%**  |
| cable | 9 | 90 | 10 | 1024×1024 | 100% |
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

We found that `bottle` is a strong candidate for the following reasons:

- Balanced per-class counts — bottle provides approximately **20–22 images per damage type**, which is among the highest per-class counts in DS‑MVTec and gives better within-class representation than many other products that include classes with $\le10$ samples or highly imbalanced `good` categories for few‑shot training.
- No ambiguous `combined` class — several products (e.g., `cable`, `pill`, `wood`, `zipper`) include a `combined` label that mixes defect types; bottle has mutually exclusive labels, avoiding label noise.
- Reliable mask coverage — bottle has ground‑truth masks for defects; most products include masks, though a few (e.g., `grid`, `tile`, `transistor`) show partial or irregular mask coverage that would complicate mask‑conditioned training.
 - Structural distinctiveness — bottle's damage types provide clear, complementary signal modalities: `broken_large` (large fractures), `broken_small` (small chips), `contamination` (surface stains), and `good` (intact baseline). This mix of structural and non‑structural defects yields strong "what" (semantic) and "where" (spatial) cues, making bottle easier to learn in few‑shot, class‑conditional settings than products dominated by subtle color/texture changes (e.g., `pill`) or texture defects (e.g., `carpet`, `leather`, `tile`), especially when some alternatives also include ambiguous `combined` labels.

---

## 3. Sample Size Selection: ~20 Per Damage Type

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
```

Note: we use the full available per-class images (20, 22, 21, 20) rather than truncating to a fixed number — the counts are sufficiently close that keeping all samples preserves intra-class variation without introducing imbalance.

### 3.2 Why ~20 Samples Per Type?

We chose to use the available ~20 images per bottle damage type for the following reasons:

- Dataset availability: bottle naturally provides 20–22 samples per class; using these avoids discarding useful data and preserves intra‑class variability.
- Sample efficiency with LoRA: LoRA keeps trainable parameters small (millions rather than hundreds of millions), so ~20 samples/class provides a more favorable samples‑to‑trainable‑param ratio and reduces overfitting risk compared with full fine‑tuning.
- Structural signal strength: bottle defects are structural or clearly localized (cracks vs stains). Structural cues are more sample‑efficient than subtle texture/color changes, so ~20 examples capture meaningful variation in defect appearance and geometry.
- Evaluation stability: more examples per class improve the reliability of generation‑based metrics (DCA, mask IoU, FID) and make per‑class diagnostics less noisy.

