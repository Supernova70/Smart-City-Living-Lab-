# Model Card — ImageGuard AI Deep Model

**Artifact:** `artifacts/image_quality_model.pt`  
**Architecture:** MobileNetV3-Small (multi-task)  
**Date trained:** 2026-08-29  
**Status:** ⚠️ CANDIDATE — macro-F1 gate not met; not auto-promoted to production

---

## Model Overview

Multi-task transfer-learning model built on MobileNetV3-Small (ImageNet pretrained),
fine-tuned on real photographic quality and industrial defect datasets.  
Two output heads:
- **Issue classifier** — 6-class multi-label sigmoid (one threshold per class)
- **Quality regressor** — single scalar in [0, 1] mapped to [0, 100]

---

## Training Data

| Dataset | Records | Role | License |
|---------|---------|------|---------|
| KADID-10k | 10,125 distorted + 81 pristine | Quality regression + blur/noise/exposure/degradation | CC BY 4.0 |
| MVTec AD | 5,354 images (15 object categories) | `potential_defect` head only | CC BY-NC-SA 4.0 |
| **Combined** | **15,479** | Train 11,620 / Val 1,943 / Test 1,916 | — |

**Note:** MVTec AD data is non-commercial only. This model must not be used commercially.

**Split strategy:** KADID splits grouped by reference photograph (no reference leakage across splits). MVTec splits stratified by category.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | `mobilenet_v3_small` |
| Image size | 224 × 224 |
| Normalisation | ImageNet (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) |
| Batch size | 20 |
| Freeze epochs | 1 (backbone frozen, heads only) |
| Total epochs run | 12 (early stopped; best at epoch 6) |
| Patience | 6 |
| Optimiser | AdamW, lr=8e-4 → 1.5e-4 cosine |
| Loss | BCE (pos-weighted) + MSE quality |
| AMP | FP16 autocast on CUDA |
| Device | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) |
| Seed | 42 |

**EfficientNet-B0 candidate crashed** with `cudaErrorIllegalAddress` during epoch 1 training
(channels_last memory format incompatibility on the RTX 3050). MobileNetV3-Small is the sole
evaluated candidate.

---

## Test-Set Results (held-out, unseen during training)

### Overall

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Quality MAE | **13.96** | ≤ 15 | ✅ PASS |
| Quality RMSE | 19.12 | — | — |
| Macro-F1 | **0.658** | ≥ 0.70 | ❌ FAIL |
| Selection score | 0.719 | — | — |
| PD ROC-AUC | **0.924** | ≥ 0.80 | ✅ PASS |
| Nonzero support (all classes) | Yes | required | ✅ PASS |
| Failure cases documented | 10 cases | required | ✅ PASS |

### Per-Class (test set, n=1,916 records)

| Class | Observed | Positives | Precision | Recall | F1 | ROC-AUC | Threshold |
|-------|---------|----------|-----------|--------|-----|---------|-----------|
| blur | 1500 | 180 | 0.903 | 0.728 | **0.806** | 0.947 | 0.70 |
| underexposure | 1500 | 60 | 0.211 | 0.717 | **0.326** | 0.882 | 0.425 |
| overexposure | 1500 | 60 | 0.218 | 0.617 | **0.322** | 0.909 | 0.80 |
| noise | 1500 | 300 | 0.984 | 0.837 | **0.905** | 0.990 | 0.85 |
| severe_degradation | 1500 | 420 | 0.658 | 0.733 | **0.694** | 0.887 | 0.275 |
| potential_defect | 416 | 293 | 0.921 | 0.877 | **0.899** | 0.924 | 0.40 |

---

## Acceptance Gate Analysis

### ❌ Macro-F1 Gate Failure

**Root cause:** `underexposure` (F1=0.326) and `overexposure` (F1=0.322) have very low
precision due to extreme class imbalance in the KADID test partition:

- Only **60 positives out of 1,500 observations** (4% positive rate) per class
- At this prevalence, achieving F1≥0.70 would require precision≥0.70 at recall≥0.70,
  which demands near-perfect discrimination — mathematically difficult without much more
  training data for these conditions
- **ROC-AUC is strong** (underexposure 0.882, overexposure 0.909), confirming the model
  has real discriminative ability; the F1 shortfall is a threshold/imbalance artefact
- The existing **explainable gates** in `app/service.py` (`_apply_explainable_gates`)
  catch under/overexposure via brightness statistics (`dark_pixel_ratio`,
  `highlight_clip_ratio`) as a safety net, partially compensating for low precision

**What was tried:**
- Extended training to epoch 12 (early stopped at epoch 12, best was epoch 6)
- Threshold re-tuning from KADID validation negatives (increased precision to 1.0 for PD
  but reduced overall macro-F1 further; reverted)
- EfficientNet-B0 alternative candidate crashed (CUDA error)

**Verdict:** Gate failure is attributable to dataset imbalance and a single missing candidate
(EfficientNet-B0). Model is NOT auto-promoted. User review required before promoting
`image_quality_model.pt` over `image_quality_model.joblib`.

---

## Known Limitations

1. **Macro-F1 < 0.70** — underexposure and overexposure classes underperform due to 4%
   positive rate in KADID test set.
2. **Potential-defect cross-domain false positives** — the PD head assigns high probability
   (>0.87) to many KADID distorted images. This is a domain-shift effect: heavily degraded
   KADID images superficially resemble MVTec defects. The current threshold (0.40) was
   tuned on mixed validation data; in practice the PD flag should only be trusted for images
   that stylistically resemble industrial inspection scenes.
3. **EfficientNet-B0 unavailable** — CUDA illegal memory access on RTX 3050 Laptop GPU
   with channels_last memory format. Only MobileNetV3-Small was evaluated.
4. **MVTec licence restriction** — CC BY-NC-SA 4.0; this model must not be used in
   commercial products.
5. **Quality regression MAE inflated on validation** — validation MAE hovered 18–21
   during training, but test MAE is 13.96. The quality head generalises better than
   validation metrics suggested.

---

## Comparison vs Baseline

| Metric | Random Forest (.joblib) | MobileNetV3-Small (.pt) |
|--------|------------------------|------------------------|
| Trained on real data | ❌ synthetic only | ✅ KADID + MVTec |
| potential_defect head | ❌ not available | ✅ F1=0.899, AUC=0.924 |
| Quality MAE (test) | not evaluated on real data | **13.96** |
| blur AUC | not evaluated | **0.947** |
| noise AUC | not evaluated | **0.990** |
| Mac-F1 gate | n/a | ❌ 0.658 |

---

## Intended Use

- Non-commercial educational and assessment demonstration only
- Supplementary quality signal; not a replacement for human review
- Potential-defect predictions should be treated as a preliminary alert requiring
  human verification

## How to Promote

To switch the application to the `.pt` model:
1. Set `MODEL_PATH=artifacts/image_quality_model.pt` in `.env`
2. Verify `pytest` passes (all `test_deep_model.py` tests run)
3. Review failure cases in `real_examples/results.json`
4. Accept the macro-F1 limitation (0.658) as a known deficiency

The `.joblib` baseline remains as fallback:
```
MODEL_PATH=artifacts/image_quality_model.joblib
```
