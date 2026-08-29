# ImageGuard AI

A full-stack application that accepts an image and automatically evaluates its visual quality using a hybrid computer-vision and deep-learning pipeline. Built as a technical assessment for a software engineering internship role.

---

## What it does

Upload any JPEG, PNG, or WebP image and get back a structured quality report covering six defect categories:

| Detected condition | Description |
|--------------------|-------------|
| Blur / sharpness | Laplacian variance below learned threshold |
| Underexposure | Dark pixel dominance or low mean brightness |
| Overexposure | Highlight clipping or extreme brightness |
| Image noise | High-frequency residual above estimated floor |
| Severe degradation | Blockiness artifacts or compound distortion |
| Potential visual defect | Industrial anomaly proxy (MVTec-trained head) |

Every result includes a 0–100 quality score, an ACCEPTABLE / DEGRADED / POTENTIALLY\_DEFECTIVE label, per-issue confidence values, raw image statistics, and explainability evidence showing exactly which threshold or model output drove each decision.

No external AI services are used. All inference runs locally in under 50 ms on CPU.

---

## Architecture overview

```
Browser (HTML + CSS + JavaScript)
    │
    │  multipart upload / JSON history
    ▼
FastAPI (Python 3.13)
    │
    ├─ Safe image decode (Pillow — JPEG, PNG, WebP)
    ├─ 21 explainable CV statistics (NumPy)
    ├─ MobileNetV3-Small inference  ← dual-head deep model (.pt)
    │     ├─ Quality regression head  →  score 0–100
    │     └─ 6-class sigmoid head     →  issue probabilities
    ├─ Explainable gate overrides     ←  CV stats hard-floor model probs
    ├─ Decision policy (label + score cap)
    └─ SQLite persistence (standard library sqlite3)
```

The **explainable gate layer** is key: if a CV measurement (e.g. `brightness_mean < 0.18`) clearly signals underexposure, the model probability is boosted regardless of what the neural network predicted. This makes every decision traceable back to a measurable image property.

---

## Model

### Why MobileNetV3-Small

Several architectures were evaluated. The selection came down to:

| Architecture | Result | Reason |
|---|---|---|
| EfficientNet-B0 | ❌ Crashed | `cudaErrorIllegalAddress` on RTX 3050 Laptop with `channels_last` memory layout — unrecoverable |
| MobileNetV3-Small | ✅ Used | Stable on RTX 3050, fast on CPU (~14 ms), ImageNet pretrained, 2.5M parameters |

MobileNetV3-Small was chosen because it is lightweight enough to run on CPU at deployment time without a GPU, yet deep enough to learn meaningful texture features for both quality regression and defect detection. The pretrained ImageNet weights give it a strong feature initialisation for photographic images.

### Architecture detail

```
ImageNet-pretrained MobileNetV3-Small backbone
    └─ Features (576-dim pooled output)
         ├─ Quality head:  Linear(576 → 1)   → sigmoid × 100  → score 0–100
         └─ Issue head:    Linear(576 → 6)   → sigmoid         → 6 independent probabilities
```

The two heads share the same frozen-then-unfrozen backbone. The issue head uses **6 independent sigmoid outputs** (not softmax) because issues are not mutually exclusive — an image can be simultaneously blurry and underexposed.

**Issue order (fixed):**
```
0: blur
1: underexposure
2: overexposure
3: noise
4: severe_degradation
5: potential_defect
```

### Training configuration

| Hyperparameter | Value |
|---|---|
| Backbone | `mobilenet_v3_small` (torchvision, ImageNet-1k pretrained) |
| Input size | 224 × 224 |
| Batch size | 24 |
| Optimiser | Adam (lr = 1e-4, weight_decay = 1e-4) |
| LR schedule | StepLR — ×0.5 every 4 epochs |
| Mixed precision | AMP FP16 (torch.cuda.amp) |
| Epochs | 12 (early stopped at epoch 6 — best validation score) |
| Early stop patience | 6 epochs |
| Loss — quality head | MSE against DMOS score (scaled 0–100) |
| Loss — issue head | Binary cross-entropy (each issue independently) |
| Combined loss | `0.5 × MSE + 0.5 × BCE` |
| Hardware | RTX 3050 Laptop GPU (4 GB VRAM) |

### Data preprocessing

All images go through the same pipeline at both train and inference time:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                     std= [0.229, 0.224, 0.225])    # ImageNet std
```

Training additionally applies random horizontal flip and colour jitter for augmentation. Inference uses the exact same resize + normalize without augmentation.

### Dataset splits and leakage prevention

| Split | KADID strategy | MVTec strategy | Size |
|---|---|---|---|
| Train | Group by pristine reference image | Stratify by category | ~70% |
| Validation | Same grouping | Same | ~15% |
| Test | Same grouping | Same | ~15% |

KADID has 81 pristine reference images, each with 125 distorted derivatives. Splitting by reference ensures the model never sees a distorted version of a reference it trained on — preventing the most common leakage in IQA benchmarks.

### Issue detection thresholds

Each issue has a per-class detection threshold tuned on the validation set:

| Issue | Threshold | Notes |
|---|---|---|
| blur | 0.40 | Laplacian-variance gate also active |
| underexposure | 0.40 | CV gate: brightness_mean < 0.18 → prob ≥ 0.92 |
| overexposure | 0.40 | CV gate: brightness_mean > 0.84 → prob ≥ 0.92 |
| noise | 0.40 | CV gate: noise_estimate > 0.075 → prob ≥ 0.88 |
| severe_degradation | 0.40 | CV gate: blockiness > 0.055 → prob ≥ 0.82 |
| potential_defect | 0.40 | MVTec-trained; high false-positive rate on non-industrial images |

### Explainable gate layer

The gate layer runs **after** the neural network and **before** the threshold decision. It floors probabilities using hard CV measurements:

```python
# Underexposure gate
if brightness_mean < 0.18 or dark_pixel_ratio > 0.40:
    underexposure = max(model_prob, 0.92)   # definite underexposure
elif brightness_mean < 0.28:
    underexposure = max(model_prob, 0.68)   # likely underexposure

# Overexposure gate
if brightness_mean > 0.84 or highlight_clip_ratio > 0.35:
    overexposure = max(model_prob, 0.92)
elif brightness_mean > 0.74:
    overexposure = max(model_prob, 0.68)

# Noise gate
if noise_estimate > 0.075:
    noise = max(model_prob, 0.88)

# Blockiness / compression gate
if blockiness > 0.055:
    severe_degradation = max(model_prob, 0.82)
```

The gate never *suppresses* a model prediction — it only raises it. The final result returned by the API includes both `model_probability` (raw neural network output) and `confidence` (after gate), so the UI can show when a gate override fired.

### Training data

| Dataset | Role | Size |
|---|---|---|
| [KADID-10k](https://database.mmsp-kn.de/kadid-10k-database.html) | Quality regression + blur / noise / degradation / exposure labels | 10,125 distorted images, 81 reference images |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Defect detection head only | 5,354 images, 15 industrial categories |

**Total manifest:** 15,479 records across train / validation / test splits.

### Test-set evaluation (real data, unseen during training)

| Metric | Value | Gate |
|---|---|---|
| Quality MAE | **13.96** | ≤ 15 ✅ |
| PD head ROC-AUC | **0.924** | ≥ 0.80 ✅ |
| Macro-F1 | **0.658** | ≥ 0.70 ❌ |

The macro-F1 gate is not met. Underexposure and overexposure have only 4 % positive rate in the KADID test partition, yielding per-class F1 ≈ 0.32 for those two heads. The PD head (ROC-AUC 0.924) and quality regression (MAE 13.96) perform well. The explainable brightness gates compensate for exposure detection at inference time. See `artifacts/model_card.md` for the full per-class breakdown and failure cases.

### Inference latency

| Environment | Avg latency |
|---|---|
| CPU (warm, single image) | ~14 ms |
| First request (model load) | ~2.7 s |

---

## Quick start (local Python)

Requirements: Python 3.11 or later (tested on Python 3.13).

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment configuration
Copy-Item .env.example .env

# 4. Start the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser. Interactive API docs are at **http://127.0.0.1:8000/docs**.

The trained model artifact (`artifacts/image_quality_model.pt`) is included in the repository. No training step is required to run the application.

---

## Docker Compose

Start Docker Desktop, then:

```powershell
docker compose up --build
```

Open **http://localhost:8000**. The named volume `imageguard_data` retains the SQLite database and uploaded previews across container restarts.

**Health check:**

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "database": "ready",
  "model": "ready",
  "model_name": "real_data_mobilenet_v3_small_multitask",
  "model_version": "2.0.0",
  "detail": null
}
```

---

## Project structure

```
app/
  config.py          Environment configuration and .env loading
  database.py        SQLite schema and queries
  features.py        Safe image decoding and 21 CV statistics
  model_service.py   Dual-format artifact loading and inference
  service.py         Analysis orchestration and decision policy
  deep_model.py      MobileNetV3-Small architecture definition
  main.py            FastAPI routes and error handlers
  static/            Frontend (HTML / CSS / JS — no framework)
scripts/
  train_real_model.py       Full KADID + MVTec training pipeline
  build_real_manifest.py    Grouped, leakage-safe dataset manifests
  export_real_examples.py   Held-out test examples with predictions
  generate_samples.py       Synthetic sample image generator
  test_real_images.py       Live-server upload test for real examples
artifacts/
  image_quality_model.pt    MobileNetV3-Small checkpoint (v2.0.0, 14 MB)
  image_quality_model.joblib  Legacy Random Forest fallback
  model_card.md             Full metrics, gates, failure cases
  metrics.json              Machine-readable test metrics
tests/
  test_deep_model.py        12 integration tests for the .pt model
  test_api.py               API flow, pagination, and error codes
  test_features.py          CV feature extraction behaviour
  test_model.py             Legacy model load and range checks
  test_real_data.py         Manifest integrity and label constraints
  test_samples.py           Sample image coverage
  test_frontend.py          Static asset availability
datasets/
  SOURCES.md                Dataset provenance, licenses, download commands
real_examples/              7 held-out images + predictions JSON
sample_images/              Demonstration images (all quality conditions)
```

---

## REST API

### Analyze an image

```powershell
curl.exe -X POST http://localhost:8000/api/v1/analyses `
  -F "file=@real_examples/01_acceptable.jpg"
```

**Response (201):**

```json
{
  "id": "3b2f...",
  "original_filename": "01_acceptable.jpg",
  "quality_score": 27.8,
  "quality_label": "POTENTIALLY_DEFECTIVE",
  "issues": [
    {
      "type": "potential_defect",
      "severity": "high",
      "confidence": 0.8706,
      "model_probability": 0.8706,
      "evidence": ["Anomaly pattern detected by defect head."]
    }
  ],
  "statistics": {
    "width": 768, "height": 512,
    "brightness_mean": 0.512,
    "laplacian_variance": 312.4,
    "noise_estimate": 0.021,
    ...
  },
  "model_name": "real_data_mobilenet_v3_small_multitask",
  "model_version": "2.0.0",
  "timing_ms": { "decode": 4.1, "features": 2.3, "inference": 38.6, "total": 47.2 },
  "image_url": "/api/v1/analyses/3b2f.../image"
}
```

### Retrieve history

```powershell
curl.exe "http://localhost:8000/api/v1/analyses?limit=20&offset=0"
```

### Retrieve a single analysis

```powershell
curl.exe http://localhost:8000/api/v1/analyses/ANALYSIS_ID
```

### HTTP status codes

| Code | Meaning |
|------|---------|
| 201 | Analysis created successfully |
| 400 | Empty or malformed file |
| 413 | File exceeds 10 MB or image exceeds pixel limit |
| 415 | Unsupported format (not JPEG / PNG / WebP) |
| 422 | Unreadable image data or invalid query parameters |
| 404 | Analysis ID or preview not found |
| 503 | Model or database not ready |
| 500 | Unexpected internal error (no stack trace exposed) |

---

## Database

SQLite is initialised automatically at `data/quality.db`. No migration command is needed. Each successful analysis persists:

- Image metadata (filename, dimensions, MIME type, file size, SHA-256)
- Quality score and label
- Full issue list with severity, confidence, and evidence
- All 21 image statistics
- Timing breakdown and model version

Change the database path or upload directory by editing `.env`.

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed. No API keys or secrets are required.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `artifacts/image_quality_model.pt` | Path to the model artifact |
| `DATABASE_PATH` | `data/quality.db` | SQLite database file |
| `UPLOAD_DIR` | `data/uploads` | Stored image previews |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size |
| `MAX_IMAGE_PIXELS` | `40000000` | Maximum decoded pixel count |
| `HISTORY_PAGE_SIZE` | `20` | Default history page size |

---

## Training (optional — model is pre-included)

Re-training from scratch requires the KADID-10k and MVTec AD datasets. See `datasets/SOURCES.md` for download instructions.

```powershell
# Build the grouped dataset manifest
python -m scripts.build_real_manifest

# Train (resumes from checkpoint if interrupted)
python -m scripts.train_real_model

# Export held-out test examples
python -m scripts.export_real_examples
```

Training runs for up to 30 epochs with early stopping (patience 6). On an RTX 3050 Laptop GPU, one epoch takes approximately 3 minutes. EfficientNet-B0 was tested but caused `cudaErrorIllegalAddress` on this GPU; MobileNetV3-Small is the validated architecture.

---

## Tests

```powershell
pytest -q
```

All 27 tests pass. Coverage includes:

- Deep model checkpoint structure and inference ranges
- API upload, pagination, and error handling
- CV feature extraction with controlled synthetic images
- Manifest integrity, label constraints, and dataset split determinism
- Frontend asset availability
- Restart persistence (records survive process restart)

---

## Real-image evaluation

Seven held-out images from KADID-10k and MVTec AD are included in `real_examples/` with pre-computed predictions. To run a live test against the running server:

```powershell
python scripts/test_real_images.py
```

Results from the running v2.0.0 model:

| File | Label | Score | Detected issues |
|------|-------|-------|-----------------|
| 01_acceptable.jpg (KADID reference) | POTENTIALLY_DEFECTIVE | 27.8 | potential_defect, severe_degradation |
| 02_blur.jpg (KADID blur distortion) | POTENTIALLY_DEFECTIVE | 27.8 | potential_defect, severe_degradation |
| 03_underexposure.jpg | POTENTIALLY_DEFECTIVE | 28.7 | potential_defect, severe_degradation |
| 04_overexposure.jpg | POTENTIALLY_DEFECTIVE | 26.9 | severe_degradation, potential_defect |
| 05_noise.jpg (KADID noise) | POTENTIALLY_DEFECTIVE | 26.4 | potential_defect, severe_degradation |
| 06_severe_degradation.jpg | POTENTIALLY_DEFECTIVE | 28.2 | potential_defect, severe_degradation |
| 07_potential_defect.jpg (MVTec) | POTENTIALLY_DEFECTIVE | 35.3 | noise, potential_defect |

**Analysis:** All KADID images are flagged as `POTENTIALLY_DEFECTIVE` — a known domain-shift artefact. The defect head was trained exclusively on MVTec AD industrial images (15 categories of manufactured objects). When presented with natural photographs from KADID, it predicts high defect probability because the texture distribution does not match its training domain. The explainable gate layer partially compensates for exposure issues but cannot override the defect head for this type of shift. This is documented as a primary limitation in `artifacts/model_card.md`.

The MVTec image (07) is correctly flagged as `POTENTIALLY_DEFECTIVE` with noise co-detection, matching the industrial defect annotation.

---

## Failure cases and known limitations

- **Domain shift on the defect head:** The `potential_defect` head was trained on MVTec AD industrial textures. Natural photographs consistently trigger false positives. A production deployment would either retrain on in-domain normal samples or gate the defect head behind a domain classifier.
- **Underexposure / overexposure F1 = 0.32:** These classes have 4 % positive rate in the KADID test partition. The model learns them but lacks enough test examples to score well on macro-F1. The CV brightness gates compensate at inference time.
- **Quality score is relative, not absolute:** DMOS scores from KADID are pairwise human preference judgments, not a universal image quality standard. The regression head learns the KADID distribution.
- **Artistic intent vs. defect:** A deliberately dark scene, shallow depth of field, or high-grain aesthetic will be flagged as defective. The system is designed for quality control use cases, not creative photography assessment.

---

## Sample images

```powershell
python -m scripts.generate_samples
```

Regenerates all images in `sample_images/` covering: acceptable, blur, underexposure, overexposure, noise, severe degradation, and local-defect proxy conditions.
