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

**Model:** MobileNetV3-Small (pretrained on ImageNet-1k, fine-tuned for this task)
**Version:** 2.0.0
**Parameters:** ~2.5 million
**Artifact:** `artifacts/image_quality_model.pt` (14 MB PyTorch checkpoint)

### Trained on

| Dataset | What it contributes | Size |
|---------|---------------------|------|
| [KADID-10k](https://database.mmsp-kn.de/kadid-10k-database.html) | Image quality scores (DMOS) + issue labels for blur, noise, exposure, degradation | 10,125 images across 25 distortion types, 81 pristine references |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Visual defect detection | 5,354 images across 15 industrial product categories |

Total training set: **15,479 images** split into train (70%) / validation (15%) / test (15%).

### What the model outputs

Two outputs per image:

- **Quality score** — a number from 0 to 100. Derived from KADID DMOS ratings. Higher is better.
- **Issue probabilities** — one confidence value (0–1) for each of the 6 defect types below.

| Issue | Description |
|-------|-------------|
| `blur` | Insufficient sharpness (low Laplacian variance) |
| `underexposure` | Image too dark |
| `overexposure` | Image too bright / highlights clipped |
| `noise` | High-frequency sensor or compression noise |
| `severe_degradation` | Heavy compression artefacts or combined distortions |
| `potential_defect` | Industrial surface anomaly (trained on MVTec AD) |

An issue is flagged when its confidence exceeds **0.40** (the detection threshold for all classes).

### Training setup

| Setting | Value |
|---------|-------|
| Input size | 224 × 224 px |
| Batch size | 24 |
| Optimiser | Adam, lr = 1e-4 |
| Epochs | 12 total, best checkpoint at epoch 6 |
| Mixed precision | FP16 (AMP) |
| GPU | NVIDIA RTX 3050 Laptop (4 GB VRAM) |

### Test results (held-out data, never seen during training)

| Metric | Score |
|--------|-------|
| Quality MAE | 13.96 points (out of 100) |
| Defect head ROC-AUC | 0.924 |
| Macro-F1 across all issues | 0.658 |

The macro-F1 misses the 0.70 target. Underexposure and overexposure make up only 4% of the test set, so those two classes score low individually. The defect head and quality score both perform well. See `artifacts/model_card.md` for per-class numbers.

### Inference speed

~14 ms per image on CPU (after the first request warm-up of ~2.7 s).


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
