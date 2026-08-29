# AI Image Quality Assessment - Implementation TODO

Status rule: mark an item complete only after its stated verification passes.

## 0. Scope lock

- [x] Read and map every requirement from `Software_Internship_Assessment.pdf`.
- [x] Choose deadline-safe architecture: FastAPI + SQLite + static responsive frontend + trained scikit-learn model.
- [x] Define boundaries: single-image analysis, local persistence, no external AI APIs, potential defect is an anomaly proxy.
- [x] Keep `project.md` aligned with the final implementation.

## 1. Project scaffold

- [x] Create backend/application package structure.
- [x] Add pinned Python dependencies and `.gitignore`.
- [x] Add configuration and required data/artifact directories.
- [x] Verify all application modules import successfully.

## 2. Computer-vision feature pipeline

- [x] Implement safe image decoding and validation for JPEG/PNG/WebP.
- [x] Implement brightness/exposure, contrast, sharpness, noise, saturation, entropy, and blockiness features.
- [x] Implement human-readable evidence generation.
- [x] Add feature tests using controlled generated images.
- [x] Verify feature tests pass (5 passed).

## 3. AI/ML component

- [x] Implement controlled training-data generation with clean and degraded images.
- [x] Generate blur, underexposure, overexposure, noise, severe-degradation, and local-defect proxy labels.
- [x] Train multi-label issue classifier and quality-score regressor.
- [x] Save versioned model artifact and evaluation metrics.
- [x] Add optional KADID-10k indexing adapter and documented label mapping.
- [x] Verify model loads and predicts valid score/probability ranges (7 total tests passed).

## 4. Persistence and analysis service

- [x] Implement SQLite schema and initialization.
- [x] Implement analysis orchestration: validate, feature extraction, inference, decision policy, storage.
- [x] Persist uploaded preview safely using generated filenames.
- [x] Implement history and detail retrieval.
- [x] Verify records survive application/service restart.

## 5. REST API

- [x] Implement `POST /api/v1/analyses`.
- [x] Implement `GET /api/v1/analyses` with pagination.
- [x] Implement `GET /api/v1/analyses/{id}`.
- [x] Implement safe image preview endpoint.
- [x] Implement `GET /health` with database/model readiness.
- [x] Implement structured error responses and correct HTTP status codes.
- [x] Verify API integration tests pass (3 passed, including restart persistence).

## 6. Frontend

- [x] Build responsive upload/drop-zone interface.
- [x] Add client-side preview, validation, loading, success, and error states.
- [x] Display quality score, label, detected issues, severity, confidence, evidence, and statistics.
- [x] Build persistent analysis-history view and result reopening.
- [x] Add accessible labels, keyboard behavior, and non-color-only statuses.
- [x] Verify frontend assets load and the complete analysis/history API flow passes (4 tests).

## 7. Deployment

- [x] Add Dockerfile and Docker Compose configuration (`docker compose config` validated).
- [x] Add environment-variable configuration and `.env.example`.
- [x] Add persistent volumes for SQLite database and uploads.
- [x] Add container health check.
- [x] Verify production application starts successfully (Uvicorn health and page checks passed).

Docker verification: image built successfully; the container reports healthy, accepted a real defect-proxy upload, and retained its record after restart.

## 8. Documentation and submission evidence

- [x] Write README with setup, training, API, database, Docker, and troubleshooting instructions.
- [x] Add example API requests and response.
- [x] Add model methodology, evaluation metrics, limitations, and failure-case discussion.
- [x] Generate sample images covering all required conditions.
- [x] Add final requirement-to-implementation checklist.
- [x] Run complete test suite and record baseline results (updated setup suite: 15 passed; Python compilation pass).
- [x] Perform final clean-start/container smoke test, real upload, health check, and restart-persistence check.
- [x] Update this TODO so every completed checkbox reflects verified work.

## 9. Real-dataset model replacement (active)

- [x] Record dataset sources, licenses, published sizes, and reproducible download commands in `datasets/SOURCES.md` and download scripts.
- [x] Download and validate KADID-10k: exact 3,067,408,471-byte archive, 10,125 labeled distorted images, 81 pristine images, and 10,125 DMOS rows.
- [x] Download and validate MVTec AD images and defect labels. (5,354 PNGs, samples.json SHA-256 recorded in `datasets/manifests/integrity.json`.)
- [x] Add CUDA-enabled PyTorch dependencies and verify RTX 3050 forward/backward training with PyTorch 2.11.0+cu128.
- [x] Create a self-contained training/testing runbook and Claude Code handoff prompt.
- [x] Implement grouped, leakage-safe real-dataset manifests and train/validation/test splits. (15,479 records; KADID grouped by reference; MVTec stratified by category; `real_dataset.json` written.)
- [x] Implement MobileNetV3-Small multi-task training with mixed precision and resumable checkpoints. (12 epochs, early stopped at epoch 6 best; AMP FP16; checkpoint in `artifacts/checkpoints/real/`.)
- [x] Benchmark candidate checkpoints and select the best model by validation metrics. (Only MobileNetV3-Small evaluated; EfficientNet-B0 crashed with `cudaErrorIllegalAddress` on RTX 3050; best val sel=0.719 at epoch 6.)
- [x] Evaluate the selected model on untouched real-dataset test images. (Quality MAE=13.96 ✅; PD ROC-AUC=0.924 ✅; macro-F1=0.658 ❌ gate fails — underexposure/overexposure F1 low due to 4% class imbalance; 10 failure cases documented in `artifacts/real_metrics.json`.)
- [x] Run inference on separately downloaded real-life example images. (7 held-out real examples exported to `real_examples/results.json`.)
- [ ] Replace the synthetic Random Forest artifact only after the real model passes acceptance checks. (BLOCKED: macro-F1 gate not met — `.joblib` remains default; `.pt` ready at `artifacts/image_quality_model.pt` for manual promotion.)
- [x] Update API inference compatibility, automated tests, model card, metrics, README, and `project.md`. (`app/model_service.py` and `app/service.py` updated for dual .pt/.joblib support; `tests/test_deep_model.py` added — all 27 tests pass; `artifacts/model_card.md` written.)
- [ ] Re-run full API, frontend-contract, Docker, and restart-persistence verification. (Pending promotion of `.pt` model.)

## Explicitly deferred until after real-data training

- [ ] Grad-CAM/localization heatmaps.
- [ ] Batch uploads, authentication, PostgreSQL, cloud deployment, CI/CD, and monitoring stack.
