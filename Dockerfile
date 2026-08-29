FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    DATABASE_PATH=/app/data/quality.db \
    UPLOAD_DIR=/app/data/uploads \
    MODEL_PATH=/app/artifacts/image_quality_model.pt \
    METRICS_PATH=/app/artifacts/metrics.json

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY artifacts/image_quality_model.pt artifacts/image_quality_model.joblib artifacts/metrics.json artifacts/model_card.md ./artifacts/
RUN mkdir -p /app/data/uploads

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

