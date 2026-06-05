# Multi-stage Dockerfile for UAV Edge Microservice
# Optimized for ARM64/AMD64 heterogeneous edge nodes
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# System deps for OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/app.py .

# Create model directory
RUN mkdir -p /app/models

# Default environment (overridden by K8s env)
ENV MS_ROLE=ms-1 \
    MS_PORT=8000 \
    DOWNSTREAM_URLS="" \
    ONNX_MODEL_PATH=/app/models/yolov8n.onnx \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317 \
    PYTHONUNBUFFERED=1

# Download YOLOv8n ONNX model at build time
# Try multiple release versions for robustness
RUN curl -fSL -o /app/models/yolov8n.onnx \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx" || \
    curl -fSL -o /app/models/yolov8n.onnx \
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx" || \
    (pip install --no-cache-dir ultralytics && \
     python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); m.export(format='onnx', imgsz=640)" && \
     mv yolov8n.onnx /app/models/yolov8n.onnx && \
     pip uninstall -y ultralytics) || \
    echo "WARN: Model download failed — mount model at /app/models/yolov8n.onnx at runtime"

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:${MS_PORT}/health || exit 1

CMD ["python", "app.py"]
