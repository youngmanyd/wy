#!/usr/bin/env bash
# =============================================================================
# build_images.sh — Build per-service Docker images for UAV Edge microservices
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Building UAV Edge per-service Docker images"
echo "Project root: $PROJECT_DIR"
echo "=========================================="

cd "$PROJECT_DIR"

# Step 1: Build base image
echo ""
echo "[1/6] Building base image: uav-edge-base:latest"
docker build -t uav-edge-base:latest -f dockerfiles/Dockerfile.base .

# Step 2: Build compute-tier services (no cv2, no onnx)
# gateway, feature-fusion, object-tracker, situation-awareness, decision-maker
COMPUTE_SERVICES=("gateway" "feature-fusion" "object-tracker" "situation-awareness" "decision-maker")
echo ""
echo "[2/6] Building compute-tier services (${#COMPUTE_SERVICES[@]} images)..."
for svc in "${COMPUTE_SERVICES[@]}"; do
    echo "  -> uav-edge-${svc}:latest"
    docker build -t "uav-edge-${svc}:latest" \
        --build-arg BASE_IMAGE=uav-edge-base:latest \
        --build-arg SERVICE_NAME="${svc}" \
        -f dockerfiles/Dockerfile.compute .
done

# Step 3: Build preprocessor-tier services (with cv2)
# rgb-preprocessor, ir-preprocessor
PREPROC_SERVICES=("rgb-preprocessor" "ir-preprocessor")
echo ""
echo "[3/6] Building preprocessor-tier services (${#PREPROC_SERVICES[@]} images)..."
for svc in "${PREPROC_SERVICES[@]}"; do
    echo "  -> uav-edge-${svc}:latest"
    docker build -t "uav-edge-${svc}:latest" \
        --build-arg BASE_IMAGE=uav-edge-base:latest \
        --build-arg SERVICE_NAME="${svc}" \
        -f dockerfiles/Dockerfile.preprocessor .
done

# Step 4: Build detector-tier services (with cv2 + onnxruntime + YOLOv8n model)
# rgb-detector, ir-detector
DETECTOR_SERVICES=("rgb-detector" "ir-detector")
echo ""
echo "[4/6] Building detector-tier services (${#DETECTOR_SERVICES[@]} images)..."
for svc in "${DETECTOR_SERVICES[@]}"; do
    echo "  -> uav-edge-${svc}:latest"
    docker build -t "uav-edge-${svc}:latest" \
        --build-arg BASE_IMAGE=uav-edge-base:latest \
        --build-arg SERVICE_NAME="${svc}" \
        -f dockerfiles/Dockerfile.detector .
done

# Step 5: Build dashboard
echo ""
echo "[5/6] Building dashboard image: uav-edge-telemetry-dashboard:latest"
docker build -t "uav-edge-telemetry-dashboard:latest" \
    --build-arg BASE_IMAGE=uav-edge-base:latest \
    --build-arg SERVICE_NAME="telemetry-dashboard" \
    -f dockerfiles/Dockerfile.dashboard .

# Step 6: Summary
echo ""
echo "[6/6] Build complete! All images:"
echo "=========================================="
ALL_SERVICES=("gateway" "rgb-preprocessor" "ir-preprocessor" "rgb-detector" "ir-detector" \
              "feature-fusion" "object-tracker" "situation-awareness" "decision-maker" "telemetry-dashboard")
for svc in "${ALL_SERVICES[@]}"; do
    SIZE=$(docker image inspect "uav-edge-${svc}:latest" --format='{{.Size}}' 2>/dev/null || echo "0")
    SIZE_MB=$(echo "scale=1; $SIZE / 1048576" | bc 2>/dev/null || echo "?")
    echo "  uav-edge-${svc}:latest  (${SIZE_MB} MB)"
done
echo "=========================================="
echo "Done."
