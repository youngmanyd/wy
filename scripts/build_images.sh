#!/usr/bin/env bash
# =============================================================================
# UAV Edge — Layered Docker Image Build Script
# Builds 4 layered images and exports them as tar archives for offline transfer.
#
# Usage:
#   ./build_images.sh [--export-dir ./images] [--no-export]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXPORT_DIR="${EXPORT_DIR:-${PROJECT_DIR}/images}"
NO_EXPORT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --export-dir) EXPORT_DIR="$2"; shift 2 ;;
        --no-export) NO_EXPORT=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  UAV Edge — Layered Image Build"
echo "  Project:    ${PROJECT_DIR}"
echo "  Export Dir: ${EXPORT_DIR}"
echo "============================================================"

cd "$PROJECT_DIR"

# --- Step 1: Build base image ---
echo ""
echo "[1/4] Building base image: uav-edge-base:latest"
docker build -t uav-edge-base:latest -f dockerfiles/Dockerfile.base .

# --- Step 2: Build compute image (gateway, fusion, tracker, SA, decision) ---
echo ""
echo "[2/4] Building compute image: uav-edge-compute:latest"
docker build -t uav-edge-compute:latest \
    --build-arg BASE_IMAGE=uav-edge-base:latest \
    -f dockerfiles/Dockerfile.compute .

# --- Step 3: Build preprocessor image (rgb-preprocessor, ir-preprocessor) ---
echo ""
echo "[3/4] Building preprocessor image: uav-edge-preprocessor:latest"
docker build -t uav-edge-preprocessor:latest \
    --build-arg BASE_IMAGE=uav-edge-base:latest \
    -f dockerfiles/Dockerfile.preprocessor .

# --- Step 4: Build detector image (rgb-detector, ir-detector) ---
echo ""
echo "[4/4] Building detector image: uav-edge-detector:latest"
docker build -t uav-edge-detector:latest \
    --build-arg BASE_IMAGE=uav-edge-base:latest \
    -f dockerfiles/Dockerfile.detector .

# --- Step 5: Build dashboard image ---
echo ""
echo "[5/5] Building dashboard image: uav-edge-dashboard:latest"
docker build -t uav-edge-dashboard:latest \
    --build-arg BASE_IMAGE=uav-edge-base:latest \
    -f dockerfiles/Dockerfile.dashboard .

echo ""
echo "============================================================"
echo "  All images built successfully"
echo "============================================================"
docker images | grep uav-edge
echo ""

# --- Export as tar archives ---
if $NO_EXPORT; then
    echo "Skipping export (--no-export)."
    exit 0
fi

mkdir -p "$EXPORT_DIR"

echo "Exporting images as tar archives..."

declare -A IMAGE_MAP=(
    ["uav-edge-base"]="uav-edge-base:latest"
    ["uav-edge-compute"]="uav-edge-compute:latest"
    ["uav-edge-preprocessor"]="uav-edge-preprocessor:latest"
    ["uav-edge-detector"]="uav-edge-detector:latest"
    ["uav-edge-dashboard"]="uav-edge-dashboard:latest"
)

for name in "${!IMAGE_MAP[@]}"; do
    tag="${IMAGE_MAP[$name]}"
    tarfile="${EXPORT_DIR}/${name}.tar"
    echo "  Saving ${tag} -> ${tarfile}"
    docker save -o "$tarfile" "$tag"
    echo "    Size: $(du -h "$tarfile" | cut -f1)"
done

echo ""
echo "============================================================"
echo "  Export Complete"
echo "  Location: ${EXPORT_DIR}/"
echo "============================================================"
ls -lh "$EXPORT_DIR"/*.tar
echo ""
echo "Next step: run ./scripts/distribute_images.sh to deploy to UAV nodes"
