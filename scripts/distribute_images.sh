#!/usr/bin/env bash
# =============================================================================
# UAV Edge — Offline Image Distribution Script
# Transfers pre-built tar images to edge nodes via SCP and imports via ctr.
#
# K3s uses containerd (ctr) as its container runtime.
# Images are imported into the k8s.io namespace for K3s compatibility.
#
# Usage:
#   ./distribute_images.sh [--images-dir ./images]
#
# Environment variables (configure your cluster):
#   MASTER_HOST   - Master node SSH address (default: master)
#   NODE1_HOST    - UAV Node1 SSH address (default: uav-node1)
#   NODE2_HOST    - UAV Node2 SSH address (default: uav-node2)
#   NODE3_HOST    - UAV Node3 SSH address (default: uav-node3)
#   SSH_USER      - SSH username (default: root)
#   SSH_KEY       - Path to SSH private key (optional)
#   REMOTE_TMP    - Remote temp directory (default: /tmp/uav-images)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGES_DIR="${IMAGES_DIR:-${PROJECT_DIR}/images}"

# Cluster node addresses (override with env vars)
MASTER_HOST="${MASTER_HOST:-master}"
NODE1_HOST="${NODE1_HOST:-uav-node1}"
NODE2_HOST="${NODE2_HOST:-uav-node2}"
NODE3_HOST="${NODE3_HOST:-uav-node3}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-}"
REMOTE_TMP="${REMOTE_TMP:-/tmp/uav-images}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images-dir) IMAGES_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build SSH command
SSH_CMD="ssh"
SCP_CMD="scp"
if [ -n "$SSH_KEY" ]; then
    SSH_CMD="ssh -i $SSH_KEY"
    SCP_CMD="scp -i $SSH_KEY"
fi

echo "============================================================"
echo "  UAV Edge — Offline Image Distribution"
echo "  Images Dir:  ${IMAGES_DIR}"
echo "  SSH User:    ${SSH_USER}"
echo "============================================================"

# Verify images exist
if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    echo "Run ./scripts/build_images.sh first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Node-to-image mapping:
#   Master: base, compute, dashboard (gateway, feature-fusion, decision-maker, telemetry-dashboard)
#   Node1:  base, preprocessor, detector (rgb-preprocessor, ir-detector)
#   Node2:  base, compute, preprocessor (ir-preprocessor, situation-awareness)
#   Node3:  base, compute, detector (rgb-detector, object-tracker)
# ---------------------------------------------------------------------------
declare -A NODE_IMAGES
NODE_IMAGES["$MASTER_HOST"]="uav-edge-base uav-edge-compute uav-edge-dashboard"
NODE_IMAGES["$NODE1_HOST"]="uav-edge-base uav-edge-preprocessor uav-edge-detector"
NODE_IMAGES["$NODE2_HOST"]="uav-edge-base uav-edge-compute uav-edge-preprocessor"
NODE_IMAGES["$NODE3_HOST"]="uav-edge-base uav-edge-compute uav-edge-detector"

# Transfer and import function
distribute_to_node() {
    local host="$1"
    local images="$2"

    echo ""
    echo "--- Distributing to ${host} ---"

    # Create remote temp dir
    $SSH_CMD ${SSH_USER}@${host} "mkdir -p ${REMOTE_TMP}" 2>/dev/null || true

    for img_name in $images; do
        local tarfile="${IMAGES_DIR}/${img_name}.tar"
        if [ ! -f "$tarfile" ]; then
            echo "  WARN: ${tarfile} not found, skipping"
            continue
        fi

        local size=$(du -h "$tarfile" | cut -f1)
        echo "  [SCP] ${img_name}.tar (${size}) -> ${host}:${REMOTE_TMP}/"
        $SCP_CMD "$tarfile" "${SSH_USER}@${host}:${REMOTE_TMP}/${img_name}.tar"

        echo "  [CTR] Importing ${img_name} into containerd (k8s.io namespace)..."
        $SSH_CMD ${SSH_USER}@${host} \
            "sudo ctr -n k8s.io image import ${REMOTE_TMP}/${img_name}.tar && \
             echo '  OK: ${img_name} imported' && \
             rm -f ${REMOTE_TMP}/${img_name}.tar"
    done

    # Verify images on node
    echo "  Verifying images on ${host}..."
    $SSH_CMD ${SSH_USER}@${host} "sudo ctr -n k8s.io images ls | grep uav-edge || true"
}

# Execute distribution
for host in "$MASTER_HOST" "$NODE1_HOST" "$NODE2_HOST" "$NODE3_HOST"; do
    images="${NODE_IMAGES[$host]}"
    distribute_to_node "$host" "$images"
done

echo ""
echo "============================================================"
echo "  Distribution Complete!"
echo "============================================================"
echo ""
echo "Image layout per node:"
echo "  Master ($MASTER_HOST): uav-edge-compute, uav-edge-dashboard"
echo "    -> gateway, feature-fusion, decision-maker, telemetry-dashboard"
echo ""
echo "  Node1 ($NODE1_HOST): uav-edge-preprocessor, uav-edge-detector"
echo "    -> rgb-preprocessor, ir-detector"
echo ""
echo "  Node2 ($NODE2_HOST): uav-edge-compute, uav-edge-preprocessor"
echo "    -> ir-preprocessor, situation-awareness"
echo ""
echo "  Node3 ($NODE3_HOST): uav-edge-compute, uav-edge-detector"
echo "    -> rgb-detector, object-tracker"
echo ""
echo "Next: kubectl apply -f k8s/services/ -f k8s/observability/"
