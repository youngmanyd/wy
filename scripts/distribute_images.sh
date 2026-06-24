#!/usr/bin/env bash
# =============================================================================
# distribute_images.sh — Export Docker images as tar.gz and import via ctr
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="${SCRIPT_DIR}/../dist"
mkdir -p "$DIST_DIR"

# Node list — edit these to match your environment
NODES=("uav-node1" "uav-node2" "uav-node3")
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-}"
REMOTE_TMP="/tmp/uav-images"

# Map services to target nodes
declare -A SVC_NODES
SVC_NODES["gateway"]="MASTER"
SVC_NODES["rgb-preprocessor"]="uav-node1"
SVC_NODES["ir-preprocessor"]="uav-node2"
SVC_NODES["rgb-detector"]="uav-node3"
SVC_NODES["ir-detector"]="uav-node1"
SVC_NODES["feature-fusion"]="MASTER"
SVC_NODES["object-tracker"]="uav-node3"
SVC_NODES["situation-awareness"]="uav-node2"
SVC_NODES["decision-maker"]="MASTER"
SVC_NODES["telemetry-dashboard"]="MASTER"

ALL_SERVICES=("gateway" "rgb-preprocessor" "ir-preprocessor" "rgb-detector" "ir-detector" \
              "feature-fusion" "object-tracker" "situation-awareness" "decision-maker" "telemetry-dashboard")

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
[ -n "$SSH_KEY" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

echo "=========================================="
echo "Exporting Docker images to tar.gz"
echo "=========================================="

for svc in "${ALL_SERVICES[@]}"; do
    IMG="uav-edge-${svc}:latest"
    TAR="${DIST_DIR}/uav-edge-${svc}.tar.gz"
    if [ -f "$TAR" ]; then
        echo "  [skip] $TAR already exists"
    else
        echo "  [export] $IMG -> $TAR"
        docker save "$IMG" | gzip > "$TAR"
    fi
done

echo ""
echo "=========================================="
echo "Importing images on local master (ctr)"
echo "=========================================="
for svc in "${ALL_SERVICES[@]}"; do
    NODE="${SVC_NODES[$svc]}"
    if [ "$NODE" == "MASTER" ]; then
        TAR="${DIST_DIR}/uav-edge-${svc}.tar.gz"
        echo "  [import] uav-edge-${svc}:latest on MASTER"
        sudo ctr -n k8s.io images import <(gunzip -c "$TAR") 2>/dev/null || \
        sudo k3s ctr images import <(gunzip -c "$TAR") 2>/dev/null || \
        echo "    WARNING: ctr import failed for $svc on master"
    fi
done

echo ""
echo "=========================================="
echo "Distributing images to worker nodes via SCP"
echo "=========================================="
for svc in "${ALL_SERVICES[@]}"; do
    NODE="${SVC_NODES[$svc]}"
    if [ "$NODE" != "MASTER" ]; then
        TAR="${DIST_DIR}/uav-edge-${svc}.tar.gz"
        echo "  [scp] $TAR -> ${NODE}:${REMOTE_TMP}/"
        ssh $SSH_OPTS "${SSH_USER}@${NODE}" "mkdir -p ${REMOTE_TMP}" 2>/dev/null || true
        scp $SSH_OPTS "$TAR" "${SSH_USER}@${NODE}:${REMOTE_TMP}/" 2>/dev/null
        echo "  [import] ctr import on ${NODE}"
        ssh $SSH_OPTS "${SSH_USER}@${NODE}" \
            "gunzip -c ${REMOTE_TMP}/uav-edge-${svc}.tar.gz | sudo ctr -n k8s.io images import - || \
             gunzip -c ${REMOTE_TMP}/uav-edge-${svc}.tar.gz | sudo k3s ctr images import -" 2>/dev/null || \
            echo "    WARNING: import failed for $svc on $NODE"
    fi
done

echo ""
echo "=========================================="
echo "Done. Images distributed to all nodes."
echo "=========================================="
