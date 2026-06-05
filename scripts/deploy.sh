#!/usr/bin/env bash
# =============================================================================
# UAV Edge Microservice — One-Click K3s Deployment Script
# Generates 10 microservice manifests from ms-template.yaml and applies them.
#
# Usage:
#   ./deploy.sh [--namespace uav-edge] [--image myregistry/uav-ms:latest]
#   ./deploy.sh --delete   # Tear down all resources
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEMPLATE="${PROJECT_DIR}/k8s/ms-template.yaml"
GEN_DIR="${PROJECT_DIR}/k8s/generated"

# Defaults
NAMESPACE="${NAMESPACE:-uav-edge}"
IMAGE_REPO="${IMAGE_REPO:-uav-edge-ms}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DELETE_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --image)
            IMAGE_REPO="${2%:*}"
            IMAGE_TAG="${2#*:}"
            shift 2 ;;
        --delete) DELETE_MODE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if $DELETE_MODE; then
    echo "🗑  Deleting namespace ${NAMESPACE}..."
    kubectl delete namespace "$NAMESPACE" --ignore-not-found
    echo "Done."
    exit 0
fi

echo "============================================================"
echo "  UAV Edge Microservice Deployment"
echo "  Namespace:  ${NAMESPACE}"
echo "  Image:      ${IMAGE_REPO}:${IMAGE_TAG}"
echo "============================================================"

# Create namespace
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Create output directory
mkdir -p "$GEN_DIR"

# ---------------------------------------------------------------------------
# Microservice definitions
# Each line: MS_NAME|MS_ROLE|MS_PORT|REPLICAS|CPU_REQ|CPU_LIM|MEM_REQ|MEM_LIM|DOWNSTREAM_URLS|NODE_HINT|OMP_THREADS
# NODE_HINT: master / node1 / node2 / node3 / any
# ---------------------------------------------------------------------------
# DAG topology:
#   ms-1 -> ms-2, ms-3
#   ms-2 -> ms-4
#   ms-3 -> ms-5
#   ms-4 -> ms-6
#   ms-5 -> ms-6
#   ms-6 -> ms-7, ms-8
#   ms-7 -> ms-9
#   ms-8 -> ms-9
#   ms-9 -> ms-10
# ---------------------------------------------------------------------------

SVC_BASE="http://%s.${NAMESPACE}.svc.cluster.local:%s"

declare -a MS_DEFS=(
  # Gateway on Master (plenty of resources)
  "ms-1|ms-1|8001|1|200m|500m|256Mi|512Mi|$(printf "$SVC_BASE" ms-2 8002),$(printf "$SVC_BASE" ms-3 8003)|master|1"
  # RGB Pre-processor on Node1 (2 CPU / 4GB)
  "ms-2|ms-2|8002|1|200m|500m|256Mi|512Mi|$(printf "$SVC_BASE" ms-4 8004)|node1|1"
  # IR Pre-processor on Node2 (1 CPU / 2GB — most constrained)
  "ms-3|ms-3|8003|1|100m|300m|128Mi|384Mi|$(printf "$SVC_BASE" ms-5 8005)|node2|1"
  # RGB Detector on Node3 (4 CPU / 8GB — most capable UAV)
  "ms-4|ms-4|8004|1|500m|1000m|512Mi|1024Mi|$(printf "$SVC_BASE" ms-6 8006)|node3|1"
  # IR Detector on Node1 (share with RGB pre-proc)
  "ms-5|ms-5|8005|1|300m|800m|384Mi|768Mi|$(printf "$SVC_BASE" ms-6 8006)|node1|1"
  # Feature Fusion on Master
  "ms-6|ms-6|8006|1|200m|500m|256Mi|512Mi|$(printf "$SVC_BASE" ms-7 8007),$(printf "$SVC_BASE" ms-8 8008)|master|1"
  # Object Tracker on Node3
  "ms-7|ms-7|8007|1|300m|600m|256Mi|512Mi|$(printf "$SVC_BASE" ms-9 8009)|node3|1"
  # Situation Awareness on Node2
  "ms-8|ms-8|8008|1|100m|300m|128Mi|384Mi|$(printf "$SVC_BASE" ms-9 8009)|node2|1"
  # Decision Maker on Master
  "ms-9|ms-9|8009|1|200m|500m|256Mi|512Mi|$(printf "$SVC_BASE" ms-10 8010)|master|1"
  # Telemetry Aggregator on Master
  "ms-10|ms-10|8010|1|100m|300m|128Mi|256Mi||master|1"
)

# Map NODE_HINT to actual nodeSelector YAML
node_selector() {
    local hint="$1"
    case "$hint" in
        master)
            echo "nodeSelector:
        node-role.kubernetes.io/master: \"true\""
            ;;
        node1)
            echo "nodeSelector:
        kubernetes.io/hostname: \"uav-node1\""
            ;;
        node2)
            echo "nodeSelector:
        kubernetes.io/hostname: \"uav-node2\""
            ;;
        node3)
            echo "nodeSelector:
        kubernetes.io/hostname: \"uav-node3\""
            ;;
        *)
            echo "# no nodeSelector (any node)"
            ;;
    esac
}

echo ""
echo "Generating and applying 10 microservice manifests..."
echo ""

for def in "${MS_DEFS[@]}"; do
    IFS='|' read -r MS_NAME MS_ROLE MS_PORT REPLICAS CPU_REQ CPU_LIM MEM_REQ MEM_LIM DOWNSTREAM_URLS NODE_HINT OMP_THREADS <<< "$def"

    NODE_SEL="$(node_selector "$NODE_HINT")"
    OUTPUT="${GEN_DIR}/${MS_NAME}.yaml"

    sed \
        -e "s|\${MS_NAME}|${MS_NAME}|g" \
        -e "s|\${MS_ROLE}|${MS_ROLE}|g" \
        -e "s|\${MS_PORT}|${MS_PORT}|g" \
        -e "s|\${REPLICAS}|${REPLICAS}|g" \
        -e "s|\${CPU_REQ}|${CPU_REQ}|g" \
        -e "s|\${CPU_LIM}|${CPU_LIM}|g" \
        -e "s|\${MEM_REQ}|${MEM_REQ}|g" \
        -e "s|\${MEM_LIM}|${MEM_LIM}|g" \
        -e "s|\${DOWNSTREAM_URLS}|${DOWNSTREAM_URLS}|g" \
        -e "s|\${OMP_THREADS}|${OMP_THREADS}|g" \
        -e "s|\${IMAGE_REPO}|${IMAGE_REPO}|g" \
        -e "s|\${IMAGE_TAG}|${IMAGE_TAG}|g" \
        -e "s|\${NAMESPACE}|${NAMESPACE}|g" \
        "$TEMPLATE" > "$OUTPUT"

    # Replace NODE_SELECTOR block (multi-line)
    # Use python for reliable multi-line replacement
    python3 -c "
import sys
with open('${OUTPUT}', 'r') as f:
    content = f.read()
content = content.replace('\${NODE_SELECTOR}', '''${NODE_SEL}''')
with open('${OUTPUT}', 'w') as f:
    f.write(content)
"

    echo "  [+] ${MS_NAME} (role=${MS_ROLE}, port=${MS_PORT}, node=${NODE_HINT}, cpu=${CPU_LIM}, mem=${MEM_LIM})"
    kubectl apply -f "$OUTPUT"
done

echo ""
echo "============================================================"
echo "  All 10 microservices deployed to namespace: ${NAMESPACE}"
echo "============================================================"
echo ""
echo "Verify with:"
echo "  kubectl -n ${NAMESPACE} get pods -o wide"
echo "  kubectl -n ${NAMESPACE} get svc"
echo ""
echo "View logs:"
echo "  kubectl -n ${NAMESPACE} logs -l tier=uav-edge -f --max-log-requests=10"
echo ""
echo "Run traffic generator:"
echo "  GATEWAY_IP=\$(kubectl -n ${NAMESPACE} get svc ms-1 -o jsonpath='{.spec.clusterIP}')"
echo "  python traffic_gen.py --gateway http://\${GATEWAY_IP}:8001 --fps 5 --total 50"
