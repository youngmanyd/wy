#!/usr/bin/env bash
# =============================================================================
# UAV Edge — K3s One-Click Deployment Script
# Applies all microservice YAMLs and observability stack.
#
# Usage:
#   ./deploy.sh [--namespace uav-edge] [--dry-run]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="${NAMESPACE:-uav-edge}"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="--dry-run=client"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  UAV Edge — K3s Deployment"
echo "  Namespace: ${NAMESPACE}"
echo "============================================================"

# Create namespace if not exists
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo "--- Deploying Observability Stack ---"
kubectl apply -f "${PROJECT_DIR}/k8s/observability/" -n "$NAMESPACE" $DRY_RUN
echo ""

echo "--- Deploying Microservices (10 services) ---"
for yaml in "${PROJECT_DIR}"/k8s/services/*.yaml; do
    svc_name=$(basename "$yaml" .yaml)
    echo "  Applying: ${svc_name}"
    kubectl apply -f "$yaml" $DRY_RUN
done
echo ""

echo "--- Waiting for Deployments to become Ready ---"
SERVICES=(
    gateway rgb-preprocessor ir-preprocessor
    rgb-detector ir-detector feature-fusion
    object-tracker situation-awareness decision-maker
    telemetry-dashboard
)
for svc in "${SERVICES[@]}"; do
    echo -n "  Waiting for ${svc}... "
    kubectl rollout status deployment/"$svc" -n "$NAMESPACE" --timeout=120s 2>/dev/null && echo "Ready" || echo "TIMEOUT"
done

echo ""
echo "============================================================"
echo "  Deployment Complete!"
echo "============================================================"
echo ""
echo "Service Status:"
kubectl get pods -n "$NAMESPACE" -o wide | head -20
echo ""
echo "Access Points:"
echo "  Dashboard:    kubectl port-forward svc/telemetry-dashboard 8010:8010 -n $NAMESPACE"
echo "  Jaeger UI:    kubectl port-forward svc/jaeger 16686:16686 -n $NAMESPACE"
echo "  Prometheus:   kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE"
echo "  Gateway API:  kubectl port-forward svc/gateway 8001:8001 -n $NAMESPACE"
