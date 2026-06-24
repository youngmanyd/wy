#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Deploy all UAV Edge microservices and observability stack to K3s
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "UAV Edge Microservices K3s Deployment"
echo "=========================================="

# Step 1: Create namespace
echo ""
echo "[1/4] Creating namespace uav-edge..."
kubectl create namespace uav-edge 2>/dev/null || echo "  Namespace already exists"

# Step 2: Deploy observability stack
echo ""
echo "[2/4] Deploying observability stack..."
kubectl apply -f "$PROJECT_DIR/k8s/observability/observability-stack.yaml"

# Step 3: Deploy microservices
echo ""
echo "[3/4] Deploying 10 microservices..."
for yaml in "$PROJECT_DIR"/k8s/services/*.yaml; do
    svc_name=$(basename "$yaml" .yaml)
    echo "  -> $svc_name"
    kubectl apply -f "$yaml"
done

# Step 4: Verify
echo ""
echo "[4/4] Waiting for pods to be ready..."
sleep 5
echo ""
echo "Pod status:"
kubectl get pods -n uav-edge -o wide
echo ""
echo "Services:"
kubectl get svc -n uav-edge
echo ""
echo "=========================================="
echo "Deployment complete!"
echo ""
echo "Dashboard access:"
echo "  kubectl port-forward -n uav-edge svc/telemetry-dashboard 8010:8010"
echo "  Then open http://localhost:8010"
echo ""
echo "Jaeger UI:"
echo "  kubectl port-forward -n uav-edge svc/jaeger 16686:16686"
echo "  Then open http://localhost:16686"
echo ""
echo "Prometheus:"
echo "  kubectl port-forward -n uav-edge svc/prometheus 9090:9090"
echo "  Then open http://localhost:9090"
echo "=========================================="
