# UAV Edge Microservice DAG — v3 (Per-Service App Architecture)

## Architecture

10-node heterogeneous DAG for UAV multi-modal perception:

```
                    ┌─── rgb-preprocessor ─── rgb-detector ───┐
Gateway ───┤                                                   ├── feature-fusion ──┬── object-tracker ────┬── decision-maker ── telemetry-dashboard
                    └─── ir-preprocessor ──── ir-detector ────┘                     └── situation-awareness┘
```

## Key Changes in v3

- **Per-service app.py**: Each microservice has its own `app.py` — no more monolithic code
- **Per-service Docker images**: 10 individual images, each containing only the dependencies it needs
- **Fixed Jaeger image**: Uses `jaegertracing/jaeger:2.17.0` (matches local availability)
- **No cv2 in compute nodes**: gateway, feature-fusion, object-tracker, situation-awareness, decision-maker do NOT import cv2/onnxruntime

## Project Structure

```
├── services/                    # Per-service Python code
│   ├── common.py                # Shared utilities (OTel, Prometheus, FastAPI factory)
│   ├── gateway/app.py           # Gateway entry point (no cv2)
│   ├── rgb-preprocessor/app.py  # OpenCV resize+normalize (cv2)
│   ├── ir-preprocessor/app.py   # OpenCV resize+normalize (cv2)
│   ├── rgb-detector/app.py      # YOLOv8 ONNX inference (cv2 + onnxruntime)
│   ├── ir-detector/app.py       # YOLOv8 ONNX inference (cv2 + onnxruntime)
│   ├── feature-fusion/app.py    # Sync fusion with OTel Span Links (no cv2)
│   ├── object-tracker/app.py    # Matrix computation (no cv2)
│   ├── situation-awareness/app.py # Matrix computation (no cv2)
│   ├── decision-maker/app.py    # Sync fusion with OTel Span Links (no cv2)
│   └── telemetry-dashboard/app.py # Web UI + CSV output (no cv2)
├── dockerfiles/
│   ├── Dockerfile.base          # Python 3.10 + FastAPI + OTel + Prometheus
│   ├── Dockerfile.compute       # Base only — for gateway/fusion/tracker/SA/DM
│   ├── Dockerfile.preprocessor  # Base + opencv — for RGB/IR preprocessors
│   ├── Dockerfile.detector      # Base + opencv + onnxruntime + YOLOv8n — for detectors
│   └── Dockerfile.dashboard     # Base only — for telemetry dashboard
├── requirements/
│   ├── base.txt                 # FastAPI, httpx, numpy, OTel, Prometheus
│   ├── preprocessor.txt         # opencv-python-headless
│   └── detector.txt             # opencv-python-headless + onnxruntime
├── k8s/
│   ├── services/                # 10 individual K8s Deployment+Service YAMLs
│   │   ├── 01-gateway.yaml
│   │   ├── 02-rgb-preprocessor.yaml
│   │   ├── ...
│   │   └── 10-telemetry-dashboard.yaml
│   └── observability/
│       └── observability-stack.yaml  # Jaeger 2.17.0 + Prometheus + Kubenurse
├── scripts/
│   ├── build_images.sh          # Build all 10 per-service images
│   ├── distribute_images.sh     # Export tar.gz + SCP + ctr import
│   ├── deploy.sh                # kubectl apply all YAMLs
│   └── collect_logs.sh          # Grep CSV_RESULT from telemetry-dashboard
└── src/
    └── traffic_gen.py           # Image POST client with configurable FPS
```

## Dependency Matrix

| Service | Image Tier | cv2 | onnxruntime | numpy |
|---------|-----------|-----|------------|-------|
| gateway | compute | - | - | yes |
| rgb-preprocessor | preprocessor | yes | - | yes |
| ir-preprocessor | preprocessor | yes | - | yes |
| rgb-detector | detector | yes | yes | yes |
| ir-detector | detector | yes | yes | yes |
| feature-fusion | compute | - | - | - |
| object-tracker | compute | - | - | yes |
| situation-awareness | compute | - | - | yes |
| decision-maker | compute | - | - | - |
| telemetry-dashboard | dashboard | - | - | - |

## Deployment Steps

### 1. Build images (on master node)

```bash
./scripts/build_images.sh
```

This builds:
- 1 base image (`uav-edge-base`)
- 5 compute images (gateway, feature-fusion, object-tracker, situation-awareness, decision-maker)
- 2 preprocessor images (rgb-preprocessor, ir-preprocessor)
- 2 detector images (rgb-detector, ir-detector)
- 1 dashboard image (telemetry-dashboard)

### 2. Distribute images to worker nodes

```bash
# Edit SSH_USER and node hostnames in the script first
./scripts/distribute_images.sh
```

### 3. Label nodes

```bash
kubectl label node <master-node> node-role.kubernetes.io/master=true --overwrite
kubectl label node <node1> kubernetes.io/hostname=uav-node1 --overwrite
kubectl label node <node2> kubernetes.io/hostname=uav-node2 --overwrite
kubectl label node <node3> kubernetes.io/hostname=uav-node3 --overwrite
```

### 4. Deploy

```bash
./scripts/deploy.sh
```

### 5. Send traffic

```bash
python src/traffic_gen.py \
    --gateway http://<GATEWAY_ClusterIP>:8001 \
    --fps 5 --total 100 --image-dir ./test_images
```

### 6. Collect experiment results

```bash
./scripts/collect_logs.sh experiment_results.csv
```

## CSV Output Format (25 columns)

```
request_id, e2e_ms, rgb_branch_ms, ir_branch_ms,
gateway_compute_ms, rgb_preproc_compute_ms, ir_preproc_compute_ms,
rgb_detect_compute_ms, ir_detect_compute_ms, fusion_compute_ms,
tracker_compute_ms, sa_compute_ms, decision_compute_ms, dashboard_compute_ms,
net_gw_rgb, net_gw_ir, net_rgb_rgbd, net_ir_ird,
net_rgbd_fusion, net_ird_fusion, net_fusion_tracker, net_fusion_sa,
net_tracker_dm, net_sa_dm, net_dm_dashboard
```

## Node Distribution

| Node | CPU/RAM | Services |
|------|---------|----------|
| Master | 16 CPU / 32 GB | gateway, feature-fusion, decision-maker, telemetry-dashboard, Jaeger, Prometheus |
| Node1 (UAV) | 2 CPU / 4 GB | rgb-preprocessor, ir-detector |
| Node2 (UAV) | 1 CPU / 2 GB | ir-preprocessor, situation-awareness |
| Node3 (UAV) | 4 CPU / 8 GB | rgb-detector, object-tracker |
