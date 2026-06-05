# UAV Edge Microservice DAG — Heterogeneous Edge Computing Platform

A production-grade, 10-node microservice DAG for UAV swarm multi-modal perception research. Designed for IEEE Transactions-level experiments on K3s heterogeneous edge clusters.

## Architecture

```
                    ┌──────────┐
                    │  MS-1    │  Gateway
                    │ (Master) │
                    └────┬─────┘
                   ┌─────┴─────┐
              ┌────▼───┐  ┌────▼───┐
              │  MS-2  │  │  MS-3  │  RGB / IR Pre-processor
              │(Node1) │  │(Node2) │
              └────┬───┘  └────┬───┘
              ┌────▼───┐  ┌────▼───┐
              │  MS-4  │  │  MS-5  │  YOLOv8 ONNX Detector
              │(Node3) │  │(Node1) │
              └────┬───┘  └────┬───┘
                   └─────┬─────┘
                    ┌────▼─────┐
                    │  MS-6    │  Feature Fusion (Span Links)
                    │ (Master) │
                    └────┬─────┘
                   ┌─────┴─────┐
              ┌────▼───┐  ┌────▼───┐
              │  MS-7  │  │  MS-8  │  Tracker / Situation Awareness
              │(Node3) │  │(Node2) │
              └────┬───┘  └────┬───┘
                   └─────┬─────┘
                    ┌────▼─────┐
                    │  MS-9    │  Decision Maker (Span Links)
                    │ (Master) │
                    └────┬─────┘
                    ┌────▼─────┐
                    │  MS-10   │  Telemetry & CSV Output
                    │ (Master) │
                    └──────────┘
```

## Key Features

- **Real Computer Vision**: OpenCV preprocessing + YOLOv8n ONNX inference (no `time.sleep`)
- **Async-safe**: All CPU-bound work runs via `run_in_executor()` — no event loop blocking
- **ONNX Thread Control**: `intra/inter_op_num_threads=1` prevents context-switch storms on constrained UAV nodes
- **OpenTelemetry**: Full distributed tracing with Span Links at fusion nodes (MS-6, MS-9)
- **Prometheus + Kubenurse**: Metrics collection with physical network RTT measurement
- **CSV Telemetry**: MS-10 outputs structured per-request latency breakdowns

## Quick Start

### 1. Build Docker Image
```bash
docker build -t uav-edge-ms:latest .
```

### 2. Deploy to K3s
```bash
# Label your nodes first
kubectl label node <master-node> node-role.kubernetes.io/master=true
kubectl label node <node1> kubernetes.io/hostname=uav-node1
kubectl label node <node2> kubernetes.io/hostname=uav-node2
kubectl label node <node3> kubernetes.io/hostname=uav-node3

# Deploy everything
chmod +x scripts/deploy.sh
./scripts/deploy.sh --image uav-edge-ms:latest

# Deploy observability stack
kubectl apply -f k8s/observability-stack.yaml
```

### 3. Run Traffic Generator
```bash
pip install httpx opencv-python-headless numpy
python src/traffic_gen.py --gateway http://<GATEWAY_IP>:8001 --fps 5 --total 100
```

### 4. Collect Results
```bash
chmod +x scripts/collect_logs.sh
./scripts/collect_logs.sh --output experiment_results.csv
```

## CSV Output Format

Each row from MS-10 contains:
```
request_id, e2e_ms, rgb_branch_ms, ir_branch_ms,
ms1-ms10_compute_ms (10 values),
network_latency for each DAG edge (11 values)
```

## Network Simulation (tc)

```bash
# On UAV nodes, simulate dynamic bandwidth/RTT:
sudo tc qdisc add dev eth0 root netem delay 20ms 5ms distribution normal rate 10mbit
```

## Project Structure

```
├── src/
│   ├── app.py              # Unified data-driven microservice
│   └── traffic_gen.py      # Traffic generator client
├── k8s/
│   ├── ms-template.yaml    # Deployment template
│   ├── observability-stack.yaml  # Jaeger + Prometheus + Kubenurse
│   └── generated/          # Auto-generated manifests (by deploy.sh)
├── scripts/
│   ├── deploy.sh           # One-click deployment
│   └── collect_logs.sh     # CSV log collector
├── Dockerfile
├── requirements.txt
└── README.md
```
