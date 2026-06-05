# UAV Edge Microservice DAG — Phase 2

面向无人机集群微服务边缘计算的 10 节点异构 DAG 系统。  
基于 FastAPI + ONNXRuntime + OpenTelemetry 实现真实视觉推理与全链路时延追踪。

## 架构拓扑

```
                          ┌─────────────────┐
                          │    gateway       │  (Master)
                          │    Port 8001     │
                          └─────┬───────┬────┘
                                │       │
                 ┌──────────────┘       └──────────────┐
                 ▼                                     ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │  rgb-preprocessor   │  (Node1)     │  ir-preprocessor    │  (Node2)
    │  Port 8002          │              │  Port 8003          │
    └──────────┬──────────┘              └──────────┬──────────┘
               ▼                                    ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │  rgb-detector       │  (Node3)     │  ir-detector        │  (Node1)
    │  Port 8004 [ONNX]   │              │  Port 8005 [ONNX]   │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               └──────────────┬─────────────────────┘
                              ▼
                 ┌─────────────────────┐
                 │  feature-fusion     │  (Master) [Span Links]
                 │  Port 8006          │
                 └─────┬───────┬───────┘
                       │       │
          ┌────────────┘       └────────────┐
          ▼                                 ▼
 ┌──────────────────┐           ┌────────────────────────┐
 │  object-tracker  │ (Node3)   │  situation-awareness   │  (Node2)
 │  Port 8007       │           │  Port 8008             │
 └────────┬─────────┘           └──────────┬─────────────┘
          │                                │
          └──────────────┬─────────────────┘
                         ▼
              ┌─────────────────────┐
              │  decision-maker     │  (Master) [Span Links]
              │  Port 8009          │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ telemetry-dashboard │  (Master)
              │ Port 8010 [Web UI]  │
              └─────────────────────┘
```

## 异构节点资源分配

| 节点       | 硬件              | 部署的微服务                                       |
|-----------|-------------------|--------------------------------------------------|
| Master    | 16 CPU / 32GB RAM | gateway, feature-fusion, decision-maker, telemetry-dashboard |
| UAV Node1 | 2 CPU / 4GB RAM   | rgb-preprocessor, ir-detector                     |
| UAV Node2 | 1 CPU / 2GB RAM   | ir-preprocessor, situation-awareness              |
| UAV Node3 | 4 CPU / 8GB RAM   | rgb-detector, object-tracker                      |

## 镜像分层架构

```
uav-edge-base         (~150MB) — Python + FastAPI + OTel + Prometheus
  ├── uav-edge-compute     (~+5MB)  — gateway / fusion / tracker / SA / decision
  ├── uav-edge-preprocessor (~+80MB)  — + OpenCV (rgb/ir-preprocessor)
  ├── uav-edge-detector    (~+300MB) — + OpenCV + ONNXRuntime + YOLOv8n
  └── uav-edge-dashboard   (~+80MB)  — + OpenCV (telemetry-dashboard + Web UI)
```

## 项目结构

```
uav-edge-v2/
├── README.md
├── src/
│   ├── app.py                  # 统一数据驱动微服务 (SERVICE_ROLE 区分角色)
│   └── traffic_gen.py          # 真实图片流量发生器
├── dockerfiles/
│   ├── Dockerfile.base         # 基础镜像
│   ├── Dockerfile.preprocessor # 预处理器镜像 (+ OpenCV)
│   ├── Dockerfile.detector     # 检测器镜像 (+ ONNX + YOLOv8n)
│   ├── Dockerfile.compute      # 计算镜像 (轻量)
│   └── Dockerfile.dashboard    # 仪表盘镜像 (+ OpenCV)
├── requirements/
│   ├── base.txt                # 基础依赖
│   ├── preprocessor.txt        # OpenCV
│   └── detector.txt            # OpenCV + ONNXRuntime
├── k8s/
│   ├── services/               # 10 个独立的微服务 YAML
│   │   ├── 01-gateway.yaml
│   │   ├── 02-rgb-preprocessor.yaml
│   │   ├── 03-ir-preprocessor.yaml
│   │   ├── 04-rgb-detector.yaml
│   │   ├── 05-ir-detector.yaml
│   │   ├── 06-feature-fusion.yaml
│   │   ├── 07-object-tracker.yaml
│   │   ├── 08-situation-awareness.yaml
│   │   ├── 09-decision-maker.yaml
│   │   └── 10-telemetry-dashboard.yaml
│   └── observability/
│       └── observability-stack.yaml  # Jaeger + Prometheus + Kubenurse
└── scripts/
    ├── build_images.sh         # 镜像分层构建 + tar 导出
    ├── distribute_images.sh    # SCP + ctr 离线导入到 UAV 节点
    ├── deploy.sh               # K3s 一键部署
    └── collect_logs.sh         # CSV 实验数据收集
```

## 快速开始

### 1. 构建镜像

```bash
# 在有 Docker 的构建机上执行
chmod +x scripts/*.sh
./scripts/build_images.sh
```

构建结果：`images/` 目录下生成 5 个 `.tar` 文件。

### 2. 离线分发到 UAV 节点

```bash
# 配置节点地址
export MASTER_HOST=192.168.1.10
export NODE1_HOST=192.168.1.11
export NODE2_HOST=192.168.1.12
export NODE3_HOST=192.168.1.13
export SSH_USER=root

./scripts/distribute_images.sh
```

脚本会根据各节点所需的镜像类型，通过 SCP 传输对应的 tar 文件，然后通过 `ctr -n k8s.io image import` 导入到 K3s 的 containerd。

### 3. 部署到 K3s

```bash
# 确保节点已打标签
kubectl label node <master> node-role.kubernetes.io/master=true
kubectl label node <n1> kubernetes.io/hostname=uav-node1
kubectl label node <n2> kubernetes.io/hostname=uav-node2
kubectl label node <n3> kubernetes.io/hostname=uav-node3

# 一键部署
./scripts/deploy.sh
```

### 4. 发送流量

```bash
python src/traffic_gen.py \
    --gateway http://<GATEWAY_ClusterIP>:8001 \
    --fps 5 \
    --total 100
```

### 5. 查看仪表盘

```bash
kubectl port-forward svc/telemetry-dashboard 8010:8010 -n uav-edge
# 浏览器访问 http://localhost:8010
```

仪表盘功能：
- 实时 E2E 时延趋势图
- RGB/IR 分支时延对比柱状图
- 各服务计算耗时水平条形图
- DAG 各跳网络时延图
- 最近请求列表

### 6. 收集实验数据

```bash
./scripts/collect_logs.sh --output experiment_results.csv

# 实时跟踪模式
./scripts/collect_logs.sh --follow
```

### 7. 查看 Jaeger 链路追踪

```bash
kubectl port-forward svc/jaeger 16686:16686 -n uav-edge
# 浏览器访问 http://localhost:16686
```

## CSV 输出格式

```
request_id, e2e_ms, rgb_branch_ms, ir_branch_ms,
gateway_compute_ms, rgb_preprocessor_compute_ms, ..., telemetry_dashboard_compute_ms,
net_gateway→rgb_preproc_ms, ..., net_decision→dashboard_ms
```

共 25 列：1(ID) + 3(汇总时延) + 10(节点计算时延) + 11(网络跳延时延)

## Web Dashboard 截图

仪表盘通过 ECharts 实时展示：

| 组件 | 说明 |
|------|------|
| E2E Trend Chart | 端到端时延折线图，显示最近 50 个请求 |
| Branch Comparison | RGB vs IR 分支时延柱状对比 |
| Per-Service Compute | 各微服务计算耗时水平图 |
| Network Hop Chart | DAG 11 条边的网络传输时延 |
| Recent Requests Table | 最近 20 个请求的时延摘要 |

## 关键技术细节

- **ONNX 线程控制**：`intra_op_num_threads=1`, `inter_op_num_threads=1`, `OMP_NUM_THREADS=1`
- **异步无阻塞**：所有 CPU 密集操作通过 `run_in_executor()` 下放到线程池
- **OTel Span Links**：feature-fusion 和 decision-maker 通过 Span Links 关联两个上游 trace
- **离线部署**：`docker save` → SCP → `ctr -n k8s.io image import`
- **containerd 兼容**：K3s 默认使用 containerd，导入命名空间必须为 `k8s.io`
