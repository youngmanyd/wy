# UAV Embodied Intelligence Operating System

低空无人机具身智能操作系统 v0.2 - Gazebo 真实仿真 + LLM 自然语言解析

## Architecture

```
                    ┌─────────────────────────────────┐
                    │  Natural Language / Mission YAML │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │  LLM Task Parser (Qwen2.5-7B-Instruct)  │
              │  ┌───────────────────────────────────┐   │
              │  │ OpenAI-compatible API → JSON       │   │
              │  │ Fallback: rule-based parsing       │   │
              │  └───────────────────────────────────┘   │
              └────────────────────┬────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────┐
│              Mission Parser (YAML + dict + NL)                  │
│              Capability Registry (10 capabilities)              │
├─────────────────────────────────────────────────────────────────┤
│              SayCan-style Orchestrator                          │
│              score = R × F × S × U × P                         │
├─────────────────────────────────────────────────────────────────┤
│              Safety Shield (hard/soft constraints)              │
├───────────────────┬──────────────────┬──────────────────────────┤
│  ROS2 Executor    │ ROS2 World Model │ Mission Logger           │
│  (AS2 DroneIf)    │ (real topics)    │ (JSONL + Markdown)       │
├───────────────────┼──────────────────┼──────────────────────────┤
│  Image Analyzer   │                  │                          │
│  (OpenCV quality) │                  │                          │
├───────────────────┴──────────────────┴──────────────────────────┤
│  PX4 SITL + Gazebo (baylands) + MicroXRCEAgent + Image Bridge  │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | File | Description |
|--------|------|-------------|
| Capability Registry | `capability_registry.py` | Loads & queries YAML-defined capabilities |
| Mission Parser | `mission_parser.py` | Parses YAML, dict (from LLM), NL; supports waypoints |
| Orchestrator | `orchestrator.py` | SayCan multiplicative scoring & plan execution |
| Safety Shield | `safety_shield.py` | Hard/soft constraint checking |
| **ROS2 World Model** | `ros2_world_model.py` | Subscribes to real Gazebo topics (pose, battery, GPS, camera) |
| **ROS2 Executor** | `ros2_executor.py` | AS2 DroneInterface flight control + image capture |
| **Image Analyzer** | `image_analyzer.py` | OpenCV image quality evaluation (blur, exposure, contrast) |
| **LLM Task Parser** | `llm_task_parser.py` | Qwen2.5 natural language → structured mission JSON |
| World Model (Mock) | `world_model.py` | Mock world model for unit testing |
| Executor (Mock) | `executor.py` | Mock executor for unit testing |
| Logger | `logger.py` | Waypoint-centric JSONL logs + markdown reports |
| World Model Base | `world_model_base.py` | Abstract interface for Mock/ROS2 world models |

## Installation

```bash
cd uav_embodied_os

# Base install (no ROS2/LLM)
pip install -e ".[dev]"

# ROS2 packages (installed via apt on your drone machine)
# sudo apt install ros-humble-cv-bridge
# pip install as2_python_api  (or via Aerostack2 workspace)
```

## Quick Start - Real Gazebo Simulation

### One-Click Launch (recommended)

```bash
./scripts/start_inspection.sh
```

This tmux script starts:
1. **PX4 SITL** (x500_depth, baylands world)
2. **MicroXRCEAgent** (PX4-ROS2 bridge, UDP port 8888)
3. **RGB image bridge** (Gazebo → `/camera/image_raw`)
4. **Depth image bridge** (Gazebo → `/camera/depth_raw`)
5. **Camera info bridge**
6. **Python inspection script** (auto-runs after systems settle)

### Manual Launch

```bash
# Terminal 1: PX4 SITL
cd ~/PX4-Autopilot
export PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001
export PX4_SIM_MODEL=gz_x500_depth PX4_GZ_WORLD=baylands
./build/px4_sitl_default/bin/px4

# Terminal 2: MicroXRCEAgent
MicroXRCEAgent udp4 -p 8888

# Terminal 3-5: Image bridges
ros2 run ros_gz_image image_bridge /world/baylands/model/gz_x500_depth/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image@gz.msgs.Image
ros2 run ros_gz_image image_bridge /world/baylands/model/gz_x500_depth/link/camera_link/sensor/depth_camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image
ros2 run ros_gz_bridge parameter_bridge /world/baylands/model/gz_x500_depth/link/camera_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo

# Terminal 6: Run inspection
python3 examples/run_campus_inspection.py
```

### Natural Language Mode (with LLM)

```bash
python3 examples/run_campus_inspection.py \
    --natural-language "巡检baylands的3个目标点，检测烟雾和屋顶异常，低质量图像自动复拍"
```

## LLM Configuration

Configure your Qwen2.5-7B-Instruct endpoint in `configs/llm_config.yaml`:

```yaml
llm:
  api_base_url: "http://<your-4090-server-ip>:8000/v1"
  api_key: "not-needed"      # vLLM default doesn't need key
  model_name: "Qwen2.5-7B-Instruct"
  temperature: 0.1
  max_tokens: 1024
  timeout_s: 30.0
```

**Deploy on your 4090 server:**

```bash
# Option 1: vLLM (recommended for performance)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000

# Option 2: Ollama (simpler setup)
ollama serve
ollama run qwen2.5:7b-instruct
# API at http://localhost:11434/v1
```

If LLM is unavailable, the parser automatically falls back to rule-based extraction (Chinese + English keywords).

## Mission Definition (Waypoint Format)

```yaml
kind: Mission
metadata:
  name: baylands_3point_inspection
  description: "Inspect 3 designated points in baylands"
spec:
  objective: fly_inspect_report
  area: baylands
  waypoints:
    - name: point_1
      position: [10.0, 10.0, 5.0]     # NED coordinates (meters)
      tasks: [fly_to_area, hold_position, scan_area, evaluate_image_quality]
    - name: point_2
      position: [30.0, 10.0, 5.0]
      tasks: [fly_to_area, hold_position, scan_area, evaluate_image_quality]
    - name: point_3
      position: [20.0, 30.0, 5.0]
      tasks: [fly_to_area, hold_position, scan_area, evaluate_image_quality]
  priority_targets: [smoke, crowd, rooftop_anomaly]
  constraints:
    max_altitude_m: 120.0
    reserve_battery_percent: 25.0
  policies:
    low_image_quality:
      action: reobserve_from_new_angle
      threshold: 0.6
```

## Image Quality Metrics

| Metric | Weight | Method | Threshold |
|--------|--------|--------|-----------|
| Blur | 50% | Laplacian variance | 100.0 (higher = sharper) |
| Exposure | 25% | Mean brightness deviation from 127.5 | 40-220 range |
| Contrast | 25% | Pixel intensity std dev | 30.0 minimum |

**Overall quality** = 0.5 * blur + 0.25 * exposure + 0.25 * contrast

If quality < 0.6 at a waypoint, the system automatically shifts position (+3m X, +1m Z) and recaptures.

## ROS2 Topics Used

| Topic | Type | Source |
|-------|------|--------|
| `/drone0/self_localization/pose` | PoseStamped | Drone position/heading |
| `/drone0/sensor_measurements/battery` | BatteryState | Battery percentage |
| `/drone0/sensor_measurements/gps` | NavSatFix | GPS quality |
| `/camera/image_raw` | Image | RGB camera (via bridge) |
| `/camera/depth_raw` | Image | Depth camera (via bridge) |

## Testing

```bash
# All 81 tests
pytest tests/ -v

# By module
pytest tests/test_image_analyzer.py -v      # 13 cases: blur/exposure/contrast scoring
pytest tests/test_llm_task_parser.py -v      # 17 cases: rule-based, validation, mock LLM
pytest tests/test_mission_parser_v2.py -v    # 7 cases: waypoint parsing, dict input
pytest tests/test_capability_registry.py -v  # 8 cases
pytest tests/test_safety_shield.py -v        # 7 cases
pytest tests/test_orchestrator.py -v         # 10 cases
pytest tests/test_integration.py -v          # 5 cases

# Coverage
pytest tests/ --cov=uav_eios --cov-report=term-missing
```

## Project Structure

```
uav_embodied_os/
├── configs/
│   ├── capabilities/           # 10 capability YAML definitions
│   ├── missions/
│   │   ├── baylands_3point_inspection.yaml  # Phase 2 real mission
│   │   └── campus_3_inspection.yaml         # Phase 1 mock mission
│   ├── safety/                 # Safety rule configs
│   └── llm_config.yaml        # LLM API endpoint config
├── src/uav_eios/
│   ├── ros2_world_model.py     # Real Gazebo topic subscriptions
│   ├── ros2_executor.py        # AS2 DroneInterface flight control
│   ├── image_analyzer.py       # OpenCV image quality evaluation
│   ├── llm_task_parser.py      # Qwen2.5 natural language parsing
│   ├── world_model_base.py     # Abstract WorldModel interface
│   ├── world_model.py          # Mock world model (for testing)
│   ├── executor.py             # Mock executor (for testing)
│   ├── orchestrator.py         # SayCan scoring engine
│   ├── mission_parser.py       # YAML/dict/NL mission parser
│   ├── capability_registry.py  # Capability definitions
│   ├── safety_shield.py        # Hard/soft safety constraints
│   └── logger.py               # Waypoint-centric logging
├── scripts/
│   └── start_inspection.sh     # One-click tmux launcher
├── tests/                      # 81 test cases
├── examples/
│   ├── run_campus_inspection.py       # Real Gazebo inspection (Phase 2)
│   └── run_campus_inspection_mock.py  # Mock demo (Phase 1)
├── outputs/                    # Generated logs and reports
├── pyproject.toml
└── README.md
```

## Environment Variables

```bash
# PX4 SITL
export ROS_DOMAIN_ID=3
export PX4_GZ_STANDALONE=1
export PX4_SYS_AUTOSTART=4001
export PX4_SIM_MODEL=gz_x500_depth
export PX4_GZ_WORLD=baylands
```
