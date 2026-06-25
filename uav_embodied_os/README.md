# UAV Embodied Intelligence Operating System

低空无人机具身智能操作系统 v0.3 - 原生 PX4 FMU 控制 + NED 坐标系 + LLM 自然语言解析

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
              │  │ NED Z-axis auto-correction         │   │
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
│  (Native FMU)     │ (FMU + Camera)   │ (JSONL + Markdown)       │
├───────────────────┼──────────────────┼──────────────────────────┤
│  Image Analyzer   │                  │                          │
│  (OpenCV quality) │                  │                          │
├───────────────────┴──────────────────┴──────────────────────────┤
│  PX4 SITL + Gazebo (baylands) + MicroXRCEAgent + Image Bridge  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. NED Coordinate System (Z-Down)

PX4 uses NED (North-East-Down) coordinate frame:
- **X** = North (positive forward)
- **Y** = East (positive right)
- **Z** = Down (positive downward)

**Flying altitude MUST be negative Z**:
```
5 meters altitude  → z = -5.0
10 meters altitude → z = -10.0
Ground level       → z = 0.0
```

The system auto-corrects LLM output: if Z > 0 in any waypoint, it is negated.

### 2. Offboard → Arm Sequence (Critical for PX4)

PX4 requires OffboardControlMode BEFORE arming:
```
1. Publish OffboardControlMode at 10Hz (heartbeat)
2. Send VEHICLE_CMD_DO_SET_MODE → OFFBOARD (after ~10 messages)
3. Send VEHICLE_CMD_COMPONENT_ARM_DISARM → ARM
4. Now send TrajectorySetpoint for position/velocity control
```

If Arm is sent before Offboard mode is active, PX4 safety rejects it.

### 3. No Aerostack2 Dependency

This version uses **native PX4 FMU ROS2 topics** exclusively:

| Direction | Topic | Message Type | Purpose |
|-----------|-------|--------------|---------|
| **Publish** | `/fmu/in/offboard_control_mode` | OffboardControlMode | Enable offboard + position control |
| **Publish** | `/fmu/in/trajectory_setpoint` | TrajectorySetpoint | Target NED position |
| **Publish** | `/fmu/in/vehicle_command` | VehicleCommand | Arm/Disarm, mode changes |
| **Subscribe** | `/fmu/out/vehicle_local_position_v1` | VehicleLocalPosition | Current NED position & velocity |
| **Subscribe** | `/fmu/out/vehicle_status_v1` | VehicleStatus | Armed state, nav mode |
| **Subscribe** | `/fmu/out/vehicle_land_detected` | VehicleLandDetected | Landed flag |
| **Subscribe** | `/fmu/out/battery_status_v1` | BatteryStatus | Battery remaining |

Camera topics (via Gazebo bridge):
| Topic | Type | Source |
|-------|------|--------|
| `/camera/image_raw` | sensor_msgs/Image | RGB camera |
| `/camera/depth_raw` | sensor_msgs/Image | Depth camera |
| `/drone0/sensor_measurements/gps` | NavSatFix | GPS quality |

## Modules

| Module | File | Description |
|--------|------|-------------|
| Capability Registry | `capability_registry.py` | Loads & queries YAML-defined capabilities |
| Mission Parser | `mission_parser.py` | Parses YAML, dict (from LLM), NL; supports waypoints |
| Orchestrator | `orchestrator.py` | SayCan multiplicative scoring & plan execution |
| Safety Shield | `safety_shield.py` | Hard/soft constraint checking |
| **ROS2 World Model** | `ros2_world_model.py` | FMU telemetry + camera subscriptions (native PX4) |
| **ROS2 Executor** | `ros2_executor.py` | PX4FlightController via FMU topics (offboard→arm→fly) |
| **Image Analyzer** | `image_analyzer.py` | OpenCV image quality evaluation (blur, exposure, contrast) |
| **LLM Task Parser** | `llm_task_parser.py` | Qwen2.5 natural language → structured mission JSON + NED correction |
| World Model (Mock) | `world_model.py` | Mock world model for unit testing |
| Executor (Mock) | `executor.py` | Mock executor for unit testing |
| Logger | `logger.py` | Waypoint-centric JSONL logs + markdown reports |
| World Model Base | `world_model_base.py` | Abstract interface for Mock/ROS2 world models |

## Installation

```bash
cd uav_embodied_os

# Base install (no ROS2/LLM - sufficient for running tests)
pip install -e ".[dev]"

# ROS2 packages (on your drone/simulation machine)
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs
# px4_msgs: either install from apt or build from source
# See: https://github.com/PX4/px4_msgs
```

**No Aerostack2 installation required.**

## Quick Start - Gazebo Simulation

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
ros2 run ros_gz_image image_bridge /camera/image_raw
ros2 run ros_gz_image image_bridge /camera/depth_raw
ros2 run ros_gz_bridge parameter_bridge /camera/camera_info

# Terminal 6: Run inspection
export ROS_DOMAIN_ID=3
python3 examples/run_campus_inspection.py
```

### Natural Language Mode (with LLM)

```bash
python3 examples/run_campus_inspection.py \
    --natural-language "巡检baylands的3个目标点，检测烟雾和屋顶异常，低质量图像自动复拍"
```

## Flight Control Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  PX4FlightController (10Hz heartbeat loop)                      │
│                                                                 │
│  1. start_offboard()                                            │
│     └─ Publish OffboardControlMode at 10Hz                      │
│     └─ Wait 1s for PX4 to register offboard stream              │
│                                                                 │
│  2. set_offboard_mode()                                         │
│     └─ VehicleCommand(CMD_DO_SET_MODE, param1=1, param2=6)      │
│     └─ Wait for vehicle_status.nav_state == OFFBOARD            │
│                                                                 │
│  3. arm()                                                       │
│     └─ VehicleCommand(CMD_ARM_DISARM, param1=1.0)               │
│     └─ Wait for vehicle_status.arming_state == ARMED            │
│                                                                 │
│  4. set_target(x, y, z_ned, yaw)                                │
│     └─ Update TrajectorySetpoint (sent every heartbeat)         │
│     └─ Monitor VehicleLocalPosition for arrival                 │
│                                                                 │
│  5. land()                                                      │
│     └─ Set target z = 0.0 (descend to ground)                   │
│     └─ Wait for VehicleLandDetected.landed == true              │
│     └─ disarm()                                                 │
└─────────────────────────────────────────────────────────────────┘
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

## NED Z-Axis Auto-Correction

The LLM Task Parser includes automatic Z-axis correction:

```python
# LLM might output: {"position": [10, 10, 5]}   ← WRONG (would fly into ground)
# System corrects:  {"position": [10, 10, -5]}   ← CORRECT (5m altitude)
```

This correction is applied:
1. After LLM JSON parsing (before returning result)
2. After rule-based fallback parsing
3. Logged for debugging: `"NED Z-axis corrected: waypoint z set to -5.0"`

## Mission Definition (Waypoint Format)

```yaml
kind: Mission
metadata:
  name: baylands_3point_inspection
  description: "Inspect 3 designated points in baylands"
spec:
  objective: fly_inspect_report
  area: baylands
  # Coordinate system: NED (North-East-Down)
  # Z is NEGATIVE for altitude: -5.0 = 5 meters above ground
  waypoints:
    - name: point_1
      position: [10.0, 10.0, -5.0]     # 5m altitude
      tasks: [fly_to_area, hold_position, scan_area, evaluate_image_quality]
    - name: point_2
      position: [30.0, 10.0, -5.0]
      tasks: [fly_to_area, hold_position, scan_area, evaluate_image_quality]
    - name: point_3
      position: [20.0, 30.0, -5.0]
      tasks: [fly_to_area, hold_position, scan_area, evaluate_image_quality]
  priority_targets: [smoke, crowd, rooftop_anomaly]
  constraints:
    max_altitude_m: 120.0
    reserve_battery_percent: 25.0
    max_wind_speed_mps: 12.0
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

## Testing

```bash
# All 88 tests
pytest tests/ -v

# By module
pytest tests/test_image_analyzer.py -v      # 13 cases: blur/exposure/contrast scoring
pytest tests/test_llm_task_parser.py -v     # 23 cases: rule-based, NED correction, mock LLM
pytest tests/test_mission_parser_v2.py -v   # 7 cases: waypoint parsing, dict input
pytest tests/test_capability_registry.py -v # 8 cases
pytest tests/test_safety_shield.py -v       # 7 cases
pytest tests/test_orchestrator.py -v        # 10 cases
pytest tests/test_integration.py -v         # 5 cases
pytest tests/test_mission_parser.py -v      # 7 cases

# Coverage
pytest tests/ --cov=uav_eios --cov-report=term-missing
```

## Project Structure

```
uav_embodied_os/
├── configs/
│   ├── capabilities/           # 10 capability YAML definitions
│   ├── missions/
│   │   ├── baylands_3point_inspection.yaml  # Real mission (NED coords, -Z altitude)
│   │   └── campus_3_inspection.yaml         # Mock mission for testing
│   ├── safety/                 # Safety rule configs
│   └── llm_config.yaml        # LLM API endpoint config
├── src/uav_eios/
│   ├── ros2_executor.py        # Native PX4 FMU flight control (no AS2)
│   ├── ros2_world_model.py     # FMU telemetry + camera topics
│   ├── image_analyzer.py       # OpenCV image quality evaluation
│   ├── llm_task_parser.py      # Qwen2.5 + NED Z-axis correction
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
├── tests/                      # 88 test cases
├── examples/
│   ├── run_campus_inspection.py       # Real Gazebo inspection
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

## Requirements

- Python 3.10+
- PX4-Autopilot v1.16.0+ with SITL
- ROS2 Humble
- px4_msgs (ROS2 package)
- cv_bridge, sensor_msgs (ROS2 packages)
- MicroXRCEDDSAgent
- Gazebo Harmonic (for simulation)
- **NO Aerostack2 required**
