# UAV Embodied Intelligence Operating System (MVP)

低空无人机具身智能操作系统 - 最小可行产品

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Mission DSL (YAML)                   │
├──────────────────────────────────────────────────────┤
│  Mission Parser  │  Capability Registry (10 caps)    │
├──────────────────┼───────────────────────────────────┤
│          SayCan-style Orchestrator                    │
│  score = R × F × S × U × P                          │
├──────────────────────────────────────────────────────┤
│  Safety Shield (hard/soft constraints)               │
├──────────────────────────────────────────────────────┤
│  Mock Executor  │  World Model  │  Mission Logger    │
└──────────────────────────────────────────────────────┘
```

## Modules

| Module | File | Description |
|--------|------|-------------|
| Capability Registry | `src/uav_eios/capability_registry.py` | Loads & queries YAML-defined capabilities |
| Mission Parser | `src/uav_eios/mission_parser.py` | Parses mission YAML + rule-based NL extraction |
| Orchestrator | `src/uav_eios/orchestrator.py` | SayCan multiplicative scoring & plan execution |
| Safety Shield | `src/uav_eios/safety_shield.py` | Hard/soft constraint checking |
| World Model | `src/uav_eios/world_model.py` | Mock UAV + environment state |
| Executor | `src/uav_eios/executor.py` | Simulated capability execution |
| Logger | `src/uav_eios/logger.py` | JSONL logs + markdown report generation |

## 10 Core Capabilities

| Capability | Type | Description |
|-----------|------|-------------|
| fly_to_area | navigation | Navigate to target area |
| scan_area | perception | Systematic area scanning |
| detect_smoke | detection | Smoke/fire detection |
| detect_crowd | detection | Crowd density analysis |
| inspect_rooftop | inspection | Rooftop structural inspection |
| evaluate_image_quality | evaluation | Image quality assessment |
| reobserve_from_new_angle | perception_action | Re-capture from new angle |
| hold_position | navigation | Position hold (safety fallback) |
| return_home | navigation | Safe return to launch point |
| generate_report | reporting | Generate inspection report |

## Installation

```bash
cd uav_embodied_os
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the campus inspection demo
python examples/run_campus_inspection.py

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=uav_eios --cov-report=term-missing
```

## Demo Output

The demo executes a full campus inspection mission:
1. Loads 10 capabilities from YAML configs
2. Parses `campus_3_inspection` mission
3. Initializes mock world model
4. Plans mission using SayCan scoring
5. Executes with safety checks at each step
6. Triggers replanning on degraded results
7. Generates JSONL log and markdown report in `outputs/`

## Scoring Formula (SayCan)

```
total_score = task_relevance × feasibility × safety × resource_utility × user_preference
```

- **task_relevance (R)**: Capability-to-mission alignment based on type and targets
- **feasibility (F)**: Precondition satisfaction in current world state
- **safety (S)**: Safety shield pass/penalty score
- **resource_utility (U)**: Energy/time efficiency scoring
- **user_preference (P)**: Priority target weighting

## Safety Rules

**Hard constraints** (violation blocks execution):
- Altitude limit (120m)
- Battery reserve (25%)
- No-fly zone avoidance
- Geofence boundary

**Soft constraints** (violation penalizes score):
- High wind (>10 m/s)
- Low GPS quality (<0.5)
- Communication degradation (<0.3)
- Obstacle proximity (<5m)

## Project Structure

```
uav_embodied_os/
├── configs/
│   ├── capabilities/    # 10 capability YAML definitions
│   ├── missions/        # Mission definitions
│   └── safety/          # Safety rule configs
├── src/uav_eios/        # Core Python modules
├── tests/               # Unit + integration tests (36 cases)
├── examples/            # Runnable demos
├── outputs/             # Generated logs and reports
├── pyproject.toml       # Project config
└── README.md
```

## Testing & Verification

### Unit Tests

```bash
# All tests
pytest tests/ -v

# Individual modules
pytest tests/test_capability_registry.py -v
pytest tests/test_safety_shield.py -v
pytest tests/test_orchestrator.py -v
pytest tests/test_mission_parser.py -v
pytest tests/test_integration.py -v
```

### What to Verify

1. **Capability loading**: All 10 capabilities load without error
2. **Mission parsing**: YAML and NL text correctly parsed
3. **Safety enforcement**: Hard violations block, soft violations penalize
4. **Scoring correctness**: Multiplicative formula produces expected values
5. **Replanning**: Degraded results trigger `reobserve_from_new_angle`
6. **Output generation**: `outputs/` contains JSONL log + markdown report

### Integration Test Coverage

- Full mission pipeline end-to-end
- Safety blocking dangerous actions (altitude >120m)
- Replanning triggered on low image quality
- Log file generation with correct structure
- Report generation with mission summary

## Next Steps (Post-MVP)

1. **Phase 2**: Replace mock executor with ROS2 action client
2. **Phase 3**: Integrate Aerostack2 behaviors
3. **Phase 4**: PX4 SITL + Gazebo simulation
4. **Phase 5**: LLM-based task parsing (replace rule-based parser)
5. **Phase 6**: BehaviorTree.CPP execution engine
6. **Phase 7**: Real-world flight testing
