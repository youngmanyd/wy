#!/usr/bin/env python3
"""Baylands 3-Point Inspection - UAV Embodied Intelligence OS Phase 2.

Real Gazebo simulation execution via native PX4 FMU ROS2 topics:
1. Initialize ROS2 world model (subscribes to /fmu/out/* topics)
2. Parse mission (YAML or natural language via LLM)
3. For each waypoint: fly -> hold -> capture image -> evaluate quality
4. If image quality low: reobserve from new angle
5. Return home and generate report

NED Coordinate System: Z negative = altitude (e.g., -5.0 = 5m above ground)
Flight Sequence: Offboard mode → Arm → Trajectory setpoints

Requirements:
  - PX4 SITL running with x500_depth in baylands
  - Image bridges active (/camera/image_raw, /camera/depth_raw)
  - MicroXRCEAgent running (PX4-ROS2 bridge)
  - px4_msgs ROS2 package installed
  - NO Aerostack2 required
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from uav_eios.capability_registry import CapabilityRegistry
from uav_eios.logger import MissionLogger
from uav_eios.mission_parser import MissionParser
from uav_eios.orchestrator import Orchestrator
from uav_eios.safety_shield import SafetyShield

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inspection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UAV Baylands Inspection")
    parser.add_argument(
        "--mission", type=str,
        default=str(project_root / "configs" / "missions" / "baylands_3point_inspection.yaml"),
        help="Path to mission YAML file",
    )
    parser.add_argument(
        "--natural-language", type=str, default="",
        help="Natural language task description (uses LLM parser)",
    )
    parser.add_argument(
        "--llm-config", type=str,
        default=str(project_root / "configs" / "llm_config.yaml"),
        help="Path to LLM config YAML",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(project_root / "outputs"),
        help="Output directory for logs and reports",
    )
    parser.add_argument(
        "--drone-id", type=str, default="drone0",
        help="PX4 drone namespace (used for GPS topic prefix)",
    )
    parser.add_argument(
        "--speed", type=float, default=2.0,
        help="Default flight speed (m/s)",
    )
    parser.add_argument(
        "--hold-time", type=float, default=3.0,
        help="Hold time at each waypoint (seconds)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs_dir = project_root / "configs"
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  UAV Embodied Intelligence OS - Phase 2")
    print("  Baylands Real Simulation Inspection")
    print("=" * 60)
    print()

    # --- Import ROS2 modules (fail fast if not available) ---
    try:
        import rclpy
        from uav_eios.ros2_world_model import ROS2WorldModel
        from uav_eios.ros2_executor import ROS2Executor
    except ImportError as e:
        logger.error("ROS2 dependencies not available: %s", e)
        logger.error(
            "Ensure rclpy, px4_msgs, cv_bridge, sensor_msgs are installed.\n"
            "  No Aerostack2 required - uses native PX4 FMU topics."
        )
        sys.exit(1)

    # 1. Load capabilities
    print("[1/8] Loading capabilities...")
    registry = CapabilityRegistry()
    cap_count = registry.load_from_directory(configs_dir / "capabilities")
    print(f"  Loaded {cap_count} capabilities: {registry.get_names()}")
    print()

    # 2. Parse mission (YAML or natural language)
    print("[2/8] Parsing mission...")
    parser = MissionParser()

    if args.natural_language:
        print(f"  Natural language input: {args.natural_language}")
        try:
            from uav_eios.llm_task_parser import LLMTaskParser
            llm_parser = LLMTaskParser(config_path=args.llm_config)
            mission_dict = llm_parser.parse_natural_language(args.natural_language)
            mission = parser.parse_dict(mission_dict)
            print(f"  Parsed via {'LLM' if llm_parser.is_llm_available else 'rules'}")
        except Exception as e:
            logger.warning("LLM parsing failed: %s, falling back to YAML", e)
            mission = parser.parse_file(args.mission)
    else:
        mission = parser.parse_file(args.mission)

    print(f"  Mission: {mission.name}")
    print(f"  Area: {mission.area}")
    print(f"  Waypoints: {len(mission.waypoints)}")
    for wp in mission.waypoints:
        print(f"    - {wp.name}: {wp.position}")
    print(f"  Priority targets: {mission.priority_targets}")
    print()

    if not mission.waypoints:
        logger.error("No waypoints defined in mission. Cannot proceed.")
        sys.exit(1)

    # 3. Initialize ROS2 world model
    print("[3/8] Initializing ROS2 world model...")
    if not rclpy.ok():
        rclpy.init()

    world_model = ROS2WorldModel()
    world_model.initialize_mission(mission.area)

    print("  Waiting for sensor data...")
    time.sleep(3.0)

    drone_state = world_model.get_drone_state()
    env_state = world_model.get_environment_state()
    print(f"  Drone: pos=({drone_state['position'][0]:.1f}, {drone_state['position'][1]:.1f}, {drone_state['position'][2]:.1f}), "
          f"battery={drone_state['battery_percent']:.1f}%")
    print(f"  Environment: GPS={env_state['gps_quality']:.2f}")
    print()

    # 4. Initialize executor (native PX4 FMU topics - no Aerostack2)
    print("[4/8] Initializing ROS2 executor (native PX4 FMU control)...")
    executor = ROS2Executor(
        world_model=world_model,
        output_dir=output_dir,
    )
    print("  Control: direct /fmu/in/* topics (offboard + trajectory)")
    print("  Sequence: Offboard mode → Arm → Trajectory setpoints")
    print()

    # 5. Load safety rules
    print("[5/8] Loading safety rules...")
    safety_shield = SafetyShield()
    rule_count = safety_shield.load_rules(configs_dir / "safety" / "default_rules.yaml")
    print(f"  Loaded {rule_count} safety rules")
    print()

    # 6. Initialize orchestrator and logger
    print("[6/8] Initializing orchestrator and logger...")
    orchestrator = Orchestrator(registry, safety_shield, world_model)
    mission_logger = MissionLogger(output_dir)
    mission_logger.start_mission(mission.name)
    print()

    # 7. Execute waypoint inspection loop
    print("[7/8] Starting waypoint inspection...")
    print("=" * 60)

    step_number = 0
    replan_count = 0
    waypoint_results: list[dict] = []

    fly_to_cap = registry.get("fly_to_area")
    hold_cap = registry.get("hold_position")
    scan_cap = registry.get("scan_area")
    reobserve_cap = registry.get("reobserve_from_new_angle")
    return_home_cap = registry.get("return_home")

    for wp_idx, waypoint in enumerate(mission.waypoints):
        print(f"\n--- Waypoint {wp_idx + 1}/{len(mission.waypoints)}: {waypoint.name} ---")
        print(f"  Target: ({waypoint.position[0]:.1f}, {waypoint.position[1]:.1f}, {waypoint.position[2]:.1f})")

        # Safety check before flying
        safety_state = world_model.get_safety_state()
        mission_constraints = {
            "max_altitude_m": mission.constraints.max_altitude_m,
            "reserve_battery_percent": mission.constraints.reserve_battery_percent,
        }
        safety_result = safety_shield.check(safety_state, mission_constraints)

        if not safety_result.passed:
            print(f"  SAFETY VIOLATION: {safety_result.hard_violations}")
            print(f"  Aborting mission, returning home.")
            if return_home_cap:
                executor.execute(return_home_cap)
            break

        # --- Fly to waypoint ---
        step_number += 1
        print(f"  [{step_number}] Flying to {waypoint.name}...")
        if fly_to_cap:
            fly_result = executor.execute(
                fly_to_cap, target_position=waypoint.position, speed=args.speed,
            )
            mission_logger.log_step(
                step=step_number, capability="fly_to_area", score=1.0,
                safety_passed=True, result=fly_result["status"],
                details=fly_result, waypoint_name=waypoint.name,
                position=waypoint.position,
            )
            print(f"    Result: {fly_result['status']}")

        # --- Hold position ---
        step_number += 1
        print(f"  [{step_number}] Holding position for {args.hold_time}s...")
        if hold_cap:
            hold_result = executor.execute(hold_cap, duration=args.hold_time)
            mission_logger.log_step(
                step=step_number, capability="hold_position", score=1.0,
                safety_passed=True, result=hold_result["status"],
                details=hold_result, waypoint_name=waypoint.name,
                position=waypoint.position,
            )

        # --- Capture and evaluate image ---
        step_number += 1
        print(f"  [{step_number}] Capturing and evaluating image...")
        if scan_cap:
            scan_result = executor.execute(
                scan_cap, waypoint_name=waypoint.name,
            )
            image_quality = scan_result.get("image_quality", 0.0)
            mission_logger.log_step(
                step=step_number, capability="scan_area", score=1.0,
                safety_passed=True, result=scan_result["status"],
                details=scan_result, waypoint_name=waypoint.name,
                position=waypoint.position,
            )
            print(f"    Image quality: {image_quality:.3f}")
            print(f"    Blur: {scan_result.get('blur_score', 0):.3f}, "
                  f"Exposure: {scan_result.get('exposure_score', 0):.3f}, "
                  f"Contrast: {scan_result.get('contrast_score', 0):.3f}")

            # --- Reobserve if quality is low ---
            quality_threshold = mission.success_criteria.min_image_quality
            if image_quality < quality_threshold and replan_count < mission.max_replan_attempts:
                replan_count += 1
                step_number += 1
                print(f"  [{step_number}] Quality {image_quality:.3f} < {quality_threshold:.1f}, "
                      f"reobserving (attempt {replan_count}/{mission.max_replan_attempts})...")

                if reobserve_cap:
                    reobs_result = executor.execute(
                        reobserve_cap, waypoint_name=f"{waypoint.name}_reobs",
                    )
                    reobs_quality = reobs_result.get("image_quality", 0.0)
                    mission_logger.log_step(
                        step=step_number, capability="reobserve_from_new_angle",
                        score=0.0, safety_passed=True,
                        result=reobs_result["status"], details=reobs_result,
                        replan_triggered=True, waypoint_name=waypoint.name,
                        position=waypoint.position,
                    )
                    print(f"    Reobserve quality: {reobs_quality:.3f}")
                    image_quality = reobs_quality

            wp_result = {
                "waypoint": waypoint.name,
                "position": waypoint.position,
                "image_quality": image_quality,
                "reobserved": replan_count > 0 and image_quality < quality_threshold,
                "image_path": scan_result.get("image_path", ""),
            }
            waypoint_results.append(wp_result)

    # --- Return home ---
    print(f"\n--- Returning home ---")
    step_number += 1
    if return_home_cap:
        rh_result = executor.execute(return_home_cap)
        mission_logger.log_step(
            step=step_number, capability="return_home", score=1.0,
            safety_passed=True, result=rh_result["status"],
            details=rh_result,
        )
        print(f"  Result: {rh_result['status']}")

    print("=" * 60)
    print()

    # 8. Generate outputs
    print("[8/8] Generating outputs...")
    log_path = mission_logger.save_jsonl()
    report_path = mission_logger.generate_report()
    print(f"  Log: {log_path}")
    print(f"  Report: {report_path}")
    print()

    # Summary
    print("=" * 60)
    print("  MISSION SUMMARY")
    print("=" * 60)
    print(f"  Total steps: {step_number}")
    print(f"  Waypoints visited: {len(waypoint_results)}/{len(mission.waypoints)}")
    print(f"  Replanning events: {replan_count}")
    final_state = world_model.get_drone_state()
    print(f"  Final battery: {final_state['battery_percent']:.1f}%")
    print()
    print("  Waypoint Results:")
    for wr in waypoint_results:
        reobs_tag = " [REOBSERVED]" if wr["reobserved"] else ""
        print(f"    {wr['waypoint']}: quality={wr['image_quality']:.3f}{reobs_tag}")
    print("=" * 60)

    # Cleanup
    executor.shutdown()
    world_model.shutdown()


if __name__ == "__main__":
    main()
