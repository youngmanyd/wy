#!/usr/bin/env python3
"""Campus 3 Inspection Demo - UAV Embodied Intelligence OS MVP.

This example demonstrates the full mission execution pipeline:
1. Load capabilities from YAML configs
2. Parse mission definition
3. Initialize mock world model
4. Plan and execute mission with SayCan-style scoring
5. Handle replanning on degraded conditions
6. Generate execution log and report
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from uav_eios.capability_registry import CapabilityRegistry
from uav_eios.executor import MockExecutor
from uav_eios.logger import MissionLogger
from uav_eios.mission_parser import MissionParser
from uav_eios.orchestrator import Orchestrator, OrchestratorStep
from uav_eios.safety_shield import SafetyShield
from uav_eios.world_model import MockWorldModel


def main() -> None:
    """Run the campus inspection demo."""
    # Set random seed for reproducible demo (remove for real testing)
    random.seed(42)

    configs_dir = project_root / "configs"
    output_dir = project_root / "outputs"

    print("=" * 60)
    print("  UAV Embodied Intelligence OS - MVP Demo")
    print("  Mission: Campus 3 Inspection")
    print("=" * 60)
    print()

    # 1. Load capabilities
    print("[1/6] Loading capabilities...")
    registry = CapabilityRegistry()
    cap_count = registry.load_from_directory(configs_dir / "capabilities")
    print(f"  Loaded {cap_count} capabilities: {registry.get_names()}")
    print()

    # 2. Parse mission
    print("[2/6] Parsing mission...")
    parser = MissionParser()
    mission = parser.parse_file(configs_dir / "missions" / "campus_3_inspection.yaml")
    print(f"  Mission: {mission.name}")
    print(f"  Area: {mission.area}")
    print(f"  Priority targets: {mission.priority_targets}")
    print(f"  Constraints: altitude<={mission.constraints.max_altitude_m}m, "
          f"battery>={mission.constraints.reserve_battery_percent}%")
    print()

    # 3. Initialize world model
    print("[3/6] Initializing world model...")
    world_model = MockWorldModel()
    world_model.initialize_mission(mission.area)
    drone_state = world_model.get_drone_state()
    env_state = world_model.get_environment_state()
    print(f"  Drone: battery={drone_state['battery_percent']:.1f}%, "
          f"altitude={drone_state['altitude_m']}m")
    print(f"  Environment: wind={env_state['wind_speed_mps']:.1f}m/s, "
          f"GPS={env_state['gps_quality']:.2f}")
    print()

    # 4. Load safety rules
    print("[4/6] Loading safety rules...")
    safety_shield = SafetyShield()
    rule_count = safety_shield.load_rules(configs_dir / "safety" / "default_rules.yaml")
    print(f"  Loaded {rule_count} safety rules")
    print()

    # 5. Plan and execute mission
    print("[5/6] Planning and executing mission...")
    print("-" * 60)

    orchestrator = Orchestrator(registry, safety_shield, world_model)
    executor = MockExecutor(world_model)
    logger = MissionLogger(output_dir)
    logger.start_mission(mission.name)

    # Generate initial plan
    plan = orchestrator.plan_mission(mission)
    print(f"  Initial plan: {plan}")
    print()

    # Execute mission
    execution_history: list[OrchestratorStep] = []
    step_number = 0
    replan_count = 0
    max_steps = 15  # Safety limit

    while step_number < max_steps:
        # Select next action
        action_score = orchestrator.select_next_action(
            mission, step_number, execution_history
        )

        if action_score is None:
            print(f"  No viable action found. Mission ending.")
            break

        cap_name = action_score.capability_name
        capability = registry.get(cap_name)
        if capability is None:
            break

        # Safety check
        safety_state = world_model.get_safety_state()
        mission_constraints = {
            "max_altitude_m": mission.constraints.max_altitude_m,
            "reserve_battery_percent": mission.constraints.reserve_battery_percent,
        }
        safety_result = safety_shield.check(safety_state, mission_constraints)

        if not safety_result.passed:
            print(f"  Step {step_number + 1}: {cap_name} - BLOCKED by safety: "
                  f"{safety_result.hard_violations}")
            if safety_result.recommended_fallback:
                cap_name = safety_result.recommended_fallback
                capability = registry.get(cap_name)
                if capability is None:
                    break
            else:
                break

        # Execute capability
        result = executor.execute(capability)
        result_status = result.get("status", "unknown")

        # Log step
        step_number += 1
        logger.log_step(
            step=step_number,
            capability=cap_name,
            score=action_score.total_score,
            safety_passed=safety_result.passed,
            result=result_status,
            details=result,
        )

        # Print step result
        print(f"  Step {step_number}: {cap_name}")
        print(f"    Score: {action_score.total_score:.4f} "
              f"(R={action_score.task_relevance:.2f} "
              f"F={action_score.capability_feasibility:.2f} "
              f"S={action_score.safety_feasibility:.2f} "
              f"U={action_score.resource_utility:.2f} "
              f"P={action_score.user_preference:.2f})")
        print(f"    Safety: {'PASS' if safety_result.passed else 'FAIL'}")
        print(f"    Result: {result_status}")

        # Record step
        orch_step = OrchestratorStep(
            step_number=step_number,
            selected_capability=cap_name,
            score=action_score,
            safety_passed=safety_result.passed,
            result=result_status,
        )
        execution_history.append(orch_step)

        # Check if replanning is needed
        should_replan, replan_action = orchestrator.should_replan(result, mission)
        if should_replan and replan_count < mission.max_replan_attempts:
            replan_count += 1
            print(f"    >>> REPLAN triggered: {replan_action} "
                  f"(attempt {replan_count}/{mission.max_replan_attempts})")
            # Insert replan action into execution
            replan_cap = registry.get(replan_action)
            if replan_cap:
                replan_result = executor.execute(replan_cap)
                step_number += 1
                logger.log_step(
                    step=step_number,
                    capability=replan_action,
                    score=0.0,
                    safety_passed=True,
                    result=replan_result.get("status", "unknown"),
                    details=replan_result,
                    replan_triggered=True,
                )
                print(f"  Step {step_number}: {replan_action} [REPLAN]")
                print(f"    Result: {replan_result.get('status', 'unknown')}")

        # Check if mission is complete (reached report step)
        if cap_name == "generate_report":
            print()
            print("  Mission completed successfully!")
            break

        print()

    print("-" * 60)
    print()

    # 6. Generate outputs
    print("[6/6] Generating outputs...")
    log_path = logger.save_jsonl()
    report_path = logger.generate_report()
    print(f"  Log saved to: {log_path}")
    print(f"  Report saved to: {report_path}")
    print()

    # Summary
    print("=" * 60)
    print("  MISSION SUMMARY")
    print("=" * 60)
    print(f"  Total steps: {step_number}")
    print(f"  Replanning events: {replan_count}")
    print(f"  Final battery: {world_model.drone.battery_percent:.1f}%")
    print(f"  Area coverage: {world_model.get_area_coverage(mission.area) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
