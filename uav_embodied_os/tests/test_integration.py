"""Integration tests - full mission execution pipeline."""

import pytest
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_eios.capability_registry import CapabilityRegistry
from uav_eios.executor import MockExecutor
from uav_eios.logger import MissionLogger
from uav_eios.mission_parser import MissionParser
from uav_eios.orchestrator import Orchestrator
from uav_eios.safety_shield import SafetyShield
from uav_eios.world_model import MockWorldModel


CONFIGS_DIR = Path(__file__).parent.parent / "configs"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "test"


class TestFullPipeline:
    """Test complete mission execution pipeline."""

    def setup_method(self):
        random.seed(123)

        self.registry = CapabilityRegistry()
        self.registry.load_from_directory(CONFIGS_DIR / "capabilities")

        self.safety_shield = SafetyShield()
        self.safety_shield.load_rules(CONFIGS_DIR / "safety" / "default_rules.yaml")

        self.world_model = MockWorldModel()
        self.world_model.initialize_mission("campus_3")

        self.orchestrator = Orchestrator(
            self.registry, self.safety_shield, self.world_model
        )

        self.executor = MockExecutor(self.world_model)
        self.logger = MissionLogger(OUTPUT_DIR)

        parser = MissionParser()
        self.mission = parser.parse_file(
            CONFIGS_DIR / "missions" / "campus_3_inspection.yaml"
        )

    def test_full_mission_execution(self):
        """Test that a full mission can execute without errors."""
        self.logger.start_mission(self.mission.name)
        plan = self.orchestrator.plan_mission(self.mission)

        assert len(plan) >= 4

        steps_executed = 0
        max_steps = 15

        for step_idx in range(max_steps):
            action_score = self.orchestrator.select_next_action(
                self.mission, step_idx, []
            )
            if action_score is None:
                break

            cap = self.registry.get(action_score.capability_name)
            if cap is None:
                break

            # Safety check
            safety_state = self.world_model.get_safety_state()
            safety_result = self.safety_shield.check(safety_state, {
                "max_altitude_m": 120.0,
                "reserve_battery_percent": 25.0,
            })

            if not safety_result.passed:
                break

            # Execute
            result = self.executor.execute(cap)
            steps_executed += 1

            self.logger.log_step(
                step=steps_executed,
                capability=cap.name,
                score=action_score.total_score,
                safety_passed=safety_result.passed,
                result=result.get("status", "unknown"),
                details=result,
            )

            if cap.name == "generate_report":
                break

        assert steps_executed >= 4
        assert self.logger.step_count == steps_executed

    def test_mission_generates_log(self):
        """Test that mission generates valid JSONL log."""
        self.logger.start_mission("test_mission")
        self.logger.log_step(1, "fly_to_area", 0.9, True, "success")
        self.logger.log_step(2, "scan_area", 0.85, True, "success")

        log_path = self.logger.save_jsonl("test_log.jsonl")
        assert log_path.exists()
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_mission_generates_report(self):
        """Test that mission generates valid markdown report."""
        self.logger.start_mission("test_mission")
        self.logger.log_step(1, "fly_to_area", 0.9, True, "success")
        self.logger.log_step(2, "scan_area", 0.85, True, "success")

        report_path = self.logger.generate_report("test_report.md")
        assert report_path.exists()
        content = report_path.read_text()
        assert "test_mission" in content
        assert "fly_to_area" in content

    def test_safety_blocks_dangerous_action(self):
        """Test that safety shield blocks actions in unsafe conditions."""
        # Drain battery below reserve
        self.world_model.drone.battery_percent = 10.0

        safety_state = self.world_model.get_safety_state()
        result = self.safety_shield.check(safety_state, {
            "max_altitude_m": 120.0,
            "reserve_battery_percent": 25.0,
        })
        assert result.passed is False
        assert "battery_reserve" in result.hard_violations

    def test_replanning_on_low_quality(self):
        """Test that orchestrator triggers replanning on degraded conditions."""
        self.orchestrator.plan_mission(self.mission)
        result = {"image_quality": 0.3}
        should_replan, action = self.orchestrator.should_replan(result, self.mission)
        assert should_replan is True
        assert action == "reobserve_from_new_angle"
