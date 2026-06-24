"""Tests for Orchestrator module."""

import pytest
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_eios.capability_registry import CapabilityRegistry
from uav_eios.mission_parser import MissionParser
from uav_eios.orchestrator import Orchestrator, ActionScore
from uav_eios.safety_shield import SafetyShield
from uav_eios.world_model import MockWorldModel


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestOrchestrator:
    def setup_method(self):
        random.seed(42)
        self.registry = CapabilityRegistry()
        self.registry.load_from_directory(CONFIGS_DIR / "capabilities")

        self.safety_shield = SafetyShield()
        self.safety_shield.load_rules(CONFIGS_DIR / "safety" / "default_rules.yaml")

        self.world_model = MockWorldModel()
        self.world_model.initialize_mission("campus_3")

        self.orchestrator = Orchestrator(
            self.registry, self.safety_shield, self.world_model
        )

        parser = MissionParser()
        self.mission = parser.parse_file(
            CONFIGS_DIR / "missions" / "campus_3_inspection.yaml"
        )

    def test_plan_mission(self):
        plan = self.orchestrator.plan_mission(self.mission)
        assert len(plan) >= 4
        assert plan[0] == "fly_to_area"
        assert "generate_report" in plan
        # Should include detection capabilities based on priority targets
        assert "detect_smoke" in plan or "detect_crowd" in plan

    def test_select_next_action(self):
        self.orchestrator.plan_mission(self.mission)
        score = self.orchestrator.select_next_action(self.mission, 0, [])
        assert score is not None
        assert score.capability_name == "fly_to_area"
        assert score.total_score > 0

    def test_action_score_components(self):
        self.orchestrator.plan_mission(self.mission)
        score = self.orchestrator.select_next_action(self.mission, 0, [])
        assert score is not None
        assert 0 <= score.task_relevance <= 1.0
        assert 0 <= score.capability_feasibility <= 1.0
        assert 0 <= score.safety_feasibility <= 1.0
        assert 0 <= score.resource_utility <= 1.0
        assert 0 <= score.user_preference <= 1.0

    def test_should_replan_low_quality(self):
        self.orchestrator.plan_mission(self.mission)
        result = {"image_quality": 0.4}  # Below threshold
        should_replan, action = self.orchestrator.should_replan(result, self.mission)
        assert should_replan is True
        assert action == "reobserve_from_new_angle"

    def test_should_not_replan_good_quality(self):
        self.orchestrator.plan_mission(self.mission)
        result = {"image_quality": 0.9}  # Above threshold
        should_replan, action = self.orchestrator.should_replan(result, self.mission)
        assert should_replan is False

    def test_should_replan_high_wind(self):
        self.orchestrator.plan_mission(self.mission)
        # Simulate high wind
        self.world_model.environment.wind_speed_mps = 15.0
        result = {}
        should_replan, action = self.orchestrator.should_replan(result, self.mission)
        assert should_replan is True
        assert action == "hold_position"


class TestActionScore:
    def test_compute_total(self):
        score = ActionScore(
            capability_name="test",
            task_relevance=0.9,
            capability_feasibility=0.8,
            safety_feasibility=1.0,
            resource_utility=0.85,
            user_preference=0.9,
        )
        total = score.compute_total()
        expected = 0.9 * 0.8 * 1.0 * 0.85 * 0.9
        assert abs(total - expected) < 1e-6

    def test_zero_safety_blocks(self):
        score = ActionScore(
            capability_name="test",
            task_relevance=0.9,
            capability_feasibility=0.8,
            safety_feasibility=0.0,  # Safety fails
            resource_utility=0.85,
            user_preference=0.9,
        )
        total = score.compute_total()
        assert total == 0.0
