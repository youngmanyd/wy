"""Tests for Mission Parser module."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_eios.mission_parser import MissionParser, Mission


MISSION_FILE = Path(__file__).parent.parent / "configs" / "missions" / "campus_3_inspection.yaml"


class TestMissionParser:
    def setup_method(self):
        self.parser = MissionParser()

    def test_parse_file(self):
        mission = self.parser.parse_file(MISSION_FILE)
        assert mission.name == "campus_3_inspection"
        assert mission.area == "campus_3"
        assert "smoke" in mission.priority_targets
        assert "crowd" in mission.priority_targets
        assert "rooftop_anomaly" in mission.priority_targets

    def test_parse_constraints(self):
        mission = self.parser.parse_file(MISSION_FILE)
        assert mission.constraints.max_altitude_m == 120.0
        assert mission.constraints.reserve_battery_percent == 25.0
        assert mission.constraints.max_wind_speed_mps == 12.0
        assert mission.constraints.avoid_no_fly_zones is True

    def test_parse_policies(self):
        mission = self.parser.parse_file(MISSION_FILE)
        assert len(mission.policies) >= 3
        policy_conditions = [p.condition for p in mission.policies]
        assert "low_image_quality" in policy_conditions
        assert "high_wind" in policy_conditions
        assert "low_battery" in policy_conditions

    def test_parse_success_criteria(self):
        mission = self.parser.parse_file(MISSION_FILE)
        assert mission.success_criteria.min_area_coverage == 0.90
        assert mission.success_criteria.min_detection_confidence == 0.75
        assert mission.success_criteria.report_required is True

    def test_natural_language_chinese(self):
        text = "巡检3号园区，优先检查烟雾、人员聚集和屋顶设备异常，遇到遮挡自动换角度确认。"
        result = self.parser.parse_natural_language(text)
        assert result["area"] == "campus_3"
        assert "smoke" in result["priority_targets"]
        assert "crowd" in result["priority_targets"]
        assert "rooftop_anomaly" in result["priority_targets"]

    def test_natural_language_english(self):
        text = "Inspect campus 3, check for smoke and crowd."
        result = self.parser.parse_natural_language(text)
        assert result["area"] == "campus_3"
        assert "smoke" in result["priority_targets"]
        assert "crowd" in result["priority_targets"]

    def test_natural_language_policy_extraction(self):
        text = "巡检3号园区，遇到遮挡自动换角度确认。"
        result = self.parser.parse_natural_language(text)
        assert len(result["policies"]) > 0
