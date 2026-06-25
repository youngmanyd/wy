"""Tests for updated MissionParser - waypoint support and dict parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from uav_eios.mission_parser import Mission, MissionParser, Waypoint


class TestWaypointParsing:
    """Test waypoint parsing from YAML."""

    def test_parse_baylands_mission(self) -> None:
        configs_dir = Path(__file__).parent.parent / "configs" / "missions"
        yaml_path = configs_dir / "baylands_3point_inspection.yaml"
        if not yaml_path.exists():
            pytest.skip("baylands_3point_inspection.yaml not found")

        parser = MissionParser()
        mission = parser.parse_file(yaml_path)

        assert mission.name == "baylands_3point_inspection"
        assert mission.area == "baylands"
        assert len(mission.waypoints) == 3

        assert mission.waypoints[0].name == "point_1"
        assert mission.waypoints[0].position == (10.0, 10.0, -5.0)
        assert "fly_to_area" in mission.waypoints[0].tasks

        assert mission.waypoints[1].position == (30.0, 10.0, -5.0)
        assert mission.waypoints[2].position == (20.0, 30.0, -5.0)

    def test_parse_dict_from_llm_output(self) -> None:
        llm_output = {
            "name": "test_mission",
            "description": "Test mission from LLM",
            "objective": "fly_inspect_report",
            "area": "campus_1",
            "waypoints": [
                {"name": "p1", "position": [5.0, 5.0, 3.0], "tasks": ["scan_area"]},
                {"name": "p2", "position": [15.0, 5.0, 3.0], "tasks": ["detect_smoke"]},
            ],
            "priority_targets": ["smoke"],
            "constraints": {
                "max_altitude_m": 100.0,
                "reserve_battery_percent": 30.0,
            },
            "policies": {
                "low_image_quality": {
                    "action": "reobserve_from_new_angle",
                    "threshold": 0.6,
                },
            },
        }

        parser = MissionParser()
        mission = parser.parse_dict(llm_output)

        assert mission.name == "test_mission"
        assert mission.area == "campus_1"
        assert len(mission.waypoints) == 2
        assert mission.waypoints[0].position == (5.0, 5.0, 3.0)
        assert mission.waypoints[1].name == "p2"
        assert mission.constraints.max_altitude_m == 100.0
        assert "smoke" in mission.priority_targets

    def test_parse_dict_empty_waypoints(self) -> None:
        llm_output = {
            "name": "empty_wp",
            "objective": "fly_inspect_report",
            "area": "test",
        }
        parser = MissionParser()
        mission = parser.parse_dict(llm_output)
        assert mission.waypoints == []

    def test_waypoint_dataclass(self) -> None:
        wp = Waypoint(name="test", position=(1.0, 2.0, 3.0), tasks=["fly", "scan"])
        assert wp.name == "test"
        assert wp.position == (1.0, 2.0, 3.0)
        assert len(wp.tasks) == 2

    def test_mission_success_criteria_min_quality(self) -> None:
        """Mission should have configurable min_image_quality in success_criteria."""
        data = {
            "metadata": {"name": "test"},
            "spec": {
                "objective": "fly_inspect_report",
                "area": "test",
                "success_criteria": {"min_image_quality": 0.7},
            },
        }
        mission = Mission.from_yaml(data)
        assert mission.success_criteria.min_image_quality == 0.7

    def test_mission_default_min_quality(self) -> None:
        data = {
            "metadata": {"name": "test"},
            "spec": {"objective": "fly_inspect_report", "area": "test"},
        }
        mission = Mission.from_yaml(data)
        assert mission.success_criteria.min_image_quality == 0.6


class TestLoggerWaypoints:
    """Test MissionLogger waypoint-related features."""

    def test_log_with_waypoint(self, tmp_path: Path) -> None:
        from uav_eios.logger import MissionLogger

        logger = MissionLogger(tmp_path)
        logger.start_mission("test_wp_mission")
        logger.log_step(
            step=1, capability="fly_to_area", score=1.0,
            safety_passed=True, result="success",
            waypoint_name="point_1", position=(10.0, 10.0, 5.0),
        )
        logger.log_step(
            step=2, capability="scan_area", score=1.0,
            safety_passed=True, result="success",
            details={"image_quality": 0.75, "image_path": "/tmp/test.jpg"},
            waypoint_name="point_1", position=(10.0, 10.0, 5.0),
        )

        log_path = logger.save_jsonl()
        assert log_path.exists()

        report_path = logger.generate_report()
        assert report_path.exists()

        report_text = report_path.read_text()
        assert "Waypoint Summary" in report_text
        assert "point_1" in report_text
        assert "0.75" in report_text
