"""Tests for Safety Shield module."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_eios.safety_shield import SafetyShield


SAFETY_RULES_FILE = Path(__file__).parent.parent / "configs" / "safety" / "default_rules.yaml"


class TestSafetyShield:
    def setup_method(self):
        self.shield = SafetyShield()
        self.shield.load_rules(SAFETY_RULES_FILE)

    def test_load_rules(self):
        assert self.shield.rule_count >= 4

    def test_all_safe(self):
        """All conditions met - should pass."""
        state = {
            "altitude_m": 80.0,
            "battery_percent": 60.0,
            "wind_speed_mps": 5.0,
            "gps_quality": 0.95,
            "comm_quality": 0.9,
            "obstacle_distance_m": 50.0,
            "in_no_fly_zone": False,
            "within_geofence": True,
        }
        constraints = {
            "max_altitude_m": 120.0,
            "reserve_battery_percent": 25.0,
        }
        result = self.shield.check(state, constraints)
        assert result.passed is True
        assert len(result.hard_violations) == 0

    def test_hard_violation_altitude(self):
        """Altitude exceeds limit - hard violation."""
        state = {
            "altitude_m": 150.0,
            "battery_percent": 60.0,
            "wind_speed_mps": 5.0,
            "gps_quality": 0.95,
            "comm_quality": 0.9,
            "obstacle_distance_m": 50.0,
            "in_no_fly_zone": False,
            "within_geofence": True,
        }
        constraints = {
            "max_altitude_m": 120.0,
            "reserve_battery_percent": 25.0,
        }
        result = self.shield.check(state, constraints)
        assert result.passed is False
        assert "altitude_limit" in result.hard_violations

    def test_hard_violation_battery(self):
        """Battery below reserve - hard violation."""
        state = {
            "altitude_m": 80.0,
            "battery_percent": 15.0,
            "wind_speed_mps": 5.0,
            "gps_quality": 0.95,
            "comm_quality": 0.9,
            "obstacle_distance_m": 50.0,
            "in_no_fly_zone": False,
            "within_geofence": True,
        }
        constraints = {
            "max_altitude_m": 120.0,
            "reserve_battery_percent": 25.0,
        }
        result = self.shield.check(state, constraints)
        assert result.passed is False
        assert "battery_reserve" in result.hard_violations

    def test_soft_violation_wind(self):
        """High wind - soft violation with penalty."""
        state = {
            "altitude_m": 80.0,
            "battery_percent": 60.0,
            "wind_speed_mps": 12.0,
            "gps_quality": 0.95,
            "comm_quality": 0.9,
            "obstacle_distance_m": 50.0,
            "in_no_fly_zone": False,
            "within_geofence": True,
        }
        constraints = {
            "max_altitude_m": 120.0,
            "reserve_battery_percent": 25.0,
        }
        result = self.shield.check(state, constraints)
        assert result.passed is True  # Soft violations don't block
        assert "high_wind" in result.soft_violations
        assert result.total_penalty > 0

    def test_no_fly_zone_violation(self):
        """In no-fly zone - hard violation."""
        state = {
            "altitude_m": 80.0,
            "battery_percent": 60.0,
            "wind_speed_mps": 5.0,
            "gps_quality": 0.95,
            "comm_quality": 0.9,
            "obstacle_distance_m": 50.0,
            "in_no_fly_zone": True,
            "within_geofence": True,
        }
        constraints = {
            "max_altitude_m": 120.0,
            "reserve_battery_percent": 25.0,
        }
        result = self.shield.check(state, constraints)
        assert result.passed is False
        assert "no_fly_zone" in result.hard_violations

    def test_safety_score(self):
        """Safety score should be 1.0 when all safe."""
        state = {
            "altitude_m": 80.0,
            "battery_percent": 60.0,
            "wind_speed_mps": 5.0,
            "gps_quality": 0.95,
            "comm_quality": 0.9,
            "obstacle_distance_m": 50.0,
            "in_no_fly_zone": False,
            "within_geofence": True,
        }
        constraints = {
            "max_altitude_m": 120.0,
            "reserve_battery_percent": 25.0,
        }
        score = self.shield.get_safety_score(state, constraints)
        assert score == 1.0

    def test_safety_score_with_violations(self):
        """Safety score should be 0.0 on hard violation."""
        state = {
            "altitude_m": 150.0,
            "battery_percent": 60.0,
            "wind_speed_mps": 5.0,
            "gps_quality": 0.95,
            "comm_quality": 0.9,
            "obstacle_distance_m": 50.0,
            "in_no_fly_zone": False,
            "within_geofence": True,
        }
        constraints = {
            "max_altitude_m": 120.0,
            "reserve_battery_percent": 25.0,
        }
        score = self.shield.get_safety_score(state, constraints)
        assert score == 0.0
