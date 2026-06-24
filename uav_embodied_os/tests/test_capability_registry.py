"""Tests for Capability Registry module."""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_eios.capability_registry import CapabilityRegistry, CapabilitySpec


CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "capabilities"


class TestCapabilitySpec:
    def test_from_yaml(self):
        data = {
            "metadata": {"name": "test_cap", "description": "A test capability"},
            "spec": {
                "type": "navigation",
                "inputs": ["position"],
                "outputs": ["status"],
                "preconditions": ["battery_above_reserve"],
                "resources": {"cpu": "low", "energy": "medium"},
                "safety": {"requires_geofence_check": True},
                "invocation": {"type": "ros2_action", "name": "/test"},
                "fallback": ["return_home"],
            },
        }
        cap = CapabilitySpec.from_yaml(data)
        assert cap.name == "test_cap"
        assert cap.description == "A test capability"
        assert cap.cap_type == "navigation"
        assert "position" in cap.inputs
        assert "status" in cap.outputs
        assert "battery_above_reserve" in cap.preconditions
        assert cap.resources["cpu"] == "low"
        assert cap.safety["requires_geofence_check"] is True
        assert "return_home" in cap.fallback


class TestCapabilityRegistry:
    def test_load_from_directory(self):
        registry = CapabilityRegistry()
        count = registry.load_from_directory(CONFIGS_DIR)
        assert count >= 8
        assert registry.count >= 8

    def test_get_capability(self):
        registry = CapabilityRegistry()
        registry.load_from_directory(CONFIGS_DIR)
        cap = registry.get("fly_to_area")
        assert cap is not None
        assert cap.name == "fly_to_area"
        assert cap.cap_type == "navigation"

    def test_get_nonexistent(self):
        registry = CapabilityRegistry()
        assert registry.get("nonexistent") is None

    def test_list_all(self):
        registry = CapabilityRegistry()
        registry.load_from_directory(CONFIGS_DIR)
        all_caps = registry.list_all()
        assert len(all_caps) >= 8
        names = [c.name for c in all_caps]
        assert "fly_to_area" in names
        assert "detect_smoke" in names

    def test_list_by_type(self):
        registry = CapabilityRegistry()
        registry.load_from_directory(CONFIGS_DIR)
        nav_caps = registry.list_by_type("navigation")
        assert len(nav_caps) >= 2
        for cap in nav_caps:
            assert cap.cap_type == "navigation"

    def test_check_preconditions_met(self):
        registry = CapabilityRegistry()
        registry.load_from_directory(CONFIGS_DIR)
        state = {
            "battery_above_reserve": True,
            "gps_quality_good": True,
            "no_fly_zone_clear": True,
        }
        assert registry.check_preconditions("fly_to_area", state) is True

    def test_check_preconditions_not_met(self):
        registry = CapabilityRegistry()
        registry.load_from_directory(CONFIGS_DIR)
        state = {
            "battery_above_reserve": False,
            "gps_quality_good": True,
            "no_fly_zone_clear": True,
        }
        assert registry.check_preconditions("fly_to_area", state) is False
