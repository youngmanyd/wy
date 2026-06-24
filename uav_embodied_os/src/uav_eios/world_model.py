"""Mock World Model - simulates UAV and environment state."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DroneState:
    """Current state of the UAV."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z
    battery_percent: float = 100.0
    altitude_m: float = 0.0
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    is_flying: bool = False
    is_armed: bool = False


@dataclass
class EnvironmentState:
    """Current environment state."""

    wind_speed_mps: float = 3.0
    gps_quality: float = 0.95
    comm_quality: float = 0.9
    obstacle_distance_m: float = 50.0
    visibility: float = 1.0
    temperature_c: float = 25.0


@dataclass
class AreaState:
    """State of a mission area."""

    area_id: str = ""
    coverage_percent: float = 0.0
    targets_found: list[str] = field(default_factory=list)
    images_captured: int = 0


class MockWorldModel:
    """Simulated world model for MVP testing."""

    def __init__(self) -> None:
        self.drone = DroneState()
        self.environment = EnvironmentState()
        self.areas: dict[str, AreaState] = {}
        self._step_count = 0
        self._events: list[dict[str, Any]] = []

    def initialize_mission(self, area_id: str) -> None:
        """Initialize world state for a mission."""
        self.drone = DroneState(
            position=(0.0, 0.0, 0.0),
            battery_percent=95.0,
            altitude_m=0.0,
            is_flying=False,
            is_armed=True,
        )
        self.environment = EnvironmentState(
            wind_speed_mps=random.uniform(2.0, 6.0),
            gps_quality=random.uniform(0.85, 0.99),
            comm_quality=random.uniform(0.8, 0.99),
            obstacle_distance_m=random.uniform(20.0, 100.0),
        )
        self.areas[area_id] = AreaState(area_id=area_id)
        self._step_count = 0
        self._events = []

    def get_drone_state(self) -> dict[str, Any]:
        return {
            "position": self.drone.position,
            "battery_percent": self.drone.battery_percent,
            "altitude_m": self.drone.altitude_m,
            "speed_mps": self.drone.speed_mps,
            "is_flying": self.drone.is_flying,
        }

    def get_environment_state(self) -> dict[str, Any]:
        return {
            "wind_speed_mps": self.environment.wind_speed_mps,
            "gps_quality": self.environment.gps_quality,
            "comm_quality": self.environment.comm_quality,
            "obstacle_distance_m": self.environment.obstacle_distance_m,
            "visibility": self.environment.visibility,
        }

    def get_precondition_state(self, area_id: str = "") -> dict[str, bool]:
        """Get current precondition satisfaction state."""
        return {
            "battery_above_reserve": self.drone.battery_percent > 25.0,
            "gps_quality_good": self.environment.gps_quality >= 0.7,
            "no_fly_zone_clear": True,
            "at_target_area": self.drone.is_flying and self.drone.altitude_m > 10,
            "weather_acceptable": self.environment.wind_speed_mps < 12.0,
            "sensor_available": True,
            "altitude_safe": self.drone.altitude_m <= 120.0,
            "target_position_known": True,
            "safe_viewpoint_available": True,
            "image_available": self._step_count > 1,
            "mission_data_available": self._step_count > 2,
        }

    def get_safety_state(self) -> dict[str, float]:
        """Get state values for safety rule evaluation."""
        return {
            "altitude_m": self.drone.altitude_m,
            "battery_percent": self.drone.battery_percent,
            "wind_speed_mps": self.environment.wind_speed_mps,
            "gps_quality": self.environment.gps_quality,
            "comm_quality": self.environment.comm_quality,
            "obstacle_distance_m": self.environment.obstacle_distance_m,
            "in_no_fly_zone": False,
            "within_geofence": True,
        }

    def update_after_action(self, capability_name: str, result: dict[str, Any]) -> None:
        """Update world state after a capability execution."""
        self._step_count += 1

        # Simulate battery drain
        energy_cost = {"high": 8.0, "medium": 4.0, "low": 1.0}
        drain = energy_cost.get(result.get("energy_used", "low"), 2.0)
        self.drone.battery_percent = max(0.0, self.drone.battery_percent - drain)

        # Simulate state changes based on capability
        if capability_name == "fly_to_area":
            self.drone.is_flying = True
            self.drone.altitude_m = 80.0
            self.drone.position = (100.0, 100.0, 80.0)
        elif capability_name == "scan_area":
            for area in self.areas.values():
                area.coverage_percent = min(1.0, area.coverage_percent + 0.3)
                area.images_captured += 5
        elif capability_name == "hold_position":
            pass  # no state change
        elif capability_name == "return_home":
            self.drone.position = (0.0, 0.0, 0.0)
            self.drone.altitude_m = 0.0
            self.drone.is_flying = False

        # Random environment perturbation
        self.environment.wind_speed_mps += random.uniform(-1.0, 1.5)
        self.environment.wind_speed_mps = max(0.0, self.environment.wind_speed_mps)

        self._events.append({
            "step": self._step_count,
            "capability": capability_name,
            "battery": self.drone.battery_percent,
            "wind": self.environment.wind_speed_mps,
        })

    def get_area_coverage(self, area_id: str) -> float:
        area = self.areas.get(area_id)
        return area.coverage_percent if area else 0.0
