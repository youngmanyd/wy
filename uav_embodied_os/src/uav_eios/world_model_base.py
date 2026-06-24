"""World Model Base - abstract interface for UAV and environment state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DroneState:
    """Current state of the UAV."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
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


class WorldModelBase(ABC):
    """Abstract base class for world models.

    Both MockWorldModel and ROS2WorldModel implement this interface,
    allowing the Orchestrator and Executor to work with either.
    """

    drone: DroneState
    environment: EnvironmentState
    areas: dict[str, AreaState]

    @abstractmethod
    def initialize_mission(self, area_id: str) -> None:
        """Initialize world state for a mission."""

    @abstractmethod
    def get_drone_state(self) -> dict[str, Any]:
        """Get current drone state as dict."""

    @abstractmethod
    def get_environment_state(self) -> dict[str, Any]:
        """Get current environment state as dict."""

    @abstractmethod
    def get_precondition_state(self, area_id: str = "") -> dict[str, bool]:
        """Get current precondition satisfaction state."""

    @abstractmethod
    def get_safety_state(self) -> dict[str, float]:
        """Get state values for safety rule evaluation."""

    @abstractmethod
    def update_after_action(self, capability_name: str, result: dict[str, Any]) -> None:
        """Update world state after a capability execution."""

    def get_area_coverage(self, area_id: str) -> float:
        area = self.areas.get(area_id)
        return area.coverage_percent if area else 0.0

    def get_latest_image(self) -> np.ndarray | None:
        """Get the latest captured RGB image (if available)."""
        return None

    def get_latest_depth(self) -> np.ndarray | None:
        """Get the latest captured depth image (if available)."""
        return None
