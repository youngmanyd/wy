"""Mock Executor - simulates capability execution for MVP testing."""

from __future__ import annotations

import random
from typing import Any

from .capability_registry import CapabilitySpec
from .world_model import MockWorldModel


class MockExecutor:
    """Simulates capability execution without real UAV hardware.

    Each capability produces realistic mock results including
    occasional degraded conditions to test replanning logic.
    """

    def __init__(self, world_model: MockWorldModel) -> None:
        self.world_model = world_model
        self._execution_count = 0

    def execute(self, capability: CapabilitySpec) -> dict[str, Any]:
        """Execute a capability and return simulated results."""
        self._execution_count += 1
        handler = self._handlers.get(capability.name, self._default_handler)
        result = handler(self, capability)

        # Update world model
        self.world_model.update_after_action(capability.name, result)

        return result

    def _execute_fly_to_area(self, capability: CapabilitySpec) -> dict[str, Any]:
        return {
            "status": "success",
            "arrival_status": "arrived",
            "final_position": (100.0, 100.0, 80.0),
            "energy_used": "high",
            "duration_s": random.uniform(30, 60),
        }

    def _execute_scan_area(self, capability: CapabilitySpec) -> dict[str, Any]:
        coverage = random.uniform(0.25, 0.40)
        return {
            "status": "success",
            "coverage_percent": coverage,
            "images_captured": random.randint(3, 8),
            "energy_used": "high",
            "duration_s": random.uniform(60, 120),
        }

    def _execute_detect_smoke(self, capability: CapabilitySpec) -> dict[str, Any]:
        detected = random.random() < 0.15  # 15% chance of smoke
        return {
            "status": "success",
            "smoke_detected": detected,
            "smoke_confidence": random.uniform(0.8, 0.95) if detected else 0.0,
            "energy_used": "low",
            "duration_s": random.uniform(2, 5),
        }

    def _execute_detect_crowd(self, capability: CapabilitySpec) -> dict[str, Any]:
        detected = random.random() < 0.25  # 25% chance of crowd
        return {
            "status": "success",
            "crowd_detected": detected,
            "crowd_count": random.randint(5, 30) if detected else 0,
            "crowd_density": random.uniform(0.3, 0.8) if detected else 0.0,
            "energy_used": "low",
            "duration_s": random.uniform(2, 5),
        }

    def _execute_inspect_rooftop(self, capability: CapabilitySpec) -> dict[str, Any]:
        # 30% chance of low image quality to trigger replanning
        image_quality = random.uniform(0.4, 0.95)
        anomaly = random.random() < 0.2
        return {
            "status": "success" if image_quality >= 0.6 else "degraded",
            "image_quality": image_quality,
            "anomaly_detected": anomaly,
            "anomaly_confidence": random.uniform(0.7, 0.9) if anomaly else 0.0,
            "energy_used": "medium",
            "duration_s": random.uniform(10, 30),
        }

    def _execute_reobserve(self, capability: CapabilitySpec) -> dict[str, Any]:
        return {
            "status": "success",
            "image_quality": random.uniform(0.75, 0.95),
            "target_confidence": random.uniform(0.7, 0.92),
            "energy_used": "medium",
            "duration_s": random.uniform(15, 30),
        }

    def _execute_hold_position(self, capability: CapabilitySpec) -> dict[str, Any]:
        return {
            "status": "success",
            "hold_duration_s": random.uniform(5, 15),
            "energy_used": "medium",
            "duration_s": random.uniform(5, 15),
        }

    def _execute_return_home(self, capability: CapabilitySpec) -> dict[str, Any]:
        return {
            "status": "success",
            "return_status": "landed",
            "landing_position": (0.0, 0.0, 0.0),
            "energy_used": "high",
            "duration_s": random.uniform(30, 90),
        }

    def _execute_generate_report(self, capability: CapabilitySpec) -> dict[str, Any]:
        return {
            "status": "success",
            "report_path": "outputs/mission_report.md",
            "report_summary": "Mission completed successfully.",
            "energy_used": "low",
            "duration_s": 1.0,
        }

    def _default_handler(self, capability: CapabilitySpec) -> dict[str, Any]:
        return {
            "status": "success",
            "energy_used": "low",
            "duration_s": random.uniform(1, 10),
        }

    _handlers: dict[str, Any] = {
        "fly_to_area": _execute_fly_to_area,
        "scan_area": _execute_scan_area,
        "detect_smoke": _execute_detect_smoke,
        "detect_crowd": _execute_detect_crowd,
        "inspect_rooftop": _execute_inspect_rooftop,
        "reobserve_from_new_angle": _execute_reobserve,
        "hold_position": _execute_hold_position,
        "return_home": _execute_return_home,
        "generate_report": _execute_generate_report,
    }
