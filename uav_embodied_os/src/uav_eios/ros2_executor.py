"""ROS2 Executor - executes capabilities via Aerostack2 DroneInterface.

Controls the UAV through AS2 Python API for real Gazebo simulation.
Replaces MockExecutor for real flight execution.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from as2_python_api.drone_interface import DroneInterface
    AS2_AVAILABLE = True
except ImportError:
    AS2_AVAILABLE = False

from .capability_registry import CapabilitySpec
from .image_analyzer import ImageAnalyzer
from .world_model_base import WorldModelBase


class ROS2Executor:
    """Executes capabilities using Aerostack2 DroneInterface and real sensors.

    Flight control: AS2 DroneInterface (arm, takeoff, go_to, land)
    Image capture:  From ROS2WorldModel camera subscriptions
    Image quality:  ImageAnalyzer (OpenCV-based evaluation)
    """

    def __init__(
        self,
        world_model: WorldModelBase,
        drone_id: str = "drone0",
        use_sim_time: bool = True,
        output_dir: str | Path = "outputs",
    ) -> None:
        if not AS2_AVAILABLE:
            raise RuntimeError(
                "as2_python_api not available. "
                "Install Aerostack2 Python API: https://aerostack2.github.io/"
            )

        self.world_model = world_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._image_analyzer = ImageAnalyzer()
        self._execution_count = 0

        self.drone = DroneInterface(drone_id, verbose=False, use_sim_time=use_sim_time)
        self._is_armed = False
        self._is_flying = False

    def execute(self, capability: CapabilitySpec, **kwargs: Any) -> dict[str, Any]:
        """Execute a capability and return results from real simulation."""
        self._execution_count += 1
        handler = self._handlers.get(capability.name, self._default_handler)
        result = handler(self, capability, **kwargs)
        self.world_model.update_after_action(capability.name, result)
        return result

    def _ensure_armed_and_offboard(self) -> None:
        if not self._is_armed:
            self.drone.arm()
            self.drone.offboard()
            self._is_armed = True
            time.sleep(1.0)

    def _execute_fly_to_area(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Takeoff and fly to target position."""
        target = kwargs.get("target_position", (10.0, 10.0, 5.0))
        speed = kwargs.get("speed", 2.0)

        self._ensure_armed_and_offboard()

        start_time = time.time()

        if not self._is_flying:
            takeoff_height = target[2] if len(target) >= 3 else 5.0
            self.drone.takeoff(takeoff_height, speed=1.0)
            self._wait_for_takeoff(takeoff_height, timeout=15.0)
            self._is_flying = True

        self.drone.go_to.go_to_point(
            [float(target[0]), float(target[1]), float(target[2])],
            speed=speed,
        )
        self._wait_for_arrival(target, timeout=30.0)

        duration = time.time() - start_time
        final_pos = self.world_model.drone.position

        return {
            "status": "success",
            "arrival_status": "arrived",
            "target_position": target,
            "final_position": final_pos,
            "energy_used": "high",
            "duration_s": duration,
        }

    def _execute_hold_position(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Hold current position for a specified duration."""
        hold_duration = kwargs.get("duration", 3.0)
        time.sleep(hold_duration)
        return {
            "status": "success",
            "hold_duration_s": hold_duration,
            "position": self.world_model.drone.position,
            "energy_used": "medium",
            "duration_s": hold_duration,
        }

    def _execute_capture_and_evaluate(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Capture image from camera and evaluate quality."""
        waypoint_name = kwargs.get("waypoint_name", f"point_{self._execution_count}")

        time.sleep(0.5)

        image = self.world_model.wait_for_fresh_image(timeout_s=5.0)
        if image is None:
            return {
                "status": "degraded",
                "image_quality": 0.0,
                "error": "no_image_received",
                "energy_used": "low",
                "duration_s": 5.0,
            }

        quality = self._image_analyzer.evaluate(image)

        img_filename = f"{waypoint_name}_{self._execution_count}.jpg"
        img_path = self._image_analyzer.save_image(
            image, self.output_dir / "images" / img_filename
        )

        status = "success" if quality.is_acceptable else "degraded"

        return {
            "status": status,
            "image_quality": quality.overall_quality,
            "blur_score": quality.blur_score,
            "exposure_score": quality.exposure_score,
            "contrast_score": quality.contrast_score,
            "laplacian_variance": quality.laplacian_variance,
            "mean_brightness": quality.mean_brightness,
            "image_path": str(img_path),
            "image_shape": list(image.shape),
            "is_acceptable": quality.is_acceptable,
            "energy_used": "low",
            "duration_s": 1.0,
        }

    def _execute_scan_area(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Scan area by capturing image at current position."""
        return self._execute_capture_and_evaluate(capability, **kwargs)

    def _execute_inspect_rooftop(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Inspect rooftop - capture and evaluate image."""
        result = self._execute_capture_and_evaluate(capability, **kwargs)
        result["anomaly_detected"] = False
        result["anomaly_confidence"] = 0.0
        return result

    def _execute_detect_smoke(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Detect smoke - capture image (real detection requires ML model)."""
        result = self._execute_capture_and_evaluate(capability, **kwargs)
        result["smoke_detected"] = False
        result["smoke_confidence"] = 0.0
        return result

    def _execute_detect_crowd(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Detect crowd - capture image (real detection requires ML model)."""
        result = self._execute_capture_and_evaluate(capability, **kwargs)
        result["crowd_detected"] = False
        result["crowd_count"] = 0
        result["crowd_density"] = 0.0
        return result

    def _execute_evaluate_image_quality(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Evaluate quality of the latest captured image."""
        image = self.world_model.get_latest_image()
        if image is None:
            return {
                "status": "degraded",
                "image_quality": 0.0,
                "error": "no_image_available",
                "energy_used": "low",
                "duration_s": 0.1,
            }

        quality = self._image_analyzer.evaluate(image)
        return {
            "status": "success" if quality.is_acceptable else "degraded",
            "image_quality": quality.overall_quality,
            "blur_score": quality.blur_score,
            "exposure_score": quality.exposure_score,
            "contrast_score": quality.contrast_score,
            "is_acceptable": quality.is_acceptable,
            "energy_used": "low",
            "duration_s": 0.1,
        }

    def _execute_reobserve(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Reobserve from a new angle - shift position slightly and recapture."""
        offset_x = kwargs.get("offset_x", 3.0)
        offset_y = kwargs.get("offset_y", 0.0)
        offset_z = kwargs.get("offset_z", 1.0)

        current_pos = self.world_model.drone.position
        new_pos = (
            current_pos[0] + offset_x,
            current_pos[1] + offset_y,
            current_pos[2] + offset_z,
        )

        start_time = time.time()

        self.drone.go_to.go_to_point(
            [float(new_pos[0]), float(new_pos[1]), float(new_pos[2])],
            speed=1.5,
        )
        self._wait_for_arrival(new_pos, timeout=15.0)

        time.sleep(1.0)

        result = self._execute_capture_and_evaluate(
            capability, waypoint_name=f"reobserve_{self._execution_count}", **kwargs
        )
        result["new_position"] = new_pos
        result["duration_s"] = time.time() - start_time
        result["energy_used"] = "medium"
        return result

    def _execute_return_home(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Return to home position and land."""
        speed = kwargs.get("speed", 2.0)
        start_time = time.time()

        self.drone.go_to.go_to_point([0.0, 0.0, 5.0], speed=speed)
        self._wait_for_arrival((0.0, 0.0, 5.0), timeout=60.0)

        self.drone.land(speed=0.5)
        self._wait_for_landing(timeout=15.0)
        self._is_flying = False

        return {
            "status": "success",
            "return_status": "landed",
            "landing_position": self.world_model.drone.position,
            "energy_used": "high",
            "duration_s": time.time() - start_time,
        }

    def _execute_generate_report(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate report (handled by MissionLogger, not executor)."""
        return {
            "status": "success",
            "report_path": str(self.output_dir / "mission_report.md"),
            "report_summary": "Mission completed. Report generated by MissionLogger.",
            "energy_used": "low",
            "duration_s": 0.1,
        }

    def _default_handler(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "info": f"Capability {capability.name} executed (no specialized handler)",
            "energy_used": "low",
            "duration_s": 0.1,
        }

    def _wait_for_takeoff(self, target_height: float, timeout: float = 15.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.world_model.drone.altitude_m >= target_height * 0.8:
                return True
            time.sleep(0.2)
        return False

    def _wait_for_arrival(
        self, target: tuple[float, ...], timeout: float = 30.0, tolerance: float = 1.0
    ) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            pos = self.world_model.drone.position
            dist = math.sqrt(
                (pos[0] - target[0]) ** 2
                + (pos[1] - target[1]) ** 2
                + (pos[2] - target[2]) ** 2
            )
            if dist < tolerance:
                return True
            time.sleep(0.2)
        return False

    def _wait_for_landing(self, timeout: float = 15.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.world_model.drone.altitude_m < 0.3:
                return True
            time.sleep(0.2)
        return False

    def shutdown(self) -> None:
        """Clean up AS2 DroneInterface."""
        try:
            self.drone.shutdown()
        except Exception:
            pass

    _handlers: dict[str, Any] = {
        "fly_to_area": _execute_fly_to_area,
        "scan_area": _execute_scan_area,
        "detect_smoke": _execute_detect_smoke,
        "detect_crowd": _execute_detect_crowd,
        "inspect_rooftop": _execute_inspect_rooftop,
        "evaluate_image_quality": _execute_evaluate_image_quality,
        "reobserve_from_new_angle": _execute_reobserve,
        "hold_position": _execute_hold_position,
        "return_home": _execute_return_home,
        "generate_report": _execute_generate_report,
    }
