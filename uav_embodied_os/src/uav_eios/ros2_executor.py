"""ROS2 Executor - controls PX4 via native FMU ROS2 topics.

Uses /fmu/in/offboard_control_mode, /fmu/in/trajectory_setpoint,
/fmu/in/vehicle_command to directly control PX4 SITL.
Completely independent of Aerostack2 - pure PX4 ROS2 interface.

NED coordinate system: X=North, Y=East, Z=Down (negative Z = altitude).
Offboard mode must be set BEFORE arming.
"""

from __future__ import annotations

import math
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile,
        ReliabilityPolicy,
        DurabilityPolicy,
        HistoryPolicy,
    )
    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
        VehicleLocalPosition,
        VehicleStatus,
        VehicleLandDetected,
        BatteryStatus,
    )

    PX4_MSGS_AVAILABLE = True
except ImportError:
    PX4_MSGS_AVAILABLE = False

from .capability_registry import CapabilitySpec
from .image_analyzer import ImageAnalyzer
from .world_model_base import WorldModelBase


# PX4 vehicle command codes
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
VEHICLE_CMD_DO_SET_MODE = 176

# PX4 navigation state for offboard
NAVIGATION_STATE_OFFBOARD = 14


class PX4FlightController:
    """Low-level PX4 flight controller via ROS2 FMU topics.

    Implements the full offboard control protocol:
    1. Publish OffboardControlMode at >2Hz (required by PX4)
    2. Switch to Offboard mode via VehicleCommand
    3. Arm via VehicleCommand
    4. Publish TrajectorySetpoint for position control

    NED Frame: Z negative = up. Altitude 5m = z=-5.0
    """

    def __init__(self, node: Node) -> None:
        self._node = node

        # QoS for PX4 topics (must match px4_ros_com)
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self._offboard_pub = node.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos
        )
        self._trajectory_pub = node.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos
        )
        self._command_pub = node.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", px4_qos
        )

        # Subscribers for state feedback
        self._local_pos: VehicleLocalPosition | None = None
        self._vehicle_status: VehicleStatus | None = None
        self._land_detected: VehicleLandDetected | None = None
        self._battery_status: BatteryStatus | None = None
        self._state_lock = threading.Lock()

        node.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self._local_pos_cb,
            px4_qos,
        )
        node.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self._vehicle_status_cb,
            px4_qos,
        )
        node.create_subscription(
            VehicleLandDetected,
            "/fmu/out/vehicle_land_detected",
            self._land_detected_cb,
            px4_qos,
        )
        node.create_subscription(
            BatteryStatus,
            "/fmu/out/battery_status_v1",
            self._battery_status_cb,
            px4_qos,
        )

        # Offboard heartbeat timer (10 Hz - required by PX4)
        self._target_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._target_yaw: float = 0.0
        self._offboard_active = False
        self._heartbeat_timer = node.create_timer(0.1, self._heartbeat_callback)

        self._is_armed = False
        self._is_offboard = False
        self._offboard_setpoint_count = 0

    def _local_pos_cb(self, msg: VehicleLocalPosition) -> None:
        with self._state_lock:
            self._local_pos = msg

    def _vehicle_status_cb(self, msg: VehicleStatus) -> None:
        with self._state_lock:
            self._vehicle_status = msg
            self._is_armed = msg.arming_state == 2  # ARMED
            self._is_offboard = msg.nav_state == NAVIGATION_STATE_OFFBOARD

    def _land_detected_cb(self, msg: VehicleLandDetected) -> None:
        with self._state_lock:
            self._land_detected = msg

    def _battery_status_cb(self, msg: BatteryStatus) -> None:
        with self._state_lock:
            self._battery_status = msg

    def _heartbeat_callback(self) -> None:
        """Publish offboard control mode and trajectory setpoint at 10Hz.

        PX4 requires continuous offboard commands to stay in offboard mode.
        Must send at least 2Hz; we send at 10Hz for stability.
        """
        if not self._offboard_active:
            return

        # Publish offboard control mode (position control)
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
        self._offboard_pub.publish(msg)

        # Publish trajectory setpoint
        sp = TrajectorySetpoint()
        sp.position[0] = float(self._target_position[0])  # X (North)
        sp.position[1] = float(self._target_position[1])  # Y (East)
        sp.position[2] = float(self._target_position[2])  # Z (Down, negative=up)
        sp.yaw = self._target_yaw
        sp.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
        self._trajectory_pub.publish(sp)

        self._offboard_setpoint_count += 1

    def _publish_vehicle_command(self, command: int, param1: float = 0.0,
                                  param2: float = 0.0, param7: float = 0.0) -> None:
        """Publish a VehicleCommand message."""
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
        self._command_pub.publish(msg)

    def start_offboard(self, initial_position: tuple[float, float, float]) -> None:
        """Start offboard mode: begin publishing setpoints, then switch mode.

        PX4 requires receiving offboard setpoints BEFORE switching to offboard mode.
        We publish for ~1 second (10+ messages) before commanding the mode switch.
        """
        self._target_position = initial_position
        self._offboard_active = True
        self._offboard_setpoint_count = 0

        # Wait for enough setpoints to be sent (PX4 needs > 10 messages)
        self._node.get_logger().info("Sending initial offboard setpoints...")
        start = time.time()
        while self._offboard_setpoint_count < 15 and time.time() - start < 3.0:
            time.sleep(0.1)

    def set_offboard_mode(self) -> None:
        """Command PX4 to switch to Offboard mode."""
        self._publish_vehicle_command(
            VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,  # base mode
            param2=6.0,  # custom mode: offboard
        )
        self._node.get_logger().info("Commanded: Switch to Offboard mode")

    def arm(self) -> None:
        """Command PX4 to arm."""
        self._publish_vehicle_command(
            VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,  # 1.0 = arm
        )
        self._node.get_logger().info("Commanded: ARM")

    def disarm(self) -> None:
        """Command PX4 to disarm."""
        self._publish_vehicle_command(
            VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,  # 0.0 = disarm
        )
        self._node.get_logger().info("Commanded: DISARM")

    def set_target(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        """Set target position in NED frame. z negative = altitude."""
        self._target_position = (x, y, z)
        self._target_yaw = yaw

    def land(self) -> None:
        """Command PX4 to land (switch to auto land mode)."""
        self._publish_vehicle_command(
            VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=4.0,  # auto mode
            param7=6.0,  # auto land sub-mode
        )
        self._node.get_logger().info("Commanded: LAND")

    def get_position(self) -> tuple[float, float, float]:
        """Get current local position (NED)."""
        with self._state_lock:
            if self._local_pos is not None:
                return (self._local_pos.x, self._local_pos.y, self._local_pos.z)
        return (0.0, 0.0, 0.0)

    def get_altitude(self) -> float:
        """Get altitude in meters (positive value, converted from NED z)."""
        with self._state_lock:
            if self._local_pos is not None:
                return -self._local_pos.z  # NED: -z = altitude above ground
        return 0.0

    def get_battery_percent(self) -> float:
        """Get battery percentage."""
        with self._state_lock:
            if self._battery_status is not None:
                return self._battery_status.remaining * 100.0
        return 100.0

    @property
    def is_armed(self) -> bool:
        return self._is_armed

    @property
    def is_offboard(self) -> bool:
        return self._is_offboard

    @property
    def is_landed(self) -> bool:
        with self._state_lock:
            if self._land_detected is not None:
                return self._land_detected.landed
        return True

    def stop_offboard(self) -> None:
        """Stop the offboard heartbeat."""
        self._offboard_active = False


class ROS2Executor:
    """Executes capabilities using native PX4 ROS2 FMU topics.

    Flight control via /fmu/in/offboard_control_mode + /fmu/in/trajectory_setpoint.
    NO Aerostack2 dependency - direct PX4 ROS2 interface.

    Coordinate system: NED (X=North, Y=East, Z=Down)
    - Flying at 5m altitude = z = -5.0
    - Waypoint [10, 10, 5] in user spec means NED position (10, 10, -5)

    Startup sequence: Offboard mode FIRST, then Arm.
    """

    def __init__(
        self,
        world_model: WorldModelBase,
        node_name: str = "uav_eios_executor",
        output_dir: str | Path = "outputs",
    ) -> None:
        if not PX4_MSGS_AVAILABLE:
            raise RuntimeError(
                "px4_msgs not available. Install: "
                "sudo apt install ros-humble-px4-msgs or build from source."
            )

        self.world_model = world_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._image_analyzer = ImageAnalyzer()
        self._execution_count = 0

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(node_name)
        self._flight_ctrl = PX4FlightController(self._node)

        # Spin in background thread
        self._spinning = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        self._is_flying = False
        self._home_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

        self._node.get_logger().info("ROS2Executor initialized (native PX4 FMU topics)")

    def _spin_loop(self) -> None:
        while self._spinning and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.02)

    @staticmethod
    def ensure_ned_altitude(z: float) -> float:
        """Ensure z is negative for NED (altitude above ground).

        In NED, flying up means z < 0. Users/LLM often specify altitude as
        positive (e.g., 5.0 for 5m altitude). We convert: if z > 0, negate it.
        """
        if z > 0:
            return -z
        return z

    def execute(self, capability: CapabilitySpec, **kwargs: Any) -> dict[str, Any]:
        """Execute a capability and return results."""
        self._execution_count += 1
        handler = self._handlers.get(capability.name, self._default_handler)
        result = handler(self, capability, **kwargs)
        self.world_model.update_after_action(capability.name, result)
        return result

    def _arm_and_takeoff(self, altitude_m: float) -> bool:
        """Full startup sequence: Offboard → Arm → climb to altitude.

        PX4 requires:
        1. Publish offboard setpoints for >1 second
        2. Switch to offboard mode
        3. Arm the vehicle
        4. Vehicle climbs to target altitude
        """
        ned_z = self.ensure_ned_altitude(altitude_m)
        self._home_position = self._flight_ctrl.get_position()

        # Start at current XY position, target altitude
        current_pos = self._flight_ctrl.get_position()
        target = (current_pos[0], current_pos[1], ned_z)

        # Step 1: Begin sending offboard setpoints (PX4 needs >10 before mode switch)
        self._flight_ctrl.start_offboard(target)

        # Step 2: Switch to Offboard mode FIRST
        self._flight_ctrl.set_offboard_mode()
        time.sleep(0.5)

        # Retry mode switch if needed
        for _ in range(5):
            if self._flight_ctrl.is_offboard:
                break
            self._flight_ctrl.set_offboard_mode()
            time.sleep(0.5)

        # Step 3: Arm AFTER offboard mode is set
        self._flight_ctrl.arm()
        time.sleep(0.5)

        # Retry arm if needed
        for _ in range(5):
            if self._flight_ctrl.is_armed:
                break
            self._flight_ctrl.arm()
            time.sleep(0.5)

        if not self._flight_ctrl.is_armed:
            self._node.get_logger().error("Failed to arm vehicle!")
            return False

        self._node.get_logger().info(
            f"Armed and in Offboard mode. Climbing to altitude {altitude_m}m (z={ned_z})"
        )

        # Step 4: Wait for takeoff (reach target altitude)
        self._is_flying = self._wait_for_altitude(altitude_m, timeout=15.0)
        return self._is_flying

    def _execute_fly_to_area(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Takeoff (if needed) and fly to target position."""
        target = kwargs.get("target_position", (10.0, 10.0, 5.0))
        speed = kwargs.get("speed", 2.0)

        # Convert user altitude to NED (z negative)
        ned_x = float(target[0])
        ned_y = float(target[1])
        ned_z = self.ensure_ned_altitude(float(target[2]))

        start_time = time.time()

        # Takeoff if not already flying
        if not self._is_flying:
            altitude = abs(ned_z)  # positive altitude for takeoff
            if not self._arm_and_takeoff(altitude):
                return {
                    "status": "failed",
                    "error": "arm_takeoff_failed",
                    "energy_used": "low",
                    "duration_s": time.time() - start_time,
                }

        # Fly to target position
        self._flight_ctrl.set_target(ned_x, ned_y, ned_z)
        self._node.get_logger().info(
            f"Flying to NED ({ned_x:.1f}, {ned_y:.1f}, {ned_z:.1f})"
        )

        arrived = self._wait_for_arrival((ned_x, ned_y, ned_z), timeout=60.0)
        duration = time.time() - start_time

        return {
            "status": "success" if arrived else "timeout",
            "arrival_status": "arrived" if arrived else "timeout",
            "target_position": target,
            "target_ned": (ned_x, ned_y, ned_z),
            "final_position_ned": self._flight_ctrl.get_position(),
            "energy_used": "high",
            "duration_s": duration,
        }

    def _execute_hold_position(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Hold current position for a specified duration."""
        hold_duration = kwargs.get("duration", 3.0)
        # Target stays at current position (heartbeat keeps publishing)
        time.sleep(hold_duration)
        return {
            "status": "success",
            "hold_duration_s": hold_duration,
            "position_ned": self._flight_ctrl.get_position(),
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
        """Reobserve from a new angle - shift position and recapture.

        Offset in NED: positive offset_x = move north, offset_z positive = go HIGHER
        (we negate offset_z since NED z is down).
        """
        offset_x = kwargs.get("offset_x", 3.0)
        offset_y = kwargs.get("offset_y", 0.0)
        offset_z = kwargs.get("offset_z", 1.0)  # user means "go 1m higher"

        current_pos = self._flight_ctrl.get_position()
        # NED: to go higher, subtract from z
        new_pos = (
            current_pos[0] + offset_x,
            current_pos[1] + offset_y,
            current_pos[2] - offset_z,  # higher = more negative z in NED
        )

        start_time = time.time()

        self._flight_ctrl.set_target(new_pos[0], new_pos[1], new_pos[2])
        self._wait_for_arrival(new_pos, timeout=15.0)

        time.sleep(1.0)

        result = self._execute_capture_and_evaluate(
            capability, waypoint_name=f"reobserve_{self._execution_count}", **kwargs
        )
        result["new_position_ned"] = new_pos
        result["duration_s"] = time.time() - start_time
        result["energy_used"] = "medium"
        return result

    def _execute_return_home(
        self, capability: CapabilitySpec, **kwargs: Any
    ) -> dict[str, Any]:
        """Return to home position and land."""
        start_time = time.time()

        # Fly back to home position at current altitude first
        current_pos = self._flight_ctrl.get_position()
        home_xy_z = (self._home_position[0], self._home_position[1], current_pos[2])
        self._flight_ctrl.set_target(home_xy_z[0], home_xy_z[1], home_xy_z[2])
        self._node.get_logger().info("Returning to home position...")
        self._wait_for_arrival(home_xy_z, timeout=60.0)

        # Land
        self._flight_ctrl.land()
        self._wait_for_landing(timeout=20.0)
        self._is_flying = False

        # Disarm after landing
        time.sleep(2.0)
        self._flight_ctrl.disarm()

        return {
            "status": "success",
            "return_status": "landed",
            "landing_position_ned": self._flight_ctrl.get_position(),
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

    # --- Wait helpers ---

    def _wait_for_altitude(self, target_alt_m: float, timeout: float = 15.0) -> bool:
        """Wait until drone reaches target altitude (positive meters)."""
        start = time.time()
        while time.time() - start < timeout:
            current_alt = self._flight_ctrl.get_altitude()
            if current_alt >= target_alt_m * 0.8:
                return True
            time.sleep(0.2)
        self._node.get_logger().warn(
            f"Altitude timeout: current={self._flight_ctrl.get_altitude():.1f}m, "
            f"target={target_alt_m:.1f}m"
        )
        return False

    def _wait_for_arrival(
        self, target_ned: tuple[float, float, float],
        timeout: float = 30.0, tolerance: float = 1.0
    ) -> bool:
        """Wait until drone reaches target position in NED coordinates."""
        start = time.time()
        while time.time() - start < timeout:
            pos = self._flight_ctrl.get_position()
            dist = math.sqrt(
                (pos[0] - target_ned[0]) ** 2
                + (pos[1] - target_ned[1]) ** 2
                + (pos[2] - target_ned[2]) ** 2
            )
            if dist < tolerance:
                return True
            time.sleep(0.2)
        self._node.get_logger().warn(
            f"Arrival timeout: dist={dist:.1f}m to target {target_ned}"
        )
        return False

    def _wait_for_landing(self, timeout: float = 20.0) -> bool:
        """Wait until drone has landed."""
        start = time.time()
        while time.time() - start < timeout:
            if self._flight_ctrl.is_landed:
                return True
            time.sleep(0.3)
        return False

    def shutdown(self) -> None:
        """Stop offboard control and clean up."""
        self._flight_ctrl.stop_offboard()
        self._spinning = False
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._node.destroy_node()

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
