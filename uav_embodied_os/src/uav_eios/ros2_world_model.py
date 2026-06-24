"""ROS2 World Model - real UAV and environment state from Gazebo simulation.

Subscribes to ROS2 topics for drone pose, battery, GPS, and camera data.
Replaces MockWorldModel for real simulation execution.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image, BatteryState, NavSatFix
    from cv_bridge import CvBridge

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

from .world_model_base import AreaState, DroneState, EnvironmentState, WorldModelBase


class ROS2WorldModel(WorldModelBase):
    """World model backed by real ROS2 topic subscriptions.

    Subscribes to:
      - /drone0/self_localization/pose   -> drone position, altitude, heading
      - /drone0/sensor_measurements/battery -> battery percentage
      - /drone0/sensor_measurements/gps  -> GPS quality
      - /camera/image_raw               -> RGB images
      - /camera/depth_raw               -> depth images
    """

    def __init__(self, node_name: str = "uav_eios_world_model") -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "ROS2 Python packages not available. "
                "Ensure rclpy, sensor_msgs, geometry_msgs, cv_bridge are installed."
            )

        self.drone = DroneState()
        self.environment = EnvironmentState()
        self.areas: dict[str, AreaState] = {}
        self._step_count = 0

        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._rgb_timestamp: float = 0.0
        self._depth_timestamp: float = 0.0
        self._lock = threading.Lock()

        self._bridge = CvBridge()

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(node_name)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._node.create_subscription(
            PoseStamped,
            "/drone0/self_localization/pose",
            self._pose_callback,
            sensor_qos,
        )
        self._node.create_subscription(
            BatteryState,
            "/drone0/sensor_measurements/battery",
            self._battery_callback,
            sensor_qos,
        )
        self._node.create_subscription(
            NavSatFix,
            "/drone0/sensor_measurements/gps",
            self._gps_callback,
            sensor_qos,
        )
        self._node.create_subscription(
            Image,
            "/camera/image_raw",
            self._rgb_callback,
            sensor_qos,
        )
        self._node.create_subscription(
            Image,
            "/camera/depth_raw",
            self._depth_callback,
            sensor_qos,
        )

        self._spinning = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        self._node.get_logger().info("ROS2WorldModel initialized, waiting for topics...")

    def _spin_loop(self) -> None:
        while self._spinning and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _pose_callback(self, msg: PoseStamped) -> None:
        with self._lock:
            pos = msg.pose.position
            self.drone.position = (pos.x, pos.y, pos.z)
            self.drone.altitude_m = pos.z

            import math
            q = msg.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            self.drone.heading_deg = math.degrees(yaw_rad)

            self.drone.is_flying = pos.z > 0.3

    def _battery_callback(self, msg: BatteryState) -> None:
        with self._lock:
            self.drone.battery_percent = msg.percentage * 100.0

    def _gps_callback(self, msg: NavSatFix) -> None:
        with self._lock:
            status = msg.status.status
            if status >= 0:
                self.environment.gps_quality = min(1.0, 0.6 + status * 0.2)
            else:
                self.environment.gps_quality = 0.3

    def _rgb_callback(self, msg: Image) -> None:
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self._latest_rgb = cv_image
                self._rgb_timestamp = time.time()
        except Exception as e:
            self._node.get_logger().warn(f"Failed to convert RGB image: {e}")

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._lock:
                self._latest_depth = depth_image
                self._depth_timestamp = time.time()
        except Exception as e:
            self._node.get_logger().warn(f"Failed to convert depth image: {e}")

    def initialize_mission(self, area_id: str) -> None:
        self.areas[area_id] = AreaState(area_id=area_id)
        self._step_count = 0
        self._node.get_logger().info(f"Mission initialized for area: {area_id}")

    def get_drone_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "position": self.drone.position,
                "battery_percent": self.drone.battery_percent,
                "altitude_m": self.drone.altitude_m,
                "speed_mps": self.drone.speed_mps,
                "heading_deg": self.drone.heading_deg,
                "is_flying": self.drone.is_flying,
                "is_armed": self.drone.is_armed,
            }

    def get_environment_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "wind_speed_mps": self.environment.wind_speed_mps,
                "gps_quality": self.environment.gps_quality,
                "comm_quality": self.environment.comm_quality,
                "obstacle_distance_m": self.environment.obstacle_distance_m,
                "visibility": self.environment.visibility,
            }

    def get_precondition_state(self, area_id: str = "") -> dict[str, bool]:
        with self._lock:
            return {
                "battery_above_reserve": self.drone.battery_percent > 25.0,
                "gps_quality_good": self.environment.gps_quality >= 0.7,
                "no_fly_zone_clear": True,
                "at_target_area": self.drone.is_flying and self.drone.altitude_m > 1.0,
                "weather_acceptable": self.environment.wind_speed_mps < 12.0,
                "sensor_available": self._latest_rgb is not None,
                "altitude_safe": self.drone.altitude_m <= 120.0,
                "target_position_known": True,
                "safe_viewpoint_available": True,
                "image_available": self._latest_rgb is not None,
                "mission_data_available": self._step_count > 0,
            }

    def get_safety_state(self) -> dict[str, float]:
        with self._lock:
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
        self._step_count += 1

        if capability_name in ("scan_area", "inspect_rooftop", "detect_smoke", "detect_crowd"):
            for area in self.areas.values():
                area.images_captured += 1
                area.coverage_percent = min(
                    1.0, area.coverage_percent + 1.0 / max(len(self.areas), 1) * 0.33
                )

    def get_latest_image(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_rgb is not None:
                return self._latest_rgb.copy()
            return None

    def get_latest_depth(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_depth is not None:
                return self._latest_depth.copy()
            return None

    def wait_for_fresh_image(self, timeout_s: float = 5.0) -> np.ndarray | None:
        """Wait for a new RGB image captured after this call."""
        start = time.time()
        threshold = time.time()
        while time.time() - start < timeout_s:
            with self._lock:
                if self._rgb_timestamp > threshold and self._latest_rgb is not None:
                    return self._latest_rgb.copy()
            time.sleep(0.1)
        return self.get_latest_image()

    def shutdown(self) -> None:
        self._spinning = False
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._node.destroy_node()
