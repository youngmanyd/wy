"""UAV Embodied Intelligence Operating System - Core Package."""

__version__ = "0.2.0"

from uav_eios.capability_registry import CapabilityRegistry
from uav_eios.executor import MockExecutor
from uav_eios.logger import MissionLogger
from uav_eios.mission_parser import MissionParser
from uav_eios.orchestrator import Orchestrator
from uav_eios.safety_shield import SafetyShield
from uav_eios.world_model import MockWorldModel
from uav_eios.world_model_base import WorldModelBase
from uav_eios.image_analyzer import ImageAnalyzer

try:
    from uav_eios.ros2_world_model import ROS2WorldModel
except ImportError:
    ROS2WorldModel = None  # type: ignore[assignment,misc]

try:
    from uav_eios.ros2_executor import ROS2Executor
except ImportError:
    ROS2Executor = None  # type: ignore[assignment,misc]

try:
    from uav_eios.llm_task_parser import LLMTaskParser, LLMConfig
except ImportError:
    LLMTaskParser = None  # type: ignore[assignment,misc]
    LLMConfig = None  # type: ignore[assignment,misc]

__all__ = [
    "CapabilityRegistry",
    "ImageAnalyzer",
    "LLMConfig",
    "LLMTaskParser",
    "MissionLogger",
    "MissionParser",
    "MockExecutor",
    "MockWorldModel",
    "Orchestrator",
    "ROS2Executor",
    "ROS2WorldModel",
    "SafetyShield",
    "WorldModelBase",
]
