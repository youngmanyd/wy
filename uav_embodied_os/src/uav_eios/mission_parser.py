"""Mission Parser - parses mission YAML into structured mission objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MissionConstraints:
    """Mission-level constraints."""

    max_altitude_m: float = 120.0
    reserve_battery_percent: float = 25.0
    max_wind_speed_mps: float = 12.0
    avoid_no_fly_zones: bool = True
    require_safe_distance: bool = True
    max_mission_duration_s: float = 1800.0


@dataclass
class MissionPolicy:
    """A policy defining response to a specific condition."""

    condition: str
    action: str
    threshold: float = 0.0
    fallback: str = ""


@dataclass
class SuccessCriteria:
    """Mission success criteria."""

    min_area_coverage: float = 0.90
    min_detection_confidence: float = 0.75
    min_image_quality: float = 0.6
    report_required: bool = True


@dataclass
class Waypoint:
    """A single inspection waypoint."""

    name: str
    position: tuple[float, float, float]
    tasks: list[str] = field(default_factory=list)


@dataclass
class Mission:
    """Parsed mission object."""

    name: str
    description: str
    objective: str
    area: str
    priority_targets: list[str] = field(default_factory=list)
    constraints: MissionConstraints = field(default_factory=MissionConstraints)
    policies: list[MissionPolicy] = field(default_factory=list)
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    max_replan_attempts: int = 5
    waypoints: list[Waypoint] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> Mission:
        metadata = data.get("metadata", {})
        spec = data.get("spec", {})

        constraints_data = spec.get("constraints", {})
        constraints = MissionConstraints(
            max_altitude_m=constraints_data.get("max_altitude_m", 120.0),
            reserve_battery_percent=constraints_data.get("reserve_battery_percent", 25.0),
            max_wind_speed_mps=constraints_data.get("max_wind_speed_mps", 12.0),
            avoid_no_fly_zones=constraints_data.get("avoid_no_fly_zones", True),
            require_safe_distance=constraints_data.get("require_safe_distance", True),
            max_mission_duration_s=constraints_data.get("max_mission_duration_s", 1800.0),
        )

        policies = []
        for condition, policy_data in spec.get("policies", {}).items():
            if isinstance(policy_data, dict):
                policies.append(MissionPolicy(
                    condition=condition,
                    action=policy_data.get("action", ""),
                    threshold=policy_data.get("threshold", 0.0),
                    fallback=policy_data.get("fallback", ""),
                ))

        criteria_data = spec.get("success_criteria", {})
        success_criteria = SuccessCriteria(
            min_area_coverage=criteria_data.get("min_area_coverage", 0.90),
            min_detection_confidence=criteria_data.get("min_detection_confidence", 0.75),
            min_image_quality=criteria_data.get("min_image_quality", 0.6),
            report_required=criteria_data.get("report_required", True),
        )

        waypoints = []
        for wp_data in spec.get("waypoints", []):
            pos = wp_data.get("position", [0.0, 0.0, 0.0])
            waypoints.append(Waypoint(
                name=wp_data.get("name", ""),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                tasks=wp_data.get("tasks", []),
            ))

        return cls(
            name=metadata.get("name", ""),
            description=metadata.get("description", ""),
            objective=spec.get("objective", ""),
            area=spec.get("area", ""),
            priority_targets=spec.get("priority_targets", []),
            constraints=constraints,
            policies=policies,
            success_criteria=success_criteria,
            max_replan_attempts=spec.get("max_replan_attempts", 5),
            waypoints=waypoints,
        )


class MissionParser:
    """Parser interface for missions. Supports YAML and dict input."""

    def parse_file(self, filepath: str | Path) -> Mission:
        """Parse a mission from a YAML file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Mission.from_yaml(data)

    def parse_dict(self, data: dict[str, Any]) -> Mission:
        """Parse a mission from a dict (e.g. from LLM output)."""
        yaml_format = {
            "metadata": {
                "name": data.get("name", ""),
                "description": data.get("description", ""),
            },
            "spec": {
                "objective": data.get("objective", "fly_inspect_report"),
                "area": data.get("area", ""),
                "priority_targets": data.get("priority_targets", []),
                "constraints": data.get("constraints", {}),
                "policies": data.get("policies", {}),
                "success_criteria": data.get("success_criteria", {}),
                "waypoints": data.get("waypoints", []),
                "max_replan_attempts": data.get("max_replan_attempts", 5),
            },
        }
        return Mission.from_yaml(yaml_format)

    def parse_natural_language(self, text: str) -> dict[str, Any]:
        """Parse natural language task into structured mission fields.

        Rule-based extraction. For LLM-based parsing, use LLMTaskParser.
        """
        result: dict[str, Any] = {
            "area": "",
            "priority_targets": [],
            "policies": {},
        }

        area_keywords = {
            "baylands": "baylands",
            "1号园区": "campus_1", "2号园区": "campus_2", "3号园区": "campus_3",
            "campus 1": "campus_1", "campus 2": "campus_2", "campus 3": "campus_3",
        }
        for keyword, area_id in area_keywords.items():
            if keyword in text.lower():
                result["area"] = area_id
                break

        target_keywords = {
            "烟雾": "smoke", "smoke": "smoke",
            "人员聚集": "crowd", "crowd": "crowd",
            "屋顶": "rooftop_anomaly", "rooftop": "rooftop_anomaly",
            "设备异常": "rooftop_anomaly",
        }
        for keyword, target in target_keywords.items():
            if keyword in text and target not in result["priority_targets"]:
                result["priority_targets"].append(target)

        policy_keywords = {
            "遮挡": ("occlusion_detected", "reobserve_from_new_angle"),
            "换角度": ("low_image_quality", "reobserve_from_new_angle"),
            "occlusion": ("occlusion_detected", "reobserve_from_new_angle"),
        }
        for keyword, (condition, action) in policy_keywords.items():
            if keyword in text:
                result["policies"][condition] = action

        return result
