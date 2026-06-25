"""Embodied Mission Orchestrator - SayCan-style capability selection and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capability_registry import CapabilityRegistry, CapabilitySpec
from .mission_parser import Mission
from .safety_shield import SafetyShield
from .world_model_base import WorldModelBase


@dataclass
class ActionScore:
    """Score breakdown for a candidate action."""

    capability_name: str
    task_relevance: float = 0.0
    capability_feasibility: float = 0.0
    safety_feasibility: float = 0.0
    resource_utility: float = 0.0
    user_preference: float = 0.0
    total_score: float = 0.0

    def compute_total(self) -> float:
        self.total_score = (
            self.task_relevance
            * self.capability_feasibility
            * self.safety_feasibility
            * self.resource_utility
            * self.user_preference
        )
        return self.total_score


@dataclass
class OrchestratorStep:
    """A single orchestrator decision step."""

    step_number: int
    selected_capability: str
    score: ActionScore
    safety_passed: bool
    result: str = ""
    replan_triggered: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """SayCan-style mission orchestrator.

    Selects capabilities based on multiplicative scoring:
    score = task_relevance * feasibility * safety * resource * preference
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        safety_shield: SafetyShield,
        world_model: WorldModelBase,
    ) -> None:
        self.registry = registry
        self.safety_shield = safety_shield
        self.world_model = world_model
        self._score_threshold = 0.3
        self._mission_plan: list[str] = []
        self._plan_index: int = 0
        self._executed_capabilities: list[str] = []

    def plan_mission(self, mission: Mission) -> list[str]:
        """Generate initial capability sequence for a mission.

        MVP: rule-based plan generation based on mission targets.
        """
        plan: list[str] = []

        # Always start with navigation to area
        plan.append("fly_to_area")

        # Scan the area
        plan.append("scan_area")

        # Detection capabilities based on priority targets
        target_to_capability = {
            "smoke": "detect_smoke",
            "crowd": "detect_crowd",
            "rooftop_anomaly": "inspect_rooftop",
        }
        for target in mission.priority_targets:
            cap = target_to_capability.get(target)
            if cap and cap in self.registry.get_names():
                plan.append(cap)

        # Always end with report
        plan.append("generate_report")

        self._mission_plan = plan
        return plan

    def select_next_action(
        self,
        mission: Mission,
        current_step: int,
        execution_history: list[OrchestratorStep],
    ) -> ActionScore | None:
        """Select the next best action using SayCan-style scoring."""
        # If we have a plan and haven't finished it, follow the plan
        if self._plan_index < len(self._mission_plan):
            planned_cap = self._mission_plan[self._plan_index]
            self._plan_index += 1
            # Score the planned capability
            score = self._score_capability(planned_cap, mission)
            if score.total_score >= self._score_threshold:
                self._executed_capabilities.append(planned_cap)
                return score

        # Plan exhausted or planned action scored too low - mission complete
        # Only select alternatives if plan failed, not after plan completion
        if self._plan_index >= len(self._mission_plan):
            return None

        # Planned action scored too low, find best alternative (excluding already executed)
        candidates = self.registry.list_all()
        best_score: ActionScore | None = None

        for cap in candidates:
            if cap.name in self._executed_capabilities:
                continue
            score = self._score_capability(cap.name, mission)
            if best_score is None or score.total_score > best_score.total_score:
                best_score = score

        if best_score and best_score.total_score >= self._score_threshold:
            self._executed_capabilities.append(best_score.capability_name)
            return best_score
        return None

    def _score_capability(self, cap_name: str, mission: Mission) -> ActionScore:
        """Compute SayCan-style score for a capability."""
        cap = self.registry.get(cap_name)
        if cap is None:
            return ActionScore(capability_name=cap_name)

        score = ActionScore(capability_name=cap_name)

        # Task relevance: how relevant is this capability to current mission state
        score.task_relevance = self._compute_task_relevance(cap, mission)

        # Capability feasibility: can this capability execute now
        score.capability_feasibility = self._compute_feasibility(cap)

        # Safety feasibility: does it pass safety checks
        score.safety_feasibility = self._compute_safety_score(cap, mission)

        # Resource utility: is it resource-efficient
        score.resource_utility = self._compute_resource_utility(cap)

        # User preference: does it align with user priorities
        score.user_preference = self._compute_user_preference(cap, mission)

        score.compute_total()
        return score

    def _compute_task_relevance(self, cap: CapabilitySpec, mission: Mission) -> float:
        """Compute task relevance score."""
        # High relevance for capabilities matching priority targets
        target_capabilities = {
            "smoke": "detect_smoke",
            "crowd": "detect_crowd",
            "rooftop_anomaly": "inspect_rooftop",
        }
        for target in mission.priority_targets:
            if target_capabilities.get(target) == cap.name:
                return 0.95

        # Navigation and scanning are always relevant
        if cap.cap_type == "navigation":
            return 0.85
        if cap.cap_type == "perception":
            return 0.80
        if cap.cap_type == "reporting":
            return 0.70

        return 0.5

    def _compute_feasibility(self, cap: CapabilitySpec) -> float:
        """Compute capability feasibility based on preconditions."""
        precond_state = self.world_model.get_precondition_state()
        if not cap.preconditions:
            return 1.0

        met = sum(1 for p in cap.preconditions if precond_state.get(p, False))
        return met / len(cap.preconditions)

    def _compute_safety_score(self, cap: CapabilitySpec, mission: Mission) -> float:
        """Compute safety feasibility."""
        safety_state = self.world_model.get_safety_state()
        mission_constraints = {
            "max_altitude_m": mission.constraints.max_altitude_m,
            "reserve_battery_percent": mission.constraints.reserve_battery_percent,
            "max_wind_speed_mps": mission.constraints.max_wind_speed_mps,
        }
        return self.safety_shield.get_safety_score(safety_state, mission_constraints)

    def _compute_resource_utility(self, cap: CapabilitySpec) -> float:
        """Compute resource utility score."""
        energy_scores = {"low": 1.0, "medium": 0.8, "high": 0.6}
        time_scores = {"low": 1.0, "medium": 0.85, "high": 0.7}

        energy_score = energy_scores.get(cap.resources.get("energy", "low"), 0.7)
        time_score = time_scores.get(cap.resources.get("time", "low"), 0.7)

        # Factor in remaining battery
        battery = self.world_model.drone.battery_percent
        battery_factor = min(1.0, battery / 50.0)

        return energy_score * time_score * battery_factor

    def _compute_user_preference(self, cap: CapabilitySpec, mission: Mission) -> float:
        """Compute user preference alignment."""
        # Higher priority targets get higher preference
        target_caps = {
            "smoke": "detect_smoke",
            "crowd": "detect_crowd",
            "rooftop_anomaly": "inspect_rooftop",
        }
        for i, target in enumerate(mission.priority_targets):
            if target_caps.get(target) == cap.name:
                return 1.0 - (i * 0.05)  # Slight decay by priority order

        return 0.85  # Default preference for non-target capabilities

    def should_replan(self, last_result: dict[str, Any], mission: Mission) -> tuple[bool, str]:
        """Determine if replanning is needed based on execution result."""
        # Check policies
        for policy in mission.policies:
            if policy.condition == "low_image_quality":
                quality = last_result.get("image_quality", 1.0)
                if quality < policy.threshold:
                    return True, policy.action
            elif policy.condition == "high_wind":
                wind = self.world_model.environment.wind_speed_mps
                if wind > policy.threshold:
                    return True, policy.action
            elif policy.condition == "low_battery":
                battery = self.world_model.drone.battery_percent
                if battery < policy.threshold:
                    return True, policy.action
            elif policy.condition == "low_confidence_detection":
                confidence = last_result.get("target_confidence", 1.0)
                if confidence < policy.threshold:
                    return True, policy.action

        return False, ""
