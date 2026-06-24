"""Safety Shield - enforces safety constraints on capability execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SafetyRule:
    """A single safety rule."""

    name: str
    rule_type: str  # "hard" or "soft"
    description: str
    condition_field: str
    condition_op: str  # "<=", ">=", "==", "not"
    condition_value: Any
    penalty: float = 0.0
    fallback: str = ""
    action: str = ""


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    passed: bool
    hard_violations: list[str] = field(default_factory=list)
    soft_violations: list[str] = field(default_factory=list)
    total_penalty: float = 0.0
    recommended_fallback: str = ""


class SafetyShield:
    """Safety constraint checker for capability execution."""

    def __init__(self) -> None:
        self._rules: list[SafetyRule] = []

    def load_rules(self, filepath: str | Path) -> int:
        """Load safety rules from YAML file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        rules_data = data.get("rules", [])
        for rule_data in rules_data:
            rule = self._parse_rule(rule_data)
            if rule:
                self._rules.append(rule)
        return len(self._rules)

    def _parse_rule(self, data: dict[str, Any]) -> SafetyRule | None:
        """Parse a rule from YAML data."""
        condition_str = data.get("condition", "")
        field_name, op, value = self._parse_condition(condition_str)
        if not field_name:
            return None

        return SafetyRule(
            name=data.get("name", ""),
            rule_type=data.get("type", "soft"),
            description=data.get("description", ""),
            condition_field=field_name,
            condition_op=op,
            condition_value=value,
            penalty=data.get("penalty", 0.0),
            fallback=data.get("fallback", ""),
            action=data.get("action", ""),
        )

    def _parse_condition(self, condition: str) -> tuple[str, str, Any]:
        """Parse condition string into (field, operator, value)."""
        # Handle "not" conditions
        if condition.startswith("not "):
            field_name = condition[4:].strip()
            return field_name, "not", True

        # Handle comparison operators
        for op in ["<=", ">=", "==", "<", ">"]:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    field_name = parts[0].strip()
                    # Skip mission.constraints references (resolved at check time)
                    value_str = parts[1].strip()
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str
                    return field_name, op, value

        return "", "", None

    def check(self, world_state: dict[str, Any], mission_constraints: dict[str, Any] | None = None) -> SafetyCheckResult:
        """Check all safety rules against current world state."""
        result = SafetyCheckResult(passed=True)

        for rule in self._rules:
            violation = self._evaluate_rule(rule, world_state, mission_constraints)
            if violation:
                if rule.rule_type == "hard":
                    result.passed = False
                    result.hard_violations.append(rule.name)
                    if rule.action:
                        result.recommended_fallback = rule.action
                else:
                    result.soft_violations.append(rule.name)
                    result.total_penalty += rule.penalty
                    if rule.fallback and not result.recommended_fallback:
                        result.recommended_fallback = rule.fallback

        return result

    def _evaluate_rule(
        self,
        rule: SafetyRule,
        world_state: dict[str, Any],
        mission_constraints: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluate a single rule. Returns True if VIOLATED."""
        field_value = world_state.get(rule.condition_field)
        if field_value is None:
            return False  # Cannot evaluate, assume safe

        # Resolve condition value (may reference mission constraints)
        threshold = rule.condition_value
        if isinstance(threshold, str) and threshold.startswith("mission.constraints."):
            if mission_constraints:
                key = threshold.replace("mission.constraints.", "")
                threshold = mission_constraints.get(key, threshold)
            else:
                return False  # Cannot resolve, assume safe

        # Evaluate condition - returns True if VIOLATED (condition NOT met)
        try:
            if rule.condition_op == "<=":
                return not (float(field_value) <= float(threshold))
            elif rule.condition_op == ">=":
                return not (float(field_value) >= float(threshold))
            elif rule.condition_op == "==":
                return not (field_value == threshold)
            elif rule.condition_op == "not":
                return bool(field_value)  # Violated if field is True
            elif rule.condition_op == "<":
                return not (float(field_value) < float(threshold))
            elif rule.condition_op == ">":
                return not (float(field_value) > float(threshold))
        except (ValueError, TypeError):
            return False

        return False

    def get_safety_score(self, world_state: dict[str, Any], mission_constraints: dict[str, Any] | None = None) -> float:
        """Get overall safety score (1.0 = fully safe, 0.0 = critical)."""
        result = self.check(world_state, mission_constraints)
        if not result.passed:
            return 0.0
        return max(0.0, 1.0 - result.total_penalty)

    @property
    def rule_count(self) -> int:
        return len(self._rules)
