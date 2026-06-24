"""Capability Registry - manages UAV capability units."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CapabilitySpec:
    """Specification of a single capability unit."""

    name: str
    description: str
    cap_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    resources: dict[str, str] = field(default_factory=dict)
    safety: dict[str, bool] = field(default_factory=dict)
    invocation: dict[str, str] = field(default_factory=dict)
    fallback: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> CapabilitySpec:
        metadata = data.get("metadata", {})
        spec = data.get("spec", {})
        return cls(
            name=metadata.get("name", ""),
            description=metadata.get("description", ""),
            cap_type=spec.get("type", ""),
            inputs=spec.get("inputs", []),
            outputs=spec.get("outputs", []),
            preconditions=spec.get("preconditions", []),
            resources=spec.get("resources", {}),
            safety=spec.get("safety", {}),
            invocation=spec.get("invocation", {}),
            fallback=spec.get("fallback", []),
        )


class CapabilityRegistry:
    """Registry for managing all available capability units."""

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilitySpec] = {}

    def load_from_directory(self, directory: str | Path) -> int:
        """Load all capability YAML files from a directory."""
        directory = Path(directory)
        count = 0
        for yaml_file in sorted(directory.glob("*.yaml")):
            self.load_from_file(yaml_file)
            count += 1
        return count

    def load_from_file(self, filepath: str | Path) -> CapabilitySpec:
        """Load a single capability from a YAML file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cap = CapabilitySpec.from_yaml(data)
        self._capabilities[cap.name] = cap
        return cap

    def get(self, name: str) -> CapabilitySpec | None:
        """Get a capability by name."""
        return self._capabilities.get(name)

    def list_all(self) -> list[CapabilitySpec]:
        """List all registered capabilities."""
        return list(self._capabilities.values())

    def list_by_type(self, cap_type: str) -> list[CapabilitySpec]:
        """List capabilities filtered by type."""
        return [c for c in self._capabilities.values() if c.cap_type == cap_type]

    def get_names(self) -> list[str]:
        """Get all registered capability names."""
        return list(self._capabilities.keys())

    def check_preconditions(self, name: str, world_state: dict[str, Any]) -> bool:
        """Check if all preconditions for a capability are met."""
        cap = self.get(name)
        if cap is None:
            return False
        for precond in cap.preconditions:
            if not world_state.get(precond, False):
                return False
        return True

    @property
    def count(self) -> int:
        return len(self._capabilities)
