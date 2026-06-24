"""Mission Logger - records execution steps and generates reports."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LogEntry:
    """A single log entry for mission execution."""

    step: int
    capability: str
    score: float
    safety_passed: bool
    result: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    replan_triggered: bool = False

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class MissionLogger:
    """Records mission execution and generates reports."""

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[LogEntry] = []
        self._mission_name: str = ""
        self._start_time: str = ""

    def start_mission(self, mission_name: str) -> None:
        """Mark mission start."""
        self._mission_name = mission_name
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._entries = []

    def log_step(
        self,
        step: int,
        capability: str,
        score: float,
        safety_passed: bool,
        result: str,
        details: dict[str, Any] | None = None,
        replan_triggered: bool = False,
    ) -> None:
        """Log a single execution step."""
        entry = LogEntry(
            step=step,
            capability=capability,
            score=score,
            safety_passed=safety_passed,
            result=result,
            details=details or {},
            replan_triggered=replan_triggered,
        )
        self._entries.append(entry)

    def save_jsonl(self, filename: str = "") -> Path:
        """Save log as JSONL file."""
        if not filename:
            filename = f"{self._mission_name}_log.jsonl"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return filepath

    def generate_report(self, filename: str = "") -> Path:
        """Generate a markdown mission report."""
        if not filename:
            filename = f"{self._mission_name}_report.md"
        filepath = self.output_dir / filename

        report_lines = [
            f"# Mission Report: {self._mission_name}",
            "",
            f"**Start Time:** {self._start_time}",
            f"**End Time:** {datetime.now(timezone.utc).isoformat()}",
            f"**Total Steps:** {len(self._entries)}",
            "",
            "## Execution Summary",
            "",
            "| Step | Capability | Score | Safety | Result |",
            "|------|-----------|-------|--------|--------|",
        ]

        for entry in self._entries:
            safety_str = "PASS" if entry.safety_passed else "FAIL"
            report_lines.append(
                f"| {entry.step} | {entry.capability} | {entry.score:.2f} | "
                f"{safety_str} | {entry.result} |"
            )

        # Statistics
        total_steps = len(self._entries)
        successful = sum(1 for e in self._entries if e.result in ("success", "no_smoke"))
        replans = sum(1 for e in self._entries if e.replan_triggered)
        safety_violations = sum(1 for e in self._entries if not e.safety_passed)

        report_lines.extend([
            "",
            "## Statistics",
            "",
            f"- **Total Steps:** {total_steps}",
            f"- **Successful:** {successful}",
            f"- **Replanning Events:** {replans}",
            f"- **Safety Violations:** {safety_violations}",
            "",
            "## Detailed Logs",
            "",
        ])

        for entry in self._entries:
            replan_marker = " [REPLAN]" if entry.replan_triggered else ""
            report_lines.append(f"### Step {entry.step}: {entry.capability}{replan_marker}")
            report_lines.append("")
            report_lines.append(f"- Score: {entry.score:.4f}")
            report_lines.append(f"- Safety: {'PASS' if entry.safety_passed else 'FAIL'}")
            report_lines.append(f"- Result: {entry.result}")
            if entry.details:
                report_lines.append(f"- Details: `{json.dumps(entry.details, ensure_ascii=False)}`")
            report_lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        return filepath

    @property
    def entries(self) -> list[LogEntry]:
        return list(self._entries)

    @property
    def step_count(self) -> int:
        return len(self._entries)
