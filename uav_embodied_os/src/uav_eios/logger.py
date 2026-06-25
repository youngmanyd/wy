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
    waypoint_name: str = ""
    position: tuple[float, float, float] | None = None

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
        waypoint_name: str = "",
        position: tuple[float, float, float] | None = None,
    ) -> None:
        entry = LogEntry(
            step=step,
            capability=capability,
            score=score,
            safety_passed=safety_passed,
            result=result,
            details=details or {},
            replan_triggered=replan_triggered,
            waypoint_name=waypoint_name,
            position=position,
        )
        self._entries.append(entry)

    def save_jsonl(self, filename: str = "") -> Path:
        if not filename:
            filename = f"{self._mission_name}_log.jsonl"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in self._entries:
                data = asdict(entry)
                if data.get("position") is not None:
                    data["position"] = list(data["position"])
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return filepath

    def generate_report(self, filename: str = "") -> Path:
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
        ]

        waypoint_entries: dict[str, list[LogEntry]] = {}
        for entry in self._entries:
            if entry.waypoint_name:
                waypoint_entries.setdefault(entry.waypoint_name, []).append(entry)

        if waypoint_entries:
            report_lines.extend([
                "## Waypoint Summary",
                "",
                "| Waypoint | Position | Image Quality | Reobserved | Status |",
                "|----------|----------|--------------|------------|--------|",
            ])
            for wp_name, entries in waypoint_entries.items():
                pos_str = ""
                quality_str = "N/A"
                reobserved = "No"
                status = "unknown"

                for e in entries:
                    if e.position:
                        pos_str = f"({e.position[0]:.1f}, {e.position[1]:.1f}, {e.position[2]:.1f})"
                    if "image_quality" in e.details:
                        quality_str = f"{e.details['image_quality']:.3f}"
                    if e.replan_triggered:
                        reobserved = "Yes"
                    status = e.result

                report_lines.append(
                    f"| {wp_name} | {pos_str} | {quality_str} | {reobserved} | {status} |"
                )
            report_lines.append("")

        report_lines.extend([
            "## Execution Summary",
            "",
            "| Step | Capability | Score | Safety | Result | Waypoint |",
            "|------|-----------|-------|--------|--------|----------|",
        ])

        for entry in self._entries:
            safety_str = "PASS" if entry.safety_passed else "FAIL"
            wp_str = entry.waypoint_name or "-"
            report_lines.append(
                f"| {entry.step} | {entry.capability} | {entry.score:.2f} | "
                f"{safety_str} | {entry.result} | {wp_str} |"
            )

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
        ])

        image_entries = [e for e in self._entries if "image_path" in e.details]
        if image_entries:
            report_lines.extend([
                "## Captured Images",
                "",
            ])
            for entry in image_entries:
                img_path = entry.details.get("image_path", "")
                quality = entry.details.get("image_quality", 0.0)
                wp = entry.waypoint_name or "unknown"
                replan_tag = " [REOBSERVED]" if entry.replan_triggered else ""
                report_lines.extend([
                    f"### {wp}{replan_tag}",
                    f"- Image: `{img_path}`",
                    f"- Quality: {quality:.3f}",
                    f"- Blur: {entry.details.get('blur_score', 0.0):.3f}",
                    f"- Exposure: {entry.details.get('exposure_score', 0.0):.3f}",
                    f"- Contrast: {entry.details.get('contrast_score', 0.0):.3f}",
                    "",
                ])

        report_lines.extend([
            "## Detailed Logs",
            "",
        ])

        for entry in self._entries:
            replan_marker = " [REPLAN]" if entry.replan_triggered else ""
            wp_marker = f" @ {entry.waypoint_name}" if entry.waypoint_name else ""
            report_lines.append(f"### Step {entry.step}: {entry.capability}{replan_marker}{wp_marker}")
            report_lines.append("")
            report_lines.append(f"- Score: {entry.score:.4f}")
            report_lines.append(f"- Safety: {'PASS' if entry.safety_passed else 'FAIL'}")
            report_lines.append(f"- Result: {entry.result}")
            if entry.position:
                report_lines.append(
                    f"- Position: ({entry.position[0]:.2f}, {entry.position[1]:.2f}, {entry.position[2]:.2f})"
                )
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
