"""LLM Task Parser - natural language mission parsing via Qwen2.5 / OpenAI-compatible API.

Connects to a remote LLM server (vLLM, Ollama, etc.) to parse natural language
task descriptions into structured mission definitions.
Falls back to rule-based parsing if LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个无人机巡检任务解析助手。请将用户的自然语言任务描述解析为结构化的JSON格式。

输出格式必须是严格的JSON（不要添加markdown代码块标记），包含以下字段：
{
    "name": "任务名称（英文下划线格式）",
    "description": "任务描述",
    "objective": "fly_inspect_report",
    "area": "区域标识（如 baylands, campus_1 等）",
    "waypoints": [
        {"name": "point_1", "position": [x, y, z], "tasks": ["capture_image", "evaluate_quality"]}
    ],
    "priority_targets": ["smoke", "crowd", "rooftop_anomaly"],
    "constraints": {
        "max_altitude_m": 120.0,
        "reserve_battery_percent": 25.0,
        "max_wind_speed_mps": 12.0
    },
    "policies": {
        "low_image_quality": {"action": "reobserve_from_new_angle", "threshold": 0.6}
    }
}

注意事项：
- waypoints的position是[x, y, z]格式的NED坐标（米），z为正值表示高度
- priority_targets只能是: smoke, crowd, rooftop_anomaly
- 如果用户没有指定具体坐标，请根据合理的巡检路线自行规划航点
- 默认飞行高度5米，除非用户另有指定
- 所有字段必须填写，没有明确提到的使用默认值"""


@dataclass
class LLMConfig:
    """Configuration for the LLM API connection."""

    api_base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    model_name: str = "Qwen2.5-7B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_s: float = 30.0

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> LLMConfig:
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        llm_data = data.get("llm", data)
        return cls(
            api_base_url=llm_data.get("api_base_url", cls.api_base_url),
            api_key=llm_data.get("api_key", cls.api_key),
            model_name=llm_data.get("model_name", cls.model_name),
            temperature=llm_data.get("temperature", cls.temperature),
            max_tokens=llm_data.get("max_tokens", cls.max_tokens),
            timeout_s=llm_data.get("timeout_s", cls.timeout_s),
        )


class LLMTaskParser:
    """Parses natural language task descriptions using a remote LLM.

    Uses OpenAI-compatible API (works with vLLM, Ollama, text-generation-inference).
    Falls back to rule-based parsing if LLM call fails.
    """

    def __init__(self, config: LLMConfig | None = None, config_path: str | Path | None = None) -> None:
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = LLMConfig.from_yaml(config_path)
        else:
            self.config = LLMConfig()

        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.config.api_base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_s,
            )
            logger.info(
                "LLM client initialized: %s model=%s",
                self.config.api_base_url, self.config.model_name,
            )
        except ImportError:
            logger.warning("openai package not installed. LLM parsing disabled.")
            self._client = None
        except Exception as e:
            logger.warning("Failed to initialize LLM client: %s", e)
            self._client = None

    def parse_natural_language(self, text: str) -> dict[str, Any]:
        """Parse natural language task description into structured mission fields.

        Tries LLM first, falls back to rule-based parsing on failure.
        """
        if self._client is not None:
            try:
                return self._parse_with_llm(text)
            except Exception as e:
                logger.warning("LLM parsing failed, falling back to rules: %s", e)

        return self._parse_with_rules(text)

    def _parse_with_llm(self, text: str) -> dict[str, Any]:
        """Parse using the remote LLM."""
        response = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        content = response.choices[0].message.content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        result = json.loads(content)
        self._validate_result(result)

        logger.info("LLM parsed task: %s", result.get("name", "unknown"))
        return result

    def _validate_result(self, result: dict[str, Any]) -> None:
        """Validate LLM output has required fields."""
        required_fields = ["name", "objective", "area"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"LLM output missing required field: {field}")

        if "waypoints" in result:
            for wp in result["waypoints"]:
                if "position" not in wp:
                    raise ValueError(f"Waypoint missing position: {wp}")
                if len(wp["position"]) != 3:
                    raise ValueError(f"Waypoint position must be [x,y,z]: {wp}")

        valid_targets = {"smoke", "crowd", "rooftop_anomaly"}
        for target in result.get("priority_targets", []):
            if target not in valid_targets:
                logger.warning("Unknown priority target from LLM: %s", target)

    def _parse_with_rules(self, text: str) -> dict[str, Any]:
        """Fallback rule-based parsing (same as MVP MissionParser)."""
        result: dict[str, Any] = {
            "name": "parsed_mission",
            "description": text[:200],
            "objective": "fly_inspect_report",
            "area": "",
            "waypoints": [],
            "priority_targets": [],
            "constraints": {
                "max_altitude_m": 120.0,
                "reserve_battery_percent": 25.0,
                "max_wind_speed_mps": 12.0,
            },
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
            "烟雾": "smoke", "smoke": "smoke", "火": "smoke",
            "人员聚集": "crowd", "crowd": "crowd", "人群": "crowd",
            "屋顶": "rooftop_anomaly", "rooftop": "rooftop_anomaly",
            "设备异常": "rooftop_anomaly", "异常": "rooftop_anomaly",
        }
        for keyword, target in target_keywords.items():
            if keyword in text.lower() and target not in result["priority_targets"]:
                result["priority_targets"].append(target)

        policy_keywords = {
            "复拍": ("low_image_quality", {"action": "reobserve_from_new_angle", "threshold": 0.6}),
            "重拍": ("low_image_quality", {"action": "reobserve_from_new_angle", "threshold": 0.6}),
            "换角度": ("low_image_quality", {"action": "reobserve_from_new_angle", "threshold": 0.6}),
            "reobserve": ("low_image_quality", {"action": "reobserve_from_new_angle", "threshold": 0.6}),
        }
        for keyword, (condition, policy) in policy_keywords.items():
            if keyword in text.lower():
                result["policies"][condition] = policy

        return result

    @property
    def is_llm_available(self) -> bool:
        return self._client is not None
