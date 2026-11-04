"""Core records shared by every evaluation mode."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    """One reproducible episode setup: a seed plus the randomized parameters."""

    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    task: str = ""

    @property
    def scenario_id(self) -> str:
        blob = json.dumps({"seed": self.seed, "params": self.params, "task": self.task}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {"scenario_id": self.scenario_id, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        return cls(seed=int(data["seed"]), params=dict(data.get("params", {})), task=data.get("task", ""))


@dataclass
class EpisodeResult:
    scenario: Scenario
    success: bool
    steps: int
    episode_return: float
    failure_reason: str | None = None
    policy_latency_ms: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        lat = self.policy_latency_ms
        return {
            **self.scenario.to_dict(),
            "success": self.success,
            "steps": self.steps,
            "episode_return": self.episode_return,
            "failure_reason": self.failure_reason,
            "latency_ms_p50": _pct(lat, 50),
            "latency_ms_p95": _pct(lat, 95),
            "latency_ms_max": max(lat) if lat else None,
            **{f"metric.{k}": v for k, v in self.metrics.items()},
        }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
