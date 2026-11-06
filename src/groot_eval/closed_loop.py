"""Closed-loop evaluation: the policy drives the simulator, episode by episode."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from groot_eval.envs.base import Env
from groot_eval.policies.base import Policy
from groot_eval.stats import wilson_interval
from groot_eval.trace import TraceRecorder
from groot_eval.types import EpisodeResult, Scenario


def run_episode(env: Env, policy: Policy, scenario: Scenario, recorder: TraceRecorder | None = None) -> EpisodeResult:
    """Run one episode. With a ``recorder``, every step is also written to a trace."""
    if recorder is None:
        return _run_episode(env, policy, scenario, None)
    recorder.start_episode(scenario.task, scenario=scenario.to_dict())
    result = _run_episode(env, policy, scenario, recorder)
    recorder.end_episode(success=result.success, failure_reason=result.failure_reason)
    return result


def _run_episode(env: Env, policy: Policy, scenario: Scenario, recorder: TraceRecorder | None) -> EpisodeResult:
    obs = env.reset(scenario)
    policy.reset(scenario.task)
    seed_episode = getattr(policy, "set_episode_seed", None)
    if seed_episode is not None:  # randomness inside the policy follows the scenario, not the run order
        seed_episode(scenario.seed)
    total = 0.0
    latencies: list[float] = []
    info: dict[str, Any] = {}
    steps = 0
    for steps in range(1, env.max_steps + 1):
        t0 = time.perf_counter()
        try:
            action = policy.act(obs)
        except Exception as exc:  # a crashing policy is a failed episode, not a crashed sweep
            return EpisodeResult(scenario, False, steps, total, f"policy_error: {type(exc).__name__}: {exc}", latencies)
        latencies.append((time.perf_counter() - t0) * 1000)
        modelled = getattr(policy, "last_latency_ms", None)  # RealTimePolicy reports the latency it simulated
        if modelled is not None:
            latencies[-1] = float(modelled)
        if not np.all(np.isfinite(np.asarray(action, dtype=float))):
            return EpisodeResult(scenario, False, steps, total, "non_finite_action", latencies)
        seen = obs
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if recorder is not None:
            recorder.record_step(seen, action, latency_ms=latencies[-1], reward=float(reward), wall_time=t0,
                                 chunk_index=getattr(policy, "last_chunk_index", -1),
                                 chunk_step=getattr(policy, "last_chunk_step", -1))
        if terminated or truncated:
            break
    else:
        info = {**info, "success": False, "failure_reason": info.get("failure_reason", "timeout")}
    success = bool(info.get("success", False))
    return EpisodeResult(
        scenario=scenario,
        success=success,
        steps=steps,
        episode_return=total,
        failure_reason=None if success else str(info.get("failure_reason", "unknown")),
        policy_latency_ms=latencies,
        metrics={k: float(v) for k, v in (info.get("metrics") or {}).items()},
    )


@dataclass
class EvalRun:
    policy: str
    env: str
    results: list[EpisodeResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def successes(self) -> int:
        return sum(r.success for r in self.results)

    @property
    def trials(self) -> int:
        return len(self.results)

    def summary(self, confidence: float = 0.95) -> dict[str, Any]:
        lo, hi = wilson_interval(self.successes, self.trials, confidence)
        lat = [x for r in self.results for x in r.policy_latency_ms]
        reasons = Counter(r.failure_reason for r in self.results if not r.success)
        return {
            "policy": self.policy,
            "env": self.env,
            "episodes": self.trials,
            "successes": self.successes,
            "success_rate": self.successes / self.trials if self.trials else math.nan,
            "ci_low": lo,
            "ci_high": hi,
            "confidence": confidence,
            "failure_reasons": dict(reasons.most_common()),
            "latency_ms_p50": float(np.percentile(lat, 50)) if lat else None,
            "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
            "latency_ms_p99": float(np.percentile(lat, 99)) if lat else None,
            "mean_steps": float(np.mean([r.steps for r in self.results])) if self.results else None,
        }


def evaluate(
    env: Env,
    policy: Policy,
    scenarios: Iterable[Scenario],
    env_name: str = "",
    progress: Callable[[int, EpisodeResult], None] | None = None,
    recorder: TraceRecorder | None = None,
) -> EvalRun:
    run = EvalRun(policy=getattr(policy, "name", type(policy).__name__), env=env_name)
    for i, scenario in enumerate(scenarios):
        result = run_episode(env, policy, scenario, recorder)
        run.results.append(result)
        if progress:
            progress(i, result)
    return run
