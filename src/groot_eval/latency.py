"""Make simulated rollouts feel inference latency the way a robot does.

The closed-loop runner is synchronous: it waits for the policy, then steps the
sim, so the action computed from the observation at tick ``k`` acts at tick
``k``. A robot's control loop does not wait. Inference that takes ``L`` ms
means the command reaches the actuators at tick ``k + ceil(L / dt)``, and
something has to be executed in between.

:class:`RealTimePolicy` wraps any policy (or chunked policy) and models that:

* Latency comes from a :class:`LatencyModel`: ``measured`` (time the real
  call), ``fixed:<ms>``, ``normal:<mean_ms>,<std_ms>``, or ``trace:<dir>``
  (sample the latencies recorded in a trace, e.g. the real robot's).
* While an inference is in flight, ``during="hold"`` keeps sending the last
  command (a blocking deploy script), and ``during="continue"`` keeps
  executing the rest of the current action chunk (an asynchronous executor).
  With ``continue``, the new chunk starts at the step that matches the time
  that has passed, skipping the actions that are already stale.

With zero latency it behaves exactly like the wrapped policy.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from groot_eval.policies.base import ChunkedPolicy, Obs, Policy

DURING = ("hold", "continue")


@dataclass
class LatencyModel:
    kind: str  # measured | fixed | normal | trace
    args: tuple = ()
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def parse(cls, spec: str, seed: int = 0) -> "LatencyModel":
        kind, _, rest = spec.partition(":")
        if kind == "measured" and not rest:
            return cls("measured", (), seed)
        if kind == "fixed":
            return cls("fixed", (_ms(float(rest)),), seed)
        if kind == "normal":
            mean, _, std = rest.partition(",")
            return cls("normal", (_ms(float(mean)), _ms(float(std or 0))), seed)
        if kind == "trace":
            return cls("trace", (latencies_from_trace(rest),), seed)
        raise ValueError(f"latency mode must be measured, fixed:<ms>, normal:<mean>,<std> or trace:<dir>; got {spec!r}")

    def sample(self, measured_ms: float) -> float:
        if self.kind == "measured":
            return measured_ms
        if self.kind == "fixed":
            return self.args[0]
        if self.kind == "normal":
            return max(0.0, float(self._rng.normal(*self.args)))
        return float(self._rng.choice(self.args[0]))

    def describe(self) -> str:
        if self.kind == "trace":
            v = self.args[0]
            return f"trace(n={len(v)}, p50={np.percentile(v, 50):.1f}ms, p95={np.percentile(v, 95):.1f}ms)"
        return self.kind + (":" + ",".join(f"{a:g}" for a in self.args) if self.args else "")


def _ms(v: float) -> float:
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"latency must be a finite, non-negative number of ms; got {v}")
    return v


def latencies_from_trace(root: str) -> np.ndarray:
    """Per-inference latencies recorded in a trace.

    With chunk information, each episode ran one inference per chunk, so its
    ``n_chunks`` largest latencies are the inference calls (the other steps only
    read a cached action). Without it, every step counts as an inference.
    """
    from groot_eval.trace import TraceSet

    out = []
    for f in TraceSet.open(root).frames():
        if "inference_latency_ms" not in f:
            continue
        lat = f["inference_latency_ms"].to_numpy(dtype=float)
        ok = np.isfinite(lat) & (lat >= 0)
        if "chunk_index" in f:
            ci = f["chunk_index"].to_numpy(dtype=float)
            n = len(np.unique(ci[ci >= 0]))
            if n:
                out.append(np.sort(lat[ok])[::-1][:n])
                continue
        out.append(lat[ok])
    lat = np.concatenate(out) if out else np.zeros(0)
    if not lat.size:
        raise ValueError(f"{root} has no inference_latency_ms values to sample from")
    return lat


class RealTimePolicy:
    """Run a policy as a robot's non-blocking control loop would (see the module docstring)."""

    def __init__(self, policy: Policy, latency: LatencyModel, dt: float, during: str = "hold") -> None:
        if during not in DURING:
            raise ValueError(f"during must be one of {DURING}")
        if not dt or dt <= 0:
            raise ValueError("latency modelling needs the env's control period (env.dt)")
        self.inner, self.latency, self.dt_ms, self.during = policy, latency, dt * 1000.0, during
        if isinstance(policy, ChunkedPolicy):
            self._predict = policy.predictor.predict_chunk
            self.execute_steps = policy.execute_steps
            self._reset = policy.predictor.reset
        else:
            self._predict = lambda obs: np.asarray(policy.act(obs), dtype=float)[None]
            self.execute_steps = 1
            self._reset = policy.reset
        self.name = f"{getattr(policy, 'name', type(policy).__name__)}+latency({latency.describe()},{during})"
        self.reset("")

    def reset(self, task: str) -> None:
        self._reset(task)
        self._k = 0
        self._chunk: np.ndarray | None = None
        self._i = self._executed = 0
        self._pending: tuple[np.ndarray, int, int] | None = None  # (chunk, ready tick, start tick)
        self._last: np.ndarray | None = None
        self.inference_calls = 0
        self.last_latency_ms = 0.0
        self.delays: list[int] = []

    def act(self, obs: Obs) -> np.ndarray:
        k = self._k
        self.last_latency_ms = 0.0
        if self._pending is not None and self._pending[1] <= k:
            self._adopt(*self._pending, k)
        if self._pending is None and (self._chunk is None or self._executed >= self.execute_steps):
            t0 = time.perf_counter()
            chunk = np.asarray(self._predict(obs), dtype=float)
            if chunk.ndim != 2 or len(chunk) == 0:
                raise ValueError(f"predict_chunk must return (horizon, dim); got shape {chunk.shape}")
            lat = self.latency.sample((time.perf_counter() - t0) * 1000.0)
            self.last_latency_ms = lat
            delay = math.ceil(lat / self.dt_ms - 1e-9)
            self.delays.append(delay)
            self.inference_calls += 1
            if delay == 0:
                self._adopt(chunk, k, k, k)
            else:
                self._pending = (chunk, k + delay, k)
        action = self._next_action()
        self._k += 1
        return action

    def _adopt(self, chunk: np.ndarray, ready: int, start: int, now: int) -> None:
        self._chunk, self._executed, self._pending = chunk, 0, None
        self._i = now - start if self.during == "continue" else 0
        if self._i >= len(chunk):  # the whole chunk went stale while it was computed
            self._i = len(chunk) - 1
        self._chunk_id = self.inference_calls - 1

    def _next_action(self) -> np.ndarray:
        c = self._chunk
        running = c is not None and self._i < len(c) and (self.during == "continue" or self._executed < self.execute_steps)
        if running:
            action = c[self._i]
            self._i += 1
            self._executed += 1
        elif self._last is not None:
            action = self._last  # nothing new has arrived: keep sending the last command
        else:
            ref = c if c is not None else self._pending[0]  # type: ignore[index]
            action = np.zeros(ref.shape[1])  # nothing has ever arrived: stand still
        self._last = action
        self._ran = running
        return action

    # Read by the trace recorder and the closed-loop runner.
    @property
    def last_chunk_index(self) -> int:
        return getattr(self, "_chunk_id", -1) if getattr(self, "_ran", False) else -1

    @property
    def last_chunk_step(self) -> int:
        return self._i - 1 if getattr(self, "_ran", False) else -1


def with_latency(policy: Policy, spec: str | None, dt: float | None, during: str = "hold", seed: int = 0) -> Policy:
    """Wrap ``policy`` in a :class:`RealTimePolicy` when ``spec`` is given (CLI helper)."""
    if not spec:
        return policy
    return RealTimePolicy(policy, LatencyModel.parse(spec, seed), dt or 0.0, during)
