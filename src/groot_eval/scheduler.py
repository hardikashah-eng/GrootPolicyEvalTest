"""Multi-rate stepping with timing taken from real traces.

A deployed robot runs several loops at different rates, and none of them wait
for each other: physics (or the motor drivers) at ~1 kHz, the whole-body or
joint controller at 50-500 Hz, and the policy every few control ticks, with
each inference taking a variable time. Sim loops usually run all of this in
lockstep. This module reproduces the robot's timing instead:

* :class:`TimingProfile` - what the robot's timing looks like, **extracted from
  a trace** (:meth:`TimingProfile.from_trace`): the distribution of inference
  latency (``inference_latency_ms``), the control-period jitter and the rate
  of missed control ticks (from the recorded ``timestamp`` column), and how
  many control ticks pass between inference requests (from ``chunk_index``).
  Saved as YAML; goes into a robot profile.
* :class:`MultiRateScheduler` - a batched clock for ``num_envs`` environments
  stepped at the physics rate. Each call of :meth:`~MultiRateScheduler.step`
  says, per environment, whether a control tick happens, whether it was
  missed (the drive keeps the last command), whether a policy inference
  starts, and whether a pending inference result arrives. Each environment
  draws its own latencies and jitter.
* :class:`ChunkBuffer` - holds each environment's action chunk and the one in
  flight, and hands out the command for each control tick: ``hold`` starts a
  new chunk at its first action, ``continue`` skips the actions that went
  stale while the inference ran (as an asynchronous executor does). Array-API
  code, so it runs on NumPy and on Isaac Lab torch tensors.
* :class:`ScheduledEnv` + :meth:`TimingProfile.wrap_policy` - the same timing
  for the single-environment CPU simulators: missed and late control ticks
  hold the previous command, and latency goes through the existing
  :class:`~groot_eval.latency.RealTimePolicy`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from array_api_compat import array_namespace, device

from groot_eval.degradation.components import EnvWrapper
from groot_eval.envs.base import Obs
from groot_eval.types import Scenario

DURING = ("hold", "continue")
QUANTILES = 101  # empirical distributions are stored as this many quantiles


@dataclass
class Distribution:
    """A non-negative scalar distribution: ``fixed``, ``normal`` (truncated at 0) or ``empirical`` (quantiles)."""

    kind: str = "fixed"
    value: float = 0.0  # fixed
    mean: float = 0.0  # normal
    std: float = 0.0
    quantiles: list[float] = field(default_factory=list)  # empirical, evenly spaced probabilities 0..1
    n: int = 0  # samples behind an empirical distribution

    @classmethod
    def from_samples(cls, samples: Any) -> "Distribution":
        v = np.asarray(samples, dtype=float)
        v = v[np.isfinite(v)]
        if not v.size:
            raise ValueError("no finite samples")
        q = np.percentile(v, np.linspace(0, 100, QUANTILES))
        return cls("empirical", quantiles=[float(x) for x in q], n=int(v.size))

    def sample(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        if self.kind == "fixed":
            return np.full(size, self.value, dtype=float)
        if self.kind == "normal":
            return np.maximum(rng.normal(self.mean, self.std, size), 0.0)
        if self.kind == "empirical":
            q = np.asarray(self.quantiles)
            return np.interp(rng.uniform(0.0, 1.0, size), np.linspace(0.0, 1.0, q.size), q)
        raise ValueError(f"unknown distribution kind {self.kind!r}")

    def percentile(self, p: float) -> float:
        if self.kind == "fixed":
            return self.value
        if self.kind == "normal":
            return max(0.0, self.mean + self.std * _z(p / 100.0))
        q = np.asarray(self.quantiles)
        return float(np.interp(p / 100.0, np.linspace(0.0, 1.0, q.size), q))

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "fixed":
            return {"kind": "fixed", "value": self.value}
        if self.kind == "normal":
            return {"kind": "normal", "mean": self.mean, "std": self.std}
        return {"kind": "empirical", "n": self.n, "quantiles": [round(x, 4) for x in self.quantiles]}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | float | None) -> "Distribution":
        if d is None:
            return cls()
        if isinstance(d, (int, float)):
            return cls("fixed", value=float(d))
        kind = d.get("kind", "fixed")
        if kind == "fixed":
            return cls("fixed", value=float(d.get("value", 0.0)))
        if kind == "normal":
            return cls("normal", mean=float(d["mean"]), std=float(d.get("std", 0.0)))
        if kind == "empirical":
            q = [float(x) for x in d["quantiles"]]
            if len(q) < 2 or any(b < a for a, b in zip(q, q[1:])):
                raise ValueError("empirical quantiles must be at least two non-decreasing values")
            return cls("empirical", quantiles=q, n=int(d.get("n", 0)))
        raise ValueError(f"unknown distribution kind {kind!r}")

    def describe(self) -> str:
        if self.kind == "fixed":
            return f"{self.value:g}"
        return f"p50={self.percentile(50):.1f}, p95={self.percentile(95):.1f}, p99={self.percentile(99):.1f}"


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, |error| < 1.2e-9)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -_z(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


@dataclass
class TimingProfile:
    """A robot's loop timing: control rate, inference cadence and latency, control jitter, missed ticks."""

    control_hz: float
    policy_period_ticks: int = 1  # control ticks between inference requests
    latency_ms: Distribution = field(default_factory=Distribution)
    control_jitter_ms: Distribution = field(default_factory=Distribution)  # control period minus nominal
    missed_tick_rate: float = 0.0  # fraction of control ticks the controller missed (command held)
    during_inference: str = "continue"
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz

    @classmethod
    def ideal(cls, control_hz: float, policy_period_ticks: int = 1) -> "TimingProfile":
        """No latency, no jitter, no missed ticks: the lockstep loop sims usually run."""
        return cls(control_hz, policy_period_ticks, source={"kind": "ideal"})

    @classmethod
    def from_trace(cls, trace: Any, during_inference: str = "continue",
                   control_hz: float | None = None) -> "TimingProfile":
        """Measure timing from a trace (a :class:`~groot_eval.trace.TraceSet` or its path).

        * latency: ``inference_latency_ms``; with ``chunk_index``, the largest value per chunk (the inference calls)
        * control period: diffs of ``timestamp`` (the robot clock); a gap of ~k periods = k-1 missed ticks
        * policy period: control ticks between chunk starts (median), else 1
        """
        from groot_eval.trace import TraceSet

        ts = trace if isinstance(trace, TraceSet) else TraceSet.open(trace)
        hz = control_hz or ts.fps
        if not hz:
            raise ValueError("trace has no fps; pass control_hz")
        nominal_ms = 1000.0 / hz
        lat, jitter, starts_gap = [], [], []
        missed = ticks = 0
        for f in ts.frames():
            if "inference_latency_ms" in f:
                v = f["inference_latency_ms"].to_numpy(dtype=float)
                ok = np.isfinite(v) & (v >= 0)
                ci = f["chunk_index"].to_numpy(dtype=float) if "chunk_index" in f else np.full(len(v), -1.0)
                n_chunks = len(np.unique(ci[ci >= 0]))
                if n_chunks:
                    # one inference per chunk; the other steps read a cached action (same rule as
                    # groot_eval.latency.latencies_from_trace)
                    lat.append(np.sort(v[ok])[::-1][:n_chunks])
                    idx = np.flatnonzero(np.r_[True, ci[1:] != ci[:-1]] & (ci >= 0))
                    if idx.size > 1:
                        starts_gap.append(np.diff(idx))
                else:
                    lat.append(v[ok])
            if "timestamp" in f and len(f) > 1:
                d = np.diff(f["timestamp"].to_numpy(dtype=float)) * 1000.0
                d = d[np.isfinite(d) & (d > 0)]
                k = np.rint(d / nominal_ms)
                ticks += int(k.sum())
                missed += int(np.maximum(k - 1, 0).sum())
                jitter.append(d[k == 1] - nominal_ms)
        latv = np.concatenate(lat) if lat else np.zeros(0)
        jit = np.concatenate(jitter) if jitter else np.zeros(0)
        gaps = np.concatenate(starts_gap) if starts_gap else np.zeros(0)
        prof = cls(float(hz), int(np.median(gaps)) if gaps.size else 1,
                   Distribution.from_samples(latv) if latv.size else Distribution(),
                   Distribution.from_samples(jit) if jit.size else Distribution(),
                   missed / ticks if ticks else 0.0, during_inference,
                   {"kind": "trace", "trace": str(ts.root), "episodes": len(ts.frames()),
                    "latency_samples": int(latv.size), "period_samples": int(jit.size), "domain": ts.domain})
        return prof

    # -- persistence -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"control_hz": self.control_hz, "policy_period_ticks": self.policy_period_ticks,
                "latency_ms": self.latency_ms.to_dict(), "control_jitter_ms": self.control_jitter_ms.to_dict(),
                "missed_tick_rate": round(self.missed_tick_rate, 6), "during_inference": self.during_inference,
                "source": self.source}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimingProfile":
        d = d.get("timing", d)
        during = str(d.get("during_inference", "continue"))
        if during not in DURING:
            raise ValueError(f"during_inference must be one of {DURING}")
        rate = float(d.get("missed_tick_rate", 0.0))
        if not 0.0 <= rate < 1.0:
            raise ValueError("missed_tick_rate must be in [0, 1)")
        return cls(float(d["control_hz"]), int(d.get("policy_period_ticks", 1)), Distribution.from_dict(d.get("latency_ms")),
                   Distribution.from_dict(d.get("control_jitter_ms")), rate, during, dict(d.get("source") or {}))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"timing": self.to_dict()}, sort_keys=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TimingProfile":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def describe(self) -> str:
        return (f"{self.control_hz:g} Hz control, inference every {self.policy_period_ticks} tick(s), latency ms "
                f"{self.latency_ms.describe()}, jitter ms {self.control_jitter_ms.describe()}, "
                f"missed ticks {100 * self.missed_tick_rate:.2f}%, {self.during_inference}")

    # -- CPU path ----------------------------------------------------------------------
    def latency_model(self, seed: int = 0) -> Any:
        """A :class:`~groot_eval.latency.LatencyModel` drawing from this profile's latency distribution."""
        from groot_eval.latency import LatencyModel

        if self.latency_ms.kind == "fixed":
            return LatencyModel("fixed", (self.latency_ms.value,), seed)
        if self.latency_ms.kind == "normal":
            return LatencyModel("normal", (self.latency_ms.mean, self.latency_ms.std), seed)
        draws = self.latency_ms.sample(np.random.default_rng(seed), 4096)
        return LatencyModel("trace", (draws,), seed)

    def wrap_policy(self, policy: Any, dt: float, seed: int = 0) -> Any:
        """``policy`` as the robot runs it: inference latency from this profile, ``hold`` or ``continue``."""
        from groot_eval.latency import RealTimePolicy

        return RealTimePolicy(policy, self.latency_model(seed), dt, self.during_inference)


@dataclass
class Tick:
    """What happens in each environment on one physics step (all ``(num_envs,)`` bool unless noted)."""

    control: np.ndarray  # a control tick happens
    missed: np.ndarray  # ... but the controller missed it (keep the last command)
    request: np.ndarray  # a policy inference starts from this step's observation
    ready: np.ndarray  # a pending inference result arrives
    latency_ms: np.ndarray  # (num_envs,) float, latency drawn for this step's requests (nan elsewhere)
    time_s: float


class MultiRateScheduler:
    """Batched clock: physics every call, control ticks and policy inferences on each env's own timing."""

    def __init__(self, timing: TimingProfile, num_envs: int, physics_dt: float, seed: int = 0) -> None:
        if physics_dt <= 0 or physics_dt > timing.control_dt + 1e-12:
            raise ValueError("physics_dt must be positive and no longer than the control period")
        self.timing = timing
        self.num_envs = num_envs
        self.physics_dt = physics_dt
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, env_ids: Any = None) -> None:
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids).reshape(-1)
        if env_ids is None:
            self.k = 0
            self.next_control = np.zeros(self.num_envs)
            self.ready_at = np.full(self.num_envs, np.inf)
            self.ticks_since_request = np.full(self.num_envs, 10**9)
            self.control_ticks = np.zeros(self.num_envs, np.int64)
        else:
            now = self.k * self.physics_dt
            self.next_control[ids] = now
            self.ready_at[ids] = np.inf
            self.ticks_since_request[ids] = 10**9
            self.control_ticks[ids] = 0

    def step(self) -> Tick:
        t = self.k * self.physics_dt
        eps = 1e-9 * max(1.0, t)
        control = t + eps >= self.next_control
        n = int(control.sum())
        if n:
            period = self.timing.control_dt + self.timing.control_jitter_ms.sample(self._rng, n) / 1000.0
            self.next_control[control] += np.maximum(period, self.physics_dt)
        missed = np.zeros(self.num_envs, bool)
        if n and self.timing.missed_tick_rate > 0:
            missed[control] = self._rng.uniform(size=n) < self.timing.missed_tick_rate
        live = control & ~missed
        self.control_ticks += live
        self.ticks_since_request += live
        ready = t + eps >= self.ready_at
        self.ready_at[ready] = np.inf
        request = live & ~np.isfinite(self.ready_at) & (self.ticks_since_request >= self.timing.policy_period_ticks)
        latency = np.full(self.num_envs, np.nan)
        m = int(request.sum())
        if m:
            latency[request] = self.timing.latency_ms.sample(self._rng, m)
            self.ready_at[request] = t + latency[request] / 1000.0
            self.ticks_since_request[request] = 0
        self.k += 1
        return Tick(control, missed, request, ready, latency, t)


class ChunkBuffer:
    """Per-env action chunks: the one executing and the one in flight (array-API: NumPy or torch)."""

    def __init__(self, during: str = "continue") -> None:
        if during not in DURING:
            raise ValueError(f"during must be one of {DURING}")
        self.during = during
        self.chunk: Any = None  # (E, H, A)
        self.ptr: np.ndarray | None = None  # (E,) next index into chunk
        self.pending: Any = None
        self.pending_ticks: np.ndarray | None = None  # live control ticks since each pending request
        self.has_chunk: np.ndarray | None = None
        self.last: Any = None

    def request(self, mask: np.ndarray, chunk: Any) -> None:
        """Store ``chunk`` (E, H, A; rows outside ``mask`` ignored) as in flight for the masked envs."""
        xp = array_namespace(chunk)
        e = chunk.shape[0]
        if self.pending is None:
            self.pending = xp.zeros_like(chunk)
            self.pending_ticks = np.zeros(e, np.int64)
            self.chunk = xp.zeros_like(chunk)
            self.ptr = np.zeros(e, np.int64)
            self.has_chunk = np.zeros(e, bool)
        sel = _sel(mask, chunk, 3)
        self.pending = sel * chunk + (1.0 - sel) * self.pending
        assert self.pending_ticks is not None
        self.pending_ticks[mask] = 0

    def ready(self, mask: np.ndarray) -> None:
        """Adopt the in-flight chunk for the masked envs."""
        if self.pending is None or not mask.any():
            return
        assert self.ptr is not None and self.pending_ticks is not None and self.has_chunk is not None
        sel = _sel(mask, self.pending, 3)
        self.chunk = sel * self.pending + (1.0 - sel) * self.chunk
        h = self.chunk.shape[1]
        start = np.minimum(self.pending_ticks, h - 1) if self.during == "continue" else np.zeros_like(self.ptr)
        self.ptr = np.where(mask, start, self.ptr)
        self.has_chunk |= mask

    def command(self, live: np.ndarray) -> Any:
        """The command for this control tick ``(E, A)``. Advances envs in ``live``; others keep their last command."""
        if self.chunk is None:
            raise RuntimeError("no chunk has been requested yet")
        assert self.ptr is not None and self.pending_ticks is not None and self.has_chunk is not None
        xp = array_namespace(self.chunk)
        h = self.chunk.shape[1]
        idx = np.minimum(self.ptr, h - 1)
        onehot = _as(np.eye(h)[idx], self.chunk)  # (E, H)
        cur = xp.sum(self.chunk * onehot[..., None], axis=1)
        if self.last is None:
            self.last = xp.zeros_like(cur)
        use = live & self.has_chunk
        cmd = _sel(use, cur, 2) * cur + (1.0 - _sel(use, cur, 2)) * self.last
        self.ptr = np.where(use, np.minimum(self.ptr + 1, h), self.ptr)
        self.pending_ticks += live
        self.last = cmd
        return cmd


def _as(a: np.ndarray, ref: Any) -> Any:
    xp = array_namespace(ref)
    return xp.asarray(a, dtype=ref.dtype, device=device(ref))


def _sel(mask: np.ndarray, ref: Any, ndim: int) -> Any:
    shape = (mask.shape[0],) + (1,) * (ndim - 1)
    return _as(mask.astype(np.float64).reshape(shape), ref)


class ScheduledEnv(EnvWrapper):
    """Single-env CPU path: missed and late control ticks keep the previous command.

    The wrapped env steps once per control tick. With the profile's missed-tick
    rate the drive misses the new command (the last one is applied again);
    a control period longer than 1.5 nominal periods (from the jitter
    distribution) is treated the same way. Combine with
    :meth:`TimingProfile.wrap_policy` for inference latency.
    """

    def __init__(self, env: Any, timing: TimingProfile, seed: int = 0) -> None:
        super().__init__(env)
        self.timing = timing
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._last: np.ndarray | None = None
        self.held = 0

    def reset(self, scenario: Scenario) -> Obs:
        self._rng = np.random.default_rng((scenario.seed * 7907 + self._seed) % 2**32)
        self._last = None
        self.held = 0
        return self.env.reset(scenario)

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=float)
        late = self.timing.control_jitter_ms.sample(self._rng, 1)[0] > 0.5 * 1000.0 * self.timing.control_dt
        if self._last is not None and (late or self._rng.uniform() < self.timing.missed_tick_rate):
            a = self._last
            self.held += 1
        self._last = a
        obs, reward, term, trunc, info = self.env.step(a)
        if term or trunc:
            info.setdefault("metrics", {})["timing.held_ticks"] = float(self.held)
        return obs, reward, term, trunc, info
