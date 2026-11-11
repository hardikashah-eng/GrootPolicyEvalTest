"""Test rig: many publishers on a real-time (or simulated) clock feeding a controller, with real or fake data.

Each input the controller reads is a **topic** (an observation key such as ``state.joint_pos`` or
``video.front``) with its own publisher:

* **rate and jitter**: messages go out at ``rate_hz`` with jitter, a random drop rate and a transport delay,
  as measured on the robot (a ``TimingProfile`` distribution format);
* **source**:
  * ``twin`` / ``plant`` - the plant's current value (the twin, or ``mock`` hardware that mirrors commands
    like ros2_control's mock components);
  * ``replay`` - a recorded trace's value at the same time since the episode start, restamped to now;
  * ``signal`` - constant, sine, step or noise;
  * ``live`` - a real device or any callable (a camera, an IMU driver, a ROS subscriber: see ``rig_ros2``);
* **faults** on the stream: ``drop``, ``freeze``, ``delay``, ``noise``, ``bias``, in time windows.

At every control tick the rig assembles the observation from the latest message of each topic, records how
old each one is (``rig.age_ms.<topic>``) and whether it is stale (older than ``max_age_ms``), calls the policy,
and sends the command to the plant. The trace records the observation exactly as the controller saw it.

Everything random comes from one :class:`~groot_eval.seeding.RunSeed`, so a run repeats exactly. Each
episode's trace carries its run seed.

:func:`compare_to_ideal` runs the same seeded scenarios twice: once with the rig's streams and once with
ideal streams (no jitter, delay, drops or faults). It reports success both ways, a paired McNemar test, and
how far the commands drifted from the ideal run.

``clock: real`` paces publishers and the control loop on the system's monotonic clock and counts overruns.
``clock: sim`` runs the same schedule as fast as possible, for CI.
"""

from __future__ import annotations

import heapq
import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.seeding import RunSeed, episode_meta
from groot_eval.types import Scenario

Obs = dict[str, Any]
SOURCE_KINDS = ("twin", "plant", "replay", "signal", "live")
FAULT_KINDS = ("drop", "freeze", "delay", "noise", "bias")


# =====================================================================================
# Clocks
# =====================================================================================
class SimClock:
    """Jumps from event to event: the same schedule as real time, as fast as the CPU allows."""

    real = False

    def start(self) -> None:
        self.t = 0.0

    def wait_until(self, t: float) -> float:
        self.t = t
        return 0.0  # lateness

    def now(self) -> float:
        return self.t


class RealClock:
    """Monotonic wall clock: waits for each event and reports how late it fired."""

    real = True

    def start(self) -> None:
        self.t0 = time.monotonic()

    def wait_until(self, t: float) -> float:
        delay = self.t0 + t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return max(0.0, time.monotonic() - self.t0 - t)

    def now(self) -> float:
        return time.monotonic() - self.t0


# =====================================================================================
# Sources
# =====================================================================================
class Source:
    def reset(self, scenario: Scenario) -> None:
        pass

    def value(self, t: float, plant_obs: Obs) -> Any:
        raise NotImplementedError


class PlantSource(Source):
    def __init__(self, key: str) -> None:
        self.key = key

    def value(self, t: float, plant_obs: Obs) -> Any:
        if self.key not in plant_obs:
            raise KeyError(f"the plant does not publish {self.key}; it has {sorted(plant_obs)}")
        return np.array(plant_obs[self.key], copy=True)


class ReplaySource(Source):
    """The recorded value of ``key`` at the same time since the episode start (episode matched by seed)."""

    def __init__(self, key: str, trace: str, episode: int | None = None) -> None:
        from groot_eval.bridge import scenarios_from_meta
        from groot_eval.trace import TraceSet

        ts = TraceSet.open(trace)
        name = key.removeprefix("state.")
        sl = ts.state_slices()
        if name not in sl:
            raise KeyError(f"{trace} has no {key}; it has {sorted('state.' + k for k in sl)}")
        self.key, self.fixed = key, episode
        dt = 1.0 / float(ts.fps or 30.0)
        self.by_seed: dict[int, np.ndarray] = {}
        self.by_index: list[np.ndarray] = []
        for sc, states in zip(scenarios_from_meta(ts.episode_meta()), ts.per_episode("observation.state")):
            arr = np.asarray(states, dtype=float)[:, sl[name]]
            self.by_seed[sc.seed] = arr
            self.by_index.append(arr)
        self.dt = dt
        self.cur = self.by_index[0]

    def reset(self, scenario: Scenario) -> None:
        if self.fixed is not None:
            self.cur = self.by_index[self.fixed]
        elif scenario.seed in self.by_seed:
            self.cur = self.by_seed[scenario.seed]
        else:
            self.cur = self.by_index[scenario.seed % len(self.by_index)]

    def value(self, t: float, plant_obs: Obs) -> Any:
        i = min(int(t / self.dt), len(self.cur) - 1)
        return self.cur[i].astype(np.float32)


class SignalSource(Source):
    def __init__(self, shape: str = "constant", value: Any = 0.0, amplitude: float = 1.0, freq_hz: float = 1.0,
                 at_s: float = 1.0, std: float = 0.1, dim: int = 1, rng: np.random.Generator | None = None) -> None:
        if shape not in ("constant", "sine", "step", "noise"):
            raise ValueError("signal shape must be constant, sine, step or noise")
        self.shape, self.base = shape, np.broadcast_to(np.asarray(value, dtype=float), (dim,)).copy()
        self.amp, self.freq, self.at, self.std = float(amplitude), float(freq_hz), float(at_s), float(std)
        self.rng = rng or np.random.default_rng(0)

    def value(self, t: float, plant_obs: Obs) -> Any:
        v = self.base.copy()
        if self.shape == "sine":
            v += self.amp * math.sin(2 * math.pi * self.freq * t)
        elif self.shape == "step" and t >= self.at:
            v += self.amp
        elif self.shape == "noise":
            v += self.rng.normal(0.0, self.std, size=v.shape)
        return v.astype(np.float32)


class LiveSource(Source):
    """A real device or any callable returning the current value.

    ``fn: package.module:function`` (called with no arguments), or ``device: <n>`` for a camera through OpenCV.
    """

    def __init__(self, fn: str | Callable[[], Any] | None = None, device: int | str | None = None) -> None:
        if callable(fn):
            self.fn = fn
        elif isinstance(fn, str):
            module, name = fn.split(":", 1)
            self.fn = getattr(importlib.import_module(module), name)
        elif device is not None:
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - depends on the machine
                raise RuntimeError("a camera source needs opencv-python (pip install opencv-python)") from exc
            cap = cv2.VideoCapture(device)
            if not cap.isOpened():  # pragma: no cover
                raise RuntimeError(f"cannot open camera {device}")

            def grab() -> Any:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"camera {device} returned no frame")
                return frame[:, :, ::-1]  # BGR -> RGB

            self.fn = grab
        else:
            raise ValueError("a live source needs fn or device")

    def value(self, t: float, plant_obs: Obs) -> Any:
        return np.asarray(self.fn())


# =====================================================================================
# Topics, faults, plants
# =====================================================================================
@dataclass
class StreamFault:
    topic: str
    kind: str
    start_s: float
    duration_s: float = math.inf
    severity: float = 0.0  # delay: ms; noise: std; bias: offset; drop / freeze: unused

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown stream fault {self.kind!r}; known: {FAULT_KINDS}")

    def active(self, t: float) -> bool:
        return self.start_s <= t < self.start_s + self.duration_s


@dataclass
class TopicSpec:
    key: str
    rate_hz: float
    source: dict[str, Any] = field(default_factory=lambda: {"kind": "plant"})
    jitter_ms: Any = 0.0  # Distribution spec (number, or {kind: normal, mean, std} / empirical quantiles)
    delay_ms: Any = 0.0  # transport delay, Distribution spec
    drop_rate: float = 0.0
    max_age_ms: float | None = None  # older than this at a control tick = stale (default: 3 periods)

    def __post_init__(self) -> None:
        if not (self.key.startswith("state.") or self.key.startswith("video.")):
            raise ValueError(f"topic {self.key!r} must be a state.* or video.* key")
        if self.source.get("kind") not in SOURCE_KINDS:
            raise ValueError(f"topic {self.key}: source kind must be one of {SOURCE_KINDS}")
        if self.rate_hz <= 0:
            raise ValueError(f"topic {self.key}: rate_hz must be positive")

    @property
    def stale_after_s(self) -> float:
        return (self.max_age_ms / 1000.0) if self.max_age_ms is not None else 3.0 / self.rate_hz


class MockPlant:
    """Ideal hardware, like ros2_control's mock components: joint positions follow position commands."""

    def __init__(self, dim: int, dt: float, max_steps: int = 10 ** 9) -> None:
        self.dim, self.dt, self.max_steps = dim, dt, max_steps
        self.q: np.ndarray = np.zeros(dim)
        self.qd: np.ndarray = np.zeros(dim)

    def reset(self, scenario: Scenario) -> Obs:
        self.q, self.qd = np.zeros(self.dim), np.zeros(self.dim)
        return self._obs()

    def _obs(self) -> Obs:
        return {"state.joint_pos": self.q.astype(np.float32), "state.joint_vel": self.qd.astype(np.float32)}

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        cmd = np.asarray(action, dtype=float).reshape(-1)[: self.dim]
        self.qd = (cmd - self.q) / self.dt
        self.q = cmd.copy()
        return self._obs(), 0.0, False, False, {}

    def close(self) -> None:
        pass


@dataclass
class RigConfig:
    topics: list[TopicSpec]
    control_hz: float = 50.0
    duration_s: float = 10.0
    clock: str = "sim"
    plant: dict[str, Any] = field(default_factory=lambda: {"kind": "twin", "env": "arm3"})
    policy: str = "arm-ik"
    faults: list[StreamFault] = field(default_factory=list)
    passthrough: bool = True  # plant keys no topic covers reach the controller untouched (ideal)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RigConfig:
        d = dict(d)
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown rig config keys {sorted(unknown)}; known: {sorted(known)}")
        topics = [TopicSpec(**t) for t in d.pop("topics", [])]
        if not topics:
            raise ValueError("a rig needs at least one topic")
        if len({t.key for t in topics}) != len(topics):
            raise ValueError("each topic key may appear once")
        faults = [StreamFault(**f) for f in d.pop("faults", [])]
        unknown_topics = {f.topic for f in faults} - {t.key for t in topics}
        if unknown_topics:
            raise ValueError(f"faults on topics the rig does not publish: {sorted(unknown_topics)}")
        cfg = cls(topics=topics, faults=faults, **d)
        if cfg.clock not in ("sim", "real"):
            raise ValueError("clock must be sim or real")
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> RigConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def ideal(self) -> RigConfig:
        """The same rig with perfect streams: no jitter, delay, drops or faults."""
        topics = [TopicSpec(t.key, t.rate_hz, t.source, 0.0, 0.0, 0.0, t.max_age_ms) for t in self.topics]
        return RigConfig(topics, self.control_hz, self.duration_s, "sim", self.plant, self.policy, [], self.passthrough)


# =====================================================================================
# Session
# =====================================================================================
@dataclass
class _Latest:
    value: Any
    sampled: float
    arrived: float


@dataclass
class EpisodeStats:
    published: dict[str, int] = field(default_factory=dict)
    delivered: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    stale_ticks: dict[str, int] = field(default_factory=dict)
    missing_ticks: dict[str, int] = field(default_factory=dict)
    max_age_ms: dict[str, float] = field(default_factory=dict)
    ticks: int = 0
    overruns: int = 0
    max_lateness_ms: float = 0.0
    faults_fired: list[str] = field(default_factory=list)
    commands: list[np.ndarray] = field(default_factory=list)
    success: bool | None = None
    failure_reason: str | None = None


def _make_plant(spec: dict[str, Any], dt: float) -> Any:
    kind = spec.get("kind", "twin")
    if kind == "mock":
        return MockPlant(int(spec.get("dim", 3)), dt)
    if kind == "twin":
        from groot_eval.envs import make_env

        return make_env(str(spec.get("env", "arm3")), **dict(spec.get("env_args") or {}))
    raise ValueError("plant kind must be twin or mock")


class RigSession:
    def __init__(self, cfg: RigConfig, run: RunSeed, policy: Any = None, plant: Any = None,
                 recorder: Any = None, on_arrive: Callable[[str, Any, float], None] | None = None) -> None:
        from groot_eval.scheduler import Distribution

        self.cfg, self.run, self.recorder, self.on_arrive = cfg, run, recorder, on_arrive
        self.dt = 1.0 / cfg.control_hz
        if policy is None:
            from groot_eval.policies import make_policy

            policy = make_policy(cfg.policy)
        self.policy = policy
        self.plant = plant if plant is not None else _make_plant(cfg.plant, self.dt)
        self.jitter = {t.key: Distribution.from_dict(t.jitter_ms) for t in cfg.topics}
        self.delay = {t.key: Distribution.from_dict(t.delay_ms) for t in cfg.topics}
        self.clock = RealClock() if cfg.clock == "real" else SimClock()

    def _source(self, t: TopicSpec, episode: int) -> Source:
        s = dict(t.source)
        kind = s.pop("kind")
        if kind in ("twin", "plant"):
            return PlantSource(s.pop("key", t.key))
        if kind == "replay":
            return ReplaySource(s.pop("key", t.key), **s)
        if kind == "signal":
            return SignalSource(rng=self.run.rng("rig", episode, t.key, "signal"), **s)
        if "ros_topic" in s:
            from groot_eval.rig_ros2 import Ros2LiveSource

            return Ros2LiveSource(s.pop("ros_topic"), **s)
        return LiveSource(**s)

    def run_episode(self, scenario: Scenario, episode: int = 0) -> EpisodeStats:
        cfg, run = self.cfg, self.run
        plant_obs = self.plant.reset(scenario)
        if hasattr(self.plant, "max_steps"):
            self.plant.max_steps = max(int(self.plant.max_steps), int(cfg.duration_s * cfg.control_hz) + 2)
        self.policy.reset(scenario.task)
        sources = {tp.key: self._source(tp, episode) for tp in cfg.topics}
        for s in sources.values():
            s.reset(scenario)
        rngs = {tp.key: run.rng("rig", episode, tp.key) for tp in cfg.topics}
        keys = [tp.key for tp in cfg.topics]
        stats = EpisodeStats(published=dict.fromkeys(keys, 0), delivered=dict.fromkeys(keys, 0),
                             dropped=dict.fromkeys(keys, 0), stale_ticks=dict.fromkeys(keys, 0),
                             missing_ticks=dict.fromkeys(keys, 0), max_age_ms=dict.fromkeys(keys, 0.0))
        latest: dict[str, _Latest] = {}
        frozen: dict[str, Any] = {}
        fired: set[int] = set()
        # event heap: (time, order, kind, payload)
        heap: list[tuple[float, int, str, Any]] = []
        order = 0

        def push(t: float, kind: str, payload: Any) -> None:
            nonlocal order
            heapq.heappush(heap, (t, order, kind, payload))
            order += 1

        specs = {tp.key: tp for tp in cfg.topics}
        for tp in cfg.topics:
            push(0.0, "publish", (tp.key, 0))
        n_ticks = int(round(cfg.duration_s * cfg.control_hz))
        for k in range(n_ticks):
            push((k + 1) * self.dt, "control", k)
        if self.recorder is not None:
            self.recorder.start_episode(scenario.task, scenario=scenario.to_dict(), rig=True,
                                        **episode_meta(run, episode))
        self.clock.start()
        wall0 = time.time()
        done = False
        while heap and not done:
            t, _, kind, payload = heapq.heappop(heap)
            lateness = self.clock.wait_until(t)
            if self.clock.real and kind == "control":
                stats.max_lateness_ms = max(stats.max_lateness_ms, 1000 * lateness)
                if lateness > 0.5 * self.dt:
                    stats.overruns += 1
            if kind == "publish":
                key, n = payload
                spec = specs[key]
                nominal_next = (n + 1) / spec.rate_hz
                jit = float(self.jitter[key].sample(rngs[key], 1)[0]) / 1000.0
                push(max(t + 1e-6, nominal_next + jit), "publish", (key, n + 1))
                stats.published[key] += 1
                active = [f for f in cfg.faults if f.topic == key and f.active(t)]
                for f in active:
                    i = cfg.faults.index(f)
                    if i not in fired:
                        fired.add(i)
                        stats.faults_fired.append(f"{f.kind}:{key}@{t:.2f}s")
                if any(f.kind == "drop" for f in active) or rngs[key].random() < spec.drop_rate:
                    stats.dropped[key] += 1
                    continue
                value = sources[key].value(t, plant_obs)
                if any(f.kind == "freeze" for f in active):
                    value = frozen.setdefault(key, value)
                else:
                    frozen.pop(key, None)
                for f in active:
                    if f.kind == "noise":
                        value = np.asarray(value, dtype=float) + rngs[key].normal(0.0, f.severity, np.shape(value))
                    elif f.kind == "bias":
                        value = np.asarray(value, dtype=float) + f.severity
                delay = max(0.0, float(self.delay[key].sample(rngs[key], 1)[0]) / 1000.0)
                delay += sum(f.severity / 1000.0 for f in active if f.kind == "delay")
                push(t + delay, "arrive", (key, value, t))
            elif kind == "arrive":
                key, value, sampled = payload
                prev = latest.get(key)
                if prev is None or sampled >= prev.sampled:  # a late message never overwrites a newer one
                    latest[key] = _Latest(value, sampled, t)
                stats.delivered[key] += 1
                if self.on_arrive is not None:
                    self.on_arrive(key, value, wall0 + sampled)
            else:  # control tick
                obs: Obs = {}
                extra: dict[str, float] = {}
                if cfg.passthrough:
                    obs.update({k: v for k, v in plant_obs.items() if k not in specs})
                for key, spec in specs.items():
                    m = latest.get(key)
                    if m is None:
                        stats.missing_ticks[key] += 1
                        extra[f"rig.age_ms.{key}"] = math.nan
                        continue
                    age = t - m.sampled
                    obs[key] = np.asarray(m.value, dtype=np.float32) if key.startswith("state.") else m.value
                    stats.max_age_ms[key] = max(stats.max_age_ms[key], 1000 * age)
                    stale = age > spec.stale_after_s
                    stats.stale_ticks[key] += int(stale)
                    extra[f"rig.age_ms.{key}"] = 1000 * age
                    extra[f"rig.stale.{key}"] = float(stale)
                if any(k not in obs for k in specs):
                    # a topic has not delivered yet: the controller has nothing to act on, so hold the last
                    # command (or send zeros before the first one)
                    command = stats.commands[-1] if stats.commands else np.zeros(self._action_dim())
                else:
                    command = np.asarray(self.policy.act(obs), dtype=float)
                stats.commands.append(command)
                stats.ticks += 1
                if self.recorder is not None:
                    self.recorder.record_step(obs, command, timestamp=t, extra=extra)
                plant_obs, _, term, trunc, info = self.plant.step(command)
                if payload == n_ticks - 1 and not (term or trunc):
                    done = True  # the run's duration is up; publishers stop with it
                if term or trunc:
                    stats.success = bool(info.get("success", False))
                    stats.failure_reason = None if stats.success else str(info.get("failure_reason", "unknown"))
                    done = True
        if stats.success is None:
            stats.success, stats.failure_reason = None, None
        if self.recorder is not None:
            self.recorder.end_episode(success=stats.success, failure_reason=stats.failure_reason,
                                      rig_stats=episode_summary(stats))
        return stats

    def _action_dim(self) -> int:
        dim = getattr(self.plant, "action_dim", None) or getattr(self.plant, "dim", None)
        if dim is None:
            raise ValueError("the plant has no action_dim; give the topics' first messages time to arrive")
        return int(dim)

    def close(self) -> None:
        self.plant.close()


def episode_summary(s: EpisodeStats) -> dict[str, Any]:
    ticks = max(s.ticks, 1)
    return {"ticks": s.ticks, "success": s.success, "failure_reason": s.failure_reason, "overruns": s.overruns,
            "max_lateness_ms": s.max_lateness_ms, "faults_fired": list(s.faults_fired),
            "topics": {k: {"published": s.published[k], "delivered": s.delivered[k], "dropped": s.dropped[k],
                           "stale_frac": s.stale_ticks[k] / ticks, "missing_ticks": s.missing_ticks[k],
                           "max_age_ms": s.max_age_ms[k]} for k in s.published}}


def run_rig(cfg: RigConfig, seed: int, episodes: int, recorder: Any = None,
            policy_factory: Callable[[], Any] | None = None,
            on_arrive: Callable[[str, Any, float], None] | None = None) -> list[EpisodeStats]:
    run = RunSeed(seed)
    session = RigSession(cfg, run, policy_factory() if policy_factory else None, recorder=recorder,
                         on_arrive=on_arrive)
    try:
        return [session.run_episode(Scenario(seed=s), i) for i, s in enumerate(run.scenario_seeds(episodes))]
    finally:
        session.close()


def compare_to_ideal(cfg: RigConfig, seed: int, episodes: int, recorder: Any = None,
                     on_arrive: Callable[[str, Any, float], None] | None = None) -> dict[str, Any]:
    """The same seeded scenarios with the rig's streams and with ideal streams: success and command drift."""
    from groot_eval.stats import mcnemar_exact, wilson_interval

    rig = run_rig(cfg, seed, episodes, recorder=recorder, on_arrive=on_arrive)
    ideal = run_rig(cfg.ideal(), seed, episodes)
    lost = sum(1 for r, i in zip(rig, ideal) if i.success and not r.success)
    gained = sum(1 for r, i in zip(rig, ideal) if r.success and not i.success)
    drift = []
    for r, i in zip(rig, ideal):
        n = min(len(r.commands), len(i.commands))
        if n:
            d = np.stack(r.commands[:n]) - np.stack(i.commands[:n])
            drift.append(float(np.sqrt(np.mean(d ** 2))))
    ok_r = sum(1 for r in rig if r.success)
    ok_i = sum(1 for i in ideal if i.success)
    return {"episodes": episodes, "seed": seed,
            "rig": {"success": ok_r, "ci": wilson_interval(ok_r, episodes)},
            "ideal": {"success": ok_i, "ci": wilson_interval(ok_i, episodes)},
            "lost": lost, "gained": gained, "mcnemar_p": mcnemar_exact(lost, gained),
            "command_rms_drift": float(np.mean(drift)) if drift else math.nan,
            "per_episode": [episode_summary(r) for r in rig]}


def rig_markdown(cfg: RigConfig, cmp: dict[str, Any], title: str = "") -> str:
    n = cmp["episodes"]
    lines = [f"# Rig run{': ' + title if title else ''}", "",
             f"Policy `{cfg.policy}` on {cfg.plant.get('kind')} plant `{cfg.plant.get('env', '')}`, control "
             f"{cfg.control_hz:g} Hz, {len(cfg.topics)} topics, {len(cfg.faults)} stream faults, clock `{cfg.clock}`, "
             f"run seed {cmp['seed']}.", "",
             "| Streams | Success | 95% CI |", "|---|---|---|",
             f"| ideal | {cmp['ideal']['success']}/{n} | [{100 * cmp['ideal']['ci'][0]:.0f}, {100 * cmp['ideal']['ci'][1]:.0f}] |",
             f"| rig | {cmp['rig']['success']}/{n} | [{100 * cmp['rig']['ci'][0]:.0f}, {100 * cmp['rig']['ci'][1]:.0f}] |", "",
             f"Paired on the same scenarios: {cmp['lost']} succeed only with ideal streams, {cmp['gained']} only with the "
             f"rig's (McNemar p = {cmp['mcnemar_p']:.3g}). Commands drifted from the ideal run by RMS "
             f"{cmp['command_rms_drift']:.4f}.", "", "## Streams (summed over episodes)", "",
             "| Topic | Rate (Hz) | Source | Published | Dropped | Stale ticks | Max age (ms) |",
             "|---|---|---|---|---|---|---|"]
    per = cmp["per_episode"]
    for t in cfg.topics:
        rows = [p["topics"][t.key] for p in per]
        pub = sum(r["published"] for r in rows)
        drop = sum(r["dropped"] for r in rows)
        stale = float(np.mean([r["stale_frac"] for r in rows]))
        age = max(r["max_age_ms"] for r in rows)
        lines.append(f"| `{t.key}` | {t.rate_hz:g} | {t.source.get('kind')} | {pub} | {drop} | {100 * stale:.1f}% | {age:.1f} |")
    fired = sorted({f.split("@")[0] for p in per for f in p["faults_fired"]})
    if fired:
        lines += ["", "Stream faults fired: " + ", ".join(f"`{f}`" for f in fired)]
    over = sum(p["overruns"] for p in per)
    if cfg.clock == "real":
        lines += ["", f"Real-time clock: {over} control-tick overruns; worst lateness "
                      f"{max(p['max_lateness_ms'] for p in per):.2f} ms."]
    return "\n".join(lines) + "\n"
