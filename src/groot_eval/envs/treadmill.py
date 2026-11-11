"""Treadmill for the walking twin: split belts, incline, recentering, scripted perturbations, belt replay.

``make_env("treadmill")`` is :class:`TreadmillEnv` around a
:class:`~groot_eval.envs.walker.Walker`. Treadmill settings are scenario
parameters with a ``treadmill.`` prefix, so sweeps, system identification,
repro and the bridge set them like any other parameter.

| Parameter | Meaning |
|---|---|
| ``treadmill.speed`` | belt speed, m/s (the belt surface moves toward -x) |
| ``treadmill.ramp_s`` | time to ramp from 0 to ``speed`` at the start |
| ``treadmill.mode`` | ``fixed``, or ``recenter`` (the belt speeds up as the robot drifts forward) |
| ``treadmill.kp`` / ``treadmill.kd`` | recentering gains on deck position and room velocity |
| ``treadmill.accel_limit`` | belt motor acceleration limit, m/s^2 |
| ``treadmill.deck_length`` | usable deck length, m (leaving it is ``off_belt``) |
| ``treadmill.split`` | right belt speed / left belt speed |
| ``treadmill.incline_deg`` | deck incline (positive = walking uphill) |
| ``treadmill.perturbations`` | scripted perturbations (below) |
| ``treadmill.belt`` | replay recorded belt speeds instead: ``trace:<dir>`` (the episode with the same seed) or a list of ``[left, right]`` per control step |

Perturbations are dicts ``{type, side, start_s, trigger, duration_s, severity}``:

| Type | Severity | Effect |
|---|---|---|
| ``trip`` | belt acceleration, m/s^2 | the side's belt slows for 50 ms, surges for 270 ms, and returns over 220 ms. This is the timed trip protocol used on instrumented split-belt treadmills; the surge pulls the stance foot back. |
| ``slip`` | friction coefficient | the side's belt is slippery for ``duration_s`` |
| ``speed_step`` | m/s | both belts change speed for ``duration_s`` |
| ``split`` | speed ratio | the side's belt runs at ``severity`` times speed |
| ``incline_step`` | degrees | the deck tilts for ``duration_s`` |
| ``push`` | newtons | a horizontal push on the CoM (a gantry pusher) |

The ``trigger`` field sets when a perturbation starts:

* ``touchdown`` (the default for trip and slip): the first touchdown of that side after ``start_s``, so the perturbation hits the same point of the gait every time;
* ``time``: at ``start_s`` exactly.

``state.belt_speed`` (left, right) is added to the observations, so every
trace records the belt. A real robot's trace carries it too, and
``treadmill.belt: trace:<dir>`` replays it into the twin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.degradation.components import EnvWrapper
from groot_eval.types import Scenario

Obs = dict[str, Any]
PREFIX = "treadmill."
TREADMILL_DEFAULTS: dict[str, Any] = {
    "speed": 0.8, "ramp_s": 2.0, "mode": "fixed", "kp": 0.8, "kd": 0.3, "accel_limit": 3.0,
    "deck_length": 1.6, "split": 1.0, "incline_deg": 0.0, "perturbations": [], "belt": None,
}
TYPES = ("trip", "slip", "speed_step", "split", "incline_step", "push")
SIDES = {"left": 0, "right": 1, "both": -1}
TRIP_PHASES = (0.05, 0.27, 0.22)  # decelerate, accelerate, return (s)


@dataclass
class Perturbation:
    type: str
    severity: float
    side: str = "both"
    start_s: float = 0.0
    trigger: str = ""
    duration_s: float = 0.0
    t_on: float | None = None  # when it actually started (filled in by the treadmill)

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"unknown perturbation {self.type!r}; known: {TYPES}")
        if self.side not in SIDES:
            raise ValueError(f"side must be left, right or both, not {self.side!r}")
        if not self.trigger:
            self.trigger = "touchdown" if self.type in ("trip", "slip") and self.side != "both" else "time"
        if self.trigger not in ("time", "touchdown"):
            raise ValueError("trigger must be time or touchdown")
        if self.type == "trip":
            self.duration_s = sum(TRIP_PHASES)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Perturbation:
        known = {"type", "severity", "side", "start_s", "trigger", "duration_s"}
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown perturbation keys {sorted(extra)}")
        return cls(**{k: d[k] for k in known if k in d})

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "severity": self.severity, "side": self.side, "start_s": self.start_s,
                "trigger": self.trigger, "duration_s": self.duration_s}

    def active(self, t: float) -> float | None:
        """Seconds since the perturbation started, or None when it is not active."""
        if self.t_on is None or t < self.t_on or t > self.t_on + self.duration_s:
            return None
        return t - self.t_on

    def hits(self, side: int) -> bool:
        return self.side == "both" or SIDES[self.side] == side


def trip_offset(tau: float, accel: float) -> float:
    """Belt speed change during a trip: slower for 50 ms, a surge for 270 ms, back to normal over 220 ms."""
    d, a, r = TRIP_PHASES
    if tau < d:
        return -accel * tau
    if tau < d + a:
        return -accel * d + accel * (tau - d)
    peak = accel * (a - d)
    return max(0.0, peak - peak * (tau - d - a) / r)


def _belt_from_trace(spec: str) -> dict[int, np.ndarray]:
    """``trace:<dir>`` -> {scenario seed: (T, 2) recorded belt speeds}."""
    from groot_eval.bridge import scenarios_from_meta
    from groot_eval.trace import TraceSet

    ts = TraceSet.open(spec.removeprefix("trace:"))
    sl = ts.state_slices()
    if "belt_speed" not in sl:
        raise ValueError(f"{ts.root} has no state.belt_speed; it was not recorded on a treadmill")
    out = {}
    for sc, states in zip(scenarios_from_meta(ts.episode_meta()), ts.per_episode("observation.state")):
        out[sc.seed] = np.asarray(states, dtype=float)[:, sl["belt_speed"]]
    return out


_BELT_CACHE: dict[str, dict[int, np.ndarray]] = {}


class Treadmill:
    """The ground under the walker: two belts, a deck, and whatever the perturbations do to them."""

    def __init__(self, cfg: dict[str, Any], seed: int, dt: float) -> None:
        unknown = set(cfg) - set(TREADMILL_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown treadmill parameters {sorted(PREFIX + k for k in unknown)}")
        self.cfg = {**TREADMILL_DEFAULTS, **cfg}
        if self.cfg["mode"] not in ("fixed", "recenter"):
            raise ValueError("treadmill.mode must be fixed or recenter")
        self.dt = dt
        self.deck_half = float(self.cfg["deck_length"]) / 2
        self.perturbations = [Perturbation.from_dict(p) if isinstance(p, dict) else p
                              for p in (self.cfg["perturbations"] or [])]
        self.replay: np.ndarray | None = None
        belt = self.cfg["belt"]
        if isinstance(belt, str):
            if belt not in _BELT_CACHE:
                _BELT_CACHE[belt] = _belt_from_trace(belt)
            table = _BELT_CACHE[belt]
            if seed not in table:
                raise KeyError(f"{belt} has no episode with seed {seed}; seeds: {sorted(table)}")
            self.replay = table[seed]
        elif belt is not None:
            self.replay = np.asarray(belt, dtype=float).reshape(-1, 2)
        self.t = 0.0
        self.speed: np.ndarray = np.zeros(2)  # belt surface speed magnitude, left/right
        self.motor: np.ndarray = np.zeros(2)  # what the speed loop drives, before trip surges
        self.log: list[dict[str, Any]] = []

    # -- what the walker asks ---------------------------------------------------------
    def reset(self, scenario: Scenario, robot: Any) -> None:
        self.t = 0.0
        if self.replay is not None:
            self.speed = self.replay[0].copy()
        else:
            self.speed = np.zeros(2) if float(self.cfg["ramp_s"]) > 0 else self._target(robot)
        self.motor = self.speed.copy()

    def velocity(self, side: int) -> float:
        return -float(self.speed[side])

    def friction(self, side: int, mu: float) -> float:
        for p in self.perturbations:
            if p.type == "slip" and p.hits(side) and p.active(self.t) is not None:
                mu = min(mu, float(p.severity))
        return mu

    def incline(self) -> float:
        inc = float(self.cfg["incline_deg"])
        for p in self.perturbations:
            if p.type == "incline_step" and p.active(self.t) is not None:
                inc += float(p.severity)
        return inc

    def push(self) -> float:
        return sum(float(p.severity) for p in self.perturbations if p.type == "push" and p.active(self.t) is not None)

    def on_touchdown(self, side: int, robot: Any) -> None:
        for p in self.perturbations:
            if p.t_on is None and p.trigger == "touchdown" and p.hits(side) and self.t >= float(p.start_s):
                p.t_on = self.t
                self.log.append({"type": p.type, "side": p.side, "t": self.t, "step": robot.s.steps})

    def _target(self, robot: Any) -> np.ndarray:
        cfg = self.cfg
        base = float(cfg["speed"])
        ramp = float(cfg["ramp_s"])
        if ramp > 0:
            base *= min(1.0, self.t / ramp)
        if cfg["mode"] == "recenter":
            base += float(cfg["kp"]) * robot.s.x + float(cfg["kd"]) * robot.s.xd
        target = np.array([base, base * float(cfg["split"])])
        for p in self.perturbations:
            tau = p.active(self.t)
            if tau is None:
                continue
            for side in (0, 1):
                if not p.hits(side):
                    continue
                if p.type == "speed_step":
                    target[side] += float(p.severity)
                elif p.type == "split":
                    target[side] *= float(p.severity)
        return np.clip(target, 0.0, 6.0)

    def update(self, h: float, robot: Any) -> None:
        for p in self.perturbations:
            if p.t_on is None and p.trigger == "time" and self.t >= float(p.start_s):
                p.t_on = self.t
                self.log.append({"type": p.type, "side": p.side, "t": self.t, "step": robot.s.steps})
        self.t += h
        if self.replay is not None:  # follow the recording, interpolating between control steps
            k = self.t / self.dt
            i = min(int(k), len(self.replay) - 1)
            j = min(i + 1, len(self.replay) - 1)
            f = k - int(k)
            self.speed = (1 - f) * self.replay[i] + f * self.replay[j]
            return
        target = self._target(robot)
        step = float(self.cfg["accel_limit"]) * h
        self.motor = self.motor + np.clip(target - self.motor, -step, step)
        surge = np.zeros(2)  # a trip drives the belt past the speed loop, then hands it back
        for p in self.perturbations:
            tau = p.active(self.t)
            if p.type == "trip" and tau is not None:
                for side in (0, 1):
                    if p.hits(side):
                        surge[side] += trip_offset(tau, float(p.severity))
        self.speed = np.maximum(0.0, self.motor + surge)

    def set_state(self, state: Obs) -> None:
        v = state.get("state.belt_speed")
        if v is not None:
            self.speed = np.asarray(v, dtype=float).reshape(-1)[:2].copy()
            self.motor = self.speed.copy()

    def observe(self) -> Obs:
        return {"state.belt_speed": self.speed.astype(np.float32)}


class TreadmillEnv(EnvWrapper):
    """A walker on a treadmill. ``treadmill.*`` scenario parameters configure the belts for each episode."""

    def __init__(self, env: Any = None, **defaults: Any) -> None:
        if env is None:
            from groot_eval.envs.walker import Walker

            env = Walker()
        super().__init__(env)
        unknown = set(defaults) - set(TREADMILL_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown treadmill defaults {sorted(unknown)}")
        self.defaults = dict(defaults)
        self.treadmill: Treadmill | None = None

    def reset(self, scenario: Scenario) -> Obs:
        tm = {k.removeprefix(PREFIX): v for k, v in scenario.params.items() if k.startswith(PREFIX)}
        rest = {k: v for k, v in scenario.params.items() if not k.startswith(PREFIX)}
        self.treadmill = Treadmill({**self.defaults, **tm}, scenario.seed, self.env.dt)
        self.env.ground = self.treadmill
        return self.env.reset(Scenario(seed=scenario.seed, params=rest, task=scenario.task, name=scenario.name))

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        obs, r, term, trunc, info = self.env.step(action)
        if (term or trunc) and self.treadmill is not None:
            m = info.setdefault("metrics", {})
            sp = self.treadmill.speed
            m["belt_speed_end"] = float(np.mean(sp))
            m["perturbations_fired"] = float(len(self.treadmill.log))
            if self.treadmill.log:
                info["perturbations"] = list(self.treadmill.log)
        return obs, r, term, trunc, info


# =====================================================================================
# Perturbation catalogue: sampled, per scenario, from one seed
# =====================================================================================
@dataclass
class CatalogueEntry:
    type: str
    severity: list[float] | float
    side: list[str] | str = "both"
    duration_s: list[float] | float = 0.0
    trigger: str = ""
    weight: float = 1.0


@dataclass
class PerturbationConfig:
    """``configs/treadmill/*.yaml``: what can happen to the robot on the treadmill, and when.

    ``mode: single`` gives every scenario one perturbation, started in ``start_window_s``;
    ``mode: random`` gives Poisson arrivals at ``rate_per_min``, at most ``max_events``.
    """

    catalogue: list[CatalogueEntry]
    mode: str = "single"
    start_window_s: list[float] = field(default_factory=lambda: [3.0, 10.0])
    rate_per_min: float = 6.0
    max_events: int = 4
    base: dict[str, Any] = field(default_factory=dict)  # treadmill settings for every scenario

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerturbationConfig:
        d = dict(data.get("perturbations", data))
        cat = [CatalogueEntry(**e) for e in d.pop("catalogue", [])]
        for e in cat:
            if e.type not in TYPES:
                raise ValueError(f"unknown perturbation {e.type!r}; known: {TYPES}")
        cfg = cls(catalogue=cat, base=dict(data.get("treadmill") or {}), **d)
        if cfg.mode not in ("single", "random"):
            raise ValueError("perturbations.mode must be single or random")
        if not cat:
            raise ValueError("the perturbation catalogue is empty")
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> PerturbationConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def _draw(self, rng: np.random.Generator, value: Any) -> Any:
        if isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return float(rng.uniform(value[0], value[1]))
        if isinstance(value, list):
            return value[int(rng.integers(len(value)))]
        return value

    def sample(self, rng: np.random.Generator, horizon_s: float) -> list[dict[str, Any]]:
        weights = np.array([e.weight for e in self.catalogue], dtype=float)
        weights /= weights.sum()
        if self.mode == "single":
            lo, hi = self.start_window_s
            starts = [float(rng.uniform(float(lo), float(hi)))]
        else:
            starts, t = [], float(self.start_window_s[0])
            while len(starts) < self.max_events:
                t += float(rng.exponential(60.0 / max(self.rate_per_min, 1e-9)))
                if t >= horizon_s:
                    break
                starts.append(t)
        out = []
        for t0 in starts:
            e = self.catalogue[int(rng.choice(len(self.catalogue), p=weights))]
            p = Perturbation(type=e.type, severity=round(float(self._draw(rng, e.severity)), 4),
                             side=str(self._draw(rng, e.side)), start_s=round(t0, 3), trigger=e.trigger,
                             duration_s=round(float(self._draw(rng, e.duration_s)), 3))
            out.append(p.to_dict())
        return out


def perturbed_scenarios(base: list[Scenario], cfg: PerturbationConfig, seed: int,
                        horizon_s: float = 20.0) -> list[Scenario]:
    """One perturbation schedule per scenario, fixed by (``seed``, scenario seed): rerunning gives the same schedule."""
    out = []
    for sc in base:
        rng = np.random.default_rng([int(seed), int(sc.seed), 0x7EAD])
        sched = cfg.sample(rng, horizon_s)
        params = {**{PREFIX + k: v for k, v in cfg.base.items()}, **sc.params, PREFIX + "perturbations": sched}
        out.append(Scenario(seed=sc.seed, params=params, task=sc.task or "walk",
                            name=sc.name or "+".join(p["type"] for p in sched)))
    return out


def schedule_label(params: dict[str, Any]) -> str:
    sched = params.get(PREFIX + "perturbations") or []
    return "+".join(p["type"] for p in sched) or "none"


def describe_schedule(params: dict[str, Any]) -> str:
    return json.dumps(params.get(PREFIX + "perturbations") or [])
