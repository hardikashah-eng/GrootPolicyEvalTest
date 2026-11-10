"""Fault injection: components fail mid-episode, and the robot has to handle it.

Wear changes a robot slowly; faults change it during an episode. ``FaultEnv``
wraps any env and plays a **fault schedule**: timed events, each on one
component. Long scenarios draw several faults at random times, so one sweep
asks whether the robot detects, contains and recovers, or stops safely, as
subsystems go down.

A schedule is a scenario parameter, written compactly so it is part of the
scenario id, saved in regression suites and replayed exactly::

    faults: "stuck@encoder[1]:40+60;motor_off@motor[0]:120+inf*1.0"
            type @ target : start step + duration steps (inf = permanent) * severity

Fault types:

==================  =========================  ========================================
``stuck``           encoder / sensor           reading frozen at its last value
``dropout``         encoder / sensor           reading is NaN
``bias_step``       encoder / sensor           sudden offset (severity, default 0.3)
``noise_burst``     encoder / sensor           Gaussian noise (std = severity, default 0.05)
``blackout``        camera                     black frames
``motor_off``       motor                      no torque (joint goes limp)
``motor_runaway``   motor                      command forced to +severity (watchdog test)
``joint_lock``      joint / motor              joint seized
``gripper_fail``    gripper                    stuck open (severity < 0.5) or closed
``comm_stall``      bus                        last command repeated, new ones lost
``latency_spike``   bus                        commands arrive ``severity`` steps late (default 5)
``power_sag``       bus                        all motors limited to ``severity`` (default 0.4) of torque
==================  =========================  ========================================

Physics faults use the env's ``set_component_param`` when it has the lever
(``Arm3`` does), otherwise the command boundary (the action is zeroed, forced
or clipped).

Metrics added to every episode (``metrics.fault.*``): faults started, step of
the first onset, whether an unsafe event followed (collision, instability or
joint-limit hits after onset), whether the
episode ended in a **controlled stop** (not done, but nothing unsafe and the
robot at rest), **safe outcome** (success or controlled stop, and nothing
unsafe), steps until the policy's commands changed after onset (reaction),
steps after a transient fault ends until the task error is back below its
level at onset (recovery), the worst task-error increase after onset, and the
steps with saturated torque after onset (reported, not counted as unsafe).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.degradation.components import (EnvWrapper, discover, env_action_dim, get_physics, match,
                                               split_params, state_slices, try_physics)
from groot_eval.envs.base import ComponentSpec, Obs
from groot_eval.types import Scenario

PREFIX = "faults"
SENSOR_FAULTS = ("stuck", "dropout", "bias_step", "noise_burst", "blackout")
ACTUATOR_FAULTS = ("motor_off", "motor_runaway", "joint_lock", "gripper_fail")
BUS_FAULTS = ("comm_stall", "latency_spike", "power_sag")
FAULT_TYPES = SENSOR_FAULTS + ACTUATOR_FAULTS + BUS_FAULTS
UNSAFE_REASONS = {"collision", "unstable", "out_of_bounds", "fall", "self_collision", "joint_limit"}
DEFAULT_SEVERITY = {"bias_step": 0.3, "noise_burst": 0.05, "motor_runaway": 1.0, "gripper_fail": 0.0,
                    "latency_spike": 5.0, "power_sag": 0.4}


@dataclass(frozen=True)
class Fault:
    type: str
    target: str
    start: int
    duration: int | None  # None = permanent
    severity: float | None = None

    def active(self, t: int) -> bool:
        return self.start <= t and (self.duration is None or t < self.start + self.duration)

    @property
    def end(self) -> float:
        return math.inf if self.duration is None else self.start + self.duration

    @property
    def sev(self) -> float:
        return self.severity if self.severity is not None else DEFAULT_SEVERITY.get(self.type, 1.0)

    def encode(self) -> str:
        d = "inf" if self.duration is None else str(self.duration)
        s = "" if self.severity is None else f"*{self.severity:g}"
        return f"{self.type}@{self.target}:{self.start}+{d}{s}"


def encode_schedule(faults: list[Fault]) -> str:
    return ";".join(f.encode() for f in sorted(faults, key=lambda f: (f.start, f.target, f.type)))


def parse_schedule(text: str | list | None) -> list[Fault]:
    if not text:
        return []
    if isinstance(text, list):  # list of dicts, e.g. from YAML
        return [Fault(d["type"], d["target"], int(d.get("start", 0)),
                      None if d.get("duration") in (None, "inf", "permanent") else int(d["duration"]),
                      None if d.get("severity") is None else float(d["severity"])) for d in text]
    out = []
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            kind, rest = part.split("@", 1)
            target, timing = rest.rsplit(":", 1)
            sev = None
            if "*" in timing:
                timing, s = timing.split("*", 1)
                sev = float(s)
            start, dur = timing.split("+", 1)
            out.append(Fault(kind, target, int(start), None if dur in ("inf", "permanent") else int(dur), sev))
        except ValueError as exc:
            raise ValueError(f"bad fault {part!r}; expected type@target:start+duration[*severity]") from exc
        if kind not in FAULT_TYPES:
            raise ValueError(f"unknown fault type {kind!r}; known: {FAULT_TYPES}")
    return out


# -- sampling ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CatalogueEntry:
    type: str
    targets: tuple[str, ...]
    duration: tuple[int, int] | None  # None = permanent
    severity: tuple[float, float] | None = None
    weight: float = 1.0


@dataclass
class FaultConfig:
    """How to draw fault schedules: Poisson arrivals over the episode, capped concurrency."""

    catalogue: list[CatalogueEntry]
    mode: str = "random"  # random | single | list
    rate_per_min: float = 1.0
    max_concurrent: int = 2
    max_faults: int = 6
    earliest_step: int = 10
    start_window: tuple[int, int] | None = None  # single mode: onset drawn from [lo, hi)
    explicit: list[Fault] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FaultConfig":
        data = data.get("faults", data)
        cat = []
        for e in data.get("catalogue") or []:
            if e["type"] not in FAULT_TYPES:
                raise ValueError(f"unknown fault type {e['type']!r}")
            d = e.get("duration", [20, 100])
            dur = None if d in ("permanent", "inf", None) else (int(d), int(d)) if isinstance(d, (int, float)) else (int(d[0]), int(d[1]))
            s = e.get("severity")
            sev = None if s is None else (float(s), float(s)) if isinstance(s, (int, float)) else (float(s[0]), float(s[1]))
            targets = e.get("targets") or [e.get("target")]
            cat.append(CatalogueEntry(e["type"], tuple(str(t) for t in targets), dur, sev, float(e.get("weight", 1.0))))
        mode = str(data.get("mode", "random"))
        explicit = parse_schedule(data.get("schedule")) if data.get("schedule") else []
        if mode not in ("random", "single", "list"):
            raise ValueError("faults.mode must be random, single or list")
        if mode != "list" and not cat:
            raise ValueError("fault config needs a catalogue")
        win = data.get("start_window")
        return cls(cat, mode, float(data.get("rate_per_min", 1.0)), int(data.get("max_concurrent", 2)),
                   int(data.get("max_faults", 6)), int(data.get("earliest_step", 10)),
                   (int(win[0]), int(win[1])) if win else None, explicit)

    @classmethod
    def load(cls, path: str | Path) -> "FaultConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def sample(self, rng: np.random.Generator, components: dict[str, ComponentSpec], horizon_steps: int,
               dt: float) -> list[Fault]:
        if self.mode == "list":
            return list(self.explicit)
        weights = np.array([c.weight for c in self.catalogue], dtype=float)
        weights /= weights.sum()
        if self.mode == "single":
            n = 1
            lo, hi = self.start_window or (self.earliest_step, max(self.earliest_step + 1, horizon_steps // 2))
            starts = [int(rng.integers(lo, max(lo + 1, hi)))]
        else:
            rate_per_step = self.rate_per_min * dt / 60.0
            starts, t = [], self.earliest_step
            while len(starts) < self.max_faults:
                t += int(math.ceil(rng.exponential(1.0 / max(rate_per_step, 1e-9))))
                if t >= horizon_steps:
                    break
                starts.append(t)
            n = len(starts)
        out: list[Fault] = []
        for start in starts[:n]:
            entry = self.catalogue[int(rng.choice(len(self.catalogue), p=weights))]
            names = [n for pat in entry.targets for n in match(components, pat)]
            if not names:
                continue
            target = names[int(rng.integers(len(names)))]
            dur = None if entry.duration is None else int(rng.integers(entry.duration[0], entry.duration[1] + 1))
            sev = None if entry.severity is None else float(round(rng.uniform(*entry.severity), 3))
            cand = Fault(entry.type, target, start, dur, sev)
            if sum(1 for f in out if f.active(start)) >= self.max_concurrent:
                continue
            out.append(cand)
        return out


def faulted_scenarios(base: list[Scenario], cfg: FaultConfig, components: dict[str, ComponentSpec],
                      horizon_steps: int, dt: float) -> list[Scenario]:
    """Attach a seeded fault schedule to each scenario (``faults`` + ``faults.n`` + ``faults.kinds``)."""
    out = []
    for s in base:
        rng = np.random.default_rng((s.seed * 104729 + 7) % 2**32)
        sched = cfg.sample(rng, components, horizon_steps, dt)
        params = {**s.params, PREFIX: encode_schedule(sched), f"{PREFIX}.n": len(sched),
                  f"{PREFIX}.kinds": "+".join(sorted({f.type for f in sched})) or "none"}
        out.append(Scenario(seed=s.seed, params=params, task=s.task, name=s.name))
    return out


# -- the wrapper --------------------------------------------------------------------------
class FaultEnv(EnvWrapper):
    def __init__(self, env: Any, action_dim: int | None = None) -> None:
        super().__init__(env)
        self._action_dim = action_dim
        self.schedule: list[Fault] = []
        self.parts: dict[str, ComponentSpec] = {}

    def reset(self, scenario: Scenario) -> Obs:
        mine, rest = split_params(scenario.params, PREFIX)
        self.schedule = parse_schedule(mine.get(PREFIX))
        obs = self.env.reset(Scenario(seed=scenario.seed, params=rest, task=scenario.task, name=scenario.name))
        self.parts = discover(self.env, obs, self._action_dim or env_action_dim(self.env))
        for f in self.schedule:
            if f.target not in self.parts:
                raise ValueError(f"fault target {f.target!r} is not a component; have {sorted(self.parts)}")
        self.t = 0
        self._rng = np.random.default_rng((scenario.seed * 31337 + 3) % 2**32)
        self._held: dict[str, dict[str, np.ndarray]] = {}
        self._physics_on: dict[Fault, Any] = {}
        self._physics_tried: set[Fault] = set()
        self._last_sent: np.ndarray | None = None
        self._delay: deque[np.ndarray] = deque()
        self._prev_action: np.ndarray | None = None
        self._changes: list[float] = []
        self._onset: int | None = None
        self._react: int | None = None
        self._err: list[float] = []
        self._err_onset: float | None = None
        self._unsafe_base: dict[str, float] | None = None
        self._last_obs = obs
        self._started = 0
        dim = self._action_dim or env_action_dim(self.env)
        self._dims = self._reaction_dims(dim) if self.schedule and dim else None
        return self._observe(obs, 0)

    # -- commands ------------------------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        t = self.t
        a = np.asarray(action, dtype=float).reshape(-1).copy()
        self._track_reaction(a, t)
        self._physics(t)
        for f in self.schedule:
            if not f.active(t):
                continue
            spec = self.parts[f.target]
            if f.type in ("motor_off", "joint_lock") and f not in self._physics_on:
                for i in spec.action:
                    if i < a.size:
                        a[i] = 0.0
            elif f.type == "motor_runaway":
                for i in spec.action:
                    if i < a.size:
                        a[i] = f.sev
            elif f.type == "gripper_fail" and f not in self._physics_on:
                for i in spec.action:
                    if i < a.size:
                        a[i] = 1.0 if f.sev >= 0.5 else 0.0
            elif f.type == "power_sag" and f not in self._physics_on:
                a = np.clip(a, -f.sev, f.sev)
        stall = any(f.type == "comm_stall" and f.active(t) for f in self.schedule)
        if stall and self._last_sent is not None:
            a = self._last_sent
        spike = max([int(f.sev) for f in self.schedule if f.type == "latency_spike" and f.active(t)] or [0])
        self._delay.append(a)
        while len(self._delay) > spike + 1:
            self._delay.popleft()
        if spike:
            a = self._delay[0]
        self._last_sent = a
        obs, reward, term, trunc, info = self.env.step(a)
        self.t += 1
        self._track_error(info, reward)
        if term or trunc:
            for f in list(self._physics_on):
                self._physics_off(f)
            info = self._finish(info, obs)
        info = dict(info)
        active = [f for f in self.schedule if f.active(t)]
        info["step_columns"] = {**(info.get("step_columns") or {}), "fault.active": float(len(active)),
                                "fault.type_id": float(FAULT_TYPES.index(active[0].type)) if active else -1.0}
        return self._observe(obs, self.t), reward, term, trunc, info

    def _physics(self, t: int) -> None:
        for f in self.schedule:
            if f.type not in ("motor_off", "joint_lock", "gripper_fail", "power_sag"):
                continue
            if f.active(t) and f not in self._physics_on and f not in self._physics_tried:
                self._physics_tried.add(f)
                self._physics_onset(f)
            elif not f.active(t) and f in self._physics_on:
                self._physics_off(f)

    def _physics_onset(self, f: Fault) -> None:
        spec = self.parts[f.target]
        idx = spec.name[spec.name.find("["):] if "[" in spec.name else ""
        if f.type == "motor_off":
            name, key, value = (f"motor{idx}" if idx else spec.name), "enabled", False
        elif f.type == "joint_lock":
            name, key, value = (f"joint{idx}" if idx else spec.name), "locked", True
        elif f.type == "gripper_fail":
            name, key, value = spec.name, "stuck", "closed" if f.sev >= 0.5 else "open"
        else:  # power_sag on every motor
            motors = [n for n, c in self.parts.items() if c.kind == "motor"]
            before = {m: get_physics(self.env, m, "torque_limit_scale", 1.0) for m in motors}
            if motors and all(try_physics(self.env, m, "torque_limit_scale", before[m] * f.sev) for m in motors):
                self._physics_on[f] = ("power_sag", before)
            return
        before = get_physics(self.env, name, key, None)
        if try_physics(self.env, name, key, value):
            self._physics_on[f] = (name, key, before)

    def _physics_off(self, f: Fault) -> None:
        state = self._physics_on.pop(f, None)
        if not state:
            return
        if state[0] == "power_sag":
            for m, v in state[1].items():
                try_physics(self.env, m, "torque_limit_scale", v)
        else:
            name, key, before = state
            if before is not None:
                try_physics(self.env, name, key, before)

    # -- observations ----------------------------------------------------------------------
    def _observe(self, obs: Obs, t: int) -> Obs:
        sensor = [f for f in self.schedule if f.type in SENSOR_FAULTS]
        if not sensor:
            self._last_obs = obs
            return obs
        out = dict(obs)
        for f in sensor:
            spec = self.parts[f.target]
            if f.type == "blackout":
                if f.active(t):
                    for key in spec.state:
                        if key in out:
                            out[key] = np.zeros_like(np.asarray(out[key]))
                continue
            for key, sl in state_slices(spec, out):
                v = np.asarray(out[key], dtype=np.float64).reshape(-1).copy()
                if not f.active(t):
                    self._held.get(f.encode(), {}).pop(key, None)
                    continue
                if f.type == "stuck":
                    held = self._held.setdefault(f.encode(), {})
                    if key not in held:
                        prev = np.asarray(self._last_obs.get(key, v), dtype=np.float64).reshape(-1)
                        held[key] = prev[sl].copy()
                    v[sl] = held[key]
                elif f.type == "dropout":
                    v[sl] = np.nan
                elif f.type == "bias_step":
                    if key == next(iter(spec.state)):
                        v[sl] += f.sev
                elif f.type == "noise_burst":
                    v[sl] += self._rng.normal(0.0, f.sev, v[sl].shape)
                out[key] = v.astype(np.float32)
        self._last_obs = out
        return self._derive(out)

    # -- metrics ---------------------------------------------------------------------------
    def _reaction_dims(self, size: int) -> list[int]:
        """Command channels to watch for a reaction: the faulted component's own (by index), never the gripper's."""
        first = min(self.schedule, key=lambda f: f.start)
        spec = self.parts[first.target]
        grip = {i for c in self.parts.values() if c.kind == "gripper" for i in c.action}
        dims = set(spec.action) if spec.kind != "bus" else set()
        if not dims and "[" in spec.name:
            idx = spec.name[spec.name.find("["):]
            dims = {i for c in self.parts.values() if c.name.endswith(idx) and c.kind in ("motor", "joint") for i in c.action}
        if first.type == "gripper_fail":
            return sorted(i for i in dims if i < size) or list(range(size))
        dims = (dims or set(range(size))) - grip
        return sorted(i for i in dims if i < size) or list(range(size))

    def _track_reaction(self, a: np.ndarray, t: int) -> None:
        if self._prev_action is not None and a.size == self._prev_action.size:
            dims = self._dims if getattr(self, "_dims", None) is not None else list(range(a.size))
            change = float(np.linalg.norm((a - self._prev_action)[dims]))
            if self._onset is None:
                self._changes.append(change)
            elif self._react is None and t > self._onset:
                base = np.array(self._changes[-50:] or [0.0])
                if change > max(0.05, float(base.mean() + 3 * base.std())):
                    self._react = t - self._onset
        self._prev_action = a.copy()
        if self._onset is None and any(f.start == t for f in self.schedule):
            self._onset = t
        self._started = sum(1 for f in self.schedule if f.start <= t)

    def _track_error(self, info: dict[str, Any], reward: float) -> None:
        sm = info.get("step_metrics") or {}
        err = float(sm.get("ee_error", -reward if reward is not None and math.isfinite(reward) else math.nan))
        self._err.append(err)
        if self._onset is not None and self._err_onset is None:
            self._err_onset = self._err[min(self._onset, len(self._err) - 1)]
            self._unsafe_base = {k: float(sm.get(k, 0.0)) for k in ("joint_limit_hits", "torque_saturation_steps")}
        self._last_sm = sm

    def _finish(self, info: dict[str, Any], obs: Obs) -> dict[str, Any]:
        info = dict(info)
        m = dict(info.get("metrics") or {})
        success = bool(info.get("success", False))
        reason = info.get("failure_reason")
        started = self._onset is not None
        unsafe = False
        if started:
            if reason in UNSAFE_REASONS:
                unsafe = True
            sm = getattr(self, "_last_sm", {}) or {}
            base = self._unsafe_base or {}
            if float(sm.get("joint_limit_hits", 0.0)) > base.get("joint_limit_hits", 0.0):
                unsafe = True
        saturation = (float((getattr(self, "_last_sm", {}) or {}).get("torque_saturation_steps", 0.0))
                      - (self._unsafe_base or {}).get("torque_saturation_steps", 0.0)) if started else math.nan
        at_rest = _at_rest(obs, m)
        controlled = (not success) and (not unsafe) and reason not in UNSAFE_REASONS and at_rest
        recovery = math.nan
        transient = [f for f in self.schedule if f.duration is not None and f.end <= self.t]
        if started and transient and self._err_onset is not None:
            end = int(max(f.end for f in transient))
            after = self._err[end:]
            hit = next((i for i, e in enumerate(after) if e <= self._err_onset), None)
            recovery = float(hit) if hit is not None else math.nan
        errs = np.array(self._err[self._onset:] if started else [], dtype=float)
        m.update({
            "fault.scheduled": float(len(self.schedule)),
            "fault.started": float(self._started),
            "fault.onset_step": float(self._onset) if started else math.nan,
            "fault.unsafe": float(unsafe),
            "fault.controlled_stop": float(controlled),
            "fault.safe_outcome": float((success or controlled) and not unsafe),
            "fault.reaction_steps": float(self._react) if self._react is not None else math.nan,
            "fault.recovery_steps": recovery,
            "fault.saturation_steps": saturation,
            "fault.error_increase": float(np.nanmax(errs) - self._err_onset) if errs.size and self._err_onset is not None else math.nan,
        })
        info["metrics"] = m
        return info


def _at_rest(obs: Obs, metrics: dict[str, float]) -> bool:
    if "final_joint_speed" in metrics:
        return float(metrics["final_joint_speed"]) < 0.05
    speeds = [np.abs(np.asarray(v, dtype=float)) for k, v in obs.items() if k.startswith("state.") and "vel" in k]
    vals = np.concatenate([s.reshape(-1) for s in speeds]) if speeds else np.zeros(0)
    vals = vals[np.isfinite(vals)]
    return bool(vals.size) and float(vals.max()) < 0.05
