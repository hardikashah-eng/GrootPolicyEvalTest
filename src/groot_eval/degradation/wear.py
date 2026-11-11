"""Wear and tear: run the same scenarios on a brand-new robot, a 1-week-old one, a 1-year-old one.

Age is modelled as a **population**, not a point: a "1-year-old robot" is a
spread of units, because wear depends on duty cycle and load. A profile file
says how each effect grows with operating hours, and every scenario gets its
own unit, drawn from the scenario seed. The same unit is used at every age, so
"new vs year_1" is a paired comparison of the same scenarios.

Profile file (YAML)::

    duty_hours_per_day: 8
    eol_hours: 10000                 # age at which effects reach ``at_eol``
    profiles:
      new:    {age_hours: 0}
      week_1: {age_hours: 56}
      year_1: {age_hours: 2920}
    effects:                         # <component kind>.<key>
      motor.gain_scale:     {curve: linear,  start: 1.0, at_eol: 0.85, spread: 0.03}
      joint.backlash_rad:   {curve: bathtub, start: 0.0, at_eol: 0.006, spread: 0.4}
      encoder.bias_rad:     {curve: linear,  start: 0.0, at_eol: 0.01, signed: true}
    load_factor: {"joint[0]": 1.6, default: 1.0}   # shoulders wear faster

The wear values become scenario parameters named ``wear.<component>.<key>``
(``wear.joint[1].backlash_rad``), plus ``wear.profile`` and
``wear.age_hours``. So they are part of the scenario id, saved by
``--export-suite``, replayed exactly by ``regress``, and ranked by failure
attribution like any other parameter. DOF loss is the same mechanism:
``wear.joint[2].dof_mode: locked | limp | limited``.

Effects and where they act (physics when the env offers the lever, else the
command or observation boundary):

=====================================  ==========================================
``motor.gain_scale``                   physics, else action x scale
``motor.torque_limit_scale``           physics, else action clipped
``motor.thermal_derate`` / ``thermal_tau_s``  command boundary, stateful: the motor heats with load
                                       (first-order, time constant ``thermal_tau_s``) and loses up to
                                       ``thermal_derate`` of its torque limit when hot, so long episodes
                                       get weaker towards the end
``joint.friction_scale`` / ``damping_scale`` / ``backlash_rad`` / ``range_scale``   physics only
``joint.stiction``                     physics, else a command dead zone
``joint.dof_mode``                     physics (lock / limp motor / 30% range), else command zeroed
``gripper.force_scale`` / ``close_rate_scale``      physics only
``encoder|sensor.bias_rad``            observation: constant offset (first key)
``encoder|sensor.noise``               observation: Gaussian noise
``encoder|sensor.quant``               observation: quantization step (first key)
``encoder|sensor.dropout_p``           observation: NaN with this probability per step
``encoder|sensor.scale``               observation: gain error, reading x scale (first key)
``encoder|sensor.bias_walk``           observation: bias random walk, std per sqrt(second) (IMU-style drift)
``encoder|sensor.saturation``          observation: reading clipped to +-saturation (F/T sensors)
``camera.blur_px`` / ``exposure`` / ``dead_px_frac`` / ``shift_px``   observation: image effects
                                       (``shift_px``: extrinsic drift as a horizontal image shift)
``bus.latency_steps`` / ``bus.drop_p``  command delayed / previous command repeated
=====================================  ==========================================

The numbers in the shipped profiles are **placeholders**. Calibrate them from
durability rigs or fleet logs; ``groot-eval replay --config`` (system
identification) fits gains, friction and backlash from an old robot's trace.
"""

from __future__ import annotations

import math
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.degradation.components import (EnvWrapper, discover, env_action_dim, split_params, state_slices,
                                               try_physics)
from groot_eval.envs.base import ComponentSpec, Obs
from groot_eval.types import Scenario

PREFIX = "wear"
DOF_MODES = ("locked", "limp", "limited")
SENSOR_KEYS = ("bias_rad", "noise", "quant", "dropout_p", "scale", "bias_walk", "saturation")
CURVES = ("linear", "bathtub", "sqrt", "step")


@dataclass(frozen=True)
class Effect:
    kind: str  # component kind: motor, joint, encoder, sensor, camera, gripper, bus
    key: str
    curve: str = "linear"
    start: float = 0.0
    at_eol: float = 0.0
    spread: float = 0.0  # log-normal sigma on the change from ``start``, per unit
    signed: bool = False  # random sign per unit (biases)
    integer: bool = False

    def value(self, age_hours: float, eol_hours: float, load: float, unit_factor: float, sign: float) -> float:
        x = max(0.0, age_hours / eol_hours) * load
        if self.curve == "linear":
            y = x
        elif self.curve == "sqrt":  # fast early wear that levels off
            y = math.sqrt(x)
        elif self.curve == "bathtub":  # small wear-in, slow middle, accelerating wear-out
            y = 0.1 * (1 - math.exp(-x / 0.02)) + 0.5 * x + 0.4 * x ** 3
        elif self.curve == "step":
            y = 1.0 if x >= 1.0 else 0.0
        else:
            raise ValueError(f"unknown curve {self.curve}")
        y = min(y, 2.0)
        v = self.start + (self.at_eol - self.start) * y * unit_factor * (sign if self.signed else 1.0)
        return float(round(v)) if self.integer else float(v)


@dataclass
class WearConfig:
    profiles: dict[str, float]  # name -> age in hours
    effects: list[Effect]
    eol_hours: float = 10000.0
    load_factor: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WearConfig":
        profiles = {}
        for name, spec in (data.get("profiles") or {}).items():
            if isinstance(spec, dict):
                if "age_hours" in spec:
                    profiles[name] = float(spec["age_hours"])
                elif "age_days" in spec:
                    profiles[name] = float(spec["age_days"]) * float(data.get("duty_hours_per_day", 8))
                else:
                    raise ValueError(f"profile {name}: give age_hours or age_days")
            else:
                profiles[name] = float(spec)
        if not profiles:
            raise ValueError("wear config has no profiles")
        effects = []
        for full, spec in (data.get("effects") or {}).items():
            kind, _, key = full.partition(".")
            if not key:
                raise ValueError(f"effect {full!r}: expected <kind>.<key>")
            spec = dict(spec or {})
            curve = str(spec.pop("curve", "linear"))
            if curve not in CURVES:
                raise ValueError(f"effect {full}: curve must be one of {CURVES}")
            effects.append(Effect(kind, key, curve, float(spec.pop("start", 1.0 if key.endswith("_scale") else 0.0)),
                                  float(spec.pop("at_eol")), float(spec.pop("spread", 0.0)),
                                  bool(spec.pop("signed", False)), bool(spec.pop("integer", False))))
            if spec:
                raise ValueError(f"effect {full}: unknown fields {sorted(spec)}")
        eol = float(data.get("eol_hours") or max(profiles.values()) or 1.0)
        return cls(profiles, effects, eol, {str(k): float(v) for k, v in (data.get("load_factor") or {}).items()})

    @classmethod
    def load(cls, path: str | Path) -> "WearConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def ordered_profiles(self) -> list[str]:
        return sorted(self.profiles, key=lambda p: self.profiles[p])

    def sample(self, profile: str, components: dict[str, ComponentSpec], unit_seed: int) -> dict[str, Any]:
        """Wear parameters of one unit at one age. The same ``unit_seed`` gives the same unit at every age."""
        if profile not in self.profiles:
            raise KeyError(f"unknown profile {profile!r}; have {sorted(self.profiles)}")
        age = self.profiles[profile]
        out: dict[str, Any] = {f"{PREFIX}.profile": profile, f"{PREFIX}.age_hours": age}
        for eff in self.effects:
            for name in sorted(n for n, c in components.items() if c.kind == eff.kind):
                rng = np.random.default_rng(_unit_seed(unit_seed, name, eff.key))
                factor = float(np.exp(rng.normal(0.0, eff.spread))) if eff.spread else 1.0
                sign = float(rng.choice([-1.0, 1.0]))
                load = self.load_factor.get(name, self.load_factor.get("default", 1.0))
                out[f"{PREFIX}.{name}.{eff.key}"] = eff.value(age, self.eol_hours, load, factor, sign)
        return out

    def typical_unit(self, age_hours: float, components: dict[str, ComponentSpec]) -> dict[str, Any]:
        """The median unit at ``age_hours``: no unit-to-unit spread, and signed effects (biases) at zero,
        since their sign is unknown. What a fleet-average robot of that age looks like."""
        out: dict[str, Any] = {f"{PREFIX}.age_hours": float(age_hours)}
        for eff in self.effects:
            for name in sorted(n for n, c in components.items() if c.kind == eff.kind):
                load = self.load_factor.get(name, self.load_factor.get("default", 1.0))
                v = eff.value(age_hours, self.eol_hours, load, 1.0, 0.0 if eff.signed else 1.0)
                out[f"{PREFIX}.{name}.{eff.key}"] = v
        return out


def _unit_seed(seed: int, name: str, key: str) -> int:
    return (int(seed) * 1_000_003 + zlib.crc32(f"{name}|{key}".encode())) % (2**32)


def wear_params(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{"wear.joint[1].backlash_rad": 0.004}`` -> ``{"joint[1]": {"backlash_rad": 0.004}}``."""
    out: dict[str, dict[str, Any]] = {}
    for k, v in params.items():
        if not k.startswith(PREFIX + "."):
            continue
        rest = k[len(PREFIX) + 1:]
        name, _, key = rest.rpartition(".")
        if name:
            out.setdefault(name, {})[key] = v
    return out


class WearEnv(EnvWrapper):
    """Apply a scenario's ``wear.*`` parameters to any env.

    Unknown components or effects without a physics lever or a boundary
    fallback are collected in ``unsupported`` instead of failing silently.
    """

    def __init__(self, env: Any, action_dim: int | None = None) -> None:
        super().__init__(env)
        self._action_dim = action_dim
        self.unsupported: set[str] = set()
        self._reset_state({})

    def _reset_state(self, wear: dict[str, dict[str, Any]]) -> None:
        self.wear = wear
        self.parts: dict[str, ComponentSpec] = {}
        self._gain = None
        self._limit = None
        self._dead = None
        self._zero: set[int] = set()
        self._obs_fx: list[tuple[ComponentSpec, dict[str, Any]]] = []
        self._cam_fx: list[tuple[ComponentSpec, dict[str, Any]]] = []
        self._latency = 0
        self._drop_p = 0.0
        self._queue: deque[np.ndarray] = deque()
        self._last_sent: np.ndarray | None = None
        self._rng = np.random.default_rng(0)
        self._thermal: dict[int, tuple[float, float]] = {}  # action index -> (derate, tau_s)
        self._heat: dict[int, float] = {}
        self._walk: dict[tuple[str, str], np.ndarray] = {}

    def reset(self, scenario: Scenario) -> Obs:
        mine, rest = split_params(scenario.params, PREFIX)
        obs = self.env.reset(Scenario(seed=scenario.seed, params=rest, task=scenario.task, name=scenario.name))
        self._reset_state(wear_params(mine))
        self._rng = np.random.default_rng((scenario.seed * 7919 + 17) % 2**32)
        dim = self._action_dim or env_action_dim(self.env) or _guess_action_dim(self.env)
        self.parts = discover(self.env, obs, dim)
        if dim:
            self._gain, self._limit, self._dead = np.ones(dim), np.ones(dim), np.zeros(dim)
        for name, effects in self.wear.items():
            spec = self.parts.get(name)
            if spec is None:
                self.unsupported.add(f"{name} (no such component)")
                continue
            self._apply(spec, effects)
        return self._observe(obs)

    def _apply(self, spec: ComponentSpec, fx: dict[str, Any]) -> None:
        obs_fx, cam_fx, thermal = {}, {}, {}
        for key, value in fx.items():
            if key in ("thermal_derate", "thermal_tau_s") and spec.action:
                thermal[key] = float(value)
            elif key == "dof_mode":
                self._dof(spec, str(value))
            elif spec.kind == "bus":
                if key == "latency_steps":
                    self._latency = int(value)
                elif key == "drop_p":
                    self._drop_p = float(value)
                else:
                    self.unsupported.add(f"{spec.name}.{key}")
            elif spec.kind in ("encoder", "sensor") and key in SENSOR_KEYS:
                obs_fx[key] = float(value)
            elif spec.kind == "camera" and key in ("blur_px", "exposure", "dead_px_frac", "shift_px"):
                cam_fx[key] = float(value)
            elif try_physics(self.env, spec.name, key, value):
                continue
            elif key == "gain_scale" and spec.action and self._gain is not None:
                self._gain[list(spec.action)] *= float(value)
            elif key == "torque_limit_scale" and spec.action and self._limit is not None:
                self._limit[list(spec.action)] = np.minimum(self._limit[list(spec.action)], float(value))
            elif key == "stiction" and spec.action and self._dead is not None:
                self._dead[list(spec.action)] = float(value)
            else:
                self.unsupported.add(f"{spec.name}.{key}")
        if thermal.get("thermal_derate"):
            for i in spec.action:
                self._thermal[i] = (thermal["thermal_derate"], max(thermal.get("thermal_tau_s", 60.0), 1e-3))
                self._heat[i] = 0.0
        if obs_fx:
            self._obs_fx.append((spec, obs_fx))
        if cam_fx:
            self._cam_fx.append((spec, cam_fx))

    def _dof(self, spec: ComponentSpec, mode: str) -> None:
        if mode in ("", "none", "ok"):
            return
        if mode not in DOF_MODES:
            raise ValueError(f"{spec.name}: dof_mode must be one of {DOF_MODES}")
        idx = spec.name[spec.name.find("["):] if "[" in spec.name else ""
        motor = f"motor{idx}" if idx else spec.name
        done = False
        if mode == "locked":
            done = try_physics(self.env, spec.name, "locked", True)
        elif mode == "limp":
            done = try_physics(self.env, motor, "enabled", False)
        elif mode == "limited":
            done = try_physics(self.env, spec.name, "range_scale", 0.3)
        if not done:  # boundary fallback: the joint ignores its command
            if spec.action:
                self._zero.update(spec.action)
            else:
                self.unsupported.add(f"{spec.name}.dof_mode={mode}")

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=float).reshape(-1).copy()
        if self._gain is not None and self._gain.size == a.size:
            a = a * self._gain
            a = np.clip(a, -self._limit, self._limit)
            a = np.where(np.abs(a) < self._dead, 0.0, a)
        if self._thermal:
            dt = float(getattr(self.env, "dt", None) or 0.05)
            for i, (derate, tau) in self._thermal.items():
                if i >= a.size:
                    continue
                # load heats the motor; a hot motor is derated (normalized temperature 0 cold .. 1 hot)
                self._heat[i] += min(1.0, dt / tau) * (min(a[i] * a[i], 1.0) - self._heat[i])
                lim = (self._limit[i] if self._limit is not None and i < self._limit.size else 1.0) * (1 - derate * self._heat[i])
                a[i] = float(np.clip(a[i], -lim, lim))
        for i in self._zero:
            if i < a.size:
                a[i] = 0.0
        if self._latency:
            self._queue.append(a)
            a = self._queue.popleft() if len(self._queue) > self._latency else np.zeros_like(a)
        if self._drop_p and self._last_sent is not None and self._rng.random() < self._drop_p:
            a = self._last_sent
        self._last_sent = a
        obs, reward, term, trunc, info = self.env.step(a)
        return self._observe(obs), reward, term, trunc, info

    def _observe(self, obs: Obs) -> Obs:
        if not self._obs_fx and not self._cam_fx:
            return obs
        obs = dict(obs)
        for spec, fx in self._obs_fx:
            for i, (key, sl) in enumerate(state_slices(spec, obs)):
                v = np.asarray(obs[key], dtype=np.float64).reshape(-1).copy()
                part = v[sl]
                if i == 0 and fx.get("scale"):
                    part = part * fx["scale"]
                if i == 0 and fx.get("bias_rad"):
                    part = part + fx["bias_rad"]
                if i == 0 and fx.get("bias_walk"):
                    dt = float(getattr(self.env, "dt", None) or 0.05)
                    w = self._walk.setdefault((spec.name, key), np.zeros(part.shape))
                    w += self._rng.normal(0.0, fx["bias_walk"] * math.sqrt(dt), part.shape)
                    part = part + w
                if fx.get("saturation"):
                    part = np.clip(part, -fx["saturation"], fx["saturation"])
                if fx.get("noise"):
                    part = part + self._rng.normal(0.0, fx["noise"], part.shape)
                if i == 0 and fx.get("quant"):
                    part = np.round(part / fx["quant"]) * fx["quant"]
                if fx.get("dropout_p") and self._rng.random() < fx["dropout_p"]:
                    part = np.full(part.shape, np.nan)
                v[sl] = part
                obs[key] = v.astype(np.float32)
        for spec, fx in self._cam_fx:
            for key in spec.state:
                if key in obs:
                    obs[key] = degrade_image(np.asarray(obs[key]), fx, self._rng)
        return self._derive(obs)


def degrade_image(img: np.ndarray, fx: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float64)
    blur = int(round(fx.get("blur_px", 0.0)))
    if blur > 0 and out.ndim >= 2:  # box blur along both image axes
        k = 2 * blur + 1
        for axis in (0, 1):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (blur, blur)
            padded = np.pad(out, pad, mode="edge")
            c = np.cumsum(padded, axis=axis)
            c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
            n = out.shape[axis]
            out = (np.take(c, range(k, k + n), axis=axis) - np.take(c, range(0, n), axis=axis)) / k
    if fx.get("exposure"):
        out = out * (2.0 ** fx["exposure"])
    shift = int(round(fx.get("shift_px", 0.0)))
    if shift and out.ndim >= 2:  # camera mount drift: the scene moves sideways in the frame
        out = np.roll(out, shift, axis=1)
        if shift > 0:
            out[:, :shift] = out[:, shift:shift + 1]
        else:
            out[:, shift:] = out[:, shift - 1:shift]
    if fx.get("dead_px_frac"):
        mask = rng.random(out.shape[:2]) < fx["dead_px_frac"]
        out[mask] = 0.0
    top = 1.0 if img.dtype.kind == "f" and float(np.nanmax(img)) <= 1.0 else 255.0
    return np.clip(out, 0, top).astype(img.dtype)


def _guess_action_dim(env: Any) -> int | None:
    if hasattr(env, "components"):
        acts = [i for c in env.components().values() for i in c.action]
        return max(acts) + 1 if acts else None
    return None


def aged_scenarios(base: list[Scenario], cfg: WearConfig, profile: str, components: dict[str, ComponentSpec],
                   extra: dict[str, Any] | None = None) -> list[Scenario]:
    """The same scenarios on units of a given age. Unit = scenario seed, so ages pair up."""
    out = []
    for s in base:
        params = {**s.params, **cfg.sample(profile, components, s.seed), **(extra or {})}
        out.append(Scenario(seed=s.seed, params=params, task=s.task, name=s.name))
    return out
