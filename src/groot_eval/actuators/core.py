"""Batched, wear- and fault-aware actuator model: ``num_envs x num_joints`` joints per call.

One call of :meth:`WearAwareActuatorCore.compute` does, for every environment
and joint:

1. **Bus:** a ``comm_stall`` fault holds the previous command; a
   ``latency_spike`` fault applies the command from ``severity`` steps ago.
2. **Motor model:** PD on the position and velocity targets plus feed-forward
   effort, with the joint's backlash as a dead band on the position error, or
   a learned :class:`~groot_eval.actuators.motor_models.MLPMotorModel`.
3. **Wear:** motor gain loss, torque-limit loss, thermal derating (the motor
   heats with load and loses up to ``thermal_derate`` of its limit), extra
   Coulomb and viscous friction, and stiction (a joint at rest stays put until
   the drive exceeds ``stiction x effort limit``).
4. **Limits:** optional DC-motor torque-speed curve, ``power_sag``, effort limit.
5. **Actuator faults:** ``motor_off`` (no torque), ``motor_runaway`` (torque
   forced to ``severity x limit``), ``joint_lock`` (a stiff brake against any
   motion; an explicit actuator cannot pin a joint exactly, so set the
   simulator's joint drive if you need a hard lock).

:meth:`observe` applies the sensor side: encoder scale, bias, slow bias walk
and noise from wear, and ``stuck`` / ``dropout`` / ``bias_step`` /
``noise_burst`` faults.

Wear parameters come from a :class:`WearTable` (one unit per environment, at
any age, from a ``WearConfig`` profile or a calibrated robot profile); fault
windows from a :class:`FaultTable`. Every array op goes through the array-API
namespace of the inputs, so the same object runs on NumPy and on torch.
Random draws (sensor noise) use NumPy and are copied to the input's device.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from array_api_compat import array_namespace, device

from groot_eval.actuators.motor_models import DCMotorSaturation, MLPMotorModel, _like
from groot_eval.degradation.faults import FaultConfig, Fault, parse_schedule
from groot_eval.degradation.wear import WearConfig, _unit_seed
from groot_eval.envs.base import ComponentSpec

WEAR_KEYS = {  # key -> (component kind, effect key, default)
    "gain": ("motor", "gain_scale", 1.0),
    "torque_limit": ("motor", "torque_limit_scale", 1.0),
    "thermal_derate": ("motor", "thermal_derate", 0.0),
    "thermal_tau_s": ("motor", "thermal_tau_s", 60.0),
    "friction": ("joint", "friction_scale", 1.0),
    "damping": ("joint", "damping_scale", 1.0),
    "backlash": ("joint", "backlash_rad", 0.0),
    "stiction": ("joint", "stiction", 0.0),
    "enc_bias": ("encoder", "bias_rad", 0.0),
    "enc_scale": ("encoder", "scale", 1.0),
    "enc_noise": ("encoder", "noise", 0.0),
    "enc_bias_walk": ("encoder", "bias_walk", 0.0),
}
JOINT_FAULTS = ("motor_off", "motor_runaway", "joint_lock")
SENSOR_FAULTS = ("stuck", "dropout", "bias_step", "noise_burst")
BUS_FAULTS = ("comm_stall", "latency_spike", "power_sag")


def joint_components(num_joints: int) -> dict[str, ComponentSpec]:
    """Component names the wear and fault configs refer to: ``motor[j]``, ``joint[j]``, ``encoder[j]``, ``bus``."""
    out: dict[str, ComponentSpec] = {}
    for j in range(num_joints):
        out[f"motor[{j}]"] = ComponentSpec(f"motor[{j}]", "motor", action=(j,))
        out[f"joint[{j}]"] = ComponentSpec(f"joint[{j}]", "joint", action=(j,))
        out[f"encoder[{j}]"] = ComponentSpec(f"encoder[{j}]", "encoder", action=(j,))
    out["bus"] = ComponentSpec("bus", "bus", action=tuple(range(num_joints)))
    return out


def wear_at_age(cfg: WearConfig, num_joints: int, age_hours: float, unit_seed: int | None) -> dict[str, np.ndarray]:
    """One unit's wear, per joint, at ``age_hours``. ``unit_seed=None`` = the fleet-median unit (no spread, zero biases).

    Same draw as :meth:`WearConfig.sample`, so a unit seed is the same unit at every age.
    """
    out = {k: np.full(num_joints, d, dtype=np.float64) for k, (_, _, d) in WEAR_KEYS.items()}
    by_effect = {(kind, key): name for name, (kind, key, _) in WEAR_KEYS.items()}
    for eff in cfg.effects:
        slot = by_effect.get((eff.kind, eff.key))
        if slot is None:
            continue
        for j in range(num_joints):
            comp = f"{eff.kind}[{j}]"
            load = cfg.load_factor.get(comp, cfg.load_factor.get("default", 1.0))
            if unit_seed is None:
                factor, sign = 1.0, (0.0 if eff.signed else 1.0)
            else:
                rng = np.random.default_rng(_unit_seed(unit_seed, comp, eff.key))
                factor = float(np.exp(rng.normal(0.0, eff.spread))) if eff.spread else 1.0
                sign = float(rng.choice([-1.0, 1.0]))
            out[slot][j] = eff.value(age_hours, cfg.eol_hours, load, factor, sign)
    return out


@dataclass
class WearTable:
    """Per-environment, per-joint wear parameters, shape ``(num_envs, num_joints)`` each (NumPy, float64)."""

    values: dict[str, np.ndarray]
    ages_hours: np.ndarray
    unit_seeds: np.ndarray
    source: str = ""

    @property
    def num_envs(self) -> int:
        return int(self.ages_hours.shape[0])

    @classmethod
    def new(cls, num_envs: int, num_joints: int) -> "WearTable":
        vals = {k: np.full((num_envs, num_joints), d, dtype=np.float64) for k, (_, _, d) in WEAR_KEYS.items()}
        return cls(vals, np.zeros(num_envs), np.arange(num_envs), "new")

    @classmethod
    def from_config(cls, cfg: WearConfig, num_joints: int, ages_hours: Any, unit_seeds: Any = None,
                    typical: bool = False) -> "WearTable":
        """One unit per environment: ``ages_hours[e]`` with unit ``unit_seeds[e]`` (default: the env index)."""
        ages = np.asarray(ages_hours, dtype=np.float64).reshape(-1)
        seeds = np.arange(ages.size) if unit_seeds is None else np.asarray(unit_seeds, dtype=np.int64).reshape(-1)
        if seeds.size != ages.size:
            raise ValueError("unit_seeds and ages_hours must have one entry per environment")
        rows = [wear_at_age(cfg, num_joints, float(a), None if typical else int(s)) for a, s in zip(ages, seeds)]
        vals = {k: np.stack([r[k] for r in rows]) for k in WEAR_KEYS}
        return cls(vals, ages, seeds, "typical" if typical else "sampled")

    @classmethod
    def fleet(cls, cfg: WearConfig, num_joints: int, ages_hours: list[float], envs_per_age: int,
              first_seed: int = 0) -> "WearTable":
        """``envs_per_age`` units at each age; unit ``first_seed + i`` appears at every age (paired comparison)."""
        ages = np.repeat(np.asarray(ages_hours, dtype=np.float64), envs_per_age)
        seeds = np.tile(np.arange(first_seed, first_seed + envs_per_age), len(ages_hours))
        return cls.from_config(cfg, num_joints, ages, seeds)

    @classmethod
    def from_robot_profile(cls, wear: dict[str, Any], num_envs: int, num_joints: int) -> "WearTable":
        """Every environment gets the same measured unit (``wear`` = per-key list or scalar, e.g. from a robot profile)."""
        table = cls.new(num_envs, num_joints)
        for k, v in wear.items():
            if k not in WEAR_KEYS:
                raise ValueError(f"unknown wear key {k!r}; known: {sorted(WEAR_KEYS)}")
            arr = np.broadcast_to(np.asarray(v, dtype=np.float64), (num_joints,))
            table.values[k] = np.tile(arr, (num_envs, 1))
        table.source = "robot_profile"
        return table


@dataclass
class FaultTable:
    """Fault windows per environment, padded to ``(num_envs, max_faults)``. Steps count control steps."""

    kind: np.ndarray  # (E, K) index into ALL_FAULTS, -1 = padding
    joint: np.ndarray  # (E, K) joint index, -1 = bus-wide
    start: np.ndarray  # (E, K) float, inf for padding
    end: np.ndarray  # (E, K) float, inf = permanent
    severity: np.ndarray  # (E, K)
    schedules: list[list[Fault]] = field(default_factory=list)

    ALL = JOINT_FAULTS + SENSOR_FAULTS + BUS_FAULTS

    @classmethod
    def none(cls, num_envs: int) -> "FaultTable":
        return cls.from_schedules([[] for _ in range(num_envs)], 1)

    @classmethod
    def from_schedules(cls, schedules: Sequence[Sequence[Fault] | str], num_joints: int) -> "FaultTable":
        parsed: list[list[Fault]] = [parse_schedule(s) if isinstance(s, str) else list(s) for s in schedules]
        k = max(1, max((len(s) for s in parsed), default=0))
        e = len(parsed)
        kind = np.full((e, k), -1, np.int64)
        joint = np.full((e, k), -1, np.int64)
        start = np.full((e, k), np.inf)
        end = np.full((e, k), np.inf)
        sev = np.zeros((e, k))
        for i, sched in enumerate(parsed):
            for n, f in enumerate(sched):
                if f.type not in cls.ALL:
                    continue  # cameras, grippers: handled by the env, not the joint actuators
                j = _joint_index(f.target)
                if j is not None and j >= num_joints:
                    raise ValueError(f"fault {f.encode()} targets joint {j}; only {num_joints} joints")
                kind[i, n] = cls.ALL.index(f.type)
                joint[i, n] = -1 if j is None else j
                start[i, n] = f.start
                end[i, n] = f.end
                sev[i, n] = f.sev
        return cls(kind, joint, start, end, sev, parsed)

    @classmethod
    def sample(cls, cfg: FaultConfig, num_envs: int, num_joints: int, horizon_steps: int, dt: float,
               seeds: Any = None) -> "FaultTable":
        """Draw one schedule per environment from ``cfg`` (same seeding as ``faulted_scenarios``)."""
        comps = joint_components(num_joints)
        seeds = np.arange(num_envs) if seeds is None else np.asarray(seeds).reshape(-1)
        scheds = [cfg.sample(np.random.default_rng((int(s) * 104729 + 7) % 2**32), comps, horizon_steps, dt) for s in seeds]
        return cls.from_schedules(scheds, num_joints)

    def masks(self, t: np.ndarray, num_joints: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Active faults at per-env step ``t``: name -> (mask (E, J) bool, severity (E, J))."""
        active = (self.start <= t[:, None]) & (t[:, None] < self.end)
        out = {}
        jidx = np.arange(num_joints)
        for code, name in enumerate(self.ALL):
            hit = active & (self.kind == code)
            if not hit.any():
                continue
            on_joint = (self.joint[..., None] == jidx) | (self.joint[..., None] == -1)  # (E, K, J)
            m = hit[..., None] & on_joint
            mask = m.any(axis=1)
            sev = np.where(m, self.severity[..., None], -np.inf).max(axis=1)
            out[name] = (mask, np.where(mask, sev, 0.0))
        return out


def _joint_index(target: str) -> int | None:
    if "[" in target and target.endswith("]"):
        return int(target[target.index("[") + 1:-1])
    return None


@dataclass
class ActuatorParams:
    """Nominal (new-robot) drive parameters, per joint (scalars broadcast)."""

    effort_limit: Any = 100.0  # N·m
    stiffness: Any = 100.0  # PD P gain, N·m/rad
    damping: Any = 5.0  # PD D gain, N·m·s/rad
    coulomb_friction: Any = 0.0  # N·m, scaled by wear "friction"
    viscous_friction: Any = 0.0  # N·m·s/rad, scaled by wear "damping"
    lock_damping: float = 1.0e4  # brake gain used for joint_lock
    rest_velocity: float = 1.0e-3  # rad/s below which stiction holds a joint
    dt: float = 0.005  # control period, s (thermal model and bias walk)


class WearAwareActuatorCore:
    """Compute applied efforts for ``num_envs x num_joints`` joints with per-env wear and faults.

    Typical use (any array namespace; NumPy shown)::

        cfg = WearConfig.load("configs/wear/arm3_profiles_extended.yaml")
        wear = WearTable.fleet(cfg, num_joints=12, ages_hours=[0, 2920, 8760], envs_per_age=256)
        act = WearAwareActuatorCore(ActuatorParams(effort_limit=80.0), wear, FaultTable.none(wear.num_envs))
        for step in range(T):
            tau = act.compute(q_target, qd_target, tau_ff, q, qd)   # (768, 12)
            q_meas = act.observe(q)
    """

    def __init__(self, params: ActuatorParams, wear: WearTable, faults: FaultTable | None = None,
                 motor: MLPMotorModel | None = None, saturation: DCMotorSaturation | None = None,
                 noise_seed: int = 0) -> None:
        self.p = params
        self.wear = wear
        self.faults = faults or FaultTable.none(wear.num_envs)
        if self.faults.kind.shape[0] != wear.num_envs:
            raise ValueError("fault table and wear table must have the same number of environments")
        self.motor = motor
        self.saturation = saturation
        self.num_envs = wear.num_envs
        self.num_joints = next(iter(wear.values.values())).shape[1]
        self._rng = np.random.default_rng(noise_seed)
        lat = self.faults.kind == FaultTable.ALL.index("latency_spike")
        self._max_lag = int(self.faults.severity[lat].max(initial=0.0))
        self._cache: dict[tuple[str, Any], Any] = {}
        self.reset()

    # -- state ------------------------------------------------------------------------
    def reset(self, env_ids: Any = None) -> None:
        """Zero the step counter, motor temperature, command history and sensor state (all envs, or ``env_ids``)."""
        e, j = self.num_envs, self.num_joints
        if env_ids is None or not hasattr(self, "step_count"):
            self.step_count = np.zeros(e, np.int64)
            self.heat = np.zeros((e, j))
            self.walk = np.zeros((e, j))
            self._cmd_hist: list[np.ndarray] = []
            self._err_hist: list[Any] = []
            self._vel_hist: list[Any] = []
            self._last_reading: np.ndarray | None = None
            self._prev_cmd: Any = None
            return
        ids = np.asarray(env_ids).reshape(-1)
        self.step_count[ids] = 0
        self.heat[ids] = 0.0
        self.walk[ids] = 0.0

    def _w(self, key: str, ref: Any) -> Any:
        """Wear array ``key`` in ``ref``'s namespace/device (cached)."""
        ck = (key, _ns_key(ref))
        if ck not in self._cache:
            self._cache[ck] = _like(self.wear.values[key], ref)
        return self._cache[ck]

    def _p(self, name: str, ref: Any) -> Any:
        ck = ("param:" + name, _ns_key(ref))
        if ck not in self._cache:
            self._cache[ck] = _like(getattr(self.p, name), ref)
        return self._cache[ck]

    # -- actuation --------------------------------------------------------------------
    def compute(self, q_target: Any, qd_target: Any, tau_ff: Any, q: Any, qd: Any) -> Any:
        """Applied effort ``(E, J)`` for this control step. Advances the step counter by one."""
        xp = array_namespace(q)
        zeros = xp.zeros_like(q)
        if qd_target is None:
            qd_target = zeros
        if tau_ff is None:
            tau_ff = zeros
        f = self.faults.masks(self.step_count, self.num_joints)

        # 1. bus: stalled or late commands
        cmd = xp.stack([q_target, qd_target, tau_ff])
        self._cmd_hist.insert(0, cmd)
        del self._cmd_hist[self._max_lag + 2:]
        if "latency_spike" in f:
            mask, sev = f["latency_spike"]
            delayed = cmd
            for lag in sorted({int(s) for s in np.unique(sev[mask])}):
                old = self._cmd_hist[min(lag, len(self._cmd_hist) - 1)]
                sel = _like((mask & (sev.astype(int) == lag)).astype(np.float64), q)
                delayed = sel * old + (1.0 - sel) * delayed
            cmd = delayed
        if "comm_stall" in f and self._prev_cmd is not None:
            sel = _like(f["comm_stall"][0].astype(np.float64), q)
            cmd = sel * self._prev_cmd + (1.0 - sel) * cmd
        self._prev_cmd = cmd
        q_t, qd_t, ff = cmd[0, ...], cmd[1, ...], cmd[2, ...]

        # 2. motor model
        err = q_t - q
        half_play = 0.5 * self._w("backlash", q)
        err = xp.sign(err) * xp.maximum(xp.abs(err) - half_play, zeros)
        if self.motor is not None:
            self._err_hist.insert(0, err)
            self._vel_hist.insert(0, qd)
            del self._err_hist[self.motor.depth:]
            del self._vel_hist[self.motor.depth:]
            pad = self.motor.depth - len(self._err_hist)
            eh = xp.stack(self._err_hist + [self._err_hist[-1]] * pad)
            vh = xp.stack(self._vel_hist + [self._vel_hist[-1]] * pad)
            tau = self.motor(eh, vh) + ff
        else:
            tau = self._p("stiffness", q) * err + self._p("damping", q) * (qd_t - qd) + ff

        # 3. wear: gain, friction, stiction
        tau = tau * self._w("gain", q)
        tau = tau - self._p("viscous_friction", q) * self._w("damping", q) * qd
        tau = tau - self._p("coulomb_friction", q) * self._w("friction", q) * xp.tanh(qd / 0.01)
        limit = self._p("effort_limit", q)
        stick = self._w("stiction", q) * limit
        at_rest = xp.abs(qd) < self.p.rest_velocity
        tau = xp.where(at_rest & (xp.abs(tau) < stick), zeros, tau)

        # 4. limits: torque limit wear, thermal derating, power sag, torque-speed curve
        heat = _like(self.heat, q)
        lim = limit * self._w("torque_limit", q) * (1.0 - self._w("thermal_derate", q) * heat)
        if "power_sag" in f:
            m, s = f["power_sag"]
            lim = lim * _like(np.where(m, s, 1.0), q)
        if self.saturation is not None:
            tau = self.saturation.clip(tau, qd, lim)
        tau = xp.minimum(xp.maximum(tau, -lim), lim)

        # 5. actuator faults
        if "motor_runaway" in f:
            m, s = f["motor_runaway"]
            tau = xp.where(_like(m.astype(np.float64), q) > 0.5, _like(s, q) * lim, tau)
        if "joint_lock" in f:
            m, _ = f["joint_lock"]
            tau = xp.where(_like(m.astype(np.float64), q) > 0.5, -self.p.lock_damping * qd, tau)
        if "motor_off" in f:
            m, _ = f["motor_off"]
            tau = xp.where(_like(m.astype(np.float64), q) > 0.5, zeros, tau)

        # thermal state (NumPy; small): heating ~ current^2 relative to what the worn motor is rated for
        rated = _to_numpy(limit * self._w("torque_limit", q))
        load = np.minimum((_to_numpy(tau) / np.maximum(rated, 1e-9)) ** 2, 1.0)
        tau_th = np.maximum(self.wear.values["thermal_tau_s"], 1e-3)
        self.heat += np.minimum(1.0, self.p.dt / tau_th) * (load - self.heat)
        self.step_count += 1
        return tau

    # -- sensing ----------------------------------------------------------------------
    def observe(self, q: Any) -> Any:
        """Encoder reading of joint positions ``(E, J)``: wear (scale, bias, bias walk, noise) and sensor faults.

        Call once per control step, after :meth:`compute` (fault windows use the same step count).
        """
        xp = array_namespace(q)
        t = np.maximum(self.step_count - 1, 0)
        f = self.faults.masks(t, self.num_joints)
        noise_std = self.wear.values["enc_noise"].copy()
        bias = self.wear.values["enc_bias"].copy()
        walk_std = self.wear.values["enc_bias_walk"]
        if walk_std.any():
            self.walk += self._rng.normal(0.0, 1.0, self.walk.shape) * walk_std * math.sqrt(self.p.dt)
            bias = bias + self.walk
        if "bias_step" in f:
            m, s = f["bias_step"]
            bias = bias + np.where(m, s, 0.0)
        if "noise_burst" in f:
            m, s = f["noise_burst"]
            noise_std = np.where(m, np.sqrt(noise_std ** 2 + s ** 2), noise_std)
        reading = q * self._w("enc_scale", q) + _like(bias, q)
        if noise_std.any():
            reading = reading + _like(self._rng.normal(0.0, 1.0, noise_std.shape) * noise_std, q)
        if "stuck" in f and self._last_reading is not None:
            m, _ = f["stuck"]
            reading = xp.where(_like(m.astype(np.float64), q) > 0.5, _like(self._last_reading, q), reading)
        if "dropout" in f:
            m, _ = f["dropout"]
            reading = xp.where(_like(m.astype(np.float64), q) > 0.5, _like(np.full(m.shape, np.nan), q), reading)
        self._last_reading = _to_numpy(reading)
        return reading

    def telemetry(self) -> dict[str, np.ndarray]:
        """Per-env motor temperature (0 cold .. 1 hot) and step count, for logging."""
        return {"heat": self.heat.copy(), "step": self.step_count.copy()}


def _ns_key(x: Any) -> tuple[str, str]:
    return (type(x).__module__, str(device(x)))


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch
        return x.detach().cpu().numpy()
    try:
        return np.asarray(x)
    except (TypeError, ValueError, RuntimeError):
        return np.from_dlpack(x)
