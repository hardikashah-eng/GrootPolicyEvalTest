"""Walker: a reduced-order humanoid walking twin (sagittal plane), for treadmill and overground tests.

The body is a linear inverted pendulum (LIPM): the centre of mass (CoM) at
constant height ``z0`` accelerates away from the centre of pressure (CoP),
``xdd = w^2 (x - cop)`` with ``w = sqrt(g / z0)``. Two feet alternate as the
stance foot on a fixed step clock:

* the **stance foot rides its ground**: it moves with the belt under it
  (left or right belt on a split-belt treadmill, zero overground);
* the **ankle** places the CoP inside the foot (``action[0]``), so it can only
  push within the foot length;
* the **swing foot** is in the air, moves toward a target placed relative to
  the CoM (``action[1]``) at a limited speed, and lands where it is when the
  step ends;
* **friction** limits the horizontal ground force to ``mu * m * g``. Beyond
  it, the stance foot slips on the belt and the CoM gets only the friction
  force;
* an **incline** adds ``-g sin(theta)`` along the deck, and an external push
  (a gantry pusher, say) adds ``F / m``.

The episode fails with ``fall`` when the CoM leans more than ``max_lean``
metres from the stance foot, and with ``off_belt`` when the robot leaves the
deck (treadmill only). It succeeds after ``duration_s`` seconds of walking.

What is under the feet comes from ``self.ground``: :class:`StaticGround` by
default, or a treadmill (:class:`groot_eval.envs.treadmill.Treadmill`) set by
the treadmill wrapper.

Observations (``state.*``), as the robot would estimate them:

* ``com_rel`` / ``com_vel_rel``: CoM relative to the stance foot, in position
  and velocity (leg odometry);
* ``swing_rel``: swing foot relative to the CoM;
* ``phase`` (fraction of the step) and ``stance`` (0 left, 1 right);
* ``com_pos`` / ``com_vel``: position and velocity in the room (motion capture).

Actions, both in [-1, 1]:

* ``action[0]``: CoP command, scaled to the foot half-length;
* ``action[1]``: swing-foot target relative to the CoM, scaled by ``reach``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from groot_eval.types import Scenario

Obs = dict[str, Any]

DEFAULTS: dict[str, Any] = {
    "z0": 0.9,  # CoM height, m
    "mass": 60.0,  # kg
    "foot_half": 0.10,  # CoP range from the ankle, m
    "reach": 0.6,  # swing target scale, m
    "swing_speed": 3.0,  # max swing foot speed, m/s
    "step_time": 0.40,  # s per step
    "mu": 0.8,  # foot-belt friction
    "max_lean": 0.45,  # CoM-to-stance-foot distance at which the robot falls, m
    "duration_s": 20.0,  # walking time for success
    "start_speed": 0.0,  # initial CoM speed relative to the ground, m/s
    "action_delay": 0,  # control steps between command and actuation
    "ankle_gain": 1.0,  # actuator wear: the CoP the ankle achieves per unit command
    "sensor_noise": 0.0,  # std of noise on the odometry signals
}


class StaticGround:
    """Overground walking: nothing moves under the feet."""

    deck_half: float | None = None  # no deck edges

    def reset(self, scenario: Scenario, robot: Walker) -> None:
        pass

    def velocity(self, side: int) -> float:
        return 0.0

    def friction(self, side: int, mu: float) -> float:
        return mu

    def incline(self) -> float:
        return 0.0

    def push(self) -> float:
        return 0.0

    def update(self, dt: float, robot: Walker) -> None:
        pass

    def on_touchdown(self, side: int, robot: Walker) -> None:
        pass

    def observe(self) -> Obs:
        return {}


@dataclass
class _State:
    x: float = 0.0  # CoM, room frame
    xd: float = 0.0
    p_st: float = 0.0  # stance ankle
    p_sw: float = 0.0  # swing foot
    stance: int = 0  # 0 left, 1 right
    t_phase: float = 0.0
    slip_vel: float = 0.0  # stance foot velocity relative to its belt
    slipped: float = 0.0  # total slip distance this episode
    t: float = 0.0
    steps: int = 0
    step_starts: list[float] = field(default_factory=list)  # touchdown positions relative to the CoM, per side
    lean_max: float = 0.0
    deck_max: float = 0.0
    cop_sat: int = 0
    ticks: int = 0


class Walker:
    action_dim = 2
    substeps = 10

    def __init__(self, dt: float = 0.02, max_steps: int | None = None, **params: Any) -> None:
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Walker does not know parameters: {sorted(unknown)}")
        self.dt = float(dt)
        self.base = {**DEFAULTS, **params}
        self.max_steps = int(max_steps if max_steps is not None else math.ceil(self.base["duration_s"] / self.dt) + 5)
        self.ground: Any = StaticGround()
        self.p: dict[str, Any] = dict(self.base)
        self.s = _State()
        self._queue: list[np.ndarray] = []
        self._rng = np.random.default_rng(0)
        self._step_lengths: dict[int, list[float]] = {0: [], 1: []}

    # -- helpers ----------------------------------------------------------------------
    @property
    def omega(self) -> float:
        return math.sqrt(9.81 / float(self.p["z0"]))

    def ground_velocity(self) -> float:
        return float(self.ground.velocity(self.s.stance))

    def _obs(self) -> Obs:
        s, n = self.s, float(self.p["sensor_noise"])

        def noisy(v: float) -> float:
            return v + (float(self._rng.normal(0.0, n)) if n > 0 else 0.0)

        obs: Obs = {
            "state.com_rel": np.array([noisy(s.x - s.p_st)], dtype=np.float32),
            "state.com_vel_rel": np.array([noisy(s.xd - self.ground_velocity() - s.slip_vel)], dtype=np.float32),
            "state.swing_rel": np.array([noisy(s.p_sw - s.x)], dtype=np.float32),
            "state.phase": np.array([s.t_phase / float(self.p["step_time"])], dtype=np.float32),
            "state.stance": np.array([float(s.stance)], dtype=np.float32),
            "state.com_pos": np.array([s.x], dtype=np.float32),
            "state.com_vel": np.array([s.xd], dtype=np.float32),
            "annotation.task": "walk",
        }
        obs.update(self.ground.observe())
        return obs

    # -- episode ----------------------------------------------------------------------
    def reset(self, scenario: Scenario) -> Obs:
        unknown = set(scenario.params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Walker does not know parameters: {sorted(unknown)}")
        self.p = {**self.base, **scenario.params}
        self._rng = np.random.default_rng(scenario.seed)
        jitter = float(self._rng.uniform(-0.02, 0.02))  # where on the deck the robot starts
        v0 = float(self.p["start_speed"])
        self.s = _State(x=jitter, xd=v0, p_st=jitter, p_sw=jitter, stance=0)
        self._queue = [np.zeros(2)] * int(self.p["action_delay"])
        self._step_lengths = {0: [], 1: []}
        self.ground.reset(scenario, self)
        self.s.xd = v0 + self.ground_velocity()  # start moving with the ground under the stance foot
        return self._obs()

    def set_state(self, state: Obs, recent_actions: np.ndarray | None = None) -> Obs:
        def get(key: str, default: float) -> float:
            v = state.get(f"state.{key}")
            return default if v is None else float(np.asarray(v, dtype=float).reshape(-1)[0])

        s = self.s
        s.x = get("com_pos", s.x)
        s.xd = get("com_vel", s.xd)
        s.p_st = s.x - get("com_rel", s.x - s.p_st)
        s.p_sw = s.x + get("swing_rel", s.p_sw - s.x)
        s.stance = int(round(get("stance", s.stance)))
        s.t_phase = get("phase", s.t_phase / float(self.p["step_time"])) * float(self.p["step_time"])
        s.slip_vel = 0.0
        if hasattr(self.ground, "set_state"):
            self.ground.set_state(state)
        if recent_actions is not None and int(self.p["action_delay"]) > 0:
            d = int(self.p["action_delay"])
            past = [np.asarray(a, dtype=float) for a in np.asarray(recent_actions, dtype=float)[-d:]]
            self._queue = [np.zeros(2)] * (d - len(past)) + past
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=float).reshape(-1)[:2], nan=0.0), -1.0, 1.0)
        if len(a) < 2:
            a = np.pad(a, (0, 2 - len(a)))
        if self._queue:
            self._queue.append(a)
            a = self._queue.pop(0)
        p, s = self.p, self.s
        w2 = self.omega ** 2
        cop_cmd = float(a[0]) * float(p["ankle_gain"])
        if abs(cop_cmd) > 0.999:
            s.cop_sat += 1
        target_rel = float(a[1]) * float(p["reach"])
        h = self.dt / self.substeps
        reason = None
        for _ in range(self.substeps):
            self.ground.update(h, self)
            vg = self.ground_velocity()
            cop = s.p_st + float(np.clip(cop_cmd, -1.0, 1.0)) * float(p["foot_half"])
            a_contact = w2 * (s.x - cop)
            mu = float(self.ground.friction(s.stance, float(p["mu"])))
            limit = mu * 9.81
            if abs(a_contact) > limit:  # the foot cannot hold: it slides, the CoM gets only friction
                excess = abs(a_contact) - limit
                s.slip_vel += -math.copysign(excess, a_contact) * h * 4.0
                a_contact = math.copysign(limit, a_contact)
            else:
                s.slip_vel *= max(0.0, 1.0 - 20.0 * h)  # static friction catches the foot again
            xdd = a_contact - 9.81 * math.sin(math.radians(float(self.ground.incline())))
            xdd += float(self.ground.push()) / float(p["mass"])
            s.xd += xdd * h
            s.x += s.xd * h
            s.p_st += (vg + s.slip_vel) * h
            s.slipped += abs(s.slip_vel) * h
            # swing foot: in the air, heads for its target at a limited speed
            target = s.x + target_rel
            move = float(np.clip(target - s.p_sw, -float(p["swing_speed"]) * h, float(p["swing_speed"]) * h))
            s.p_sw += move
            s.t_phase += h
            s.t += h
            if s.t_phase >= float(p["step_time"]):
                self._step_lengths[1 - s.stance].append(s.p_sw - s.p_st)
                s.p_st, s.p_sw = s.p_sw, s.p_st
                s.stance = 1 - s.stance
                s.t_phase = 0.0
                s.slip_vel = 0.0
                s.steps += 1
                self.ground.on_touchdown(s.stance, self)
            lean = abs(s.x - s.p_st)
            s.lean_max = max(s.lean_max, lean)
            if self.ground.deck_half is not None:
                s.deck_max = max(s.deck_max, abs(s.x))
            if lean > float(p["max_lean"]) or not math.isfinite(s.x):
                reason = "fall"
                break
            if self.ground.deck_half is not None and abs(s.x) > self.ground.deck_half:
                reason = "off_belt"
                break
        s.ticks += 1
        obs = self._obs()
        done_ok = s.t >= float(p["duration_s"]) - 1e-9
        terminated = reason is not None or done_ok
        truncated = not terminated and s.ticks >= self.max_steps
        info: dict[str, Any] = {}
        if terminated or truncated:
            info = {"success": reason is None and done_ok, "metrics": self.metrics()}
            if reason is not None:
                info["failure_reason"] = reason
            elif not done_ok:
                info["failure_reason"] = "timeout"
        reward = -abs(s.x) * self.dt
        return obs, reward, terminated, truncated, info

    def metrics(self) -> dict[str, float]:
        s = self.s
        left, right = self._step_lengths[0], self._step_lengths[1]
        ml = float(np.mean(left)) if left else math.nan
        mr = float(np.mean(right)) if right else math.nan
        asym = abs(ml - mr) / max(abs(ml + mr) / 2, 1e-6) if left and right else math.nan
        return {"walk_time_s": s.t, "steps": float(s.steps), "cadence_spm": 60.0 * s.steps / max(s.t, 1e-9),
                "step_len_left": ml, "step_len_right": mr, "step_asymmetry": asym, "max_lean": s.lean_max,
                "deck_excursion": s.deck_max, "slip_distance": s.slipped,
                "cop_saturation": s.cop_sat / max(s.ticks, 1)}

    def close(self) -> None:
        pass


WalkerEnv = Walker
