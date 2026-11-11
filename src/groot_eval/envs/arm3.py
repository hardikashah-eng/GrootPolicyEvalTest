"""An articulated CPU simulator: a planar arm with N revolute joints and a gripper.

``Reach2D`` is a point mass, so it has no joints that can wear out or lose a
degree of freedom. ``Arm3`` is the smallest model that does, and it still runs
hundreds of episodes per second on a laptop:

* a vertical-plane arm (3 joints by default) whose end effector must reach a
  target and close its gripper there;
* per-joint motors (torque limit, gain), gearboxes (backlash, Coulomb and
  viscous friction, stiction), joint limits and gravity, with the robot's own
  gravity compensation computed for the *nominal* payload, so an extra
  payload or a limp motor makes the arm sag;
* an optional obstacle and an optional tiny camera image (``video.wrist``)
  whose brightness follows a ``lighting`` parameter.

It is not a realistic arm. Its job is to give every wear and fault effect in
:mod:`groot_eval.degradation` a named component to act on, which is what
:meth:`components` and :meth:`set_component_param` expose. An Isaac Lab
adapter can provide the same two hooks through ``param_setters``.

Observations (GR00T-style keys): ``state.joint_pos``, ``state.joint_vel``,
``state.joint_torque`` (applied, normalized to the torque limit),
``state.ee_pos`` (forward kinematics of the *reported* joint angles, as a real
controller computes it), ``state.target``, ``state.gripper`` (0 open, 1
closed) and ``state.obstacle`` when present. Actions: one normalized torque
per joint in [-1, 1], then the gripper command (> 0.5 closes).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from groot_eval.envs.base import ComponentSpec, Obs
from groot_eval.types import Scenario

GRAVITY = 9.81
LINK_MASS = 0.3
REACH = 0.9

DEFAULTS: dict[str, Any] = {
    "friction": 1.0,  # scales every joint's Coulomb and viscous friction
    "payload_kg": 0.0,  # extra mass at the end effector (gravity comp assumes 0)
    "sensor_noise": 0.0,  # encoder noise, rad
    "action_delay": 0,  # control steps between command and actuation
    "obstacle": False,
    "obstacle_radius": 0.08,
    "target_radius": None,  # distance of the target from the base (None = random 0.35-0.8)
    "target_angle": None,  # direction of the target, rad from +x (None = random 0.35-2.8)
    "lighting": 1.0,  # camera brightness factor (only with render=True)
    "target_tolerance": 0.03,
    "waypoints": 1,  # targets to visit in order (long-horizon tasks); the gripper closes on the last
}

# Physics levers per component kind, applied through set_component_param.
MOTOR_KEYS = {"gain_scale": 1.0, "torque_limit_scale": 1.0, "enabled": True}
JOINT_KEYS = {"friction_scale": 1.0, "damping_scale": 1.0, "backlash_rad": 0.0, "stiction": 0.0,
              "locked": False, "range_scale": 1.0}
GRIPPER_KEYS = {"force_scale": 1.0, "close_rate_scale": 1.0, "stuck": None}


def arm_fk(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Joint positions (N+1, 2) of a planar serial arm, base at the origin."""
    phi = np.cumsum(q)
    steps = np.stack([lengths * np.cos(phi), lengths * np.sin(phi)], axis=1)
    return np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])


def arm_jacobian(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    pts = arm_fk(q, lengths)
    ee = pts[-1]
    J = np.zeros((2, len(q)))
    for j in range(len(q)):
        r = ee - pts[j]
        J[:, j] = [-r[1], r[0]]
    return J


class Arm3Env:
    dt = 0.02

    def __init__(self, n_joints: int = 3, max_steps: int = 300, gripper: bool = True, render: bool = False,
                 image_size: int = 32) -> None:
        if n_joints < 2:
            raise ValueError("n_joints must be >= 2")
        self.n = int(n_joints)
        self.max_steps = max_steps
        self.gripper = bool(gripper)
        self.render = bool(render)
        self.image_size = int(image_size)
        self.lengths = np.full(self.n, REACH / self.n)
        self.tau_max = np.array([4.0 + 8.0 * (self.n - j) / self.n for j in range(self.n)])
        self.inertia = np.array([0.04 + 0.25 * (self.n - j) / self.n for j in range(self.n)])
        self.limits = np.array([[-0.2, math.pi + 0.2]] + [[-2.6, 2.6]] * (self.n - 1))
        self.p: dict[str, Any] = dict(DEFAULTS)
        self._rng = np.random.default_rng(0)
        self._reset_components()

    # -- component hooks -------------------------------------------------------------
    def components(self) -> dict[str, ComponentSpec]:
        out: dict[str, ComponentSpec] = {}
        for j in range(self.n):
            out[f"motor[{j}]"] = ComponentSpec(f"motor[{j}]", "motor", action=(j,), physics_keys=tuple(MOTOR_KEYS))
            out[f"joint[{j}]"] = ComponentSpec(f"joint[{j}]", "joint", action=(j,), physics_keys=tuple(JOINT_KEYS))
            out[f"encoder[{j}]"] = ComponentSpec(f"encoder[{j}]", "encoder",
                                                 state={"state.joint_pos": (j, j + 1), "state.joint_vel": (j, j + 1)})
        if self.gripper:
            out["gripper"] = ComponentSpec("gripper", "gripper", action=(self.n,), state={"state.gripper": (0, 1)},
                                           physics_keys=tuple(GRIPPER_KEYS))
        if self.render:
            out["camera.wrist"] = ComponentSpec("camera.wrist", "camera", state={"video.wrist": (0, 0)})
        out["bus"] = ComponentSpec("bus", "bus", action=tuple(range(self.action_dim)))
        return out

    def state_limits(self) -> dict[str, list]:
        """Ranges the sim can represent, per ``state.*`` key (``groot-eval transfer`` checks real data against them)."""
        return {"joint_pos": self.limits.tolist(), "gripper": [[0.0, 1.0]]}

    @property
    def action_dim(self) -> int:
        return self.n + (1 if self.gripper else 0)

    def set_component_param(self, name: str, key: str, value: Any) -> None:
        """Change a physics parameter of one component for the rest of the episode."""
        if name not in self.comp:
            raise KeyError(f"Arm3 has no component {name!r}")
        if key not in self.comp[name]:
            raise NotImplementedError(f"{name} has no physics lever {key!r}")
        self.comp[name][key] = value

    def get_component_param(self, name: str, key: str) -> Any:
        return self.comp[name][key]

    def _reset_components(self) -> None:
        self.comp: dict[str, dict[str, Any]] = {}
        for j in range(self.n):
            self.comp[f"motor[{j}]"] = dict(MOTOR_KEYS)
            self.comp[f"joint[{j}]"] = dict(JOINT_KEYS)
        if self.gripper:
            self.comp["gripper"] = dict(GRIPPER_KEYS)

    # -- episode ----------------------------------------------------------------------
    def reset(self, scenario: Scenario) -> Obs:
        unknown = set(scenario.params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Arm3 does not know parameters: {sorted(unknown)}")
        self.p = {**DEFAULTS, **scenario.params}
        self._rng = np.random.default_rng(scenario.seed)
        self._reset_components()
        home = np.array([math.pi / 2] + [-0.6] * (self.n - 1))
        self.q = home + self._rng.normal(0, 0.15, self.n)
        self.m = self.q.copy()  # motor-side angle (differs from q by backlash play)
        self.qd = np.zeros(self.n)
        self.md = np.zeros(self.n)
        self.tau = np.zeros(self.n)
        self.grip = 0.0
        r = self.p["target_radius"] if self.p["target_radius"] is not None else self._rng.uniform(0.35, 0.8)
        th = self.p["target_angle"] if self.p["target_angle"] is not None else self._rng.uniform(0.35, math.pi - 0.35)
        self.targets = [np.array([r * math.cos(th), r * math.sin(th)])]
        for _ in range(int(self.p["waypoints"]) - 1):
            r, th = self._rng.uniform(0.35, 0.8), self._rng.uniform(0.35, math.pi - 0.35)
            self.targets.append(np.array([r * math.cos(th), r * math.sin(th)]))
        self.waypoint = 0
        self.target = self.targets[0]
        ee0 = arm_fk(self.q, self.lengths)[-1]
        mid, d = (ee0 + self.target) / 2, self.target - ee0
        normal = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-6)
        self.obstacle_centre = mid + normal * self._rng.uniform(-0.04, 0.04)  # near the straight path
        r = float(self.p["obstacle_radius"])
        pts0 = arm_fk(self.q, self.lengths)
        if min(np.linalg.norm(p - self.obstacle_centre) for p in pts0[1:]) < 2 * r:
            self.obstacle_centre = mid + normal * 3 * r  # never start in contact
        self._queue: deque[np.ndarray] = deque(np.zeros(self.action_dim) for _ in range(int(self.p["action_delay"])))
        self.t = 0
        self.limit_hits = 0
        self.saturated_steps = 0
        self.max_speed = 0.0
        self.path = 0.0
        self.task = scenario.task or "reach the target and close the gripper"
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=float).reshape(self.action_dim), -1.0, 1.0)
        self._queue.append(action)
        applied = self._queue.popleft()
        u = applied[: self.n]
        ee_before = arm_fk(self.q, self.lengths)[-1]

        tau_motor = np.zeros(self.n)
        g_actual = self._gravity(self.q, float(self.p["payload_kg"]))
        g_nominal = self._gravity(self.q, 0.0)
        net = np.zeros(self.n)
        for j in range(self.n):
            mo, jo = self.comp[f"motor[{j}]"], self.comp[f"joint[{j}]"]
            limit = self.tau_max[j] * float(mo["torque_limit_scale"])
            if mo["enabled"]:
                raw = u[j] * self.tau_max[j] * float(mo["gain_scale"])
                tau_motor[j] = np.clip(raw, -limit, limit)
                comp = g_nominal[j]  # the controller's gravity feed-forward assumes no payload
                if abs(raw) >= limit * 0.999:
                    self.saturated_steps += 1
            else:
                comp = 0.0  # limp: no torque, no gravity compensation
            fric = float(self.p["friction"]) * float(jo["friction_scale"])
            visc = 0.6 * fric * float(jo["damping_scale"]) * self.md[j]
            coul = 0.08 * fric * math.tanh(self.md[j] / 0.01)
            drive = tau_motor[j] + comp - g_actual[j]
            stiction = float(jo["stiction"]) * self.tau_max[j]
            if abs(self.md[j]) < 1e-3 and abs(drive) < stiction:
                drive = 0.0
                self.md[j] = 0.0
            net[j] = drive - visc - coul
        self.tau = tau_motor / self.tau_max

        q_prev = self.q.copy()
        for j in range(self.n):
            jo = self.comp[f"joint[{j}]"]
            if jo["locked"]:
                self.md[j] = 0.0
                continue
            self.md[j] += self.dt * net[j] / self.inertia[j]
            self.m[j] += self.dt * self.md[j]
            b = float(jo["backlash_rad"])
            if b > 0:  # play operator: the link follows the motor only once the slack is taken up
                if self.m[j] - self.q[j] > b / 2:
                    self.q[j] = self.m[j] - b / 2
                elif self.q[j] - self.m[j] > b / 2:
                    self.q[j] = self.m[j] + b / 2
            else:
                self.q[j] = self.m[j]
            lo, hi = self.limits[j]
            rs = float(jo["range_scale"])
            mid = (lo + hi) / 2
            lo, hi = mid + (lo - mid) * rs, mid + (hi - mid) * rs
            if self.q[j] < lo or self.q[j] > hi:
                self.q[j] = float(np.clip(self.q[j], lo, hi))
                self.m[j] = self.q[j]
                self.md[j] = 0.0
                self.limit_hits += 1
        self.qd = (self.q - q_prev) / self.dt
        self.max_speed = max(self.max_speed, float(np.max(np.abs(self.qd))))

        if self.gripper:
            g = self.comp["gripper"]
            want = 1.0 if applied[self.n] > 0.5 else 0.0
            if g["stuck"] == "open":
                want = 0.0
            elif g["stuck"] == "closed":
                want = 1.0
            rate = 0.08 * float(g["close_rate_scale"])
            self.grip = float(np.clip(self.grip + np.clip(want - self.grip, -rate, rate), 0.0, 1.0))

        pts = arm_fk(self.q, self.lengths)
        ee = pts[-1]
        self.path += float(np.linalg.norm(ee - ee_before))
        self.t += 1
        dist = float(np.linalg.norm(self.target - ee))
        ee_speed = float(np.linalg.norm(ee - ee_before) / self.dt)
        info: dict[str, Any] = {}
        terminated = truncated = False
        if self.p["obstacle"] and self._collides(pts):
            terminated, info = True, {"success": False, "failure_reason": "collision"}
        elif np.max(np.abs(self.qd)) > 25.0:
            terminated, info = True, {"success": False, "failure_reason": "unstable"}
        elif dist < float(self.p["target_tolerance"]) and ee_speed < 0.1 and self.waypoint < len(self.targets) - 1:
            self.waypoint += 1  # intermediate waypoint reached: move on
            self.target = self.targets[self.waypoint]
            dist = float(np.linalg.norm(self.target - ee))
        elif dist < float(self.p["target_tolerance"]) and ee_speed < 0.1 and (not self.gripper or self._grasped()):
            terminated, info = True, {"success": True}
        elif self.t >= self.max_steps:
            truncated, info = True, {"success": False, "failure_reason": "timeout"}
        info["step_metrics"] = {"ee_error": dist, "max_joint_speed": float(np.max(np.abs(self.qd))),
                                "joint_limit_hits": float(self.limit_hits),
                                "torque_saturation_steps": float(self.saturated_steps)}
        if terminated or truncated:
            info["metrics"] = {
                "final_distance": dist,
                "path_length": self.path,
                "joint_limit_hits": float(self.limit_hits),
                "torque_saturation_steps": float(self.saturated_steps),
                "max_joint_speed": self.max_speed,
                "final_joint_speed": float(np.max(np.abs(self.qd))),
                "waypoints_reached": float(self.waypoint + (1 if info.get("success") else 0)),
            }
        return self._obs(), -dist, terminated, truncated, info

    def close(self) -> None:
        pass

    def set_state(self, state: Obs, recent_actions: np.ndarray | None = None) -> Obs:
        if "state.joint_pos" in state:
            self.q = np.asarray(state["state.joint_pos"], dtype=float).reshape(self.n).copy()
            self.m = self.q.copy()
        if "state.joint_vel" in state:
            self.qd = np.asarray(state["state.joint_vel"], dtype=float).reshape(self.n).copy()
            self.md = self.qd.copy()
        if "state.target" in state:
            self.target = np.asarray(state["state.target"], dtype=float).reshape(2).copy()
        if "state.gripper" in state and self.gripper:
            self.grip = float(np.asarray(state["state.gripper"]).reshape(-1)[0])
        if "state.obstacle" in state and self.p["obstacle"]:
            self.obstacle_centre = np.asarray(state["state.obstacle"], dtype=float).reshape(-1)[:2].copy()
        n = int(self.p["action_delay"])
        past = [] if recent_actions is None else list(np.asarray(recent_actions, dtype=float).reshape(-1, self.action_dim))
        history = [np.zeros(self.action_dim)] * n + past
        self._queue = deque(history[len(history) - n:]) if n else deque()
        return self._obs()

    def derive_obs(self, obs: Obs) -> Obs:
        """Recompute derived signals (end-effector pose) from the reported joint angles.

        Wrappers that corrupt encoders call this, so ``state.ee_pos`` stays what a
        real controller would compute from the readings it actually has.
        """
        q = np.asarray(obs.get("state.joint_pos"), dtype=float)
        if q.shape == (self.n,):
            obs["state.ee_pos"] = arm_fk(q, self.lengths)[-1].astype(np.float32)
        return obs

    # -- helpers ----------------------------------------------------------------------
    def _gravity(self, q: np.ndarray, payload: float) -> np.ndarray:
        """Gravity torque on each joint (positive = counter-clockwise)."""
        pts = arm_fk(q, self.lengths)
        com = (pts[:-1] + pts[1:]) / 2
        tau = np.zeros(self.n)
        for j in range(self.n):
            lever = com[j:, 0] - pts[j, 0]
            tau[j] = -GRAVITY * (LINK_MASS * float(np.sum(lever)) + payload * (pts[-1, 0] - pts[j, 0]))
        return -tau  # torque needed to hold against gravity

    def _grasped(self) -> bool:
        return self.grip > 0.8 and float(self.comp["gripper"]["force_scale"]) >= 0.3

    def _collides(self, pts: np.ndarray) -> bool:
        r = float(self.p["obstacle_radius"])
        for a, b in zip(pts[1:-1], pts[2:]):  # links beyond the shoulder
            for s in (0.0, 0.5, 1.0):
                if np.linalg.norm(a + s * (b - a) - self.obstacle_centre) < r:
                    return True
        return False

    def _obs(self) -> Obs:
        noise = float(self.p["sensor_noise"])
        q_seen = self.q + (self._rng.normal(0, noise, self.n) if noise > 0 else 0.0)
        obs: Obs = {
            "state.joint_pos": q_seen.astype(np.float32),
            "state.joint_vel": self.qd.astype(np.float32),
            "state.joint_torque": self.tau.astype(np.float32),
            "state.target": self.target.astype(np.float32),
            "annotation.task": self.task,
        }
        if self.gripper:
            obs["state.gripper"] = np.array([self.grip], dtype=np.float32)
        if self.p["obstacle"]:
            obs["state.obstacle"] = np.append(self.obstacle_centre, self.p["obstacle_radius"]).astype(np.float32)
        if self.render:
            obs["video.wrist"] = self._image()
        return self.derive_obs(obs)

    def _image(self) -> np.ndarray:
        s = self.image_size
        light = float(self.p["lighting"])
        img = np.full((s, s, 3), 60.0 * light)
        yy, xx = np.mgrid[0:s, 0:s]

        def blob(p: np.ndarray, value: float, rad: float) -> None:
            cx, cy = (p[0] + 1) / 2 * (s - 1), (1 - (p[1] + 0.2) / 1.2) * (s - 1)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2] = value * light

        blob(self.target, 220.0, 2.0)
        blob(arm_fk(self.q, self.lengths)[-1], 140.0, 1.5)
        img += self._rng.normal(0, 4.0 / max(light, 0.05), img.shape)  # dim scenes are noisier
        return np.clip(img, 0, 255).astype(np.uint8)
