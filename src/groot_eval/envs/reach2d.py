"""A small CPU simulator for exercising the harness without a GPU.

A point mass must reach a target in the unit square. It is deliberately
simple, but it has the properties that make real robot evaluation hard:

* physics parameters (friction, mass, actuator gain) that a sweep varies;
* sensor noise and actuation delay, which destabilize aggressive controllers;
* an optional obstacle that a naive policy drives straight into;
* several distinct failure modes (collision, out of bounds, timeout).

That is enough to show sweeps, confidence intervals, failure attribution and
regression suites behaving the way they would on Isaac Lab, in seconds.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from groot_eval.envs.base import Obs
from groot_eval.types import Scenario

DEFAULTS: dict[str, Any] = {
    "friction": 0.6,
    "mass": 1.0,
    "actuator_gain": 1.0,
    "sensor_noise": 0.0,
    "action_delay": 0,
    "obstacle": False,
    "obstacle_radius": 0.15,
    "target_tolerance": 0.05,
}


class Reach2DEnv:
    dt = 0.05

    def __init__(self, max_steps: int = 200) -> None:
        self.max_steps = max_steps
        self._rng = np.random.default_rng(0)
        self.p: dict[str, Any] = dict(DEFAULTS)

    # -- episode ----------------------------------------------------------
    def reset(self, scenario: Scenario) -> Obs:
        unknown = set(scenario.params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Reach2D does not know parameters: {sorted(unknown)}")
        self.p = {**DEFAULTS, **scenario.params}
        self._rng = np.random.default_rng(scenario.seed)
        self.pos = self._rng.uniform(-0.8, -0.4, size=2)
        self.target = self._rng.uniform(0.4, 0.8, size=2)
        self.vel = np.zeros(2)
        self.t = 0
        self.path_length = 0.0
        self.min_clearance = np.inf
        self.obstacle_centre = (self.pos + self.target) / 2 + self._rng.normal(0, 0.03, size=2)
        self._queue: deque[np.ndarray] = deque(np.zeros(2) for _ in range(int(self.p["action_delay"])))
        self.task = scenario.task or "move the end effector to the target"
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        self._queue.append(action)
        applied = self._queue.popleft()

        accel = applied * float(self.p["actuator_gain"]) * 4.0 / float(self.p["mass"])
        accel -= float(self.p["friction"]) * 2.0 * self.vel
        self.vel = self.vel + self.dt * accel
        new_pos = self.pos + self.dt * self.vel
        self.path_length += float(np.linalg.norm(new_pos - self.pos))
        self.pos = new_pos
        self.t += 1

        dist = float(np.linalg.norm(self.target - self.pos))
        info: dict[str, Any] = {}
        terminated = truncated = False

        if self.p["obstacle"]:
            clearance = float(np.linalg.norm(self.pos - self.obstacle_centre)) - float(self.p["obstacle_radius"])
            self.min_clearance = min(self.min_clearance, clearance)
            if clearance < 0:
                terminated, info = True, {"success": False, "failure_reason": "collision"}
        if not terminated and np.any(np.abs(self.pos) > 1.0):
            terminated, info = True, {"success": False, "failure_reason": "out_of_bounds"}
        if not terminated and dist < float(self.p["target_tolerance"]) and np.linalg.norm(self.vel) < 0.3:
            terminated, info = True, {"success": True}
        if not terminated and self.t >= self.max_steps:
            truncated, info = True, {"success": False, "failure_reason": "timeout"}

        reward = -dist
        if terminated or truncated:
            info["metrics"] = {
                "final_distance": dist,
                "path_length": self.path_length,
                "min_obstacle_clearance": float(self.min_clearance) if self.p["obstacle"] else float("nan"),
            }
        return self._obs(), reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # -- helpers ----------------------------------------------------------
    def _obs(self) -> Obs:
        noise = float(self.p["sensor_noise"])
        seen = self.pos + (self._rng.normal(0, noise, size=2) if noise > 0 else 0)
        obs: Obs = {
            "state.position": seen.astype(np.float32),
            "state.velocity": self.vel.astype(np.float32),
            "state.target": self.target.astype(np.float32),
            "annotation.task": self.task,
        }
        if self.p["obstacle"]:
            obs["state.obstacle"] = np.append(self.obstacle_centre, self.p["obstacle_radius"]).astype(np.float32)
        return obs
