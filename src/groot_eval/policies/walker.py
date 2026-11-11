"""Reference walking controller for the walker twin: divergent-component-of-motion (capture point) stepping.

Each control tick it predicts where the divergent component of motion (DCM, ``xi = x + xd / w``) will be when
the swing foot lands, and places the foot so the next step starts from the steady-gait offset for the desired
speed. The ankle (CoP) tracks the nominal DCM within the foot in the meantime.

On a treadmill (``state.belt_speed`` present) it walks at the belt speed, minus a correction toward the deck
centre (from ``state.com_pos``), so it stays in place. ``track_belt=False`` makes it ignore the belt, as a
controller unaware of the treadmill would.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _f(obs: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = obs.get(f"state.{key}")
    return default if v is None else float(np.mean(np.asarray(v, dtype=float).reshape(-1)))


class WalkerCapturePolicy:
    def __init__(self, speed: float = 0.6, track_belt: bool = True, k_center: float = 0.6, step_time: float = 0.4,
                 z0: float = 0.9, reach: float = 0.6, foot_half: float = 0.10, k_cop: float = 1.5) -> None:
        self.speed, self.track_belt, self.k_center = float(speed), bool(track_belt), float(k_center)
        self.T, self.z0, self.reach, self.foot_half, self.k_cop = float(step_time), float(z0), float(reach), float(foot_half), float(k_cop)
        self.w = math.sqrt(9.81 / self.z0)

    def reset(self, task: str = "") -> None:
        pass

    def desired_speed(self, obs: dict[str, Any]) -> float:
        if self.track_belt and "state.belt_speed" in obs:
            return _f(obs, "belt_speed") - self.k_center * _f(obs, "com_pos")
        return self.speed

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        w, T = self.w, self.T
        xr, vr = _f(obs, "com_rel"), _f(obs, "com_vel_rel")
        phase = min(max(_f(obs, "phase"), 0.0), 1.0)
        v_des = max(0.0, self.desired_speed(obs))
        b = v_des * T / (math.exp(w * T) - 1.0)
        xi = xr + vr / w
        xi_nom = b * math.exp(w * phase * T)
        cop = float(np.clip(self.k_cop * (xi - xi_nom), -self.foot_half, self.foot_half))
        t_rem = (1.0 - phase) * T
        e, c, s = math.exp(w * t_rem), math.cosh(w * t_rem), math.sinh(w * t_rem)
        xi_T = cop + (xi - cop) * e  # DCM at touchdown, relative to the stance foot
        x_T = cop + (xr - cop) * c + vr / w * s  # CoM at touchdown, relative to the stance foot
        target_rel = (xi_T - b) - x_T
        return np.array([cop / self.foot_half, float(np.clip(target_rel / self.reach, -1.0, 1.0))])
