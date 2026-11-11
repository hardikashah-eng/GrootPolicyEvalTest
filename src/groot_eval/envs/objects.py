"""A small time-stepped rigid-object simulator for the standard real-to-sim physics tests.

``real2sim`` estimates an object's mass, friction and restitution from closed-form models
of five bench tests (weigh, slide, push, tilt, drop). This module runs the *same tests
through a simulator*, so the ``physics.sim_rollout`` hook has a working reference:

    physics:
      sim_rollout: groot_eval.envs.objects:rollout
      sim_rollout_args: {surface_friction: 0.8, friction_combine: multiply}

Replace it with a function that runs each test in your own engine (Isaac Lab, MuJoCo, ...)
and returns the same observation: positions at the test's times (slide, push), successive
bounce apex heights (drop), the incline angle at which the object starts to slide (tilt),
or the mass (weigh). That catches engine behaviour the parameters alone do not show:
friction combine modes, contact restitution thresholds, solver substeps.

Integration is semi-implicit Euler at ``dt`` (1 ms default) with Coulomb friction and a
coefficient of restitution applied at each impact.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

GRAVITY = 9.81


def _combine(obj: float, surface: float, mode: str) -> float:
    if mode == "param":
        return obj
    if mode == "multiply":
        return obj * surface
    if mode == "average":
        return (obj + surface) / 2
    if mode == "min":
        return min(obj, surface)
    if mode == "max":
        return max(obj, surface)
    raise ValueError(f"unknown friction_combine {mode!r}; use param, multiply, average, min or max")


def _slide(times: np.ndarray, v0: float, mu: float, dt: float, force: float = 0.0, duration: float = 0.0,
           mass: float = 1.0, mu_s: float | None = None) -> np.ndarray:
    x, v, t = 0.0, v0, 0.0
    xs, ts = [0.0], [0.0]
    end = float(times.max()) + dt
    while t < end:
        f_app = force if t < duration else 0.0
        if abs(v) < 1e-9:
            if abs(f_app) <= (mu_s if mu_s is not None else mu) * mass * GRAVITY:
                v = 0.0
                a = 0.0
            else:
                a = f_app / mass - math.copysign(mu * GRAVITY, f_app)
        else:
            a = f_app / mass - math.copysign(mu * GRAVITY, v)
            if v * (v + a * dt) < 0 and abs(f_app) <= mu * mass * GRAVITY:
                a, v = 0.0, 0.0  # friction stops it within the step
        v += a * dt
        x += v * dt
        t += dt
        xs.append(x)
        ts.append(t)
    return np.interp(times, ts, xs)


def _drop(h0: float, e: float, n: int, dt: float) -> np.ndarray:
    z, v, apexes, rising = h0, 0.0, [], False
    steps = 0
    while len(apexes) < n and steps < 10_000_000:
        v -= GRAVITY * dt
        z += v * dt
        steps += 1
        if z <= 0.0:
            z, v = 0.0, -v * e
            rising = True
            if v < 1e-4:
                apexes += [0.0] * (n - len(apexes))
                break
        elif rising and v <= 0.0:
            apexes.append(z)
            rising = False
    return np.asarray(apexes[:n], dtype=float)


def rollout(test: dict[str, Any], params: dict[str, float], dt: float = 0.001, surface_friction: float = 1.0,
            friction_combine: str = "param") -> np.ndarray:
    """Run one bench test with the given object parameters; returns the test's observation."""
    kind = test["kind"]
    mu = _combine(float(params.get("friction", 0.5)), surface_friction, friction_combine)
    mu_s = params.get("static_friction")
    mu_s = _combine(float(mu_s), surface_friction, friction_combine) if mu_s is not None else None
    if kind == "slide":
        return _slide(np.asarray(test["t"], float), float(test["v0"]), mu, dt)
    if kind == "push":
        return _slide(np.asarray(test["t"], float), 0.0, mu, dt, float(test["force"]), float(test["duration"]),
                      float(params["mass"]), mu_s)
    if kind == "drop":
        return _drop(float(test["h0"]), float(params["restitution"]), len(test["bounce_heights"]), dt)
    if kind == "tilt":
        static = mu_s if mu_s is not None else mu
        angle = 0.0
        while math.tan(math.radians(angle)) <= static and angle < 89.9:
            angle += 0.05
        return np.array([angle])
    if kind == "weigh":
        return np.array([float(params["mass"])])
    raise ValueError(f"unknown test kind {kind!r}")
