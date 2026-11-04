"""Reference controllers for the built-in Reach2D simulator.

They stand in for a learned policy so the whole harness can be demonstrated
on a laptop. ``pd`` is a naive controller that ignores obstacles; ``pd-avoid``
adds a repulsive term. Comparing them shows how a sweep separates a real
improvement from noise. ``pd-chunked`` predicts open-loop action chunks the
way a VLA model does, so chunk length becomes something you can evaluate.
"""

from __future__ import annotations

import numpy as np

from groot_eval.policies.base import Obs


class PDPolicy:
    def __init__(self, kp: float = 2.0, kd: float = 1.2, avoid: float = 0.0) -> None:
        self.kp, self.kd, self.avoid = kp, kd, avoid
        self.name = f"pd(kp={kp},kd={kd}{',avoid=' + str(avoid) if avoid else ''})"

    def reset(self, task: str) -> None:
        pass

    def act(self, obs: Obs) -> np.ndarray:
        pos = np.asarray(obs["state.position"], dtype=float)
        vel = np.asarray(obs["state.velocity"], dtype=float)
        target = np.asarray(obs["state.target"], dtype=float)
        action = self.kp * (target - pos) - self.kd * vel
        if self.avoid and "state.obstacle" in obs:
            ox, oy, r = np.asarray(obs["state.obstacle"], dtype=float)
            away = pos - np.array([ox, oy])
            d = np.linalg.norm(away)
            influence = r + 0.3
            if 1e-6 < d < influence:
                n = away / d
                closeness = (influence - d) / (influence - r)  # 0 at the edge, 1 at contact
                # Cancel the part of the command that drives into the obstacle,
                # brake any velocity toward it, and slide around it instead.
                inward = min(0.0, float(np.dot(action, n)))
                radial_v = min(0.0, float(np.dot(vel, n)))
                tangent = np.array([-n[1], n[0]])
                if np.dot(tangent, target - pos) < 0:
                    tangent = -tangent
                action += self.avoid * closeness * (-inward * n - 2.0 * radial_v * n + 0.8 * tangent + 0.3 * n)
        return np.clip(action, -1.0, 1.0)


class PDChunkPredictor:
    """Rolls a PD controller forward on an internal model to emit an action chunk."""

    def __init__(self, horizon: int = 16, kp: float = 2.0, kd: float = 1.2, avoid: float = 0.0) -> None:
        self.horizon = horizon
        self.inner = PDPolicy(kp, kd, avoid)
        self.name = f"pd-chunk(h={horizon})"

    def reset(self, task: str) -> None:
        pass

    def predict_chunk(self, obs: Obs) -> np.ndarray:
        # Nominal model: default friction and mass. Mismatch with the real
        # (randomized) dynamics is exactly the sim-to-real style error that
        # long open-loop chunks amplify.
        pos = np.asarray(obs["state.position"], dtype=float).copy()
        vel = np.asarray(obs["state.velocity"], dtype=float).copy()
        sim_obs = dict(obs)
        out = []
        for _ in range(self.horizon):
            sim_obs["state.position"], sim_obs["state.velocity"] = pos, vel
            a = self.inner.act(sim_obs)
            out.append(a)
            vel = vel + 0.05 * (a * 4.0 - 1.2 * vel)
            pos = pos + 0.05 * vel
        return np.stack(out)


class ReplayPolicy:
    """Returns recorded actions in order; used to sanity-check open-loop scoring."""

    name = "replay"

    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=float)
        self._i = 0

    def reset(self, task: str) -> None:
        self._i = 0

    def act(self, obs: Obs) -> np.ndarray:
        a = self.actions[min(self._i, len(self.actions) - 1)]
        self._i += 1
        return a
