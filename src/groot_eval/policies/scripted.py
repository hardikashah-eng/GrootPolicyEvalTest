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


class ArmIKPolicy:
    """Reference controller for :class:`~groot_eval.envs.arm3.Arm3Env` (``arm-ik``).

    Damped least-squares inverse kinematics turns the end-effector error into
    joint velocity targets; a PI velocity loop turns those into normalized
    torques; the gripper closes once the end effector is on the target.

    ``robust=True`` (``arm-ik-robust``) adds the fault handling a deployed
    controller would have, so fault reports can tell a policy that copes from
    one that doesn't:

    * non-finite readings are replaced by the last valid one;
    * a joint whose encoder stops changing while its velocity says it moves, or
      that does not move while its torque is saturated, is marked untrusted;
      its Jacobian column is dropped so the other joints take over the motion,
      and it is commanded to hold;
    * with more than one untrusted joint it slows to a controlled stop;
    * a joint is trusted again after ``retry_steps``, and re-flagged if the
      problem is still there, so a transient fault does not disable it for good.
    """

    def __init__(self, kp: float = 3.0, kv: float = 0.8, ki: float = 2.0, max_qd: float = 2.0,
                 robust: bool = False, stall_steps: int = 12, retry_steps: int = 40) -> None:
        self.kp, self.kv, self.ki, self.max_qd = kp, kv, ki, max_qd
        self.robust, self.stall_steps, self.retry_steps = robust, stall_steps, retry_steps
        self.name = f"arm-ik{'-robust' if robust else ''}(kp={kp},kv={kv})"
        self.dt = 0.02
        self.reset("")

    def reset(self, task: str) -> None:
        self._integ: np.ndarray | None = None
        self._last: dict[str, np.ndarray] = {}
        self._prev_q: np.ndarray | None = None
        self._frozen: np.ndarray | None = None
        self._stalled: np.ndarray | None = None
        self.untrusted: set[int] = set()
        self._flagged_at: dict[int, int] = {}
        self._t = 0

    def _clean(self, obs: Obs, key: str) -> np.ndarray:
        v = np.asarray(obs.get(key, self._last.get(key, np.zeros(0))), dtype=float).copy()
        if self.robust:
            last = self._last.get(key)
            bad = ~np.isfinite(v)
            if bad.any() and last is not None and last.shape == v.shape:
                v[bad] = last[bad]
            v = np.nan_to_num(v)
            self._last[key] = v.copy()
        return v

    def act(self, obs: Obs) -> np.ndarray:
        from groot_eval.envs.arm3 import arm_fk, arm_jacobian

        q = self._clean(obs, "state.joint_pos")
        qd = self._clean(obs, "state.joint_vel")
        target = self._clean(obs, "state.target")
        n = q.size
        lengths = np.full(n, 0.9 / n)
        has_grip = "state.gripper" in obs
        if self._integ is None:
            self._integ = np.zeros(n)
            self._frozen = np.zeros(n, dtype=int)
            self._stalled = np.zeros(n, dtype=int)
        ee = arm_fk(q, lengths)[-1]
        err = target - ee
        J = arm_jacobian(q, lengths)
        if self.robust:
            self._diagnose(q, qd, obs)
            for j in self.untrusted:
                J[:, j] = 0.0
        lam = 0.05
        qd_des = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(2), self.kp * err)
        qd_des = np.clip(qd_des, -self.max_qd, self.max_qd)
        # respect joint limits: slow down toward a limit and stop short of it (margin 0.15 rad)
        lim = np.array([[-0.2, np.pi + 0.2]] + [[-2.6, 2.6]] * (n - 1))
        room_hi, room_lo = lim[:, 1] - 0.15 - q, q - (lim[:, 0] + 0.15)
        qd_des = np.where(qd_des > 0, np.minimum(qd_des, np.maximum(room_hi, 0.0) * 4.0), qd_des)
        qd_des = np.where(qd_des < 0, np.maximum(qd_des, -np.maximum(room_lo, 0.0) * 4.0), qd_des)
        if self.robust and len(self.untrusted) > 1:
            qd_des[:] = 0.0  # controlled stop
        vel_err = qd_des - qd
        self._integ = np.clip(self._integ + self.dt * self.ki * vel_err, -0.6, 0.6)
        u = np.clip(self.kv * vel_err + self._integ, -1.0, 1.0)
        if self.robust:
            for j in self.untrusted:
                u[j] = np.clip(-self.kv * qd[j], -0.3, 0.3)  # hold, gently
        self._prev_q = q.copy()
        if not has_grip:
            return u
        close = 1.0 if np.linalg.norm(err) < 0.04 else 0.0
        return np.append(u, close)

    def _diagnose(self, q: np.ndarray, qd: np.ndarray, obs: Obs) -> None:
        tau = np.asarray(obs.get("state.joint_torque", np.zeros_like(q)), dtype=float)
        tau = np.nan_to_num(tau)
        self._t += 1
        for j, t0 in list(self._flagged_at.items()):
            if self._t - t0 >= self.retry_steps:  # give it another chance
                self.untrusted.discard(j)
                del self._flagged_at[j]
                self._frozen[j] = self._stalled[j] = 0
        if self._prev_q is None:
            return
        dq = np.abs(q - self._prev_q)
        for j in range(q.size):
            frozen = dq[j] < 1e-7 and abs(qd[j]) > 0.05  # encoder stuck while the joint moves
            stalled = abs(tau[j]) > 0.95 and abs(qd[j]) < 0.02  # torque saturated, nothing moves
            self._frozen[j] = self._frozen[j] + 1 if frozen else 0
            self._stalled[j] = self._stalled[j] + 1 if stalled else 0
            if (self._frozen[j] >= 3 or self._stalled[j] >= self.stall_steps) and j not in self.untrusted:
                self.untrusted.add(j)
                self._flagged_at[j] = self._t
