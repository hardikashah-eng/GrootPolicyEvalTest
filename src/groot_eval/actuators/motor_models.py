"""Motor models used by :class:`~groot_eval.actuators.core.WearAwareActuatorCore`.

* :class:`DCMotorSaturation`: torque-speed curve. The torque a motor can make
  drops linearly with speed (back-EMF), so a fast joint cannot also be strong.
  Same shape as Isaac Lab's ``DCMotor``.
* :class:`MLPMotorModel`: a learned actuator network in the style of Isaac
  Lab's ``ActuatorNetMLP``: a small MLP maps a short history of position
  errors and joint velocities to torque, capturing gearbox compliance, friction
  and controller behaviour a PD law misses. Weights are stored as ``.npz`` so
  they can be fitted from robot logs (``groot-robot calibrate``) without
  TorchScript, and evaluated on NumPy or torch.

All functions take and return arrays of shape ``(num_envs, num_joints)`` and
work on any array-API namespace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from array_api_compat import array_namespace, device


@dataclass
class DCMotorSaturation:
    """Torque-speed limit: ``|tau| <= saturation * (1 -/+ qd / v_max)``, then the hard effort limit."""

    saturation_effort: Any  # (J,) or scalar, N·m at zero speed
    velocity_limit: Any  # (J,) or scalar, rad/s at which available torque reaches zero

    def clip(self, tau: Any, qd: Any, effort_limit: Any) -> Any:
        xp = array_namespace(tau)
        sat = _like(self.saturation_effort, tau)
        vmax = _like(self.velocity_limit, tau)
        lim = _like(effort_limit, tau)
        upper = xp.minimum(sat * (1.0 - qd / vmax), lim)
        lower = xp.maximum(sat * (-1.0 - qd / vmax), -lim)
        upper = xp.maximum(upper, xp.zeros_like(upper))
        lower = xp.minimum(lower, xp.zeros_like(lower))
        return xp.minimum(xp.maximum(tau, lower), upper)


ACTIVATIONS = ("softsign", "tanh", "elu", "relu")


@dataclass
class MLPMotorModel:
    """Learned actuator: torque = torque_scale * MLP([pos_err history * pos_scale, vel history * vel_scale]).

    ``history`` lists which past control steps feed the network (0 = now), as
    in Isaac Lab's ``input_idx``. Inputs are ordered position errors first,
    then velocities, each oldest-to-newest in ``history`` order.
    """

    weights: list[np.ndarray]  # W_i with shape (in, out)
    biases: list[np.ndarray]
    history: tuple[int, ...] = (0, 1, 2)
    pos_scale: float = 1.0
    vel_scale: float = 1.0
    torque_scale: float = 1.0
    activation: str = "softsign"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.biases) or not self.weights:
            raise ValueError("MLPMotorModel needs one bias per weight matrix")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.weights[0].shape[0] != 2 * len(self.history):
            raise ValueError(f"first layer takes {self.weights[0].shape[0]} inputs; history needs {2 * len(self.history)}")
        if self.weights[-1].shape[1] != 1:
            raise ValueError("last layer must output one torque per joint")

    @property
    def depth(self) -> int:
        return max(self.history) + 1

    def __call__(self, pos_err_hist: Any, vel_hist: Any) -> Any:
        """``pos_err_hist``, ``vel_hist``: (depth, E, J), index 0 = now. Returns torque (E, J)."""
        xp = array_namespace(pos_err_hist)
        cols = [pos_err_hist[i, ...] * self.pos_scale for i in self.history]
        cols += [vel_hist[i, ...] * self.vel_scale for i in self.history]
        x = xp.stack(cols, axis=-1)  # (E, J, 2H)
        dev = device(x)
        n = len(self.weights)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = xp.matmul(x, xp.asarray(w, dtype=x.dtype, device=dev)) + xp.asarray(b, dtype=x.dtype, device=dev)
            if k < n - 1:
                x = _act(xp, x, self.activation)
        return x[..., 0] * self.torque_scale

    # -- persistence ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        arrays: dict[str, np.ndarray] = {f"W{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        arrays["history"] = np.asarray(self.history)
        arrays["scales"] = np.asarray([self.pos_scale, self.vel_scale, self.torque_scale])
        arrays["activation"] = np.asarray(self.activation)
        np.savez(path, **arrays)  # type: ignore[arg-type]
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> "MLPMotorModel":
        with np.load(path, allow_pickle=False) as z:
            n = sum(1 for k in z.files if k.startswith("W"))
            s = z["scales"]
            return cls([z[f"W{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)],
                       tuple(int(i) for i in z["history"]), float(s[0]), float(s[1]), float(s[2]), str(z["activation"]))


def _act(xp: Any, x: Any, kind: str) -> Any:
    if kind == "softsign":
        return x / (1.0 + xp.abs(x))
    if kind == "tanh":
        return xp.tanh(x)
    if kind == "relu":
        return xp.maximum(x, xp.zeros_like(x))
    return xp.where(x > 0, x, xp.exp(xp.minimum(x, xp.zeros_like(x))) - 1.0)  # elu


def _like(value: Any, ref: Any) -> Any:
    """Broadcast a scalar or (J,) value to ``ref``'s namespace, dtype and device."""
    xp = array_namespace(ref)
    if isinstance(value, (int, float)):
        return xp.full(ref.shape, float(value), dtype=ref.dtype, device=device(ref))
    try:
        if array_namespace(value) is xp:
            return value
    except TypeError:
        pass
    return xp.asarray(np.asarray(value, dtype=np.float64), dtype=ref.dtype, device=device(ref))
