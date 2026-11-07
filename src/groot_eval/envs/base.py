"""The environment interface every simulator adapter implements.

It follows Gymnasium's step signature so Isaac Lab tasks fit with a thin
wrapper, but ``reset`` takes a full :class:`Scenario` so randomized physics
parameters are applied on every episode, not just the seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from groot_eval.types import Scenario

Obs = dict[str, Any]


@runtime_checkable
class Env(Protocol):
    max_steps: int

    def reset(self, scenario: Scenario) -> Obs:
        """Start an episode with this scenario's seed and parameters."""

    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        """Advance one control step.

        Returns ``(obs, reward, terminated, truncated, info)``. When an episode
        ends, ``info`` should carry ``success`` (bool) and, on failure, a short
        ``failure_reason`` such as ``"collision"`` or ``"timeout"``. Optional
        ``info["metrics"]`` (dict of floats) is copied into the episode result.
        """

    def close(self) -> None: ...


@runtime_checkable
class StatefulEnv(Env, Protocol):
    """An env that can be put into a recorded state (needed by ``groot-eval replay``).

    ``set_state`` is optional: envs without it can still be replayed, but the
    replay then starts from whatever ``reset`` produces for the recorded seed.
    """

    def set_state(self, state: Obs, recent_actions: np.ndarray | None = None) -> Obs:
        """Move the current episode to ``state`` (``state.*`` keys) and return the new obs.

        ``recent_actions`` (oldest first) are the commands sent just before this
        state, so an env with actuation delay can refill its in-flight queue.
        Raise ``NotImplementedError`` if the env cannot do this.
        """


@dataclass(frozen=True)
class ComponentSpec:
    """Something that can wear out or fail: a motor, joint, sensor, the gripper, the comm bus.

    ``action`` lists the action indices the component drives and ``state`` the
    ``state.*`` / ``video.*`` slices it produces, so generic wrappers can
    degrade it at the command or observation boundary. ``physics_keys`` are the
    levers the env can change inside its own physics through
    ``set_component_param`` (optional; see :mod:`groot_eval.degradation`).
    """

    name: str
    kind: str  # joint | motor | encoder | gripper | camera | bus | sensor
    action: tuple[int, ...] = ()
    state: dict[str, tuple[int, int]] = field(default_factory=dict)
    physics_keys: tuple[str, ...] = ()
