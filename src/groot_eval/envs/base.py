"""The environment interface every simulator adapter implements.

It follows Gymnasium's step signature so Isaac Lab tasks fit with a thin
wrapper, but ``reset`` takes a full :class:`Scenario` so randomized physics
parameters are applied on every episode, not just the seed.
"""

from __future__ import annotations

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
