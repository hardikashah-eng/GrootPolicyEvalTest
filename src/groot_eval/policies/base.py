"""The policy interface, plus action chunking.

Foundation-model policies like GR00T predict a *chunk* of future actions per
inference call (an action horizon) and the controller executes some of them
before asking again. :class:`ChunkedPolicy` turns any chunk predictor into a
per-step policy, so evaluation code never has to care, and so the number of
executed steps per chunk is an explicit, sweepable setting.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

Obs = dict[str, Any]


@runtime_checkable
class Policy(Protocol):
    name: str

    def reset(self, task: str) -> None: ...

    def act(self, obs: Obs) -> np.ndarray:
        """Return one action for the current step."""


@runtime_checkable
class ChunkPredictor(Protocol):
    name: str

    def reset(self, task: str) -> None: ...

    def predict_chunk(self, obs: Obs) -> np.ndarray:
        """Return an ``(horizon, action_dim)`` array of future actions."""


class ChunkedPolicy:
    """Execute ``execute_steps`` actions of each predicted chunk, then re-plan."""

    def __init__(self, predictor: ChunkPredictor, execute_steps: int = 8) -> None:
        if execute_steps < 1:
            raise ValueError("execute_steps must be >= 1")
        self.predictor = predictor
        self.execute_steps = execute_steps
        self.name = f"{predictor.name}@exec{execute_steps}"
        self._chunk: np.ndarray | None = None
        self._i = 0
        self.inference_calls = 0

    def reset(self, task: str) -> None:
        self.predictor.reset(task)
        self._chunk, self._i = None, 0
        self.inference_calls = 0

    def act(self, obs: Obs) -> np.ndarray:
        if self._chunk is None or self._i >= min(self.execute_steps, len(self._chunk)):
            chunk = np.asarray(self.predictor.predict_chunk(obs), dtype=float)
            if chunk.ndim != 2 or len(chunk) == 0:
                raise ValueError(f"predict_chunk must return (horizon, dim); got shape {chunk.shape}")
            self._chunk, self._i = chunk, 0
            self.inference_calls += 1
        action = self._chunk[self._i]
        self._i += 1
        return action

    # Where the last returned action came from; trace recording reads these.
    @property
    def last_chunk_index(self) -> int:
        return self.inference_calls - 1

    @property
    def last_chunk_step(self) -> int:
        return self._i - 1
