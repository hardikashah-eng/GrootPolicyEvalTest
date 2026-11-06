"""Adapter for Isaac Lab tasks.

Isaac Lab registers its tasks with Gymnasium, but they only work after the
Omniverse app has been launched, and they run many environments in parallel
on the GPU. This adapter launches the app (headless by default), creates one
environment, and maps scenario parameters onto the task's config before each
episode. Per-episode physics changes go through ``param_setters``: callables
that receive the unwrapped env and a value, because the right lever (an event
term, a physics material, an asset's mass) is task-specific.

NOTE: not exercised in this repository's CI (no GPU / Isaac Sim). Written
against Isaac Lab 2.x conventions (``isaaclab.app.AppLauncher``,
``isaaclab_tasks.utils.parse_env_cfg``); older releases used the
``omni.isaac.lab`` package names, which are tried as a fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from groot_eval.types import Scenario

ParamSetter = Callable[[Any, Any], None]


def _launch_app(headless: bool):
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher  # Isaac Lab 1.x
    return AppLauncher(headless=headless).app


def _parse_env_cfg(task: str, num_envs: int):
    try:
        from isaaclab_tasks.utils import parse_env_cfg
    except ImportError:
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    return parse_env_cfg(task, num_envs=num_envs)


class IsaacLabEnv:
    def __init__(
        self,
        task: str,
        max_steps: int = 500,
        headless: bool = True,
        param_setters: dict[str, ParamSetter] | None = None,
        obs_key: str = "policy",
        success_fn: Callable[[Any, dict[str, Any]], bool] | None = None,
    ) -> None:
        self.app = _launch_app(headless)
        import gymnasium as gym

        try:
            import isaaclab_tasks  # noqa: F401  (registers tasks)
        except ImportError:
            import omni.isaac.lab_tasks  # noqa: F401

        self.task = task
        self.max_steps = max_steps
        self.obs_key = obs_key
        self.param_setters = param_setters or {}
        self.success_fn = success_fn
        cfg = _parse_env_cfg(task, num_envs=1)
        self.env = gym.make(task, cfg=cfg)
        self.t = 0

    def reset(self, scenario: Scenario) -> dict[str, Any]:
        unknown = set(scenario.params) - set(self.param_setters)
        if unknown:
            raise ValueError(f"no setter registered for parameters {sorted(unknown)} on {self.task}")
        for name, value in scenario.params.items():
            self.param_setters[name](self.env.unwrapped, value)
        obs, _ = self.env.reset(seed=scenario.seed)
        self.t = 0
        return self._obs(obs, scenario.task)

    def step(self, action: np.ndarray):
        import torch

        act = torch.as_tensor(np.asarray(action, dtype=np.float32)[None], device=self.env.unwrapped.device)
        obs, reward, terminated, truncated, extras = self.env.step(act)
        self.t += 1
        done_term, done_trunc = bool(terminated[0]), bool(truncated[0]) or self.t >= self.max_steps
        info: dict[str, Any] = {}
        if done_term or done_trunc:
            success = self.success_fn(self.env.unwrapped, extras) if self.success_fn else bool(done_term and not done_trunc)
            info = {"success": success}
            if not success:
                info["failure_reason"] = "timeout" if done_trunc else "terminated"
        return self._obs(obs, ""), float(reward[0]), done_term, done_trunc, info

    @property
    def dt(self) -> float | None:
        """Control period in seconds (Isaac Lab's ``step_dt`` = sim dt x decimation)."""
        return getattr(self.env.unwrapped, "step_dt", None)

    def close(self) -> None:
        self.env.close()
        self.app.close()

    def _obs(self, obs: Any, task: str) -> dict[str, Any]:
        tensor = obs[self.obs_key] if isinstance(obs, dict) else obs
        return {"state.policy": tensor[0].detach().cpu().numpy(), "annotation.task": task}
