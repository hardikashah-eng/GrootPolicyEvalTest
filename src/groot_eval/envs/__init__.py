"""Environment adapters, created from a short spec string.

* ``reach2d`` - the built-in CPU simulator.
* ``arm3`` - a planar N-joint arm with a gripper (wear, faults and DOF loss).
* ``isaaclab:<TaskId>`` - an Isaac Lab Gymnasium task (needs Isaac Sim / Isaac Lab).
* ``package.module:ClassName`` - any class implementing :class:`groot_eval.envs.base.Env`.
"""

from __future__ import annotations

import importlib
from typing import Any

from groot_eval.envs.base import Env


def make_env(spec: str, **kwargs: Any) -> Env:
    if spec == "reach2d":
        from groot_eval.envs.reach2d import Reach2DEnv

        return Reach2DEnv(**kwargs)
    if spec == "arm3":
        from groot_eval.envs.arm3 import Arm3Env

        return Arm3Env(**kwargs)
    if spec.startswith("isaaclab:"):
        from groot_eval.envs.isaaclab_env import IsaacLabEnv

        return IsaacLabEnv(task=spec.split(":", 1)[1], **kwargs)
    if ":" in spec:
        module, cls = spec.split(":", 1)
        return getattr(importlib.import_module(module), cls)(**kwargs)
    raise ValueError(f"unknown env spec {spec!r}")
