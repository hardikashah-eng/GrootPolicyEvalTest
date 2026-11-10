"""Shared plumbing for the wear and fault wrappers.

Both wrappers sit between the policy and any :class:`~groot_eval.envs.base.Env`:

* commands are degraded on the way in (gain, clipping, dead zones, dropped or
  delayed packets, disabled motors);
* observations are degraded on the way out (bias, noise, quantization,
  dropouts, frozen readings, camera effects);
* effects that need physics (friction, backlash, a seized joint) go through the
  env's optional ``set_component_param(name, key, value)`` hook.

An env that implements ``components()`` (``Arm3`` does) names its parts. For
any other env, components are derived from the first observation: one
``motor[i]`` per action dimension, one ``sensor.<key>`` per ``state.*`` key and
one ``camera.<key>`` per ``video.*`` key. Those have no physics levers, so only
the boundary effects apply.
"""

from __future__ import annotations

import fnmatch
from typing import Any

import numpy as np

from groot_eval.envs.base import ComponentSpec, Obs


def discover(env: Any, obs: Obs, action_dim: int | None) -> dict[str, ComponentSpec]:
    comps = getattr(env, "components", None)
    if callable(comps):
        return dict(comps())
    out: dict[str, ComponentSpec] = {}
    for i in range(action_dim or 0):
        out[f"motor[{i}]"] = ComponentSpec(f"motor[{i}]", "motor", action=(i,))
    for key, value in obs.items():
        if key.startswith("state."):
            n = int(np.asarray(value).size)
            out[f"sensor.{key[6:]}"] = ComponentSpec(f"sensor.{key[6:]}", "sensor", state={key: (0, n)})
        elif key.startswith("video."):
            out[f"camera.{key[6:]}"] = ComponentSpec(f"camera.{key[6:]}", "camera", state={key: (0, 0)})
    if action_dim:
        out["bus"] = ComponentSpec("bus", "bus", action=tuple(range(action_dim)))
    return out


def match(components: dict[str, ComponentSpec], pattern: str) -> list[str]:
    """Component names matching a glob such as ``encoder[*]`` or ``motor[1]``."""
    if pattern in components:
        return [pattern]
    esc = pattern.replace("[", "\x00").replace("]", "\x01")
    names = {n.replace("[", "\x00").replace("]", "\x01"): n for n in components}
    hits = [names[k] for k in names if fnmatch.fnmatchcase(k, esc)]
    return sorted(hits) or sorted(n for n in components if components[n].kind == pattern)


def env_action_dim(env: Any) -> int | None:
    dim = getattr(env, "action_dim", None)
    return int(dim) if dim else None


def try_physics(env: Any, name: str, key: str, value: Any) -> bool:
    """Apply a physics lever if the env has it; False means use the boundary fallback."""
    setter = getattr(env, "set_component_param", None)
    if setter is None:
        return False
    try:
        setter(name, key, value)
        return True
    except (NotImplementedError, KeyError):
        return False


def get_physics(env: Any, name: str, key: str, default: Any) -> Any:
    getter = getattr(env, "get_component_param", None)
    if getter is None:
        return default
    try:
        return getter(name, key)
    except (NotImplementedError, KeyError):
        return default


def state_slices(spec: ComponentSpec, obs: Obs) -> list[tuple[str, slice]]:
    out = []
    for key, (a, b) in spec.state.items():
        if key in obs and key.startswith("state."):
            n = int(np.asarray(obs[key]).size)
            out.append((key, slice(a, b if b > a else n)))
    return out


class EnvWrapper:
    """Forwards everything the runner, replay and checks use to the wrapped env."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:  # dt, lengths, set_state, ...
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def max_steps(self) -> int:
        return self.env.max_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self.env.max_steps = value

    def close(self) -> None:
        self.env.close()

    def _derive(self, obs: Obs) -> Obs:
        derive = getattr(self.env, "derive_obs", None)
        return derive(obs) if callable(derive) else obs


def split_params(params: dict[str, Any], prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(params for this wrapper, params for the inner env)."""
    mine = {k: v for k, v in params.items() if k == prefix or k.startswith(prefix + ".")}
    rest = {k: v for k, v in params.items() if k not in mine}
    return mine, rest


def probe_components(env: Any, action_dim: int | None = None, scenario: Any = None) -> dict[str, ComponentSpec]:
    """Reset the (unwrapped) env once and list its components, for sampling wear and fault targets."""
    from groot_eval.types import Scenario

    obs = env.reset(scenario or Scenario(seed=0))
    return discover(env, obs, action_dim or env_action_dim(env))
