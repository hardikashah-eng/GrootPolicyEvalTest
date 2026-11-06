"""Randomization spaces and scenario generation.

A sweep config names the parameters a simulator exposes and how to vary
them. Scenarios are generated deterministically from a seed, so any failing
episode can be replayed exactly.

Example (YAML)::

    seed: 7
    episodes: 200
    task: "reach the target"
    params:
      friction:      {uniform: [0.2, 1.2]}
      mass:          {uniform: [0.5, 2.0]}
      sensor_noise:  {uniform: [0.0, 0.05]}
      action_delay:  {choice: [0, 1, 2, 3]}
      obstacle:      {choice: [false, true], weights: [0.7, 0.3]}
      gravity:       {fixed: 9.81}

To mirror a real test set, list named scenarios instead of sampling them; the
names are what ``sweep --export-pairs`` writes, so they join with the robot's
results in ``groot-eval correlate``::

    task: "pick up the cup"
    params:
      friction: {fixed: 0.8}       # fixed params apply to every named scenario
    repeats: 3                     # seeds seed, seed+1, seed+2 for each
    scenarios:
      - {name: pick_place_00, seed: 0, params: {mass: 0.4}}
      - {name: pick_place_01, seed: 1}
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.types import Scenario


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # uniform | loguniform | choice | fixed
    args: tuple

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "uniform":
            lo, hi = self.args
            return float(rng.uniform(lo, hi))
        if self.kind == "loguniform":
            lo, hi = self.args
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if self.kind == "choice":
            options, weights = self.args
            i = rng.choice(len(options), p=weights)
            return _plain(options[i])
        if self.kind == "fixed":
            return _plain(self.args[0])
        raise ValueError(f"unknown kind {self.kind}")

    def grid(self, points: int) -> list[Any]:
        if self.kind in ("uniform", "loguniform"):
            lo, hi = self.args
            if self.kind == "uniform":
                return [float(v) for v in np.linspace(lo, hi, points)]
            return [float(v) for v in np.geomspace(lo, hi, points)]
        if self.kind == "choice":
            return [_plain(v) for v in self.args[0]]
        return [_plain(self.args[0])]


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_param(name: str, spec: Any) -> ParamSpec:
    if not isinstance(spec, dict):
        return ParamSpec(name, "fixed", (spec,))
    if "uniform" in spec:
        lo, hi = spec["uniform"]
        if lo > hi:
            raise ValueError(f"{name}: uniform low > high")
        return ParamSpec(name, "uniform", (float(lo), float(hi)))
    if "loguniform" in spec:
        lo, hi = spec["loguniform"]
        if lo <= 0 or lo > hi:
            raise ValueError(f"{name}: loguniform needs 0 < low <= high")
        return ParamSpec(name, "loguniform", (float(lo), float(hi)))
    if "choice" in spec:
        options = list(spec["choice"])
        if not options:
            raise ValueError(f"{name}: empty choice")
        weights = spec.get("weights")
        if weights is None:
            weights = [1 / len(options)] * len(options)
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(options) or weights.min() < 0 or weights.sum() == 0:
            raise ValueError(f"{name}: bad weights")
        return ParamSpec(name, "choice", (options, weights / weights.sum()))
    if "fixed" in spec:
        return ParamSpec(name, "fixed", (spec["fixed"],))
    raise ValueError(f"{name}: expected one of uniform, loguniform, choice, fixed")


@dataclass
class SweepConfig:
    params: list[ParamSpec]
    episodes: int = 100
    seed: int = 0
    task: str = ""
    mode: str = "random"  # random | grid
    grid_points: int = 3
    repeats: int = 1  # grid mode: seeds per grid cell; named scenarios: runs per scenario
    named: list[Scenario] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SweepConfig":
        params = [parse_param(k, v) for k, v in (data.get("params") or {}).items()]
        task = str(data.get("task", ""))
        named = []
        fixed = {p.name: p.sample(np.random.default_rng(0)) for p in params if p.kind == "fixed"}
        for i, entry in enumerate(data.get("scenarios") or []):
            if not isinstance(entry, dict) or not entry.get("name"):
                raise ValueError(f"scenarios[{i}] needs a name")
            named.append(Scenario(seed=int(entry.get("seed", i)), params={**fixed, **(entry.get("params") or {})},
                                  task=str(entry.get("task", task)), name=str(entry["name"])))
        if len({s.name for s in named}) != len(named):
            raise ValueError("scenario names must be unique")
        if named and any(p.kind != "fixed" for p in params):
            raise ValueError("with named scenarios, params may only be fixed (put varying values in each scenario)")
        return cls(
            params=params,
            episodes=int(data.get("episodes", 100)),
            seed=int(data.get("seed", 0)),
            task=task,
            mode=str(data.get("mode", "random")),
            grid_points=int(data.get("grid_points", 3)),
            repeats=int(data.get("repeats", 1)),
            named=named,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SweepConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def scenarios(self) -> Iterator[Scenario]:
        if self.named:
            for s in self.named:
                for r in range(self.repeats):
                    yield Scenario(seed=s.seed + r, params=dict(s.params), task=s.task, name=s.name)
            return
        rng = np.random.default_rng(self.seed)
        if self.mode == "grid":
            axes = [p.grid(self.grid_points) for p in self.params]
            for combo in itertools.product(*axes):
                for _ in range(self.repeats):
                    yield Scenario(
                        seed=int(rng.integers(0, 2**31 - 1)),
                        params={p.name: v for p, v in zip(self.params, combo)},
                        task=self.task,
                    )
            return
        if self.mode != "random":
            raise ValueError(f"unknown mode {self.mode}")
        for _ in range(self.episodes):
            yield Scenario(
                seed=int(rng.integers(0, 2**31 - 1)),
                params={p.name: p.sample(rng) for p in self.params},
                task=self.task,
            )
