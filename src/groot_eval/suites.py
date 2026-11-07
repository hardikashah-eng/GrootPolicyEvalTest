"""Skill-focused condition suites: one skill, the conditions that stress it, min / nominal / max.

Random sweeps estimate average success well but rarely land in the corners,
and corners are where field failures happen. A suite names a skill (walk,
grasp, reach, ...) and its conditions, each with explicit ``min``, ``nominal``
and ``max`` levels, and builds scenarios with structured designs:

* ``oat`` (one at a time): each condition at min and at max, the rest nominal,
  plus the all-nominal centre. Cheap; answers "which single condition hurts".
* ``corners``: all 2^k min/max combinations plus the centre. Finds
  interactions (fast *and* curved *and* dim). Use for k up to about 6;
  ``corner_conditions`` limits it to a subset.
* ``lhs``: Latin-hypercube samples inside the box, for the average and for
  failure attribution over continuous ranges.

Example (YAML)::

    skill: reach
    task: "reach the target and close the gripper"
    design: [oat, corners, lhs]
    repeats: 10           # seeds per oat/corner cell
    lhs_samples: 200
    fixed: {sensor_noise: 0.0}
    conditions:
      target_radius: {min: 0.35, nominal: 0.6, max: 0.8}
      payload_kg:    {min: 0.0,  nominal: 0.1, max: 0.5}
      action_delay:  {min: 0,    nominal: 0,   max: 2, integer: true}

Condition names are env parameters (Isaac Lab: map them with
``param_setters``). Each scenario's ``name`` records its design cell
(``oat:payload_kg=max``, ``corner:target_radius=min,payload_kg=max``,
``lhs:0042``), which ``--trace`` writes into ``meta/episodes.jsonl``, so the
trace doubles as a per-skill, per-condition dataset.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval import analysis
from groot_eval.closed_loop import EvalRun
from groot_eval.stats import wilson_interval
from groot_eval.types import Scenario

DESIGNS = ("oat", "corners", "lhs")
LEVELS = ("min", "nominal", "max")


@dataclass(frozen=True)
class Condition:
    name: str
    min: Any
    nominal: Any
    max: Any
    integer: bool = False

    def level(self, which: str) -> Any:
        return {"min": self.min, "nominal": self.nominal, "max": self.max}[which]

    def at(self, u: float) -> Any:
        """Value at fraction ``u`` of [min, max] (numeric conditions only)."""
        v = float(self.min) + u * (float(self.max) - float(self.min))
        return int(round(v)) if self.integer else float(v)

    @property
    def numeric(self) -> bool:
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (self.min, self.nominal, self.max))


@dataclass
class SuiteConfig:
    skill: str
    conditions: list[Condition]
    design: list[str] = field(default_factory=lambda: ["oat", "corners"])
    task: str = ""
    repeats: int = 5
    lhs_samples: int = 100
    seed: int = 0
    fixed: dict[str, Any] = field(default_factory=dict)
    corner_conditions: list[str] | None = None
    metrics: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuiteConfig":
        conds = []
        for name, spec in (data.get("conditions") or {}).items():
            if not isinstance(spec, dict) or not {"min", "max"} <= set(spec):
                raise ValueError(f"condition {name}: needs min and max (and usually nominal)")
            nominal = spec.get("nominal", spec["min"])
            conds.append(Condition(name, spec["min"], nominal, spec["max"], bool(spec.get("integer", False))))
        if not conds:
            raise ValueError("suite has no conditions")
        design = data.get("design") or ["oat", "corners"]
        design = [design] if isinstance(design, str) else list(design)
        bad = set(design) - set(DESIGNS)
        if bad:
            raise ValueError(f"unknown designs {sorted(bad)}; use {DESIGNS}")
        cc = data.get("corner_conditions")
        if cc and set(cc) - {c.name for c in conds}:
            raise ValueError("corner_conditions must name conditions")
        return cls(str(data.get("skill", "skill")), conds, design, str(data.get("task", "")),
                   int(data.get("repeats", 5)), int(data.get("lhs_samples", 100)), int(data.get("seed", 0)),
                   dict(data.get("fixed") or {}), list(cc) if cc else None, list(data.get("metrics") or []))

    @classmethod
    def load(cls, path: str | Path) -> "SuiteConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def nominal(self) -> dict[str, Any]:
        return {c.name: c.nominal for c in self.conditions}

    def scenarios(self) -> list[Scenario]:
        rng = np.random.default_rng(self.seed)
        out: list[Scenario] = []

        def cell(name: str, params: dict[str, Any], repeats: int) -> None:
            for _ in range(repeats):
                out.append(Scenario(seed=int(rng.integers(0, 2**31 - 1)), params={**self.fixed, **params},
                                    task=self.task, name=name))

        if "oat" in self.design or "corners" in self.design:
            cell("centre", self.nominal(), self.repeats)
        if "oat" in self.design:
            for c in self.conditions:
                for lv in ("min", "max"):
                    if c.level(lv) != c.nominal:
                        cell(f"oat:{c.name}={lv}", {**self.nominal(), c.name: c.level(lv)}, self.repeats)
        if "corners" in self.design:
            cs = [c for c in self.conditions if not self.corner_conditions or c.name in self.corner_conditions]
            if len(cs) > 8:
                raise ValueError(f"{len(cs)} conditions make {2 ** len(cs)} corners; set corner_conditions")
            for combo in itertools.product(("min", "max"), repeat=len(cs)):
                label = ",".join(f"{c.name}={lv}" for c, lv in zip(cs, combo))
                cell(f"corner:{label}", {**self.nominal(), **{c.name: c.level(lv) for c, lv in zip(cs, combo)}},
                     self.repeats)
        if "lhs" in self.design:
            num = [c for c in self.conditions if c.numeric]
            n = self.lhs_samples
            cols = {c.name: (rng.permutation(n) + rng.random(n)) / n for c in num}
            for i in range(n):
                params = {**self.nominal(), **{c.name: c.at(float(cols[c.name][i])) for c in num}}
                for c in self.conditions:
                    if not c.numeric:
                        params[c.name] = c.level(str(rng.choice(LEVELS)))
                cell(f"lhs:{i:04d}", params, 1)
        return out


# -- analysis --------------------------------------------------------------------------------
def _rate(results: list[Any], confidence: float) -> dict[str, Any]:
    s, n = sum(r.success for r in results), len(results)
    lo, hi = wilson_interval(s, n, confidence)
    return {"successes": s, "episodes": n, "rate": s / n if n else math.nan, "ci_low": lo, "ci_high": hi}


def suite_summary(cfg: SuiteConfig, run: EvalRun, confidence: float = 0.95) -> dict[str, Any]:
    by_cell: dict[str, list] = {}
    for r in run.results:
        by_cell.setdefault(r.scenario.name, []).append(r)
    centre = _rate(by_cell.get("centre", []), confidence) if "centre" in by_cell else None
    oat = []
    for c in cfg.conditions:
        row: dict[str, Any] = {"condition": c.name, "levels": {}}
        for lv in ("min", "max"):
            key = f"oat:{c.name}={lv}"
            if key in by_cell:
                row["levels"][lv] = {"value": c.level(lv), **_rate(by_cell[key], confidence)}
        if centre:
            row["levels"]["nominal"] = {"value": c.nominal, **centre}
        if row["levels"]:
            rates = [v for v in row["levels"].values()]
            worst = min(rates, key=lambda v: v["rate"])
            best = max(rates, key=lambda v: v["rate"])
            row["gap"] = best["rate"] - worst["rate"]
            row["significant"] = worst["ci_high"] < best["ci_low"]
            row["worst_level"] = next(k for k, v in row["levels"].items() if v is worst)
            oat.append(row)
    oat.sort(key=lambda r: (not r["significant"], -r["gap"]))
    corners = sorted(([k.removeprefix("corner:"), _rate(v, confidence)] for k, v in by_cell.items() if k.startswith("corner:")),
                     key=lambda kv: kv[1]["rate"])
    pairs = _pair_table(cfg, by_cell, confidence)
    lhs = [r for r in run.results if r.scenario.name.startswith("lhs:")]
    attribution = analysis.attribute_failures(lhs, confidence=confidence) if len(lhs) >= 20 else []
    attribution = [a for a in attribution if a["param"] in {c.name for c in cfg.conditions}]
    metrics = {}
    for m in cfg.metrics:
        vals = [r.metrics.get(m) for r in run.results if r.metrics.get(m) is not None and math.isfinite(r.metrics.get(m))]
        if vals:
            metrics[m] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    return {"skill": cfg.skill, "policy": run.policy, "overall": _rate(run.results, confidence), "centre": centre,
            "oat": oat, "corners": [{"corner": k, **v} for k, v in corners], "worst_pairs": pairs,
            "lhs": _rate(lhs, confidence) if lhs else None, "attribution": attribution, "metrics": metrics}


def _pair_table(cfg: SuiteConfig, by_cell: dict[str, list], confidence: float) -> list[dict[str, Any]]:
    """Success for every 2-way min/max combination, pooled over the corners that contain it."""
    corner_runs = [(dict(p.split("=") for p in k.removeprefix("corner:").split(",")), v)
                   for k, v in by_cell.items() if k.startswith("corner:")]
    if not corner_runs:
        return []
    names = list(corner_runs[0][0])
    out = []
    for a, b in itertools.combinations(names, 2):
        for la, lb in itertools.product(("min", "max"), repeat=2):
            res = [r for lv, rs in corner_runs if lv[a] == la and lv[b] == lb for r in rs]
            out.append({"pair": f"{a}={la}, {b}={lb}", **_rate(res, confidence)})
    out.sort(key=lambda r: r["rate"])
    return out[:10]


def suite_gate(summary: dict[str, Any], min_oat: float | None, min_corner: float | None) -> list[str]:
    """A cell fails only when its success is *significantly* below the bar (CI high under it)."""
    out = []
    if min_oat is not None:
        for row in summary["oat"]:
            for lv, v in row["levels"].items():
                if v["ci_high"] < min_oat:
                    out.append(f"{row['condition']}={lv} ({v['value']}): {v['rate']:.1%}, CI high {v['ci_high']:.1%} < {min_oat:.0%}")
    if min_corner is not None:
        for c in summary["corners"]:
            if c["ci_high"] < min_corner:
                out.append(f"corner {c['corner']}: {c['rate']:.1%}, CI high {c['ci_high']:.1%} < {min_corner:.0%}")
    return out


def suite_markdown(s: dict[str, Any], cfg: SuiteConfig, gate_failures: list[str] | None = None) -> str:
    p = lambda x: "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"  # noqa: E731
    rate = lambda d: f"{p(d['rate'])} [{p(d['ci_low'])}, {p(d['ci_high'])}] (n={d['episodes']})" if d else "–"  # noqa: E731
    lines = [f"# Skill suite: {s['skill']} – {s['policy']}", "",
             f"Overall {rate(s['overall'])} across {', '.join(cfg.design)}."]
    if s["centre"]:
        lines.append(f"All conditions nominal: {rate(s['centre'])}.")
    if gate_failures is not None:
        lines.append("")
        lines.append(("**Gate: FAIL** – " + "; ".join(gate_failures[:8]) + (" …" if len(gate_failures) > 8 else ""))
                     if gate_failures else "**Gate: pass**")
    if s["oat"]:
        lines += ["", "## One condition at a time", "",
                  "Each condition at its min and max, every other condition nominal. Significant = the worst and best "
                  "levels' intervals do not overlap.", "",
                  "| Condition | Min | Nominal | Max | Gap | Significant |", "|---|---|---|---|---|---|"]
        for row in s["oat"]:
            cond = next(c for c in cfg.conditions if c.name == row["condition"])
            cell = lambda lv: (f"{row['levels'][lv]['value']}: {rate(row['levels'][lv])}" if lv in row["levels"]  # noqa: E731
                               else "= nominal" if cond.level(lv) == cond.nominal else "–")
            lines.append(f"| `{row['condition']}` | {cell('min')} | {cell('nominal')} | {cell('max')} | {p(row['gap'])} | "
                         f"{'**yes**' if row['significant'] else 'no'} |")
    if s["worst_pairs"]:
        lines += ["", "## Worst two-way combinations", "",
                  "Pooled over every corner that contains the pair; the other conditions vary over min and max.", "",
                  "| Combination | Success |", "|---|---|"]
        for r in s["worst_pairs"][:8]:
            lines.append(f"| {r['pair']} | {rate(r)} |")
    if s["corners"]:
        lines += ["", "## Worst corners", "", "| Corner (all conditions at min or max) | Success |", "|---|---|"]
        for c in s["corners"][:8]:
            lines.append(f"| {c['corner']} | {rate(c)} |")
    if s["lhs"]:
        lines += ["", f"## Space-filling sample (LHS): {rate(s['lhs'])}", ""]
        if s["attribution"]:
            lines += ["| Condition | Gap | Significant | Worst bucket |", "|---|---|---|---|"]
            for a in s["attribution"]:
                lines.append(f"| `{a['param']}` | {p(a['gap'])} | {'**yes**' if a['significant'] else 'no'} | {a['worst_bucket']} |")
    if s["metrics"]:
        lines += ["", "## Skill metrics", "", "| Metric | Mean | Min | Max |", "|---|---|---|---|"]
        for k, v in s["metrics"].items():
            lines.append(f"| {k} | {v['mean']:.4g} | {v['min']:.4g} | {v['max']:.4g} |")
    return "\n".join(lines) + "\n"
