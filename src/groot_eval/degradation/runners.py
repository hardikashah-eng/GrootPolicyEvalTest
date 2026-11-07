"""Run and summarize aging sweeps, DOF-loss matrices and fault campaigns."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from groot_eval import analysis
from groot_eval.closed_loop import EvalRun, evaluate
from groot_eval.degradation.faults import parse_schedule
from groot_eval.degradation.wear import DOF_MODES, WearConfig, WearEnv, aged_scenarios
from groot_eval.envs.base import ComponentSpec
from groot_eval.stats import mcnemar_exact, wilson_interval
from groot_eval.types import EpisodeResult, Scenario

EnvFactory = Callable[[], Any]


def _rate(succ: int, n: int, confidence: float) -> dict[str, Any]:
    lo, hi = wilson_interval(succ, n, confidence)
    return {"successes": succ, "episodes": n, "rate": succ / n if n else math.nan, "ci_low": lo, "ci_high": hi}


def _mean_metric(results: list[EpisodeResult], key: str) -> float:
    vals = [r.metrics.get(key, math.nan) for r in results]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


# -- aging ----------------------------------------------------------------------------------
def run_aging(make_env: EnvFactory, policy: Any, base: list[Scenario], cfg: WearConfig, profiles: list[str],
              components: dict[str, ComponentSpec], progress: Callable[[str, EvalRun], None] | None = None,
              recorder: Any = None) -> dict[str, EvalRun]:
    runs = {}
    for prof in profiles:
        env = WearEnv(make_env())
        run = evaluate(env, policy, aged_scenarios(base, cfg, prof, components), env_name=f"wear:{prof}",
                       recorder=recorder)
        run.meta = {"profile": prof, "age_hours": cfg.profiles[prof], "unsupported": sorted(env.unsupported)}
        env.close()
        runs[prof] = run
        if progress:
            progress(prof, run)
    return runs


def aging_summary(runs: dict[str, EvalRun], gate: float | None = None, confidence: float = 0.95,
                  metrics: tuple[str, ...] = ("final_distance", "path_length", "torque_saturation_steps",
                                              "joint_limit_hits")) -> dict[str, Any]:
    profiles = list(runs)
    ref = profiles[0]
    ref_by = {r.scenario.seed: r.success for r in runs[ref].results}
    rows = []
    for prof in profiles:
        run = runs[prof]
        row = {"profile": prof, "age_hours": run.meta.get("age_hours"), **_rate(run.successes, run.trials, confidence),
               "mean_steps": float(np.mean([r.steps for r in run.results])) if run.results else math.nan,
               "metrics": {m: _mean_metric(run.results, m) for m in metrics},
               "failure_reasons": run.summary(confidence)["failure_reasons"],
               "unsupported": run.meta.get("unsupported", [])}
        if prof != ref:
            b = sum(1 for r in run.results if ref_by.get(r.scenario.seed) and not r.success)
            c = sum(1 for r in run.results if ref_by.get(r.scenario.seed) is False and r.success)
            row["paired"] = {"vs": ref, "lost": b, "gained": c, "p_value": mcnemar_exact(b, c),
                             "significant": mcnemar_exact(b, c) < 1 - confidence}
        rows.append(row)
    margin = None
    if gate is not None:
        failing = [r for r in rows if r["ci_low"] < gate]
        passing = [r for r in rows if r["ci_low"] >= gate]
        margin = {"gate": gate, "first_failing": failing[0]["profile"] if failing else None,
                  "first_failing_age_hours": failing[0]["age_hours"] if failing else None,
                  "last_passing": passing[-1]["profile"] if passing else None}
    oldest = runs[profiles[-1]]
    attribution = [a for a in analysis.attribute_failures(oldest.results, confidence=confidence)
                   if a["param"].startswith("wear.") and a["param"] not in ("wear.profile", "wear.age_hours")]
    return {"reference": ref, "profiles": rows, "margin": margin, "attribution_oldest": attribution,
            "policy": oldest.policy}


def dof_loss_matrix(make_env: EnvFactory, policy: Any, base: list[Scenario], components: dict[str, ComponentSpec],
                    modes: tuple[str, ...] = DOF_MODES, confidence: float = 0.95) -> list[dict[str, Any]]:
    """Success with one joint locked, limp or range-limited, on an otherwise new robot."""
    joints = sorted((n for n, c in components.items() if c.kind == "joint"), key=_natural)
    if not joints:
        joints = sorted((n for n, c in components.items() if c.kind == "motor"), key=_natural)
    nominal = evaluate(WearEnv(make_env()), policy, base)
    rows = [{"joint": "(none)", "cells": {m: _rate(nominal.successes, nominal.trials, confidence) for m in modes}}]
    for j in joints:
        cells = {}
        for mode in modes:
            scen = [Scenario(seed=s.seed, params={**s.params, f"wear.{j}.dof_mode": mode}, task=s.task, name=s.name)
                    for s in base]
            run = evaluate(WearEnv(make_env()), policy, scen)
            cells[mode] = _rate(run.successes, run.trials, confidence)
        rows.append({"joint": j, "cells": cells})
    return rows


def _natural(name: str) -> tuple:
    import re

    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


# -- faults ------------------------------------------------------------------------------------
def _fault_view(r: EpisodeResult) -> dict[str, Any]:
    """Per-episode fault outcome, also for episodes the runner ended itself (a NaN action)."""
    m = r.metrics
    sched = parse_schedule(r.scenario.params.get("faults"))
    if "fault.safe_outcome" in m:
        return {"started": m.get("fault.started", 0) > 0, "safe": bool(m["fault.safe_outcome"]),
                "unsafe": bool(m["fault.unsafe"]), "controlled": bool(m["fault.controlled_stop"]),
                "reaction": m.get("fault.reaction_steps", math.nan), "recovery": m.get("fault.recovery_steps", math.nan),
                "saturation": m.get("fault.saturation_steps", math.nan), "sched": sched}
    started = any(f.start <= r.steps for f in sched)
    bad = (r.failure_reason or "").startswith(("non_finite_action", "policy_error"))
    return {"started": started, "safe": r.success, "unsafe": bool(started and bad), "controlled": False,
            "reaction": math.nan, "recovery": math.nan, "saturation": math.nan, "sched": sched}


def fault_summary(run: EvalRun, confidence: float = 0.95) -> dict[str, Any]:
    views = [(r, _fault_view(r)) for r in run.results]
    hit = [(r, v) for r, v in views if v["started"]]

    def block(items: list[tuple[EpisodeResult, dict[str, Any]]]) -> dict[str, Any]:
        n = len(items)
        if not n:
            return {"episodes": 0}
        reasons: dict[str, int] = defaultdict(int)
        for r, _ in items:
            if not r.success:
                reasons[(r.failure_reason or "unknown").split(":")[0]] += 1
        med = lambda k: float(np.nanmedian([v[k] for _, v in items])) if any(math.isfinite(v[k]) for _, v in items) else math.nan  # noqa: E731
        return {"episodes": n,
                "success": _rate(sum(r.success for r, _ in items), n, confidence),
                "safe": _rate(sum(v["safe"] for _, v in items), n, confidence),
                "unsafe": _rate(sum(v["unsafe"] for _, v in items), n, confidence),
                "controlled_stop_rate": sum(v["controlled"] for _, v in items) / n,
                "reaction_steps_median": med("reaction"), "recovery_steps_median": med("recovery"),
                "saturation_steps_median": med("saturation"),
                "failure_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}

    by_type: dict[str, list] = defaultdict(list)
    by_target_kind: dict[str, list] = defaultdict(list)
    by_count: dict[int, list] = defaultdict(list)
    for r, v in hit:
        started = [f for f in v["sched"] if f.start <= r.steps]
        for t in sorted({f.type for f in started}):
            by_type[t].append((r, v))
        for k in sorted({f.target.split("[")[0] for f in started}):
            by_target_kind[k].append((r, v))
        by_count[len(started)].append((r, v))
    return {"policy": run.policy, "episodes": len(views), "faulted_episodes": len(hit),
            "no_fault_started": len(views) - len(hit),
            "overall": block(hit),
            "by_type": {k: block(v) for k, v in sorted(by_type.items())},
            "by_component": {k: block(v) for k, v in sorted(by_target_kind.items())},
            "by_count": {k: block(v) for k, v in sorted(by_count.items())}}


def fault_gate(summary: dict[str, Any], min_safe: float | None, max_unsafe: float | None) -> list[str]:
    """Failed gate descriptions; empty = pass.

    Safety gates must be *shown* to hold: the lower bound of the safe-outcome
    rate must reach ``min_safe`` and the upper bound of the unsafe-event rate
    must stay under ``max_unsafe``. Too few episodes therefore fails the gate
    (``groot-eval plan`` says how many are needed).
    """
    out = []
    o = summary["overall"]
    if not o.get("episodes"):
        return ["no episode had a fault start; nothing was tested"]
    if min_safe is not None and o["safe"]["ci_low"] < min_safe:
        out.append(f"safe-outcome rate {o['safe']['rate']:.1%} has CI low {o['safe']['ci_low']:.1%} < {min_safe:.1%}")
    if max_unsafe is not None and o["unsafe"]["ci_high"] > max_unsafe:
        out.append(f"unsafe-event rate {o['unsafe']['rate']:.1%} has CI high {o['unsafe']['ci_high']:.1%} > {max_unsafe:.1%}")
    return out
