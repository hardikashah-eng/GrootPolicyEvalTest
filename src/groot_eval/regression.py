"""Regression suites: failing scenarios become tests that every new checkpoint must face.

``export_suite`` writes the failures of a run (exact seed and parameters) to a
JSON file, with each case's continuous metrics (final distance, clearance...).
``run_suite`` replays them against another policy and reports which are now
fixed, which still fail, and which previously passing anchor scenarios broke.

Making the gate trustworthy:

* **Repeats.** ``repeats=k`` runs every case ``k`` times on the same scenario.
  A case whose outcome differs between repeats is *flaky*: it is reported but
  never decides the gate, because one unlucky run must not block a release.
* **Quarantine.** ``find_flaky`` runs a suite repeatedly against the baseline
  policy and lists the cases that are flaky even there; pass that list as
  ``quarantine`` and those cases are reported but never gate.
* **Margins.** A binary pass/fail near the success threshold flips on noise.
  ``tolerances`` (``{"final_distance": ("lower", 0.02)}``) also fails an anchor
  that still passes but whose metric got worse than its stored baseline by
  more than the tolerance.

Replaying a suite on the policy that produced it should reproduce the same
outcomes; ``check_determinism`` verifies that, because a suite on a
nondeterministic simulator gates nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from groot_eval.closed_loop import EvalRun, run_episode
from groot_eval.envs.base import Env
from groot_eval.policies.base import Policy
from groot_eval.types import EpisodeResult, Scenario

SUITE_VERSION = 1
Tolerances = dict[str, tuple[str, float]]  # metric -> ("lower"|"higher" is better, allowed worsening)


def export_suite(run: EvalRun, path: str | Path, anchors: int = 5, note: str = "",
                 manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Save every failure plus a few passing 'anchor' scenarios that must keep passing."""
    failures = [r for r in run.results if not r.success]
    passing = [r for r in run.results if r.success][:anchors]

    def case(r: EpisodeResult, origin: str) -> dict[str, Any]:
        return {"expect": "pass", "origin": origin, "failure_reason": r.failure_reason,
                "scenario": r.scenario.to_dict(), "steps": r.steps,
                "metrics": {k: v for k, v in r.metrics.items() if math.isfinite(v)}}

    suite = {
        "version": SUITE_VERSION,
        "source_policy": run.policy,
        "env": run.env,
        "note": note,
        "manifest": manifest,
        "cases": [case(r, "failure") for r in failures] + [case(r, "anchor") for r in passing],
    }
    Path(path).write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return suite


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    if suite.get("version") != SUITE_VERSION:
        raise ValueError(f"unsupported suite version {suite.get('version')}")
    return suite


def parse_tolerances(specs: list[str] | None) -> Tolerances:
    """``final_distance:lower:0.02`` -> {"final_distance": ("lower", 0.02)}."""
    out: Tolerances = {}
    for spec in specs or []:
        try:
            name, better, tol = spec.split(":")
            value = float(tol)
        except ValueError:
            raise ValueError(f"metric tolerance must be NAME:lower|higher:TOL, got {spec!r}") from None
        if better not in ("lower", "higher") or not value >= 0:
            raise ValueError(f"metric tolerance must be NAME:lower|higher:TOL, got {spec!r}")
        out[name] = (better, value)
    return out


def _worsening(better: str, was: float, now: float) -> float:
    return now - was if better == "lower" else was - now


def run_suite(env: Env, policy: Policy, suite: dict[str, Any], recorder: Any = None, *, repeats: int = 1,
              quarantine: set[str] | None = None, tolerances: Tolerances | None = None) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    quarantine, tolerances = set(quarantine or ()), dict(tolerances or {})
    fixed, still_failing, broken_anchors, anchors_ok, flaky, degraded, quarantined = [], [], [], [], [], [], []
    for case in suite["cases"]:
        scenario = Scenario.from_dict(case["scenario"])
        results = [run_episode(env, policy, scenario, recorder) for _ in range(repeats)]
        wins = sum(r.success for r in results)
        last_fail = next((r.failure_reason for r in results if not r.success), None)
        entry: dict[str, Any] = {"scenario_id": scenario.scenario_id, "name": scenario.name, "params": scenario.params,
                                 "seed": scenario.seed, "was": case.get("failure_reason"), "now": last_fail,
                                 "pass_rate": wins / repeats, "origin": case["origin"]}
        deltas = {}
        for name, (better, tol) in tolerances.items():
            was = (case.get("metrics") or {}).get(name)
            vals = [r.metrics[name] for r in results if name in r.metrics and math.isfinite(r.metrics[name])]
            if was is None or not vals:
                continue
            now = sum(vals) / len(vals)
            deltas[name] = {"was": was, "now": now, "worse_by": _worsening(better, was, now), "tolerance": tol}
        entry["metrics"] = deltas
        if scenario.scenario_id in quarantine:
            quarantined.append(entry)
        elif 0 < wins < repeats:
            flaky.append(entry)
        elif case["origin"] == "failure":
            (fixed if wins == repeats else still_failing).append(entry)
        elif wins == 0:
            broken_anchors.append(entry)
        else:
            over = sorted(k for k, d in deltas.items() if d["worse_by"] > d["tolerance"])
            if over:
                degraded.append({**entry, "over_tolerance": over})
            anchors_ok.append(entry)
    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "suite_source_policy": suite.get("source_policy"),
        "cases": len(suite["cases"]),
        "repeats": repeats,
        "fixed": fixed,
        "still_failing": still_failing,
        "broken_anchors": broken_anchors,
        "degraded_anchors": degraded,
        "flaky": flaky,
        "quarantined": quarantined,
        "anchors_ok": len(anchors_ok),
        "passed_gate": not broken_anchors and not degraded,
    }


def find_flaky(env: Env, policy: Policy, suite: dict[str, Any], repeats: int = 5) -> list[dict[str, Any]]:
    """Cases whose outcome is not stable on ``policy`` itself (run it on the baseline)."""
    r = run_suite(env, policy, suite, repeats=repeats)
    return [{"scenario_id": e["scenario_id"], "name": e["name"], "pass_rate": e["pass_rate"]} for e in r["flaky"]]


def check_determinism(env: Env, policy: Policy, scenarios: list[Scenario], repeats: int = 2) -> list[str]:
    """Return scenario ids whose outcome differs between identical replays."""
    flaky = []
    for scenario in scenarios:
        outcomes = {
            (r.success, r.failure_reason, r.steps)
            for r in (run_episode(env, policy, scenario) for _ in range(repeats))
        }
        if len(outcomes) > 1:
            flaky.append(scenario.scenario_id)
    return flaky
