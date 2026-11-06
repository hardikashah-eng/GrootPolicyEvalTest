"""Regression suites: failing scenarios become tests that every new checkpoint must face.

``export_suite`` writes the failures of a run (exact seed and parameters) to a
JSON file. ``run_suite`` replays them against another policy and reports which
are now fixed, which still fail, and which previously passing anchor scenarios
broke. Replaying a suite on the policy that produced it should reproduce the
same outcomes; ``check_determinism`` verifies that, because a suite on a
nondeterministic simulator gates nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from groot_eval.closed_loop import EvalRun, run_episode
from groot_eval.envs.base import Env
from groot_eval.policies.base import Policy
from groot_eval.types import Scenario

SUITE_VERSION = 1


def export_suite(run: EvalRun, path: str | Path, anchors: int = 5, note: str = "") -> dict[str, Any]:
    """Save every failure plus a few passing 'anchor' scenarios that must keep passing."""
    failures = [r for r in run.results if not r.success]
    passing = [r for r in run.results if r.success][:anchors]
    suite = {
        "version": SUITE_VERSION,
        "source_policy": run.policy,
        "env": run.env,
        "note": note,
        "cases": [
            {"expect": "pass", "origin": "failure", "failure_reason": r.failure_reason, "scenario": r.scenario.to_dict()}
            for r in failures
        ]
        + [{"expect": "pass", "origin": "anchor", "failure_reason": None, "scenario": r.scenario.to_dict()} for r in passing],
    }
    Path(path).write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return suite


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    if suite.get("version") != SUITE_VERSION:
        raise ValueError(f"unsupported suite version {suite.get('version')}")
    return suite


def run_suite(env: Env, policy: Policy, suite: dict[str, Any], recorder: Any = None) -> dict[str, Any]:
    fixed, still_failing, broken_anchors, anchors_ok = [], [], [], []
    for case in suite["cases"]:
        scenario = Scenario.from_dict(case["scenario"])
        result = run_episode(env, policy, scenario, recorder)
        entry = {"scenario_id": scenario.scenario_id, "params": scenario.params, "seed": scenario.seed,
                 "was": case.get("failure_reason"), "now": None if result.success else result.failure_reason}
        if case["origin"] == "failure":
            (fixed if result.success else still_failing).append(entry)
        else:
            (anchors_ok if result.success else broken_anchors).append(entry)
    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "suite_source_policy": suite.get("source_policy"),
        "cases": len(suite["cases"]),
        "fixed": fixed,
        "still_failing": still_failing,
        "broken_anchors": broken_anchors,
        "anchors_ok": len(anchors_ok),
        "passed_gate": not broken_anchors,
    }


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
