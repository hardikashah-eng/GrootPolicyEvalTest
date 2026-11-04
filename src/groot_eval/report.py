"""Markdown reports a reviewer can read without running anything."""

from __future__ import annotations

import math
from typing import Any


def _pct(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _num(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def closed_loop_report(summary: dict[str, Any], attribution: list[dict[str, Any]], worst: list[dict[str, Any]]) -> str:
    s = summary
    lines = [
        f"# Closed-loop evaluation: `{s['policy']}` on `{s['env']}`",
        "",
        f"**Success rate {_pct(s['success_rate'])}** ({s['successes']}/{s['episodes']} episodes), "
        f"{int(100 * s['confidence'])}% CI {_pct(s['ci_low'])} – {_pct(s['ci_high'])}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Episodes | {s['episodes']} |",
        f"| Mean steps | {_num(s['mean_steps'], 1)} |",
        f"| Policy latency p50 / p95 / p99 (ms) | {_num(s['latency_ms_p50'])} / {_num(s['latency_ms_p95'])} / {_num(s['latency_ms_p99'])} |",
        "",
        "## Failure reasons",
        "",
    ]
    if s["failure_reasons"]:
        lines += ["| Reason | Episodes |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["failure_reasons"].items()]
    else:
        lines.append("No failures.")
    lines += ["", "## What drives failure", "",
              "Success rate by bucket of each randomized parameter. **Significant** means the best and worst "
              "buckets' confidence intervals do not overlap.", ""]
    if attribution:
        lines += ["| Parameter | Gap | Significant | Worst bucket | Buckets (success rate, n) |", "|---|---|---|---|---|"]
        for a in attribution:
            buckets = "; ".join(f"{b['bucket']}: {_pct(b['success_rate'])} (n={b['episodes']})" for b in a["buckets"])
            lines.append(f"| {a['param']} | {_pct(a['gap'])} | {'**yes**' if a['significant'] else 'no'} | {a['worst_bucket']} | {buckets} |")
    else:
        lines.append("No randomized parameters varied.")
    lines += ["", "## Worst cases", ""]
    if worst:
        lines += ["| Scenario | Seed | Reason | Steps | Params |", "|---|---|---|---|---|"]
        for w in worst:
            params = ", ".join(f"{k}={_num(v, 3) if isinstance(v, float) else v}" for k, v in w["params"].items())
            lines.append(f"| `{w['scenario_id']}` | {w['seed']} | {w['failure_reason']} | {w['steps']} | {params} |")
        lines += ["", "Replay any of these exactly with `groot-eval regress` using the exported suite."]
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def comparison_report(baseline: dict[str, Any], candidate: dict[str, Any], z: float, regression: bool) -> str:
    verdict = "**REGRESSION**: candidate is significantly worse." if regression else (
        "**Improvement**: candidate is significantly better." if z > 1.6449 else "No significant difference.")
    rows = []
    for label, s in (("baseline", baseline), ("candidate", candidate)):
        rows.append(f"| {label} | `{s['policy']}` | {_pct(s['success_rate'])} | {_pct(s['ci_low'])} – {_pct(s['ci_high'])} | "
                    f"{s['successes']}/{s['episodes']} | {_num(s['latency_ms_p95'])} |")
    reasons = sorted(set(baseline["failure_reasons"]) | set(candidate["failure_reasons"]))
    lines = [
        "# Checkpoint comparison", "", verdict, "",
        "| Run | Policy | Success | CI | n | Latency p95 (ms) |", "|---|---|---|---|---|---|", *rows, "",
        f"Two-proportion z = {z:.2f} (one-sided 95% threshold: ±1.645).", "",
        "## Failure reasons", "", "| Reason | Baseline | Candidate |", "|---|---|---|",
        *[f"| {r} | {baseline['failure_reasons'].get(r, 0)} | {candidate['failure_reasons'].get(r, 0)} |" for r in reasons],
    ]
    return "\n".join(lines) + "\n"


def open_loop_report(summary: dict[str, Any]) -> str:
    s = summary
    horizon = s["per_horizon_mse"]
    lines = [
        f"# Open-loop evaluation: `{s['policy']}`", "",
        f"Action MSE **{_num(s['mse_mean'], 4)}** (95% bootstrap CI {_num(s['mse_ci_low'], 4)} – {_num(s['mse_ci_high'], 4)}) "
        f"over {s['episodes']} episodes, horizon {s['horizon']}. MAE {_num(s['mae_mean'], 4)}.", "",
        "## Error by action group", "", "| Group | MSE |", "|---|---|",
        *[f"| {k} | {_num(v, 4)} |" for k, v in s["per_group_mse"].items()], "",
        "## Error by step within the chunk", "",
        "Error that grows along the chunk means the policy's internal model drifts from the demonstrator's; "
        "it predicts how much executing long chunks open-loop will hurt.", "",
        "| Step | MSE |", "|---|---|",
        *[f"| {i} | {_num(v, 4)} |" for i, v in enumerate(horizon)], "",
        "## Worst episodes", "", "| Episode | Task | MSE | Max abs error |", "|---|---|---|---|",
        *[f"| {e['episode_index']} | {e['task']} | {_num(e['mse'], 4)} | {_num(e['max_abs_error'], 3)} |" for e in s["worst_episodes"]],
    ]
    return "\n".join(lines) + "\n"


def regression_report(r: dict[str, Any]) -> str:
    def table(entries: list[dict[str, Any]]) -> list[str]:
        if not entries:
            return ["None.", ""]
        return ["| Scenario | Seed | Was | Now |", "|---|---|---|---|",
                *[f"| `{e['scenario_id']}` | {e['seed']} | {e['was'] or 'pass'} | {e['now'] or 'pass'} |" for e in entries], ""]

    lines = [
        f"# Regression suite: `{r['policy']}`", "",
        f"Suite from `{r['suite_source_policy']}`, {r['cases']} cases. Gate: **{'PASS' if r['passed_gate'] else 'FAIL'}** "
        f"({len(r['broken_anchors'])} previously passing anchors broke).", "",
        f"## Fixed ({len(r['fixed'])})", "", *table(r["fixed"]),
        f"## Still failing ({len(r['still_failing'])})", "", *table(r["still_failing"]),
        f"## Broken anchors ({len(r['broken_anchors'])})", "", *table(r["broken_anchors"]),
    ]
    return "\n".join(lines)


def correlation_report(c: dict[str, Any]) -> str:
    lines = [
        "# Sim-to-real correlation", "",
        f"* Ranking agreement (Spearman over policies): **{_num(c['ranking_spearman'])}**",
        f"* Outcome agreement on {c['matched_pairs']} matched policy/scenario pairs: **{_pct(c['outcome_agreement'])}**",
        f"* Share of pairs where sim is more optimistic than real: {_pct(c['sim_more_optimistic_share'])}", "",
        "| Policy | Sim | Real | Gap (sim − real) |", "|---|---|---|---|",
    ]
    for p in c["policies"]:
        sim, real = p["sim"], p["real"]
        lines.append(f"| {p['policy']} | {_pct(sim['rate'])} [{_pct(sim['ci_low'])}–{_pct(sim['ci_high'])}] n={sim['trials']} | "
                     f"{_pct(real['rate'])} [{_pct(real['ci_low'])}–{_pct(real['ci_high'])}] n={real['trials']} | {_pct(p.get('gap'))} |")
    lines += ["", "## Scenarios where sim and real disagree most", "",
              "Candidates for better assets, physics tuning, or domain randomization.", "",
              "| Scenario | Mean abs disagreement |", "|---|---|",
              *[f"| {d['scenario_id']} | {_num(d['mean_abs_disagreement'])} |" for d in c["most_divergent_scenarios"]]]
    return "\n".join(lines) + "\n"
