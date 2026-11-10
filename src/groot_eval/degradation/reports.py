"""Markdown reports for aging, DOF loss and fault campaigns."""

from __future__ import annotations

import math
from typing import Any


def _p(x: float | None) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _n(x: float | None, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}g}"


def _rate(d: dict[str, Any]) -> str:
    return f"{_p(d['rate'])} [{_p(d['ci_low'])}, {_p(d['ci_high'])}] (n={d['episodes']})"


def aging_markdown(s: dict[str, Any], dof: list[dict[str, Any]] | None = None) -> str:
    lines = [f"# Aging report: {s['policy']}", ""]
    rows = s["profiles"]
    first, last = rows[0], rows[-1]
    lines.append(f"Success goes from {_p(first['rate'])} on a `{first['profile']}` robot to {_p(last['rate'])} at "
                 f"`{last['profile']}` ({last['age_hours']:g} h), on the same {first['episodes']} scenarios and units.")
    m = s.get("margin")
    if m:
        if m["first_failing"]:
            lines.append(f"**Robustness margin:** the success lower bound first drops below {_p(m['gate'])} at "
                         f"`{m['first_failing']}` ({m['first_failing_age_hours']:g} h); last passing profile: "
                         f"`{m['last_passing'] or 'none'}`.")
        else:
            lines.append(f"**Robustness margin:** every profile keeps the success lower bound at or above {_p(m['gate'])}.")
    lines += ["", "## Degradation curve", "",
              "Paired test: McNemar on the same scenario and unit across ages (lost = succeeded on the reference "
              "robot, failed on this one).", "",
              "| Profile | Age (h) | Success [95% CI] | Lost / gained vs ref | p | Mean steps | Final dist | Torque-sat steps | Limit hits |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pr = r.get("paired")
        paired = f"{pr['lost']} / {pr['gained']}" if pr else "reference"
        pv = f"**{pr['p_value']:.3g}**" if pr and pr["significant"] else (f"{pr['p_value']:.3g}" if pr else "–")
        mt = r["metrics"]
        lines.append(f"| {r['profile']} | {r['age_hours']:g} | {_rate(r)} | {paired} | {pv} | {_n(r['mean_steps'])} | "
                     f"{_n(mt.get('final_distance'))} | {_n(mt.get('torque_saturation_steps'))} | {_n(mt.get('joint_limit_hits'))} |")
    reasons = ", ".join(f"{k}: {v}" for k, v in last["failure_reasons"].items()) or "none"
    lines += ["", f"Failure reasons at `{last['profile']}`: {reasons}."]
    unsupported = sorted({u for r in rows for u in r.get("unsupported", [])})
    if unsupported:
        lines.append(f"Effects with no physics lever or fallback in this env (not applied): {', '.join(unsupported)}.")
    att = s.get("attribution_oldest") or []
    if att:
        lines += ["", f"## Which wear drives failure at `{last['profile']}`", "",
                  "Units at the same age differ (log-normal spread), so per-unit wear values can be ranked like any "
                  "randomized parameter. Only gaps whose intervals do not overlap are significant.", "",
                  "| Wear parameter | Gap | Significant | Worst bucket |", "|---|---|---|---|"]
        for a in att[:12]:
            lines.append(f"| `{a['param']}` | {_p(a['gap'])} | {'**yes**' if a['significant'] else 'no'} | {a['worst_bucket']} |")
    if dof:
        modes = list(dof[0]["cells"])
        lines += ["", "## Loss of a degree of freedom", "",
                  "One joint at a time, on an otherwise new robot. `locked`: seized at its current angle; `limp`: motor "
                  "off, joint swings under gravity; `limited`: 30% of its range. The policy still sends commands to that "
                  "joint, as on a real robot.", "",
                  "| Joint | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
        for row in dof:
            lines.append(f"| {row['joint']} | " + " | ".join(_rate(row["cells"][m]) for m in modes) + " |")
        spof = [row["joint"] for row in dof[1:] if all(row["cells"][m]["ci_high"] < 0.2 for m in modes)]
        if spof:
            lines.append(f"\nSingle points of failure for this skill (every mode below 20%): {', '.join(spof)}.")
    return "\n".join(lines) + "\n"


def faults_markdown(s: dict[str, Any], gate_failures: list[str] | None = None) -> str:
    o = s["overall"]
    lines = [f"# Fault-injection report: {s['policy']}", ""]
    if not o.get("episodes"):
        return "\n".join(lines + ["No fault started before the episodes ended; nothing was tested. Move the fault "
                                  "start window earlier or lengthen the task."]) + "\n"
    lines.append(f"{s['faulted_episodes']} of {s['episodes']} episodes had at least one fault start "
                 f"({s['no_fault_started']} ended before their first fault). Safe outcome means the task succeeded or the "
                 "robot came to a controlled stop, with no collision, instability or joint-limit hit after the fault.")
    if gate_failures is not None:
        lines.append("")
        lines.append("**Gate: FAIL** – " + "; ".join(gate_failures) if gate_failures else "**Gate: pass**")
    lines += ["", "| | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction (median steps) | Recovery (median steps) |",
              "|---|---|---|---|---|---|---|---|"]

    def row(label: str, b: dict[str, Any]) -> str:
        return (f"| {label} | {b['episodes']} | {_rate(b['success'])} | {_rate(b['safe'])} | {_rate(b['unsafe'])} | "
                f"{_p(b['controlled_stop_rate'])} | {_n(b['reaction_steps_median'])} | {_n(b['recovery_steps_median'])} |")

    lines.append(row("**all faulted**", o))
    lines += ["", "## By fault type", "", "An episode with several faults counts under each type it saw.", "",
              "| Fault | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |",
              "|---|---|---|---|---|---|---|---|"]
    worst = sorted(s["by_type"].items(), key=lambda kv: kv[1]["safe"]["rate"])
    for k, b in worst:
        lines.append(row(f"`{k}`", b))
    lines += ["", "## By component", "", "| Component | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |",
              "|---|---|---|---|---|---|---|---|"]
    for k, b in s["by_component"].items():
        lines.append(row(f"`{k}`", b))
    if len(s["by_count"]) > 1:
        lines += ["", "## Degradation with concurrent faults", "",
                  "A cliff between one and two faults means the system has no redundancy for the second.", "",
                  "| Faults started | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, b in s["by_count"].items():
            lines.append(row(str(k), b))
    reasons = ", ".join(f"{k}: {v}" for k, v in o["failure_reasons"].items()) or "none"
    lines += ["", f"Failure reasons in faulted episodes: {reasons}. `non_finite_action` means the policy passed a NaN "
              "reading straight through to its command."]
    return "\n".join(lines) + "\n"
