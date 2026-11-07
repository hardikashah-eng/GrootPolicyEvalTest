"""Wear and tear, DOF loss, fault injection, skill suites and data-fidelity gates on a CPU.

    uv run python examples/run_robustness_demo.py [output_dir]

Everything runs on the Arm3 simulator (a planar 3-joint arm with a gripper)
with two reference controllers: ``arm-ik`` (naive) and ``arm-ik-robust``
(detects faulty joints and works around them). The "real robot" in the
fidelity part is SYNTHETIC: the same simulator on a worn unit. The wear
profiles are placeholders. The point is to show what each report says and
that the tools find known, planted causes; it is not a result about any
particular robot.

1. ``aging``: the same 200 scenarios on a new, 1-week, 6-month, 1-, 2-, 3-year-old and end-of-life robot,
   plus the DOF-loss matrix;
2. ``faults``: one fault per episode, naive vs robust controller;
3. ``faults``: long-horizon task (8 waypoints) with several random faults, on a 1-year-old robot;
4. ``suite``: reach-and-grasp skill at min / nominal / max conditions (one-at-a-time, corners, LHS);
5. ``fidelity``: a units bug stops the gates at G1; once fixed, G3-G5 measure the worn "robot";
   system identification shows how much of the gap the sim's friction parameter can close.

Writes every report plus SUMMARY.md to the output directory (default
``demo-output/robustness``). Takes a few minutes on a laptop CPU.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from groot_eval.cli import main
from groot_eval.trace import TraceSet

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "configs"


def run(*args: str) -> int:
    print("$ groot-eval " + " ".join(args))
    code = main(list(args))
    print(f"  -> exit {code}")
    return code


def corrupt_velocity_units(src: Path, dst: Path, factor: float = 57.2958) -> None:
    """Copy a trace as a 'driver' that logs joint velocity in deg/s instead of rad/s."""
    shutil.copytree(src, dst)
    ts = TraceSet.open(dst)
    sl = ts.state_slices()["joint_vel"]
    for meta in ts.dataset.episodes:
        path = dst / "data" / "chunk-000" / f"episode_{meta['episode_index']:06d}.parquet"
        df = pd.read_parquet(path)
        df["observation.state"] = [np.concatenate([v[: sl.start], v[sl] * factor, v[sl.stop:]]) for v in df["observation.state"]]
        df.to_parquet(path)
    manifest = dst / "meta" / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def main_demo(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    codes = {}

    # 1. aging + DOF loss
    codes["aging"] = run("aging", "--env", "arm3", "--policy", "arm-ik", "--config", str(C / "arm3_reach.yaml"),
                         "--episodes", "200", "--wear", str(C / "wear/arm3_profiles.yaml"), "--dof-episodes", "40",
                         "--min-success-at", "year_1", "0.9", "--out", str(out / "aging"))

    # 2. single faults, naive vs robust
    for pol in ("arm-ik", "arm-ik-robust"):
        codes[f"faults_{pol}"] = run("faults", "--env", "arm3", "--policy", pol, "--episodes", "300",
                                     "--faults", str(C / "faults/arm3_single.yaml"), "--min-safe", "0.9",
                                     "--out", str(out / f"faults_single_{pol}"))

    # 3. long horizon, several random faults, on a 1-year-old robot
    codes["faults_long"] = run("faults", "--env", "arm3", "--env-arg", "max_steps=900", "--policy", "arm-ik-robust",
                               "--config", str(C / "arm3_long.yaml"), "--episodes", "150",
                               "--faults", str(C / "faults/arm3_long_horizon.yaml"),
                               "--wear", str(C / "wear/arm3_profiles.yaml"), "--profile", "year_1",
                               "--export-suite", "--out", str(out / "faults_long_horizon"))

    # 4. skill suite
    codes["suite"] = run("suite", "--env", "arm3", "--policy", "arm-ik", "--config", str(C / "suites/arm3_reach.yaml"),
                         "--min-corner", "0.5", "--out", str(out / "suite_reach"))

    # 5. data fidelity: fixed physics on both sides, so replay can be exact; the "robot" is a 3-year-old unit
    fixed = {"seed": 21, "episodes": 60, "params": {"payload_kg": {"fixed": 0.2}, "friction": {"fixed": 1.0}}}
    (out / "fixed.yaml").write_text(yaml.safe_dump(fixed), encoding="utf-8")
    run("sweep", "--env", "arm3", "--policy", "arm-ik", "--config", str(out / "fixed.yaml"),
        "--trace", str(out / "traces/sim"), "--out", str(out / "runs/sim"))
    run("sweep", "--env", "arm3", "--policy", "arm-ik", "--config", str(out / "fixed.yaml"),
        "--wear", str(C / "wear/arm3_profiles.yaml"), "--profile", "year_3",
        "--trace", str(out / "traces/robot_fixed"), "--out", str(out / "runs/robot"))
    run("fidelity", "--write-manifest", str(out / "traces/sim"), str(out / "traces/robot_fixed"))
    corrupt_velocity_units(out / "traces/robot_fixed", out / "traces/robot_bad_units")
    gates = C / "fidelity/arm3_gates.yaml"
    codes["fidelity_units_bug"] = run("fidelity", "--sim", str(out / "traces/sim"), "--real", str(out / "traces/robot_bad_units"),
                                      "--gates", str(gates), "--direction", "real2sim", "--env", "arm3",
                                      "--param", "payload_kg=0.2", "--param", "friction=1.0",
                                      "--out", str(out / "fidelity_units_bug"))
    codes["fidelity_worn_robot"] = run("fidelity", "--sim", str(out / "traces/sim"), "--real", str(out / "traces/robot_fixed"),
                                       "--gates", str(gates), "--direction", "real2sim", "--env", "arm3",
                                       "--param", "payload_kg=0.2", "--param", "friction=1.0",
                                       "--out", str(out / "fidelity_worn_robot"))
    sysid = {"mode": "grid", "grid_points": 8, "params": {"friction": {"uniform": [1.0, 8.0]}, "payload_kg": {"fixed": 0.2}}}
    (out / "sysid.yaml").write_text(yaml.safe_dump(sysid), encoding="utf-8")
    codes["fidelity_after_sysid"] = run("fidelity", "--sim", str(out / "traces/sim"), "--real", str(out / "traces/robot_fixed"),
                                        "--gates", str(gates), "--direction", "real2sim", "--env", "arm3",
                                        "--sysid-config", str(out / "sysid.yaml"), "--out", str(out / "fidelity_after_sysid"))
    codes["fidelity_self"] = run("fidelity", "--sim", str(out / "traces/sim"), "--real", str(out / "traces/sim"),
                                 "--gates", str(gates), "--env", "arm3", "--param", "payload_kg=0.2",
                                 "--param", "friction=1.0", "--out", str(out / "fidelity_self_check"))
    (out / "SUMMARY.md").write_text(summary(out, codes), encoding="utf-8")
    print(f"\nWrote {out / 'SUMMARY.md'}")


def summary(out: Path, codes: dict[str, int]) -> str:
    aging = json.loads((out / "aging/aging.json").read_text(encoding="utf-8"))
    fn = {p: json.loads((out / f"faults_single_{p}/faults.json").read_text(encoding="utf-8")) for p in ("arm-ik", "arm-ik-robust")}
    fl = json.loads((out / "faults_long_horizon/faults.json").read_text(encoding="utf-8"))
    su = json.loads((out / "suite_reach/suite.json").read_text(encoding="utf-8"))
    fid = {k: json.loads((out / k / "fidelity.json").read_text(encoding="utf-8"))
           for k in ("fidelity_units_bug", "fidelity_worn_robot", "fidelity_after_sysid", "fidelity_self_check")}
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = ["# Robustness demo (SYNTHETIC: Arm3 simulator, placeholder wear profiles)", "",
             "## Aging", "", "| Profile | Age (h) | Success [95% CI] | Lost vs new (McNemar p) |", "|---|---|---|---|"]
    for r in aging["profiles"]:
        pr = r.get("paired")
        lines.append(f"| {r['profile']} | {r['age_hours']:g} | {pct(r['rate'])} [{pct(r['ci_low'])}, {pct(r['ci_high'])}] | "
                     + (f"{pr['lost']} (p={pr['p_value']:.2g})" if pr else "reference") + " |")
    m = aging["margin"]
    lines += ["", f"Robustness margin at 90%: first failing profile `{m['first_failing']}`, last passing `{m['last_passing']}`. "
              f"Gate `--min-success-at year_1 0.9`: exit {codes['aging']}."]
    att = [a for a in aging["attribution_oldest"] if a["significant"]]
    if att:
        lines.append("Significant wear drivers at end of life: " + ", ".join(f"`{a['param']}`" for a in att) + ".")
    lines += ["", "## Single faults: naive vs robust controller", "",
              "| Controller | Success | Safe outcome [95% CI] | Unsafe | NaN commands |", "|---|---|---|---|---|"]
    for p, s in fn.items():
        o = s["overall"]
        nan = o["failure_reasons"].get("non_finite_action", 0)
        lines.append(f"| {p} | {pct(o['success']['rate'])} | {pct(o['safe']['rate'])} [{pct(o['safe']['ci_low'])}, "
                     f"{pct(o['safe']['ci_high'])}] | {pct(o['unsafe']['rate'])} | {nan} |")
    worst = sorted(fn["arm-ik-robust"]["by_type"].items(), key=lambda kv: kv[1]["safe"]["rate"])[:3]
    lines += ["", "Worst fault types for the robust controller: " + ", ".join(f"`{k}` (safe {pct(v['safe']['rate'])})" for k, v in worst) + "."]
    lines += ["", "## Long horizon, several random faults, 1-year-old robot", "", "| Faults started | Episodes | Success | Safe outcome |", "|---|---|---|---|"]
    for k, b in fl["by_count"].items():
        lines.append(f"| {k} | {b['episodes']} | {pct(b['success']['rate'])} | {pct(b['safe']['rate'])} |")
    lines += ["", "## Skill suite (reach and grasp)", "",
              f"Overall {pct(su['overall']['rate'])}; all-nominal centre {pct(su['centre']['rate'])}. "
              f"One-at-a-time significant conditions: {', '.join(r['condition'] for r in su['oat'] if r['significant']) or 'none'}. "
              f"Worst two-way combination: {su['worst_pairs'][0]['pair']} at {pct(su['worst_pairs'][0]['rate'])}. "
              f"Gate `--min-corner 0.5`: exit {codes['suite']} ({len(su['gate_failures'] or [])} corners significantly below)."]
    lines += ["", "## Data-fidelity gates", "", "| Run | Exit | " + " | ".join(f"G{i}" for i in range(7)) + " |", "|---|---|" + "---|" * 7]
    for k, d in fid.items():
        st = {g["gate"][:2]: g["status"] for g in d["gates"]}
        lines.append(f"| {k} | {d['exit_code']} | " + " | ".join(st.get(f"G{i}", "–") for i in range(7)) + " |")
    g5 = {k: next((g["metrics"] for g in d["gates"] if g["gate"] == "G5_dynamics"), {}) for k, d in fid.items()}
    lines += ["", f"G5 replay error at 10 steps: self-check {g5['fidelity_self_check'].get('nrmse_at_horizon', float('nan')):.3f}, "
              f"worn robot vs nominal sim {g5['fidelity_worn_robot'].get('nrmse_at_horizon', float('nan')):.3f}, after fitting "
              f"friction {g5['fidelity_after_sysid'].get('nrmse_at_horizon', float('nan')):.3f} "
              f"(best friction {g5['fidelity_after_sysid'].get('params', {}).get('friction', float('nan')):.2f}, at the edge of the "
              "searched range, which G5 flags). One global friction parameter is absorbing per-joint wear (backlash, gain loss, "
              "encoder bias) it cannot represent: the sim needs per-joint wear parameters, which is what `--wear` provides."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main_demo(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo-output" / "robustness")
