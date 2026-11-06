"""The whole sim-to-real debugging loop on a CPU, against a SYNTHETIC "robot".

    uv run python examples/run_sim2real_demo.py [output_dir]

There is no robot here. The "real" side is the Reach2D simulator with physics
the debugging tools are not told about (``HIDDEN`` below), sensor noise, about
110 ms of inference latency handled the way an asynchronous robot executor
handles it, and a unit bug in its "driver". The point is to show each tool
finding its part of the gap, and to check the answers against known values:

1. record the robot and a naive sim in one trace format, three checkpoints each;
2. ``check`` catches the driver's unit bug before anything else is compared;
3. ``gap`` ranks the remaining differences and calls out the missing latency;
4. ``replay`` + a parameter grid recovers the hidden physics from the robot's trajectories;
5. a calibrated sim (fitted physics + the robot's recorded latencies) is compared again;
6. ``correlate`` scores sim-vs-real agreement per checkpoint and scenario, before and after.

Writes every report plus ``SUMMARY.md`` to the output directory (default
``demo-output/sim2real``). Takes about a minute on a laptop CPU.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

from groot_eval.cli import main
from groot_eval.open_loop import write_lerobot_dataset
from groot_eval.trace import TraceSet, import_real

ROOT = Path(__file__).resolve().parent.parent

# What the robot "really" is. Only the robot runs see these values.
HIDDEN = {"friction": 1.0, "actuator_gain": 0.9, "action_delay": 1}
SENSOR_NOISE = 0.02  # treated as known (e.g. measured with the robot at rest), not fitted
ROBOT_LATENCY = "normal:110,25"  # ms; about 2-3 control steps at 20 Hz
CHECKPOINTS = {  # label -> (policy spec, policy args)
    "ckpt_a": ("pd", []),
    "ckpt_b": ("pd-avoid", []),
    "ckpt_c": ("pd-chunk", ["avoid=2.0", "execute_steps=8"]),
}
TRACED = "ckpt_c"  # the chunked checkpoint is the one compared step by step


def step(title: str, argv: list[str], expect: int = 0) -> None:
    print(f"\n=== {title}\n$ groot-eval {' '.join(argv)}")
    code = main(argv)
    if code != expect:
        raise SystemExit(f"step failed: exit {code}, expected {expect}")


def scenarios() -> list[dict]:
    """A 16-scenario test set, named the way a robot lab would name it."""
    rng = np.random.default_rng(5)
    out = []
    for i in range(16):
        params = {"obstacle": bool(i % 2), "obstacle_radius": round(float(rng.uniform(0.1, 0.2)), 3)}
        out.append({"name": f"reach_cell_{i:02d}", "seed": 100 + i, "params": params})
    return out


def write_config(path: Path, fixed: dict, repeats: int) -> Path:
    cfg = {"task": "move the end effector to the target", "repeats": repeats,
           "params": {k: {"fixed": v} for k, v in fixed.items()}, "scenarios": scenarios()}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def relabel(trace: Path, domain: str) -> None:
    p = trace / "meta" / "trace_info.json"
    info = json.loads(p.read_text(encoding="utf-8"))
    info["domain"] = domain
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")


def buggy_export(clean: Path, out: Path) -> Path:
    """What the robot's logger wrote before the driver bug was fixed: velocity x 57.3 (rad -> deg)."""
    ts = TraceSet.open(clean)
    vel = ts.state_slices()["velocity"]
    eps = []
    for meta, f in zip(ts.episode_meta(), ts.frames()):
        states = np.stack(f["observation.state"].to_numpy())
        states[:, vel] *= 180 / np.pi
        eps.append({"task": meta["tasks"][0], "states": states, "actions": np.stack(f["action"].to_numpy())})
    raw = write_lerobot_dataset(out / "raw", eps, fps=ts.fps, modality=ts.dataset.modality)
    path, _ = import_real(raw, out / "trace")
    return path


def gap_max(path: Path) -> tuple[float, str]:
    sig = json.loads((path / "gap.json").read_text(encoding="utf-8"))["signals"]
    top = max(sig, key=lambda s: s["w1_norm"])
    return top["w1_norm"], f"{top['group']}:{top['signal']}"


def corr(path: Path) -> dict:
    from groot_eval.correlation import correlate, load_pairs

    return correlate(load_pairs(path))


def run(out: Path) -> dict:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    robot_cfg = write_config(out / "robot_cell.yaml", {**HIDDEN, "sensor_noise": SENSOR_NOISE}, repeats=1)
    naive_cfg = write_config(out / "sim_naive.yaml", {"sensor_noise": SENSOR_NOISE}, repeats=3)
    real_pairs, before, after = out / "pairs_real.csv", out / "pairs_before.csv", out / "pairs_after.csv"

    # 1. Record the robot and the naive sim.
    for label, (spec, args) in CHECKPOINTS.items():
        pa = sum((["--policy-arg", a] for a in args), [])
        trace = ["--trace", str(out / "traces" / "real")] if label == TRACED else []
        step(f"Robot (SYNTHETIC): {label}", ["sweep", "--policy", spec, *pa, "--config", str(robot_cfg),
             "--latency-mode", ROBOT_LATENCY, "--during-inference", "continue",
             "--out", str(out / "runs" / f"real_{label}"), "--export-pairs", str(out / "tmp_real.csv"),
             "--pairs-policy", label, *trace])
        trace = ["--trace", str(out / "traces" / "sim_naive")] if label == TRACED else []
        step(f"Naive sim: {label}", ["sweep", "--policy", spec, *pa, "--config", str(naive_cfg),
             "--out", str(out / "runs" / f"naive_{label}"), "--export-pairs", str(before), "--pairs-policy", label, *trace])
    relabel(out / "traces" / "real", "real")
    # The robot's rows: same scenario names, domain=real.
    rows = (out / "tmp_real.csv").read_text(encoding="utf-8").replace(",sim,", ",real,")
    real_pairs.write_text(rows, encoding="utf-8")
    (out / "tmp_real.csv").unlink()
    body = "".join(rows.splitlines(keepends=True)[1:])
    with open(before, "a", encoding="utf-8", newline="") as fh:
        fh.write(body)

    # 2. Contract check: the logger's export first, then the fixed one.
    bad = buggy_export(out / "traces" / "real", out / "traces" / "real_buggy_export")
    step("Contract check against the robot's first export (expect a units error)",
         ["check", "--reference", str(bad), "--traces", str(out / "traces" / "sim_naive"),
          "--out", str(out / "check_buggy.md")], expect=1)
    step("Contract check after the driver fix",
         ["check", "--reference", str(out / "traces" / "real"), "--traces", str(out / "traces" / "sim_naive"),
          "--out", str(out / "check.md")])

    # 3. Where is the gap?
    step("Gap: naive sim vs robot", ["gap", str(out / "traces" / "sim_naive"), str(out / "traces" / "real"),
                                     "--out", str(out / "gap_naive")])

    # 4. Which physics reproduces the robot's trajectories?
    step("Replay robot actions in sim and fit physics", ["replay", str(out / "traces" / "real"), "--config",
         str(ROOT / "configs" / "reach2d_sysid.yaml"), "--out", str(out / "replay")])
    fitted = json.loads((out / "replay" / "replay.json").read_text(encoding="utf-8"))["ranked"][0]["params"]
    fitted = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fitted.items()}

    # 5. Calibrated sim: fitted physics plus the robot's recorded latency.
    cal_cfg = write_config(out / "sim_calibrated.yaml", {**fitted, "sensor_noise": SENSOR_NOISE}, repeats=3)
    for label, (spec, args) in CHECKPOINTS.items():
        pa = sum((["--policy-arg", a] for a in args), [])
        trace = ["--trace", str(out / "traces" / "sim_calibrated")] if label == TRACED else []
        step(f"Calibrated sim: {label}", ["sweep", "--policy", spec, *pa, "--config", str(cal_cfg),
             "--latency-mode", f"trace:{out / 'traces' / 'real'}", "--during-inference", "continue",
             "--out", str(out / "runs" / f"cal_{label}"), "--export-pairs", str(after), "--pairs-policy", label, *trace])
    with open(after, "a", encoding="utf-8", newline="") as fh:
        fh.write(body)
    step("Gap: calibrated sim vs robot", ["gap", str(out / "traces" / "sim_calibrated"), str(out / "traces" / "real"),
                                          "--out", str(out / "gap_calibrated")])

    # 6. Does sim now agree with the robot on outcomes?
    step("Correlate before calibration", ["correlate", str(before), "--out", str(out / "correlation_before.md")])
    step("Correlate after calibration", ["correlate", str(after), "--out", str(out / "correlation_after.md")])

    result = {"hidden": HIDDEN, "fitted": fitted, "gap_naive": gap_max(out / "gap_naive"),
              "gap_calibrated": gap_max(out / "gap_calibrated"),
              "corr_before": corr(before), "corr_after": corr(after),
              "replay": json.loads((out / "replay" / "replay.json").read_text(encoding="utf-8"))}
    summary = summary_markdown(result)
    (out / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"All reports are in {out}")
    return result


def _pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def _rate(c: dict, label: str, domain: str) -> str:
    p = next(p for p in c["policies"] if p["policy"] == label)[domain]
    return f"{_pct(p['rate'])} ({p['successes']}/{p['trials']})"


def summary_markdown(r: dict) -> str:
    rows = "\n".join(f"| `{k}` | {v} | {r['fitted'].get(k, '-')} |" for k, v in r["hidden"].items())
    rep, cb, ca = r["replay"], r["corr_before"], r["corr_after"]
    best = rep["ranked"][0]
    lines = [
        "# Sim-to-real debugging demo (SYNTHETIC)",
        "",
        "> **SYNTHETIC.** The \"robot\" is the Reach2D simulator with hidden physics, sensor noise "
        f"{SENSOR_NOISE}, inference latency `{ROBOT_LATENCY}` ms executed asynchronously, and a logger that "
        "exported velocity x 57.3. It shows the tools finding a gap whose causes are known. It is not a real robot result.",
        "",
        "## 1. Contract check",
        "",
        "The first export from the robot fails `groot-eval check` with a units error on `velocity` "
        "(`check_buggy.md`). After the fix, the check is clean (`check.md`). Comparing physics before this "
        "would have compared a unit bug.",
        "",
        "## 2. Gap, naive sim vs robot",
        "",
        f"Largest gap: **{r['gap_naive'][0]:.2f} σ** on `{r['gap_naive'][1]}` (`gap_naive/gap.md`). The timing "
        "section reports the robot's 2-3 steps of inference latency that the sim never sees.",
        "",
        "## 3. Replay and parameter fit",
        "",
        f"Replaying the robot's actions in the default sim: tracking error {rep['baseline']['score']:.3f} nRMSE. "
        f"The best of {len(rep['ranked'])} settings: {best['score']:.3f} (`replay/replay.md`).",
        "",
        "| Parameter | Hidden (true) | Recovered |",
        "|---|---|---|",
        rows,
        "",
        "Replay recovers the actuator's own delay, not the inference latency: recorded actions are the ones the "
        "robot applied, so latency is already inside them. Latency is modelled separately in the next step.",
        "",
        "## 4. Calibrated sim vs robot",
        "",
        "Fitted physics, the known sensor noise, and `--latency-mode trace:traces/real --during-inference continue`.",
        "",
        f"Largest gap: **{r['gap_calibrated'][0]:.2f} σ** on `{r['gap_calibrated'][1]}` "
        f"(was {r['gap_naive'][0]:.2f} σ; `gap_calibrated/gap.md`).",
        "",
        "## 5. Outcomes per checkpoint",
        "",
        "| Checkpoint | Robot | Naive sim | Calibrated sim |",
        "|---|---|---|---|",
        *[f"| {k} | {_rate(cb, k, 'real')} | {_rate(cb, k, 'sim')} | {_rate(ca, k, 'sim')} |" for k in CHECKPOINTS],
        "",
        f"Outcome agreement on {ca['matched_pairs']} matched checkpoint/scenario pairs: "
        f"{_pct(cb['outcome_agreement'])} naive, **{_pct(ca['outcome_agreement'])} calibrated** "
        "(`correlation_before.md`, `correlation_after.md`). Checkpoint ranking agreement (Spearman): "
        f"{cb['ranking_spearman']:.2f} naive, **{ca['ranking_spearman']:.2f} calibrated**.",
        "",
        "The naive sim says `ckpt_b` (obstacle avoidance) never fails; on the robot it collides in half the "
        "scenarios, because it steers on observations that are 2-3 steps old. Only the sim with the robot's "
        "latency shows that.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo-output" / "sim2real")
