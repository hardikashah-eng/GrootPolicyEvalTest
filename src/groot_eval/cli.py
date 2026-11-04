"""Command line: ``groot-eval <command>``. Run ``groot-eval -h`` for the list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from groot_eval import analysis, report, stats
from groot_eval.closed_loop import EvalRun, evaluate
from groot_eval.envs import make_env
from groot_eval.policies import make_policy
from groot_eval.randomization import SweepConfig
from groot_eval.types import EpisodeResult, Scenario


def _kv(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path}")


def _save_run(run: EvalRun, out: Path, confidence: float) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w", encoding="utf-8") as fh:
        for r in run.results:
            fh.write(json.dumps({**r.to_dict(), "latencies_ms": r.policy_latency_ms}) + "\n")
    summary = run.summary(confidence)
    attribution = analysis.attribute_failures(run.results, confidence=confidence)
    _write(out / "summary.json", json.dumps({**summary, "attribution": attribution}, indent=2))
    _write(out / "report.md", report.closed_loop_report(summary, attribution, analysis.worst_cases(run.results)))
    return summary


def _load_run(path: Path) -> EvalRun:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    run = EvalRun(policy=summary["policy"], env=summary["env"])
    for line in (path / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        run.results.append(EpisodeResult(Scenario.from_dict(d), d["success"], d["steps"], d["episode_return"],
                                         d["failure_reason"], d.get("latencies_ms", [])))
    return run


# -- commands -------------------------------------------------------------
def cmd_sweep(a: argparse.Namespace) -> int:
    cfg = SweepConfig.load(a.config)
    if a.episodes:
        cfg.episodes = a.episodes
    env = make_env(a.env, **_kv(a.env_arg))
    policy = make_policy(a.policy, **_kv(a.policy_arg))
    scenarios = list(cfg.scenarios())
    print(f"Evaluating {getattr(policy, 'name', a.policy)} on {a.env}: {len(scenarios)} episodes")

    def progress(i: int, r: EpisodeResult) -> None:
        if (i + 1) % max(1, len(scenarios) // 10) == 0:
            print(f"  {i + 1}/{len(scenarios)}")

    run = evaluate(env, policy, scenarios, env_name=a.env, progress=progress)
    env.close()
    out = Path(a.out)
    s = _save_run(run, out, a.confidence)
    if a.export_suite:
        from groot_eval.regression import export_suite

        suite = export_suite(run, out / "suite.json", anchors=a.anchors)
        print(f"  wrote {out / 'suite.json'} ({len(suite['cases'])} cases)")
    print(f"Success {100 * s['success_rate']:.1f}% [{100 * s['ci_low']:.1f}, {100 * s['ci_high']:.1f}] "
          f"({s['successes']}/{s['episodes']}); failures: {s['failure_reasons'] or 'none'}")
    if a.min_success is not None and s["ci_low"] < a.min_success:
        print(f"GATE FAIL: CI lower bound {s['ci_low']:.3f} < required {a.min_success}")
        return 1
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    base, cand = _load_run(Path(a.baseline)), _load_run(Path(a.candidate))
    bs, cs = base.summary(a.confidence), cand.summary(a.confidence)
    common = {r.scenario.scenario_id for r in base.results} & {r.scenario.scenario_id for r in cand.results}
    if len(common) < min(base.trials, cand.trials):
        print(f"warning: runs share only {len(common)} scenarios; compare sweeps generated from the same config and seed")
    z = stats.two_proportion_z(cs["successes"], cs["episodes"], bs["successes"], bs["episodes"])
    regressed = stats.is_regression((bs["successes"], bs["episodes"]), (cs["successes"], cs["episodes"]), a.confidence)
    text = report.comparison_report(bs, cs, z, regressed)
    if a.out:
        _write(Path(a.out), text)
    print(text)
    return 1 if regressed else 0


def cmd_regress(a: argparse.Namespace) -> int:
    from groot_eval.regression import load_suite, run_suite

    env = make_env(a.env, **_kv(a.env_arg))
    policy = make_policy(a.policy, **_kv(a.policy_arg))
    result = run_suite(env, policy, load_suite(a.suite))
    env.close()
    text = report.regression_report(result)
    if a.out:
        _write(Path(a.out), text)
    print(text)
    return 0 if result["passed_gate"] else 1


def cmd_open_loop(a: argparse.Namespace) -> int:
    from groot_eval.open_loop import LeRobotDataset, evaluate_open_loop

    ds = LeRobotDataset.open(a.dataset)
    kwargs = _kv(a.policy_arg)
    kwargs.pop("execute_steps", None)
    # Open-loop scoring wants the raw chunk predictor, not the per-step wrapper.
    policy = make_policy(a.policy, **kwargs)
    policy = getattr(policy, "predictor", policy)
    episodes = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    result = evaluate_open_loop(ds, policy, horizon=a.horizon, episodes=episodes, use_video=a.video)
    s = result.summary()
    out = Path(a.out)
    _write(out / "open_loop.json", json.dumps({**s, "per_episode": result.per_episode}, indent=2))
    _write(out / "open_loop.md", report.open_loop_report(s))
    print(f"MSE {s['mse_mean']:.4f} [{s['mse_ci_low']:.4f}, {s['mse_ci_high']:.4f}] over {s['episodes']} episodes")
    return 0


def cmd_make_demo_data(a: argparse.Namespace) -> int:
    """Record demonstrations from the reference controller in the Reach2D sim."""
    from groot_eval.envs.reach2d import Reach2DEnv
    from groot_eval.open_loop import write_lerobot_dataset
    from groot_eval.policies.scripted import PDPolicy

    env, demo = Reach2DEnv(max_steps=120), PDPolicy(kp=2.0, kd=1.2)
    rng = np.random.default_rng(a.seed)
    episodes = []
    for _ in range(a.episodes):
        scenario = Scenario(seed=int(rng.integers(0, 2**31 - 1)),
                            params={"friction": float(rng.uniform(0.3, 1.0)), "mass": float(rng.uniform(0.7, 1.5))},
                            task="move the end effector to the target")
        obs = env.reset(scenario)
        states, actions = [], []
        for _ in range(env.max_steps):
            act = demo.act(obs)
            states.append(np.concatenate([obs["state.position"], obs["state.velocity"], obs["state.target"]]))
            actions.append(act)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
        episodes.append({"task": scenario.task, "states": np.stack(states), "actions": np.stack(actions)})
    modality = {
        "state": {"position": {"start": 0, "end": 2}, "velocity": {"start": 2, "end": 4}, "target": {"start": 4, "end": 6}},
        "action": {"end_effector": {"start": 0, "end": 2}},
        "annotation": {"task": {}},
    }
    root = write_lerobot_dataset(a.out, episodes, fps=20, modality=modality)
    print(f"wrote {len(episodes)} episodes to {root}")
    return 0


def cmd_correlate(a: argparse.Namespace) -> int:
    from groot_eval.correlation import correlate, load_pairs

    c = correlate(load_pairs(a.pairs), a.confidence)
    text = report.correlation_report(c)
    if a.out:
        _write(Path(a.out), text)
    print(text)
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    n = stats.required_trials(a.margin, a.confidence, a.expected)
    print(f"{n} episodes give a ±{100 * a.margin:.1f}-point {int(100 * a.confidence)}% interval "
          f"at an expected success rate of {100 * a.expected:.0f}%.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="groot-eval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, needs_env: bool = True) -> None:
        if needs_env:
            sp.add_argument("--env", default="reach2d", help="reach2d | isaaclab:<Task> | module:Class")
            sp.add_argument("--env-arg", action="append", metavar="K=V")
        sp.add_argument("--policy", required=True, help="pd | pd-avoid | pd-chunk | groot://host:port | module:Class")
        sp.add_argument("--policy-arg", action="append", metavar="K=V")

    sp = sub.add_parser("sweep", help="closed-loop evaluation over a randomization sweep")
    common(sp)
    sp.add_argument("--config", required=True)
    sp.add_argument("--episodes", type=int, help="override the config's episode count")
    sp.add_argument("--out", required=True)
    sp.add_argument("--confidence", type=float, default=0.95)
    sp.add_argument("--export-suite", action="store_true", help="save failures as a regression suite")
    sp.add_argument("--anchors", type=int, default=5)
    sp.add_argument("--min-success", type=float, help="exit 1 if the CI lower bound is below this")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("compare", help="is a candidate checkpoint significantly better or worse?")
    sp.add_argument("baseline")
    sp.add_argument("candidate")
    sp.add_argument("--confidence", type=float, default=0.95)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("regress", help="replay a regression suite against a policy")
    common(sp)
    sp.add_argument("--suite", required=True)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_regress)

    sp = sub.add_parser("open-loop", help="score action chunks against recorded demonstrations")
    common(sp, needs_env=False)
    sp.add_argument("--dataset", required=True, help="LeRobot v2 dataset root")
    sp.add_argument("--horizon", type=int, default=16)
    sp.add_argument("--episodes", help="comma-separated episode indices")
    sp.add_argument("--video", action="store_true", help="decode videos listed in modality.json")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_open_loop)

    sp = sub.add_parser("make-demo-data", help="record a small LeRobot-format dataset from the Reach2D sim")
    sp.add_argument("--out", required=True)
    sp.add_argument("--episodes", type=int, default=30)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_make_demo_data)

    sp = sub.add_parser("correlate", help="sim-to-real agreement from paired results")
    sp.add_argument("pairs", help="CSV with policy, scenario_id, domain (sim|real), success")
    sp.add_argument("--confidence", type=float, default=0.95)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_correlate)

    sp = sub.add_parser("plan", help="how many episodes for a given confidence interval width")
    sp.add_argument("--margin", type=float, default=0.05)
    sp.add_argument("--confidence", type=float, default=0.95)
    sp.add_argument("--expected", type=float, default=0.5)
    sp.set_defaults(func=cmd_plan)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # Windows consoles default to cp1252; reports contain en dashes and minus signs.
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
