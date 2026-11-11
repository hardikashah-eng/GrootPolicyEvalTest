"""Command line: ``groot-eval <command>``. Run ``groot-eval -h`` for the list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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


def _recorder(a: argparse.Namespace, env: Any, policy: Any):
    if not getattr(a, "trace", None):
        return None
    from groot_eval.trace import TraceRecorder

    return TraceRecorder(domain="sim", env=a.env, policy=getattr(policy, "name", a.policy), control_dt=getattr(env, "dt", None))


def _latency(a: argparse.Namespace, policy: Any, env: Any, seed: int = 0) -> Any:
    if not getattr(a, "latency_mode", None):
        return policy
    from groot_eval.latency import with_latency

    wrapped = with_latency(policy, a.latency_mode, getattr(env, "dt", None), a.during_inference, seed)
    print(f"Latency model: {wrapped.name}")
    return wrapped


def _write_trace(recorder: Any, a: argparse.Namespace) -> None:
    if recorder is not None and recorder.episodes:
        root = recorder.write(a.trace)
        print(f"  wrote trace of {len(recorder.episodes)} episodes to {root}")


def _print_findings(findings: list[Any]) -> None:
    for f in findings:
        if f.level != "ok":
            print(f"  {f.level.upper():5s} {f.check}: {f.detail}")
    n_err = sum(f.level == "error" for f in findings)
    n_warn = sum(f.level == "warn" for f in findings)
    print(f"{n_err} errors, {n_warn} warnings")


# -- commands -------------------------------------------------------------
def cmd_import_real(a: argparse.Namespace) -> int:
    from groot_eval.checks import findings_markdown
    from groot_eval.trace import import_real

    root, findings = import_real(a.dataset, a.out, video=a.video, domain=a.domain, env=a.env_name or "", policy=a.policy_name or "")
    _write(root / "meta" / "import_report.md", findings_markdown(findings, f"Import of {a.dataset}"))
    _print_findings(findings)
    return 1 if any(f.level == "error" for f in findings) else 0


def cmd_check(a: argparse.Namespace) -> int:
    from groot_eval.checks import check_policy, check_sources, findings_markdown, source_from_env, source_from_traces
    from groot_eval.trace import TraceSet

    ref_ts = TraceSet.open(a.reference)
    ref = source_from_traces(ref_ts, "reference")
    policy = None
    if a.traces:
        cand = source_from_traces(TraceSet.open(a.traces), "candidate")
    else:
        env = make_env(a.env, **_kv(a.env_arg))
        policy = make_policy(a.policy, **_kv(a.policy_arg)) if a.policy else None
        scenarios = list(SweepConfig.load(a.config).scenarios())[: a.episodes] if a.config else [Scenario(seed=i) for i in range(a.episodes)]
        dim = ref.action.shape[1] if ref.action.size else None
        cand = source_from_env(env, policy, scenarios, action_dim=dim, name=a.env)
        env.close()
    findings = check_sources(cand, ref)
    if policy is not None:
        findings += check_policy(policy, ref)
    text = findings_markdown(findings, f"Contract check: {cand.name} vs {a.reference}")
    if a.out:
        _write(Path(a.out), text)
    _print_findings(findings)
    return 1 if any(f.level == "error" for f in findings) else 0


def cmd_gap(a: argparse.Namespace) -> int:
    from groot_eval.gap import gap_analysis, gap_markdown, write_plots
    from groot_eval.trace import TraceSet

    sim, real = TraceSet.open(a.sim), TraceSet.open(a.real)
    result = gap_analysis(sim, real, labels=(a.label_a, a.label_b))
    out = Path(a.out)
    _write(out / "gap.json", json.dumps(result, indent=2, default=str))
    _write(out / "gap.md", gap_markdown(result, top=a.top))
    if a.plots:
        paths = write_plots(result, sim, real, out / "plots", top=a.top)
        print(f"  wrote {len(paths)} plots to {out / 'plots'}" if paths else "  matplotlib not installed; no plots")
    errors = sum(c["level"] == "error" for c in result["contract"])
    print(f"Contract: {errors} errors. Largest gaps:")
    for r in result["signals"][:5]:
        print(f"  {r['group']}:{r['signal']}  W1/sigma={r['w1_norm']:.2f}  ({r['hint']})")
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    from groot_eval.replay import format_params, replay_markdown, replay_trace, system_id
    from groot_eval.trace import TraceSet

    ts = TraceSet.open(a.traces)
    env = make_env(a.env, **_kv(a.env_arg))
    base = _kv(a.param)
    signals = a.signals.split(",") if a.signals else None
    opts = {"reset_every": a.reset_every, "signals": signals, "episodes": a.episodes}
    try:
        if a.config:
            fit = system_id(env, ts, SweepConfig.load(a.config), base, **opts)
            baseline = fit["baseline"]
        else:
            fit, baseline = None, replay_trace(env, ts, base, label="baseline", **opts)
    finally:
        env.close()
    out = Path(a.out)
    payload: dict[str, Any] = {"trace": str(ts.root), "env": a.env, "baseline": baseline.to_dict()}
    if fit:
        payload.update(ranked=[r.to_dict() for r in fit["ranked"]], sensitivity=fit["sensitivity"], notes=fit["notes"])
    _write(out / "replay.json", json.dumps(payload, indent=2, default=str))
    _write(out / "replay.md", replay_markdown(ts, a.env, baseline, fit, top=a.top, reset_every=a.reset_every))
    print(f"Baseline tracking error (mean nRMSE): {baseline.score:.3f}")
    if fit and fit["ranked"]:
        best = fit["ranked"][0]
        print(f"Best of {len(fit['ranked'])} settings: {best.score:.3f}  ({format_params(best.params)})")
        for note in fit["notes"]:
            print(f"  - {note}")
    return 0


def cmd_sweep(a: argparse.Namespace) -> int:
    cfg = SweepConfig.load(a.config)
    if a.episodes:
        cfg.episodes = a.episodes
    env = make_env(a.env, **_kv(a.env_arg))
    policy = _latency(a, make_policy(a.policy, **_kv(a.policy_arg)), env, cfg.seed)
    scenarios = list(cfg.scenarios())
    if a.wear or a.faults:
        env, scenarios = _degraded_env(a, env, scenarios)
    print(f"Evaluating {getattr(policy, 'name', a.policy)} on {a.env}: {len(scenarios)} episodes")

    def progress(i: int, r: EpisodeResult) -> None:
        if (i + 1) % max(1, len(scenarios) // 10) == 0:
            print(f"  {i + 1}/{len(scenarios)}")

    recorder = _recorder(a, env, policy)
    run = evaluate(env, policy, scenarios, env_name=a.env, progress=progress, recorder=recorder)
    env.close()
    out = Path(a.out)
    s = _save_run(run, out, a.confidence)
    _write_trace(recorder, a)
    if a.export_pairs:
        from groot_eval.correlation import pairs_from_run, write_pairs

        rows = pairs_from_run(run, a.pairs_policy)
        write_pairs(rows, a.export_pairs)
        print(f"  appended {len(rows)} sim rows to {a.export_pairs}")
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
    policy = _latency(a, make_policy(a.policy, **_kv(a.policy_arg)), env)
    recorder = _recorder(a, env, policy)
    result = run_suite(env, policy, load_suite(a.suite), recorder=recorder)
    env.close()
    _write_trace(recorder, a)
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
    for w in c["join"]["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    text = report.correlation_report(c)
    if a.out:
        _write(Path(a.out), text)
    print(text)
    return 0


def cmd_pairs(a: argparse.Namespace) -> int:
    from groot_eval.correlation import pairs_from_trace, write_pairs
    from groot_eval.trace import TraceSet

    rows = pairs_from_trace(TraceSet.open(a.trace), a.policy, a.domain)
    if not rows:
        print("no episodes with a success mark in this trace")
        return 1
    write_pairs(rows, a.out, append=not a.overwrite)
    print(f"  wrote {len(rows)} {rows[0]['domain']} rows for {rows[0]['policy']} to {a.out}")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    n = stats.required_trials(a.margin, a.confidence, a.expected)
    print(f"{n} episodes give a ±{100 * a.margin:.1f}-point {int(100 * a.confidence)}% interval "
          f"at an expected success rate of {100 * a.expected:.0f}%.")
    return 0



# -- robustness: aging, faults, skill suites, data fidelity -----------------------------------
def _save_episodes(run: EvalRun, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w", encoding="utf-8") as fh:
        for r in run.results:
            fh.write(json.dumps({**r.to_dict(), "latencies_ms": r.policy_latency_ms}, default=str) + "\n")
    _write(out / "summary.json", json.dumps(run.summary(), indent=2))


def _base_scenarios(a: argparse.Namespace) -> list[Scenario]:
    if not a.config:
        return [Scenario(seed=i) for i in range(a.episodes or 100)]
    cfg = SweepConfig.load(a.config)
    if a.episodes:
        cfg.episodes = a.episodes
    scen = list(cfg.scenarios())
    return scen[: a.episodes] if a.episodes else scen


def _degraded_env(a: argparse.Namespace, base_env: Any, scenarios: list[Scenario]) -> tuple[Any, list[Scenario]]:
    """Wrap with WearEnv / FaultEnv when --wear/--profile or --faults are given."""
    from groot_eval.degradation import FaultConfig, FaultEnv, WearConfig, WearEnv, aged_scenarios, faulted_scenarios
    from groot_eval.degradation.components import probe_components

    env = base_env
    comps = None
    if getattr(a, "wear", None):
        if not a.profile:
            raise SystemExit("--wear needs --profile")
        comps = probe_components(make_env(a.env, **_kv(a.env_arg)))
        scenarios = aged_scenarios(scenarios, WearConfig.load(a.wear), a.profile, comps)
        env = WearEnv(env)
    if getattr(a, "faults", None):
        comps = comps or probe_components(make_env(a.env, **_kv(a.env_arg)))
        fcfg = FaultConfig.load(a.faults)
        scenarios = faulted_scenarios(scenarios, fcfg, comps, int(base_env.max_steps), float(getattr(base_env, "dt", 0.05)))
        env = FaultEnv(env)
    return env, scenarios


def cmd_aging(a: argparse.Namespace) -> int:
    from groot_eval.degradation import WearConfig
    from groot_eval.degradation.components import probe_components
    from groot_eval.degradation.reports import aging_markdown
    from groot_eval.degradation.runners import aging_summary, dof_loss_matrix, run_aging

    wear = WearConfig.load(a.wear)
    profiles = a.profiles.split(",") if a.profiles else wear.ordered_profiles()
    base = _base_scenarios(a)
    env_args = _kv(a.env_arg)
    comps = probe_components(make_env(a.env, **env_args))
    policy = make_policy(a.policy, **_kv(a.policy_arg))
    print(f"Aging {getattr(policy, 'name', a.policy)} on {a.env}: {len(base)} scenarios x {len(profiles)} profiles")
    out = Path(a.out)

    def progress(prof: str, run: EvalRun) -> None:
        s = run.summary(a.confidence)
        print(f"  {prof:>12s}: {100 * s['success_rate']:.1f}% [{100 * s['ci_low']:.1f}, {100 * s['ci_high']:.1f}]")
        _save_episodes(run, out / "profiles" / prof)

    runs = run_aging(lambda: make_env(a.env, **env_args), policy, base, wear, profiles, comps, progress)
    gate_profile, gate_value = (a.min_success_at[0], float(a.min_success_at[1])) if a.min_success_at else (None, None)
    summary = aging_summary(runs, gate=gate_value if gate_value is not None else a.margin_gate, confidence=a.confidence)
    dof = None
    if a.dof_episodes:
        print(f"  DOF-loss matrix: {a.dof_episodes} scenarios per cell")
        dof = dof_loss_matrix(lambda: make_env(a.env, **env_args), policy, base[: a.dof_episodes], comps,
                              confidence=a.confidence)
    _write(out / "aging.json", json.dumps({**summary, "dof_loss": dof}, indent=2, default=str))
    _write(out / "aging.md", aging_markdown(summary, dof))
    if gate_profile:
        row = next((r for r in summary["profiles"] if r["profile"] == gate_profile), None)
        if row is None:
            print(f"GATE ERROR: profile {gate_profile!r} was not run")
            return 2
        if row["ci_low"] < gate_value:
            print(f"GATE FAIL: at {gate_profile} the success lower bound {row['ci_low']:.3f} < {gate_value}")
            return 1
    return 0


def cmd_faults(a: argparse.Namespace) -> int:
    from groot_eval.degradation.reports import faults_markdown
    from groot_eval.degradation.runners import fault_gate, fault_summary

    base = _base_scenarios(a)
    env0 = make_env(a.env, **_kv(a.env_arg))
    env, scenarios = _degraded_env(a, env0, base)
    policy = make_policy(a.policy, **_kv(a.policy_arg))
    print(f"Fault campaign: {getattr(policy, 'name', a.policy)} on {a.env}, {len(scenarios)} episodes")
    recorder = _recorder(a, env0, policy)
    run = evaluate(env, policy, scenarios, env_name=a.env, recorder=recorder)
    out = Path(a.out)
    _save_episodes(run, out)
    _write_trace(recorder, a)
    summary = fault_summary(run, a.confidence)
    failures = fault_gate(summary, a.min_safe, a.max_unsafe) if (a.min_safe is not None or a.max_unsafe is not None) else None
    _write(out / "faults.json", json.dumps({**summary, "gate_failures": failures}, indent=2, default=str))
    _write(out / "faults.md", faults_markdown(summary, failures))
    if a.export_suite:
        from groot_eval.regression import export_suite

        suite = export_suite(run, out / "suite.json", anchors=a.anchors, note="fault campaign")
        print(f"  wrote {out / 'suite.json'} ({len(suite['cases'])} cases)")
    o = summary["overall"]
    if o.get("episodes"):
        print(f"Faulted episodes {o['episodes']}: success {100 * o['success']['rate']:.1f}%, safe outcome "
              f"{100 * o['safe']['rate']:.1f}% [{100 * o['safe']['ci_low']:.1f}, {100 * o['safe']['ci_high']:.1f}], "
              f"unsafe {100 * o['unsafe']['rate']:.1f}%")
    if failures:
        print("GATE FAIL: " + "; ".join(failures))
        return 1
    return 0


def cmd_suite(a: argparse.Namespace) -> int:
    from groot_eval.suites import SuiteConfig, suite_gate, suite_markdown, suite_summary

    cfg = SuiteConfig.load(a.config)
    if a.repeats:
        cfg.repeats = a.repeats
    if a.lhs_samples is not None:
        cfg.lhs_samples = a.lhs_samples
    if a.design:
        cfg.design = a.design.split(",")
    env0 = make_env(a.env, **_kv(a.env_arg))
    env, scenarios = _degraded_env(a, env0, cfg.scenarios())
    policy = make_policy(a.policy, **_kv(a.policy_arg))
    print(f"Suite {cfg.skill}: {getattr(policy, 'name', a.policy)} on {a.env}, {len(scenarios)} episodes ({', '.join(cfg.design)})")
    recorder = _recorder(a, env0, policy)
    run = evaluate(env, policy, scenarios, env_name=a.env, recorder=recorder)
    out = Path(a.out)
    _save_episodes(run, out)
    _write_trace(recorder, a)
    summary = suite_summary(cfg, run, a.confidence)
    failures = suite_gate(summary, a.min_oat, a.min_corner) if (a.min_oat is not None or a.min_corner is not None) else None
    _write(out / "suite.json", json.dumps({**summary, "gate_failures": failures}, indent=2, default=str))
    _write(out / "suite.md", suite_markdown(summary, cfg, failures))
    o = summary["overall"]
    print(f"Overall {100 * o['rate']:.1f}% [{100 * o['ci_low']:.1f}, {100 * o['ci_high']:.1f}]")
    if failures:
        print(f"GATE FAIL: {len(failures)} cell(s) significantly below the bar, e.g. {failures[0]}")
        return 1
    return 0


def cmd_fidelity(a: argparse.Namespace) -> int:
    from groot_eval import fidelity
    from groot_eval.trace import TraceSet

    if a.write_manifest:
        for root in a.write_manifest:
            print(f"  wrote {fidelity.write_manifest(root)}")
        return 0
    if not (a.sim and a.real and a.out):
        raise SystemExit("fidelity needs --sim, --real and --out (or --write-manifest DIR)")
    cfg, cfg_direction = fidelity.load_gates(a.gates)
    direction = a.direction or cfg_direction or "sim2real"
    env = make_env(a.env, **_kv(a.env_arg)) if a.env else None
    sysid = SweepConfig.load(a.sysid_config) if a.sysid_config else None
    results = fidelity.run_gates(TraceSet.open(a.sim), TraceSet.open(a.real), cfg, direction, env=env,
                                 params=_kv(a.param), sysid=sysid, pairs=a.pairs)
    out = Path(a.out)
    _write(out / "fidelity.json", json.dumps({"direction": direction, "exit_code": fidelity.exit_code(results),
                                             "gates": [r.to_dict() for r in results]}, indent=2))
    _write(out / "fidelity.md", fidelity.fidelity_markdown(results, a.sim, a.real, direction))
    for r in results:
        bad = [f for f in r.findings if f.level == "error"]
        print(f"  {r.gate:17s} {r.status.upper():5s}" + (f"  {bad[0].check}: {bad[0].detail[:110]}" if bad else ""))
    code = fidelity.exit_code(results)
    print({0: "All gates pass", 1: "GATE FAIL", 2: "INCOMPLETE: a required gate could not run"}[code])
    return code


def cmd_real2sim(a: argparse.Namespace) -> int:
    from groot_eval import real2sim

    bundle, base = real2sim.load_bundle(a.bundle)
    results = real2sim.run_checks(bundle, base, real2sim.load_thresholds(a.thresholds))
    out = Path(a.out)
    corr = real2sim.corrections(results)
    code = real2sim.exit_code(results)
    _write(out / "real2sim.json", json.dumps({"bundle": str(a.bundle), "exit_code": code, "corrections": corr,
                                             "checks": [r.to_dict() for r in results]}, indent=2))
    _write(out / "real2sim.md", real2sim.real2sim_markdown(results, a.bundle, str(bundle.get("name", ""))))
    _write(out / "corrections.yaml", yaml.safe_dump(corr, sort_keys=False) if corr else "{}\n")
    for r in results:
        bad = [f for f in r.findings if f.level == "error"]
        print(f"  {r.gate:14s} {r.status.upper():5s}" + (f"  {bad[0].check}: {bad[0].detail[:110]}" if bad else ""))
    print({0: "All checks pass", 1: "CHECK FAIL", 2: "INCOMPLETE: a required section is missing"}[code])
    return code


def cmd_transfer(a: argparse.Namespace) -> int:
    from groot_eval import transfer
    from groot_eval.trace import TraceSet

    if not ((a.sim and a.real) or a.roundtrip):
        raise SystemExit("transfer needs --sim and --real, and/or --roundtrip ORIGINAL CONVERTED")
    cfg, cfg_direction = transfer.load_config(a.config)
    direction = a.direction or cfg_direction or "sim2real"
    env = make_env(a.env, **_kv(a.env_arg)) if a.env else None
    policy = make_policy(a.policy, **_kv(a.policy_arg)) if a.policy else None
    wear_profiles = None
    if a.wear:
        from groot_eval.degradation import WearConfig

        wear_profiles = WearConfig.load(a.wear).profiles
    pair = (TraceSet.open(a.roundtrip[0]), TraceSet.open(a.roundtrip[1])) if a.roundtrip else None
    results = transfer.run_transfer(TraceSet.open(a.sim) if a.sim else None, TraceSet.open(a.real) if a.real else None,
                                    cfg, direction, env=env, policy=policy, robot=transfer.load_robot(a.robot),
                                    roundtrip_pair=pair, wear_profiles=wear_profiles)
    out = Path(a.out)
    _write(out / "transfer.json", json.dumps(transfer.to_json(results, direction), indent=2, default=str))
    _write(out / "transfer.md", transfer.transfer_markdown(results, a.sim, a.real, direction))
    for r in results:
        bad = [f for f in r.findings if f.level == "error"]
        print(f"  {r.gate:17s} {r.status.upper():5s}" + (f"  {bad[0].check}: {bad[0].detail[:110]}" if bad else ""))
    code = transfer.exit_code(results)
    print({0: "Transfer verified", 1: "GATE FAIL", 2: "INCOMPLETE: a required gate could not run"}[code])
    return code


def cmd_wear_fit(a: argparse.Namespace) -> int:
    from groot_eval.degradation import WearConfig
    from groot_eval.degradation.calibrate import fit_age, fit_markdown
    from groot_eval.degradation.components import probe_components
    from groot_eval.trace import TraceSet

    cfg = WearConfig.load(a.wear)
    env_args = _kv(a.env_arg)
    comps = probe_components(make_env(a.env, **env_args))
    ages = [float(x) for x in a.ages.split(",")] if a.ages else None
    fit = fit_age(lambda: make_env(a.env, **env_args), TraceSet.open(a.real), cfg, comps, ages, _kv(a.param), a.episodes)
    out = Path(a.out)
    _write(out / "wear_fit.json", json.dumps(fit, indent=2, default=str))
    _write(out / "wear_fit.md", fit_markdown(fit, a.real, a.env))
    if not fit["best"]:
        print("no age could be scored")
        return 1
    print(f"Best fit: {fit['best']['age_hours']:g} h (closest profile {fit['closest_profile']}), "
          f"replay error {fit['best']['score']:.3f}")
    for n in fit["notes"]:
        print(f"  - {n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="groot-eval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, needs_env: bool = True) -> None:
        if needs_env:
            sp.add_argument("--env", default="reach2d", help="reach2d | arm3 | isaaclab:<Task> | module:Class")
            sp.add_argument("--env-arg", action="append", metavar="K=V")
        sp.add_argument("--policy", required=True, help="pd | pd-avoid | pd-chunk | arm-ik | arm-ik-robust | groot://host:port | module:Class")
        sp.add_argument("--policy-arg", action="append", metavar="K=V")

    def latency_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--latency-mode", metavar="MODE",
                        help="make the sim feel inference latency: measured | fixed:<ms> | normal:<mean>,<std> | trace:<dir>")
        sp.add_argument("--during-inference", choices=["hold", "continue"], default="hold",
                        help="while inference runs: hold the last command, or continue the current chunk")

    sp = sub.add_parser("sweep", help="closed-loop evaluation over a randomization sweep")
    common(sp)
    sp.add_argument("--config", required=True)
    sp.add_argument("--episodes", type=int, help="override the config's episode count")
    sp.add_argument("--out", required=True)
    sp.add_argument("--confidence", type=float, default=0.95)
    sp.add_argument("--export-suite", action="store_true", help="save failures as a regression suite")
    sp.add_argument("--anchors", type=int, default=5)
    sp.add_argument("--min-success", type=float, help="exit 1 if the CI lower bound is below this")
    sp.add_argument("--trace", metavar="DIR", help="also write step-level traces (LeRobot v2) for sim-to-real comparison")
    latency_args(sp)
    sp.add_argument("--export-pairs", metavar="CSV", help="append policy,scenario_id,domain=sim,success rows for `correlate`")
    sp.add_argument("--pairs-policy", metavar="LABEL", help="checkpoint label for --export-pairs (match the real runs)")
    sp.add_argument("--wear", metavar="YAML", help="run on a worn robot: wear profiles YAML (needs --profile)")
    sp.add_argument("--profile", help="age profile, e.g. year_1")
    sp.add_argument("--faults", metavar="YAML", help="inject faults from this catalogue")
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
    sp.add_argument("--trace", metavar="DIR", help="also write step-level traces (LeRobot v2)")
    latency_args(sp)
    sp.set_defaults(func=cmd_regress)

    sp = sub.add_parser("import-real", help="convert a real-robot LeRobot recording into a trace and validate it")
    sp.add_argument("dataset", help="LeRobot v2 dataset root")
    sp.add_argument("--out", required=True, help="where to write the trace")
    sp.add_argument("--video", action="store_true", help="decode camera streams to compute image statistics (needs opencv)")
    sp.add_argument("--domain", default="real", choices=["real", "sim"])
    sp.add_argument("--env-name", help="label stored in trace_info.json")
    sp.add_argument("--policy-name", help="label stored in trace_info.json")
    sp.set_defaults(func=cmd_import_real)

    sp = sub.add_parser("check", help="contract check: units, ranges, keys, rates, image range vs a reference dataset")
    sp.add_argument("--reference", required=True, help="trace or LeRobot dataset to check against (usually the real one)")
    sp.add_argument("--traces", help="compare this trace instead of rolling out an env")
    sp.add_argument("--env", default="reach2d")
    sp.add_argument("--env-arg", action="append", metavar="K=V")
    sp.add_argument("--policy", help="policy to drive the env (zero actions if omitted)")
    sp.add_argument("--policy-arg", action="append", metavar="K=V")
    sp.add_argument("--config", help="sweep config whose first scenarios are rolled out")
    sp.add_argument("--episodes", type=int, default=3)
    sp.add_argument("--out", help="write the findings as Markdown")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("gap", help="signal-by-signal sim-to-real gap report from two traces")
    sp.add_argument("sim", help="simulation trace (or any LeRobot dataset)")
    sp.add_argument("real", help="real-robot trace (or any LeRobot dataset)")
    sp.add_argument("--out", required=True)
    sp.add_argument("--top", type=int, default=15, help="rows in the ranked table")
    sp.add_argument("--plots", action="store_true", help="histograms of the top signals (needs matplotlib)")
    sp.add_argument("--label-a", default="sim")
    sp.add_argument("--label-b", default="real")
    sp.set_defaults(func=cmd_gap)

    sp = sub.add_parser("replay", help="play recorded actions through the sim; with --config, rank sim parameters by fit")
    sp.add_argument("traces", help="trace to reproduce (usually the real-robot one)")
    sp.add_argument("--env", default="reach2d", help="reach2d | isaaclab:<Task> | module:Class")
    sp.add_argument("--env-arg", action="append", metavar="K=V")
    sp.add_argument("--param", action="append", metavar="K=V", help="fixed sim parameter for every replay")
    sp.add_argument("--config", help="sweep YAML of candidate parameters (grid mode recommended)")
    sp.add_argument("--reset-every", type=int, help="re-sync the sim to the recording every N steps")
    sp.add_argument("--signals", help="comma-separated state signals to score (default: all non-constant)")
    sp.add_argument("--episodes", type=int, help="replay only the first N episodes")
    sp.add_argument("--top", type=int, default=10, help="rows in the ranking table")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_replay)

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

    sp = sub.add_parser("pairs", help="append a trace's episode outcomes to a pairs CSV for `correlate`")
    sp.add_argument("trace", help="trace (sim or real) with per-episode success marks")
    sp.add_argument("--out", required=True, help="pairs CSV (appended to unless --overwrite)")
    sp.add_argument("--policy", help="checkpoint label (default: the trace's policy)")
    sp.add_argument("--domain", choices=["sim", "real"], help="override the trace's domain")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_pairs)

    def degrade_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--wear", metavar="YAML", help="wear profiles (configs/wear/*.yaml); needs --profile")
        sp.add_argument("--profile", help="age profile to apply, e.g. year_1")
        sp.add_argument("--faults", metavar="YAML", help="fault catalogue (configs/faults/*.yaml)")

    sp = sub.add_parser("aging", help="same scenarios on robots of different ages (wear and tear), plus DOF loss")
    common(sp)
    sp.add_argument("--config", help="sweep YAML for the base scenarios (default: plain seeds)")
    sp.add_argument("--episodes", type=int, help="number of base scenarios")
    sp.add_argument("--wear", required=True, metavar="YAML", help="wear profiles (configs/wear/*.yaml)")
    sp.add_argument("--profiles", help="comma-separated profiles in age order (default: all)")
    sp.add_argument("--dof-episodes", type=int, default=0, help="also run the DOF-loss matrix with this many scenarios per cell")
    sp.add_argument("--margin-gate", type=float, default=0.8, help="success level for the robustness-margin line")
    sp.add_argument("--min-success-at", nargs=2, metavar=("PROFILE", "RATE"),
                    help="exit 1 if the success CI lower bound at PROFILE is below RATE")
    sp.add_argument("--confidence", type=float, default=0.95)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_aging)

    sp = sub.add_parser("faults", help="inject subsystem faults mid-episode and measure graceful degradation")
    common(sp)
    sp.add_argument("--config", help="sweep YAML for the base scenarios (default: plain seeds)")
    sp.add_argument("--episodes", type=int)
    degrade_args(sp)
    sp.add_argument("--min-safe", type=float, help="exit 1 unless the safe-outcome CI lower bound reaches this")
    sp.add_argument("--max-unsafe", type=float, help="exit 1 unless the unsafe-event CI upper bound stays under this")
    sp.add_argument("--export-suite", action="store_true", help="save failing fault scenarios as a regression suite")
    sp.add_argument("--anchors", type=int, default=5)
    sp.add_argument("--trace", metavar="DIR", help="also write step-level traces (with fault.active columns)")
    sp.add_argument("--confidence", type=float, default=0.95)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_faults)

    sp = sub.add_parser("suite", help="skill-focused condition suite: min / nominal / max, one-at-a-time, corners, LHS")
    common(sp)
    sp.add_argument("--config", required=True, help="suite YAML (configs/suites/*.yaml)")
    sp.add_argument("--design", help="override the designs, e.g. oat,corners")
    sp.add_argument("--repeats", type=int, help="seeds per oat/corner cell")
    sp.add_argument("--lhs-samples", type=int)
    degrade_args(sp)
    sp.add_argument("--min-oat", type=float, help="exit 1 if any one-at-a-time level is significantly below this")
    sp.add_argument("--min-corner", type=float, help="exit 1 if any corner is significantly below this")
    sp.add_argument("--trace", metavar="DIR", help="also write traces; scenario names tag each episode's design cell")
    sp.add_argument("--confidence", type=float, default=0.95)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_suite)

    sp = sub.add_parser("fidelity", help="data-fidelity gates G0-G6 between a sim and a real trace")
    sp.add_argument("--sim", help="simulation trace")
    sp.add_argument("--real", help="real-robot trace")
    sp.add_argument("--gates", help="thresholds YAML (configs/fidelity/gates.yaml)")
    sp.add_argument("--direction", choices=["sim2real", "real2sim"], help="which side is being promoted")
    sp.add_argument("--env", help="env for G3 derived-signal checks and G5 replay (e.g. arm3)")
    sp.add_argument("--env-arg", action="append", metavar="K=V")
    sp.add_argument("--param", action="append", metavar="K=V", help="sim parameters for the G5 replay")
    sp.add_argument("--sysid-config", help="G5: fit sim parameters first (grid sweep YAML), then judge the best fit")
    sp.add_argument("--pairs", help="G6: pairs CSV of sim and real outcomes")
    sp.add_argument("--write-manifest", nargs="+", metavar="DIR", help="write SHA-256 manifests for these traces and exit")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_fidelity)

    sp = sub.add_parser("real2sim", help="calibration checks R1-R5: scene, kinematics, cameras, object placement, object physics")
    sp.add_argument("--bundle", required=True, help="calibration bundle YAML (configs/real2sim/example_bundle.yaml)")
    sp.add_argument("--thresholds", help="thresholds YAML (configs/real2sim/thresholds.yaml)")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_real2sim)

    sp = sub.add_parser("transfer", help="sim<->real transfer verification: fidelity, compatibility, usability, "
                                         "direction-specific checks, round trip")
    sp.add_argument("--sim", help="simulation trace")
    sp.add_argument("--real", help="real-robot trace")
    sp.add_argument("--direction", choices=["sim2real", "real2sim"], help="which side is being promoted")
    sp.add_argument("--config", help="thresholds YAML (configs/transfer/defaults.yaml)")
    sp.add_argument("--robot", help="robot profile YAML with the hardware's limits (configs/robots/*.yaml)")
    sp.add_argument("--env", help="target env: are keys, widths, action size and ranges accepted? (e.g. arm3)")
    sp.add_argument("--env-arg", action="append", metavar="K=V")
    sp.add_argument("--policy", help="policy that must consume the promoted observations (e.g. arm-ik)")
    sp.add_argument("--policy-arg", action="append", metavar="K=V")
    sp.add_argument("--wear", metavar="YAML", help="wear profiles, to name the profile matching the robot's age")
    sp.add_argument("--roundtrip", nargs=2, metavar=("ORIGINAL", "CONVERTED"),
                    help="the same episodes before and after a conversion: verify nothing changed")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_transfer)

    sp = sub.add_parser("wear-fit", help="real to sim: estimate a robot's effective age and wear from its trace")
    sp.add_argument("real", help="real-robot trace (recorded actions are replayed in the worn sim)")
    sp.add_argument("--env", default="arm3")
    sp.add_argument("--env-arg", action="append", metavar="K=V")
    sp.add_argument("--wear", required=True, metavar="YAML", help="wear profiles (configs/wear/*.yaml)")
    sp.add_argument("--ages", help="comma-separated candidate ages in hours (default: profiles + a grid to end of life)")
    sp.add_argument("--param", action="append", metavar="K=V", help="fixed sim parameter for every replay")
    sp.add_argument("--episodes", type=int, help="replay only the first N episodes")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_wear_fit)

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
