"""Digital-twin commands: ``groot-twin <command>``.

* ``repro``          reproduce failed runs in the twin and score the twin on them
* ``triage``         why did a robot run fail: plumbing, timing, actuation, sensing, policy or not modelled
* ``metrics``        tier-1 twin usefulness metrics from a table of sim and real results
* ``credibility``    twin credibility per scenario class, and a twin certificate for this robot profile
* ``cert-check``     may a sim result in this scenario class stand in for robot runs?
* ``convert-check``  HDF5 -> LeRobot conversion fidelity gates (C0-C4)
* ``convert``        reference HDF5 -> LeRobot converter (to compare a production pipeline against)
* ``convert-sidecar`` keep the sim-only HDF5 data (initial_state, states) next to a LeRobot dataset
* ``roundtrip``      rebuild the HDF5 from LeRobot + sidecar and compare it with the source (C5)
* ``replay-check``   replay the source and the rebuilt HDF5 in the simulator (C6)

Run ``groot-twin <command> -h`` for options. Exit codes: 0 pass, 1 a gate failed,
2 a required check could not run.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _kv(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _knobs(items: list[str] | None) -> dict[str, list[Any]]:
    """``--knob friction=0.4,0.8,1.2`` -> {"friction": [0.4, 0.8, 1.2]}."""
    out: dict[str, list[Any]] = {}
    for item in items or []:
        name, _, raw = item.partition("=")
        vals = []
        for v in raw.split(","):
            try:
                vals.append(json.loads(v))
            except json.JSONDecodeError:
                vals.append(v)
        out[name] = vals
    return out


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path}")


def _load_callable(spec: str | None) -> Any:
    if not spec:
        return None
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)


# -- repro -------------------------------------------------------------------------------
def cmd_repro(a: argparse.Namespace) -> int:
    from groot_eval import repro

    cfg = repro.load_config(a.config, env=a.env, policy=a.policy, repeats=a.repeats, seed_jitter=a.seed_jitter,
                            latency_mode=a.latency_mode)
    if a.env_arg:
        cfg["env_args"] = {**cfg.get("env_args", {}), **_kv(a.env_arg)}
    if a.policy_arg:
        cfg["policy_args"] = {**cfg.get("policy_args", {}), **_kv(a.policy_arg)}
    if a.knob:
        cfg["knobs"] = {**cfg.get("knobs", {}), **_knobs(a.knob)}
    if a.no_replay:
        cfg["replay"] = False
    episodes = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    cases = repro.load_failures(a.failed, episodes, a.domain)
    if a.limit:
        cases = cases[: a.limit]
    if not cases:
        print(f"no failed episodes found in {a.failed}")
        return 2
    print(f"Reproducing {len(cases)} failure(s) in `{cfg['env']}` with policy `{cfg['policy']}`")

    def progress(i: int, r: dict[str, Any]) -> None:
        print(f"  [{i + 1}/{len(cases)}] {r['case_id']}: {r['failure_reason']} -> {r['verdict']}"
              + (f"  ({r['twin_ticket']['summary']})" if r.get("twin_ticket") else ""))

    summary = repro.reproduce_batch(cases, cfg, progress)
    out = Path(a.out)
    _write(out / "repro.json", repro.to_json({"config": cfg, **summary}))
    _write(out / "repro.md", repro.repro_markdown(summary, cfg))
    lo, hi = summary["reproduction_ci"]
    print(f"Failure reproduction rate {100 * summary['reproduction_rate']:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]; "
          f"with knobs {100 * summary['reproduction_rate_with_knobs']:.0f}%")
    if summary["twin_tickets"]:
        print("Twin tickets: " + ", ".join(f"{k} ({c})" for k, c in summary["twin_tickets"].items()))
    if a.min_rate is not None and summary["reproduction_ci"][0] < a.min_rate:
        print(f"GATE FAIL: reproduction-rate lower bound {lo:.2f} < {a.min_rate}")
        return 1
    return 0


# -- triage --------------------------------------------------------------------------------
def cmd_triage(a: argparse.Namespace) -> int:
    from groot_eval import repro, triage
    from groot_eval.envs import make_env
    from groot_eval.policies import make_policy

    cases = repro.load_failures(a.failed, [int(x) for x in a.episodes.split(",")] if a.episodes else None, "real")
    cases = [c for c in cases if c.has_recording][: a.limit or None]
    if not cases:
        print(f"no failed runs with recorded states and actions in {a.failed}")
        return 2
    env = make_env(a.env, **_kv(a.env_arg))
    policy = make_policy(a.policy, **_kv(a.policy_arg)) if a.policy else None
    cfg = {k: float(v) for k, v in _kv(a.set).items()}
    for k in ("repeats", "seed_jitter", "max_lag"):
        if k in cfg:
            cfg[k] = int(cfg[k])
    print(f"Triaging {len(cases)} failed run(s) against `{a.env}`" + (f" with policy `{a.policy}`" if a.policy else ""))

    def progress(i: int, r: dict[str, Any]) -> None:
        print(f"  [{i + 1}/{len(cases)}] {r['case_id']}: {r['failure_reason']} -> {r['cause']} ({r['confidence']})")

    summary = triage.triage_batch(cases, env, policy, cfg, progress)
    out = Path(a.out)
    _write(out / "triage.json", triage.to_json(summary))
    _write(out / "triage.md", triage.triage_markdown(summary))
    return 0


# -- metrics -----------------------------------------------------------------------------
def _read_rows(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _metrics_markdown(pv: dict[str, Any], cv: dict[str, Any] | None) -> str:
    def f(x: Any, fmt: str = ".2f") -> str:
        return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)

    cal = pv["calibration"]
    lines = ["# Digital-twin usefulness: predictive validity", "",
             "| Metric | Value | Reading |", "|---|---|---|",
             f"| Pearson r (sim vs real success) | {f(pv['pearson'])} | scores move together (1 = perfectly) |",
             f"| Spearman / Kendall tau | {f(pv['spearman'])} / {f(pv['kendall_tau'])} | same ranking of policies |",
             f"| MMRV | {f(pv['mmrv'], '.3f')} | worst mis-ranking, in success-rate units (0 = never mis-ranks) |",
             f"| Bias (sim - real) | {f(cal['bias'], '+.3f')} [{f(cal['bias_ci_low'], '+.3f')}, {f(cal['bias_ci_high'], '+.3f')}] "
             "| positive = twin optimistic |",
             f"| Slope / offset (real ≈ slope x sim + offset) | {f(cal['slope'])} / {f(cal['offset'], '+.2f')} "
             "| slope < 1: twin exaggerates differences |"]
    sens = pv["sensitivity"]
    if sens["pairs"]:
        lines.append(f"| Perturbation sensitivity alignment | r {f(sens['correlation'])}, same sign "
                     f"{f(100 * sens['sign_agreement'], '.0f')}%, mean error {f(sens['mean_abs_error'], '.3f')} "
                     "| does each perturbation hurt as much in the twin as on the robot? |")
    if pv.get("decisions"):
        d = pv["decisions"]
        lines.append(f"| Gate decisions | recall {f(d['recall'])}, false alarms {f(d['false_alarm_rate'])} "
                     "| would gating on the twin make the robot's calls? |")
    if cv:
        lines.append(f"| Control-variate value | rho {f(cv['rho'])}: each real run worth {f(cv['real_runs_multiplier'])} "
                     f"| real runs saved at equal confidence: {f(100 * cv['real_runs_saved_share'], '.0f')}% |")
    lines += ["", "| Policy | Twin | Robot |", "|---|---|---|",
              *[f"| {t['policy']} | {100 * t['sim_rate']:.0f}% (n={t['sim_n']}) | {100 * t['real_rate']:.0f}% (n={t['real_n']}) |"
                for t in pv["table"]]]
    if sens.get("worst"):
        lines += ["", "Perturbations the twin gets most wrong:", "",
                  *[f"- {w['policy']} / {w['condition']}: twin {w['sim_delta']:+.2f}, robot {w['real_delta']:+.2f}"
                    for w in sens["worst"]]]
    lines += ["", *[f"- warning: {w}" for w in pv["warnings"]]]
    return "\n".join(lines) + "\n"


def cmd_metrics(a: argparse.Namespace) -> int:
    from groot_eval import repro, twin_metrics as tm

    rows = _read_rows(a.results)
    pv = tm.predictive_validity(rows, nominal=a.nominal, gate=a.gate)
    cv = None
    if rows and "scenario_id" in rows[0]:
        idx: dict[tuple[str, str], dict[str, list[float]]] = {}
        for r in rows:
            ok = str(r.get("success", "")).strip().lower() in ("1", "true", "yes")
            idx.setdefault((r["policy"], r["scenario_id"]), {}).setdefault(r["domain"], []).append(float(ok))
        pairs = [(np.mean(v["sim"]), np.mean(v["real"])) for v in idx.values() if "sim" in v and "real" in v]
        if len(pairs) >= 3:
            cv = tm.control_variate_savings([p[0] for p in pairs], [p[1] for p in pairs])
    out = Path(a.out)
    _write(out / "twin_metrics.json", repro.to_json({"predictive_validity": pv, "control_variate": cv}))
    _write(out / "twin_metrics.md", _metrics_markdown(pv, cv))
    print(f"Pearson {pv['pearson']:.2f}, MMRV {pv['mmrv']:.3f}, bias {pv['calibration']['bias']:+.3f} "
          f"over {pv['policies']} policies")
    for w in pv["warnings"]:
        print(f"  warning: {w}")
    if a.max_mmrv is not None and not (pv["mmrv"] <= a.max_mmrv):
        print(f"GATE FAIL: MMRV {pv['mmrv']:.3f} > {a.max_mmrv}")
        return 1
    return 0


# -- credibility ---------------------------------------------------------------------------
def cmd_credibility(a: argparse.Namespace) -> int:
    from groot_eval import credibility as cr

    rows = cr.read_rows(a.results)
    if not rows:
        print(f"no rows in {a.results}")
        return 2
    th = {k: float(v) for k, v in _kv(a.threshold).items()}
    result = cr.assess(rows, class_column=a.class_column, gate=a.gate, thresholds=th)
    profile = None
    if a.profile:
        from groot_eval.profiles import resolve_profile

        prof = resolve_profile(a.profile)
        profile = {"robot_id": prof.robot_id, "name": prof.name or prof.robot_id, "version": prof.version,
                   "status": prof.status, "hash": prof.content_hash}
        if prof.status != "approved":
            print(f"  warning: profile {prof.robot_id} v{prof.version} is {prof.status}, not approved")
    cert = cr.make_certificate(result, a.twin, a.twin_version, profile, a.valid_days,
                               a.decision or None) if a.twin_version else None
    out = Path(a.out)
    _write(out / "credibility.json", json.dumps(result, indent=2, default=lambda x: None))
    _write(out / "credibility.md", cr.credibility_markdown(result, cert))
    if cert:
        _write(out / "twin_certificate.json", json.dumps(cert, indent=2, default=lambda x: None))
    for cls, m in result["classes"].items():
        print(f"  {cls:24s} {m['verdict']:12s} policies {m['policies']}")
    missing = [c for c in a.require or [] if result["classes"].get(c, {}).get("verdict") != "trusted"]
    if missing:
        print(f"GATE FAIL: not trusted for required class(es): {', '.join(missing)}")
        return 1
    return 0


def cmd_cert_check(a: argparse.Namespace) -> int:
    from groot_eval import credibility as cr

    cert = json.loads(Path(a.certificate).read_text(encoding="utf-8"))
    phash = None
    if a.profile:
        from groot_eval.profiles import resolve_profile

        phash = resolve_profile(a.profile).content_hash
    ok, why = cr.check_certificate(cert, a.scenario_class, phash, a.twin_version, a.allow_provisional)
    print(("OK: " if ok else "NO: ") + why)
    return 0 if ok else 1


# -- conversion ----------------------------------------------------------------------------
def cmd_convert_check(a: argparse.Namespace) -> int:
    from groot_eval import conversion_fidelity as cf
    from groot_eval.fidelity import exit_code

    cfg = cf.load_config(a.config)
    results = cf.check_conversion(a.hdf5, a.lerobot, cfg, encoder=_load_callable(a.encoder))
    out = Path(a.out)
    code = exit_code(results)
    _write(out / "conversion.json", json.dumps({"exit_code": code, "gates": [r.to_dict() for r in results]},
                                               indent=2, default=lambda x: None))
    _write(out / "conversion.md", cf.conversion_markdown(results, a.hdf5, a.lerobot))
    for r in results:
        bad = [f for f in r.findings if f.level == "error"]
        print(f"  {r.gate:13s} {r.status.upper():5s}" + (f"  {bad[0].check}: {bad[0].detail[:110]}" if bad else ""))
    print({0: "Conversion kept the information checked", 1: "GATE FAIL: information lost or changed",
           2: "INCOMPLETE: a required gate could not run"}[code])
    return code


def cmd_convert(a: argparse.Namespace) -> int:
    from groot_eval import conversion_fidelity as cf

    cfg = cf.load_config(a.config)
    root = cf.convert_hdf5_to_lerobot(a.hdf5, a.out, cfg["mapping"], fps=a.fps, task=a.task or "", codec=a.codec)
    print(f"wrote {root}")
    if a.sidecar:
        from groot_eval import conversion_roundtrip as rt

        print(f"wrote {rt.write_sidecar(a.hdf5, root, rt.load_config(a.config))}")
    return 0


def cmd_convert_sidecar(a: argparse.Namespace) -> int:
    from groot_eval import conversion_roundtrip as rt

    print(f"wrote {rt.write_sidecar(a.hdf5, a.lerobot, rt.load_config(a.config))}")
    return 0


def _roundtrip_report(results: list[Any], a: argparse.Namespace, name: str) -> int:
    from groot_eval import conversion_roundtrip as rt
    from groot_eval.fidelity import exit_code

    out = Path(a.out)
    code = exit_code(results)
    _write(out / f"{name}.json", json.dumps({"exit_code": code, "gates": [r.to_dict() for r in results]},
                                            indent=2, default=lambda x: None))
    _write(out / f"{name}.md", rt.roundtrip_markdown(results, a.hdf5, getattr(a, "lerobot", "") or ""))
    for r in results:
        bad = [f for f in r.findings if f.level == "error"]
        print(f"  {r.gate:13s} {r.status.upper():5s}" + (f"  {bad[0].check}: {bad[0].detail[:110]}" if bad else ""))
    print({0: "Round trip kept what was checked", 1: "GATE FAIL", 2: "INCOMPLETE: a required gate could not run"}[code])
    return code


def cmd_roundtrip(a: argparse.Namespace) -> int:
    from groot_eval import conversion_roundtrip as rt

    cfg = rt.load_config(a.config)
    recon = Path(a.recon or Path(a.out) / "rebuilt.hdf5")
    result = rt.check_roundtrip(a.hdf5, a.lerobot, cfg, recon)
    if recon.exists():
        print(f"  rebuilt {recon}")
    return _roundtrip_report([result], a, "roundtrip")


def cmd_replay_check(a: argparse.Namespace) -> int:
    from groot_eval import conversion_roundtrip as rt

    cfg = rt.load_config(a.config)
    if a.replayer == "isaaclab":
        if not a.task:
            raise SystemExit("--replayer isaaclab needs --task")
        from groot_eval.isaaclab_demo_replay import IsaacLabDemoReplayer

        replayer = IsaacLabDemoReplayer(a.task, dict(kv.split("=", 1) for kv in a.camera_map or []))
    else:
        replayer = _load_callable(a.replayer)()
    try:
        result = rt.check_replay(a.hdf5, a.rebuilt, cfg, replayer, joint_key=a.joint_key, render=not a.no_render,
                                 demos=a.demos.split(",") if a.demos else None)
    finally:
        close = getattr(replayer, "close", None)
        if callable(close):
            close()
    return _roundtrip_report([result], a, "replay")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="groot-twin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("repro", help="reproduce failed runs in the digital twin")
    sp.add_argument("failed", help="trace / LeRobot dataset, sweep run dir, suite.json or failure-record JSON")
    sp.add_argument("--config", help="repro YAML (configs/repro/*.yaml)")
    sp.add_argument("--env", help="twin env: reach2d | arm3 | isaaclab:<Task> | module:Class")
    sp.add_argument("--env-arg", action="append", metavar="K=V")
    sp.add_argument("--policy", help="policy that failed: groot://host:port | module:Class | pd ...")
    sp.add_argument("--policy-arg", action="append", metavar="K=V")
    sp.add_argument("--knob", action="append", metavar="NAME=V1,V2,...", help="twin knob values to search")
    sp.add_argument("--repeats", type=int, help="identical re-runs per failure")
    sp.add_argument("--seed-jitter", type=int, help="neighbouring seeds per failure")
    sp.add_argument("--latency-mode", help="trace:<dir> | fixed:<ms> | normal:<mean>,<std>")
    sp.add_argument("--episodes", help="comma-separated episode indices to take (default: every failed one)")
    sp.add_argument("--limit", type=int, help="at most this many failures")
    sp.add_argument("--domain", choices=["real", "sim"], help="where the failures happened")
    sp.add_argument("--no-replay", action="store_true", help="skip open-loop replay of recorded actions")
    sp.add_argument("--min-rate", type=float, help="exit 1 if the reproduction-rate CI lower bound is below this")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_repro)

    sp = sub.add_parser("triage", help="why did a robot run fail: plumbing, timing, actuation, sensing, policy, not modelled")
    sp.add_argument("failed", help="trace / LeRobot dataset of the robot's failed runs, or a failure-record JSON")
    sp.add_argument("--env", required=True, help="twin env: reach2d | arm3 | isaaclab:<Task> | module:Class")
    sp.add_argument("--env-arg", action="append", metavar="K=V")
    sp.add_argument("--policy", help="the policy that ran on the robot (needed for the consistency and closed-loop tests)")
    sp.add_argument("--policy-arg", action="append", metavar="K=V")
    sp.add_argument("--episodes", help="comma-separated episode indices (default: every failed one)")
    sp.add_argument("--limit", type=int, help="at most this many runs")
    sp.add_argument("--set", action="append", metavar="K=V", help="override a triage setting, e.g. max_lag=10")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_triage)

    sp = sub.add_parser("metrics", help="tier-1 twin metrics: correlation, MMRV, calibration, sensitivity, decisions")
    sp.add_argument("results", help="CSV: policy, domain (sim|real), [condition], success | successes,trials, [scenario_id]")
    sp.add_argument("--nominal", default="nominal", help="condition name of the unperturbed setting")
    sp.add_argument("--gate", type=float, help="success-rate bar: also score pass/fail decision agreement")
    sp.add_argument("--max-mmrv", type=float, help="exit 1 if MMRV exceeds this")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_metrics)

    sp = sub.add_parser("credibility", help="twin credibility per scenario class, and a twin certificate")
    sp.add_argument("results", help="CSV: policy, domain (sim|real), scenario_class, success | successes,trials, [scenario_id]")
    sp.add_argument("--class-column", default="scenario_class", help="column that names the scenario class")
    sp.add_argument("--gate", type=float, help="success-rate bar: also score pass/fail decision agreement")
    sp.add_argument("--threshold", action="append", metavar="K=V", help="override a threshold, e.g. max_mmrv=0.05")
    sp.add_argument("--twin", default="twin", help="twin name for the certificate")
    sp.add_argument("--twin-version", help="twin version; writes twin_certificate.json")
    sp.add_argument("--profile", help="robot profile the evidence was taken on: path or registry:<root>:<robot>@approved")
    sp.add_argument("--valid-days", type=int, default=90, help="certificate lifetime")
    sp.add_argument("--decision", action="append", help="decision the certificate covers (repeatable)")
    sp.add_argument("--require", action="append", help="exit 1 unless this class is trusted (repeatable)")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_credibility)

    sp = sub.add_parser("cert-check", help="may a sim result in this scenario class stand in for robot runs?")
    sp.add_argument("certificate", help="twin_certificate.json")
    sp.add_argument("--class", dest="scenario_class", required=True)
    sp.add_argument("--profile", help="the robot's current profile: refuse if it differs from the certified one")
    sp.add_argument("--twin-version", help="the twin version in use: refuse if it differs")
    sp.add_argument("--allow-provisional", action="store_true", help="accept provisional classes too")
    sp.set_defaults(func=cmd_cert_check)

    sp = sub.add_parser("convert-check", help="HDF5 -> LeRobot conversion fidelity (visual and actuator)")
    sp.add_argument("--hdf5", required=True)
    sp.add_argument("--lerobot", required=True, help="converted LeRobot v2 dataset root")
    sp.add_argument("--config", help="mapping + thresholds YAML (configs/conversion/*.yaml)")
    sp.add_argument("--encoder", metavar="MODULE:FN", help="frame -> embedding function for the policy-space gap")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_convert_check)

    sp = sub.add_parser("convert", help="reference HDF5 -> LeRobot v2 converter")
    sp.add_argument("--hdf5", required=True)
    sp.add_argument("--config", required=True, help="YAML with a `mapping:` section")
    sp.add_argument("--fps", type=float, required=True)
    sp.add_argument("--task", help="task string when the HDF5 has none")
    sp.add_argument("--codec", default="h264-lossless", choices=["h264-lossless", "h264", "av1", "mp4v"])
    sp.add_argument("--sidecar", action="store_true", help="also write the sim sidecar needed by `roundtrip`")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_convert)

    sp = sub.add_parser("convert-sidecar", help="keep initial_state / states next to a LeRobot dataset (run after converting)")
    sp.add_argument("--hdf5", required=True)
    sp.add_argument("--lerobot", required=True, help="LeRobot v2 dataset converted from --hdf5")
    sp.add_argument("--config", help="mapping YAML; `roundtrip.sidecar` lists the sim-only prefixes")
    sp.set_defaults(func=cmd_convert_sidecar)

    sp = sub.add_parser("roundtrip", help="rebuild the HDF5 from LeRobot + sidecar and compare with the source (C5)")
    sp.add_argument("--hdf5", required=True)
    sp.add_argument("--lerobot", required=True)
    sp.add_argument("--config", help="mapping + `roundtrip` thresholds YAML (configs/conversion/*.yaml)")
    sp.add_argument("--recon", help="where to write the rebuilt HDF5 (default: OUT/rebuilt.hdf5)")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_roundtrip)

    sp = sub.add_parser("replay-check", help="replay the source and the rebuilt HDF5 in the simulator (C6)")
    sp.add_argument("--hdf5", required=True, help="source HDF5")
    sp.add_argument("--rebuilt", required=True, help="HDF5 written by `roundtrip`")
    sp.add_argument("--config", help="mapping + `roundtrip` thresholds YAML")
    sp.add_argument("--replayer", default="isaaclab", help="isaaclab | module:Class (a DemoReplayer)")
    sp.add_argument("--task", help="Isaac Lab task id for --replayer isaaclab")
    sp.add_argument("--camera-map", nargs="*", metavar="HDF5_KEY=SENSOR", help="camera datasets to re-render")
    sp.add_argument("--joint-key", default="obs/joint_pos", help="recorded joint positions, for the source check")
    sp.add_argument("--demos", help="comma-separated episode names (default: all)")
    sp.add_argument("--no-render", action="store_true", help="skip re-rendering recorded states")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_replay_check)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
