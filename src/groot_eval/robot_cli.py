"""Command line for robot-side models: ``groot-robot <command>``.

Commands are registered by the modules that implement them, so each feature
(actuators, scheduler, hardware bridge, calibration and robot profiles) adds
its own subcommands here without touching the others.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

Register = Callable[[argparse._SubParsersAction], None]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path}")


# -- actuator layer ---------------------------------------------------------------
def cmd_actuator_sweep(a: argparse.Namespace) -> int:
    from groot_eval.actuators.core import ActuatorParams, FaultTable, WearTable
    from groot_eval.actuators.sweep import run_sweep, sweep_markdown
    from groot_eval.degradation import FaultConfig, WearConfig

    cfg = WearConfig.load(a.wear)
    ages = [float(x) for x in a.ages.split(",")] if a.ages else sorted(cfg.profiles.values())
    wear = WearTable.fleet(cfg, a.joints, ages, a.units)
    params = ActuatorParams(effort_limit=a.effort_limit, stiffness=a.stiffness, damping=a.damping,
                            coulomb_friction=a.coulomb, viscous_friction=a.viscous, dt=a.dt)
    faults = (FaultTable.sample(FaultConfig.load(a.faults), wear.num_envs, a.joints, a.steps, a.dt,
                                seeds=wear.unit_seeds) if a.faults else None)
    res = run_sweep(wear, params, faults, steps=a.steps)
    out = Path(a.out)
    _write(out / "actuator_sweep.json", json.dumps(res, indent=2))
    _write(out / "actuator_sweep.md", sweep_markdown(res, f"{a.wear}, {a.units} units x {len(ages)} ages, {a.joints} joints"))
    for r in res["rows"]:
        print(f"  {r['age_hours']:>8g} h  RMSE {r['rmse_mean']:.4f} rad (worst {r['rmse_worst']:.4f}), "
              f"saturated {100 * r['saturated_frac']:.1f}%")
    return 0


def register_actuators(sub: argparse._SubParsersAction) -> None:
    sp = sub.add_parser("actuator-sweep", help="sweep a fleet of ages and units through the wear-aware actuator in one batch")
    sp.add_argument("--wear", required=True, metavar="YAML", help="wear profiles (configs/wear/*.yaml)")
    sp.add_argument("--ages", help="comma-separated ages in hours (default: every profile)")
    sp.add_argument("--units", type=int, default=64, help="units per age (the same units at every age)")
    sp.add_argument("--joints", type=int, default=3)
    sp.add_argument("--faults", metavar="YAML", help="fault catalogue; each unit gets its own schedule")
    sp.add_argument("--steps", type=int, default=2000)
    sp.add_argument("--dt", type=float, default=0.005)
    sp.add_argument("--effort-limit", type=float, default=10.0)
    sp.add_argument("--stiffness", type=float, default=60.0)
    sp.add_argument("--damping", type=float, default=4.0)
    sp.add_argument("--coulomb", type=float, default=0.2, help="nominal Coulomb friction, N·m (scaled by wear)")
    sp.add_argument("--viscous", type=float, default=0.1, help="nominal viscous friction, N·m·s/rad (scaled by wear)")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_actuator_sweep)


# -- multi-rate scheduler -----------------------------------------------------------
def cmd_timing_profile(a: argparse.Namespace) -> int:
    from groot_eval.scheduler import TimingProfile

    prof = TimingProfile.from_trace(a.trace, during_inference=a.during, control_hz=a.control_hz)
    if a.policy_period:
        prof.policy_period_ticks = a.policy_period
    path = prof.save(a.out)
    print(f"  wrote {path}")
    print(f"  {prof.describe()}")
    if not prof.source.get("latency_samples"):
        print("  warning: the trace has no inference_latency_ms values; latency is set to 0")
    return 0


def cmd_timing_check(a: argparse.Namespace) -> int:
    from groot_eval.closed_loop import evaluate
    from groot_eval.envs import make_env
    from groot_eval.policies import make_policy
    from groot_eval.scheduler import ScheduledEnv, TimingProfile
    from groot_eval.stats import mcnemar_exact
    from groot_eval.types import Scenario

    prof = TimingProfile.load(a.profile)
    scenarios = [Scenario(seed=i) for i in range(a.episodes)]
    rows = {}
    for label in ("ideal", "profiled"):
        env = make_env(a.env)
        dt = float(getattr(env, "dt", 0.0)) or prof.control_dt
        policy = make_policy(a.policy)
        if label == "profiled":
            env = ScheduledEnv(env, prof, seed=a.seed)
            policy = prof.wrap_policy(policy, dt, seed=a.seed)
        run = evaluate(env, policy, scenarios, env_name=a.env)
        env.close()
        rows[label] = run
    ideal, prof_run = rows["ideal"], rows["profiled"]
    lost = sum(1 for x, y in zip(ideal.results, prof_run.results) if x.success and not y.success)
    gained = sum(1 for x, y in zip(ideal.results, prof_run.results) if y.success and not x.success)
    p = mcnemar_exact(lost, gained)
    s_i, s_p = ideal.summary(), prof_run.summary()
    lines = [f"# Timing check: {a.policy} on {a.env}", "", f"Profile: {prof.describe()}", "",
             "| Loop timing | Success | 95% CI | Failures |", "|---|---|---|---|"]
    for name, s in (("ideal (lockstep)", s_i), ("robot timing", s_p)):
        lines.append(f"| {name} | {100 * s['success_rate']:.1f}% | [{100 * s['ci_low']:.1f}, {100 * s['ci_high']:.1f}] | "
                     f"{s['failure_reasons'] or 'none'} |")
    lines += ["", f"Same scenarios in both runs: {lost} succeed only with ideal timing, {gained} only with robot timing "
                  f"(McNemar p = {p:.3g})."]
    out = Path(a.out)
    _write(out / "timing_check.md", "\n".join(lines) + "\n")
    _write(out / "timing_check.json", json.dumps({"profile": prof.to_dict(), "ideal": s_i, "profiled": s_p,
                                                  "lost": lost, "gained": gained, "mcnemar_p": p}, indent=2, default=str))
    print(f"  ideal {100 * s_i['success_rate']:.1f}%  robot timing {100 * s_p['success_rate']:.1f}%  "
          f"({lost} lost, {gained} gained, p = {p:.3g})")
    return 0


def register_scheduler(sub: argparse._SubParsersAction) -> None:
    sp = sub.add_parser("timing-profile", help="measure a robot's loop timing (latency, jitter, missed ticks) from a trace")
    sp.add_argument("trace", help="trace directory (usually the real robot's)")
    sp.add_argument("--out", required=True, help="timing profile YAML to write")
    sp.add_argument("--during", choices=["hold", "continue"], default="continue",
                    help="what the robot executes while an inference is in flight")
    sp.add_argument("--control-hz", type=float, help="override the trace's fps")
    sp.add_argument("--policy-period", type=int, help="override control ticks between inference requests")
    sp.set_defaults(func=cmd_timing_profile)

    sp = sub.add_parser("timing-check", help="same scenarios with lockstep vs the robot's timing: does the policy survive?")
    sp.add_argument("--profile", required=True, metavar="YAML", help="timing profile (groot-robot timing-profile)")
    sp.add_argument("--env", default="arm3")
    sp.add_argument("--policy", default="arm-ik")
    sp.add_argument("--episodes", type=int, default=100)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_timing_check)


# -- hardware bridge --------------------------------------------------------------
def cmd_bridge(a: argparse.Namespace) -> int:
    from groot_eval.bridge import (BridgeConfig, BridgeSession, PolicySource, TraceSource, build_report,
                                   make_transport, parse_env_spec)
    from groot_eval.policies import make_policy
    from groot_eval.types import Scenario

    cfg = BridgeConfig.load(a.config, mode=a.mode, resync_every=a.resync_every, max_ticks=a.max_ticks)
    twin_name, twin_params = parse_env_spec(a.twin)
    if a.twin_profile:
        from groot_eval.profiles import resolve_profile

        prof = resolve_profile(a.twin_profile)
        twin_params = {**prof.twin_params(), **twin_params}
        print(f"  twin is {prof.robot_id} v{prof.version}: {len(twin_params)} parameters from the profile")
    robot, recorded = make_transport(a.robot, a.robot_policy)
    source: PolicySource | TraceSource | None = None
    from_source = None
    if cfg.mode == "mirror":
        if not a.source:
            print("mirror mode needs --source policy:<name> or trace:<dir>", file=sys.stderr)
            return 2
        kind, _, rest = a.source.partition(":")
        if kind == "policy":
            source = PolicySource(make_policy(rest))
        elif kind == "trace":
            tsrc = TraceSource(rest)
            source, from_source = tsrc, tsrc.scenarios()
        else:
            print(f"--source must be policy:<name> or trace:<dir>, got {a.source!r}", file=sys.stderr)
            return 2
    base = recorded or from_source or [Scenario(seed=a.seed + i) for i in range(a.episodes or 1)]
    scenarios = list(base[:a.episodes] if a.episodes else base)
    twin = _twin_env(twin_name, twin_params)
    session = BridgeSession(robot, twin, cfg, source, robot_name=a.robot, twin_name=a.twin)
    try:
        session.run(scenarios)
    finally:
        robot.close()
        twin.close()
    res = build_report(session, a.out, robot=a.robot, twin_spec=a.twin, source=a.source,
                       make_twin=lambda: _twin_env(twin_name, twin_params), twin_params=twin_params, g5=not a.no_g5,
                       every=a.every, run_repro={"policy": a.repro_policy, "config": a.repro_config}
                       if a.repro_policy else None)
    out = Path(a.out)
    print(f"  wrote {out / 'bridge.md'}, {out / 'bridge.json'}, traces in {out / 'robot'} and {out / 'twin'}")
    c = res["comparison"]
    print(f"  {c.get('ticks', 0)} ticks compared, nRMSE {c.get('nrmse', float('nan')):.3f}; "
          f"trips: {sum(1 for e in res['episodes'] if e['trip'])}"
          + (f"; G5 {res['g5']['status']}" if res.get("g5") else ""))
    tripped = any(e["trip"] for e in res["episodes"])
    return 1 if (tripped and cfg.mode == "mirror") else 0


class _ParamTwin:
    """The twin env with its own scenario parameters (from --twin or a robot profile) added on every reset.

    Kept on the twin side, so a stand-in robot never receives the twin's parameters.
    """

    def __init__(self, env: Any, params: dict[str, Any]) -> None:
        self.env, self.params = env, dict(params)

    def reset(self, scenario: Any) -> Any:
        from groot_eval.types import Scenario

        return self.env.reset(Scenario(seed=scenario.seed, params={**scenario.params, **self.params},
                                       task=scenario.task, name=scenario.name))

    def __getattr__(self, name: str) -> Any:  # step, set_state, max_steps, dt, close, ...
        return getattr(self.env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("env", "params"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.env, name, value)


def _twin_env(name: str, params: dict[str, Any]) -> Any:
    from groot_eval.envs import make_env

    env = make_env(name)
    if any(k.startswith("wear.") for k in params):
        from groot_eval.degradation.wear import WearEnv

        env = WearEnv(env)
    return _ParamTwin(env, params) if params else env


def register_bridge(sub: argparse._SubParsersAction) -> None:
    sp = sub.add_parser("bridge", help="run the twin next to the robot (shadow or mirror) with a watchdog and a "
                                       "step-by-step comparison report")
    sp.add_argument("--mode", choices=["shadow", "mirror"], help="default: from --config, else shadow")
    sp.add_argument("--robot", required=True,
                    help="sim:<env>[:k=v,...] (stand-in robot), trace:<dir> (recorded robot) or ros2:<config.yaml>")
    sp.add_argument("--robot-policy", help="the robot's own controller for a sim:<env> robot (shadow mode, and "
                                           "dimensions outside command_mask in mirror mode)")
    sp.add_argument("--twin", required=True, help="twin env, e.g. arm3 or arm3:friction=1.2")
    sp.add_argument("--twin-profile", help="make the twin this robot: a profile YAML or "
                                           "registry:<root>:<robot>[@approved|@latest|@N]")
    sp.add_argument("--source", help="mirror mode: policy:<name> or trace:<dir> (open-loop replay on the robot)")
    sp.add_argument("--config", metavar="YAML", help="bridge config (watchdog limits, resync, command mask)")
    sp.add_argument("--episodes", type=int, help="default: every recorded episode, else 1")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--resync-every", type=int)
    sp.add_argument("--max-ticks", type=int)
    sp.add_argument("--every", type=int, default=10, help="report every Nth tick (all ticks around a trip)")
    sp.add_argument("--no-g5", action="store_true", help="skip the G5 dynamics gate")
    sp.add_argument("--repro-policy", help="also run groot-twin repro on tripped episodes with this policy")
    sp.add_argument("--repro-config", metavar="YAML", help="repro config for --repro-policy")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_bridge)


# -- calibration loop and robot profiles ----------------------------------------------
def _kv(items: list[str] | None) -> dict[str, object]:
    import yaml

    out: dict[str, object] = {}
    for item in items or []:
        k, _, v = item.partition("=")
        out[k] = yaml.safe_load(v)
    return out


def cmd_calibrate(a: argparse.Namespace) -> int:
    from groot_eval.calibration import calibrate
    from groot_eval.profiles import ProfileRegistry

    prof, rep = calibrate(a.trace, robot_id=a.robot_id, env=a.env, env_args=_kv(a.env_arg),
                          joint_names=a.joints.split(",") if a.joints else None, wear_config=a.wear,
                          sysid_config=a.sysid, refine=[k for k in (a.refine or "").split(",") if k],
                          rounds=a.rounds, episodes=a.episodes, during=a.during, operating_hours=a.operating_hours,
                          limits=a.limits)
    for st in rep["stages"]:
        print(f"  {st['stage']}: replay error {st['score']:.3f}")
    for n in rep["notes"]:
        print(f"  note: {n}")
    if a.registry:
        stored, created = ProfileRegistry(a.registry).publish(prof, note=a.note or "")
        print(f"  {'published' if created else 'unchanged, kept'} {stored.robot_id} v{stored.version} "
              f"in {a.registry} (candidate; approve it with `groot-robot profile approve`)")
    if a.out:
        print(f"  wrote {prof.save(a.out)}")
    if not a.out and not a.registry:
        print(prof.describe())
    return 0


def cmd_profile(a: argparse.Namespace) -> int:
    from groot_eval.profiles import ProfileError, ProfileRegistry, RobotProfile, diff, resolve_profile

    if a.action == "validate":
        bad = 0
        for path in a.items:
            try:
                p = RobotProfile.load(path)
                print(f"  ok  {path}: {p.robot_id}, {p.num_joints} joints, {p.content_hash}")
            except (ProfileError, OSError) as exc:
                bad += 1
                problems = exc.problems if isinstance(exc, ProfileError) else [str(exc)]
                print(f"  bad {path}:")
                for pr in problems:
                    print(f"      - {pr}")
        return 1 if bad else 0
    if a.action == "show":
        for spec in a.items:
            print(resolve_profile(spec).describe())
        return 0
    if a.action == "diff":
        if len(a.items) != 2:
            print("profile diff takes two profiles", file=sys.stderr)
            return 2
        changes = diff(resolve_profile(a.items[0]), resolve_profile(a.items[1]))
        print("\n".join(f"  {c}" for c in changes) or "  no changes")
        return 0
    if not a.registry:
        print(f"profile {a.action} needs --registry", file=sys.stderr)
        return 2
    reg = ProfileRegistry(a.registry)
    if a.action == "publish":
        for path in a.items:
            stored, created = reg.publish(RobotProfile.load(path), note=a.note or "")
            print(f"  {'published' if created else 'unchanged'} {stored.robot_id} v{stored.version}")
        return 0
    if a.action == "approve":
        if len(a.items) != 2:
            print("profile approve <robot_id> <version>", file=sys.stderr)
            return 2
        p = reg.approve(a.items[0], int(a.items[1]), by=a.by or "")
        print(f"  {p.robot_id} v{p.version} approved")
        return 0
    if a.action == "list":
        for rid in (a.items or reg.robots()):
            approved = reg._index()["robots"].get(rid, {}).get("approved")
            for v in reg.versions(rid):
                mark = "approved" if v["version"] == approved else ""
                score = v.get("fit_score")
                print(f"  {rid} v{v['version']:<3} {v['created']}  fit {'-' if score is None else f'{score:.3f}'}  "
                      f"{mark} {v.get('note') or ''}")
        return 0
    if a.action == "get":
        rid = a.items[0]
        which = a.items[1] if len(a.items) > 1 else "approved"
        p = reg.resolve(rid, which)
        if a.out:
            p.save(a.out)
            print(f"  wrote {a.out}")
        else:
            print(p.path)
        return 0
    print(f"unknown action {a.action}", file=sys.stderr)
    return 2


def cmd_fleet_fit(a: argparse.Namespace) -> int:
    import yaml

    from groot_eval.calibration import fit_fleet, fleet_markdown
    from groot_eval.profiles import ProfileRegistry, RobotProfile

    profiles = [RobotProfile.load(p) for p in a.profiles or []]
    if a.registry:
        profiles += ProfileRegistry(a.registry).all_profiles(a.which)
    if not profiles:
        print("no profiles given (--registry or profile files)", file=sys.stderr)
        return 2
    cfg, report = fit_fleet(profiles, a.wear)
    out = Path(a.out)
    header = ("# Wear profiles fitted from measured robots by `groot-robot fleet-fit`.\n"
              f"# Robots: {', '.join(report['robots'])}. Base: {a.wear}.\n")
    _write(out, header + yaml.safe_dump(cfg, sort_keys=False))
    _write(out.with_suffix(".md"), fleet_markdown(report))
    fitted = [r for r in report["effects"] if "fitted_at_eol" in r]
    print(f"  {len(fitted)} of {len(report['effects'])} effects refitted from {len(report['robots'])} robots")
    return 0


def register_profiles(sub: argparse._SubParsersAction) -> None:
    sp = sub.add_parser("calibrate", help="fit one robot's profile (sysid, effective age, per-joint wear, timing) "
                                          "from its trace")
    sp.add_argument("trace", help="the robot's trace directory")
    sp.add_argument("--robot-id", required=True, help="serial or asset id")
    sp.add_argument("--env", required=True, help="twin env, e.g. arm3")
    sp.add_argument("--env-arg", action="append", metavar="K=V")
    sp.add_argument("--joints", help="comma-separated joint names, in the env's joint order")
    sp.add_argument("--wear", metavar="YAML", help="wear profiles to fit the effective age against")
    sp.add_argument("--sysid", metavar="YAML", help="sweep config for system identification (groot-eval replay)")
    sp.add_argument("--refine", default="gain,friction", help="per-joint wear values to refine ('' = none)")
    sp.add_argument("--rounds", type=int, default=2)
    sp.add_argument("--episodes", type=int)
    sp.add_argument("--during", choices=["hold", "continue"], default="continue")
    sp.add_argument("--operating-hours", type=float, help="the robot's hour meter (needed for fleet-fit)")
    sp.add_argument("--limits", metavar="YAML", help="robot limits to copy in (configs/robots/*.yaml)")
    sp.add_argument("--out", help="profile YAML to write")
    sp.add_argument("--registry", help="publish the profile into this registry as a candidate")
    sp.add_argument("--note")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("profile", help="validate, show, diff, publish, approve, list or get robot profiles")
    sp.add_argument("action", choices=["validate", "show", "diff", "publish", "approve", "list", "get"])
    sp.add_argument("items", nargs="*", help="profile files, registry:<root>:<robot>[@approved|@latest|@N], "
                                             "or robot id and version")
    sp.add_argument("--registry")
    sp.add_argument("--note")
    sp.add_argument("--by", help="who approved (recorded in the registry)")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("fleet-fit", help="refit the wear curves from measured robot profiles")
    sp.add_argument("profiles", nargs="*", help="profile files")
    sp.add_argument("--registry")
    sp.add_argument("--which", default="latest", choices=["latest", "approved"])
    sp.add_argument("--wear", required=True, metavar="YAML", help="the wear config to refit (configs/wear/*.yaml)")
    sp.add_argument("--out", required=True, metavar="YAML")
    sp.set_defaults(func=cmd_fleet_fit)


# -- treadmill ------------------------------------------------------------------------
def cmd_treadmill(a: argparse.Namespace) -> int:
    from groot_eval.closed_loop import evaluate
    from groot_eval.envs import make_env
    from groot_eval.envs.treadmill import PerturbationConfig, perturbed_scenarios, schedule_label
    from groot_eval.policies import make_policy
    from groot_eval.stats import wilson_interval
    from groot_eval.trace import TraceRecorder
    from groot_eval.types import Scenario

    extra = _kv(a.param)
    base = [Scenario(seed=a.seed + i, params=dict(extra), task="walk") for i in range(a.episodes)]
    if a.config:
        pcfg = PerturbationConfig.load(a.config)
        scenarios = perturbed_scenarios(base, pcfg, a.seed)
    else:
        scenarios = base
    rec = TraceRecorder(domain="sim", env="treadmill", policy=a.policy, control_dt=0.02) if a.record else None
    env = make_env("treadmill")
    run = evaluate(env, make_policy(a.policy), scenarios, recorder=rec, env_name="treadmill")
    env.close()
    rows: dict[str, list[Any]] = {}
    for sc, r in zip(scenarios, run.results):
        rows.setdefault(schedule_label(sc.params), []).append(r)
    lines = [f"# Treadmill test: {a.policy}", "",
             f"{len(scenarios)} episodes, seed {a.seed}" + (f", perturbations from `{a.config}`" if a.config else "")
             + (f", parameters {extra}" if extra else ""), "",
             "| Perturbation | Episodes | Kept walking | 95% CI | Failures | Max lean (m) | Deck excursion (m) |",
             "|---|---|---|---|---|---|---|"]
    table = []
    for label, rs in sorted(rows.items()):
        ok = sum(1 for r in rs if r.success)
        lo, hi = wilson_interval(ok, len(rs))
        fails: dict[str, int] = {}
        for r in rs:
            if not r.success:
                fails[str(r.failure_reason)] = fails.get(str(r.failure_reason), 0) + 1
        lean = max(float(r.metrics.get("max_lean", 0.0)) for r in rs)
        deck = max(float(r.metrics.get("deck_excursion", 0.0)) for r in rs)
        table.append({"perturbation": label, "episodes": len(rs), "kept_walking": ok, "ci": [lo, hi],
                      "failures": fails, "max_lean": lean, "deck_excursion": deck})
        lines.append(f"| {label} | {len(rs)} | {100 * ok / len(rs):.0f}% | [{100 * lo:.0f}, {100 * hi:.0f}] | "
                     f"{fails or 'none'} | {lean:.3f} | {deck:.3f} |")
    detail = [{"seed": sc.seed, "schedule": sc.params.get("treadmill.perturbations") or [], "success": r.success,
               "failure_reason": r.failure_reason, "metrics": r.metrics} for sc, r in zip(scenarios, run.results)]
    lines += ["", "Each perturbation schedule is fixed by the seed and the scenario: rerunning with the same "
                  "`--seed` repeats every trip at the same point in the gait."]
    out = Path(a.out)
    _write(out / "treadmill.md", "\n".join(lines) + "\n")
    _write(out / "treadmill.json", json.dumps({"policy": a.policy, "seed": a.seed, "config": a.config,
                                               "summary": table, "episodes": detail}, indent=2, default=str))
    if rec is not None:
        print(f"  trace: {rec.write(out / 'trace')}")
    s = run.summary()
    print(f"  kept walking in {100 * s['success_rate']:.0f}% of {len(scenarios)} episodes; failures {s['failure_reasons'] or 'none'}")
    return 0


def cmd_treadmill_replay(a: argparse.Namespace) -> int:
    from groot_eval.envs import make_env
    from groot_eval.replay import replay_trace
    from groot_eval.trace import TraceSet

    ts = TraceSet.open(a.trace)
    if "belt_speed" not in ts.state_slices():
        print(f"{a.trace} has no state.belt_speed (not recorded on a treadmill)", file=sys.stderr)
        return 2
    base = _kv(a.param)
    env = make_env("treadmill")
    try:
        nominal = replay_trace(env, ts, {**base, "treadmill.speed": a.speed}, episodes=a.episodes,
                               reset_every=a.reset_every, label="nominal belt")
        recorded = replay_trace(env, ts, {**base, "treadmill.belt": f"trace:{a.trace}"}, episodes=a.episodes,
                                reset_every=a.reset_every, label="recorded belt")
    finally:
        env.close()
    lines = [f"# Treadmill replay: {a.trace}", "",
             f"Recorded actions replayed open loop, with the twin put back into the recorded state every "
             f"{a.reset_every} steps: walking is unstable (an inverted pendulum), so without resyncing any "
             "difference grows exponentially and swamps the comparison.", "", "| Belt in the twin | Replay error | Worst signals |", "|---|---|---|"]
    for r in (nominal, recorded):
        worst = ", ".join(f"{k} {v:.3f}" for k, v in sorted(r.signals.items(), key=lambda kv: -kv[1])[:3])
        lines.append(f"| {r.label} | {r.score:.4f} | {worst} |")
    lines += ["", "The recorded belt removes the belt from the sim/real gap: what is left is the walker's dynamics."]
    out = Path(a.out)
    _write(out / "treadmill_replay.md", "\n".join(lines) + "\n")
    _write(out / "treadmill_replay.json", json.dumps({"nominal": nominal.score, "recorded": recorded.score,
                                                      "per_signal_recorded": recorded.signals}, indent=2))
    print(f"  replay error: nominal belt {nominal.score:.4f}, recorded belt {recorded.score:.4f}")
    return 0


def register_treadmill(sub: argparse._SubParsersAction) -> None:
    sp = sub.add_parser("treadmill", help="walk the twin on a treadmill with scripted perturbations; survival by type")
    sp.add_argument("--policy", default="walker-capture")
    sp.add_argument("--config", metavar="YAML", help="perturbation catalogue (configs/treadmill/*.yaml)")
    sp.add_argument("--episodes", type=int, default=50)
    sp.add_argument("--seed", type=int, default=0, help="fixes every scenario and perturbation schedule")
    sp.add_argument("--param", action="append", metavar="K=V", help="extra scenario parameters, e.g. treadmill.speed=1.2")
    sp.add_argument("--record", action="store_true", help="also write the episodes as a trace (with state.belt_speed)")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_treadmill)

    sp = sub.add_parser("treadmill-replay", help="replay a treadmill trace in the twin with the recorded vs a nominal belt")
    sp.add_argument("trace")
    sp.add_argument("--speed", type=float, default=0.8, help="nominal belt speed to compare against")
    sp.add_argument("--reset-every", type=int, default=10, help="resync the twin to the recording every N steps")
    sp.add_argument("--episodes", type=int)
    sp.add_argument("--param", action="append", metavar="K=V")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_treadmill_replay)


# -- test rig and run seeds -----------------------------------------------------------
def cmd_rig(a: argparse.Namespace) -> int:
    from groot_eval.rig import RigConfig, compare_to_ideal, rig_markdown
    from groot_eval.trace import TraceRecorder

    cfg = RigConfig.load(a.config)
    if a.clock:
        cfg.clock = a.clock
    rec = TraceRecorder(domain="sim", env=f"rig:{cfg.plant.get('env', cfg.plant.get('kind'))}", policy=cfg.policy,
                        control_dt=1.0 / cfg.control_hz) if a.record else None
    publisher = None
    if a.ros2_publish:
        from groot_eval.rig_ros2 import Ros2RigPublisher

        publisher = Ros2RigPublisher([t.key for t in cfg.topics], prefix=a.ros2_publish)
    try:
        cmp = compare_to_ideal(cfg, a.seed, a.episodes, recorder=rec, on_arrive=publisher)
    finally:
        if publisher is not None:
            publisher.close()
    out = Path(a.out)
    _write(out / "rig.md", rig_markdown(cfg, cmp, a.config))
    _write(out / "rig.json", json.dumps(cmp, indent=2, default=str))
    if rec is not None:
        print(f"  trace (observations as the controller saw them): {rec.write(out / 'trace')}")
    print(f"  ideal {cmp['ideal']['success']}/{a.episodes}, rig {cmp['rig']['success']}/{a.episodes}; "
          f"command drift RMS {cmp['command_rms_drift']:.4f}")
    return 0


def cmd_seed_check(a: argparse.Namespace) -> int:
    from groot_eval.closed_loop import evaluate
    from groot_eval.envs import make_env
    from groot_eval.policies import make_policy
    from groot_eval.seeding import determinism_check, episode_meta
    from groot_eval.trace import TraceRecorder
    from groot_eval.types import Scenario

    extra = _kv(a.param)

    def one_run(run: Any, out: Path) -> Path:
        rec = TraceRecorder(domain="sim", env=a.env, policy=a.policy, control_dt=0.02)
        if a.rig:
            from groot_eval.rig import RigConfig, run_rig

            run_rig(RigConfig.load(a.rig), run.value, a.episodes, recorder=rec)
            return rec.write(out)
        base = [Scenario(seed=s, params=dict(extra)) for s in run.scenario_seeds(a.episodes)]
        if a.treadmill:
            from groot_eval.envs.treadmill import PerturbationConfig, perturbed_scenarios

            base = perturbed_scenarios(base, PerturbationConfig.load(a.treadmill), run.derive("treadmill"))
        env, pol = make_env(a.env), make_policy(a.policy)
        for i, sc in enumerate(base):
            # stamp each episode with its run seed so the trace can be replayed with the same draws
            evaluate(env, pol, [sc], recorder=_Stamped(rec, episode_meta(run, i)))  # type: ignore[arg-type]
        env.close()
        return rec.write(out)

    rep = determinism_check(one_run, a.seed, Path(a.out) / "runs", runs=a.runs)
    title = f"{a.policy} on {a.rig or a.env}"
    _write(Path(a.out) / "seed_check.md", rep.markdown(title))
    _write(Path(a.out) / "seed_check.json", json.dumps({"seed": rep.seed, "deterministic": rep.deterministic,
                                                        "fingerprints": rep.fingerprints,
                                                        "differences": rep.differences}, indent=2))
    print(f"  {'deterministic' if rep.deterministic else 'NOT deterministic'}: "
          f"{len(set(rep.fingerprints))} distinct fingerprint(s) over {rep.runs} runs")
    return 0 if rep.deterministic else 1


class _Stamped:
    """Adds run-seed metadata to every episode a recorder starts."""

    def __init__(self, rec: Any, meta: dict[str, Any]) -> None:
        self.rec, self.meta = rec, meta

    def start_episode(self, task: str = "", **meta: Any) -> None:
        self.rec.start_episode(task, **{**meta, **self.meta})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.rec, name)


def register_rig(sub: argparse._SubParsersAction) -> None:
    sp = sub.add_parser("rig", help="feed a controller from many publishers (twin, replay, signal, live) on a real or "
                                    "simulated clock, with stream faults; compare to ideal streams")
    sp.add_argument("--config", required=True, metavar="YAML", help="rig config (configs/rig/*.yaml)")
    sp.add_argument("--episodes", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0, help="run seed: fixes scenarios, jitter, drops and stream faults")
    sp.add_argument("--clock", choices=["sim", "real"], help="override the config's clock")
    sp.add_argument("--record", action="store_true", help="record the observations as the controller saw them")
    sp.add_argument("--ros2-publish", metavar="PREFIX", help="also publish every delivered message on ROS 2 under PREFIX")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_rig)

    sp = sub.add_parser("seed-check", help="run the same seed twice and check the runs are identical")
    sp.add_argument("--env", default="arm3")
    sp.add_argument("--policy", default="arm-ik")
    sp.add_argument("--episodes", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--runs", type=int, default=2)
    sp.add_argument("--param", action="append", metavar="K=V")
    sp.add_argument("--treadmill", metavar="YAML", help="perturbation catalogue, for --env treadmill")
    sp.add_argument("--rig", metavar="YAML", help="check a rig config instead of a plain env run")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_seed_check)


REGISTRARS: list[Register] = [register_actuators, register_scheduler, register_bridge, register_profiles,
                              register_treadmill, register_rig]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="groot-robot", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for reg in REGISTRARS:
        reg(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
