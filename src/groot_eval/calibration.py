"""Calibration loop: turn a robot's own traces into its profile, and a fleet of profiles into wear curves.

``calibrate`` runs, against one robot's recording:

1. **System identification** (optional, ``replay.system_id``): env-level
   physics such as friction or actuation delay, from a sweep config.
2. **Effective age** (``degradation.calibrate.fit_age``, i.e. ``wear-fit``):
   the typical unit of each candidate age is replayed and the best age kept.
3. **Per-joint refinement**: coordinate descent over per-joint multipliers on
   the wear values replay can see (by default gain and friction), so the
   profile describes this unit rather than the fleet median.
4. **Timing** (``scheduler.TimingProfile.from_trace``): latency, jitter and
   missed ticks.

The result is a :class:`~groot_eval.profiles.RobotProfile`. Its provenance
records the trace hash, episodes, the replay error before and after each
stage, and notes. A step that does not improve tracking is not kept.

``fit_fleet`` closes the loop on the placeholder curves in ``configs/wear``.
Given profiles of several robots and each robot's operating hours, it fits
every effect's end-of-life value (and its unit-to-unit spread) by least
squares on the effect's curve, and writes a wear config in the usual format.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from groot_eval.actuators.core import WEAR_KEYS
from groot_eval.profiles import RobotProfile, other_wear_from_params, tree_sha256, wear_lists_from_params
from groot_eval.trace import TraceSet

EnvFactory = Callable[[], Any]
REFINE_FACTORS = (0.6, 0.8, 1.0, 1.25, 1.6)
SCALE_KEYS = ("gain", "torque_limit", "friction", "damping")  # multiplicative wear values replay can see


def _score(make_env: EnvFactory, ts: TraceSet, params: dict[str, Any], episodes: int | None) -> float:
    from groot_eval.degradation.wear import WearEnv
    from groot_eval.replay import replay_trace

    env = WearEnv(make_env())
    try:
        return float(replay_trace(env, ts, params, episodes=episodes).score)
    finally:
        env.close()


def _better(new: float, old: float, tol: float = 1e-9) -> bool:
    return math.isfinite(new) and (not math.isfinite(old) or new < old - tol)


def refine_joints(make_env: EnvFactory, ts: TraceSet, params: dict[str, Any], num_joints: int,
                  keys: Sequence[str] = ("gain", "friction"), factors: Sequence[float] = REFINE_FACTORS,
                  rounds: int = 2, episodes: int | None = None) -> dict[str, Any]:
    """Coordinate descent on per-joint multipliers of ``keys``. Returns the new params, scores and moves."""
    for k in keys:
        if k not in SCALE_KEYS:
            raise ValueError(f"can refine only multiplicative wear values {SCALE_KEYS}, not {k!r}")
    cur = dict(params)
    best = _score(make_env, ts, cur, episodes)
    start = best
    moves = []
    for r in range(rounds):
        changed = False
        for k in keys:
            kind, eff, default = WEAR_KEYS[k]
            for j in range(num_joints):
                name = f"wear.{kind}[{j}].{eff}"
                base = float(cur.get(name, default))
                for f in factors:
                    if f == 1.0:
                        continue
                    trial = {**cur, name: base * f}
                    s = _score(make_env, ts, trial, episodes)
                    if _better(s, best, tol=1e-4 * max(best, 1e-9)):
                        best, cur, changed = s, trial, True
                        moves.append({"round": r, "key": k, "joint": j, "from": base, "to": base * f, "score": s})
                        base = base * f
        if not changed:
            break
    return {"params": cur, "score": best, "start_score": start, "moves": moves}


def trace_provenance(ts: TraceSet) -> dict[str, Any]:
    steps = sum(len(f) for f in ts.frames())
    meta = ts.episode_meta()
    return {"path": str(ts.root), "sha256": tree_sha256(ts.root), "domain": ts.domain, "episodes": len(meta),
            "steps": int(steps), "fps": ts.fps}


def calibrate(trace: str | Path, *, robot_id: str, env: str, joint_names: list[str] | None = None,
              env_args: dict[str, Any] | None = None, wear_config: str | Path | None = None,
              sysid_config: str | Path | None = None, refine: Sequence[str] = ("gain", "friction"),
              rounds: int = 2, episodes: int | None = None, during: str = "continue",
              operating_hours: float | None = None, limits: str | Path | None = None,
              ages: list[float] | None = None) -> tuple[RobotProfile, dict[str, Any]]:
    """Fit one robot's profile from its trace. Returns ``(profile, report)``."""
    from groot_eval.degradation.calibrate import fit_age
    from groot_eval.degradation.components import probe_components
    from groot_eval.degradation.wear import WearConfig
    from groot_eval.envs import make_env
    from groot_eval.scheduler import TimingProfile

    env_args = dict(env_args or {})

    def factory() -> Any:
        return make_env(env, **env_args)

    ts = TraceSet.open(trace)
    comps = probe_components(factory())
    J = sum(1 for c in comps.values() if c.kind == "motor") or len(joint_names or [])
    if not J:
        raise ValueError(f"cannot tell how many joints {env!r} has; pass joint_names")
    names = list(joint_names) if joint_names else [f"joint{j}" for j in range(J)]
    if len(names) != J:
        raise ValueError(f"{len(names)} joint names for {J} joints in {env!r}")
    notes: list[str] = []
    stages: list[dict[str, Any]] = []
    sim_params: dict[str, Any] = {}
    baseline = _score(factory, ts, {}, episodes)
    stages.append({"stage": "new unit, default physics", "score": baseline})
    current = baseline

    if sysid_config:
        from groot_eval.randomization import SweepConfig
        from groot_eval.replay import system_id

        e = factory()
        try:
            fit = system_id(e, ts, SweepConfig.load(sysid_config), episodes=episodes)
        finally:
            e.close()
        best = next((r for r in fit["ranked"] if math.isfinite(r.score)), None)
        if best is not None and _better(best.score, current):
            sim_params, current = dict(best.params), best.score
            stages.append({"stage": "system identification", "score": best.score, "params": sim_params})
        else:
            notes.append("system identification did not beat the default physics; sim params left at defaults")
        notes += [n for n in fit["notes"] if "edge" in n or "not pinned" in n]

    wear_params: dict[str, Any] = {}
    age = None
    if wear_config:
        wcfg = WearConfig.load(wear_config)
        fa = fit_age(factory, ts, wcfg, comps, ages, sim_params, episodes)
        if fa["best"] is not None:
            cand = {k: v for k, v in fa["best"]["params"].items() if k.startswith("wear.")}
            s = _score(factory, ts, {**sim_params, **cand}, episodes)
            if _better(s, current) or not cand:
                wear_params, current, age = cand, s, float(fa["best"]["age_hours"])
                stages.append({"stage": f"effective age {age:g} h (typical unit)", "score": s})
            else:
                notes.append("no age tracked better than a new unit")
            notes += [n for n in fa["notes"] if "replay" not in n]
    if refine:
        rf = refine_joints(factory, ts, {**sim_params, **wear_params}, J, refine, rounds=rounds, episodes=episodes)
        if _better(rf["score"], current):
            wear_params = {k: v for k, v in rf["params"].items() if k.startswith("wear.")}
            current = rf["score"]
            stages.append({"stage": f"per-joint refinement ({', '.join(refine)})", "score": current,
                           "moves": rf["moves"]})
        else:
            notes.append("per-joint refinement did not improve tracking; kept the typical unit")
    if math.isfinite(baseline) and math.isfinite(current) and baseline > 0:
        notes.append(f"replay error {baseline:.3f} -> {current:.3f} ({1 - current / baseline:.0%} lower than a new unit)")
    timing = None
    try:
        timing = TimingProfile.from_trace(ts.root, during_inference=during).to_dict()
        if not timing.get("source", {}).get("latency_samples"):
            notes.append("the trace has no inference_latency_ms; timing latency is 0")
    except (KeyError, ValueError) as exc:
        notes.append(f"timing not measured: {exc}")

    limits_data: dict[str, Any] = {}
    if limits:
        import yaml

        with open(limits, encoding="utf-8") as fh:
            limits_data = yaml.safe_load(fh) or {}
    from groot_eval import __version__ as tool_version

    prof = RobotProfile(
        robot_id=robot_id, joint_names=names, embodiment=str(limits_data.get("embodiment") or env),
        name=str(limits_data.get("name") or robot_id), operating_hours=operating_hours,
        control_hz=limits_data.get("control_hz") or (timing or {}).get("control_hz"),
        state_limits=limits_data.get("state_limits") or {}, action_range=limits_data.get("action_range"),
        max_action_rate=limits_data.get("max_action_rate"),
        wear={"effective_age_hours": age, "values": wear_lists_from_params(wear_params, J),
              **({"other": other_wear_from_params(wear_params, J)} if other_wear_from_params(wear_params, J) else {}),
              "source": "calibrated" if wear_params else "default"} if (wear_params or wear_config) else {},
        sim={"env": env, **({"env_args": env_args} if env_args else {}), "params": sim_params},
        timing=timing,
        provenance={"method": "groot-robot calibrate", "tool_version": tool_version,
                    "traces": [trace_provenance(ts)], "wear_config": str(wear_config) if wear_config else None,
                    "sysid_config": str(sysid_config) if sysid_config else None,
                    "fit": {"baseline_score": baseline, "score": current, "episodes": episodes or len(ts.episode_meta()),
                            "stages": stages},
                    "notes": notes},
    )
    if prof.wear.get("effective_age_hours") is None:
        prof.wear.pop("effective_age_hours", None)
    return prof, {"stages": stages, "notes": notes, "baseline": baseline, "score": current}


# =====================================================================================
# Fleet
# =====================================================================================
def _curve(curve: str, age: float, eol: float, load: float) -> float:
    from groot_eval.degradation.wear import Effect

    return Effect("x", "y", curve, 0.0, 1.0).value(age, eol, load, 1.0, 1.0)


def fit_fleet(profiles: Sequence[RobotProfile], base: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit every effect of the ``base`` wear config from measured robots.

    Each profile contributes one point per joint: (operating hours, measured value). For an effect with
    curve ``c``, ``value = start + (at_eol - start) * c(age / eol * load)``, so ``at_eol`` has a closed-form
    least-squares solution; ``spread`` is the standard deviation of the log ratio of measured to fitted change.
    Signed effects (biases) are fitted on their magnitude. Effects with fewer than two informative points keep
    their placeholder value and are listed in the report.
    """
    import yaml

    from groot_eval.degradation.wear import WearConfig

    with open(base, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = WearConfig.from_dict(raw)
    usable = [p for p in profiles if p.operating_hours is not None and (p.wear or {}).get("values")]
    report: dict[str, Any] = {"robots": [p.robot_id for p in usable],
                              "skipped": [p.robot_id for p in profiles if p not in usable], "effects": []}
    by_effect = {(kind, eff): key for key, (kind, eff, _) in WEAR_KEYS.items()}
    effects_out = dict(raw.get("effects") or {})
    for e in cfg.effects:
        name = f"{e.kind}.{e.key}"
        key = by_effect.get((e.kind, e.key))
        row: dict[str, Any] = {"effect": name, "placeholder_at_eol": e.at_eol, "points": 0}
        if key is None:
            row["note"] = "not a per-joint actuator value; left as is"
            report["effects"].append(row)
            continue
        ys, vs = [], []
        for p in usable:
            values = p.wear["values"].get(key)
            if not values:
                continue
            for j, v in enumerate(values):
                load = cfg.load_factor.get(f"{e.kind}[{j}]", cfg.load_factor.get("default", 1.0))
                y = _curve(e.curve, float(p.operating_hours or 0.0), cfg.eol_hours, load)
                ys.append(y)
                vs.append(abs(float(v) - e.start) if e.signed else float(v) - e.start)
        y_arr, v_arr = np.asarray(ys), np.asarray(vs)
        info = y_arr > 1e-9
        row["points"] = int(info.sum())
        if info.sum() < 2:
            row["note"] = "fewer than two robots past zero hours; placeholder kept"
            report["effects"].append(row)
            continue
        delta = float(np.sum(v_arr[info] * y_arr[info]) / np.sum(y_arr[info] ** 2))
        at_eol = e.start + delta
        ratio = v_arr[info] / (delta * y_arr[info]) if delta != 0 else np.full(int(info.sum()), np.nan)
        pos = ratio[np.isfinite(ratio) & (ratio > 0)]
        spread = float(np.std(np.log(pos))) if pos.size >= 2 else e.spread
        resid = v_arr[info] - delta * y_arr[info]
        r2 = 1 - float(np.sum(resid ** 2)) / max(float(np.sum((v_arr[info] - v_arr[info].mean()) ** 2)), 1e-18)
        spec = dict(effects_out.get(name) or {})
        spec.update({"at_eol": round(at_eol, 6), "spread": round(spread, 4)})
        effects_out[name] = spec
        row.update({"fitted_at_eol": at_eol, "spread": spread, "r2": r2})
        report["effects"].append(row)
    out = dict(raw)
    out["effects"] = effects_out
    out["fitted_from"] = {"robots": report["robots"], "base": str(base)}
    return out, report


def fleet_markdown(report: dict[str, Any]) -> str:
    lines = ["# Fleet wear fit", "", f"Robots: {', '.join(report['robots']) or 'none'}"
             + (f"; skipped (no operating hours or wear): {', '.join(report['skipped'])}" if report["skipped"] else ""),
             "", "| Effect | Points | Placeholder at EOL | Fitted at EOL | Spread | R² | Note |",
             "|---|---|---|---|---|---|---|"]
    for r in report["effects"]:
        f = r.get("fitted_at_eol")
        lines.append(f"| {r['effect']} | {r['points']} | {r['placeholder_at_eol']:g} | "
                     f"{'-' if f is None else f'{f:.4g}'} | {'-' if 'spread' not in r else format(r['spread'], '.3g')} | "
                     f"{'-' if 'r2' not in r else format(r['r2'], '.2f')} | {r.get('note', '')} |")
    return "\n".join(lines) + "\n"
