"""Failure reproduction: take a failed run and try to make the digital twin fail the same way.

The single most useful test of a twin is whether it reproduces the failures
seen on the robot (or in a higher-fidelity sim). This module does that for one
failure or a batch, in stages, cheapest evidence first:

1. **Load** the failure: a sweep run directory (``episodes.jsonl``), a
   regression suite (``suite.json``), a trace / LeRobot recording (failed
   episodes, with the recorded states and actions), or a single JSON record.
2. **Open-loop replay** (when states and actions were recorded): put the twin
   in the recorded start state, play the recorded actions, and measure where
   the twin's trajectory leaves the recorded one (per-signal nRMSE, divergence
   step, DTW) and whether the twin ends the episode the same way. The policy is
   out of the loop, so a gap here is the twin's dynamics.
3. **Closed-loop re-run**: run the same policy on the same scenario ``repeats``
   times (non-determinism check) and on ``seed_jitter`` neighbouring seeds with
   the same parameters (is the failure a knife edge or a region?). Score the
   twin's failure rate and same-mode rate with Wilson intervals.
4. **Hypothesis search** (when the baseline twin does not reproduce it): sweep
   the twin's fidelity knobs (physics, actuator delay, sensor noise, latency
   model, wear) and rank settings by how well they reproduce the failure. The
   winning setting is a concrete ticket against the twin ("needs 2 steps of
   actuator delay"); a knob that only reproduces at the edge of its range is
   flagged so the range is widened.

Verdicts: ``reproduced``, ``reproduced_with_knobs``, ``partial`` (the twin
fails, but differently or rarely), ``not_reproduced``, ``flaky`` (identical
re-runs disagree, so the twin's determinism must be fixed before anything
else). A batch reports the failure reproduction rate with an interval and a
failure-mode confusion matrix (:mod:`groot_eval.twin_metrics`).
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval import twin_metrics as tm
from groot_eval.stats import wilson_interval
from groot_eval.types import Scenario

VERDICTS = ("reproduced", "reproduced_with_knobs", "partial", "not_reproduced", "flaky")

DEFAULTS: dict[str, Any] = {
    "env": "reach2d",
    "env_args": {},
    "policy": None,
    "policy_args": {},
    "repeats": 5,  # identical re-runs of the failing scenario
    "seed_jitter": 10,  # neighbouring seeds, same parameters
    "confidence": 0.95,
    "min_same_mode_rate": 0.5,  # twin reproduces when at least this share of runs fail the same way
    "min_fail_rate": 0.3,  # below this the twin is "not reproducing" even with a mode match
    "divergence_threshold": 0.5,  # replay: error in units of the signal's spread
    "replay": True,
    "knobs": {},  # name -> list of values to try
    "knob_repeats": 3,
    "knob_seed_jitter": 5,
    "max_knob_settings": 64,
    "latency_mode": None,  # e.g. "trace:<dir>" or "fixed:120" (groot_eval.latency)
    "during_inference": "hold",
    "mode_aliases": {},  # map labels across domains, e.g. {"dropped_object": "drop"}
}

# Which twin knobs usually explain which failure mode. Used to suggest a search when the
# config has none; override or extend under `knob_hints` in the repro config.
KNOB_HINTS: dict[str, list[str]] = {
    "collision": ["action_delay", "latency_mode", "sensor_noise", "obstacle_radius"],
    "out_of_bounds": ["actuator_gain", "mass", "friction", "action_delay"],
    "timeout": ["friction", "actuator_gain", "mass"],
    "drop": ["friction", "mass", "gripper_force", "sensor_noise"],
    "slip": ["friction", "mass", "gripper_force"],
    "missed_grasp": ["sensor_noise", "camera_offset", "latency_mode", "action_delay"],
    "joint_limit": ["actuator_gain", "action_delay"],
    "fall": ["friction", "action_delay", "latency_mode", "mass"],
    "non_finite_action": ["sensor_noise"],
}


def load_config(path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        unknown = set(data) - set(DEFAULTS) - {"knob_hints"}
        if unknown:
            raise ValueError(f"unknown repro config keys {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        cfg.update(data)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# =====================================================================================
# 1. Loading failures
# =====================================================================================
@dataclass
class FailureCase:
    case_id: str
    scenario: Scenario
    failure_reason: str
    source: str  # sweep | suite | trace | record
    domain: str = "unknown"  # where the failure happened: real | sim | unknown
    states: np.ndarray | None = None  # (T, S) recorded observation.state
    actions: np.ndarray | None = None  # (T, A) recorded actions
    slices: dict[str, slice] | None = None  # state key -> columns of `states`
    fail_step: int | None = None
    latencies_ms: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_recording(self) -> bool:
        return self.states is not None and self.actions is not None and len(self.states) > 1


def _norm_mode(mode: Any, aliases: dict[str, str]) -> str:
    m = str(mode or "unknown").strip()
    return aliases.get(m, m)


def load_failures(path: str | Path, episodes: list[int] | None = None, domain: str | None = None) -> list[FailureCase]:
    """Read failed episodes from a sweep run dir, a suite JSON, a trace / LeRobot dataset or a record JSON."""
    p = Path(path)
    if p.is_dir() and (p / "meta" / "info.json").exists():
        return _from_trace(p, episodes, domain)
    if p.is_dir() and (p / "episodes.jsonl").exists():
        return _from_sweep(p, episodes, domain)
    if p.is_file() and p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "cases" in data:
            return _from_suite(data, episodes, domain)
        records = data if isinstance(data, list) else [data]
        return [_from_record(r, i, p.parent, domain) for i, r in enumerate(records)]
    raise ValueError(f"{path}: not a trace/LeRobot dataset, sweep run directory, suite.json or failure record JSON")


def _from_sweep(root: Path, episodes: list[int] | None, domain: str | None) -> list[FailureCase]:
    out = []
    for i, line in enumerate((root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()):
        if not line.strip() or (episodes is not None and i not in episodes):
            continue
        d = json.loads(line)
        if d.get("success"):
            continue
        sc = Scenario.from_dict(d)
        out.append(FailureCase(f"sweep:{i}:{sc.scenario_id}", sc, str(d.get("failure_reason") or "unknown"), "sweep",
                               domain or "sim", fail_step=d.get("steps"), latencies_ms=list(d.get("latencies_ms") or [])))
    return out


def _from_suite(suite: dict[str, Any], episodes: list[int] | None, domain: str | None) -> list[FailureCase]:
    out = []
    for i, case in enumerate(suite["cases"]):
        if case.get("origin") != "failure" or (episodes is not None and i not in episodes):
            continue
        sc = Scenario.from_dict(case["scenario"])
        out.append(FailureCase(f"suite:{i}:{sc.scenario_id}", sc, str(case.get("failure_reason") or "unknown"), "suite",
                               domain or "sim", meta={"source_policy": suite.get("source_policy")}))
    return out


def _from_trace(root: Path, episodes: list[int] | None, domain: str | None) -> list[FailureCase]:
    from groot_eval.trace import TraceSet

    ts = TraceSet.open(root)
    slices = ts.state_slices()
    frames = ts.frames()
    out = []
    for meta, frame in zip(ts.episode_meta(), frames):
        idx = int(meta["episode_index"])
        if episodes is not None and idx not in episodes:
            continue
        if meta.get("success") is not False and episodes is None:
            continue  # only explicitly failed episodes, unless the caller picked them
        scen = meta.get("scenario") or {}
        sc = Scenario(seed=int(scen.get("seed", idx)), params=dict(scen.get("params") or {}),
                      task=str(scen.get("task") or (meta.get("tasks") or [""])[0]), name=str(scen.get("name") or ""))
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
        lat = frame["inference_latency_ms"].to_numpy(dtype=float).tolist() if "inference_latency_ms" in frame else []
        out.append(FailureCase(f"trace:{root.name}:{idx}", sc, str(meta.get("failure_reason") or "unknown"), "trace",
                               domain or ts.domain, states, actions, slices, fail_step=len(states),
                               latencies_ms=[x for x in lat if math.isfinite(x)],
                               meta={"trace": str(root), "episode_index": idx,
                                     "scenario_known": bool(scen)}))
    return out


def _from_record(rec: dict[str, Any], i: int, base: Path, domain: str | None) -> FailureCase:
    """``{"scenario": {...}, "failure_reason": "...", "trace": "dir", "episode_index": 3, "domain": "real"}``."""
    if rec.get("trace"):
        tr = Path(rec["trace"])
        tr = tr if tr.is_absolute() else base / tr
        cases = _from_trace(tr, [int(rec.get("episode_index", 0))], domain or rec.get("domain"))
        if not cases:
            raise ValueError(f"record {i}: episode {rec.get('episode_index', 0)} not found in {tr}")
        c = cases[0]
        if rec.get("scenario"):
            c.scenario = Scenario.from_dict(rec["scenario"])
            c.meta["scenario_known"] = True
        if rec.get("failure_reason"):
            c.failure_reason = str(rec["failure_reason"])
        c.source = "record"
        return c
    sc = Scenario.from_dict(rec["scenario"])
    return FailureCase(f"record:{i}:{sc.scenario_id}", sc, str(rec.get("failure_reason") or "unknown"), "record",
                       domain or rec.get("domain", "unknown"), fail_step=rec.get("fail_step"))


# =====================================================================================
# 2. Open-loop replay
# =====================================================================================
def replay_case(env: Any, case: FailureCase, params: dict[str, Any] | None = None,
                threshold: float = 0.5) -> dict[str, Any]:
    """Play the recorded actions through the twin from the recorded start state."""
    from groot_eval.replay import _flatten, _state_dict

    states, actions, slices = case.states, case.actions, case.slices or {}
    T = len(states)
    obs = env.reset(Scenario(seed=case.scenario.seed, params={**case.scenario.params, **(params or {})},
                             task=case.scenario.task))
    if hasattr(env, "max_steps") and env.max_steps < T + 1:
        env.max_steps = T + 1
    synced = True
    try:
        obs = env.set_state(_state_dict(states[0], slices), actions[:0])
    except (AttributeError, NotImplementedError):
        synced = False
    pred = np.full_like(states, np.nan)
    pred[0] = _flatten(obs, slices)
    end_step, end_info = None, {}
    for t in range(T - 1):
        obs, _, term, trunc, info = env.step(actions[t])
        pred[t + 1] = _flatten(obs, slices)
        if term or trunc:
            end_step, end_info = t + 1, info
            break
    live = slice(0, (end_step or T - 1) + 1)
    scale = np.nanstd(states, axis=0)
    moving = np.nanstd(states, axis=0) > 1e-9
    per_dim = tm.nrmse(pred[live], states[live], np.where(moving, scale, 0.0))
    per_signal = {}
    for k, s in slices.items():
        vals = per_dim[s][np.isfinite(per_dim[s])]
        if vals.size:
            per_signal[k] = float(np.sqrt(np.mean(vals ** 2)))
    div = tm.divergence_step(pred[live], states[live], np.where(moving, scale, 0.0), threshold)
    ended_as = None
    if end_step is not None:
        ended_as = "success" if end_info.get("success") else str(end_info.get("failure_reason", "unknown"))
    score = [v for v in per_signal.values() if math.isfinite(v)]
    return {
        "synced": synced,
        "steps_recorded": T,
        "steps_replayed": int(end_step or T - 1),
        "nrmse": float(np.mean(score)) if score else math.nan,
        "nrmse_by_signal": per_signal,
        "divergence_step": div,
        "dtw": tm.dtw_distance(pred[live][::max(1, T // 200)], states[live][::max(1, T // 200)]) if T > 2 else math.nan,
        "ended_early_at": end_step if end_step is not None and end_step < T - 1 else None,
        "ended_as": ended_as,
    }


# =====================================================================================
# 3. Closed-loop re-runs
# =====================================================================================
def _make(cfg: dict[str, Any]) -> tuple[Any, Any]:
    from groot_eval.envs import make_env
    from groot_eval.policies import make_policy

    if not cfg.get("policy"):
        raise ValueError("closed-loop reproduction needs a policy (config `policy:` or --policy)")
    return make_env(cfg["env"], **(cfg.get("env_args") or {})), make_policy(cfg["policy"], **(cfg.get("policy_args") or {}))


def _with_latency(policy: Any, env: Any, mode: str | None, during: str, seed: int) -> Any:
    if not mode:
        return policy
    from groot_eval.latency import with_latency  # present in the harness; optional here

    return with_latency(policy, mode, getattr(env, "dt", None), during, seed)


def closed_loop_runs(env: Any, policy: Any, case: FailureCase, params: dict[str, Any] | None, repeats: int,
                     seed_jitter: int, latency_mode: str | None = None, during: str = "hold",
                     aliases: dict[str, str] | None = None) -> dict[str, Any]:
    """Re-run the policy on the failing scenario (same seed) and on neighbouring seeds."""
    from groot_eval.closed_loop import run_episode

    aliases = aliases or {}
    sc = case.scenario
    merged = {**sc.params, **(params or {})}
    same, near = [], []
    for k in range(repeats):
        pol = _with_latency(policy, env, latency_mode, during, sc.seed + 7919 * k)
        r = run_episode(env, pol, Scenario(seed=sc.seed, params=merged, task=sc.task, name=sc.name))
        same.append((r.success, None if r.success else _norm_mode(r.failure_reason, aliases), r.steps))
    for k in range(1, seed_jitter + 1):
        pol = _with_latency(policy, env, latency_mode, during, sc.seed + k)
        r = run_episode(env, pol, Scenario(seed=sc.seed + k, params=merged, task=sc.task, name=sc.name))
        near.append((r.success, None if r.success else _norm_mode(r.failure_reason, aliases), r.steps))
    return {"same_seed": same, "neighbours": near}


def _score(case: FailureCase, runs: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    mode = _norm_mode(case.failure_reason, cfg.get("mode_aliases") or {})
    conf = float(cfg["confidence"])
    same = [(ok, m) for ok, m, _ in runs["same_seed"]]
    near = [(ok, m) for ok, m, _ in runs["neighbours"]]
    pooled = same + near
    s_same = tm.reproduction_score(mode, same, conf) if same else None
    s_near = tm.reproduction_score(mode, near, conf) if near else None
    s_all = tm.reproduction_score(mode, pooled, conf)
    flaky = len({(ok, m, st) for ok, m, st in runs["same_seed"]}) > 1
    ttf = [st for ok, m, st in runs["same_seed"] + runs["neighbours"] if not ok and m == mode]
    # The case is about *this* scenario: judge it on its own seed; neighbours say whether it is a region or a knife edge.
    primary = s_same or s_all
    return {"mode": mode, "same_seed": s_same, "neighbours": s_near, "pooled": s_all, "primary": primary,
            "flaky": flaky, "twin_time_to_failure_median": float(np.median(ttf)) if ttf else None}


def _verdict(score: dict[str, Any], cfg: dict[str, Any]) -> str:
    p = score["primary"]
    if score["flaky"] and p["same_mode_rate"] < cfg["min_same_mode_rate"]:
        return "flaky"
    if p["same_mode_rate"] >= cfg["min_same_mode_rate"] and p["fail_rate"] >= cfg["min_fail_rate"]:
        return "reproduced"
    if p["fail_rate"] >= cfg["min_fail_rate"] or p["same_mode_rate"] > 0 or score["pooled"]["same_mode_rate"] > 0:
        return "partial"
    return "not_reproduced"


# =====================================================================================
# 4. Hypothesis search over twin knobs
# =====================================================================================
def knob_settings(knobs: dict[str, list[Any]], limit: int) -> list[dict[str, Any]]:
    names = list(knobs)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*[knobs[n] for n in names])]
    if len(combos) > limit:
        idx = np.random.default_rng(0).choice(len(combos), size=limit, replace=False)
        combos = [combos[i] for i in sorted(idx)]
    return combos


def suggest_knobs(failure_reason: str, hints: dict[str, list[str]] | None = None) -> list[str]:
    hints = {**KNOB_HINTS, **(hints or {})}
    key = next((k for k in hints if k in failure_reason.lower()), None)
    return hints.get(key, []) if key else []


def _objective(score: dict[str, Any], replay_nrmse: float | None) -> float:
    """Rank knob settings. With a recording, the replay error is direct evidence about the twin's dynamics and
    outweighs closed-loop mode matching (which a wrong physics setting can also hit by accident); without one,
    the failure on its own seed counts most and neighbouring seeds break ties."""
    same = score["primary"]["same_mode_rate"]
    near = (score["neighbours"] or {}).get("same_mode_rate") or 0.0
    if replay_nrmse is not None:
        return same - min(replay_nrmse, 2.0)
    return 0.7 * same + 0.3 * near


def search_knobs(env: Any, policy: Any, case: FailureCase, cfg: dict[str, Any]) -> dict[str, Any]:
    knobs = {k: list(v) for k, v in (cfg.get("knobs") or {}).items()}
    if not knobs:
        hint = ", ".join(suggest_knobs(case.failure_reason, cfg.get("knob_hints"))) or "none known"
        return {"ranked": [], "notes": [f"no knobs configured; suggested for this failure mode: {hint}"]}
    ranked = []
    for setting in knob_settings(knobs, int(cfg["max_knob_settings"])):
        lat = setting.get("latency_mode", cfg.get("latency_mode"))
        params = {k: v for k, v in setting.items() if k != "latency_mode"}
        try:
            runs = closed_loop_runs(env, policy, case, params, int(cfg["knob_repeats"]), int(cfg["knob_seed_jitter"]),
                                    lat, cfg["during_inference"], cfg.get("mode_aliases"))
        except ValueError as exc:  # the env rejected a knob
            return {"ranked": [], "notes": [f"env rejected knob setting {setting}: {exc}"]}
        sc = _score(case, runs, cfg)
        rep = replay_case(env, case, params, cfg["divergence_threshold"]) if (cfg["replay"] and case.has_recording) else None
        r_err = rep["nrmse"] if rep and math.isfinite(rep["nrmse"]) else None
        objective = _objective(sc, r_err)
        ranked.append({"knobs": setting, "same_mode_rate": sc["primary"]["same_mode_rate"],
                       "same_mode_ci": sc["primary"]["same_mode_ci"], "fail_rate": sc["primary"]["fail_rate"],
                       "neighbour_same_mode_rate": (sc["neighbours"] or {}).get("same_mode_rate"),
                       "modes": sc["pooled"]["modes"], "replay_nrmse": r_err, "objective": objective})
    ranked.sort(key=lambda r: -r["objective"])
    notes, sens = [], []
    for name, values in knobs.items():
        best_by = {}
        for r in ranked:
            v = str(r["knobs"].get(name))
            best_by[v] = max(best_by.get(v, -math.inf), r["objective"])
        if len(best_by) > 1:
            sens.append({"knob": name, "spread": max(best_by.values()) - min(best_by.values()), "best_by_value": best_by})
        if ranked and len(values) > 2 and isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
            bv = ranked[0]["knobs"].get(name)
            if bv in (min(values), max(values)) and ranked[0]["same_mode_rate"] >= cfg["min_same_mode_rate"]:
                notes.append(f"`{name}` = {bv} is at the edge of the searched values {values}; widen the range")
    sens.sort(key=lambda s: -s["spread"])
    if ranked:
        top = ranked[0]["objective"]
        ties = [r for r in ranked if r["objective"] >= top - 1e-9]
        if len(ties) > 1:
            loose = sorted({k for r in ties for k in r["knobs"] if r["knobs"][k] != ranked[0]["knobs"].get(k)})
            if loose:
                notes.append(f"{len(ties)} settings tie for best; not pinned down by this failure: {loose}")
    return {"ranked": ranked, "sensitivity": sens, "notes": notes}


# =====================================================================================
# Orchestration
# =====================================================================================
def reproduce(case: FailureCase, cfg: dict[str, Any], env: Any = None, policy: Any = None) -> dict[str, Any]:
    """Run every stage for one failure and return a JSON-friendly result."""
    own = env is None
    if own:
        env, policy = _make(cfg)
    try:
        result: dict[str, Any] = {"case_id": case.case_id, "source": case.source, "domain": case.domain,
                                  "failure_reason": case.failure_reason, "scenario": case.scenario.to_dict(),
                                  "recorded_steps": len(case.states) if case.has_recording else None, "notes": []}
        if case.source == "trace" and not case.meta.get("scenario_known"):
            result["notes"].append("the recording carries no scenario (seed and parameters); the twin used seed = "
                                   "episode index and default parameters, so closed-loop runs are only indicative")
        if cfg["replay"] and case.has_recording:
            result["replay"] = replay_case(env, case, None, cfg["divergence_threshold"])
            if not result["replay"]["synced"]:
                result["notes"].append("env has no set_state(): the replay started from the seed's reset state")
        runs = closed_loop_runs(env, policy, case, None, int(cfg["repeats"]), int(cfg["seed_jitter"]),
                                cfg.get("latency_mode"), cfg["during_inference"], cfg.get("mode_aliases"))
        score = _score(case, runs, cfg)
        result["baseline"] = score
        verdict = _verdict(score, cfg)
        if verdict != "reproduced" and verdict != "flaky":
            search = search_knobs(env, policy, case, cfg)
            result["search"] = search
            best = search["ranked"][0] if search["ranked"] else None
            if best and best["same_mode_rate"] >= cfg["min_same_mode_rate"] and best["fail_rate"] >= cfg["min_fail_rate"]:
                verdict = "reproduced_with_knobs"
                result["twin_ticket"] = {"knobs": best["knobs"], "same_mode_rate": best["same_mode_rate"],
                                         "summary": "twin reproduces this failure only with " +
                                                    ", ".join(f"{k}={v}" for k, v in best["knobs"].items())}
        if score["flaky"]:
            result["notes"].append("identical re-runs of the failing scenario disagree: the twin (or policy) is not "
                                   "deterministic for a fixed seed, so single-run comparisons are unreliable")
        result["verdict"] = verdict
        result["twin_outcome"] = (Counter(score["primary"]["modes"]).most_common(1) or [("unknown", 0)])[0][0]
        if case.fail_step and score["twin_time_to_failure_median"] is not None:
            result["time_to_failure_error_steps"] = score["twin_time_to_failure_median"] - case.fail_step
        return result
    finally:
        if own:
            env.close()


def reproduce_batch(cases: list[FailureCase], cfg: dict[str, Any], progress: Any = None) -> dict[str, Any]:
    env, policy = _make(cfg)
    results = []
    try:
        for i, c in enumerate(cases):
            results.append(reproduce(c, cfg, env, policy))
            if progress:
                progress(i, results[-1])
    finally:
        env.close()
    return summarize(results, cfg)


def summarize(results: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    n = len(results)
    conf = float(cfg.get("confidence", 0.95))
    counts = Counter(r["verdict"] for r in results)
    base = counts["reproduced"]
    with_knobs = base + counts["reproduced_with_knobs"]
    confusion = tm.failure_mode_confusion([(r["baseline"]["mode"], r["twin_outcome"]) for r in results]) if n else {}
    replays = [r["replay"]["nrmse"] for r in results if r.get("replay") and math.isfinite(r["replay"]["nrmse"])]
    tickets = Counter()
    for r in results:
        if r.get("twin_ticket"):
            for k, v in r["twin_ticket"]["knobs"].items():
                tickets[f"{k}={v}"] += 1
    return {
        "cases": n,
        "verdicts": dict(counts),
        "reproduction_rate": base / n if n else math.nan,
        "reproduction_ci": wilson_interval(base, n, conf),
        "reproduction_rate_with_knobs": with_knobs / n if n else math.nan,
        "reproduction_ci_with_knobs": wilson_interval(with_knobs, n, conf),
        "failure_mode_confusion": confusion,
        "replay_nrmse_mean": float(np.mean(replays)) if replays else None,
        "twin_tickets": dict(tickets.most_common()),
        "results": results,
    }


# =====================================================================================
# Report
# =====================================================================================
def _pct(x: float | None) -> str:
    return "-" if x is None or not math.isfinite(x) else f"{100 * x:.0f}%"


def _num(x: float | None) -> str:
    return "-" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _ci(ci: tuple[float, float]) -> str:
    return f"[{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]"


def repro_markdown(summary: dict[str, Any], cfg: dict[str, Any]) -> str:
    lines = [
        "# Failure reproduction in the digital twin",
        "",
        f"- Twin: `{cfg['env']}` {cfg.get('env_args') or ''}; policy `{cfg.get('policy')}`",
        f"- {summary['cases']} failure(s); each re-run {cfg['repeats']}x on its seed and on {cfg['seed_jitter']} "
        f"neighbouring seeds; reproduced = at least {_pct(cfg['min_same_mode_rate'])} of twin runs fail the same way",
        "",
        f"**Failure reproduction rate: {_pct(summary['reproduction_rate'])} {_ci(summary['reproduction_ci'])}** at the "
        f"baseline twin; {_pct(summary['reproduction_rate_with_knobs'])} "
        f"{_ci(summary['reproduction_ci_with_knobs'])} once the knob search is allowed.",
        "",
        "| Verdict | Cases |",
        "|---|---|",
        *[f"| {v} | {summary['verdicts'].get(v, 0)} |" for v in VERDICTS],
    ]
    if summary.get("replay_nrmse_mean") is not None:
        lines += ["", f"Mean open-loop replay error over recorded failures: nRMSE {summary['replay_nrmse_mean']:.3f} "
                      "(0 = the twin moves exactly like the recording under the same commands)."]
    if summary["twin_tickets"]:
        lines += ["", "## Twin tickets", "",
                  "Knob values the twin needed to reproduce failures it otherwise missed. Each is a candidate fix to "
                  "the twin's default configuration; confirm with `groot-eval replay --config` on clean recordings.", "",
                  "| Knob value | Failures it explains |", "|---|---|",
                  *[f"| `{k}` | {c} |" for k, c in summary["twin_tickets"].items()]]
    conf = summary.get("failure_mode_confusion") or {}
    if conf.get("labels"):
        labels = conf["labels"]
        lines += ["", "## Failure-mode confusion (rows: original failure, columns: twin's most common outcome)", "",
                  "| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels),
                  *[f"| {r} | " + " | ".join(str(conf["matrix"][r][t]) for t in labels) + " |" for r in labels
                    if sum(conf["matrix"][r].values())],
                  "", f"Mode agreement {_pct(conf['agreement'])}; JS divergence between the mode distributions "
                      f"{conf['js_divergence']:.2f} (0 = same mix of failure modes)."]
    lines += ["", "## Cases", ""]
    for r in summary["results"]:
        b = r["baseline"]["primary"]
        lines += [f"### `{r['case_id']}`: {r['verdict']}", "",
                  f"- Original: **{r['failure_reason']}** ({r['domain']}, from {r['source']}); scenario seed "
                  f"{r['scenario']['seed']}, params {r['scenario']['params'] or '{}'}",
                  f"- Twin, baseline, same seed: fails {_pct(b['fail_rate'])} {_ci(b['fail_ci'])}, same mode "
                  f"{_pct(b['same_mode_rate'])} {_ci(b['same_mode_ci'])}; outcomes {b['modes']}"]
        nb = r["baseline"].get("neighbours")
        if nb:
            edge = " (reproduced on its seed but not nearby: a knife-edge failure)" if (
                b["same_mode_rate"] >= cfg["min_same_mode_rate"] and nb["same_mode_rate"] < 0.2) else ""
            lines.append(f"- Neighbouring seeds, same parameters: same mode {_pct(nb['same_mode_rate'])}; outcomes "
                         f"{nb['modes']}{edge}")
        if r.get("replay"):
            rp = r["replay"]
            lines.append(f"- Replay of the recorded actions: nRMSE {rp['nrmse']:.3f}, diverges at step "
                         f"{rp['divergence_step'] if rp['divergence_step'] is not None else 'never'} of "
                         f"{rp['steps_recorded']}, DTW {rp['dtw']:.3f}; the twin {('ended as `' + rp['ended_as'] + '`') if rp['ended_as'] else 'did not end within the recorded steps'}"
                         + "; worst signals " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(rp["nrmse_by_signal"].items(), key=lambda kv: -kv[1])[:3]))
        if r.get("time_to_failure_error_steps") is not None:
            lines.append(f"- Twin fails {r['time_to_failure_error_steps']:+.0f} steps relative to the original")
        if r.get("search", {}).get("ranked"):
            top = r["search"]["ranked"][:5]
            lines += ["", "| Knobs | Same mode (seed) | Fail (seed) | Replay nRMSE |", "|---|---|---|---|",
                      *[f"| {t['knobs']} | {_pct(t['same_mode_rate'])} | {_pct(t['fail_rate'])} | "
                        f"{_num(t['replay_nrmse'])} |" for t in top]]
            for s in r["search"].get("sensitivity", [])[:3]:
                lines.append(f"- knob `{s['knob']}` moves the objective by {s['spread']:.2f}")
        for note in r.get("notes", []) + r.get("search", {}).get("notes", []):
            lines.append(f"- note: {note}")
        if r.get("twin_ticket"):
            lines.append(f"- **Ticket:** {r['twin_ticket']['summary']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def to_json(summary: dict[str, Any]) -> str:
    def plain(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    return json.dumps(plain(summary), indent=2)
