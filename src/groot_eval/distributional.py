"""Time-to-success distributions, safety metrics and paired comparisons of two runs.

A success rate at a fixed timeout throws away most of what a run measured. Two policies with the
same 80% can differ by 2x in how long they take, and a policy can "succeed" while slamming into
joint limits. This module scores runs as distributions:

* **Time to success** with failures treated as *censored* (the robot had not succeeded by the
  time the episode ended), so a timeout is not counted as a fast run or dropped. Kaplan-Meier
  curve, median time to success, and **RMST** (restricted mean time to success up to a horizon:
  the average time spent before succeeding, where a failure costs the whole horizon). Bootstrap
  intervals throughout.
* **Safety** per episode from traces: action jerk, the largest command jump between steps (chunk
  boundaries show up here), time near joint limits, actuator saturation, plus any ``metric.*``
  column you name (contact force, minimum clearance). Each is reported as a distribution and as
  the share of episodes over a threshold, with a Wilson interval.
* **Paired comparison** of two runs on the same scenarios (A vs B: lockstep vs measured timing,
  two checkpoints, new vs worn robot): McNemar on success, the RMST difference with a paired
  bootstrap interval, and the Kolmogorov-Smirnov distance between time-to-success distributions.

Input runs are directories written by ``groot-eval sweep`` (``episodes.jsonl``) or any JSONL
with ``scenario_id``, ``success`` and ``steps`` per episode. Safety metrics read traces
(``--trace``), the LeRobot v2 datasets the harness writes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from groot_eval.stats import mcnemar_exact, wilson_interval


def load_episodes(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    f = p / "episodes.jsonl" if p.is_dir() else p
    if not f.exists():
        raise FileNotFoundError(f"no episodes.jsonl in {p}")
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def _times(episodes: list[dict[str, Any]], dt: float) -> tuple[np.ndarray, np.ndarray]:
    """(time, event) per episode: event=1 at success, 0 when censored (failed or timed out)."""
    t = np.array([float(e["steps"]) * dt for e in episodes], dtype=float)
    ev = np.array([1 if e.get("success") in (True, 1, "true", "True") else 0 for e in episodes], dtype=int)
    return t, ev


# -- survival ---------------------------------------------------------------------------------
def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of P(not yet succeeded by t). Returns (event times, survival after each)."""
    order = np.argsort(times, kind="stable")
    t, e = times[order], events[order]
    uniq = np.unique(t[e == 1])
    surv, s = [], 1.0
    for u in uniq:
        at_risk = int(np.sum(t >= u))
        d = int(np.sum((t == u) & (e == 1)))
        if at_risk:
            s *= 1.0 - d / at_risk
        surv.append(s)
    return uniq, np.array(surv, dtype=float)


def rmst(times: np.ndarray, events: np.ndarray, horizon: float) -> float:
    """Restricted mean time to success up to ``horizon`` (area under the KM curve)."""
    if times.size == 0:
        return math.nan
    ut, s = kaplan_meier(times, events)
    area, prev_t, prev_s = 0.0, 0.0, 1.0
    for u, v in zip(ut, s):
        if u >= horizon:
            break
        area += prev_s * (u - prev_t)
        prev_t, prev_s = u, v
    return float(area + prev_s * (horizon - prev_t))


def median_time(times: np.ndarray, events: np.ndarray) -> float:
    ut, s = kaplan_meier(times, events)
    below = np.nonzero(s <= 0.5)[0]
    return float(ut[below[0]]) if below.size else math.inf


def _boot(fn: Any, n: int, resamples: int, seed: int, confidence: float) -> tuple[float, float]:
    if n < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(resamples)]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return (math.nan, math.nan)
    a = (1 - confidence) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def time_to_success(episodes: list[dict[str, Any]], dt: float, horizon: float | None = None,
                    confidence: float = 0.95, resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    t, e = _times(episodes, dt)
    H = float(horizon if horizon is not None else (t.max() if t.size else 0.0))
    succ = int(e.sum())
    out: dict[str, Any] = {
        "episodes": int(t.size), "successes": succ, "success_rate": succ / t.size if t.size else math.nan,
        "success_ci": wilson_interval(succ, int(t.size), confidence), "horizon_s": H,
        "rmst_s": rmst(t, e, H), "rmst_ci": _boot(lambda i: rmst(t[i], e[i], H), t.size, resamples, seed, confidence),
        "median_s": median_time(t, e),
    }
    for q in (0.25, 0.75, 0.9):
        ok = np.sort(t[e == 1])
        out[f"p{int(q * 100)}_success_s"] = float(np.quantile(ok, q)) if ok.size else math.nan
    ut, s = kaplan_meier(t, e)
    out["km"] = {"t": ut.tolist(), "not_yet": s.tolist()}
    return out


# -- safety -----------------------------------------------------------------------------------
DEFAULT_SAFETY: dict[str, float] = {
    "max_jerk": 50.0,  # rms 3rd difference of the commanded action, per s^3 (action units)
    "max_jump": 0.5,  # largest change of any command dimension between two steps (action units)
    "limit_margin": 0.05,  # within this fraction of a state limit counts as near the limit
    "max_near_limit_share": 0.05,
    "max_saturation_share": 0.10,
}


def episode_safety(actions: np.ndarray, dt: float, states: np.ndarray | None = None,
                   limits: tuple[np.ndarray, np.ndarray] | None = None, action_range: tuple[float, float] | None = None,
                   margin: float = 0.05) -> dict[str, float]:
    a = np.asarray(actions, dtype=float)
    out: dict[str, float] = {"steps": float(len(a))}
    if len(a) >= 4:
        j = np.diff(a, n=3, axis=0) / dt ** 3
        out["jerk_rms"] = float(np.sqrt(np.mean(j ** 2)))
    if len(a) >= 2:
        out["max_jump"] = float(np.max(np.abs(np.diff(a, axis=0))))
    if action_range is not None and a.size:
        lo, hi = action_range
        span = hi - lo
        out["saturation_share"] = float(np.mean((a <= lo + 1e-3 * span) | (a >= hi - 1e-3 * span)))
    if states is not None and limits is not None and len(states):
        s = np.asarray(states, dtype=float)
        lower, upper = (np.asarray(x, dtype=float) for x in limits)
        lim_span = np.where(upper - lower > 1e-12, upper - lower, np.inf)
        near = (s - lower < margin * lim_span) | (upper - s < margin * lim_span)
        out["near_limit_share"] = float(np.mean(np.any(near, axis=1)))
    return out


def safety_from_trace(trace_root: str | Path, dt: float | None = None, limits: tuple[list[float], list[float]] | None = None,
                      action_range: tuple[float, float] | None = None, metrics: list[str] | None = None,
                      margin: float = 0.05) -> list[dict[str, float]]:
    from groot_eval.trace import TraceSet

    ts = TraceSet.open(trace_root)
    step = float(dt or (1.0 / ts.fps if ts.fps else 0.0)) or 1.0
    acts, states = ts.per_episode("action"), ts.per_episode("observation.state")
    lim = (np.asarray(limits[0]), np.asarray(limits[1])) if limits else None
    rows = []
    for i, a in enumerate(acts):
        row = episode_safety(a, step, states[i] if i < len(states) else None, lim, action_range, margin)
        frame = ts.frames()[i]
        for m in metrics or []:
            col = m if m in frame else f"metric.{m}"
            if col in frame:
                v = frame[col].to_numpy(dtype=float)
                row[m] = float(np.nanmax(np.abs(v))) if v.size else math.nan
        rows.append(row)
    return rows


def safety_summary(rows: list[dict[str, float]], thresholds: dict[str, float] | None = None,
                   confidence: float = 0.95) -> dict[str, Any]:
    th = {**DEFAULT_SAFETY, **(thresholds or {})}
    checks = {"jerk_rms": th["max_jerk"], "max_jump": th["max_jump"],
              "near_limit_share": th["max_near_limit_share"], "saturation_share": th["max_saturation_share"]}
    for k, v in th.items():
        if k.startswith("max_") and k[4:] not in ("jerk", "jump", "near_limit_share", "saturation_share"):
            checks[k[4:]] = v
    out: dict[str, Any] = {"episodes": len(rows), "thresholds": th, "metrics": {}}
    for key, limit in checks.items():
        vals = np.array([r[key] for r in rows if key in r and math.isfinite(r[key])], dtype=float)
        if not vals.size:
            continue
        over = int(np.sum(vals > limit))
        out["metrics"][key] = {"median": float(np.median(vals)), "p95": float(np.quantile(vals, 0.95)),
                               "max": float(vals.max()), "limit": limit, "over": over,
                               "over_share": over / vals.size, "over_ci": wilson_interval(over, int(vals.size), confidence)}
    return out


# -- paired comparison ------------------------------------------------------------------------
def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return math.nan
    grid = np.union1d(a, b)
    fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def compare(a: list[dict[str, Any]], b: list[dict[str, Any]], dt: float, horizon: float | None = None,
            confidence: float = 0.95, resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """A vs B on the scenarios both ran (joined on ``scenario_id``, or by position when ids are missing)."""
    ida = {e.get("scenario_id"): e for e in a if e.get("scenario_id")}
    idb = {e.get("scenario_id"): e for e in b if e.get("scenario_id")}
    common = [k for k in ida if k in idb]
    if common:
        pa, pb = [ida[k] for k in common], [idb[k] for k in common]
        joined_by = "scenario_id"
    else:
        n = min(len(a), len(b))
        pa, pb, joined_by = a[:n], b[:n], "position"
    ta, ea = _times(pa, dt)
    tb, eb = _times(pb, dt)
    H = float(horizon if horizon is not None else max(ta.max(initial=0), tb.max(initial=0)))
    only_a, only_b = int(np.sum((ea == 1) & (eb == 0))), int(np.sum((ea == 0) & (eb == 1)))
    diff = rmst(tb, eb, H) - rmst(ta, ea, H)
    ci = _boot(lambda i: rmst(tb[i], eb[i], H) - rmst(ta[i], ea[i], H), ta.size, resamples, seed, confidence)
    out = {
        "pairs": int(ta.size), "joined_by": joined_by, "horizon_s": H,
        "success_a": float(ea.mean()) if ea.size else math.nan, "success_b": float(eb.mean()) if eb.size else math.nan,
        "only_a_succeeded": only_a, "only_b_succeeded": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b),
        "rmst_a_s": rmst(ta, ea, H), "rmst_b_s": rmst(tb, eb, H), "rmst_diff_s": diff, "rmst_diff_ci": ci,
        "ks_time_to_success": ks_distance(ta[ea == 1], tb[eb == 1]),
    }
    out["verdict"] = verdict(out, confidence)
    return out


def verdict(c: dict[str, Any], confidence: float = 0.95) -> str:
    alpha = 1 - confidence
    lo, hi = c["rmst_diff_ci"]
    slower = math.isfinite(lo) and lo > 0
    faster = math.isfinite(hi) and hi < 0
    worse_success = c["mcnemar_p"] < alpha and c["only_a_succeeded"] > c["only_b_succeeded"]
    better_success = c["mcnemar_p"] < alpha and c["only_b_succeeded"] > c["only_a_succeeded"]
    if worse_success or (slower and not better_success):
        return "B worse"
    if better_success or (faster and not worse_success):
        return "B better"
    return "no significant difference"


# -- report -----------------------------------------------------------------------------------
def _f(x: Any, fmt: str = ".2f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "not reached"
    return format(x, fmt)


def dist_markdown(name_a: str, tts_a: dict[str, Any], tts_b: dict[str, Any] | None = None, name_b: str = "",
                  cmp: dict[str, Any] | None = None, safety: dict[str, dict[str, Any]] | None = None) -> str:
    lines = ["# Time to success and safety", "", "| Run | Episodes | Success | Median time to success (s) | "
             "p90 (s) | RMST (s) |", "|---|---|---|---|---|---|"]
    for name, t in ((name_a, tts_a), (name_b, tts_b)):
        if t:
            lines.append(f"| {name} | {t['episodes']} | {t['success_rate']:.0%} [{t['success_ci'][0]:.0%}, "
                         f"{t['success_ci'][1]:.0%}] | {_f(t['median_s'])} | {_f(t['p90_success_s'])} | "
                         f"{_f(t['rmst_s'])} [{_f(t['rmst_ci'][0])}, {_f(t['rmst_ci'][1])}] |")
    lines += ["", f"RMST is the mean time spent before succeeding, up to {tts_a['horizon_s']:.1f} s; a failure costs the "
              "whole horizon. Lower is better.", ""]
    if cmp:
        lines += ["## Paired comparison", "",
                  f"{cmp['pairs']} scenarios run by both (joined by {cmp['joined_by']}). **{cmp['verdict']}**.", "",
                  f"- Success: {cmp['success_a']:.0%} vs {cmp['success_b']:.0%}; only {name_a} succeeded on "
                  f"{cmp['only_a_succeeded']}, only {name_b} on {cmp['only_b_succeeded']} (McNemar p = {cmp['mcnemar_p']:.3g}).",
                  f"- RMST: {_f(cmp['rmst_a_s'])} s vs {_f(cmp['rmst_b_s'])} s; difference {_f(cmp['rmst_diff_s'], '+.2f')} s "
                  f"[{_f(cmp['rmst_diff_ci'][0], '+.2f')}, {_f(cmp['rmst_diff_ci'][1], '+.2f')}].",
                  f"- Time-to-success distributions differ by KS {_f(cmp['ks_time_to_success'])}.", ""]
    for name, s in (safety or {}).items():
        lines += [f"## Safety: {name}", "", "| Metric | Median | p95 | Max | Limit | Episodes over |",
                  "|---|---|---|---|---|---|"]
        for k, m in s["metrics"].items():
            lines.append(f"| {k} | {m['median']:.3g} | {m['p95']:.3g} | {m['max']:.3g} | {m['limit']:.3g} | "
                         f"{m['over']} ({m['over_share']:.0%}, CI up to {m['over_ci'][1]:.0%}) |")
        lines.append("")
    return "\n".join(lines)
