"""Failure triage: why did this robot run fail? Plumbing, timing, actuation, sensing, policy or not modelled.

``groot-twin repro`` asks whether the twin can reproduce a failure. Triage asks the question a
team actually has when a real run fails: **whose problem is it?** It runs a fixed sequence of
cheap, targeted tests on the recorded run and the twin, each of which can rule a cause in or out:

====  ===================  ==========================================================================
T1    policy consistency   Re-query the policy on the recorded observations, in order. Same actions
                           as recorded: the deployment ran the policy faithfully. Same actions k
                           steps late: **timing** (inference latency). Actions that are a clean
                           scale, offset, sign flip or reordering of the policy's: **plumbing**
                           (action mapping, units, normalization, joint order). Otherwise the
                           robot ran something else, or the policy is not deterministic.
T2    open-loop replay     Play the recorded actions through the twin from the recorded start state.
                           Tracks the robot: actuation is modelled well. Tracks once the actions are
                           shifted by k steps: **timing** (actuation delay). Residual that grows:
                           **actuation / dynamics**. Residual that is large but does not grow and
                           looks like noise: **sensing** (the robot's sensors are noisier or biased).
T3    closed-loop twin     Re-run the policy on the same scenario in the twin (same seed, repeats,
                           neighbouring seeds). The twin fails the same way: a **policy** failure
                           the twin reproduces. With a timing finding, re-run with that latency to
                           confirm it.
====  ===================  ==========================================================================

The verdict names the most likely cause, the evidence for it, the tests that ruled the others out,
and the next step. ``not_modelled`` means the policy ran faithfully, the twin tracks the robot,
and the twin still does not fail: something in the real scene is missing from the twin (an
object, a contact, a disturbance), so the next step is ``groot-twin repro`` with knobs.

T1 needs a policy that acts on the recorded state (``state.*`` observations); for camera
policies, T1 is reported as not checkable and triage relies on T2 and T3.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from groot_eval import twin_metrics as tm
from groot_eval.repro import FailureCase, _norm_mode, closed_loop_runs, replay_case
from groot_eval.types import Scenario

DEFAULTS: dict[str, Any] = {
    "max_lag": 6,  # steps searched for inference latency (T1) and actuation delay (T2)
    "consistent_nrmse": 0.05,  # T1: re-queried vs recorded actions this close = same policy
    "affine_r2": 0.98,  # T1: per-dimension fit this good with a non-identity map = plumbing
    "replay_good_nrmse": 0.25,  # T2: the twin tracks the robot
    "growth_ratio": 2.0,  # T2: late residual / early residual above this = drift (dynamics)
    "noise_autocorr": 0.5,  # T2: residual lag-1 autocorrelation below this = noise-like (sensing)
    "shift_gain": 0.5,  # T2: a shift must cut the replay error by this fraction to count
    "noise_floor": 0.05,  # T2: residual above this, steady and noise-like, points at sensing
    "repeats": 3,
    "seed_jitter": 2,
    "min_reproduce_rate": 0.5,
}

CAUSES = ("plumbing", "timing", "actuation", "sensing", "policy", "not_modelled", "inconclusive")

NEXT_STEP = {
    "plumbing": "Check the action mapping between policy and robot: joint order, units (rad/deg), normalization "
                "statistics, gripper scaling and sign. `groot-eval check` compares the two contracts.",
    "timing": "Measure the deployed loop with `groot-robot timing-profile` and re-test with `groot-robot timing-check`; "
              "consider asynchronous chunk execution or a faster inference path.",
    "actuation": "Recalibrate the twin's actuators for this robot (`groot-robot calibrate`), or search actuator knobs "
                 "with `groot-twin repro --knob`.",
    "sensing": "Compare sensor noise and bias between robot and twin (`groot-eval gap`); add the measured noise to the "
               "twin and check sensor calibration on the robot.",
    "policy": "A genuine policy failure that the twin reproduces: add the scenario to the regression suite and fix it "
              "in training data or fine-tuning; iterate in the twin.",
    "not_modelled": "The twin tracks the robot but does not fail: something in the real scene is missing from the twin. "
                    "Run `groot-twin repro` with knobs and record the scenario (objects, poses) on the next run.",
    "inconclusive": "Not enough recorded data to tell: record observation.state and actions for the run (TraceRecorder).",
}


# -- T1: policy consistency -------------------------------------------------------------------
def requery_policy(policy: Any, case: FailureCase) -> np.ndarray | None:
    """Actions the policy gives on the recorded observations, queried in order like the deployment did."""
    from groot_eval.replay import _state_dict

    if case.states is None or case.slices is None:
        return None
    if hasattr(policy, "reset"):
        policy.reset(case.scenario.task or "")
    out = []
    for s in case.states:
        obs: dict[str, Any] = _state_dict(np.asarray(s, dtype=float), case.slices)
        obs["annotation.task"] = case.scenario.task or ""
        try:
            out.append(np.asarray(policy.act(obs), dtype=float).reshape(-1))
        except (KeyError, ValueError, TypeError):
            return None
    return np.stack(out)


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return math.nan
    scale = np.std(b, axis=0)
    scale = np.where(scale > 1e-9, scale, np.maximum(np.abs(b).mean(axis=0), 1e-9))
    return float(np.sqrt(np.mean(((a - b) / scale) ** 2)))


def best_lag(pred: np.ndarray, recorded: np.ndarray, max_lag: int) -> tuple[int, float]:
    """The lag k for which recorded[t] ≈ pred[t - k] fits best, and its error."""
    best = (0, _nrmse(pred, recorded))
    for k in range(1, min(max_lag, len(recorded) - 2) + 1):
        e = _nrmse(pred[:-k], recorded[k:])
        if e < best[1] - 1e-12:
            best = (k, e)
    return best


def stale_match(pred: np.ndarray, recorded: np.ndarray, max_lag: int, tol: float) -> dict[str, Any]:
    """For each step, the smallest k with recorded[t] ≈ pred[t - k] (a held or late command), if any.

    Catches inference latency in both forms: a blocking loop that holds the last command while
    waiting (offsets vary between k and 2k) and an asynchronous one (a constant offset).
    """
    scale = np.std(recorded, axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    offsets: list[int | None] = []
    for t in range(len(recorded)):
        hit = None
        for k in range(0, min(max_lag, t) + 1):
            if float(np.sqrt(np.mean(((recorded[t] - pred[t - k]) / scale) ** 2))) <= tol:
                hit = k
                break
        offsets.append(hit)
    # Only steps where the policy's output just changed tell a faithful loop (offset 0) from a late one;
    # while a command is saturated or steady, every offset matches.
    changing = [t for t in range(1, len(pred))
                if float(np.sqrt(np.mean(((pred[t] - pred[t - 1]) / scale) ** 2))) > tol]
    informative = [offsets[t] for t in changing] or offsets
    matched = [k for k in informative if k is not None]
    late = [k for k in matched if k > 0]
    n = len(informative) or 1
    return {"informative_steps": len(informative), "matched_share": len(matched) / n, "late_share": len(late) / n,
            "median_offset": float(np.median(matched)) if matched else math.nan,
            "max_offset": max(matched) if matched else None}


def affine_map(pred: np.ndarray, recorded: np.ndarray) -> dict[str, Any]:
    """Is the recorded action a per-dimension scale/offset, sign flip or reordering of the policy's?"""
    D = min(pred.shape[1], recorded.shape[1])
    order, scales, offsets, r2s = [], [], [], []
    for j in range(D):
        y = recorded[:, j]
        best = (-math.inf, j, 1.0, 0.0)
        for i in range(pred.shape[1]):
            x = pred[:, i]
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                continue
            a, b = np.polyfit(x, y, 1)
            res = y - (a * x + b)
            r2 = 1.0 - float(np.var(res) / np.var(y))
            if r2 > best[0]:
                best = (r2, i, float(a), float(b))
        r2s.append(best[0])
        order.append(best[1])
        scales.append(best[2])
        offsets.append(best[3])
    fit = min(r2s) if r2s else -math.inf
    identity = (order == list(range(D)) and all(abs(s - 1) < 0.05 for s in scales)
                and all(abs(o) < 0.05 * (np.std(recorded) + 1e-9) for o in offsets))
    return {"r2_min": fit, "order": order, "scale": scales, "offset": offsets, "identity": identity}


def t1_consistency(policy: Any, case: FailureCase, cfg: dict[str, Any]) -> dict[str, Any]:
    if policy is None or case.actions is None:
        return {"checkable": False, "why": "no policy or no recorded actions"}
    pred = requery_policy(policy, case)
    if pred is None:
        return {"checkable": False, "why": "the policy cannot act on the recorded state alone (camera inputs?)"}
    rec = np.asarray(case.actions, dtype=float)
    T = min(len(pred), len(rec))
    pred, rec = pred[:T], rec[:T]
    again = requery_policy(policy, case)
    deterministic = again is not None and _nrmse(again[:T], pred) < 1e-6
    e0 = _nrmse(pred, rec)
    lag, e_lag = best_lag(pred, rec, int(cfg["max_lag"]))
    stale = stale_match(pred, rec, 2 * int(cfg["max_lag"]), float(cfg["consistent_nrmse"]))
    aff = affine_map(pred, rec) if pred.shape[1] and rec.shape[1] else {"r2_min": -math.inf, "identity": True}
    out: dict[str, Any] = {"checkable": True, "nrmse": e0, "deterministic": deterministic, "best_lag": lag, "best_lag_nrmse": e_lag,
           "affine": aff, "stale": stale}
    tol = float(cfg["consistent_nrmse"])
    if e0 <= tol:
        out["finding"] = "consistent"
    elif lag > 0 and e_lag <= tol:
        out["finding"] = "lagged"
    elif stale["informative_steps"] >= 5 and stale["matched_share"] >= 0.8 and stale["late_share"] >= 0.5:
        out["finding"] = "lagged"
        out["best_lag"] = max(1, int(round(stale["median_offset"])))
    elif aff["r2_min"] >= float(cfg["affine_r2"]) and not aff["identity"]:
        out["finding"] = "mapped"
    elif not deterministic:
        out["finding"] = "nondeterministic"
    else:
        out["finding"] = "different"
    return out


# -- T2: open-loop replay ---------------------------------------------------------------------
def _shifted(case: FailureCase, k: int) -> FailureCase:
    acts = np.asarray(case.actions, dtype=float)
    shifted = np.vstack([np.repeat(acts[:1] * 0.0, k, axis=0), acts[:-k]]) if k else acts
    return FailureCase(case.case_id, case.scenario, case.failure_reason, case.source, case.domain,
                       case.states, shifted, case.slices, case.fail_step, case.latencies_ms, case.meta)


def _residual_shape(env: Any, case: FailureCase) -> dict[str, float]:
    """Growth and noise-likeness of the replay residual (late vs early error; lag-1 autocorrelation)."""
    from groot_eval.replay import _flatten, _state_dict

    states, acts, slices = np.asarray(case.states, float), np.asarray(case.actions, float), case.slices or {}
    env.reset(case.scenario)
    if hasattr(env, "max_steps") and env.max_steps < len(states) + 1:
        env.max_steps = len(states) + 1
    try:
        obs = env.set_state(_state_dict(states[0], slices), acts[:0])
    except (AttributeError, NotImplementedError):
        return {"growth": math.nan, "autocorr": math.nan}
    pred = [_flatten(obs, slices)]
    for t in range(len(states) - 1):
        obs, _, term, trunc, _ = env.step(acts[t])
        pred.append(_flatten(obs, slices))
        if term or trunc:
            break
    P = np.stack(pred)
    R = states[: len(P)] - P
    moving = np.std(states, axis=0) > 1e-9
    if len(R) < 8 or not moving.any():
        return {"growth": math.nan, "autocorr": math.nan}
    R = R[:, moving] / np.std(states[:, moving], axis=0)
    err = np.sqrt(np.mean(R ** 2, axis=1))
    q = max(2, len(err) // 4)
    early, late = float(np.mean(err[1:q + 1])), float(np.mean(err[-q:]))
    flat = R - R.mean(axis=0)
    num = float(np.sum(flat[1:] * flat[:-1]))
    den = float(np.sum(flat * flat)) or 1e-12
    return {"growth": late / max(early, 1e-9), "autocorr": num / den, "early": early, "late": late}


def t2_replay(env: Any, case: FailureCase, cfg: dict[str, Any]) -> dict[str, Any]:
    if not case.has_recording:
        return {"checkable": False, "why": "no recorded states and actions"}
    base = replay_case(env, case)
    out: dict[str, Any] = {"checkable": True, "nrmse": base["nrmse"], "divergence_step": base["divergence_step"],
                           "synced": base["synced"], "nrmse_by_signal": base["nrmse_by_signal"]}
    best_k, best_e = 0, base["nrmse"]
    for k in range(1, int(cfg["max_lag"]) + 1):
        e = replay_case(env, _shifted(case, k))["nrmse"]
        if math.isfinite(e) and e < best_e:
            best_k, best_e = k, e
    out["best_shift"], out["best_shift_nrmse"] = best_k, best_e
    out.update(_residual_shape(env, case))
    good = float(cfg["replay_good_nrmse"])
    g, ac = out.get("growth", math.nan), out.get("autocorr", math.nan)
    out["noise_like"] = bool(math.isfinite(base["nrmse"]) and base["nrmse"] > float(cfg["noise_floor"])
                             and math.isfinite(g) and g < 1.5 and math.isfinite(ac) and ac < float(cfg["noise_autocorr"]))
    if math.isfinite(base["nrmse"]) and base["nrmse"] <= good:
        out["finding"] = "tracks"
    elif best_k and best_e <= good and best_e <= (1 - float(cfg["shift_gain"])) * base["nrmse"]:
        out["finding"] = "delayed"
    elif math.isfinite(out.get("growth", math.nan)) and out["growth"] >= float(cfg["growth_ratio"]):
        out["finding"] = "drifts"
    elif math.isfinite(out.get("autocorr", math.nan)) and out["autocorr"] < float(cfg["noise_autocorr"]):
        out["finding"] = "noisy"
    else:
        out["finding"] = "drifts"
    return out


# -- T3: closed-loop in the twin --------------------------------------------------------------
def t3_closed_loop(env: Any, policy: Any, case: FailureCase, cfg: dict[str, Any],
                   latency_mode: str | None = None) -> dict[str, Any]:
    if policy is None:
        return {"checkable": False, "why": "no policy"}
    aliases = cfg.get("mode_aliases") or {}
    runs = closed_loop_runs(env, policy, case, None, int(cfg["repeats"]), int(cfg["seed_jitter"]),
                            latency_mode=latency_mode, aliases=aliases)
    mode = _norm_mode(case.failure_reason, aliases)
    same = [(ok, m) for ok, m, _ in runs["same_seed"]]
    s = tm.reproduction_score(mode, same, 0.95)
    out = {"checkable": True, "mode": mode, "latency_mode": latency_mode, "same_seed": s,
           "neighbours_failed": sum(not ok for ok, _, _ in runs["neighbours"]), "neighbours": len(runs["neighbours"])}
    out["finding"] = "reproduced" if s["fail_rate"] >= float(cfg["min_reproduce_rate"]) else "not_reproduced"
    return out


# -- verdict ----------------------------------------------------------------------------------
def triage_case(case: FailureCase, env: Any, policy: Any = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run T1-T3 on one failed run and name the most likely cause."""
    cfg = {**DEFAULTS, **(cfg or {})}
    t1 = t1_consistency(policy, case, cfg)
    t2 = t2_replay(env, case, cfg)
    t3 = t3_closed_loop(env, policy, case, cfg)
    evidence: list[str] = []
    ruled_out: list[str] = []
    cause, confidence = "inconclusive", "low"
    f1, f2, f3 = t1.get("finding"), t2.get("finding"), t3.get("finding")
    dt = float(getattr(env, "dt", 0.0) or 0.0)

    if f1 == "mapped":
        cause, confidence = "plumbing", "high"
        a = t1["affine"]
        evidence.append(f"recorded actions are a remapped copy of the policy's (fit R² ≥ {a['r2_min']:.3f}; order "
                        f"{a['order']}, scale {[round(x, 3) for x in a['scale']]})")
    elif f1 == "lagged":
        k = t1["best_lag"]
        cause, confidence = "timing", "medium"
        st = t1["stale"]
        evidence.append(f"the robot executed the policy's actions late: {st['matched_share']:.0%} of recorded actions "
                        f"match the policy's output from up to {st['max_offset']} step(s) earlier (median {k}): "
                        f"inference latency ≈ {k * dt * 1000:.0f} ms")
        if dt > 0:
            t3_lat = t3_closed_loop(env, policy, case, cfg, latency_mode=f"fixed:{k * dt * 1000:.1f}")
            t3["with_latency"] = t3_lat
            if t3_lat.get("finding") == "reproduced" and f3 != "reproduced":
                confidence = "high"
                evidence.append("the twin fails the same way once that latency is added, and not without it")
    elif f2 == "delayed":
        k = t2["best_shift"]
        cause, confidence = "timing", "high" if f1 in ("consistent", None) else "medium"
        evidence.append(f"the twin tracks the robot once actions are delayed {k} step(s) (replay error "
                        f"{t2['best_shift_nrmse']:.3f} vs {t2['nrmse']:.3f}): actuation delay ≈ {k * dt * 1000:.0f} ms")
    elif f2 == "drifts":
        cause, confidence = "actuation", "medium" if f1 == "consistent" else "low"
        evidence.append(f"replaying the robot's actions, the twin drifts away (error {t2['nrmse']:.3f}, residual grows "
                        f"{t2.get('growth', math.nan):.1f}x; diverges at step {t2['divergence_step']})")
    elif f3 == "reproduced":
        cause = "policy"
        confidence = "high" if f1 == "consistent" and f2 == "tracks" else "medium"
        s = t3["same_seed"]
        evidence.append(f"the twin fails the same way on the same scenario ({s['same_mode_rate']:.0%} same mode, "
                        f"{s['fail_rate']:.0%} fail)")
    elif t2.get("noise_like") and f3 == "not_reproduced":
        cause, confidence = "sensing", "medium" if f1 == "consistent" else "low"
        evidence.append(f"replay error {t2['nrmse']:.3f} does not grow ({t2.get('growth', math.nan):.1f}x) and is "
                        f"noise-like (lag-1 autocorrelation {t2.get('autocorr', math.nan):.2f}): the robot's sensors "
                        "report noise or bias the twin does not have, and the policy succeeds on the twin's clean inputs")
    elif f2 == "tracks" and f3 == "not_reproduced":
        cause, confidence = "not_modelled", "medium" if f1 == "consistent" else "low"
        evidence.append("the twin tracks the robot's motion and the policy succeeds in the twin: the real scene has "
                        "something the twin does not")
    if f1 == "nondeterministic":
        evidence.append("the policy gives different actions on the same inputs: fix its seed before trusting T1")
    if f1 == "different" and cause in ("inconclusive", "not_modelled"):
        cause, confidence = "plumbing", "low"
        evidence.append(f"the robot's recorded actions differ from the policy's on the same inputs (error {t1['nrmse']:.3f}) "
                        "with no clean mapping: check the checkpoint, server and pre/post-processing actually deployed")

    if f1 == "consistent":
        ruled_out.append("plumbing (the policy's actions reached the robot unchanged)")
    if f1 == "consistent" and f2 not in ("delayed",):
        ruled_out.append("timing (no latency or actuation delay found)")
    if f2 == "tracks" and not t2.get("noise_like"):
        ruled_out.append("actuation and sensing (the twin follows the robot under the same commands)")
    if f3 == "not_reproduced" and cause != "policy":
        ruled_out.append("a policy failure the twin reproduces")
    return {"case_id": case.case_id, "failure_reason": case.failure_reason, "cause": cause, "confidence": confidence,
            "evidence": evidence, "ruled_out": ruled_out, "next_step": NEXT_STEP[cause],
            "tests": {"t1_consistency": t1, "t2_replay": t2, "t3_closed_loop": t3}}


def triage_batch(cases: list[FailureCase], env: Any, policy: Any = None, cfg: dict[str, Any] | None = None,
                 progress: Any = None) -> dict[str, Any]:
    results = []
    for i, c in enumerate(cases):
        r = triage_case(c, env, policy, cfg)
        results.append(r)
        if progress:
            progress(i, r)
    counts = {k: sum(r["cause"] == k for r in results) for k in CAUSES}
    return {"cases": len(results), "counts": {k: v for k, v in counts.items() if v}, "results": results}


def triage_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Failure triage", "", f"{summary['cases']} failed run(s): "
             + ", ".join(f"{k} {v}" for k, v in summary["counts"].items()) + ".", "",
             "| Run | Failure | Likely cause | Confidence | Key evidence |", "|---|---|---|---|---|"]
    for r in summary["results"]:
        lines.append(f"| {r['case_id']} | {r['failure_reason']} | **{r['cause']}** | {r['confidence']} | "
                     f"{(r['evidence'] or ['-'])[0]} |")
    for r in summary["results"]:
        lines += ["", f"## {r['case_id']}: {r['cause']} ({r['confidence']})", ""]
        lines += [f"- {e}" for e in r["evidence"]] or ["- no test ruled a cause in"]
        if r["ruled_out"]:
            lines += ["", "Ruled out: " + "; ".join(r["ruled_out"]) + "."]
        lines += ["", f"Next step: {r['next_step']}"]
    return "\n".join(lines) + "\n"


def to_json(summary: dict[str, Any]) -> str:
    def plain(x: Any) -> Any:
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Scenario):
            return x.to_dict()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return str(x)

    return json.dumps(summary, indent=2, default=plain)
