"""Digital-twin usefulness metrics: does the twin predict reality, where is it wrong, is it worth it?

A twin is judged by whether it leads to the same decisions as the real robot,
not by how real it looks. The metrics come in three tiers:

Tier 1, predictive validity (needs paired sim and real results)
    :func:`pearson`, :func:`kendall_tau`, :func:`mmrv` (mean maximum rank
    violation, from SIMPLER), :func:`calibration` (bias, slope, offset),
    :func:`sensitivity_alignment` (does a perturbation hurt as much in sim as
    on the robot?), :func:`decision_agreement` (would a sim-based gate make
    the same go/no-go calls?). :func:`predictive_validity` runs them all on a
    table of results.

Tier 2, component fidelity (where is it wrong?)
    Trajectories (:func:`nrmse`, :func:`dtw_distance`, :func:`divergence_step`),
    actuators (:func:`actuator_tracking`, :func:`saturation_fraction`), timing
    (:func:`latency_match`), failures (:func:`failure_mode_confusion`,
    :func:`reproduction_score`), and vision (:func:`psnr`, :func:`ssim`,
    :func:`channel_order_check`, :func:`best_frame_offset`,
    :func:`sharpness_ratio`, :func:`policy_visual_gap`).

Tier 3, value (is it worth it?)
    :func:`control_variate_savings` (how many real rollouts the twin saves at
    the same confidence), :func:`coverage_vs_field` (does the twin's scenario
    distribution cover what the fleet sees?).

Everything is plain NumPy so it runs anywhere the harness runs. Definitions
and thresholds are documented in ``docs/twin-failure-repro.md``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from groot_eval.stats import bootstrap_ci, spearman, wilson_interval

_EPS = 1e-12


# =====================================================================================
# Tier 1: predictive validity
# =====================================================================================
def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; nan when either side is constant or n < 2."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.size != b.size:
        raise ValueError("x and y must have equal length")
    if a.size < 2 or a.std() < _EPS or b.std() < _EPS:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b: share of policy pairs ranked the same way, ties handled."""
    a, b = np.asarray(x, float), np.asarray(y, float)
    n = a.size
    if n != b.size:
        raise ValueError("x and y must have equal length")
    if n < 2:
        return math.nan
    conc = disc = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                tie_a += 1
            elif db == 0:
                tie_b += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tie_a) * (conc + disc + tie_b))
    return (conc - disc) / denom if denom else math.nan


def mmrv(sim: Sequence[float], real: Sequence[float]) -> float:
    """Mean maximum rank violation (Li et al., SIMPLER, 2024).

    For each policy i, find the worst pair (i, j) that sim orders the wrong way
    round, weighted by how far apart the two are on the robot:
    ``max_j |R_i - R_j| * 1[(S_i - S_j)(R_i - R_j) < 0]``, then average over i.
    0 means sim never ranks two policies the wrong way round. Lower is better;
    it is in success-rate units, so 0.05 means the typical worst mis-ranking is
    between policies 5 points apart on the robot.
    """
    s, r = np.asarray(sim, float), np.asarray(real, float)
    if s.size != r.size:
        raise ValueError("sim and real must have equal length")
    if s.size < 2:
        return math.nan
    worst = []
    for i in range(s.size):
        v = [abs(r[i] - r[j]) for j in range(s.size) if j != i and (s[i] - s[j]) * (r[i] - r[j]) < 0]
        worst.append(max(v) if v else 0.0)
    return float(np.mean(worst))


def calibration(sim: Sequence[float], real: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """How sim scores map onto real ones: bias, least-squares slope and offset, MAE.

    A twin that is consistently 15 points optimistic is usable once the bias is
    known; a slope far from 1 means the twin compresses or stretches
    differences between policies.
    """
    s, r = np.asarray(sim, float), np.asarray(real, float)
    diff = s - r
    out: dict[str, float] = {
        "bias": float(diff.mean()) if diff.size else math.nan,
        "mae": float(np.abs(diff).mean()) if diff.size else math.nan,
        "slope": math.nan,
        "offset": math.nan,
    }
    lo, hi = bootstrap_ci(diff, confidence) if diff.size >= 2 else (math.nan, math.nan)
    out["bias_ci_low"], out["bias_ci_high"] = lo, hi
    if s.size >= 2 and s.std() > _EPS:
        slope, offset = np.polyfit(s, r, 1)  # real ≈ slope * sim + offset
        out["slope"], out["offset"] = float(slope), float(offset)
    return out


def _rates(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Aggregate rows into (policy, condition, domain) -> (successes, trials).

    A row is ``{policy, domain, condition?, successes, trials}`` or
    ``{policy, domain, condition?, success}`` (one episode per row).
    """
    agg: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        key = (str(row["policy"]), str(row.get("condition") or "nominal"), str(row["domain"]))
        s, n = agg.setdefault(key, [0, 0])
        if "trials" in row:
            agg[key] = [s + int(row["successes"]), n + int(row["trials"])]
        else:
            ok = row["success"]
            ok = ok if isinstance(ok, bool) else str(ok).strip().lower() in ("1", "true", "yes")
            agg[key] = [s + int(ok), n + 1]
    return {k: (v[0], v[1]) for k, v in agg.items()}


def sensitivity_alignment(rows: list[dict[str, Any]], nominal: str = "nominal") -> dict[str, Any]:
    """Does each perturbation change success by the same amount in sim as on the robot?

    For every policy and perturbation, delta = rate(perturbation) - rate(nominal),
    in each domain. Reports the correlation of sim deltas against real deltas,
    the share with the same sign, the mean absolute delta error, and the
    perturbations the twin gets most wrong. A twin can match nominal success
    and still miss that lighting changes hurt the real camera policy.
    """
    rates = _rates(rows)
    policies = sorted({k[0] for k in rates})
    conds = sorted({k[1] for k in rates} - {nominal})
    sim_d: list[float] = []
    real_d: list[float] = []
    per: list[dict[str, Any]] = []
    for p in policies:
        base_s, base_r = rates.get((p, nominal, "sim")), rates.get((p, nominal, "real"))
        if not base_s or not base_r or not base_s[1] or not base_r[1]:
            continue
        for c in conds:
            cs, cr = rates.get((p, c, "sim")), rates.get((p, c, "real"))
            if not cs or not cr or not cs[1] or not cr[1]:
                continue
            ds = cs[0] / cs[1] - base_s[0] / base_s[1]
            dr = cr[0] / cr[1] - base_r[0] / base_r[1]
            sim_d.append(ds)
            real_d.append(dr)
            per.append({"policy": p, "condition": c, "sim_delta": ds, "real_delta": dr, "error": ds - dr})
    if not per:
        return {"pairs": 0, "correlation": math.nan, "sign_agreement": math.nan, "mean_abs_error": math.nan, "worst": []}
    signs = [np.sign(a) == np.sign(b) or (abs(a) < 0.02 and abs(b) < 0.02) for a, b in zip(sim_d, real_d)]
    by_cond: dict[str, list[float]] = {}
    for e in per:
        by_cond.setdefault(e["condition"], []).append(abs(e["error"]))
    return {
        "pairs": len(per),
        "correlation": pearson(sim_d, real_d),
        "sign_agreement": float(np.mean(signs)),
        "mean_abs_error": float(np.mean(np.abs(np.subtract(sim_d, real_d)))),
        "by_condition": {c: float(np.mean(v)) for c, v in sorted(by_cond.items(), key=lambda kv: -np.mean(kv[1]))},
        "worst": sorted(per, key=lambda e: -abs(e["error"]))[:5],
    }


def decision_agreement(sim_flags: Sequence[bool], real_flags: Sequence[bool]) -> dict[str, float]:
    """Would gating on the twin make the same calls as gating on the robot?

    ``*_flags[i]`` is True when decision i is "block" (regression found, gate
    failed) in that domain. Recall is the share of real regressions the twin
    catches; false-alarm rate is the share of real passes the twin blocks.
    """
    s, r = np.asarray(sim_flags, bool), np.asarray(real_flags, bool)
    if s.size != r.size:
        raise ValueError("sim_flags and real_flags must have equal length")
    tp, fp = int(np.sum(s & r)), int(np.sum(s & ~r))
    fn, tn = int(np.sum(~s & r)), int(np.sum(~s & ~r))

    def div(a: int, b: int) -> float:
        return a / b if b else math.nan

    return {"decisions": int(s.size), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": div(tp, tp + fn), "precision": div(tp, tp + fp),
            "false_alarm_rate": div(fp, fp + tn), "agreement": div(tp + tn, int(s.size))}


def predictive_validity(rows: list[dict[str, Any]], nominal: str = "nominal", gate: float | None = None,
                        confidence: float = 0.95) -> dict[str, Any]:
    """Tier-1 summary from a table of sim and real results.

    ``rows`` as in :func:`sensitivity_alignment`. Rates are compared per policy
    at the nominal condition (or pooled over conditions when there is no
    nominal row). With ``gate`` (a success-rate bar), also reports whether the
    twin's pass/fail call per policy matches the robot's.
    """
    rates = _rates(rows)
    policies = sorted({k[0] for k in rates})
    has_nominal = any(k[1] == nominal for k in rates)
    sim, real, table = [], [], []
    for p in policies:
        if has_nominal:
            s, r = rates.get((p, nominal, "sim")), rates.get((p, nominal, "real"))
        else:
            s = tuple(map(sum, zip(*[v for k, v in rates.items() if k[0] == p and k[2] == "sim"]))) or None
            r = tuple(map(sum, zip(*[v for k, v in rates.items() if k[0] == p and k[2] == "real"]))) or None
        if not s or not r or not s[1] or not r[1]:
            continue
        sim.append(s[0] / s[1])
        real.append(r[0] / r[1])
        table.append({"policy": p, "sim_rate": s[0] / s[1], "sim_n": s[1], "real_rate": r[0] / r[1], "real_n": r[1],
                      "sim_ci": wilson_interval(s[0], s[1], confidence), "real_ci": wilson_interval(r[0], r[1], confidence)})
    out: dict[str, Any] = {
        "policies": len(table),
        "table": table,
        "pearson": pearson(sim, real),
        "spearman": spearman(sim, real) if len(sim) >= 2 else math.nan,
        "kendall_tau": kendall_tau(sim, real),
        "mmrv": mmrv(sim, real),
        "calibration": calibration(sim, real, confidence),
        "sensitivity": sensitivity_alignment(rows, nominal),
        "warnings": [],
    }
    if len(table) < 5:
        out["warnings"].append(f"only {len(table)} policies on both sides: correlation and MMRV are unreliable below "
                               "about 5 policies of clearly different quality")
    if len(sim) >= 2 and np.ptp(real) < 0.1:
        out["warnings"].append("real success rates span less than 10 points: there is little ranking to predict")
    small = [t["policy"] for t in table if min(t["sim_n"], t["real_n"]) < 20]
    if small:
        out["warnings"].append(f"fewer than 20 episodes on a side for {small[:5]}; rate noise can dominate the metrics")
    if gate is not None and table:
        out["decisions"] = decision_agreement([t["sim_rate"] < gate for t in table], [t["real_rate"] < gate for t in table])
    return out


# =====================================================================================
# Tier 2: component fidelity
# =====================================================================================
# -- trajectories -----------------------------------------------------------------
def nrmse(pred: np.ndarray, rec: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    """Per-dimension RMS error divided by the recorded signal's spread (NaN-safe)."""
    p, r = np.atleast_2d(np.asarray(pred, float)), np.atleast_2d(np.asarray(rec, float))
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {r.shape}")
    sc = np.nanstd(r, axis=0) if scale is None else np.asarray(scale, float)
    err = p - r
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(np.nanmean(err ** 2, axis=0)) / np.where(sc > _EPS, sc, np.nan)
    return out


def divergence_step(pred: np.ndarray, rec: np.ndarray, scale: np.ndarray | None = None,
                    threshold: float = 0.5) -> int | None:
    """First step where any dimension's error exceeds ``threshold`` x its spread (None if never)."""
    p, r = np.asarray(pred, float), np.asarray(rec, float)
    sc = np.nanstd(r, axis=0) if scale is None else np.asarray(scale, float)
    live = sc > _EPS
    if not live.any():
        return None
    norm = np.abs(p[:, live] - r[:, live]) / sc[live]
    bad = np.where(np.nan_to_num(norm, nan=np.inf).max(axis=1) > threshold)[0]
    return int(bad[0]) if bad.size else None


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int | None = None, scale: np.ndarray | None = None) -> float:
    """Dynamic-time-warping distance per aligned step, in units of the signal's spread.

    Unlike step-by-step error, DTW forgives a trajectory that does the right
    thing slightly early or late, so ``dtw`` small with ``nrmse`` large means
    timing (latency, delay), not path shape, is what differs.
    """
    x, y = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    if x.ndim == 2 and x.shape[0] == 1 and x.shape[1] > 1 and y.shape[0] == 1:
        x, y = x.T, y.T
    sc = np.nanstd(np.concatenate([x, y]), axis=0) if scale is None else np.asarray(scale, float)
    sc = np.where(sc > _EPS, sc, 1.0)
    x, y = x / sc, y / sc
    n, m = len(x), len(y)
    w = max(window or max(n, m), abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            d = float(np.linalg.norm(x[i - 1] - y[j - 1]))
            prev = min((cost[i - 1, j], steps[i - 1, j]), (cost[i, j - 1], steps[i, j - 1]),
                       (cost[i - 1, j - 1], steps[i - 1, j - 1]))
            cost[i, j] = d + prev[0]
            steps[i, j] = prev[1] + 1
    return float(cost[n, m] / max(steps[n, m], 1))


# -- actuators ----------------------------------------------------------------------
def actuator_tracking(commanded: np.ndarray, measured: np.ndarray, max_lag: int = 10) -> list[dict[str, float]]:
    """Per joint: tracking nRMSE, the lag (steps) that best aligns measured with commanded, and the gain.

    Compare the numbers between sim and real: a real joint with a lag of 3
    steps that the twin tracks with lag 0 is a missing actuator delay; a gain
    of 0.8 against 1.0 is a weak or worn motor the twin does not model.
    """
    c, m = np.atleast_2d(np.asarray(commanded, float)), np.atleast_2d(np.asarray(measured, float))
    if c.shape != m.shape:
        raise ValueError(f"shape mismatch {c.shape} vs {m.shape}")
    out = []
    for j in range(c.shape[1]):
        cj, mj = c[:, j], m[:, j]
        best_lag, best_err = 0, math.inf
        for lag in range(0, min(max_lag, len(cj) - 2) + 1):
            e = float(np.nanmean((mj[lag:] - cj[: len(cj) - lag]) ** 2))
            if e < best_err:
                best_lag, best_err = lag, e
        a, b = cj[: len(cj) - best_lag], mj[best_lag:]
        ok = np.isfinite(a) & np.isfinite(b)
        gain = float(np.dot(a[ok], b[ok]) / max(np.dot(a[ok], a[ok]), _EPS)) if ok.any() else math.nan
        sc = float(np.nanstd(cj)) or 1.0
        out.append({"joint": j, "nrmse": float(np.sqrt(np.nanmean((mj - cj) ** 2)) / sc), "lag_steps": best_lag,
                    "gain": gain, "nrmse_at_lag": math.sqrt(best_err) / sc})
    return out


def saturation_fraction(values: np.ndarray, limit: float | Sequence[float] | None = None, tol: float = 1e-3) -> np.ndarray:
    """Share of steps each dimension sits at its limit (given, or the observed max |value|)."""
    v = np.atleast_2d(np.asarray(values, float))
    lim = np.nanmax(np.abs(v), axis=0) if limit is None else np.broadcast_to(np.asarray(limit, float), (v.shape[1],))
    lim = np.where(lim > _EPS, lim, np.inf)
    return np.mean(np.abs(v) >= lim * (1 - tol), axis=0)


# -- timing -------------------------------------------------------------------------
def wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    """1-D Wasserstein (earth mover's) distance between two samples."""
    x, y = np.sort(np.asarray(a, float)[np.isfinite(a)]), np.sort(np.asarray(b, float)[np.isfinite(b)])
    if not x.size or not y.size:
        return math.nan
    q = np.linspace(0, 1, max(x.size, y.size, 2))
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (largest CDF gap)."""
    x, y = np.sort(np.asarray(a, float)[np.isfinite(a)]), np.sort(np.asarray(b, float)[np.isfinite(b)])
    if not x.size or not y.size:
        return math.nan
    grid = np.concatenate([x, y])
    return float(np.max(np.abs(np.searchsorted(x, grid, "right") / x.size - np.searchsorted(y, grid, "right") / y.size)))


def latency_match(sim_ms: Sequence[float], real_ms: Sequence[float], control_dt_ms: float | None = None) -> dict[str, float]:
    """Does the twin feel the robot's inference latency? Distances in ms and in control steps."""
    s, r = np.asarray(sim_ms, float), np.asarray(real_ms, float)
    s, r = s[np.isfinite(s)], r[np.isfinite(r)]
    out: dict[str, float] = {"w1_ms": wasserstein1(s, r), "ks": ks_statistic(s, r)}
    for q in (50, 95, 99):
        out[f"p{q}_gap_ms"] = float(np.percentile(s, q) - np.percentile(r, q)) if s.size and r.size else math.nan
    if control_dt_ms:
        out["w1_steps"] = out["w1_ms"] / control_dt_ms
        out["p95_gap_steps"] = out["p95_gap_ms"] / control_dt_ms
    return out


# -- failures -----------------------------------------------------------------------
def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (base 2, 0..1) between two label distributions."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys], float)
    b = np.array([q.get(k, 0.0) for k in keys], float)
    if a.sum() <= 0 or b.sum() <= 0:
        return math.nan
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def failure_mode_confusion(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Confusion between real failure modes and what the twin produced for the same scenarios.

    ``pairs`` = (real_mode, twin_mode); use ``"success"`` when a side succeeded.
    A twin can match the success rate and still fail for different reasons, so
    the mode agreement and the JS divergence between the two mode
    distributions matter as much as the rate.
    """
    labels = sorted({x for p in pairs for x in p})
    matrix = {r: {t: 0 for t in labels} for r in labels}
    for r, t in pairs:
        matrix[r][t] += 1
    n = len(pairs)
    agree = sum(matrix[l][l] for l in labels)
    real_d, twin_d = Counter(r for r, _ in pairs), Counter(t for _, t in pairs)
    per_mode = {}
    for l in labels:
        row = sum(matrix[l].values())
        if row and l != "success":
            per_mode[l] = {"real_count": row, "twin_reproduced": matrix[l][l], "recall": matrix[l][l] / row,
                           "twin_said": dict(Counter({t: c for t, c in matrix[l].items() if c}).most_common())}
    return {"labels": labels, "matrix": matrix, "pairs": n, "agreement": agree / n if n else math.nan,
            "js_divergence": js_divergence(dict(real_d), dict(twin_d)), "per_mode": per_mode}


def reproduction_score(real_mode: str, twin_outcomes: Sequence[tuple[bool, str | None]],
                       confidence: float = 0.95) -> dict[str, Any]:
    """How often the twin reproduces one real failure, from repeated twin runs of that scenario.

    ``twin_outcomes`` = (success, failure_reason) per twin run. Returns the
    twin's failure rate and same-mode rate, each with a Wilson interval.
    """
    n = len(twin_outcomes)
    fails = sum(1 for ok, _ in twin_outcomes if not ok)
    same = sum(1 for ok, mode in twin_outcomes if not ok and (mode or "unknown") == real_mode)
    modes = Counter((mode or "unknown") if not ok else "success" for ok, mode in twin_outcomes)
    return {"runs": n, "fail_rate": fails / n if n else math.nan, "fail_ci": wilson_interval(fails, n, confidence),
            "same_mode_rate": same / n if n else math.nan, "same_mode_ci": wilson_interval(same, n, confidence),
            "modes": dict(modes.most_common())}


# -- vision -------------------------------------------------------------------------
def _as_float(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    f = a.astype(np.float64)
    if np.issubdtype(a.dtype, np.floating) and a.size and float(np.nanmax(a)) <= 1.0 + 1e-6:
        f = f * 255.0
    return f


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB (inf for identical frames). >40 dB is visually lossless."""
    x, y = _as_float(a), _as_float(b)
    if x.shape != y.shape:
        raise ValueError(f"frame shapes differ: {x.shape} vs {y.shape}")
    mse = float(np.mean((x - y) ** 2))
    return math.inf if mse <= _EPS else 10 * math.log10(data_range ** 2 / mse)


def _gauss_filter(img: np.ndarray, sigma: float = 1.5, radius: int = 5) -> np.ndarray:
    t = np.arange(-radius, radius + 1)
    k = np.exp(-(t ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    pad = np.pad(img, ((radius, radius), (radius, radius)), mode="reflect")
    tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, tmp)


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Structural similarity (Wang et al. 2004, Gaussian window), averaged over channels. 1 = identical."""
    x, y = _as_float(a), _as_float(b)
    if x.shape != y.shape:
        raise ValueError(f"frame shapes differ: {x.shape} vs {y.shape}")
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    vals = []
    for ch in range(x.shape[-1]):
        u, v = x[..., ch], y[..., ch]
        mu_u, mu_v = _gauss_filter(u), _gauss_filter(v)
        suu = _gauss_filter(u * u) - mu_u ** 2
        svv = _gauss_filter(v * v) - mu_v ** 2
        suv = _gauss_filter(u * v) - mu_u * mu_v
        m = ((2 * mu_u * mu_v + c1) * (2 * suv + c2)) / ((mu_u ** 2 + mu_v ** 2 + c1) * (suu + svv + c2))
        vals.append(float(m.mean()))
    return float(np.mean(vals))


def channel_order_check(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Detect an RGB/BGR swap: compare PSNR as-is against PSNR with the candidate's channels reversed."""
    as_is = psnr(reference, candidate)
    swapped = psnr(reference, np.asarray(candidate)[..., ::-1])
    return {"psnr_as_is": as_is, "psnr_swapped": swapped,
            "swapped": bool(np.isfinite(swapped) and swapped > as_is + 3.0 or (math.isinf(swapped) and not math.isinf(as_is)))}


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian on a 0-1 gray image (lower = blurrier)."""
    g = _as_float(img)
    g = g.mean(axis=-1) if g.ndim == 3 else g
    if g.shape[0] < 3 or g.shape[1] < 3:
        return math.nan
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]) / 255.0
    return float(lap.var())


def sharpness_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Candidate sharpness over reference sharpness; well below 1 = detail lost (blur, downscale, codec)."""
    r = sharpness(reference)
    return sharpness(candidate) / r if r and math.isfinite(r) and r > _EPS else math.nan


def _thumb(img: np.ndarray, size: int = 32) -> np.ndarray:
    g = _as_float(img)
    g = g.mean(axis=-1) if g.ndim == 3 else g
    ys = np.linspace(0, g.shape[0] - 1, size).astype(int)
    xs = np.linspace(0, g.shape[1] - 1, size).astype(int)
    return g[np.ix_(ys, xs)]


def best_frame_offset(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray], max_offset: int = 5) -> dict[str, Any]:
    """Which shift aligns the candidate frame sequence with the reference (catches dropped or extra frames).

    Uses small grayscale thumbnails, so differing resolutions still align.
    Offset k means candidate[i] matches reference[i + k].
    """
    ref = [_thumb(f) for f in reference]
    cand = [_thumb(f) for f in candidate]
    scores = {}
    for k in range(-max_offset, max_offset + 1):
        pairs = [(ref[i + k], cand[i]) for i in range(len(cand)) if 0 <= i + k < len(ref)]
        if len(pairs) >= max(3, min(len(ref), len(cand)) // 2):
            scores[k] = float(np.mean([np.mean((a - b) ** 2) for a, b in pairs]))
    if not scores:
        return {"offset": 0, "mse_by_offset": {}}
    best = min(scores, key=scores.get)
    return {"offset": int(best), "mse_at_best": scores[best], "mse_at_zero": scores.get(0, math.nan),
            "mse_by_offset": scores}


def policy_visual_gap(reference_frames: Sequence[np.ndarray], candidate_frames: Sequence[np.ndarray],
                      encoder: Callable[[np.ndarray], np.ndarray] | None = None,
                      action_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> dict[str, float]:
    """The gap the policy actually sees: embedding distance and action divergence on matched frames.

    ``encoder(frame) -> vector`` (e.g. the VLA's vision tower) and
    ``action_fn(frame) -> action`` (the policy with everything else held
    fixed) are optional hooks; pixel metrics alone over- and under-state
    what matters to the policy.
    """
    out: dict[str, float] = {}
    if encoder is not None:
        d = []
        for a, b in zip(reference_frames, candidate_frames):
            u, v = np.asarray(encoder(a), float).ravel(), np.asarray(encoder(b), float).ravel()
            d.append(1 - float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), _EPS)))
        out["embedding_cosine_distance_mean"] = float(np.mean(d)) if d else math.nan
        out["embedding_cosine_distance_p95"] = float(np.percentile(d, 95)) if d else math.nan
    if action_fn is not None:
        e = [float(np.linalg.norm(np.asarray(action_fn(a), float) - np.asarray(action_fn(b), float)))
             for a, b in zip(reference_frames, candidate_frames)]
        out["action_divergence_mean"] = float(np.mean(e)) if e else math.nan
        out["action_divergence_max"] = float(np.max(e)) if e else math.nan
    return out


# =====================================================================================
# Tier 3: value
# =====================================================================================
def control_variate_savings(paired_sim: Sequence[float], paired_real: Sequence[float]) -> dict[str, float]:
    """How many real rollouts the twin saves when used as a control variate.

    With per-scenario sim and real outcomes correlated at rho, a
    regression estimator that combines a few real runs with many cheap twin
    runs has variance (1 - rho^2) times that of real runs alone, so each real
    run is worth 1 / (1 - rho^2) of them (the idea behind SureSim-style
    evaluation). rho near 0 means the twin saves nothing, whatever its
    average success rate.
    """
    s, r = np.asarray(paired_sim, float), np.asarray(paired_real, float)
    rho = pearson(s, r)
    if not math.isfinite(rho):
        return {"pairs": int(s.size), "rho": math.nan, "variance_factor": 1.0, "real_runs_multiplier": 1.0,
                "real_runs_saved_share": 0.0}
    vf = max(1 - rho ** 2, 1e-6)
    return {"pairs": int(s.size), "rho": rho, "variance_factor": vf, "real_runs_multiplier": 1 / vf,
            "real_runs_saved_share": 1 - vf}


def coverage_vs_field(twin: np.ndarray, field: np.ndarray, margin: float = 0.05,
                      names: Sequence[str] | None = None) -> dict[str, Any]:
    """Share of field (deployment) samples whose scenario parameters fall inside the twin's tested envelope.

    ``twin`` and ``field`` are (N, D) arrays of the same scenario parameters
    (lighting lux, payload kg, floor friction...). Low coverage means the twin
    is validated on conditions the fleet does not live in.
    """
    t, f = np.atleast_2d(np.asarray(twin, float)), np.atleast_2d(np.asarray(field, float))
    lo, hi = np.nanpercentile(t, 0.5, axis=0), np.nanpercentile(t, 99.5, axis=0)
    pad = margin * np.maximum(hi - lo, _EPS)
    inside = (f >= lo - pad) & (f <= hi + pad)
    names = list(names) if names else [f"dim[{i}]" for i in range(t.shape[1])]
    return {"coverage": float(np.mean(np.all(inside, axis=1))),
            "per_dimension": {n: float(v) for n, v in zip(names, inside.mean(axis=0))},
            "field_samples": int(f.shape[0])}
