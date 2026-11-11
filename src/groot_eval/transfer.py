"""Sim-to-real and real-to-sim transfer verification: is the data still faithful, compatible and usable?

The fidelity gates (:mod:`groot_eval.fidelity`, G0-G6) ask whether sim and real
data describe the *same world*. This module asks the questions that come up
every time data or a policy crosses the boundary:

* **Fidelity**: did the transfer lose information? Precision (quantization),
  clipping, high-frequency content (filtering or resampling), NaNs, columns and
  metadata that did not survive, and, given the same episodes before and after
  a conversion, any value that changed (round trip).
* **Compatibility**: does the receiving side read it the same way? Schema,
  key order inside ``observation.state``, action groups, control rate, dtypes,
  embodiment and calibration ids, declared units, swapped joints, flipped
  action signs or very different actuator gain, quaternion order.
* **Usability**: can the receiving side use it? Loadable with ``modality.json``,
  long enough for action-chunk training, task strings, no dead action channels
  or duplicate episodes, accepted by the target env (keys, widths, action size,
  representable ranges, ``set_state`` for replay) and by the policy (finite,
  right-sized, in-range actions; open-loop error not much worse than on the
  reference data).
* **Direction-specific** checks, because the two directions fail differently:

  - ``sim2real`` (sim data or a sim-validated policy goes to the robot): the
    data must respect the *robot's* limits and actuator rate, the sim must model
    the robot's inference latency, sensor noise and action jerkiness, camera
    resolution must match, and the sim must have covered the deployed robot's
    wear (its age in hours).
  - ``real2sim`` (robot logs go into sim: calibration, digital twins, replay
    seeds): calibration and robot age must be recorded, actions must be the
    applied ones and inside the robot's range, states inside its limits, the
    latency recorded, episodes named for pairing, and the sim's reset
    distribution must cover the robot's start states.

Gates ``T0`` (round trip, only with ``--roundtrip``) to ``T4`` run in order. If
compatibility (T1) fails, the cross-domain comparisons (T2, T4) are skipped,
because they would compare the wrong signals; usability (T3) still runs. Exit
codes match ``fidelity``: 0 pass, 1 a gate failed, 2 a required gate could not
run. Thresholds live in ``configs/transfer/defaults.yaml``; the robot's limits
in a robot profile (``configs/robots/*.yaml``). Both are starting points, not
validated limits.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.fidelity import DIRECTIONS, GateResult, _plain, _status, exit_code
from groot_eval.trace import TraceSet, dim_names
from groot_eval.types import Finding, Scenario

STANDARD_COLUMNS = {"observation.state", "action", "frame_index", "episode_index", "index", "task_index"}
DEFAULTS: dict[str, dict[str, Any]] = {
    "T0_roundtrip": {"required": False, "max_rel_error": 1e-4},
    "T1_compatibility": {"required": True, "permutation_margin": 0.5, "min_response_corr": 0.3,
                         "max_gain_ratio": 3.0},
    "T2_fidelity": {"required": True, "max_levels": 200, "level_ratio": 0.2, "max_clipped_frac": 0.02,
                    "clip_factor": 5.0, "min_hf_ratio": 0.2, "hf_cut": 0.25, "max_nan_frac": 0.0},
    "T3_usability": {"required": True, "min_episode_steps": 16, "policy_samples": 50, "horizon": 8,
                     "open_loop_episodes": 5, "max_open_loop_ratio": 3.0},
    "T4_direction": {"required": True, "max_latency_gap_steps": 0.5, "max_noise_ratio": 2.0,
                     "max_jerk_ratio": 2.0, "min_start_coverage": 0.9, "envelope_margin": 0.05},
}
WHAT = {
    "T0_roundtrip": "the same episodes before and after a conversion: every value, column, length and metadata key survived",
    "T1_compatibility": "both sides use the same schema, key order, action layout, rate, dtypes, ids, units, joint "
                        "order, action signs and quaternion convention",
    "T2_fidelity": "the promoted data kept its precision, range, high-frequency content, columns and metadata",
    "T3_usability": "the promoted data loads, is fit for training, and the target env and policy accept it",
    "T4_direction": "the checks specific to this direction (robot limits, latency, noise, wear coverage, "
                    "calibration, applied actions, start states)",
}


def load_config(path: str | Path | None) -> tuple[dict[str, dict[str, Any]], str | None]:
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    direction = None
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        direction = data.get("direction")
        for k, v in (data.get("gates") or {}).items():
            if k not in cfg:
                raise ValueError(f"unknown transfer gate {k!r}; known: {sorted(cfg)}")
            cfg[k].update(v or {})
    return cfg, direction


def load_robot(path: str | Path | None) -> dict[str, Any]:
    """Robot profile: the limits the hardware accepts (``configs/robots/*.yaml``)."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# -- helpers ---------------------------------------------------------------------------------------
def _w1(a: np.ndarray, b: np.ndarray) -> float:
    from groot_eval.gap import wasserstein1

    return wasserstein1(a, b)


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _dims(ts: TraceSet, column: str) -> tuple[list[str], list[np.ndarray]]:
    eps = ts.per_episode(column)
    if not eps:
        return [], []
    slices = ts.state_slices() if column == "observation.state" else ts.action_slices()
    names = dim_names(slices)
    if len(names) != eps[0].shape[1]:
        names = [f"{column.split('.')[-1]}[{i}]" for i in range(eps[0].shape[1])]
    return names, eps


def _levels(x: np.ndarray) -> tuple[bool, float, float]:
    """(quantized?, step, number of levels across the 1-99% range)."""
    x = _finite(x)
    if x.size < 20:
        return False, math.nan, math.inf
    u = np.unique(x)
    lo, hi = np.percentile(x, [1, 99])
    spread = hi - lo
    if spread <= 0 or u.size > 0.2 * x.size:
        return False, math.nan, math.inf
    d = np.diff(u)
    d = d[d > 1e-12 * max(1.0, float(np.max(np.abs(u))))]
    if not d.size:
        return True, math.inf, 1.0
    step = float(np.median(d))
    return True, step, spread / step


def _pinned(x: np.ndarray) -> float:
    """Share of samples sitting exactly at the signal's min or max: clipping leaves a pile there."""
    x = _finite(x)
    if x.size < 20:
        return 0.0
    lo, hi = x.min(), x.max()
    if hi - lo <= 0:
        return 0.0
    tol = 1e-6 * (hi - lo)
    return float(np.mean((x <= lo + tol) | (x >= hi - tol)))


def _hf_share(episodes: list[np.ndarray], cut: float) -> float:
    """Share of (detrended) signal power above ``cut`` x the sample rate, averaged over episodes."""
    shares = []
    for x in episodes:
        x = x[np.isfinite(x)]
        if x.size < 32:
            continue
        t = np.arange(x.size)
        x = x - np.polyval(np.polyfit(t, x, 1), t)
        p = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(x.size)
        total = p[1:].sum()
        if total > 1e-18:
            shares.append(float(p[f > cut].sum() / total))
    return float(np.mean(shares)) if shares else math.nan


def _response(states: list[np.ndarray], actions: list[np.ndarray], i: int, j: int) -> tuple[float, float]:
    """Correlation and slope between action ``i`` at t and the change of state ``j`` from t to t+1."""
    a = np.concatenate([A[:-1, i] for A in actions if len(A) > 2])
    ds = np.concatenate([np.diff(S[:, j]) for S in states if len(S) > 2])
    ok = np.isfinite(a) & np.isfinite(ds)
    a, ds = a[ok], ds[ok]
    if a.size < 20 or a.std() < 1e-12 or ds.std() < 1e-12:
        return math.nan, math.nan
    c = float(np.corrcoef(a, ds)[0, 1])
    return c, float(np.cov(a, ds)[0, 1] / a.var())


# -- T0 round trip ---------------------------------------------------------------------------------
def roundtrip(original: TraceSet, converted: TraceSet, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    """Compare the same episodes before and after a conversion (export, import, resampling, re-encoding)."""
    out: list[Finding] = []
    tol = float(cfg["max_rel_error"])
    fo, fc = original.frames(), converted.frames()
    m: dict[str, Any] = {"episodes": [len(fo), len(fc)]}
    if len(fo) != len(fc):
        out.append(Finding("error", "episodes", f"{len(fo)} episodes before, {len(fc)} after"))
    lengths = sum(len(a) != len(b) for a, b in zip(fo, fc))
    if lengths:
        out.append(Finding("error", "length", f"{lengths} episode(s) changed length (dropped or resampled steps)"))
    worst: dict[str, float] = {}
    nan_new: dict[str, int] = {}
    lost: set[str] = set()
    for a, b in zip(fo, fc):
        if len(a) != len(b):
            continue
        for col in a.columns:
            if col in ("frame_index", "episode_index", "index", "task_index"):
                continue
            if col not in b.columns:
                lost.add(col)
                continue
            try:
                x = np.stack(a[col].to_numpy()).astype(float) if col in ("observation.state", "action") else a[col].to_numpy(dtype=float)
                y = np.stack(b[col].to_numpy()).astype(float) if col in ("observation.state", "action") else b[col].to_numpy(dtype=float)
            except (TypeError, ValueError):
                continue
            if x.shape != y.shape:
                worst[col] = math.inf
                continue
            new_nan = int(np.sum(np.isfinite(x) & ~np.isfinite(y)))
            if new_nan:
                nan_new[col] = nan_new.get(col, 0) + new_nan
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            spread = np.nanstd(x, axis=0) if x.ndim > 1 else np.nanstd(x)
            scale = np.maximum(np.abs(spread), 1e-9)
            err = np.abs(np.where(ok, x - y, 0.0)) / scale
            worst[col] = max(worst.get(col, 0.0), float(np.max(err)))
    for col, e in sorted(worst.items(), key=lambda kv: -kv[1]):
        if e > tol:
            out.append(Finding("error", "values", f"{col}: values changed by up to {e:.3g} of the signal's spread "
                                                  f"(limit {tol:g}): unit conversion, rounding, re-encoding or resampling"))
    if nan_new:
        out.append(Finding("error", "finite", f"conversion introduced NaN/Inf: {dict(list(nan_new.items())[:6])}"))
    if lost:
        out.append(Finding("warn", "columns", f"columns dropped by the conversion: {sorted(lost)[:10]}"))
    mo, mc = original.episode_meta(), converted.episode_meta()
    keys_lost = sorted({k for a, b in zip(mo, mc) for k in a if k not in b and k not in ("length",)})
    if keys_lost:
        out.append(Finding("warn", "metadata", f"episode metadata dropped: {keys_lost[:10]}"))
    tasks = sum((a.get("tasks") or [""])[0] != (b.get("tasks") or [""])[0] for a, b in zip(mo, mc))
    if tasks:
        out.append(Finding("error", "tasks", f"{tasks} episode(s) changed their task string"))
    if original.fps and converted.fps and abs(original.fps - converted.fps) > 1e-6:
        out.append(Finding("error", "fps", f"control rate changed from {original.fps:g} to {converted.fps:g} Hz"))
    if original.dataset.modality != converted.dataset.modality:
        out.append(Finding("error", "modality", "modality.json differs (key layout of state, action or video changed)"))
    ignore = {"imported_from", "episodes", "fps", "control_dt"}
    prov = sorted(k for k, v in original.info.items() if k not in ignore and v not in (None, "", [])
                  and converted.info.get(k) != v)
    if prov:
        out.append(Finding("warn", "provenance", f"trace_info fields lost or changed: {prov[:8]}"))
    m["max_rel_error"] = max(worst.values()) if worst else 0.0
    if not any(f.level != "ok" for f in out):
        out.append(Finding("ok", "roundtrip", f"{len(fo)} episodes identical within {tol:g} of each signal's spread; "
                                              "all columns, metadata and layout preserved"))
    return out, m


# -- T1 compatibility ------------------------------------------------------------------------------
def compatibility(p: TraceSet, r: TraceSet, pl: str, rl: str, cfg: dict[str, Any],
                  direction: str) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    m: dict[str, Any] = {}
    for key, where in (("schema_version", p.info), ("codebase_version", p.dataset.info)):
        other = r.info if where is p.info else r.dataset.info
        if where.get(key) is not None and other.get(key) is not None and where.get(key) != other.get(key):
            out.append(Finding("warn", "schema", f"{key} {where.get(key)} in {pl} vs {other.get(key)} in {rl}"))
    ps, rs = p.state_slices(), r.state_slices()
    missing, extra = sorted(set(rs) - set(ps)), sorted(set(ps) - set(rs))
    if missing:
        out.append(Finding("error", "state.keys", f"{pl} lacks {missing}, which {rl} has"))
    if extra:
        out.append(Finding("warn", "state.keys", f"{pl} has {extra}, which {rl} lacks (ignored by the other side)"))
    common = [k for k in rs if k in ps]
    widths = [k for k in common if ps[k].stop - ps[k].start != rs[k].stop - rs[k].start]
    for k in widths:
        out.append(Finding("error", "state.width", f"state.{k}: {ps[k].stop - ps[k].start} values in {pl}, "
                                                    f"{rs[k].stop - rs[k].start} in {rl}"))
    if not missing and not extra and [k for k in ps] != [k for k in rs]:
        out.append(Finding("error", "state.order", f"observation.state concatenates the keys in a different order: "
                                                    f"{list(ps)} in {pl}, {list(rs)} in {rl}; a policy reading by "
                                                    "position gets the wrong signal"))
    pa, ra = p.stacked("action"), r.stacked("action")
    if pa.size and ra.size and pa.shape[1] != ra.shape[1]:
        out.append(Finding("error", "action.dim", f"action has {pa.shape[1]} dims in {pl}, {ra.shape[1]} in {rl}"))
    pg, rg = list(p.action_slices()), list(r.action_slices())
    if set(pg) == set(rg) and pg != rg:
        out.append(Finding("error", "action.order", f"action groups in a different order: {pg} vs {rg}"))
    if p.fps and r.fps and abs(p.fps - r.fps) / r.fps > 0.02:
        out.append(Finding("error", "fps", f"{p.fps:g} Hz in {pl} vs {r.fps:g} Hz in {rl}"))
    for col in ("observation.state", "action"):
        dp = (p.dataset.info.get("features", {}).get(col) or {}).get("dtype")
        dr = (r.dataset.info.get("features", {}).get(col) or {}).get("dtype")
        if dp and dr and dp != dr:
            out.append(Finding("warn", "dtype", f"{col} is {dp} in {pl}, {dr} in {rl}"))
        if dp == "float16":
            out.append(Finding("error", "dtype", f"{col} stored as float16 in {pl}: about 3 significant digits"))
    pv, rv = set((p.dataset.modality.get("video") or {})), set((r.dataset.modality.get("video") or {}))
    if rv - pv:
        out.append(Finding("error", "video.keys", f"{pl} has no camera stream for {sorted(rv - pv)}"))
    pin, rin = {**p.dataset.info, **p.info}, {**r.dataset.info, **r.info}
    for key in ("embodiment", "embodiment_tag", "robot_type"):
        if pin.get(key) and rin.get(key) and pin[key] != rin[key]:
            out.append(Finding("error", key, f"{pin[key]!r} in {pl} vs {rin[key]!r} in {rl}: different robots"))
    if pin.get("calibration_id") and rin.get("calibration_id") and pin["calibration_id"] != rin["calibration_id"]:
        lvl = "error" if direction == "real2sim" else "warn"
        out.append(Finding(lvl, "calibration_id", f"{pin['calibration_id']!r} in {pl} vs {rin['calibration_id']!r} "
                                                   f"in {rl}: joint zeros may differ"))
    pu, ru = pin.get("units") or {}, rin.get("units") or {}
    for k in sorted(set(pu) & set(ru)):
        if str(pu[k]).lower() != str(ru[k]).lower():
            out.append(Finding("error", "units", f"{k}: declared {pu[k]!r} in {pl}, {ru[k]!r} in {rl}"))
    # swapped dimensions: dims whose distributions match the *other* index much better than their own
    for key in common:
        if key in widths or ps[key].stop - ps[key].start < 2:
            continue
        P, R = p.stacked("observation.state")[:, ps[key]], r.stacked("observation.state")[:, rs[key]]
        out += _swaps(f"state.{key}", P, R, cfg["permutation_margin"])
    if pa.size and ra.size and pa.shape[1] == ra.shape[1] and pa.shape[1] >= 2:
        out += _swaps("action", pa, ra, cfg["permutation_margin"])
    # action -> response: sign and gain of what each command does to the state
    if pa.size and ra.size and pa.shape[1] == ra.shape[1] and not missing and not widths and not extra:
        out += _responses(p, r, pl, rl, cfg, m)
    # quaternions
    for key in common:
        if ps[key].stop - ps[key].start == 4 and any(t in key.lower() for t in ("quat", "orient", "rot")):
            out += _quaternion(key, p.stacked("observation.state")[:, ps[key]], r.stacked("observation.state")[:, rs[key]], pl, rl)
    if not any(f.level != "ok" for f in out):
        out.append(Finding("ok", "compatibility", f"same layout, rate, ids and conventions on both sides"))
    return out, m


def _swaps(label: str, P: np.ndarray, R: np.ndarray, margin: float) -> list[Finding]:
    n = P.shape[1]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = _finite(P[:, i]), _finite(R[:, j])
            if not a.size or not b.size:
                C[i, j] = math.nan
                continue
            pooled = math.sqrt((np.var(a) + np.var(b)) / 2) or 1e-9
            C[i, j] = _w1(a, b) / pooled
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            own, swapped = C[i, i] + C[j, j], C[i, j] + C[j, i]
            if np.isfinite(own) and np.isfinite(swapped) and own > 1.0 and swapped < margin * own:
                out.append(Finding("error", "order", f"{label}[{i}] and [{j}] look swapped: each matches the other "
                                                      f"side's other index ({swapped:.2f} vs {own:.2f} pooled std)"))
    return out


def _responses(p: TraceSet, r: TraceSet, pl: str, rl: str, cfg: dict[str, Any], m: dict[str, Any]) -> list[Finding]:
    out = []
    Sr, Ar = r.per_episode("observation.state"), r.per_episode("action")
    Sp, Ap = p.per_episode("observation.state"), p.per_episode("action")
    names = dim_names(r.state_slices())
    anames = dim_names(r.action_slices())
    if len(anames) != Ar[0].shape[1]:
        anames = [f"action[{i}]" for i in range(Ar[0].shape[1])]
    gains = {}
    for i in range(Ar[0].shape[1]):
        best, bj = 0.0, None
        for j in range(Sr[0].shape[1]):
            c, _ = _response(Sr, Ar, i, j)
            if math.isfinite(c) and abs(c) > abs(best):
                best, bj = c, j
        if bj is None or abs(best) < cfg["min_response_corr"]:
            continue
        cr, sr = _response(Sr, Ar, i, bj)
        cp, sp = _response(Sp, Ap, i, bj)
        if not math.isfinite(cp):
            continue
        tag = f"{anames[i]} -> d({names[bj]})"
        if abs(cp) >= cfg["min_response_corr"] / 2 and np.sign(cp) != np.sign(cr):
            out.append(Finding("error", "sign", f"{tag}: the command moves the state the opposite way in {pl} "
                                                f"(corr {cp:+.2f}) and {rl} (corr {cr:+.2f}): sign convention flipped"))
            continue
        ratio = sp / sr if sr else math.nan
        gains[tag] = ratio
        if math.isfinite(ratio) and ratio > 0 and (ratio > cfg["max_gain_ratio"] or ratio < 1 / cfg["max_gain_ratio"]):
            out.append(Finding("warn", "gain", f"{tag}: the same command moves the state {ratio:.2f}x as much in {pl} "
                                               f"as in {rl}: action scaling, actuator gain or wear differ"))
    m["action_gain_ratio"] = gains
    if gains and not out:
        out.append(Finding("ok", "response", "every responsive command moves the state the same way on both sides "
                                             f"(gain ratios {min(gains.values()):.2f}-{max(gains.values()):.2f})"))
    return out


def _quaternion(key: str, P: np.ndarray, R: np.ndarray, pl: str, rl: str) -> list[Finding]:
    out = []
    for label, Q in ((pl, P), (rl, R)):
        nrm = np.linalg.norm(Q[np.all(np.isfinite(Q), axis=1)], axis=1)
        if nrm.size and abs(float(np.median(nrm)) - 1.0) > 0.01:
            out.append(Finding("error", "quaternion", f"state.{key} in {label}: median norm {np.median(nrm):.3f}, not unit quaternions"))
    big_p, big_r = int(np.argmax(np.nanmean(np.abs(P), axis=0))), int(np.argmax(np.nanmean(np.abs(R), axis=0)))
    if {big_p, big_r} == {0, 3}:
        out.append(Finding("error", "quaternion", f"state.{key}: the dominant component is index {big_p} in {pl} and "
                                                   f"{big_r} in {rl}: wxyz vs xyzw order"))
    return out


# -- T2 fidelity -----------------------------------------------------------------------------------
def information(p: TraceSet, r: TraceSet, pl: str, rl: str, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    m: dict[str, Any] = {"quantized": {}, "clipped": {}, "hf_share": {}}
    for column in ("observation.state", "action"):
        pn, pe = _dims(p, column)
        rn, re_ = _dims(r, column)
        if not pe or not re_ or pe[0].shape[1] != re_[0].shape[1]:
            continue
        P, R = np.concatenate(pe), np.concatenate(re_)
        for d, name in enumerate(pn):
            x, y = P[:, d], R[:, d]
            label = f"{'state' if column == 'observation.state' else 'action'}.{name}"
            nan = float(np.mean(~np.isfinite(x))) if x.size else 0.0
            if nan > cfg["max_nan_frac"]:
                out.append(Finding("error", "finite", f"{label}: {100 * nan:.2f}% NaN/Inf in {pl}"))
            moving = np.median([np.nanstd(e[:, d]) for e in pe]) > 1e-9 and np.median([np.nanstd(e[:, d]) for e in re_]) > 1e-9
            if moving:
                qp, sp, lp = _levels(x)
                qr, _, lr = _levels(y)
                if qp and lp < cfg["max_levels"] and (not qr or lp < cfg["level_ratio"] * lr):
                    m["quantized"][label] = lp
                    out.append(Finding("error", "precision", f"{label}: only ~{lp:.0f} distinct levels across its range in "
                                                             f"{pl} (step {sp:.3g}) vs {'continuous' if not qr else f'~{lr:.0f}'} "
                                                             f"in {rl}: rounded, quantized or stored too coarsely"))
            cp, cr = _pinned(x), _pinned(y)
            if cp > cfg["max_clipped_frac"] and cp > cfg["clip_factor"] * cr + 0.005:
                m["clipped"][label] = cp
                out.append(Finding("error", "clipping", f"{label}: {100 * cp:.1f}% of samples sit exactly at the min or max "
                                                        f"in {pl} ({100 * cr:.1f}% in {rl}): clipped or saturated on the way"))
            if moving and column == "observation.state":
                hp = _hf_share([e[:, d] for e in pe], cfg["hf_cut"])
                hr = _hf_share([e[:, d] for e in re_], cfg["hf_cut"])
                m["hf_share"][label] = (hp, hr)
                if math.isfinite(hp) and math.isfinite(hr) and hr > 0.01 and hp < cfg["min_hf_ratio"] * hr:
                    out.append(Finding("warn", "bandwidth", f"{label}: {100 * hp:.1f}% of its power is above "
                                                            f"{cfg['hf_cut']:g} x the sample rate in {pl} vs {100 * hr:.1f}% in "
                                                            f"{rl}: smoothed, low-pass filtered or interpolated (or the sim lacks the "
                                                            "robot's sensor noise and vibration)"))
    out += _lost_columns(p, r, pl, rl)
    out += _metadata(p, r, pl, rl)
    if not any(f.level != "ok" for f in out):
        out.append(Finding("ok", "fidelity", f"no precision, range, bandwidth, column or metadata loss in {pl}"))
    return out, m


def _lost_columns(p: TraceSet, r: TraceSet, pl: str, rl: str) -> list[Finding]:
    out = []
    pc = set(p.scalar_columns("")) - STANDARD_COLUMNS
    rc = set(r.scalar_columns("")) - STANDARD_COLUMNS
    gone = sorted(rc - pc)
    empty = sorted(c for c in rc & pc if not np.isfinite(p.scalar(c)).any() and np.isfinite(r.scalar(c)).any())
    if "timestamp" in gone:
        out.append(Finding("error", "columns", f"{pl} has no timestamp column: timing cannot be checked or replayed"))
        gone.remove("timestamp")
    if gone:
        out.append(Finding("warn", "columns", f"{rl} records {gone[:8]}{' ...' if len(gone) > 8 else ''}, {pl} does not"))
    if empty:
        out.append(Finding("warn", "columns", f"{pl} has these columns but every value is NaN: {empty[:8]} "
                                              "(e.g. inference latency not recorded)"))
    return out


def _metadata(p: TraceSet, r: TraceSet, pl: str, rl: str) -> list[Finding]:
    def shares(ts: TraceSet) -> dict[str, float]:
        eps = ts.episode_meta() or [{}]
        n = len(eps)
        return {"success label": sum(isinstance(e.get("success"), bool) for e in eps) / n,
                "task string": sum(bool((e.get("tasks") or [""])[0]) for e in eps) / n,
                "scenario": sum(bool(e.get("scenario") or e.get("scenario_name")) for e in eps) / n}

    sp, sr = shares(p), shares(r)
    lost = [f"{k} ({100 * sp[k]:.0f}% vs {100 * sr[k]:.0f}%)" for k in sp if sp[k] + 0.01 < sr[k]]
    return [Finding("warn", "metadata", f"fewer {pl} episodes carry: {', '.join(lost)}")] if lost else []


# -- T3 usability ----------------------------------------------------------------------------------
def usability(p: TraceSet, r: TraceSet, pl: str, rl: str, cfg: dict[str, Any], env: Any = None,
              policy: Any = None) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    m: dict[str, Any] = {}
    if not p.dataset.modality:
        out.append(Finding("error", "modality", f"{pl} has no meta/modality.json: GR00T fine-tuning and the open-loop "
                                                "reader cannot split observation.state into keys"))
    lengths = [len(f) for f in p.frames()]
    short = sum(n < cfg["min_episode_steps"] for n in lengths)
    if short:
        lvl = "error" if short == len(lengths) else "warn"
        out.append(Finding(lvl, "length", f"{short}/{len(lengths)} episodes shorter than {cfg['min_episode_steps']} steps "
                                          "(one action chunk): unusable for chunked training"))
    no_task = sum(not (e.get("tasks") or [""])[0] for e in p.episode_meta())
    if no_task:
        out.append(Finding("warn", "task", f"{no_task} episode(s) without a task string: language-conditioned policies "
                                           "get an empty instruction"))
    A, Ar = p.stacked("action"), r.stacked("action")
    if A.size:
        dead = [i for i in range(A.shape[1]) if np.nanstd(A[:, i]) < 1e-9]
        alive_ref = [i for i in dead if Ar.size and Ar.shape[1] == A.shape[1] and np.nanstd(Ar[:, i]) > 1e-9]
        if alive_ref:
            out.append(Finding("error", "dead_action", f"action dims {alive_ref} never change in {pl} but do in {rl}: "
                                                       "a policy trained on this never commands them"))
        elif dead:
            out.append(Finding("warn", "dead_action", f"action dims {dead} are constant in {pl}"))
    hashes = [hashlib.sha1(np.ascontiguousarray(S).tobytes()).hexdigest() for S in p.per_episode("observation.state")]
    dup = len(hashes) - len(set(hashes))
    if dup:
        out.append(Finding("warn", "duplicates", f"{dup} duplicate episode(s) in {pl}: they inflate sample counts"))
    labels = [e.get("success") for e in p.episode_meta() if isinstance(e.get("success"), bool)]
    if not labels:
        out.append(Finding("warn", "labels", f"{pl} episodes carry no success label: no outcome pairing (G6) or "
                                             "success-filtered training"))
    if env is not None:
        out += _env_usability(env, p, r, pl, rl, m)
    if policy is not None:
        out += _policy_usability(policy, p, r, pl, rl, cfg, m)
    if not any(f.level != "ok" for f in out):
        out.append(Finding("ok", "usability", f"{pl} loads, is fit for training"
                                              + (" and is accepted by the env" if env is not None else "")
                                              + (" and the policy" if policy is not None else "")))
    return out, m


def _env_usability(env: Any, p: TraceSet, r: TraceSet, pl: str, rl: str, m: dict[str, Any]) -> list[Finding]:
    out = []
    obs = env.reset(Scenario(seed=0))
    have = {k.removeprefix("state."): int(np.asarray(v).size) for k, v in obs.items() if k.startswith("state.")}
    for label, ts in ((pl, p), (rl, r)):
        sl = ts.state_slices()
        lost = sorted(k for k in sl if k not in have)
        wrong = sorted(k for k in sl if k in have and have[k] != sl[k].stop - sl[k].start)
        if lost:
            out.append(Finding("error" if label == pl else "warn", "env.keys",
                               f"the env does not produce {lost}, which {label} records: those signals cannot be "
                               "simulated, replayed or compared"))
        if wrong:
            out.append(Finding("error", "env.width", f"{wrong}: the env and {label} disagree on the number of values"))
    dim = getattr(env, "action_dim", None)
    A = p.stacked("action")
    if dim and A.size and int(dim) != A.shape[1]:
        out.append(Finding("error", "env.action", f"env takes {dim} action values, {pl} has {A.shape[1]}"))
    limits_fn = getattr(env, "state_limits", None)
    if callable(limits_fn):
        real = p if p.domain == "real" else r
        rlab = pl if real is p else rl
        S, sl = real.stacked("observation.state"), real.state_slices()
        for key, lim in limits_fn().items():
            if key not in sl or not S.size:
                continue
            lim = np.asarray(lim, dtype=float)
            vals = S[:, sl[key]]
            if lim.shape[0] != vals.shape[1]:
                continue
            tol = 1e-6 + 0.001 * (lim[:, 1] - lim[:, 0])
            frac = float(np.mean(np.any((vals < lim[:, 0] - tol) | (vals > lim[:, 1] + tol), axis=1)))
            m[f"outside_env_limits.{key}"] = frac
            if frac > 0:
                out.append(Finding("error", "env.range", f"{100 * frac:.2f}% of {rlab} steps have state.{key} outside what "
                                                         "the sim can represent: the sim would clip it (calibration offset, "
                                                         "encoder wrap or a wrong zero)"))
    if not hasattr(env, "set_state"):
        out.append(Finding("warn", "env.set_state", "the env has no set_state(): real episodes cannot seed replays"))
    return out


def _policy_usability(policy: Any, p: TraceSet, r: TraceSet, pl: str, rl: str, cfg: dict[str, Any],
                      m: dict[str, Any]) -> list[Finding]:
    from groot_eval.open_loop import evaluate_open_loop

    out = []
    ds = p.dataset
    A = p.stacked("action")
    dim = A.shape[1] if A.size else None
    rng = np.random.default_rng(0)
    bad, wrong, samples = 0, 0, 0
    frames = p.frames()
    try:
        for _ in range(int(cfg["policy_samples"])):
            e = int(rng.integers(len(frames)))
            if not len(frames[e]):
                continue
            t = int(rng.integers(len(frames[e])))
            task = (p.episode_meta()[e].get("tasks") or [""])[0]
            policy.reset(task)
            act = np.asarray(policy.act(ds.observation(frames[e], t, task)), dtype=float).reshape(-1)
            samples += 1
            bad += int(not np.all(np.isfinite(act)))
            wrong += int(dim is not None and act.size != dim)
    except Exception as exc:  # the point is to find out whether it can
        return [Finding("error", "policy", f"the policy cannot consume {pl} observations: {type(exc).__name__}: {exc}")]
    if wrong:
        out.append(Finding("error", "policy.shape", f"{wrong}/{samples} actions have the wrong size for {pl} (expected {dim})"))
    if bad:
        out.append(Finding("error", "policy.finite", f"{bad}/{samples} actions on {pl} observations are NaN/Inf"))
    if wrong or not hasattr(policy, "act"):
        return out
    try:
        n = int(cfg["open_loop_episodes"])
        h = int(cfg["horizon"])
        mp = evaluate_open_loop(p.dataset, policy, horizon=h, episodes=[e["episode_index"] for e in p.episode_meta()[:n]]).summary()
        mr = evaluate_open_loop(r.dataset, policy, horizon=h, episodes=[e["episode_index"] for e in r.episode_meta()[:n]]).summary()
    except Exception as exc:
        return out + [Finding("warn", "policy.open_loop", f"open-loop scoring failed: {type(exc).__name__}: {exc}")]
    m["open_loop_mse"] = {pl: mp["mse_mean"], rl: mr["mse_mean"]}
    ratio = mp["mse_mean"] / mr["mse_mean"] if mr["mse_mean"] > 1e-12 else (1.0 if mp["mse_mean"] < 1e-12 else math.inf)
    m["open_loop_ratio"] = ratio
    lvl = "warn" if ratio > cfg["max_open_loop_ratio"] else "ok"
    out.append(Finding(lvl, "policy.open_loop", f"open-loop action error on {pl} is {ratio:.2f}x that on {rl} "
                                                f"(MSE {mp['mse_mean']:.4g} vs {mr['mse_mean']:.4g}; limit "
                                                f"{cfg['max_open_loop_ratio']}x)"))
    return out


# -- T4 direction-specific -------------------------------------------------------------------------
def _limits_findings(ts: TraceSet, label: str, robot: dict[str, Any], what: str) -> list[Finding]:
    out = []
    S, sl = ts.stacked("observation.state"), ts.state_slices()
    for key, lim in (robot.get("state_limits") or {}).items():
        if key not in sl or not S.size:
            continue
        lim = np.asarray(lim, dtype=float)
        vals = S[:, sl[key]]
        lo, hi = (lim[:, 0], lim[:, 1]) if lim.ndim == 2 else (lim[0], lim[1])
        tol = 1e-6 + 0.001 * np.abs(np.asarray(hi) - np.asarray(lo))
        frac = float(np.mean(np.any((vals < lo - tol) | (vals > hi + tol), axis=1)))
        if frac > 0:
            out.append(Finding("error", "robot.limits", f"{100 * frac:.2f}% of {label} steps have state.{key} outside the "
                                                         f"robot's limits {lim.tolist()}: {what}"))
    A = ts.stacked("action")
    if robot.get("action_range") and A.size:
        lo, hi = robot["action_range"]
        frac = float(np.mean(np.any((A < lo - 1e-6) | (A > hi + 1e-6), axis=1)))
        if frac > 0:
            out.append(Finding("error", "robot.action_range", f"{100 * frac:.2f}% of {label} actions outside [{lo}, {hi}]: "
                                                              "the driver clips them, so commanded and applied differ"))
    if robot.get("max_action_rate") and A.size:
        steps = np.concatenate([np.max(np.abs(np.diff(E, axis=0)), axis=1) for E in ts.per_episode("action") if len(E) > 1])
        frac = float(np.mean(steps > float(robot["max_action_rate"]) + 1e-9)) if steps.size else 0.0
        if frac > 0:
            out.append(Finding("error", "robot.action_rate", f"{100 * frac:.2f}% of {label} steps change a command by more "
                                                             f"than the robot's {robot['max_action_rate']} per step "
                                                             f"(max {steps.max():.3g}): actuator bandwidth exceeded"))
    return out


def _noise(ts: TraceSet) -> dict[str, float]:
    """Per state dim: high-frequency noise estimate (std of the second difference / sqrt(6))."""
    names, eps = _dims(ts, "observation.state")
    out = {}
    for d, n in enumerate(names):
        dd = np.concatenate([np.diff(e[:, d], 2) for e in eps if len(e) > 3] or [np.zeros(0)])
        dd = _finite(dd)
        if dd.size > 20:
            out[n] = float(np.median(np.abs(dd - np.median(dd))) * 1.4826 / math.sqrt(6))
    return out


def direction_checks(sim: TraceSet, real: TraceSet, direction: str, cfg: dict[str, Any], robot: dict[str, Any],
                     wear_profiles: dict[str, float] | None = None) -> tuple[list[Finding], dict[str, Any]]:
    from groot_eval.gap import timing

    out: list[Finding] = []
    m: dict[str, Any] = {}
    rin = {**real.dataset.info, **real.info}
    if direction == "sim2real":
        if robot:
            out += _limits_findings(sim, "sim", robot, "sim data or a sim-trained policy would command what the robot cannot do")
        else:
            out.append(Finding("warn", "robot", "no --robot profile: sim data not checked against the robot's limits"))
        ts, tr = timing(sim), timing(real)
        ls, lr = ts.get("latency_p95_steps"), tr.get("latency_p95_steps")
        m["latency_p95_steps"] = {"sim": ls, "real": lr}
        if lr is not None and lr >= 1.0 and (ls is None or ls < lr - cfg["max_latency_gap_steps"]):
            out.append(Finding("error", "latency", f"the robot acts {lr:.1f} control steps after it observes (p95) but the sim "
                                                   f"{'records no latency' if ls is None else f'only {ls:.1f}'}: a policy "
                                                   "validated in this sim has never seen stale observations; rerun with "
                                                   "`--latency-mode trace:<real trace>`"))
        def jerk(t: TraceSet) -> float:
            d = [np.abs(np.diff(e, axis=0)) for e in t.per_episode("action") if len(e) > 1]
            return float(np.percentile(np.concatenate(d), 95)) if d else math.nan
        js, jr = jerk(sim), jerk(real)
        m["action_change_p95"] = {"sim": js, "real": jr}
        if math.isfinite(js) and math.isfinite(jr) and js > 1e-9 and jr / js > cfg["max_jerk_ratio"]:
            out.append(Finding("warn", "jerk", f"commands change {jr / js:.1f}x more per step on the robot than in sim "
                                               "(p95): latency, noise or chunk boundaries the sim does not reproduce"))
        ns, nr = _noise(sim), _noise(real)
        quiet = [k for k in nr if k in ns and nr[k] > cfg["max_noise_ratio"] * ns[k] + 1e-6]
        m["noise"] = {"sim": ns, "real": nr}
        if quiet:
            out.append(Finding("warn", "noise", f"the robot's readings are noisier than the sim's on {quiet[:6]}: "
                                                "add a sensor-noise model (or randomize it) before trusting sim robustness"))
        cams = robot.get("camera") or {}
        for cam, spec in cams.items():
            h, w = sim.scalar(f"image.{cam}.height"), sim.scalar(f"image.{cam}.width")
            if h.size and np.isfinite(h).any() and (int(np.nanmedian(h)), int(np.nanmedian(w))) != (int(spec.get("height", 0)), int(spec.get("width", 0))):
                out.append(Finding("error", "camera", f"sim renders {cam} at {int(np.nanmedian(h))}x{int(np.nanmedian(w))}, "
                                                      f"the robot's camera is {spec.get('height')}x{spec.get('width')}"))
        age = rin.get("robot_age_hours", robot.get("age_hours"))
        sim_ages = sorted({float(e["scenario"]["params"]["wear.age_hours"]) for e in sim.episode_meta()
                           if isinstance(e.get("scenario"), dict) and "wear.age_hours" in (e["scenario"].get("params") or {})})
        m["wear"] = {"robot_age_hours": age, "sim_age_hours": sim_ages}
        if age is not None:
            age = float(age)
            prof = (min(wear_profiles, key=lambda k: abs(wear_profiles[k] - age)) if wear_profiles else None)
            hint = f" (closest profile `{prof}`: run the sim with `--wear ... --profile {prof}`)" if prof else ""
            if not sim_ages:
                out.append(Finding("warn" if age < 100 else "error", "wear", f"the robot has {age:g} operating hours but the "
                                                                              f"sim data is from a new robot only{hint}"))
            elif age > max(sim_ages) * 1.1 + 1:
                out.append(Finding("error", "wear", f"the robot ({age:g} h) is older than any wear profile in the sim data "
                                                    f"(max {max(sim_ages):g} h){hint}"))
            else:
                out.append(Finding("ok", "wear", f"sim data covers {min(sim_ages):g}-{max(sim_ages):g} h; the robot has {age:g} h"))
        sp, rp = sim.info.get("action_postprocess"), real.info.get("action_postprocess")
        if sp and rp and sp != rp:
            out.append(Finding("error", "postprocess", f"actions are post-processed differently: {sp!r} in sim, {rp!r} on the robot"))
    else:
        if not rin.get("calibration_id"):
            out.append(Finding("warn", "calibration", "the real trace records no calibration_id: nobody can tell which joint "
                                                      "zeros it used, or whether they were applied once or twice"))
        if robot:
            out += _limits_findings(real, "real", robot, "out-of-range readings (calibration offset, encoder wrap)")
        if real.info.get("action_source") != "applied":
            out.append(Finding("warn", "actions", "the real trace does not say its actions are the applied ones "
                                                  "(trace_info action_source: applied); if the driver clips or filters, "
                                                  "replay error will blame the sim for it"))
        if not rin.get("robot_age_hours"):
            out.append(Finding("warn", "wear", "the real trace records no robot_age_hours: the sim cannot be given matching "
                                               "wear; estimate it with `groot-eval wear-fit`"))
        lat = real.scalar("inference_latency_ms")
        if not (lat.size and np.isfinite(lat).any()):
            out.append(Finding("warn", "latency", "no inference_latency_ms in the real trace: the sim cannot replay the "
                                                  "robot's latency (`--latency-mode trace:`)"))
        named = [bool(e.get("scenario_name") or e.get("scenario")) for e in real.episode_meta()]
        if named and not all(named):
            out.append(Finding("warn", "pairing", f"{named.count(False)}/{len(named)} real episodes have no scenario name: "
                                                  "they cannot be paired with sim runs (`correlate`, G6)"))
        s0 = [e[0] for e in sim.per_episode("observation.state") if len(e)]
        r0 = [e[0] for e in real.per_episode("observation.state") if len(e)]
        if s0 and r0 and s0[0].size == r0[0].size and len(s0) >= 3:
            from groot_eval.fidelity import _coverage

            cov = _coverage(np.stack(s0), np.stack(r0), cfg["envelope_margin"])
            m["start_coverage"] = cov["share"]
            out.append(Finding("warn" if cov["share"] < cfg["min_start_coverage"] else "ok", "start_states",
                               f"{100 * cov['share']:.0f}% of real start states fall inside the sim's reset envelope"
                               + (f"; widen the sim's reset randomization on dims {cov['worst_dims']}" if cov["worst_dims"] and cov["share"] < cfg["min_start_coverage"] else "")))
    if not any(f.level != "ok" for f in out):
        out.append(Finding("ok", direction, f"no {direction}-specific problems found"))
    return out, m


# -- runner ----------------------------------------------------------------------------------------
def run_transfer(sim: TraceSet | None, real: TraceSet | None, cfg: dict[str, dict[str, Any]],
                 direction: str = "sim2real", env: Any = None, policy: Any = None, robot: dict[str, Any] | None = None,
                 roundtrip_pair: tuple[TraceSet, TraceSet] | None = None,
                 wear_profiles: dict[str, float] | None = None) -> list[GateResult]:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    results: list[GateResult] = []

    def add(name: str, findings: list[Finding], metrics: dict[str, Any]) -> GateResult:
        res = GateResult(name, _status(findings), bool(cfg[name]["required"]), findings, metrics)
        results.append(res)
        return res

    def skip(name: str, why: str) -> None:
        results.append(GateResult(name, "skip", bool(cfg[name]["required"]), [Finding("warn", "skipped", why)]))

    if roundtrip_pair is not None:
        add("T0_roundtrip", *roundtrip(roundtrip_pair[0], roundtrip_pair[1], cfg["T0_roundtrip"]))
    elif cfg["T0_roundtrip"]["required"]:
        skip("T0_roundtrip", "no --roundtrip ORIGINAL CONVERTED pair given")
    if sim is None or real is None:
        for g in ("T1_compatibility", "T2_fidelity", "T3_usability", "T4_direction"):
            skip(g, "needs both --sim and --real")
        return results
    p, r = (sim, real) if direction == "sim2real" else (real, sim)
    pl, rl = ("sim", "real") if direction == "sim2real" else ("real", "sim")
    t1 = add("T1_compatibility", *compatibility(p, r, pl, rl, cfg["T1_compatibility"], direction))
    if t1.status == "fail":
        skip("T2_fidelity", "compatibility errors: signals do not line up, so comparing them would mislead")
    else:
        add("T2_fidelity", *information(p, r, pl, rl, cfg["T2_fidelity"]))
    add("T3_usability", *usability(p, r, pl, rl, cfg["T3_usability"], env, policy))
    if t1.status == "fail":
        skip("T4_direction", "compatibility errors: fix them first")
    else:
        add("T4_direction", *direction_checks(sim, real, direction, cfg["T4_direction"], robot or {}, wear_profiles))
    return results


def transfer_markdown(results: list[GateResult], sim_root: str | None, real_root: str | None, direction: str) -> str:
    code = exit_code(results)
    verdict = {0: "PASS", 1: "FAIL", 2: "INCOMPLETE (a required gate could not run)"}[code]
    icon = {"pass": "pass", "warn": "pass (warnings)", "fail": "**FAIL**", "skip": "skipped"}
    promoted = "sim" if direction == "sim2real" else "real"
    lines = [f"# Transfer verification ({direction}): {verdict}", "",
             f"Promoted side: **{promoted}** (sim `{sim_root}`, real `{real_root}`). Errors on the promoted data fail a "
             "gate; findings about the reference side are warnings.", "",
             "| Gate | Status | Required | Errors | Warnings | What a pass means |", "|---|---|---|---|---|---|"]
    for res in results:
        ne = sum(f.level == "error" for f in res.findings)
        nw = sum(f.level == "warn" for f in res.findings)
        lines.append(f"| {res.gate} | {icon[res.status]} | {'yes' if res.required else 'no'} | {ne} | {nw} | {WHAT[res.gate]} |")
    for res in results:
        lines += ["", f"## {res.gate}: {icon[res.status]}", ""]
        for f in sorted(res.findings, key=lambda f: {"error": 0, "warn": 1, "ok": 2}[f.level]):
            tag = {"error": "**ERROR**", "warn": "warn", "ok": "ok"}[f.level]
            lines.append(f"- {tag} `{f.check}`: {f.detail}")
    lines += ["", "---", "See `docs/transfer-verification.md` for what each direction can and cannot verify offline."]
    return "\n".join(lines) + "\n"


def to_json(results: list[GateResult], direction: str) -> dict[str, Any]:
    return {"direction": direction, "exit_code": exit_code(results), "gates": [_plain(r.to_dict()) for r in results]}
