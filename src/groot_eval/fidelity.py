"""Data-fidelity gates: is sim and real data good enough to cross the boundary?

Data should go from simulation to the robot (training data, sim-validated
policies) or from the robot to simulation (calibration, digital twins, replay
seeds) only through an ordered set of gates, cheapest first, so nobody tunes
physics for a week to fix a degrees-vs-radians bug:

====  ======================  ===================================================================
G0    integrity               columns present, lengths agree, no NaN/Inf, timestamps increase,
                              files match their SHA-256 manifest, provenance recorded
G1    contract                keys, dims, units, scale, gripper range, action order, image range,
                              control rate (:mod:`groot_eval.checks`)
G2    timing                  control-period jitter, dropped steps, camera-to-state sync offset
G3    physical plausibility   finite-difference velocity vs reported velocity, position / action
                              limits, teleports, derived signals vs the env's own kinematics,
                              success labels vs final task error
G4    distribution            per-signal distance (Wasserstein / pooled std), image statistics, and
                              coverage: the share of real states inside the sim's envelope
G5    dynamics                real actions replayed in sim: tracking error at 10 steps (optionally
                              after system identification)
G6    outcome                 sim and real agree on per-scenario outcomes and checkpoint ranking
====  ======================  ===================================================================

``direction`` names the side being promoted: ``sim2real`` (sim data or policy
goes to the robot) or ``real2sim`` (robot data goes into sim). G0, G2 and G3
run on both sides; a problem on the promoted side fails the gate, one on the
reference side is a warning, because it weakens the comparison but is not the
data under test. G0 and G1 are prerequisites: if either fails, the later gates are skipped,
because they would measure the bug instead of the data. G5 needs ``--env``;
G6 needs a pairs CSV. A required gate
that cannot run makes the command exit 2; a failed gate exits 1.

All thresholds live in a YAML file (``configs/fidelity/gates.yaml``). The
defaults are starting points to tune on your own data, not validated limits.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.checks import check_sources, source_from_traces
from groot_eval.trace import TraceSet
from groot_eval.types import Finding

MANIFEST = "manifest.json"
DIRECTIONS = ("sim2real", "real2sim")
DEFAULTS: dict[str, dict[str, Any]] = {
    "G0_integrity": {"required": True, "provenance": [], "provenance_required": False},
    "G1_contract": {"required": True},
    "G2_timing": {"required": True, "max_jitter_p99_frac": 0.25, "max_drop_frac": 0.005, "max_sync_offset_frac": 0.5},
    "G3_plausibility": {"required": True, "max_velocity_nrmse": 0.5, "velocity_pairs": "auto", "limits": {},
                        "action_range": None, "teleport_factor": 4.0, "max_teleport_frac": 0.001,
                        "max_derived_error": 0.01, "label_check": True},
    "G4_distribution": {"required": True, "max_state_w1": 0.3, "max_action_w1": 0.5, "max_image_w1": 0.5,
                        "min_coverage": 0.95, "envelope_margin": 0.05},
    "G5_dynamics": {"required": False, "horizon": 10, "max_nrmse": 0.1, "episodes": 20},
    "G6_outcome": {"required": False, "min_agreement": 0.85, "min_spearman": 0.8},
}


@dataclass
class GateResult:
    gate: str
    status: str  # pass | fail | warn | skip
    required: bool
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "status": self.status, "required": self.required,
                "findings": [f.to_dict() for f in self.findings], "metrics": _plain(self.metrics)}


def load_gates(path: str | Path | None) -> tuple[dict[str, dict[str, Any]], str | None]:
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    direction = None
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        direction = data.get("direction")
        for k, v in (data.get("gates") or {}).items():
            if k not in cfg:
                raise ValueError(f"unknown gate {k!r}; known: {sorted(cfg)}")
            cfg[k].update(v or {})
    return cfg, direction


def _status(findings: list[Finding]) -> str:
    if any(f.level == "error" for f in findings):
        return "fail"
    if any(f.level == "warn" for f in findings):
        return "warn"
    return "pass"


def _side(findings: list[Finding], promoted: bool, label: str) -> list[Finding]:
    """Prefix with the side; errors on the reference side become warnings."""
    out = []
    for f in findings:
        level = f.level if promoted or f.level != "error" else "warn"
        extra = "" if promoted or f.level != "error" else " (reference side: weakens the comparison)"
        out.append(Finding(level, f"{label}.{f.check}", f.detail + extra))
    return out


# -- G0 integrity ------------------------------------------------------------------------------
def write_manifest(root: str | Path) -> Path:
    """SHA-256 of every data and metadata file, so later readers can prove nothing changed."""
    root = Path(root)
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != MANIFEST:
            files[str(p.relative_to(root)).replace("\\", "/")] = _sha256(p)
    out = root / "meta" / MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"algorithm": "sha256", "files": files}, indent=2), encoding="utf-8")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def integrity(ts: TraceSet, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    n_eps = len(ts.dataset.episodes)
    declared = ts.dataset.info.get("total_episodes")
    if declared is not None and int(declared) != n_eps:
        out.append(Finding("error", "episodes", f"info.json declares {declared} episodes, episodes.jsonl has {n_eps}"))
    bad_len = nan_rows = nonmono = missing = 0
    for meta, frame in zip(ts.episode_meta(), ts.frames()):
        cols = set(frame.columns)
        if not {"observation.state", "action"} <= cols:
            missing += 1
            continue
        if meta.get("length") is not None and int(meta["length"]) != len(frame):
            bad_len += 1
        for col in ("observation.state", "action"):
            arr = np.stack(frame[col].to_numpy()).astype(float) if len(frame) else np.zeros((0, 0))
            nan_rows += int((~np.isfinite(arr)).any(axis=1).sum()) if arr.size else 0
        if "timestamp" in cols and len(frame) > 1 and np.any(np.diff(frame["timestamp"].to_numpy(dtype=float)) <= 0):
            nonmono += 1
    if missing:
        out.append(Finding("error", "columns", f"{missing} episode(s) lack observation.state or action"))
    if bad_len:
        out.append(Finding("error", "length", f"{bad_len} episode(s) have a different row count than episodes.jsonl says"))
    if nan_rows:
        out.append(Finding("error", "finite", f"{nan_rows} step(s) with NaN/Inf in state or action"))
    if nonmono:
        out.append(Finding("error", "timestamp", f"{nonmono} episode(s) with timestamps that do not increase"))
    man = ts.root / "meta" / MANIFEST
    changed: list[str] = []
    if man.exists():
        data = json.loads(man.read_text(encoding="utf-8"))
        for rel, digest in data.get("files", {}).items():
            p = ts.root / rel
            if not p.exists():
                changed.append(f"{rel} (missing)")
            elif _sha256(p) != digest:
                changed.append(rel)
        if changed:
            out.append(Finding("error", "manifest", f"{len(changed)} file(s) differ from the manifest: {changed[:5]}"))
        else:
            out.append(Finding("ok", "manifest", f"{len(data.get('files', {}))} files match the SHA-256 manifest"))
    else:
        out.append(Finding("warn", "manifest", "no meta/manifest.json; run `groot-eval fidelity --write-manifest DIR` "
                                               "when the data is produced, so later changes are detectable"))
    want = list(cfg.get("provenance") or [])
    have = {**ts.info, **ts.dataset.info}
    lacking = [k for k in want if not have.get(k)]
    if lacking:
        level = "error" if cfg.get("provenance_required") else "warn"
        out.append(Finding(level, "provenance", f"trace_info.json / info.json lack {lacking}"))
    if not any(f.level != "ok" for f in out):
        out.append(Finding("ok", "integrity", f"{n_eps} episodes, {sum(len(f) for f in ts.frames())} steps, all finite"))
    return out, {"episodes": n_eps, "nan_steps": nan_rows, "changed_files": len(changed)}


# -- G2 timing -----------------------------------------------------------------------------------
def timing_checks(ts: TraceSet, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    if not ts.fps:
        return [Finding("error", "fps", "no control rate recorded")], {}
    dt = 1.0 / ts.fps
    periods = [np.diff(f["timestamp"].to_numpy(dtype=float)) for f in ts.frames() if "timestamp" in f and len(f) > 1]
    if not periods:
        return [Finding("warn", "timestamp", "no timestamps to check")], {}
    d = np.concatenate(periods)
    dev = np.abs(d - dt)
    p99 = float(np.percentile(dev, 99)) / dt
    drops = float(np.mean(d > 1.5 * dt))
    m = {"dt_s": dt, "jitter_p99_frac": p99, "drop_frac": drops}
    out.append(Finding("error" if p99 > cfg["max_jitter_p99_frac"] else "ok", "jitter",
                       f"p99 deviation from the {1000 * dt:.1f} ms period is {100 * p99:.1f}% of a step "
                       f"(limit {100 * cfg['max_jitter_p99_frac']:.0f}%)"))
    out.append(Finding("error" if drops > cfg["max_drop_frac"] else "ok", "drops",
                       f"{100 * drops:.2f}% of steps are more than 1.5 periods apart (limit {100 * cfg['max_drop_frac']:.2f}%)"))
    for col in ts.scalar_columns("camera_time."):
        cam = col.removeprefix("camera_time.")
        offs = np.concatenate([(f[col].to_numpy(dtype=float) - f["timestamp"].to_numpy(dtype=float))
                               for f in ts.frames() if col in f and "timestamp" in f])
        offs = offs[np.isfinite(offs)]
        if offs.size:
            off = float(np.median(np.abs(offs))) / dt
            m[f"sync_offset_frac.{cam}"] = off
            out.append(Finding("error" if off > cfg["max_sync_offset_frac"] else "ok", "sync",
                               f"camera {cam} is {off:.2f} steps away from the state timestamp (limit {cfg['max_sync_offset_frac']})"))
    return out, m


# -- G3 plausibility --------------------------------------------------------------------------
def _velocity_pairs(slices: dict[str, slice], spec: Any) -> list[tuple[str, str]]:
    if isinstance(spec, dict):
        return [(k, v) for k, v in spec.items() if k in slices and v in slices]
    pairs = []
    for k in slices:
        for a, b in (("_pos", "_vel"), ("position", "velocity"), ("pos", "vel")):
            if k.endswith(a) and k[: -len(a)] + b in slices:
                pairs.append((k, k[: -len(a)] + b))
                break
    return pairs


def plausibility(ts: TraceSet, cfg: dict[str, Any], env: Any = None) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    m: dict[str, Any] = {}
    slices = ts.state_slices()
    eps_s = ts.per_episode("observation.state")
    eps_a = ts.per_episode("action")
    stamps = [f["timestamp"].to_numpy(dtype=float) if "timestamp" in f else None for f in ts.frames()]
    dt = 1.0 / ts.fps if ts.fps else None
    # velocity consistency
    for pk, vk in _velocity_pairs(slices, cfg.get("velocity_pairs", "auto")):
        errs, refs = [], []
        for S, T in zip(eps_s, stamps):
            if len(S) < 3:
                continue
            pos, vel = S[:, slices[pk]], S[:, slices[vk]]
            step = np.diff(T)[:, None] if T is not None else (dt or 1.0)
            fd = np.diff(pos, axis=0) / step
            # a robot may report the velocity at either end of the interval; take the better alignment
            e1, e2 = fd - vel[1:], fd - vel[:-1]
            errs.append(e1 if np.nanmean(e1 ** 2) <= np.nanmean(e2 ** 2) else e2)
            refs.append(vel)
        if not errs:
            continue
        e, r = np.concatenate(errs), np.concatenate(refs)
        scale = np.nanstd(r, axis=0)
        live = scale > 1e-9
        if not live.any():
            continue
        nrmse = float(np.sqrt(np.nanmean((e[:, live] / scale[live]) ** 2)))
        m[f"velocity_nrmse.{pk}"] = nrmse
        out.append(Finding("error" if nrmse > cfg["max_velocity_nrmse"] else "ok", "velocity",
                           f"d({pk})/dt vs {vk}: nRMSE {nrmse:.3f} (limit {cfg['max_velocity_nrmse']}); a mismatch means "
                           "wrong units, a stale velocity filter, timestamps that are not the sample times, or noisy positions"))
    # limits
    allS = np.concatenate(eps_s) if eps_s else np.zeros((0, 0))
    for key, lim in (cfg.get("limits") or {}).items():
        if key not in slices or not allS.size:
            continue
        vals = allS[:, slices[key]]
        lim = np.asarray(lim, dtype=float)
        lo, hi = (lim[:, 0], lim[:, 1]) if lim.ndim == 2 else (lim[0], lim[1])
        tol = 1e-6 + 0.001 * np.abs(np.asarray(hi) - np.asarray(lo))
        viol = float(np.mean(np.any((vals < lo - tol) | (vals > hi + tol), axis=1)))
        m[f"limit_violation_frac.{key}"] = viol
        out.append(Finding("error" if viol > 0 else "ok", "limits",
                           f"{key}: {100 * viol:.2f}% of steps outside the limits {lim.tolist()}"))
    if cfg.get("action_range") and eps_a:
        lo, hi = cfg["action_range"]
        A = np.concatenate(eps_a)
        viol = float(np.mean(np.any((A < lo - 1e-6) | (A > hi + 1e-6), axis=1)))
        m["action_range_violation_frac"] = viol
        out.append(Finding("error" if viol > 0 else "ok", "action_range",
                           f"{100 * viol:.2f}% of actions outside [{lo}, {hi}] (commanded vs applied clipping differs?)"))
    # teleports: position-like signals only (torques, commands and gripper states may legitimately jump)
    pos_keys = {pk for pk, _ in _velocity_pairs(slices, cfg.get("velocity_pairs", "auto"))}
    pos_keys |= {k for k in slices if any(t in k for t in ("pos", "position", "pose"))}
    cols = [i for k in sorted(pos_keys) for i in range(slices[k].start, slices[k].stop)]
    jumps, total = 0, 0
    for S in eps_s if cols else []:
        if len(S) < 3:
            continue
        d = np.abs(np.diff(S[:, cols], axis=0))
        ref = np.percentile(d, 99, axis=0) + 1e-9
        jumps += int(np.any(d > cfg["teleport_factor"] * np.maximum(ref, np.median(d, axis=0) * 50 + 1e-9), axis=1).sum())
        total += len(d)
    if total:
        frac = jumps / total
        m["teleport_frac"] = frac
        out.append(Finding("error" if frac > cfg["max_teleport_frac"] else ("warn" if jumps else "ok"), "teleport",
                           f"{jumps} step(s) where a state jumps far beyond its usual step size ({100 * frac:.3f}%)"))
    # derived signals vs the env's kinematics
    derive = getattr(env, "derive_obs", None) if env is not None else None
    if callable(derive) and eps_s:
        worst = 0.0
        compared: set[str] = set()
        for S in eps_s[:20]:
            for row in S[:: max(1, len(S) // 50)]:
                obs = {f"state.{k}": row[s].astype(np.float32) for k, s in slices.items()}
                got = derive(dict(obs))
                for k, v in got.items():
                    if k in obs and k.startswith("state."):
                        diff = float(np.max(np.abs(np.asarray(v, dtype=float) - np.asarray(obs[k], dtype=float))))
                        if diff > 0:
                            compared.add(k)
                        worst = max(worst, diff)
        m["derived_max_error"] = worst
        out.append(Finding("error" if worst > cfg["max_derived_error"] else "ok", "derived",
                           f"recorded derived signals vs the env's own computation from the recorded joints: max error "
                           f"{worst:.4g} (limit {cfg['max_derived_error']}){'; ' + ', '.join(sorted(compared)) if compared else ''}"))
    # success labels vs final task error
    if cfg.get("label_check"):
        finals, labels = [], []
        for meta, f in zip(ts.episode_meta(), ts.frames()):
            if isinstance(meta.get("success"), bool) and "reward" in f and len(f):
                r = f["reward"].to_numpy(dtype=float)
                if np.isfinite(r[-1]):
                    finals.append(r[-1])
                    labels.append(meta["success"])
        if finals and any(labels) and not all(labels):
            finals_a, labels_a = np.array(finals), np.array(labels)
            fail_med = float(np.median(finals_a[~labels_a]))
            succ_med = float(np.median(finals_a[labels_a]))
            if succ_med > fail_med:
                odd = int(np.sum(labels_a & (finals_a < fail_med)) + np.sum(~labels_a & (finals_a > succ_med)))
                m["suspicious_labels"] = odd
                out.append(Finding("warn" if odd else "ok", "labels",
                                   f"{odd} episode(s) whose success label disagrees with their final reward "
                                   "(a success that ends worse than the typical failure, or the reverse)"))
            else:
                out.append(Finding("warn", "labels", "successful episodes end with a lower reward than failed ones: "
                                                     "labels inverted, or reward is not task progress"))
    if not out:
        out.append(Finding("warn", "plausibility", "nothing to check: no position/velocity pairs, limits or env given"))
    return out, m


# -- G4 distribution --------------------------------------------------------------------------
def distribution(sim: TraceSet, real: TraceSet, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    from groot_eval.gap import gap_analysis

    g = gap_analysis(sim, real, ("sim", "real"))
    out: list[Finding] = []
    m: dict[str, Any] = {}
    for group, key in (("state", "max_state_w1"), ("action", "max_action_w1"), ("image", "max_image_w1")):
        rows = [r for r in g["signals"] if r["group"] == group]
        if not rows:
            continue
        worst = max(rows, key=lambda r: r["w1_norm"] if math.isfinite(r["w1_norm"]) else 1e9)
        m[f"{group}_w1_max"] = worst["w1_norm"]
        m[f"{group}_w1_worst_signal"] = worst["signal"]
        over = [r["signal"] for r in rows if not (r["w1_norm"] <= cfg[key])]
        out.append(Finding("error" if over else "ok", f"{group}.distance",
                           f"largest {group} gap {worst['w1_norm']:.3g} pooled std ({worst['signal']}; limit {cfg[key]})"
                           + (f"; over the limit: {over[:6]}" if over else "")))
    S, R = sim.stacked("observation.state"), real.stacked("observation.state")
    if S.size and R.size and S.shape[1] == R.shape[1]:
        cov, rev = _coverage(S, R, cfg["envelope_margin"]), _coverage(R, S, cfg["envelope_margin"])
        m.update({"coverage_real_in_sim": cov["share"], "coverage_sim_in_real": rev["share"]})
        detail = f"{100 * cov['share']:.1f}% of real states fall inside the sim's envelope (limit {100 * cfg['min_coverage']:.0f}%)"
        if cov["worst_dims"]:
            detail += f"; most often outside on dims {cov['worst_dims']}"
        out.append(Finding("error" if cov["share"] < cfg["min_coverage"] else "ok", "coverage", detail))
        out.append(Finding("ok", "coverage.reverse", f"{100 * rev['share']:.1f}% of sim states fall inside the real envelope "
                                                      "(low = sim explores states the robot never visits; informational)"))
    return out, m


def _coverage(ref: np.ndarray, test: np.ndarray, margin: float) -> dict[str, Any]:
    lo, hi = np.nanpercentile(ref, 0.5, axis=0), np.nanpercentile(ref, 99.5, axis=0)
    pad = margin * np.maximum(hi - lo, 1e-9)
    inside = (test >= lo - pad) & (test <= hi + pad)
    share = float(np.mean(np.all(inside, axis=1)))
    miss = 1 - inside.mean(axis=0)
    worst = [int(i) for i in np.argsort(-miss)[:3] if miss[i] > 0.01]
    return {"share": share, "worst_dims": worst}


# -- G5 dynamics -------------------------------------------------------------------------------
def dynamics(real: TraceSet, env: Any, cfg: dict[str, Any], params: dict[str, Any] | None = None,
             sysid: Any = None) -> tuple[list[Finding], dict[str, Any]]:
    from groot_eval.replay import replay_trace, system_id

    h = int(cfg["horizon"])
    n = int(cfg.get("episodes") or 0) or None
    out: list[Finding] = []
    if sysid is not None:
        fit = system_id(env, real, sysid, params, episodes=n)
        best = next((r for r in fit["ranked"] if math.isfinite(r.score)), fit["baseline"])
        for note in fit["notes"]:
            if "edge" in note or "not pinned" in note:
                out.append(Finding("warn", "sysid", note))
        score, label = best, f"best fit {best.params}"
    else:
        score, label = replay_trace(env, real, params, episodes=n), "given parameters"
    err = score.horizons.get(h, math.nan)
    m = {"nrmse_at_horizon": err, "horizon": h, "score": score.score, "params": score.params, "synced": score.synced}
    if not math.isfinite(err):
        out.append(Finding("warn", "replay", f"no replayed step reached {h} steps after sync"))
    else:
        out.append(Finding("error" if err > cfg["max_nrmse"] else "ok", "replay",
                           f"real actions replayed in sim ({label}): normalized state error {err:.3f} after {h} steps "
                           f"(limit {cfg['max_nrmse']})"))
    if not score.synced:
        out.append(Finding("warn", "replay", "env has no set_state(); replays did not start from the recorded state"))
    return out, m


# -- G6 outcome --------------------------------------------------------------------------------
def outcome(pairs_csv: str | Path, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    from groot_eval.correlation import correlate, load_pairs

    c = correlate(load_pairs(pairs_csv))
    out: list[Finding] = []
    agree, rho = c["outcome_agreement"], c["ranking_spearman"]
    m = {"outcome_agreement": agree, "ranking_spearman": rho, "matched_pairs": c["matched_pairs"]}
    if agree is None:
        out.append(Finding("error", "agreement", "no sim/real pairs matched; see `groot-eval correlate`"))
    else:
        out.append(Finding("error" if agree < cfg["min_agreement"] else "ok", "agreement",
                           f"sim and real agree on {100 * agree:.1f}% of {c['matched_pairs']} matched scenarios "
                           f"(limit {100 * cfg['min_agreement']:.0f}%)"))
    if rho is None or not math.isfinite(rho):
        out.append(Finding("warn", "ranking", "ranking agreement needs at least two checkpoints run on both sides"))
    else:
        out.append(Finding("error" if rho < cfg["min_spearman"] else "ok", "ranking",
                           f"checkpoint ranking Spearman {rho:.2f} (limit {cfg['min_spearman']})"))
    for w in c["join"]["warnings"]:
        out.append(Finding("warn", "join", w))
    return out, m


# -- runner ------------------------------------------------------------------------------------
def run_gates(sim: TraceSet, real: TraceSet, cfg: dict[str, dict[str, Any]], direction: str = "sim2real",
              env: Any = None, params: dict[str, Any] | None = None, sysid: Any = None,
              pairs: str | Path | None = None) -> list[GateResult]:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    promoted = "sim" if direction == "sim2real" else "real"
    sides = (("sim", sim), ("real", real))
    results: list[GateResult] = []
    stop = False

    def add(name: str, findings: list[Finding], metrics: dict[str, Any]) -> None:
        nonlocal stop
        req = bool(cfg[name]["required"])
        r = GateResult(name, _status(findings), req, findings, metrics)
        results.append(r)
        if r.status == "fail":
            stop = True

    def skip(name: str, why: str) -> None:
        results.append(GateResult(name, "skip", bool(cfg[name]["required"]), [Finding("warn", "skipped", why)]))

    f0, m0 = [], {}
    for label, ts in sides:
        f, m = integrity(ts, cfg["G0_integrity"])
        f0 += _side(f, label == promoted or any(x.check in ("finite", "columns", "manifest") for x in f if x.level == "error"), label)
        m0[label] = m
    add("G0_integrity", f0, m0)
    if stop:
        for g in list(DEFAULTS)[1:]:
            skip(g, "an earlier gate failed; fix it first (later gates would measure the bug, not the data)")
        return results
    add("G1_contract", check_sources(source_from_traces(sim, "sim"), source_from_traces(real, "real")), {})
    f2, m2, f3, m3 = [], {}, [], {}
    for label, ts in sides:
        f, m = timing_checks(ts, cfg["G2_timing"])
        f2 += _side(f, label == promoted, label)
        m2[label] = m
        f, m = plausibility(ts, cfg["G3_plausibility"], env)
        f3 += _side(f, label == promoted, label)
        m3[label] = m
    if stop:
        for g in list(DEFAULTS)[2:]:
            skip(g, "contract errors: units, keys or rates differ, so every later number would be wrong")
        return results
    add("G2_timing", f2, m2)
    add("G3_plausibility", f3, m3)
    add("G4_distribution", *distribution(sim, real, cfg["G4_distribution"]))
    if env is None:
        skip("G5_dynamics", "no --env given to replay the real actions in")
    else:
        add("G5_dynamics", *dynamics(real, env, cfg["G5_dynamics"], params, sysid))
    if pairs is None:
        skip("G6_outcome", "no --pairs CSV of paired sim/real outcomes")
    else:
        add("G6_outcome", *outcome(pairs, cfg["G6_outcome"]))
    return results


def exit_code(results: list[GateResult]) -> int:
    if any(r.status == "fail" for r in results):
        return 1
    if any(r.status == "skip" and r.required for r in results):
        return 2
    return 0


def fidelity_markdown(results: list[GateResult], sim_root: str, real_root: str, direction: str) -> str:
    code = exit_code(results)
    verdict = {0: "PASS", 1: "FAIL", 2: "INCOMPLETE (a required gate could not run)"}[code]
    icon = {"pass": "pass", "warn": "pass (warnings)", "fail": "**FAIL**", "skip": "skipped"}
    lines = [f"# Data-fidelity gates: {verdict}", "",
             f"Direction `{direction}`: sim `{sim_root}`, real `{real_root}`. Gates run in order; after a failure the "
             "later gates are skipped, because they would measure the bug instead of the data.", "",
             "| Gate | Status | Required | Key numbers |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.gate} | {icon[r.status]} | {'yes' if r.required else 'no'} | {_key_numbers(r)} |")
    for r in results:
        bad = [f for f in r.findings if f.level != "ok"]
        good = [f for f in r.findings if f.level == "ok"]
        lines += ["", f"## {r.gate}: {icon[r.status]}", ""]
        for f in bad + good:
            tag = {"error": "**ERROR**", "warn": "warn", "ok": "ok"}[f.level]
            lines.append(f"- {tag} `{f.check}`: {f.detail}")
    return "\n".join(lines) + "\n"


def _key_numbers(r: GateResult) -> str:
    m = r.metrics
    if r.gate == "G2_timing":
        return "; ".join(f"{k}: jitter p99 {100 * v.get('jitter_p99_frac', math.nan):.1f}% of a step"
                         for k, v in m.items() if isinstance(v, dict) and v)
    if r.gate == "G4_distribution":
        parts = [f"state gap {m['state_w1_max']:.2f} σ" if "state_w1_max" in m else "",
                 f"coverage {100 * m['coverage_real_in_sim']:.1f}%" if "coverage_real_in_sim" in m else ""]
        return "; ".join(p for p in parts if p)
    if r.gate == "G5_dynamics" and "nrmse_at_horizon" in m:
        return f"nRMSE@{m['horizon']} {m['nrmse_at_horizon']:.3f}"
    if r.gate == "G6_outcome" and m.get("outcome_agreement") is not None:
        return f"agreement {100 * m['outcome_agreement']:.0f}%, Spearman {m['ranking_spearman']:.2f}"
    n_err = sum(f.level == "error" for f in r.findings)
    n_warn = sum(f.level == "warn" for f in r.findings)
    return f"{n_err} errors, {n_warn} warnings"


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
