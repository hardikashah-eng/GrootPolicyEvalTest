"""Contract checks: do two data sources describe the robot the same way?

Many sim-to-real "gaps" are not physics at all. They are a state in degrees on
one side and radians on the other, a camera frame in [0, 1] floats that turns
black when cast to uint8, a gripper commanded 0-1 in sim and 0-100 on the
robot, action groups concatenated in a different order, or a dataset recorded
at 30 Hz driving a controller stepped at 20 Hz. Those are cheap to detect and
expensive to find by watching the robot, so check them first.

A :class:`SignalSource` is built from a trace / LeRobot dataset or by rolling
out an environment, and :func:`check_sources` compares two of them. The checks
are heuristics: an ``error`` is very likely a bug, a ``warn`` is worth a look.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from groot_eval.trace import TraceSet, dim_names
from groot_eval.types import Finding, Scenario

DEG_PER_RAD = 180.0 / math.pi  # 57.3


@dataclass
class SignalSource:
    name: str
    fps: float | None
    state: dict[str, np.ndarray]  # key (without "state.") -> (N, d)
    action: np.ndarray  # (N, A)
    action_groups: dict[str, slice] = field(default_factory=dict)
    video: dict[str, dict[str, Any]] = field(default_factory=dict)  # key -> dtype, shape, max_value


def source_from_traces(ts: TraceSet, name: str | None = None) -> SignalSource:
    states = ts.stacked("observation.state")
    state = {k: states[:, sl] for k, sl in ts.state_slices().items()} if states.size else {}
    video: dict[str, dict[str, Any]] = {}
    features = ts.dataset.info.get("features", {})
    for key, spec in (ts.dataset.modality.get("video") or {}).items():
        original = (spec or {}).get("original_key", f"observation.images.{key}")
        feat = features.get(original, {})
        entry: dict[str, Any] = {"dtype": feat.get("dtype"), "shape": tuple(feat["shape"]) if feat.get("shape") else None, "max_value": None}
        mx = ts.scalar(f"image.{key}.max_value")
        if mx.size and np.isfinite(mx).any():
            entry["max_value"] = float(np.nanmax(mx))
            h, w = ts.scalar(f"image.{key}.height"), ts.scalar(f"image.{key}.width")
            if entry["shape"] is None and np.isfinite(h).any():
                entry["shape"] = (int(np.nanmedian(h)), int(np.nanmedian(w)))
        video[key] = entry
    return SignalSource(name or f"{ts.domain}:{ts.root.name}", ts.fps, state, ts.stacked("action"), ts.action_slices(), video)


def source_from_env(env: Any, policy: Any = None, scenarios: list[Scenario] | None = None,
                    action_dim: int | None = None, name: str = "env") -> SignalSource:
    """Roll out ``env`` and collect what it emits. Without a policy, zero actions of ``action_dim`` are sent."""
    if policy is None and action_dim is None:
        raise ValueError("need a policy or an action_dim to step the env")
    scenarios = scenarios or [Scenario(seed=i) for i in range(3)]
    states: dict[str, list[np.ndarray]] = {}
    actions: list[np.ndarray] = []
    video: dict[str, dict[str, Any]] = {}
    for sc in scenarios:
        obs = env.reset(sc)
        if policy is not None:
            policy.reset(sc.task)
        for _ in range(env.max_steps):
            for k, v in obs.items():
                if k.startswith("state."):
                    states.setdefault(k.removeprefix("state."), []).append(np.asarray(v, dtype=np.float64).reshape(-1))
                elif k.startswith("video."):
                    arr = np.asarray(v)
                    shape = arr.shape[1:] if arr.ndim == 4 else arr.shape
                    e = video.setdefault(k.removeprefix("video."), {"dtype": str(arr.dtype), "shape": tuple(shape), "max_value": -math.inf})
                    e["max_value"] = max(e["max_value"], float(np.max(arr)) if arr.size else -math.inf)
            act = np.asarray(policy.act(obs) if policy is not None else np.zeros(action_dim), dtype=np.float64).reshape(-1)
            actions.append(act)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    dt = getattr(env, "dt", None)
    act_arr = np.stack(actions) if actions else np.zeros((0, action_dim or 0))
    return SignalSource(name, 1.0 / dt if dt else None, {k: np.stack(v) for k, v in states.items()},
                        act_arr, {"action": slice(0, act_arr.shape[1])}, video)


# -- checks ---------------------------------------------------------------------
def check_sources(a: SignalSource, b: SignalSource) -> list[Finding]:
    """Compare ``a`` against reference ``b``."""
    out: list[Finding] = []
    out += _check_keys(a, b)
    common = [k for k in b.state if k in a.state]
    widths_ok = []
    for key in common:
        da, db = a.state[key].shape[1], b.state[key].shape[1]
        if da != db:
            out.append(Finding("error", "state.shape", f"state.{key}: {da} dims in {a.name}, {db} in {b.name}"))
        else:
            widths_ok.append(key)
    for key in widths_ok:
        names = dim_names({key: slice(0, a.state[key].shape[1])})
        out += _compare_dims("state", names, a.state[key], b.state[key], a.name, b.name, hint=key)
    out += _check_actions(a, b)
    out += _check_fps(a, b)
    out += _check_video(a, b)
    if not any(f.level != "ok" for f in out):
        out.append(Finding("ok", "summary", f"no contract mismatches between {a.name} and {b.name}"))
    return out


def _check_keys(a: SignalSource, b: SignalSource) -> list[Finding]:
    out = []
    only_a, only_b = sorted(set(a.state) - set(b.state)), sorted(set(b.state) - set(a.state))
    if only_b:
        out.append(Finding("error", "state.keys", f"{a.name} lacks state keys present in {b.name}: {only_b}"))
    if only_a:
        out.append(Finding("warn", "state.keys", f"{a.name} has state keys {b.name} lacks: {only_a}"))
    if not only_a and not only_b:
        out.append(Finding("ok", "state.keys", f"same {len(a.state)} state keys"))
    return out


def _check_actions(a: SignalSource, b: SignalSource) -> list[Finding]:
    out = []
    if a.action.size == 0 or b.action.size == 0:
        return [Finding("warn", "action", "no actions recorded on one side; action checks skipped")]
    da, db = a.action.shape[1], b.action.shape[1]
    if da != db:
        return [Finding("error", "action.shape", f"action has {da} dims in {a.name}, {db} in {b.name}")]
    ga, gb = list(a.action_groups), list(b.action_groups)
    generic = ga == ["action"] or gb == ["action"]
    if not generic:
        if set(ga) == set(gb) and ga != gb:
            out.append(Finding("error", "action.order", f"action groups in a different order: {ga} in {a.name}, {gb} in {b.name}"))
        elif set(ga) != set(gb):
            out.append(Finding("warn", "action.groups", f"action group names differ: {ga} vs {gb}"))
        elif any(a.action_groups[g] != b.action_groups[g] for g in ga):
            out.append(Finding("error", "action.slices", "action groups have the same names but different index ranges"))
    slices = b.action_groups if gb != ["action"] else a.action_groups
    names = dim_names(slices) if sum(s.stop - s.start for s in slices.values()) == db else [f"action[{i}]" for i in range(db)]
    for i, n in enumerate(names):
        hint = "gripper" if "gripper" in n.lower() else None
        out += _compare_dims("action", [n], a.action[:, i : i + 1], b.action[:, i : i + 1], a.name, b.name, hint=hint)
    return out


def _check_fps(a: SignalSource, b: SignalSource) -> list[Finding]:
    if not a.fps or not b.fps:
        return [Finding("warn", "fps", f"control rate unknown for {a.name if not a.fps else b.name}")]
    if abs(a.fps - b.fps) / b.fps > 0.02:
        return [Finding("error", "fps", f"control rate {a.fps:g} Hz in {a.name} vs {b.fps:g} Hz in {b.name}: "
                                        "action chunks cover different durations, velocities and delays scale differently")]
    return [Finding("ok", "fps", f"both at {b.fps:g} Hz")]


def _check_video(a: SignalSource, b: SignalSource) -> list[Finding]:
    out = []
    if not a.video and not b.video:
        return out
    missing = sorted(set(b.video) - set(a.video))
    if missing:
        out.append(Finding("error", "video.keys", f"{a.name} has no camera stream for {missing}"))
    for key in sorted(set(a.video) & set(b.video)):
        va, vb = a.video[key], b.video[key]
        ma, mb = va.get("max_value"), vb.get("max_value")
        if ma is not None and mb is not None and ma > -math.inf and mb > -math.inf and (ma <= 1.0) != (mb <= 1.0):
            out.append(Finding("error", "video.range", f"video.{key}: pixel max {ma:.3g} in {a.name} vs {mb:.3g} in {b.name}; "
                                                      "one side is a [0, 1] float image, the other 0-255"))
        sa, sb = va.get("shape"), vb.get("shape")
        if sa and sb and tuple(sa)[:2] != tuple(sb)[:2]:
            out.append(Finding("warn", "video.shape", f"video.{key}: {tuple(sa)} in {a.name} vs {tuple(sb)} in {b.name}; "
                                                     "resizing changes field of view and apparent object size unless the aspect ratio and crop match"))
        if not any(f.check.startswith("video.") and key in f.detail for f in out):
            out.append(Finding("ok", "video", f"video.{key}: range and resolution consistent"))
    return out


def _compare_dims(kind: str, names: list[str], A: np.ndarray, B: np.ndarray, a_name: str, b_name: str,
                  hint: str | None = None) -> list[Finding]:
    out = []
    for j, name in enumerate(names):
        xa, xb = A[:, j], B[:, j]
        xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
        if not xa.size or not xb.size:
            continue
        sa, sb = _spread(xa), _spread(xb)
        const_a, const_b = sa < 1e-9, sb < 1e-9
        label = f"{kind}.{name}"
        if const_a and const_b:
            if abs(np.median(xa) - np.median(xb)) > 1e-6 * max(1.0, abs(np.median(xb))):
                out.append(Finding("warn", f"{kind}.offset", f"{label}: constant {np.median(xa):.4g} in {a_name} vs {np.median(xb):.4g} in {b_name}"))
            continue
        if const_a != const_b:
            which, other = (a_name, b_name) if const_a else (b_name, a_name)
            out.append(Finding("warn", f"{kind}.constant", f"{label} never changes in {which} but varies in {other}: "
                                                           "unconnected sensor, zeroed field or frozen joint?"))
            continue
        r = sa / sb
        extra = " (gripper scaling: 0-1 vs 0-100 or 0-255?)" if hint == "gripper" else ""
        if 45 <= r <= 70 or 45 <= 1 / r <= 70:
            out.append(Finding("error", f"{kind}.units", f"{label}: spread is {r:.1f}x larger in {a_name} than in {b_name}; "
                                                        f"looks like degrees vs radians (x{DEG_PER_RAD:.1f}){extra}"))
            continue
        if r > 10 or r < 0.1:
            out.append(Finding("warn", f"{kind}.scale", f"{label}: spread is {r:.2g}x that of {b_name}: units, normalization "
                                                        f"or scaling differ{extra}"))
            continue
        lo, hi = np.percentile(xb, [1, 99])
        margin = 0.5 * (hi - lo)
        med = float(np.median(xa))
        if med < lo - margin or med > hi + margin:
            out.append(Finding("warn", f"{kind}.offset", f"{label}: typical value {med:.4g} in {a_name} lies outside "
                                                         f"{b_name}'s range [{lo:.4g}, {hi:.4g}]: offset, calibration or a different zero pose"))
    return out


def _spread(x: np.ndarray) -> float:
    p5, p95 = np.percentile(x, [5, 95])
    return float(p95 - p5) if p95 > p5 else float(np.std(x))


def check_policy(policy: Any, reference: SignalSource) -> list[Finding]:
    """Adapter settings that silently scramble actions (GR00T server adapter only)."""
    inner = getattr(policy, "predictor", policy)
    if not (hasattr(inner, "action_keys") and hasattr(inner, "client")):
        return []
    keys = inner.action_keys
    if not keys:
        return [Finding("warn", "policy.action_keys", "GR00T adapter has no action_keys: reply keys are concatenated "
                                                      f"alphabetically; reference order is {list(reference.action_groups)}")]
    names = [k.removeprefix("action.") for k in keys]
    ref = list(reference.action_groups)
    if ref != ["action"] and set(names) == set(ref) and names != ref:
        return [Finding("error", "policy.action_keys", f"action_keys order {names} differs from reference order {ref}")]
    if ref != ["action"] and set(names) != set(ref):
        return [Finding("warn", "policy.action_keys", f"action_keys {names} do not match reference groups {ref}")]
    return [Finding("ok", "policy.action_keys", f"action_keys match reference order {ref}")]


def findings_markdown(findings: list[Finding], title: str) -> str:
    icon = {"error": "**ERROR**", "warn": "warn", "ok": "ok"}
    order = {"error": 0, "warn": 1, "ok": 2}
    n_err = sum(f.level == "error" for f in findings)
    n_warn = sum(f.level == "warn" for f in findings)
    lines = [f"# {title}", "", f"{n_err} errors, {n_warn} warnings.", "", "| Level | Check | Detail |", "|---|---|---|"]
    for f in sorted(findings, key=lambda f: order.get(f.level, 3)):
        lines.append(f"| {icon.get(f.level, f.level)} | {f.check} | {f.detail.replace('|', '/')} |")
    return "\n".join(lines) + "\n"
