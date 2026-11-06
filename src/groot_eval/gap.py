"""Where is the sim-to-real gap? Signal-by-signal comparison of two traces.

Given a simulation trace and a real-robot trace (see :mod:`groot_eval.trace`),
this compares every recorded signal and ranks them by how far apart the two
distributions are:

* each **state** and **action** dimension, plus per-step **action changes**
  (how jerky the commands are);
* **image statistics** per camera (brightness, contrast, sharpness);
* **timing**: control period and its jitter, policy inference latency, and
  that latency expressed in control steps (sim usually waits for the policy,
  a robot does not);
* **action-chunk boundaries**: how much the command jumps when a new chunk
  starts, compared with steps inside a chunk;
* **outcomes**, when episodes carry ``success``;
* **contract mismatches** (units, ranges, rates, key order) from
  :mod:`groot_eval.checks`, which should be fixed before reading anything else.

The ranking statistic is the 1-D Wasserstein distance divided by the pooled
standard deviation ("how many standard deviations apart"), which is
comparable across signals with different units. The Kolmogorov-Smirnov p-value
uses the number of *episodes*, not steps, as the sample size, because steps in
one episode are strongly correlated; that makes it conservative.

This is a comparison of distributions, not of matched trajectories: if the
sim and real runs start from different states or the policy did different
things, some differences are behaviour, not physics. Paired replay (feeding
real actions to the sim) is the next step for separating the two.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from groot_eval.checks import check_sources, source_from_traces
from groot_eval.stats import wilson_interval
from groot_eval.trace import TraceSet, dim_names

RANKED_IMAGE_STATS = ("brightness", "contrast", "sharpness")


# -- statistics -------------------------------------------------------------------
def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1-D earth mover's distance, via quantile functions on a fixed grid."""
    u = np.linspace(0.0025, 0.9975, 400)
    return float(np.mean(np.abs(np.quantile(a, u) - np.quantile(b, u))))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_pvalue(d: float, n: int, m: int) -> float:
    """Asymptotic two-sample Kolmogorov-Smirnov p-value (Stephens' correction)."""
    if n < 1 or m < 1:
        return math.nan
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.2:
        return 1.0
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(min(1.0, max(0.0, p)))


def _summary(x: np.ndarray) -> dict[str, float]:
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    return {"mean": float(np.mean(x)), "std": float(np.std(x)), "p5": float(p5), "p50": float(p50), "p95": float(p95), "n": int(x.size)}


def compare_signal(group: str, name: str, a: np.ndarray, b: np.ndarray, eps_a: int, eps_b: int) -> dict[str, Any] | None:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return None
    w1 = wasserstein1(a, b)
    pooled = math.sqrt((np.var(a) + np.var(b)) / 2)
    w1_norm = w1 / pooled if pooled > 1e-12 else (0.0 if w1 < 1e-12 else math.inf)
    d = ks_statistic(a, b)
    sa, sb = _summary(a), _summary(b)
    return {
        "group": group,
        "signal": name,
        "a": sa,
        "b": sb,
        "mean_shift": sa["mean"] - sb["mean"],
        "std_ratio": sa["std"] / sb["std"] if sb["std"] > 1e-12 else (1.0 if sa["std"] < 1e-12 else math.inf),
        "w1": w1,
        "w1_norm": w1_norm,
        "ks": d,
        "ks_p_episodes": ks_pvalue(d, eps_a, eps_b),
    }


# -- signal extraction --------------------------------------------------------------
def signal_samples(ts: TraceSet) -> dict[tuple[str, str], np.ndarray]:
    """Every rankable signal as a flat sample array, keyed by (group, name)."""
    out: dict[tuple[str, str], np.ndarray] = {}
    states = ts.stacked("observation.state")
    if states.size:
        slices = ts.state_slices()
        names = dim_names(slices)
        cols = [i for sl in slices.values() for i in range(sl.start, sl.stop)]
        for name, i in zip(names, cols):
            out[("state", name)] = states[:, i]
    actions = ts.stacked("action")
    if actions.size:
        slices = ts.action_slices()
        names = dim_names(slices)
        if len(names) != actions.shape[1]:
            names = [f"action[{i}]" for i in range(actions.shape[1])]
        for i, name in enumerate(names):
            out[("action", name)] = actions[:, i]
        deltas = [np.abs(np.diff(ep, axis=0)) for ep in ts.per_episode("action") if len(ep) > 1]
        if deltas:
            d = np.concatenate(deltas)
            for i, name in enumerate(names):
                out[("action_change", name)] = d[:, i]
    for col in ts.scalar_columns("image."):
        _, key, stat = col.split(".", 2)
        if stat in RANKED_IMAGE_STATS:
            out[("image", f"{key}.{stat}")] = ts.scalar(col)
    return out


def timing(ts: TraceSet) -> dict[str, Any]:
    def stats_ms(values: np.ndarray) -> dict[str, float] | None:
        v = values[np.isfinite(values)]
        if not v.size:
            return None
        return {"mean": float(np.mean(v)), "std": float(np.std(v)), "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "max": float(np.max(v)), "n": int(v.size)}

    frames = ts.frames()
    period = np.concatenate([np.diff(f["timestamp"].to_numpy(dtype=float)) * 1000 for f in frames if "timestamp" in f and len(f) > 1] or [np.zeros(0)])
    wall = np.concatenate([np.diff(f["wall_time"].to_numpy(dtype=float)) * 1000 for f in frames if "wall_time" in f and len(f) > 1] or [np.zeros(0)])
    latency = ts.scalar("inference_latency_ms")
    out = {
        "fps": ts.fps,
        "period_ms": stats_ms(period),
        "wall_period_ms": stats_ms(wall),
        "latency_ms": stats_ms(latency),
    }
    nominal = 1000.0 / ts.fps if ts.fps else (out["period_ms"] or {}).get("mean")
    if out["latency_ms"] and nominal:
        out["latency_p95_steps"] = out["latency_ms"]["p95"] / nominal
    return out


def chunking(ts: TraceSet) -> dict[str, Any] | None:
    jumps, inner = [], []
    for f in ts.frames():
        if "chunk_step" not in f or len(f) < 2:
            continue
        cs = f["chunk_step"].to_numpy(dtype=float)
        if not np.any(cs >= 0):
            continue
        a = np.stack(f["action"].to_numpy()).astype(float)
        d = np.linalg.norm(np.diff(a, axis=0), axis=1)
        nxt = cs[1:]
        jumps.append(d[nxt == 0])
        inner.append(d[nxt > 0])
    if not jumps:
        return None
    j, i = np.concatenate(jumps), np.concatenate(inner)
    jm = float(np.mean(j)) if j.size else math.nan
    im = float(np.mean(i)) if i.size else math.nan
    return {"boundary_jump": jm, "within_chunk_change": im, "ratio": jm / im if im and im > 1e-12 else math.nan,
            "boundaries": int(j.size)}


def outcomes(ts: TraceSet) -> dict[str, Any] | None:
    marks = [e["success"] for e in ts.episode_meta() if isinstance(e.get("success"), bool)]
    if not marks:
        return None
    s, n = sum(marks), len(marks)
    lo, hi = wilson_interval(s, n)
    reasons: dict[str, int] = {}
    for e in ts.episode_meta():
        if e.get("success") is False:
            r = e.get("failure_reason") or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
    return {"successes": s, "episodes": n, "rate": s / n, "ci_low": lo, "ci_high": hi, "failure_reasons": reasons}


# -- analysis ---------------------------------------------------------------------
def gap_analysis(a: TraceSet, b: TraceSet, labels: tuple[str, str] = ("sim", "real")) -> dict[str, Any]:
    """Compare trace ``a`` (normally sim) with trace ``b`` (normally real)."""
    la, lb = labels
    contract = check_sources(source_from_traces(a, la), source_from_traces(b, lb))
    sa, sb = signal_samples(a), signal_samples(b)
    ea, eb = len(a.frames()), len(b.frames())
    rows = []
    for key in sa:
        if key in sb:
            row = compare_signal(key[0], key[1], sa[key], sb[key], ea, eb)
            if row:
                row["hint"] = _hint(row, la, lb)
                rows.append(row)
    rows.sort(key=lambda r: -(r["w1_norm"] if math.isfinite(r["w1_norm"]) else 1e9))
    only = sorted({f"{g}:{n}" for g, n in set(sa) ^ set(sb)})
    ta, tb = timing(a), timing(b)
    return {
        "labels": [la, lb],
        "sources": {la: {"root": str(a.root), "domain": a.domain, "episodes": ea, "steps": int(sum(len(f) for f in a.frames()))},
                    lb: {"root": str(b.root), "domain": b.domain, "episodes": eb, "steps": int(sum(len(f) for f in b.frames()))}},
        "contract": [f.to_dict() for f in contract],
        "signals": rows,
        "unmatched_signals": only,
        "timing": {la: ta, lb: tb},
        "timing_notes": _timing_notes(ta, tb, la, lb),
        "chunking": {la: chunking(a), lb: chunking(b)},
        "outcomes": {la: outcomes(a), lb: outcomes(b)},
    }


def _hint(row: dict[str, Any], la: str, lb: str) -> str:
    g, sa, sb = row["group"], row["a"], row["b"]
    pooled = math.sqrt((sa["std"] ** 2 + sb["std"] ** 2) / 2) or 1e-12
    shift = abs(row["mean_shift"]) / pooled
    ratio = row["std_ratio"]
    if row["w1_norm"] < 0.2:
        return "close"
    prefix = "small; " if row["w1_norm"] < 0.5 else ""
    return prefix + _cause(g, row, sa, sb, shift, ratio, la, lb)


def _cause(g: str, row: dict[str, Any], sa: dict, sb: dict, shift: float, ratio: float, la: str, lb: str) -> str:
    if g == "image":
        stat = row["signal"].rsplit(".", 1)[-1]
        return {"brightness": "lighting / exposure / white balance differ",
                "contrast": "lighting, tone mapping or textures differ",
                "sharpness": "blur differs: focus, motion blur, compression or render resolution"}.get(stat, "camera differs")
    if g == "action_change":
        more = la if sa["mean"] > sb["mean"] else lb
        return f"commands are jerkier in {more}: latency, chunk boundaries or noisier observations"
    if shift > 1.0 and 0.5 < ratio < 2:
        return "offset: calibration, zero pose or a bias in sim"
    if ratio >= 2 or ratio <= 0.5:
        more = la if ratio > 1 else lb
        return f"much more variation in {more}: noise model, dynamics or different behaviour"
    if g == "action":
        return "policy acts differently: check its inputs (state and image rows) first"
    return "distribution shape differs: dynamics (friction, mass, damping, gains) or behaviour"


def _timing_notes(ta: dict[str, Any], tb: dict[str, Any], la: str, lb: str) -> list[str]:
    notes = []
    if ta.get("fps") and tb.get("fps") and abs(ta["fps"] - tb["fps"]) / tb["fps"] > 0.02:
        notes.append(f"Control rate differs: {ta['fps']:g} Hz in {la}, {tb['fps']:g} Hz in {lb}.")
    pa, pb = ta.get("period_ms"), tb.get("period_ms")
    if pb and pb["mean"] > 0 and pb["std"] / pb["mean"] > 0.1:
        notes.append(f"{lb} control period jitters: std {pb['std']:.2f} ms on a {pb['mean']:.2f} ms period (p95 {pb['p95']:.2f}, max {pb['max']:.2f}). "
                     f"{la} steps are {'perfectly regular' if pa and pa['std'] < 1e-6 else 'more regular'}.")
    sb = tb.get("latency_p95_steps")
    if sb is not None and sb >= 0.5:
        notes.append(f"{lb} policy latency p95 is {sb:.2f} control steps. If {la} waits for the policy (the closed-loop runner does), "
                     f"{la} never sees this delay: add about {math.ceil(sb)} step(s) of action delay in {la} to match.")
    la_ms, lb_ms = ta.get("latency_ms"), tb.get("latency_ms")
    if la_ms and lb_ms and lb_ms["p95"] > 2 * la_ms["p95"]:
        notes.append(f"Inference is slower in {lb} (p95 {lb_ms['p95']:.1f} ms vs {la_ms['p95']:.1f} ms): different hardware, network hop or batch settings.")
    if not lb_ms:
        notes.append(f"{lb} has no inference_latency_ms: record it (TraceRecorder.record_step(latency_ms=...)) to see the delay the robot really gets.")
    return notes


# -- output -----------------------------------------------------------------------
def _n(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a" if x is None or math.isnan(x) else "inf"
    if abs(x) < 1e-9:
        return "0"
    return f"{x:.{nd}g}"


def gap_markdown(result: dict[str, Any], top: int = 15) -> str:
    la, lb = result["labels"]
    src = result["sources"]
    lines = [f"# Sim-to-real gap: `{la}` vs `{lb}`", "",
             f"`{la}`: {src[la]['episodes']} episodes, {src[la]['steps']} steps (`{Path(src[la]['root']).name}`).  ",
             f"`{lb}`: {src[lb]['episodes']} episodes, {src[lb]['steps']} steps (`{Path(src[lb]['root']).name}`).", ""]

    contract = result["contract"]
    bad = [c for c in contract if c["level"] != "ok"]
    lines += ["## 1. Contract mismatches (fix these first)", ""]
    if bad:
        lines += ["| Level | Check | Detail |", "|---|---|---|"]
        lines += [f"| {'**ERROR**' if c['level'] == 'error' else 'warn'} | {c['check']} | {c['detail'].replace('|', '/')} |"
                  for c in sorted(bad, key=lambda c: c["level"] != "error")]
    else:
        lines.append("None found: keys, dimensions, units, ranges, control rate and image range are consistent.")

    lines += ["", "## 2. Largest gaps", "",
              f"Ranked by Wasserstein distance in pooled standard deviations (0 = identical, above about 0.5 is a real difference). "
              f"KS p-values use episodes as the sample size.", "",
              f"| # | Signal | {la} mean ± std | {lb} mean ± std | W1/σ | KS | p | Likely cause |",
              "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(result["signals"][:top], 1):
        lines.append(f"| {i} | {r['group']}: `{r['signal']}` | {_n(r['a']['mean'])} ± {_n(r['a']['std'])} | "
                     f"{_n(r['b']['mean'])} ± {_n(r['b']['std'])} | {_n(r['w1_norm'], 2)} | {_n(r['ks'], 2)} | "
                     f"{_n(r['ks_p_episodes'], 2)} | {r['hint']} |")
    if not result["signals"]:
        lines.append("| | no signals in common | | | | | | |")
    if result["unmatched_signals"]:
        lines += ["", f"Signals present on only one side (not compared): {', '.join(result['unmatched_signals'][:20])}"]

    lines += ["", "## 3. Timing", "", f"| Measure | {la} | {lb} |", "|---|---|---|"]
    t = result["timing"]

    def cell(d: dict | None, key: str) -> str:
        return "n/a" if not d else f"{_n(d['mean'], 3)} ± {_n(d['std'], 2)} (p95 {_n(d['p95'], 3)}, max {_n(d['max'], 3)})" if key == "full" else _n(d.get(key))

    lines.append(f"| Control rate (Hz) | {_n(t[la]['fps'])} | {_n(t[lb]['fps'])} |")
    lines.append(f"| Control period from timestamps (ms) | {cell(t[la]['period_ms'], 'full')} | {cell(t[lb]['period_ms'], 'full')} |")
    lines.append(f"| Host-clock step period (ms) | {cell(t[la]['wall_period_ms'], 'full')} | {cell(t[lb]['wall_period_ms'], 'full')} |")
    lines.append(f"| Inference latency (ms) | {cell(t[la]['latency_ms'], 'full')} | {cell(t[lb]['latency_ms'], 'full')} |")
    lines.append(f"| Latency p95 in control steps | {_n(t[la].get('latency_p95_steps'), 2)} | {_n(t[lb].get('latency_p95_steps'), 2)} |")
    lines += ["", "In simulation the host-clock period measures compute speed, not a control rate; compare it only between real runs.", ""]
    lines += [f"* {n}" for n in result["timing_notes"]] or ["* No timing problems detected."]

    lines += ["", "## 4. Action chunking", ""]
    ch = result["chunking"]
    if ch[la] or ch[lb]:
        lines += [f"| Measure | {la} | {lb} |", "|---|---|---|"]
        for key, label in (("boundary_jump", "Mean action change at a new chunk"), ("within_chunk_change", "Mean action change inside a chunk"),
                           ("ratio", "Ratio (1 = seamless)")):
            lines.append(f"| {label} | {_n((ch[la] or {}).get(key))} | {_n((ch[lb] or {}).get(key))} |")
        lines += ["", "A ratio well above 1 means each new chunk starts from a different place than the last one ended: "
                      "the policy's prediction of the next state is off (worse when latency is unmodelled)."]
        ra, rb = (ch[la] or {}).get("ratio"), (ch[lb] or {}).get("ratio")
        if ra and rb and math.isfinite(ra) and math.isfinite(rb) and rb > 1.5 * ra:
            lines += ["", f"**Chunk boundaries jump {rb / ra:.1f}x more in `{lb}` than in `{la}`.** The policy's internal model of "
                          f"what happens during a chunk fits `{la}` better: look for actuation delay, latency or dynamics that `{la}` lacks."]
    else:
        lines.append("No chunk information recorded (policy without chunking, or a recording without `chunk_step`).")

    lines += ["", "## 5. Outcomes", ""]
    oc = result["outcomes"]
    if oc[la] or oc[lb]:
        lines += [f"| | Success | 95% CI | Failure reasons |", "|---|---|---|---|"]
        for lab in (la, lb):
            o = oc[lab]
            lines.append(f"| {lab} | " + (f"{o['successes']}/{o['episodes']} ({100 * o['rate']:.1f}%) | {100 * o['ci_low']:.1f}–{100 * o['ci_high']:.1f}% | "
                                          f"{o['failure_reasons'] or 'none'} |" if o else "n/a | | |"))
    else:
        lines.append("No success labels in either trace.")
    lines += ["", "---", "Distribution comparison only: differences can come from behaviour as well as physics. "
                     "Fix contract mismatches, then check the top-ranked observation rows before the action rows, since the policy reacts to what it sees."]
    return "\n".join(lines) + "\n"


def write_plots(result: dict[str, Any], a: TraceSet, b: TraceSet, out: str | Path, top: int = 6) -> list[Path]:
    """Overlaid histograms of the top-ranked signals (needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    la, lb = result["labels"]
    sa, sb = signal_samples(a), signal_samples(b)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in enumerate(result["signals"][:top], 1):
        key = (row["group"], row["signal"])
        xa, xb = sa[key][np.isfinite(sa[key])], sb[key][np.isfinite(sb[key])]
        fig, ax = plt.subplots(figsize=(5, 3))
        bins = np.histogram_bin_edges(np.concatenate([xa, xb]), bins=40)
        ax.hist(xa, bins=bins, alpha=0.55, density=True, label=la)
        ax.hist(xb, bins=bins, alpha=0.55, density=True, label=lb)
        ax.set_title(f"{row['group']}: {row['signal']}  (W1/σ={row['w1_norm']:.2f})", fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = "".join(c if c.isalnum() else "_" for c in f"{i:02d}_{row['group']}_{row['signal']}")
        path = out / f"{safe}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths
