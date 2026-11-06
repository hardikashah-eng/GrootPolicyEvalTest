"""Paired replay and a system-identification sweep.

``replay`` puts the simulator into each recorded episode's first state, plays
the recorded actions open loop, and measures how far the simulated states
drift from the recorded ones. Run against a real-robot trace, that answers
"given the same commands, does the sim move like the robot?" without the
policy in the loop, so policy feedback cannot hide (or cause) the gap.

With a sweep config, every candidate parameter setting is replayed and ranked
by tracking error: the setting that reproduces the real trajectories best is
the sim's best guess at the real physics, and the per-parameter spread says
which knob matters.

Error is reported per state signal as nRMSE: RMS error divided by the standard
deviation of that signal in the recording (0 is perfect; 1 is about as wrong as
always predicting the signal's mean). Signals that are constant in the
recording, or within every episode (a fixed target, say), are excluded from the score.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from groot_eval.envs.base import Env
from groot_eval.randomization import SweepConfig
from groot_eval.trace import TraceSet
from groot_eval.types import Scenario

HORIZONS = (1, 5, 10, 20, 40, 80)
_CONSTANT = 1e-9


@dataclass
class ReplayEpisode:
    recorded: np.ndarray  # (T, D) recorded states
    predicted: np.ndarray  # (T, D) simulated states, NaN after the sim ended
    since_sync: np.ndarray  # (T,) steps since the sim was last put into the recorded state
    synced: bool
    ended_at: int | None  # step at which the sim terminated before the recording did


@dataclass
class ReplayScore:
    params: dict[str, Any]
    score: float
    signals: dict[str, float]
    horizons: dict[int, float]
    episodes: int
    steps: int
    synced: bool
    ended_early: int
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "params": self.params, "score": self.score, "signals": self.signals,
                "horizons": {str(k): v for k, v in self.horizons.items()}, "episodes": self.episodes,
                "steps": self.steps, "synced": self.synced, "ended_early": self.ended_early}


# -- one episode ----------------------------------------------------------------
def _state_dict(vec: np.ndarray, slices: dict[str, slice]) -> dict[str, np.ndarray]:
    return {f"state.{k}": vec[s].astype(np.float32) for k, s in slices.items()}


def _flatten(obs: dict[str, Any], slices: dict[str, slice]) -> np.ndarray:
    parts = []
    for k, s in slices.items():
        key = f"state.{k}"
        if key not in obs:
            raise ValueError(f"env observation has no {key!r}, which the trace records; run `groot-eval check` first")
        v = np.asarray(obs[key], dtype=np.float64).reshape(-1)
        if v.size != s.stop - s.start:
            raise ValueError(f"{key}: env gives {v.size} values, trace has {s.stop - s.start}")
        parts.append(v)
    return np.concatenate(parts)


def replay_episode(env: Env, states: np.ndarray, actions: np.ndarray, slices: dict[str, slice], *,
                   seed: int = 0, task: str = "", params: dict[str, Any] | None = None,
                   reset_every: int | None = None) -> ReplayEpisode:
    """Replay one recorded episode open loop. ``states[t]`` is what the robot saw before ``actions[t]``."""
    T = len(states)
    obs = env.reset(Scenario(seed=seed, params=dict(params or {}), task=task))
    if hasattr(env, "max_steps") and env.max_steps < T + 1:
        env.max_steps = T + 1  # the recording's length decides when to stop, not the env's timeout
    synced = True
    try:
        obs = env.set_state(_state_dict(states[0], slices), actions[:0])  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError):
        synced = False  # start from the seed's reset state instead
    predicted = np.full((T, states.shape[1]), np.nan)
    since = np.zeros(T, dtype=int)
    predicted[0] = _flatten(obs, slices)
    ended_at = None
    k = 0
    for t in range(T - 1):
        if synced and reset_every and t > 0 and t % reset_every == 0:
            obs = env.set_state(_state_dict(states[t], slices), actions[:t])  # type: ignore[attr-defined]
            k = 0
        obs, _, terminated, truncated, _ = env.step(actions[t])
        k += 1
        predicted[t + 1], since[t + 1] = _flatten(obs, slices), k
        if (terminated or truncated) and t + 2 < T:
            ended_at = t + 1
            break
    return ReplayEpisode(states, predicted, since, synced, ended_at)


# -- scoring --------------------------------------------------------------------
def _episodes(ts: TraceSet, limit: int | None) -> list[dict[str, Any]]:
    out = []
    for i, (meta, states, actions) in enumerate(zip(ts.episode_meta(), ts.per_episode("observation.state"),
                                                    ts.per_episode("action"))):
        scen = meta.get("scenario") or {}
        out.append({"seed": int(scen.get("seed", i)), "task": (meta.get("tasks") or [""])[0],
                    "states": states, "actions": actions})
    return out[:limit] if limit else out


def score_replays(runs: list[ReplayEpisode], slices: dict[str, slice], scale: np.ndarray,
                  signals: list[str] | None = None) -> tuple[float, dict[str, float], dict[int, float]]:
    """Return (score, nRMSE per signal, mean normalized error at each horizon)."""
    err = np.concatenate([(r.predicted - r.recorded)[1:] for r in runs])
    since = np.concatenate([r.since_sync[1:] for r in runs])
    live = scale > _CONSTANT
    norm = np.where(live, np.abs(err) / np.where(live, scale, 1.0), np.nan)
    per_signal: dict[str, float] = {}
    for name, s in slices.items():
        cols = norm[:, s][:, live[s]]
        if cols.size == 0:
            continue
        vals = cols[np.isfinite(cols)]
        per_signal[name] = float(np.sqrt(np.mean(vals ** 2))) if vals.size else math.nan
    chosen = [per_signal[n] for n in (signals or per_signal) if n in per_signal and math.isfinite(per_signal[n])]
    score = float(np.mean(chosen)) if chosen else math.nan
    horizons: dict[int, float] = {}
    for h in HORIZONS:
        rows = norm[since == h]
        vals = rows[np.isfinite(rows)]
        if vals.size:
            horizons[h] = float(np.mean(vals))
    return score, per_signal, horizons


def replay_trace(env: Env, ts: TraceSet, params: dict[str, Any] | None = None, *, reset_every: int | None = None,
                 signals: list[str] | None = None, episodes: int | None = None, label: str = "") -> ReplayScore:
    slices = ts.state_slices()
    eps = _episodes(ts, episodes)
    if not eps:
        raise ValueError(f"{ts.root} has no episodes")
    runs = [replay_episode(env, e["states"], e["actions"], slices, seed=e["seed"], task=e["task"],
                           params=params, reset_every=reset_every) for e in eps]
    scale = np.concatenate([e["states"] for e in eps]).std(axis=0)
    # A signal that never changes within an episode (a target set at reset) is copied in by set_state,
    # so it would score a free 0; leave it out.
    moving = np.max([e["states"].std(axis=0) for e in eps], axis=0) > _CONSTANT
    scale = np.where(moving, scale, 0.0)
    score, per_signal, horizons = score_replays(runs, slices, scale, signals)
    return ReplayScore(dict(params or {}), score, per_signal, horizons, len(runs),
                       int(sum(len(e["states"]) for e in eps)), all(r.synced for r in runs),
                       sum(r.ended_at is not None for r in runs), label)


# -- system identification --------------------------------------------------------
def candidate_params(cfg: SweepConfig) -> list[dict[str, Any]]:
    """Parameter settings to try: the grid in grid mode, ``episodes`` random draws otherwise."""
    if cfg.mode == "grid":
        axes = [p.grid(cfg.grid_points) for p in cfg.params]
        return [{p.name: v for p, v in zip(cfg.params, combo)} for combo in itertools.product(*axes)]
    seen, out = set(), []
    for scen in cfg.scenarios():
        key = repr(sorted(scen.params.items()))
        if key not in seen:
            seen.add(key)
            out.append(scen.params)
    return out


def system_id(env: Env, ts: TraceSet, cfg: SweepConfig, base: dict[str, Any] | None = None, *,
              reset_every: int | None = None, signals: list[str] | None = None,
              episodes: int | None = None) -> dict[str, Any]:
    base = dict(base or {})
    baseline = replay_trace(env, ts, base, reset_every=reset_every, signals=signals, episodes=episodes,
                            label="baseline")
    ranked = []
    for params in candidate_params(cfg):
        ranked.append(replay_trace(env, ts, {**base, **params}, reset_every=reset_every, signals=signals,
                                   episodes=episodes))
    ranked.sort(key=lambda r: (math.isnan(r.score), r.score))
    for i, r in enumerate(ranked, 1):
        r.label = f"#{i}"
    swept = [p.name for p in cfg.params]
    return {"baseline": baseline, "ranked": ranked, "swept": swept,
            "sensitivity": _sensitivity(ranked, cfg), "notes": _fit_notes(baseline, ranked, cfg)}


def _sensitivity(ranked: list[ReplayScore], cfg: SweepConfig) -> list[dict[str, Any]]:
    """For each swept parameter: the best score reachable at each of its values."""
    rows = []
    for p in cfg.params:
        best: dict[str, float] = {}
        for r in ranked:
            v = _value(r.params.get(p.name))
            if math.isfinite(r.score):
                best[v] = min(best.get(v, math.inf), r.score)
        if len(best) < 2:
            continue
        values = sorted(best.values())
        rows.append({"param": p.name, "spread": values[-1] - values[0], "best_by_value": best})
    rows.sort(key=lambda r: -r["spread"])
    return rows


def _fit_notes(baseline: ReplayScore, ranked: list[ReplayScore], cfg: SweepConfig) -> list[str]:
    notes: list[str] = []
    finite = [r for r in ranked if math.isfinite(r.score)]
    if not finite:
        return ["no candidate could be scored"]
    best = finite[0]
    if math.isfinite(baseline.score) and baseline.score > 0:
        gain = 1 - best.score / baseline.score
        notes.append(f"best setting cuts tracking error by {gain:.0%} versus the baseline "
                     f"({baseline.score:.3f} -> {best.score:.3f})")
    for p in cfg.params:
        v = best.params.get(p.name)
        if p.kind in ("uniform", "loguniform") and isinstance(v, (int, float)) and p.args[0] != p.args[1]:
            lo, hi = p.args
            if math.isclose(v, lo) or math.isclose(v, hi):
                notes.append(f"`{p.name}` = {v:g} is at the edge of the searched range [{lo:g}, {hi:g}]; widen it")
    close = [r for r in finite if r.score <= best.score * 1.05 + 1e-12]
    if len(close) > 1:
        loose = sorted({k for r in close for k in r.params if r.params[k] != best.params.get(k)})
        if loose:
            notes.append(f"{len(close)} settings score within 5% of the best; not pinned down by this data: "
                         + ", ".join(f"`{k}`" for k in loose))
    if any("noise" in p.name for p in cfg.params):
        notes.append("noise parameters only add error to an open-loop replay; fit them with `groot-eval gap` "
                     "(state spread, action jerk) instead")
    if not best.synced:
        notes.append("the env has no set_state(): replays started from the seed's reset state, not the recorded one")
    if best.ended_early:
        notes.append(f"{best.ended_early} replayed episode(s) ended (terminated) before the recording did")
    return notes


# -- report -----------------------------------------------------------------------
def _f(x: float) -> str:
    return "-" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _value(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def format_params(p: dict[str, Any]) -> str:
    return ", ".join(f"{k}={_value(v)}" for k, v in p.items()) or "(env defaults)"


def replay_markdown(ts: TraceSet, env_name: str, baseline: ReplayScore, fit: dict[str, Any] | None = None,
                    top: int = 10, reset_every: int | None = None) -> str:
    sync = "yes" if baseline.synced else "no (env has no set_state; started from the seed's reset state)"
    lines = [
        "# Replay: recorded actions through the simulator",
        "",
        f"- Trace: `{ts.root.name}` (domain: {ts.domain}), {baseline.episodes} episodes, {baseline.steps} steps",
        f"- Env: `{env_name}`; started from recorded state: {sync}",
        f"- Re-synced to the recording every {reset_every} steps" if reset_every else
        "- Pure open loop: never re-synced after the first step",
        "",
        "nRMSE = RMS(sim state - recorded state) / std(recorded state). 0 is perfect; around 1 means the "
        "sim is about as wrong as always predicting the mean. Signals that never change within an episode (a target set at reset) are not scored.",
        "",
        f"## 1. Baseline: {format_params(baseline.params)}",
        "",
        f"Score (mean nRMSE): **{_f(baseline.score)}**"
        + (f"; {baseline.ended_early} episode(s) ended early in sim" if baseline.ended_early else ""),
        "",
        "| Signal | nRMSE |",
        "|---|---|",
        *[f"| {k} | {_f(v)} |" for k, v in baseline.signals.items()],
        "",
        "Error growth (mean |error| / std, by steps since the sim was last put in the recorded state):",
        "",
        "| Steps | " + " | ".join(str(h) for h in baseline.horizons) + " |",
        "|---|" + "---|" * len(baseline.horizons),
        "| error | " + " | ".join(_f(v) for v in baseline.horizons.values()) + " |",
        "",
        "Error at step 1 that is already large points at a one-step model error (gain, mass, units, "
        "a missing delay); error that starts small and grows points at slow drift (friction, damping).",
    ]
    if fit:
        ranked = fit["ranked"][:top]
        names = list(ranked[0].signals) if ranked else []
        lines += [
            "",
            f"## 2. Parameter fit ({len(fit['ranked'])} settings of {', '.join(f'`{n}`' for n in fit['swept'])})",
            "",
            "| Rank | Score | " + " | ".join(names) + " | Parameters |",
            "|---|---|" + "---|" * len(names) + "---|",
            f"| baseline | {_f(fit['baseline'].score)} | "
            + " | ".join(_f(fit["baseline"].signals.get(n, math.nan)) for n in names)
            + f" | {format_params(fit['baseline'].params)} |",
            *[f"| {r.label} | {_f(r.score)} | " + " | ".join(_f(r.signals.get(n, math.nan)) for n in names)
              + f" | {format_params(r.params)} |" for r in ranked],
        ]
        if fit["notes"]:
            lines += ["", *[f"- {n}" for n in fit["notes"]]]
        if fit["sensitivity"]:
            lines += [
                "",
                "### Which knob matters",
                "",
                "Best score reachable with each value of a parameter (others free). A large spread means "
                "the recording constrains that parameter; a flat row means it does not.",
                "",
                "| Parameter | Spread | Best score by value |",
                "|---|---|---|",
                *[f"| {s['param']} | {s['spread']:.3f} | "
                  + ", ".join(f"{v}: {sc:.3f}" for v, sc in s["best_by_value"].items()) + " |"
                  for s in fit["sensitivity"]],
            ]
    return "\n".join(lines) + "\n"
