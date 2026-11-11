"""Real to sim for wear: how old does this robot behave, and which wear explains its trace?

The age profiles in ``configs/wear/`` are placeholders until they are fitted to
real robots. :func:`fit_age` closes that loop with the replay machinery that
already exists: it plays a real (worn) robot's recorded actions through
``WearEnv(sim)`` configured as a typical unit of each candidate age, and ranks
the ages by how well the simulated states track the recorded ones. The best age
is the sim's estimate of the robot's *effective* age; the per-effect values at
that age are the wear parameters to give a digital twin of that unit.

What replay can and cannot pin down:

* gain, torque limit, friction, damping, backlash, stiction and thermal derating
  change how the robot responds to commands, so they show up in replay error;
* signed sensor biases are set to zero (the sign of one unit's bias is not known
  from the profile), and noise only adds open-loop error: fit those with
  ``groot-eval gap`` (state offsets and spread) instead;
* if the best age is at the edge of the grid, widen the grid; if many ages score
  within 5% of the best, the trace does not constrain age (record longer or more
  varied motions).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from groot_eval.degradation.wear import WearConfig, WearEnv
from groot_eval.envs.base import ComponentSpec
from groot_eval.trace import TraceSet

EnvFactory = Callable[[], Any]


def age_grid(cfg: WearConfig, points: int = 12) -> list[float]:
    """Candidate ages: every profile age plus an even grid up to end of life."""
    ages = set(cfg.profiles.values()) | {float(a) for a in np.linspace(0.0, cfg.eol_hours, points)}
    return sorted(ages)


def fit_age(make_env: EnvFactory, real: TraceSet, cfg: WearConfig, components: dict[str, ComponentSpec],
            ages: list[float] | None = None, base_params: dict[str, Any] | None = None,
            episodes: int | None = None) -> dict[str, Any]:
    from groot_eval.replay import replay_trace

    ages = sorted(ages) if ages else age_grid(cfg)
    rows = []
    for age in ages:
        params = {**(base_params or {}), **cfg.typical_unit(age, components)}
        params.pop("wear.age_hours", None)
        env = WearEnv(make_env())
        try:
            score = replay_trace(env, real, params, episodes=episodes, label=f"{age:g} h")
        finally:
            env.close()
        rows.append({"age_hours": age, "score": score.score, "horizons": score.horizons, "params": params,
                     "synced": score.synced})
    finite = [r for r in rows if math.isfinite(r["score"])]
    if not finite:
        return {"ages": rows, "best": None, "notes": ["no age could be scored (replay produced no finite error)"]}
    best = min(finite, key=lambda r: r["score"])
    notes = []
    new = next((r for r in rows if r["age_hours"] == min(ages)), None)
    if new and math.isfinite(new["score"]) and new["score"] > 0 and best is not new:
        notes.append(f"a {best['age_hours']:g} h unit tracks the recording {1 - best['score'] / new['score']:.0%} better "
                     f"than a new one ({new['score']:.3f} -> {best['score']:.3f})")
    if best["age_hours"] in (min(ages), max(ages)) and len(ages) > 2:
        edge = "youngest" if best["age_hours"] == min(ages) else "oldest"
        notes.append(f"best fit is the {edge} age tried; " + ("the robot behaves like a new one" if edge == "youngest"
                                                               else "widen the grid or check eol_hours"))
    close = [r["age_hours"] for r in finite if r["score"] <= best["score"] * 1.05 + 1e-12]
    if len(close) > 1:
        notes.append(f"ages {min(close):g}-{max(close):g} h all score within 5% of the best: the trace does not pin age down")
    if not best["synced"]:
        notes.append("env has no set_state(): replays started from the seed's reset state, not the recorded one")
    notes.append("signed biases and noise are not fitted by replay; compare offsets and spread with `groot-eval gap`")
    profile = min(cfg.profiles, key=lambda p: abs(cfg.profiles[p] - best["age_hours"]))
    return {"ages": rows, "best": best, "closest_profile": profile, "notes": notes}


def fit_markdown(fit: dict[str, Any], trace: str, env_name: str) -> str:
    lines = ["# Wear calibration: effective age of a real robot", "",
             f"Trace `{trace}` replayed through `WearEnv({env_name})` configured as a typical unit of each age. "
             "Lower error = the sim at that age moves more like the recorded robot.", ""]
    best = fit.get("best")
    if best:
        lines += [f"**Best fit: {best['age_hours']:g} h** (closest profile `{fit['closest_profile']}`), "
                  f"replay error {best['score']:.3f}.", ""]
    lines += ["| Age (h) | Replay error (mean nRMSE) | Error after 10 steps |", "|---|---|---|"]
    for r in fit["ages"]:
        mark = " **best**" if best is r else ""
        h10 = r["horizons"].get(10, math.nan)
        lines.append(f"| {r['age_hours']:g}{mark} | {r['score']:.3f} | {h10:.3f} |" if math.isfinite(r["score"])
                     else f"| {r['age_hours']:g} | – | – |")
    if best:
        wear = {k: v for k, v in best["params"].items() if k.startswith("wear.")}
        lines += ["", "## Wear parameters for a digital twin of this unit", "",
                  "Pass these as scenario parameters (or `--param`) with `WearEnv` to make the sim behave like this robot.", "",
                  "| Parameter | Value |", "|---|---|"]
        lines += [f"| `{k}` | {v:.4g} |" if isinstance(v, float) else f"| `{k}` | {v} |" for k, v in sorted(wear.items())]
    if fit.get("notes"):
        lines += ["", *[f"- {n}" for n in fit["notes"]]]
    return "\n".join(lines) + "\n"
