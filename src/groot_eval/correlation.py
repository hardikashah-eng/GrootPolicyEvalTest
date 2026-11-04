"""Sim-to-real correlation: can simulation results be trusted to stand in for the robot?

Input is a CSV of paired runs, one row per (policy, scenario, domain)::

    policy,scenario_id,domain,success
    ckpt_a,pick_cup_01,sim,1
    ckpt_a,pick_cup_01,real,0
    ...

The questions it answers:

* **Ranking agreement** - do sim and real order the checkpoints the same way?
  (Spearman over per-policy success rates.) This is what decides whether sim
  can choose which checkpoint goes to the robot.
* **Gap** - how much more optimistic is sim, per policy, with intervals.
* **Scenario agreement** - on matched scenarios, how often do sim and real
  give the same outcome, and which scenarios disagree most (candidates for
  better simulation assets or physics tuning).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from groot_eval.stats import spearman, wilson_interval


def load_pairs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"policy", "scenario_id", "domain", "success"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df["domain"] = df["domain"].str.lower()
    bad = set(df["domain"]) - {"sim", "real"}
    if bad:
        raise ValueError(f"domain must be sim or real; got {sorted(bad)}")
    df["success"] = df["success"].astype(int)
    return df


def correlate(df: pd.DataFrame, confidence: float = 0.95) -> dict[str, Any]:
    per_policy = []
    for policy, group in df.groupby("policy", sort=True):
        row: dict[str, Any] = {"policy": policy}
        for domain in ("sim", "real"):
            d = group[group["domain"] == domain]["success"]
            s, n = int(d.sum()), len(d)
            lo, hi = wilson_interval(s, n, confidence)
            row[domain] = {"successes": s, "trials": n, "rate": s / n if n else None, "ci_low": lo, "ci_high": hi}
        if row["sim"]["rate"] is not None and row["real"]["rate"] is not None:
            row["gap"] = row["sim"]["rate"] - row["real"]["rate"]
        per_policy.append(row)

    both = [p for p in per_policy if p["sim"]["trials"] and p["real"]["trials"]]
    rho = spearman([p["sim"]["rate"] for p in both], [p["real"]["rate"] for p in both])

    # Matched outcomes: average repeated trials per (policy, scenario, domain).
    pivot = df.pivot_table(index=["policy", "scenario_id"], columns="domain", values="success", aggfunc="mean").dropna()
    agreement = float(((pivot["sim"] >= 0.5) == (pivot["real"] >= 0.5)).mean()) if len(pivot) else None
    pivot["disagreement"] = (pivot["sim"] - pivot["real"]).abs()
    by_scenario = pivot.groupby(level="scenario_id")["disagreement"].mean().sort_values(ascending=False)
    sim_optimistic = float((pivot["sim"] > pivot["real"]).mean()) if len(pivot) else None

    return {
        "policies": per_policy,
        "ranking_spearman": rho,
        "matched_pairs": int(len(pivot)),
        "outcome_agreement": agreement,
        "sim_more_optimistic_share": sim_optimistic,
        "most_divergent_scenarios": [{"scenario_id": k, "mean_abs_disagreement": float(v)} for k, v in by_scenario.head(10).items()],
    }
