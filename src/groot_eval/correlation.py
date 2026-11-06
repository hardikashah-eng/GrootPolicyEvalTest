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

Producing the CSV: ``sweep --export-pairs`` writes the sim rows and
``groot-eval pairs TRACE`` writes rows from any trace (a real-robot one
included). Both use the scenario's ``name`` when it has one, so naming sim
scenarios after the real test set (see :mod:`groot_eval.randomization`) is
what makes them join. :func:`join_diagnostics` reports what did not match.
"""

from __future__ import annotations

import csv
import re
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


PAIR_COLUMNS = ["policy", "scenario_id", "domain", "success"]
_HASH_ID = re.compile(r"^[0-9a-f]{12}$")


def pairs_from_run(run: Any, policy: str | None = None, domain: str = "sim") -> list[dict[str, Any]]:
    """Rows for an :class:`~groot_eval.closed_loop.EvalRun`, keyed by scenario name when set."""
    label = policy or run.policy
    return [{"policy": label, "scenario_id": r.scenario.pair_id, "domain": domain, "success": int(r.success)}
            for r in run.results]


def pairs_from_trace(ts: Any, policy: str | None = None, domain: str | None = None) -> list[dict[str, Any]]:
    """Rows for a trace. The scenario is ``scenario_name`` from the episode metadata
    (``TraceRecorder.start_episode(task, scenario_name=...)``), else the sim scenario's
    name or id, else ``episode_<index>``. Episodes without a success mark are skipped."""
    domain = domain or ts.domain
    if domain not in ("sim", "real"):
        raise ValueError(f"trace domain is {domain!r}; pass domain='sim' or 'real'")
    label = policy or ts.info.get("policy") or "policy"
    rows = []
    for i, meta in enumerate(ts.episode_meta()):
        if not isinstance(meta.get("success"), bool):
            continue
        scen = meta.get("scenario") if isinstance(meta.get("scenario"), dict) else {}
        sid = meta.get("scenario_name") or scen.get("name") or scen.get("scenario_id") or f"episode_{i:03d}"
        rows.append({"policy": label, "scenario_id": str(sid), "domain": domain, "success": int(meta["success"])})
    return rows


def write_pairs(rows: list[dict[str, Any]], path: str | Path, append: bool = True) -> Path:
    """Write pair rows; with ``append``, add to an existing file so checkpoints and domains accumulate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not (append and path.exists() and path.stat().st_size > 0)
    with open(path, "w" if new else "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return path


def join_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """What the sim/real join matched, what it did not, and the likely reason."""
    keys = {d: set(map(tuple, df[df["domain"] == d][["policy", "scenario_id"]].astype(str).drop_duplicates().to_numpy()))
            for d in ("sim", "real")}
    scen = {d: {k[1] for k in keys[d]} for d in keys}
    pol = {d: {k[0] for k in keys[d]} for d in keys}
    matched = keys["sim"] & keys["real"]
    warnings: list[str] = []
    if keys["sim"] and keys["real"] and not matched:
        if not pol["sim"] & pol["real"]:
            warnings.append(f"no policy appears in both domains (sim: {sorted(pol['sim'])[:3]}, real: {sorted(pol['real'])[:3]}); "
                            "use the same checkpoint label on both sides (`--pairs-policy` / `--policy`)")
        else:
            warnings.append("no scenario appears in both domains for the same policy")
    hashed = {d: (sum(bool(_HASH_ID.match(s)) for s in scen[d]) / len(scen[d])) if scen[d] else 0.0 for d in scen}
    if scen["sim"] and scen["real"] and (hashed["sim"] > 0.5) != (hashed["real"] > 0.5):
        side = "sim" if hashed["sim"] > 0.5 else "real"
        warnings.append(f"{side} scenario ids look like generated hashes while the other side uses names: give the sim "
                        "scenarios the real test set's names (`scenarios:` in the sweep config)")
    only_sim, only_real = sorted(scen["sim"] - scen["real"]), sorted(scen["real"] - scen["sim"])
    if matched and (only_sim or only_real):
        warnings.append(f"{len(only_sim)} scenario(s) only in sim and {len(only_real)} only in real are left out of the matched comparison")
    return {"matched": len(matched), "sim_keys": len(keys["sim"]), "real_keys": len(keys["real"]),
            "only_sim": only_sim[:20], "only_real": only_real[:20],
            "policies_only_sim": sorted(pol["sim"] - pol["real"]), "policies_only_real": sorted(pol["real"] - pol["sim"]),
            "warnings": warnings}


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
        "join": join_diagnostics(df),
        "policies": per_policy,
        "ranking_spearman": rho,
        "matched_pairs": int(len(pivot)),
        "outcome_agreement": agreement,
        "sim_more_optimistic_share": sim_optimistic,
        "most_divergent_scenarios": [{"scenario_id": k, "mean_abs_disagreement": float(v)} for k, v in by_scenario.head(10).items()],
    }
