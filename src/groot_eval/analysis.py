"""Failure attribution: which randomized parameters make the policy fail.

For each parameter, episodes are bucketed (quantile bins for numbers, one
bucket per value otherwise) and each bucket gets a success rate with a
confidence interval. Parameters are ranked by the gap between their best and
worst bucket, but only gaps where the intervals do not overlap are flagged as
significant, so a lucky bucket of five episodes doesn't send anyone chasing
friction when the real problem is latency.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from groot_eval.stats import wilson_interval
from groot_eval.types import EpisodeResult


def results_frame(results: list[EpisodeResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"scenario_id": r.scenario.scenario_id, "seed": r.scenario.seed, "success": r.success,
                     "failure_reason": r.failure_reason, "steps": r.steps, **{f"param.{k}": v for k, v in r.scenario.params.items()}})
    return pd.DataFrame(rows)


def attribute_failures(results: list[EpisodeResult], bins: int = 4, confidence: float = 0.95) -> list[dict[str, Any]]:
    df = results_frame(results)
    out = []
    for col in [c for c in df.columns if c.startswith("param.")]:
        series = df[col]
        if series.nunique(dropna=True) < 2:
            continue
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if numeric and series.nunique() > bins:
            codes, edges = pd.qcut(series, q=bins, labels=False, retbins=True, duplicates="drop")
            names = [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(edges) - 1)]
            # Zero-padded index keeps buckets in numeric order when grouped.
            labels = codes.map(lambda c: f"{int(c):02d}|{names[int(c)]}")
        else:
            labels = series.astype(str)
        rows = []
        for label, group in df.groupby(labels, sort=True, observed=True):
            s, n = int(group["success"].sum()), len(group)
            lo, hi = wilson_interval(s, n, confidence)
            rows.append({"bucket": str(label).split("|", 1)[-1], "episodes": n, "success_rate": s / n, "ci_low": lo, "ci_high": hi})
        if len(rows) < 2:
            continue
        best = max(rows, key=lambda r: r["success_rate"])
        worst = min(rows, key=lambda r: r["success_rate"])
        out.append({
            "param": col.removeprefix("param."),
            "gap": best["success_rate"] - worst["success_rate"],
            "significant": worst["ci_high"] < best["ci_low"],
            "worst_bucket": worst["bucket"],
            "buckets": rows,
        })
    out.sort(key=lambda r: (not r["significant"], -r["gap"]))
    return out


def worst_cases(results: list[EpisodeResult], limit: int = 10) -> list[dict[str, Any]]:
    """Failed episodes, closest-to-success last, so the list leads with the clearest failures."""
    failed = [r for r in results if not r.success]
    failed.sort(key=lambda r: (-r.metrics.get("final_distance", 0.0), r.steps))
    return [r.to_dict() for r in failed[:limit]]


