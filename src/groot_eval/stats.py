"""Statistics for non-deterministic policy evaluation.

A policy that succeeds 46 times out of 50 has not "passed 92%". It has a
success rate somewhere in an interval, and two policies whose intervals
overlap heavily are not shown to differ. Everything a report says about
success rates goes through these helpers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# Two-sided z for common confidence levels, so no scipy dependency is needed.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def z_for(confidence: float) -> float:
    try:
        return _Z[round(confidence, 2)]
    except KeyError as exc:
        raise ValueError(f"unsupported confidence {confidence}; use one of {sorted(_Z)}") from exc


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Behaves sensibly at 0/n and n/n, unlike the normal approximation, which
    is exactly where release gates live.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid counts: {successes}/{trials}")
    if trials == 0:
        return (0.0, 1.0)
    z = z_for(confidence)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
    stat=np.mean,
) -> tuple[float, float]:
    """Percentile bootstrap interval for any statistic of continuous values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    stats = np.apply_along_axis(stat, 1, arr[idx])
    lo = (1 - confidence) / 2
    return (float(np.quantile(stats, lo)), float(np.quantile(stats, 1 - lo)))


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> float:
    """z statistic for H0: p1 == p2 (pooled). Positive means p1 > p2."""
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    return (p1 - p2) / se


def is_regression(
    baseline: tuple[int, int], candidate: tuple[int, int], confidence: float = 0.95
) -> bool:
    """True when the candidate's success rate is significantly below baseline's (one-sided)."""
    z = two_proportion_z(candidate[0], candidate[1], baseline[0], baseline[1])
    one_sided = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}[round(confidence, 2)]
    return z < -one_sided


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    arr = np.asarray(values, dtype=float)
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    sorted_vals = arr[order]
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; nan when either side is constant or n < 2."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 2:
        return math.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def required_trials(margin: float, confidence: float = 0.95, p: float = 0.5) -> int:
    """Episodes needed so a success-rate interval's half-width is at most ``margin``.

    Useful for answering "how many runs do we need before we can gate on this?"
    """
    if not 0 < margin < 1:
        raise ValueError("margin must be in (0, 1)")
    z = z_for(confidence)
    return math.ceil(z * z * p * (1 - p) / (margin * margin))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    ``b`` = pairs where only the first condition succeeded, ``c`` = only the
    second. Pairs that agree carry no information. Running the same scenarios
    under two conditions (a new and a 1-year-old robot) and testing the
    discordant pairs detects smaller differences than two independent sweeps.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))
