"""Per-scenario-class twin credibility and the twin certificate.

A single sim-vs-real correlation hides where a twin is trustworthy and where it is not: a twin
can rank policies well on tabletop picks and badly on articulated objects. This module judges
the twin **separately for each scenario class** (any label in the results: task family, object
type, skill) and writes a **certificate** that says, for this twin version and this robot's
approved profile, in which classes a sim result may stand in for robot runs.

Input rows, one per episode or per aggregate (CSV or dicts)::

    policy, domain (sim|real), scenario_class, success            # one episode per row
    policy, domain, scenario_class, successes, trials             # aggregated
    ... optional scenario_id: pairs sim and real episodes of the same scenario

Per class it reports the number of policies, Pearson r, Spearman, MMRV, bias (sim - real) with a
bootstrap interval, the pass/fail decision agreement at a gate, and, when episodes are paired by
``scenario_id``, the episode outcome agreement with a Wilson interval. Each class gets a verdict:

* ``trusted``: enough evidence and every threshold met. Sim results in this class may replace
  robot runs for the decisions the certificate names.
* ``provisional``: thresholds met, but too little evidence (few policies, few episodes, or the
  real policies are too similar to rank). Use sim to screen, confirm on the robot.
* ``not_trusted``: a threshold failed. Do not use sim results in this class for decisions.

The thresholds are starting points, not validated limits (see ``DEFAULT_THRESHOLDS``).
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from groot_eval.stats import bootstrap_ci, spearman, wilson_interval
from groot_eval.twin_metrics import decision_agreement, mmrv, pearson

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_policies": 5,  # below this, correlation and MMRV are noise
    "min_real_episodes": 20,  # per policy and class, on the robot
    "min_real_spread": 0.10,  # real success rates must span this much to have a ranking to predict
    "min_pearson": 0.80,
    "max_mmrv": 0.10,
    "max_abs_bias": 0.10,  # |sim - real| success rate, upper end of its interval
    "min_decision_agreement": 0.80,  # share of policies the twin calls pass/fail like the robot
    "min_outcome_agreement": 0.70,  # paired episodes with the same outcome (lower interval bound)
}

VERDICTS = ("trusted", "provisional", "not_trusted")


def _ok(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def rows_hash(rows: list[dict[str, Any]]) -> str:
    """Content hash of the evidence, so a certificate names exactly the data it was issued on."""
    blob = json.dumps(sorted(json.dumps(r, sort_keys=True, default=str) for r in rows))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _aggregate(rows: list[dict[str, Any]], class_column: str) -> dict[str, dict[tuple[str, str], list[int]]]:
    """class -> (policy, domain) -> [successes, trials]."""
    out: dict[str, dict[tuple[str, str], list[int]]] = {}
    for r in rows:
        cls = str(r.get(class_column) or "all")
        key = (str(r["policy"]), str(r["domain"]).strip().lower())
        s, n = out.setdefault(cls, {}).setdefault(key, [0, 0])
        if r.get("trials") not in (None, ""):
            out[cls][key] = [s + int(r["successes"]), n + int(r["trials"])]
        else:
            out[cls][key] = [s + int(_ok(r["success"])), n + 1]
    return out


def _paired_outcomes(rows: list[dict[str, Any]], class_column: str) -> dict[str, list[tuple[bool, bool]]]:
    """class -> [(sim_ok, real_ok)] for episodes sharing (policy, scenario_id)."""
    idx: dict[tuple[str, str, str], dict[str, list[bool]]] = {}
    for r in rows:
        sid = r.get("scenario_id")
        if not sid or r.get("trials") not in (None, ""):
            continue
        key = (str(r.get(class_column) or "all"), str(r["policy"]), str(sid))
        idx.setdefault(key, {}).setdefault(str(r["domain"]).strip().lower(), []).append(_ok(r["success"]))
    out: dict[str, list[tuple[bool, bool]]] = {}
    for (cls, _, _), d in idx.items():
        for s_ok, r_ok in zip(d.get("sim", []), d.get("real", [])):
            out.setdefault(cls, []).append((s_ok, r_ok))
    return out


def class_credibility(agg: dict[tuple[str, str], list[int]], pairs: list[tuple[bool, bool]],
                      thresholds: dict[str, float], gate: float | None = None,
                      confidence: float = 0.95) -> dict[str, Any]:
    """Metrics and verdict for one scenario class."""
    th = {**DEFAULT_THRESHOLDS, **thresholds}
    policies = sorted({p for p, _ in agg})
    table: list[dict[str, Any]] = []
    for p in policies:
        s, r = agg.get((p, "sim")), agg.get((p, "real"))
        if not s or not r or not s[1] or not r[1]:
            continue
        table.append({"policy": p, "sim_rate": s[0] / s[1], "sim_n": s[1], "real_rate": r[0] / r[1], "real_n": r[1],
                      "real_ci": wilson_interval(r[0], r[1], confidence)})
    sim_l: list[float] = [float(t["sim_rate"]) for t in table]
    real_l: list[float] = [float(t["real_rate"]) for t in table]
    sim, real = np.array(sim_l, dtype=float), np.array(real_l, dtype=float)
    n = len(table)
    bias = float(np.mean(sim - real)) if n else math.nan
    bias_ci = bootstrap_ci(sim - real, confidence) if n >= 2 else (math.nan, math.nan)
    m: dict[str, Any] = {
        "policies": n,
        "table": table,
        "pearson": pearson(sim_l, real_l) if n >= 2 else math.nan,
        "spearman": spearman(sim_l, real_l) if n >= 2 else math.nan,
        "mmrv": mmrv(sim_l, real_l) if n >= 2 else math.nan,
        "bias": bias,
        "bias_ci": bias_ci,
        "real_spread": float(np.ptp(real)) if n else 0.0,
        "min_real_episodes": min((int(t["real_n"]) for t in table), default=0),
    }
    if gate is not None and n:
        m["decisions"] = decision_agreement([t["sim_rate"] < gate for t in table], [t["real_rate"] < gate for t in table])
    if pairs:
        same = sum(a == b for a, b in pairs)
        m["paired"] = {"episodes": len(pairs), "agreement": same / len(pairs),
                       "agreement_ci": wilson_interval(same, len(pairs), confidence)}

    failed, thin = [], []
    if n >= 2:
        if not (m["pearson"] >= th["min_pearson"]):
            failed.append(f"Pearson r {m['pearson']:.2f} < {th['min_pearson']}")
        if not (m["mmrv"] <= th["max_mmrv"]):
            failed.append(f"MMRV {m['mmrv']:.3f} > {th['max_mmrv']}")
    worst_bias = max(abs(b) for b in bias_ci) if all(math.isfinite(b) for b in bias_ci) else abs(bias)
    if math.isfinite(worst_bias) and worst_bias > th["max_abs_bias"]:
        failed.append(f"bias up to {worst_bias:.2f} (interval {bias_ci[0]:+.2f}..{bias_ci[1]:+.2f}) > {th['max_abs_bias']}")
    dec = m.get("decisions")
    if dec and dec.get("agreement") is not None and dec["agreement"] < th["min_decision_agreement"]:
        failed.append(f"pass/fail agreement {dec['agreement']:.2f} < {th['min_decision_agreement']}")
    pr = m.get("paired")
    if pr and pr["agreement_ci"][0] < th["min_outcome_agreement"]:
        if pr["agreement"] < th["min_outcome_agreement"]:
            failed.append(f"paired outcome agreement {pr['agreement']:.2f} < {th['min_outcome_agreement']}")
        else:
            thin.append(f"paired outcome agreement {pr['agreement']:.2f}, but its interval reaches "
                        f"{pr['agreement_ci'][0]:.2f} ({pr['episodes']} episodes)")
    if n < th["min_policies"]:
        thin.append(f"{n} policies with both sim and real results (need {int(th['min_policies'])})")
    if m["min_real_episodes"] < th["min_real_episodes"]:
        thin.append(f"as few as {m['min_real_episodes']} robot episodes for a policy (need {int(th['min_real_episodes'])})")
    if n and m["real_spread"] < th["min_real_spread"]:
        thin.append(f"real success rates span {m['real_spread']:.2f} (need {th['min_real_spread']}): little to rank")
    m["failed"], m["thin_evidence"] = failed, thin
    m["verdict"] = "not_trusted" if failed or n == 0 else ("provisional" if thin else "trusted")
    return m


def assess(rows: list[dict[str, Any]], class_column: str = "scenario_class", gate: float | None = None,
           thresholds: dict[str, float] | None = None, confidence: float = 0.95) -> dict[str, Any]:
    """Credibility of the twin in every scenario class found in ``rows``."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    agg, pairs = _aggregate(rows, class_column), _paired_outcomes(rows, class_column)
    classes = {c: class_credibility(a, pairs.get(c, []), th, gate, confidence) for c, a in sorted(agg.items())}
    counts = {v: sum(c["verdict"] == v for c in classes.values()) for v in VERDICTS}
    return {"class_column": class_column, "gate": gate, "thresholds": th, "confidence": confidence,
            "classes": classes, "counts": counts, "evidence_hash": rows_hash(rows)}


# -- certificate ------------------------------------------------------------------------------
def make_certificate(result: dict[str, Any], twin: str, twin_version: str, profile: dict[str, Any] | None = None,
                     valid_days: int = 90, decisions: list[str] | None = None) -> dict[str, Any]:
    """The twin certificate: which classes this twin may stand in for, for this robot profile, until when."""
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    return {
        "kind": "groot-eval/twin-certificate",
        "version": 1,
        "twin": twin,
        "twin_version": twin_version,
        "profile": profile or None,
        "issued": now.isoformat(),
        "expires": (now + _dt.timedelta(days=valid_days)).isoformat(),
        "decisions": decisions or ["checkpoint ranking", "regression screening"],
        "evidence_hash": result["evidence_hash"],
        "thresholds": result["thresholds"],
        "classes": {c: {"verdict": m["verdict"], "policies": m["policies"], "pearson": m["pearson"], "mmrv": m["mmrv"],
                        "bias": m["bias"], "reasons": m["failed"] or m["thin_evidence"]}
                    for c, m in result["classes"].items()},
    }


def check_certificate(cert: dict[str, Any], scenario_class: str, profile_hash: str | None = None,
                      twin_version: str | None = None, allow_provisional: bool = False,
                      now: _dt.datetime | None = None) -> tuple[bool, str]:
    """May a sim result in ``scenario_class`` stand in for robot runs under this certificate?"""
    if cert.get("kind") != "groot-eval/twin-certificate":
        return False, "not a twin certificate"
    now = now or _dt.datetime.now(_dt.timezone.utc)
    if now > _dt.datetime.fromisoformat(cert["expires"]):
        return False, f"certificate expired on {cert['expires']}: re-run the paired evaluation"
    if twin_version and twin_version != cert.get("twin_version"):
        return False, f"twin version {twin_version} differs from the certified {cert.get('twin_version')}"
    certified = (cert.get("profile") or {}).get("hash")
    if profile_hash and certified and profile_hash != certified:
        return False, f"robot profile {profile_hash} differs from the certified {certified}: the robot changed"
    entry = cert.get("classes", {}).get(scenario_class)
    if entry is None:
        return False, f"class {scenario_class!r} was not assessed"
    v = entry["verdict"]
    if v == "trusted" or (v == "provisional" and allow_provisional):
        return True, f"class {scenario_class!r} is {v}"
    why = "; ".join(entry.get("reasons") or []) or v
    return False, f"class {scenario_class!r} is {v}: {why}"


# -- report -----------------------------------------------------------------------------------
def _f(x: Any, fmt: str = ".2f") -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def credibility_markdown(result: dict[str, Any], cert: dict[str, Any] | None = None) -> str:
    c = result["counts"]
    lines = ["# Twin credibility by scenario class", "",
             f"{c['trusted']} trusted, {c['provisional']} provisional, {c['not_trusted']} not trusted "
             f"(class column `{result['class_column']}`; evidence `{result['evidence_hash']}`).", ""]
    if cert:
        prof = cert.get("profile") or {}
        lines += [f"Certificate for twin `{cert['twin']}` {cert['twin_version']}"
                  + (f", robot profile `{prof.get('name')}` v{prof.get('version')} (`{prof.get('hash')}`)" if prof else "")
                  + f", valid until {cert['expires'][:10]}.", ""]
    lines += ["| Class | Verdict | Policies | Pearson r | MMRV | Bias (sim - real) | Pass/fail agreement | Paired agreement |",
              "|---|---|---|---|---|---|---|---|"]
    for cls, m in result["classes"].items():
        dec = m.get("decisions") or {}
        pr = m.get("paired")
        lines.append(f"| {cls} | **{m['verdict']}** | {m['policies']} | {_f(m['pearson'])} | {_f(m['mmrv'], '.3f')} | "
                     f"{_f(m['bias'], '+.2f')} [{_f(m['bias_ci'][0], '+.2f')}, {_f(m['bias_ci'][1], '+.2f')}] | "
                     f"{_f(dec.get('agreement'))} | "
                     + (f"{pr['agreement']:.2f} [{pr['agreement_ci'][0]:.2f}, {pr['agreement_ci'][1]:.2f}], n={pr['episodes']}"
                        if pr else "-") + " |")
    lines += ["", "## Why", ""]
    for cls, m in result["classes"].items():
        reasons = m["failed"] + m["thin_evidence"]
        lines.append(f"- **{cls}** ({m['verdict']}): " + ("; ".join(reasons) if reasons else "every threshold met."))
    lines += ["", "Trusted: sim may replace robot runs in this class for the certified decisions. Provisional: screen in "
              "sim, confirm on the robot. Not trusted: do not use sim results in this class for decisions.", ""]
    return "\n".join(lines)
