"""Three-level episode logging: policy, action and joint. An episode without complete logs does not count.

Every task spec declares a ``logging`` block with a rate and fields for each level::

    logging:
      policy: {rate_hz: 10}    # one record per model query
      action: {rate_hz: 50}    # one record per command sent to the controller
      joint:  {rate_hz: 200}   # one record per joint-state sample

The required fields below are always needed; a spec may ask for more (``fields: [...]``). The three levels let
any result be traced end to end: which model and input produced which action chunk (policy), what was
commanded versus applied and whether it was clipped (action), and what every joint actually did (joint).
That is what failure triage and sim-vs-real comparison need.

A log is three columnar streams (lists of equal length per field)::

    {"policy": {"t": [...], "step": [...], ...}, "action": {...}, "joint": {...}}

:func:`check_episode_log` returns findings; :func:`episode_valid` is False if any is an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

LEVELS = ("policy", "action", "joint")
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    # model identity and inputs, so a result can be re-queried; the chunk it returned; how long it took
    "policy": ("t", "step", "model_id", "checkpoint_hash", "instruction", "obs_hash", "action_chunk", "latency_ms"),
    # which policy step it came from; commanded vs applied (after limits and smoothing); whether it was clipped
    "action": ("t", "policy_step", "commanded", "applied", "clipped"),
    # what every joint did
    "joint": ("t", "position", "velocity", "effort"),
}
MIN_RATE_HZ = {"policy": 1.0, "action": 10.0, "joint": 50.0}


@dataclass
class Finding:
    level: str  # error | warn
    check: str
    detail: str


def validate_logging(spec_id: str, logging: dict[str, Any]) -> None:
    """Spec-time check of the logging block (raises SpecError)."""
    from livechorebench.spec import SpecError

    if not logging:
        raise SpecError(f"{spec_id}: logging is required at policy, action and joint level")
    rates = {}
    for lv in LEVELS:
        cfg = logging.get(lv)
        if not isinstance(cfg, dict) or not isinstance(cfg.get("rate_hz"), (int, float)):
            raise SpecError(f"{spec_id}: logging.{lv}.rate_hz is required")
        if cfg["rate_hz"] < MIN_RATE_HZ[lv]:
            raise SpecError(f"{spec_id}: logging.{lv}.rate_hz {cfg['rate_hz']} is below the minimum {MIN_RATE_HZ[lv]}")
        rates[lv] = float(cfg["rate_hz"])
    if not rates["policy"] <= rates["action"] <= rates["joint"]:
        raise SpecError(f"{spec_id}: logging rates must satisfy policy <= action <= joint")


def required_fields(spec_logging: dict[str, Any], level: str) -> list[str]:
    extra = list((spec_logging.get(level) or {}).get("fields") or [])
    return list(REQUIRED_FIELDS[level]) + [f for f in extra if f not in REQUIRED_FIELDS[level]]


def check_episode_log(spec_logging: dict[str, Any], log: dict[str, Any], rate_tolerance: float = 0.9,
                      max_gap_periods: float = 3.0) -> list[Finding]:
    out: list[Finding] = []
    for lv in LEVELS:
        stream = log.get(lv)
        if not stream:
            out.append(Finding("error", f"{lv}.present", f"no {lv}-level log"))
            continue
        missing = [f for f in required_fields(spec_logging, lv) if f not in stream]
        if missing:
            out.append(Finding("error", f"{lv}.fields", f"missing fields: {', '.join(missing)}"))
        t = [float(x) for x in stream.get("t", [])]
        lengths = {f: len(v) for f, v in stream.items()}
        if len(set(lengths.values())) > 1:
            out.append(Finding("error", f"{lv}.lengths", f"fields have different lengths: {lengths}"))
        if len(t) < 2:
            out.append(Finding("error", f"{lv}.samples", f"only {len(t)} record(s)"))
            continue
        if any(b <= a for a, b in zip(t, t[1:])):
            out.append(Finding("error", f"{lv}.time", "timestamps are not strictly increasing"))
        want = float((spec_logging.get(lv) or {}).get("rate_hz", MIN_RATE_HZ[lv]))
        got = (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else math.inf
        if got < rate_tolerance * want:
            out.append(Finding("error", f"{lv}.rate", f"{got:.1f} Hz logged, {want:g} Hz required"))
        gap = max(b - a for a, b in zip(t, t[1:]))
        if gap > max_gap_periods / want:
            out.append(Finding("warn", f"{lv}.gap", f"largest gap {gap * 1000:.0f} ms (> {max_gap_periods:g} periods)"))
    pol, act, jnt = log.get("policy") or {}, log.get("action") or {}, log.get("joint") or {}
    if pol.get("step") is not None and act.get("policy_step") is not None:
        known = set(pol["step"])
        orphans = sum(1 for s in act["policy_step"] if s not in known)
        if orphans:
            out.append(Finding("error", "trace.policy_to_action", f"{orphans} action(s) name a policy step that was not logged"))
    if act.get("t") and jnt.get("t"):
        if float(jnt["t"][0]) > float(act["t"][0]) or float(jnt["t"][-1]) < float(act["t"][-1]):
            out.append(Finding("error", "trace.action_to_joint", "joint log does not cover the whole action log"))
    if act.get("clipped"):
        share = sum(bool(c) for c in act["clipped"]) / len(act["clipped"])
        if share > 0.1:
            out.append(Finding("warn", "action.clipped", f"{share:.0%} of commands were clipped by limits"))
    return out


def episode_valid(findings: list[Finding]) -> bool:
    return not any(f.level == "error" for f in findings)
