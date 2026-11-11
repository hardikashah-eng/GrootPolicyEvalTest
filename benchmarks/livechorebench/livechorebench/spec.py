"""LiveChoreBench task specs: load, check, sample and score.

A task spec is one YAML file that both a simulator environment (Isaac Lab or similar) and a real-room runbook
read. Everything that decides success is a predicate on a **world state** (object poses, joint states, timers),
so the same scoring code runs on simulator state and on states measured in the real room (tags, motion capture,
scales, encoders). See ../README.md.

World state, as given to :func:`evaluate`::

    {
      "objects": {"mug": {"pos": [x, y, z], "yaw_deg": 30.0, "tilt_deg": 4.0, "held": False}},
      "regions": {"rack.slots": ["mug"]},            # objects inside each named region
      "joints": {"drawer": 0.004},                   # joint positions (m or rad)
      "targets": {"plate_1": {"pos": [...], "yaw_deg": 0.0}},
      "stable_s": 2.5,                               # seconds the scene has been still
      "fallen": False, "dropped": [], "said": ["Which cup, the red or the blue one?"],
    }
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TRACKS = ("execute", "self_repair", "clarify", "long_horizon", "disturbance", "loco_manipulation")
PREDICATES = ("all", "any", "inside", "tilt_deg", "hold_s", "within", "joint_max", "joint_min",
              "not_fallen", "not_dropped", "said_any", "object_at")
VARIATION_KINDS = ("uniform", "choice")


class SpecError(ValueError):
    """A task spec that cannot be run the same way in sim and in the real room."""


@dataclass
class TaskSpec:
    id: str
    release: str
    track: str
    instruction: list[str]
    scene: dict[str, Any]
    variation: dict[str, Any]
    success: dict[str, Any]
    subgoals: list[dict[str, Any]]
    limits: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    feedback: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    path: str = ""

    @property
    def objects(self) -> dict[str, Any]:
        return dict(self.scene.get("objects", {}))


def load(path: str | Path) -> TaskSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spec = TaskSpec(
        id=raw["id"], release=str(raw["release"]), track=raw["track"],
        instruction=list(raw["instruction"]), scene=raw.get("scene", {}), variation=raw.get("variation", {}),
        success=raw["success"], subgoals=list(raw.get("subgoals", [])), limits=raw.get("limits", {}),
        events=list(raw.get("events", [])), feedback=raw.get("feedback", {}), scoring=raw.get("scoring", {}),
        path=str(path),
    )
    validate(spec)
    return spec


def load_all(root: str | Path) -> list[TaskSpec]:
    return [load(p) for p in sorted(Path(root).glob("*.yaml"))]


# -- checks ---------------------------------------------------------------------------------
def _predicate_kinds(pred: Any) -> list[str]:
    if not isinstance(pred, dict) or len(pred) != 1:
        raise SpecError(f"a predicate is a one-key mapping, got {pred!r}")
    (kind, arg), = pred.items()
    if kind not in PREDICATES:
        raise SpecError(f"unknown predicate {kind!r} (known: {', '.join(PREDICATES)})")
    if kind in ("all", "any"):
        out = [kind]
        for p in arg:
            out += _predicate_kinds(p)
        return out
    return [kind]


def validate(spec: TaskSpec) -> None:
    """Refuse a spec that could not be run and scored the same way in sim and in the real room."""
    if spec.track not in TRACKS:
        raise SpecError(f"{spec.id}: unknown track {spec.track!r}")
    if not spec.instruction or not all(isinstance(s, str) and s.strip() for s in spec.instruction):
        raise SpecError(f"{spec.id}: needs at least one instruction")
    if not spec.objects:
        raise SpecError(f"{spec.id}: scene has no objects")
    for name, obj in spec.objects.items():
        for key in ("usd", "real_sku"):  # every object must exist in both worlds
            if not obj.get(key):
                raise SpecError(f"{spec.id}: object {name!r} has no {key}")
        if not obj.get("fixed") and not isinstance(obj.get("mass_kg"), (int, float)):
            raise SpecError(f"{spec.id}: movable object {name!r} needs mass_kg (sim and real must match)")
    for key, v in spec.variation.items():
        if not isinstance(v, dict) or len(v) != 1 or next(iter(v)) not in VARIATION_KINDS:
            raise SpecError(f"{spec.id}: variation {key!r} must be uniform: [lo, hi] or choice: [...]")
        kind, arg = next(iter(v.items()))
        if kind == "uniform":
            lo, hi = arg
            if not _le(lo, hi):
                raise SpecError(f"{spec.id}: variation {key!r} has lo > hi")
        elif not arg:
            raise SpecError(f"{spec.id}: variation {key!r} has an empty choice")
    _predicate_kinds(spec.success)
    for sg in spec.subgoals:
        if "name" not in sg or "check" not in sg:
            raise SpecError(f"{spec.id}: each subgoal needs name and check")
        _predicate_kinds(sg["check"])
    if not isinstance(spec.limits.get("timeout_s"), (int, float)) or spec.limits["timeout_s"] <= 0:
        raise SpecError(f"{spec.id}: limits.timeout_s is required")
    for ev in spec.events:
        if not ev.get("trigger") or not ev.get("action") or not ev.get("real"):
            raise SpecError(f"{spec.id}: each event needs trigger, action (sim) and real (how a person or rig does it)")
    if spec.track == "self_repair" and not spec.feedback.get("text"):
        raise SpecError(f"{spec.id}: self-repair tasks need feedback.text")
    if spec.track == "disturbance" and not spec.events:
        raise SpecError(f"{spec.id}: disturbance tasks need at least one event")
    if spec.track == "clarify" and "said_any" not in _predicate_kinds(spec.success):
        raise SpecError(f"{spec.id}: clarify tasks must score what the robot says (said_any)")


def _le(lo: Any, hi: Any) -> bool:
    if isinstance(lo, (list, tuple)):
        return all(a <= b for a, b in zip(lo, hi))
    return bool(lo <= hi)


# -- sampling (sim randomizer and real placement sheet) ----------------------------------------
def instance_seed(spec: TaskSpec, seed: int) -> int:
    h = hashlib.sha256(f"{spec.id}:{seed}".encode()).hexdigest()
    return int(h[:12], 16)


def sample(spec: TaskSpec, seed: int) -> dict[str, Any]:
    """One task instance. The same seed gives the same instance in sim and on the placement sheet."""
    rng = random.Random(instance_seed(spec, seed))
    out: dict[str, Any] = {"task": spec.id, "seed": seed, "instruction": rng.choice(spec.instruction)}
    for key in sorted(spec.variation):
        kind, arg = next(iter(spec.variation[key].items()))
        if kind == "choice":
            out[key] = rng.choice(list(arg))
        else:
            lo, hi = arg
            if isinstance(lo, (list, tuple)):
                out[key] = [round(rng.uniform(a, b), 4) for a, b in zip(lo, hi)]
            else:
                out[key] = round(rng.uniform(lo, hi), 4)
    return out


def placement_sheet(spec: TaskSpec, seeds: list[int]) -> str:
    """What a person sets up in the real room for these instances (Markdown)."""
    lines = [f"# Placement sheet: {spec.id}", "", "| Seed | Instruction | Settings |", "|---|---|---|"]
    for s in seeds:
        inst = sample(spec, s)
        settings = "; ".join(f"{k} = {json.dumps(v)}" for k, v in inst.items() if k not in ("task", "seed", "instruction"))
        lines.append(f"| {s} | {inst['instruction']} | {settings} |")
    return "\n".join(lines) + "\n"


# -- scoring (the same code for sim and real) --------------------------------------------------
def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _yaw_err(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def evaluate(pred: dict[str, Any], world: dict[str, Any]) -> bool:
    (kind, arg), = pred.items()
    objs = world.get("objects", {})
    if kind == "all":
        return all(evaluate(p, world) for p in arg)
    if kind == "any":
        return any(evaluate(p, world) for p in arg)
    if kind == "inside":
        return arg["object"] in world.get("regions", {}).get(arg["region"], [])
    if kind == "tilt_deg":
        return float(objs.get(arg["object"], {}).get("tilt_deg", math.inf)) <= float(arg["max"])
    if kind == "hold_s":
        return float(world.get("stable_s", 0.0)) >= float(arg)
    if kind == "within":
        o, t = objs.get(arg["object"]), world.get("targets", {}).get(arg["target"])
        if not o or not t:
            return False
        ok = _dist(o["pos"], t["pos"]) <= float(arg["pos_tol_m"])
        if "yaw_tol_deg" in arg:
            ok = ok and _yaw_err(float(o.get("yaw_deg", 0.0)), float(t.get("yaw_deg", 0.0))) <= float(arg["yaw_tol_deg"])
        return ok
    if kind == "object_at":
        o = objs.get(arg["object"])
        return bool(o) and _dist(o["pos"], arg["pos"]) <= float(arg["tol_m"])
    if kind == "joint_max":
        return float(world.get("joints", {}).get(arg["joint"], math.inf)) <= float(arg["max"])
    if kind == "joint_min":
        return float(world.get("joints", {}).get(arg["joint"], -math.inf)) >= float(arg["min"])
    if kind == "not_fallen":
        return not world.get("fallen", False)
    if kind == "not_dropped":
        return not [d for d in world.get("dropped", []) if arg in ("any", d)]
    if kind == "said_any":
        said = " ".join(world.get("said", [])).lower()
        return any(phrase.lower() in said for phrase in arg)
    raise SpecError(f"unknown predicate {kind!r}")


def score(spec: TaskSpec, world: dict[str, Any]) -> dict[str, Any]:
    """Success, partial credit and per-subgoal results for one finished episode."""
    done = {sg["name"]: evaluate(sg["check"], world) for sg in spec.subgoals}
    success = evaluate(spec.success, world)
    partial = (sum(done.values()) / len(done)) if done else float(success)
    out: dict[str, Any] = {"task": spec.id, "success": success, "partial_credit": 1.0 if success else partial,
                           "subgoals": done}
    rubric = spec.scoring.get("graded")
    if rubric:  # e.g. clarify: full marks for asking, half for a silent but plausible choice
        out["grade"] = next((float(r["score"]) for r in rubric if evaluate(r["when"], world)), 0.0)
    return out
