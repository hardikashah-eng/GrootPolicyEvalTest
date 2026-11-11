"""Robot profiles: one measured robot, versioned, ready to load in production.

A profile is a YAML file holding everything the twin needs to behave like one
physical robot:

* the robot's limits (the same keys as ``configs/robots/*.yaml``, so a profile
  also works with ``groot-eval transfer --robot``);
* its joint order;
* actuator parameters and an optional learned motor model;
* per-joint wear and its effective age (plus wear on other components, such as
  the gripper, under ``wear.other``);
* sim parameters found by system identification;
* loop timing (the :class:`~groot_eval.scheduler.TimingProfile` format);
* optional fault rates;
* provenance: source traces with hashes, fit quality, tool and date.

``content_hash`` is a SHA-256 over everything else except ``version`` and
``status``, so a loaded profile can be checked against what was calibrated.

Consumers load it directly:

* Isaac Lab: ``WearAwareActuatorCfg(robot_profile="profile.yaml")`` reads
  ``RobotProfile.load(path).wear_for(joint_names)`` (reordered by name).
* Scheduler: ``TimingProfile.load("profile.yaml")`` (the ``timing`` key) or
  ``profile.timing_profile()``.
* CPU twin: ``profile.make_twin()`` returns a ``WearEnv`` and the scenario
  parameters for this unit.
* Transfer checks: ``groot-eval transfer --robot profile.yaml``.

:class:`ProfileRegistry` keeps every version of every robot's profile, with a
``latest`` and an ``approved`` pointer, so production code loads the approved
version and calibration keeps publishing candidates.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.actuators.core import WEAR_KEYS

SCHEMA_VERSION = 1
STATUSES = ("candidate", "approved", "retired")
ACTUATOR_KEYS = ("effort_limit", "stiffness", "damping", "coulomb_friction", "viscous_friction", "velocity_limit")
# Physically meaningful ranges for wear values; outside them a profile is rejected.
WEAR_RANGES: dict[str, tuple[float, float]] = {
    "gain": (0.05, 2.0), "torque_limit": (0.05, 2.0), "thermal_derate": (0.0, 1.0), "thermal_tau_s": (0.1, 1e5),
    "friction": (0.0, 20.0), "damping": (0.0, 20.0), "backlash": (0.0, 0.5), "stiction": (0.0, 5.0),
    "enc_bias": (-0.5, 0.5), "enc_scale": (0.5, 1.5), "enc_noise": (0.0, 0.5), "enc_bias_walk": (0.0, 0.5),
}
_UNHASHED = ("content_hash", "version", "status")


class ProfileError(ValueError):
    """The profile does not follow the schema; ``problems`` lists every issue found."""

    def __init__(self, problems: list[str], where: str = "") -> None:
        self.problems = problems
        super().__init__((f"{where}: " if where else "") + "; ".join(problems))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def content_hash(data: dict[str, Any]) -> str:
    body = {k: v for k, v in data.items() if k not in _UNHASHED}
    blob = json.dumps(_plain(body), sort_keys=True, separators=(",", ":"), allow_nan=True)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(root: str | Path) -> str:
    """One hash over every file of a trace directory (relative path + content)."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file() and "manifest" not in x.name):
        h.update(str(p.relative_to(root)).replace("\\", "/").encode())
        h.update(file_sha256(p).encode())
    return h.hexdigest()


# =====================================================================================
# Wear <-> twin parameters
# =====================================================================================
def wear_lists_from_params(params: dict[str, Any], num_joints: int) -> dict[str, list[float]]:
    """``{"wear.joint[1].friction_scale": 1.4, ...}`` -> ``{"friction": [1.0, 1.4, 1.0], ...}`` (every WEAR_KEY)."""
    out: dict[str, list[float]] = {}
    for key, (kind, eff, default) in WEAR_KEYS.items():
        out[key] = [float(params.get(f"wear.{kind}[{j}].{eff}", default)) for j in range(num_joints)]
    return out


def other_wear_from_params(params: dict[str, Any], num_joints: int) -> dict[str, float]:
    """Wear on components outside the per-joint actuator model (gripper, bus, cameras): ``{"gripper.close_rate_scale": 0.8}``."""
    per_joint = {f"wear.{kind}[{j}].{eff}" for kind, eff, _ in WEAR_KEYS.values() for j in range(num_joints)}
    return {k.removeprefix("wear."): float(v) for k, v in params.items()
            if k.startswith("wear.") and k not in per_joint and k != "wear.age_hours" and isinstance(v, (int, float))}


def params_from_wear_lists(wear: dict[str, list[float]]) -> dict[str, float]:
    """The inverse: twin scenario parameters for :class:`~groot_eval.degradation.wear.WearEnv` (defaults left out)."""
    out: dict[str, float] = {}
    for key, values in wear.items():
        kind, eff, default = WEAR_KEYS[key]
        for j, v in enumerate(values):
            if not math.isclose(float(v), default, rel_tol=0.0, abs_tol=1e-12):
                out[f"wear.{kind}[{j}].{eff}"] = float(v)
    return out


# =====================================================================================
# Profile
# =====================================================================================
@dataclass
class RobotProfile:
    robot_id: str
    joint_names: list[str]
    embodiment: str = ""
    name: str = ""
    operating_hours: float | None = None  # the robot's own hour meter, used by the fleet fit
    control_hz: float | None = None
    state_limits: dict[str, Any] = field(default_factory=dict)
    action_range: list[float] | None = None
    max_action_rate: float | None = None
    actuator: dict[str, Any] = field(default_factory=dict)
    wear: dict[str, Any] = field(default_factory=dict)  # effective_age_hours, values: {key: [per joint]}
    sim: dict[str, Any] = field(default_factory=dict)  # env, params
    timing: dict[str, Any] | None = None
    faults: dict[str, Any] = field(default_factory=dict)  # rates_per_hour: {kind: rate}
    provenance: dict[str, Any] = field(default_factory=dict)
    created: str = field(default_factory=_now)
    schema_version: int = SCHEMA_VERSION
    version: int | None = None
    status: str = "candidate"
    content_hash: str = ""
    path: Path | None = field(default=None, repr=False, compare=False)

    # -- dict / file -------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version, "robot_id": self.robot_id, "version": self.version,
            "status": self.status, "created": self.created, "name": self.name or self.robot_id,
            "embodiment": self.embodiment, "operating_hours": self.operating_hours, "joint_names": list(self.joint_names),
            "control_hz": self.control_hz, "state_limits": self.state_limits, "action_range": self.action_range,
            "max_action_rate": self.max_action_rate, "actuator": self.actuator, "wear": self.wear, "sim": self.sim,
            "timing": self.timing, "faults": self.faults, "provenance": self.provenance,
        }
        d = {k: v for k, v in d.items() if v not in (None, {}, [])}
        d["content_hash"] = content_hash(d)
        out: dict[str, Any] = _plain(d)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any], where: str = "", verify: bool = True) -> RobotProfile:
        problems = validate_dict(data)
        if verify and data.get("content_hash") and data["content_hash"] != content_hash(data):
            problems.append("content_hash does not match the contents (the file was edited after calibration; "
                            "re-publish it, or load with verify=False)")
        if problems:
            raise ProfileError(problems, where)
        known = {f for f in cls.__dataclass_fields__ if f != "path"}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        self.content_hash = d["content_hash"]
        path.write_text(yaml.safe_dump(d, sort_keys=False, width=120), encoding="utf-8")
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> RobotProfile:
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        prof = cls.from_dict(data, str(path), verify)
        prof.path = path
        return prof

    # -- consumers ---------------------------------------------------------------------
    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def _reorder(self, values: list[Any], joint_names: list[str] | None, default: float, missing: str) -> list[float]:
        if joint_names is None:
            return [float(v) for v in values]
        index = {n: i for i, n in enumerate(self.joint_names)}
        absent = [n for n in joint_names if n not in index]
        if absent and missing == "error":
            raise KeyError(f"profile {self.robot_id} has no joints {absent}; it has {self.joint_names}")
        return [float(values[index[n]]) if n in index else default for n in joint_names]

    def wear_for(self, joint_names: list[str] | None = None, missing: str = "error") -> dict[str, list[float]]:
        """Per-joint wear in ``joint_names`` order (the simulator's), for ``WearTable.from_robot_profile``.

        ``missing="default"`` gives joints the profile does not cover a new unit's values instead of raising.
        """
        values = (self.wear or {}).get("values") or {}
        return {k: self._reorder(v, joint_names, WEAR_KEYS[k][2], missing) for k, v in values.items()}

    def actuator_for(self, joint_names: list[str] | None = None) -> dict[str, list[float]]:
        out = {}
        for k in ACTUATOR_KEYS:
            v = self.actuator.get(k)
            if v is not None:
                vals = v if isinstance(v, list) else [v] * self.num_joints
                out[k] = self._reorder(vals, joint_names, math.nan, "error")
        return out

    def motor_model_path(self) -> Path | None:
        mm = self.actuator.get("motor_model")
        if not mm:
            return None
        p = Path(mm["path"])
        return p if p.is_absolute() or self.path is None else self.path.parent / p

    def load_motor_model(self) -> Any:
        from groot_eval.actuators.motor_models import MLPMotorModel

        p = self.motor_model_path()
        if p is None:
            return None
        want = self.actuator["motor_model"].get("sha256")
        if want and file_sha256(p) != want:
            raise ProfileError([f"motor model {p} does not match the sha256 in the profile"])
        return MLPMotorModel.load(p)

    def timing_profile(self) -> Any:
        from groot_eval.scheduler import TimingProfile

        if not self.timing:
            raise ProfileError([f"profile {self.robot_id} has no timing section"])
        return TimingProfile.from_dict(self.timing)

    def twin_params(self) -> dict[str, Any]:
        """Scenario parameters that make the CPU twin this unit: identified sim params plus its wear."""
        values = (self.wear or {}).get("values") or {}
        other = {f"wear.{k}": float(v) for k, v in ((self.wear or {}).get("other") or {}).items()}
        return {**(self.sim.get("params") or {}), **params_from_wear_lists(values), **other}

    def make_twin(self, **env_args: Any) -> tuple[Any, dict[str, Any]]:
        """``(WearEnv(env), scenario params)`` for this robot. Pass the params in every ``Scenario``."""
        from groot_eval.degradation.wear import WearEnv
        from groot_eval.envs import make_env

        env = self.sim.get("env")
        if not env:
            raise ProfileError([f"profile {self.robot_id} names no sim env (sim.env)"])
        return WearEnv(make_env(env, **env_args)), self.twin_params()

    def describe(self) -> str:
        w = self.wear or {}
        lines = [f"{self.robot_id} v{self.version if self.version is not None else '-'} ({self.status}), "
                 f"{self.embodiment or '?'}; {self.num_joints} joints: {', '.join(self.joint_names)}"]
        if self.operating_hours is not None:
            lines.append(f"operating hours {self.operating_hours:g}")
        if w.get("effective_age_hours") is not None:
            lines.append(f"behaves like a {w['effective_age_hours']:g} h unit")
        changed = {k: v for k, v in (w.get("values") or {}).items()
                   if any(not math.isclose(float(x), WEAR_KEYS[k][2], abs_tol=1e-9) for x in v)}
        for k, v in changed.items():
            lines.append(f"  wear {k}: " + ", ".join(f"{x:.3g}" for x in v))
        if self.sim.get("params"):
            lines.append(f"sim params {self.sim['params']}")
        if self.timing:
            lines.append("timing: " + self.timing_profile().describe())
        fit = self.provenance.get("fit") or {}
        if fit.get("score") is not None:
            lines.append(f"fit: replay error {fit.get('baseline_score', math.nan):.3f} (new unit) -> {fit['score']:.3f}")
        lines.append(f"hash {self.content_hash or content_hash(self.to_dict())}")
        return "\n".join(lines)


# =====================================================================================
# Schema validation
# =====================================================================================
def _num_list(v: Any, n: int) -> bool:
    return isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                       and math.isfinite(float(x)) for x in v)


def validate_dict(d: dict[str, Any]) -> list[str]:
    """Every schema problem in a profile dict (empty list = valid)."""
    p: list[str] = []
    if not isinstance(d, dict):
        return ["a profile is a mapping"]
    if d.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        p.append(f"schema_version {d.get('schema_version')} is not supported (expected {SCHEMA_VERSION})")
    if not isinstance(d.get("robot_id"), str) or not d.get("robot_id"):
        p.append("robot_id is required (the robot's serial or asset id)")
    elif any(c in d["robot_id"] for c in "/\\ "):
        p.append("robot_id may not contain slashes or spaces")
    names = d.get("joint_names")
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        p.append("joint_names must be a non-empty list of names")
        names = []
    elif len(set(names)) != len(names):
        p.append("joint_names has duplicates")
    n = len(names)
    known = {"schema_version", "robot_id", "version", "status", "created", "name", "embodiment", "operating_hours",
             "joint_names", "control_hz", "state_limits", "action_range", "max_action_rate", "actuator", "wear", "sim",
             "timing", "faults", "provenance", "content_hash", "camera", "age_hours"}
    extra = set(d) - known
    if extra:
        p.append(f"unknown keys {sorted(extra)}")
    if d.get("status", "candidate") not in STATUSES:
        p.append(f"status must be one of {STATUSES}")
    for key in ("operating_hours", "control_hz", "max_action_rate"):
        v = d.get(key)
        if v is not None and (not isinstance(v, (int, float)) or v < 0):
            p.append(f"{key} must be a non-negative number")
    ar = d.get("action_range")
    if ar is not None and not (isinstance(ar, list) and len(ar) == 2 and ar[0] < ar[1]):
        p.append("action_range must be [low, high]")
    act = d.get("actuator") or {}
    for k, v in act.items():
        if k == "motor_model":
            if not isinstance(v, dict) or not v.get("path"):
                p.append("actuator.motor_model needs a path (and should carry its sha256)")
        elif k not in ACTUATOR_KEYS:
            p.append(f"unknown actuator key {k!r}; known: {list(ACTUATOR_KEYS)}")
        elif not (isinstance(v, (int, float)) or _num_list(v, n)):
            p.append(f"actuator.{k} must be a number or one value per joint ({n})")
    wear = d.get("wear") or {}
    unknown_w = set(wear) - {"effective_age_hours", "values", "other", "source"}
    if unknown_w:
        p.append(f"unknown wear keys {sorted(unknown_w)}")
    for k, v in (wear.get("values") or {}).items():
        if k not in WEAR_KEYS:
            p.append(f"unknown wear value {k!r}; known: {sorted(WEAR_KEYS)}")
        elif not _num_list(v, n):
            p.append(f"wear.values.{k} must have one number per joint ({n})")
        else:
            lo, hi = WEAR_RANGES[k]
            bad = [x for x in v if not lo <= float(x) <= hi]
            if bad:
                p.append(f"wear.values.{k} outside the plausible range [{lo}, {hi}]: {bad}")
    for k, v in (wear.get("other") or {}).items():
        if "." not in str(k) or not isinstance(v, (int, float)) or isinstance(v, bool):
            p.append(f"wear.other.{k} must be <component>.<key>: number")
    sim = d.get("sim") or {}
    if set(sim) - {"env", "env_args", "params"}:
        p.append(f"unknown sim keys {sorted(set(sim) - {'env', 'env_args', 'params'})}")
    if d.get("timing"):
        try:
            from groot_eval.scheduler import TimingProfile

            TimingProfile.from_dict(d["timing"])
        except (KeyError, ValueError, TypeError) as exc:
            p.append(f"timing: {exc}")
    rates = (d.get("faults") or {}).get("rates_per_hour") or {}
    from groot_eval.actuators.core import FaultTable

    for k, v in rates.items():
        if k not in FaultTable.ALL:
            p.append(f"unknown fault kind {k!r}; known: {list(FaultTable.ALL)}")
        elif not isinstance(v, (int, float)) or v < 0:
            p.append(f"faults.rates_per_hour.{k} must be a non-negative number")
    return p


# =====================================================================================
# Registry
# =====================================================================================
class ProfileRegistry:
    """Versioned profiles on disk: ``<root>/<robot_id>/v0001.yaml`` plus ``<root>/registry.json``.

    ``publish`` adds a new version (a candidate) unless its content is identical to the latest;
    ``approve`` points production at one version; ``resolve`` returns ``approved``, ``latest`` or a number.
    Published files are never rewritten: status lives in ``registry.json``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "registry.json"

    def _index(self) -> dict[str, Any]:
        if self.index_path.exists():
            data: dict[str, Any] = json.loads(self.index_path.read_text(encoding="utf-8"))
            return data
        return {"schema_version": SCHEMA_VERSION, "robots": {}}

    def _save(self, idx: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    def robots(self) -> list[str]:
        return sorted(self._index()["robots"])

    def versions(self, robot_id: str) -> list[dict[str, Any]]:
        return list(self._index()["robots"].get(robot_id, {}).get("versions", []))

    def publish(self, profile: RobotProfile, note: str = "") -> tuple[RobotProfile, bool]:
        """Store a new version. Returns ``(profile, created)``; ``created`` is False if nothing changed."""
        idx = self._index()
        entry = idx["robots"].setdefault(profile.robot_id, {"versions": [], "approved": None})
        h = content_hash(profile.to_dict())
        if entry["versions"] and entry["versions"][-1]["content_hash"] == h:
            return self.get(profile.robot_id, entry["versions"][-1]["version"]), False
        version = (entry["versions"][-1]["version"] + 1) if entry["versions"] else 1
        profile.version, profile.status = version, "candidate"
        path = profile.save(self.root / profile.robot_id / f"v{version:04d}.yaml")
        entry["versions"].append({"version": version, "content_hash": profile.content_hash, "created": profile.created,
                                  "file": str(path.relative_to(self.root)), "note": note,
                                  "fit_score": (profile.provenance.get("fit") or {}).get("score")})
        self._save(idx)
        return profile, True

    def approve(self, robot_id: str, version: int, by: str = "") -> RobotProfile:
        prof = self.get(robot_id, version)  # loads and verifies the hash
        idx = self._index()
        entry = idx["robots"][robot_id]
        entry["approved"] = int(version)
        entry.setdefault("history", []).append({"approved": int(version), "by": by, "at": _now()})
        self._save(idx)
        prof.status = "approved"
        return prof

    def get(self, robot_id: str, version: int) -> RobotProfile:
        entry = self._index()["robots"].get(robot_id)
        if not entry:
            raise KeyError(f"no robot {robot_id!r} in {self.root}; have {self.robots()}")
        row = next((v for v in entry["versions"] if v["version"] == int(version)), None)
        if row is None:
            raise KeyError(f"{robot_id} has no version {version}; have {[v['version'] for v in entry['versions']]}")
        prof = RobotProfile.load(self.root / row["file"])
        if prof.content_hash != row["content_hash"]:
            raise ProfileError([f"{row['file']} does not match the hash recorded at publish time"])
        prof.status = "approved" if entry.get("approved") == prof.version else "candidate"
        return prof

    def resolve(self, robot_id: str, which: str | int = "approved") -> RobotProfile:
        entry = self._index()["robots"].get(robot_id)
        if not entry or not entry["versions"]:
            raise KeyError(f"no profiles for {robot_id!r} in {self.root}")
        if which == "approved":
            if entry.get("approved") is None:
                raise KeyError(f"{robot_id} has no approved profile yet (`groot-robot profile approve`)")
            return self.get(robot_id, entry["approved"])
        if which == "latest":
            return self.get(robot_id, entry["versions"][-1]["version"])
        return self.get(robot_id, int(which))

    def all_profiles(self, which: str = "latest") -> list[RobotProfile]:
        out = []
        for rid in self.robots():
            try:
                out.append(self.resolve(rid, which))
            except KeyError:
                continue
        return out


def resolve_profile(spec: str) -> RobotProfile:
    """``path/to/profile.yaml`` or ``registry:<root>:<robot_id>[@approved|@latest|@<n>]``."""
    if spec.startswith("registry:"):
        _, root, rest = spec.split(":", 2)
        rid, _, which = rest.partition("@")
        return ProfileRegistry(root).resolve(rid, which or "approved")
    return RobotProfile.load(spec)


def diff(a: RobotProfile, b: RobotProfile) -> list[str]:
    """What changed between two versions, in words."""
    out = []
    da, db = a.to_dict(), b.to_dict()
    for k in sorted(set(da) | set(db)):
        if k in ("content_hash", "created", "version", "status", "provenance"):
            continue
        if da.get(k) != db.get(k):
            if k == "wear":
                va = (da.get("wear") or {}).get("values") or {}
                vb = (db.get("wear") or {}).get("values") or {}
                for wk in sorted(set(va) | set(vb)):
                    if va.get(wk) != vb.get(wk):
                        out.append(f"wear.{wk}: {va.get(wk)} -> {vb.get(wk)}")
                ea, eb = (da.get("wear") or {}).get("effective_age_hours"), (db.get("wear") or {}).get("effective_age_hours")
                if ea != eb:
                    out.append(f"effective age: {ea} -> {eb} h")
            else:
                out.append(f"{k}: {da.get(k)} -> {db.get(k)}")
    return out
