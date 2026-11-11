"""Real-to-sim calibration checks: does the simulator's scene match the measured real cell?

The fidelity gates (:mod:`groot_eval.fidelity`) compare *recorded behaviour*. Before that
comparison means anything, the static and physical setup of the sim must match the real
cell, or every later gap is a calibration error in disguise. This module checks five
groups of measurements from the real cell against the sim's configuration, cheapest and
most fundamental first, and writes the corrections it can estimate:

====  =====================  =============================================================
R1    scene                  measured vs configured dimensions: table height, object sizes,
                             gravity, control rate, ... (any named number with a tolerance)
R2    robot kinematics       forward kinematics of recorded joint angles (sim model) vs
                             externally measured end-effector positions (laser tracker,
                             mocap, touch probe); fits joint zero offsets and a base shift
R3    cameras                intrinsics and extrinsics, sim config vs real calibration;
                             reprojection of known 3-D points (checkerboard corners,
                             fiducials, a marker on the gripper) through the sim camera vs
                             the pixels detected in real images; fits a corrected pose
R4    object placement       real object start poses vs the sim's: per named scenario
                             (paired), coverage of the sim's randomization ranges,
                             distribution distance, resting on the support surface, overlap
R5    object physics         mass, kinetic and static friction, restitution estimated from
                             standard tests on the real object (weigh, slide, push, tilt,
                             drop) vs the sim's configured values; how well the sim's values
                             reproduce each recorded test; optional replay of the same tests
                             in your own simulator (``physics.sim_rollout``)
====  =====================  =============================================================

Everything is driven by one **calibration bundle** (YAML; tables inline or as CSV paths
relative to the bundle). See ``configs/real2sim/example_bundle.yaml`` for every field.
Sections that are absent are skipped; list them under ``required`` to make a missing
section fail the run (exit 2). Thresholds live in ``configs/real2sim/thresholds.yaml``;
the defaults are starting points, not validated limits.

Conventions: metres, radians in data (degrees where the field name says ``_deg``), robot
base frame. Cameras use the OpenCV frame (x right, y down, z forward); a camera pose is the
camera's position in the base frame plus its orientation as roll/pitch/yaw in degrees,
composed R = Rz(yaw) Ry(pitch) Rx(roll).

The physics models for the standard tests are closed-form (Coulomb friction, constant
restitution). They estimate what the real object *is*; whether your simulator then
*behaves* like that is what ``sim_rollout`` (and later G5 replay) checks, because contact
models, friction-combine modes and solver settings make engines differ from the formulas.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from groot_eval.types import Finding

GRAVITY = 9.81
SECTIONS = ("scene", "kinematics", "cameras", "placement", "physics")
GATE_NAMES = {"scene": "R1_scene", "kinematics": "R2_kinematics", "cameras": "R3_cameras",
              "placement": "R4_placement", "physics": "R5_physics"}
DEFAULTS: dict[str, dict[str, Any]] = {
    "scene": {"default_rel_tol": 0.02},
    "kinematics": {"max_rms_mm": 3.0, "max_error_mm": 8.0, "max_fitted_rms_mm": 1.0},
    "cameras": {"max_translation_mm": 10.0, "max_rotation_deg": 1.0, "max_focal_pct": 2.0,
                "max_principal_px": 5.0, "max_reprojection_px": 3.0, "max_calibration_px": 1.5},
    "placement": {"max_paired_position_mm": 20.0, "max_paired_yaw_deg": 10.0, "min_coverage": 0.95,
                  "max_w1": 0.5, "support_tol_mm": 5.0, "range_margin": 0.0},
    "physics": {"mass_rel_tol": 0.10, "friction_rel_tol": 0.20, "static_friction_rel_tol": 0.20,
                "restitution_abs_tol": 0.10, "max_sim_nrmse": 0.15, "max_fit_nrmse": 0.10},
}


@dataclass
class GateResult:
    gate: str
    status: str  # pass | warn | fail | skip
    required: bool
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"gate": self.gate, "status": self.status, "required": self.required,
                "findings": [f.to_dict() for f in self.findings], "metrics": _plain(self.metrics)}


def _status(findings: list[Finding]) -> str:
    if any(f.level == "error" for f in findings):
        return "fail"
    return "warn" if any(f.level == "warn" for f in findings) else "pass"


# -- loading -------------------------------------------------------------------------------
def load_thresholds(path: str | Path | None) -> dict[str, dict[str, Any]]:
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for k, v in data.items():
            if k not in cfg:
                raise ValueError(f"unknown threshold section {k!r}; known: {sorted(cfg)}")
            cfg[k].update(v or {})
    return cfg


def load_bundle(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    unknown = set(data) - set(SECTIONS) - {"required", "name", "notes"}
    if unknown:
        raise ValueError(f"unknown bundle sections {sorted(unknown)}; known: {SECTIONS}")
    return data, path.parent


def _table(spec: Any, base: Path) -> pd.DataFrame:
    """A table given inline (list of rows) or as a CSV path relative to the bundle."""
    if spec is None:
        return pd.DataFrame()
    if isinstance(spec, str):
        return pd.read_csv(base / spec)
    return pd.DataFrame(list(spec))


def _callable(spec: str) -> Callable:
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"expected module:function, got {spec!r}")
    return getattr(importlib.import_module(module), name)


# -- R1 scene ------------------------------------------------------------------------------
def check_scene(section: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    out, metrics, corr = [], {}, {}
    for name, spec in section.items():
        sim, real = float(spec["sim"]), float(spec["real"])
        if "tol" in spec:
            tol, how = float(spec["tol"]), f"tolerance {spec['tol']}"
            bad = abs(sim - real) > tol
        else:
            rel = float(spec.get("rel_tol", cfg["default_rel_tol"]))
            tol, how = rel * max(abs(real), 1e-12), f"tolerance {100 * rel:g}%"
            bad = abs(sim - real) > tol
        unit = spec.get("unit", "")
        metrics[name] = {"sim": sim, "real": real, "diff": sim - real}
        msg = f"{name}: sim {sim:g}{unit}, measured {real:g}{unit} (diff {sim - real:+.4g}{unit}, {how})"
        out.append(Finding("error" if bad else "ok", "measurement", msg))
        if bad:
            corr[name] = real
    return out, {"items": metrics, "corrections": corr}


# -- R2 kinematics ---------------------------------------------------------------------------
def resolve_fk(spec: str, n_joints: int) -> Callable[[np.ndarray], np.ndarray]:
    """``arm3`` (the built-in planar arm, equal links summing to 0.9 m) or ``module:function``."""
    if spec == "arm3":
        from groot_eval.envs.arm3 import REACH, arm_fk

        lengths = np.full(n_joints, REACH / n_joints)
        return lambda q: arm_fk(np.asarray(q, dtype=float), lengths)[-1]
    fn = _callable(spec)
    return lambda q: np.asarray(fn(np.asarray(q, dtype=float)), dtype=float).reshape(-1)


def _gauss_newton(residual: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, iters: int = 50,
                  eps: float = 1e-6) -> np.ndarray:
    """Levenberg-Marquardt with a numerical Jacobian (small problems only)."""
    x, lam = x0.astype(float).copy(), 1e-3
    r = residual(x)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((r.size, x.size))
        for i in range(x.size):
            d = np.zeros_like(x)
            d[i] = eps
            J[:, i] = (residual(x + d) - r) / eps
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        x_new = x + step
        r_new = residual(x_new)
        c_new = float(r_new @ r_new)
        if c_new < cost:
            x, r, cost, lam = x_new, r_new, c_new, lam * 0.3
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            lam *= 10
            if lam > 1e8:
                break
    return x


def check_kinematics(section: dict[str, Any], base: Path, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    df = _table(section.get("samples"), base)
    qcols = sorted((c for c in df.columns if c.startswith("q")), key=lambda c: int(c[1:]))
    pcols = [c for c in ("x", "y", "z") if c in df.columns]
    if not qcols or not pcols or df.empty:
        return [Finding("error", "samples", "kinematics.samples needs columns q0..qN and measured x, y[, z]")], {}
    Q, P = df[qcols].to_numpy(float), df[pcols].to_numpy(float)
    raw_fk = resolve_fk(str(section.get("fk", "arm3")), len(qcols))
    off0 = np.asarray(section.get("joint_offsets_rad") or np.zeros(len(qcols)), dtype=float)
    shift0 = np.asarray(section.get("base_shift_m") or np.zeros(len(pcols)), dtype=float)

    def fk(q: np.ndarray) -> np.ndarray:  # the sim's model, including its current calibration
        return raw_fk(q + off0)[: len(pcols)] + shift0

    pred = np.array([fk(q) for q in Q])
    err = np.linalg.norm(pred - P, axis=1) * 1000
    rms, worst = float(np.sqrt(np.mean(err ** 2))), float(err.max())
    out = [Finding("error" if rms > cfg["max_rms_mm"] else "ok", "fk.rms",
                   f"sim FK vs measured end effector: RMS {rms:.2f} mm over {len(df)} poses (limit {cfg['max_rms_mm']} mm)"),
           Finding("error" if worst > cfg["max_error_mm"] else "ok", "fk.max",
                   f"worst pose error {worst:.2f} mm (limit {cfg['max_error_mm']} mm)")]
    n, d = len(qcols), len(pcols)

    def resid(x: np.ndarray) -> np.ndarray:
        dq, shift = x[:n], x[n:]
        return (np.array([fk(q + dq)[:d] for q in Q]) + shift - P).reshape(-1)

    x = _gauss_newton(resid, np.zeros(n + d))
    fit_err = np.linalg.norm(resid(x).reshape(-1, d), axis=1) * 1000
    fit_rms = float(np.sqrt(np.mean(fit_err ** 2)))
    offsets, shift = x[:n], x[n:]
    if len(df) < n + d + 2:
        out.append(Finding("warn", "fit", f"only {len(df)} poses for {n + d} unknowns; the fit is not trustworthy"))
    out.append(Finding("ok" if fit_rms <= cfg["max_fitted_rms_mm"] else "warn", "fit",
                       f"after fitting further joint zero offsets {np.round(np.degrees(offsets), 3).tolist()} deg and base shift "
                       f"{np.round(shift * 1000, 2).tolist()} mm: RMS {fit_rms:.2f} mm"
                       + ("" if fit_rms <= cfg["max_fitted_rms_mm"] else
                          " (offsets alone do not explain the error: check link lengths, the URDF, or the measurement setup)")))
    corr = {}
    if rms > cfg["max_rms_mm"] or worst > cfg["max_error_mm"]:
        corr = {"joint_offsets_rad": np.round(off0 + offsets, 6).tolist(), "base_shift_m": np.round(shift0 + shift, 6).tolist()}
    return out, {"rms_mm": rms, "max_mm": worst, "fitted_rms_mm": fit_rms, "fitted_offsets_rad": offsets.tolist(),
                 "fitted_base_shift_m": shift.tolist(), "corrections": corr}


# -- R3 cameras ------------------------------------------------------------------------------
def rotation(rpy_deg: Any) -> np.ndarray:
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=float))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def rpy_from_matrix(R: np.ndarray) -> list[float]:
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _rotvec(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


@dataclass
class Camera:
    position: np.ndarray
    R: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Camera:
        return cls(np.asarray(d["position"], dtype=float), rotation(d.get("rpy_deg", [0, 0, 0])), float(d["fx"]),
                   float(d.get("fy", d["fx"])), float(d["cx"]), float(d["cy"]), d.get("width"), d.get("height"))

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xc = (np.asarray(X, dtype=float) - self.position) @ self.R  # = R^T (X - t) per row
        z = Xc[:, 2]
        valid = z > 1e-6
        zs = np.where(valid, z, 1.0)
        uv = np.stack([self.fx * Xc[:, 0] / zs + self.cx, self.fy * Xc[:, 1] / zs + self.cy], axis=1)
        return uv, valid

    def to_dict(self) -> dict[str, Any]:
        out = {"position": np.round(self.position, 5).tolist(), "rpy_deg": np.round(rpy_from_matrix(self.R), 4).tolist(),
               "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}
        if self.width:
            out.update(width=self.width, height=self.height)
        return out


def _reproj(cam: Camera, X: np.ndarray, uv: np.ndarray) -> tuple[float, float, int]:
    pred, valid = cam.project(X)
    e = np.linalg.norm(pred - uv, axis=1)
    e = np.where(valid, e, np.inf)
    finite = e[np.isfinite(e)]
    rms = float(np.sqrt(np.mean(finite ** 2))) if finite.size else math.inf
    return rms, float(finite.max()) if finite.size else math.inf, int((~valid).sum())


def fit_extrinsics(cam: Camera, X: np.ndarray, uv: np.ndarray) -> Camera:
    """Keep the intrinsics, adjust the camera pose to minimize reprojection error."""

    def resid(p: np.ndarray) -> np.ndarray:
        c = Camera(cam.position + p[:3], cam.R @ _rotvec(p[3:]), cam.fx, cam.fy, cam.cx, cam.cy)
        pred, valid = c.project(X)
        return np.where(valid[:, None], pred - uv, 1e3).reshape(-1)

    p = _gauss_newton(resid, np.zeros(6), iters=100)
    return Camera(cam.position + p[:3], cam.R @ _rotvec(p[3:]), cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height)


def _pose_delta(a: Camera, b: Camera) -> tuple[float, float]:
    dt = float(np.linalg.norm(a.position - b.position)) * 1000
    Rd = a.R.T @ b.R
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rd) - 1) / 2))))
    return dt, ang


def check_cameras(section: dict[str, Any], base: Path, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    metrics: dict[str, Any] = {}
    corr: dict[str, Any] = {}
    for name, spec in section.items():
        m: dict[str, Any] = {}
        sim = Camera.from_dict(spec["sim"])
        real = Camera.from_dict(spec["real"]) if spec.get("real") else None
        if real is not None:
            if sim.width and real.width and (sim.width, sim.height) != (real.width, real.height):
                out.append(Finding("error", f"{name}.resolution", f"sim renders {sim.width}x{sim.height}, real camera "
                                   f"is {real.width}x{real.height}: intrinsics and policy inputs will not match"))
            dt, ang = _pose_delta(sim, real)
            fpct = 100 * max(abs(sim.fx - real.fx) / real.fx, abs(sim.fy - real.fy) / real.fy)
            dpp = math.hypot(sim.cx - real.cx, sim.cy - real.cy)
            m.update(translation_mm=dt, rotation_deg=ang, focal_pct=fpct, principal_px=dpp)
            out.append(Finding("error" if dt > cfg["max_translation_mm"] else "ok", f"{name}.translation",
                               f"sim camera is {dt:.1f} mm from the calibrated position (limit {cfg['max_translation_mm']} mm)"))
            out.append(Finding("error" if ang > cfg["max_rotation_deg"] else "ok", f"{name}.rotation",
                               f"sim camera orientation differs by {ang:.2f} deg (limit {cfg['max_rotation_deg']} deg)"))
            out.append(Finding("error" if fpct > cfg["max_focal_pct"] else "ok", f"{name}.focal",
                               f"focal length differs by {fpct:.2f}% (limit {cfg['max_focal_pct']}%): field of view"))
            out.append(Finding("error" if dpp > cfg["max_principal_px"] else "ok", f"{name}.principal_point",
                               f"principal point differs by {dpp:.1f} px (limit {cfg['max_principal_px']} px)"))
        obs = _table(spec.get("observations"), base)
        if not obs.empty:
            need = {"X", "Y", "Z", "u", "v"}
            if not need <= set(obs.columns):
                out.append(Finding("error", f"{name}.observations", f"observations need columns {sorted(need)}"))
            else:
                X, uv = obs[["X", "Y", "Z"]].to_numpy(float), obs[["u", "v"]].to_numpy(float)
                rms, worst, behind = _reproj(sim, X, uv)
                m.update(sim_reprojection_px=rms, sim_reprojection_max_px=worst, points=len(obs))
                out.append(Finding("error" if rms > cfg["max_reprojection_px"] or behind else "ok", f"{name}.reprojection",
                                   f"{len(obs)} known points through the sim camera land {rms:.2f} px RMS (max {worst:.1f}) from "
                                   f"where the real camera saw them (limit {cfg['max_reprojection_px']} px)"
                                   + (f"; {behind} behind the sim camera" if behind else "")))
                if real is not None:
                    rrms, _, _ = _reproj(real, X, uv)
                    m["calibration_reprojection_px"] = rrms
                    out.append(Finding("warn" if rrms > cfg["max_calibration_px"] else "ok", f"{name}.calibration",
                                       f"the real calibration itself reprojects at {rrms:.2f} px RMS "
                                       f"(limit {cfg['max_calibration_px']} px); "
                                       + ("recalibrate before trusting it" if rrms > cfg["max_calibration_px"] else "it is consistent")))
                if len(obs) >= 6:
                    fitted = fit_extrinsics(sim, X, uv)
                    frms, _, _ = _reproj(fitted, X, uv)
                    fdt, fang = _pose_delta(sim, fitted)
                    m.update(fitted_reprojection_px=frms, correction_translation_mm=fdt, correction_rotation_deg=fang)
                    out.append(Finding("ok" if frms <= cfg["max_reprojection_px"] else "warn", f"{name}.fit",
                                       f"moving the sim camera {fdt:.1f} mm and {fang:.2f} deg brings reprojection to {frms:.2f} px"
                                       + ("" if frms <= cfg["max_reprojection_px"] else
                                          "; the pose alone does not explain it: check intrinsics, distortion or point detection")))
                    if rms > cfg["max_reprojection_px"]:
                        corr[name] = fitted.to_dict()
                else:
                    out.append(Finding("warn", f"{name}.fit", "fewer than 6 points: no pose correction estimated"))
        if real is not None and name not in corr and any(f.level == "error" for f in out if f.check.startswith(name)):
            corr[name] = real.to_dict()
        if not spec.get("real") and obs.empty:
            out.append(Finding("warn", name, "neither a real calibration nor observations given; nothing checked"))
        metrics[name] = m
    return out, {"cameras": metrics, "corrections": corr}


# -- R4 placement ----------------------------------------------------------------------------
POSE_KEYS = ("x", "y", "z", "yaw_deg")


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    u = np.linspace(0.0025, 0.9975, 400)
    w = float(np.mean(np.abs(np.quantile(a, u) - np.quantile(b, u))))
    pooled = math.sqrt((np.var(a) + np.var(b)) / 2)
    return w / pooled if pooled > 1e-12 else (0.0 if w < 1e-12 else math.inf)


def check_placement(section: dict[str, Any], base: Path, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    from groot_eval.randomization import SweepConfig

    real = _table(section.get("real"), base)
    if real.empty or "object" not in real.columns:
        return [Finding("error", "real", "placement.real needs rows with object and x, y[, z, yaw_deg]")], {}
    sim_cfg = SweepConfig.load(base / section["sim_config"]) if section.get("sim_config") else None
    named_cfg = SweepConfig.load(base / section["sim_scenarios"]) if section.get("sim_scenarios") else None
    if sim_cfg is None and named_cfg is None:
        return [Finding("error", "sim", "placement needs sim_config (randomized ranges) and/or sim_scenarios "
                                        "(named scenarios to pair with)")], {}
    specs = {p.name: p for p in sim_cfg.params} if sim_cfg else {}
    named = {s.name: s.params for s in named_cfg.named} if named_cfg else {}
    sim_samples = [s.params for s in sim_cfg.scenarios()] if sim_cfg and not sim_cfg.named else []
    out: list[Finding] = []
    metrics: dict[str, Any] = {}
    widen: dict[str, Any] = {}
    margin = float(cfg["range_margin"])
    for obj, rows in real.groupby("object"):
        m: dict[str, Any] = {"poses": len(rows)}
        keys = [k for k in POSE_KEYS if k in rows.columns]
        # paired, by scenario name
        if named and "scenario" in rows.columns:
            dpos, dyaw, missing = [], [], 0
            for _, r in rows.iterrows():
                p = named.get(str(r["scenario"]))
                if p is None:  # a placement outside the named test set (e.g. logged during data collection)
                    continue
                if f"object.{obj}.x" not in p:
                    missing += 1
                    continue
                d = [float(r[k]) - float(p.get(f"object.{obj}.{k}", r[k])) for k in ("x", "y", "z") if k in keys]
                dpos.append(math.hypot(*d) * 1000)
                if "yaw_deg" in keys and f"object.{obj}.yaw_deg" in p:
                    dyaw.append(abs((float(r["yaw_deg"]) - float(p[f"object.{obj}.yaw_deg"]) + 180) % 360 - 180))
            if dpos:
                worst = max(dpos)
                m.update(paired=len(dpos), paired_position_mm_max=worst, paired_position_mm_mean=float(np.mean(dpos)))
                out.append(Finding("error" if worst > cfg["max_paired_position_mm"] else "ok", f"{obj}.paired_position",
                                   f"real start position vs the sim's for the same scenario: mean {np.mean(dpos):.1f} mm, worst "
                                   f"{worst:.1f} mm over {len(dpos)} scenarios (limit {cfg['max_paired_position_mm']} mm)"))
                if dyaw:
                    wy = max(dyaw)
                    m["paired_yaw_deg_max"] = wy
                    out.append(Finding("error" if wy > cfg["max_paired_yaw_deg"] else "ok", f"{obj}.paired_yaw",
                                       f"worst yaw difference {wy:.1f} deg (limit {cfg['max_paired_yaw_deg']} deg)"))
            if missing:
                out.append(Finding("warn", f"{obj}.paired", f"{missing} named scenarios place {obj} in the real cell but "
                                   f"set no object.{obj}.* pose in the sim: the sim uses its default placement"))
        # coverage of the sim's randomization ranges
        inside = np.ones(len(rows), dtype=bool)
        ranged = []
        for k in keys:
            spec = specs.get(f"object.{obj}.{k}")
            if spec is None or spec.kind not in ("uniform", "loguniform", "fixed"):
                continue
            lo, hi = (spec.args[0], spec.args[1]) if spec.kind != "fixed" else (spec.args[0], spec.args[0])
            pad = margin * (hi - lo)
            vals = rows[k].to_numpy(float)
            ok = (vals >= lo - pad - 1e-9) & (vals <= hi + pad + 1e-9)
            inside &= ok
            ranged.append(k)
            if not ok.all():
                widen[f"object.{obj}.{k}"] = {"uniform": [round(float(min(lo, vals.min())), 4), round(float(max(hi, vals.max())), 4)]}
        if ranged:
            cov = float(inside.mean())
            m["coverage"] = cov
            out.append(Finding("error" if cov < cfg["min_coverage"] else "ok", f"{obj}.coverage",
                               f"{100 * cov:.1f}% of real placements fall inside the sim's randomization ranges for "
                               f"{', '.join(ranged)} (limit {100 * cfg['min_coverage']:.0f}%)"
                               + ("" if cov >= cfg["min_coverage"] else "; the sim never trains or tests these placements")))
        # distribution distance against sampled sim scenarios
        for k in keys:
            sim_vals = np.array([float(s[f"object.{obj}.{k}"]) for s in sim_samples if f"object.{obj}.{k}" in s])
            if sim_vals.size >= 5 and len(rows) >= 5 and np.ptp(sim_vals) > 0:
                w = _w1(sim_vals, rows[k].to_numpy(float))
                m[f"w1.{k}"] = w
                if w > cfg["max_w1"]:
                    out.append(Finding("warn", f"{obj}.distribution", f"{k}: real placements are distributed differently from "
                                       f"the sim's ({w:.2f} pooled std; limit {cfg['max_w1']}): the sim samples the wrong region"))
        # resting on the support surface
        support = section.get("support_height")
        size = (section.get("objects") or {}).get(obj, {})
        if support is not None and "z" in keys and "height" in size:
            expect = float(support) + float(size["height"]) / 2
            dz = (rows["z"].to_numpy(float) - expect) * 1000
            worst = float(np.max(np.abs(dz)))
            m["support_error_mm_max"] = worst
            out.append(Finding("error" if worst > cfg["support_tol_mm"] else "ok", f"{obj}.support",
                               f"measured object centre height vs resting on the surface: worst {worst:.1f} mm, median "
                               f"{np.median(dz):+.1f} mm (positive = floating, negative = sunk; limit {cfg['support_tol_mm']} mm)"
                               + ("; fix the pose source before importing these placements into the sim"
                                  if worst > cfg["support_tol_mm"] else "")))
        metrics[str(obj)] = m
    # overlapping objects in the same scenario
    radii = {o: float(s["radius"]) for o, s in (section.get("objects") or {}).items() if "radius" in s}
    if radii and "scenario" in real.columns:
        overlaps = []
        for scen, rows in real.groupby("scenario"):
            rs = rows[rows["object"].isin(radii)]
            pts = rs[["x", "y"]].to_numpy(float)
            names = rs["object"].tolist()
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    if np.linalg.norm(pts[i] - pts[j]) < radii[names[i]] + radii[names[j]]:
                        overlaps.append(f"{scen}: {names[i]}/{names[j]}")
        out.append(Finding("error" if overlaps else "ok", "overlap",
                           f"{len(overlaps)} scenario(s) with interpenetrating objects" + (f": {overlaps[:5]}" if overlaps else "")))
    return out, {"objects": metrics, "corrections": {"widen_ranges": widen} if widen else {}}


# -- R5 physics ------------------------------------------------------------------------------
def slide_x(t: np.ndarray, v0: float, mu: float, g: float = GRAVITY) -> np.ndarray:
    """Released at speed v0 on a flat surface, decelerating at mu*g until it stops."""
    a = max(mu, 1e-9) * g
    ts = v0 / a
    tc = np.minimum(t, ts)
    return v0 * tc - 0.5 * a * tc ** 2


def push_x(t: np.ndarray, force: float, duration: float, mass: float, mu: float, mu_s: float | None = None,
           g: float = GRAVITY) -> np.ndarray:
    """From rest, constant horizontal force for ``duration``, then released."""
    if force <= (mu_s if mu_s is not None else mu) * mass * g:
        return np.zeros_like(t)
    a1 = force / mass - mu * g
    x1, v1 = 0.5 * a1 * duration ** 2, a1 * duration
    return np.where(t <= duration, 0.5 * a1 * np.minimum(t, duration) ** 2, x1 + slide_x(np.maximum(t - duration, 0.0), v1, mu, g))


def bounce_heights(h0: float, e: float, n: int) -> np.ndarray:
    return h0 * e ** (2 * np.arange(1, n + 1))


def _nrmse(pred: np.ndarray, obs: np.ndarray) -> float:
    scale = float(np.ptp(obs)) or float(np.max(np.abs(obs))) or 1.0
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(obs)) ** 2)) / scale)


def _predict(test: dict[str, Any], p: dict[str, float]) -> np.ndarray | None:
    kind = test["kind"]
    if kind == "slide":
        return slide_x(np.asarray(test["t"], float), float(test["v0"]), p["friction"])
    if kind == "push":
        return push_x(np.asarray(test["t"], float), float(test["force"]), float(test["duration"]), p["mass"],
                      p["friction"], p.get("static_friction"))
    if kind == "drop":
        return bounce_heights(float(test["h0"]), p["restitution"], len(test["bounce_heights"]))
    if kind == "tilt":
        return np.array([math.degrees(math.atan(p["static_friction"]))])
    if kind == "weigh":
        return np.array([p["mass"]])
    raise ValueError(f"unknown physics test kind {kind!r}; use weigh, slide, push, tilt or drop")


def _observed(test: dict[str, Any]) -> np.ndarray:
    kind = test["kind"]
    if kind in ("slide", "push"):
        return np.asarray(test["x"], float)
    if kind == "drop":
        return np.asarray(test["bounce_heights"], float)
    if kind == "tilt":
        return np.array([float(test["angle_deg"])])
    return np.array([float(test["mass"])])


def _golden(f: Callable[[float], float], lo: float, hi: float, iters: int = 80) -> float:
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return (a + b) / 2


def _sse(test: dict[str, Any], params: Callable[[Any], dict[str, float]]) -> Callable[[Any], float]:
    """Sum of squared errors of one test's model prediction, as a function of the free parameter(s)."""
    obs = _observed(test)
    return lambda x: float(np.sum((_predict(test, params(x)) - obs) ** 2))


def estimate_object(tests: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Physical parameters of one object from its tests: value, spread across tests, number of tests."""
    est: dict[str, list[float]] = {"mass": [], "friction": [], "static_friction": [], "restitution": []}
    for t in tests:
        if t["kind"] == "weigh":
            est["mass"].append(float(t["mass"]))
        elif t["kind"] == "tilt":
            est["static_friction"].append(math.tan(math.radians(float(t["angle_deg"]))))
        elif t["kind"] == "drop":
            h = np.asarray([float(t["h0"])] + list(t["bounce_heights"]), float)
            h = h[: int(np.argmax(h <= 0)) if np.any(h <= 0) else h.size]  # stop at the first non-positive reading
            if h.size >= 2:  # least squares on log heights: h_k = h0 * e^(2k)
                k = np.arange(h.size)
                est["restitution"].append(float(math.exp(np.polyfit(k, np.log(h), 1)[0] / 2)))
        elif t["kind"] == "slide":
            est["friction"].append(_golden(_sse(t, lambda p: {"friction": p}), 0.01, 2.0))
    mu = float(np.mean(est["friction"])) if est["friction"] else None
    for t in tests:
        if t["kind"] != "push":
            continue
        if mu is not None:  # friction known from slides: push gives the mass (log grid, then refine)
            cost = _sse(t, lambda m, mu=mu: {"mass": m, "friction": mu})
            grid = np.geomspace(0.01, 50.0, 200)
            i = int(np.argmin([cost(m) for m in grid]))
            est["mass"].append(_golden(cost, grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]))
        else:  # mass and friction together: joint grid
            joint = _sse(t, lambda mass_mu: {"mass": mass_mu[0], "friction": mass_mu[1]})
            best = min((joint((m, u)), m, u) for m in np.geomspace(0.02, 50, 60) for u in np.linspace(0.02, 1.5, 60))
            est["mass"].append(best[1])
            est["friction"].append(best[2])
    return {k: {"value": float(np.mean(v)), "spread": float(np.std(v)), "n": len(v)} for k, v in est.items() if v}


def check_physics(section: dict[str, Any], base: Path, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    tests = section.get("tests") or []
    if isinstance(tests, str):
        with open(base / tests, encoding="utf-8") as fh:
            tests = yaml.safe_load(fh) or []
    sim_params = section.get("sim_params") or {}
    rollout_fn = _callable(section["sim_rollout"]) if section.get("sim_rollout") else None
    rollout_args = dict(section.get("sim_rollout_args") or {})
    rollout = (lambda t, p: rollout_fn(t, p, **rollout_args)) if rollout_fn else None
    out: list[Finding] = []
    metrics: dict[str, Any] = {}
    corr: dict[str, Any] = {}
    tol = {"mass": ("rel", cfg["mass_rel_tol"]), "friction": ("rel", cfg["friction_rel_tol"]),
           "static_friction": ("rel", cfg["static_friction_rel_tol"]), "restitution": ("abs", cfg["restitution_abs_tol"])}
    by_obj: dict[str, list[dict[str, Any]]] = {}
    for t in tests:
        by_obj.setdefault(str(t["object"]), []).append(t)
    for obj, ts in by_obj.items():
        est = estimate_object(ts)
        sim = {k: float(v) for k, v in (sim_params.get(obj) or {}).items()}
        m: dict[str, Any] = {"estimated": est, "sim": sim, "tests": len(ts)}
        fix: dict[str, float] = {}
        for k, e in est.items():
            spread = f" ± {e['spread']:.3g}" if e["n"] > 1 else ""
            if k not in sim:
                out.append(Finding("warn", f"{obj}.{k}", f"measured {k} {e['value']:.3g}{spread} (n={e['n']}) but the sim "
                                   "config does not set it: the engine default is being used"))
                fix[k] = round(e["value"], 4)
                continue
            how, lim = tol[k]
            diff = sim[k] - e["value"]
            bad = abs(diff) > (lim * abs(e["value"]) if how == "rel" else lim)
            out.append(Finding("error" if bad else "ok", f"{obj}.{k}",
                               f"sim {k} {sim[k]:.3g} vs measured {e['value']:.3g}{spread} (n={e['n']}); "
                               f"diff {diff:+.3g} (limit {100 * lim:.0f}%)" if how == "rel" else
                               f"sim {k} {sim[k]:.3g} vs measured {e['value']:.3g}{spread} (n={e['n']}); diff {diff:+.3g} (limit {lim})"))
            if bad:
                fix[k] = round(e["value"], 4)
        mu, mus = est.get("friction", {}).get("value"), est.get("static_friction", {}).get("value")
        if mu is not None and mus is not None and mus < mu * 0.95:
            out.append(Finding("warn", f"{obj}.friction_order", f"static friction {mus:.3g} is below kinetic {mu:.3g}: "
                               "physically odd; check the tilt test (vibration, rolling) or the slide timing"))
        # how well do the sim's values and the fitted values reproduce each recorded test
        fitted = {k: v["value"] for k, v in est.items()}
        sim_full = {**fitted, **sim}
        rows = []
        for i, t in enumerate(ts):
            if t["kind"] in ("weigh", "tilt"):
                continue
            obs = _observed(t)
            need = {"slide": ["friction"], "push": ["mass", "friction"], "drop": ["restitution"]}[t["kind"]]
            if not all(k in fitted for k in need):
                continue
            e_fit = _nrmse(_predict(t, fitted), obs)
            e_sim = _nrmse(_predict(t, sim_full), obs) if all(k in sim_full for k in need) else math.nan
            row = {"test": i, "kind": t["kind"], "fit_nrmse": e_fit, "sim_params_nrmse": e_sim}
            if rollout is not None:
                pred = np.asarray(rollout(t, sim_full), float).reshape(-1)
                row["sim_rollout_nrmse"] = _nrmse(pred[: obs.size], obs) if pred.size >= obs.size else math.inf
            rows.append(row)
        m["test_errors"] = rows
        if rows:
            worst_fit = max(r["fit_nrmse"] for r in rows)
            out.append(Finding("ok" if worst_fit <= cfg["max_fit_nrmse"] else "warn", f"{obj}.model",
                               f"the fitted Coulomb/restitution model reproduces the recorded tests to {worst_fit:.3f} nRMSE "
                               f"(worst; limit {cfg['max_fit_nrmse']})" + ("" if worst_fit <= cfg["max_fit_nrmse"] else
                               ": the object does something the simple model lacks (rolling, tipping, rate-dependent friction)")))
            sim_errs = [r["sim_params_nrmse"] for r in rows if math.isfinite(r["sim_params_nrmse"])]
            if sim_errs:
                ws = max(sim_errs)
                out.append(Finding("error" if ws > cfg["max_sim_nrmse"] else "ok", f"{obj}.sim_params_replay",
                                   f"the sim's configured values reproduce the recorded tests to {ws:.3f} nRMSE (worst; limit "
                                   f"{cfg['max_sim_nrmse']})"))
            if rollout is not None:
                wr = max(r["sim_rollout_nrmse"] for r in rows)
                out.append(Finding("error" if wr > cfg["max_sim_nrmse"] else "ok", f"{obj}.sim_rollout",
                                   f"your simulator, run with these parameters, reproduces the recorded tests to {wr:.3f} nRMSE "
                                   f"(worst; limit {cfg['max_sim_nrmse']})" + ("" if wr <= cfg["max_sim_nrmse"] else
                                   ": the engine does not behave like the measured object even with matching parameters "
                                   "(friction combine mode, contact stiffness, solver substeps)")))
        if fix:
            corr[obj] = fix
        metrics[obj] = m
    if not by_obj:
        out.append(Finding("warn", "tests", "no physics tests in the bundle"))
    return out, {"objects": metrics, "corrections": corr}


# -- runner ----------------------------------------------------------------------------------
def run_checks(bundle: dict[str, Any], base: Path, cfg: dict[str, dict[str, Any]]) -> list[GateResult]:
    required = set(bundle.get("required") or [])
    bad = required - set(SECTIONS)
    if bad:
        raise ValueError(f"required lists unknown sections {sorted(bad)}")
    runners = {"scene": lambda s: check_scene(s, cfg["scene"]),
               "kinematics": lambda s: check_kinematics(s, base, cfg["kinematics"]),
               "cameras": lambda s: check_cameras(s, base, cfg["cameras"]),
               "placement": lambda s: check_placement(s, base, cfg["placement"]),
               "physics": lambda s: check_physics(s, base, cfg["physics"])}
    results = []
    for sec in SECTIONS:
        gate = GATE_NAMES[sec]
        if not bundle.get(sec):
            results.append(GateResult(gate, "skip", sec in required,
                                      [Finding("warn", "skipped", f"no `{sec}` section in the bundle")]))
            continue
        try:
            findings, metrics = runners[sec](bundle[sec])
        except (KeyError, ValueError, FileNotFoundError) as exc:
            findings, metrics = [Finding("error", "input", f"{type(exc).__name__}: {exc}")], {}
        results.append(GateResult(gate, _status(findings), sec in required, findings, metrics))
    return results


def exit_code(results: list[GateResult]) -> int:
    if any(r.status == "fail" for r in results):
        return 1
    if any(r.status == "skip" and r.required for r in results):
        return 2
    return 0


def corrections(results: list[GateResult]) -> dict[str, Any]:
    """Suggested sim-config changes, keyed by section (only where a check failed or a value was missing)."""
    out = {}
    for r in results:
        c = r.metrics.get("corrections")
        if c:
            out[r.gate] = _plain(c)
    return out


def real2sim_markdown(results: list[GateResult], bundle_path: str, name: str = "") -> str:
    code = exit_code(results)
    verdict = {0: "PASS", 1: "FAIL", 2: "INCOMPLETE (a required section is missing)"}[code]
    icon = {"pass": "pass", "warn": "pass (warnings)", "fail": "**FAIL**", "skip": "skipped"}
    lines = [f"# Real-to-sim calibration: {verdict}", "",
             f"Bundle `{bundle_path}`{f' ({name})' if name else ''}. Each check compares a measurement from the real cell with "
             "the sim's configuration. Fix failures in order: a wrong camera or kinematic model makes every later comparison "
             "measure the calibration error instead of the physics.", "",
             "| Check | Status | Required | Errors | Warnings |", "|---|---|---|---|---|"]
    for r in results:
        ne = sum(f.level == "error" for f in r.findings)
        nw = sum(f.level == "warn" for f in r.findings)
        lines.append(f"| {r.gate} | {icon[r.status]} | {'yes' if r.required else 'no'} | {ne} | {nw} |")
    for r in results:
        lines += ["", f"## {r.gate}: {icon[r.status]}", ""]
        for f in sorted(r.findings, key=lambda f: {"error": 0, "warn": 1, "ok": 2}[f.level]):
            tag = {"error": "**ERROR**", "warn": "warn", "ok": "ok"}[f.level]
            lines.append(f"- {tag} `{f.check}`: {f.detail}")
        if r.gate == "R5_physics":
            rows = [(o, t) for o, m in r.metrics.get("objects", {}).items() for t in m.get("test_errors", [])]
            if rows:
                lines += ["", "| Object | Test | Kind | nRMSE, fitted values | nRMSE, sim's values | nRMSE, your sim |",
                          "|---|---|---|---|---|---|"]
                for o, t in rows:
                    lines.append(f"| {o} | {t['test']} | {t['kind']} | {t['fit_nrmse']:.3f} | {_f(t['sim_params_nrmse'])} | "
                                 f"{_f(t.get('sim_rollout_nrmse', math.nan))} |")
    corr = corrections(results)
    lines += ["", "## Suggested sim corrections", ""]
    if corr:
        lines += ["Estimated from the real measurements; also written to `corrections.yaml`. Apply them, then rerun this check "
                  "and the fidelity gates.", "", "```yaml", yaml.safe_dump(corr, sort_keys=False).rstrip(), "```"]
    else:
        lines.append("None: every checked value is within tolerance.")
    return "\n".join(lines) + "\n"


def _f(x: float) -> str:
    return "-" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
