"""Real-to-sim calibration checks on a SYNTHETIC "real" cell with planted errors.

    uv run python examples/run_real2sim_demo.py [output_dir]

There is no real cell here. The script invents measurements from a "real" cell whose true
values differ from the sim's configuration in known ways, runs ``groot-eval real2sim``,
applies the corrections it suggests, and runs it again:

* scene: the real table is 8 mm lower than the sim's;
* kinematics (Arm3): joint 1 and 2 encoders are zeroed 1.15 and -0.57 deg off, base 3.6 mm off;
* camera: the sim camera is 15 mm and 2 deg away from the calibrated pose;
* placement: real cube positions spill past the sim's x range, one operator mis-placement,
  two perception poses floating 12 mm above the table, one scenario with cube and cup
  interpenetrating;
* physics: the real cube has friction 0.45 (sim 0.30), restitution 0.35 (sim 0.60), static
  friction 0.55 (sim leaves it unset); and the reference engine combines friction with a
  table material of 0.8, which the parameters alone do not show.

The corrected run should fix scene, kinematics, camera and physics *parameters*, and keep
failing exactly where the problem is not a sim parameter: the real placement data (fix the
data or the perception pipeline) and the engine's friction combine mode.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from groot_eval.cli import main
from groot_eval.envs.arm3 import REACH, arm_fk
from groot_eval.real2sim import Camera, rpy_from_matrix

ROOT = Path(__file__).resolve().parent.parent
TRUE = {
    "table_height": 0.742,
    "joint_offsets": [0.0, 0.02, -0.01],
    "base_shift": [0.003, -0.002],
    "cube": {"mass": 0.21, "friction": 0.45, "static_friction": 0.55, "restitution": 0.35},
}
CAM_R = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # looks along -x, image y down
CAM_REAL = {"position": [1.2, 0.0, 0.6], "rpy_deg": None, "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
            "width": 640, "height": 480}


def _real_camera() -> dict:
    tilt = np.array([[1, 0, 0], [0, math.cos(-0.35), -math.sin(-0.35)], [0, math.sin(-0.35), math.cos(-0.35)]])
    cam = dict(CAM_REAL)
    cam["rpy_deg"] = [round(v, 6) for v in rpy_from_matrix(CAM_R @ tilt)]  # pitched down about 20 deg
    return cam


def make_bundle(out: Path) -> Path:
    rng = np.random.default_rng(7)
    out.mkdir(parents=True, exist_ok=True)

    # kinematics: recorded joints, end effector measured by a "laser tracker" (0.3 mm noise)
    q = rng.uniform([0.3, -1.6, -1.6], [2.8, 0.6, 0.6], size=(40, 3))
    lengths = np.full(3, REACH / 3)
    ee = np.array([arm_fk(qi + TRUE["joint_offsets"], lengths)[-1] for qi in q]) + TRUE["base_shift"]
    ee += rng.normal(0, 0.0003, ee.shape)
    pd.DataFrame({"q0": q[:, 0], "q1": q[:, 1], "q2": q[:, 2], "x": ee[:, 0], "y": ee[:, 1]}).round(6).to_csv(
        out / "kinematics.csv", index=False)

    # camera: fiducials in the cell, detected in a real image (0.3 px noise)
    real_cam = _real_camera()
    pts = np.column_stack([rng.uniform(0.2, 0.7, 30), rng.uniform(-0.3, 0.3, 30), rng.uniform(0.0, 0.3, 30)])
    uv, valid = Camera.from_dict(real_cam).project(pts)
    assert valid.all()
    uv += rng.normal(0, 0.3, uv.shape)
    pd.DataFrame({"X": pts[:, 0], "Y": pts[:, 1], "Z": pts[:, 2], "u": uv[:, 0], "v": uv[:, 1]}).round(4).to_csv(
        out / "camera_front_points.csv", index=False)
    sim_cam = dict(real_cam)
    sim_cam["position"] = [1.2, 0.015, 0.6]
    sim_cam["rpy_deg"] = [real_cam["rpy_deg"][0], real_cam["rpy_deg"][1], real_cam["rpy_deg"][2] + 2.0]

    # placement: the sim randomizes the cube over x 0.35-0.55, y +-0.15, yaw +-45 deg
    sim_cfg = {"seed": 3, "episodes": 200, "params": {"object.cube.x": {"uniform": [0.35, 0.55]},
                                                      "object.cube.y": {"uniform": [-0.15, 0.15]},
                                                      "object.cube.yaw_deg": {"uniform": [-45, 45]}}}
    (out / "sim_scene_randomization.yaml").write_text(yaml.safe_dump(sim_cfg), encoding="utf-8")
    named = []
    for i in range(10):
        named.append({"name": f"cell_{i:02d}", "seed": i,
                      "params": {"object.cube.x": round(float(rng.uniform(0.38, 0.52)), 4),
                                 "object.cube.y": round(float(rng.uniform(-0.12, 0.12)), 4),
                                 "object.cube.yaw_deg": round(float(rng.uniform(-30, 30)), 2)}})
    (out / "sim_test_scenarios.yaml").write_text(yaml.safe_dump({"scenarios": named}), encoding="utf-8")
    rows = []
    z_cube = TRUE["table_height"] + 0.025
    for i, s in enumerate(named):  # the real test set: same scenarios, placed by an operator (3 mm, 2 deg)
        p = s["params"]
        dx = 0.035 if i == 4 else 0.0  # one mis-placement
        rows.append({"scenario": s["name"], "object": "cube", "x": p["object.cube.x"] + dx + rng.normal(0, 0.003),
                     "y": p["object.cube.y"] + rng.normal(0, 0.003), "z": z_cube + rng.normal(0, 0.001),
                     "yaw_deg": p["object.cube.yaw_deg"] + rng.normal(0, 2)})
        cup = (p["object.cube.x"] + 0.05, p["object.cube.y"]) if i == 7 else (0.62, -0.2 + 0.04 * i)
        rows.append({"scenario": s["name"], "object": "cup", "x": cup[0], "y": cup[1], "z": TRUE["table_height"] + 0.05,
                     "yaw_deg": 0.0})
    for j in range(40):  # free placements logged by perception during data collection
        rows.append({"scenario": f"collect_{j:02d}", "object": "cube", "x": rng.normal(0.50, 0.05),
                     "y": rng.uniform(-0.14, 0.14), "z": z_cube + (0.012 if j in (5, 17) else rng.normal(0, 0.001)),
                     "yaw_deg": rng.uniform(-40, 40)})
    pd.DataFrame(rows).round(5).to_csv(out / "placements_real.csv", index=False)

    # physics bench tests on the real cube
    c = TRUE["cube"]
    from groot_eval.real2sim import bounce_heights, push_x, slide_x

    tests = [{"object": "cube", "kind": "weigh", "mass": round(c["mass"] + rng.normal(0, 0.001), 4)} for _ in range(3)]
    t = np.round(np.arange(0, 0.4, 0.02), 3)
    for v0 in (0.8, 1.0, 1.2):
        x = slide_x(t, v0, c["friction"]) + rng.normal(0, 0.001, t.size)
        tests.append({"object": "cube", "kind": "slide", "v0": v0, "t": t.tolist(), "x": np.round(x, 5).tolist()})
    t2 = np.round(np.arange(0, 0.8, 0.02), 3)
    for f in (2.5, 3.0):
        x = push_x(t2, f, 0.3, c["mass"], c["friction"]) + rng.normal(0, 0.001, t2.size)
        tests.append({"object": "cube", "kind": "push", "force": f, "duration": 0.3, "t": t2.tolist(),
                      "x": np.round(x, 5).tolist()})
    for _ in range(3):
        tests.append({"object": "cube", "kind": "tilt",
                      "angle_deg": round(math.degrees(math.atan(c["static_friction"])) + rng.normal(0, 0.4), 2)})
    for h0 in (0.3, 0.4):
        b = bounce_heights(h0, c["restitution"], 3) * (1 + rng.normal(0, 0.01, 3))
        tests.append({"object": "cube", "kind": "drop", "h0": h0, "bounce_heights": np.round(b, 5).tolist()})
    (out / "physics_tests.yaml").write_text(yaml.safe_dump(tests, default_flow_style=None), encoding="utf-8")

    bundle = {
        "name": "SYNTHETIC demo cell",
        "required": ["scene", "kinematics", "cameras", "placement", "physics"],
        "scene": {"table_height": {"sim": 0.75, "real": TRUE["table_height"], "tol": 0.003, "unit": " m"},
                  "gravity": {"sim": 9.81, "real": 9.806, "unit": " m/s^2"},
                  "cube_size": {"sim": 0.05, "real": 0.0502, "tol": 0.001, "unit": " m"}},
        "kinematics": {"fk": "arm3", "samples": "kinematics.csv"},
        "cameras": {"front": {"sim": sim_cam, "real": real_cam, "observations": "camera_front_points.csv"}},
        "placement": {"real": "placements_real.csv", "sim_config": "sim_scene_randomization.yaml",
                      "sim_scenarios": "sim_test_scenarios.yaml", "support_height": TRUE["table_height"],
                      "objects": {"cube": {"height": 0.05, "radius": 0.03}, "cup": {"height": 0.10, "radius": 0.04}}},
        "physics": {"sim_params": {"cube": {"mass": 0.2, "friction": 0.3, "restitution": 0.6}},
                    "tests": "physics_tests.yaml", "sim_rollout": "groot_eval.envs.objects:rollout",
                    "sim_rollout_args": {"surface_friction": 0.8, "friction_combine": "multiply"}},
    }
    path = out / "bundle.yaml"
    path.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
    return path


def apply_corrections(bundle_path: Path, corrections: dict, out: Path) -> Path:
    """Write a bundle whose sim side has the suggested corrections applied (the real side is unchanged)."""
    b = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    for name, value in (corrections.get("R1_scene") or {}).items():
        b["scene"][name]["sim"] = value
    k = corrections.get("R2_kinematics") or {}
    b["kinematics"].update({key: k[key] for key in ("joint_offsets_rad", "base_shift_m") if key in k})
    for cam, spec in (corrections.get("R3_cameras") or {}).items():
        b["cameras"][cam]["sim"] = spec
    widen = (corrections.get("R4_placement") or {}).get("widen_ranges") or {}
    if widen:
        cfg = yaml.safe_load((bundle_path.parent / b["placement"]["sim_config"]).read_text(encoding="utf-8"))
        cfg["params"].update(widen)
        (out / "sim_scene_randomization_corrected.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        b["placement"]["sim_config"] = "sim_scene_randomization_corrected.yaml"
    for obj, vals in (corrections.get("R5_physics") or {}).items():
        b["physics"]["sim_params"].setdefault(obj, {}).update(vals)
    path = out / "bundle_corrected.yaml"
    path.write_text(yaml.safe_dump(b, sort_keys=False), encoding="utf-8")
    return path


def run(out: Path) -> dict:
    shutil.rmtree(out, ignore_errors=True)
    bundle = make_bundle(out / "bundle")
    code_before = main(["real2sim", "--bundle", str(bundle), "--out", str(out / "before")])
    before = json.loads((out / "before" / "real2sim.json").read_text(encoding="utf-8"))
    corrected = apply_corrections(bundle, before["corrections"], out / "bundle")
    code_after = main(["real2sim", "--bundle", str(corrected), "--out", str(out / "after")])
    after = json.loads((out / "after" / "real2sim.json").read_text(encoding="utf-8"))

    def status(d: dict) -> dict:
        return {g["gate"]: g["status"] for g in d["checks"]}

    lines = ["# Real-to-sim calibration demo (SYNTHETIC)", "",
             "> **SYNTHETIC.** The \"real\" measurements are generated with planted errors (see "
             "`examples/run_real2sim_demo.py`). This shows what each check finds and that the estimates recover known "
             "values. It is not a result about a real cell.", "",
             "| Check | Before | After applying corrections |", "|---|---|---|"]
    sb, sa = status(before), status(after)
    lines += [f"| {g} | {sb[g]} | {sa[g]} |" for g in sb]
    k = next(g for g in before["checks"] if g["gate"] == "R2_kinematics")["metrics"]
    cam = next(g for g in before["checks"] if g["gate"] == "R3_cameras")["metrics"]["cameras"]["front"]
    phys = next(g for g in before["checks"] if g["gate"] == "R5_physics")["metrics"]["objects"]["cube"]["estimated"]
    lines += ["", "## Recovered values", "", "| Quantity | True (planted) | Estimated |", "|---|---|---|",
              f"| joint zero offsets (deg) | {np.round(np.degrees(TRUE['joint_offsets']), 3).tolist()} | "
              f"{np.round(np.degrees(k['fitted_offsets_rad']), 3).tolist()} |",
              f"| base shift (mm) | {np.round(np.array(TRUE['base_shift']) * 1000, 2).tolist()} | "
              f"{np.round(np.array(k['fitted_base_shift_m']) * 1000, 2).tolist()} |",
              f"| camera correction | 15 mm, 2.00 deg | {cam['correction_translation_mm']:.1f} mm, "
              f"{cam['correction_rotation_deg']:.2f} deg |"]
    for key in ("mass", "friction", "static_friction", "restitution"):
        lines.append(f"| cube {key} | {TRUE['cube'][key]} | {phys[key]['value']:.3f} ± {phys[key]['spread']:.3f} |")
    lines += ["", "After the corrections, what still fails is not a sim parameter:", ""]
    for g in after["checks"]:
        for f in g["findings"]:
            if f["level"] == "error":
                lines.append(f"- `{g['gate']}` `{f['check']}`: {f['detail']}")
    summary = "\n".join(lines) + "\n"
    (out / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    return {"before": before, "after": after, "codes": (code_before, code_after)}


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo-output" / "real2sim")
