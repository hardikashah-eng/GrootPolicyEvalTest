"""Real-to-sim calibration checks: scene, kinematics, cameras, placement, physics."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from groot_eval import real2sim
from groot_eval.cli import main
from groot_eval.envs.arm3 import REACH, arm_fk
from groot_eval.envs.objects import rollout
from groot_eval.real2sim import Camera, bounce_heights, push_x, rotation, rpy_from_matrix, slide_x

ROOT = Path(__file__).resolve().parent.parent
CFG = real2sim.load_thresholds(ROOT / "configs" / "real2sim" / "thresholds.yaml")


def _demo():
    spec = importlib.util.spec_from_file_location("real2sim_demo", ROOT / "examples" / "run_real2sim_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _levels(findings):
    return {f.check: f.level for f in findings}


def test_rotation_roundtrip():
    for rpy in ([10, -20, 30], [-110, 5, 90], [0, 0, 0]):
        assert np.allclose(rpy_from_matrix(rotation(rpy)), rpy, atol=1e-9)


def test_scene_tolerances():
    f, m = real2sim.check_scene({"table": {"sim": 0.75, "real": 0.742, "tol": 0.003},
                                 "g": {"sim": 9.81, "real": 9.806}}, CFG["scene"])
    assert [x.level for x in f] == ["error", "ok"] and m["corrections"] == {"table": 0.742}


def test_kinematics_recovers_zero_offsets(tmp_path):
    rng = np.random.default_rng(0)
    q = rng.uniform([0.3, -1.6, -1.6], [2.8, 0.6, 0.6], size=(30, 3))
    lengths = np.full(3, REACH / 3)
    off, shift = np.array([0.0, 0.03, -0.02]), np.array([0.003, 0.0])
    ee = np.array([arm_fk(qi + off, lengths)[-1] for qi in q]) + shift
    rows = [{"q0": a, "q1": b, "q2": c, "x": x, "y": y} for (a, b, c), (x, y) in zip(q, ee)]
    f, m = real2sim.check_kinematics({"fk": "arm3", "samples": rows}, tmp_path, CFG["kinematics"])
    assert _levels(f)["fk.rms"] == "error"
    assert np.allclose(m["fitted_offsets_rad"], off, atol=1e-4) and m["fitted_rms_mm"] < 0.05
    # with those offsets configured in the sim, the check passes
    f2, m2 = real2sim.check_kinematics({"fk": "arm3", "samples": rows, **m["corrections"]}, tmp_path, CFG["kinematics"])
    assert _levels(f2)["fk.rms"] == "ok" and not m2["corrections"]


def _cam(**kw):
    d = {"position": [1.2, 0.0, 0.6], "rpy_deg": [-110, 0, 90], "fx": 600, "fy": 600, "cx": 320, "cy": 240,
         "width": 640, "height": 480}
    d.update(kw)
    return d


def test_camera_extrinsics_fit():
    rng = np.random.default_rng(1)
    real = _cam()
    pts = np.column_stack([rng.uniform(0.2, 0.7, 20), rng.uniform(-0.3, 0.3, 20), rng.uniform(0, 0.3, 20)])
    uv, ok = Camera.from_dict(real).project(pts)
    assert ok.all()
    obs = [{"X": a, "Y": b, "Z": c, "u": u, "v": v} for (a, b, c), (u, v) in zip(pts, uv)]
    sim = _cam(position=[1.2, 0.02, 0.6], rpy_deg=[-110, 0, 91.5])
    f, m = real2sim.check_cameras({"front": {"sim": sim, "real": real, "observations": obs}}, Path("."), CFG["cameras"])
    lv = _levels(f)
    assert lv["front.translation"] == lv["front.rotation"] == lv["front.reprojection"] == "error"
    assert lv["front.calibration"] == "ok" and lv["front.fit"] == "ok"
    fixed = Camera.from_dict(m["corrections"]["front"])
    assert np.allclose(fixed.position, real["position"], atol=1e-4)
    assert np.allclose(rpy_from_matrix(fixed.R), real["rpy_deg"], atol=1e-3)


def test_camera_resolution_and_intrinsics_mismatch():
    f, _ = real2sim.check_cameras({"c": {"sim": _cam(fx=640, width=320, height=240), "real": _cam()}}, Path("."),
                                  CFG["cameras"])
    lv = _levels(f)
    assert lv["c.resolution"] == "error" and lv["c.focal"] == "error"


def test_physics_models_match_the_stepped_engine():
    t = np.arange(0, 0.6, 0.02)
    p = {"mass": 0.3, "friction": 0.4, "static_friction": 0.5, "restitution": 0.5}
    assert np.allclose(rollout({"kind": "slide", "v0": 1.0, "t": t}, p), slide_x(t, 1.0, 0.4), atol=2e-3)
    assert np.allclose(rollout({"kind": "push", "force": 3.0, "duration": 0.3, "t": t}, p),
                       push_x(t, 3.0, 0.3, 0.3, 0.4), atol=2e-3)
    assert np.allclose(rollout({"kind": "drop", "h0": 0.4, "bounce_heights": [0, 0, 0]}, p),
                       bounce_heights(0.4, 0.5, 3), rtol=0.05)  # 1 ms Euler steps lose a little energy
    assert rollout({"kind": "tilt"}, p)[0] == pytest.approx(math.degrees(math.atan(0.5)), abs=0.1)
    # a push below static friction does not move the object
    assert np.all(rollout({"kind": "push", "force": 1.0, "duration": 0.3, "t": t}, p) == 0)


def test_physics_estimates_and_sim_mismatch():
    true = {"mass": 0.25, "friction": 0.5, "static_friction": 0.6, "restitution": 0.4}
    t = np.arange(0, 0.4, 0.02).tolist()
    tests = [{"object": "box", "kind": "weigh", "mass": 0.25},
             {"object": "box", "kind": "slide", "v0": 1.0, "t": t, "x": slide_x(np.array(t), 1.0, 0.5).tolist()},
             {"object": "box", "kind": "push", "force": 3.0, "duration": 0.3, "t": t,
              "x": push_x(np.array(t), 3.0, 0.3, 0.25, 0.5).tolist()},
             {"object": "box", "kind": "tilt", "angle_deg": math.degrees(math.atan(0.6))},
             {"object": "box", "kind": "drop", "h0": 0.5, "bounce_heights": bounce_heights(0.5, 0.4, 3).tolist()}]
    est = real2sim.estimate_object(tests)
    for k, v in true.items():
        assert est[k]["value"] == pytest.approx(v, rel=0.01), k
    section = {"sim_params": {"box": {"mass": 0.25, "friction": 0.25, "restitution": 0.4}}, "tests": tests,
               "sim_rollout": "groot_eval.envs.objects:rollout"}
    f, m = real2sim.check_physics(section, Path("."), CFG["physics"])
    lv = _levels(f)
    assert lv["box.mass"] == lv["box.restitution"] == "ok" and lv["box.friction"] == "error"
    assert lv["box.static_friction"] == "warn"  # not set in the sim
    assert lv["box.sim_params_replay"] == "error" and lv["box.model"] == "ok"
    assert m["corrections"]["box"] == {"friction": 0.5, "static_friction": 0.6}
    # matching parameters, but an engine that combines friction with the table's: only the rollout check fails
    section["sim_params"]["box"].update(friction=0.5, static_friction=0.6)
    section["sim_rollout_args"] = {"surface_friction": 0.7, "friction_combine": "multiply"}
    f, _ = real2sim.check_physics(section, Path("."), CFG["physics"])
    lv = _levels(f)
    assert lv["box.friction"] == lv["box.sim_params_replay"] == "ok" and lv["box.sim_rollout"] == "error"


def test_placement_checks(tmp_path):
    (tmp_path / "rand.yaml").write_text(yaml.safe_dump({"seed": 1, "episodes": 50, "params": {
        "object.cube.x": {"uniform": [0.3, 0.5]}, "object.cube.y": {"uniform": [-0.1, 0.1]}}}), encoding="utf-8")
    (tmp_path / "named.yaml").write_text(yaml.safe_dump({"scenarios": [
        {"name": "s0", "params": {"object.cube.x": 0.4, "object.cube.y": 0.0}},
        {"name": "s1", "params": {"object.cube.x": 0.45, "object.cube.y": 0.05}}]}), encoding="utf-8")
    real = [{"scenario": "s0", "object": "cube", "x": 0.401, "y": 0.0, "z": 0.775},
            {"scenario": "s1", "object": "cube", "x": 0.49, "y": 0.05, "z": 0.775},  # 40 mm off
            {"scenario": "s1", "object": "ball", "x": 0.50, "y": 0.06, "z": 0.79},   # overlaps the cube
            {"scenario": "c0", "object": "cube", "x": 0.6, "y": 0.0, "z": 0.775}]     # outside the x range
    sec = {"real": real, "sim_config": "rand.yaml", "sim_scenarios": "named.yaml", "support_height": 0.75,
           "objects": {"cube": {"height": 0.05, "radius": 0.03}, "ball": {"radius": 0.03}}}
    f, m = real2sim.check_placement(sec, tmp_path, CFG["placement"])
    lv = _levels(f)
    assert lv["cube.paired_position"] == "error" and lv["cube.coverage"] == "error"
    assert lv["cube.support"] == "ok" and lv["overlap"] == "error"
    assert m["corrections"]["widen_ranges"]["object.cube.x"] == {"uniform": [0.3, 0.6]}


def test_missing_required_section_exits_2(tmp_path):
    b = tmp_path / "b.yaml"
    b.write_text(yaml.safe_dump({"required": ["cameras"], "scene": {"t": {"sim": 1, "real": 1}}}), encoding="utf-8")
    assert main(["real2sim", "--bundle", str(b), "--out", str(tmp_path / "o")]) == 2
    data = json.loads((tmp_path / "o" / "real2sim.json").read_text(encoding="utf-8"))
    assert {c["gate"]: c["status"] for c in data["checks"]}["R3_cameras"] == "skip"
    (tmp_path / "bad.yaml").write_text("camera: {}\n", encoding="utf-8")  # typo for `cameras`
    with pytest.raises(ValueError, match="unknown bundle sections"):
        real2sim.load_bundle(tmp_path / "bad.yaml")


def test_example_bundle_template_is_valid():
    bundle, _ = real2sim.load_bundle(ROOT / "configs" / "real2sim" / "example_bundle.yaml")
    assert set(bundle["required"]) == set(real2sim.SECTIONS)


def test_demo_finds_planted_errors_and_corrections_fix_parameters(tmp_path):
    r = _demo().run(tmp_path / "out")
    assert r["codes"] == (1, 1)
    before = {c["gate"]: c["status"] for c in r["before"]["checks"]}
    after = {c["gate"]: c["status"] for c in r["after"]["checks"]}
    assert set(before.values()) == {"fail"}
    assert after["R1_scene"] == after["R2_kinematics"] == after["R3_cameras"] == "pass"
    remaining = {(c["gate"], f["check"]) for c in r["after"]["checks"] for f in c["findings"] if f["level"] == "error"}
    assert remaining == {("R4_placement", "cube.paired_position"), ("R4_placement", "cube.support"),
                         ("R4_placement", "overlap"), ("R5_physics", "cube.sim_rollout")}
    est = next(c for c in r["before"]["checks"] if c["gate"] == "R5_physics")["metrics"]["objects"]["cube"]["estimated"]
    assert est["friction"]["value"] == pytest.approx(0.45, rel=0.02)
    assert est["restitution"]["value"] == pytest.approx(0.35, rel=0.02)
    assert (tmp_path / "out" / "SUMMARY.md").read_text(encoding="utf-8").startswith("# Real-to-sim calibration demo (SYNTHETIC)")
