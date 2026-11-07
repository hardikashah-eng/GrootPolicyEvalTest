"""Wear, DOF loss, fault injection, skill suites and data-fidelity gates."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from groot_eval.cli import main
from groot_eval.closed_loop import evaluate
from groot_eval.degradation import FaultConfig, FaultEnv, WearConfig, WearEnv, aged_scenarios, faulted_scenarios
from groot_eval.degradation.components import probe_components
from groot_eval.degradation.faults import Fault, encode_schedule, parse_schedule
from groot_eval.degradation.runners import aging_summary, dof_loss_matrix, fault_gate, fault_summary, run_aging
from groot_eval.envs.arm3 import Arm3Env, arm_fk
from groot_eval.envs.reach2d import Reach2DEnv
from groot_eval.policies import make_policy
from groot_eval.stats import mcnemar_exact
from groot_eval.suites import SuiteConfig, suite_gate, suite_summary
from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario

WEAR = {
    "eol_hours": 1000,
    "profiles": {"new": {"age_hours": 0}, "old": {"age_hours": 1000}},
    "effects": {
        "motor.gain_scale": {"curve": "linear", "at_eol": 0.5, "spread": 0.2},
        "joint.friction_scale": {"curve": "bathtub", "at_eol": 3.0},
        "encoder.bias_rad": {"curve": "linear", "at_eol": 0.05, "signed": True, "spread": 0.3},
    },
}


# -- Arm3 -------------------------------------------------------------------------------------------
def test_arm3_reaches_and_exposes_components():
    run = evaluate(Arm3Env(), make_policy("arm-ik"), [Scenario(seed=i) for i in range(20)])
    assert run.successes == 20
    comps = Arm3Env().components()
    assert {"motor[0]", "joint[2]", "encoder[1]", "gripper", "bus"} <= set(comps)
    assert comps["motor[1]"].action == (1,)


def test_arm3_ee_pos_is_forward_kinematics_of_reported_joints():
    env = Arm3Env()
    obs = env.reset(Scenario(seed=1))
    q = np.asarray(obs["state.joint_pos"], dtype=float)
    assert np.allclose(obs["state.ee_pos"], arm_fk(q, env.lengths)[-1], atol=1e-6)


def test_arm3_locked_joint_does_not_move():
    env = Arm3Env()
    env.reset(Scenario(seed=0))
    env.set_component_param("joint[1]", "locked", True)
    q0 = env.q[1]
    for _ in range(30):
        env.step(np.array([0.0, 1.0, 0.0, 0.0]))
    assert env.q[1] == pytest.approx(q0)


# -- wear -------------------------------------------------------------------------------------------
def test_wear_new_profile_is_nominal_and_units_pair_across_ages():
    cfg = WearConfig.from_dict(WEAR)
    comps = probe_components(Arm3Env())
    new = cfg.sample("new", comps, unit_seed=4)
    old_a, old_b = cfg.sample("old", comps, 4), cfg.sample("old", comps, 4)
    assert new["wear.motor[0].gain_scale"] == pytest.approx(1.0)
    assert new["wear.encoder[2].bias_rad"] == pytest.approx(0.0)
    assert old_a == old_b  # same unit, same draw
    assert old_a["wear.motor[0].gain_scale"] != cfg.sample("old", comps, 5)["wear.motor[0].gain_scale"]
    assert old_a["wear.joint[0].friction_scale"] > 2.5


def test_wear_env_applies_physics_and_sensor_effects():
    env = WearEnv(Arm3Env())
    params = {"wear.motor[0].gain_scale": 0.6, "wear.encoder[1].bias_rad": 0.1}
    obs = env.reset(Scenario(seed=2, params=params))
    assert env.env.comp["motor[0]"]["gain_scale"] == pytest.approx(0.6)
    true_q = env.env.q.copy()
    assert obs["state.joint_pos"][1] == pytest.approx(true_q[1] + 0.1, abs=1e-5)
    # the end effector is computed from the biased reading, as a real controller would
    assert np.allclose(obs["state.ee_pos"], arm_fk(np.asarray(obs["state.joint_pos"], float), env.env.lengths)[-1], atol=1e-5)


def test_wear_env_falls_back_to_command_boundary_on_generic_envs():
    env = WearEnv(Reach2DEnv())
    env.reset(Scenario(seed=0, params={"wear.motor[0].gain_scale": 0.0, "wear.motor[1].backlash_rad": 0.01}))
    before = env.env.vel.copy()
    env.step(np.array([1.0, 0.0]))
    assert env.env.vel[0] == pytest.approx(before[0])  # gain 0 on axis 0
    assert "motor[1].backlash_rad" in env.unsupported


def test_aging_summary_pairs_and_finds_margin():
    cfg = WearConfig.from_dict(WEAR)
    comps = probe_components(Arm3Env())
    base = [Scenario(seed=i) for i in range(40)]
    runs = run_aging(lambda: Arm3Env(), make_policy("arm-ik"), base, cfg, ["new", "old"], comps)
    s = aging_summary(runs, gate=0.9)
    assert s["profiles"][0]["rate"] == 1.0
    old = s["profiles"][1]
    assert old["rate"] < 1.0 and old["paired"]["lost"] > 0
    assert s["margin"]["first_failing"] == "old"


def test_dof_loss_matrix_limp_shoulder_is_worst():
    rows = dof_loss_matrix(lambda: Arm3Env(), make_policy("arm-ik"), [Scenario(seed=i) for i in range(12)],
                           probe_components(Arm3Env()))
    by = {r["joint"]: r["cells"] for r in rows}
    assert by["(none)"]["locked"]["rate"] == 1.0
    assert by["joint[0]"]["limp"]["rate"] < by["joint[0]"]["limited"]["rate"]


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0


# -- faults -----------------------------------------------------------------------------------------
def test_schedule_roundtrip():
    faults = [Fault("stuck", "encoder[1]", 40, 60), Fault("motor_off", "motor[0]", 120, None, 1.0)]
    text = encode_schedule(faults)
    assert text == "stuck@encoder[1]:40+60;motor_off@motor[0]:120+inf*1"
    assert parse_schedule(text) == faults
    with pytest.raises(ValueError):
        parse_schedule("melt@motor[0]:1+2")


def test_stuck_encoder_freezes_reading_then_releases():
    env = FaultEnv(Arm3Env())
    obs = env.reset(Scenario(seed=0, params={"faults": "stuck@encoder[0]:2+5"}))
    seen = []
    for _ in range(10):
        obs, *_ = env.step(np.array([0.5, 0.0, 0.0, 0.0]))
        seen.append(float(obs["state.joint_pos"][0]))
    frozen = seen[1:6]
    assert len(set(frozen)) == 1
    assert seen[7] != frozen[0]


def test_motor_off_uses_physics_and_restores():
    env = FaultEnv(Arm3Env())
    env.reset(Scenario(seed=0, params={"faults": "motor_off@motor[2]:1+3"}))
    env.step(np.zeros(4))
    env.step(np.zeros(4))
    assert env.env.comp["motor[2]"]["enabled"] is False
    for _ in range(4):
        env.step(np.zeros(4))
    assert env.env.comp["motor[2]"]["enabled"] is True


def test_fault_sampling_is_deterministic_and_reported():
    cfg = FaultConfig.from_dict({"faults": {"mode": "single", "start_window": [5, 30], "catalogue": [
        {"type": "dropout", "targets": ["encoder[*]"], "duration": [3, 10]},
        {"type": "joint_lock", "targets": ["joint[*]"], "duration": [20, 40]}]}})
    comps = probe_components(Arm3Env())
    base = [Scenario(seed=i) for i in range(30)]
    a = faulted_scenarios(base, cfg, comps, 300, 0.02)
    assert a == faulted_scenarios(base, cfg, comps, 300, 0.02)
    assert all(s.params["faults.n"] == 1 for s in a)
    naive = evaluate(FaultEnv(Arm3Env()), make_policy("arm-ik"), a)
    robust = evaluate(FaultEnv(Arm3Env()), make_policy("arm-ik-robust"), a)
    sn, sr = fault_summary(naive), fault_summary(robust)
    assert sn["faulted_episodes"] >= 27  # an episode may finish before its fault starts
    assert set(sn["by_type"]) == {"dropout", "joint_lock"}
    # the naive controller passes NaN readings into its commands; the robust one does not
    assert any((r.failure_reason or "").startswith("non_finite") for r in naive.results)
    assert not any((r.failure_reason or "").startswith("non_finite") for r in robust.results)
    assert sr["overall"]["safe"]["rate"] > sn["overall"]["safe"]["rate"]
    assert fault_gate(sr, min_safe=0.999, max_unsafe=None)  # 30 episodes cannot prove 99.9%


# -- suites -------------------------------------------------------------------------------------------
SUITE = {"skill": "reach", "design": ["oat", "corners", "lhs"], "repeats": 3, "lhs_samples": 20,
         "conditions": {"payload_kg": {"min": 0.0, "nominal": 0.1, "max": 0.5},
                        "action_delay": {"min": 0, "nominal": 0, "max": 3, "integer": True},
                        "friction": {"min": 0.6, "nominal": 1.0, "max": 2.0}}}


def test_suite_designs_and_counts():
    cfg = SuiteConfig.from_dict(SUITE)
    sc = cfg.scenarios()
    names = [s.name for s in sc]
    # centre + oat (payload min/max, delay max only, friction min/max = 5 cells) + 8 corners, x3 repeats, + 20 lhs
    assert names.count("centre") == 3
    assert sum(n.startswith("oat:") for n in names) == 5 * 3
    assert sum(n.startswith("corner:") for n in names) == 8 * 3
    lhs = [s for s in sc if s.name.startswith("lhs:")]
    assert len(lhs) == 20
    bins = np.floor(np.array([s.params["payload_kg"] for s in lhs]) / 0.5 * 20).astype(int)
    assert sorted(np.clip(bins, 0, 19)) == list(range(20))  # one sample per stratum


def test_suite_summary_finds_delay_and_gates():
    cfg = SuiteConfig.from_dict(SUITE)
    run = evaluate(Arm3Env(), make_policy("arm-ik"), cfg.scenarios())
    s = suite_summary(cfg, run)
    delay = next(r for r in s["oat"] if r["condition"] == "action_delay")
    assert delay["worst_level"] == "max"
    assert s["corners"][0]["rate"] <= s["corners"][-1]["rate"]
    assert suite_gate(s, min_oat=None, min_corner=0.99)


# -- fidelity -----------------------------------------------------------------------------------------
def _trace(tmp_path, name, params=None, wear=None, n=8, scale_vel=None):
    env = WearEnv(Arm3Env()) if wear else Arm3Env()
    rec = TraceRecorder(domain="real" if wear else "sim", env="arm3", policy="arm-ik", control_dt=0.02)
    sc = [Scenario(seed=i, params={**(params or {}), **(wear or {})}) for i in range(n)]
    evaluate(env, make_policy("arm-ik"), sc, recorder=rec)
    root = rec.write(tmp_path / name)
    if scale_vel:
        ts = TraceSet.open(root)
        sl = ts.state_slices()["joint_vel"]
        for meta in ts.dataset.episodes:
            path = root / "data" / "chunk-000" / f"episode_{meta['episode_index']:06d}.parquet"
            df = pd.read_parquet(path)
            df["observation.state"] = [np.concatenate([v[: sl.start], v[sl] * scale_vel, v[sl.stop:]]) for v in df["observation.state"]]
            df.to_parquet(path)
    return root


def test_fidelity_identical_traces_pass_and_tamper_is_caught(tmp_path):
    from groot_eval import fidelity

    a = _trace(tmp_path, "a", {"payload_kg": 0.2})
    fidelity.write_manifest(a)
    cfg, _ = fidelity.load_gates(None)
    res = fidelity.run_gates(TraceSet.open(a), TraceSet.open(a), cfg, env=Arm3Env(), params={"payload_kg": 0.2})
    status = {r.gate: r.status for r in res}
    assert status["G1_contract"] == "pass" and status["G4_distribution"] == "pass"
    assert status["G5_dynamics"] == "pass"
    assert fidelity.exit_code(res) == 0
    (a / "meta" / "episodes.jsonl").write_text((a / "meta" / "episodes.jsonl").read_text() + "\n", encoding="utf-8")
    res = fidelity.run_gates(TraceSet.open(a), TraceSet.open(a), cfg)
    assert res[0].status == "fail" and all(r.status == "skip" for r in res[1:])


def test_fidelity_units_bug_stops_at_contract(tmp_path):
    from groot_eval import fidelity

    sim = _trace(tmp_path, "sim")
    real = _trace(tmp_path, "real", scale_vel=57.2958)
    cfg, _ = fidelity.load_gates(None)
    res = fidelity.run_gates(TraceSet.open(sim), TraceSet.open(real), cfg, direction="real2sim")
    by = {r.gate: r for r in res}
    assert by["G1_contract"].status == "fail"
    assert any("degrees" in f.detail for f in by["G1_contract"].findings)
    assert by["G4_distribution"].status == "skip"
    assert fidelity.exit_code(res) == 1


def test_fidelity_required_gate_missing_exits_2(tmp_path):
    from groot_eval import fidelity

    a = _trace(tmp_path, "a")
    cfg, _ = fidelity.load_gates(None)
    cfg["G5_dynamics"]["required"] = True
    res = fidelity.run_gates(TraceSet.open(a), TraceSet.open(a), cfg)
    assert fidelity.exit_code(res) == 2


# -- CLI ----------------------------------------------------------------------------------------------
def test_cli_robustness_commands(tmp_path, capsys):
    wear = tmp_path / "wear.yaml"
    wear.write_text(json.dumps(WEAR), encoding="utf-8")
    out = tmp_path / "aging"
    code = main(["aging", "--env", "arm3", "--policy", "arm-ik", "--episodes", "20", "--wear", str(wear),
                 "--dof-episodes", "6", "--min-success-at", "old", "0.99", "--out", str(out)])
    assert code == 1
    assert "Loss of a degree of freedom" in (out / "aging.md").read_text(encoding="utf-8")

    code = main(["faults", "--env", "arm3", "--policy", "arm-ik-robust", "--episodes", "20",
                 "--faults", "configs/faults/arm3_single.yaml", "--out", str(tmp_path / "faults")])
    assert code == 0
    assert "By fault type" in (tmp_path / "faults" / "faults.md").read_text(encoding="utf-8")

    suite = tmp_path / "suite.yaml"
    suite.write_text(json.dumps({**SUITE, "design": ["oat"], "repeats": 2}), encoding="utf-8")
    code = main(["suite", "--env", "arm3", "--policy", "arm-ik", "--config", str(suite), "--wear", str(wear),
                 "--profile", "old", "--out", str(tmp_path / "suite")])
    assert code == 0
    data = json.loads((tmp_path / "suite" / "suite.json").read_text(encoding="utf-8"))
    assert data["oat"]

    sim = _trace(tmp_path, "s")
    code = main(["fidelity", "--sim", str(sim), "--real", str(sim), "--out", str(tmp_path / "fid")])
    assert code == 0
    assert "Data-fidelity gates: PASS" in (tmp_path / "fid" / "fidelity.md").read_text(encoding="utf-8")
