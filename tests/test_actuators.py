"""Wear- and fault-aware batched actuator: wear per unit and age, faults per env, NumPy == strict array API."""

import json

import numpy as np
import pytest

from groot_eval.actuators import (ActuatorParams, DCMotorSaturation, FaultTable, MLPMotorModel, WearAwareActuatorCore,
                                  WearTable, wear_at_age)
from groot_eval.degradation import WearConfig
from groot_eval.degradation.faults import parse_schedule
from groot_eval.robot_cli import main

WEAR = WearConfig.from_dict({
    "eol_hours": 10000,
    "profiles": {"new": {"age_hours": 0}, "old": {"age_hours": 10000}},
    "effects": {"motor.gain_scale": {"at_eol": 0.6, "spread": 0.3},
                "motor.torque_limit_scale": {"at_eol": 0.5},
                "motor.thermal_derate": {"at_eol": 0.5},
                "motor.thermal_tau_s": {"start": 1.0, "at_eol": 1.0},
                "joint.friction_scale": {"at_eol": 3.0},
                "joint.stiction": {"at_eol": 0.2},
                "encoder.bias_rad": {"at_eol": 0.05, "signed": True, "spread": 0.3}},
})
P = ActuatorParams(effort_limit=10.0, stiffness=50.0, damping=2.0, coulomb_friction=0.5, dt=0.01)


def _zeros(e, j):
    return np.zeros((e, j))


def test_wear_table_same_unit_at_every_age_and_typical_unit():
    t = WearTable.fleet(WEAR, 2, [0, 5000, 10000], envs_per_age=4)
    assert t.num_envs == 12
    g = t.values["gain"].reshape(3, 4, 2)
    assert np.allclose(g[0], 1.0)  # new robots: no wear yet
    assert (g[2] < g[1]).all() and (g[1] < 1.0).all()  # wear grows with age for every unit
    # unit i keeps its spread factor at every age: (1 - g) scales the same way
    assert np.allclose((1 - g[2]) / (1 - g[1]), (1 - g[2, 0]) / (1 - g[1, 0]))
    typ = wear_at_age(WEAR, 2, 10000, None)
    assert typ["gain"] == pytest.approx([0.6, 0.6]) and (typ["enc_bias"] == 0).all()


def test_older_units_track_worse_and_hit_torque_limits_sooner():
    t = WearTable.fleet(WEAR, 1, [0, 10000], envs_per_age=8)
    act = WearAwareActuatorCore(P, t)
    tau = act.compute(np.full((16, 1), 1.0), None, None, _zeros(16, 1), _zeros(16, 1))
    assert np.allclose(tau[:8], 10.0)  # new: PD wants 50 N·m, clipped at the full 10 N·m limit
    assert (np.abs(tau[8:]) <= 5.0 + 1e-9).all()  # old: limit halved


def test_thermal_derating_and_stiction():
    t = WearTable.fleet(WEAR, 1, [10000], envs_per_age=1)
    act = WearAwareActuatorCore(P, t)
    first = act.compute(np.ones((1, 1)), None, None, _zeros(1, 1), np.full((1, 1), 0.1))[0, 0]
    for _ in range(300):
        last = act.compute(np.ones((1, 1)), None, None, _zeros(1, 1), np.full((1, 1), 0.1))[0, 0]
    assert last < 0.8 * first  # a saturated motor heats up and loses part of its limit (equilibrium ~27% here)
    small = act.compute(np.full((1, 1), 0.001), None, None, _zeros(1, 1), _zeros(1, 1))[0, 0]
    assert small == 0.0  # below stiction a resting joint does not move


def test_fault_windows_apply_per_env_and_joint():
    scheds = [parse_schedule("motor_off@motor[1]:2+3"), parse_schedule("power_sag@bus:0+inf*0.5"), []]
    act = WearAwareActuatorCore(P, WearTable.new(3, 2), FaultTable.from_schedules(scheds, 2))
    out = [act.compute(np.ones((3, 2)), None, None, _zeros(3, 2), _zeros(3, 2)) for _ in range(6)]
    assert out[1][0, 1] == 10.0 and out[2][0, 1] == 0.0 and out[4][0, 1] == 0.0 and out[5][0, 1] == 10.0
    assert (out[3][0, 0] == 10.0)  # the other joint of the same robot is unaffected
    assert np.allclose(out[0][1], 5.0)  # power sag halves every motor of env 1
    assert np.allclose(out[3][2], 10.0)  # env 2 has no faults


def test_comm_stall_holds_and_latency_spike_delays_commands():
    scheds = [parse_schedule("comm_stall@bus:2+2"), parse_schedule("latency_spike@bus:0+inf*2")]
    act = WearAwareActuatorCore(ActuatorParams(effort_limit=1e9, stiffness=1.0, damping=0.0),
                                WearTable.new(2, 1), FaultTable.from_schedules(scheds, 1))
    targets = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    out = [act.compute(np.full((2, 1), x), None, None, _zeros(2, 1), _zeros(2, 1))[:, 0] for x in targets]
    assert [o[0] for o in out] == [0.0, 1.0, 1.0, 1.0, 4.0, 5.0]  # stalled at steps 2-3
    assert [o[1] for o in out][2:] == [0.0, 1.0, 2.0, 3.0]  # two steps late


def test_runaway_lock_and_sensor_faults():
    scheds = [parse_schedule("motor_runaway@motor[0]:0+inf*0.8;stuck@encoder[0]:2+inf"),
              parse_schedule("joint_lock@joint[0]:0+inf;dropout@encoder[0]:0+inf")]
    act = WearAwareActuatorCore(P, WearTable.new(2, 1), FaultTable.from_schedules(scheds, 1))
    qd = np.array([[0.0], [0.3]])
    tau = act.compute(_zeros(2, 1), None, None, _zeros(2, 1), qd)
    assert tau[0, 0] == pytest.approx(8.0)  # forced to 0.8 x limit regardless of the command
    assert tau[1, 0] == pytest.approx(-3000.0)  # a seized joint brakes hard against any motion (not motor-limited)
    r1 = act.observe(np.array([[0.1], [0.1]]))
    assert np.isnan(r1[1, 0])
    for q in (0.2, 0.3):
        act.compute(_zeros(2, 1), None, None, _zeros(2, 1), _zeros(2, 1))
        r = act.observe(np.array([[q], [q]]))
    assert r[0, 0] == pytest.approx(0.2)  # frozen at the last reading before the fault began


def test_numpy_and_strict_array_api_agree():
    xs = pytest.importorskip("array_api_strict")
    t = WearTable.fleet(WEAR, 3, [0, 10000], envs_per_age=4)
    sch = FaultTable.from_schedules([parse_schedule("motor_off@motor[2]:3+inf")] * 8, 3)
    a, b = WearAwareActuatorCore(P, t, sch), WearAwareActuatorCore(P, t, sch)
    rng = np.random.default_rng(0)
    for _ in range(5):
        q, qd, tgt = rng.normal(size=(8, 3)), rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
        ref = a.compute(tgt, None, None, q, qd)
        got = b.compute(xs.asarray(tgt), None, None, xs.asarray(q), xs.asarray(qd))
        assert isinstance(got, xs._array_object.Array)
        assert np.allclose(np.from_dlpack(got), ref)


def test_mlp_motor_model_roundtrip_and_dc_saturation(tmp_path):
    rng = np.random.default_rng(1)
    m = MLPMotorModel([rng.normal(size=(6, 8)), rng.normal(size=(8, 1))], [np.zeros(8), np.zeros(1)],
                      history=(0, 1, 2), pos_scale=2.0, vel_scale=0.5, torque_scale=3.0)
    m2 = MLPMotorModel.load(m.save(tmp_path / "net.npz"))
    err, vel = rng.normal(size=(3, 4, 2)), rng.normal(size=(3, 4, 2))
    x = np.concatenate([err[i][..., None] * 2.0 for i in (0, 1, 2)] + [vel[i][..., None] * 0.5 for i in (0, 1, 2)], -1)
    h = x @ m.weights[0]
    manual = ((h / (1 + np.abs(h))) @ m.weights[1])[..., 0] * 3.0
    assert np.allclose(m2(err, vel), manual)
    sat = DCMotorSaturation(saturation_effort=20.0, velocity_limit=10.0)
    out = sat.clip(np.full((1, 3), 100.0), np.array([[0.0, 5.0, 12.0]]), 15.0)
    assert out[0].tolist() == [15.0, 10.0, 0.0]  # full limit at rest, half at half speed, none past v_max


def test_mlp_actuator_in_core_uses_history():
    m = MLPMotorModel([np.array([[1.0], [0.0], [0.0], [0.0]])], [np.zeros(1)], history=(0, 1))
    act = WearAwareActuatorCore(ActuatorParams(effort_limit=100.0), WearTable.new(1, 1), motor=m)
    out = [act.compute(np.full((1, 1), x), None, None, _zeros(1, 1), _zeros(1, 1))[0, 0] for x in (1.0, 2.0)]
    assert out == [1.0, 2.0]  # torque = current position error, through the learned model


def test_cli_actuator_sweep(tmp_path):
    code = main(["actuator-sweep", "--wear", "configs/wear/arm3_profiles_extended.yaml", "--ages", "0,8760",
                 "--units", "6", "--joints", "2", "--steps", "400", "--faults", "configs/faults/arm3_single.yaml",
                 "--out", str(tmp_path)])
    assert code == 0
    res = json.loads((tmp_path / "actuator_sweep.json").read_text())
    assert [r["age_hours"] for r in res["rows"]] == [0.0, 8760.0]
    assert "Paired increase" in (tmp_path / "actuator_sweep.md").read_text()


def test_isaaclab_adapter_builds_fleet_core_without_isaaclab():
    from types import SimpleNamespace

    from groot_eval.actuators.isaaclab_actuator import build_core

    cfg = SimpleNamespace(wear_config="configs/wear/arm3_profiles_extended.yaml", ages_hours=[0, 8760], typical_units=False,
                          fault_config="configs/faults/arm3_single.yaml", horizon_steps=200, dt=0.005, fault_seed=0,
                          coulomb_friction=0.1, viscous_friction=0.0, motor_model=None, saturation_effort=None,
                          velocity_limit_dc=float("inf"))
    core = build_core(cfg, num_envs=6, num_joints=3, effort_limit=[20.0] * 3, stiffness=[80.0] * 3, damping=[2.0] * 3)
    assert core.wear.ages_hours.tolist() == [0, 0, 0, 8760, 8760, 8760]
    assert core.wear.unit_seeds.tolist() == [0, 1, 2, 0, 1, 2]  # same units at both ages
    assert len(core.faults.schedules) == 6
    tau = core.compute(np.ones((6, 3)), None, None, np.zeros((6, 3)), np.zeros((6, 3)))
    assert tau.shape == (6, 3)
