"""Wear additions, real-to-sim wear calibration, and sim<->real transfer verification."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from groot_eval import transfer
from groot_eval.cli import main
from groot_eval.closed_loop import evaluate
from groot_eval.degradation import WearConfig, WearEnv
from groot_eval.degradation.calibrate import fit_age
from groot_eval.degradation.components import probe_components
from groot_eval.degradation.wear import degrade_image
from groot_eval.envs.arm3 import Arm3Env
from groot_eval.policies import make_policy
from groot_eval.trace import TraceRecorder, TraceSet, import_real
from groot_eval.types import Scenario

ROBOT = {"state_limits": {"joint_pos": [[-0.2, 3.3416], [-2.6, 2.6], [-2.6, 2.6]]}, "action_range": [-1, 1],
         "max_action_rate": 2.0}
WEAR = {"eol_hours": 10000, "profiles": {"new": {"age_hours": 0}, "year_3": {"age_hours": 8760}},
        "effects": {"motor.gain_scale": {"at_eol": 0.6, "spread": 0.3},
                    "joint.friction_scale": {"curve": "linear", "at_eol": 4.0, "spread": 0.3},
                    "encoder.bias_rad": {"at_eol": 0.05, "signed": True, "spread": 0.3}}}


def _trace(tmp_path, name, domain="sim", params=None, n=8, env=None, latency_ms=None, info=None, seed0=0):
    env = env or Arm3Env()
    rec = TraceRecorder(domain=domain, env="arm3", policy="arm-ik", control_dt=0.02)
    pol = make_policy("arm-ik")
    if latency_ms is not None:
        from groot_eval.latency import with_latency

        pol = with_latency(pol, f"fixed:{latency_ms}", 0.02, "hold")
    evaluate(env, pol, [Scenario(seed=seed0 + i, params=dict(params or {}), name=f"s{i:02d}") for i in range(n)],
             recorder=rec)
    root = rec.write(tmp_path / name)
    if info:
        p = root / "meta" / "trace_info.json"
        p.write_text(json.dumps({**json.loads(p.read_text()), **info}), encoding="utf-8")
    return root


def _edit(root, fn):
    """Rewrite every episode's parquet with fn(df) -> df."""
    for p in sorted((root / "data").rglob("*.parquet")):
        fn(pd.read_parquet(p)).to_parquet(p)


def _edit_state(root, fn):
    def go(df):
        df["observation.state"] = [np.asarray(fn(np.asarray(v, dtype=np.float32)), dtype=np.float32) for v in df["observation.state"]]
        return df
    _edit(root, go)


def _status(results):
    return {r.gate: r.status for r in results}


def _checks(results, gate):
    r = next(x for x in results if x.gate == gate)
    return {f.check for f in r.findings if f.level == "error"}, {f.check for f in r.findings if f.level == "warn"}


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pair")
    return _trace(tmp, "sim", n=10), _trace(tmp, "real", domain="real", n=10, seed0=100, params={"payload_kg": 0.1})


def _copy(src, dst):
    import shutil

    shutil.copytree(src, dst)
    return dst


# -- wear additions -------------------------------------------------------------------------------
def test_thermal_derate_weakens_a_loaded_motor_over_time():
    env = WearEnv(Arm3Env())
    env.reset(Scenario(seed=0, params={"wear.motor[0].thermal_derate": 0.5, "wear.motor[0].thermal_tau_s": 0.2}))
    sent = []
    for _ in range(60):
        env.step(np.array([1.0, 0.0, 0.0, 0.0]))
        sent.append(env._last_sent[0])
    assert sent[0] > 0.9 and sent[-1] == pytest.approx(0.5, abs=0.02)  # full load -> half the torque when hot


def test_sensor_scale_walk_saturation_and_camera_shift():
    env = WearEnv(Arm3Env())
    obs = env.reset(Scenario(seed=3, params={"wear.encoder[0].scale": 0.5, "wear.encoder[1].saturation": 0.1}))
    q = env.env.q
    assert obs["state.joint_pos"][0] == pytest.approx(0.5 * q[0], abs=1e-5)
    assert abs(obs["state.joint_pos"][1]) <= 0.1 + 1e-6
    env = WearEnv(Arm3Env())
    env.reset(Scenario(seed=3, params={"wear.encoder[2].bias_walk": 0.5}))
    drift = []
    for _ in range(100):
        obs, *_ = env.step(np.zeros(4))
        drift.append(float(obs["state.joint_pos"][2] - env.env.q[2]))
    assert np.std(drift) > 0.01 and abs(drift[-1]) > 0.0  # bias wanders, it is not white noise
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, 2] = 200
    out = degrade_image(img, {"shift_px": 3}, np.random.default_rng(0))
    assert out[0, 5, 0] == 200 and out[0, 2, 0] == 0


def test_typical_unit_has_no_spread_and_zero_biases():
    cfg = WearConfig.from_dict(WEAR)
    comps = probe_components(Arm3Env())
    u = cfg.typical_unit(10000, comps)
    assert u["wear.motor[1].gain_scale"] == pytest.approx(0.6)
    assert u["wear.encoder[0].bias_rad"] == 0.0


def test_wear_fit_recovers_the_robot_age(tmp_path):
    cfg = WearConfig.from_dict(WEAR)
    comps = probe_components(Arm3Env())
    worn = cfg.typical_unit(6000, comps)
    worn.pop("wear.age_hours")
    real = _trace(tmp_path, "worn", domain="real", params=worn, env=WearEnv(Arm3Env()), n=6)
    fit = fit_age(lambda: Arm3Env(), TraceSet.open(real), cfg, comps, ages=[0, 2000, 4000, 6000, 8000, 10000])
    assert fit["best"]["age_hours"] == 6000
    assert fit["best"]["score"] < fit["ages"][0]["score"]  # better than assuming a new robot
    assert fit["closest_profile"] == "year_3"


# -- round trip -----------------------------------------------------------------------------------
def test_import_real_round_trip_is_lossless(tmp_path):
    src = _trace(tmp_path, "src", domain="real", n=4, info={"calibration_id": "cal-7", "robot_age_hours": 120})
    _edit(src, lambda df: df.assign(**{"fault.active": np.arange(len(df)) % 2 * 1.0}))
    out, _ = import_real(src, tmp_path / "imported")
    cfg, _ = transfer.load_config(None)
    findings, m = transfer.roundtrip(TraceSet.open(src), TraceSet.open(out), cfg["T0_roundtrip"])
    assert not [f for f in findings if f.level != "ok"], findings
    assert TraceSet.open(out).info["calibration_id"] == "cal-7"


def test_round_trip_catches_lossy_conversion(tmp_path):
    src = _trace(tmp_path, "src", n=4)
    conv = _copy(src, tmp_path / "conv")
    _edit_state(conv, lambda v: np.round(v, 2))  # e.g. an exporter that writes 2 decimals
    _edit(conv, lambda df: df.drop(columns=["inference_latency_ms"]))
    cfg, _ = transfer.load_config(None)
    findings, m = transfer.roundtrip(TraceSet.open(src), TraceSet.open(conv), cfg["T0_roundtrip"])
    errs = {f.check for f in findings if f.level == "error"}
    warns = {f.check for f in findings if f.level == "warn"}
    assert "values" in errs and "columns" in warns
    assert m["max_rel_error"] > 1e-3


# -- cross-domain gates -------------------------------------------------------------------------------
def test_clean_pair_passes_every_gate(pair, tmp_path):
    sim, real = pair
    cfg, _ = transfer.load_config(None)
    res = transfer.run_transfer(TraceSet.open(sim), TraceSet.open(real), cfg, "sim2real", env=Arm3Env(),
                                policy=make_policy("arm-ik"), robot=ROBOT)
    assert all(s in ("pass", "warn") for s in _status(res).values()), transfer.transfer_markdown(res, "s", "r", "sim2real")
    assert transfer.exit_code(res) == 0
    t1 = next(r for r in res if r.gate == "T1_compatibility")
    assert t1.metrics["action_gain_ratio"]  # responsive commands were found and compared


def test_state_key_order_mismatch_is_a_compatibility_error(pair, tmp_path):
    sim, real = pair
    bad = _copy(sim, tmp_path / "reordered")
    mod = json.loads((bad / "meta" / "modality.json").read_text())
    keys = list(mod["state"])
    widths = {k: v["end"] - v["start"] for k, v in mod["state"].items()}
    start, new = 0, {}
    for k in reversed(keys):  # same keys, concatenated in another order
        new[k] = {"start": start, "end": start + widths[k]}
        start += widths[k]
    old = {k: slice(v["start"], v["end"]) for k, v in mod["state"].items()}
    _edit_state(bad, lambda v: np.concatenate([v[old[k]] for k in reversed(keys)]))
    mod["state"] = new
    (bad / "meta" / "modality.json").write_text(json.dumps(mod))
    cfg, _ = transfer.load_config(None)
    res = transfer.run_transfer(TraceSet.open(bad), TraceSet.open(real), cfg, "sim2real")
    errs, _ = _checks(res, "T1_compatibility")
    assert "state.order" in errs
    assert _status(res)["T2_fidelity"] == "skip" and transfer.exit_code(res) == 1


def test_swapped_joints_and_flipped_sign_are_found(pair, tmp_path):
    sim, real = pair
    ts = TraceSet.open(real)
    sl = ts.state_slices()["joint_pos"]
    swapped = _copy(real, tmp_path / "swapped")

    def swap(v):
        v = v.copy()
        v[sl.start], v[sl.start + 1] = v[sl.start + 1], v[sl.start]
        return v
    _edit_state(swapped, swap)
    cfg, _ = transfer.load_config(None)
    errs, _ = _checks(transfer.run_transfer(TraceSet.open(sim), TraceSet.open(swapped), cfg, "real2sim"), "T1_compatibility")
    assert "order" in errs

    flipped = _copy(real, tmp_path / "flipped")
    _edit(flipped, lambda df: df.assign(action=[np.concatenate([-v[:3], v[3:]]).astype(np.float32) for v in df["action"]]))
    errs, _ = _checks(transfer.run_transfer(TraceSet.open(sim), TraceSet.open(flipped), cfg, "real2sim"), "T1_compatibility")
    assert "sign" in errs


def test_fidelity_finds_quantization_clipping_and_smoothing(pair, tmp_path):
    sim, real = pair
    ts = TraceSet.open(real)
    sl = ts.state_slices()
    vel, pos = sl["joint_vel"], sl["joint_pos"]
    lossy = _copy(real, tmp_path / "lossy")

    def damage(v):
        v = v.copy()
        v[pos.start] = np.round(v[pos.start] / 0.05) * 0.05  # coarse encoder / 2-decimal export
        v[vel.start + 1] = np.clip(v[vel.start + 1], -0.2, 0.2)  # saturating logger
        return v
    _edit_state(lossy, damage)

    def smooth(df):
        S = np.stack(df["observation.state"].to_numpy())
        k = np.ones(9) / 9
        S[:, vel.start + 2] = np.convolve(np.pad(S[:, vel.start + 2], 4, mode="edge"), k, mode="valid")
        df["observation.state"] = list(S.astype(np.float32))
        return df
    noisy_ref = _copy(sim, tmp_path / "noisy_ref")
    _edit_state(noisy_ref, lambda v: v + np.random.default_rng(int(abs(v[0]) * 1e6) % 2**32).normal(0, 0.05, v.shape).astype(np.float32))
    _edit(lossy, smooth)
    cfg, _ = transfer.load_config(None)
    res = transfer.run_transfer(TraceSet.open(noisy_ref), TraceSet.open(lossy), cfg, "real2sim")
    errs, warns = _checks(res, "T2_fidelity")
    assert {"precision", "clipping"} <= errs
    assert "bandwidth" in warns


def test_sim2real_direction_checks(pair, tmp_path):
    sim, _ = pair
    real = _trace(tmp_path, "slow_real", domain="real", n=8, seed0=50, latency_ms=70, info={"robot_age_hours": 9000})
    cfg, _ = transfer.load_config(None)
    tight = {**ROBOT, "max_action_rate": 0.05}
    res = transfer.run_transfer(TraceSet.open(sim), TraceSet.open(real), cfg, "sim2real", robot=tight,
                                wear_profiles={"new": 0, "year_3": 8760})
    errs, warns = _checks(res, "T4_direction")
    assert {"latency", "robot.action_rate", "wear"} <= errs
    t4 = next(r for r in res if r.gate == "T4_direction")
    assert any("year_3" in f.detail for f in t4.findings if f.check == "wear")


def test_real2sim_direction_and_env_usability(pair, tmp_path):
    sim, real = pair
    off = _copy(real, tmp_path / "offset")
    sl = TraceSet.open(real).state_slices()["joint_pos"]
    _edit_state(off, lambda v: np.concatenate([v[: sl.start], v[sl] + 1.0, v[sl.stop:]]))  # zero offset applied twice
    cfg, _ = transfer.load_config(None)
    res = transfer.run_transfer(TraceSet.open(sim), TraceSet.open(off), cfg, "real2sim", env=Arm3Env(), robot=ROBOT)
    t3e, _ = _checks(res, "T3_usability")
    t4e, t4w = _checks(res, "T4_direction")
    assert "env.range" in t3e
    assert "robot.limits" in t4e
    assert {"calibration", "actions", "wear"} <= t4w


def test_policy_usability_catches_a_policy_that_cannot_read_the_data(pair):
    sim, real = pair

    class Broken:
        name = "broken"

        def reset(self, task):
            pass

        def act(self, obs):
            return np.full(4, np.nan) if "state.joint_pos" in obs else np.zeros(4)

    cfg, _ = transfer.load_config(None)
    errs, _ = _checks(transfer.run_transfer(TraceSet.open(sim), TraceSet.open(real), cfg, "sim2real", policy=Broken()),
                      "T3_usability")
    assert "policy.finite" in errs


def test_cli_transfer_and_wear_fit(pair, tmp_path):
    sim, real = pair
    code = main(["transfer", "--sim", str(sim), "--real", str(real), "--robot", "configs/robots/arm3.yaml",
                 "--env", "arm3", "--policy", "arm-ik", "--config", "configs/transfer/defaults.yaml",
                 "--roundtrip", str(real), str(real), "--out", str(tmp_path / "t")])
    assert code == 0
    md = (tmp_path / "t" / "transfer.md").read_text(encoding="utf-8")
    assert "Transfer verification (sim2real): PASS" in md and "T0_roundtrip" in md
    data = json.loads((tmp_path / "t" / "transfer.json").read_text(encoding="utf-8"))
    assert [g["gate"] for g in data["gates"]] == ["T0_roundtrip", "T1_compatibility", "T2_fidelity", "T3_usability", "T4_direction"]

    wear = tmp_path / "wear.yaml"
    wear.write_text(json.dumps(WEAR), encoding="utf-8")
    code = main(["wear-fit", str(real), "--env", "arm3", "--wear", str(wear), "--ages", "0,5000,10000",
                 "--episodes", "3", "--out", str(tmp_path / "wf")])
    assert code == 0
    assert "Wear parameters for a digital twin" in (tmp_path / "wf" / "wear_fit.md").read_text(encoding="utf-8")
    assert math.isfinite(json.loads((tmp_path / "wf" / "wear_fit.json").read_text())["best"]["score"])
