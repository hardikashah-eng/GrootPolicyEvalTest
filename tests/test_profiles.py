"""Calibration loop and robot profiles: fit from a robot's trace, version, approve, load in every consumer."""

import json

import numpy as np
import pytest
import yaml

from groot_eval.actuators.core import WearTable
from groot_eval.calibration import calibrate, fit_fleet, refine_joints
from groot_eval.closed_loop import evaluate
from groot_eval.degradation.components import probe_components
from groot_eval.degradation.wear import WearConfig, WearEnv
from groot_eval.envs.arm3 import Arm3Env
from groot_eval.policies import make_policy
from groot_eval.profiles import (ProfileError, ProfileRegistry, RobotProfile, diff, params_from_wear_lists,
                                 resolve_profile, wear_lists_from_params)
from groot_eval.robot_cli import main
from groot_eval.scheduler import TimingProfile
from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario

WEAR = "configs/wear/arm3_profiles.yaml"


def _unit(age, joint1_friction=1.0):
    cfg = WearConfig.load(WEAR)
    p = cfg.typical_unit(age, probe_components(Arm3Env()))
    p.pop("wear.age_hours")
    p["wear.joint[1].friction_scale"] *= joint1_friction
    return p


def _worn_trace(tmp_path, params, n=4, name="real"):
    """A trace like a real robot's: the wear is in the physics, not written down in the episode metadata."""
    rec = TraceRecorder(domain="real", env="arm3", policy="arm-ik", control_dt=0.02)
    evaluate(WearEnv(Arm3Env()), make_policy("arm-ik"), [Scenario(seed=i, params=params) for i in range(n)], recorder=rec)
    root = rec.write(tmp_path / name)
    meta = root / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    for r in rows:
        r["scenario"]["params"] = {}
    meta.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _profile(robot_id="arm3-sn1", hours=None, friction=(1.0, 1.0, 1.0), **kw):
    vals = wear_lists_from_params({}, 3)
    vals["friction"] = list(friction)
    return RobotProfile(robot_id=robot_id, joint_names=["shoulder", "elbow", "wrist"], embodiment="arm3",
                        operating_hours=hours, wear={"values": vals}, sim={"env": "arm3", "params": {}}, **kw)


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cal")
    truth = _unit(4000, joint1_friction=1.6)
    root = _worn_trace(tmp, truth)
    prof, rep = calibrate(root, robot_id="arm3-sn1", env="arm3", joint_names=["shoulder", "elbow", "wrist"],
                          wear_config=WEAR, operating_hours=4000, limits="configs/robots/arm3.yaml")
    return prof, rep, truth, root


def test_calibration_recovers_age_and_the_worn_joint(calibrated):
    prof, rep, truth, root = calibrated
    fit = prof.provenance["fit"]
    assert fit["score"] < 0.3 * fit["baseline_score"]  # much closer than a new unit
    assert 2000 <= prof.wear["effective_age_hours"] <= 7000
    f = prof.wear["values"]["friction"]
    assert f[1] > f[0] and f[1] > f[2]  # the joint that was made stiffer is found
    assert [s["stage"].split(" ")[0] for s in rep["stages"]][:2] == ["new", "effective"]
    tr = prof.provenance["traces"][0]
    assert tr["sha256"] and tr["episodes"] == 4 and tr["domain"] == "real"
    assert prof.state_limits["joint_pos"] and prof.control_hz == 50  # limits copied from the robot config
    assert prof.timing["control_hz"] == pytest.approx(50.0)


def test_profile_roundtrip_hash_and_tamper_detection(calibrated, tmp_path):
    prof = calibrated[0]
    path = prof.save(tmp_path / "p.yaml")
    q = RobotProfile.load(path)
    assert q.content_hash == prof.content_hash and q.to_dict() == prof.to_dict()
    data = yaml.safe_load(path.read_text())
    data["wear"]["values"]["friction"][0] = 9.0
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ProfileError, match="content_hash"):
        RobotProfile.load(path)
    assert RobotProfile.load(path, verify=False).wear["values"]["friction"][0] == 9.0


def test_schema_validation_lists_every_problem(tmp_path):
    bad = {"robot_id": "a b", "joint_names": ["x", "x"], "wear": {"values": {"friction": [1.0], "warp": [1]}},
           "actuator": {"stiffness": [1, 2, 3, 4]}, "faults": {"rates_per_hour": {"meteor": 1}}, "colour": "red",
           "timing": {"control_hz": 50, "missed_tick_rate": 2}}
    with pytest.raises(ProfileError) as e:
        RobotProfile.from_dict(bad)
    text = " ".join(e.value.problems)
    for word in ("slashes or spaces", "duplicates", "warp", "one number per joint", "actuator.stiffness", "meteor",
                 "colour", "missed_tick_rate"):
        assert word in text, word
    out = _profile(friction=(1.0, 50.0, 1.0)).to_dict()
    with pytest.raises(ProfileError, match="plausible range"):
        RobotProfile.from_dict(out, verify=False)


def test_consumers_load_the_profile_directly(calibrated, tmp_path):
    prof = calibrated[0]
    path = prof.save(tmp_path / "p.yaml")
    # Isaac Lab actuator: by joint name, in the simulator's order
    w = RobotProfile.load(path).wear_for(["wrist", "shoulder", "elbow"])
    table = WearTable.from_robot_profile(w, num_envs=4, num_joints=3)
    assert table.values["friction"][0].tolist() == pytest.approx([prof.wear["values"]["friction"][i] for i in (2, 0, 1)])
    with pytest.raises(KeyError, match="knee"):
        prof.wear_for(["knee"])
    assert prof.wear_for(["knee", "elbow"], missing="default")["friction"][0] == 1.0
    # scheduler
    assert TimingProfile.load(path).control_hz == pytest.approx(50.0)
    # transfer checks read the same file as a robot config
    from groot_eval.transfer import load_robot

    assert load_robot(path)["state_limits"]["joint_pos"]
    # CPU twin: replaying the robot's trace in the profiled twin matches the fitted score
    from groot_eval.replay import replay_trace

    env, params = prof.make_twin()
    score = replay_trace(env, TraceSet.open(calibrated[3]), params).score
    assert score == pytest.approx(prof.provenance["fit"]["score"], rel=1e-6)


def test_wear_params_roundtrip():
    p = _unit(6000)
    lists = wear_lists_from_params(p, 3)
    back = params_from_wear_lists(lists)
    known = {k: v for k, v in p.items() if any(k.endswith("." + e) for e in ("gain_scale", "friction_scale", "backlash_rad"))}
    for k, v in known.items():
        assert back[k] == pytest.approx(v)


def test_refine_does_not_accept_additive_values(tmp_path):
    with pytest.raises(ValueError, match="multiplicative"):
        refine_joints(Arm3Env, None, {}, 3, keys=("backlash",))


def test_registry_versions_approval_and_resolution(tmp_path):
    reg = ProfileRegistry(tmp_path / "reg")
    p1, created = reg.publish(_profile(hours=1000))
    assert created and p1.version == 1
    again, created = reg.publish(RobotProfile.load(p1.path))
    assert not created and again.version == 1  # same content: no new version
    p2, _ = reg.publish(_profile(hours=1000, friction=(1.0, 1.3, 1.0)), note="after gearbox swap")
    assert p2.version == 2 and reg.resolve("arm3-sn1", "latest").version == 2
    with pytest.raises(KeyError, match="no approved"):
        reg.resolve("arm3-sn1")
    reg.approve("arm3-sn1", 1, by="qa")
    assert reg.resolve("arm3-sn1").version == 1 and reg.resolve("arm3-sn1").status == "approved"
    assert resolve_profile(f"registry:{tmp_path / 'reg'}:arm3-sn1@2").version == 2
    assert diff(reg.get("arm3-sn1", 1), reg.get("arm3-sn1", 2)) == ["wear.friction: [1.0, 1.0, 1.0] -> [1.0, 1.3, 1.0]"]
    (tmp_path / "reg" / "arm3-sn1" / "v0001.yaml").write_text(
        (tmp_path / "reg" / "arm3-sn1" / "v0002.yaml").read_text())
    with pytest.raises(ProfileError):
        reg.get("arm3-sn1", 1)  # a published file was replaced


def test_fleet_fit_recovers_the_end_of_life_value(tmp_path):
    cfg = WearConfig.load(WEAR)
    eff = next(e for e in cfg.effects if (e.kind, e.key) == ("joint", "friction_scale"))
    true_eol = 2.2  # the placeholder says 3.0
    profiles = []
    for i, hours in enumerate((1000, 3000, 5000, 8000)):
        vals = []
        for j in range(3):
            load = cfg.load_factor.get(f"joint[{j}]", cfg.load_factor.get("default", 1.0))
            from groot_eval.degradation.wear import Effect

            y = Effect("j", "f", eff.curve, 0.0, 1.0).value(hours, cfg.eol_hours, load, 1.0, 1.0)
            vals.append(1.0 + (true_eol - 1.0) * y * (1.0 + 0.05 * ((i + j) % 3 - 1)))
        profiles.append(_profile(robot_id=f"sn{i}", hours=hours, friction=tuple(vals)))
    profiles.append(_profile(robot_id="no-hours"))
    out, report = fit_fleet(profiles, WEAR)
    row = next(r for r in report["effects"] if r["effect"] == "joint.friction_scale")
    assert row["fitted_at_eol"] == pytest.approx(true_eol, abs=0.05) and row["points"] == 12
    assert out["effects"]["joint.friction_scale"]["at_eol"] == pytest.approx(true_eol, abs=0.05)
    assert report["skipped"] == ["no-hours"]
    assert WearConfig.from_dict(out).eol_hours == cfg.eol_hours  # still a valid wear config


def test_cli_calibrate_publish_approve_and_bridge(tmp_path, calibrated):
    root = calibrated[3]
    reg = tmp_path / "reg"
    assert main(["calibrate", str(root), "--robot-id", "arm3-sn7", "--env", "arm3", "--wear", WEAR,
                 "--refine", "friction", "--rounds", "1", "--operating-hours", "4000", "--registry", str(reg),
                 "--out", str(tmp_path / "p.yaml")]) == 0
    assert main(["profile", "validate", str(tmp_path / "p.yaml")]) == 0
    assert main(["profile", "approve", "arm3-sn7", "1", "--registry", str(reg), "--by", "qa"]) == 0
    assert main(["profile", "list", "--registry", str(reg)]) == 0
    assert main(["profile", "get", "arm3-sn7", "--registry", str(reg), "--out", str(tmp_path / "prod.yaml")]) == 0
    assert RobotProfile.load(tmp_path / "prod.yaml").robot_id == "arm3-sn7"
    # the profiled twin shadows the recorded robot more closely than a new-unit twin
    for name, extra in (("new", []), ("prof", ["--twin-profile", f"registry:{reg}:arm3-sn7"])):
        assert main(["bridge", "--robot", f"trace:{root}", "--twin", "arm3", "--no-g5", "--out",
                     str(tmp_path / name), *extra]) == 0
    new = json.loads((tmp_path / "new" / "bridge.json").read_text())["comparison"]["nrmse"]
    prof = json.loads((tmp_path / "prof" / "bridge.json").read_text())["comparison"]["nrmse"]
    assert prof < 0.5 * new
    bad = tmp_path / "bad.yaml"
    bad.write_text("robot_id: x\njoint_names: []\n")
    assert main(["profile", "validate", str(bad)]) == 1


def test_cli_fleet_fit(tmp_path):
    files = []
    for i, hours in enumerate((2000, 6000)):
        files.append(str(_profile(robot_id=f"sn{i}", hours=hours, friction=(1.2 + i, 1.3 + i, 1.1 + i)).save(
            tmp_path / f"p{i}.yaml")))
    assert main(["fleet-fit", *files, "--wear", WEAR, "--out", str(tmp_path / "fitted.yaml")]) == 0
    fitted = WearConfig.load(tmp_path / "fitted.yaml")
    assert np.isfinite([e.at_eol for e in fitted.effects]).all()
    assert "Fleet wear fit" in (tmp_path / "fitted.md").read_text()


def test_calibrate_with_system_identification(tmp_path):
    rec = TraceRecorder(domain="real", env="arm3", policy="arm-ik", control_dt=0.02)
    evaluate(Arm3Env(), make_policy("arm-ik"), [Scenario(seed=i, params={"friction": 1.5, "action_delay": 1})
                                                for i in range(3)], recorder=rec)
    root = rec.write(tmp_path / "real")
    prof, rep = calibrate(root, robot_id="sn9", env="arm3", sysid_config="configs/arm3_sysid.yaml", refine=())
    assert prof.sim["params"]["action_delay"] == 1
    assert prof.sim["params"]["friction"] == pytest.approx(1.5, abs=0.31)
    assert rep["score"] < 0.5 * rep["baseline"] and "wear" not in prof.to_dict()


def test_bridge_twin_parameters_stay_on_the_twin(tmp_path):
    # the twin gets friction 1.8; the stand-in robot keeps its own (default) physics, so they diverge
    assert main(["bridge", "--robot", "sim:arm3", "--robot-policy", "arm-ik", "--twin", "arm3:friction=1.8",
                 "--no-g5", "--out", str(tmp_path / "b")]) == 0
    res = json.loads((tmp_path / "b" / "bridge.json").read_text())
    assert res["comparison"]["nrmse"] > 0.05
