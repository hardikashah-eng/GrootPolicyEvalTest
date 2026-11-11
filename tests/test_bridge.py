"""Hardware bridge: shadow and mirror modes, watchdog, step-by-step report, G5 and repro hand-off."""

import json
import sys
import types

import numpy as np
import pytest

from groot_eval.bridge import (BridgeConfig, BridgeSession, PolicySource, SimRobot, TraceRobot, TraceSource,
                               build_report, repro_records)
from groot_eval.bridge_ros2 import joint_state_to_obs
from groot_eval.closed_loop import evaluate
from groot_eval.envs.arm3 import Arm3Env
from groot_eval.policies import make_policy
from groot_eval.robot_cli import main
from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario


class Const:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def reset(self, task):
        pass

    def act(self, obs):
        return self.value.copy()


def _cfg(**kw):
    wd = kw.pop("watchdog", {})
    return BridgeConfig.load(None, **kw, watchdog=wd)


def _session(mode="shadow", robot_params=None, source=None, robot_policy="arm-ik", **kw):
    robot = SimRobot(Arm3Env(), make_policy(robot_policy) if robot_policy else None, robot_params,
                     drop=kw.pop("drop", ()))
    return robot, BridgeSession(robot, Arm3Env(), _cfg(mode=mode, **kw), source)


def test_shadow_identical_twin_tracks_exactly():
    _, s = _session()
    logs = s.run([Scenario(seed=i) for i in range(2)])
    assert all(lg.trip is None for lg in logs)
    assert max(max(e.values()) for lg in logs for e in lg.errors) < 1e-5
    assert all(lg.robot_result.get("success") is not None for lg in logs)  # robot side ended the episode


def test_shadow_gap_trips_once_and_keeps_watching(tmp_path):
    robot, s = _session(robot_params={"friction": 1.8})
    log = s.run_episode(Scenario(seed=0))
    assert log.trip is not None and log.trip.reason == "divergence"
    assert len(log.errors) - 1 > log.trip.tick  # shadow does not stop at the trip
    assert robot.stops == []  # and never commands the robot
    res = build_report(s, tmp_path, robot="sim", twin_spec="arm3", make_twin=Arm3Env)
    assert res["episodes"][0]["trip"]["tick"] == log.trip.tick
    shown = [r["tick"] for r in res["episodes"][0]["steps"]]
    assert all(t in shown for t in range(log.trip.tick - 5, log.trip.tick + 6))  # every tick around the trip
    assert res["g5"]["status"] in ("pass", "warn", "fail") and "nrmse_at_horizon" in res["g5"]["metrics"]
    assert (tmp_path / "steps_episode0.csv").read_text().startswith("tick,stamp,since_sync,worst")
    real, twin = TraceSet.open(tmp_path / "robot"), TraceSet.open(tmp_path / "twin")
    assert real.domain == "real" and twin.domain == "sim"
    assert real.episode_meta()[0]["failure_reason"] == "watchdog:divergence"
    assert "bridge.err.joint_pos" in real.frames()[0]


def test_mirror_trip_safe_stops_and_ends_session():
    robot, s = _session("mirror", robot_params={"friction": 1.8, "action_delay": 3},
                        source=PolicySource(make_policy("arm-ik")), robot_policy=None,
                        watchdog={"on_trip": "stop"})
    logs = s.run([Scenario(seed=i) for i in range(3)])
    assert len(logs) == 1 and logs[0].trip is not None
    assert len(robot.stops) == 1 and robot.stops[0][1] == "stop" and "divergence" in robot.stops[0][0]
    assert len(logs[0].errors) - 1 == logs[0].trip.tick  # stopped at the trip


def test_partial_mirror_drives_only_masked_dims():
    robot = SimRobot(Arm3Env(), Const([-0.1, -0.1, -0.1, 0.0]))
    s = BridgeSession(robot, Arm3Env(), _cfg(mode="mirror", command_mask=[0], max_ticks=5,
                                             watchdog={"max_error": {}}), PolicySource(Const([0.2, 0.2, 0.2, 1.0])))
    log = s.run_episode(Scenario(seed=0))
    assert np.allclose(log.commands[0], [0.2, -0.1, -0.1, 0.0])  # bridge drives joint 0, the robot the rest
    assert max(max(e.values()) for e in log.errors) < 1e-5  # the twin got the merged command too


def test_missing_samples_and_state_limits_trip():
    _, s = _session(drop=(3, 4, 5), watchdog={"max_missing": 3})
    log = s.run_episode(Scenario(seed=0))
    assert log.trip.reason == "missing" and log.missing == 3
    _, s = _session(watchdog={"state_limits": {"joint_vel": [-0.05, 0.05]}})
    assert s.run_episode(Scenario(seed=0)).trip.reason == "state_limit"


def test_command_step_watchdog():
    class Jumpy(Const):
        def act(self, obs):
            self.value = -self.value
            return self.value.copy()

    _, s = _session("mirror", source=PolicySource(Jumpy([0.5, 0.5, 0.5, 0.0])), robot_policy=None,
                    watchdog={"max_command_step": 0.5})
    assert s.run_episode(Scenario(seed=0)).trip.reason == "command_step"


def _recorded(tmp_path, n=2, gap_at=None):
    rec = TraceRecorder(domain="real", env="arm3", policy="arm-ik", control_dt=0.02)
    evaluate(Arm3Env(), make_policy("arm-ik"), [Scenario(seed=i) for i in range(n)], recorder=rec)
    root = rec.write(tmp_path / "real")
    if gap_at is not None:
        import pandas as pd

        p = sorted((root / "data").rglob("*.parquet"))[0]
        df = pd.read_parquet(p)
        df.loc[gap_at:, "timestamp"] += 1.0
        df.to_parquet(p)
    return root


def test_offline_shadow_from_recorded_trace(tmp_path):
    robot = TraceRobot(_recorded(tmp_path))
    s = BridgeSession(robot, Arm3Env(), _cfg())
    logs = s.run(robot.scenarios())
    assert len(logs) == 2 and all(lg.trip is None for lg in logs)
    assert max(max(e.values()) for lg in logs for e in lg.errors) < 1e-3  # float32 recording vs the twin
    with pytest.raises(ValueError, match="shadow"):
        BridgeSession(TraceRobot(_recorded(tmp_path / "b")), Arm3Env(), _cfg(mode="mirror"), PolicySource(Const([0])))


def test_stale_robot_clock_trips(tmp_path):
    robot = TraceRobot(_recorded(tmp_path, n=1, gap_at=10))
    s = BridgeSession(robot, Arm3Env(), _cfg(watchdog={"max_stale_s": 0.1}))
    assert s.run(robot.scenarios())[0].trip.reason == "stale"


def test_mirror_replays_recorded_commands_on_the_robot(tmp_path):
    root = _recorded(tmp_path, n=2)
    src = TraceSource(root)
    robot = SimRobot(Arm3Env())
    s = BridgeSession(robot, Arm3Env(), _cfg(mode="mirror"), src)
    logs = s.run(src.scenarios())
    acts = TraceSet.open(root).per_episode("action")
    assert [len(lg.commands) for lg in logs] == [len(a) for a in acts]
    assert np.allclose(np.stack(logs[1].commands), acts[1], atol=1e-6)


def test_repro_records_match_the_failure_record_format(tmp_path):
    _, s = _session(robot_params={"friction": 1.8})
    s.run([Scenario(seed=1), Scenario(seed=0)])
    build_report(s, tmp_path, robot="sim", twin_spec="arm3", g5=False)
    recs = json.loads((tmp_path / "repro_cases.json").read_text())
    ts = TraceSet.open(tmp_path / recs[0]["trace"])  # relative to the records file, as repro.load_failures reads it
    for r in recs:
        meta = ts.episode_meta()[r["episode_index"]]
        assert meta["success"] is False and meta["failure_reason"] == r["failure_reason"]
        assert meta["scenario"]["seed"] == r["scenario"]["seed"]
    assert repro_records(tmp_path / "robot", [], tmp_path) == []


def test_inline_repro_handoff(tmp_path, monkeypatch):
    calls = {}
    fake = types.ModuleType("groot_eval.repro")
    fake.load_config = lambda path=None, **kw: calls.setdefault("cfg", {"path": path, **kw})
    fake.load_failures = lambda p: calls.setdefault("cases", json.loads(open(p).read()))
    fake.reproduce_batch = lambda cases, cfg: {"reproduction_rate": 1.0, "verdicts": {"reproduced": len(cases)}}
    fake.repro_markdown = lambda summary, cfg: "# repro\n"
    fake.to_json = lambda summary: json.dumps(summary)
    monkeypatch.setitem(sys.modules, "groot_eval.repro", fake)
    _, s = _session(robot_params={"friction": 1.8})
    s.run([Scenario(seed=0)])
    res = build_report(s, tmp_path, robot="sim", twin_spec="arm3:friction=1.0", g5=False,
                       run_repro={"policy": "arm-ik", "config": None})
    assert calls["cfg"]["env"] == "arm3" and calls["cfg"]["policy"] == "arm-ik"
    assert res["repro"]["verdicts"] == {"reproduced": 1} and (tmp_path / "repro.md").exists()


def test_config_validation():
    from groot_eval.bridge_ros2 import Ros2Config

    assert BridgeConfig.load("configs/bridge/shadow.yaml").watchdog.max_error["joint_pos"] == 0.25
    assert BridgeConfig.load("configs/bridge/mirror_arm.yaml").command_mask == [0, 1, 2]
    assert len(Ros2Config.load("configs/bridge/ros2.yaml").joint_names) == 3
    with pytest.raises(ValueError, match="unknown bridge config"):
        BridgeConfig.load(None, speed=2)
    with pytest.raises(ValueError, match="mode"):
        BridgeConfig.load(None, mode="drive")
    with pytest.raises(ValueError, match="command source"):
        BridgeSession(SimRobot(Arm3Env()), Arm3Env(), _cfg(mode="mirror"))


def test_joint_state_reorder():
    obs = joint_state_to_obs(["a", "b"], ["b", "x", "a"], [2.0, 9.0, 1.0], [0.2, 0.9, 0.1], [], ["a", "b"])
    assert obs["state.joint_pos"].tolist() == [1.0, 2.0]
    assert np.allclose(obs["state.joint_vel"], [0.1, 0.2]) and "state.joint_torque" not in obs
    with pytest.raises(KeyError):
        joint_state_to_obs(["c"], ["a"], [1.0], [], [], ["c"])


def test_cli_shadow_and_mirror(tmp_path):
    assert main(["bridge", "--robot", "sim:arm3", "--robot-policy", "arm-ik", "--twin", "arm3", "--episodes", "2",
                 "--out", str(tmp_path / "s")]) == 0
    res = json.loads((tmp_path / "s" / "bridge.json").read_text())
    assert res["comparison"]["episodes"] == 2 and res["g5"]["status"] == "pass"
    assert main(["bridge", "--mode", "mirror", "--robot", "sim:arm3:friction=1.8,action_delay=3", "--twin", "arm3",
                 "--source", "policy:arm-ik", "--out", str(tmp_path / "m")]) == 1  # tripped
    assert "Watchdog tripped" in (tmp_path / "m" / "bridge.md").read_text()
