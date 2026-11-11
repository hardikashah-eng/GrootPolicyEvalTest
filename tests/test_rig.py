"""Run seeds and the test rig: independent streams, seeds in traces, determinism check, rig publishers and faults."""

import json
import sys
import types

import numpy as np
import pytest

from groot_eval.rig import MockPlant, RigConfig, RigSession, StreamFault, TopicSpec, compare_to_ideal, run_rig
from groot_eval.rig_ros2 import array_payload, image_payload, topic_name
from groot_eval.robot_cli import main
from groot_eval.seeding import RunSeed, determinism_check, episode_fingerprints, first_difference
from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario

RIG = "configs/rig/arm3_streams.yaml"


# -- run seeds ---------------------------------------------------------------------------
def test_streams_are_named_independent_and_stable():
    r = RunSeed(42)
    assert r.derive("faults", 3) == RunSeed(42).derive("faults", 3)
    assert r.derive("faults", 3) != r.derive("faults", 4) != r.derive("latency", 3)
    assert r.derive("faults", 3) != RunSeed(43).derive("faults", 3)
    a = r.rng("latency", 0).normal(size=5)
    r.rng("faults", 0).normal(size=1000)  # drawing from one stream does not shift another
    assert np.array_equal(a, r.rng("latency", 0).normal(size=5))
    assert r.scenario_seeds(10)[:4] == r.scenario_seeds(4)  # growing the run keeps earlier scenarios
    assert set(r.streams(0)) >= {"scenario", "faults", "treadmill", "rig"}


def test_seed_travels_with_the_trace_and_check_finds_nondeterminism(tmp_path):
    assert main(["seed-check", "--episodes", "3", "--seed", "11", "--out", str(tmp_path / "ok")]) == 0
    assert RunSeed.from_trace(tmp_path / "ok" / "runs" / "run0") == RunSeed(11)
    meta = TraceSet.open(tmp_path / "ok" / "runs" / "run0").episode_meta()[2]
    assert meta["run_seed"]["episode"] == 2 and meta["run_seed"]["streams"]["faults"] == RunSeed(11).derive("faults", 2)

    class Jittery:  # a controller with an unseeded random source
        def reset(self, task):
            pass

        def act(self, obs):
            return np.random.default_rng().normal(0, 0.3, 4)

    def run(seed, out):
        from groot_eval.closed_loop import evaluate
        from groot_eval.envs.arm3 import Arm3Env

        rec = TraceRecorder(domain="sim", env="arm3", policy="jittery", control_dt=0.02)
        evaluate(Arm3Env(), Jittery(), [Scenario(seed=s) for s in seed.scenario_seeds(2)], recorder=rec)
        return rec.write(out)

    rep = determinism_check(run, 5, tmp_path / "bad")
    assert not rep.deterministic and rep.differences[0]["first_step"] == 0
    assert "Not deterministic" in rep.markdown()
    with pytest.raises(KeyError, match="run_seed"):
        RunSeed.from_trace(tmp_path / "bad" / "run0")


def test_treadmill_and_rig_runs_are_deterministic(tmp_path):
    assert main(["seed-check", "--env", "treadmill", "--policy", "walker-capture", "--treadmill",
                 "configs/treadmill/trip_ladder.yaml", "--episodes", "4", "--out", str(tmp_path / "t")]) == 0
    assert main(["seed-check", "--rig", RIG, "--episodes", "3", "--out", str(tmp_path / "r")]) == 0
    a, b = (tmp_path / "t" / "runs" / f"run{i}" for i in (0, 1))
    assert episode_fingerprints(a) == episode_fingerprints(b)
    assert all(d["first_step"] is None for d in first_difference(a, b))


# -- rig ---------------------------------------------------------------------------------
def _cfg(**kw):
    base = {"control_hz": 50, "duration_s": 1.0, "plant": {"kind": "twin", "env": "arm3"}, "policy": "arm-ik",
            "topics": [{"key": "state.joint_pos", "rate_hz": 100}, {"key": "state.joint_vel", "rate_hz": 100}]}
    base.update(kw)
    return RigConfig.from_dict(base)


def test_ideal_streams_reproduce_the_plain_closed_loop_run():
    from groot_eval.closed_loop import run_episode
    from groot_eval.envs.arm3 import Arm3Env
    from groot_eval.policies import make_policy

    cfg = _cfg(duration_s=3.0,
               topics=[{"key": "state.joint_pos", "rate_hz": 1000}, {"key": "state.joint_vel", "rate_hz": 1000}])
    stats = run_rig(cfg, seed=1, episodes=1)[0]
    ref = run_episode(Arm3Env(), make_policy("arm-ik"), Scenario(seed=RunSeed(1).scenario_seeds(1)[0]))
    assert stats.success == ref.success
    assert stats.ticks == pytest.approx(ref.steps, abs=2)  # the first tick waits for the first messages


def test_rates_drops_delay_and_staleness():
    cfg = _cfg(duration_s=2.0, topics=[
        {"key": "state.joint_pos", "rate_hz": 200},
        {"key": "state.joint_vel", "rate_hz": 200, "drop_rate": 0.25},
        {"key": "state.ee_pos", "rate_hz": 20, "delay_ms": 80, "max_age_ms": 60}])
    cfg.plant = {"kind": "mock", "dim": 3}
    cfg.policy = "pkg:none"
    s = RigSession(cfg, RunSeed(3), policy=_Hold(), plant=_MockWithEE())
    st = s.run_episode(Scenario(seed=0))
    assert st.published["state.joint_pos"] == pytest.approx(400, abs=2)
    assert 0.18 < st.dropped["state.joint_vel"] / st.published["state.joint_vel"] < 0.32
    assert st.max_age_ms["state.ee_pos"] >= 80  # never fresher than the transport delay
    assert st.stale_ticks["state.ee_pos"] > 0.5 * st.ticks  # 80 ms delay vs a 60 ms limit
    assert st.stale_ticks["state.joint_pos"] == 0


class _Hold:
    def reset(self, task):
        pass

    def act(self, obs):
        return np.zeros(3)


class _MockWithEE(MockPlant):
    def __init__(self):
        super().__init__(3, 0.02)

    def _obs(self):
        return {**super()._obs(), "state.ee_pos": np.zeros(2, dtype=np.float32)}


def test_stream_faults_freeze_drop_bias_noise():
    seen = []

    class Spy(_Hold):
        def act(self, obs):
            seen.append(float(obs["state.joint_pos"][0]))
            return np.full(3, len(seen) * 0.01)  # the mock plant follows: joint_pos ramps up

    cfg = _cfg(duration_s=1.0, topics=[{"key": "state.joint_pos", "rate_hz": 100}],
               faults=[{"topic": "state.joint_pos", "kind": "freeze", "start_s": 0.4, "duration_s": 0.2},
                       {"topic": "state.joint_pos", "kind": "bias", "start_s": 0.8, "severity": 5.0}])
    st = RigSession(cfg, RunSeed(0), policy=Spy(), plant=MockPlant(3, 0.02)).run_episode(Scenario(seed=0))
    ticks = np.array(seen)
    frozen = ticks[21:29]  # control ticks at ~0.44-0.58 s
    assert np.ptp(frozen) == 0 and ticks[35] > frozen[0]  # held during the freeze, moving after
    assert ticks[-1] > 4.0  # the bias reached the controller
    assert {f.split("@")[0] for f in st.faults_fired} == {"freeze:state.joint_pos", "bias:state.joint_pos"}
    with pytest.raises(ValueError, match="unknown stream fault"):
        StreamFault("state.joint_pos", "teleport", 0.0)


def test_signal_replay_and_live_sources(tmp_path, monkeypatch):
    from groot_eval.closed_loop import evaluate
    from groot_eval.envs.arm3 import Arm3Env
    from groot_eval.policies import make_policy

    rec = TraceRecorder(domain="real", env="arm3", policy="arm-ik", control_dt=0.02)
    seeds = RunSeed(2).scenario_seeds(2)
    evaluate(Arm3Env(), make_policy("arm-ik"), [Scenario(seed=s) for s in seeds], recorder=rec)
    root = rec.write(tmp_path / "real")
    fake = types.ModuleType("fake_cam")
    fake.frame = lambda: np.full((4, 6, 3), 7, dtype=np.uint8)
    monkeypatch.setitem(sys.modules, "fake_cam", fake)
    got = {}

    class Look(_Hold):
        def act(self, obs):
            got.setdefault("target", obs["state.target"].copy())
            got["video"] = obs["video.front"]
            got["sig"] = float(obs["state.gripper"][0])
            return np.zeros(4)

    cfg = _cfg(duration_s=0.5, topics=[
        {"key": "state.target", "rate_hz": 50, "source": {"kind": "replay", "trace": str(root)}},
        {"key": "state.gripper", "rate_hz": 50, "source": {"kind": "signal", "shape": "step", "value": 0.0,
                                                           "amplitude": 1.0, "at_s": 0.2}},
        {"key": "video.front", "rate_hz": 30, "source": {"kind": "live", "fn": "fake_cam:frame"}}])
    s = RigSession(cfg, RunSeed(2), policy=Look())
    s.run_episode(Scenario(seed=seeds[1]), 1)
    recorded = TraceSet.open(root).per_episode("observation.state")[1][0, TraceSet.open(root).state_slices()["target"]]
    assert np.allclose(got["target"], recorded)  # the episode with the same seed was replayed
    assert got["video"].shape == (4, 6, 3) and got["sig"] == 1.0


def test_compare_to_ideal_and_real_clock():
    cmp = compare_to_ideal(RigConfig.load(RIG), seed=7, episodes=6)
    assert cmp["ideal"]["success"] == 6 and cmp["command_rms_drift"] > 0
    assert cmp == compare_to_ideal(RigConfig.load(RIG), seed=7, episodes=6)  # same seed, same result
    cfg = _cfg(duration_s=0.3, clock="real")
    import time

    t0 = time.monotonic()
    st = run_rig(cfg, seed=0, episodes=1)[0]
    assert time.monotonic() - t0 >= 0.28 and st.ticks > 0  # paced on the wall clock


def test_config_validation_and_ros_payloads():
    with pytest.raises(ValueError, match="state"):
        TopicSpec("joint_pos", 10)
    with pytest.raises(ValueError, match="faults on topics"):
        _cfg(faults=[{"topic": "state.nope", "kind": "drop", "start_s": 0}])
    with pytest.raises(ValueError, match="unknown rig config"):
        _cfg(speed=3)
    assert array_payload(np.array([[1, 2], [3, 4]])) == [1.0, 2.0, 3.0, 4.0]
    img = image_payload(np.zeros((2, 3, 3), dtype=np.uint8))
    assert img["step"] == 9 and img["encoding"] == "rgb8" and len(img["data"]) == 18
    assert topic_name("/rig/", "state.joint_pos") == "/rig/state/joint_pos"


def test_cli_rig(tmp_path):
    assert main(["rig", "--config", RIG, "--episodes", "4", "--seed", "3", "--record", "--out", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "rig.json").read_text())
    assert res["episodes"] == 4 and "state.ee_pos" in res["per_episode"][0]["topics"]
    ts = TraceSet.open(tmp_path / "trace")
    assert "rig.age_ms.state.ee_pos" in ts.frames()[0]
    assert RunSeed.from_trace(tmp_path / "trace") == RunSeed(3)
    assert "Streams" in (tmp_path / "rig.md").read_text()
