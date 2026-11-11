"""Multi-rate scheduler: timing profiles from traces, batched control/inference clocks, chunk execution."""

import json

import numpy as np
import pandas as pd
import pytest

from groot_eval.closed_loop import evaluate
from groot_eval.envs.arm3 import Arm3Env
from groot_eval.latency import with_latency
from groot_eval.policies import make_policy
from groot_eval.robot_cli import main
from groot_eval.scheduler import ChunkBuffer, Distribution, MultiRateScheduler, ScheduledEnv, TimingProfile
from groot_eval.trace import TraceRecorder
from groot_eval.types import Scenario


def _robot_trace(tmp_path, latency_ms=70, drop_every=None, n=4):
    """A trace like a robot's: ~70 ms inference on a 50 Hz loop; optionally a missed tick every ``drop_every`` steps."""
    rec = TraceRecorder(domain="real", env="arm3", policy="arm-ik", control_dt=0.02)
    pol = with_latency(make_policy("arm-ik"), f"normal:{latency_ms},8", 0.02, "continue", seed=1)
    evaluate(Arm3Env(), pol, [Scenario(seed=i) for i in range(n)], recorder=rec)
    root = rec.write(tmp_path / "real")
    if drop_every:
        for p in sorted((root / "data").rglob("*.parquet")):
            df = pd.read_parquet(p)
            k = np.arange(len(df))
            df["timestamp"] = (k + k // drop_every) * 0.02  # every drop_every-th period is two periods long
            df.to_parquet(p)
    return root


def test_distribution_roundtrip_and_sampling():
    rng = np.random.default_rng(0)
    d = Distribution.from_samples(rng.gamma(4.0, 20.0, 5000))
    d2 = Distribution.from_dict(d.to_dict())
    s = d2.sample(np.random.default_rng(1), 20000)
    assert np.percentile(s, 50) == pytest.approx(d.percentile(50), rel=0.03)
    assert np.percentile(s, 95) == pytest.approx(d.percentile(95), rel=0.05)
    assert Distribution.from_dict(12.5).sample(rng, 3).tolist() == [12.5] * 3
    assert Distribution("normal", mean=10, std=2).percentile(97.5) == pytest.approx(13.92, abs=0.01)
    with pytest.raises(ValueError):
        Distribution.from_dict({"kind": "empirical", "quantiles": [3, 2]})


def test_timing_profile_from_trace_measures_latency_period_and_missed_ticks(tmp_path):
    prof = TimingProfile.from_trace(_robot_trace(tmp_path, drop_every=10))
    assert prof.control_hz == pytest.approx(50.0)
    assert prof.latency_ms.percentile(50) == pytest.approx(70, abs=6)  # one value per inference call
    assert prof.missed_tick_rate == pytest.approx(1 / 11, abs=0.01)  # 1 missed tick per 11 periods
    assert abs(prof.control_jitter_ms.percentile(50)) < 1e-6
    p2 = TimingProfile.load(prof.save(tmp_path / "timing.yaml"))
    assert p2.to_dict() == prof.to_dict()
    assert p2.source["latency_samples"] > 0 and p2.source["domain"] == "real"


def test_scheduler_rates_latency_and_missed_ticks():
    prof = TimingProfile(100.0, policy_period_ticks=4, latency_ms=Distribution("fixed", value=25.0),
                         control_jitter_ms=Distribution("normal", mean=0.0, std=0.5), missed_tick_rate=0.05)
    sch = MultiRateScheduler(prof, num_envs=64, physics_dt=0.002, seed=0)  # 500 Hz physics, 100 Hz control
    control = missed = requests = 0
    lat_steps = []
    pending_since = np.full(64, -1)
    for k in range(5000):  # 10 s
        t = sch.step()
        control += t.control.sum()
        missed += t.missed.sum()
        requests += t.request.sum()
        for e in np.flatnonzero(t.ready):
            lat_steps.append(k - pending_since[e])
        pending_since[t.request] = k
        assert not (t.request & t.missed).any()  # a missed tick cannot start an inference
    assert control / 64 == pytest.approx(1000, rel=0.02)  # 100 Hz for 10 s
    assert missed / control == pytest.approx(0.05, abs=0.01)
    # one inference per 4 live ticks, but never two in flight: 25 ms latency = 2.5 ticks, so period stays 4 ticks
    assert requests / 64 == pytest.approx(0.95 * 1000 / 4, rel=0.06)
    assert np.median(lat_steps) == pytest.approx(25e-3 / 0.002, abs=1)  # 12-13 physics steps


def test_chunk_buffer_hold_vs_continue():
    for during, expected in (("hold", [0.0, 0.0, 0.0, 10.0, 11.0, 12.0]), ("continue", [0.0, 0.0, 0.0, 12.0, 13.0, 14.0])):
        buf = ChunkBuffer(during)
        first = np.zeros((1, 8, 1))
        buf.request(np.array([True]), first)
        buf.ready(np.array([True]))
        out = [float(buf.command(np.array([True]))[0, 0])]
        buf.request(np.array([True]), (10.0 + np.arange(8.0)).reshape(1, 8, 1))
        for _ in range(2):  # inference in flight for 2 ticks: continue skips those 2 stale actions
            out.append(float(buf.command(np.array([True]))[0, 0]))
        buf.ready(np.array([True]))
        for _ in range(3):
            out.append(float(buf.command(np.array([True]))[0, 0]))
        assert out == expected, during


def test_chunk_buffer_strict_array_api_and_missed_ticks():
    xs = pytest.importorskip("array_api_strict")
    buf = ChunkBuffer("hold")
    chunk = xs.asarray(np.arange(2 * 4 * 1, dtype=float).reshape(2, 4, 1))
    buf.request(np.array([True, True]), chunk)
    buf.ready(np.array([True, True]))
    a = buf.command(np.array([True, True]))
    b = buf.command(np.array([True, False]))  # env 1 misses the tick: keeps its last command
    assert np.from_dlpack(a)[:, 0].tolist() == [0.0, 4.0]
    assert np.from_dlpack(b)[:, 0].tolist() == [1.0, 4.0]


def test_scheduled_env_holds_commands_on_missed_ticks():
    class Echo:
        max_steps = 100
        dt = 0.02

        def reset(self, scenario):
            return {}

        def step(self, action):
            self.seen = float(action[0])
            return {}, 0.0, False, False, {}

        def close(self):
            pass

    prof = TimingProfile(50.0, missed_tick_rate=0.3)
    env = ScheduledEnv(Echo(), prof, seed=3)
    env.reset(Scenario(seed=0))
    seen = []
    for k in range(400):
        env.step(np.array([float(k)]))
        seen.append(env.env.seen)
    held = sum(1 for k, s in enumerate(seen) if s != k)
    assert held == env.held and 0.2 < held / 400 < 0.4


def test_cli_timing_profile_and_check(tmp_path):
    trace = _robot_trace(tmp_path, latency_ms=90, n=3)
    assert main(["timing-profile", str(trace), "--out", str(tmp_path / "t.yaml")]) == 0
    assert main(["timing-check", "--profile", str(tmp_path / "t.yaml"), "--env", "arm3", "--policy", "arm-ik",
                 "--episodes", "8", "--out", str(tmp_path / "chk")]) == 0
    res = json.loads((tmp_path / "chk" / "timing_check.json").read_text())
    assert res["ideal"]["episodes"] == res["profiled"]["episodes"] == 8
    assert "robot timing" in (tmp_path / "chk" / "timing_check.md").read_text()
