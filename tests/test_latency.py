import json

import numpy as np
import pytest

from groot_eval.cli import main
from groot_eval.closed_loop import evaluate
from groot_eval.envs.reach2d import Reach2DEnv
from groot_eval.gap import chunking
from groot_eval.latency import LatencyModel, RealTimePolicy, latencies_from_trace
from groot_eval.policies import make_policy
from groot_eval.policies.base import ChunkedPolicy
from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario


class Counter:
    """Chunk predictor whose actions encode (call, step): row j of call c is 100*c + j."""

    name = "counter"

    def __init__(self, horizon=8):
        self.horizon, self.calls = horizon, 0

    def reset(self, task):
        self.calls = 0

    def predict_chunk(self, obs):
        self.calls += 1
        return (100 * self.calls + np.arange(self.horizon, dtype=float))[:, None]


def _drive(policy, steps):
    policy.reset("")
    return [float(policy.act({})[0]) for _ in range(steps)]


@pytest.mark.parametrize("during", ["hold", "continue"])
def test_zero_latency_is_the_plain_policy(during):
    plain = _drive(ChunkedPolicy(Counter(), execute_steps=3), 20)
    timed = _drive(RealTimePolicy(ChunkedPolicy(Counter(), execute_steps=3), LatencyModel.parse("fixed:0"), 0.05, during), 20)
    assert timed == plain


def test_hold_waits_then_starts_the_chunk_from_the_top():
    p = RealTimePolicy(ChunkedPolicy(Counter(), execute_steps=3), LatencyModel.parse("fixed:100"), 0.05, "hold")
    acts = _drive(p, 12)
    # 2 steps of nothing, chunk 1 steps 0-2, hold its last command while chunk 2 is computed, ...
    assert acts == [0, 0, 100, 101, 102, 102, 102, 200, 201, 202, 202, 202]
    assert p.delays == [2, 2, 2] and p.inference_calls == 3  # the third is still in flight at step 12


def test_continue_keeps_executing_and_skips_stale_steps():
    p = RealTimePolicy(ChunkedPolicy(Counter(), execute_steps=3), LatencyModel.parse("fixed:100"), 0.05, "continue")
    acts = _drive(p, 12)
    # chunk 1 arrives 2 steps late, so it starts at its step 2; after 3 steps the next is requested
    # while chunk 1 keeps running; chunk 2 also starts 2 steps in.
    assert acts == [0, 0, 102, 103, 104, 105, 106, 202, 203, 204, 205, 206]


def test_plain_policy_with_latency_acts_on_old_observations():
    env = Reach2DEnv()
    p = RealTimePolicy(make_policy("pd"), LatencyModel.parse("fixed:120"), env.dt)  # ceil(120/50) = 3 steps
    obs = env.reset(Scenario(seed=0))
    p.reset("")
    acts = []
    for _ in range(9):
        acts.append(p.act(obs))
        obs = env.step(acts[-1])[0]
    assert all(np.allclose(a, 0) for a in acts[:3]) and not np.allclose(acts[3], 0)
    assert p.inference_calls == 3  # busy in between: a robot cannot start the next call early


def test_parse_and_sample():
    assert LatencyModel.parse("fixed:80").sample(1.0) == 80
    assert LatencyModel.parse("measured").sample(3.5) == 3.5
    draws = [LatencyModel.parse("normal:100,10", seed=1).sample(0) for _ in range(3)]
    assert all(d >= 0 for d in draws)
    for bad in ("fast", "fixed:-1", "fixed:nan", "measured:3"):
        with pytest.raises(ValueError):
            LatencyModel.parse(bad)
    with pytest.raises(ValueError, match="control period"):
        RealTimePolicy(make_policy("pd"), LatencyModel.parse("fixed:1"), 0)


def _trace(tmp_path, name, policy, episodes=4, domain="real"):
    env = Reach2DEnv(max_steps=60)
    rec = TraceRecorder(domain=domain, env="reach2d", policy=policy.name, control_dt=env.dt)
    run = evaluate(env, policy, [Scenario(seed=i) for i in range(episodes)], recorder=rec)
    return run, TraceSet.open(rec.write(tmp_path / name))


def test_modelled_latency_is_recorded_and_resampled(tmp_path):
    pol = RealTimePolicy(make_policy("pd-chunk", execute_steps=4), LatencyModel.parse("fixed:80"), 0.05, "continue")
    run, ts = _trace(tmp_path, "real", pol)
    assert run.summary()["latency_ms_p95"] == pytest.approx(80)
    lat = latencies_from_trace(str(ts.root))
    assert np.allclose(lat, 80) and len(lat) == sum(len(set(f["chunk_index"][f["chunk_index"] >= 0])) for f in ts.frames())
    ch = chunking(ts)  # chunks start mid-way (stale steps skipped) and are still counted as boundaries
    assert ch and ch["boundaries"] > 0


def test_cli_sweep_with_latency_from_a_trace(tmp_path):
    _, real = _trace(tmp_path, "real", RealTimePolicy(make_policy("pd-chunk", execute_steps=4),
                                                      LatencyModel.parse("normal:110,20"), 0.05, "continue"))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seed: 1\nepisodes: 5\nparams: {}\n", encoding="utf-8")
    out = tmp_path / "run"
    assert main(["sweep", "--policy", "pd-chunk", "--policy-arg", "execute_steps=4", "--config", str(cfg),
                 "--latency-mode", f"trace:{real.root}", "--during-inference", "continue", "--out", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "+latency(trace(" in s["policy"] and s["latency_ms_p95"] > 50
