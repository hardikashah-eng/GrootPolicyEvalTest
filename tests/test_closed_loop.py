import numpy as np
import pytest

from groot_eval.closed_loop import evaluate, run_episode
from groot_eval.envs.reach2d import Reach2DEnv
from groot_eval.policies import make_policy
from groot_eval.policies.base import ChunkedPolicy
from groot_eval.policies.scripted import PDChunkPredictor, PDPolicy
from groot_eval.types import Scenario


def test_nominal_scenario_succeeds_and_replays_identically():
    env, policy = Reach2DEnv(), PDPolicy()
    s = Scenario(seed=11, params={"sensor_noise": 0.02})
    a, b = run_episode(env, policy, s), run_episode(env, policy, s)
    assert a.success
    assert (a.steps, a.episode_return) == (b.steps, b.episode_return)
    assert a.metrics["final_distance"] < 0.05


def test_naive_policy_collides_and_avoiding_policy_does_not():
    env = Reach2DEnv()
    s = Scenario(seed=5, params={"obstacle": True})
    assert run_episode(env, PDPolicy(), s).failure_reason == "collision"
    assert run_episode(env, make_policy("pd-avoid"), s).success


def test_unknown_parameter_is_an_error_not_silently_ignored():
    with pytest.raises(ValueError, match="does not know"):
        Reach2DEnv().reset(Scenario(seed=0, params={"fricton": 0.5}))


class Boom:
    name = "boom"

    def reset(self, task):
        pass

    def act(self, obs):
        raise RuntimeError("CUDA out of memory")


class NaNPolicy(Boom):
    def act(self, obs):
        return np.array([np.nan, 0.0])


def test_policy_errors_and_nan_actions_are_failures_not_crashes():
    env = Reach2DEnv()
    run = evaluate(env, Boom(), [Scenario(seed=1)])
    assert run.results[0].failure_reason.startswith("policy_error: RuntimeError")
    assert run_episode(env, NaNPolicy(), Scenario(seed=1)).failure_reason == "non_finite_action"


def test_summary_reports_interval_and_reasons():
    env = Reach2DEnv()
    scenarios = [Scenario(seed=i, params={"obstacle": i % 2 == 0}) for i in range(20)]
    s = evaluate(env, PDPolicy(), scenarios, env_name="reach2d").summary()
    assert s["episodes"] == 20
    assert s["ci_low"] <= s["success_rate"] <= s["ci_high"]
    assert s["failure_reasons"].get("collision", 0) >= 5


def test_chunked_policy_replans_every_execute_steps():
    policy = ChunkedPolicy(PDChunkPredictor(horizon=16), execute_steps=4)
    obs = Reach2DEnv().reset(Scenario(seed=0))
    policy.reset("")
    for _ in range(12):
        policy.act(obs)
    assert policy.inference_calls == 3
    with pytest.raises(ValueError):
        ChunkedPolicy(PDChunkPredictor(), execute_steps=0)
