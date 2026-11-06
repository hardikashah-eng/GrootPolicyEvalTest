import pytest

from groot_eval.closed_loop import evaluate
from groot_eval.envs.reach2d import Reach2DEnv
from groot_eval.policies import make_policy
from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario


@pytest.fixture
def make_trace(tmp_path):
    """Record Reach2D episodes as a trace and open it."""

    def _make(name, params=None, domain="sim", policy="pd", episodes=6, **policy_args):
        env = Reach2DEnv(max_steps=80)
        pol = make_policy(policy, **policy_args)
        rec = TraceRecorder(domain=domain, env="reach2d", policy=pol.name, control_dt=env.dt)
        scenarios = [Scenario(seed=i, params=dict(params or {}), task="reach") for i in range(episodes)]
        evaluate(env, pol, scenarios, env_name="reach2d", recorder=rec)
        return TraceSet.open(rec.write(tmp_path / name))

    return _make
