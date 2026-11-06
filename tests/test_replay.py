import json

import numpy as np
import pytest

from groot_eval.cli import main
from groot_eval.envs.reach2d import Reach2DEnv
from groot_eval.randomization import SweepConfig
from groot_eval.replay import candidate_params, replay_episode, replay_trace, system_id
from groot_eval.types import Scenario

HIDDEN = {"friction": 1.0, "actuator_gain": 0.9, "action_delay": 2}


def _grid(**params):
    return SweepConfig.from_dict({"mode": "grid", "grid_points": 3, "params": params})


def test_replay_with_the_recording_physics_is_exact(make_trace):
    ts = make_trace("sim", params=HIDDEN)
    r = replay_trace(Reach2DEnv(), ts, HIDDEN)
    assert r.synced and r.ended_early == 0
    assert r.score == pytest.approx(0, abs=1e-5)
    assert "target" not in r.signals  # constant within each episode: copied in by set_state, not scored


def test_wrong_physics_shows_up_as_tracking_error(make_trace):
    ts = make_trace("real", params=HIDDEN, domain="real")
    r = replay_trace(Reach2DEnv(), ts, {})
    assert r.score > 0.3 and set(r.signals) == {"position", "velocity"}
    assert list(r.horizons) == sorted(r.horizons) and r.horizons[1] < r.horizons[max(r.horizons)]


def test_set_state_refills_the_delay_queue():
    env = Reach2DEnv()
    env.reset(Scenario(seed=0, params={"action_delay": 2}))
    obs = env.set_state({"state.position": np.zeros(2), "state.velocity": np.zeros(2)},
                        recent_actions=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
    assert np.allclose(obs["state.position"], 0)
    env.step(np.zeros(2))  # the command sent two steps ago is applied now
    assert env.vel[1] > 0 and env.vel[0] == pytest.approx(0)


def test_system_id_recovers_hidden_physics(make_trace):
    ts = make_trace("real", params={**HIDDEN, "sensor_noise": 0.01}, domain="real", policy="pd-chunk", execute_steps=4)
    cfg = _grid(friction={"uniform": [0.6, 1.4]}, actuator_gain={"uniform": [0.7, 1.1]}, action_delay={"choice": [0, 2, 4]})
    assert len(candidate_params(cfg)) == 27
    fit = system_id(Reach2DEnv(), ts, cfg)
    best = fit["ranked"][0]
    assert best.params == pytest.approx(HIDDEN)
    assert best.score < 0.2 * fit["baseline"].score
    assert {s["param"] for s in fit["sensitivity"]} == {"friction", "actuator_gain", "action_delay"}
    assert any("cuts tracking error" in n for n in fit["notes"])


def test_edge_of_range_and_noise_notes(make_trace):
    ts = make_trace("real", params=HIDDEN, domain="real")
    cfg = _grid(friction={"uniform": [0.2, 0.6]}, sensor_noise={"choice": [0.0, 0.01]})
    notes = " ".join(system_id(Reach2DEnv(), ts, cfg, {"actuator_gain": 0.9, "action_delay": 2})["notes"])
    assert "edge of the searched range" in notes and "noise parameters" in notes


class _NoSetState(Reach2DEnv):
    def set_state(self, state, recent_actions=None):
        raise NotImplementedError


def test_env_without_set_state_falls_back_to_seed(make_trace):
    ts = make_trace("sim")  # recorded from Reach2D itself, so the seed reproduces the start state
    r = replay_trace(_NoSetState(), ts, {})
    assert not r.synced and r.score == pytest.approx(0, abs=1e-5)


def test_state_key_mismatch_is_an_error(make_trace):
    ts = make_trace("sim")
    states, actions = ts.per_episode("observation.state")[0], ts.per_episode("action")[0]
    with pytest.raises(ValueError, match="state.joints"):
        replay_episode(Reach2DEnv(), states, actions, {"joints": slice(0, states.shape[1])})


def test_cli_replay_with_config(make_trace, tmp_path):
    ts = make_trace("real", params=HIDDEN, domain="real")
    cfg = tmp_path / "sysid.yaml"
    cfg.write_text("mode: grid\nparams:\n  action_delay: {choice: [0, 2]}\n  friction: {choice: [0.6, 1.0]}\n",
                   encoding="utf-8")
    out = tmp_path / "rep"
    assert main(["replay", str(ts.root), "--config", str(cfg), "--param", "actuator_gain=0.9",
                 "--reset-every", "10", "--out", str(out)]) == 0
    text = (out / "replay.md").read_text(encoding="utf-8")
    assert "## 2. Parameter fit (4 settings" in text and "### Which knob matters" in text
    data = json.loads((out / "replay.json").read_text(encoding="utf-8"))
    assert data["ranked"][0]["params"] == {"actuator_gain": 0.9, "action_delay": 2, "friction": 1.0}
    assert main(["replay", str(ts.root), "--out", str(tmp_path / "plain")]) == 0
