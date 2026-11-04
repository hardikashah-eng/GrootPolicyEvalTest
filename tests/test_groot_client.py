"""The GR00T adapter against a fake client (no server or GPU needed)."""

import numpy as np
import pytest

from groot_eval.policies import ChunkedPolicy
from groot_eval.policies.groot_client import DEFAULT_LANGUAGE_KEY, GrootServerPredictor


class FakeClient:
    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def get_action(self, obs):
        self.requests.append(obs)
        return self.reply


def test_request_shapes_and_language():
    client = FakeClient({"action.left_arm": np.ones((16, 7)), "action.gripper": np.zeros(16)})
    p = GrootServerPredictor(client=client, obs_map={"state.position": "state.left_arm"},
                             action_keys="action.left_arm, action.gripper")
    p.reset("pick up the cup")
    chunk = p.predict_chunk({"state.position": np.arange(7.0), "video.ego_view": np.zeros((4, 4, 3)),
                             "annotation.task": "ignored when reset gave a task"})
    req = client.requests[0]
    assert req["state.left_arm"].shape == (1, 7)
    assert req["video.ego_view"].shape == (1, 4, 4, 3) and req["video.ego_view"].dtype == np.uint8
    assert req[DEFAULT_LANGUAGE_KEY] == ["pick up the cup"]
    assert "annotation.task" not in req
    assert chunk.shape == (16, 8)
    assert np.all(chunk[:, :7] == 1) and np.all(chunk[:, 7] == 0)


def test_horizon_mismatch_and_missing_actions_raise():
    with pytest.raises(ValueError, match="horizon"):
        GrootServerPredictor(client=FakeClient({"action.a": np.ones((16, 2)), "action.b": np.ones((8, 1))})).predict_chunk({})
    with pytest.raises(ValueError, match="no action"):
        GrootServerPredictor(client=FakeClient({"foo": 1})).predict_chunk({})


def test_works_inside_chunked_policy():
    client = FakeClient({"action.arm": np.tile(np.arange(16.0)[:, None], (1, 2))})
    policy = ChunkedPolicy(GrootServerPredictor(client=client), execute_steps=4)
    policy.reset("t")
    actions = [policy.act({"state.x": np.zeros(2)})[0] for _ in range(6)]
    assert actions == [0, 1, 2, 3, 0, 1]
    assert len(client.requests) == 2


def test_missing_gr00t_package_gives_actionable_error():
    with pytest.raises(ImportError, match="Isaac-GR00T"):
        GrootServerPredictor(client_cls="definitely_not_installed_mod:Client")
