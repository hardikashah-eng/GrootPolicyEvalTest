import numpy as np
import pytest

from groot_eval.open_loop import LeRobotDataset, evaluate_open_loop, write_lerobot_dataset
from groot_eval.policies.scripted import ReplayPolicy

MODALITY = {"state": {"a": {"start": 0, "end": 2}, "b": {"start": 2, "end": 3}},
            "action": {"arm": {"start": 0, "end": 2}, "gripper": {"start": 2, "end": 3}},
            "annotation": {"task": {}}}


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    eps = [{"task": "pick", "states": rng.normal(size=(40, 3)), "actions": rng.normal(size=(40, 3))}]
    write_lerobot_dataset(tmp_path, eps, modality=MODALITY)
    return LeRobotDataset.open(tmp_path), eps[0]


def test_perfect_policy_scores_zero(dataset):
    ds, ep = dataset
    res = evaluate_open_loop(ds, ReplayPolicy(ep["actions"]), horizon=8)
    s = res.summary()
    assert s["mse_mean"] == pytest.approx(0.0, abs=1e-10)
    assert set(s["per_group_mse"]) == {"arm", "gripper"}


class ConstantChunk:
    name = "zeros"

    def reset(self, task):
        self.task = task

    def predict_chunk(self, obs):
        assert set(obs) == {"state.a", "state.b", "annotation.task"}
        assert obs["state.a"].shape == (2,)
        return np.zeros((8, 3))


def test_chunk_predictor_gets_modality_split_observations(dataset):
    ds, ep = dataset
    policy = ConstantChunk()
    res = evaluate_open_loop(ds, policy, horizon=8)
    assert policy.task == "pick"
    assert res.summary()["mse_mean"] == pytest.approx(float(np.mean(ep["actions"] ** 2)), rel=1e-5)
    assert len(res.per_horizon_mse) == 8


def test_shape_mismatch_is_reported(dataset):
    ds, _ = dataset

    class Wrong(ConstantChunk):
        def predict_chunk(self, obs):
            return np.zeros((8, 2))

    with pytest.raises(ValueError, match="predicted"):
        evaluate_open_loop(ds, Wrong(), horizon=8)
