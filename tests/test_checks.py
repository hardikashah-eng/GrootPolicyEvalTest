import numpy as np
import pytest

from groot_eval.checks import SignalSource, check_policy, check_sources, source_from_env, source_from_traces
from groot_eval.cli import main
from groot_eval.envs.reach2d import Reach2DEnv
from groot_eval.open_loop import write_lerobot_dataset
from groot_eval.policies.groot_client import GrootServerPredictor
from groot_eval.policies.scripted import PDPolicy
from groot_eval.trace import TraceSet, import_real


def _errors(findings):
    return [f for f in findings if f.level == "error"]


def _rewrite(ts: TraceSet, root, scale_state=None, fps=None, modality=None):
    """Copy a trace as a plain LeRobot recording, optionally corrupted, and import it as 'real'."""
    eps = []
    for meta, f in zip(ts.episode_meta(), ts.frames()):
        states = np.stack(f["observation.state"].to_numpy())
        if scale_state:
            sl, factor = scale_state
            states[:, sl] *= factor
        eps.append({"task": meta["tasks"][0], "states": states, "actions": np.stack(f["action"].to_numpy())})
    src = write_lerobot_dataset(root / "raw", eps, fps=fps or 20, modality=modality or ts.dataset.modality)
    out, _ = import_real(src, root / "real")
    return TraceSet.open(out)


def test_same_source_has_no_errors(make_trace):
    ts = make_trace("sim")
    assert not _errors(check_sources(source_from_traces(ts, "a"), source_from_traces(ts, "b")))


def test_degrees_vs_radians_is_an_error(make_trace, tmp_path):
    sim = make_trace("sim")
    vel = sim.state_slices()["velocity"]
    real = _rewrite(sim, tmp_path, scale_state=(vel, 180 / np.pi))
    errs = _errors(check_sources(source_from_traces(sim, "sim"), source_from_traces(real, "real")))
    assert errs and all(e.check == "state.units" and "velocity" in e.detail for e in errs)


def test_rate_and_action_order_mismatch(make_trace, tmp_path):
    sim = make_trace("sim")
    mod = dict(sim.dataset.modality)
    mod["action"] = {"y": {"start": 0, "end": 1}, "x": {"start": 1, "end": 2}}  # real robot packs y first
    real = _rewrite(sim, tmp_path, fps=30, modality=mod)
    sim_src = source_from_traces(sim, "sim")
    sim_src.action_groups = {"x": slice(0, 1), "y": slice(1, 2)}
    checks = {e.check for e in _errors(check_sources(sim_src, source_from_traces(real, "real")))}
    assert {"fps", "action.order"} <= checks


def test_missing_state_key_and_video_range():
    n = 50
    rng = np.random.default_rng(0)
    a = SignalSource("sim", 20, {"q": rng.normal(size=(n, 2))}, rng.normal(size=(n, 1)), {"action": slice(0, 1)},
                     {"cam": {"dtype": "float32", "shape": (64, 64, 3), "max_value": 1.0}})
    b = SignalSource("real", 20, {"q": rng.normal(size=(n, 2)), "gripper": rng.normal(size=(n, 1))}, rng.normal(size=(n, 1)),
                     {"action": slice(0, 1)}, {"cam": {"dtype": "uint8", "shape": (480, 640, 3), "max_value": 255.0}})
    found = {(f.level, f.check) for f in check_sources(a, b)}
    assert ("error", "state.keys") in found and ("error", "video.range") in found and ("warn", "video.shape") in found


def test_gripper_scale_mismatch_is_named():
    n = 200
    rng = np.random.default_rng(0)
    a = SignalSource("sim", 20, {}, rng.uniform(0, 1, size=(n, 1)), {"gripper": slice(0, 1)})
    b = SignalSource("real", 20, {}, rng.uniform(0, 100, size=(n, 1)), {"gripper": slice(0, 1)})
    warns = [f for f in check_sources(a, b) if f.check == "action.scale"]
    assert warns and "gripper scaling" in warns[0].detail


def test_env_rollout_source_matches_its_own_trace(make_trace):
    ts = make_trace("sim")
    env_src = source_from_env(Reach2DEnv(max_steps=80), PDPolicy(), name="env")
    assert env_src.fps == pytest.approx(20)
    assert not _errors(check_sources(env_src, source_from_traces(ts)))


class FakeClient:
    def get_action(self, obs):
        self.last = obs
        return {"action.arm": np.zeros((4, 2)), "action.gripper": np.zeros(4)}


def test_groot_action_key_order_checked():
    ref = SignalSource("real", 20, {}, np.zeros((1, 3)), {"arm": slice(0, 2), "gripper": slice(2, 3)})
    wrong = GrootServerPredictor(client=FakeClient(), action_keys="action.gripper,action.arm")
    assert _errors(check_policy(wrong, ref))
    assert check_policy(GrootServerPredictor(client=FakeClient(), action_keys="action.arm,action.gripper"), ref)[0].level == "ok"
    assert check_policy(GrootServerPredictor(client=FakeClient()), ref)[0].level == "warn"


def test_float_images_are_scaled_not_zeroed():
    client = FakeClient()
    p = GrootServerPredictor(client=client, action_keys="action.arm")
    with pytest.warns(UserWarning, match="float image"):
        p.predict_chunk({"video.wrist_cam_test": np.full((2, 2, 3), 0.5, np.float32)})
    assert client.last["video.wrist_cam_test"].dtype == np.uint8 and client.last["video.wrist_cam_test"].max() == 128


def test_cli_check(make_trace, tmp_path):
    ts = make_trace("ref")
    assert main(["check", "--reference", str(ts.root), "--env", "reach2d", "--policy", "pd", "--out", str(tmp_path / "c.md")]) == 0
    assert (tmp_path / "c.md").read_text(encoding="utf-8").startswith("# Contract check")
    bad = _rewrite(ts, tmp_path, scale_state=(ts.state_slices()["velocity"], 180 / np.pi))
    assert main(["check", "--reference", str(bad.root), "--traces", str(ts.root)]) == 1
