import json

import numpy as np
import pytest

from groot_eval.cli import main
from groot_eval.open_loop import LeRobotDataset, evaluate_open_loop, write_lerobot_dataset
from groot_eval.policies.scripted import ReplayPolicy
from groot_eval.trace import TraceRecorder, TraceSet, image_stats, import_real


def test_sim_rollout_is_a_lerobot_trace(make_trace):
    ts = make_trace("sim")
    assert ts.domain == "sim" and ts.fps == pytest.approx(20)
    f = ts.frames()[0]
    for col in ("observation.state", "action", "timestamp", "wall_time", "inference_latency_ms", "chunk_index", "chunk_step", "reward"):
        assert col in f.columns
    assert np.allclose(f["timestamp"].to_numpy(), np.arange(len(f)) * 0.05)
    assert (f["chunk_step"] == -1).all()
    assert np.all(np.diff(f["wall_time"].to_numpy()) >= 0)
    meta = ts.episode_meta()[0]
    assert isinstance(meta["success"], bool) and "scenario" in meta and meta["tasks"] == ["reach"]
    assert set(ts.dataset.modality["state"]) == {"position", "target", "velocity"}

    # The open-loop scorer reads a trace like any LeRobot dataset.
    ds = LeRobotDataset.open(ts.root)
    gt = np.stack(f["action"].to_numpy())
    res = evaluate_open_loop(ds, ReplayPolicy(gt), horizon=8, episodes=[0])
    assert res.summary()["mse_mean"] == pytest.approx(0.0, abs=1e-10)


def test_chunk_positions_are_recorded(make_trace):
    ts = make_trace("chunked", policy="pd-chunk", execute_steps=4, episodes=2)
    f = ts.frames()[0]
    assert list(f["chunk_step"][:6]) == [0, 1, 2, 3, 0, 1]
    assert list(f["chunk_index"][:6]) == [0, 0, 0, 0, 1, 1]


def test_image_stats_are_dtype_independent():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32, 3)).astype(np.uint8)
    a, b = image_stats(img), image_stats(img.astype(np.float32) / 255.0)
    assert a["brightness"] == pytest.approx(b["brightness"], abs=1e-6)
    assert a["sharpness"] == pytest.approx(b["sharpness"], rel=1e-5)
    assert a["max_value"] > 1 and b["max_value"] <= 1
    flat = np.full((32, 32, 3), 128, np.uint8)
    assert image_stats(flat)["sharpness"] == 0 < a["sharpness"]


def test_recorder_writes_image_columns(tmp_path):
    rec = TraceRecorder(domain="real", control_dt=0.1)
    rec.start_episode("look")
    for t in range(5):
        rec.record_step({"state.q": np.array([t, 0.0]), "video.cam": np.full((8, 8, 3), 40 * t, np.uint8)}, np.zeros(2), latency_ms=30.0)
    rec.end_episode(success=True)
    ts = TraceSet.open(rec.write(tmp_path / "t"))
    assert ts.domain == "real" and ts.dataset.modality["video"] == {"cam": {}}
    assert np.allclose(ts.scalar("image.cam.brightness"), np.arange(5) * 40 / 255)
    assert np.allclose(ts.scalar("inference_latency_ms"), 30.0)


def test_recorder_rejects_changing_shapes():
    rec = TraceRecorder()
    rec.start_episode()
    rec.record_step({"state.q": np.zeros(2)}, np.zeros(1))
    with pytest.raises(ValueError, match="changed size"):
        rec.record_step({"state.q": np.zeros(3)}, np.zeros(1))


def _plain_dataset(root, fps=20, step=None):
    rng = np.random.default_rng(1)
    eps = [{"task": "pick", "states": rng.normal(size=(30, 3)), "actions": rng.normal(size=(30, 2))} for _ in range(2)]
    if step is not None:
        for ep in eps:
            ep["extra"] = {"timestamp": np.arange(30) * step}
    modality = {"state": {"q": {"start": 0, "end": 3}}, "action": {"arm": {"start": 0, "end": 2}}}
    return write_lerobot_dataset(root, eps, fps=fps, modality=modality)


def test_import_real_fills_trace_columns(tmp_path):
    src = _plain_dataset(tmp_path / "rec")
    root, findings = import_real(src, tmp_path / "trace")
    ts = TraceSet.open(root)
    assert ts.domain == "real" and len(ts.frames()) == 2
    f = ts.frames()[0]
    assert np.allclose(f["wall_time"], f["timestamp"])
    assert f["inference_latency_ms"].isna().all() and (f["chunk_step"] == -1).all()
    assert not [x for x in findings if x.level == "error"]


def test_import_real_flags_wrong_declared_rate(tmp_path):
    src = _plain_dataset(tmp_path / "rec", fps=20, step=0.1)  # recorded at 10 Hz, declared 20 Hz
    _, findings = import_real(src, tmp_path / "trace")
    assert any(f.level == "error" and f.check == "fps" for f in findings)


def test_cli_sweep_writes_trace(tmp_path):
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text("seed: 1\nepisodes: 4\nparams:\n  friction: {uniform: [0.4, 0.8]}\n", encoding="utf-8")
    assert main(["sweep", "--policy", "pd", "--config", str(cfg), "--out", str(tmp_path / "run"), "--trace", str(tmp_path / "trace")]) == 0
    info = json.loads((tmp_path / "trace" / "meta" / "trace_info.json").read_text(encoding="utf-8"))
    assert info["domain"] == "sim" and info["episodes"] == 4 and info["control_dt"] == 0.05
