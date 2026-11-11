"""HDF5 -> LeRobot conversion fidelity on synthetic Isaac Lab-style demos, with one injected loss per test."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")
cv2 = pytest.importorskip("cv2")

from groot_eval import conversion_fidelity as cf  # noqa: E402
from groot_eval.fidelity import exit_code  # noqa: E402

LOSSLESS = shutil.which("ffmpeg") is not None
MAPPING = {
    "episodes": "data/demo_*", "fps": 20, "task_attr": "task",
    "state": [{"key": "obs/joint_pos", "modality": "joint_pos"}, {"key": "obs/joint_vel", "modality": "joint_vel"}],
    "action": [{"key": "actions", "modality": "arm"}],
    "video": {"front": "obs/front_rgb"},
    "ignore": ["states/*"],
}


def _frames(T, H=64, W=80, seed=0):
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.stack([xx * 255 / W, yy * 255 / H, (xx + yy) * 127 / (W + H)], -1)
    out = []
    for t in range(T):
        f = base.copy()
        f[(xx - 10 - t * 0.8) ** 2 + (yy - H / 2 - 8 * np.sin(t / 6 + seed)) ** 2 < 64] = [250, 40, 40]
        f[(xx // 8 + yy // 8) % 2 == 0] *= 0.85
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(out)


@pytest.fixture()
def demo(tmp_path):
    path = tmp_path / "demo.hdf5"
    T = 60
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        for i in range(2):
            g = d.create_group(f"demo_{i}")
            t = np.arange(T) / 20
            q = np.stack([np.sin(t * (k + 1) * 0.7 + i) + 0.01 * np.random.default_rng(10 * i + k).normal(size=T)
                          for k in range(7)], 1)
            g.create_dataset("obs/joint_pos", data=q)
            g.create_dataset("obs/joint_vel", data=np.gradient(q, axis=0) * 20)
            g.create_dataset("obs/ft_wrench", data=np.random.default_rng(i).normal(size=(T, 6)))
            g.create_dataset("actions", data=q + 0.05 * np.gradient(q, axis=0))
            g.create_dataset("obs/front_rgb", data=_frames(T, seed=i))
            g.create_dataset("states/articulation", data=np.zeros((T, 3)))
            g.attrs["success"] = bool(i % 2 == 0)
            g.attrs["task"] = "stack the red cube"
    ref = cf.convert_hdf5_to_lerobot(path, tmp_path / "ref", MAPPING, fps=20)
    cfg = cf.load_config(None)
    cfg["mapping"] = MAPPING
    if not LOSSLESS:  # OpenCV's mp4v fallback is lossy: relax the visual bar so the clean case can pass
        cfg["gates"]["C3_visual"].update(min_psnr_mean=25, min_psnr_p5=20, min_ssim_mean=0.8)
    return path, ref, cfg, tmp_path


def _variant(ref, dst):
    shutil.copytree(ref, dst)
    return dst


def _edit_parquet(root, fn):
    for p in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(p)
        fn(df)
        df.to_parquet(p)


def _status(results):
    return {r.gate: r.status for r in results}


def _errors(results, gate):
    return [f for r in results if r.gate == gate for f in r.findings if f.level == "error"]


def test_clean_conversion_passes_and_reports_lost_sensing(demo):
    path, ref, cfg, _ = demo
    res = cf.check_conversion(path, ref, cfg)
    assert exit_code(res) == 0, cf.conversion_markdown(res, str(path), str(ref))
    inv = next(r for r in res if r.gate == "C0_inventory")
    assert "obs/ft_wrench" in inv.metrics["lost"] and "states/articulation" in inv.metrics["ignored"]
    bits = next(r for r in res if r.gate == "C2_actuator").metrics["observation.state.min_effective_bits"]
    assert bits > 18  # float64 -> float32 keeps ~24 bits
    if LOSSLESS:
        assert next(r for r in res if r.gate == "C3_visual").metrics["front"]["psnr_mean"] > 60
    assert "Information kept" in cf.conversion_markdown(res, str(path), str(ref))


def test_lost_sensing_can_be_an_error(demo):
    path, ref, cfg, _ = demo
    cfg["gates"]["C0_inventory"]["lost_is_error"] = True
    assert _status(cf.check_conversion(path, ref, cfg))["C0_inventory"] == "fail"


def test_degrees_and_joint_swap_detected(demo):
    path, ref, cfg, tmp = demo
    deg = _variant(ref, tmp / "deg")

    def to_deg(df):
        df["observation.state"] = [np.r_[np.asarray(v[:7]) * 57.29577951308232, v[7:]].astype(np.float32)
                                   for v in df["observation.state"]]

    _edit_parquet(deg, to_deg)
    errs = _errors(cf.check_conversion(path, deg, cfg), "C2_actuator")
    assert any("radians -> degrees" in f.detail for f in errs)
    sw = _variant(ref, tmp / "swap")

    def swap(df):
        df["action"] = [np.asarray(v)[[1, 0, 2, 3, 4, 5, 6]].astype(np.float32) for v in df["action"]]

    _edit_parquet(sw, swap)
    errs = _errors(cf.check_conversion(path, sw, cfg), "C2_actuator")
    assert any(f.check == "action.order" for f in errs)


def test_quantization_and_shift_detected(demo):
    path, ref, cfg, tmp = demo
    q = _variant(ref, tmp / "q")
    _edit_parquet(q, lambda df: df.__setitem__("action", [np.round(np.asarray(v) * 50) / 50 for v in df["action"]]))
    errs = _errors(cf.check_conversion(path, q, cfg), "C2_actuator")
    assert errs and any("effective bits" in f.detail for f in errs)
    sh = _variant(ref, tmp / "shift")

    def shift(df):
        vals = list(df["observation.state"])
        df["observation.state"] = vals[1:] + vals[-1:]

    _edit_parquet(sh, shift)
    assert any(f.check == "shift" for f in _errors(cf.check_conversion(path, sh, cfg), "C1_timing"))


def test_resampling_detected(demo):
    path, ref, cfg, tmp = demo
    rs = tmp / "rs"
    shutil.copytree(ref, rs)
    for p in sorted((rs / "data").rglob("*.parquet")):
        df = pd.read_parquet(p).iloc[::2].reset_index(drop=True)
        df["timestamp"] = np.arange(len(df), dtype=np.float32) / 10
        df.to_parquet(p)
    info = json.loads((rs / "meta" / "info.json").read_text())
    info["fps"] = 10
    (rs / "meta" / "info.json").write_text(json.dumps(info))
    res = cf.check_conversion(path, rs, cfg)
    assert _status(res)["C0_inventory"] == "fail"
    assert any(f.check in ("fps", "resample") for f in _errors(res, "C1_timing"))


def _reencode(root, fn):
    for p in sorted((root / "videos").rglob("*.mp4")):
        frames = np.stack(cf._frames(p))
        cf.write_video(fn(frames), p, 20)


def test_bgr_swap_and_downscale_detected(demo):
    path, ref, cfg, tmp = demo
    bgr = _variant(ref, tmp / "bgr")
    _reencode(bgr, lambda fr: fr[..., ::-1].copy())
    assert any(f.check == "front.channels" for f in _errors(cf.check_conversion(path, bgr, cfg), "C3_visual"))
    small = _variant(ref, tmp / "small")
    _reencode(small, lambda fr: np.stack([cv2.resize(f, (40, 32), interpolation=cv2.INTER_AREA) for f in fr]))
    errs = _errors(cf.check_conversion(path, small, cfg), "C3_visual")
    assert any(f.check == "front.resolution" for f in errs)


def test_dropped_first_frame_detected(demo):
    path, ref, cfg, tmp = demo
    drop = _variant(ref, tmp / "drop")
    _reencode(drop, lambda fr: np.concatenate([fr[1:], fr[-1:]]))
    errs = _errors(cf.check_conversion(path, drop, cfg), "C3_visual")
    assert any(f.check == "front.alignment" for f in errs)


def test_success_label_and_task_checks(demo):
    path, ref, cfg, tmp = demo
    m = _variant(ref, tmp / "meta")
    lines = [json.loads(x) for x in (m / "meta" / "episodes.jsonl").read_text().splitlines() if x.strip()]
    lines[0]["success"] = not lines[0]["success"]
    lines[1]["tasks"] = [""]
    (m / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    errs = _errors(cf.check_conversion(path, m, cfg), "C4_metadata")
    assert {f.check for f in errs} >= {"success", "task"}


def test_inferred_mapping_from_modality(demo):
    path, ref, cfg, _ = demo
    cfg["mapping"] = {"fps": 20, "ignore": ["states/*"]}  # state/action/video inferred from modality.json names
    res = cf.check_conversion(path, ref, cfg)
    assert _status(res)["C2_actuator"] == "pass"
    assert any("inferred" in f.detail for r in res for f in r.findings)


def test_cli_convert_check(demo):
    from groot_eval.twin_cli import main

    path, ref, cfg, tmp = demo
    import yaml

    (tmp / "cfg.yaml").write_text(yaml.safe_dump({"mapping": MAPPING, "gates": cfg["gates"]}))
    code = main(["convert-check", "--hdf5", str(path), "--lerobot", str(ref), "--config", str(tmp / "cfg.yaml"),
                 "--out", str(tmp / "out")])
    assert code == 0 and (tmp / "out" / "conversion.md").exists()
