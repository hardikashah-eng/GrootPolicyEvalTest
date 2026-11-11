# Conversion round trip and sim replay (C5, C6)

`groot-twin convert-check` (C0–C4, see [twin-failure-repro.md](twin-failure-repro.md)) checks that the LeRobot output kept the information GR00T trains on. Two questions need the trip back to HDF5:

* **C5 round trip:** can the source HDF5 be rebuilt from the LeRobot dataset, and does the rebuilt file match it?
* **C6 replay:** do the source and the rebuilt demos play back the same way in the simulator? If they don't, is the conversion at fault or the sim?

Code: [`conversion_roundtrip.py`](../src/groot_eval/conversion_roundtrip.py), [`isaaclab_demo_replay.py`](../src/groot_eval/isaaclab_demo_replay.py).

## The sim sidecar

LeRobot keeps video, state, action and the task. It does not keep what Isaac Lab needs to replay a demo:

* `initial_state`: the articulation and rigid-object state at reset;
* the per-step `states`;
* the file's `env_args`.

The mapping usually lists these under `ignore`. Without them the demos can't be replayed, and no LeRobot-only check can show it.

`write_sidecar` copies those datasets byte for byte into `<dataset>/sim_sidecar/`, along with every episode and file attribute and a SHA-256 per episode. LeRobot and GR00T readers ignore the folder. Write the sidecar when the dataset is produced:

```bash
groot-twin convert-sidecar --hdf5 demos.hdf5 --lerobot datasets/my_task --config configs/conversion/mapping_example.yaml
# or, with the reference converter
groot-twin convert --hdf5 demos.hdf5 --config configs/conversion/mapping_example.yaml --fps 30 --sidecar --out datasets/ref
```

## C5: round trip (`groot-twin roundtrip`)

```bash
groot-twin roundtrip --hdf5 demos.hdf5 --lerobot datasets/my_task --config configs/conversion/mapping_example.yaml --out runs/rt
```

This rebuilds `runs/rt/rebuilt.hdf5` from the LeRobot data and the sidecar, using only the mapping, then compares it with the source.

| Check | Fails when |
|---|---|
| `<key>.values` | A mapped state, action or scalar stream differs by more than `max_rel_error` of the signal's spread |
| `<camera>.psnr` | Rebuilt frames fall below `min_psnr_mean` / `min_psnr_p5` against the source frames |
| `<camera>.pix_fmt` | The video's pixel format alone, with no codec, caps PSNR below `min_psnr_mean`. The cap is measured with ffmpeg's scaler on a few source frames. When this fails, no CRF setting can pass, so change the pixel format (yuv444p, rgb) or lower the bar knowingly. On the synthetic test frames, yuv420p capped PSNR at 39 dB |
| `sidecar` | Sim state isn't bit-exact after the round trip |
| `sidecar.hash` | The sidecar changed after it was written |
| `shape` | A dataset's shape or dtype wasn't restored |
| `attrs.root`, `attrs.episode` | File or episode attributes (`env_args`, `success`, the task) differ |
| `ignored` (warn) | A dataset listed under `ignore` is in neither the LeRobot data nor the sidecar, so it can't be rebuilt |
| `lost` (warn, or error with `lost_is_error`) | A dataset is neither mapped, ignored nor in the sidecar |

Without a sidecar, C5 fails and says how to write one.

## C6: replay (`groot-twin replay-check`)

```bash
./isaaclab.sh -p -m groot_eval.twin_cli replay-check --hdf5 demos.hdf5 --rebuilt runs/rt/rebuilt.hdf5 \
    --config configs/conversion/mapping_example.yaml --task Isaac-<Task>-v0 \
    --camera-map obs/robot_pov_cam=robot_pov_cam --joint-key obs/joint_pos --out runs/replay
```

Each demo is replayed several times, and the comparisons are kept separate so a failure names its cause:

| Check | Compares | A failure means |
|---|---|---|
| `determinism` (warn) | Source replayed twice | The sim is nondeterministic. The conversion limit widens to `floor_multiplier` × this floor instead of failing |
| `source` | Source replay against the recorded `--joint-key` | The source doesn't reproduce in this sim build (Isaac Sim / Lab version, USD assets, physics dt, decimation). The conversion isn't at fault |
| `conversion` | Rebuilt replay against source replay | The conversion damaged actions or start states. The first diverging step is reported |
| `success` | Task success, source against rebuilt | Same as `conversion`. A warning is also raised when the source itself doesn't succeed on replay |
| `<camera>.render` | Recorded states re-rendered against the recorded frames | The sim doesn't reproduce its own visuals (renderer, lighting, assets, camera pose, channel order) |

`--replayer module:Class` plugs in any other simulator. The class needs a `replay(h5_path, demo, mode, render)` method that returns a `ReplayResult`.

## Status and limits

* **Tested here:** C5 and C6 on synthetic demos with a mock replayer that uses the same dynamics as the data (`tests/test_conversion_roundtrip.py`). The tests inject a unit bug, lossy re-encoding, sidecar tampering, a missing sidecar, ignored streams, sim drift, nondeterminism and a render channel swap. Each lands on the check written for it, and a clean run passes.
* **Not run here:** `IsaacLabDemoReplayer`, because there's no GPU. It follows the Isaac Lab 2.x `replay_demos.py` API. Check the lines marked `VERIFY` on first use.
* The pixel-format ceiling and lossless reference encodes need the `ffmpeg` command line. Without it, the ceiling check is skipped.
* Rebuilt camera frames come from the LeRobot video, so C5's PSNR includes the codec loss. C3 measures the same loss, frame by frame and with more checks. C5 only confirms that the frames come back in a usable state.
