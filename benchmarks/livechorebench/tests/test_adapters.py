"""Adapters: Isaac Lab state and real-room measurements give the same world state, and the same score."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from livechorebench import load, score
from livechorebench.adapters import IsaacLabAdapter, RealRoomAdapter, on_scale, tag_pose_for
from livechorebench.world import Pose, WorldBuilder, quat_from_yaw, route_progress, tilt_deg

ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "cells" / "example_cell.yaml"
E1 = load(ROOT / "tasks" / "e1_mug_to_rack.yaml")
E2 = load(ROOT / "tasks" / "e2_items_to_drawer.yaml")
L1 = load(ROOT / "tasks" / "l1_set_table.yaml")
M1 = load(ROOT / "tasks" / "m1_laundry_basket.yaml")
D2 = load(ROOT / "tasks" / "d2_tray_push.yaml")

RACK = Pose.of([0.60, -0.30, 0.90])  # matches the cell file
TABLE = Pose.of([0.0, 0.0, 0.75])


def tilt_quat(deg: float) -> list[float]:
    h = np.radians(deg) / 2
    return [float(np.cos(h)), float(np.sin(h)), 0.0, 0.0]  # about x


# -- a fake Isaac Lab scene ---------------------------------------------------------------------
def entity(pos, quat=(1, 0, 0, 0), joint_names=None, joint_pos=None, n_envs=2):
    data = SimpleNamespace(root_pos_w=np.tile(np.asarray(pos, float), (n_envs, 1)),
                           root_quat_w=np.tile(np.asarray(quat, float), (n_envs, 1)))
    if joint_names:
        data.joint_pos = np.tile(np.asarray(joint_pos, float), (n_envs, 1))
    return SimpleNamespace(data=data, joint_names=joint_names or [])


def fake_env(scene):
    return SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))


def test_isaac_mug_in_rack_scores_success_after_it_settles():
    ad = IsaacLabAdapter(E1)
    scene = {"mug": entity([0.10, -0.35, 0.75]), "rack": entity(RACK.pos), "robot": entity([0, 0, 0.95])}
    w = ad.read(fake_env(scene), 0.0)
    assert not score(E1, w)["success"]
    scene["mug"] = entity([0.62, -0.28, 1.00], tilt_quat(8))  # in the slots, 8 deg tilt
    for t in np.arange(0.1, 2.6, 0.1):
        w = ad.read(fake_env(scene), float(t))
    r = score(E1, w)
    assert w["regions"]["rack.slots"] == ["mug"]
    assert r["success"] and r["subgoals"] == {"lifted": True, "in_rack": True, "upright": True}
    assert w["objects"]["mug"]["tilt_deg"] == pytest.approx(8, abs=1e-6)


def test_isaac_tilted_mug_fails_and_fall_is_detected():
    ad = IsaacLabAdapter(E1)
    scene = {"mug": entity([0.62, -0.28, 1.00], tilt_quat(35)), "rack": entity(RACK.pos), "robot": entity([0, 0, 0.30])}
    w = ad.read(fake_env(scene), 0.0)
    assert w["fallen"] and not score(E1, w)["subgoals"]["upright"]


def test_isaac_reads_drawer_joint_by_name_and_env_index():
    ad = IsaacLabAdapter(E2, env_index=1)
    drawer = entity([0.55, 0.40, 0.75], joint_names=["hinge_dummy", "drawer_slide"], joint_pos=[0.0, 0.004])
    drawer.data.joint_pos[1, 1] = 0.006  # env 1 differs from env 0
    items = {f"item_{i}": entity([0.55 + 0.05 * i, 0.40, 0.70]) for i in (1, 2, 3)}
    w = ad.read(fake_env({"drawer": drawer, **items}), 0.0)
    assert w["joints"]["drawer"] == pytest.approx(0.006)
    assert sorted(w["regions"]["drawer.volume"]) == ["item_1", "item_2", "item_3"]
    assert score(E2, w)["success"]


def test_isaac_accepts_torch_like_tensors():
    class T:  # stands in for a CUDA tensor
        def __init__(self, a): self.a = np.asarray(a, float)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.a
    ent = SimpleNamespace(data=SimpleNamespace(root_pos_w=T([[0.6, -0.3, 1.0]]), root_quat_w=T([[1, 0, 0, 0]])),
                          joint_names=[])
    w = IsaacLabAdapter(E1).read({"mug": ent, "rack": entity(RACK.pos, n_envs=1)}, 0.0)
    assert w["regions"]["rack.slots"] == ["mug"]


# -- the real room ------------------------------------------------------------------------------
def frame_with_mug(t, mug: Pose, **extra):
    return {"t": t, "tags": {"7": tag_pose_for(mug, [0.0, 0.0, -0.05])}, "robot": {"base_height_m": 0.95}, **extra}


def test_real_room_tags_give_the_same_world_as_the_sim():
    real, sim = RealRoomAdapter(E1, CELL), IsaacLabAdapter(E1)
    mug = Pose.of([0.62, -0.28, 1.00], tilt_quat(8))
    for t in np.arange(0.0, 2.6, 0.1):
        wr = real.update(frame_with_mug(float(t), mug))
        ws = sim.read({"mug": entity(mug.pos, mug.quat), "rack": entity(RACK.pos), "robot": entity([0, 0, 0.95])}, float(t))
    assert np.allclose(wr["objects"]["mug"]["pos"], ws["objects"]["mug"]["pos"])
    assert wr["regions"] == ws["regions"] and wr["stable_s"] == pytest.approx(ws["stable_s"])
    assert score(E1, wr) == score(E1, ws)


def test_real_room_keeps_last_pose_when_tag_is_occluded_and_collects_speech():
    real = RealRoomAdapter(E1, CELL)
    mug = Pose.of([0.62, -0.28, 1.00])
    real.update(frame_with_mug(0.0, mug, speech=["Putting the mug in the rack."]))
    w = real.update({"t": 0.1, "tags": {}, "speech": ["Done."]})
    assert w["regions"]["rack.slots"] == ["mug"]
    assert w["said"] == ["Putting the mug in the rack.", "Done."]


def test_real_room_mocap_overrides_tags():
    real = RealRoomAdapter(E1, CELL)
    w = real.update(frame_with_mug(0.0, Pose.of([0.0, 0.0, 0.8]), mocap={"mug": {"pos": [0.6, -0.3, 1.0]}}))
    assert w["regions"]["rack.slots"] == ["mug"]


def test_real_room_drawer_encoder_and_scale_prove_items_in_closed_drawer():
    real = RealRoomAdapter(E2, CELL)
    # tags hidden inside the closed drawer: only the scale knows what is in it
    w = real.update({"t": 0.0, "encoders": {"drawer_encoder": 0.003}, "scales": {"drawer_scale": 1.40 + 0.45}})
    assert w["joints"]["drawer"] == pytest.approx(0.003)
    assert sorted(w["regions"]["drawer.volume"]) == ["item_1", "item_2", "item_3"]
    assert score(E2, w)["success"]
    w = real.update({"t": 0.1, "encoders": {"drawer_encoder": 0.003}, "scales": {"drawer_scale": 1.40 + 0.35}})
    assert sorted(w["regions"]["drawer.volume"]) == ["item_1", "item_2"]
    assert not score(E2, w)["success"]


def test_scale_only_claims_what_the_weight_proves():
    assert on_scale(0.30, {"a": 0.1, "b": 0.2, "c": 0.3}, 0.01) == []  # a+b or c: nothing is certain
    assert on_scale(0.50, {"a": 0.1, "b": 0.2, "c": 0.3}, 0.01) == ["b", "c"]
    assert on_scale(0.0, {"a": 0.1}, 0.01) == []


def test_real_room_set_table_targets_follow_the_mat():
    cell = {"fixtures": {"table_mat": {"pos": [1.8, 0.0, 0.75], "yaw_deg": 90}},
            "tags": {n: {"id": i, "offset": [0, 0, 0]} for i, n in enumerate(
                ["plate_a", "plate_b", "fork_a", "fork_b", "cup_a", "cup_b"])}}
    real, mat = RealRoomAdapter(L1, cell), Pose.of([1.8, 0.0, 0.75], quat_from_yaw(90))
    from livechorebench.world import from_frame
    tags = {}
    for i, (_obj, tgt) in enumerate([("plate_a", "plate_1"), ("plate_b", "plate_2"), ("fork_a", "fork_1"),
                                    ("fork_b", "fork_2"), ("cup_a", "cup_1"), ("cup_b", "cup_2")]):
        spot = L1.scene["targets"][tgt]
        pose = Pose.of(from_frame(spot["pos"], mat), quat_from_yaw(90 + spot.get("yaw_deg", 0)))
        tags[str(i)] = {"pos": pose.pos.tolist(), "quat": pose.quat.tolist()}
    w = real.update({"t": 0.0, "tags": tags})
    r = score(L1, w)
    assert r["success"]
    assert r["subgoals"]["looks_set"] is False  # no judge answer yet: abstention does not pass
    w = real.update({"t": 0.1, "tags": tags, "judge": {L1.subgoals[-1]["check"]["judge"]: True}})
    assert score(L1, w)["subgoals"]["looks_set"] is True


# -- shared world rules ----------------------------------------------------------------------------
def test_stillness_resets_when_anything_moves():
    b = WorldBuilder(E1)
    b.build(0.0, {"mug": Pose.of([0, 0, 1])})
    assert b.build(1.0, {"mug": Pose.of([0, 0, 1])})["stable_s"] == pytest.approx(1.0)
    assert b.build(1.1, {"mug": Pose.of([0.05, 0, 1])})["stable_s"] == 0.0
    b.reset()
    assert b.build(5.0, {"mug": Pose.of([0, 0, 1])})["stable_s"] == 0.0


def test_drop_needs_a_fall_from_above_the_threshold():
    b = WorldBuilder(M1)
    w = b.build(0.0, {"basket": Pose.of([0, 0, 0.1]), "washer_area": Pose.of([3.5, 3.5, 0])})
    assert w["dropped"] == []  # starts on the floor: not a drop
    b = WorldBuilder(D2)
    b.build(0.0, {"tray": Pose.of([0, 0, 0.9]), "cup_on_tray": Pose.of([0, 0, 0.95]), "coffee_table": TABLE})
    w = b.build(0.5, {"tray": Pose.of([0, 0, 0.9]), "cup_on_tray": Pose.of([0.3, 0, 0.02]), "coffee_table": TABLE})
    assert w["dropped"] == ["cup_on_tray"]
    assert not score(D2, w)["subgoals"]["cup_kept"]


def test_route_progress_and_lift_height_for_loco_task():
    assert route_progress(np.array([3.5, 0.0, 0.5]), M1.scene["route"]["points"]) == pytest.approx(0.5)
    assert route_progress(np.array([3.6, 3.4, 0.5]), M1.scene["route"]["points"]) == pytest.approx(6.9 / 7.0)
    b = WorldBuilder(M1)
    b.build(0.0, {"basket": Pose.of([0, 0, 0.1]), "washer_area": Pose.of([3.5, 3.5, 0])})
    w = b.build(30.0, {"basket": Pose.of([3.5, 3.4, 0.1]), "washer_area": Pose.of([3.5, 3.5, 0])}, base_height_m=0.9)
    w2 = b.build(10.0, {"basket": Pose.of([1.0, 0.0, 0.5]), "washer_area": Pose.of([3.5, 3.5, 0])})
    assert w2["joints"]["basket.height_m"] == pytest.approx(0.4)
    r = score(M1, w)
    assert r["success"] and r["subgoals"]["through_door"]


def test_tilt_of_upright_and_sideways():
    assert tilt_deg(np.array([1.0, 0, 0, 0])) == pytest.approx(0)
    assert tilt_deg(np.asarray(tilt_quat(90))) == pytest.approx(90)
