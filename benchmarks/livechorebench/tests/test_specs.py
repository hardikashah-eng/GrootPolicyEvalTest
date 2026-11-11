"""LiveChoreBench: the sample task specs load, run the same way in sim and real, and score correctly."""

from pathlib import Path

import pytest

from livechorebench import SpecError, evaluate, load, load_all, placement_sheet, sample, score
from livechorebench.spec import TRACKS, TaskSpec, validate

TASKS = Path(__file__).resolve().parents[1] / "tasks"
SPECS = load_all(TASKS)
BY_ID = {s.id: s for s in SPECS}


def test_every_track_has_a_sample_task():
    assert len(SPECS) == 8
    assert {s.track for s in SPECS} == set(TRACKS)


@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_every_object_exists_in_sim_and_real(spec):
    for name, obj in spec.objects.items():
        assert obj["usd"].endswith(".usd"), name
        assert obj["real_sku"], name


@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_sampling_is_seeded_and_in_range(spec):
    a, b = sample(spec, 7), sample(spec, 7)
    assert a == b  # the sim randomizer and the real placement sheet see the same instance
    assert sample(spec, 8) != a or len(spec.variation) == 0
    assert a["instruction"] in spec.instruction
    for key, v in spec.variation.items():
        kind, arg = next(iter(v.items()))
        if kind == "choice":
            assert a[key] in arg
        else:
            lo, hi = arg
            vals, los, his = (a[key], lo, hi) if isinstance(lo, list) else ([a[key]], [lo], [hi])
            assert all(lo_ <= x <= hi_ for x, lo_, hi_ in zip(vals, los, his))


def test_placement_sheet_lists_each_seed():
    sheet = placement_sheet(BY_ID["execute/mug-to-rack/v3"], [1, 2, 3])
    assert sheet.count("\n| ") == 4 and "mug.xy_m" in sheet


def mug_world(in_rack=True, tilt=5.0, stable=3.0, height=0.0):
    return {"objects": {"mug": {"pos": [0, 0, 0.9], "tilt_deg": tilt}},
            "regions": {"rack.slots": ["mug"] if in_rack else []},
            "joints": {"mug.height_m": height}, "stable_s": stable}


def test_mug_to_rack_scoring():
    spec = BY_ID["execute/mug-to-rack/v3"]
    assert score(spec, mug_world())["success"]
    tipped = score(spec, mug_world(tilt=45))
    assert not tipped["success"] and tipped["subgoals"] == {"lifted": False, "in_rack": True, "upright": False}
    assert not score(spec, mug_world(stable=0.5))["success"]  # placed but still wobbling


def test_drawer_partial_credit():
    spec = BY_ID["execute/items-to-drawer/v1"]
    world = {"regions": {"drawer.volume": ["item_1", "item_2"]}, "joints": {"drawer": 0.08}}
    r = score(spec, world)
    assert not r["success"] and r["partial_credit"] == pytest.approx(0.5)
    world = {"regions": {"drawer.volume": ["item_1", "item_2", "item_3"]}, "joints": {"drawer": 0.004}}
    assert score(spec, world)["success"]


def test_set_table_tolerances():
    spec = BY_ID["long_horizon/set-table-for-two/v1"]
    targets = {f"{k}_{i}": {"pos": [i * 0.6, j * 0.2, 0.75], "yaw_deg": 0.0}
               for i in (1, 2) for j, k in enumerate(("plate", "fork", "cup"))}
    objects = {f"{k}_{s}": {"pos": list(targets[f"{k}_{i}"]["pos"]), "yaw_deg": 5.0}
               for i, s in ((1, "a"), (2, "b")) for k in ("plate", "fork", "cup")}
    world = {"objects": objects, "targets": targets}
    assert score(spec, world)["success"]
    objects["fork_b"]["pos"][0] += 0.05  # 5 cm off: outside the 3 cm tolerance
    r = score(spec, world)
    assert not r["success"] and r["partial_credit"] == pytest.approx(5 / 6)
    objects["fork_b"]["pos"][0] -= 0.05
    objects["plate_a"]["yaw_deg"] = 40.0  # rotated too far
    assert not score(spec, world)["success"]


def test_clarify_grading():
    spec = BY_ID["clarify/bring-the-cup/v1"]
    asked = {"said": ["Which cup would you like, the red or the blue one?"], "objects": {}}
    assert score(spec, asked)["success"] and score(spec, asked)["grade"] == 1.0
    silent = {"said": [], "objects": {"cup_red": {"pos": [0.05, 0.0, 0.9]}}}
    r = score(spec, silent)
    assert not r["success"] and r["grade"] == 0.5
    explained = {"said": ["I'll bring the closer one."], "objects": {"cup_blue": {"pos": [0.0, 0.1, 0.9]}}}
    assert score(spec, explained)["success"]
    assert score(spec, {"said": [], "objects": {}})["grade"] == 0.0


def test_disturbance_and_loco_tasks_fail_on_falls_and_drops():
    tray = BY_ID["disturbance/tray-carry-push/v1"]
    ok = {"regions": {"coffee_table.top": ["tray"], "tray.top": ["cup_on_tray"]}, "fallen": False}
    assert score(tray, ok)["success"]
    assert not score(tray, {**ok, "fallen": True})["success"]
    r = score(tray, {**ok, "dropped": ["cup_on_tray"], "regions": {"coffee_table.top": ["tray"]}})
    assert not r["success"] and r["subgoals"]["cup_kept"] is False
    basket = BY_ID["loco_manipulation/laundry-to-washer/v1"]
    assert score(basket, {"regions": {"washer_area": ["basket"]}})["success"]
    assert not score(basket, {"regions": {"washer_area": ["basket"]}, "dropped": ["sock"]})["success"]


def test_events_say_how_to_run_them_in_the_real_room():
    for spec in SPECS:
        for ev in spec.events:
            assert ev["trigger"] and ev["action"] and len(ev["real"]) > 20


def minimal(**over):
    base = dict(id="t/x/v1", release="2026-12", track="execute", instruction=["do it"],
                scene={"objects": {"box": {"usd": "a.usd", "real_sku": "BOX", "mass_kg": 1.0}}},
                variation={"x": {"uniform": [0, 1]}}, success={"hold_s": 1.0}, subgoals=[],
                limits={"timeout_s": 10})
    base.update(over)
    return TaskSpec(**base)


@pytest.mark.parametrize("bad, msg", [
    ({"track": "juggling"}, "unknown track"),
    ({"scene": {"objects": {"box": {"usd": "a.usd", "mass_kg": 1.0}}}}, "real_sku"),
    ({"scene": {"objects": {"box": {"usd": "a.usd", "real_sku": "BOX"}}}}, "mass_kg"),
    ({"variation": {"x": {"gaussian": [0, 1]}}}, "uniform"),
    ({"variation": {"x": {"uniform": [2, 1]}}}, "lo > hi"),
    ({"success": {"looks_done": True}}, "unknown predicate"),
    ({"limits": {}}, "timeout_s"),
    ({"track": "disturbance"}, "at least one event"),
    ({"track": "self_repair"}, "feedback.text"),
    ({"track": "clarify"}, "said_any"),
    ({"events": [{"trigger": {"t": 1}, "action": {"push": 1}}]}, "real"),
])
def test_validation_refuses_specs_that_cannot_run_in_both_worlds(bad, msg):
    validate(minimal())
    with pytest.raises(SpecError, match=msg):
        validate(minimal(**bad))


def test_unknown_predicate_at_evaluation():
    with pytest.raises(SpecError):
        evaluate({"vibes": 1}, {})


def test_load_roundtrip(tmp_path):
    src = (TASKS / "e1_mug_to_rack.yaml").read_text()
    p = tmp_path / "copy.yaml"
    p.write_text(src)
    assert load(p).id == "execute/mug-to-rack/v3"
