"""The AI judge (prompt, parsing, abstention, audit against humans) and the three-level episode logs."""

import json
from pathlib import Path

import pytest

from livechorebench import load
from livechorebench.judge import Judge, JudgeError, agreement, audit_gate, build_prompt, judge_questions, parse_reply
from livechorebench.logs import REQUIRED_FIELDS, check_episode_log, episode_valid

ROOT = Path(__file__).resolve().parents[1]
L1 = load(ROOT / "tasks" / "l1_set_table.yaml")
E1 = load(ROOT / "tasks" / "e1_mug_to_rack.yaml")
Q = L1.subgoals[-1]["check"]["judge"]


# -- judge ------------------------------------------------------------------------------------
def test_judge_questions_come_from_the_spec():
    assert judge_questions(L1) == [Q]
    assert judge_questions(E1) == []
    prompt = build_prompt(L1, "Set the table for two", [Q])
    assert Q in prompt and "null" in prompt and "JSON" in prompt


def test_parse_reply_tolerates_fences_and_rejects_non_bool():
    text = 'Here you go:\n```json\n{"answers": {"1": true, "2": "yes"}, "rationale": "ok", "confidence": 0.9}\n```'
    v = parse_reply(text, ["q1?", "q2?"])
    assert v["answers"] == {"q1?": True, "q2?": None} and v["confidence"] == 0.9
    with pytest.raises(JudgeError):
        parse_reply("I think it is fine.", ["q1?"])
    with pytest.raises(JudgeError):
        parse_reply("{not json}", ["q1?"])


def test_judge_grades_with_any_backend_and_abstains_when_unsure():
    seen = {}

    def backend(prompt, images):
        seen["prompt"], seen["n"] = prompt, len(images)
        return json.dumps({"answers": {"1": True}, "rationale": "both settings neat", "confidence": 0.95})

    v = Judge(backend, model_id="fake-vlm").grade(L1, "Set the table for two", [b"jpg1", b"jpg2"])
    assert v["answers"] == {Q: True} and v["model_id"] == "fake-vlm" and seen["n"] == 2
    assert Judge.for_world(v) == {Q: True}

    def unsure(prompt, images):
        return json.dumps({"answers": {"1": True}, "confidence": 0.4})

    v = Judge(unsure, min_confidence=0.7).grade(L1, "x", [b"jpg"])
    assert v["answers"] == {Q: None} and Judge.for_world(v) == {}


def test_judge_not_called_when_task_has_no_questions():
    def boom(prompt, images):
        raise AssertionError("should not be called")
    assert Judge(boom).grade(E1, "x", [])["answers"] == {}


def test_agreement_and_gate_pass_a_good_judge():
    human = [True, False] * 100
    judge = list(human)
    judge[0] = False  # one false fail
    s = agreement(judge, human)
    assert s["answered"] == 200 and s["false_pass"] == 0 and s["false_fail"] == 1
    assert s["kappa"] > 0.95
    ok, why = audit_gate(s)
    assert ok, why


def test_gate_blocks_judges_that_pass_failures_or_abstain_or_have_too_few_items():
    human = [True, False] * 100
    lenient = [True] * 200  # passes everything
    ok, why = audit_gate(agreement(lenient, human))
    assert not ok and any("false passes" in w for w in why) and any("kappa" in w for w in why)
    shy = [h if i % 4 else None for i, h in enumerate(human)]  # abstains on a quarter
    ok, why = audit_gate(agreement(shy, human))
    assert not ok and any("answers only" in w for w in why)
    ok, why = audit_gate(agreement([True, False] * 5, [True, False] * 5))  # perfect but n = 10
    assert not ok and any("lower bound" in w for w in why)
    assert audit_gate(agreement([None, None], [True, False])) == (False, ["no answered items"])


# -- three-level logs ----------------------------------------------------------------------------
def stream(level, rate, seconds=2.0, t0=0.0, **over):
    n = int(seconds * rate) + 1
    t = [t0 + i / rate for i in range(n)]
    cols = {f: [0] * n for f in REQUIRED_FIELDS[level]}
    cols["t"] = t
    if level == "policy":
        cols["step"] = list(range(n))
    if level == "action":
        cols["policy_step"] = [int(x * 10) for x in t]  # policy runs at 10 Hz
        cols["clipped"] = [False] * n
    cols.update(over)
    return cols


def good_log(spec):
    lg = spec.logging
    return {"policy": stream("policy", lg["policy"]["rate_hz"]),
            "action": stream("action", lg["action"]["rate_hz"]),
            "joint": {**stream("joint", lg["joint"]["rate_hz"], seconds=2.05, t0=-0.02),
                      **{f: [0] * (int(2.05 * lg["joint"]["rate_hz"]) + 1) for f in lg["joint"].get("fields", [])}}}


def test_every_task_requires_policy_action_and_joint_logs():
    for p in sorted((ROOT / "tasks").glob("*.yaml")):
        lg = load(p).logging
        assert lg["policy"]["rate_hz"] <= lg["action"]["rate_hz"] <= lg["joint"]["rate_hz"], p.name


def test_a_complete_log_is_valid():
    f = check_episode_log(E1.logging, good_log(E1))
    assert episode_valid(f), f


@pytest.mark.parametrize("break_it, check", [
    (lambda log: log.pop("joint"), "joint.present"),
    (lambda log: log["policy"].pop("checkpoint_hash"), "policy.fields"),
    (lambda log: log["action"].pop("applied"), "action.fields"),
    (lambda log: log.update(joint=stream("joint", 100, seconds=2.05, t0=-0.02)), "joint.rate"),
    (lambda log: log["action"]["t"].reverse(), "action.time"),
    (lambda log: log["action"].update(policy_step=[999] * len(log["action"]["t"])), "trace.policy_to_action"),
    (lambda log: log.update(joint=stream("joint", 200, seconds=1.0)), "trace.action_to_joint"),
    (lambda log: log["policy"]["latency_ms"].append(5), "policy.lengths"),
])
def test_incomplete_logs_invalidate_the_episode(break_it, check):
    log = good_log(E1)
    break_it(log)
    f = check_episode_log(E1.logging, log)
    assert not episode_valid(f)
    assert check in {x.check for x in f}, f


def test_loco_tasks_also_need_base_pose_and_foot_contact():
    m1 = load(ROOT / "tasks" / "m1_laundry_basket.yaml")
    log = good_log(m1)
    assert episode_valid(check_episode_log(m1.logging, log))
    log["joint"].pop("foot_contact")
    assert "joint.fields" in {x.check for x in check_episode_log(m1.logging, log)}


def test_gaps_and_clipping_are_warnings():
    log = good_log(E1)
    t = log["action"]["t"]
    log["action"]["t"] = t[:50] + [x + 0.2 for x in t[50:]]  # a 220 ms hole
    log["action"]["clipped"] = [i % 5 == 0 for i in range(len(t))]
    f = check_episode_log(E1.logging, log)
    assert {"action.gap", "action.clipped"} <= {x.check for x in f if x.level == "warn"}
