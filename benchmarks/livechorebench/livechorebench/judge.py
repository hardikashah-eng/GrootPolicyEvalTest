"""AI judge: a vision-language model grades what sensors don't measure, and is audited against humans.

Instrumented checks (poses, encoders, scales) always win. The judge answers only the yes/no questions a spec
asks with the ``judge`` predicate, e.g. ``judge: "Is the towel folded in half with its edges aligned?"``, from
camera frames of the finished scene. Its answers go into the world state under ``judge`` and are scored like
any other predicate.

A judge is trusted only after it has been measured: :func:`agreement` compares its answers with human grades
on the same episodes (agreement with a Wilson interval, Cohen's kappa, false passes and false fails), and
:func:`audit_gate` says whether it may grade unsupervised.

The model behind the judge is a plain callable ``backend(prompt, images) -> text``, so any provider works.
:func:`anthropic_backend` builds one on the Anthropic SDK if it is installed.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable
from typing import Any

from livechorebench.spec import TaskSpec

Backend = Callable[[str, list[bytes]], str]


class JudgeError(ValueError):
    """The judge's reply could not be read as a verdict."""


def judge_questions(spec: TaskSpec) -> list[str]:
    """Every question the spec asks the judge, in order, without repeats."""
    out: list[str] = []

    def walk(p: Any) -> None:
        (kind, arg), = p.items()
        if kind in ("all", "any"):
            for q in arg:
                walk(q)
        elif kind == "judge" and arg not in out:
            out.append(arg)

    walk(spec.success)
    for sg in spec.subgoals:
        walk(sg["check"])
    for r in (spec.scoring.get("graded") or []):
        walk(r["when"])
    return out


def build_prompt(spec: TaskSpec, instruction: str, questions: list[str]) -> str:
    qs = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    return (
        "You are grading one episode of a robot doing a household chore. The images show the scene at the end "
        "of the episode, in time order if there is more than one.\n\n"
        f"Instruction given to the robot: {instruction!r}\n\n"
        "Answer each question strictly from what is visible. If the images do not show enough to decide, "
        "answer null rather than guessing.\n\n"
        f"Questions:\n{qs}\n\n"
        'Reply with JSON only: {"answers": {"<question number>": true | false | null}, '
        '"rationale": "<one or two sentences>", "confidence": <0 to 1>}'
    )


def parse_reply(text: str, questions: list[str]) -> dict[str, Any]:
    """Read the judge's JSON, tolerating code fences or text around it."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise JudgeError("no JSON object in the judge's reply")
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeError(f"judge reply is not valid JSON: {e}") from e
    ans = raw.get("answers") or {}
    answers: dict[str, bool | None] = {}
    for i, q in enumerate(questions):
        v = ans.get(str(i + 1), ans.get(i + 1, ans.get(q)))
        answers[q] = v if isinstance(v, bool) else None
    conf = raw.get("confidence")
    return {"answers": answers, "rationale": str(raw.get("rationale", "")),
            "confidence": float(conf) if isinstance(conf, (int, float)) else None}


class Judge:
    def __init__(self, backend: Backend, model_id: str = "unknown", min_confidence: float = 0.0) -> None:
        self.backend, self.model_id, self.min_confidence = backend, model_id, min_confidence

    def grade(self, spec: TaskSpec, instruction: str, images: list[bytes]) -> dict[str, Any]:
        """Answers to the spec's judge questions. Unanswerable or low-confidence answers come back as None."""
        questions = judge_questions(spec)
        if not questions:
            return {"answers": {}, "rationale": "no judge questions in this task", "confidence": None,
                    "model_id": self.model_id}
        verdict = parse_reply(self.backend(build_prompt(spec, instruction, questions), images), questions)
        if verdict["confidence"] is not None and verdict["confidence"] < self.min_confidence:
            verdict["answers"] = {q: None for q in questions}
        verdict["model_id"] = self.model_id
        return verdict

    @staticmethod
    def for_world(verdict: dict[str, Any]) -> dict[str, bool]:
        """The answers the world state carries (abstentions left out, so they score as not met)."""
        return {q: v for q, v in verdict["answers"].items() if isinstance(v, bool)}


def anthropic_backend(model: str = "claude-sonnet-5-5", max_tokens: int = 512) -> Backend:
    """A backend on the Anthropic Messages API (needs the ``anthropic`` package and an API key)."""
    import anthropic

    client = anthropic.Anthropic()

    def call(prompt: str, images: list[bytes]) -> str:
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(img).decode()}} for img in images]
        content.append({"type": "text", "text": prompt})
        msg = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": content}])
        return "".join(getattr(b, "text", "") for b in msg.content)

    return call


# -- audit against humans -----------------------------------------------------------------------
def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def agreement(judge: list[bool | None], human: list[bool]) -> dict[str, Any]:
    """Judge vs human on the same items. Abstentions count against coverage, not agreement."""
    pairs = [(j, h) for j, h in zip(judge, human) if j is not None]
    n = len(pairs)
    same = sum(j == h for j, h in pairs)
    po = same / n if n else math.nan
    pj = sum(j for j, _ in pairs) / n if n else math.nan
    ph = sum(h for _, h in pairs) / n if n else math.nan
    pe = pj * ph + (1 - pj) * (1 - ph) if n else math.nan
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else (1.0 if n and po == 1 else math.nan)
    return {"items": len(human), "answered": n, "coverage": n / len(human) if human else math.nan,
            "agreement": po, "agreement_ci": _wilson(same, n), "kappa": kappa,
            "false_pass": sum(j and not h for j, h in pairs), "false_fail": sum((not j) and h for j, h in pairs)}


def audit_gate(stats: dict[str, Any], min_agreement_lower: float = 0.85, min_kappa: float = 0.7,
               min_coverage: float = 0.9, max_false_pass_rate: float = 0.05) -> tuple[bool, list[str]]:
    """May this judge grade without a human checking every episode?"""
    why = []
    if not stats["answered"]:
        return False, ["no answered items"]
    if stats["agreement_ci"][0] < min_agreement_lower:
        why.append(f"agreement lower bound {stats['agreement_ci'][0]:.2f} < {min_agreement_lower}")
    if not (stats["kappa"] >= min_kappa):
        why.append(f"kappa {stats['kappa']:.2f} < {min_kappa}")
    if stats["coverage"] < min_coverage:
        why.append(f"answers only {stats['coverage']:.0%} of items")
    if stats["false_pass"] / stats["answered"] > max_false_pass_rate:
        why.append(f"false passes {stats['false_pass']}/{stats['answered']}: the judge passes failed chores")
    return (not why), why
