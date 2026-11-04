"""Policy adapters, created from a short spec string.

* ``pd``, ``pd-avoid`` - reference controllers for the Reach2D simulator.
* ``pd-chunk`` - a chunked (VLA-style) reference controller.
* ``groot://host:port`` - a running Isaac-GR00T inference server.
* ``package.module:ClassName`` - any object with ``reset(task)`` and ``act(obs)``
  or ``predict_chunk(obs)``.

Keyword options (``--policy-arg kp=3``) are passed to the constructor.
``execute_steps`` is consumed here and wraps chunk predictors in
:class:`ChunkedPolicy`.
"""

from __future__ import annotations

import importlib
from typing import Any

from groot_eval.policies.base import ChunkedPolicy, ChunkPredictor, Policy


def make_policy(spec: str, **kwargs: Any) -> Policy:
    execute_steps = int(kwargs.pop("execute_steps", 8))
    obj: Any
    if spec == "pd":
        from groot_eval.policies.scripted import PDPolicy

        obj = PDPolicy(**kwargs)
    elif spec == "pd-avoid":
        from groot_eval.policies.scripted import PDPolicy

        kwargs.setdefault("avoid", 2.0)
        obj = PDPolicy(**kwargs)
    elif spec == "pd-chunk":
        from groot_eval.policies.scripted import PDChunkPredictor

        obj = PDChunkPredictor(**kwargs)
    elif spec.startswith("groot://"):
        from groot_eval.policies.groot_client import GrootServerPredictor

        host, _, port = spec[len("groot://") :].partition(":")
        obj = GrootServerPredictor(host=host or "localhost", port=int(port or 5555), **kwargs)
    elif ":" in spec:
        module, cls = spec.split(":", 1)
        obj = getattr(importlib.import_module(module), cls)(**kwargs)
    else:
        raise ValueError(f"unknown policy spec {spec!r}")

    if hasattr(obj, "predict_chunk") and not hasattr(obj, "act"):
        return ChunkedPolicy(obj, execute_steps=execute_steps)
    return obj


__all__ = ["ChunkPredictor", "ChunkedPolicy", "Policy", "make_policy"]
