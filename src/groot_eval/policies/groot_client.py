"""Adapter for a running Isaac-GR00T inference server.

Start the server from an Isaac-GR00T checkout (example from the N1.5
README; check the flags for your version)::

    python scripts/inference_service.py --server \\
        --model_path nvidia/GR00T-N1.5-3B \\
        --embodiment-tag gr1 --data-config fourier_gr1_arms_waist

then evaluate against it with ``--policy groot://localhost:5555``.

Isaac-GR00T's client API has moved between releases, so the client class is
looked up from a list of known import paths and can be overridden with
``client_cls="package.module:Class"``. The client must offer
``get_action(obs: dict) -> dict[str, np.ndarray]``.

NOTE: written against the N1.5 client contract and not run against a live
server in this repository's CI (no GPU). Verify key names against your
checkpoint's ``modality.json`` / data config before trusting results.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np

KNOWN_CLIENTS = [
    "gr00t.eval.robot:RobotInferenceClient",          # N1 / N1.5
    "gr00t.eval.service:ExternalRobotInferenceClient",
    "gr00t.policy.server_client:PolicyClient",        # newer layouts
]

DEFAULT_LANGUAGE_KEY = "annotation.human.action.task_description"


_warned_float_video: set[str] = set()


def _to_uint8(arr: np.ndarray, key: str) -> np.ndarray:
    """Convert a camera frame to uint8 without silently destroying it.

    A plain ``astype(uint8)`` turns a float image in [0, 1] (common from
    simulators and torchvision) into an all-black frame, and wraps values above
    255. Floats in [0, 1] are scaled to [0, 255]; everything else is rounded
    and clipped.
    """
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
        if key not in _warned_float_video:
            _warned_float_video.add(key)
            warnings.warn(f"{key} is a float image in [0, 1]; scaling to 0-255 uint8. Check the real camera "
                          "pipeline sends the same range.", stacklevel=3)
        arr = arr * 255.0
    return np.clip(np.rint(np.nan_to_num(arr)), 0, 255).astype(np.uint8)


def _load_client_cls(spec: str | None):
    candidates = [spec] if spec else KNOWN_CLIENTS
    errors = []
    for cand in candidates:
        module, _, cls = cand.partition(":")
        try:
            return getattr(importlib.import_module(module), cls)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{cand}: {exc}")
    raise ImportError(
        "Could not import a GR00T inference client. Install Isaac-GR00T "
        "(pip install -e . in its checkout) or pass client_cls=module:Class.\n" + "\n".join(errors)
    )


class GrootServerPredictor:
    """A :class:`ChunkPredictor` backed by a GR00T inference server.

    Args:
        obs_map: env observation key -> GR00T observation key. Keys not listed
            pass through unchanged (so an env that already emits
            ``state.left_arm`` / ``video.ego_view`` needs no map).
        action_keys: GR00T action keys to concatenate, in order, into the
            env's action vector. Defaults to every ``action.*`` key returned,
            sorted, which is only safe if the env expects that order.
        language_key: where the task instruction goes.
        client: an already-constructed client (tests, custom transports).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        obs_map: dict[str, str] | None = None,
        action_keys: list[str] | str | None = None,
        language_key: str = DEFAULT_LANGUAGE_KEY,
        client_cls: str | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            client = _load_client_cls(client_cls)(host=host, port=int(port))
        self.client = client
        self.obs_map = obs_map or {}
        if isinstance(action_keys, str):
            action_keys = [k.strip() for k in action_keys.split(",") if k.strip()]
        self.action_keys = action_keys
        self.language_key = language_key
        self.task = ""
        self.name = f"groot@{host}:{port}"
        self._warned_order = False

    def reset(self, task: str) -> None:
        self.task = task

    def build_request(self, obs: dict[str, Any]) -> dict[str, Any]:
        req: dict[str, Any] = {}
        for key, value in obs.items():
            if key.startswith("annotation."):
                continue
            out_key = self.obs_map.get(key, key)
            arr = np.asarray(value)
            if out_key.startswith("video."):
                arr = _to_uint8(arr, out_key)
                if arr.ndim == 3:  # H, W, C -> 1, H, W, C
                    arr = arr[None]
            elif out_key.startswith("state."):
                arr = arr.astype(np.float64)
                if arr.ndim == 1:
                    arr = arr[None]
            req[out_key] = arr
        task = self.task or next((v for k, v in obs.items() if k.startswith("annotation.")), "")
        req[self.language_key] = [task]
        return req

    def predict_chunk(self, obs: dict[str, Any]) -> np.ndarray:
        reply = self.client.get_action(self.build_request(obs))
        keys = self.action_keys or sorted(k for k in reply if k.startswith("action."))
        if not keys:
            raise ValueError(f"server reply has no action.* keys: {sorted(reply)}")
        if not self.action_keys and len(keys) > 1 and not self._warned_order:
            self._warned_order = True
            warnings.warn(f"no action_keys given: concatenating {keys} in alphabetical order. If the robot expects "
                          "another order (check modality.json), every joint gets the wrong command; pass action_keys.",
                          stacklevel=2)
        parts = []
        for key in keys:
            arr = np.asarray(reply[key], dtype=float)
            if arr.ndim == 1:  # a scalar joint per step, e.g. a gripper
                arr = arr[:, None]
            parts.append(arr)
        horizons = {p.shape[0] for p in parts}
        if len(horizons) != 1:
            raise ValueError(f"action keys disagree on horizon: { {k: p.shape for k, p in zip(keys, parts)} }")
        return np.concatenate(parts, axis=1)
