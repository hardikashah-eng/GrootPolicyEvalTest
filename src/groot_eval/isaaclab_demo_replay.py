"""Isaac Lab :class:`~groot_eval.conversion_roundtrip.DemoReplayer` for ``groot-twin replay-check``.

Written against Isaac Lab 2.x ``scripts/tools/replay_demos.py`` (``HDF5DatasetFileHandler``,
``EpisodeData.get_initial_state / get_next_action / get_next_state``,
``env.reset_to(..., is_relative=True)``). Not run here: there is no GPU. The lines most likely
to need adjusting for your Isaac Lab version are marked VERIFY. Run under Isaac Sim's python
(``./isaaclab.sh -p -m groot_eval.twin_cli replay-check ...``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from groot_eval.conversion_roundtrip import ReplayResult


class IsaacLabDemoReplayer:
    def __init__(self, task: str, camera_map: dict[str, str] | None = None, robot_name: str = "robot",
                 success_term: str | None = "success", device: str = "cuda:0", headless: bool = True):
        self.task, self.camera_map, self.robot_name = task, camera_map or {}, robot_name
        self.success_term, self.device, self.headless = success_term, device, headless
        self._env = None
        self._app = None

    def _ensure_env(self):
        if self._env is not None:
            return self._env
        from isaaclab.app import AppLauncher

        self._app = AppLauncher(headless=self.headless, enable_cameras=bool(self.camera_map)).app
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  registers the tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        cfg = parse_env_cfg(self.task, device=self.device, num_envs=1)
        cfg.recorders = {}
        terms = getattr(cfg, "terminations", None)  # keep the success term, drop the rest so a replay runs to the end
        if terms is not None:
            for name in list(vars(terms)):
                if name != self.success_term and not name.startswith("_"):
                    setattr(terms, name, None)
        self._env = gym.make(self.task, cfg=cfg).unwrapped
        return self._env

    def _load(self, h5_path: str, demo: str):
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        h = HDF5DatasetFileHandler()
        h.open(h5_path)
        ep = h.load_episode(demo, self._env.device)
        h.close()
        return ep

    def _frames(self) -> dict[str, np.ndarray]:
        out = {}
        for key, sensor in self.camera_map.items():
            rgb = self._env.scene[sensor].data.output["rgb"][0]  # VERIFY: output key and shape for your camera cfg
            out[key] = rgb[..., :3].detach().cpu().numpy().astype(np.uint8)
        return out

    def _success(self) -> bool:
        if not self.success_term:
            return False
        try:
            return bool(self._env.termination_manager.get_term(self.success_term)[0].item())
        except Exception:  # the task defines no such term
            return False

    def replay(self, h5_path: str, demo: str, mode: Literal["actions", "states"], render: bool = False) -> ReplayResult:
        import torch

        env = self._ensure_env()
        ep = self._load(h5_path, demo)
        ids = torch.tensor([0], device=env.device)
        env.reset_to(ep.get_initial_state(), ids, is_relative=True)
        robot = env.scene[self.robot_name]
        joints, frames, success = [], {k: [] for k in self.camera_map}, False
        if mode == "actions":
            while (a := ep.get_next_action()) is not None:
                env.step(a.unsqueeze(0))
                joints.append(robot.data.joint_pos[0].cpu().numpy().copy())
                if render:
                    for k, v in self._frames().items():
                        frames[k].append(v)
                success = success or self._success()
        else:
            while (s := ep.get_next_state()) is not None:
                env.reset_to(s, ids, is_relative=True)
                env.sim.render()  # VERIFY: some versions also need env.scene.update before sensors refresh
                env.scene.update(dt=env.physics_dt)
                joints.append(robot.data.joint_pos[0].cpu().numpy().copy())
                if render:
                    for k, v in self._frames().items():
                        frames[k].append(v)
        return ReplayResult(joint_pos=np.stack(joints) if joints else np.zeros((0, 0)),
                            frames={k: np.stack(v) for k, v in frames.items() if v},
                            success=success if mode == "actions" and self.success_term else None)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        if self._app is not None:
            self._app.close()
