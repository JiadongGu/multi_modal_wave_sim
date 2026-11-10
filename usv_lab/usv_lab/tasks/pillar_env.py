"""Find the orange pillar on your own plot and reach it, using only camera and lidar.

The policy gets the colour image and the lidar ranges and nothing else: no position, no velocity, no
heading, no direction to the goal. The true distance is used only for the reward during training.
The pillar is visible to this boat's camera and lidar only: it lives under this environment in the
scene, and the lidar's cylinder list is per boat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.images import normalize_camera_image
from isaaclab.utils.math import sample_uniform

from usv_core import Cylinders

from .pillar_env_cfg import PILLAR_CENTRE_Z, PILLAR_HEIGHT, PILLAR_RADIUS, UsvPillarEnvCfg
from .usv_env import UsvEnv


class UsvPillarEnv(UsvEnv):
    cfg: UsvPillarEnvCfg

    def __init__(self, cfg: UsvPillarEnvCfg, render_mode: str | None = None, **kwargs):
        # the first reset happens after this, but the lidar hook needs these as soon as it runs
        super().__init__(cfg, render_mode, **kwargs)
        self.pillar = self.scene["pillar"]
        n, dev = self.num_envs, self.device
        self.pillar_xy = torch.zeros(n, 2, device=dev)              # on the boat's own plot
        self.previous_distance = torch.zeros(n, device=dev)
        self.arrived = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.success_rate = torch.zeros(1, device=dev)

    def lidar_cylinders(self):
        return Cylinders(self.pillar_xy[:, None, :], PILLAR_RADIUS,
                         PILLAR_CENTRE_Z - PILLAR_HEIGHT / 2, PILLAR_CENTRE_Z + PILLAR_HEIGHT / 2)

    def distance_to_pillar(self) -> torch.Tensor:
        return (self.plot_position()[:, :2] - self.pillar_xy).norm(dim=-1)

    def _get_observations(self) -> dict:
        camera = normalize_camera_image(self.camera.data.output["rgb"].torch, "rgb")
        lidar = (self.lidar_ranges.clamp(max=self.cfg.lidar.max_range) / self.cfg.lidar.max_range)
        return {"policy": {"camera": camera, "lidar": lidar}}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        capsized_or_bad, time_out = super()._get_dones()
        off_plot = self.plot_position()[:, :2].norm(dim=-1) > self.cfg.plot_limit
        self.failed = capsized_or_bad | off_plot
        self.arrived = self.distance_to_pillar() < self.cfg.arrive_radius
        ended = self.failed | self.arrived | time_out
        if bool(ended.any()):                                      # share of finished episodes that reached the pillar
            self.success_rate = 0.98 * self.success_rate + 0.02 * self.arrived[ended].float().mean()
        return self.failed | self.arrived, time_out

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d = self.distance_to_pillar()
        reward = (c.progress_reward * (self.previous_distance - d) + c.arrival_reward * self.arrived.float()
                  - c.failure_penalty * self.failed.float() - c.step_cost)
        self.previous_distance = d.clone()
        log = self.extras.setdefault("log", {})
        log["Nav/mean_distance_m"] = d.mean()
        log["Nav/success_rate"] = self.success_rate
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        ids = torch.as_tensor(self.boat._ALL_INDICES if env_ids is None else env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(ids)
        n = ids.numel()
        angle = sample_uniform(0.0, 2.0 * math.pi, (n,), self.device)
        radius = sample_uniform(*self.cfg.pillar_distance, (n,), self.device)
        self.pillar_xy[ids] = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], -1)

        pose = torch.zeros(n, 7, device=self.device)                # (x, y, z, qx, qy, qz, qw)
        pose[:, :2] = self.scene.env_origins[ids, :2] + self.pillar_xy[ids]
        pose[:, 2] = PILLAR_CENTRE_Z
        pose[:, 6] = 1.0
        self.pillar.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.previous_distance[ids] = (self.spawn_xy[ids] - self.pillar_xy[ids]).norm(dim=-1)
