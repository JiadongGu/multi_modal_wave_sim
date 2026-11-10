"""Station keeping: reach the centre of your own plot and stay there, in waves.

The goal is given in the boat's own frame, so the policy sees "the centre is 6 m
ahead and 2 m to port" and never an absolute position. Heading is free: the boat
may point however it likes as long as it holds position.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from usv_core import world_to_body

from .station_env_cfg import UsvStationEnvCfg
from .usv_env import UsvEnv


class UsvStationEnv(UsvEnv):
    cfg: UsvStationEnvCfg

    def __init__(self, cfg: UsvStationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.previous_request = torch.zeros(self.num_envs, 2, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_request = self.requested.clone()
        super()._pre_physics_step(actions)

    def distance_to_centre(self) -> torch.Tensor:
        return self.plot_position()[:, :2].norm(dim=-1)

    def _get_observations(self) -> dict:
        s = self.boat_state()
        to_centre = -self.plot_position()
        to_centre[:, 2] = 0.0
        obs = torch.cat([
            world_to_body(s.quat, to_centre)[:, :2], s.lin_vel_body, s.ang_vel_body,
            world_to_body(s.quat, self._down), self.actuators.state,
        ], -1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d = self.distance_to_centre()
        speed2 = (self.boat_state().lin_vel[:, :2] ** 2).sum(-1)
        rate2 = ((self.requested - self.previous_request) ** 2).sum(-1)
        reward = (torch.exp(-((d / c.position_scale_wide) ** 2)) + torch.exp(-((d / c.position_scale_tight) ** 2))
                  - c.speed_penalty * speed2 - c.action_rate_penalty * rate2
                  - c.termination_penalty * self.reset_terminated.float())
        log = self.extras.setdefault("log", {})
        log["Station/mean_distance_m"] = d.mean()
        log["Station/within_2m_fraction"] = (d < 2.0).float().mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        failed, time_out = super()._get_dones()
        return failed | (self.distance_to_centre() > self.cfg.max_distance), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        ids = self.boat._ALL_INDICES if env_ids is None else env_ids
        self.previous_request[ids] = 0.0
