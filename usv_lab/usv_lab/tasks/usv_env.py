"""The Isaac Lab adapter: boats in the scene, physics from usv_core.

Each physics substep (`_apply_action`, `decimation` times per policy step):
  1. read every boat's CoM position, orientation and velocities from the engine
  2. convert at the boundary: world -> env-local (subtract env origins);
     Isaac Lab quaternions are already (x, y, z, w), usv_core's order
  3. move the actuators toward the policy's commands
  4. ask usv_core's Composite for each boat's force and torque (world frame, about the CoM)
  5. hand them to the engine's wrench composer (is_global: world frame, applied at the CoM)
The engine integrates; gravity stays on in the engine and buoyancy cancels it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from usv_core import (
    Actuators,
    BodyState,
    ColumnBuoyancy,
    Composite,
    HullGeometry,
    HydroDamping,
    ModelContext,
    SineWaveField,
    TwinThruster,
    world_to_body,
)

from .usv_env_cfg import UsvFloatEnvCfg


class UsvEnv(DirectRLEnv):
    cfg: UsvFloatEnvCfg

    def __init__(self, cfg: UsvFloatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.boat = self.scene["boat"]
        dev = self.device
        self.geo = HullGeometry(cfg.hull, device=dev, dtype=torch.float32)
        self.sea = SineWaveField(cfg.sea, self.num_envs, device=dev, seed=cfg.sea_seed)
        self.model = Composite({
            "buoyancy": ColumnBuoyancy(self.geo),
            "damping": HydroDamping(self.geo, cfg.damping),
            "thrust": TwinThruster(self.geo, cfg.thruster),
        })
        self.actuators = Actuators(self.num_envs, cfg.actuators, device=dev)
        self.requested = torch.zeros(self.num_envs, 2, device=dev)
        self.sim_time = 0.0
        self._down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.num_envs, 3)

    # ------------------------------------------------------------------ physics

    def boat_state(self) -> BodyState:
        d = self.boat.data
        return BodyState(
            pos=d.root_com_pos_w.torch - self.scene.env_origins,
            quat=d.root_link_quat_w.torch,
            lin_vel=d.root_com_lin_vel_w.torch,
            ang_vel=d.root_com_ang_vel_w.torch,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.requested = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        dt = self.physics_dt
        commands = self.actuators.step(self.requested, dt)
        ctx = ModelContext(t=self.sim_time, dt=dt, sea=self.sea, actions=commands)
        w = self.model(self.boat_state(), ctx)
        self.boat.permanent_wrench_composer.set_forces_and_torques_index(
            forces=w.force[:, None, :], torques=w.torque[:, None, :], is_global=True
        )
        self.sim_time += dt

    # ---------------------------------------------------------------------- RL

    def _get_observations(self) -> dict:
        s = self.boat_state()
        obs = torch.cat([s.lin_vel_body, s.ang_vel_body, world_to_body(s.quat, self._down), self.actuators.state], -1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)   # no task yet: this env only floats

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.boat_state()
        up_z = -world_to_body(s.quat, self._down)[:, 2]          # z of the hull's up axis in the world
        capsized = up_z < self.cfg.capsize_up_z
        drifted = s.pos[:, :2].norm(dim=-1) > self.cfg.max_distance
        bad = ~torch.isfinite(s.pos).all(-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return capsized | drifted | bad, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.boat._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = env_ids.numel()

        pose = self.boat.data.default_root_pose.torch[env_ids].clone()      # (x, y, z, qx, qy, qz, qw)
        pose[:, :3] += self.scene.env_origins[env_ids]
        yaw = sample_uniform(*self.cfg.reset_yaw, (n,), self.device)
        pose[:, 3:7] = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)
        self.boat.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.boat.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        self.sea.resample(env_ids)
        self.actuators.reset(env_ids)
        self.model.reset(env_ids)
        self.requested[env_ids] = 0.0
