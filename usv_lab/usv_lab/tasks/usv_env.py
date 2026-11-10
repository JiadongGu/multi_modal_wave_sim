"""The Isaac Lab adapter: boats in the scene, physics from usv_core.

Each physics substep (`_apply_action`, `decimation` times per policy step):
  1. read every boat's CoM position, orientation and velocities from the engine
  2. convert at the boundary: world -> env-local (subtract env origins);
     Isaac Lab quaternions are already (x, y, z, w), usv_core's order
  3. move the actuators toward the policy's commands
  4. ask usv_core's Composite for each boat's force and torque (world frame, about the CoM)
  5. hand them to the engine's wrench composer (is_global: world frame, applied at the CoM)
The engine integrates; gravity stays on in the engine and buoyancy cancels it.

Treadmill (keeping boats on their plot)
---------------------------------------
Each boat has a plot of `env_spacing` metres around its env origin. When a boat
gets more than `recenter_radius` from the plot centre, it is moved back to the
centre with its velocity, heading and tilt unchanged, and the distance it was
moved is added to its `sea_offset`. Every position the physics sees is
plot position + sea_offset, so the boat keeps meeting the waves it would have
met had it kept going: forces depend only on the sea there and on relative
positions, so the move is invisible to the physics. The boat feels open sea;
in the scene it never leaves its plot.
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
        self.imu = self.scene.sensors.get("imu")          # None when switched off in the config
        self.camera = self.scene.sensors.get("top_cam")
        dev = self.device
        self.geo = HullGeometry(cfg.hull, device=dev, dtype=torch.float32)
        # Mass and inertia come from the hull, not the engine. The rigid body has no collision shape, so the
        # engine has nothing to compute an inertia from and falls back to a tiny default (~18 kg m^2 for a
        # 4.5 t hull, ~170x too small); with that, the buoyancy righting moment rings faster than the step
        # can follow and the hull capsizes on its own in flat water.
        inertia = torch.diag(torch.tensor(self.geo.inertia(), device=dev)).reshape(1, 1, 9).repeat(self.num_envs, 1, 1)
        self.boat.set_masses_index(masses=torch.full((self.num_envs, 1), self.geo.mass, device=dev))
        self.boat.set_inertias_index(inertias=inertia)
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
        self.sea_offset = torch.zeros(self.num_envs, 3, device=dev)          # treadmill: distance moved back so far
        if self.water is not None:
            self.water.bind_fabric()

    def _setup_scene(self) -> None:
        """Runs after the scene is built and BEFORE the simulation resets, which is when Fabric (what the
        renderer draws from) is filled: the water meshes must exist by then or they are never drawn."""
        self.water = None
        if self.cfg.draw_water_envs > 0:
            from .water_visual import WaterPatches

            n = self.scene.num_envs
            self.water = WaterPatches(list(range(min(self.cfg.draw_water_envs, n))),
                                      self.cfg.water_patch_size, self.cfg.water_resolution, self.cfg.recenter_radius,
                                      self.scene.env_origins.device)

    # ------------------------------------------------------------------ physics

    def plot_position(self) -> torch.Tensor:
        """[N, 3] CoM position relative to the boat's own plot centre (what the scene shows)."""
        return self.boat.data.root_com_pos_w.torch - self.scene.env_origins

    def boat_state(self) -> BodyState:
        """The boats as the physics sees them: plot position + treadmill offset (the 'true' voyage position)."""
        d = self.boat.data
        return BodyState(
            pos=self.plot_position() + self.sea_offset,
            quat=d.root_link_quat_w.torch,
            lin_vel=d.root_com_lin_vel_w.torch,
            ang_vel=d.root_com_ang_vel_w.torch,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.requested = actions.clamp(-1.0, 1.0)
        self.recenter()
        if self.water is not None:
            # drawn for the moment the frame renders, at the end of this policy step
            self.water.update(self.sea, self.sim_time + self.step_dt, self.plot_position()[:, :2], self.sea_offset)

    def recenter(self) -> None:
        """Treadmill: move boats that strayed back to their plot centre and remember how far."""
        plot = self.plot_position()
        far = plot[:, :2].norm(dim=-1) > self.cfg.recenter_radius
        if not bool(far.any()):
            return
        ids = far.nonzero(as_tuple=False).squeeze(-1)
        shift = torch.zeros_like(plot[ids])
        shift[:, :2] = plot[ids, :2]                        # horizontal only; height and attitude untouched
        self.sea_offset[ids] += shift
        pose = self.boat.data.root_link_pose_w.torch[ids].clone()   # (x, y, z, qx, qy, qz, qw)
        pose[:, :3] -= shift
        self.boat.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)   # velocities are not touched

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
        bad = ~torch.isfinite(s.pos).all(-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return capsized | bad, time_out

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
        self.sea_offset[env_ids] = 0.0
