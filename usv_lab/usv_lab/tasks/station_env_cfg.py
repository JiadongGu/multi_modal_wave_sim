"""Station keeping: hold position at the centre of your own plot, in waves."""

from __future__ import annotations

from isaaclab.utils import configclass

from .usv_env_cfg import UsvFloatEnvCfg, UsvSceneCfg


@configclass
class UsvStationEnvCfg(UsvFloatEnvCfg):
    """Observations: centre in the boat's frame 2, body lin vel 3, body ang vel 3, gravity in body 3, commands 2."""

    episode_length_s = 30.0
    observation_space = 13
    scene: UsvSceneCfg = UsvSceneCfg(num_envs=1024, env_spacing=100.0, replicate_physics=True)

    recenter_radius = None               # no treadmill: the task is to stay on the plot
    water_follows_boat = False           # a fixed patch, so holding station is visible
    spawn_radius = 15.0                  # start up to this far from the centre [m]
    max_distance = 35.0                  # episode ends beyond this [m]

    # reward
    position_scale_wide = 5.0            # [m] exp(-(d/scale)^2): pulls in from far away
    position_scale_tight = 1.0           # [m] rewards actually sitting on the centre
    speed_penalty = 0.01                 # per (m/s)^2 of horizontal speed
    action_rate_penalty = 0.02           # per unit^2 of change in the command between policy steps
    termination_penalty = 20.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.top_cam = None        # state-only policy: no rendering cost while training
        self.scene.imu = None


@configclass
class UsvStationPlayEnvCfg(UsvStationEnvCfg):
    """Watch a trained policy: 16 boats, each with its sea drawn."""

    draw_water_envs = 16

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
