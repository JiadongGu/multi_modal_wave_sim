"""Pillar navigation: find the bright orange pillar on your own plot and reach it, using only camera and lidar."""

from __future__ import annotations

from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from usv_core import LidarCfg

from .usv_env_cfg import DECK_Z, UsvFloatEnvCfg, UsvSceneCfg

CAMERA_WIDTH, CAMERA_HEIGHT = 96, 72
LIDAR = LidarCfg(channels=3, vertical_fov=(-4.0, 4.0), horizontal_res=2.0, max_range=60.0,
                 mount=(0.0, 0.0, DECK_Z + 1.0), update_period=0.1)     # 3 x 180 = 540 beams
LIDAR_BEAMS = 540
PILLAR_RADIUS, PILLAR_HEIGHT, PILLAR_CENTRE_Z = 1.0, 12.0, 3.0          # spans z = -3 .. +9 m


@configclass
class UsvPillarSceneCfg(UsvSceneCfg):
    pillar: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pillar",
        spawn=sim_utils.CylinderCfg(
            radius=PILLAR_RADIUS,
            height=PILLAR_HEIGHT,
            rigid_props=[
                sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                PhysxRigidBodyCfg(disable_gravity=True),
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0), emissive_color=(0.5, 0.12, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(30.0, 0.0, PILLAR_CENTRE_Z)),
    )


@configclass
class UsvPillarEnvCfg(UsvFloatEnvCfg):
    """Policy input: {"camera": [H, W, 3] colour image, "lidar": beam ranges scaled to 0..1}. Nothing else.

    Reward uses the true distance to the pillar (training only; the policy never sees it). The waves
    act on the boat but are not drawn: drawing them is too slow for this many boats.
    """

    episode_length_s = 40.0
    observation_space = spaces.Dict({
        "camera": spaces.Box(low=float("-inf"), high=float("inf"), shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3)),
        "lidar": spaces.Box(low=0.0, high=1.0, shape=(LIDAR_BEAMS,)),
    })
    scene: UsvPillarSceneCfg = UsvPillarSceneCfg(num_envs=256, env_spacing=100.0, replicate_physics=True)
    lidar = LIDAR

    recenter_radius = None               # stays on its plot, like the pillar
    spawn_radius = 5.0                   # boat starts near the plot centre [m]
    pillar_distance = (20.0, 40.0)       # pillar this far from the plot centre [m]
    arrive_radius = 5.0                  # success: boat's centre this close to the pillar's centre [m]
    plot_limit = 48.0                    # episode fails beyond this from the plot centre [m]

    progress_reward = 1.0                # per metre closer to the pillar
    arrival_reward = 20.0
    failure_penalty = 10.0
    step_cost = 0.01

    def __post_init__(self):
        super().__post_init__()
        self.scene.imu = None
        cam = self.scene.top_cam
        cam.width, cam.height = CAMERA_WIDTH, CAMERA_HEIGHT
        cam.data_types = ["rgb"]
        cam.update_latest_camera_pose = False
        # the pillar must be the only orange thing the camera can see
        self.scene.boat.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.78))
        self.scene.backdrop.init_state.pos = (0.0, 0.0, 0.0)   # no drawn waves here: the backdrop is the sea


@configclass
class UsvPillarPlayEnvCfg(UsvPillarEnvCfg):
    """Watch a trained policy: 16 boats. Same look as training, so the camera sees what it trained on."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
