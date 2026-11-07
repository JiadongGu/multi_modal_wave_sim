"""Configuration for the floating-boat environment.

The hull is described once (HullCfg) and both sides read it: usv_core for the
buoyancy columns, damping and thrusters, and Isaac Lab for the rigid body's
mass and visual size. Mass comes from HullGeometry (it floats at its design
draft), so the two cannot disagree.

For now the rigid body is drawn as a box the size of the hull's bounding box.
Its centre is the centre of mass, so HullCfg.com is set to the box centre.

Sensors (Isaac Lab 3.0; mount offsets are relative to the boat's body frame,
x forward, y port, z up, origin at the CoM; quaternions (x, y, z, w)):
    imu      on the boat body at the deck centre. Proper acceleration
             (+9.81 up when floating still) and angular velocity, IMU frame.
    top_cam  child prim of the boat, so it rides and tilts with the hull.
             Each env's camera renders only its own env plus the shared
             background (RTX per-env scene partitioning, on by default).
Set either to None to leave it out (physics-only runs skip the rendering cost).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from usv_core import ActuatorCfg, DampingCfg, HullCfg, HullGeometry, SeaCfg, ThrusterCfg

DRAFT, FREEBOARD = 0.60, 0.40
HULL = HullCfg(
    shape="wigley", length=6.10, beam=2.72, draft=DRAFT, freeboard=FREEBOARD,
    columns=(8, 4), com=(0.0, 0.0, (FREEBOARD - DRAFT) / 2),
)
HULL_MASS = HullGeometry(HULL).mass
DECK_Z = (HULL.draft + HULL.freeboard) / 2          # deck height above the CoM (box centre)

CAMERA_PITCH_DOWN = math.radians(10.0)


def pitch_down(angle: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) rotation that tilts a +X-looking ('world' convention) camera down by `angle`."""
    return (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))


@configclass
class UsvSceneCfg(InteractiveSceneCfg):
    boat: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Boat",
        spawn=sim_utils.CuboidCfg(
            size=(HULL.length, HULL.beam, HULL.draft + HULL.freeboard),
            rigid_props=[
                sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
                # usv_core does all the damping; no hidden engine damping, no sleeping
                PhysxRigidBodyCfg(linear_damping=0.0, angular_damping=0.0, sleep_threshold=0.0),
            ],
            mass_props=sim_utils.MassPropertiesCfg(mass=HULL_MASS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HULL.com[2])),
    )
    # Distant backdrop, shared by every camera. Each boat's own sea is drawn by water_visual.py
    # (when draw_water_envs > 0); this sits 3 m down so it never shows through wave troughs.
    backdrop = AssetBaseCfg(
        prim_path="/World/Backdrop",
        spawn=sim_utils.CuboidCfg(
            size=(4000.0, 4000.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.18, 0.3)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0)),
    )
    imu: ImuCfg | None = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Boat",
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, DECK_Z)),          # deck centre
    )
    top_cam: CameraCfg | None = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Boat/top_cam",
        update_period=1 / 30,                                        # once per policy step
        update_latest_camera_pose=True,                              # keep data.pos_w / quat_w current
        width=128,
        height=96,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, 200.0)),   # default lens: ~47 deg horizontal FOV
        offset=CameraCfg.OffsetCfg(
            pos=(1.0, 0.0, DECK_Z + 1.0),                            # 1 m forward of centre, 1 m above the deck
            rot=pitch_down(CAMERA_PITCH_DOWN),
            convention="world",                                      # +X forward, +Z up
        ),
    )
    # A dim sky and a low sun (30 degrees up): side light is what makes wave slopes visible.
    # A distant light shines along its local -Z; tilting it 60 degrees about Y gives a 30 degree sun.
    sky = AssetBaseCfg(prim_path="/World/Sky", spawn=sim_utils.DomeLightCfg(intensity=400.0))
    sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.0, math.sin(math.radians(30)), 0.0, math.cos(math.radians(30)))),
    )


@configclass
class UsvFloatEnvCfg(DirectRLEnvCfg):
    """16 boats floating in irregular seas. Actions: (surge, turn) in [-1, 1]. No task reward yet."""

    decimation = 4                       # physics 120 Hz, policy 30 Hz
    episode_length_s = 60.0
    action_space = 2
    observation_space = 11               # body lin vel 3, body ang vel 3, gravity in body 3, commands 2
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PhysxCfg())
    scene: UsvSceneCfg = UsvSceneCfg(num_envs=16, env_spacing=100.0, replicate_physics=True)

    hull: HullCfg = HULL
    sea: SeaCfg = SeaCfg(hs=(0.5, 2.0), tp=(5.0, 9.0))
    damping: DampingCfg = DampingCfg()
    thruster: ThrusterCfg = ThrusterCfg(max_thrust=1000.0)
    actuators: ActuatorCfg = ActuatorCfg()
    sea_seed: int = 0

    capsize_up_z = 0.2                   # reset when the hull's up axis has z below this (> ~78 degrees of tilt)
    recenter_radius = 10.0               # treadmill: move a boat back to its plot centre beyond this [m]
    reset_yaw = (-math.pi, math.pi)

    # Drawn water (visual only): the first `draw_water_envs` environments get their own sea mesh.
    # Patch + 2 x recenter_radius must stay below env_spacing so patches never overlap.
    draw_water_envs = 0
    water_patch_size = 80.0              # [m], centred on the boat
    water_resolution = 96                # vertices per side

    def __post_init__(self):
        # Overhead view of the grid (16 envs at 100 m spacing span about 300 m).
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(-230.0, -230.0, 170.0), lookat=(0.0, 0.0, 0.0))
