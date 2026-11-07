"""Configuration for the floating-boat environment.

The hull is described once (HullCfg) and both sides read it: usv_core for the
buoyancy columns, damping and thrusters, and Isaac Lab for the rigid body's
mass and visual size. Mass comes from HullGeometry (it floats at its design
draft), so the two cannot disagree.

For now the rigid body is drawn as a box the size of the hull's bounding box.
Its centre is the centre of mass, so HullCfg.com is set to the box centre.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
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
    # Visual only: a translucent plane at the mean water level, so motion is visible.
    # The physics sea is usv_core's; nothing draws its waves yet.
    sea_level = AssetBaseCfg(
        prim_path="/World/SeaLevel",
        spawn=sim_utils.CuboidCfg(
            size=(600.0, 600.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.3, 0.5), opacity=0.55),
        ),
    )
    sky = AssetBaseCfg(prim_path="/World/Sky", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    sun = AssetBaseCfg(prim_path="/World/Sun", spawn=sim_utils.DistantLightCfg(intensity=2500.0))


@configclass
class UsvFloatEnvCfg(DirectRLEnvCfg):
    """16 boats floating in irregular seas. Actions: (surge, turn) in [-1, 1]. No task reward yet."""

    decimation = 4                       # physics 120 Hz, policy 30 Hz
    episode_length_s = 60.0
    action_space = 2
    observation_space = 11               # body lin vel 3, body ang vel 3, gravity in body 3, commands 2
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PhysxCfg())
    scene: UsvSceneCfg = UsvSceneCfg(num_envs=16, env_spacing=30.0, replicate_physics=True)

    hull: HullCfg = HULL
    sea: SeaCfg = SeaCfg(hs=(0.5, 2.0), tp=(5.0, 9.0))
    damping: DampingCfg = DampingCfg()
    thruster: ThrusterCfg = ThrusterCfg(max_thrust=1000.0)
    actuators: ActuatorCfg = ActuatorCfg()
    sea_seed: int = 0

    capsize_up_z = 0.2                   # reset when the hull's up axis has z below this (> ~78 degrees of tilt)
    max_distance = 100.0                 # reset when a boat drifts this far from its env origin [m]
    reset_yaw = (-math.pi, math.pi)

    def __post_init__(self):
        # Overhead view of the whole grid (16 envs at 30 m spacing span about 90 m).
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(-70.0, -70.0, 55.0), lookat=(0.0, 0.0, 0.0))
