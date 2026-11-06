"""The contract every other module in usv_core follows.

Batches
-------
Every tensor has a leading dimension N, one row per environment. A single
boat's position is [3]; all boats' positions are [N, 3].

Units
-----
SI only: m, s, kg, N, rad, rad/s. No degrees anywhere. USD reports angular
velocity in deg/s; the environment converts at its boundary, once.

Frames
------
world      z up, x/y horizontal. Isaac Sim's frame.
env-local  the world frame shifted to this environment's origin. Isaac Lab
           reports positions in the world frame, which includes each
           environment's grid offset; the environment subtracts
           `scene.env_origins` before calling usv_core. Everything that
           depends on position (wave sampling, waypoints, observations) must
           use env-local positions, or a boat's grid slot leaks into the
           physics and the policy.
body       attached to the hull at its centre of mass: x forward (bow),
           y left (port), z up. FLU, the ROS/Isaac convention.

Fossen's textbook body frame is FRD (forward, right, down). A coefficient
ported from Fossen flips sign on its y and z components. Orientation and
velocity are the same in world and env-local frames; only positions shift.

Quaternions
-----------
(x, y, z, w), Isaac Lab 3.0's order. USD and the mmws.wave_sim extension use
(w, x, y, z). Reading one as the other gives a wrong rotation and no error,
so the conversions below are the only place the order changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) -> (x, y, z, w). For quaternions coming from USD or the extension."""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """(x, y, z, w) -> (w, x, y, z). For quaternions going to USD or the extension."""
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(x, y, z, w) quaternions [..., 4] -> rotation matrices [..., 3, 3].

    R maps body-frame vectors to world-frame vectors: v_world = R @ v_body.
    Normalises as it goes, so a slightly stale quaternion from the physics
    engine does not scale every force. A zero quaternion gives the identity.
    """
    x, y, z, w = q.unbind(-1)
    n = x * x + y * y + z * z + w * w
    s = torch.where(n > 1e-12, 2.0 / n.clamp_min(1e-12), torch.zeros_like(n))
    R = torch.stack(
        [
            1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y),
            s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x),
            s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))


def body_to_world(q: torch.Tensor, v_body: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vectors into the world frame.

    q [N, 4]; v_body [N, 3] (one vector per boat) or [N, M, 3] (M per boat).
    """
    R = quat_to_matrix(q)
    if v_body.dim() == R.dim() - 1:
        return (R @ v_body.unsqueeze(-1)).squeeze(-1)
    return v_body @ R.transpose(-1, -2)


def world_to_body(q: torch.Tensor, v_world: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the body frame (the inverse of body_to_world)."""
    R = quat_to_matrix(q)
    if v_world.dim() == R.dim() - 1:
        return (R.transpose(-1, -2) @ v_world.unsqueeze(-1)).squeeze(-1)
    return v_world @ R


@dataclass
class BodyState:
    """Everything a force model may know about the boats. One row per environment.

    pos      [N, 3]  centre of mass, env-local frame
    quat     [N, 4]  orientation, (x, y, z, w)
    lin_vel  [N, 3]  centre-of-mass velocity, world frame
    ang_vel  [N, 3]  angular velocity, world frame, rad/s
    """

    pos: torch.Tensor
    quat: torch.Tensor
    lin_vel: torch.Tensor
    ang_vel: torch.Tensor

    @property
    def num_envs(self) -> int:
        return self.pos.shape[0]

    @property
    def device(self) -> torch.device:
        return self.pos.device

    @property
    def lin_vel_body(self) -> torch.Tensor:
        """Velocity as the boat feels it: x forward, y left, z up."""
        return world_to_body(self.quat, self.lin_vel)

    @property
    def ang_vel_body(self) -> torch.Tensor:
        """Roll, pitch and yaw rates about the boat's own axes."""
        return world_to_body(self.quat, self.ang_vel)


@dataclass
class Wrench:
    """Force and torque on each boat, world frame, torque about the centre of mass.

    force [N, 3], torque [N, 3]. World frame because buoyancy points up in the
    world however the hull is tilted; the environment converts to whatever
    frame Isaac Lab expects, in one place.
    """

    force: torch.Tensor
    torque: torch.Tensor

    @classmethod
    def zeros(cls, num_envs: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> "Wrench":
        z = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        return cls(z, z.clone())

    def __add__(self, other: "Wrench") -> "Wrench":
        return Wrench(self.force + other.force, self.torque + other.torque)
