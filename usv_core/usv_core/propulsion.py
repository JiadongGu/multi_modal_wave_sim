"""Propulsion: the policy's actions become a push on the hull.

Two pieces:

Actuators      what the motors actually do. The policy asks for (surge, turn)
               in [-1, 1]; real throttles and steering legs do not jump, so each
               command moves toward the request with a first-order lag and an
               optional rate limit. One state row per environment; reset(env_ids)
               zeroes the boats that restarted. Kept here (not in the environment)
               so it is tested once, and so a model that wants propeller/rudder
               state (a ROM might) can read it.

TwinThruster   two outboards at the stern, after the extension's TwinScrewThruster.
               surge is throttle (-1 full astern .. +1 full ahead), turn is helm
               (+1 full to port). The legs swivel by turn * steer_max and push
               partly sideways, which turns the boat; a little differential thrust
               (port screw surge - d*turn, starboard surge + d*turn) lets it pivot
               with no way on, when swivelling alone gives no side force worth having.
               Torque is r x F with r from the CoM to each screw, so the boat also
               pitches bow-up under throttle: the screws sit below the CoM.
               A screw above the local water surface gives no thrust (ventilation).

Reads the commands from ctx.actions [N, 2] = (surge, turn), already passed
through Actuators by the environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .conventions import BodyState, Wrench, body_to_world
from .hull import HullGeometry
from .model import HullModel, ModelContext


@dataclass
class ActuatorCfg:
    """time_constant  seconds to reach 63 % of a step change, per command (surge, turn); 0 = instant
    rate_limit        max change per second, per command, in command units (1.0 = full scale in 1 s); None = none
    """

    time_constant: tuple[float, float] = (0.3, 0.5)
    rate_limit: tuple[float, float] | None = (2.0, 1.0)


class Actuators:
    """Per-environment command state, moving toward the policy's request each step."""

    def __init__(self, num_envs: int, cfg: ActuatorCfg | None = None, device="cpu", dtype=torch.float32):
        self.cfg = cfg or ActuatorCfg()
        self.state = torch.zeros(num_envs, 2, device=device, dtype=dtype)
        self._tau = torch.tensor(self.cfg.time_constant, device=device, dtype=dtype)
        self._rate = None if self.cfg.rate_limit is None else torch.tensor(self.cfg.rate_limit, device=device, dtype=dtype)

    def step(self, target: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance by dt toward target [N, 2] (clamped to [-1, 1]); returns the new state."""
        target = target.to(self.state).clamp(-1.0, 1.0)
        alpha = torch.where(self._tau > 0, 1.0 - torch.exp(-dt / self._tau.clamp_min(1e-9)), torch.ones_like(self._tau))
        delta = (target - self.state) * alpha
        if self._rate is not None:
            delta = torch.maximum(torch.minimum(delta, self._rate * dt), -self._rate * dt)
        self.state = self.state + delta
        return self.state

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.state.zero_()
        else:
            self.state[env_ids] = 0.0


@dataclass
class ThrusterCfg:
    """max_thrust  per screw [N]
    positions      ((x, y, z), (x, y, z)) port then starboard, hull frame (x forward from midship,
                   z up from the design waterline); None places them like the extension:
                   0.42 L aft, 0.22 B out, 0.55 T down
    steer_max      leg deflection at full helm [rad] (0.61 = 35 degrees, a typical outboard)
    differential   share of helm applied as thrust imbalance
    ventilate      a screw above the water gives no thrust
    """

    max_thrust: float = 1000.0
    positions: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    steer_max: float = 0.61
    differential: float = 0.3
    ventilate: bool = True


class TwinThruster(HullModel):
    def __init__(self, geometry: HullGeometry, cfg: ThrusterCfg | None = None):
        self.cfg = cfg or ThrusterCfg()
        c = geometry.cfg
        pos = self.cfg.positions or (
            (-0.42 * c.length, +0.22 * c.beam, -0.55 * c.draft),
            (-0.42 * c.length, -0.22 * c.beam, -0.55 * c.draft),
        )
        self.screws = torch.tensor(pos, dtype=torch.float64) - torch.tensor(c.com, dtype=torch.float64)  # [2, 3] from CoM

    def commands(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(per-screw command [N, 2] port/starboard in [-1, 1], steer angle [N]) from (surge, turn)."""
        surge, turn = actions[:, 0].clamp(-1, 1), actions[:, 1].clamp(-1, 1)
        d = self.cfg.differential
        cmd = torch.stack([surge - d * turn, surge + d * turn], -1)
        cmd = cmd / cmd.abs().amax(-1, keepdim=True).clamp_min(1.0)
        return cmd, turn * self.cfg.steer_max

    def wrench(self, state: BodyState, ctx: ModelContext) -> Wrench:
        dev, dtype, n = state.device, state.pos.dtype, state.num_envs
        if ctx.actions is None:
            return Wrench.zeros(n, dev, dtype)
        cmd, steer = self.commands(ctx.actions.to(dev, dtype))                         # [N, 2], [N]
        cache = self.__dict__.setdefault("_screw_cache", {})                            # moved to the device once
        if (dev, dtype) not in cache:
            cache[(dev, dtype)] = self.screws.to(dev, dtype)
        screws = cache[(dev, dtype)].expand(n, -1, -1)                                   # [N, 2, 3] body, from CoM

        if self.cfg.ventilate and ctx.sea is not None:
            world = state.pos[:, None, :] + body_to_world(state.quat, screws)
            cmd = torch.where(world[..., 2] < ctx.sea.height(world[..., :2], ctx.t), cmd, torch.zeros_like(cmd))

        # Axial part follows the signed command (astern is astern); the side part follows its magnitude,
        # since the leg points the same way whichever way the screw turns.
        T = self.cfg.max_thrust
        fx = T * cmd * torch.cos(steer)[:, None]
        fy = -T * cmd.abs() * torch.sin(steer)[:, None]
        force_b = torch.stack([fx, fy, torch.zeros_like(fx)], -1)                       # [N, 2, 3]
        torque_b = torch.cross(screws, force_b, dim=-1)
        return Wrench(body_to_world(state.quat, force_b.sum(1)), body_to_world(state.quat, torque_b.sum(1)))
