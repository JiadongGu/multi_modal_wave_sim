"""Wrap a model written in someone else's conventions so it fits the socket.

The model inside is used exactly as its author wrote it. The adapter builds
its inputs from our BodyState/ModelContext and turns its output back into our
standard Wrench (world frame, torque about the centre of mass).

Generic pieces here are usable on their own:

    flu_to_frd / frd_to_flu   Fossen's body frame (forward, right, down) vs ours
                              (forward, left, up): flip y and z. Applies to
                              vectors and to pseudo-vectors (torque, angular
                              velocity) alike, because the flip is a proper
                              rotation (180 degrees about x).
    split6 / join6            [N, 6] tau = (X, Y, Z, K, M, N) <-> force, torque
    torque_about_com          move a torque from the point it was computed
                              about to the centre of mass
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

from .conventions import BodyState, Wrench, body_to_world
from .model import HullModel, ModelContext

_FLIP = (1.0, -1.0, -1.0)


def flu_to_frd(v: torch.Tensor) -> torch.Tensor:
    """Body vectors [..., 3] from FLU (ours) to FRD (Fossen). Its own inverse."""
    return v * v.new_tensor(_FLIP)


frd_to_flu = flu_to_frd


def split6(tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, 6] (X, Y, Z, K, M, N) -> force [N, 3], torque [N, 3]."""
    return tau[..., 0:3], tau[..., 3:6]


def join6(force: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
    return torch.cat([force, torque], dim=-1)


def torque_about_com(torque_at_p: torch.Tensor, force: torch.Tensor, p_minus_com: torch.Tensor) -> torch.Tensor:
    """Torque about the centre of mass, given the torque about point P.

    tau_com = tau_p + (P - com) x F. All three in the same frame.
    """
    return torque_at_p + torch.cross(p_minus_com, force, dim=-1)


def body_velocity_inputs(state: BodyState, ctx: ModelContext, convention: Literal["FLU", "FRD"] = "FLU") -> torch.Tensor:
    """[N, 6] (u, v, w, p, q, r): body-frame linear then angular velocity. The usual Fossen nu."""
    v, w = state.lin_vel_body, state.ang_vel_body
    if convention == "FRD":
        v, w = flu_to_frd(v), flu_to_frd(w)
    return torch.cat([v, w], dim=-1)


class ConventionAdapter(HullModel):
    """Makes `model` a HullModel.

    model          callable taking whatever `build_inputs` returns, returning either
                   a [N, 6] tau or a (force, torque) pair
    build_inputs   (state, ctx) -> the model's input, in the model's conventions
    output_frame   "body" or "world": the frame the model's force/torque are in
    body_convention  "FLU" or "FRD": only matters when output_frame is "body"
    torque_point   [3] body-frame (FLU) point, relative to the centre of mass,
                   that the model's torque is taken about; None means the CoM
    reset_fn       called with env_ids on reset, for models with memory
    """

    def __init__(
        self,
        model: Callable,
        build_inputs: Callable[[BodyState, ModelContext], object],
        output_frame: Literal["body", "world"] = "body",
        body_convention: Literal["FLU", "FRD"] = "FLU",
        torque_point: torch.Tensor | None = None,
        reset_fn: Callable[[torch.Tensor | None], None] | None = None,
    ) -> None:
        if output_frame not in ("body", "world"):
            raise ValueError(f"output_frame must be 'body' or 'world', got {output_frame!r}")
        if body_convention not in ("FLU", "FRD"):
            raise ValueError(f"body_convention must be 'FLU' or 'FRD', got {body_convention!r}")
        self.model = model
        self.build_inputs = build_inputs
        self.output_frame = output_frame
        self.body_convention = body_convention
        self.torque_point = torque_point
        self.reset_fn = reset_fn

    def wrench(self, state: BodyState, ctx: ModelContext) -> Wrench:
        out = self.model(self.build_inputs(state, ctx))
        force, torque = split6(out) if isinstance(out, torch.Tensor) else out

        if self.output_frame == "body":
            if self.body_convention == "FRD":
                force, torque = frd_to_flu(force), frd_to_flu(torque)
            if self.torque_point is not None:
                p = self.torque_point.to(force).expand_as(force)
                torque = torque_about_com(torque, force, p)
            force, torque = body_to_world(state.quat, force), body_to_world(state.quat, torque)
        elif self.torque_point is not None:
            p = body_to_world(state.quat, self.torque_point.to(force).expand_as(force))
            torque = torque_about_com(torque, force, p)

        return Wrench(force, torque)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self.reset_fn is not None:
            self.reset_fn(env_ids)
