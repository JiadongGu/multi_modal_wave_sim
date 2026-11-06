"""The socket every force model plugs into.

The environment only ever calls `model.wrench(state, ctx)` and applies the
result, so buoyancy, damping, thrusters and a future ROM are interchangeable,
and "replace Fossen" versus "supplement it" is a choice of what goes in a
`Composite`, not a code change.

    state  BodyState (conventions.py): where the boats are and how they move.
    ctx    ModelContext: everything else a model might want. New inputs are
           added here as optional fields, so adding one breaks no existing model.

A model that keeps memory between steps (a history window, a recurrent net, a
radiation state) must clear it per environment in `reset(env_ids)`, or a new
episode starts with the previous boat's memory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from .conventions import BodyState, Wrench


@dataclass
class ModelContext:
    """Inputs beyond the body state. Optional fields are None when not provided.

    t        simulation time [s]
    dt       physics step [s]
    sea      the wave field (waves.py); models sample what they need from it
    actions  [N, A] actuator commands, already scaled and limited by the environment
    lin_acc  [N, 3] world-frame linear acceleration from the previous step
    ang_acc  [N, 3] world-frame angular acceleration from the previous step
    extras   anything else, by name, for experiments that should not touch this class
    """

    t: float = 0.0
    dt: float = 0.0
    sea: Any = None
    actions: torch.Tensor | None = None
    lin_acc: torch.Tensor | None = None
    ang_acc: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class HullModel(ABC):
    """A force model: body state and context in, world-frame wrench about the centre of mass out."""

    @abstractmethod
    def wrench(self, state: BodyState, ctx: ModelContext) -> Wrench:
        ...

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear per-environment memory. None means all environments. Stateless models ignore it."""

    def __call__(self, state: BodyState, ctx: ModelContext) -> Wrench:
        return self.wrench(state, ctx)


class ZeroModel(HullModel):
    """Contributes nothing. A placeholder and a baseline for benchmarks."""

    def wrench(self, state: BodyState, ctx: ModelContext) -> Wrench:
        return Wrench.zeros(state.num_envs, state.device, state.pos.dtype)


class Composite(HullModel):
    """Sum of several models, e.g. buoyancy + damping + thrusters (+ a ROM).

    Accepts a list, or a dict of name -> model so `parts()` can report each
    contribution by name for debugging and plotting.
    """

    def __init__(self, models: list[HullModel] | dict[str, HullModel]) -> None:
        if isinstance(models, dict):
            self.models = dict(models)
        else:
            self.models = {f"{type(m).__name__}_{i}": m for i, m in enumerate(models)}

    def parts(self, state: BodyState, ctx: ModelContext) -> dict[str, Wrench]:
        return {name: m.wrench(state, ctx) for name, m in self.models.items()}

    def wrench(self, state: BodyState, ctx: ModelContext) -> Wrench:
        total = Wrench.zeros(state.num_envs, state.device, state.pos.dtype)
        for m in self.models.values():
            total = total + m.wrench(state, ctx)
        return total

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        for m in self.models.values():
            m.reset(env_ids)
