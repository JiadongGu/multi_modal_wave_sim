"""Several force models acting on one hull.

A vessel is never one force. Buoyancy, damping, thrusters and whatever comes
later all act at once, and the physics engine wants one wrench. This sums
them, so the extension holds a single force model however many are active and
nothing downstream has to know the difference.
"""

from __future__ import annotations

from ..registry import FORCES
from .base import BodyState, ForceModel, Wrench


@FORCES.register("composite")
class CompositeForceModel(ForceModel):
    """Sum of several force models, evaluated in order."""

    def __init__(self, models=None) -> None:
        self.models = list(models or [])
        for m in self.models:
            if not isinstance(m, ForceModel):
                raise TypeError(f"not a ForceModel: {m!r}")

    def wrench(self, body: BodyState, wave_field, t: float) -> Wrench:
        total = Wrench()
        for model in self.models:
            total = total + model.wrench(body, wave_field, t)
        return total

    def __repr__(self) -> str:
        return f"CompositeForceModel({[type(m).__name__ for m in self.models]})"
