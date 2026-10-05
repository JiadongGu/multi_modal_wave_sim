"""Flat water - the field that exists to prove the plumbing works.

Zero everywhere, at all times. Useless as water, but it makes every other part
of the system testable in isolation: if a hull floats correctly on flat water
at exactly the right draft, then the force model is right and any remaining
problem is in the waves. Debugging buoyancy and waves at the same time is how
you end up chasing a hovering boat for an afternoon.

Keep this. It is also the fastest sanity check that the extension loaded: swap
to `flat`, press play, and the vessel should sit still at its design waterline
forever.
"""

from __future__ import annotations

import numpy as np

from ..registry import FIELDS
from .base import WaveField


@FIELDS.register("flat")
class FlatWaterField(WaveField):
    """A motionless plane at z = 0."""

    def advance(self, t: float) -> None:
        self.time = float(t)

    def height(self, x, y):
        return np.zeros_like(np.asarray(x, dtype=float))

    def normal(self, x, y, eps: float | None = None):
        # Short-circuit the finite-difference default: exact, and it avoids
        # four pointless height queries per sample.
        shape = np.asarray(x, dtype=float).shape
        n = np.zeros(shape + (3,))
        n[..., 2] = 1.0
        return n
