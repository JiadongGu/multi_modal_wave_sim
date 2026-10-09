"""The force model contract.

A force model takes a rigid body's state and a wave field, and returns a
wrench: three forces and three moments, in the body frame.

Why the contract is a full wrench
---------------------------------
ASTRA's buoyancy computed a submerged volume, turned it into a scalar force
magnitude, and applied it along world +Z only::

    body_force = R^T . (0, 0, rho g V)

That is Archimedes for a boat sitting in still water, and it is wrong the
moment the water is not flat: a wave face is a tilted pressure surface, so the
true force has a horizontal component that pushes the vessel along and across
the wave. Without it there is no surf-riding, no broaching, no wave-induced
sway - the vessel bobs but is never pushed.

So the contract here is the complete six-component wrench. Horizontal force is
the default rather than a feature to be added later, and a model that genuinely
has no horizontal component has to return zeros for it on purpose.

Why the field is passed in, not the heights
-------------------------------------------
A force model gets the whole `WaveField`, so it can ask for whatever it needs -
heights, normals, horizontal displacement, the surface velocity - at whatever
points its own method wants. Passing a precomputed array of heights would lock
every model into one sampling pattern, and a pressure-integral model samples
quite differently from a column model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field

import numpy as np


def quat_to_matrix(q):
    """(w, x, y, z) quaternion -> 3x3 rotation matrix.

    USD's Gf.Quatd is (w, x, y, z), so matching that order here avoids a whole
    class of silent bug at the USD boundary. Normalises as it goes, so a
    slightly stale quaternion from the physics engine does not quietly scale
    every force.
    """
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=float).ravel()[:4])
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


@dataclass
class Wrench:
    """Force and moment in the body frame. SNAME ordering: X Y Z K M N."""

    force: np.ndarray = dataclass_field(
        default_factory=lambda: np.zeros(3)
    )   # [N]  surge, sway, heave
    moment: np.ndarray = dataclass_field(
        default_factory=lambda: np.zeros(3)
    )   # [N m] roll, pitch, yaw

    def as_array(self) -> np.ndarray:
        """The 6-vector tau, which is what gets written to the physics engine."""
        return np.concatenate([self.force, self.moment])

    def __add__(self, other: "Wrench") -> "Wrench":
        return Wrench(self.force + other.force, self.moment + other.moment)


@dataclass
class BodyState:
    """Everything a force model is allowed to know about the vessel.

    Deliberately a plain dataclass with no USD or PhysX types in it. That keeps
    every force model testable without Isaac Sim running - you construct a
    BodyState by hand, call the model, and check the numbers. Converting from
    USD prims to this struct is the controller's job, not the model's.
    """

    position: np.ndarray = dataclass_field(default_factory=lambda: np.zeros(3))
    # Orientation as a quaternion in (w, x, y, z) order. USD uses that order
    # for Gf.Quatd, so matching it avoids a whole class of silent bug.
    orientation: np.ndarray = dataclass_field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )
    linear_velocity: np.ndarray = dataclass_field(default_factory=lambda: np.zeros(3))
    # Body-frame angular velocity in rad/s. Note USD reports degrees per second;
    # convert at the boundary, once, and never again.
    angular_velocity: np.ndarray = dataclass_field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia: np.ndarray = dataclass_field(default_factory=lambda: np.ones(3))
    center_of_mass: np.ndarray = dataclass_field(default_factory=lambda: np.zeros(3))


class ForceModel(ABC):
    """Base class for all hydrodynamic force models."""

    name: str = "<unregistered>"

    @abstractmethod
    def wrench(self, body: BodyState, wave_field, t: float) -> Wrench:
        """Forces and moments on `body` from `wave_field` at time `t`.

        Body frame, about the body's centre of mass. Must not mutate `body` or
        `wave_field`: a model that caches state across calls cannot be used in
        a parallel Isaac Lab environment, which is where this is all going.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
