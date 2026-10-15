"""Twin screws: a known thrust, at a known point, in the body frame.

Why a force model and not a special case in the extension
---------------------------------------------------------
Thrust is a wrench on the hull, the same as buoyancy and damping, so it goes
through the same interface and into the same composite. That also means the
run recorder captures it, the arrows could draw it, and swapping it out for a
real propeller curve later changes nothing else.

Why TWO screws and not one force on the centreline
--------------------------------------------------
Because the moments are the interesting part and a single centreline force
has none of them. Two screws at known positions give you:

  * surge, from the sum
  * yaw, from the difference across the beam - which is how a twin-screw
    vessel actually turns, with no rudder in the model at all
  * a pitch moment, because the screws are BELOW the centre of gravity, so
    opening the throttle pushes the bow down. A centreline point force
    through G would silently lose that.

Each screw is a simple commanded fraction of a fixed thrust, not a propeller
curve - no advance ratio, no wake fraction, no thrust deduction. That is the
point: the force is a number you set, so when the hull moves you know exactly
what moved it. Replace it with a real curve when there is something to
validate the curve against.
"""

from __future__ import annotations

import numpy as np

from ..registry import FORCES
from .base import BodyState, ForceModel, Wrench


@FORCES.register("thruster")
class TwinScrewThruster(ForceModel):
    """Two screws, each pushing along body +x at its own commanded fraction."""

    def __init__(self, thrust: float = 1000.0, positions=None,
                 length: float = 1.0, beam: float = 1.0, draft: float = 0.3,
                 ventilate: bool = True) -> None:
        if thrust < 0.0:
            raise ValueError(f"thrust must be non-negative, got {thrust}")
        self.thrust = float(thrust)
        self.ventilate = bool(ventilate)
        if positions is None:
            # Aft, outboard, and below the waterline - where screws go. In the
            # body frame, which for these hulls is centred on the hull's own
            # volume centroid, so -0.42 L is about a tenth of the length from
            # the transom.
            x = -0.42 * float(length)
            y = 0.22 * float(beam)
            z = -0.55 * float(draft)
            positions = [(x, +y, z), (x, -y, z)]
        self.positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        # One commanded fraction per screw, -1 (full astern) to +1 (full
        # ahead). Set by whatever is driving: the keyboard, a controller, a
        # ROS topic later.
        self.command = np.zeros(len(self.positions))

    def set_drive(self, surge: float = 0.0, turn: float = 0.0) -> None:
        """Mix surge and turn into per-screw commands.

        `turn` is positive to port, which is positive yaw about +z. The port
        screw is the one at +y, so turning to port means easing it and pushing
        the starboard one - the moment of a force along +x applied at +y is
        about -z, hence the sign.
        """
        surge = float(np.clip(surge, -1.0, 1.0))
        turn = float(np.clip(turn, -1.0, 1.0))
        port, starboard = surge - turn, surge + turn
        biggest = max(abs(port), abs(starboard), 1.0)
        self.command = np.array([port, starboard]) / biggest

    def wrench(self, body: BodyState, wave_field, t: float) -> Wrench:
        from .base import quat_to_matrix

        if not np.any(self.command):
            return Wrench()

        command = self.command
        if self.ventilate and wave_field is not None:
            # A propeller out of the water does not push. Cheap, and it stops
            # a hull driving itself along the tops of the waves.
            R = quat_to_matrix(np.asarray(body.orientation, dtype=float))
            world = np.asarray(body.position, dtype=float) + self.positions @ R.T
            surface = np.asarray(wave_field.height(world[:, 0], world[:, 1]))
            command = np.where(world[:, 2] < surface, command, 0.0)
            if not np.any(command):
                return Wrench()

        force = np.zeros((len(self.positions), 3))
        force[:, 0] = self.thrust * command          # body +x is forward
        arm = self.positions - np.asarray(body.center_of_mass, dtype=float)
        moment = np.cross(arm, force)
        return Wrench(force.sum(axis=0), moment.sum(axis=0))
