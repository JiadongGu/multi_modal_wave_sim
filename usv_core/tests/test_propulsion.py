import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from usv_core import BodyState, Composite, ModelContext, world_to_body
from usv_core.damping import HydroDamping
from usv_core.hull import ColumnBuoyancy, HullCfg, HullGeometry
from usv_core.propulsion import ActuatorCfg, Actuators, ThrusterCfg, TwinThruster
from usv_core.waves import FlatWater

F64 = torch.float64
EXT = Path(__file__).resolve().parents[2] / "exts/mmws.wave_sim"


def upright(n=1, z=0.0, roll=0.0, vx=0.0):
    return BodyState(
        pos=torch.tensor([[0.0, 0.0, z]], dtype=F64).repeat(n, 1),
        quat=torch.tensor([[math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2)]], dtype=F64).repeat(n, 1),
        lin_vel=torch.tensor([[vx, 0.0, 0.0]], dtype=F64).repeat(n, 1),
        ang_vel=torch.zeros(n, 3, dtype=F64),
    )


def thruster(**cfg):
    geo = HullGeometry(HullCfg(), dtype=F64)
    return geo, TwinThruster(geo, ThrusterCfg(**cfg))


def act(surge, turn, n=1):
    return ModelContext(sea=FlatWater(n, dtype=F64), actions=torch.tensor([[surge, turn]], dtype=F64).repeat(n, 1))


def test_full_ahead_pushes_forward_and_pitches_bow_up():
    _, t = thruster()
    w = t(upright(), act(1.0, 0.0))
    assert torch.allclose(w.force, torch.tensor([[2000.0, 0.0, 0.0]], dtype=F64))
    assert abs(w.torque[0, 0]) < 1e-9 and abs(w.torque[0, 2]) < 1e-9   # no roll, no yaw
    assert w.torque[0, 1] < 0          # screws below the CoM: bow-up pitch (negative about +y in FLU)


def test_full_astern_pushes_backward():
    _, t = thruster()
    assert t(upright(), act(-1.0, 0.0)).force[0, 0] < 0


def test_helm_turns_the_boat_the_right_way():
    _, t = thruster()
    port = t(upright(), act(0.5, +1.0)).torque[0, 2]
    starboard = t(upright(), act(0.5, -1.0)).torque[0, 2]
    assert port > 0 and starboard < 0 and math.isclose(port.item(), -starboard.item(), rel_tol=1e-9)


def test_pivot_with_no_way_on_comes_from_differential_thrust():
    _, t = thruster()
    assert t(upright(), act(0.0, 1.0)).torque[0, 2] > 0
    _, no_diff = thruster(differential=0.0)
    assert abs(no_diff(upright(), act(0.0, 1.0)).torque[0, 2]) < 1e-9


@pytest.mark.parametrize("surge,turn", [(1.0, 0.0), (0.6, 0.4), (-0.7, -0.9), (0.0, 1.0), (0.3, -0.2)])
def test_matches_extension_twin_screw(surge, turn):
    sys.path.insert(0, str(EXT))
    try:
        from mmws.wave_sim.forces.base import BodyState as ExtBody
        from mmws.wave_sim.forces.thruster import TwinScrewThruster
    finally:
        sys.path.remove(str(EXT))
    geo, ours = thruster()
    c = geo.cfg
    ext = TwinScrewThruster(thrust=1000.0, length=c.length, beam=c.beam, draft=c.draft, ventilate=False)
    ext.set_drive(surge, turn)
    w = ext.wrench(ExtBody(), None, 0.0)
    got = ours(upright(), act(surge, turn))
    np.testing.assert_allclose(got.force[0].numpy(), w.force, atol=1e-9)
    np.testing.assert_allclose(got.torque[0].numpy(), w.moment, atol=1e-9)


def test_ventilation():
    _, t = thruster()
    assert torch.equal(t(upright(z=3.0), act(1.0, 0.0)).force, torch.zeros(1, 3, dtype=F64))
    # roll hard to port: the starboard screw lifts out, only the port one pushes -> yaw toward starboard
    w = t(upright(roll=math.radians(-40)), act(1.0, 0.0))
    assert 0 < w.force[0, 0] < 1500 and w.torque[0, 2] < 0


def test_no_actions_no_thrust():
    _, t = thruster()
    assert torch.equal(t(upright(), ModelContext()).force, torch.zeros(1, 3, dtype=F64))


def test_actuator_lag_rate_limit_and_reset():
    a = Actuators(3, ActuatorCfg(time_constant=(0.3, 0.5), rate_limit=None), dtype=F64)
    target = torch.tensor([[1.0, 1.0]] * 3, dtype=F64)
    dt = 0.001
    for _ in range(300):                       # one surge time constant
        a.step(target, dt)
    assert math.isclose(a.state[0, 0].item(), 1 - math.exp(-1), rel_tol=1e-3)
    assert a.state[0, 1].item() < a.state[0, 0].item()    # slower steering

    r = Actuators(1, ActuatorCfg(time_constant=(0.0, 0.0), rate_limit=(2.0, 1.0)), dtype=F64)
    r.step(torch.tensor([[1.0, 1.0]], dtype=F64), 0.1)
    assert torch.allclose(r.state, torch.tensor([[0.2, 0.1]], dtype=F64))   # rate * dt, not the full jump

    a.reset(torch.tensor([1]))
    assert torch.equal(a.state[1], torch.zeros(2, dtype=F64)) and a.state[0, 0] > 0
    a.step(torch.tensor([[5.0, -5.0]] * 3, dtype=F64), 10.0)
    assert a.state.abs().max() <= 1.0          # requests clamped to [-1, 1]


def test_top_speed_where_thrust_equals_drag():
    """Full ahead in still water from rest: surge settles where 2 T = D_lin u + D_quad u^2."""
    geo = HullGeometry(HullCfg(), dtype=F64)
    damp = HydroDamping(geo)
    model = Composite({"buoyancy": ColumnBuoyancy(geo), "damping": damp, "thrust": TwinThruster(geo)})
    a, b, T2 = damp.linear[0].item(), damp.quadratic[0].item(), 2000.0
    expected = (-a + math.sqrt(a * a + 4 * b * T2)) / (2 * b)
    u, dt = 0.0, 1 / 120
    for _ in range(int(60 / dt)):
        s = upright(vx=u)
        w = model(s, ModelContext(sea=FlatWater(1, dtype=F64), dt=dt, actions=torch.tensor([[1.0, 0.0]], dtype=F64)))
        u += w.force[0, 0].item() / geo.mass * dt
    assert math.isclose(u, expected, rel_tol=0.01), (u, expected)
