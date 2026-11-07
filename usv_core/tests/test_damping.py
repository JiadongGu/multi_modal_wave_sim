import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from usv_core import BodyState, Composite, ModelContext, SeaCfg, SineWaveField, body_to_world, world_to_body
from usv_core.damping import DampingCfg, HydroDamping
from usv_core.hull import ColumnBuoyancy, HullCfg, HullGeometry
from usv_core.waves import FlatWater

F64 = torch.float64
EXT = Path(__file__).resolve().parents[2] / "exts/mmws.wave_sim"


def yaw_quat(a):
    return torch.tensor([[0.0, 0.0, math.sin(a / 2), math.cos(a / 2)]], dtype=F64)


def roll_quat(a):
    return torch.tensor([[math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)]], dtype=F64)


def st(lin=(0, 0, 0), ang=(0, 0, 0), z=0.0, q=None, n=1):
    return BodyState(
        pos=torch.tensor([[0.0, 0.0, z]], dtype=F64).repeat(n, 1),
        quat=(yaw_quat(0.0) if q is None else q).repeat(n, 1),
        lin_vel=torch.tensor([lin], dtype=F64).repeat(n, 1),
        ang_vel=torch.tensor([ang], dtype=F64).repeat(n, 1),
    )


def make(shape="wigley", **cfg):
    geo = HullGeometry(HullCfg(shape=shape, **cfg), dtype=F64)
    return geo, HydroDamping(geo)


def still(dt=0.0):
    return ModelContext(sea=FlatWater(1, dtype=F64), dt=dt)


def test_coefficients_match_extension():
    sys.path.insert(0, str(EXT))
    try:
        from mmws.wave_sim.forces.damping import LinearQuadraticDamping
    finally:
        sys.path.remove(str(EXT))
    geo, d = make()
    c = geo.cfg
    ext = LinearQuadraticDamping(extents=(c.length, c.beam, c.draft + c.freeboard), density=c.density)
    np.testing.assert_allclose(d.quadratic.numpy(), ext.quadratic, rtol=1e-12)
    np.testing.assert_allclose(d.linear.numpy(), ext.linear, rtol=1e-12)


@pytest.mark.parametrize("axis", range(6))
def test_opposes_motion_on_every_axis(axis):
    """Velocity set along one BODY axis of a yawed boat; the damping must act on that axis only, against it."""
    _, d = make()
    q = yaw_quat(0.8)
    nu = torch.zeros(6, dtype=F64)
    nu[axis] = 1.3
    s = BodyState(torch.zeros(1, 3, dtype=F64), q, body_to_world(q, nu[None, :3]), body_to_world(q, nu[None, 3:]))
    w = d(s, still())
    tau = torch.cat([world_to_body(q, w.force), world_to_body(q, w.torque)], -1)[0]
    assert tau[axis] * nu[axis] < 0
    others = [i for i in range(6) if i != axis]
    assert torch.allclose(tau[others], torch.zeros(5, dtype=F64), atol=1e-9)


def test_moving_with_the_water_is_not_damped():
    geo, d = make()
    sea = SineWaveField(SeaCfg(hs=(2.0, 2.0)), num_envs=1, dtype=F64, seed=5)
    probe = torch.tensor([[[0.0, 0.0, -0.5 * geo.cfg.draft]]], dtype=F64)
    surface = sea.height(torch.zeros(1, 1, 2, dtype=F64), 4.0)[0, 0].item()   # float the boat on the local surface
    probe = probe + torch.tensor([0.0, 0.0, surface], dtype=F64)
    water = sea.velocity(probe, 4.0)[0]
    pos = torch.tensor([[0.0, 0.0, surface]], dtype=F64)
    w = d(BodyState(pos, yaw_quat(0.0), water, torch.zeros(1, 3, dtype=F64)), ModelContext(sea=sea, t=4.0))
    assert torch.allclose(w.force, torch.zeros(1, 3, dtype=F64), atol=1e-9)
    still_boat = d(BodyState(pos, yaw_quat(0.0), torch.zeros(1, 3, dtype=F64), torch.zeros(1, 3, dtype=F64)),
                   ModelContext(sea=sea, t=4.0))
    assert still_boat.force.abs().max() > 1.0  # a boat at rest in moving water does feel drag


@pytest.mark.parametrize("axis", range(6))
def test_stable_at_absurd_speed(axis):
    """100 m/s (or rad/s) at 60 Hz: one step of v += tau/m dt must not reverse or grow the velocity."""
    geo, d = make()
    dt = 1 / 60
    v = [0.0] * 6
    v[axis] = 100.0
    w = d(st(lin=v[:3], ang=v[3:]), still(dt))
    tau = torch.cat([w.force, w.torque], -1)[0, axis]
    m = d.generalised_mass[axis]
    new = 100.0 + tau / m * dt
    assert 0.0 <= new < 100.0
    naive = 100.0 - (d.linear[axis] + d.quadratic[axis] * 100.0) * 100.0 / m * dt
    if axis in (1, 2, 4, 5):           # the bluff axes: the naive form flips the velocity
        assert naive < 0


def test_small_dt_approaches_plain_formula():
    _, d = make()
    nu = [0.7, -0.3, 0.2, 0.1, -0.05, 0.3]
    w = d(st(lin=nu[:3], ang=nu[3:]), still(dt=1e-9))
    v = torch.tensor(nu, dtype=F64)
    plain = -(d.linear + d.quadratic * v.abs()) * v
    assert torch.allclose(torch.cat([w.force, w.torque], -1)[0], plain, rtol=1e-6)


def test_wetness_scales_damping():
    geo, d = make(shape="box", length=4.0, beam=2.0, draft=0.5, freeboard=0.5)
    full = d(st(lin=(0, 0, -1.0)), still()).force[0, 2]
    half = d(st(lin=(0, 0, -1.0), z=0.25), still()).force[0, 2]
    flying = d(st(lin=(0, 0, -1.0), z=2.0), still()).force[0, 2]
    assert math.isclose(half.item(), 0.5 * full.item(), rel_tol=1e-9)
    assert flying.item() == 0.0


def simulate(model, geo, z0=0.0, roll0=0.0, seconds=25.0, dt=1 / 120):
    """Heave and roll only, semi-implicit Euler, gravity included. Returns final (z, w, roll, roll_rate)."""
    z, w, phi, p = z0, 0.0, roll0, 0.0
    m, Ixx = geo.mass, geo.inertia()[0]
    for _ in range(int(seconds / dt)):
        s = BodyState(torch.tensor([[0.0, 0.0, z]], dtype=F64), roll_quat(phi),
                      torch.tensor([[0.0, 0.0, w]], dtype=F64), torch.tensor([[p, 0.0, 0.0]], dtype=F64))
        wr = model(s, ModelContext(sea=FlatWater(1, dtype=F64), dt=dt))
        w += (wr.force[0, 2].item() / m - geo.cfg.gravity) * dt
        p += wr.torque[0, 0].item() / Ixx * dt
        z += w * dt
        phi += p * dt
    return z, w, phi, p


def test_buoyancy_plus_damping_settles_at_design_draft():
    """Dropped 0.3 m and rolled 10 degrees: settles to the design draft and upright.

    Slow at the end: below ~0.1 m/s only the small linear term damps and there is no radiation
    damping yet, so after 25 s a few mm/s of heave remain."""
    geo = HullGeometry(HullCfg(), dtype=F64)
    model = Composite({"buoyancy": ColumnBuoyancy(geo), "damping": HydroDamping(geo)})
    z, w, phi, p = simulate(model, geo, z0=0.3, roll0=math.radians(10))
    assert abs(z) < 3e-3 and abs(w) < 3e-2, (z, w)
    # roll is lightly damped, as on real ships: 10 degrees has decayed below 0.5 degrees
    assert abs(phi) < math.radians(0.5) and abs(p) < 3e-2, (math.degrees(phi), p)


def test_without_damping_it_keeps_bouncing():
    geo = HullGeometry(HullCfg(), dtype=F64)
    z, w, *_ = simulate(ColumnBuoyancy(geo), geo, z0=0.3, seconds=25.0)
    assert abs(z) > 0.05 or abs(w) > 0.2
