import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from usv_core import BodyState, ModelContext, SeaCfg, SineWaveField, quat_to_matrix, xyzw_to_wxyz
from usv_core.hull import ColumnBuoyancy, HullCfg, HullGeometry
from usv_core.waves import FlatWater

F64 = torch.float64
EXT = Path(__file__).resolve().parents[2] / "exts/mmws.wave_sim"


def quat(roll=0.0, pitch=0.0, yaw=0.0):
    """(x, y, z, w) from roll about x, then pitch about y, then yaw about z (applied in that order)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=F64)


def state(z=0.0, roll=0.0, pitch=0.0, yaw=0.0, x=0.0, y=0.0):
    return BodyState(
        pos=torch.tensor([[x, y, z]], dtype=F64),
        quat=quat(roll, pitch, yaw)[None],
        lin_vel=torch.zeros(1, 3, dtype=F64),
        ang_vel=torch.zeros(1, 3, dtype=F64),
    )


def buoy(**cfg):
    geo = HullGeometry(HullCfg(**cfg), dtype=F64)
    return geo, ColumnBuoyancy(geo)


def flat(n=1):
    return ModelContext(sea=FlatWater(n, dtype=F64))


def test_box_floats_at_design_draft_exactly():
    geo, b = buoy(shape="box", length=4.0, beam=2.0, draft=0.5, freeboard=0.5, density=1000.0)
    assert math.isclose(geo.volume, 4.0, rel_tol=1e-12)
    assert math.isclose(geo.waterplane_area, 8.0, rel_tol=1e-12)
    w = b(state(), flat())
    assert torch.allclose(w.force, torch.tensor([[0.0, 0.0, geo.mass * geo.cfg.gravity]], dtype=F64))
    assert torch.allclose(w.torque, torch.zeros(1, 3, dtype=F64), atol=1e-9)


def test_wigley_volume_and_waterplane_match_analytic():
    L, B, T = 6.10, 2.72, 0.60
    geo, _ = buoy(columns=(8, 4))
    assert math.isclose(geo.volume, 4 / 9 * L * B * T, rel_tol=5e-3), geo.volume / (4 / 9 * L * B * T)
    assert math.isclose(geo.waterplane_area, 2 / 3 * L * B, rel_tol=5e-3)
    # 8 x 4 cells, minus the outer cells at bow and stern where the hull is too narrow to reach them
    assert 24 < geo.num_columns <= 32 and (geo.area > 0).all()
    # centre of buoyancy of a Wigley hull sits on the centreline, below the waterline
    assert abs(geo.center_of_buoyancy[0]) < 1e-9 and abs(geo.center_of_buoyancy[1]) < 1e-9
    assert -0.6 < geo.center_of_buoyancy[2] < 0


def test_still_water_equilibrium_and_heave_stiffness():
    """At the design draft lift = weight; pushed down by dz, extra lift = rho g A_wp dz (wall-sided above the waterline)."""
    geo, b = buoy()
    weight = geo.mass * geo.cfg.gravity
    assert torch.allclose(b(state(), flat()).force[0, 2], torch.tensor(weight, dtype=F64))
    dz = 0.05
    extra = b(state(z=-dz), flat()).force[0, 2] - weight
    expected = geo.cfg.density * geo.cfg.gravity * geo.waterplane_area * dz
    assert math.isclose(extra.item(), expected, rel_tol=1e-9)


def test_roll_and_pitch_are_righted():
    _, b = buoy()
    roll = b(state(roll=math.radians(5)), flat()).torque[0]
    pitch = b(state(pitch=math.radians(3)), flat()).torque[0]
    assert roll[0] < 0, roll          # rolled toward +x rotation -> torque about -x brings it back
    assert pitch[1] < 0, pitch        # bow down (+pitch about +y in FLU) -> bow-up torque


def box_roll_moment(L, B, T, zg, phi, rho=1025.0, g=9.80665):
    """Textbook righting moment of a wall-sided box: -rho g V sin(phi) (GM + BM tan^2(phi) / 2).

    KB = T/2 (centre of buoyancy above the keel), BM = B^2 / (12 T), KG = T + zg
    (zg: CoM height above the waterline), GM = KB + BM - KG.
    """
    V = L * B * T
    BM = B * B / (12 * T)
    GM = T / 2 + BM - (T + zg)
    return -rho * g * V * math.sin(phi) * (GM + BM * math.tan(phi) ** 2 / 2)


@pytest.mark.parametrize("B,T,zg", [(2.0, 0.5, 0.0), (2.0, 0.5, 0.3), (1.0, 1.0, -0.8), (1.0, 1.0, 0.0)])
def test_box_roll_moment_matches_textbook(B, T, zg):
    """Includes (1.0, 1.0, 0.0): narrow, deep, CoM at the waterline -> GM < 0, the box capsizes."""
    L, phi = 4.0, math.radians(5)
    _, b = buoy(shape="box", length=L, beam=B, draft=T, freeboard=0.5, com=(0.0, 0.0, zg), columns=(4, 32))
    got = b(state(z=zg, roll=phi), flat()).torque[0, 0].item()   # CoM at its design height, hull at design draft
    want = box_roll_moment(L, B, T, zg, phi)
    assert math.isclose(got, want, rel_tol=0.02, abs_tol=50.0), (got, want)


def test_wide_hull_resists_roll_more_than_narrow_same_volume():
    """Same volume, CoM at each hull's centre of buoyancy, so only BM = B^2/(12T) differs."""
    _, wide = buoy(shape="box", length=4.0, beam=2.0, draft=0.5, com=(0, 0, -0.25))
    _, narrow = buoy(shape="box", length=4.0, beam=1.0, draft=1.0, com=(0, 0, -0.5))
    tw = wide(state(z=-0.25, roll=math.radians(4)), flat()).torque[0, 0]
    tn = narrow(state(z=-0.5, roll=math.radians(4)), flat()).torque[0, 0]
    assert tw < 0 and tn < 0 and abs(tw) > 3 * abs(tn)


def test_out_of_the_water_and_fully_submerged():
    geo, b = buoy()
    assert torch.equal(b(state(z=5.0), flat()).force, torch.zeros(1, 3, dtype=F64))
    deep = b(state(z=-5.0), flat()).force[0, 2].item()
    full = geo.cfg.density * geo.cfg.gravity * (geo.volume + geo.waterplane_area * geo.cfg.freeboard)
    assert math.isclose(deep, full, rel_tol=1e-9)


def test_crest_under_bow_pitches_bow_up_and_pushes_downslope():
    """A long wave with its crest at the bow and the trough toward the stern."""
    from tests.test_waves import one_wave
    period = 8.0
    sea = one_wave(a=0.4, period=period, direction=0.0)
    k = sea.waves.k.item()
    # place the boat so its bow (x = +L/2) sits on a crest at t = 0: k x + phase = pi/2
    x0 = (math.pi / 2) / k - 3.05
    _, b = buoy()
    w = b(state(x=x0), ModelContext(sea=sea, t=0.0))
    assert w.torque[0, 1] < 0          # bow-up pitch moment
    assert w.force[0, 0] < 0           # surface slopes down toward -x here, so the push is toward -x


def test_batched_equals_one_at_a_time():
    sea = SineWaveField(SeaCfg(), num_envs=6, dtype=F64, seed=3)
    _, b = buoy()
    g = torch.Generator().manual_seed(4)
    s = BodyState(
        pos=torch.randn(6, 3, generator=g, dtype=F64) * torch.tensor([20, 20, 0.2], dtype=F64),
        quat=torch.stack([quat(0.1 * i, -0.05 * i, 0.7 * i) for i in range(6)]),
        lin_vel=torch.zeros(6, 3, dtype=F64), ang_vel=torch.zeros(6, 3, dtype=F64),
    )
    batch = b(s, ModelContext(sea=sea, t=2.0))
    for i in range(6):
        one_sea = SineWaveField.from_components(type(sea.waves)(**{k: getattr(sea.waves, k)[i:i + 1] for k in sea.waves.__dataclass_fields__}))
        one = b(BodyState(s.pos[i:i + 1], s.quat[i:i + 1], s.lin_vel[i:i + 1], s.ang_vel[i:i + 1]), ModelContext(sea=one_sea, t=2.0))
        assert torch.allclose(batch.force[i], one.force[0]) and torch.allclose(batch.torque[i], one.torque[0])


# ------------------------------------------------- against the extension's triangle-cutting buoyancy


@pytest.fixture(scope="module")
def ext():
    sys.path.insert(0, str(EXT))
    try:
        from mmws.wave_sim.forces import base, buoyancy_pressure, hulls
        yield {"base": base, "bp": buoyancy_pressure, "hulls": hulls}
    finally:
        sys.path.remove(str(EXT))


class ExtSea:
    """Our sea, environment 0, at a fixed time, in the shape the extension's force models expect."""

    def __init__(self, sea, t):
        self.sea, self.t = sea, t

    def height(self, x, y):
        xy = torch.as_tensor(np.stack([np.asarray(x, float), np.asarray(y, float)], -1), dtype=F64)
        return self.sea.height(xy.reshape(1, -1, 2), self.t).reshape(np.shape(x)).numpy()


def ext_wrench(ext, vertices, triangles, origin_offset, s, sea, t, density):
    """Extension buoyancy for the same pose, returned in the world frame about our CoM."""
    R = quat_to_matrix(s.quat)[0].numpy()
    model = ext["bp"].BuoyancyPressure(vertices, triangles, density=density)
    body = ext["base"].BodyState(
        position=s.pos[0].numpy() + R @ origin_offset,
        orientation=xyzw_to_wxyz(s.quat)[0].numpy(),
        center_of_mass=-origin_offset,
    )
    w = model.wrench(body, ExtSea(sea, t), t)
    return R @ w.force, R @ w.moment


POSES = [dict(z=0.0), dict(z=0.08), dict(z=-0.1, roll=math.radians(4)), dict(z=0.05, pitch=math.radians(2)),
         dict(z=0.02, roll=math.radians(-3), pitch=math.radians(1.5), yaw=0.6)]


@pytest.mark.parametrize("pose", POSES)
def test_box_force_matches_extension_in_still_water(ext, pose):
    """Forces only. The extension's 12-triangle box applies each triangle's pressure at its centroid, which on so
    coarse a mesh gives spurious torques (e.g. ~2.7 kN m of yaw on an upright box); box torques are checked
    against textbook stability above instead."""
    L, B, T, F = 4.0, 2.0, 0.5, 0.5
    v, tri = ext["hulls"].box_hull(L, B, T + F)
    _, ours = buoy(shape="box", length=L, beam=B, draft=T, freeboard=F, density=1000.0, columns=(16, 8))
    s = state(**pose)
    w = ours(s, flat())
    f, m = ext_wrench(ext, v, tri, np.array([0, 0, (F - T) / 2]), s, FlatWater(1, dtype=F64), 0.0, 1000.0)
    assert np.allclose(w.force[0].numpy(), f, atol=0.01 * abs(f[2])), (w.force, f)


@pytest.mark.parametrize("pose", [dict(z=0.1), dict(z=0.12, pitch=math.radians(1.5)), dict(z=0.15, roll=math.radians(3))])
def test_wigley_matches_extension_in_waves(ext, pose):
    """Raised so the deck stays dry (the extension's Wigley has no freeboard), in a moderate irregular sea."""
    v, tri = ext["hulls"].wigley_hull(stations=81, sections=33, end_fraction=0.999)
    from mmws.wave_sim.integration.repair import orient_faces
    tri = orient_faces(v, tri)  # the raw loft winds inward; the extension orients meshes before use
    _, ours = buoy(freeboard=0.0, density=1000.0, columns=(16, 8))
    sea = SineWaveField(SeaCfg(hs=(0.25, 0.25), tp=(6.0, 6.0), heading=(0.4, 0.4)), num_envs=1, dtype=F64, seed=11)
    s = state(**pose)
    w = ours(s, ModelContext(sea=sea, t=3.0))
    f, m = ext_wrench(ext, v, tri, np.zeros(3), s, sea, 3.0, 1000.0)
    scale = abs(f[2])
    assert np.allclose(w.force[0].numpy(), f, atol=0.03 * scale), (w.force, f)
    assert np.allclose(w.torque[0].numpy(), m, atol=0.03 * scale * 3.05), (w.torque, m)
