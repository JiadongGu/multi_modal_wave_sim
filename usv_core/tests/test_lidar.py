import math

import torch

from tests.test_waves import one_wave
from usv_core import BodyState, Cylinders, FlatWater, LidarCfg, SeaCfg, SeaLidar, SineWaveField, quat_to_matrix

F64 = torch.float64


def boat(z=0.0, n=1, roll=0.0, x=0.0):
    return BodyState(
        pos=torch.tensor([[x, 0.0, z]], dtype=F64).repeat(n, 1),
        quat=torch.tensor([[math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2)]], dtype=F64).repeat(n, 1),
        lin_vel=torch.zeros(n, 3, dtype=F64),
        ang_vel=torch.zeros(n, 3, dtype=F64),
    )


def lidar(**kw):
    return SeaLidar(LidarCfg(**{"mount": (0.0, 0.0, 2.0), **kw}), dtype=F64)


def test_ray_count_is_channels_times_azimuths():
    assert lidar(channels=16, horizontal_res=2.0).num_rays == 16 * 180


def test_flat_water_range_is_exact_geometry():
    l = lidar(channels=5, vertical_fov=(-20.0, 20.0), horizontal_res=30.0, max_range=100.0)
    scan = l.scan(boat(), FlatWater(1, dtype=F64), 0.0)
    for i, el in enumerate(torch.linspace(-20.0, 20.0, 5).tolist()):
        channel = scan.ranges[0, i * 12:(i + 1) * 12]
        if el < -1e-6:
            assert torch.allclose(channel, torch.full((12,), 2.0 / math.sin(math.radians(-el)), dtype=F64), rtol=1e-3)
        else:
            assert torch.isinf(channel).all()


def test_tilted_boat_matches_plane_intersection():
    roll = math.radians(10)
    l = lidar(channels=7, vertical_fov=(-30.0, 30.0), horizontal_res=15.0)
    s = boat(roll=roll)
    scan = l.scan(s, FlatWater(1, dtype=F64), 0.0)
    R = quat_to_matrix(s.quat)[0]
    origin_z = (R @ l.mount)[2]
    dz = (l.directions @ R.T)[:, 2]
    for r in range(l.num_rays):
        if dz[r] < -1e-3 and origin_z / -dz[r] < 100.0:
            assert math.isclose(scan.ranges[0, r].item(), (origin_z / -dz[r]).item(), rel_tol=2e-3)
        elif dz[r] > 1e-3:
            assert math.isinf(scan.ranges[0, r].item())


def test_beyond_max_range_is_no_return():
    l = lidar(channels=1, vertical_fov=(-1.0, -1.0), horizontal_res=90.0, max_range=50.0)     # would hit at 114.6 m
    assert torch.isinf(l.scan(boat(), FlatWater(1, dtype=F64), 0.0).ranges).all()


def test_hits_lie_on_the_waves():
    sea = SineWaveField(SeaCfg(hs=(2.0, 3.0)), num_envs=8, dtype=F64, seed=2)
    l = lidar(march_steps=64)
    scan = l.scan(boat(n=8), sea, 7.0)
    good = torch.isfinite(scan.ranges)
    assert good.sum() > 1000
    eta = sea.height(torch.nan_to_num(scan.hits[..., :2], posinf=0.0), 7.0)
    assert (scan.hits[..., 2] - eta)[good].abs().max() < 5e-3


def test_first_hit_matches_a_dense_march():
    sea = SineWaveField(SeaCfg(hs=(2.5, 2.5)), num_envs=1, dtype=F64, seed=9)
    l = lidar(channels=8, vertical_fov=(-12.0, 3.0), horizontal_res=45.0, march_steps=128, max_range=60.0)
    s, t = boat(), 3.0
    got = l.scan(s, sea, t).ranges[0]
    d = l.directions
    grid = torch.arange(0.0, 60.0, 0.005, dtype=F64)
    for r in range(l.num_rays):
        p = l.mount[None] + grid[:, None] * d[r][None]
        f = p[:, 2] - sea.height(p[None, :, :2], t)[0]
        wet = (f <= 0).nonzero()
        want = grid[wet[0]].item() if len(wet) else math.inf
        if math.isinf(want):
            assert math.isinf(got[r].item()), r
        else:
            assert abs(got[r].item() - want) < 0.02, (r, got[r].item(), want)


def test_a_crest_can_block_an_upward_ray():
    sea = one_wave(a=4.0, period=10.0, direction=0.0, phase=0.0)
    l = lidar(channels=1, vertical_fov=(2.0, 2.0), horizontal_res=180.0)
    k = sea.waves.k.item()                                    # crest at x = pi / (2k), above the sensor at z = 2
    scan = l.scan(boat(), sea, 0.0)
    assert math.isfinite(scan.ranges[0, 0].item()) and 0.0 < scan.ranges[0, 0].item() < math.pi / (2 * k)


def test_under_the_water_reads_zero():
    assert torch.equal(lidar().scan(boat(z=-5.0), FlatWater(1, dtype=F64), 0.0).ranges, torch.zeros(1, 2880, dtype=F64))


def test_each_boat_is_independent():
    """A batch of boats gives exactly what each boat gets alone, in its own sea."""
    sea = SineWaveField(SeaCfg(hs=(1.0, 3.0)), num_envs=4, dtype=F64, seed=4)
    l = lidar(channels=4, vertical_fov=(-15.0, 0.0), horizontal_res=30.0)
    s = boat(n=4)
    together = l.scan(s, sea, 2.0).ranges
    assert not torch.allclose(torch.nan_to_num(together[0], posinf=-1.0), torch.nan_to_num(together[1], posinf=-1.0))
    for i in range(4):
        alone_sea = SineWaveField.from_components(type(sea.waves)(**{k: getattr(sea.waves, k)[i:i + 1] for k in sea.waves.__dataclass_fields__}))
        alone = l.scan(BodyState(s.pos[i:i + 1], s.quat[i:i + 1], s.lin_vel[i:i + 1], s.ang_vel[i:i + 1]), alone_sea, 2.0).ranges
        assert torch.equal(torch.nan_to_num(together[i], posinf=-1.0), torch.nan_to_num(alone[0], posinf=-1.0))


def test_treadmill_position_is_what_counts():
    """The boat's voyage position picks the water, so a boat further along sees different waves."""
    sea = SineWaveField(SeaCfg(hs=(2.0, 2.0)), num_envs=1, dtype=F64, seed=1)
    l = lidar()
    assert not torch.equal(torch.nan_to_num(l.scan(boat(), sea, 1.0).ranges, posinf=-1.0),
                           torch.nan_to_num(l.scan(boat(x=37.0), sea, 1.0).ranges, posinf=-1.0))


def pillar(x, y=0.0, n=1, radius=1.0, z=(-3.0, 9.0)):
    return Cylinders(torch.tensor([[[x, y]]], dtype=F64).repeat(n, 1, 1), radius, *z)


def level_lidar(**kw):
    return lidar(channels=1, vertical_fov=(0.0, 0.0), horizontal_res=90.0, **kw)


def test_pillar_range_and_bearing():
    l = level_lidar()                                          # beams at 0, 90, 180, 270 degrees
    r = l.scan(boat(), FlatWater(1, dtype=F64), 0.0, pillar(20.0)).ranges[0]
    assert math.isclose(r[0].item(), 19.0, rel_tol=1e-6)       # ahead: centre at 20 m minus 1 m radius
    assert torch.isinf(r[1:]).all()


def test_pillar_beyond_range_or_height_is_not_seen():
    water = FlatWater(1, dtype=F64)
    assert torch.isinf(level_lidar(max_range=10.0).scan(boat(), water, 0.0, pillar(20.0)).ranges).all()
    assert torch.isinf(level_lidar().scan(boat(), water, 0.0, pillar(20.0, z=(5.0, 9.0))).ranges).all()


def test_nearer_of_water_and_pillar_wins():
    l = lidar(channels=1, vertical_fov=(-10.0, -10.0), horizontal_res=90.0)     # meets flat water at 11.5 m
    water = FlatWater(1, dtype=F64)
    assert math.isclose(l.scan(boat(), water, 0.0, pillar(50.0)).ranges[0, 0].item(), 2.0 / math.sin(math.radians(10)), rel_tol=1e-3)
    near = l.scan(boat(), water, 0.0, pillar(6.0)).ranges[0, 0].item()
    assert math.isclose(near, 5.0 / math.cos(math.radians(10)), rel_tol=1e-3)


def test_each_boat_only_sees_its_own_pillar():
    l = level_lidar()
    centres = torch.tensor([[[20.0, 0.0]], [[0.0, 30.0]]], dtype=F64)
    r = l.scan(boat(n=2), FlatWater(2, dtype=F64), 0.0, Cylinders(centres, 1.0, -3.0, 9.0)).ranges
    assert math.isclose(r[0, 0].item(), 19.0, rel_tol=1e-6) and torch.isinf(r[0, 1:]).all()
    assert math.isclose(r[1, 1].item(), 29.0, rel_tol=1e-6) and torch.isinf(r[1, [0, 2, 3]]).all()
