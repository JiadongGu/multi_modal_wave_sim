import math

import pytest
import torch

from usv_core.constants import GRAVITY
from usv_core.spectrum import SeaCfg, WaveComponents
from usv_core.waves import FlatWater, SineWaveField, WaveField

F64 = torch.float64


def one_wave(a=1.5, period=6.0, direction=0.0, phase=0.0, n=1):
    omega = 2 * math.pi / period
    t = lambda v: torch.full((n, 1), float(v), dtype=F64)
    return SineWaveField.from_components(WaveComponents(
        amp=t(a), omega=t(omega), k=t(omega**2 / GRAVITY), direction=t(direction),
        cos_dir=t(math.cos(direction)), sin_dir=t(math.sin(direction)), phase=t(phase),
        hs=torch.full((n,), 4 * a / math.sqrt(2), dtype=F64), tp=torch.full((n,), period, dtype=F64),
        heading=torch.full((n,), direction, dtype=F64), spread_s=torch.ones(n, dtype=F64),
    ))


def field(n=64, seed=0, **kw):
    return SineWaveField(SeaCfg(**kw), num_envs=n, dtype=F64, seed=seed)


def rand_points(n, m, dims=2, scale=200.0, seed=1):
    return (torch.rand(n, m, dims, generator=torch.Generator().manual_seed(seed), dtype=F64) - 0.5) * scale


def test_single_wave_textbook_values():
    f = one_wave(a=1.5, period=6.0)
    k = (2 * math.pi / 6.0) ** 2 / GRAVITY
    x = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=F64) * (2 * math.pi / k)   # quarter wavelengths
    xy = torch.stack([x, torch.zeros_like(x)], -1)[None]
    assert torch.allclose(f.height(xy, 0.0), torch.tensor([[0.0, 1.5, 0.0, -1.5]], dtype=F64), atol=1e-12)
    # a quarter period later the crest has moved a quarter wavelength forward
    assert torch.allclose(f.height(xy, 1.5), torch.tensor([[-1.5, 0.0, 1.5, 0.0]], dtype=F64), atol=1e-12)


def test_shapes_and_per_env_time():
    f = field(n=8)
    xy = rand_points(8, 5)
    assert f.height(xy, 3.0).shape == (8, 5)
    assert f.slope(xy, 3.0).shape == (8, 5, 2)
    assert f.velocity(torch.cat([xy, torch.zeros(8, 5, 1, dtype=F64)], -1), 3.0).shape == (8, 5, 3)
    t = torch.arange(8, dtype=F64)
    per_env = f.height(xy, t)
    for i in (0, 5):
        assert torch.allclose(per_env[i], f.height(xy, float(i))[i])


def test_variance_matches_requested_hs():
    """Over many points and times, var(eta) ~ Hs^2 / 16 for each environment's sea."""
    f = field(n=16, hs=(1.0, 4.0))
    xy = rand_points(16, 4000, scale=2000.0)
    t = torch.rand(16, generator=torch.Generator().manual_seed(2), dtype=F64) * 1000
    eta = f.height(xy, t)
    hs_measured = 4 * eta.std(dim=1)
    assert torch.allclose(hs_measured, f.waves.hs, rtol=0.08), (hs_measured / f.waves.hs)


def test_vertical_velocity_at_surface_is_d_eta_dt():
    f = field(n=8)
    xy = rand_points(8, 20)
    xyz = torch.cat([xy, torch.zeros(8, 20, 1, dtype=F64)], -1)
    h = 1e-5
    d_eta_dt = (f.height(xy, 10.0 + h) - f.height(xy, 10.0 - h)) / (2 * h)
    assert torch.allclose(f.velocity(xyz, 10.0)[..., 2], d_eta_dt, atol=1e-6)


def test_slope_is_finite_difference_gradient():
    f = field(n=8)
    xy = rand_points(8, 20)
    h = 1e-5
    ex = torch.tensor([h, 0.0], dtype=F64)
    ey = torch.tensor([0.0, h], dtype=F64)
    gx = (f.height(xy + ex, 4.0) - f.height(xy - ex, 4.0)) / (2 * h)
    gy = (f.height(xy + ey, 4.0) - f.height(xy - ey, 4.0)) / (2 * h)
    assert torch.allclose(f.slope(xy, 4.0), torch.stack([gx, gy], -1), atol=1e-7)


def test_orbital_motion_decays_with_depth_and_is_clamped_above():
    f = one_wave(a=1.0, period=8.0, direction=0.0)
    k = f.waves.k.item()
    pts = lambda z: torch.tensor([[[3.0, 0.0, z]]], dtype=F64)
    v0 = f.velocity(pts(0.0), 2.0)
    assert torch.allclose(f.velocity(pts(-5.0), 2.0), v0 * math.exp(-5.0 * k), atol=1e-12)
    assert torch.allclose(f.velocity(pts(+1.0), 2.0), v0)


def test_horizontal_velocity_points_along_wave_direction():
    f = one_wave(a=1.0, period=8.0, direction=math.pi / 3)
    v = f.velocity(torch.tensor([[[1.0, 2.0, 0.0]]], dtype=F64), 0.7)[0, 0]
    assert abs(v[0] * math.sin(math.pi / 3) - v[1] * math.cos(math.pi / 3)) < 1e-12


def test_environments_see_different_seas_and_resample_only_chosen():
    f = field(n=6)
    xy = torch.zeros(6, 1, 2, dtype=F64)
    eta = f.height(xy, 5.0)
    assert eta.unique().numel() == 6
    before = f.waves.amp.clone()
    f.resample(torch.tensor([1, 4]))
    changed = (f.waves.amp != before).any(-1)
    assert changed.tolist() == [False, True, False, False, True, False]
    f.resample(torch.tensor([], dtype=torch.long))  # empty reset is a no-op
    f.resample()
    assert (f.waves.amp != before).any(-1).all()


def test_flat_water():
    f = FlatWater(4, dtype=F64)
    xy = rand_points(4, 3)
    assert isinstance(f, WaveField)
    assert torch.equal(f.height(xy, 1.0), torch.zeros(4, 3, dtype=F64))
    assert torch.equal(f.slope(xy, 1.0), torch.zeros(4, 3, 2, dtype=F64))
    assert f.sea_state()["hs"].shape == (4,)


def test_socket_is_enforced():
    class Incomplete(WaveField):
        def height(self, xy, t):
            return xy[..., 0]

    with pytest.raises(TypeError):
        Incomplete()
