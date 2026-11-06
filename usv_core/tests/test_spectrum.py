import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from usv_core.constants import GRAVITY
from usv_core.spectrum import (
    SeaCfg,
    cos2s_spreading,
    jonswap,
    pierson_moskowitz,
    sample_components,
)

EXT = Path(__file__).resolve().parents[2] / "exts/mmws.wave_sim"


@pytest.fixture(scope="module")
def ext():
    sys.path.insert(0, str(EXT))
    try:
        from mmws.wave_sim.spectra.jonswap import Jonswap
        from mmws.wave_sim.spectra.pierson_moskowitz import PiersonMoskowitz
        yield {"jonswap": Jonswap, "pm": PiersonMoskowitz}
    finally:
        sys.path.remove(str(EXT))


OMEGA = torch.linspace(0.0, 4.0, 401, dtype=torch.float64)


@pytest.mark.parametrize("hs,tp", [(0.5, 4.0), (2.0, 7.0), (6.0, 12.0)])
def test_pm_matches_extension(ext, hs, tp):
    ours = pierson_moskowitz(OMEGA, torch.tensor(hs, dtype=torch.float64), torch.tensor(tp, dtype=torch.float64))
    theirs = ext["pm"](hs=hs, tp=tp).energy(OMEGA.numpy())
    np.testing.assert_allclose(ours.numpy(), theirs, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("hs,tp,gamma", [(2.0, 7.0, 3.3), (4.0, 10.0, 1.0), (1.0, 5.0, 7.0)])
def test_jonswap_matches_extension(ext, hs, tp, gamma):
    ours = jonswap(OMEGA, torch.tensor(hs, dtype=torch.float64), torch.tensor(tp, dtype=torch.float64), gamma)
    theirs = ext["jonswap"](hs=hs, tp=tp, gamma=gamma).energy(OMEGA.numpy())
    np.testing.assert_allclose(ours.numpy(), theirs, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("s", [1.0, 5.0, 20.0])
def test_spreading_matches_extension_and_integrates_to_one(ext, s):
    theta = torch.linspace(-math.pi, math.pi, 2001, dtype=torch.float64)
    mean = torch.tensor(0.4, dtype=torch.float64)
    ours = cos2s_spreading(theta, mean, torch.tensor(s, dtype=torch.float64))
    theirs = ext["jonswap"](hs=1.0, tp=5.0, wind_dir=0.4, spread_s=s).spreading(theta.numpy())
    np.testing.assert_allclose(ours.numpy(), theirs, rtol=1e-10, atol=1e-12)
    assert abs(torch.trapezoid(ours, theta).item() - 1.0) < 1e-6


def draw(n=256, seed=0, **kw):
    cfg = SeaCfg(**kw)
    return cfg, sample_components(cfg, n, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def test_shapes_and_ranges():
    cfg, c = draw(n=64)
    K = cfg.num_waves
    for name in ("amp", "omega", "k", "direction", "cos_dir", "sin_dir", "phase"):
        assert getattr(c, name).shape == (64, K), name
    for name, rng in (("hs", cfg.hs), ("tp", cfg.tp), ("heading", cfg.heading)):
        v = getattr(c, name)
        assert v.shape == (64,) and v.min() >= rng[0] and v.max() <= rng[1], name
    assert (c.phase >= 0).all() and (c.phase < 2 * math.pi).all()


def test_exact_hs():
    _, c = draw()
    assert torch.allclose(c.significant_height(), c.hs, rtol=1e-12)


def test_unscaled_waves_already_carry_nearly_the_right_energy():
    """Without the rescale, bin-integrated energy should already give nearly the requested Hs.

    What is left is the band cut-off (energy above 3 omega_p, about 0.8 % of
    variance) and the Goda normalisation, so Hs lands a fraction of a percent low.
    """
    _, c = draw(exact_hs=False)
    ratio = c.significant_height() / c.hs
    assert (ratio > 0.99).all() and (ratio < 1.005).all(), (ratio.min().item(), ratio.max().item())


def test_dispersion_relation():
    _, c = draw()
    assert torch.allclose(c.k, c.omega**2 / GRAVITY)


def test_frequencies_one_per_bin_inside_band():
    cfg, c = draw()
    wp = 2 * math.pi / c.tp[:, None]
    rel = c.omega / wp
    assert (rel >= cfg.band_lo).all() and (rel <= cfg.band_hi).all()
    assert (rel[:, 1:] > rel[:, :-1]).all()  # one draw per bin, bins in order


def test_directions_centre_on_heading_and_narrow_with_s():
    def spread(s):
        _, c = draw(n=512, heading=(0.7, 0.7), spread_s=(s, s), num_waves=64)
        d = c.direction - 0.7
        mean = math.atan2(torch.sin(d).mean().item(), torch.cos(d).mean().item())
        return mean, torch.cos(d).mean().item()  # mean resultant length: 1 = all one direction

    mean5, r5 = spread(5.0)
    mean20, r20 = spread(20.0)
    assert abs(mean5) < 0.02 and abs(mean20) < 0.02
    assert r20 > r5 > 0.8


def test_reproducible_and_resample_rows():
    _, a = draw(n=16, seed=3)
    _, b = draw(n=16, seed=3)
    assert torch.equal(a.amp, b.amp) and torch.equal(a.phase, b.phase)
    _, fresh = draw(n=2, seed=99)
    before = a.amp.clone()
    ids = torch.tensor([4, 11])
    a.copy_rows_(ids, fresh)
    assert torch.equal(a.amp[ids], fresh.amp)
    keep = torch.ones(16, dtype=torch.bool)
    keep[ids] = False
    assert torch.equal(a.amp[keep], before[keep])
