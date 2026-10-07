#!/usr/bin/env python3
"""Check the framework outside Isaac Sim.

    python3 tools/selftest.py

Isaac Sim takes a minute to launch, needs an RTX GPU and cannot easily be
driven from a test runner. Nothing in `spectra/`, `field/` or `forces/` depends
on it, so all of that can be checked in under a second from a plain shell. Use
this as the inner loop while writing wave math, and save Isaac Sim for the
visuals and the force path.

Tests marked EXPECTED FAIL correspond to the stubs you have not implemented
yet. They should turn into passes as you fill them in; that is the to-do list.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make the extension's python package importable without installing anything.
EXT_PY = Path(__file__).resolve().parents[1] / "exts" / "mmws.wave_sim"
sys.path.insert(0, str(EXT_PY))

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: python3 -m pip install numpy")

# numpy 2.0 renamed trapz -> trapezoid and later removed the old name.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

from mmws.wave_sim import FIELDS, FORCES, SPECTRA          # noqa: E402
from mmws.wave_sim.field import deep_water_omega           # noqa: E402
from mmws.wave_sim.forces import BodyState                 # noqa: E402

PASS, FAIL, XFAIL = [], [], []


def check(name, fn, expected_fail=False):
    try:
        fn()
    except NotImplementedError as exc:
        (XFAIL if expected_fail else FAIL).append((name, f"not implemented: {exc}"))
    except Exception as exc:                                # noqa: BLE001
        (XFAIL if expected_fail else FAIL).append((name, f"{type(exc).__name__}: {exc}"))
    else:
        PASS.append(name)


def assert_close(got, want, tol, what):
    if not math.isfinite(got):
        raise AssertionError(f"{what}: got non-finite {got}")
    if abs(got - want) > tol:
        raise AssertionError(f"{what}: got {got:.6g}, want {want:.6g} +/- {tol:.3g}")


# --------------------------------------------------------------------- registry

def test_registries_populated():
    assert len(SPECTRA) >= 3, f"expected >=3 spectra, got {SPECTRA.names()}"
    assert "flat" in FIELDS, FIELDS.names()
    assert "zero" in FORCES, FORCES.names()


def test_unknown_name_is_helpful():
    try:
        SPECTRA.get("does_not_exist")
    except KeyError as exc:
        assert "Available" in str(exc), f"unhelpful error: {exc}"
    else:
        raise AssertionError("expected KeyError for an unknown spectrum")


def test_registry_create():
    s = SPECTRA.create("pierson_moskowitz", hs=2.0, tp=8.0)
    assert s.hs == 2.0 and s.tp == 8.0
    assert s.name == "pierson_moskowitz"


# -------------------------------------------------------------------- spreading

def test_spreading_integrates_to_one():
    """The cos-2s normalisation must be right or every sea has wrong energy."""
    s = SPECTRA.create("pierson_moskowitz", spread_s=5.0)
    theta = np.linspace(-math.pi, math.pi, 20001)
    total = float(_trapezoid(s.spreading(theta), theta))
    assert_close(total, 1.0, 1e-3, "integral of D(theta)")


def test_spreading_peaks_at_wind_direction():
    s = SPECTRA.create("pierson_moskowitz", wind_dir=1.0, spread_s=5.0)
    theta = np.linspace(-math.pi, math.pi, 4001)
    peak = theta[int(np.argmax(s.spreading(theta)))]
    assert_close(peak, 1.0, 5e-3, "argmax of D(theta)")


def test_spreading_narrows_with_s():
    """Larger s must mean a more directional sea, not merely a different one."""
    def width(s_val):
        s = SPECTRA.create("pierson_moskowitz", spread_s=s_val)
        theta = np.linspace(-math.pi, math.pi, 8001)
        d = s.spreading(theta)
        return float(_trapezoid(d * theta**2, theta))     # angular variance
    assert width(25.0) < width(5.0), "s=25 should be narrower than s=5"


# --------------------------------------------------------------------- dispersion

def test_dispersion():
    """omega = sqrt(g k), so a long wave must travel faster than a short one."""
    for wavelength, want_speed in ((100.0, 12.5), (10.0, 3.95)):
        k = 2.0 * math.pi / wavelength
        speed = float(deep_water_omega(k)) / k          # phase speed c = omega/k
        assert_close(speed, want_speed, 0.1, f"phase speed at {wavelength} m")


# ----------------------------------------------------------------------- spectra

def _hs_roundtrip(name, **kwargs):
    """Ask for Hs, integrate the spectrum, get the same Hs back."""
    want = 2.5
    s = SPECTRA.create(name, hs=want, tp=7.0, **kwargs)
    assert_close(s.hs_from_spectrum(), want, 0.03 * want, f"{name} Hs round-trip")


def test_pm_hs_roundtrip():
    _hs_roundtrip("pierson_moskowitz")


def test_pm_zero_frequency_is_finite():
    """S(0) must be 0, not nan. A nan here poisons the entire wave field."""
    s = SPECTRA.create("pierson_moskowitz")
    vals = np.asarray(s.energy(np.array([0.0, 0.5, 1.0])))
    assert np.all(np.isfinite(vals)), f"non-finite energy: {vals}"
    assert vals[0] == 0.0, f"S(0) should be 0, got {vals[0]}"


def test_pm_peaks_near_omega_p():
    s = SPECTRA.create("pierson_moskowitz", tp=7.0)
    omega = np.linspace(0.05, 5.0, 20001)
    peak = omega[int(np.argmax(s.energy(omega)))]
    assert_close(peak, s.omega_p, 0.05 * s.omega_p, "PM peak location")


def test_jonswap_hs_roundtrip():
    _hs_roundtrip("jonswap", gamma=3.3)


def test_jonswap_reduces_to_pm():
    """gamma = 1 means no peak enhancement, so it must equal PM exactly."""
    omega = np.linspace(0.1, 4.0, 500)
    pm = np.asarray(SPECTRA.create("pierson_moskowitz", hs=2.0, tp=7.0).energy(omega))
    js = np.asarray(SPECTRA.create("jonswap", hs=2.0, tp=7.0, gamma=1.0).energy(omega))
    err = float(np.max(np.abs(js - pm)) / np.max(pm))
    assert err < 1e-6, f"JONSWAP(gamma=1) differs from PM by {err:.2e}"


def test_jonswap_sharper_than_pm():
    omega = np.linspace(0.1, 4.0, 4001)
    pm = SPECTRA.create("pierson_moskowitz", hs=2.0, tp=7.0)
    js = SPECTRA.create("jonswap", hs=2.0, tp=7.0, gamma=3.3)
    ratio = float(np.max(js.energy(omega)) / np.max(pm.energy(omega)))
    assert ratio > 1.3, f"gamma=3.3 should raise the peak well above PM, got {ratio:.2f}x"


def test_eckv_hs_roundtrip():
    _hs_roundtrip("eckv")


# ------------------------------------------------------------------ wave field

def test_flat_field_is_flat():
    f = FIELDS.create("flat")
    f.advance(1.234)
    assert float(np.max(np.abs(f.height(np.linspace(-50, 50, 101), 3.0)))) == 0.0
    assert_close(f.significant_height(), 0.0, 1e-12, "Hs of flat water")


def test_flat_field_normal_is_up():
    f = FIELDS.create("flat")
    n = f.normal(np.zeros(4), np.zeros(4))
    assert np.allclose(n, np.array([0.0, 0.0, 1.0])), f"normal should be +Z, got {n}"


def test_field_advance_is_reproducible():
    """Same time, same surface - or no wave field can ever be regression tested."""
    f = FIELDS.create("flat")
    x = np.linspace(0, 10, 17)
    f.advance(3.0)
    a = np.array(f.height(x, 0.0))
    f.advance(9.0)
    f.advance(3.0)
    assert np.array_equal(a, np.array(f.height(x, 0.0))), "advance(t) is not reproducible"


def _fft(seed=0, L=512.0, N=256, **kw):
    spec = SPECTRA.create("jonswap", hs=2.0, tp=7.0, spread_s=5.0, gamma=3.3)
    return FIELDS.create("fft", spectrum=spec, tile_size=L, resolution=N, seed=seed, **kw)


def test_fft_needs_a_spectrum():
    try:
        FIELDS.create("fft", spectrum=None)
    except (ValueError, TypeError):
        return
    raise AssertionError("a wave field with no spectrum should not construct")


def test_fft_surface_is_real():
    """The two-term construction is Hermitian, so the ifft must come out real."""
    f = _fft()
    z = np.asarray(f.height(np.linspace(0, 100, 32), 0.0))
    assert np.all(np.isreal(z)), "height() returned complex values"
    assert np.all(np.isfinite(z)), "height() returned non-finite values"


def test_fft_hs_roundtrip():
    """Averaged over seeds, the field must measure the Hs the spectrum promised."""
    hs = [_fft(seed=k).significant_height() for k in range(8)]
    mean = float(np.mean(hs))
    assert_close(mean, 2.0, 0.16, f"mean Hs over 8 seeds (got {hs!r})")


def test_fft_advance_is_reproducible():
    f = _fft(N=128)
    x = np.linspace(0, 200, 17)
    f.advance(3.0)
    a = np.array(f.height(x, 0.0))
    f.advance(99.0)
    f.advance(3.0)
    assert np.array_equal(a, np.array(f.height(x, 0.0))), "advance(t) is not reproducible"


def test_fft_waves_actually_move():
    f = _fft(N=128)
    x = np.linspace(0, 200, 17)
    f.advance(0.0)
    a = np.array(f.height(x, 0.0))
    f.advance(2.0)
    moved = float(np.max(np.abs(np.array(f.height(x, 0.0)) - a)))
    assert moved > 0.05, f"surface barely changed in 2 s (max delta {moved:.4f} m)"


def test_fft_stays_bounded():
    f = _fft(N=128)
    f.advance(600.0)
    assert_close(f.significant_height(), 2.0, 0.8, "Hs after 10 minutes")


def test_cell_size():
    f = FIELDS.create("flat", tile_size=128.0, resolution=256)
    assert_close(f.cell_size, 0.5, 1e-12, "cell size")


# ---------------------------------------------------------------- force models

def test_zero_force_is_zero():
    w = FORCES.create("zero").wrench(BodyState(mass=28.0), FIELDS.create("flat"), 0.0)
    assert np.allclose(w.as_array(), np.zeros(6)), w.as_array()
    assert w.as_array().shape == (6,), "wrench must be a 6-vector"


def test_wrench_addition():
    from mmws.wave_sim.forces import Wrench
    a = Wrench(np.array([1.0, 0, 0]), np.array([0, 2.0, 0]))
    total = (a + a).as_array()
    assert np.allclose(total, [2, 0, 0, 0, 4, 0]), total


# ------------------------------------------------------------------------- main

TESTS = [
    ("registries populated", test_registries_populated, False),
    ("unknown name gives a helpful error", test_unknown_name_is_helpful, False),
    ("registry create passes kwargs", test_registry_create, False),
    ("spreading integrates to 1", test_spreading_integrates_to_one, False),
    ("spreading peaks at wind direction", test_spreading_peaks_at_wind_direction, False),
    ("spreading narrows as s grows", test_spreading_narrows_with_s, False),
    ("deep-water dispersion", test_dispersion, False),
    ("PM: Hs round-trip", test_pm_hs_roundtrip, False),
    ("PM: S(0) finite and zero", test_pm_zero_frequency_is_finite, False),
    ("PM: peaks at omega_p", test_pm_peaks_near_omega_p, False),
    ("flat field is flat", test_flat_field_is_flat, False),
    ("flat field normal is +Z", test_flat_field_normal_is_up, False),
    ("advance(t) is reproducible", test_field_advance_is_reproducible, False),
    ("cell size", test_cell_size, False),
    ("zero force model returns 6 zeros", test_zero_force_is_zero, False),
    ("wrench addition", test_wrench_addition, False),
    ("FFT: rejects a missing spectrum", test_fft_needs_a_spectrum, True),
    ("FFT: surface is real and finite", test_fft_surface_is_real, True),
    ("FFT: Hs round-trip over 8 seeds", test_fft_hs_roundtrip, True),
    ("FFT: advance(t) is reproducible", test_fft_advance_is_reproducible, True),
    ("FFT: waves actually move", test_fft_waves_actually_move, True),
    ("FFT: still bounded after 10 min", test_fft_stays_bounded, True),
    # --- stubs you are going to implement -------------------------------------
    ("JONSWAP: Hs round-trip", test_jonswap_hs_roundtrip, True),
    ("JONSWAP: gamma=1 equals PM", test_jonswap_reduces_to_pm, True),
    ("JONSWAP: gamma=3.3 sharpens peak", test_jonswap_sharper_than_pm, True),
    ("ECKV: Hs round-trip", test_eckv_hs_roundtrip, True),
]


def main() -> int:
    for name, fn, xfail in TESTS:
        check(name, fn, xfail)

    for name in PASS:
        print(f"  pass   {name}")
    for name, why in XFAIL:
        print(f"  todo   {name}  ({why})")
    for name, why in FAIL:
        print(f"  FAIL   {name}  ({why})")

    print(
        f"\n{len(PASS)} passed, {len(XFAIL)} awaiting implementation, {len(FAIL)} failed"
    )
    if FAIL:
        print("\nFAILED - something that used to work is broken.")
        return 1
    print("Framework is sound. The 'todo' entries are the work queue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
