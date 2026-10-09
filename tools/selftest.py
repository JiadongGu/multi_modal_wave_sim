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
from mmws.wave_sim.forces import BodyState, box_hull, quat_to_matrix  # noqa: E402
from mmws.wave_sim.globals import GRAVITY, WATER_DENSITY   # noqa: E402

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


def test_eckv_peak_matches_the_wind():
    """ECKV takes wind, not Hs, so the peak comes from U10 - not from tp."""
    s = SPECTRA.create("eckv", u10=10.0)
    assert_close(s.k_p, 0.0692, 0.001, "peak wavenumber at U10=10")
    assert_close(2 * math.pi / s.k_p, 90.8, 1.0, "peak wavelength at U10=10 [m]")


def test_eckv_frequency_and_wavenumber_agree():
    """S(omega) is S(k) times dk/domega. Both integrals must give the same m0."""
    s = SPECTRA.create("eckv", u10=10.0)
    k = np.linspace(1e-4, 500.0, 500000)
    hs_k = 4.0 * math.sqrt(float(_trapezoid(np.asarray(s.wavenumber_energy(k)), k)))
    assert_close(s.hs_from_spectrum(), hs_k, 0.02 * hs_k, "Hs via omega vs via k")


def test_eckv_grows_with_wind():
    hs = [SPECTRA.create("eckv", u10=float(u)).hs_from_spectrum() for u in (5, 10, 20)]
    assert hs[0] < hs[1] < hs[2], f"Hs must increase with wind speed, got {hs}"
    assert_close(hs[1], 2.6, 0.5, "Hs at U10=10 m/s")


def test_eckv_zero_frequency_is_finite():
    s = SPECTRA.create("eckv", u10=10.0)
    v = np.asarray(s.energy(np.array([0.0, 0.5, 1.0])))
    assert np.all(np.isfinite(v)), f"non-finite energy: {v}"
    assert v[0] == 0.0, f"S(0) should be 0, got {v[0]}"


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


def test_mesh_points_shape_and_spacing():
    f = _fft(N=32)
    pts = f.mesh_points()
    N = f.resolution
    assert pts.shape == (N * N, 3), f"expected {(N*N, 3)}, got {pts.shape}"
    assert_close(float(pts[N, 0] - pts[0, 0]), f.cell_size, 1e-9, "x spacing between rows")
    assert_close(float(pts[1, 1] - pts[0, 1]), f.cell_size, 1e-9, "y spacing within a row")


def test_mesh_points_match_height():
    """The mesh and the physics must read the same surface - that is the design."""
    f = _fft(N=32)
    pts = f.mesh_points()
    z = np.asarray(f.height(pts[:, 0], pts[:, 1]))
    assert np.allclose(pts[:, 2], z), "mesh z does not match height() at the same xy"


def test_mesh_topology_valid():
    f = _fft(N=32)
    counts, idx = f.mesh_topology()
    N = f.resolution
    assert counts.shape == ((N - 1) ** 2,), f"expected {(N-1)**2} faces, got {counts.shape}"
    assert bool((counts == 4).all()), "every face must be a quad"
    assert idx.shape == (4 * (N - 1) ** 2,), f"wrong index count: {idx.shape}"
    assert int(idx.min()) >= 0 and int(idx.max()) < N * N, "vertex index out of range"


def test_mesh_topology_is_static():
    """Topology never changes - rebuilding it every frame is the classic perf bug."""
    f = _fft(N=32)
    a = f.mesh_topology()[1]
    f.advance(12.0)
    assert np.array_equal(a, f.mesh_topology()[1]), "topology changed after advance()"


def test_mesh_winding_faces_up():
    f = _fft(N=32)
    pts, (_, idx) = f.mesh_points(), f.mesh_topology()
    a, b, d = pts[idx[0]], pts[idx[1]], pts[idx[3]]
    n = np.cross(b - a, d - a)
    assert n[2] > 0, f"first quad normal points down: {n}"


def test_mesh_works_for_flat_field():
    f = FIELDS.create("flat", tile_size=256.0, resolution=8)
    pts = f.mesh_points()
    assert pts.shape == (64, 3)
    assert bool((pts[:, 2] == 0.0).all()), "flat water mesh is not flat"


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


# ------------------------------------------------- pressure-based buoyancy

FLAT = lambda x, y: np.zeros_like(np.asarray(x, dtype=float))


def _press(**kw):
    v, tri = box_hull(2.0, 1.0, 0.5)
    return FORCES.create("buoyancy_pressure", vertices=v, triangles=tri, **kw)


def _at(model, z, height_fn=FLAT, quat=(1.0, 0.0, 0.0, 0.0)):
    class _F:
        height = staticmethod(height_fn)
    body = BodyState(position=np.array([0.0, 0.0, float(z)]),
                     orientation=np.array(quat, dtype=float))
    return model.wrench(body, _F(), 0.0)


def test_quat_to_matrix():
    assert np.allclose(quat_to_matrix([1, 0, 0, 0]), np.eye(3)), "identity quaternion"
    import math
    a = math.radians(90) / 2
    R = quat_to_matrix([math.cos(a), 0.0, 0.0, math.sin(a)])    # 90 deg about z
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12), f"90 deg yaw gave {R @ [1,0,0]}"
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12), "rotation must be orthonormal"


def test_pressure_needs_a_hull():
    try:
        FORCES.create("buoyancy_pressure")
    except (ValueError, TypeError):
        return
    raise AssertionError("a pressure model with no hull should not construct")


def test_pressure_archimedes_exact():
    """No calibration constant: the pressure integral IS rho g V, at every draft."""
    m = _press()
    for centre_z, draft in ((0.15, 0.10), (0.05, 0.20), (0.0, 0.25), (-0.05, 0.30)):
        want = WATER_DENSITY * GRAVITY * (2.0 * 1.0 * draft)
        got = float(_at(m, centre_z).force[2])
        assert_close(got, want, 1e-6 * want, f"buoyancy at draft {draft} m")


def test_pressure_dry_hull_is_zero():
    assert np.allclose(_at(_press(), 1.0).as_array(), np.zeros(6)), "airborne hull has force"


def test_pressure_fully_submerged_is_capped():
    """Deeper than full immersion must not keep adding force - that is Archimedes."""
    m = _press()
    full = WATER_DENSITY * GRAVITY * (2.0 * 1.0 * 0.5)
    for z in (-0.25, -1.0, -50.0):
        assert_close(float(_at(m, z).force[2]), full, 1e-6 * full, f"submerged at z={z}")


def test_pressure_flat_water_has_no_horizontal_force():
    f = _at(_press(), 0.05).force
    assert abs(float(f[0])) < 1e-9 and abs(float(f[1])) < 1e-9, f"flat water pushed sideways: {f}"


def test_pressure_slope_gives_horizontal_force():
    """The headline capability: Fx/Fz equals the water slope. ASTRA cannot do this."""
    m = _press()
    for slope in (0.05, 0.10, 0.20):
        w = _at(m, 0.05, height_fn=lambda x, y, s=slope: s * np.asarray(x, dtype=float))
        ratio = float(w.force[0]) / float(w.force[2])
        assert_close(ratio, -slope, 0.02 * slope, f"Fx/Fz on a {slope} slope")


def test_pressure_rotation_does_not_change_displaced_volume():
    import math
    m = _press()
    full = WATER_DENSITY * GRAVITY * 1.0
    for deg in (0, 15, 30, 45):
        a = math.radians(deg) / 2
        q = (math.cos(a), 0.0, math.sin(a), 0.0)
        # The wrench is body-frame, so a pitched hull splits the upward world
        # force across body x and z. The magnitude is what must be invariant.
        got = float(np.linalg.norm(_at(m, -1.0, quat=q).force))
        assert_close(got, full, 1e-6 * full, f"fully submerged at {deg} deg pitch")


def test_pressure_moment_small_for_a_level_box():
    """A level box should have no moment, and very nearly does.

    Pressure is evaluated at each triangle's CENTROID, which gives the exact
    force but a slightly high application point - the resultant really acts at
    the centre of pressure, which sits lower on a face whose pressure varies.
    So a coarse hull shows a small spurious moment that converges away under
    refinement: 27.8 mm of equivalent arm at 12 triangles, 1.0 mm at 768.
    Implementing the centre-of-pressure correction removes it outright.
    """
    w = _at(_press(), 0.05)
    arm = float(np.linalg.norm(w.moment)) / float(w.force[2])
    assert arm < 0.03 * 2.0, f"moment arm {arm * 1000:.1f} mm on a 2 m hull is too large"


# --------------------------------------------------- usd conversion layer

from mmws.wave_sim.integration import (                  # noqa: E402
    body_state_from_usd, quat_to_wxyz, triangulate,
)


class _FakeGfQuat:
    """Stands in for pxr.Gf.Quatd, which keeps the scalar part separate."""

    def __init__(self, w, xyz):
        self._w, self._i = w, xyz

    def GetReal(self):
        return self._w

    def GetImaginary(self):
        return self._i


def test_quat_from_gf_object():
    """Gf.Quatd splits w from (x,y,z). Reassembling it in the wrong order
    still normalises and still rotates - just not the way you meant."""
    q = quat_to_wxyz(_FakeGfQuat(0.7071, (0.0, 0.0, 0.7071)))
    assert np.allclose(q, [0.7071, 0.0, 0.0, 0.7071]), f"got {q}"


def test_quat_from_sequence():
    assert np.allclose(quat_to_wxyz([1, 0, 0, 0]), [1, 0, 0, 0])
    try:
        quat_to_wxyz([1, 0, 0])
    except ValueError:
        return
    raise AssertionError("a 3-element quaternion should be rejected")


def test_triangulate_quads():
    """A USD mesh may hold quads; the force model needs triangles."""
    t = triangulate([4, 4], [0, 1, 2, 3, 4, 5, 6, 7])
    assert t.shape == (4, 3), f"two quads should give four triangles, got {t.shape}"
    assert np.array_equal(t[:2], [[0, 1, 2], [0, 2, 3]]), f"fan order wrong: {t[:2]}"


def test_triangulate_rejects_mismatch():
    try:
        triangulate([4, 4], [0, 1, 2, 3])
    except ValueError:
        return
    raise AssertionError("counts not matching the index list should be rejected")


def test_body_state_converts_degrees_to_radians():
    """USD reports angular velocity in DEGREES per second."""
    b = body_state_from_usd([0, 0, 0], [1, 0, 0, 0], angular_velocity_deg=[0, 0, 180.0])
    assert_close(float(b.angular_velocity[2]), math.pi, 1e-9, "180 deg/s in rad/s")


def test_body_state_rotates_into_the_body_frame():
    """USD velocities are world frame; BodyState is body frame."""
    a = math.radians(90) / 2
    yaw90 = [math.cos(a), 0.0, 0.0, math.sin(a)]
    b = body_state_from_usd([0, 0, 0], yaw90, linear_velocity=[1.0, 0.0, 0.0])
    assert np.allclose(b.linear_velocity, [0.0, -1.0, 0.0], atol=1e-9), \
        f"world +x under 90 deg yaw should be body -y, got {b.linear_velocity}"


def test_body_state_identity_is_a_passthrough():
    b = body_state_from_usd([1, 2, 3], [1, 0, 0, 0], linear_velocity=[4, 5, 6], mass=28.0)
    assert np.allclose(b.position, [1, 2, 3]) and np.allclose(b.linear_velocity, [4, 5, 6])
    assert b.mass == 28.0


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
    ("mesh: point shape and spacing", test_mesh_points_shape_and_spacing, True),
    ("mesh: z matches height()", test_mesh_points_match_height, True),
    ("mesh: topology is valid", test_mesh_topology_valid, True),
    ("mesh: topology is static", test_mesh_topology_is_static, True),
    ("mesh: winding faces up", test_mesh_winding_faces_up, True),
    ("mesh: works for the flat field", test_mesh_works_for_flat_field, True),
    ("quat_to_matrix", test_quat_to_matrix, True),
    ("pressure: rejects a missing hull", test_pressure_needs_a_hull, True),
    ("pressure: Archimedes exact at every draft", test_pressure_archimedes_exact, True),
    ("pressure: dry hull gives zero", test_pressure_dry_hull_is_zero, True),
    ("pressure: submerged force is capped", test_pressure_fully_submerged_is_capped, True),
    ("pressure: flat water has no sideways push", test_pressure_flat_water_has_no_horizontal_force, True),
    ("pressure: slope gives horizontal force", test_pressure_slope_gives_horizontal_force, True),
    ("pressure: rotation preserves displacement", test_pressure_rotation_does_not_change_displaced_volume, True),
    ("pressure: level box has a near-zero moment", test_pressure_moment_small_for_a_level_box, True),
    # --- stubs you are going to implement -------------------------------------
    ("JONSWAP: Hs round-trip", test_jonswap_hs_roundtrip, True),
    ("JONSWAP: gamma=1 equals PM", test_jonswap_reduces_to_pm, True),
    ("JONSWAP: gamma=3.3 sharpens peak", test_jonswap_sharper_than_pm, True),
    ("ECKV: peak matches the wind", test_eckv_peak_matches_the_wind, True),
    ("ECKV: omega and k integrals agree", test_eckv_frequency_and_wavenumber_agree, True),
    ("ECKV: Hs grows with wind", test_eckv_grows_with_wind, True),
    ("ECKV: S(0) finite and zero", test_eckv_zero_frequency_is_finite, True),
    ("usd: quaternion from a Gf object", test_quat_from_gf_object, True),
    ("usd: quaternion from a sequence", test_quat_from_sequence, True),
    ("usd: triangulate quads", test_triangulate_quads, True),
    ("usd: triangulate rejects a mismatch", test_triangulate_rejects_mismatch, True),
    ("usd: degrees become radians", test_body_state_converts_degrees_to_radians, True),
    ("usd: velocities rotate into the body frame", test_body_state_rotates_into_the_body_frame, True),
    ("usd: identity orientation passes through", test_body_state_identity_is_a_passthrough, True),
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
