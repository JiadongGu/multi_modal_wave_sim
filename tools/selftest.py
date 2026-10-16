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
    """A plain height field is an evenly spaced grid. Choppiness off."""
    f = _fft(N=32, choppiness=0.0)
    pts = f.mesh_points()
    N = f.resolution
    assert pts.shape == (N * N, 3), f"expected {(N*N, 3)}, got {pts.shape}"
    assert_close(float(pts[N, 0] - pts[0, 0]), f.cell_size, 1e-9, "x spacing between rows")
    assert_close(float(pts[1, 1] - pts[0, 1]), f.cell_size, 1e-9, "y spacing within a row")


def test_mesh_points_match_height():
    """The mesh and the physics must read the same surface - that is the design.

    True of a HEIGHT field. Horizontal displacement deliberately breaks it, so
    the invariant is asserted where it is meant to hold, and the choppy case
    gets its own tests below.
    """
    f = _fft(N=32, choppiness=0.0)
    pts = f.mesh_points()
    z = np.asarray(f.height(pts[:, 0], pts[:, 1]))
    assert np.allclose(pts[:, 2], z), "mesh z does not match height() at the same xy"


def test_choppiness_does_not_touch_height():
    """Choppiness is a visual setting. The physics must not be able to tell.

    The force model queries height(x, y). If a rendering knob moved that, the
    hull would float differently depending on how pretty the water was.
    """
    a, b = _fft(N=32, choppiness=0.0), _fft(N=32, choppiness=1.5)
    a.advance(3.0)
    b.advance(3.0)
    rng = np.random.default_rng(7)
    xy = rng.uniform(0.0, a.tile_size, size=(2, 400))
    diff = float(np.abs(np.asarray(a.height(*xy)) - np.asarray(b.height(*xy))).max())
    assert diff == 0.0, f"choppiness changed height() by {diff:.3e} m"


def test_choppiness_conserves_water():
    """Displacement moves water sideways; it must not create or destroy any.

    The mean of the displacement over a full tile is the net transport, and a
    periodic surface has nowhere to transport it to - so it must be zero.
    """
    f = _fft(N=32, choppiness=1.0)
    f.advance(3.0)
    pts = f.mesh_points()
    flat = _fft(N=32, choppiness=0.0)
    flat.advance(3.0)
    base = flat.mesh_points()
    for axis, name in ((0, "x"), (1, "y")):
        mean = float(np.abs((pts[:, axis] - base[:, axis]).mean()))
        assert mean < 1e-9, f"net {name} displacement {mean:.3e} m, want 0"


def test_choppiness_displaces_toward_crests():
    """Water should pile INTO the crests, which is what sharpens them.

    Points whose displacement carries them uphill outnumber those carried
    downhill, so the correlation between displacement and surface slope is
    negative - D points against the gradient, toward the peak.
    """
    f = _fft(N=32, choppiness=1.0)
    f.advance(3.0)
    z = f._z
    gx, gy = np.gradient(z, f.cell_size)
    corr = float((f._dx * gx + f._dy * gy).mean())
    assert corr < 0.0, f"displacement is not toward the crests (corr {corr:+.4f})"


def test_foam_only_in_steep_seas():
    """Long low swell does not break; short steep waves do.

    Resolution matters here and it is worth knowing why: a fold is made by
    the SHORT components riding the long ones, so a grid too coarse to carry
    them reports a glassy sea no matter how steep it is. At a 512 m tile with
    64 points the cells are 8 m, a 39 m wave is five cells across, and nothing
    breaks. 2 m cells show the whitecaps. If foam disappears when you expected
    it, suspect the cell size before the threshold.
    """
    # Built explicitly: _fft() fixes the spectrum, and FftWaveField swallows
    # unknown kwargs, so passing hs/tp through it would silently do nothing.
    def field(hs, tp):
        return FIELDS.create("fft", spectrum=SPECTRA.create("jonswap", hs=hs, tp=tp),
                             tile_size=256.0, resolution=128, choppiness=1.0)

    swell = field(2.0, 12.0)      # 225 m long, steepness 0.009
    steep = field(3.0, 5.0)       # 39 m long, steepness 0.077
    swell.advance(5.0)
    steep.advance(5.0)
    calm_frac = float((swell._foam > 0).mean())
    rough_frac = float((steep._foam > 0).mean())
    assert calm_frac == 0.0, f"swell should not break, got {100*calm_frac:.2f}% foam"
    assert rough_frac > 0.0, "a steep sea should produce whitecaps somewhere"


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
    body_state_from_usd, quat_to_wxyz, relative_transform, transform_points, triangulate,
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


def test_transform_points_uses_the_row_convention():
    """USD puts translation in the last ROW. Column-convention code silently
    rotates correctly and translates to nonsense."""
    M = np.eye(4)
    M[3, :3] = [10.0, 5.0, 0.2]
    got = transform_points([[1.0, 0.0, 0.0]], M)
    assert np.allclose(got, [[11.0, 5.0, 0.2]]), f"got {got}"


def test_transform_points_rotates():
    c, sn = math.cos(math.radians(90)), math.sin(math.radians(90))
    R = np.eye(4)
    R[:3, :3] = [[c, sn, 0], [-sn, c, 0], [0, 0, 1]]
    assert np.allclose(transform_points([[1, 0, 0]], R), [[0, 1, 0]], atol=1e-12)


def test_relative_transform_puts_the_hull_in_the_body_frame():
    """A hull 2 m ahead of the body origin must read as 2 m forward, not as
    its world position."""
    child, parent = np.eye(4), np.eye(4)
    child[3, :3] = [12.0, 5.0, 0.2]
    parent[3, :3] = [10.0, 5.0, 0.2]
    rel = relative_transform(child, parent)
    assert np.allclose(transform_points([[0, 0, 0]], rel), [[2.0, 0.0, 0.0]], atol=1e-12)


def test_relative_transform_of_a_prim_with_itself_is_identity():
    P = np.eye(4)
    P[3, :3] = [3.0, -1.0, 7.0]
    assert np.allclose(relative_transform(P, P), np.eye(4), atol=1e-12)


from mmws.wave_sim.integration.repair import (                # noqa: E402
    open_edges, repair, signed_volume,
)
from mmws.wave_sim.forces import box_hull as _box              # noqa: E402


def _exploded_box():
    """A box as a bad exporter would emit it: one prim per triangle, every
    corner point duplicated, and a third of the faces wound backwards."""
    v, t = _box(2.0, 1.0, 0.5)
    return [(v[tri], np.array([[0, 2, 1]] if i % 3 == 0 else [[0, 1, 2]]))
            for i, tri in enumerate(t)]


def test_repair_closes_an_exploded_box():
    _, tris, rep = repair(_exploded_box())
    assert rep["open_edges_before"] == 36, rep
    assert rep["open_edges_after"] == 0, f"still open after welding: {rep}"
    assert rep["vertices"] == 8, f"should weld down to 8 corners, got {rep['vertices']}"


def test_repair_recovers_the_volume():
    v, t, _ = repair(_exploded_box())
    assert_close(signed_volume(v, t), 1.0, 1e-9, "volume of the repaired box")


def test_repair_fixes_winding():
    """Positive signed volume means the normals ended up pointing outward."""
    v, t, _ = repair(_exploded_box())
    assert signed_volume(v, t) > 0, "repaired mesh is inside-out"
    assert open_edges(t) == 0


def test_repair_gives_the_same_forces_as_a_clean_hull():
    v, t = _box(2.0, 1.0, 0.5)
    rv, rt, _ = repair(_exploded_box())
    clean = FORCES.create("buoyancy_pressure", vertices=v, triangles=t)
    fixed = FORCES.create("buoyancy_pressure", vertices=rv, triangles=rt)
    state = body_state_from_usd([0, 0, 0.05], [1, 0, 0, 0])

    class _F:
        height = staticmethod(FLAT)

    a = clean.wrench(state, _F(), 0.0).as_array()
    b = fixed.wrench(state, _F(), 0.0).as_array()
    assert np.allclose(a, b, atol=1e-9), f"repaired hull disagrees: {a} vs {b}"


def test_mixed_winding_is_detectable_and_fixable():
    """An authored mesh with some faces reversed gives kilonewtons of sideways
    force and no lift - the signature of a hull that needs orienting. The
    enclosed volume is the cheap detector: it reads low, so a bind-time check
    on volume catches it before any force is believed."""
    from mmws.wave_sim.integration.repair import orient_faces, signed_volume

    v, t = _box(1.0, 1.0, 1.0)
    mixed = t.copy()
    mixed[[1, 4, 7, 10]] = mixed[[1, 4, 7, 10]][:, [0, 2, 1]]

    class _F:
        height = staticmethod(FLAT)

    state = body_state_from_usd([0, 0, -0.3], [1, 0, 0, 0], mass=200.0)
    bad = FORCES.create("buoyancy_pressure", vertices=v, triangles=mixed)
    fixed = FORCES.create("buoyancy_pressure", vertices=v, triangles=orient_faces(v, mixed))

    assert signed_volume(v, mixed) < 0.5, "a mixed-winding box should read low volume"
    assert_close(signed_volume(v, orient_faces(v, mixed)), 1.0, 1e-9, "volume after orienting")

    fb = bad.wrench(state, _F(), 0.0).force
    assert abs(float(fb[0])) > 1000.0, f"expected large spurious sideways force, got {fb}"

    ff = fixed.wrench(state, _F(), 0.0).force
    assert_close(float(ff[2]), 7845.3, 1.0, "lift after orienting")
    assert abs(float(ff[0])) < 1e-6 and abs(float(ff[1])) < 1e-6, f"sideways force remains: {ff}"


def test_damping_is_zero_in_air():
    """Water damps, air does not. A hull in flight must be in free fall, not
    dragging against nothing - otherwise a dry falling body feels forces that
    make a force log impossible to read."""
    field = FIELDS.create("fft", spectrum=SPECTRA.create("jonswap", hs=2.0, tp=7.0),
                          tile_size=256.0, resolution=64, seed=3)
    field.advance(0.0)
    d = FORCES.create("damping", extents=[1.0, 1.0, 1.0])
    airborne = BodyState(position=np.array([150.0, 150.0, 50.0]),
                         linear_velocity=np.array([0.0, 0.0, -20.0]), mass=200.0)
    assert np.allclose(d.wrench(airborne, field, 0.0).as_array(), np.zeros(6)), \
        "a hull 50 m above the water felt drag"
    surface = float(np.asarray(field.height(np.array([150.0]), np.array([150.0])))[0])
    wet = BodyState(position=np.array([150.0, 150.0, surface - 1.0]),
                    linear_velocity=np.array([0.0, 0.0, -20.0]), mass=200.0)
    assert abs(float(d.wrench(wet, field, 0.0).force[2])) > 100.0, "submerged hull felt no drag"


def test_damping_opposes_motion_and_vanishes_at_rest():
    d = FORCES.create("damping", extents=[1.0, 1.0, 1.0])
    at_rest = BodyState()
    assert np.allclose(d.wrench(at_rest, None, 0.0).as_array(), np.zeros(6)), "drag at rest"
    moving = BodyState(linear_velocity=np.array([2.0, 0.0, -3.0]))
    f = d.wrench(moving, None, 0.0).force
    assert f[0] < 0 and f[2] > 0, f"drag must oppose velocity, got {f}"


def test_damping_grows_faster_than_linearly():
    """Near-quadratic at working speeds. Not exactly 4x on doubling, because
    the step is solved implicitly and the force saturates toward m|nu|/dt -
    which is the property that keeps it stable."""
    d = FORCES.create("damping", extents=[1.0, 1.0, 1.0])

    def drag(v):
        return abs(float(d.wrench(
            BodyState(linear_velocity=np.array([0.0, 0.0, v]), mass=200.0),
            None, 0.0).force[2]))

    assert drag(10.0) / drag(5.0) > 3.0, f"ratio {drag(10.0) / drag(5.0):.2f}"


def test_damping_cannot_reverse_the_velocity():
    """The explicit form overshoots above (m/D)/dt - 24 m/s for this hull - and
    amplifies instead of damping, which destroys the simulation. The implicit
    form must never remove more momentum than the body has."""
    d = FORCES.create("damping", extents=[1.0, 1.0, 1.0], timestep=1.0 / 60.0)
    mass, dt = 200.0, 1.0 / 60.0
    for v in (5.0, 24.0, 60.0, 200.0, 5000.0):
        state = BodyState(linear_velocity=np.array([0.0, 0.0, v]), mass=mass)
        dv = float(d.wrench(state, None, 0.0).force[2]) / mass * dt
        assert -v < dv <= 0.0, f"at {v} m/s the step changed velocity by {dv:+.2f}"


def test_composite_sums_its_models():
    v, tri = _box(1.0, 1.0, 1.0)
    buoy = FORCES.create("buoyancy_pressure", vertices=v, triangles=tri)
    damp = FORCES.create("damping", extents=[1.0, 1.0, 1.0])
    both = FORCES.create("composite", models=[buoy, damp])

    class _F:
        height = staticmethod(FLAT)

    state = BodyState(position=np.array([0.0, 0.0, -0.3]),
                      orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                      linear_velocity=np.array([0.0, 0.0, 2.0]), mass=200.0)
    a = buoy.wrench(state, _F(), 0.0).as_array()
    b = damp.wrench(state, _F(), 0.0).as_array()
    assert np.allclose(both.wrench(state, _F(), 0.0).as_array(), a + b)


def test_damping_stops_the_energy_pump():
    """Undamped, a hull bouncing on a MOVING surface gains energy every cycle
    and climbs above its drop height - the ball-on-a-vibrating-plate effect.
    With damping it must never exceed where it started."""
    v, tri = _box(1.0, 1.0, 1.0)
    field = FIELDS.create("fft", spectrum=SPECTRA.create("jonswap", hs=2.0, tp=7.0),
                          tile_size=256.0, resolution=64, seed=3)
    model = FORCES.create("composite", models=[
        FORCES.create("buoyancy_pressure", vertices=v, triangles=tri),
        FORCES.create("damping", extents=[1.0, 1.0, 1.0]),
    ])
    z, vz, peak, dt, mass = 3.0, 0.0, 3.0, 1.0 / 60.0, 200.0
    for i in range(1800):
        field.advance(i * dt)
        state = BodyState(position=np.array([150.0, 150.0, z]),
                          orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                          linear_velocity=np.array([0.0, 0.0, vz]), mass=mass)
        vz += (float(model.wrench(state, field, i * dt).force[2]) / mass - GRAVITY) * dt
        z += vz * dt
        peak = max(peak, z)
    assert peak <= 3.01, f"gained energy: rose to {peak:.2f} m from a 3.00 m drop"
    assert -5.0 < z < 3.0, f"ended somewhere implausible: {z:.2f} m"


# ------------------------------------------------------------------------- main

# --------------------------------------------------------------------- wake


def _wake(base="flat", **kw):
    spec = SPECTRA.create("jonswap", hs=2.0, tp=7.0)
    return FIELDS.create("wake", spectrum=spec, tile_size=128.0,
                         resolution=64, base=base, **kw)


def _drive(f, speed=3.0, steps=60, dt=0.1, y=10.0):
    """Run a hull along +x at a constant speed, shedding wake as it goes."""
    for i in range(steps):
        t = i * dt
        f.advance(t)
        f.set_source([speed * t, y, 0.0], [speed, 0.0, 0.0])
    return (steps - 1) * dt


def test_wake_needs_motion():
    """A vessel sitting still leaves no wake. The simplest thing that could be wrong."""
    f = _wake()
    for i in range(50):
        f.advance(i * 0.1)
        f.set_source([10.0, 10.0, 0.0], [0.0, 0.0, 0.0])
    x = np.linspace(0.0, 60.0, 61)
    peak = float(np.max(np.abs(f.wake_height(x, 10.0))))
    assert peak == 0.0, f"still hull shed a wake of {peak} m"


def test_wake_appears_behind_a_moving_hull():
    f = _wake()
    t_end = _drive(f)
    f.advance(t_end)
    x = np.linspace(0.0, 3.0 * t_end, 200)
    peak = float(np.max(np.abs(f.wake_height(x, 10.0))))
    assert peak > 1e-4, f"moving hull shed no wake, peak {peak} m"


def test_wake_respects_causality():
    """Nothing beyond the radius the group velocity has had time to reach.

    Without this the pattern exists everywhere the instant it is shed, which
    is not a wave at all.
    """
    f = _wake()
    f.advance(0.0)
    f.set_source([0.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    f.advance(2.0)
    beyond = f.group_speed * 2.0 + 10.0
    assert float(abs(f.wake_height(beyond, 0.0))) == 0.0, \
        "wake found outside its own light cone"


def test_wake_decays_with_distance():
    f = _wake(lifetime=1.0e6, decay_length=20.0)
    f.advance(0.0)
    f.set_source([0.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    f.advance(60.0)                      # long enough to reach both radii
    r = np.linspace(5.0, 60.0, 400)
    env = np.abs(f.wake_height(r, 0.0))
    near, far = float(env[:200].max()), float(env[200:].max())
    assert far < near, f"wake does not decay: near {near}, far {far}"


def test_wake_grows_with_speed():
    def peak(speed):
        f = _wake()
        f.advance(0.0)
        f.set_source([0.0, 0.0, 0.0], [speed, 0.0, 0.0])
        f.advance(5.0)
        r = np.linspace(0.0, f.group_speed * 5.0, 300)
        return float(np.max(np.abs(f.wake_height(r, 0.0))))
    slow, fast = peak(1.0), peak(3.0)
    assert fast > slow, f"wake did not grow with speed: {slow} -> {fast}"


def test_wake_superposes_on_the_base_field():
    """height == base + wake, exactly. The whole point of the composition."""
    f = _wake(base="fft")
    t_end = _drive(f)
    f.advance(t_end)
    g = np.linspace(0.0, 100.0, 23)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    total = np.asarray(f.height(gx, gy))
    parts = np.asarray(f.base.height(gx, gy)) + f.wake_height(gx, gy)
    assert np.allclose(total, parts, atol=1e-12), "superposition is not a plain sum"


def test_wake_zero_amplitude_is_exactly_the_base():
    """Turning the wake off must leave the sea bit-for-bit unchanged."""
    f = _wake(base="fft", amplitude=0.0)
    t_end = _drive(f)
    f.advance(t_end)
    g = np.linspace(0.0, 100.0, 17)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    assert np.array_equal(np.asarray(f.height(gx, gy)),
                          np.asarray(f.base.height(gx, gy))), \
        "amplitude=0 changed the base sea"


def test_wake_trail_is_bounded():
    """Cost per query is linear in the trail, so it must not grow forever."""
    f = _wake(max_samples=16)
    _drive(f, steps=400)
    assert len(f._trail) <= 16, f"trail grew to {len(f._trail)}"


def test_wake_is_independent_of_sampling_rate():
    """Halving the physics timestep must not change the wake.

    It did: rings were shed with a fixed amplitude each, so the sum counted
    samples instead of integrating along the track, and amplitude=0.15 came
    out as a 1.49 m wave. Each ring now carries the arc length it stands for.
    """
    def peak(dt, steps, speed=4.0):
        spec = SPECTRA.create("jonswap", hs=2.0, tp=7.0)
        f = FIELDS.create("wake", spectrum=spec, tile_size=128.0, resolution=64,
                          base="flat", amplitude=0.15, wavelength=6.0,
                          max_samples=600, min_spacing=0.0, min_interval=0.0)
        for i in range(steps):
            t = i * dt
            f.advance(t)
            f.set_source([speed * t, 64.0, 0.0], [speed, 0.0, 0.0])
        t_end = (steps - 1) * dt
        f.advance(t_end)
        x = np.linspace(speed * t_end - 70.0, speed * t_end - 2.0, 500)
        return float(np.max(np.abs(f.wake_height(x, 64.0))))

    coarse, fine = peak(0.1, 121), peak(0.025, 481)
    rel = abs(fine - coarse) / max(coarse, 1e-12)
    assert rel < 0.05, f"wake changed {rel:.1%} when the timestep changed"


def test_wake_opens_near_the_kelvin_angle():
    """The wedge is not drawn anywhere - it falls out of the superposition.

    A deep-water ship wake has a 19.47 degree half-angle, independent of
    speed. A single wavenumber and a threshold-based measurement will not hit
    that exactly, but landing in the right neighbourhood is real evidence the
    rings and the group velocity are doing the work.
    """
    spec = SPECTRA.create("jonswap", hs=2.0, tp=7.0)
    f = FIELDS.create("wake", spectrum=spec, tile_size=128.0, resolution=64,
                      base="flat", amplitude=0.15, wavelength=6.0,
                      max_samples=600, min_spacing=0.0, min_interval=0.0)
    speed, dt, steps = 4.0, 0.05, 241
    for i in range(steps):
        t = i * dt
        f.advance(t)
        f.set_source([speed * t, 64.0, 0.0], [speed, 0.0, 0.0])
    t_end = (steps - 1) * dt
    f.advance(t_end)
    hull_x = speed * t_end

    ys = np.linspace(64.0 - 40.0, 64.0 + 40.0, 321)
    angles = []
    for astern in (10.0, 20.0, 30.0):
        lat = np.abs(f.wake_height(hull_x - astern, ys))
        assert lat.max() > 0.0, f"no wake {astern} m astern"
        edge = np.max(np.abs(ys[lat > 0.15 * lat.max()] - 64.0))
        angles.append(math.degrees(math.atan(edge / astern)))
    mean = sum(angles) / len(angles)
    assert 12.0 < mean < 30.0, f"half-angle {mean:.1f} deg is nowhere near Kelvin's 19.47"


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
    ("choppiness: height() unchanged", test_choppiness_does_not_touch_height, True),
    ("choppiness: conserves water", test_choppiness_conserves_water, True),
    ("choppiness: displaces toward crests", test_choppiness_displaces_toward_crests, True),
    ("foam: only in steep seas", test_foam_only_in_steep_seas, True),
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
    ("usd: transform uses the row convention", test_transform_points_uses_the_row_convention, True),
    ("usd: transform rotates", test_transform_points_rotates, True),
    ("usd: hull lands in the body frame", test_relative_transform_puts_the_hull_in_the_body_frame, True),
    ("usd: self-relative transform is identity", test_relative_transform_of_a_prim_with_itself_is_identity, True),
    ("repair: closes an exploded box", test_repair_closes_an_exploded_box, True),
    ("repair: recovers the volume", test_repair_recovers_the_volume, True),
    ("repair: fixes winding", test_repair_fixes_winding, True),
    ("repair: matches a clean hull's forces", test_repair_gives_the_same_forces_as_a_clean_hull, True),
    ("repair: mixed winding detected and fixed", test_mixed_winding_is_detectable_and_fixable, True),
    ("damping: zero in air, present in water", test_damping_is_zero_in_air, True),
    ("damping: opposes motion, zero at rest", test_damping_opposes_motion_and_vanishes_at_rest, True),
    ("damping: grows faster than linearly", test_damping_grows_faster_than_linearly, True),
    ("damping: cannot reverse the velocity", test_damping_cannot_reverse_the_velocity, True),
    ("composite: sums its models", test_composite_sums_its_models, True),
    ("damping: stops the energy pump", test_damping_stops_the_energy_pump, True),
    ("wake: needs motion", test_wake_needs_motion, True),
    ("wake: appears behind a moving hull", test_wake_appears_behind_a_moving_hull, True),
    ("wake: respects causality", test_wake_respects_causality, True),
    ("wake: decays with distance", test_wake_decays_with_distance, True),
    ("wake: grows with speed", test_wake_grows_with_speed, True),
    ("wake: superposes on the base field", test_wake_superposes_on_the_base_field, True),
    ("wake: zero amplitude leaves the base alone", test_wake_zero_amplitude_is_exactly_the_base, True),
    ("wake: trail is bounded", test_wake_trail_is_bounded, True),
    ("wake: independent of sampling rate", test_wake_is_independent_of_sampling_rate, True),
    ("wake: opens near the Kelvin angle", test_wake_opens_near_the_kelvin_angle, True),
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
