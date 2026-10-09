"""ECKV spectrum - YOUR IMPLEMENTATION.

What it is
----------
Elfouhaily, Chapron, Katsaros and Vandemark (1997), "A unified directional
spectrum for long and short wind-driven waves". Unlike PM and JONSWAP it is
written in *wavenumber* rather than frequency, and it spans the whole range
from long gravity waves down to centimetre capillary waves by summing two
separate regimes:

    S(k) = (B_low + B_high) / k^3

`B_low` is the gravity-wave side, shaped by wind speed at 10 m (U10) and the
inverse wave age; `B_high` is the capillary side, driven by friction velocity
through a drag law. This is the spectrum `asv_wave_sim` uses, so implementing
it is what makes a like-for-like comparison against the Gazebo baseline
possible - which is the whole point of the port.

Why it matters here
-------------------
PM and JONSWAP are parameterised by (Hs, Tp): you tell them what sea you want.
ECKV is parameterised by (U10, fetch): you tell it what *wind* there is and it
tells you what sea results. That is the physically-driven direction, and it is
what makes "set the wind to 12 m/s" a meaningful command in this simulator.

What to write
-------------
This is the long one. Mind two things in particular:

* It is a wavenumber spectrum, so either convert to S(omega) via the deep-water
  Jacobian d(k)/d(omega) = 2 omega / g, or widen the base class to allow
  wavenumber-native spectra. The second is more honest; the first is less work.
  Decide deliberately, and write down which you chose.
* ECKV has its own built-in directional spreading, so overriding `spreading()`
  rather than inheriting cos-2s is likely the right call.

Reference: the formulation, every constant and a numerically verified
cross-check against the Gazebo implementation is in
`asv_wave_sim/doc/isaac_sim_port_spec.md` section 4. That document exists
precisely so this file can be written without re-deriving anything.
"""

from __future__ import annotations

import math

import numpy as np

from ..globals import GRAVITY
from ..registry import SPECTRA
from .base import WaveSpectrum

C_D10N = 0.00144      # neutral drag coefficient at 10 m
K_M = 370.0           # capillary peak wavenumber [rad/m]
C_M = 0.23            # minimum phase speed [m/s]


@SPECTRA.register("eckv")
class Eckv(WaveSpectrum):
    """Unified wind-driven spectrum, parameterised by wind rather than Hs."""

    def __init__(self, u10: float = 10.0, omega_c: float = 0.84,
                 fetch: float = 25e3, **kwargs) -> None:
        super().__init__(**kwargs)
        if u10 < 0.0:
            raise ValueError(f"u10 must be non-negative, got {u10}")
        if omega_c <= 0.0:
            raise ValueError(f"omega_c must be positive, got {omega_c}")
        if fetch <= 0.0:
            raise ValueError(f"fetch must be positive, got {fetch}")
        self.u10 = float(u10)
        self.omega_c = float(omega_c)
        self.fetch = float(fetch)

    @property
    def k_p(self) -> float:
        """Peak wavenumber [rad/m], from the wind rather than from Tp."""
        if self.u10 < 1e-8:
            return 0.0
        return (GRAVITY / self.u10**2) * self.omega_c**2

    @property
    def omega_p(self) -> float:
        """Peak angular frequency, overriding the base class's 2*pi/tp.

        ECKV is driven by wind, so its peak comes from k_p via the dispersion
        relation, not from a period the caller supplied. The base class uses
        omega_p to choose integration limits, so getting this right is what
        makes moment() and hs_from_spectrum() meaningful.
        """
        kp = self.k_p
        return math.sqrt(GRAVITY * kp) if kp > 0.0 else 0.0

    def wavenumber_energy(self, k):
        """S(k) in m^3 - elevation variance per unit wavenumber magnitude."""
        k = np.asarray(k, dtype=float)
        if self.u10 < 1e-8:
            return np.zeros_like(k)
        k_safe = np.where(k > 1e-8, k, 1.0)

        oc = self.omega_c
        u_star = math.sqrt(C_D10N) * self.u10
        gamma = 1.7 if oc < 1.0 else 1.7 + 6.0 * math.log10(oc)
        sigma = 0.08 * (1.0 + 4.0 * oc**-3.0)
        alpha_p = 0.006 * oc**0.55
        if u_star <= C_M:
            alpha_m = 0.01 * (1.0 + math.log(u_star / C_M))
        else:
            alpha_m = 0.01 * (1.0 + 3.0 * math.log(u_star / C_M))

        k_p = self.k_p
        c_p = math.sqrt(GRAVITY / k_p)
        c = np.sqrt((GRAVITY / k_safe) * (1.0 + (k_safe / K_M) ** 2))

        sqrt_ratio = np.sqrt(k_safe / k_p)
        L_pm = np.exp(-1.25 * (k_p / k_safe) ** 2)
        Gam = np.exp(-((sqrt_ratio - 1.0) ** 2) / (2.0 * sigma**2))
        J_p = gamma**Gam

        F_p = L_pm * J_p * np.exp(-0.3162 * oc * (sqrt_ratio - 1.0))
        F_m = L_pm * J_p * np.exp(-0.25 * (k_safe / K_M - 1.0) ** 2)

        B_l = 0.5 * alpha_p * (c_p / c) * F_p
        B_h = 0.5 * alpha_m * (C_M / c) * F_m
        return np.where(k > 1e-8, (B_l + B_h) / k_safe**3, 0.0)

    def energy(self, omega):
        """S(omega) in m^2 s/rad, converted from the wavenumber spectrum.

        Deep water gives k = omega^2 / g, so dk/domega = 2 omega / g and
        S(omega) = S(k) * dk/domega. Same change-of-variables bookkeeping as
        the wave field does in the other direction.
        """
        omega = np.asarray(omega, dtype=float)
        safe = np.where(omega > 0.0, omega, 1.0)
        k = safe**2 / GRAVITY
        s_k = np.asarray(self.wavenumber_energy(k))
        return np.where(omega > 0.0, s_k * 2.0 * safe / GRAVITY, 0.0)
