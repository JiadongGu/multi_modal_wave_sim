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

from ..registry import SPECTRA
from .base import WaveSpectrum


@SPECTRA.register("eckv")
class Eckv(WaveSpectrum):
    """Unified wind-driven spectrum, parameterised by wind rather than Hs."""

    def __init__(self, u10: float = 10.0, fetch: float = 25e3, **kwargs) -> None:
        super().__init__(**kwargs)
        if u10 < 0.0:
            raise ValueError(f"u10 must be non-negative, got {u10}")
        if fetch <= 0.0:
            raise ValueError(f"fetch must be positive, got {fetch}")
        self.u10 = float(u10)
        self.fetch = float(fetch)

    def energy(self, omega):
        # TODO(jiadong): implement from isaac_sim_port_spec.md section 4.
        raise NotImplementedError(
            "ECKV energy() is not implemented yet - see the module docstring"
        )
