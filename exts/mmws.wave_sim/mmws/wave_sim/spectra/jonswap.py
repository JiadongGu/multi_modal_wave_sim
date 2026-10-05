"""JONSWAP spectrum - YOUR IMPLEMENTATION.

What it is
----------
JONSWAP (Joint North Sea Wave Project, Hasselmann et al. 1973) is
Pierson-Moskowitz with a peak-sharpening bump. A *developing* sea - one still
being driven by wind over limited fetch - has more of its energy piled up near
the peak than a fully developed sea does, so you multiply PM by an extra factor
that is largest exactly at omega_p:

    S(omega) = S_pm(omega) * gamma ** r(omega) * alpha_norm

    r(omega) = exp( -(omega - omega_p)^2 / (2 sigma^2 omega_p^2) )

    sigma = 0.07  for omega <= omega_p        (left side of the peak)
            0.09  for omega >  omega_p        (right side)

`gamma` is the peak enhancement; gamma = 1 collapses exactly to PM, gamma = 3.3
is the North Sea average and the usual default. ASTRA's disabled NVIDIA module
used JONSWAP with a fetch parameter, which is the equivalent form where gamma
and omega_p are derived from wind speed and fetch distance instead of given.

The trap
--------
Multiplying by gamma^r *adds* energy, so the spectrum no longer integrates to
Hs^2/16 and `hs_from_spectrum()` comes back too high. Hasselmann's normalising
factor restores it:

    alpha_norm = 1 - 0.287 ln(gamma)

Leave it out and every sea state in the simulator is quietly wrong by ~15% at
gamma = 3.3. The self-test will catch it.

What to write
-------------
1. `__init__`: take `gamma` alongside the base parameters, validate it is >= 1,
   call `super().__init__(...)` with the rest.
2. `energy`: evaluate PM (compose a `PiersonMoskowitz` instance, or inline the
   formula - composing is cleaner), then apply the sigma switch, gamma^r and
   alpha_norm. Use `np.where` for the sigma switch so arrays work.
3. Delete the `raise` below.
4. Run `python3 tools/selftest.py` and confirm the Hs check passes for
   gamma = 1.0 and gamma = 3.3.

Reference: the verified formulation is in
`asv_wave_sim/doc/isaac_sim_port_spec.md`.
"""

from __future__ import annotations

import numpy as np

from ..registry import SPECTRA
from .base import WaveSpectrum


@SPECTRA.register("jonswap")
class Jonswap(WaveSpectrum):
    """Developing sea with a sharpened peak. Reduces to PM at gamma = 1."""

    def __init__(self, gamma: float = 3.3, **kwargs) -> None:
        super().__init__(**kwargs)
        if gamma < 1.0:
            raise ValueError(f"gamma must be >= 1 (1 means no peak enhancement), got {gamma}")
        self.gamma = float(gamma)

    def energy(self, omega):
        # TODO(jiadong): implement. See the module docstring for the formula,
        # the sigma switch and the alpha_norm factor you must not forget.
        raise NotImplementedError(
            "JONSWAP energy() is not implemented yet - see the module docstring"
        )
