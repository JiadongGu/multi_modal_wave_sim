"""The wave spectrum contract.

A spectrum answers two separate questions:

    energy(omega)             variance density at this frequency  [m^2 s / rad]
    spreading(theta, omega)   how that energy spreads in angle    [1 / rad]

and their product is the directional spectrum used to seed a wave field.

Why spreading is in the base class
----------------------------------
ASTRA's active wave model was two axis-aligned sine waves::

    eta(x, y, t) = A sin(2 pi f x + t) + A sin(2 pi f y + t)

so the sea had a permanent 90-degree crossed pattern, no wind direction and no
notion of how tightly energy clusters around it.  Putting `spreading` in the
abstract base with a working default means no subclass can quietly omit it: a
new spectrum gets directional behaviour whether or not its author thought
about it, and overriding is a deliberate act.

Conventions
-----------
* `omega` is angular frequency [rad/s], not Hz. Mixing the two is the single
  most common error in spectrum code; there are no Hz anywhere in this package.
* `theta` is direction [rad], measured in the same frame as `wind_dir`.
* Subclasses implement `energy` only. Everything else is provided.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import lgamma, pi, sqrt, exp

import numpy as np

# numpy renamed trapz -> trapezoid in 2.0. Isaac Sim 5.1 and 6.1 may ship
# either, so bind once here rather than pinning a numpy version.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class WaveSpectrum(ABC):
    """Base class for all wave spectra.

    Parameters
    ----------
    hs:
        Significant wave height [m]. Defined as 4 * std(surface elevation),
        and historically "the average of the highest third of the waves".
    tp:
        Peak period [s] - the period carrying the most energy.
    wind_dir:
        Mean wave propagation direction [rad].
    spread_s:
        Exponent of the cos-2s directional model. Larger is a narrower,
        more unidirectional sea. s = 5 is a reasonable wind-sea default;
        a long swell is more like s = 15-25.
    """

    # Set by Registry.register(); the fallback keeps an unregistered subclass
    # from having no name at all.
    name: str = "<unregistered>"

    def __init__(
        self,
        hs: float = 1.0,
        tp: float = 5.0,
        wind_dir: float = 0.0,
        spread_s: float = 5.0,
    ) -> None:
        if hs < 0.0:
            raise ValueError(f"hs must be non-negative, got {hs}")
        if tp <= 0.0:
            raise ValueError(f"tp must be positive, got {tp}")
        if spread_s <= 0.0:
            raise ValueError(f"spread_s must be positive, got {spread_s}")
        self.hs = float(hs)
        self.tp = float(tp)
        self.wind_dir = float(wind_dir)
        self.spread_s = float(spread_s)

    # ---------------------------------------------------------------- frequency

    @property
    def omega_p(self) -> float:
        """Peak angular frequency [rad/s]."""
        return 2.0 * pi / self.tp

    @abstractmethod
    def energy(self, omega):
        """Variance density S(omega) in [m^2 s / rad].

        Must accept a scalar or a numpy array and return the same shape, and
        must return 0 (not nan, not inf) at omega = 0.
        """

    # -------------------------------------------------------------- directional

    def spreading(self, theta, omega=None):
        """cos-2s directional spreading, normalised so it integrates to 1.

            D(theta) = N(s) * cos^(2s)((theta - wind_dir) / 2)

        with N(s) = Gamma(s+1) / (2 sqrt(pi) Gamma(s+1/2)).

        `omega` is accepted and ignored here. A real sea spreads more widely
        away from the peak, so a subclass may override this with a
        frequency-dependent s; the signature exists so that is not a breaking
        change later.
        """
        s = self.spread_s
        # Wrap into [-pi, pi] so cos(half-angle) is non-negative and the
        # fractional power never sees a negative base.
        d = np.asarray(theta, dtype=float) - self.wind_dir
        d = (d + pi) % (2.0 * pi) - pi
        # lgamma rather than Gamma/2**(2s) directly: the closed form overflows
        # for large s, this one does not.
        norm = exp(lgamma(s + 1.0) - lgamma(s + 0.5)) / (2.0 * sqrt(pi))
        return norm * np.cos(0.5 * d) ** (2.0 * s)

    def directional(self, omega, theta):
        """Directional spectrum S(omega, theta) = S(omega) * D(theta; omega)."""
        return np.asarray(self.energy(omega)) * np.asarray(
            self.spreading(theta, omega)
        )

    # ----------------------------------------------------------------- moments

    def moment(self, n: int = 0, omega_max: float | None = None, count: int = 4096):
        """n-th spectral moment m_n = integral omega^n S(omega) d(omega).

        Integrated numerically on a uniform grid, which is accurate enough for
        validation and keeps subclasses from each needing an analytic moment.
        m0 is the variance of the surface elevation; m2 relates to mean period.
        """
        if omega_max is None:
            # Five times the peak captures essentially all the energy of every
            # spectrum here; they fall off as omega^-5 or faster.
            omega_max = 5.0 * self.omega_p
        omega = np.linspace(1e-6, omega_max, count)
        return float(_trapezoid(omega**n * np.asarray(self.energy(omega)), omega))

    def hs_from_spectrum(self, **kwargs) -> float:
        """Hs recovered from the spectrum itself: Hs = 4 sqrt(m0).

        This should come back equal to the `hs` passed to the constructor. If it
        does not, the spectrum formula is wrong or unnormalised - which is
        exactly the bug the self-test checks for.
        """
        return 4.0 * sqrt(max(self.moment(0, **kwargs), 0.0))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, hs={self.hs}, "
            f"tp={self.tp}, wind_dir={self.wind_dir}, spread_s={self.spread_s})"
        )
