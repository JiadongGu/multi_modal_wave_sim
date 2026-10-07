from __future__ import annotations

import numpy as np

from  ..registry import SPECTRA
from .base import WaveSpectrum
from .pierson_moskowitz import PiersonMoskowitz

@SPECTRA.register("jonswap")
class Jonswap(WaveSpectrum):
    def __init__(self,gamma: float = 3.3, **kwargs) -> None:
        super().__init__(**kwargs)
        if gamma < 1.0:
            raise ValueError(
                f"gamma must be >= 1 (1 means no enhancement), got {gamma}"
            )
        self.gamma = float(gamma)
        self._pm = PiersonMoskowitz(
            hs=self.hs,tp=self.tp,wind_dir=self.wind_dir,spread_s=self.spread_s
        )
    def energy(self,omega):
        omega = np.asarray(omega,dtype=float)
        wp = self.omega_p
        base = np.asarray(self._pm.energy(omega))
        sigma = np.where(omega <= wp, 0.07, 0.09)
        r = np.exp(-((omega - wp) ** 2) / (2.0 * sigma**2 * wp**2))
        alpha_norm = 1.0 - 0.287 * np.log(self.gamma)
        return base * self.gamma**r * alpha_norm
