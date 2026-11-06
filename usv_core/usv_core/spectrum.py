"""From a wave spectrum to concrete waves, one sea per environment.

A spectrum describes a sea statistically: how much energy sits at each
frequency, S(omega), and how it spreads around the mean direction, D(theta).
It does not say how high the water is anywhere. To get a surface we draw K
individual waves whose sum has that energy distribution:

    eta(x, y, t) = sum_i  a_i sin(k_i (x cos theta_i + y sin theta_i) - omega_i t + phi_i)

How the K waves are drawn
-------------------------
Frequencies   The band [band_lo, band_hi] * omega_p is cut into K equal bins and
              one frequency is drawn uniformly inside each bin. Random within a
              bin (not the bin centre) so the sea never repeats exactly: on an
              even grid every wave period divides a common period and the whole
              surface comes back to the same shape, which a policy could learn.
Amplitudes    a_i = sqrt(2 E_i), E_i the energy of the whole bin (S integrated
              over it). A sine of amplitude a has variance a^2 / 2, so each wave
              carries exactly its bin's energy. Integrating, rather than reading
              S at the drawn frequency, matters for JONSWAP: its peak is about
              as narrow as one bin at K = 32, and a single point can miss it.
Directions    One direction per wave, drawn from D(theta) itself (inverse CDF).
              Drawing directions from the spreading function, instead of
              putting a grid of directions under every frequency, keeps K
              waves total instead of K x K_theta.
Wavenumbers   k_i = omega_i^2 / g, the deep-water dispersion relation.
Phases        Uniform in [0, 2 pi). Without random phases every crest lines up
              at t = 0 and the sea starts as one freak wave.
Hs            Significant height is Hs = 4 sqrt(variance) = 4 sqrt(sum a_i^2 / 2).
              With `exact_hs` the amplitudes are rescaled so this equals the
              requested Hs exactly. That absorbs three small errors: the band
              cut-off, the bin discretisation, and the JONSWAP normalisation,
              which is the Goda approximation (about 0.05 % at gamma = 3.3).

Each environment draws its own Hs, Tp, heading and spreading from `SeaCfg`
ranges, so the result is [N, K]. `resample(env_ids)` redraws only those rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .constants import GRAVITY

# --------------------------------------------------------------------- spectra


def pierson_moskowitz(omega: torch.Tensor, hs: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    """PM variance density S(omega) [m^2 s/rad]. hs/tp broadcast against omega; 0 at omega <= 0.

        S = (5/16) Hs^2 wp^4 w^-5 exp(-(5/4) (wp/w)^4)
    """
    wp = 2.0 * math.pi / tp
    safe = torch.where(omega > 0, omega, torch.ones_like(omega))
    s = (5.0 / 16.0) * hs**2 * wp**4 * safe**-5 * torch.exp(-1.25 * (wp / safe) ** 4)
    return torch.where(omega > 0, s, torch.zeros_like(s))


def jonswap(omega: torch.Tensor, hs: torch.Tensor, tp: torch.Tensor, gamma: float | torch.Tensor = 3.3) -> torch.Tensor:
    """JONSWAP: PM with the peak sharpened by gamma, a sea still growing under the wind.

    Same formula as the extension's spectra/jonswap.py, including its Goda
    normalisation 1 - 0.287 ln(gamma).
    """
    gamma = torch.as_tensor(gamma, dtype=omega.dtype, device=omega.device)
    wp = 2.0 * math.pi / tp
    sigma = torch.where(omega <= wp, torch.full_like(omega, 0.07), torch.full_like(omega, 0.09))
    r = torch.exp(-((omega - wp) ** 2) / (2.0 * sigma**2 * wp**2))
    return pierson_moskowitz(omega, hs, tp) * gamma**r * (1.0 - 0.287 * torch.log(gamma))


def cos2s_spreading(theta: torch.Tensor, mean_dir: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """cos-2s directional spreading D(theta) [1/rad], integrating to 1 over a full circle.

        D = Gamma(s+1) / (2 sqrt(pi) Gamma(s+1/2)) * cos^(2s)((theta - mean_dir) / 2)
    """
    d = torch.remainder(theta - mean_dir + math.pi, 2.0 * math.pi) - math.pi
    norm = torch.exp(torch.lgamma(s + 1.0) - torch.lgamma(s + 0.5)) / (2.0 * math.sqrt(math.pi))
    return norm * torch.cos(0.5 * d) ** (2.0 * s)


# ----------------------------------------------------------------------- config


@dataclass
class SeaCfg:
    """Ranges each environment's sea is drawn from. (lo, hi) pairs are uniform; lo == hi pins a value.

    hs        significant wave height [m]
    tp        peak period [s]
    heading   mean wave direction [rad], in the env-local frame
    spread_s  cos-2s exponent: about 5 for a wind sea, 15-25 for a long swell
    gamma     JONSWAP peak enhancement (1 = Pierson-Moskowitz)
    """

    hs: tuple[float, float] = (1.0, 3.0)
    tp: tuple[float, float] = (6.0, 10.0)
    heading: tuple[float, float] = (-math.pi, math.pi)
    spread_s: tuple[float, float] = (5.0, 5.0)
    gamma: float = 3.3
    num_waves: int = 32
    band_lo: float = 0.6          # lowest frequency, as a multiple of omega_p
    band_hi: float = 3.0          # highest frequency, as a multiple of omega_p
    exact_hs: bool = True
    gravity: float = GRAVITY


# ------------------------------------------------------------------- components


@dataclass
class WaveComponents:
    """K waves for each of N environments. All [N, K] except the per-sea parameters, which are [N].

    amp, omega, k, direction, phase    the waves
    cos_dir, sin_dir                   cached, used in every height evaluation
    hs, tp, heading, spread_s          the sea each row was drawn from
    """

    amp: torch.Tensor
    omega: torch.Tensor
    k: torch.Tensor
    direction: torch.Tensor
    cos_dir: torch.Tensor
    sin_dir: torch.Tensor
    phase: torch.Tensor
    hs: torch.Tensor
    tp: torch.Tensor
    heading: torch.Tensor
    spread_s: torch.Tensor

    @property
    def num_envs(self) -> int:
        return self.amp.shape[0]

    def significant_height(self) -> torch.Tensor:
        """[N] Hs implied by the drawn waves: 4 sqrt(sum a^2 / 2)."""
        return 4.0 * torch.sqrt((self.amp**2).sum(-1) / 2.0)

    def copy_rows_(self, env_ids: torch.Tensor, other: "WaveComponents") -> None:
        """Overwrite rows env_ids in place with `other` (which has len(env_ids) rows)."""
        for name in self.__dataclass_fields__:
            getattr(self, name)[env_ids] = getattr(other, name)


def _uniform(lo_hi: tuple[float, float], n: int, gen, device, dtype) -> torch.Tensor:
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=device, dtype=dtype)


def _sample_directions(heading: torch.Tensor, s: torch.Tensor, k: int, gen, grid: int = 721) -> torch.Tensor:
    """[N, K] directions drawn from cos-2s around each env's heading, by inverse CDF on a grid."""
    n = heading.shape[0]
    rel = torch.linspace(-math.pi, math.pi, grid, device=heading.device, dtype=heading.dtype)
    pdf = cos2s_spreading(rel.expand(n, grid), torch.zeros_like(heading)[:, None], s[:, None])
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = cdf / cdf[:, -1:]
    u = torch.rand(n, k, generator=gen, device=heading.device, dtype=heading.dtype)
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp(1, grid - 1)
    c0, c1 = cdf.gather(1, idx - 1), cdf.gather(1, idx)
    t = ((u - c0) / (c1 - c0).clamp_min(1e-12)).clamp(0, 1)
    offset = rel[idx - 1] + t * (rel[1] - rel[0])
    return heading[:, None] + offset


def sample_components(
    cfg: SeaCfg,
    num_envs: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> WaveComponents:
    """Draw a sea for each of num_envs environments. Pass a seeded generator for reproducible seas."""
    n, K = num_envs, cfg.num_waves
    hs = _uniform(cfg.hs, n, generator, device, dtype)
    tp = _uniform(cfg.tp, n, generator, device, dtype)
    heading = _uniform(cfg.heading, n, generator, device, dtype)
    spread_s = _uniform(cfg.spread_s, n, generator, device, dtype)

    # Frequencies: one random draw inside each of K equal bins of [lo, hi] * omega_p.
    wp = 2.0 * math.pi / tp
    lo, hi = cfg.band_lo * wp, cfg.band_hi * wp                                   # [N]
    d_omega = (hi - lo) / K                                                       # [N]
    bin_start = lo[:, None] + d_omega[:, None] * torch.arange(K, device=device, dtype=dtype)
    omega = bin_start + d_omega[:, None] * torch.rand(n, K, generator=generator, device=device, dtype=dtype)

    # Energy of each whole bin (midpoint rule on M sub-points), not S at the drawn frequency:
    # the JONSWAP peak is about as narrow as one bin, so a single point can miss it.
    M = 8
    sub = bin_start[..., None] + d_omega[:, None, None] * (torch.arange(M, device=device, dtype=dtype) + 0.5) / M
    bin_energy = jonswap(sub, hs[:, None, None], tp[:, None, None], cfg.gamma).mean(-1) * d_omega[:, None]
    amp = torch.sqrt(2.0 * bin_energy)
    if cfg.exact_hs:
        drawn = 4.0 * torch.sqrt((amp**2).sum(-1) / 2.0)
        amp = amp * (hs / drawn.clamp_min(1e-12))[:, None]

    direction = _sample_directions(heading, spread_s, K, generator)
    phase = 2.0 * math.pi * torch.rand(n, K, generator=generator, device=device, dtype=dtype)

    return WaveComponents(
        amp=amp,
        omega=omega,
        k=omega**2 / cfg.gravity,
        direction=direction,
        cos_dir=torch.cos(direction),
        sin_dir=torch.sin(direction),
        phase=phase,
        hs=hs,
        tp=tp,
        heading=heading,
        spread_s=spread_s,
    )
