"""The sea socket, and the seas that plug into it.

Any sea answers the same questions about water at points, for all
environments at once. Row i of every query is answered with environment i's
sea. Points are env-local (conventions.py), metres, z up, z = 0 the mean
water level. A sea knows nothing about boats.

    height(xy, t)    [N, M, 2] -> [N, M]      surface elevation eta
    velocity(xyz, t) [N, M, 3] -> [N, M, 3]   water velocity (orbital motion)
    slope(xy, t)     [N, M, 2] -> [N, M, 2]   d(eta)/dx, d(eta)/dy
    sea_state()      -> dict of [N]           summary, for observations or parameter-driven models
    resample(ids)    new seas for those environments only

`t` is a number, or an [N] tensor when environments run on separate clocks.

Seas here: SineWaveField (sum of K waves per environment, spectrum.py) and
FlatWater. A recorded CFD sea, an FFT grid or anything else plugs in by
implementing the same methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .spectrum import SeaCfg, WaveComponents, sample_components


class WaveField(ABC):
    """The socket. Subclasses implement height, velocity and slope; the rest have sensible defaults."""

    num_envs: int

    @abstractmethod
    def height(self, xy: torch.Tensor, t) -> torch.Tensor:
        ...

    @abstractmethod
    def velocity(self, xyz: torch.Tensor, t) -> torch.Tensor:
        ...

    @abstractmethod
    def slope(self, xy: torch.Tensor, t) -> torch.Tensor:
        ...

    def sea_state(self) -> dict[str, torch.Tensor]:
        return {}

    def height_bound(self) -> torch.Tensor | None:
        """[N] a height the surface never exceeds in magnitude, or None if the sea cannot say."""
        return None

    def resample(self, env_ids: torch.Tensor | None = None) -> None:
        """New seas for env_ids (None: all). Seas with nothing to randomise ignore it."""


class FlatWater(WaveField):
    """Still water: zero height, velocity and slope everywhere. For still-water tests and as a baseline."""

    def __init__(self, num_envs: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.num_envs, self.device, self.dtype = num_envs, torch.device(device), dtype

    def height(self, xy, t):
        return torch.zeros(xy.shape[:-1], device=xy.device, dtype=xy.dtype)

    def velocity(self, xyz, t):
        return torch.zeros_like(xyz)

    def slope(self, xy, t):
        return torch.zeros_like(xy)

    def height_bound(self):
        return torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)

    def sea_state(self):
        z = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        return {"hs": z, "tp": z.clone(), "heading": z.clone()}


class SineWaveField(WaveField):
    """A sum of K linear waves per environment, drawn from a spectrum (spectrum.py).

    For wave i at a point (x, y, z) and time t, with phase

        psi_i = k_i (x cos theta_i + y sin theta_i) - omega_i t + phi_i

    deep-water linear (Airy) theory gives

        eta          = sum a_i sin(psi_i)
        u_horizontal = sum a_i omega_i e^(k_i z) sin(psi_i) (cos theta_i, sin theta_i)
        w_vertical   = -sum a_i omega_i e^(k_i z) cos(psi_i)          (= d eta / dt at z = 0)
        d eta / dx   = sum a_i k_i cos theta_i cos(psi_i),  likewise y with sin theta_i

    Water particles move in circles that shrink as e^(kz) with depth. Linear
    theory says nothing above z = 0 (inside a crest), so z is clamped to 0
    there and the factor never exceeds 1.

    Internally the phase is [N, M, K]: every point against every wave.
    """

    def __init__(
        self,
        cfg: SeaCfg | None = None,
        num_envs: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or SeaCfg()
        self.num_envs, self.device, self.dtype = num_envs, torch.device(device), dtype
        self._gen = torch.Generator(device=self.device).manual_seed(seed)
        self.waves = sample_components(self.cfg, num_envs, self.device, dtype, self._gen)

    @classmethod
    def from_components(cls, waves: WaveComponents) -> "SineWaveField":
        """A field with given waves, e.g. fitted from probe data. resample() still draws from the default SeaCfg."""
        obj = cls.__new__(cls)
        obj.cfg = SeaCfg(num_waves=waves.amp.shape[1])
        obj.num_envs, obj.device, obj.dtype = waves.num_envs, waves.amp.device, waves.amp.dtype
        obj._gen = torch.Generator(device=obj.device).manual_seed(0)
        obj.waves = waves
        return obj

    def _phase(self, xy: torch.Tensor, t) -> torch.Tensor:
        w = self.waves
        if torch.is_tensor(t):                                                   # per-env times [N] or a 0-d tensor
            t = t.to(xy)
            t = t.reshape(-1, 1, 1) if t.dim() == 1 else t
        # a plain number stays one: turning it into a GPU tensor would be a CPU->GPU copy (and a wait) per call
        along = xy[..., 0:1] * w.cos_dir[:, None, :] + xy[..., 1:2] * w.sin_dir[:, None, :]  # [N, M, K]
        return w.k[:, None, :] * along - w.omega[:, None, :] * t + w.phase[:, None, :]

    def height(self, xy, t):
        return (self.waves.amp[:, None, :] * torch.sin(self._phase(xy, t))).sum(-1)

    def velocity(self, xyz, t):
        w = self.waves
        psi = self._phase(xyz[..., :2], t)
        z = xyz[..., 2:3].clamp(max=0.0)                                        # [N, M, 1]
        speed = w.amp[:, None, :] * w.omega[:, None, :] * torch.exp(w.k[:, None, :] * z)  # [N, M, K]
        horiz = speed * torch.sin(psi)
        vx = (horiz * w.cos_dir[:, None, :]).sum(-1)
        vy = (horiz * w.sin_dir[:, None, :]).sum(-1)
        vz = -(speed * torch.cos(psi)).sum(-1)
        return torch.stack([vx, vy, vz], dim=-1)

    def slope(self, xy, t):
        w = self.waves
        g = w.amp[:, None, :] * w.k[:, None, :] * torch.cos(self._phase(xy, t))  # [N, M, K]
        return torch.stack([(g * w.cos_dir[:, None, :]).sum(-1), (g * w.sin_dir[:, None, :]).sum(-1)], dim=-1)

    def height_bound(self):
        return self.waves.amp.sum(-1)                     # every crest lining up at once

    def sea_state(self):
        return {"hs": self.waves.hs, "tp": self.waves.tp, "heading": self.waves.heading}

    def resample(self, env_ids=None):
        if env_ids is None:
            self.waves = sample_components(self.cfg, self.num_envs, self.device, self.dtype, self._gen)
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        fresh = sample_components(self.cfg, env_ids.numel(), self.device, self.dtype, self._gen)
        self.waves.copy_rows_(env_ids, fresh)
