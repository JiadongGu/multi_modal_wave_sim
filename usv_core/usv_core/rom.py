"""Reduced-order models (ROMs): run a trained network as a force model on every boat at once.

A ROM arrives as two files with the same name:
  <name>.pt    a TorchScript network (saved with torch.jit.save). It takes the inputs listed in the
               description, for all boats at once, and returns [N, 6]: force (3) then torque (3).
  <name>.json  its description (RomSpec): which inputs in which order, the body-axis convention,
               the frame of the output, the point the torque is about, scaling, and history length.
SI units throughout (m, s, rad, N, N m).

Inputs a ROM can ask for (names in `inputs`, in the ROM's own body convention, FLU or FRD):
  lin_vel_body       3  linear velocity in the body frame [m/s]
  lin_vel_rel_body   3  the same, relative to the water at the centre of mass (wave orbital velocity removed)
  ang_vel_body       3  angular velocity in the body frame [rad/s]
  gravity_body       3  unit "down" vector in the body frame (roll and pitch without angle wrap-around)
  heave              1  centre-of-mass height above the still-water level [m]
  actions            A  actuator commands as applied this step (A = `num_actions`)
  wave_height        P  wave height [m] at P points fixed to the hull (`wave_points`, body-frame [x, y])
  sea_state          2  significant wave height Hs [m], peak period Tp [s]
History: with `history` = H > 1 the network gets the last H steps' inputs, oldest first, either
flattened to [N, H*F] (`layout` "flat") or as [N, H, F] ("sequence"). Cleared when a boat resets.

Modes (how the environment uses it):
  add        on top of the other force models (the ROM corrects them)
  replace    instead of buoyancy and damping (the ROM owns the water forces; thrust stays)
  cost_only  run every step, output multiplied by zero: measures what a network of that size costs
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .adapters import flu_to_frd
from .conventions import BodyState, Wrench, body_to_world, world_to_body
from .model import HullModel, ModelContext

FEATURE_SIZES = {"lin_vel_body": 3, "lin_vel_rel_body": 3, "ang_vel_body": 3, "gravity_body": 3, "heave": 1,
                 "sea_state": 2}   # actions and wave_height depend on the spec


@dataclass
class RomSpec:
    inputs: list[str]
    num_actions: int = 2
    wave_points: list[list[float]] = field(default_factory=list)   # [P, 2] body-frame (x, y) [m]
    convention: str = "FLU"                  # body axes of inputs and outputs: FLU (ours) or FRD (Fossen)
    output_frame: str = "body"               # "body", or "world" (z up; only with FLU)
    reference_point: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # torque is about this point,
                                                                                    # body frame, from the CoM [m]
    history: int = 1
    layout: str = "flat"                     # "flat" [N, H*F] or "sequence" [N, H, F]
    input_mean: list[float] | None = None    # per input feature (F values); inputs become (x - mean) / std
    input_std: list[float] | None = None
    output_scale: list[float] | None = None  # 6 values; output becomes raw * scale + offset
    output_offset: list[float] | None = None
    note: str = ""

    def feature_size(self, name: str) -> int:
        if name == "actions":
            return self.num_actions
        if name == "wave_height":
            return len(self.wave_points)
        if name not in FEATURE_SIZES:
            raise ValueError(f"unknown ROM input {name!r}; known: {sorted([*FEATURE_SIZES, 'actions', 'wave_height'])}")
        return FEATURE_SIZES[name]

    @property
    def num_features(self) -> int:
        return sum(self.feature_size(n) for n in self.inputs)

    def check(self) -> None:
        if self.convention not in ("FLU", "FRD"):
            raise ValueError(f"convention must be FLU or FRD, not {self.convention!r}")
        if self.output_frame not in ("body", "world"):
            raise ValueError(f"output_frame must be body or world, not {self.output_frame!r}")
        if self.output_frame == "world" and self.convention != "FLU":
            raise ValueError("a world-frame output must use FLU (z up); give FRD outputs in the body frame")
        if self.layout not in ("flat", "sequence"):
            raise ValueError(f"layout must be flat or sequence, not {self.layout!r}")
        if self.history < 1:
            raise ValueError("history must be at least 1")
        if "wave_height" in self.inputs and not self.wave_points:
            raise ValueError("wave_height needs wave_points")
        f = self.num_features
        for key in ("input_mean", "input_std"):
            v = getattr(self, key)
            if v is not None and len(v) != f:
                raise ValueError(f"{key} has {len(v)} values, the inputs have {f} features")
        for key in ("output_scale", "output_offset"):
            v = getattr(self, key)
            if v is not None and len(v) != 6:
                raise ValueError(f"{key} needs 6 values")

    @classmethod
    def load(cls, path: str | Path) -> "RomSpec":
        spec = cls(**json.loads(Path(path).read_text()))
        spec.check()
        return spec

    def save(self, path: str | Path) -> None:
        self.check()
        Path(path).write_text(json.dumps(asdict(self), indent=2))


@dataclass
class RomCfg:
    path: str                                # the .pt file; its description is the .json beside it
    mode: str = "add"                        # add | replace | cost_only


class TorchModel(HullModel):
    """A ROM loaded from disk, as a force model. Batched over all boats, on the device it is given."""

    def __init__(self, path: str | Path, num_envs: int, device="cpu", dtype=torch.float32, mode: str = "add"):
        if mode not in ("add", "replace", "cost_only"):
            raise ValueError(f"mode must be add, replace or cost_only, not {mode!r}")
        path = Path(path).expanduser()
        self.spec = RomSpec.load(path.with_suffix(".json"))
        self.mode, self.device, self.dtype = mode, torch.device(device), dtype
        self.net = torch.jit.load(str(path), map_location=self.device).eval()
        s, kw = self.spec, {"device": self.device, "dtype": dtype}
        flip = s.convention == "FRD"
        self._flip = flip
        # constants on the device once (a copy per step would make the CPU wait for the GPU every time)
        self._mean = torch.tensor(s.input_mean, **kw) if s.input_mean is not None else None
        self._std = torch.tensor(s.input_std, **kw) if s.input_std is not None else None
        self._scale = torch.tensor(s.output_scale, **kw) if s.output_scale is not None else None
        self._offset = torch.tensor(s.output_offset, **kw) if s.output_offset is not None else None
        ref = torch.tensor(s.reference_point, **kw)
        self._ref = flu_to_frd(ref) if flip else ref                                   # FLU, from the CoM
        pts = torch.tensor(s.wave_points, **kw).reshape(-1, 2)
        pts = torch.cat([pts, torch.zeros_like(pts[:, :1])], -1)
        self._points = flu_to_frd(pts) if flip else pts                                 # [P, 3] FLU
        self._down = torch.tensor([0.0, 0.0, -1.0], **kw)
        self.history = torch.zeros(num_envs, s.history, s.num_features, **kw)

    def features(self, state: BodyState, ctx: ModelContext) -> torch.Tensor:
        """[N, F] this step's inputs, in the spec's order and convention."""
        flip = flu_to_frd if self._flip else (lambda v: v)
        n, sea, out = state.num_envs, ctx.sea, []
        for name in self.spec.inputs:
            if name == "lin_vel_body":
                out.append(flip(state.lin_vel_body))
            elif name == "lin_vel_rel_body":
                v = state.lin_vel
                if sea is not None:
                    v = v - sea.velocity(state.pos[:, None, :], ctx.t)[:, 0, :]
                out.append(flip(world_to_body(state.quat, v)))
            elif name == "ang_vel_body":
                out.append(flip(state.ang_vel_body))
            elif name == "gravity_body":
                out.append(flip(world_to_body(state.quat, self._down.expand(n, 3))))
            elif name == "heave":
                out.append(state.pos[:, 2:3])
            elif name == "actions":
                a = ctx.actions if ctx.actions is not None else state.pos.new_zeros(n, self.spec.num_actions)
                out.append(a.to(state.pos))
            elif name == "wave_height":
                p = self._points.shape[0]
                world = state.pos[:, None, :] + body_to_world(state.quat, self._points.expand(n, p, 3))
                out.append(sea.height(world[..., :2], ctx.t) if sea is not None else state.pos.new_zeros(n, p))
            elif name == "sea_state":
                if sea is None:
                    out.append(state.pos.new_zeros(n, 2))
                else:
                    ss = sea.sea_state()
                    out.append(torch.stack([ss["hs"], ss["tp"]], -1).to(state.pos))
        return torch.cat(out, -1)

    def wrench(self, state: BodyState, ctx: ModelContext) -> Wrench:
        x = self.features(state, ctx).to(self.dtype)
        if self._mean is not None:
            x = x - self._mean
        if self._std is not None:
            x = x / self._std
        # new tensors, never in place: the env steps under torch.inference_mode but resets outside it
        self.history = torch.cat([self.history[:, 1:], x[:, None, :]], 1) if self.spec.history > 1 else x[:, None, :]
        inp = self.history.flatten(1) if self.spec.layout == "flat" else self.history
        with torch.no_grad():
            y = self.net(inp).to(self.dtype)                                            # [N, 6]
        if self._scale is not None:
            y = y * self._scale
        if self._offset is not None:
            y = y + self._offset
        force, torque = y[:, :3], y[:, 3:]
        if self.spec.output_frame == "body":
            if self._flip:
                force, torque = flu_to_frd(force), flu_to_frd(torque)
            torque = torque + torch.cross(self._ref.expand_as(force), force, dim=-1)    # about the CoM
            force, torque = body_to_world(state.quat, force), body_to_world(state.quat, torque)
        else:
            ref_w = body_to_world(state.quat, self._ref.expand_as(force))
            torque = torque + torch.cross(ref_w, force, dim=-1)
        if self.mode == "cost_only":
            force, torque = force * 0.0, torque * 0.0
        return Wrench(force.to(state.pos.dtype), torque.to(state.pos.dtype))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.history = torch.zeros_like(self.history)
            return
        keep = torch.ones(self.history.shape[0], 1, 1, device=self.history.device, dtype=self.history.dtype)
        keep[torch.as_tensor(env_ids, device=self.history.device, dtype=torch.long)] = 0.0
        self.history = self.history * keep


# ---------------------------------------------------------------- stand-in networks (for measuring cost)

DEFAULT_INPUTS = ["lin_vel_rel_body", "ang_vel_body", "gravity_body", "heave", "actions", "wave_height"]


def ring(radius_x: float, radius_y: float, count: int) -> list[list[float]]:
    """`count` body-frame points on an ellipse around the hull, starting at the bow."""
    import math

    return [[round(radius_x * math.cos(2 * math.pi * i / count), 3), round(radius_y * math.sin(2 * math.pi * i / count), 3)]
            for i in range(count)]


def parse_size(text: str) -> tuple[int, int, int]:
    """'3x256' or '3x256x10' -> (hidden layers, width, history)."""
    parts = [int(p) for p in text.lower().split("x")]
    if len(parts) not in (2, 3) or min(parts) < 1:
        raise ValueError(f"ROM size must look like 3x256 or 3x256x10, not {text!r}")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else 1


def make_stand_in(path: str | Path, depth: int = 3, width: int = 256, history: int = 1,
                  wave_points: list[list[float]] | None = None, seed: int = 0) -> Path:
    """Write an untrained network of the given size, plus its description, as a ROM would arrive.

    Random weights: it predicts nothing useful. Use it with mode "cost_only" to measure what a
    network of this size costs per step, before the real ROM exists.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = RomSpec(inputs=list(DEFAULT_INPUTS), wave_points=wave_points or ring(4.0, 2.0, 8), history=history,
                   note=f"stand-in: untrained MLP, {depth} hidden layers x {width}, history {history}")
    torch.manual_seed(seed)
    layers, n_in = [], spec.num_features * history
    for _ in range(depth):
        layers += [nn.Linear(n_in, width), nn.ReLU()]
        n_in = width
    layers.append(nn.Linear(n_in, 6))
    torch.jit.script(nn.Sequential(*layers).eval()).save(str(path))
    spec.save(path.with_suffix(".json"))
    return path
