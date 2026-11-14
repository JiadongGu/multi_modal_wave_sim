"""Example: save a network as a ROM our simulator can load. Copy and adapt (see docs/ROM_INTERFACE.md).

    python3 rom_example.py            # writes my_rom.pt and my_rom.json next to this file

Your network: any torch.nn.Module that TorchScript can compile (torch.jit.script; or torch.jit.trace
with an example input). Input [N, history * F] (or [N, history, F] with layout "sequence"), output [N, 6].
"""

from pathlib import Path

import torch
from torch import nn


class MyRom(nn.Module):
    """Stand-in for your trained model: 20 inputs -> force (3) and torque (3)."""

    def __init__(self, n_in: int = 20):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export(model: nn.Module, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model.eval()).save(str(out.with_suffix(".pt")))
    description = {
        # inputs, in this order; sizes: lin_vel_rel_body 3, ang_vel_body 3, gravity_body 3, heave 1,
        # actions 2, wave_height 8 (one per point below) = 20 features per step
        "inputs": ["lin_vel_rel_body", "ang_vel_body", "gravity_body", "heave", "actions", "wave_height"],
        "num_actions": 2,
        "wave_points": [[3.0, 0.0], [2.1, 1.4], [0.0, 2.0], [-2.1, 1.4], [-3.0, 0.0], [-2.1, -1.4], [0.0, -2.0], [2.1, -1.4]],
        "convention": "FRD",             # your body axes: FRD (x forward, y starboard, z down) or FLU
        "output_frame": "body",          # force and torque in the body frame above
        "reference_point": [0.0, 0.0, 0.0],   # torque is about this point, body frame, metres from the CoM
        "history": 1,                    # steps of inputs the network sees (1 = this step only)
        "layout": "flat",
        "input_mean": None,              # 20 values each if your network expects normalised inputs
        "input_std": None,
        "output_scale": None,            # 6 values if your outputs are normalised: out * scale + offset
        "output_offset": None,
        "note": "example export",
    }
    import json

    out.with_suffix(".json").write_text(json.dumps(description, indent=2))


if __name__ == "__main__":
    export(MyRom(), Path(__file__).with_name("my_rom"))
    print("wrote my_rom.pt and my_rom.json")
