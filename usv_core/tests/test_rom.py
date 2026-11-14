"""ROM loader: hand-made networks with known answers, saved and loaded exactly as a real ROM would be."""

import math

import pytest
import torch
from torch import nn

from usv_core import (BodyState, FlatWater, ModelContext, RomSpec, SeaCfg, SineWaveField, TorchModel, body_to_world,
                      make_stand_in, parse_size, world_to_body)


def save(tmp_path, net: nn.Module, spec: RomSpec, name="rom"):
    path = tmp_path / f"{name}.pt"
    torch.jit.script(net.eval()).save(str(path))
    spec.save(path.with_suffix(".json"))
    return path


def random_state(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    return BodyState(pos=torch.randn(n, 3, generator=g), quat=q, lin_vel=torch.randn(n, 3, generator=g),
                     ang_vel=torch.randn(n, 3, generator=g))


class Drag(nn.Module):
    """force = -k * input velocity, torque = 0. Works in whatever convention it is fed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([-100.0 * x[:, :3], torch.zeros_like(x[:, :3])], -1)


class Constant(nn.Module):
    def __init__(self, values: list[float]):
        super().__init__()
        self.register_buffer("c", torch.tensor(values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c.expand(x.shape[0], 6)


class SumFirst(nn.Module):
    """force x = sum of all inputs (to read back what the loader fed in)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.reshape(x.shape[0], -1).sum(-1, keepdim=True)
        return torch.cat([s, torch.zeros(x.shape[0], 5)], -1)


def test_body_frame_drag_matches_hand_calculation(tmp_path):
    path = save(tmp_path, Drag(), RomSpec(inputs=["lin_vel_body"]))
    s = random_state()
    w = TorchModel(path, 5)(s, ModelContext())
    expected = body_to_world(s.quat, -100.0 * world_to_body(s.quat, s.lin_vel))
    assert torch.allclose(w.force, expected, atol=1e-4)
    assert torch.allclose(w.force, -100.0 * s.lin_vel, atol=1e-4)      # drag is frame-independent
    assert torch.allclose(w.torque, torch.zeros_like(w.torque))


def test_frd_rom_gives_the_same_physics_as_flu(tmp_path):
    flu = TorchModel(save(tmp_path, Drag(), RomSpec(inputs=["lin_vel_body"]), "flu"), 5)
    frd = TorchModel(save(tmp_path, Drag(), RomSpec(inputs=["lin_vel_body"], convention="FRD"), "frd"), 5)
    s = random_state()
    assert torch.allclose(flu(s, ModelContext()).force, frd(s, ModelContext()).force, atol=1e-4)


def test_torque_is_moved_to_the_centre_of_mass(tmp_path):
    # 100 N up (body z), computed about a point 1 m ahead of the CoM, no torque there:
    # about the CoM that is r x F = (1, 0, 0) x (0, 0, 100) = (0, -100, 0) in the body frame
    spec = RomSpec(inputs=["heave"], reference_point=[1.0, 0.0, 0.0])
    s = random_state()
    w = TorchModel(save(tmp_path, Constant([0, 0, 100, 0, 0, 0]), spec), 5)(s, ModelContext())
    expected = body_to_world(s.quat, torch.tensor([0.0, -100.0, 0.0]).expand(5, 3))
    assert torch.allclose(w.torque, expected, atol=1e-4)


def test_world_frame_output_is_used_as_is(tmp_path):
    spec = RomSpec(inputs=["heave"], output_frame="world")
    w = TorchModel(save(tmp_path, Constant([1, 2, 3, 4, 5, 6]), spec), 5)(random_state(), ModelContext())
    assert torch.allclose(w.force, torch.tensor([1.0, 2, 3]).expand(5, 3))
    assert torch.allclose(w.torque, torch.tensor([4.0, 5, 6]).expand(5, 3))


def test_normalisation_and_output_scaling(tmp_path):
    spec = RomSpec(inputs=["heave"], input_mean=[1.0], input_std=[2.0], output_frame="world",
                   output_scale=[10, 1, 1, 1, 1, 1], output_offset=[5, 0, 0, 0, 0, 0])
    s = random_state()
    s.pos[:, 2] = 5.0                                                   # (5 - 1) / 2 = 2 -> 2 * 10 + 5 = 25
    w = TorchModel(save(tmp_path, SumFirst(), spec), 5)(s, ModelContext())
    assert torch.allclose(w.force[:, 0], torch.full((5,), 25.0))


def test_history_keeps_the_last_steps_and_clears_on_reset(tmp_path):
    spec = RomSpec(inputs=["heave"], history=3, output_frame="world")
    rom = TorchModel(save(tmp_path, SumFirst(), spec), 2)
    s = random_state(2)
    for z, total in [(1.0, 1.0), (2.0, 3.0), (3.0, 6.0), (4.0, 9.0)]:  # window of 3: oldest drops out
        s.pos[:, 2] = z
        assert torch.allclose(rom(s, ModelContext()).force[:, 0], torch.full((2,), total))
    rom.reset(torch.tensor([1]))                                         # boat 1 forgets; boat 0 does not
    s.pos[:, 2] = 1.0
    f = rom(s, ModelContext()).force[:, 0]
    assert f[0].item() == pytest.approx(3.0 + 4.0 + 1.0) and f[1].item() == pytest.approx(1.0)


def test_sequence_layout_gets_steps_by_features(tmp_path):
    class Shape(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat([torch.full((x.shape[0], 1), float(x.shape[1] * 100 + x.shape[2])),
                              torch.zeros(x.shape[0], 5)], -1)

    spec = RomSpec(inputs=["heave", "ang_vel_body"], history=4, layout="sequence", output_frame="world")
    w = TorchModel(save(tmp_path, Shape(), spec), 3)(random_state(3), ModelContext())
    assert torch.allclose(w.force[:, 0], torch.full((3,), 404.0))          # [N, 4 steps, 4 features]


def test_wave_height_is_sampled_at_the_hull_points(tmp_path):
    pts = [[3.0, 0.0], [0.0, 1.5], [-3.0, 0.0]]
    spec = RomSpec(inputs=["wave_height"], wave_points=pts, output_frame="world")

    class Pass(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x, torch.zeros(x.shape[0], 3)], -1)

    rom = TorchModel(save(tmp_path, Pass(), spec), 4)
    s = random_state(4)
    sea = SineWaveField(SeaCfg(hs=(1.0, 2.0), tp=(6.0, 8.0)), 4, seed=1)
    got = rom(s, ModelContext(t=2.5, sea=sea)).force
    p = torch.tensor([[x, y, 0.0] for x, y in pts])
    world = s.pos[:, None, :] + body_to_world(s.quat, p.expand(4, 3, 3))
    assert torch.allclose(got, sea.height(world[..., :2], 2.5), atol=1e-5)
    assert torch.allclose(rom(s, ModelContext(t=2.5, sea=FlatWater(4))).force, torch.zeros(4, 3))


def test_cost_only_runs_but_applies_nothing(tmp_path):
    path = save(tmp_path, Constant([1, 2, 3, 4, 5, 6]), RomSpec(inputs=["heave"]))
    w = TorchModel(path, 5, mode="cost_only")(random_state(), ModelContext())
    assert torch.count_nonzero(w.force) == 0 and torch.count_nonzero(w.torque) == 0


def test_stand_in_loads_and_runs_on_a_full_context(tmp_path):
    path = make_stand_in(tmp_path / "stand_in.pt", depth=2, width=32, history=5)
    rom = TorchModel(path, 6)
    assert rom.spec.history == 5 and rom.spec.num_features == 3 + 3 + 3 + 1 + 2 + 8
    sea = SineWaveField(SeaCfg(hs=(1.0, 2.0), tp=(6.0, 8.0)), 6, seed=0)
    w = rom(random_state(6), ModelContext(t=1.0, dt=1 / 120, sea=sea, actions=torch.rand(6, 2)))
    assert w.force.shape == (6, 3) and torch.isfinite(w.force).all() and torch.isfinite(w.torque).all()


def test_parse_size():
    assert parse_size("3x256") == (3, 256, 1)
    assert parse_size("2x64x10") == (2, 64, 10)
    with pytest.raises(ValueError):
        parse_size("256")


def test_bad_specs_are_refused():
    with pytest.raises(ValueError):
        RomSpec(inputs=["speed"]).check()
    with pytest.raises(ValueError):
        RomSpec(inputs=["heave"], convention="FRD", output_frame="world").check()
    with pytest.raises(ValueError):
        RomSpec(inputs=["wave_height"]).check()
    with pytest.raises(ValueError):
        RomSpec(inputs=["heave"], input_mean=[0.0, 1.0]).check()


def test_the_example_eric_copies_loads_and_runs(tmp_path):
    """docs/rom_example.py is what we hand Eric: its files must load and run in our loader as they are."""
    import importlib.util
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "docs" / "rom_example.py"
    spec = importlib.util.spec_from_file_location("rom_example", src)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.export(example.MyRom(), tmp_path / "my_rom")
    rom = TorchModel(tmp_path / "my_rom.pt", 4)
    assert rom.spec.num_features == 20
    sea = SineWaveField(SeaCfg(hs=(1.0, 2.0), tp=(6.0, 8.0)), 4, seed=0)
    w = rom(random_state(4), ModelContext(t=1.0, dt=1 / 120, sea=sea, actions=torch.rand(4, 2)))
    assert torch.isfinite(w.force).all() and torch.isfinite(w.torque).all()
