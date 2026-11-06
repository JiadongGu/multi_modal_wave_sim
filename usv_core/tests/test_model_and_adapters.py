import math

import torch

from usv_core import BodyState, Wrench
from usv_core.adapters import (
    ConventionAdapter,
    body_velocity_inputs,
    flu_to_frd,
    frd_to_flu,
    join6,
    split6,
    torque_about_com,
)
from usv_core.model import Composite, HullModel, ModelContext, ZeroModel


def yaw_quat(angle, n=1):
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]).repeat(n, 1)


def state(n=1, yaw=0.0, lin_vel=None, ang_vel=None):
    return BodyState(
        pos=torch.zeros(n, 3),
        quat=yaw_quat(yaw, n),
        lin_vel=torch.zeros(n, 3) if lin_vel is None else lin_vel,
        ang_vel=torch.zeros(n, 3) if ang_vel is None else ang_vel,
    )


class Constant(HullModel):
    def __init__(self, f, t):
        self.f, self.t, self.resets = torch.tensor(f), torch.tensor(t), []

    def wrench(self, s, ctx):
        return Wrench(self.f.expand(s.num_envs, 3).clone(), self.t.expand(s.num_envs, 3).clone())

    def reset(self, env_ids=None):
        self.resets.append(env_ids)


def test_composite_sums_and_reports_parts():
    a, b = Constant([1.0, 0, 0], [0, 0, 1.0]), Constant([0, 2.0, 0], [0, 0, 3.0])
    m = Composite({"a": a, "b": b, "zero": ZeroModel()})
    w = m(state(4), ModelContext())
    assert w.force.shape == (4, 3)
    assert torch.equal(w.force[0], torch.tensor([1.0, 2.0, 0.0]))
    assert torch.equal(w.torque[0], torch.tensor([0.0, 0.0, 4.0]))
    parts = m.parts(state(4), ModelContext())
    assert set(parts) == {"a", "b", "zero"} and torch.equal(parts["b"].force[0], torch.tensor([0.0, 2.0, 0.0]))


def test_composite_reset_reaches_every_model():
    a, b = Constant([0.0] * 3, [0.0] * 3), Constant([0.0] * 3, [0.0] * 3)
    ids = torch.tensor([2, 5])
    Composite([a, b]).reset(ids)
    assert a.resets == [ids] and b.resets == [ids]


def test_frd_flip():
    v = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(flu_to_frd(v), torch.tensor([[1.0, -2.0, -3.0]]))
    assert torch.equal(frd_to_flu(flu_to_frd(v)), v)


def test_split_join6():
    tau = torch.arange(12.0).reshape(2, 6)
    f, t = split6(tau)
    assert torch.equal(join6(f, t), tau)


def test_frd_surge_force_on_yawed_boat_points_along_bow():
    """A Fossen-style model pushing forward (FRD +X) on a boat yawed 90 degrees pushes world +y."""
    m = ConventionAdapter(lambda _: torch.tensor([[100.0, 0, 0, 0, 0, 0]]), lambda s, c: None, "body", "FRD")
    w = m(state(1, yaw=math.pi / 2), ModelContext())
    assert torch.allclose(w.force, torch.tensor([[0.0, 100.0, 0.0]]), atol=1e-4)


def test_frd_positive_heave_is_world_down():
    m = ConventionAdapter(lambda _: torch.tensor([[0, 0, 50.0, 0, 0, 0]]), lambda s, c: None, "body", "FRD")
    w = m(state(1), ModelContext())
    assert torch.allclose(w.force, torch.tensor([[0.0, 0.0, -50.0]]))


def test_frd_positive_yaw_moment_is_clockwise_from_above():
    """FRD +N turns the bow to starboard (right), which is world -z in FLU."""
    m = ConventionAdapter(lambda _: torch.tensor([[0, 0, 0, 0, 0, 7.0]]), lambda s, c: None, "body", "FRD")
    assert torch.allclose(m(state(1), ModelContext()).torque, torch.tensor([[0.0, 0.0, -7.0]]))


def test_torque_point_shift():
    """Upward force at a point 2 m ahead of the CoM pitches the bow up: torque about CoM = r x F."""
    f = torch.tensor([[0.0, 0.0, 10.0]])
    r = torch.tensor([[2.0, 0.0, 0.0]])
    assert torch.allclose(torque_about_com(torch.zeros(1, 3), f, r), torch.tensor([[0.0, -20.0, 0.0]]))
    m = ConventionAdapter(lambda _: (f.clone(), torch.zeros(1, 3)), lambda s, c: None, "body", "FLU",
                          torque_point=torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(m(state(1), ModelContext()).torque, torch.tensor([[0.0, -20.0, 0.0]]))


def test_fossen_style_damping_through_adapter_matches_direct():
    """Linear damping written in FRD body frame, adapted, equals the same damping done directly in world FLU."""
    D = torch.tensor([10.0, 40.0, 60.0, 5.0, 8.0, 3.0])
    n = 16
    g = torch.Generator().manual_seed(1)
    yaw = torch.rand(n, generator=g) * 6.28
    s = BodyState(
        pos=torch.zeros(n, 3),
        quat=torch.stack([torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1),
        lin_vel=torch.randn(n, 3, generator=g),
        ang_vel=torch.randn(n, 3, generator=g),
    )
    adapted = ConventionAdapter(
        lambda nu_frd: -D * nu_frd,
        lambda st, c: body_velocity_inputs(st, c, "FRD"),
        "body", "FRD",
    )(s, ModelContext())
    nu = body_velocity_inputs(s, ModelContext(), "FLU")
    from usv_core import body_to_world
    direct_f = body_to_world(s.quat, -D[:3] * nu[:, :3])
    direct_t = body_to_world(s.quat, -D[3:] * nu[:, 3:])
    assert torch.allclose(adapted.force, direct_f, atol=1e-5)
    assert torch.allclose(adapted.torque, direct_t, atol=1e-5)


def test_adapter_passes_reset_through():
    seen = []
    m = ConventionAdapter(lambda _: torch.zeros(1, 6), lambda s, c: None, reset_fn=seen.append)
    m.reset(torch.tensor([3]))
    assert len(seen) == 1 and seen[0].tolist() == [3]
