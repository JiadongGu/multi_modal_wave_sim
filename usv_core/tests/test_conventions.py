import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

from usv_core import (
    BodyState,
    Wrench,
    body_to_world,
    quat_to_matrix,
    world_to_body,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

EXT_BASE = Path(__file__).resolve().parents[2] / "exts/mmws.wave_sim/mmws/wave_sim/forces/base.py"


def yaw_quat(angle):
    """(x, y, z, w) for a rotation of `angle` about world z."""
    return torch.tensor([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])


def random_quats(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def test_yaw_90_turns_forward_into_left():
    q = yaw_quat(math.pi / 2).unsqueeze(0)
    forward = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(body_to_world(q, forward), torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_round_trip_body_world_body():
    q = random_quats(64)
    v = torch.randn(64, 3, dtype=torch.float64)
    assert torch.allclose(world_to_body(q, body_to_world(q, v)), v, atol=1e-12)


def test_many_vectors_per_boat_match_one_at_a_time():
    q = random_quats(8)
    pts = torch.randn(8, 5, 3, dtype=torch.float64)
    batched = body_to_world(q, pts)
    for m in range(5):
        assert torch.allclose(batched[:, m], body_to_world(q, pts[:, m]), atol=1e-12)


def test_rotation_matrices_are_proper():
    R = quat_to_matrix(random_quats(32))
    eye = torch.eye(3, dtype=torch.float64).expand(32, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(32, dtype=torch.float64), atol=1e-12)


def test_unnormalised_and_zero_quaternions():
    q = random_quats(4)
    assert torch.allclose(quat_to_matrix(q * 3.7), quat_to_matrix(q), atol=1e-12)
    assert torch.allclose(quat_to_matrix(torch.zeros(1, 4)), torch.eye(3).unsqueeze(0))


def test_order_conversions_round_trip():
    q = random_quats(10)
    assert torch.equal(wxyz_to_xyzw(xyzw_to_wxyz(q)), q)
    assert torch.equal(xyzw_to_wxyz(torch.tensor([1.0, 2.0, 3.0, 4.0])), torch.tensor([4.0, 1.0, 2.0, 3.0]))


def test_matches_extension_quat_to_matrix():
    """Same rotation as the mmws.wave_sim extension, once the quaternion order is converted."""
    spec = importlib.util.spec_from_file_location("ext_forces_base", EXT_BASE)
    ext = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ext  # dataclasses look their module up here
    spec.loader.exec_module(ext)
    q_xyzw = random_quats(16)
    ours = quat_to_matrix(q_xyzw).numpy()
    for i, q_wxyz in enumerate(xyzw_to_wxyz(q_xyzw).numpy()):
        np.testing.assert_allclose(ours[i], ext.quat_to_matrix(q_wxyz), atol=1e-12)


def test_body_state_body_frame_velocity():
    # Boat yawed 90 degrees (bow pointing world +y), moving world +y at 2 m/s: forward at 2 m/s.
    s = BodyState(
        pos=torch.zeros(1, 3),
        quat=yaw_quat(math.pi / 2).unsqueeze(0),
        lin_vel=torch.tensor([[0.0, 2.0, 0.0]]),
        ang_vel=torch.tensor([[0.0, 0.0, 0.5]]),
    )
    assert s.num_envs == 1
    assert torch.allclose(s.lin_vel_body, torch.tensor([[2.0, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(s.ang_vel_body, torch.tensor([[0.0, 0.0, 0.5]]), atol=1e-6)


def test_wrench_zeros_and_sum():
    a = Wrench.zeros(3)
    b = Wrench(torch.ones(3, 3), torch.full((3, 3), 2.0))
    c = a + b + b
    assert c.force.shape == (3, 3)
    assert torch.equal(c.force, torch.full((3, 3), 2.0))
    assert torch.equal(c.torque, torch.full((3, 3), 4.0))
