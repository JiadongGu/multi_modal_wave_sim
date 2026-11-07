"""Verify the IMU and top camera on the lab GPU (flat water, so results are deterministic).

    ./isaaclab.sh -p <repo>/usv_lab/scripts/check_sensors.py --num_envs 4 --out ~/sensor_check

Checks
  1. IMU at rest reads proper acceleration ~(0, 0, +9.81) and no rotation.
  2. IMU under thrust: forward acceleration matches d(u)/dt measured from the boat's velocity.
  3. Camera pose stays at its mount on the moving boat (data.pos_w vs boat pose + offset).
  4. Camera isolation: boat 1 placed broadside right in front of boat 0's camera does not appear in it.
Saves rgb and depth grids for each stage into --out.
"""

import argparse
import math
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument("--num_envs", type=int, default=4)
ours.add_argument("--out", type=str, default=os.path.expanduser("~/sensor_check"))
mine, rest = ours.parse_known_args()
parser = argparse.ArgumentParser(parents=[ours])
add_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0], *rest]
for k, v in vars(mine).items():
    setattr(args, k, v)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import usv_lab.tasks  # noqa: E402,F401
from isaaclab.sensors import save_images_to_file  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from usv_core import body_to_world  # noqa: E402

TASK = "Usv-Float-Direct-v0"
launcher_argv, sys.argv = sys.argv, sys.argv[:1]
cfg, _ = resolve_task_config(TASK, "")
sys.argv = launcher_argv
cfg.scene.num_envs = args.num_envs
cfg.sea.hs = (1e-6, 1e-6)                      # flat water
cfg.reset_yaw = (0.0, 0.0)                     # every boat faces +x
os.makedirs(args.out, exist_ok=True)

results = []


def check(name, ok, detail):
    results.append((name, ok))
    print(f"CHECK {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def save(env, tag):
    cam = env.camera
    rgb = cam.data.output["rgb"].torch.float() / 255.0
    depth = cam.data.output["distance_to_image_plane"].torch.clone()
    depth = torch.nan_to_num(depth, posinf=200.0).clamp(0, 200.0) / 200.0
    save_images_to_file(rgb, os.path.join(args.out, f"{tag}_rgb.png"))
    save_images_to_file(depth.repeat(1, 1, 1, 3) if depth.shape[-1] == 1 else depth, os.path.join(args.out, f"{tag}_depth.png"))
    return rgb


def run(env, action, steps):
    for _ in range(steps):
        with torch.inference_mode():
            env.step(action)


with launch_simulation(cfg, args):
    genv = gym.make(TASK, cfg=cfg)
    genv.reset()
    u = genv.unwrapped
    n, dev, g = u.num_envs, u.device, cfg.hull.gravity
    zero = torch.zeros(n, 2, device=dev)
    full = torch.tensor([[1.0, 0.0]], device=dev).repeat(n, 1)
    print(f"CHECK info: imu={u.imu is not None} camera={u.camera is not None} "
          f"rgb shape {tuple(u.camera.data.output['rgb'].torch.shape)}", flush=True)

    # 1. IMU at rest
    run(genv, zero, 60)
    acc, gyro = u.imu.data.lin_acc_b.torch, u.imu.data.ang_vel_b.torch
    check("imu at rest", bool(((acc - torch.tensor([0, 0, g], device=dev)).abs() < 0.05).all() and (gyro.abs() < 1e-3).all()),
          f"lin_acc_b env0 {[round(v, 3) for v in acc[0].tolist()]}  ang_vel_b env0 {[round(v, 4) for v in gyro[0].tolist()]}")
    save(u, "1_rest")

    # 2. IMU under thrust vs measured acceleration
    run(genv, full, 45)                                    # 1.5 s: actuators have ramped up
    dt = u.step_dt
    u0 = u.boat_state().lin_vel_body[:, 0].clone()
    run(genv, full, 1)
    u1 = u.boat_state().lin_vel_body[:, 0].clone()
    a_measured = (u1 - u0) / dt
    a_imu = u.imu.data.lin_acc_b.torch[:, 0]
    rel = ((a_imu - a_measured).abs() / a_measured.abs().clamp_min(1e-6)).max().item()
    check("imu under thrust", rel < 0.05,
          f"imu ax {a_imu[0].item():.4f} m/s^2 vs du/dt {a_measured[0].item():.4f} m/s^2 (max rel diff {rel:.3f}); "
          f"thrust/mass would be {2000.0 / u.geo.mass:.4f} before drag")

    # 3. camera pose follows the boat
    run(genv, full, 30)
    d = u.boat.data
    off = torch.tensor(cfg.scene.top_cam.offset.pos, device=dev).expand(n, 3)
    expected = d.root_link_pos_w.torch + body_to_world(d.root_link_quat_w.torch, off)
    err = (u.camera.data.pos_w.torch - expected).norm(dim=-1).max().item()
    check("camera rides on the boat", err < 0.02, f"max |camera pos - (boat pos + R offset)| = {err * 100:.2f} cm")
    rgb_moving = save(u, "3_moving")
    check("camera images not blank", bool((rgb_moving.mean(dim=(1, 2, 3)) > 0.05).all()),
          f"mean pixel per env {[round(v, 3) for v in rgb_moving.mean(dim=(1, 2, 3)).tolist()]}")

    # 4. isolation: boat 1 broadside 12 m in front of boat 0's camera
    def settle_render():
        u.scene.write_data_to_sim()
        u.sim.step(render=True)
        u.scene.update(u.physics_dt)

    run(genv, zero, 1)
    still = torch.zeros(n, 6, device=dev)
    u.boat.write_root_velocity_to_sim_index(root_velocity=still)
    for _ in range(3):
        settle_render()
    before = u.camera.data.output["rgb"].torch.float().clone()
    save(u, "4a_before")
    cam0 = u.camera.data.pos_w.torch[0]
    pose = u.boat.data.root_link_pose_w.torch[1:2].clone()
    pose[0, 0] = cam0[0] + 12.0
    pose[0, 1] = cam0[1]
    pose[0, 3:7] = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)], device=dev)   # broadside
    u.boat.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([1], device=dev))
    for _ in range(3):
        settle_render()
    after = u.camera.data.output["rgb"].torch.float().clone()
    save(u, "4b_boat1_in_front_of_cam0")
    diff0 = (after[0] - before[0]).abs().mean().item()
    diff1 = (after[1] - before[1]).abs().mean().item()
    check("camera isolation", diff0 < 1.0,
          f"env0 image change {diff0:.2f} (0-255 scale; boat 1 now 12 m in front of it), env1 image change {diff1:.2f}")

    genv.close()

passed = sum(ok for _, ok in results)
print(f"CHECK SUMMARY {passed}/{len(results)} passed", flush=True)
