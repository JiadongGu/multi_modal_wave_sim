"""Does each boat's camera really see that boat's own drawn sea?

For the centre pixel of every camera: depth -> 3D hit point -> compare its height with the sea height there.
A camera seeing the backdrop (z = -3) or a patch in the wrong place fails this. Pictures alone do not.

    ./isaaclab.sh -p <repo>/usv_lab/scripts/check_water.py --num_envs 16 --out ~/water_check
"""

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument("--num_envs", type=int, default=16)
ours.add_argument("--out", type=str, default=os.path.expanduser("~/water_check"))
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
cfg.draw_water_envs = args.num_envs
cfg.scene.top_cam.width, cfg.scene.top_cam.height = 321, 241   # odd, so a pixel sits exactly on the optical axis
os.makedirs(args.out, exist_ok=True)


def measure(u, tag):
    cam = u.camera
    d = cam.data.output["distance_to_image_plane"].torch[:, 120, 160, 0]
    forward = body_to_world(cam.data.quat_w_world.torch, torch.tensor([1.0, 0.0, 0.0], device=u.device).expand(u.num_envs, 3))
    hit = cam.data.pos_w.torch + d[:, None] * forward
    ok = torch.isfinite(d)
    voyage = hit[:, :2] - u.scene.env_origins[:, :2] + u.sea_offset[:, :2]
    eta = u.sea.height(torch.nan_to_num(voyage, posinf=0.0, neginf=0.0)[:, None, :], u.sim_time)[:, 0]
    err = (hit[:, 2] - eta).abs()
    print(f"WATER {tag}: {int(ok.sum())}/{u.num_envs} centre pixels hit something; "
          f"|hit z - sea height| max {err[ok].max():.3f} m, mean {err[ok].mean():.3f} m; "
          f"depth env0..3 {[round(v, 2) for v in d[:4].tolist()]}", flush=True)
    for i in (err > 0.15).nonzero().flatten().tolist():
        cam_local = cam.data.pos_w.torch[i, :2] - u.scene.env_origins[i, :2] + u.sea_offset[i, :2]
        eta_cam = u.sea.height(cam_local[None, None, :].expand(u.num_envs, 1, 2), u.sim_time)[i, 0].item()
        print(f"WATER   env {i}: camera z {cam.data.pos_w.torch[i, 2]:.2f}, sea height at camera {eta_cam:.2f}, "
              f"hit z {hit[i, 2]:.2f}, depth {d[i]:.2f}, boat z {u.plot_position()[i, 2]:.2f}, "
              f"episode step {int(u.episode_length_buf[i])}, Hs {u.sea.waves.hs[i]:.2f}", flush=True)
    save_images_to_file(cam.data.output["rgb"].torch.float() / 255.0, os.path.join(args.out, f"{tag}.png"))
    return bool(ok.all()) and err[ok].max().item() < 0.15


with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped
    action = torch.zeros(u.num_envs, 2, device=u.device)
    passed = []
    for tag in ("t2s", "t4s", "t6s"):
        for _ in range(60):
            with torch.inference_mode():
                env.step(action)
        passed.append(measure(u, tag))
    print(f"WATER SUMMARY {'PASS' if all(passed) else 'FAIL'} {passed}", flush=True)
    env.close()
