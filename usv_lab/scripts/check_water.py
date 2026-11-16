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
ours.add_argument("--samples", type=int, default=3, help="how many checks to make")
ours.add_argument("--every", type=int, default=60, help="policy steps between checks")
ours.add_argument("--res", type=str, default="321x241", help="camera WxH; odd so a pixel sits on the optical axis. "
                  "Use 129x97 for ~1,000 boats (321x241 does not fit in 24 GB there)")
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
cfg.scene.top_cam.width, cfg.scene.top_cam.height = (int(v) for v in args.res.lower().split("x"))  # odd: centre pixel on axis
os.makedirs(args.out, exist_ok=True)


def measure(u, tag):
    """Centre pixel of every camera: depth -> 3D hit -> compare with the sea height there.

    Cameras that cannot see water are left out and counted: underwater (the boat dipped) or pointing
    above the horizon (the boat pitched up). Those say nothing about whether the water is drawn right."""
    cam = u.camera
    h, w = cam.data.output["distance_to_image_plane"].torch.shape[1:3]
    d = cam.data.output["distance_to_image_plane"].torch[:, h // 2, w // 2, 0]
    forward = body_to_world(cam.data.quat_w_world.torch, torch.tensor([1.0, 0.0, 0.0], device=u.device).expand(u.num_envs, 3))
    cam_pos = cam.data.pos_w.torch
    cam_voyage = cam_pos[:, :2] - u.scene.env_origins[:, :2] + u.sea_offset[:, :2]
    eta_cam = u.sea.height(cam_voyage[:, None, :], u.sim_time)[:, 0]
    underwater = cam_pos[:, 2] < eta_cam + 0.05
    skyward = forward[:, 2] > -0.01
    eligible = ~underwater & ~skyward
    hit = cam_pos + d[:, None] * forward
    ok = torch.isfinite(d)
    voyage = hit[:, :2] - u.scene.env_origins[:, :2] + u.sea_offset[:, :2]
    eta = u.sea.height(torch.nan_to_num(voyage, posinf=0.0, neginf=0.0)[:, None, :], u.sim_time)[:, 0]
    err = torch.where(ok, (hit[:, 2] - eta).abs(), torch.full_like(d, float("inf")))
    good = eligible & ok & (err < 0.15)
    e = err[eligible & ok]
    print(f"WATER {tag}: {int(good.sum())}/{int(eligible.sum())} cameras that can see water hit it within 0.15 m "
          f"(left out: {int(underwater.sum())} underwater, {int((skyward & ~underwater).sum())} looking up); "
          f"|hit z - sea height| max {e.max() if e.numel() else float('nan'):.3f} m, mean {e.mean() if e.numel() else float('nan'):.3f} m", flush=True)
    for i in (eligible & ~good).nonzero().flatten().tolist()[:10]:
        print(f"WATER   env {i}: camera z {cam_pos[i, 2]:.2f}, sea height at camera {eta_cam[i]:.2f}, "
              f"hit z {hit[i, 2]:.2f}, depth {d[i]:.2f}, boat z {u.plot_position()[i, 2]:.2f}, "
              f"episode step {int(u.episode_length_buf[i])}, Hs {u.sea.waves.hs[i]:.2f}", flush=True)
    if args.samples <= 5 and u.num_envs <= 64:
        save_images_to_file(cam.data.output["rgb"].torch.float() / 255.0, os.path.join(args.out, f"{tag}.png"))
    return bool((good == eligible).all())


with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped
    action = torch.zeros(u.num_envs, 2, device=u.device)
    passed = []
    for k in range(args.samples):
        for _ in range(args.every):
            with torch.inference_mode():
                env.step(action)
        passed.append(measure(u, f"t{u.sim_time:05.1f}s"))
    print(f"WATER SUMMARY {'PASS' if all(passed) else 'FAIL'}: {sum(passed)}/{len(passed)} samples clean", flush=True)
    env.close()
