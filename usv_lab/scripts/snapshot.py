"""Save what the boat cameras see (waves on, water drawn), headless.

    ./isaaclab.sh -p <repo>/usv_lab/scripts/snapshot.py --num_envs 16 --steps 150 --out ~/snapshot
"""

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument("--num_envs", type=int, default=16)
ours.add_argument("--steps", type=int, default=150)
ours.add_argument("--res", type=int, nargs=2, default=(320, 240))
ours.add_argument("--throttle", type=float, default=0.5)
ours.add_argument("--out", type=str, default=os.path.expanduser("~/snapshot"))
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

TASK = "Usv-Float-Direct-v0"
launcher_argv, sys.argv = sys.argv, sys.argv[:1]
cfg, _ = resolve_task_config(TASK, "")
sys.argv = launcher_argv
cfg.scene.num_envs = args.num_envs
cfg.draw_water_envs = min(16, args.num_envs)
cfg.scene.top_cam.width, cfg.scene.top_cam.height = args.res
os.makedirs(args.out, exist_ok=True)

with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped
    action = torch.zeros(u.num_envs, 2, device=u.device)
    action[:, 0] = args.throttle
    action[:, 1] = torch.linspace(-0.6, 0.6, u.num_envs, device=u.device)
    for i in range(args.steps):
        with torch.inference_mode():
            env.step(action)
        if (i + 1) % 50 == 0:
            rgb = u.camera.data.output["rgb"].torch[: cfg.draw_water_envs].float() / 255.0
            save_images_to_file(rgb, os.path.join(args.out, f"cams_step{i + 1:04d}.png"))
            print(f"SNAP saved step {i + 1}  hs of env0..3 {[round(v, 2) for v in u.sea.waves.hs[:4].tolist()]}", flush=True)
    env.close()
print("SNAP DONE", flush=True)
