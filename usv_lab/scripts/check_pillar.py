"""Check the pillar task: each boat's lidar finds its own pillar at the right range and bearing, and cameras see orange.

    ./isaaclab.sh -p <repo>/usv_lab/scripts/check_pillar.py --num_envs 16 --out ~/pillar_check
"""

import argparse
import math
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument("--num_envs", type=int, default=16)
ours.add_argument("--out", type=str, default=os.path.expanduser("~/pillar_check"))
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
from usv_core import world_to_body  # noqa: E402

TASK = "Usv-PillarNav-Play-Direct-v0"
launcher_argv, sys.argv = sys.argv, sys.argv[:1]
cfg, _ = resolve_task_config(TASK, "")
sys.argv = launcher_argv
cfg.scene.num_envs = args.num_envs
os.makedirs(args.out, exist_ok=True)

with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    action = torch.zeros(u.num_envs, 2, device=u.device)
    with torch.inference_mode():
        for _ in range(20):
            obs, *_ = env.step(action)
    print(f"PILLAR obs: camera {tuple(obs['policy']['camera'].shape)}, lidar {tuple(obs['policy']['lidar'].shape)} "
          f"range [{obs['policy']['lidar'].min():.2f}, {obs['policy']['lidar'].max():.2f}]", flush=True)
    world = u.pillar.data.root_pos_w.torch[:, :2] - u.scene.env_origins[:, :2]
    print(f"PILLAR placed where intended: max error {(world - u.pillar_xy).norm(dim=-1).max():.3f} m", flush=True)

    # lidar: scan each boat with and without its pillar; any beam that changes really saw the pillar
    s = u.boat_state()
    to_pillar = torch.cat([u.pillar_xy - s.pos[:, :2], torch.zeros(u.num_envs, 1, device=u.device)], -1)
    body = world_to_body(s.quat, to_pillar)
    bearing = torch.atan2(body[:, 1], body[:, 0])
    expected = u.distance_to_pillar() - 1.0
    with_pillar = u.lidar.scan(s, u.sea, u.sim_time, u.lidar_cylinders()).ranges
    without = u.lidar.scan(s, u.sea, u.sim_time).ranges
    saw = with_pillar < torch.nan_to_num(without, posinf=1e9) - 1e-3             # [N, R]
    boats = saw.any(1)
    near = with_pillar.clone()
    near[~saw] = float("inf")
    got = near.amin(1)
    grid = u.lidar.directions
    beam = near.argmin(1)
    az = torch.atan2(grid[beam, 1], grid[beam, 0])
    miss = torch.remainder(az - bearing + math.pi, 2 * math.pi) - math.pi
    max_ok = u.distance_to_pillar() < cfg.lidar.max_range
    print(f"PILLAR lidar: {int(boats.sum())}/{u.num_envs} boats have beams that saw their pillar "
          f"(mean {saw.sum(1).float().mean():.1f} beams); range error max {(got - expected)[boats].abs().max():.2f} m, "
          f"bearing error max {torch.rad2deg(miss[boats].abs()).max():.1f} deg; "
          f"boats with no pillar return: {(~boats).nonzero().flatten().tolist()}", flush=True)

    # camera: orange pixels, and which boats have the pillar inside the field of view
    img = u.camera.data.output["rgb"].torch.float()
    orange = (img[..., 0] > 150) & (img[..., 1] < 150) & (img[..., 2] < 90)
    pixels = orange.sum(dim=(1, 2))
    in_view = torch.rad2deg(torch.atan2(body[:, 1], body[:, 0]).abs()) < 22.0
    print(f"PILLAR camera: orange pixels per boat {pixels.tolist()}", flush=True)
    print(f"PILLAR pillar within the camera's view on {int(in_view.sum())} boats; of those, {int((pixels[in_view] > 3).sum())} show orange; "
          f"boats with it out of view but showing orange: {int((pixels[~in_view] > 3).sum())}", flush=True)
    save_images_to_file(img / 255.0, os.path.join(args.out, "cams.png"))
    print("PILLAR DONE", flush=True)
    env.close()
