"""Lidar on every boat, in the running environment: hits on that boat's own sea, plus GPU timing.

    ./isaaclab.sh -p <repo>/usv_lab/scripts/check_lidar.py --num_envs 16
"""

import argparse
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument("--num_envs", type=int, default=16)
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
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from usv_core import SeaCfg, SineWaveField, BodyState, SeaLidar  # noqa: E402

TASK = "Usv-Float-Direct-v0"
launcher_argv, sys.argv = sys.argv, sys.argv[:1]
cfg, _ = resolve_task_config(TASK, "")
sys.argv = launcher_argv
cfg.scene.num_envs = args.num_envs
cfg.scene.top_cam = None
cfg.scene.imu = None

with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped
    action = torch.zeros(u.num_envs, 2, device=u.device)
    action[:, 0] = 1.0                                   # full throttle, so the treadmill is in play
    worst, returns, snapshots = 0.0, [], []
    for i in range(900):
        with torch.inference_mode():
            env.step(action)
        if i % 30 == 29:
            r, h = u.lidar_ranges, u.lidar_hits
            good = torch.isfinite(r) & (r > 0)
            # the scan was taken at the start of a policy step, within one lidar period of the current sea time
            scan_t = u._next_lidar_time - cfg.lidar.update_period
            eta = u.sea.height(torch.nan_to_num(h[..., :2], posinf=0.0), scan_t)
            worst = max(worst, (h[..., 2] - eta)[good].abs().max().item() if bool(good.any()) else 0.0)
            returns.append(good.float().mean().item())
            snapshots.append(r[:2].clone())
    a, b = torch.nan_to_num(snapshots[-1][0], posinf=-1.0), torch.nan_to_num(snapshots[-1][1], posinf=-1.0)
    offset = u.sea_offset[:, :2].norm(dim=-1)
    print(f"LIDAR rays per boat {u.lidar.num_rays}; returns {100 * sum(returns) / len(returns):.1f}% of rays; "
          f"worst |hit z - sea height| {worst:.4f} m; boats 0 and 1 differ: {not torch.equal(a, b)}; "
          f"treadmill offset mean {offset.mean():.1f} m", flush=True)

    # timing of one scan, for the sizes we care about, on this GPU
    for n in (16, 256, 1024):
        sea = SineWaveField(cfg.sea, n, device=u.device, seed=0)
        lidar = SeaLidar(cfg.lidar, device=u.device)
        d = u.device
        pos = torch.zeros(n, 3, device=d)
        pos[:, 2] = cfg.hull.com[2]
        quat = torch.zeros(n, 4, device=d)
        quat[:, 3] = 1.0
        state = BodyState(pos, quat, torch.zeros(n, 3, device=d), torch.zeros(n, 3, device=d))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            lidar.scan(state, sea, 0.0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for k in range(5):
                lidar.scan(state, sea, k * 0.1)
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1e3
        print(f"LIDAR timing n={n:5d}: {ms:8.1f} ms per scan, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print("LIDAR DONE", flush=True)
    env.close()
