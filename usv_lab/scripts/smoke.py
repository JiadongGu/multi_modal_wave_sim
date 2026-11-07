"""Headless check: boats are created, float near their design draft, and nothing goes NaN.

    ./isaaclab.sh -p <repo>/usv_lab/scripts/smoke.py --num_envs 16 --steps 300 [--sea flat]
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300, help="policy steps (each is `decimation` physics steps)")
parser.add_argument("--sea", choices=["waves", "flat"], default="waves")
parser.add_argument("--throttle", type=float, default=0.0)
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import usv_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "Usv-Float-Direct-v0"
cfg, _ = resolve_task_config(TASK, "")
cfg.scene.num_envs = args.num_envs
if args.sea == "flat":
    cfg.sea.hs = (1e-6, 1e-6)

with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped
    design_z = cfg.hull.com[2]
    action = torch.tensor([[args.throttle, 0.0]], device=u.device).repeat(u.num_envs, 1)
    for i in range(args.steps):
        with torch.inference_mode():
            env.step(action)
        if i % 60 == 0 or i == args.steps - 1:
            s = u.boat_state()
            z = s.pos[:, 2] - design_z
            speed = s.lin_vel_body[:, 0]
            print(f"SMOKE step {i:4d}  t={u.sim_time:6.2f}s  heave vs design: mean {z.mean():+.3f} m"
                  f" [{z.min():+.3f}, {z.max():+.3f}]  surge {speed.mean():+.2f} m/s"
                  f"  finite={bool(torch.isfinite(s.pos).all())}", flush=True)
    env.close()
print("SMOKE DONE", flush=True)
