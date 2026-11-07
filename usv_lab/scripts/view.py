"""Watch the boats: random throttle and helm, held for a few seconds at a time, until you stop it.

    ./isaaclab.sh -p <repo>/usv_lab/scripts/view.py --num_envs 16 --livestream 2
    ./isaaclab.sh -p <repo>/usv_lab/scripts/view.py --num_envs 16 --livestream 2 --tiles

--tiles adds a panel beside the main view with one chase camera per boat (behind and above,
third person; not a training sensor). Camera rendering is switched on automatically for it.

(Isaac Lab's random_agent.py does not load externally registered tasks, hence this script.)
"""

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument("--num_envs", type=int, default=16)
ours.add_argument("--hold", type=float, default=4.0, help="seconds each random command is held")
ours.add_argument("--max_throttle", type=float, default=0.6)
ours.add_argument("--tiles", action="store_true", help="per-boat chase-camera panel")
ours.add_argument("--chase", type=float, nargs=3, default=(-14.0, -8.0, 7.0), help="chase camera offset [m]")
mine, rest = ours.parse_known_args()
parser = argparse.ArgumentParser(parents=[ours])
add_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0], *rest]  # the launcher re-reads argv and rejects flags it does not know
for k, v in vars(mine).items():
    setattr(args, k, v)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import usv_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "Usv-Float-Direct-v0"
launcher_argv, sys.argv = sys.argv, sys.argv[:1]   # Hydra (config loader) reads argv too; give it none
cfg, _ = resolve_task_config(TASK, "")
sys.argv = launcher_argv
cfg.scene.num_envs = args.num_envs
if args.tiles:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    viz = KitVisualizerCfg(eye=cfg.sim.default_visualizer_cfg.eye, lookat=cfg.sim.default_visualizer_cfg.lookat)
    viz.streaming_view = True                                   # the tiled panel
    viz.streaming_envs = list(range(min(16, args.num_envs)))     # which boats get a tile
    viz.streaming_cam_target_prim_path = "/World/envs/*/Boat"    # one follow-camera per boat
    viz.streaming_cam_eye = tuple(args.chase)                   # offset from the boat
    viz.dock_position = "RIGHT"                                 # beside the main view, not a hidden tab
    cfg.sim.visualizer_cfgs = [viz]

with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    env.reset()
    u = env.unwrapped
    hold_steps = max(1, int(args.hold / u.step_dt))
    action = torch.zeros(u.num_envs, 2, device=u.device)
    print("VIEW running: random throttle/helm per boat, Ctrl+C to stop", flush=True)
    step = 0
    while u.sim.is_running():
        if step % hold_steps == 0:
            action[:, 0] = torch.rand(u.num_envs, device=u.device) * args.max_throttle
            action[:, 1] = torch.rand(u.num_envs, device=u.device) * 2 - 1
        with torch.inference_mode():
            env.step(action)
        step += 1
    env.close()
