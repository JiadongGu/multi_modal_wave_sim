"""Watch the boats: random throttle and helm, held for a few seconds at a time, until you stop it.

    ./isaaclab.sh -p <repo>/usv_lab/scripts/view.py --num_envs 16 --livestream 2
    ./isaaclab.sh -p <repo>/usv_lab/scripts/view.py --num_envs 16 --livestream 2 --tiles

--tiles adds a panel beside the main view showing each boat's own top camera (the same sensor
the policy would use, rendered larger for viewing). Each boat's sea is drawn for the boats shown.

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
ours.add_argument("--view_res", type=int, nargs=2, default=(320, 240), help="boat camera resolution for viewing")
ours.add_argument("--no_water", action="store_true", help="do not draw each boat's sea")
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
shown = min(16, args.num_envs)
if not args.no_water:
    cfg.draw_water_envs = shown
if args.tiles:
    from isaaclab_visualizers.kit import KitVisualizerCfg

    cfg.scene.top_cam.width, cfg.scene.top_cam.height = args.view_res
    viz = KitVisualizerCfg(eye=cfg.sim.default_visualizer_cfg.eye, lookat=cfg.sim.default_visualizer_cfg.lookat)
    viz.streaming_view = True                                            # the tiled panel
    viz.streaming_envs = list(range(shown))                              # which boats get a tile
    viz.streaming_sensor_prim_path = "/World/envs/env_.*/Boat/top_cam"   # each boat's own camera
    viz.dock_position = "RIGHT"                                          # beside the main view, not a hidden tab
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
