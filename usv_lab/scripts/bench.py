"""One benchmark run: N boats, one sensor set, one sea. Appends one row to a CSV.

    ./isaaclab.sh -p <repo>/usv_lab/scripts/bench.py --num_envs 1024 --sensors all --out ~/bench/x/results.csv

Phases (boats driven by held random commands, re-drawn every 5 s):
  1. warm-up    not measured (first steps include one-time setup: kernels, shaders)
  2. clean      timer off, one GPU sync at each end: the headline throughput
  3. breakdown  timer on, GPU sync around every part: ms per part (the syncs slow it slightly)
Each step also reads every enabled sensor, as a policy would.
Stop anything else using the GPU (the live stream) first.
"""

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

SENSOR_SETS = {"none": (), "imu": ("imu",), "camera": ("camera",), "lidar": ("lidar",), "all": ("imu", "camera", "lidar")}

ours = argparse.ArgumentParser(add_help=False)
ours.add_argument("--num_envs", type=int, default=16)
ours.add_argument("--sensors", choices=list(SENSOR_SETS), default="none")
ours.add_argument("--sea", choices=["waves", "flat"], default="waves", help="flat: no wave maths at all (plain Fossen)")
ours.add_argument("--water", type=int, default=0, help="boats with a drawn water mesh (CPU-bound today)")
ours.add_argument("--cam_res", type=str, default="128x96", help="camera width x height")
ours.add_argument("--warmup", type=int, default=50)
ours.add_argument("--steps", type=int, default=300, help="policy steps per measured phase")
ours.add_argument("--out", type=str, default="~/bench/adhoc/results.csv")
ours.add_argument("--tag", type=str, default="")
ours.add_argument("--feed", action="store_true", help="live feed on (to measure its cost)")
ours.add_argument("--feed_sensors", action="store_true", help="feed also shows camera and lidar")
mine, rest = ours.parse_known_args()
parser = argparse.ArgumentParser(parents=[ours])
add_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0], *rest]
for k, v in vars(mine).items():
    setattr(args, k, v)

import csv  # noqa: E402
import datetime  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import usv_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

TASK = "Usv-Float-Direct-v0"
INSIDE = ["physics", "render", "write_sim", "scene_update", "forces", "lidar", "treadmill", "water", "reset"]
PARTS = INSIDE + ["other", "total", "read_sensors"]   # other = total - the parts above (Isaac Lab bookkeeping)
COLUMNS = (
    ["tag", "time", "status", "num_envs", "sensors", "sea", "water_envs", "cam_res", "live_feed", "warmup", "steps",
     "physics_dt", "decimation", "rendering", "wall_s", "env_steps_per_s", "sim_s_per_real_s_per_boat",
     "sim_s_per_real_s_total", "ms_per_step_clean", "ms_per_step_timed"]
    + [f"{p}_{s}" for p in PARTS for s in ("ms", "p95_ms")]
    + ["gpu_mem_torch_gb", "gpu_mem_process_gb", "gpu_mem_device_gb", "resets", "nonfinite", "cam_bad_frac",
       "gpu", "driver", "kit", "isaaclab", "torch", "commit", "error"]
)


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def gpu_memory_gb():
    """(this process, whole device) in GB, from nvidia-smi: includes what PhysX and the renderer hold."""
    proc = 0.0
    for line in sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        if pid == str(os.getpid()):
            proc = float(mem) / 1024
    dev = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).splitlines()
    return proc, (float(dev[0]) / 1024 if dev else 0.0)


def versions():
    v = {"torch": torch.__version__}
    q = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).split(", ")
    v["gpu"], v["driver"] = (q + ["", ""])[:2]
    try:
        import omni.kit.app

        v["kit"] = omni.kit.app.get_app().get_build_version()
    except Exception:
        v["kit"] = ""
    try:   # the release (VERSION file at the repo root), then the python package's own number
        import pathlib
        from importlib.metadata import version

        import isaaclab

        release = (pathlib.Path(isaaclab.__file__).resolve().parents[3] / "VERSION").read_text().strip()
        v["isaaclab"] = f"{release} (pkg {version('isaaclab')})"
    except Exception:
        v["isaaclab"] = ""
    here = os.path.dirname(os.path.abspath(__file__))
    commit = sh(["git", "-C", here, "rev-parse", "--short", "HEAD"])
    dirty = sh(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"])
    v["commit"] = commit + ("-dirty" if dirty else "")
    return v


def write_row(row):
    path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})
    print(f"BENCH row written to {path}", flush=True)


sensors = SENSOR_SETS[args.sensors]
launcher_argv, sys.argv = sys.argv, sys.argv[:1]
cfg, _ = resolve_task_config(TASK, "")
sys.argv = launcher_argv
cfg.scene.num_envs = args.num_envs
if "imu" not in sensors:
    cfg.scene.imu = None
if "camera" not in sensors:
    cfg.scene.top_cam = None
else:
    cfg.scene.top_cam.width, cfg.scene.top_cam.height = (int(x) for x in args.cam_res.lower().split("x"))
if "lidar" not in sensors:
    cfg.lidar = None
if args.sea == "flat":
    cfg.sea = None
cfg.draw_water_envs = args.water
cfg.live_feed = args.feed or args.feed_sensors
cfg.live_feed_sensors = args.feed_sensors

row = {"tag": args.tag, "time": datetime.datetime.now().isoformat(timespec="seconds"), "status": "started",
       "num_envs": args.num_envs, "sensors": args.sensors, "sea": args.sea, "water_envs": args.water,
       "cam_res": args.cam_res if "camera" in sensors else "", "live_feed": "sensors" if args.feed_sensors else ("metrics" if args.feed else ""), "warmup": args.warmup, "steps": args.steps,
       "physics_dt": cfg.sim.dt, "decimation": cfg.decimation}

with launch_simulation(cfg, args):
    try:
        row.update(versions())
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        n = u.num_envs
        row["rendering"] = bool(u.sim.is_rendering)
        gen = torch.Generator(device=u.device).manual_seed(0)
        mem_proc = 0.0

        def commands():
            surge = torch.rand(n, 1, device=u.device, generator=gen)
            turn = torch.rand(n, 1, device=u.device, generator=gen) * 2 - 1
            return torch.cat([surge, turn], -1)

        def read_sensors():
            """What a policy would consume each step."""
            out = []
            if u.imu is not None:
                out += [u.imu.data.lin_acc_b.torch, u.imu.data.ang_vel_b.torch]
            if u.camera is not None:
                out += [u.camera.data.output["rgb"].torch, u.camera.data.output["distance_to_image_plane"].torch]
            if u.lidar is not None:
                out.append(u.lidar_ranges)
            return out

        state = {"k": 0, "action": None, "resets": 0}

        def run(steps, timed):
            for _ in range(steps):
                if state["k"] % 150 == 0:                       # new held command every 5 s
                    state["action"] = commands()
                state["k"] += 1
                with torch.inference_mode():
                    if timed:
                        with u.timer("total"):
                            _, _, term, trunc, _ = env.step(state["action"])
                        with u.timer("read_sensors"):
                            read_sensors()
                        u.timer.end_step()
                    else:
                        _, _, term, trunc, _ = env.step(state["action"])
                        read_sensors()
                state["resets"] += int((term | trunc).sum())

        env.reset()
        print(f"BENCH n={n} sensors={args.sensors} sea={args.sea} water={args.water} rendering={row['rendering']}", flush=True)

        run(args.warmup, timed=False)
        mem_proc = max(mem_proc, gpu_memory_gb()[0])
        state["resets"] = 0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(args.steps, timed=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        mem_proc = max(mem_proc, gpu_memory_gb()[0])
        sim_per_real = args.steps * u.step_dt / wall
        row.update(wall_s=round(wall, 3), env_steps_per_s=round(n * args.steps / wall, 1),
                   sim_s_per_real_s_per_boat=round(sim_per_real, 3), sim_s_per_real_s_total=round(n * sim_per_real, 1),
                   ms_per_step_clean=round(wall / args.steps * 1e3, 3))
        print(f"BENCH clean: {row['env_steps_per_s']} env-steps/s, {row['ms_per_step_clean']} ms/step, "
              f"{row['sim_s_per_real_s_per_boat']} sim s per real s per boat", flush=True)

        u.timer.clear()
        u.timer.enabled = True
        run(args.steps, timed=True)
        u.timer.enabled = False
        total = u.timer.series("total")
        other = total - sum(u.timer.series(p) for p in INSIDE)
        summary = u.timer.summary()
        summary["other"] = (float(other.mean()), float(torch.quantile(other, 0.95)))
        for p in PARTS:
            mean, p95 = summary.get(p, (0.0, 0.0))
            row[f"{p}_ms"], row[f"{p}_p95_ms"] = round(mean, 3), round(p95, 3)
        row["ms_per_step_timed"] = row["total_ms"]
        print("BENCH breakdown (mean ms per policy step): "
              + "  ".join(f"{p} {row[p + '_ms']}" for p in PARTS), flush=True)

        s = u.boat_state()
        row["nonfinite"] = int((~torch.isfinite(s.pos).all(-1)).sum())
        row["resets"] = state["resets"]
        if u.camera is not None:
            rgb = u.camera.data.output["rgb"].torch.float() / 255.0          # [N, H, W, C]
            mean = rgb.mean(dim=(1, 2, 3))
            std = rgb.flatten(1).std(dim=1)
            bad = ~torch.isfinite(mean) | (mean < 0.01) | (std < 1e-3)    # black, NaN, or one flat colour
            row["cam_bad_frac"] = round(float(bad.float().mean()), 4)
        mem_proc_now, mem_dev = gpu_memory_gb()
        row.update(gpu_mem_torch_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   gpu_mem_process_gb=round(max(mem_proc, mem_proc_now), 3), gpu_mem_device_gb=round(mem_dev, 3),
                   status="ok")
        env.close()
    except torch.cuda.OutOfMemoryError as e:
        row.update(status="oom", error=str(e).splitlines()[0][:200])
        traceback.print_exc()
    except Exception as e:
        row.update(status="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
        traceback.print_exc()
    write_row(row)
print("BENCH DONE", flush=True)
