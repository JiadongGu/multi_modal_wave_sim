"""Run bench.py over a grid of configurations, one Isaac process each. Plain python3, no Isaac imports.

    python3 <repo>/usv_lab/scripts/sweep.py --name first --boats 128 256 512 1024 --sensors none imu camera lidar all

Writes <root>/<name>/results.csv (one row per finished run, from bench.py),
<root>/<name>/runs.csv (every run, including crashes and timeouts) and logs/<run>.log.
Within each (sensors, sea, camera resolution) series the boat counts go up; after a failure the larger counts are
skipped, since the failure is that configuration's ceiling.
"""

import argparse
import csv
import datetime
import os
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--root", default="~/bench")
p.add_argument("--boats", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
p.add_argument("--sensors", nargs="+", default=["none", "imu", "camera", "lidar", "all"])
p.add_argument("--seas", nargs="+", default=["waves", "flat"])
p.add_argument("--cam_res", nargs="+", default=["128x96"], help="camera runs repeat at each resolution")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--timeout", type=int, default=1200, help="seconds per run")
p.add_argument("--isaaclab", default="~/IsaacLab")
args = p.parse_args()

out = os.path.expanduser(os.path.join(args.root, args.name))
os.makedirs(os.path.join(out, "logs"), exist_ok=True)
bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
results, runs = os.path.join(out, "results.csv"), os.path.join(out, "runs.csv")


def rows_in(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def log_run(entry):
    new = not os.path.exists(runs)
    with open(runs, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(entry))
        if new:
            w.writeheader()
        w.writerow(entry)


plan = [(s, sea, res if s in ("camera", "all") else "", n) for s in args.sensors for sea in args.seas
        for res in (args.cam_res if s in ("camera", "all") else [""]) for n in sorted(args.boats)]
print(f"SWEEP {args.name}: {len(plan)} runs -> {out}", flush=True)
failed_series = set()
for i, (sensors, sea, res, n) in enumerate(plan, 1):
    run = f"{n}_{sensors}_{sea}" + (f"_{res}" if res else "")
    if (sensors, sea, res) in failed_series:
        log_run({"run": run, "status": "skipped", "exit": "", "seconds": 0, "time": ""})
        continue
    before = len(rows_in(results))
    cmd = [os.path.join(os.path.expanduser(args.isaaclab), "isaaclab.sh"), "-p", bench, "--num_envs", str(n),
           "--sensors", sensors, "--sea", sea, "--cam_res", res or "128x96", "--steps", str(args.steps),
           "--out", results, "--tag", args.name]
    t0 = time.time()
    with open(os.path.join(out, "logs", run + ".log"), "w") as log:
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                                  cwd=os.path.expanduser(args.isaaclab)).returncode
            status = None
        except subprocess.TimeoutExpired:
            code, status = "", "timeout"
            subprocess.run(["pkill", "-f", "bench.py"])     # the launcher's children survive a timeout
    new_rows = rows_in(results)[before:]
    if status is None:
        status = new_rows[-1]["status"] if new_rows else "crash"
    secs = round(time.time() - t0)
    log_run({"run": run, "status": status, "exit": code, "seconds": secs,
             "time": datetime.datetime.now().isoformat(timespec="seconds")})
    speed = f"{new_rows[-1]['env_steps_per_s']} env-steps/s" if new_rows and status == "ok" else ""
    print(f"SWEEP [{i}/{len(plan)}] {run}: {status} {speed} ({secs} s)", flush=True)
    if status != "ok":
        failed_series.add((sensors, sea, res))
print("SWEEP DONE", flush=True)
