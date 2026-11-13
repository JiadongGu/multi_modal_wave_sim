# PLAN: what Tim asked for, as a checklist

Scope: tooling to run a dynamics model (Eric's ROM, or our stand-ins) on many boats in Isaac Lab,
add sensors, and measure how far it scales. RL itself (policies, rewards, training runs) is out of
scope here; Cody owns that. Work top to bottom inside each section; sections B and C can run in parallel.

Legend: `[x]` done · `[ ]` to do · **blocked:** needs someone else first.

## A. Dynamics model plug-in (ready for Eric's ROM)

- [x] `HullModel` socket, `ModelContext`, `Composite` (`usv_core/model.py`)
- [x] `ConventionAdapter` for models in another convention, e.g. Fossen FRD, 6-vector tau (`usv_core/adapters.py`)
- [x] Applied in Isaac Lab every physics substep, world-frame wrench at the CoM (`usv_lab/tasks/usv_env.py`)
- [ ] Write the input/output contract for Eric: inputs, frames, units, time step, memory/reset, and which forces
      the ROM owns (buoyancy? waves? propulsion?) so nothing is counted twice. `docs/ROM_INTERFACE.md`.
      **blocked:** answers from Eric and Tim.
- [ ] `TorchModel`: load a saved PyTorch model file as a `HullModel` (batched on the GPU, optional history window)
- [ ] Cost knob: an untrained network of configurable size run every step with zero effect, standing in
      for a heavy ROM so its cost can be measured before it exists

## B. Dynamics versions to benchmark

- [ ] Plain Fossen: flat water + buoyancy + damping + thrust (config switch, no new physics)
- [x] Waves + Fossen: the current stack (sea + buoyancy + damping + thrust)
- [ ] BEM + waves + Fossen: a `HullModel` from BEM coefficients (added mass, radiation damping, excitation).
      **blocked:** the BEM data or tool the team used.
- [ ] Added mass term (needed for a complete Fossen; uses `ctx.lin_acc` from the previous step)
- [ ] Compare against the existing (Cody's) implementation. **blocked:** access to that code.

## C. Sensors

- [x] IMU on every boat (`ImuCfg`, checked at rest and under thrust)
- [x] Top camera on every boat, isolated per boat (checked with a negative control)
- [x] Lidar on every boat against its own sea (`usv_core/lidar.py`)
- [ ] Waves visible to cameras at training scale: move the water mesh update to the GPU (today: CPU, ~16 boats max)
- [ ] Radar: decide between Isaac Sim's experimental RTX radar and a simple analytic model; implement one
- [ ] Measure the per-sensor cost (see D)

## D. Scaling benchmark (Tim's main question)

- [x] Timing breakdown inside the environment: physics, usv_core forces, lidar, water drawing, rendering (`tasks/timing.py`)
- [x] Benchmark scripts (`scripts/bench.py` one run, `scripts/sweep.py` the grid): boats × flat/waves × sensor set
      × camera resolution; headless, held random commands, warm-up excluded, GPU synchronised. Cost-knob size: to do
- [x] Record per run: env-steps/s, simulated s per real s, ms per step by component, peak GPU memory,
      reset count, GPU/driver/Isaac versions; flag black or corrupted camera images at scale
- [ ] Find each configuration's ceiling (out of memory, or no further gain)
- [ ] Results table and plots for Tim (CSV plus a page or doc)
- [x] Live viewer (`tasks/live_feed.py` + `scripts/live_view.py`). Cost at 1,024 boats, all sensors:
      metrics +0.4%, with camera and lidar +0.6% (identical runs differ by 0.1%). `bench/results/feed/`
- [x] First sweep, 128-1,024 boats: `bench/results/first/`. Fixed ~20 ms/step in usv_core forces dominates
      without sensors; lidar on waves and camera read-out dominate with them
- [x] usv_core forces: no per-step CPU->GPU copies, column depths computed once. Forces 19.9 -> 16.3 ms/step,
      1,024 boats no sensors 41.5k -> 49.3k env-steps/s. `bench/results/forcefix/`
- [ ] usv_core forces: the remaining ~16 ms is hundreds of tiny GPU launches; try torch.compile / CUDA graphs
- [ ] Lidar on waves: 77 ms/step at 1,024 boats (largest single cost with sensors on); faster surface search
- [ ] Camera read-out: 63 ms/step at 1,024 boats (128x96); check Isaac Lab's tiled camera

## E. Testbeds

- [x] Wigley hull (analytic, columns)
- [ ] BlueBoat: build hull columns from a mesh (`HullCfg(shape="mesh")`), reusing `tools/load_hull.py`'s loading

## F. Housekeeping

- [ ] Verify the treadmill jump: IMU and camera readings across a recentre (untested)
- [ ] Spawn transient in waves: at full throttle from rest, one of 16 boats dipped to -6 m and 70 deg tilt
      in the first 4 s, then recovered (smoke.py, Hs 0.5-2 m). Likely zero spawn velocity on a moving surface.
- [ ] `usv_core`/`usv_lab` section in `docs/METHODS.md`
- [ ] Pull request: `usv-core` → `main`, after review

## Removed (2026-11-11), to keep the code lean

RL tasks and training configs (station keeping, pillar navigation), the old training dashboard and
sensor feed (replaced by the live viewer in D), the camera snapshot script, and lidar cylinders. All are in git history if needed.
Not planned: reward tuning, a trajectory-tracking task, photorealistic water, RTX lidar.
