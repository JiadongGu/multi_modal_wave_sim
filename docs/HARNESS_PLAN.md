# Task harness: implementation plan

Draft for review, 2026-11-25. Written without lab access (machine offline since ~11-20), so nothing
here has been tried on Isaac Lab 3.0 yet; items that depend on its behaviour are marked **verify**.

## Goal

A standard way to build custom USV environments from reusable parts, so a new task (refuelling
alongside a ship in high seas, avoiding obstacles and traffic, docking) is mostly a config file:
pick the vessels and objects in the world, the sea state, the objective, the observations, and the
success metric. Training, evaluation and benchmarking then work the same way for every task.

Not in scope here: tuning rewards to get good policies (Cody), the ROM itself (Eric), ship-to-ship
hydrodynamics (research question, phase 5).

## Layers

```
TASK      objective, rewards, success/failure, observations, curriculum, evaluation   (new)
SCENARIO  vessels, obstacles, docks, their behaviour, sea state, current            (new)
ENGINE    usv_core physics + ROM socket, sea, sensors, treadmill, scaling           (exists)
```

## Decisions needed first (proposed defaults in bold)

| # | Decision | Proposed | Why / alternative |
|---|---|---|---|
| D1 | Task config format | **Isaac Lab's manager-based environments** (observation, reward, termination, event, curriculum, command terms) | The standard RL people already know; Isaac Lab's train scripts use it. Alternative: keep the current "direct" env and write a thin term registry ourselves (only if D1 costs too much speed, see P0) |
| D2 | Ego vessel for the reference tasks | **Configurable; start with the 6 m Wigley, add BlueBoat** | Which hull matters for sea states: a 1.2 m BlueBoat in sea state 5 is mostly underwater. Should match the hull Eric's ROM is for (GPPH?) |
| D3 | Relationship with USVBench | **Reuse its task ideas and evaluation habits on our engine; ask the owners before copying code/assets** (licence TBD) | Alternatives: co-develop with them, or stay separate and compare |
| D4 | Sea-state naming | **WMO sea-state codes** mapped to JONSWAP Hs/Tp ranges | "Sea state 5" then means the same thing to everyone |
| D5 | Where it runs at scale | **Container image usable on CoreWeave / Great Lakes**, lab machine for development | The lab machine has dropped off for days twice in two weeks |

## Code layout (target)

```
usv_core/usv_core/            engine, no Isaac imports (tested on any machine)
  fleet.py                    NEW  many vessels per world: rows = (world, vessel), each row knows its world
  world.py                    NEW  obstacles and docks as shapes; distance and ray queries
  traffic.py                  NEW  scripted vessel motion: steady course, waypoints, (later) rules of the road
  waves.py, lidar.py, ...     CHANGED  sea queries take a row -> world index; lidar also hits shapes and vessels
usv_lab/usv_lab/
  env/usv_env_base.py         NEW  manager-based env base: treadmill for every entity, water, timers, live feed
  mdp/                        NEW  term library (functions, each a few lines)
    actions.py                     thrust + hydrodynamics as one action term (runs every physics substep)
    observations.py                proprio, relative pose to a target, lidar, camera, IMU
    rewards.py                     pose error, progress, clearance, collision, energy, smoothness
    terminations.py                capsize, collision, out of bounds, held-for-N-seconds success
    events.py                      sea state by WMO code, spawn randomisation, traffic encounter sampling
    curriculum.py                  sea-state ladder, traffic density, obstacle count
  scenarios/                  NEW  entity configs: VesselCfg (hull, physics or scripted, behaviour), ObstacleCfg, DockCfg
  tasks/<task>/               NEW  one config file + evaluation spec per task
scripts/
  eval_task.py                NEW  standard evaluation: fixed seeds, metrics to JSON/CSV
  bench.py, sweep.py          CHANGED  --task, so every task is benchmarked the same way
```

What a task file should look like (illustrative, final names follow Isaac Lab's conventions):

```python
@configclass
class RefuelAlongsideCfg(UsvTaskCfg):
    sea = SeaStatePreset(wmo=(3, 6))                         # curriculum from SS3 up to SS6
    scene = Scene(ego=Wigley(), others={"tanker": Vessel(length=60, behaviour=Steady(knots=5))})
    commands = RelativeStation(target="tanker", offset=(0.0, 25.0), tol_m=2.0)
    observations = Obs(policy=["proprio", "relative_pose:tanker"], sensors=["lidar"])
    rewards = Rew(pose_error=-1.0, heading_error=-0.3, action_rate=-0.01)
    terminations = Term(capsize=True, collision="tanker", success=HoldFor(seconds=60))
    evaluation = Eval(seeds=range(64), metrics=["rms_pose_error", "time_in_tolerance", "success"])
```

## Phases

Estimates are focused working days and are rough.

### P0. Check the manager-based route on Isaac Lab 3.0 (0.5–1 day, needs the lab)
- Port the current float environment to a manager-based version: same physics, same treadmill, same sensors.
- **verify:** where hydrodynamics belongs. An action term's per-substep apply step looks right; the
  treadmill and water drawing need a hook before physics.
- **Pass:** `smoke.py` / `check_*.py` numbers unchanged, and throughput within 5% of the current env at 1,024 and 4,096 boats.
- **If it fails** (more than 10% slower): keep the direct env and build a small term registry ourselves (D1 alternative).

### P1. Manager-based base env + term library basics (1–2 days)
- `UsvEnvBase`, the thrust + hydrodynamics action term, basic observation / reward / termination terms.
- The float task rewritten as the first config-only task.
- Container image (Dockerfile) with Isaac Lab + our packages; run it once on the lab machine.
- **Pass:** float task trains with Isaac Lab's standard PPO script for a few iterations; benchmark parity with P0.

### P2. Many entities per world (3–5 days, the riskiest phase)
- **2a. Fleet batching in usv_core.** Vessels of every world in one batch. Each row carries its world index, so all
  vessels in a world share one sea and one treadmill offset. One force model per vessel class (different hulls).
  - **Pass (unit tests, no lab):** a vessel's forces are identical whether simulated alone or in a fleet;
    two vessels in one world see the same wave at the same point.
- **2b. Objects and scripted traffic.** Obstacles and docks as shapes (circle, capsule, box, polygon). Scripted
  vessels move on the GPU, and optionally ride the surface so a tanker heaves and rolls. Distance and collision
  checks. The lidar hits shapes and vessels as well as the sea.
  - **Pass:** lidar and distance results match hand-computed geometry; relative positions are unchanged across a
    treadmill jump.
- **2c. Isaac scene.** Other vessels and objects appear in each world, so cameras see them. Option for a second
  physics-driven vessel.
  - **Pass:** at 1,024 worlds with 2 vessels each, cameras and lidar see the other vessel; the cost per extra
    entity is measured.

### P3. Three reference tasks (2–3 days each)

| Task | World | Success and metrics |
|---|---|---|
| T1 Refuel alongside | Ego + tanker on a steady course, sea-state curriculum SS3 to SS6 | Hold 25 m abeam ±2 m for 60 s; RMS pose error, % time in tolerance, success rate |
| T2 Obstacles + traffic | Goal behind static obstacles; N scripted vessels in head-on, crossing and overtaking encounters | Reach goal with no collision; minimum clearance, closest point of approach, path efficiency (rules-of-the-road score later) |
| T3 Docking | Pier and berth, optional current, lower sea states | Final position and heading error, impact speed, success rate |

Each task ships:
- a zero-action and random baseline,
- a short PPO run showing it learns at all,
- an evaluation script with fixed seeds,
- a README entry.

These are the reference results, not tuned policies.

### P4. Scale and sensors (2 days, needs the lab or cloud)
- Benchmark the three tasks in each observation tier (privileged, onboard, camera/lidar) with the existing sweep tooling.
- New charts: cost per extra vessel or object, and memory ceilings per task.

### P5. Fidelity (later, with Tim and Eric)
- Physics-driven tanker.
- Ship-to-ship interaction: suction between hulls, wave sheltering.
- Docking contact (fenders, collision shapes).
- Eric's ROM in the tasks.
- Radar, once Tim picks the type.

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Manager-based overhead at thousands of worlds | Slower than today | P0 measures it before anything is built on it |
| Fleet batching with mixed hull types | Bugs where vessels in one world disagree about the sea | 2a done first, with tests that run without the lab |
| More objects per camera | Render cost and memory rise; camera ceiling drops below 2,048 | Measure per entity in 2c; keep objects simple |
| Small ego boat in high seas | Boat spends its time underwater; task meaningless | Decide D2 with Tim; sea-state curriculum per hull |
| Overlap with USVBench | Duplicate work, or friction | D3 decided with Tim and the owners before P3 |
| Lab machine reliability | Days lost | Container (P1) so work can move to CoreWeave or Great Lakes |

## Order

P0 → P1 → P2a (can start now, no lab needed) → P2b → P2c → P3 (T1, then T2, then T3) → P4 → P5.

Total to three reference tasks at scale: roughly 3–4 weeks of focused work, assuming the decisions above
hold and the lab (or a cloud GPU) is available from P0 on.
