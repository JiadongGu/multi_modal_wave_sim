# ROM interface: how a reduced-order model plugs into the simulator

Draft for Eric, 2026-11-13. Section 1 describes what exists and works today. Section 2 lists what
we need to agree on. Nothing here is final: if your model needs something different, we change our
side.

## 1. What the simulator does with a ROM

Each physics step, which is 1/120 s of simulated time, the simulator gathers your model's inputs for
every boat at once (N boats, batched, on the GPU). It runs your network once on all of them and
receives a force and a torque per boat. The physics engine (PhysX) then moves the boats.

- **The engine owns the rigid body:** mass, inertia, gravity and the equations of motion. Your
  output must not include the boat's weight.
- **We keep the thrust model** (two outboard motors) unless you tell us otherwise.
- **Two ways to use a ROM:**
  - `replace`: your output replaces our buoyancy and damping, so the ROM owns all water forces.
  - `add`: your output is added on top of ours, as a correction.

  We need to know which one (question 1).

### Delivery: two files with the same name

| File | What |
|---|---|
| `name.pt` | The network, saved with TorchScript (`torch.jit.script(model).save(...)`, or `torch.jit.trace`) |
| `name.json` | A description: inputs, axes, frame, scaling, history (format below) |

`docs/rom_example.py` writes both files for a small example network. Our tests check that its files
load and run in the simulator as they are.

### Inputs you can ask for

List them in `inputs`, in the order your network expects. Units are SI. Body-frame quantities come in
your chosen axis convention: `FRD` (x forward, y starboard, z down, as in Fossen) or `FLU` (x forward,
y port, z up).

| Name | Size | Meaning |
|---|---|---|
| `lin_vel_body` | 3 | Velocity of the centre of mass, body frame [m/s] |
| `lin_vel_rel_body` | 3 | The same, relative to the water: the wave orbital velocity at the CoM subtracted [m/s] |
| `ang_vel_body` | 3 | Angular velocity, body frame [rad/s] |
| `gravity_body` | 3 | Unit "down" vector in the body frame. Carries roll and pitch without angle wrap-around |
| `heave` | 1 | Height of the CoM above the still-water level [m] |
| `actions` | 2 | Motor commands (surge, turn) in [-1, 1], after the actuator lag |
| `wave_height` | P | Wave elevation [m] at P points fixed to the hull (`wave_points`, body-frame [x, y] in metres), rotating with the boat |
| `sea_state` | 2 | Significant wave height Hs [m] and peak period Tp [s] of that boat's sea |

The waves are a sum of 32 random sine components per boat, drawn from a JONSWAP spectrum. Each boat
has its own sea. We can add any other input the simulator can compute, for example accelerations,
wave direction relative to the bow, or more wave points.

### Output

The network returns `[N, 6]`: force (3), then torque (3), in N and N·m.

- `output_frame`: `body` (in your axis convention) or `world` (z up; only with `FLU`).
- `reference_point`: the point your torque is about, in the body frame, metres from the centre of
  mass. We move the torque to the CoM ourselves.
- `output_scale` / `output_offset`: if the network's outputs are normalised; the result becomes
  `output * scale + offset`.

### Normalisation and history

- `input_mean` / `input_std`: one value per input feature. The network receives `(x - mean) / std`.
- `history`: how many steps the network sees, oldest first.
  - `layout`: `flat` gives `[N, history * features]`; `sequence` gives `[N, history, features]`,
    for recurrent or convolutional models.
  - The history is cleared when a boat is reset.

### What it costs today

Measured on an RTX 4090 with untrained stand-in networks of different sizes, on waves, no sensors.
The input list is the one in the example (20 features).

| | 1,024 boats | 4,096 boats |
|---|---|---|
| Simulator without a ROM | 21.4 ms per step | 21.4 ms |
| With a 3 × 256 network | 28.2 ms | 28.7 ms |
| With a 6 × 1,024 network | 29.0 ms | 30.9 ms |

Network size matters little in this range; most of the extra ~7.5 ms is gathering the inputs. Many
more inputs, such as hundreds of wave points, would change that.

## 2. Questions for you

1. **Which forces does the ROM produce?** All water forces (buoyancy, wave excitation, radiation,
   viscous), or only part of them, with our buoyancy and damping kept? This decides `replace` or
   `add`, and avoids counting anything twice.
2. **What does it know about the waves?** Hs and Tp only, elevation at points around the hull, the full
   wave components, or regular waves only? Which sea states and headings was it trained on?
3. **Other inputs:** does it need accelerations (added mass), the previous output, or anything not in
   the table above?
4. **Time step and memory:** what step was it trained at? We run at 1/120 s and can sub-sample. Does it
   need a history window, and how long?
5. **Axes and reference:** body or world frame, FRD or FLU, and which point the torque is about.
6. **The hull:** GPPH dimensions, mass, inertia and CoG, so our rigid body matches what the ROM was
   trained on. Should we model the GPPH hull in our simulator, or is the ROM meant for another hull?
7. **Valid range:** speeds, angles and sea states it is trustworthy for, and what we should do outside
   that range.
8. **Format and size:** is TorchScript fine? We could add ONNX. Roughly how large is the network, and
   is it recurrent?
9. **Timing:** when could a first rough version be ready for an integration test? An untrained
   network with the final inputs and outputs would already let us check the plumbing and measure the
   cost.
