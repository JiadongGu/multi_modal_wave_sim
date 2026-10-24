# Multi-Modal Wave Sim — Methods and Approach

A running document on the wave and hydrodynamics methodology. Written for
someone joining the project, and for a hydrodynamicist being asked whether
the approach is sound. Software structure is covered only where it affects
the physics.

Last updated: 24 October 2026.

---

## 1. What this is and why it exists

An Isaac Sim extension that simulates a surface vessel on a realistic sea.
It exists because the lab's previous active wave model was two axis-aligned
sine waves,

    eta(x, y, t) = A sin(2*pi*f*x + t) + A sin(2*pi*f*y + t)

which gives a permanent 90-degree crossed pattern, no wind direction, no
notion of how energy clusters around a peak, and no dispersion. Everything
below is an attempt to replace that with something that can be checked
against measurements.

The design goal is that **three things are independently swappable**: the
wave spectrum, the wave field (how a spectrum becomes a moving surface), and
the force model (how a surface becomes forces on a hull). Each is selected by
name at runtime. The reason is validation rather than elegance — being able
to hold two fixed and vary the third is what makes a disagreement
attributable.

---

## 2. Conventions

These are stated because mixing them is the most common source of silent
error in wave code.

- **Angular frequency** `omega` in rad/s throughout, never Hz. There are no
  Hz anywhere in the package.
- **Deep water** is assumed: `omega^2 = g*k`, i.e. `omega = sqrt(g*k)`. This
  holds for depth greater than about half a wavelength. A pond is *not* deep
  water for the longer components, and the finite-depth relation
  `omega^2 = g*k*tanh(k*h)` is not implemented. See §10.
- **Direction** `theta` in radians, in the same frame as the wind direction.
- `g = 9.80665 m/s^2`; fresh water `rho = 1000`, salt water `1025` available.
- Significant wave height `Hs = 4*sqrt(m0)`, where `m0` is the zeroth
  spectral moment — i.e. four times the standard deviation of the surface
  elevation.

---

## 3. Wave spectra

A spectrum answers two separate questions:

    energy(omega)             variance density            [m^2 s / rad]
    spreading(theta, omega)   angular distribution         [1 / rad]

and their product is the directional spectrum that seeds a wave field.

### 3.1 Pierson–Moskowitz

Fully developed sea, written in terms of Hs and peak frequency:

    S(omega) = (5/16) * Hs^2 * omega_p^4 * omega^-5 * exp(-1.25 * (omega_p/omega)^4)

### 3.2 JONSWAP

PM with a peak enhancement, for fetch-limited seas:

    S(omega) = S_PM(omega) * gamma^r * alpha_norm

    r = exp( -(omega - omega_p)^2 / (2 * sigma^2 * omega_p^2) )
    sigma = 0.07 for omega <= omega_p, 0.09 above
    alpha_norm = 1 - 0.287 * ln(gamma)

The `alpha_norm` factor matters: without it, raising gamma adds energy and
the realised Hs no longer matches the requested Hs. With it, JONSWAP reduces
exactly to PM at gamma = 1, which is asserted in the test suite.

### 3.3 ECKV

Elfouhaily, Chapron, Katsaros & Vandemark (1997), "A unified directional
spectrum for long and short wind-driven waves". Written in **wavenumber**
rather than frequency, and spans gravity waves down to centimetre capillary
waves by summing two regimes:

    S(k) = (B_low + B_high) / k^3

`B_low` is the gravity side, shaped by U10 and inverse wave age; `B_high` is
the capillary side, driven by friction velocity through a drag law.

This is the spectrum `asv_wave_sim` uses. It is implemented specifically so a
like-for-like comparison against the Gazebo baseline is possible — same
spectrum, different field synthesis and force model.

### 3.4 Directional spreading

A cos-2s model, normalised so it integrates to 1:

    D(theta) = N(s) * cos^(2s)( (theta - theta_wind) / 2 )
    N(s) = Gamma(s+1) / (2 * sqrt(pi) * Gamma(s+1/2))

`s ≈ 5` is a reasonable wind-sea default; a long swell is more like 15–25.
Computed via log-gamma, because the closed form overflows for large s.

Spreading lives in the abstract base class with a working default, so no
spectrum can silently omit it — a new spectrum gets directional behaviour
whether or not its author thought about it, and overriding is deliberate. A
frequency-dependent `s` (real seas spread more widely away from the peak) is
not implemented but the signature already accepts omega, so adding it is not
a breaking change.

---

## 4. Wave field: FFT synthesis

### 4.1 Method

Tessendorf-style inverse-FFT synthesis on a periodic tile. Amplitudes are
drawn once at construction from the directional spectrum:

    var(k) = S(omega(k)) * (d omega/d k) * D(theta) / k * (dk)^2
    h0(k)  = (xi / sqrt(2)) * sqrt(var / 2),   xi ~ complex Gaussian

with `d omega/d k = 0.5 * sqrt(g/k)` the chain-rule factor from the
dispersion relation, and `dk = 2*pi/L`.

Time evolution is the standard two-term Hermitian form, which keeps the
surface real:

    h(k, t) = h0(k) * exp(i*omega*t) + conj(h0(-k)) * exp(-i*omega*t)
    eta(x, t) = IFFT[ h(k, t) ]

The imaginary residual of that inverse transform is tracked at every step as
a correctness check; it should be at machine precision.

### 4.2 What the grid can and cannot represent

This is the governing constraint on the whole simulator and is worth stating
plainly. A tile of side `L` divided into `N x N` cells (`cell = L/N`) can
only carry wavelengths between:

    shortest = 2 * cell      (Nyquist)
    longest  = L             (must fit in a periodic square)

So **tile size sets the longest wave, cell size sets the shortest**, and
point count goes as `(L/cell)^2`. Big seas need big tiles; small hulls need
small cells, because a hull cannot feel a wave shorter than two cells. See
§9.2 for the consequences.

### 4.3 Horizontal displacement (choppiness)

A pure height field is sinusoidal, and a real sea is not — real waves have
sharp crests and broad flat troughs. Tessendorf's horizontal displacement

    D(x, t) = sum over k of  -i * (k/|k|) * h(k, t) * exp(i*k*x)

moves water toward crests and away from troughs, which is exactly the
Gerstner trochoid.

**Important for force work:** `height(x, y)` deliberately does *not* include
this displacement. Once points move sideways the surface is no longer a
function of (x, y) — a sharp crest overhangs — so there is no unique height
to return. The force model therefore integrates against the undisplaced
surface, and choppiness changes what you *see*, not what the hull feels. A
test asserts the force model cannot detect it. This is a known approximation,
not an oversight; breaking-wave loads would need a different force model
anyway.

### 4.4 Band limiting

The FFT field accepts `k_min` / `k_max` and zeroes the **variance** (not the
amplitude) outside that band. Zeroing variance is what keeps a partition
exact, since variance is additive across independent components: two
complementary bands sum in quadrature back to the full-spectrum Hs. Verified
to three decimals (1.869 against 1.869).

This exists to support the cascade described in §9.2 and is not yet used by
default.

---

## 5. Breaking and whitecaps

### 5.1 Breaking criterion

A wave breaks when the downward acceleration at its crest reaches a fraction
of gravity — the classical limiting condition, and the reason a crest cannot
get sharper than about 120 degrees. This falls directly out of the spectral
amplitudes already in hand, since each component oscillates at its own omega:

    d^2(eta)/dt^2 = IFFT[ -omega^2 * h(k, t) ]

Foam is born where `-acc/g` exceeds a threshold, currently **0.35**.

### 5.2 Calibration

Threshold and persistence are calibrated **jointly** against Monahan &
O'Muircheartaigh (1980) whitecap coverage, because foam lingers and their W
is a steady-state measurement. With threshold 0.35 g and a 0.8 s decay:

    Hs      Monahan     model
    1.0 m     0.27%     0.00%
    2.5 m     1.29%     0.50%
    4.0 m     2.86%     3.55%
    5.5 m     4.93%     8.12%
    8.0 m     9.34%     7.05%

Within roughly a factor of two across Hs 1–8, and **monotonic in sea state**.

Caveat worth passing on: secondary sources disagree on Monahan's exponent
(3.41 vs 3.52), and the 1980 fit is known to overestimate coverage at high
latitudes against satellite microwave estimates, with later work finding the
exponent rises with sea surface temperature. Our agreement is about as good
as the fit's own agreement with reality, and should not be presented as a
precision result.

### 5.3 What this replaced

The previous foam model keyed off the Jacobian of the horizontal
displacement — i.e. it fired when the surface folded through itself. That is
a rendering artefact, not a breaking wave, and it had two consequences:
coverage was set by `choppiness`, an artistic parameter, and was almost blind
to the sea. Measured at choppiness 1.0, it gave 0.0% coverage for every sea
from Hs 1 to Hs 8. A 19 m/s gale produced no whitecaps.

The fold term is retained as a secondary mechanism, since past choppiness
~1.5 the surface genuinely does self-intersect and that is also white water,
but the brighter of the two wins rather than the two being summed.

---

## 6. Vessel-generated wake

As far as we know no existing ASV simulator attempts this, which is why it is
here.

### 6.1 Method

The hull is treated as a moving point disturbance. Each physics step it sheds
a **ring** — a damped circular wave centred on its current position — and the
wake is the superposition of the rings left along its track:

    eta_wake(x, t) = sum over rings i of
        a_i * exp(-d_i / L_decay) * exp(-age_i / tau) * sin(k*d_i - omega*age_i)

where `d_i` is radial distance from where that ring was shed. Note the phase
depends on **distance from the source**, not on a direction — these are
concentric circles, not plane waves.

This is Huygens' construction: a Kelvin wake *is* the superposition of
circular waves emitted continuously along the path. The consequence is that
the 19.47-degree half-angle is a **result**, not an input. Measured half-angle
comes out at 20.6–22.2 degrees with nothing in the code that knows what a
wedge is.

### 6.2 Three constraints, each of which was a bug first

- **Arc-length weighting.** `a_i` carries the arc length that ring
  represents, so the sum is a discrete line integral along the track rather
  than a count of samples. Without it, halving the physics timestep doubles
  the wake: amplitude 0.15 produced a 1.49 m wave. Now invariant —
  0.1237 / 0.1230 / 0.1226 m across dt = 0.1 / 0.05 / 0.025.
- **Causality.** Each ring's energy has only reached radius `c_g * age`,
  where `c_g = 0.5*sqrt(g/k)` is the deep-water group velocity. Without this
  the entire pattern exists the instant it is shed.
- **Source core radius.** A point source has a singular near field — every
  recent ring in phase at `d = 0`. That produced 0.55 m of wake from an
  amplitude of 0.12, and since `height()` feeds the force model, the hull
  floated on a hump of its own making. Distance is softened to
  `sqrt(d^2 + core^2)`.

### 6.3 Superposition with the ambient sea

`height()` is `base.height + wake`, a plain sum. That is exact to first
order, since the linearised free-surface equations are linear. It fails at
second order in steepness, where wave–wave interaction transfers energy
between components — the same approximation the FFT field already makes by
treating Fourier modes as independent.

The wake wraps a base field rather than replacing it, so flat water is the
isolated case and a spectral sea is the same wake superposed, with no code
difference.

### 6.4 Known deficiency

Amplitude decays as `exp(-d/L_decay)`, which is *damping*, not geometric
spreading. A real circular wave spreads its energy around a circumference of
`2*pi*d`, so amplitude must fall as **1/sqrt(d)** by energy conservation
before any dissipation. The far field is therefore about 6x too strong
relative to the near field, and the decay length is a fitted parameter rather
than physics. This is a known next fix.

The kernel also uses a **single wavenumber**. Replacing it with a spectrum of
wavenumbers and applying stationary phase would give the genuine Kelvin
pattern with both transverse and divergent systems. Everything around the
kernel is bookkeeping that would not change.

---

## 7. Hydrodynamic forces

### 7.1 Buoyancy by pressure integration

Not a block coefficient or a volume lookup. The wetted hull is integrated
directly:

1. Transform hull vertices to world, query the wave field for depth at each.
2. Clip every triangle against the instantaneous waterline — fully wet
   triangles pass through; partly wet ones are cut into sub-triangles at the
   exact crossing points (one wet vertex gives a triangle, two give a
   quadrilateral split into two).
3. For each wet sub-triangle, `F = -rho * g * d_c * A * n_hat`, with `d_c`
   the depth at its centroid.
4. Sum forces and moments about the centre of mass.

This is the same family as `asv_wave_sim`'s approach. Its advantage over a
displacement-volume method is that it gives the correct **restoring moment**
on a heeled hull without assuming anything about waterplane shape, and it
handles a wave surface that is not flat across the hull — which is the whole
point once the vessel is shorter than the wave.

Verified: summed per-face forces match the reported wrench to 2.4e-4 N.

### 7.2 Damping

Fossen's form, in the body frame:

    tau = -( D_linear + D_quadratic * |nu| ) * nu

Coefficients are **derived from the hull**, not tuned per vessel:

    D_quad = 0.5 * rho * Cd * A

with A the projected area per axis from the hull's bounding box. Cd is
**anisotropic** — (0.05, 1.0, 1.0) for surge, sway, heave — because a hull is
streamlined going ahead and bluff going sideways. Using Cd = 1.0 on all three
axes gave a 5.3 m speedboat a top speed of 0.6 m/s: surge drag came out at
851 N/(m/s)^2 against the ~8 implied by a Whaler 170 doing 16.5 m/s on 90 hp.

Rotational terms are integrated along the hull rather than lumped at one
radius:

    N = 2 * integral(0, S/2) of 0.5*rho*Cd*w*(omega*r)^2 * r dr
      = rho * Cd * w * omega^2 * S^4 / 64

with S the span the rotation sweeps and w the strip width. Yaw sweeps the
hull's side along its length, roll sweeps the bottom across the beam, pitch
sweeps the bottom along the length. A lumped-lever version had yaw damping
4.4x too high, which gave the boat a 29-second turning circle.

The quadratic term is solved **implicitly**: applied explicitly it is only
stable while `(D/m)*|nu|*dt < 1`, and an undamped hull can reach speeds past
that before the damping engages, at which point the damping itself destroys
the simulation. Solving `nu' = -(k/m) nu` exactly over the step gives
`tau = -k*nu / (1 + k*dt/m)`, which equals the explicit form as dt -> 0 and
is bounded at any speed.

### 7.3 Propulsion

Twin screws at known positions in the body frame — aft, outboard, below the
waterline. Thrust is a stated fraction of vessel weight (0.25) rather than a
propeller curve, deliberately: when the hull moves, we want to know exactly
what moved it. Thrust is vectored by a steering angle (35 degrees at full
helm), because an outboard steers by pointing rather than by thrust
imbalance; differential thrust alone gave a 29-second turning circle.

A ventilation check zeroes a screw's thrust when it lifts clear of the water,
so in a head sea the drive momentarily cuts as the stern comes up.

---

## 8. Hull geometry and mass distribution

### 8.1 Generated hulls

Hull forms are generated from principal dimensions rather than downloaded:
Wigley (the analytic benchmark, and the form Journée towed), a warped-V
planing hull (Boston Whaler 170 particulars), a round-bilge displacement hull
with deckhouse (Andromeda-class coastal survey vessel particulars), and a
1 m box for closed-form checks.

The reason is that every hull file obtained so far arrived without the one
number that matters — mass. A generated hull has the opposite problem solved:
the geometry is a formula, so there is no licensing, no unit ambiguity and
nothing to repair, and real vessels come with published displacements that
can be typed in beside it.

Imported USD meshes are also supported, and go through a merge/weld/orient
repair pass — the pressure integral needs a closed surface with outward
normals, and downloaded meshes rarely have either. Inward winding makes
buoyancy push *down*.

### 8.2 Mass distribution

This matters more than it sounds. A uniform-density hull puts the centre of
gravity at the volume centroid and gives a bounding-box inertia tensor, which
makes GM wrong, which makes roll wrong.

Mass is therefore distributed over three components:

- **plating** spread over the shell area,
- **machinery** as one lump low and a little aft of amidships,
- **topside** for the superstructure, which is lighter per unit volume than
  the hull.

The centre of gravity and the full inertia tensor are computed from that
distribution, then eigendecomposed into `diagonalInertia` plus
`principalAxes` for the physics engine. GM is measured about the actual
centre of gravity:

    GM = KB + BM - KG,   BM = I_waterplane / V

The superstructure is excluded from the buoyancy mesh — it sits above the
waterline and displaces nothing — but is included in the mass distribution.
Merging it into the hull would wreck the enclosed volume and the centroid
every later number is measured from.

A hull file with no mass authored produces a mass-to-draft table and stops,
rather than a guess.

---

## 9. Coupling to the physics engine

### 9.1 The transport delay, and why it needed compensating

The simulator is a discrete feedback loop:

    state(n) -> force(n) -> applied across [n, n+1] -> state(n+1)

The force is an exact function of pose, but it is evaluated at the *old* pose
and held constant while the body moves. Hydrostatic stiffness is high —
`k = rho*g*A_wp` is about 55 kN/m for a 771 kg boat, giving `omega ≈ 8.4
rad/s` — so at 60 Hz that is roughly 8 degrees of phase per step. A restoring
force should be exactly in antiphase with displacement; shift it and part of
it becomes proportional to *velocity*, which is damping of the wrong sign.

Both position and orientation are therefore extrapolated forward before the
forces are evaluated. Compensating position but not orientation leaves the
force *magnitude* corrected while its *direction* lags the body it acts on,
which does work on the rotational mode every cycle. Measured on flat water,
with position compensated and orientation not:

    rot_lead   tilt peak-to-peak
    0.0        4.238 deg, growing
    0.5        1.439 deg, decaying
    1.0        0.119 deg, decaying
    1.5        0.000 deg, settled

A floating hull on still water now settles to zero motion to six decimal
places. Before this, it held a permanent 23.8 mm / 4.3 degree limit cycle at
1.6 Hz — quadratic damping capping a linear pump. **Any rotational result
recorded before this fix is contaminated.**

### 9.2 Sea state ceiling

From §4.2, point count goes as `(tile/cell)^2`, and the two ends are set by
different things:

    sea state   peak Tp    peak wavelength   tile needed   cell needed (5 m hull)
    SS4          4.4 s          30 m            ~90 m           <= 1 m
    SS5          5.7 s          51 m           ~150 m           <= 1 m
    SS6          7.1 s          80 m           ~240 m           <= 1 m
    SS7          8.8 s         120 m           ~360 m           <= 1 m
    SS8         12.0 s         225 m           ~700 m           <= 1 m

Measured render cost: 65k points is ~15 ms per redraw, 262k points is ~93 ms
against a 50 ms budget at 20 Hz. So:

- **SS8 already works for a large vessel today.** A 1024 m tile at resolution
  256 gives 4 m cells, 99% of requested Hs, 8.4 ms per redraw. A 31.5 m ship
  responds mainly to waves near or above its own length, so 4 m cells are
  defensible.
- **Small hull in a big sea is the real constraint.** A 5 m boat in SS8 needs
  ~500k points.

There is also a **statistical** floor independent of resolution: Hs scatters
±12% between realisations when only 2.3 wavelengths fit the tile, falling to
±4% at 9 wavelengths. For RAO measurement, tile >= 5x peak wavelength.

**Proposed fix — a cascade.** Sum two or three FFT tiles, each carrying one
band of the spectrum (§4.4), all occupying the same water:

    1024 m / 128    8.0 m cells     holds   16 - 1024 m
     128 m / 128    1.0 m cells     holds    2 -  128 m
      16 m / 128    0.125 m cells   holds 0.25 -   16 m

49k points total — a fifth of one 512/512 tile — covering 0.25 m to 1024 m.
`height()` summing three fields for a 1027-vertex hull costs 0.225 ms, about
1.4% of real time: the physics is nearly free, and it was only ever the
render mesh that was expensive. Because the tile sizes share no common
factor, the visible repeat period also becomes very long.

Not yet implemented. The band limiting it depends on is, and is verified.

---

## 10. Known approximations and what is not modelled

Listed because this is the section a hydrodynamicist should read first.

**Not modelled at all:**

- **Added mass.** A hull accelerating in water drags water with it — roughly
  50–100% extra mass in heave. Its absence makes the modelled hull too
  responsive and raises its natural frequency. Offline, heave came out at
  1.33 Hz where added mass would put it lower.
- **Radiation damping.** A heaving hull radiates waves and loses energy. It
  is roughly *linear* in velocity and dominates at small amplitude — exactly
  the regime where the quadratic drag term is asleep. This is the single
  largest gap for matching measured RAOs.
- **Diffraction and the Froude–Krylov split.** The incident pressure field is
  integrated over the hull; the hull's own scattering of the incident wave is
  not represented.
- **Finite depth.** Deep-water dispersion only.
- **Nonlinear wave–wave interaction.** Fourier modes are independent, and
  superposition is linear. Second order in steepness.
- **Planing lift.** Everything is displacement-mode. A planing hull rises and
  its resistance collapses; that is not represented, so true planing speeds
  are unreachable. The surge drag coefficient (0.05) is explicitly labelled
  in the code as a stand-in for this, not a measured value.
- **Spray, air entrainment, green water.**
- **Wind loads on the hull and superstructure.**

**Approximations that are deliberate:**

- Choppiness is visual only (§4.3).
- Wake amplitude lacks geometric spreading (§6.4) and uses one wavenumber.
- Rotational drag uses bounding-box projected areas with shape-factor Cd
  rather than a hull-form-specific value.
- Thrust is a fraction of weight, not a propeller curve.

---

## 11. Validation status

88 automated checks run outside Isaac Sim in about a second, covering
spectra (9), FFT synthesis (6), choppiness (3), foam (3), wake (10), pressure
integration (8), damping (8), thruster (3), mesh repair (5), USD transforms
(11) and mesh handling (6).

Specific numbers checked against independent references or closed forms:

| Quantity | Reference | Measured |
|---|---|---|
| Hs round-trip, spectrum -> field -> surface | requested Hs | within a few % |
| JONSWAP at gamma = 1 | reduces to PM | exact |
| Wigley block coefficient | 4/9 analytic | 0.4444 |
| 1 m box, 200 kg, fresh water | 0.200 m draft | 0.200 m |
| Whaler 170 draft | 0.30 m published | 0.315 m |
| Survey hull draught | 3.1 m at 380 t published | 3.10 m |
| Wake half-angle | Kelvin 19.47 deg | 20.6–22.2 deg |
| Whitecap coverage at Hs 4 | Monahan 2.86% | 3.55% |
| Per-face forces vs reported wrench | identity | 2.4e-4 N |
| Flat-water settling | zero motion | 0.000000 m p-p |
| Heave RAO, Whaler at Tp 4 s | ~1 expected (25 m wave, 5.2 m boat) | 0.92 |

**Not yet done:** the Journée (1992) Wigley RAO comparison, which is the real
benchmark. The hull form and the measurement exist; what is missing is
confirmation of the exact model particulars (L/B, B/T) used in those
experiments. If Eric has access to the Delft report or the Weymouth (2005)
*Journal of Ship Research* paper, that would unblock it.

---

## 12. Questions a hydrodynamicist could help with

1. **Added mass and radiation damping.** What is the cheapest defensible way
   to add these for a hull of this size — a strip-theory estimate, a
   frequency-independent constant, or precomputed BEM coefficients? There are
   `bluboat_bem*.stl` files in the lab's archive suggesting someone has run a
   panel code before.
2. **High sea state validation.** Does the cascade approach (§9.2) have a
   precedent in the hydrodynamics literature, or is it purely a graphics
   technique? Is there a better way to span four octaves of wavenumber.
3. **Towing-tank validation protocol.** What would a credible comparison
   look like, and which quantities are most diagnostic — heave/pitch RAOs,
   added resistance in waves, or something else?
4. **Breaking criterion.** We use downward crest acceleration at 0.35 g
   calibrated to whitecap coverage. Is there a criterion closer to the
   hydrodynamics literature that is still cheap to evaluate spectrally?
5. **The choppiness/force decoupling** (§4.3). Is integrating forces against
   an undisplaced surface while displaying a displaced one defensible, or
   does the horizontal particle velocity matter enough to need the full
   Gerstner surface in the force path?
