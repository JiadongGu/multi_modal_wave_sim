

from __future__ import annotations

try:
    import omni.ext

    _ExtBase = omni.ext.IExt
except ImportError:          # pragma: no cover - running outside Kit
    # Kit identifies an extension by finding omni.ext.IExt subclasses, so inside
    # Isaac Sim the real base class is essential. Outside it, a stand-in lets
    # this module be imported from a plain python shell, which is what makes
    # tools/selftest.py possible without launching the simulator.
    class _ExtBase:  # type: ignore[no-redef]
        """Stand-in for omni.ext.IExt when Kit is not running."""


from .registry import FIELDS, FORCES, SPECTRA

# Level for the one-time lifecycle messages (startup, shutdown, component swaps).
#
# "warn" on purpose. carb's default console level hides info, and so does the
# Console window's own filter, so an extension that announces itself at info
# level looks identical to one that failed to load. During bring-up, being able
# to see that it started is worth more than log-level purity. Drop this to
# "info" once the extension has a UI that makes its state visible.
BANNER_LEVEL = "warn"

# Per-step vessel diagnostics. Set False once the force path is trusted.
DIAGNOSTIC = False

# Log EVERY step for this many steps after Play, regardless of what the hull
# is doing. Conditional logging kept missing the window where things go wrong:
# whatever happens between two sampled lines is exactly what you need to see.
TRACE_STEPS = 400


# How far ahead of the measured state to evaluate the force, in timesteps.
#
# The force we write is not applied to the state we read it from. Reading at
# step n, PhysX applies that force during step n+1 - the trace shows it
# plainly, with every dv matching the PREVIOUS line's exp. A restoring force
# that arrives a step late does not just blur the answer, it ADDS energy:
# released at rest on flat water the hull's oscillation grew about 25% per
# cycle until quadratic drag capped it at a limit cycle.
#
# Evaluating at x + v * lead * dt cancels it. 1.0 step for the transport
# delay, 0.5 more to centre the force on the step it acts over. For a
# hydrostatic spring of stiffness k this is algebraically a linear damper of
# k * lead * dt N per m/s, which for this hull is about 245 - so if the hull
# now LOSES energy too fast, this number is why.
FORCE_LEAD_STEPS = 1.5


def _lead(state, dt):
    """state, with position extrapolated FORCE_LEAD_STEPS into the future.

    linear_velocity is body-frame (see convert.body_state_from_usd), and the
    position is world-frame, so the velocity has to be rotated back out before
    it can be added to it. R maps body -> world; R.T is the world -> body used
    when the state was built.
    """
    from dataclasses import replace

    import numpy as _np

    from .forces import quat_to_matrix

    if dt <= 0.0 or FORCE_LEAD_STEPS == 0.0:
        return state
    R = quat_to_matrix(_np.asarray(state.orientation, dtype=float))
    v_world = R @ _np.asarray(state.linear_velocity, dtype=float)
    return replace(
        state,
        position=_np.asarray(state.position, dtype=float)
        + v_world * (FORCE_LEAD_STEPS * dt),
    )


# The live extension object, for the Script Editor and for tools/.
#
# Kit constructs the extension itself and hands the instance to nobody, so
# without this there is no supported way to reach the running object from a
# script. Set on startup, cleared on shutdown, so a stale handle after a
# disable/enable cycle is impossible.
_INSTANCE = None


def get():
    """The running extension, or None if it is not enabled.

        import mmws.wave_sim.extension as E
        E.get().select(field="flat")
    """
    return _INSTANCE


class MultiModalWaveSimExtension(_ExtBase):
    """Lifecycle owner. Holds the current selection and the event handles."""

    def on_startup(self, ext_id: str) -> None:
        global _INSTANCE
        _INSTANCE = self
        self._ext_id = ext_id
        self._step_count = 0
        self._sim_time = 0.0
        self._last_draw = 0.0

        # Handles for the event subscriptions. Kept as attributes because
        # dropping the reference silently unsubscribes - a classic Kit bug.
        self._timeline_sub = None
        self._physics_sub = None

        # The current mode. This triple is what "multi-modal" means: each can
        # be swapped by name without the others knowing.
        self.spectrum = None
        self.field = None
        self.force_model = None
        # How the current field was built, so a spectrum swap can rebuild it.
        self._field_name = None
        self._field_kwargs = {}
        # Per-step CSV capture, off unless record() turns it on.
        self._csv = None
        self._csv_writer = None
        self._csv_rows = 0

        # Bound on Play, not here: the hull comes off the stage, so the force
        # model cannot be built until a prim actually exists to read it from.
        self.vessel = None
        self.vessel_path = "/World/Vessel"

        self._log(f"starting up (ext_id={ext_id})", BANNER_LEVEL)
        self._log(f"  spectra:      {', '.join(SPECTRA.names()) or '(none)'}", BANNER_LEVEL)
        self._log(f"  wave fields:  {', '.join(FIELDS.names()) or '(none)'}", BANNER_LEVEL)
        self._log(f"  force models: {', '.join(FORCES.names()) or '(none)'}", BANNER_LEVEL)

        # Safe defaults: a spectrum that works, a field that is flat and a
        # force model that does nothing. Nothing here touches the stage, so
        # enabling the extension cannot disturb a scene.
        try:
            self.select(
                spectrum="jonswap",
                spectrum_kwargs={"hs": 2.0, "tp": 7.0, "spread_s": 5.0},
                field="fft",
                field_kwargs={"tile_size": 256.0, "resolution": 64},
                force_model="zero",
            )
        except Exception as exc:                      # noqa: BLE001
            self._log(f"  could not build defaults: {exc}", "error")

        self._subscribe()
        self._log("ready", BANNER_LEVEL)

    def on_shutdown(self) -> None:
        global _INSTANCE
        if _INSTANCE is self:
            _INSTANCE = None
        # Unsubscribe before dropping anything else: a callback firing into a
        # half-torn-down extension is how Isaac Sim crashes on reload, and
        # Python extensions reload often during development.
        self._unsubscribe()
        if getattr(self, "_csv", None) is not None:
            self.record(None)
        self.spectrum = None
        self.field = None
        self.force_model = None

        # Bound on Play, not here: the hull comes off the stage, so the force
        # model cannot be built until a prim actually exists to read it from.
        self.vessel = None
        self.vessel_path = "/World/Vessel"
        self._log("shut down", BANNER_LEVEL)

    # ------------------------------------------------------------------ the swap

    def select(
        self,
        spectrum=None,
        spectrum_kwargs=None,
        field=None,
        field_kwargs=None,
        force_model=None,
    ) -> None:
        """Swap any of the three components by registry name.

        The force model is independent of the other two. The field is NOT
        independent of the spectrum: a spectral field samples the spectrum
        once, at construction, and never consults it again - FftWaveField
        bakes h0 in _build() - so swapping the spectrum alone would leave the
        water exactly as it was. That failed silently and looked like the
        swap had worked, with only the wave statistics to give it away. So a
        spectrum change rebuilds the current field on the same settings.

        This is the method a UI dropdown, a ROS 2 service or a test harness
        should call.
        """
        rebuild = False
        if spectrum is not None:
            self.spectrum = SPECTRA.create(spectrum, **(spectrum_kwargs or {}))
            self._log(f"spectrum  -> {self.spectrum!r}", BANNER_LEVEL)
            rebuild = field is None and self._field_name is not None

        if field is not None or rebuild:
            name = field if field is not None else self._field_name
            kwargs = field_kwargs if field is not None else self._field_kwargs
            self._field_name, self._field_kwargs = name, dict(kwargs or {})
            self.field = FIELDS.create(
                name, spectrum=self.spectrum, **self._field_kwargs
            )
            # Keep the clock. A rebuilt field starts at t = 0, and a surface
            # that jumps backwards under a floating hull is a step change in
            # depth that the force model will answer with a step change in
            # force.
            self.field.advance(self._sim_time)
            self._log(
                f"field     -> {self.field!r}"
                + ("   (rebuilt for the new spectrum)" if rebuild else ""),
                BANNER_LEVEL,
            )
        if force_model is not None:
            self.force_model = FORCES.create(force_model)
            self._log(f"forces    -> {self.force_model!r}", BANNER_LEVEL)

    def record(self, path=None):
        """Write one row per physics step to CSV, for frequency-domain work.

            E.get().record("/tmp/heave.csv")    # start
            E.get().record(None)                # stop and close

        The heartbeat is once a second. Nyquist on a 1 Hz sample is 0.5 Hz, so
        it cannot see anything with a period under 2 s - which is most of a
        wave spectrum. This writes every step, 60 Hz, giving headroom to 30 Hz.

        Columns are the hull and the water under it at the same instant, which
        is what a transfer function needs: heave RAO is the ratio of the z
        amplitude to the water amplitude at each frequency, so both have to be
        sampled together or the phase is meaningless.

        Stopping the timeline closes the file. So does calling record(None),
        and so does shutdown - an unclosed file is a truncated file.
        """
        import csv

        if self._csv is not None:
            self._csv.flush()
            self._csv.close()
            self._log(f"recording closed: {self._csv.name} ({self._csv_rows} rows)",
                      BANNER_LEVEL)
            self._csv = self._csv_writer = None
            self._csv_rows = 0
        if path is None:
            return

        self._csv = open(path, "w", newline="")
        self._csv_writer = csv.writer(self._csv)
        self._csv_writer.writerow([
            "step", "t", "x", "y", "z", "water", "depth", "tilt_deg",
            "vx", "vy", "vz", "fx", "fy", "fz",
        ])
        self._csv_rows = 0
        self._log(f"recording to {path} (one row per step)", BANNER_LEVEL)

    def _record_row(self, state, wrench) -> None:
        import numpy as _np

        try:
            x, y, z = (float(v) for v in state.position)
            wz = float(_np.asarray(self.field.height(
                _np.array([x]), _np.array([y])))[0])
            q = _np.asarray(state.orientation, dtype=float)
            tilt = _np.degrees(_np.arccos(_np.clip(
                1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0)))
            v = _np.asarray(state.linear_velocity, dtype=float)
            f = _np.asarray(wrench.force, dtype=float)
            self._csv_writer.writerow([
                self._step_count, f"{self._sim_time:.6f}",
                f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{wz:.6f}",
                f"{wz - z:.6f}", f"{tilt:.4f}",
                f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}",
                f"{f[0]:.4f}", f"{f[1]:.4f}", f"{f[2]:.4f}",
            ])
            self._csv_rows += 1
        except Exception as exc:                      # noqa: BLE001
            self._log(f"recording failed, stopping ({exc})", "error")
            self.record(None)

    # ------------------------------------------------------------------ vessel

    def bind_vessel(self, body_path: str = None, hull_path: str = None) -> bool:
        """Attach to a rigid body on the stage and build its force model.

        The hull geometry comes from the stage, so the force model cannot exist
        before this succeeds. That is why selecting "buoyancy_pressure" by name
        in `select()` is not enough on its own - it needs a mesh.
        """
        body_path = body_path or self.vessel_path
        try:
            import omni.usd

            from .integration.usd_bridge import VesselBridge

            bridge = VesselBridge(omni.usd.get_context().get_stage(), body_path, hull_path)
            bridge.bind()
            self.vessel = bridge
            # Buoyancy alone is a spring with nothing to dissipate energy, and
            # the water surface is moving, so an undamped hull bouncing on it
            # gains energy every cycle and leaves. Damping is not optional.
            extents = bridge.vertices.max(axis=0) - bridge.vertices.min(axis=0)
            self.force_model = FORCES.create(
                "composite",
                models=[
                    FORCES.create(
                        "buoyancy_pressure",
                        vertices=bridge.vertices,
                        triangles=bridge.triangles,
                    ),
                    FORCES.create("damping", extents=extents),
                ],
            )
            lo = bridge.vertices.min(axis=0)
            hi = bridge.vertices.max(axis=0)
            self._log(
                f"vessel    -> {body_path}  ({len(bridge.vertices)} verts, "
                f"{len(bridge.triangles)} tris, mass {bridge.mass:.1f} kg, "
                f"force mode {getattr(bridge, 'force_mode', '?')})",
                BANNER_LEVEL,
            )
            self._log(
                f"  hull  x {lo[0]:.2f}..{hi[0]:.2f}  y {lo[1]:.2f}..{hi[1]:.2f}  "
                f"z {lo[2]:.2f}..{hi[2]:.2f} m   volume {bridge.volume:.4f} m^3   "
                f"floats at {bridge.mass / 1000.0:.4f} m^3 displaced",
                BANNER_LEVEL,
            )
            return True
        except Exception as exc:                      # noqa: BLE001
            self._log(f"no vessel at {body_path} ({exc})", "error")
            self.vessel = None
            return False

    # ------------------------------------------------------------ subscriptions

    def _subscribe(self) -> None:
        try:
            import omni.timeline

            self._timeline_sub = (
                omni.timeline.get_timeline_interface()
                .get_timeline_event_stream()
                .create_subscription_to_pop(self._on_timeline_event)
            )
        except Exception as exc:                      # noqa: BLE001
            self._log(f"no timeline events ({exc})", "error")

        try:
            from omni.physx import get_physx_interface

            self._physics_sub = get_physx_interface().subscribe_physics_step_events(
                self._on_physics_step
            )
        except Exception as exc:                      # noqa: BLE001
            self._log(f"no physics step events ({exc})", "error")

    def _unsubscribe(self) -> None:
        if self._physics_sub is not None:
            try:
                self._physics_sub.unsubscribe()
            except Exception:                         # noqa: BLE001
                pass
            self._physics_sub = None
        # Timeline subscriptions are released by dropping the handle.
        self._timeline_sub = None

    # ---------------------------------------------------------------- callbacks

    def _on_timeline_event(self, event) -> None:
        """Play / Pause / Stop. Reset accumulated state on Stop."""
        try:
            import omni.timeline

            if event.type == int(omni.timeline.TimelineEventType.PLAY):
                if self.vessel is None:
                    self.bind_vessel()
            if event.type == int(omni.timeline.TimelineEventType.STOP):
                self._step_count = 0
                self._sim_time = 0.0
                self._last_draw = 0.0
                self._wet_steps = 0
                self._prev_vz = None
                self.vessel = None
                # Close any recording. A file left open is a file with its
                # last buffer unwritten, and the truncation lands at the end
                # of the run - exactly where the interesting part is.
                if self._csv is not None:
                    self.record(None)
                self._log("timeline stopped, counters reset", BANNER_LEVEL)
        except Exception:                             # noqa: BLE001
            pass

    def _on_physics_step(self, dt: float) -> None:
        """Called every PhysX substep. The hot path.

        Everything expensive happens here, so it stays deliberately thin: keep
        time, advance the field, and later gather forces and write them to the
        body. Per-step allocation and per-step logging both belong nowhere near
        this function.

        The full sequence, once the pieces exist:

            1. read body state from USD    -> BodyState
            2. advance the wave field      -> field.advance(t)
            3. ask the force model         -> force_model.wrench(body, field, t)
            4. sum thrusters and the rest  -> tau
            5. write to PhysxForceAPI      (body frame)
        """
        self._step_count += 1
        self._sim_time += dt

        if self.field is not None:
            self.field.advance(self._sim_time)
            if(self._sim_time - self._last_draw >= 1.0 / 20.0): #throttle limits how often the mesh is redrawn. 20Hz
                self._last_draw = self._sim_time
                import omni.usd
                self.field.write_mesh(omni.usd.get_context().get_stage())

        if self.vessel is not None and self.force_model is not None and self.field is not None:
            try:
                state = self.vessel.read_state()
                # Kept for the heartbeat below, which has to say where the
                # hull is even when the per-step trace is off. Holding the
                # reference costs nothing; the heartbeat does the arithmetic
                # once a second.
                self._last_state = state
                wrench = self.force_model.wrench(
                    _lead(state, dt), self.field,
                    self._sim_time + FORCE_LEAD_STEPS * dt,
                )
                self.vessel.apply(wrench)

                if self._csv_writer is not None:
                    self._record_row(state, wrench)

                # Diagnostic, roughly twice a second. Three questions at once:
                # is the pose updating, is the water height sane, is the force
                # the right order of magnitude for this hull and mass.
                # Log every step where the hull is actually touching water,
                # capped so a stuck vessel cannot flood the console. Sampling
                # on a step interval misses the impact entirely when the body
                # crosses the surface in a few milliseconds.
                import numpy as _np

                if DIAGNOSTIC and (self._step_count <= TRACE_STEPS
                                   or self._step_count % 60 == 0):
                    px, py, pz = (float(v) for v in state.position)
                    wz = float(_np.asarray(self.field.height(
                        _np.array([px]), _np.array([py])))[0])
                    f, m = wrench.force, wrench.moment
                    vz = float(state.linear_velocity[2])
                    # What PhysX actually did last step, against what this
                    # force and this mass say it should have done. A mismatch
                    # means the body's mass is not what we think, or something
                    # else is pushing it.
                    prev = getattr(self, "_prev_vz", None)
                    prev_f = getattr(self, "_prev_fz", 0.0)
                    if prev is None or dt <= 0.0:
                        implied = state.mass
                    else:
                        accel = (vz - prev) / dt + 9.80665
                        implied = prev_f / accel if abs(accel) > 1e-6 else float("inf")
                    self._prev_vz, self._prev_fz = vz, float(f[2])
                    # How far the hull has tipped from upright. R[2][2] of a
                    # unit quaternion (w, x, y, z) is 1 - 2(x^2 + y^2), which
                    # is the world-z component of the body's own z axis - and
                    # it reads the same whichever row/column convention the
                    # rotation matrix uses, so there is nothing to get wrong.
                    #
                    # Worth its own column: every force in this log is in the
                    # BODY frame, so a big Fx on flat water does not mean a
                    # sideways force, it means the hull is lying over. Without
                    # this number that reads as a bug in the force model.
                    q = _np.asarray(state.orientation, dtype=float)
                    tilt = _np.degrees(_np.arccos(
                        _np.clip(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0)))
                    self._log(
                        f"  z={pz:+8.3f} vz={vz:+7.2f}  water={wz:+6.3f} "
                        f"depth={wz - pz:+7.3f}  F=({f[0]:+7.0f},{f[1]:+7.0f},"
                        f"{f[2]:+8.0f})  dv={(vz - prev) if prev is not None else 0.0:+6.3f}"
                        f" exp={(float(f[2]) / max(state.mass, 1e-9) - 9.80665) * dt:+6.3f}"
                        f" m_imp={implied:+.0f} tilt={tilt:5.1f}deg",
                        "warn",
                    )
            except Exception as exc:                  # noqa: BLE001
                # Drop the vessel rather than throw sixty times a second.
                self._log(f"force step failed, detaching ({exc})", "error")
                self.vessel = None

        # One line a second, and it says where the hull is.
        #
        # This used to print only a step counter, and to drop to "info" - which
        # carb hides - after step 300. With DIAGNOSTIC off that left the sim
        # completely silent after five seconds, which is exactly when a hull
        # that is going to misbehave has got round to it. The heartbeat is the
        # floor of what you can see, so it carries the state: where the hull
        # is, where the water is under it, and whether it is still upright.
        if self._step_count % 60 == 0:
            line = f"step {self._step_count}  t={self._sim_time:.2f}s  dt={dt * 1000:.2f}ms"
            state = getattr(self, "_last_state", None)
            if state is not None and self.field is not None:
                try:
                    import numpy as _np

                    px, py, pz = (float(v) for v in state.position)
                    wz = float(_np.asarray(self.field.height(
                        _np.array([px]), _np.array([py])))[0])
                    q = _np.asarray(state.orientation, dtype=float)
                    tilt = _np.degrees(_np.arccos(_np.clip(
                        1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0)))
                    line += (f"   hull z={pz:+7.3f} water={wz:+7.3f} "
                             f"depth={wz - pz:+7.3f} tilt={tilt:5.1f}deg")
                except Exception:                     # noqa: BLE001
                    line += "   (state unavailable)"
            self._log(line, BANNER_LEVEL)

    # -------------------------------------------------------------------- utils

    def _log(self, message: str, level: str = "info") -> None:
        """carb's logger when inside Kit, print() when not.

        Lets the same code be exercised from a plain python shell, which is how
        most of the development will actually happen.
        """
        text = f"[mmws.wave_sim] {message}"
        try:
            import carb

            {"info": carb.log_info, "warn": carb.log_warn, "error": carb.log_error}[
                level
            ](text)
        except Exception:                             # noqa: BLE001
            print(text)
