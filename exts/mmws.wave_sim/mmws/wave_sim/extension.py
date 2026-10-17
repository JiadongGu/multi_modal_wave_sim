

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


def _quat_mul(a, b):
    """Hamilton product, (w, x, y, z) - the order USD's Gf.Quatd uses."""
    import numpy as _np

    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return _np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _lead(state, dt, steps=None, rot_steps=0.0):
    """state, with position extrapolated `steps` physics steps into the future.

    linear_velocity is body-frame (see convert.body_state_from_usd), and the
    position is world-frame, so the velocity has to be rotated back out before
    it can be added to it. R maps body -> world; R.T is the world -> body used
    when the state was built.
    """
    from dataclasses import replace

    import numpy as _np

    from .forces import quat_to_matrix

    steps = FORCE_LEAD_STEPS if steps is None else float(steps)
    rot_steps = float(rot_steps)
    if dt <= 0.0 or (steps == 0.0 and rot_steps == 0.0):
        return state
    R = quat_to_matrix(_np.asarray(state.orientation, dtype=float))
    v_world = R @ _np.asarray(state.linear_velocity, dtype=float)
    out = replace(
        state,
        position=_np.asarray(state.position, dtype=float)
        + v_world * (steps * dt),
    )
    if rot_steps == 0.0:
        return out
    # Orientation gets the same treatment as position. qdot = 0.5 q (x) w,
    # with w the BODY-frame angular velocity, which is the frame BodyState
    # stores it in. Without this, heave is delay-compensated and rotation is
    # not - measured: sweeping the position lead from 0 to 3 moves heave 6x
    # and tilt only 14%, because the surviving limit cycle is rotational.
    q = _np.asarray(state.orientation, dtype=float)
    w = _np.asarray(state.angular_velocity, dtype=float)
    q = q + 0.5 * _quat_mul(q, _np.array([0.0, w[0], w[1], w[2]])) * (rot_steps * dt)
    n = _np.linalg.norm(q)
    return replace(out, orientation=q / n if n > 0.0 else state.orientation)


# A distinct "argument not given" marker, because None is a real value here:
# capture(None) stops and writes.
_DEFAULT_PATH = object()


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
        self._panel = None
        self._in_select = False
        # Grid overlay, off by default: it is a diagnostic, not scenery.
        self.show_grid = False
        self._grid_stride = 8
        self._grid_drawn = False
        # Nothing is written to any stage until something asks for it.
        #
        # Kit enables an extension globally, so without this the first Play in
        # ANY scene gets a 256 m sheet of water dropped into it. The tools
        # call activate(), which also remembers WHICH stage they set up - open
        # a different file and the extension stands down by itself rather than
        # redecorating someone else's work.
        self.active = False
        self._stage_id = None
        # Drive keys, off until keyboard_drive() turns it on.
        #
        # NOT WASD: Kit binds those to the viewport camera fly-through,
        # and _on_key deliberately does not swallow events, so WASD drove
        # the boat and flew the camera at the same time. IJKL is the same
        # shape one row right, and Kit leaves it alone.
        self._drive_keys = ("I", "J", "K", "L")   # ahead, port, astern, starboard

        # How far ahead the body state is extrapolated before the forces are
        # evaluated, in physics steps. Settable at runtime so it can be swept
        # against a recording without restarting Kit:
        #
        #     E.get().force_lead = 1.0
        #
        # It exists to cancel the one-step transport delay between reading the
        # pose and PhysX applying the force. Leading by MORE than the delay
        # turns it into velocity feedback with the wrong sign - negative
        # damping - and the hull settles into a limit cycle instead of
        # settling down: measured on flat water at 1.594 Hz, 23.8 mm peak to
        # peak, constant to four significant figures over 40 s.
        self.force_lead = FORCE_LEAD_STEPS
        # Same idea for orientation. 0.0 keeps the old behaviour;
        # the measurement that justifies a non-zero default is the
        # tilt column of a sweep.
        self.rot_lead = 0.0
        self._sweep = None
        self._autorun_sub = None
        self.thruster = None
        self._keys = set()
        self._kb_sub = None
        self._kb_input = None
        # Per-face force arrows, off by default.
        self.show_forces = False
        self._forces_drawn = False
        # Full run capture, off until capture() turns it on.
        self._cap = None
        # Chase camera, off by default.
        self.follow = False
        self._cam_eye = None
        self._cam_at = None
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
        self.show_ui()
        self._log("ready - standing by. A tool or the panel checkbox activates it", BANNER_LEVEL)

        import os as _os
        _auto = _os.environ.get("MMWS_AUTORUN")
        if _auto:
            self._autorun(_auto)

    def _autorun(self, path: str) -> None:
        """Run a Python file once the app has settled, if $MMWS_AUTORUN is set.

        The Script Editor is the only way into a running Kit, and it needs a
        human at a GUI. That makes every experiment hand-driven, which is how
        the first two sweeps got confounded - and it makes a headless run over
        ssh impossible. This closes that hole: the launcher sets the variable,
        the extension execs the file a couple of seconds after startup, and
        the whole experiment is reproducible from a shell.

        Deliberately an environment variable rather than a setting in the
        repo: it only runs what the person starting Kit chose to point it at,
        and it leaves no trace in the stage or the config.
        """
        import omni.kit.app

        frames = {"n": 0}
        app = omni.kit.app.get_app()

        def _tick(_event):
            frames["n"] += 1
            if frames["n"] < 120:            # ~2 s, so the stage and UI exist
                return
            self._autorun_sub = None
            self._log(f"autorun: {path}", BANNER_LEVEL)
            try:
                with open(path) as fh:
                    code = fh.read()
                exec(compile(code, path, "exec"),
                     {"__name__": "__mmws_autorun__", "__file__": path})
            except Exception as exc:                  # noqa: BLE001
                import traceback
                self._log(f"autorun failed: {exc}\n{traceback.format_exc()}",
                          "error")

        self._autorun_sub = app.get_update_event_stream().create_subscription_to_pop(
            _tick, name="mmws.wave_sim.autorun")

    def on_shutdown(self) -> None:
        global _INSTANCE
        if _INSTANCE is self:
            _INSTANCE = None
        # Unsubscribe before dropping anything else: a callback firing into a
        # half-torn-down extension is how Isaac Sim crashes on reload, and
        # Python extensions reload often during development.
        self._unsubscribe()
        if getattr(self, "_kb_sub", None) is not None:
            try:
                self.keyboard_drive(False)
            except Exception:                         # noqa: BLE001
                pass
        if getattr(self, "_panel", None) is not None:
            self._panel.destroy()
            self._panel = None
        if getattr(self, "_csv", None) is not None:
            self.record(None)
        if getattr(self, "_cap", None) is not None:
            self._write_capture()
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
            kwargs = dict(
                (field_kwargs if field is not None else self._field_kwargs) or {})
            # The tile is the world the hull floats in, and load_hull.py puts
            # the vessel at tile_size/2. Swapping the field must not silently
            # resize it: select(field="flat") with no kwargs used to drop the
            # 256 m fft tile for WaveField's 128 m default, which left the
            # boat sitting exactly on the corner of the water with the mesh
            # ending under its bow. Carried over per-key, not wholesale -
            # choppiness and seed mean nothing to flat water and would raise.
            if self.field is not None:
                kwargs.setdefault("tile_size", self.field.tile_size)
                kwargs.setdefault("resolution", self.field.resolution)
            self._field_name, self._field_kwargs = name, kwargs
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

        # select() is also called from the Script Editor, and a panel showing
        # numbers that are no longer true is worse than no panel. Guarded
        # against re-entry: the panel calls select(), which would otherwise
        # call back into the panel mid-apply.
        panel = getattr(self, "_panel", None)
        if panel is not None and not self._in_select:
            self._in_select = True
            try:
                panel.refresh()
            except Exception:                         # noqa: BLE001
                pass
            finally:
                self._in_select = False

    # ------------------------------------------------------------------- drive

    def keyboard_drive(self, enable: bool = True, keys: str = None) -> None:
        """I/K ahead and astern, J/L turn to port and starboard.

        Kit delivers key events globally, so this stays unsubscribed unless
        asked for - otherwise typing "like" in the Script Editor would drive
        the boat.

        `keys` is four letters in the order ahead, port, astern, starboard.
        The default is IJKL rather than WASD because Kit uses WASD for the
        viewport camera and _on_key does not swallow events: with WASD you
        drove the hull and flew the camera together, which is unusable.
        Pass "WASD" back if you ever want the old behaviour.
        """
        import carb.input
        import omni.appwindow

        if keys is not None:
            k = str(keys).upper()
            if len(k) != 4 or len(set(k)) != 4:
                raise ValueError(
                    f"keys must be four distinct letters, got {keys!r}")
            self._drive_keys = tuple(k)

        if self._kb_sub is not None:
            self._kb_input.unsubscribe_to_keyboard_events(
                omni.appwindow.get_default_app_window().get_keyboard(),
                self._kb_sub)
            self._kb_sub = self._kb_input = None
            self._keys.clear()
            self._drive_from_keys()
        if not enable:
            self._log("drive off", BANNER_LEVEL)
            return

        self._kb_input = carb.input.acquire_input_interface()
        self._kb_sub = self._kb_input.subscribe_to_keyboard_events(
            omni.appwindow.get_default_app_window().get_keyboard(), self._on_key)
        ahead, port, astern, starboard = self._drive_keys
        self._log(f"drive on - {ahead}/{astern} ahead and astern, "
                  f"{port}/{starboard} turn", BANNER_LEVEL)

    def _on_key(self, event, *args) -> bool:
        import carb.input

        name = getattr(event.input, "name", str(event.input)).upper()
        if name in self._drive_keys:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self._keys.add(name)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._keys.discard(name)
            self._drive_from_keys()
        # Never swallow the event: Kit's viewport camera and its shortcuts
        # share the keyboard, and stealing keys would make the scene
        # impossible to fly around. This is why the drive keys have to be
        # ones Kit does not already use, rather than being grabbed.
        return True

    def _drive_from_keys(self) -> None:
        if self.thruster is None:
            return
        ahead, port, astern, starboard = self._drive_keys
        surge = (ahead in self._keys) - (astern in self._keys)
        turn = (port in self._keys) - (starboard in self._keys)
        self.thruster.set_drive(surge, turn)

    # ----------------------------------------------------------------- scoping

    def activate(self, enable: bool = True) -> None:
        """Start (or stop) touching the current stage.

        The tools call this, so the normal workflow never has to. Call it by
        hand to adopt a stage you built yourself, or activate(False) to leave
        a scene alone without disabling the extension.
        """
        self.active = bool(enable)
        self._stage_id = self._current_stage_id() if enable else None
        if not enable:
            self.vessel = None
        self._log("active on this stage" if enable else "standing down",
                  BANNER_LEVEL)

    @staticmethod
    def _current_stage_id():
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            return stage.GetRootLayer().identifier if stage else None
        except Exception:                             # noqa: BLE001
            return None

    def _stage_is_ours(self) -> bool:
        """False once a different file is open, and say so exactly once.

        Comparing the root layer identifier rather than the stage object: Kit
        can hand back a different wrapper for the same stage, and a new file
        always has a new identifier. Checked every step because it has to be
        right on the first one after a file opens, and it is a string compare.
        """
        if self._stage_id is None:
            return False
        if self._current_stage_id() == self._stage_id:
            return True
        self.active = False
        self._stage_id = None
        self.vessel = None
        self._log("a different stage is open - standing down. activate() to "
                  "adopt this one.", BANNER_LEVEL)
        return False

    # ------------------------------------------------------------- diagnostics

    def _face_model(self):
        """The buoyancy model inside whatever composite is bound, or None."""
        model = self.force_model
        for candidate in ([model] + list(getattr(model, "models", []) or [])):
            if hasattr(candidate, "record_faces"):
                return candidate
        return None

    def _sync_face_capture(self) -> None:
        """Turn the per-face capture on exactly when something needs it."""
        model = self._face_model()
        if model is not None:
            model.record_faces = bool(getattr(self, "show_forces", False)
                                      or getattr(self, "_cap", None) is not None)

    def show_force_vectors(self, enable: bool = True) -> None:
        """Draw the wetted triangles and their per-face forces."""
        self.show_forces = bool(enable)
        self._sync_face_capture()

    def _draw_face_forces(self, stage) -> None:
        """The wetted triangles, and one arrow per face along its force.

        This is the force model's own intermediate state made visible: the
        polygons are what `_clip` actually produced after cutting each
        triangle at the waterline, and each line is -p*A*n for that piece. If
        the hull is behaving oddly, the arrows show whether the integration is
        wrong or the hull is simply somewhere unexpected.
        """
        import numpy as _np
        from pxr import Sdf, UsdGeom, Vt

        model = self._face_model()
        wet = getattr(model, "last_wet", None) if model else None
        vis = (UsdGeom.Tokens.inherited if self.show_forces and wet is not None
               and len(wet) else UsdGeom.Tokens.invisible)

        faces = UsdGeom.Mesh.Get(stage, "/World/Debug/WetFaces")
        if not faces:
            faces = UsdGeom.Mesh.Define(stage, "/World/Debug/WetFaces")
            faces.CreateSubdivisionSchemeAttr().Set("none")
            faces.CreateDoubleSidedAttr().Set(True)
            UsdGeom.PrimvarsAPI(faces).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.constant).Set(Vt.Vec3fArray([(1.0, 0.55, 0.15)]))
        arrows = UsdGeom.BasisCurves.Get(stage, "/World/Debug/ForceVectors")
        if not arrows:
            arrows = UsdGeom.BasisCurves.Define(stage, "/World/Debug/ForceVectors")
            arrows.CreateTypeAttr().Set("linear")
            arrows.CreateWidthsAttr().SetMetadata("interpolation", "constant")
            UsdGeom.PrimvarsAPI(arrows).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.constant).Set(Vt.Vec3fArray([(0.25, 1.0, 0.45)]))

        for prim in (faces, arrows):
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(vis)
        if vis == UsdGeom.Tokens.invisible:
            return

        pts = _np.asarray(wet, dtype=float).reshape(-1, 3)
        faces.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(_np.float32)))
        faces.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(wet)))
        faces.CreateFaceVertexIndicesAttr().Set(
            Vt.IntArray.FromNumpy(_np.arange(len(wet) * 3, dtype=_np.int32)))

        centroids = _np.asarray(model.last_centroids, dtype=float)
        forces = _np.asarray(model.last_forces, dtype=float)
        # Scaled so the biggest arrow is about a sixth of the hull, every
        # frame. A fixed newtons-per-metre would be unreadable across a 200 kg
        # cube and a 245 tonne ship, and unreadable again the moment a wave
        # doubles the load.
        span = float(_np.ptp(self.vessel.vertices[:, 0])) if self.vessel else 1.0
        biggest = float(_np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
        scale = (span / 6.0) / biggest if biggest > 0.0 else 0.0
        tips = centroids + forces * scale
        line_pts = _np.empty((len(centroids) * 2, 3))
        line_pts[0::2], line_pts[1::2] = centroids, tips
        arrows.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(line_pts.astype(_np.float32)))
        arrows.CreateCurveVertexCountsAttr().Set(Vt.IntArray([2] * len(centroids)))
        arrows.GetWidthsAttr().Set(Vt.FloatArray([max(span * 0.004, 0.005)]))

    # ---------------------------------------------------------- run recording

    def capture(self, path=_DEFAULT_PATH, surface_every: int = 10,
                faces_every: int = 1, limit_mb: float = 512.0):
        """Record the whole run to one .npz, overwritten each time.

        Pass None to stop and write. Stopping the timeline also writes it, so
        a run that is simply stopped is not a run that is lost.

            E.get().capture()            # start, into out/run.npz
            E.get().capture(None)        # stop and write

        Streams, and what they cost
        ---------------------------
        The hull's pose, velocity and total wrench are small - a few hundred
        bytes a second - so they are kept every tick. The other two are not:

          surface  128x128 floats is 65 kB a tick, 3.9 MB a second at 60 Hz
          faces    one centroid and one force per wetted piece, which for the
                   survey hull is a few thousand rows a tick

        Hence `surface_every` and `faces_every`, in ticks. The surface evolves
        on a wave period of seconds, so every 10th tick still oversamples it
        by a wide margin.

        Wetted faces come and go as the hull moves, so their count changes
        every tick and they cannot be stacked into a rectangular array. They
        are stored concatenated with a per-tick count, which you split with

            np.split(v0_face_forces, np.cumsum(v0_face_counts)[:-1])

        Layout
        ------
        Per-vessel streams are prefixed v0_, v1_, ... and `vessels` lists the
        prim path each index belongs to. One hull is written today; the
        prefix exists so a second one is an extra key rather than a new file
        format, which matters because vessel-to-vessel wake is the direction
        this is going. `t` and the surface_* arrays are unprefixed: there is
        one clock and one water surface, shared by everyone in the scene.

            d = np.load(path)
            list(d["vessels"])      -> ['/World/Vessel']
            d["v0_pos"], d["t"]     -> hull track against the shared clock
        """
        import numpy as _np

        from .paths import output

        # A sentinel, not None: None has always meant "stop and write", the
        # panel calls it that way, and overloading it to also mean "use the
        # default path" would have made stopping start a new capture.
        if path is _DEFAULT_PATH:
            path = output("run.npz")
        if self._cap is not None:
            self._write_capture()
        if path is None:
            self._sync_face_capture()
            return

        self._cap = {
            "path": str(path), "surface_every": max(1, int(surface_every)),
            "faces_every": max(1, int(faces_every)),
            "limit": float(limit_mb) * 1e6, "bytes": 0, "stopped": False,
            "t": [], "pos": [], "quat": [], "lin_vel": [], "ang_vel": [],
            "force": [], "moment": [],
            "surface_t": [], "surface_z": [], "surface_dx": [], "surface_dy": [],
            "face_t": [], "face_counts": [], "face_centroids": [], "face_forces": [],
        }
        self._sync_face_capture()
        self._log(f"capturing this run to {path} (overwrites)", BANNER_LEVEL)

    def _capture_row(self, state, wrench) -> None:
        import numpy as _np

        cap = self._cap
        if cap["stopped"]:
            return
        cap["t"].append(self._sim_time)
        cap["pos"].append(_np.asarray(state.position, dtype=_np.float32))
        cap["quat"].append(_np.asarray(state.orientation, dtype=_np.float32))
        cap["lin_vel"].append(_np.asarray(state.linear_velocity, dtype=_np.float32))
        cap["ang_vel"].append(_np.asarray(state.angular_velocity, dtype=_np.float32))
        cap["force"].append(_np.asarray(wrench.force, dtype=_np.float32))
        cap["moment"].append(_np.asarray(wrench.moment, dtype=_np.float32))
        cap["bytes"] += 22 * 4

        if self._step_count % cap["surface_every"] == 0 and self.field is not None:
            z = getattr(self.field, "_z", None)
            if z is not None:
                cap["surface_t"].append(self._sim_time)
                cap["surface_z"].append(_np.asarray(z, dtype=_np.float32))
                cap["bytes"] += z.size * 4
                for key, attr in (("surface_dx", "_dx"), ("surface_dy", "_dy")):
                    d = getattr(self.field, attr, None)
                    if d is not None:
                        cap[key].append(_np.asarray(d, dtype=_np.float32))
                        cap["bytes"] += d.size * 4

        model = self._face_model()
        if (model is not None and self._step_count % cap["faces_every"] == 0
                and getattr(model, "last_forces", None) is not None):
            c = _np.asarray(model.last_centroids, dtype=_np.float32)
            f = _np.asarray(model.last_forces, dtype=_np.float32)
            cap["face_t"].append(self._sim_time)
            cap["face_counts"].append(len(c))
            cap["face_centroids"].append(c)
            cap["face_forces"].append(f)
            cap["bytes"] += (c.size + f.size) * 4

        if cap["bytes"] > cap["limit"]:
            cap["stopped"] = True
            self._log(
                f"capture hit {cap['limit']/1e6:.0f} MB at t={self._sim_time:.1f}s "
                "and stopped recording. Raise limit_mb, or sample the surface "
                "and faces less often.", "warn")

    def _write_capture(self) -> None:
        import numpy as _np

        cap, self._cap = self._cap, None
        if cap is None or not cap["t"]:
            return
        # Per-vessel streams are namespaced v0_, v1_, ... and `vessels` says
        # which prim each index is. There is one hull today, so this writes
        # v0_ only - but wake interaction between vessels is the direction
        # this is heading, and a file format that assumes a single boat would
        # have to be thrown away the moment a second one floats. The water is
        # shared by everything in the scene, so surface_* stays unprefixed.
        path = getattr(self.vessel, "body_path", None) or "/World/Vessel"
        mass = float(getattr(self.vessel, "mass", 0.0) or 0.0)
        out = {
            "vessels": _np.asarray([str(path)]),
            "v0_path": _np.asarray(str(path)),
            "v0_mass": _np.asarray(mass),
        }
        for k in ("t", "pos", "quat", "lin_vel", "ang_vel", "force", "moment"):
            out[f"v0_{k}"] = _np.asarray(cap[k])
        # The clock is shared by every stream in the file, so it is also
        # written unprefixed - every analysis starts by reading it.
        out["t"] = out["v0_t"]
        if cap["surface_t"]:
            out["surface_t"] = _np.asarray(cap["surface_t"])
            out["surface_z"] = _np.stack(cap["surface_z"])
            for key in ("surface_dx", "surface_dy"):
                if cap[key]:
                    out[key] = _np.stack(cap[key])
        if cap["face_t"]:
            out["v0_face_t"] = _np.asarray(cap["face_t"])
            out["v0_face_counts"] = _np.asarray(cap["face_counts"], dtype=_np.int32)
            out["v0_face_centroids"] = _np.concatenate(cap["face_centroids"])
            out["v0_face_forces"] = _np.concatenate(cap["face_forces"])
        try:
            _np.savez_compressed(cap["path"], **out)
            self._log(f"capture written: {cap['path']}  "
                      f"{len(out['t'])} ticks, {cap['bytes']/1e6:.1f} MB raw",
                      BANNER_LEVEL)
        except Exception as exc:                      # noqa: BLE001
            self._log(f"capture could not be written ({exc})", "error")

    # ------------------------------------------------------------------ camera

    def follow_vessel(self, enable: bool = True, distance: float = 14.0,
                      height: float = 5.0, azimuth: float = 135.0,
                      lag: float = 0.45) -> None:
        """Track the hull with a chase camera, and point the viewport at it.

        Parenting a camera under the vessel would be simpler and is the wrong
        answer: it would inherit the hull's roll and pitch, and a camera that
        rolls with a boat in a seaway is unwatchable. This follows the
        POSITION only and keeps its own level horizon.

        `lag` is the time constant of an exponential follow, in seconds. Zero
        is rigid, which looks worse than you would expect - every wave the
        hull rides shows up as camera shake. Around half a second lets the
        boat move within the frame, which is what makes it read as a boat on
        water rather than a boat held in a vice.

        `azimuth` is a WORLD heading, not relative to the hull. A camera that
        swings round to stay behind a yawing boat makes the sea appear to
        rotate, and the sea is the thing being looked at.
        """
        self.follow = bool(enable)
        self._cam_distance = float(distance)
        self._cam_height = float(height)
        self._cam_azimuth = float(azimuth)
        self._cam_lag = max(float(lag), 0.0)
        if not enable:
            return
        self._cam_eye = self._cam_at = None      # snap, do not glide, on enable
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = "/World/ChaseCam"
        except Exception as exc:                  # noqa: BLE001
            self._log(f"camera made, but could not switch the viewport ({exc}). "
                      "Pick /World/ChaseCam from the viewport camera menu.",
                      BANNER_LEVEL)

    def _update_chase_camera(self, stage) -> None:
        import math

        import numpy as _np
        from pxr import Gf, UsdGeom

        if self.vessel is None:
            return
        try:
            target = _np.asarray(self.vessel.read_state().position, dtype=float)
        except Exception:                         # noqa: BLE001
            return

        a = math.radians(self._cam_azimuth)
        desired = target + _np.array([self._cam_distance * math.cos(a),
                                      self._cam_distance * math.sin(a),
                                      self._cam_height])
        if self._cam_eye is None:
            self._cam_eye, self._cam_at = desired, target
        else:
            # Exponential smoothing on the draw tick, not the physics tick:
            # 1 - exp(-dt/tau) makes the response independent of the rate, so
            # changing the redraw throttle does not change how it feels.
            alpha = 1.0 if self._cam_lag <= 0.0 else (
                1.0 - math.exp(-(1.0 / 20.0) / self._cam_lag))
            self._cam_eye += (desired - self._cam_eye) * alpha
            self._cam_at += (target - self._cam_at) * alpha

        cam = UsdGeom.Camera.Get(stage, "/World/ChaseCam")
        if not cam:
            cam = UsdGeom.Camera.Define(stage, "/World/ChaseCam")
            cam.CreateFocalLengthAttr(22.0)       # wide: keep sea in the frame
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100000.0))
        # SetLookAt builds a VIEW matrix, which maps world into the camera.
        # A prim's transform is the other direction, so it is the inverse.
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*self._cam_eye), Gf.Vec3d(*self._cam_at), Gf.Vec3d(0, 0, 1))
        xf = UsdGeom.Xformable(cam.GetPrim())
        ops = xf.GetOrderedXformOps()
        if not ops or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
            xf.ClearXformOpOrder()
            ops = [xf.AddTransformOp()]
        ops[0].Set(view.GetInverse())

    def show_ui(self) -> None:
        """Open (or re-open) the spectrum panel.

        Imported here rather than at module scope: ui.py needs omni.ui, which
        does not exist outside Kit, and tools/selftest.py imports this package
        from a plain python shell. A missing panel must never be the reason
        the physics cannot be tested.
        """
        try:
            from .ui import SpectrumPanel

            if self._panel is not None:
                self._panel.destroy()
            self._panel = SpectrumPanel(self)
        except Exception as exc:                      # noqa: BLE001
            self._log(f"no UI panel ({exc})", "warn")
            self._panel = None

    def record(self, path=None):
        """Write one row per physics step to CSV, for frequency-domain work.

            E.get().record(output("heave.csv"))    # start
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
        # A recording that does not say what produced it is close to useless a
        # day later. Two sweeps have already been confounded - one by changing
        # two things at once, one by four runs overwriting the same filename -
        # so the conditions go in the file itself, not in anyone's memory.
        # No commas in here: a reader splitting on commas counts the
        # metadata as a row and the whole file fails to parse, which is
        # exactly what happened the first time this was used.
        kw = " ".join(f"{k}={v}" for k, v in sorted(self._field_kwargs.items()))
        self._csv.write(
            f"# force_lead={self.force_lead} rot_lead={self.rot_lead}"
            f" field={self._field_name} {kw}"
            f" mass={getattr(self.vessel, 'mass', None)}\n")
        self._csv_writer = csv.writer(self._csv)
        self._csv_writer.writerow([
            "step", "t", "x", "y", "z", "water", "depth", "tilt_deg",
            "vx", "vy", "vz", "fx", "fy", "fz",
        ])
        self._csv_rows = 0
        self._log(f"recording to {path} (one row per step)", BANNER_LEVEL)

    def sweep_lead(self, values=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
                   settle: float = 8.0, measure: float = 20.0, prefix=None,
                   attr: str = "force_lead"):
        """Sweep force_lead automatically, one recording per value.

            E.get().sweep_lead()

        Driven from the physics callback rather than by hand, because hand
        timing is how the last two attempts went wrong: the settings were
        changed two at a time, and four runs wrote to one filename. Each value
        gets `settle` seconds to reach steady state, which is the other thing
        that caught us out - a limit cycle measured while it is still growing
        reads as a much smaller one - and then `measure` seconds recorded to
        its own file.
        """
        from .paths import output

        values = [float(v) for v in values]
        if not values:
            raise ValueError("sweep_lead needs at least one value")
        self._sweep = {
            "values": values, "i": -1, "settle": float(settle),
            "measure": float(measure),
            "prefix": str(prefix) if prefix else None, "attr": str(attr),
            "phase": "settle", "until": 0.0, "output": output,
        }
        self._sweep_advance()

    def _sweep_path(self, lead):
        name = f"{self._sweep['attr']}_{lead:.2f}".replace(".", "p") + ".csv"
        pre = self._sweep["prefix"]
        return f"{pre}{name}" if pre else self._sweep["output"](name)

    def _sweep_advance(self) -> None:
        """settle -> measure -> next value. Called from the physics step."""
        sw = self._sweep
        if sw is None:
            return
        if sw["phase"] == "settle" and sw["i"] >= 0:
            self.record(self._sweep_path(sw["values"][sw["i"]]))
            sw["phase"] = "measure"
            sw["until"] = self._sim_time + sw["measure"]
            return
        if sw["phase"] == "measure":
            self.record(None)
        sw["i"] += 1
        if sw["i"] >= len(sw["values"]):
            self._sweep = None
            self._log("sweep finished", BANNER_LEVEL)
            return
        lead = sw["values"][sw["i"]]
        setattr(self, sw["attr"], lead)
        sw["phase"] = "settle"
        sw["until"] = self._sim_time + sw["settle"]
        self._log(f"sweep: {sw['attr']} = {lead} (settling {sw['settle']:.0f}s)",
                  BANNER_LEVEL)

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
                    FORCES.create(
                        "thruster",
                        # 2% of the vessel's own weight per screw. Not a
                        # propeller curve - a number, chosen so that when the
                        # hull moves you know exactly what moved it.
                        thrust=0.02 * bridge.mass * 9.80665,
                        length=float(extents[0]), beam=float(extents[1]),
                        draft=float(extents[2]) * 0.5,
                    ),
                ],
            )
            self.thruster = next(
                (m for m in self.force_model.models
                 if hasattr(m, "set_drive")), None)
            if self.thruster is not None:
                # numpy is imported per-function in this module, not at the
                # top: the package must stay importable from a plain shell for
                # tools/selftest.py, and a module-level import here was never
                # needed.
                import numpy as _np

                port, starboard = self.thruster.positions[:2]
                self._log(
                    f"  drive  {self.thruster.thrust/1000.0:.1f} kN per screw "
                    f"at {_np.round(port, 2)} and {_np.round(starboard, 2)} m "
                    "(body frame)", BANNER_LEVEL)
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
            self._sync_face_capture()
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
                # Only if this is the stage we were set up on. Pressing play
                # in an unrelated scene must do nothing at all.
                if self.active and self._stage_is_ours() and self.vessel is None:
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
                if self._cap is not None:
                    self._write_capture()
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
        if not self.active or not self._stage_is_ours():
            return

        self._step_count += 1
        self._sim_time += dt
        if self._sweep is not None and self._sim_time >= self._sweep["until"]:
            self._sweep_advance()

        if self.field is not None:
            self.field.advance(self._sim_time)
            if(self._sim_time - self._last_draw >= 1.0 / 20.0): #throttle limits how often the mesh is redrawn. 20Hz
                self._last_draw = self._sim_time
                import omni.usd
                stage = omni.usd.get_context().get_stage()
                self.field.write_mesh(stage)
                # Drawn on the same throttle as the water, so the lines never
                # lag the surface they are supposed to be lying on.
                if self.show_grid or self._grid_drawn:
                    self.field.write_grid(stage, stride=self._grid_stride,
                                          visible=self.show_grid)
                    self._grid_drawn = self.show_grid
                if self.show_forces or self._forces_drawn:
                    self._draw_face_forces(stage)
                    self._forces_drawn = self.show_forces
                if self.follow:
                    self._update_chase_camera(stage)

        if self.vessel is not None and self.force_model is not None and self.field is not None:
            try:
                state = self.vessel.read_state()
                # Kept for the heartbeat below, which has to say where the
                # hull is even when the per-step trace is off. Holding the
                # reference costs nothing; the heartbeat does the arithmetic
                # once a second.
                self._last_state = state
                # A wake field needs to know where the hull is and how fast
                # it is going; no other field does. Duck-typed rather than
                # added to the WaveField contract, so a field that does not
                # shed a wake needs no empty method and no edit.
                _feed = getattr(self.field, "set_source", None)
                if _feed is not None:
                    _feed(state.position, state.linear_velocity)
                wrench = self.force_model.wrench(
                    _lead(state, dt, self.force_lead, self.rot_lead), self.field,
                    self._sim_time + self.force_lead * dt,
                )
                self.vessel.apply(wrench)
                if self._cap is not None:
                    self._capture_row(state, wrench)

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
