

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


class MultiModalWaveSimExtension(_ExtBase):
    """Lifecycle owner. Holds the current selection and the event handles."""

    def on_startup(self, ext_id: str) -> None:
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

        self._log(f"starting up (ext_id={ext_id})", BANNER_LEVEL)
        self._log(f"  spectra:      {', '.join(SPECTRA.names()) or '(none)'}", BANNER_LEVEL)
        self._log(f"  wave fields:  {', '.join(FIELDS.names()) or '(none)'}", BANNER_LEVEL)
        self._log(f"  force models: {', '.join(FORCES.names()) or '(none)'}", BANNER_LEVEL)

        # Safe defaults: a spectrum that works, a field that is flat and a
        # force model that does nothing. Nothing here touches the stage, so
        # enabling the extension cannot disturb a scene.
        try:
            self.select(spectrum="pierson_moskowitz", field="flat", force_model="zero")
        except Exception as exc:                      # noqa: BLE001
            self._log(f"  could not build defaults: {exc}", "error")

        self._subscribe()
        self._log("ready", BANNER_LEVEL)

    def on_shutdown(self) -> None:
        # Unsubscribe before dropping anything else: a callback firing into a
        # half-torn-down extension is how Isaac Sim crashes on reload, and
        # Python extensions reload often during development.
        self._unsubscribe()
        self.spectrum = None
        self.field = None
        self.force_model = None
        self._log("shut down", BANNER_LEVEL)

    # ------------------------------------------------------------------ the swap

    def select(self, spectrum=None, field=None, force_model=None) -> None:
        """Swap any of the three components by registry name.

        Each is independent, so `select(spectrum="jonswap")` leaves the field
        and force model alone. This is the method a UI dropdown, a ROS 2
        service or a test harness should call.
        """
        if spectrum is not None:
            self.spectrum = SPECTRA.create(spectrum)
            self._log(f"spectrum  -> {self.spectrum!r}", BANNER_LEVEL)
        if field is not None:
            self.field = FIELDS.create(field, spectrum=self.spectrum)
            self._log(f"field     -> {self.field!r}", BANNER_LEVEL)
        if force_model is not None:
            self.force_model = FORCES.create(force_model)
            self._log(f"forces    -> {self.force_model!r}", BANNER_LEVEL)

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

            if event.type == int(omni.timeline.TimelineEventType.STOP):
                self._step_count = 0
                self._sim_time = 0.0
                self._last_draw = 0.0
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

        # Proof of life without flooding the console: roughly once a second at
        # a 60 Hz step. Delete once there is real work here.
        if self._step_count % 60 == 0:
            self._log(
                f"step {self._step_count}  t={self._sim_time:.2f}s  dt={dt * 1000:.2f}ms",
                "warn" if self._step_count <= 300 else "info",
            )

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
