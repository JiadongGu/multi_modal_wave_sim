"""A panel of sliders for the JONSWAP parameters.

Everything here is a view. It reads the extension's current spectrum to seed
itself and calls `select()` to change it - the same entry point the Script
Editor uses - so there is no second path into the model that could drift from
the first. If a slider does something `select()` cannot, that is a bug in the
slider.

Why the sliders apply on RELEASE and not while dragging
-------------------------------------------------------
Changing a spectral parameter rebuilds the wave field, because an FFT field
samples the spectrum once in _build() and bakes h0. That rebuild is about
5 ms at resolution 64. Cheap - but a drag emits a value change per mouse move,
so continuously applying would rebuild the sea tens of times a second and
reseed the surface under a floating hull each time. add_end_edit_fn fires once,
when the mouse comes up.

What changing a parameter does to a hull already in the water
--------------------------------------------------------------
It reseeds the surface. h0 is drawn from the spectrum, so a new spectrum is a
new realisation of the sea, not a smooth morph of the old one - the water under
the boat jumps. The clock is preserved (select() advances the new field to the
current sim time) but the waves themselves are new. That is inherent to
changing a sea state, not a defect, and it is the reason the panel reports the
Hs it actually got rather than the one you asked for.
"""

from __future__ import annotations

from math import degrees, pi, radians

# Slider name, attribute on the spectrum, min, max, unit, and what it does.
# Ranges are chosen to span the useful physics, not to be round numbers:
# gamma below 1 is unphysical and the class rejects it, and gamma = 1 is
# exactly Pierson-Moskowitz, so the slider's left end is a different spectrum.
# None means "ask the extension", which resolves to <repo>/out.
CAPTURE_PATH = None

PARAMS = (
    ("Hs", "hs", 0.0, 8.0, "m",
     "significant wave height: the mean of the highest third, = 4 * the "
     "surface standard deviation"),
    ("Tp", "tp", 1.0, 20.0, "s",
     "peak period: where the spectrum has most of its energy. "
     "wavelength = g Tp^2 / 2pi"),
    ("gamma", "gamma", 1.0, 7.0, "",
     "peak enhancement. 1.0 is Pierson-Moskowitz (fully developed); 3.3 is "
     "the JONSWAP mean; higher is a narrower, fetch-limited sea"),
    ("direction", "wind_dir", -180.0, 180.0, "deg",
     "mean wave heading. stored in radians, shown in degrees"),
    ("spread s", "spread_s", 1.0, 30.0, "",
     "cos-2s spreading exponent. low is a confused sea spread over many "
     "headings, high is a narrow swell all going one way"),
)


class SpectrumPanel:
    """Sliders for the JONSWAP parameters, wired to extension.select()."""

    def __init__(self, ext) -> None:
        import omni.ui as ui

        self._ext = ext
        self._ui = ui
        self._sliders = {}
        self._value_labels = {}
        self._readout = None
        self._window = ui.Window("Multi-Modal Wave Sim", width=420, height=320)
        with self._window.frame:
            self._build()
        self.refresh()

    # ------------------------------------------------------------------ layout

    def _build(self) -> None:
        ui = self._ui
        with ui.VStack(spacing=6, height=0):
            with ui.HStack(height=24, spacing=6):
                self._active = ui.CheckBox(width=20)
                self._active.model.add_value_changed_fn(
                    lambda m: self._ext.activate(bool(m.get_value_as_bool())))
                ui.Label("active on this stage", height=22,
                         tooltip="off means the extension touches nothing. It "
                                 "turns itself off when a different file is "
                                 "opened, so pressing play elsewhere never "
                                 "drops water into someone else's scene",
                         style={"font_size": 14, "color": 0xFFCCCCCC})
            ui.Separator(height=6)
            ui.Label("JONSWAP spectrum", height=22,
                     style={"font_size": 16, "color": 0xFFCCCCCC})
            ui.Label("sliders apply when you release them", height=16,
                     style={"font_size": 12, "color": 0xFF888888})
            ui.Spacer(height=4)

            for label, attr, lo, hi, unit, tip in PARAMS:
                with ui.HStack(height=24, spacing=6):
                    ui.Label(label, width=70, tooltip=tip)
                    slider = ui.FloatSlider(min=lo, max=hi)
                    slider.model.set_value(lo)
                    value = ui.Label("", width=74,
                                     style={"color": 0xFFAAAAAA})
                    self._sliders[attr] = slider
                    self._value_labels[attr] = (value, unit)
                    # Two callbacks on purpose. The first keeps the number
                    # beside the slider honest while you drag; the second is
                    # the only one that touches the model.
                    slider.model.add_value_changed_fn(
                        lambda m, a=attr: self._show(a, m.get_value_as_float()))
                    slider.model.add_end_edit_fn(lambda m: self.apply())

            ui.Spacer(height=8)
            ui.Label("surface", height=20,
                     style={"font_size": 14, "color": 0xFFCCCCCC})
            # Not spectrum parameters: these change the field, not the sea
            # state, so they are set on the live field rather than going
            # through select(). Rebuilding for a visual knob would reseed the
            # waves under a floating hull for no reason.
            with ui.HStack(height=24, spacing=6):
                ui.Label("choppiness", width=70,
                         tooltip="horizontal displacement toward the crests. "
                                 "0 is a plain height field, 1 is standard, "
                                 "past ~1.5 the surface folds through itself. "
                                 "Visual only - height() is untouched")
                chop = ui.FloatSlider(min=0.0, max=1.5)
                self._chop = chop
                self._chop_label = ui.Label("", width=74,
                                            style={"color": 0xFFAAAAAA})
                chop.model.add_value_changed_fn(
                    lambda m: setattr(self._chop_label, "text",
                                      f"{m.get_value_as_float():.2f}"))
                chop.model.add_end_edit_fn(lambda m: self._set_surface())

            with ui.HStack(height=24, spacing=6):
                ui.Label("foam at", width=70,
                         tooltip="Jacobian below which the surface counts as "
                                 "broken. 1 is flat water, 0 is a fold. Lower "
                                 "means only the steepest crests go white")
                foam = ui.FloatSlider(min=0.0, max=1.0)
                self._foam = foam
                self._foam_label = ui.Label("", width=74,
                                            style={"color": 0xFFAAAAAA})
                foam.model.add_value_changed_fn(
                    lambda m: setattr(self._foam_label, "text",
                                      f"{m.get_value_as_float():.2f}"))
                foam.model.add_end_edit_fn(lambda m: self._set_surface())

            with ui.HStack(height=24, spacing=6):
                ui.Label("opacity", width=70,
                         tooltip="how much of the sea you can see through. "
                                 "1 is opaque. Below about 0.8 the hull shows "
                                 "through the surface, which is the only way "
                                 "to see how deep it is actually sitting")
                opac = ui.FloatSlider(min=0.05, max=1.0)
                opac.model.set_value(1.0)
                self._opacity = opac
                self._opacity_label = ui.Label("1.00", width=74,
                                               style={"color": 0xFFAAAAAA})
                opac.model.add_value_changed_fn(
                    lambda m: setattr(self._opacity_label, "text",
                                      f"{m.get_value_as_float():.2f}"))
                opac.model.add_end_edit_fn(lambda m: self._set_opacity())

            with ui.HStack(height=24, spacing=6):
                ui.Label("colour", width=70,
                         tooltip="paint the wake onto the water - orange on "
                                 "its crests, green in its troughs. Signed, "
                                 "so a wave train reads as a wave train "
                                 "rather than as bars at twice the "
                                 "frequency. Colour only: the surface stays "
                                 "exactly the surface the hull floats on")
                self._wake_tint = ui.CheckBox(width=20)
                self._wake_tint.model.add_value_changed_fn(
                    lambda m: self._set_wake_tint(bool(m.get_value_as_bool())))
                ui.Label("colour the wake", style={"color": 0xFFAAAAAA})

            with ui.HStack(height=24, spacing=6):
                ui.Label("wake x", width=70,
                         tooltip="scale the wake's drawn HEIGHT. 1 is honest - "
                                 "what you see is what the hull floats on. "
                                 "Above that the picture and the physics "
                                 "disagree on purpose, which is useful for "
                                 "seeing a few centimetres of wake on a 5 m "
                                 "boat and useless for measuring anything")
                exag = ui.FloatSlider(min=1.0, max=10.0)
                exag.model.set_value(1.0)
                self._exaggerate = exag
                self._exaggerate_label = ui.Label("1.00", width=74,
                                                  style={"color": 0xFFAAAAAA})
                exag.model.add_value_changed_fn(
                    lambda m: setattr(self._exaggerate_label, "text",
                                      f"{m.get_value_as_float():.2f}"))
                exag.model.add_end_edit_fn(lambda m: self._set_exaggerate())

            with ui.HStack(height=24, spacing=6):
                ui.Label("grid", width=70,
                         tooltip="draw the FFT grid lines on the surface - "
                                 "shows the resolution you are working at, "
                                 "and the horizontal drag a shaded sheet hides")
                self._grid = ui.CheckBox(width=20)
                self._grid.model.add_value_changed_fn(
                    lambda m: setattr(self._ext, "show_grid",
                                      bool(m.get_value_as_bool())))
                ui.Label("show wave grid", style={"color": 0xFFAAAAAA})

            with ui.HStack(height=24, spacing=6):
                ui.Label("forces", width=70,
                         tooltip="draw the clipped wetted triangles and one "
                                 "arrow per face along its pressure force. "
                                 "Arrows rescale every frame so the largest "
                                 "is about a sixth of the hull")
                self._arrows = ui.CheckBox(width=20)
                self._arrows.model.add_value_changed_fn(
                    lambda m: self._ext.show_force_vectors(bool(m.get_value_as_bool())))
                ui.Label("show hull faces + force vectors",
                         style={"color": 0xFFAAAAAA})

            with ui.HStack(height=24, spacing=6):
                ui.Label("record", width=70,
                         tooltip="write every tick of this run to "
                                 "out/run.npz - pose, velocity, wrench, "
                                 "the wave surface, and every face force. "
                                 "Overwritten each run, written on Stop")
                self._capture = ui.CheckBox(width=20)
                self._capture.model.add_value_changed_fn(
                    lambda m: (self._ext.capture(CAPTURE_PATH)
                               if m.get_value_as_bool()
                               else self._ext.capture(None)))
                ui.Label("record run to .npz", style={"color": 0xFFAAAAAA})

            with ui.HStack(height=24, spacing=6):
                ui.Label("wake", width=70,
                         tooltip="the wake the hull leaves behind it, shed as "
                                 "damped rings along its track and summed - "
                                 "so the Kelvin wedge is a result, not a "
                                 "drawing. Rides on top of whatever sea is "
                                 "already there; unticking puts that sea back "
                                 "exactly as it was")
                self._wake = ui.CheckBox(width=20)
                self._wake.model.add_value_changed_fn(
                    lambda m: self._ext.show_wake(bool(m.get_value_as_bool())))
                ui.Label("hull wake", style={"color": 0xFFAAAAAA})

            with ui.HStack(height=24, spacing=6):
                ui.Label("drive", width=70,
                         tooltip="I/K ahead and astern, J/L turn. Two screws "
                                 "at a known point below the centre of "
                                 "gravity, so opening the throttle lifts the "
                                 "bow the way it should")
                self._drive = ui.CheckBox(width=20)
                self._drive.model.add_value_changed_fn(
                    lambda m: self._ext.keyboard_drive(bool(m.get_value_as_bool())))
                ui.Label("IJKL thrust", style={"color": 0xFFAAAAAA})

            with ui.HStack(height=24, spacing=6):
                ui.Label("camera", width=70,
                         tooltip="chase the hull with a level camera. Follows "
                                 "position only - a camera that rolled with "
                                 "the boat would be unwatchable")
                self._follow = ui.CheckBox(width=20)
                self._follow.model.add_value_changed_fn(
                    lambda m: self._ext.follow_vessel(bool(m.get_value_as_bool())))
                ui.Label("follow the vessel", style={"color": 0xFFAAAAAA})

            ui.Spacer(height=6)
            with ui.HStack(height=26, spacing=6):
                ui.Button("Apply", clicked_fn=self.apply, width=80)
                ui.Button("Pierson-Moskowitz", width=150,
                          tooltip="gamma = 1: the same spectrum with no peak "
                                  "enhancement",
                          clicked_fn=lambda: self._set("gamma", 1.0))
                ui.Button("Refresh", clicked_fn=self.refresh, width=80,
                          tooltip="re-read the sliders from the live spectrum")
            ui.Spacer(height=4)
            self._readout = ui.Label("", height=46, word_wrap=True,
                                     style={"font_size": 12,
                                            "color": 0xFF99CC99})

    # ------------------------------------------------------------------- state

    def _set_wake_tint(self, on: bool) -> None:
        field = getattr(self._ext, "field", None)
        if field is not None and hasattr(field, "highlight"):
            field.highlight = bool(on)

    def _set_exaggerate(self) -> None:
        """Visual gain on the wake. Does not touch what the force model sees.

        The wake field adds `exaggerate` to the drawn surface only -
        height(), which the hull is integrated against, is untouched. So this
        makes the wake findable without quietly changing the experiment.
        """
        field = getattr(self._ext, "field", None)
        if field is None or not hasattr(field, "exaggerate"):
            return
        field.exaggerate = float(self._exaggerate.model.get_value_as_float())

    def _set_opacity(self) -> None:
        """Write opacity onto the water material make_scene.py built.

        Reaches into the material rather than the mesh because opacity is a
        shading property, and the mesh is rewritten from scratch every frame -
        anything set on it would be gone by the next redraw.
        """
        value = float(self._opacity.model.get_value_as_float())
        try:
            import omni.usd
            from pxr import Sdf, UsdShade

            stage = omni.usd.get_context().get_stage()
            shader = UsdShade.Shader.Get(stage, "/World/Looks/Water/Shader")
            if not shader:
                self._readout.text = ("no water material - run "
                                      "tools/make_scene.py first")
                return
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(value)
            # Without a threshold of 0 the renderer treats the surface as a
            # cutout and you get stippled holes rather than translucency.
            shader.CreateInput("opacityThreshold",
                               Sdf.ValueTypeNames.Float).Set(0.0)
        except Exception as exc:                      # noqa: BLE001
            if self._readout:
                self._readout.text = f"opacity failed: {exc}"

    def _set_surface(self) -> None:
        """Push the visual knobs onto the live field, without rebuilding it.

        choppiness and foam_threshold are read fresh every advance(), so
        setting them takes effect on the next step with no reseeding - unlike
        a spectrum change, which has to rebuild h0.
        """
        field = getattr(self._ext, "field", None)
        if field is None:
            return
        if hasattr(field, "choppiness"):
            field.choppiness = self._chop.model.get_value_as_float()
            field.foam_threshold = self._foam.model.get_value_as_float()
        self._report()

    def _show(self, attr, value) -> None:
        label, unit = self._value_labels[attr]
        label.text = f"{value:.2f} {unit}".strip()

    def _set(self, attr, value) -> None:
        self._sliders[attr].model.set_value(float(value))
        self.apply()

    def refresh(self) -> None:
        """Pull the slider positions from whatever the spectrum currently is.

        The Script Editor can change the spectrum too, and a panel showing
        stale numbers is worse than no panel.
        """
        spectrum = getattr(self._ext, "spectrum", None)
        if spectrum is None:
            return
        for _, attr, lo, hi, _, _ in PARAMS:
            value = getattr(spectrum, attr, None)
            if value is None:
                continue
            if attr == "wind_dir":
                value = degrees(value)
            value = min(max(float(value), lo), hi)
            self._sliders[attr].model.set_value(value)
            self._show(attr, value)

        field = getattr(self._ext, "field", None)
        chop = float(getattr(field, "choppiness", 0.0))
        thr = float(getattr(field, "foam_threshold", 0.55))
        self._chop.model.set_value(chop)
        self._foam.model.set_value(thr)
        self._chop_label.text = f"{chop:.2f}"
        self._foam_label.text = f"{thr:.2f}"
        self._active.model.set_value(bool(getattr(self._ext, "active", False)))
        self._grid.model.set_value(bool(getattr(self._ext, "show_grid", False)))
        self._follow.model.set_value(bool(getattr(self._ext, "follow", False)))
        self._arrows.model.set_value(bool(getattr(self._ext, "show_forces", False)))
        self._wake.model.set_value(bool(getattr(self._ext, "wake_on", False)))
        self._wake_tint.model.set_value(bool(getattr(field, "highlight", False)))
        exag = float(getattr(field, "exaggerate", 1.0))
        self._exaggerate.model.set_value(exag)
        self._exaggerate_label.text = f"{exag:.2f}"
        self._drive.model.set_value(getattr(self._ext, "_kb_sub", None) is not None)
        self._capture.model.set_value(getattr(self._ext, "_cap", None) is not None)
        self._report()

    def apply(self) -> None:
        kwargs = {}
        for _, attr, _, _, _, _ in PARAMS:
            value = self._sliders[attr].model.get_value_as_float()
            kwargs[attr] = radians(value) if attr == "wind_dir" else value
        try:
            self._ext.select(spectrum="jonswap", spectrum_kwargs=kwargs)
        except Exception as exc:                      # noqa: BLE001
            if self._readout:
                self._readout.text = f"rejected: {exc}"
            return
        # select() rebuilt the field, so it came back with the class defaults.
        # Put the sliders' values back or the panel and the water disagree.
        self._set_surface()

    def _report(self) -> None:
        """What the water ACTUALLY is, which is not always what was asked for.

        Hs comes back from the realised surface, not from the parameter: a
        finite tile and a finite resolution cannot hold every wave the
        spectrum describes, so the number it achieves is the honest one.
        """
        if self._readout is None:
            return
        field = getattr(self._ext, "field", None)
        if field is None:
            self._readout.text = "no field"
            return
        tp = self._sliders["tp"].model.get_value_as_float()
        wavelength = 9.80665 * tp * tp / (2.0 * pi)
        try:
            achieved = float(field.significant_height())
        except Exception:                             # noqa: BLE001
            achieved = float("nan")
        asked = self._sliders["hs"].model.get_value_as_float()
        note = ""
        # Significant steepness, 2 pi Hs / (g Tp^2). Waves break near 1/7, so
        # a sea above that is not steep - it is impossible, and what the
        # renderer shows is a field of razor peaks that no ocean sustains.
        # Worth saying out loud: the sliders happily go there, and the result
        # looks like a bug in the wave model rather than a request for a sea
        # that cannot exist.
        steep = 2.0 * pi * asked / (9.80665 * tp * tp)
        if steep > 0.142:
            hs_max = 0.142 * 9.80665 * tp * tp / (2.0 * pi)
            note = (f"\nsteepness {steep:.3f} is past breaking (1/7 = 0.143)"
                    f" - at Tp {tp:.1f} s the limit is Hs {hs_max:.1f} m")
        elif steep > 0.100:
            note = f"\nsteepness {steep:.3f} - a very steep, storm sea"
        # Statistical, not structural: the tile holds the wave, but too few
        # of them to pin Hs down. Measured over 5 seeds at Hs 1.5 / Tp 6:
        # 2.3 waves per tile scatters 12%, 4.6 scatters 5%, 9 scatters 3%.
        # Worth saying, because an RAO measured against a nominal Hs inherits
        # that scatter as an error term.
        waves = field.tile_size / max(wavelength, 1e-6)
        if 1.0 < waves < 4.0:
            note += (f"\nonly {waves:.1f} waves per tile - Hs will scatter "
                     f"~{12.0 * 2.3 / waves:.0f}% between seeds")
        if wavelength > field.tile_size:
            note += (f"\npeak wavelength {wavelength:.0f} m is longer than the "
                    f"{field.tile_size:.0f} m tile - it will not fit")
        elif wavelength < 4.0 * field.cell_size:
            note += (f"\npeak wavelength {wavelength:.1f} m is under 4 cells "
                    f"({field.cell_size:.2f} m) - under-resolved")
        self._readout.text = (
            f"realised Hs {achieved:.3f} m   (asked {asked:.2f})\n"
            f"peak wavelength {wavelength:.1f} m   tile {field.tile_size:.0f} m"
            f"   cell {field.cell_size:.2f} m{note}"
        )

    def destroy(self) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None
