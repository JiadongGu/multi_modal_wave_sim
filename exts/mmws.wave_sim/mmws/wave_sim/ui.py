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
                chop = ui.FloatSlider(min=0.0, max=2.0)
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
        self._grid.model.set_value(bool(getattr(self._ext, "show_grid", False)))
        self._follow.model.set_value(bool(getattr(self._ext, "follow", False)))
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
        if wavelength > field.tile_size:
            note = (f"\npeak wavelength {wavelength:.0f} m is longer than the "
                    f"{field.tile_size:.0f} m tile - it will not fit")
        elif wavelength < 4.0 * field.cell_size:
            note = (f"\npeak wavelength {wavelength:.1f} m is under 4 cells "
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
