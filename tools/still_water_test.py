"""Float the hull on flat water and check the draft against hand arithmetic.

Run in Isaac Sim's Script Editor with the timeline STOPPED:

    exec(open("/home/asv/dev/multi_modal_wave_sim/tools/still_water_test.py").read())

then press Play and watch the console.

Why flat water, when the whole point of the project is waves: a 1 m cube at
200 kg has a metacentric height of about 17 mm.

    BM = I / V = (1/12) / 0.2   = 0.417 m      waterplane second moment
    KB = draft / 2              = 0.100 m      centre of buoyancy above keel
    KG = 0.5 m                                 centre of a uniform cube
    GM = KB + BM - KG           = 0.017 m

Seventeen millimetres is neutral stability for practical purposes, so in any
real sea the cube lolls and tumbles - correctly. That makes it a terrible
instrument for checking whether buoyancy is right, because every discrepancy
has two possible explanations. On flat water there is only one right answer
and it is known in advance:

    draft = mass / (rho * waterplane area) = 200 / (1000 * 1) = 0.200 m

The cube's centre then sits 0.300 m ABOVE the surface (0.5 m half-height minus
0.2 m of draft), so the extension's `depth = water - z` settles at -0.300.
"""

import mmws.wave_sim.extension as E

ext = E.get()
if ext is None:
    raise RuntimeError("extension is not enabled - turn on mmws.wave_sim first")

ext.select(field="flat")

MASS = 200.0
RHO = 1000.0
AREA = 1.0

draft = MASS / (RHO * AREA)
print()
print("  still-water test armed")
print(f"    field            {ext.field!r}")
print(f"    expected draft   {draft:.3f} m")
print(f"    expected depth   {-(0.5 - draft):+.3f}   (depth = water - z, printed each step)")
print(f"    expected z       {0.5 - draft:+.3f} m    (water is at z = 0)")
print()
print("    buoyancy spring period  "
      f"{2 * 3.14159 / (RHO * 9.81 * AREA / MASS) ** 0.5:.2f} s")
print("    so give it ~5 s to ring down before reading the draft")
print()
print("  press Play")
