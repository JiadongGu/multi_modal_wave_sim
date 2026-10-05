"""Physical constants.

ASTRA hardcoded `GRAVITY = 9.8` and `WATER_DENSITY = 1000`, and then hardcoded
`rho = 1000` a second time inside the buoyancy call.  Both live here instead,
and salt water is present from the start because the pond is not the target
environment forever.
"""

GRAVITY = 9.80665           # standard gravity [m/s^2]
WATER_DENSITY_FRESH = 1000.0   # [kg/m^3]
WATER_DENSITY_SALT = 1025.0    # [kg/m^3] typical open-ocean surface value

# Default for a new scene.  Override per-environment rather than editing this.
WATER_DENSITY = WATER_DENSITY_FRESH
