"""Multi-Modal Wave Sim - swappable ocean simulation for Isaac Sim.

Three things are swappable, and the package is laid out accordingly:

    spectra/   how much wave energy sits at each frequency and direction
    field/     how a spectrum becomes a moving water surface you can query
    forces/    how that surface becomes forces and moments on a hull

Importing this package registers every built-in implementation, so
`registry.SPECTRA.names()` and friends are populated by the time Kit calls
`on_startup`.
"""

from .globals import GRAVITY, WATER_DENSITY_FRESH, WATER_DENSITY_SALT
from .registry import SPECTRA, FIELDS, FORCES
from .paths import output, output_dir

# Importing these for their side effect: each module decorates its class with
# the matching registry, which is what makes the swap-by-name work.
from .spectra import base as _spectra_base          # noqa: F401
from .spectra import pierson_moskowitz as _pm       # noqa: F401
from .spectra import jonswap as _jonswap            # noqa: F401
from .spectra import eckv as _eckv                  # noqa: F401
from .field import flat as _flat                    # noqa: F401
from .field import fft as _fft                      # noqa: F401
from .forces import zero as _zero                   # noqa: F401
from .forces import buoyancy_pressure as _bp       # noqa: F401
from .forces import thruster as _thruster          # noqa: F401
from .forces import damping as _damping            # noqa: F401
from .forces import composite as _composite        # noqa: F401

# Kit discovers the extension by scanning this module for omni.ext.IExt
# subclasses, so the class has to be reachable from here.
from .extension import MultiModalWaveSimExtension   # noqa: F401

__all__ = [
    "GRAVITY",
    "WATER_DENSITY_FRESH",
    "WATER_DENSITY_SALT",
    "SPECTRA",
    "FIELDS",
    "FORCES",
    "MultiModalWaveSimExtension",
]
