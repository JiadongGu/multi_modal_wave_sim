"""usv_core: vessel physics for many boats at once, as plain PyTorch.

Every function takes and returns batches with a leading dimension N (one row
per environment), so one boat and four thousand boats are the same code.
Nothing here imports Isaac Sim or Isaac Lab; the environment converts at its
boundary and calls in. Conventions live in `conventions.py`.
"""

from .conventions import (
    BodyState,
    Wrench,
    body_to_world,
    quat_to_matrix,
    world_to_body,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from .model import Composite, HullModel, ModelContext, ZeroModel
from .adapters import ConventionAdapter
from .spectrum import SeaCfg, WaveComponents
from .waves import FlatWater, SineWaveField, WaveField
from .hull import ColumnBuoyancy, HullCfg, HullGeometry

__all__ = [
    "ColumnBuoyancy",
    "Composite",
    "ConventionAdapter",
    "FlatWater",
    "SeaCfg",
    "SineWaveField",
    "WaveComponents",
    "WaveField",
    "HullCfg",
    "HullGeometry",
    "HullModel",
    "ModelContext",
    "ZeroModel",
    "BodyState",
    "Wrench",
    "body_to_world",
    "quat_to_matrix",
    "world_to_body",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
