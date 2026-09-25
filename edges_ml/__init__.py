"""
edges_ml  –  Multi-code MHD simulation dataset builder for machine learning.

Public API
----------
    from edges_ml import build_dataset
    from edges_ml import M3DC1Adapter, NIMRODAdapter
    from edges_ml import SimulationAdapter
    from edges_ml import GridSpec
"""

from .utils import GridSpec
from .base import SimulationAdapter
from .edges_ml import build_dataset, group_simulation_directories

try:
    from .m3dc1_adapter import M3DC1Adapter
except Exception:
    M3DC1Adapter = None

try:
    from .nimrod_adapter import NIMRODAdapter
except Exception:
    NIMRODAdapter = None

__all__ = [
    "build_dataset",
    "group_simulation_directories",
    "SimulationAdapter",
    "M3DC1Adapter",
    "NIMRODAdapter",
    "GridSpec",
]
