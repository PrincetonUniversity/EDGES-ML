"""
edges-ml  –  Multi-code MHD simulation dataset builder for machine learning.

Public API
----------
    from edges_ml import build_dataset
    from edges_ml import M3DC1Adapter, NIMRODAdapter
    from edges_ml import SimulationAdapter
"""

from .edges_ml import build_dataset, group_simulation_directories
from .base import SimulationAdapter
from .m3dc1_adapter import M3DC1Adapter
from .nimrod_adapter import NIMRODAdapter

__all__ = [
    "build_dataset",
    "group_simulation_directories",
    "SimulationAdapter",
    "M3DC1Adapter",
    "NIMRODAdapter",
]
