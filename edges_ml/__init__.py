"""
edges_ml  –  Multi-code MHD simulation dataset builder for machine learning.

Public API
----------
    from edges_ml import build_dataset
    from edges_ml import M3DC1Adapter, NIMRODAdapter
    from edges_ml import SimulationAdapter
    from edges_ml import GridSpec
"""

__version__ = "0.1.0"

from .utils import GridSpec
from .base import SimulationAdapter
from .edges_ml import build_dataset, group_simulation_directories

# Optional, dependency-heavy adapters.  They are allowed to be unavailable
# (missing fpy / nimpy / imas ...), but we keep the original exception around
# so that a genuine bug inside an adapter is not silently hidden.
_ADAPTER_IMPORT_ERRORS = {}

try:
    from .m3dc1_adapter import M3DC1Adapter
except Exception as exc:  # pragma: no cover - depends on local environment
    M3DC1Adapter = None
    _ADAPTER_IMPORT_ERRORS["M3DC1Adapter"] = exc

try:
    from .nimrod_adapter import NIMRODAdapter
except Exception as exc:  # pragma: no cover - depends on local environment
    NIMRODAdapter = None
    _ADAPTER_IMPORT_ERRORS["NIMRODAdapter"] = exc


def adapter_import_error(name):
    """Return the exception that prevented `name` from being imported, or None.

    Example
    -------
        >>> import edges_ml
        >>> edges_ml.M3DC1Adapter is None
        True
        >>> edges_ml.adapter_import_error("M3DC1Adapter")
        ModuleNotFoundError("No module named 'fpy'")
    """
    return _ADAPTER_IMPORT_ERRORS.get(name)


def available_adapters():
    """Return a dict of the adapter classes that imported successfully."""
    return {
        name: cls
        for name, cls in (("M3DC1Adapter", M3DC1Adapter),
                          ("NIMRODAdapter", NIMRODAdapter))
        if cls is not None
    }


__all__ = [
    "__version__",
    "build_dataset",
    "group_simulation_directories",
    "SimulationAdapter",
    "M3DC1Adapter",
    "NIMRODAdapter",
    "GridSpec",
    "adapter_import_error",
    "available_adapters",
]
