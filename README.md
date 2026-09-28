# edges_ml

**edges_ml** is a standalone Python package for building machine-learning-ready datasets
from MHD simulation codes. Currently it supports M3D-C1 and NIMROD, but its design allows
for other codes to be added in the future.

It acts as a neutral orchestration layer: it discovers simulation output directories,
extracts fields on spatial grids, flux averages, time traces, and mode eigenfunctions, and
writes them to HDF5 or ADIOS2 BP files in a consistent schema. It supports either a custom
data scheme that was developed specifically for extended-MHD codes and can be expanded on
demand, or it can make use of the IMAS schema as defined by the IMAS Data Dictionary.

---

## Supported Codes

| Code      | Adapter                          | Key dependency          |
|-----------|----------------------------------|-------------------------|
| M3D-C1    | `edges_ml.adapters.m3dc1`        | `fpy` (fusion-io)       |
| NIMROD    | `edges_ml.adapters.nimrod`       | `nimpy`                 |

---

## Installation

### Minimal (no simulation-code dependencies)
```bash
pip install edges-ml
```

### M3D-C1 users
```bash
pip install "edges-ml[m3dc1]"
```

### NIMROD users
```bash
pip install "edges-ml[nimrod]"
```

### Everything
```bash
pip install "edges-ml[all]"
```

---

## External Dependencies (not on PyPI)

| Dependency    | How to provide                                                                 |
|---------------|--------------------------------------------------------------------------------|
| `fpy`         | Install fusion-io with Python bindings; ensure `fpy` is importable.           |
| `nimpy`       | Install the nimpy package into your Python environment.                        |
| `nimrod2imas` | Clone from GitHub; set `config["imas"]["nimrod2imas_path"]` or `$NIMROD2IMAS_DIR`. |
| `imas`        | Install imas-python; set `config["imas"]["python"]` if in a separate env.     |

---

## Quick Start

### Python API

```python
from edges_ml import build_dataset

sources = [
    {
        "code": "m3dc1",
        "directories": ["/path/to/m3dc1/runs"],
    }
]

config = {
    "units": "SI",
    "fcoords": "pest",
    "flux_averages": ["p", "q", "te", "ne"],
    "2d_fields": ["p", "j"],
    "output_format": "reduced_h5",
}

build_dataset(sources, "/path/to/output", config)
```

### Command Line

```bash
edges-build-dataset --config configs/example_m3dc1.yaml
```

---

## Configuration Reference

See `configs/example_m3dc1.yaml` and `configs/example_nimrod.yaml` for annotated examples.

Key top-level config keys:

| Key                      | Default          | Description                                      |
|--------------------------|------------------|--------------------------------------------------|
| `units`                  | `"codeunits"`    | `"SI"` or `"codeunits"`                         |
| `fcoords`                | `"pest"`         | Flux coordinate system for flux averages         |
| `output_format`          | `"reduced_h5"`   | `"reduced_h5"`, `"reduced_bp"`, or `"imas_h5"`  |
| `flux_averages`          | `None`           | List of quantities, or `"all"`                   |
| `2d_fields`              | `None`           | List of quantities, or `"all"`                   |
| `3d_fields`              | `None`           | List of quantities, or `"all"`                   |
| `time_traces`            | `None`           | List of quantities, or `"all"`                   |
| `mode_2d_fields`         | `None`           | Perturbation 2D fields                           |
| `mode_3d_fields`         | `None`           | Perturbation 3D fields                           |
| `time_slices`            | `[]`             | List of integer time slice indices               |
| `resolutions`            | `{"1d": 200}`    | Grid resolution settings                         |
| `include_input_namelist` | `False`          | Whether to save code input parameters            |
| `shared_grid_source`     | `None`           | `"first"` or path to a model dir for shared grid |

---

## Adding a New Code

1. Create `edges_ml/adapters/mycode.py` subclassing `SimulationAdapter`.
2. Register it in `edges_ml/adapters/__init__.py`.
3. Add it to `group_simulation_directories()` in `edges_ml/discovery.py`.

---

## Output Schema (HDF5)

```
/metadata/
/metadata/file_hashes/<mode_dir>/
/inputs/                          (optional)
/global_parameters/
/time_traces/
/grids/
/grids/time/
/grids/1d_grid
/grids/2d_grid
/grids/3d_grid
/equilibrium/flux_averages/
/equilibrium/2d_fields/
/equilibrium/3d_fields/
/total_fields/flux_averages/      (if time_slices requested)
/total_fields/2d_fields/
/total_fields/3d_fields/
/perturbations/<nXX>/mode_information/
/perturbations/<nXX>/1d_profiles/
/perturbations/<nXX>/2d_fields/
/perturbations/<nXX>/3d_fields/
```
