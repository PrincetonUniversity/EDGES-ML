# EDGES-ML

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

| Code      | Adapter class                              | Dependencies                                |
|-----------|--------------------------------------------|---------------------------------------------|
| M3D-C1    | `edges_ml.m3dc1_adapter.M3DC1Adapter`      | `fpy` (fusion-io), `m3dc1` (Python package) |
| NIMROD    | `edges_ml.nimrod_adapter.NIMRODAdapter`    | `nimpy`, `nimrod2imas`                      |

---

## Installation

### Minimal (no simulation-code dependencies)
```bash
pip install edges_ml
```

### M3D-C1 users
```bash
pip install "edges_ml[m3dc1]"
```

### NIMROD users
```bash
pip install "edges_ml[nimrod]"
```

### Everything
```bash
pip install "edges_ml[all]"
```

---

## External Dependencies (not on PyPI)

Some functionality requires external packages that are not available on PyPI and must be
installed separately before use.

### `fpy` (fusion-io)

Install fusion-io with Python bindings and ensure that `fpy.py` is importable from your
Python environment (`import fpy`). The fusion-io source code is available at:

- https://github.com/nferraro/fusion-io

### `nimpy`

Install the nimpy package into your Python environment. The nimpy package is available at:

- Source: https://gitlab.com/NIMRODteam/open/nimpy/
- Documentation (including installation instructions): https://nimrodteam.gitlab.io/open/nimpy/index.html

### `nimrod2imas`

Install the nimrod2imas package into your Python environment. Like nimpy it can be installed as a
regular Python package from its source repository using `pip install -e`:

- Source: https://github.com/PrincetonUniversity/nimrod2imas

### `imas`

Install the imas package into your Python environment. This package can be readily
installed via pip:

```bash
pip install imas-python
```
Documentation is available at https://imas-python.readthedocs.io/en/stable/index.html

---

## Quick Start

After installing EDGES-ML and its dependencies, running EDGES-ML to export data in an ML-ready format
is rather straightforward:

The user defines a `config` dictionary to select the data to be exported, and the runs
`build_dataset()`. Please see the examples in the `examples` directory on how to export
data from M3D-C1 and NIMROD in the supported output formats.

---

## Configuration Reference

See files in the `examples` directory for reference.

Top-level config keys:

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

## Output Schema

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

For information on the IMAS schema please see https://imas-data-dictionary.readthedocs.io/en/latest/reference_ids.html
