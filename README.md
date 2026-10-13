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

## Simulation Browser

EDGES-ML ships with a generator for a **static HTML browser** that lets users explore the
simulation directories without touching the raw data. It is a manual, offline step: run it
whenever new simulations have been added, and it regenerates a single HTML file that can be
opened directly in a web browser or copied to a web server.

The generator lives in `edges_ml/browser.py` and uses **only the Python standard library**.
It does not need `fpy`, `nimpy`, `h5py`, `imas` or any other simulation-code dependency, so
it can be run on a login node or a laptop that merely has the data directories mounted.

### What the browser shows

- A collapsible **directory tree** that mirrors the on-disk layout (root directories →
  intermediate folders → model directories → simulations).
- A **search box** filtering by name, relative path, full path or physics-model summary,
  plus *Expand all* / *Collapse all* controls.
- For every entry a detail pane with three tabs:
  - **Overview** – metadata (code, path, last modification time, toroidal mode number,
    number of time slices / dumps, presence of `equilibrium.h5`, `energy.bin`, …), the
    derived **physical model** table, clickable child entries and the directory listing.
  - **Inputs** – the code **input namelists** (`C1input`, `nimrod.in`, `nimeq.in`,
    `fluxgrid.in`, `oculus.in`, `README`, the M3D-C1 growth-rate table, …). Each file can be
    viewed either as a filterable parameter table grouped by `&namelist`, or as raw text.
  - **Plots** – a gallery placeholder. Plot support is already wired up but inactive by
    default (see below).
- Selecting an entry updates the URL hash, so individual simulations can be bookmarked and
  shared as links.

The **physical model** is derived from the primary input file of each simulation:

| Code   | Parameters used                                                                 |
|--------|---------------------------------------------------------------------------------|
| M3D-C1 | `linear`, `numvar`, `itwo_fluid` / `db_fac`, `eqsubtract`, `icomplex`, `ntor`, …  |
| NIMROD | `nonlinear`, `ohms`, `continuity`, `separate_pe`, `p_model`, …                   |

### How simulations are discovered

The browser applies the same rules as `group_simulation_directories()`:

| Code   | A directory is a simulation if it contains |
|--------|--------------------------------------------|
| M3D-C1 | `C1.h5`                                    |
| NIMROD | `nimhist.bin` **and** `nimrod.in`, and the directory name starts with `nNN` or `nln` |

Any path containing a `base_*` component is skipped. The **parent** of a simulation
directory becomes the *model* node (its own input files are attached there as well), and all
directories above are rendered as plain folders, so the website reflects the directory
structure.

### Running it

From the repository root (the directory containing `make_browser.py` and the `edges_ml`
package):

```bash
# Crawl one root directory; both codes are searched
python make_browser.py /scratch/mastu_runs -o simulations.html

# Open the result
xdg-open simulations.html     # Linux
open simulations.html         # macOS
```

Three equivalent entry points are available:

```bash
# a) convenience launcher at the repository root (no installation required)
python make_browser.py /scratch/mastu_runs -o simulations.html

# b) as a module (package installed, or repository root on PYTHONPATH)
python -m edges_ml.browser /scratch/mastu_runs -o simulations.html

# c) directly as a file - bypasses edges_ml/__init__.py and all heavy imports
python edges_ml/browser.py /scratch/mastu_runs -o simulations.html
```

Selecting what to crawl:

```bash
# Several roots at once (positional arguments are searched for both codes)
python make_browser.py /scratch/runs_2023 /scratch/runs_2024 -o db.html

# Restrict a root to a single code (faster and avoids false positives)
python make_browser.py --m3dc1 /data/m3dc1 --nimrod /data/nimrod

# Mix and match
python make_browser.py /scratch/shared --m3dc1 /home/me/private_runs
```

A typical run prints a summary such as:

```
edges-ml browser: 1 root directory  •  12 model(s)  •  37 simulation(s)  •  25 M3D-C1  •  12 NIMROD
Wrote /path/to/simulations.html
```

### Command-line options

| Option                 | Default              | Description                                                        |
|------------------------|----------------------|--------------------------------------------------------------------|
| `DIR [DIR ...]`        | –                    | Root directories to crawl; both codes are searched                 |
| `--m3dc1 DIR`          | –                    | Root directory crawled for M3D-C1 runs only (repeatable)           |
| `--nimrod DIR`         | –                    | Root directory crawled for NIMROD runs only (repeatable)           |
| `-o, --output FILE`    | `simulations.html`   | Output HTML file                                                   |
| `-t, --title TEXT`     | `edges-ml simulation browser` | Page title                                                |
| `--scan-plots`         | off                  | Link image files found next to the simulations                     |
| `--max-doc-bytes N`    | `262144`             | Maximum characters embedded per input file                         |
| `--max-files N`        | `400`                | Maximum entries in the per-directory file listing                  |
| `--json FILE`          | –                    | Also dump the collected data model as JSON                         |
| `--open`               | off                  | Open the generated page in a web browser                           |
| `-q, --quiet`          | off                  | Suppress progress output                                           |

Run `python make_browser.py --help` for the full list.

### Python API

```python
from edges_ml import build_browser

build_browser(
    [{"code": "m3dc1",  "directories": ["/data/m3dc1"]},
     {"code": "nimrod", "directories": ["/data/nimrod"]}],
    output="simulations.html",
    title="MAST-U pedestal database",
)
```

`build_browser()` accepts the same source dictionaries as `build_dataset()`, plain directory
paths (both codes are then searched), or `(code, directory)` tuples. Related helpers:

```python
from edges_ml import discover_simulations, describe_simulation, parse_namelist
```

### Plots

The *Plots* tab is present but empty by default and shows a placeholder note. Pass
`--scan-plots` to link image files (`*.png`, `*.jpg`, `*.svg`, `*.gif`, `*.webp`, `*.pdf`)
found in the simulation directory itself or in a `plots/`, `figures/`, `figs/` or `images/`
subdirectory:

```bash
python make_browser.py /scratch/mastu_runs --scan-plots -o simulations.html
```

Note that plot images are linked **relative to the output HTML file**. When using
`--scan-plots`, write the HTML somewhere from which those relative paths still resolve
(for example inside or next to the data tree). Without `--scan-plots` the generated file is
fully self-contained and can be copied or e-mailed anywhere.

### Troubleshooting

If the generator reports `warning: no simulations were found`, verify that the marker files
actually exist below the given path:

```bash
find /scratch/mastu_runs -name C1.h5 | head
find /scratch/mastu_runs -name nimhist.bin | head
```

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
4. Optionally teach the HTML browser about it by adding entries to `CODE_LABELS`,
   `SIM_MARKERS`, `DOC_PATTERNS` and `PRIMARY_INPUT` in `edges_ml/browser.py`, plus a
   `_describe_mycode()` helper for the physical-model summary.

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
