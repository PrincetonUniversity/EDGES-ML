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
| `mode_1d_profiles`       | `None`           | Perturbation 1D profiles (e.g. `"q_profile"`)    |
| `mode_2d_fields`         | `None`           | Perturbation 2D fields                           |
| `mode_3d_fields`         | `None`           | Perturbation 3D fields                           |
| `mode_spectra`           | `None`           | Fields for poloidal mode spectra (e.g. `["p"]`); see below |
| `time_slices`            | `[]`             | List of integer time slice indices               |
| `resolutions`            | `{"1d": 200}`    | Grid resolution settings                         |
| `include_input_namelist` | `False`          | Whether to save code input parameters            |
| `shared_grid_source`     | `None`           | `"first"` or path to a model dir for shared grid |

---

## Mode Perturbations and Poloidal Spectra

### Perturbation entries

Each mode directory is expanded into one or more entries under `/perturbations/`:

| Code   | Entries                                    | How the perturbation is obtained |
|--------|--------------------------------------------|----------------------------------|
| M3D-C1 | one per `nXX` directory                    | `field(mode run, last time slice) − field(equilibrium)` |
| NIMROD | one per toroidal mode stored in the dump (`keff` of the last dump) | complex Fourier coefficient `c_n` of the perturbation, evaluated with `EvalNimrod.eval_comp_field()` |

For NIMROD this works the same way for single-n and multi-n runs. No full fields are
subtracted, so there is no cancellation against the equilibrium. The stored 2D/3D fields are
the physical perturbation of mode n, `2 Re[c_n e^{i n φ}]` (for n = 0: `Re[c_0]`), at the
toroidal angle of the output grid (φ = 0 for 2D grids).

Additional NIMROD options:

| Key                       | Default | Description |
|---------------------------|---------|-------------|
| `nimrod_include_n0_mode`  | `False` | Write an `n00` entry in multi-n runs (skipped by default) |
| `nimrod_store_complex`    | `False` | Also write `<field>_re` / `<field>_im` datasets with the complex coefficient |

If an entry name is already taken (for example the same n in several directories), the
later entry is written as `<dir>_nXX` instead of being skipped, and a warning is printed.

### Poloidal spectra

Setting `mode_spectra` (currently only `["p"]` is supported) adds the poloidal Fourier
spectrum of each mode's eigenfunction, computed the same way as M3D-C1's
`eigenfunction(fourier=True)`:

1. Build a flux-coordinate grid of the **equilibrium** with `spectrum_points` poloidal and
   radial points: uniform in normalised poloidal flux ψ_N ∈ [1e-4, 1] and uniform in the
   poloidal angle θ.
2. Evaluate the eigenfunction on that grid at constant geometric toroidal angle
   φ = `spectrum_phi`.
3. Compute `amplitude = |rfft(eigenfunction, axis=θ, norm='forward')|` for every surface.

**The spectrum depends on the choice of poloidal angle.** Use a flux-aligned
(straight-field-line) coordinate system:

| `spectrum_fcoords`   | Poloidal angle |
|----------------------|----------------|
| `"pest"` (default)   | PEST |
| `"boozer"`           | Boozer |
| `"hamada"`           | Hamada |
| `"canonical"`        | identical to PEST after normalisation |
| `""` / `"geometric"` | geometric angle — **fallback for special cases only**, a warning is printed |

Conventions shared by both codes:

- θ = 0 lies on the outboard midplane, and θ increases clockwise in the (R, Z) plane, as in M3D-C1.
- φ follows the M3D-C1 direction. NIMROD coefficients are complex-conjugated accordingly.
- `m = 0 … n_θ/2` is the poloidal mode number. Only the magnitude is stored, so the sign of m
  (helicity) is not resolved.

**M3D-C1** calls `m3dc1.flux_coordinates.flux_coordinates()`. The flux coordinates used for
the flux averages are not affected.

**NIMROD** has no poloidal flux function. `edges_ml/nimrod_fluxcoords.py` therefore
reconstructs ψ by integrating the equilibrium magnetic field along rays from the magnetic
axis, then builds the straight-field-line angle with a faithful port of the M3D-C1 algorithm.
ψ at the LCFS is taken as the smallest of these candidates:

- the separatrix estimate from the ray integration;
- the flux at X-points found with `nimpy.fsa.find_pf_null`, starting from an automatic search
  and from `nimrod_xpoint_guess`.

`nimrod_psi_lcfs` overrides the result.

### Spectrum configuration keys

| Key                         | Default                     | Description |
|-----------------------------|-----------------------------|-------------|
| `mode_spectra`              | off                         | Fields to compute spectra for, e.g. `["p"]` |
| `spectrum_points`           | `200`                       | Number of θ **and** ψ points (like `points` in `eigenfunction()`) |
| `spectrum_fcoords`          | `fcoords`, else `"pest"`    | Flux coordinate system (table above) |
| `spectrum_phi`              | `0.0`                       | Toroidal angle at which the eigenfunction is evaluated |
| `spectrum_store_fluxgrid`   | `False`                     | Also store the eigenfunction and the (R, Z) flux grid |
| `spectrum_include_omega`    | `False`                     | NIMROD only: for Boozer/Hamada evaluate at constant toroidal flux-coordinate angle ζ instead of geometric φ |
| `spectrum_theta_oversample` | `1`                         | NIMROD only: θ oversampling of the flux-coordinate construction (1 = M3D-C1 algorithm) |
| `nimrod_psi_ray_samples`    | `4 × spectrum_points`       | NIMROD only: samples per ray for the ψ integration |
| `nimrod_xpoint_guess`       | `None`                      | NIMROD only: `[R, Z]` or list of `[R, Z]` X-point guesses |
| `nimrod_psi_lcfs`           | `None`                      | NIMROD only: override for ψ_LCFS − ψ_axis (Wb/rad, magnitude used) |

### Spectrum output

Datasets in `/perturbations/<entry>/spectra/<field>/`:

| Dataset         | Shape          | Description |
|-----------------|----------------|-------------|
| `psi_norm`      | `[n_ψ]`        | Normalised poloidal flux of the surfaces |
| `m`             | `[n_m]`        | Poloidal mode numbers |
| `amplitude`     | `[n_m, n_ψ]`   | Spectrum amplitude |
| `q`             | `[n_ψ]`        | Safety factor on the same surfaces (if available) |
| `eigenfunction` | `[n_θ, n_ψ]`   | Eigenfunction on the flux grid (`spectrum_store_fluxgrid`) |
| `R`, `Z`        | `[n_θ, n_ψ]`   | Flux grid coordinates (`spectrum_store_fluxgrid`) |
| `theta`         | `[n_θ]`        | Poloidal angle (`spectrum_store_fluxgrid`) |
| `omega`         | `[n_θ, n_ψ]`   | NIMROD only: ζ − φ for Boozer/Hamada (`spectrum_store_fluxgrid` and `spectrum_include_omega`) |

Group attributes: `code`, `field`, `n`, `fcoords`, `phi`, `toroidal_angle`, `points`,
`normalization`, `theta_convention`. NIMROD additionally stores `psi_lcfs`, `psi_lcfs_source`,
`n_clipped` and `n_unreached`, which help check the quality of the flux grid.

### Known limitations

- Only scalar fields are supported for spectra; vector fields (B, v, …) are planned.
- NIMROD: the mode axis returned by `eval_comp_field()` is assumed to follow the order of
  `keff` in the dump. A warning is printed if the mode counts disagree.
- NIMROD: the existing FSA-based flux averages and global parameters still normalise ψ_N to
  the domain boundary in diverted cases. The spectrum grid uses the separatrix.
- M3D-C1: `spectrum_include_omega` is not implemented. Spectra are evaluated at constant
  geometric φ, as in `eigenfunction()`.
- M3D-C1 campaign-archive mode: spectra depend on `flux_coordinates()` working with the
  streamed data. If it fails, spectra are skipped for that model with an error message.

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
/perturbations/<nXX>/spectra/<field>/   (if mode_spectra requested)
```

Perturbation entries are named `nXX`; on a name collision `<dir>_nXX` is used.

For information on the IMAS schema please see https://imas-data-dictionary.readthedocs.io/en/latest/reference_ids.html
