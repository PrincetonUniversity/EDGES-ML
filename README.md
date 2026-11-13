# EDGES-ML

**edges_ml** is a standalone Python package for building machine-learning-ready datasets
from MHD simulation codes. Currently it supports M3D-C1 and NIMROD, but its design allows
for other codes to be added in the future.

It acts as a neutral orchestration layer: it discovers simulation output directories,
extracts fields on spatial grids, flux averages, time traces, and mode eigenfunctions, and
writes them to HDF5 or ADIOS2 BP files in a consistent schema. It supports either a custom
data scheme that was developed specifically for extended-MHD codes and can be expanded on
demand, or it can make use of the IMAS schema as defined by the IMAS Data Dictionary.

Design decisions, conventions and the change history are described in
[`docs/design_notes.md`](docs/design_notes.md).

---

## Supported Codes

| Code      | Adapter class                              | Dependencies                                |
|-----------|--------------------------------------------|---------------------------------------------|
| M3D-C1    | `edges_ml.m3dc1_adapter.M3DC1Adapter`      | `fpy` (fusion-io), `m3dc1` (Python package) |
| NIMROD    | `edges_ml.nimrod_adapter.NIMRODAdapter`    | `nimpy`, `nimrod2imas`                      |

M3D-C1 data can be read from a local file system or streamed from a remote host through an
ADIOS2 campaign archive (`.aca`); see `examples/example_m3dc1_campaign_aca.py`.

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

You define a `config` dictionary that selects the data to export and a list of data
sources, then call `build_dataset()`:

```python
import edges_ml

config = {
    "units": "mks",
    "fcoords": "pest",
    "time_slices": [2],
    "resolutions": {"1d": 200},
    "time_traces": ["all"],

    "equilibrium":   {"flux_averages": "all", "2d_fields": ["te", "ne"], "3d_fields": []},
    "total_fields":  {"flux_averages": [],    "2d_fields": ["te", "ne"], "3d_fields": []},
    "perturbations": {"flux_averages": [],    "2d_fields": ["p"],        "3d_fields": [],
                      "spectrum": {"fcoords": "pest", "points": 200}},

    "2d_grid": edges_ml.GridSpec("inside_wall", None),
    "3d_grid": edges_ml.GridSpec("rectangular", (64, 16, 64)),
    "output_format": "reduced_h5",
}

sources = [{"code": "m3dc1", "directories": ["runs/mastu_45272"], "model": None}]

edges_ml.build_dataset(sources=sources, output_directory="./datasets", config=config)
```

Complete, commented configurations for every code and output format are in the
`examples` directory:

| Example                                  | Code   | Output format |
|------------------------------------------|--------|---------------|
| `example_m3dc1_reduced_h5.py`            | M3D-C1 | `reduced_h5`  |
| `example_m3dc1_reduced_bp.py`            | M3D-C1 | `reduced_bp`  |
| `example_m3dc1_imas.py`                  | M3D-C1 | `imas_h5`     |
| `example_m3dc1_campaign_aca.py`          | M3D-C1 (remote, `.aca`) | `reduced_h5` |
| `example_nimrod_reduced_h5.py`           | NIMROD | `reduced_h5`  |
| `example_nimrod_imas.py`                 | NIMROD | `imas_h5`     |
| `run_mastu_example.py`                   | M3D-C1 | `reduced_h5` (same as `example_m3dc1_reduced_h5.py`) |
| `plot_perturbation.py`                   | –      | plots a `reduced_h5` file |

At start-up `build_dataset()` prints a summary of the resolved export selection
("Export selection: ..."), including the quantities for which a poloidal spectrum will be
computed. Check it to confirm that the configuration was understood as intended.

---

## Configuration Reference

### Top-level keys

| Key                         | Default                              | Description |
|-----------------------------|--------------------------------------|-------------|
| `units`                     | `"codeunits"`                        | `"mks"`, `"cgs"` or `"codeunits"`. M3D-C1: passed to the flux averages, time traces and spectra. NIMROD data are always SI; the setting only affects the `unit` attributes written to the file. |
| `fcoords`                   | `"pest"`                             | Flux coordinate system of the flux averages (M3D-C1) and default for the spectrum. NIMROD: a value containing `"rho"` makes ρ the 1D grid coordinate, otherwise ψ_N is used. |
| `time_slices`               | `[]`                                 | Time slices for the `total_fields` section (see *Time slices* below). |
| `resolutions`               | `{"1d": 200}`                        | Number of points of the 1D (flux-average) grid. |
| `time_traces`               | `None`                               | List of scalar time traces, or `"all"`. |
| `equilibrium`               | –                                    | Export section, see below. |
| `total_fields`              | –                                    | Export section, see below. |
| `perturbations`             | –                                    | Export section, see below. |
| `2d_grid`                   | `GridSpec("rectangular", (128, 128))`| Grid for **all** 2D fields of every section. |
| `3d_grid`                   | `GridSpec("rectangular", (64, 16, 64))` | Grid for **all** 3D fields of every section (`(nR, nPhi, nZ)`). |
| `shared_grid_source`        | `None`                               | `"first"`: the inner-wall points of the first M3D-C1 model are used for the `inside_wall` grid of **all** models (also NIMROD). A path to an M3D-C1 model directory uses that model. `None`: every model uses its own grid. |
| `include_input_namelist`    | `False`                              | Write the code input parameters to `/inputs`. |
| `output_format`             | `"reduced_h5"`                       | `"reduced_h5"`, `"reduced_bp"` (ADIOS2) or `"imas_h5"`. |
| `m3dc1_imas_conversion_lib` | `"standalone"`                       | M3D-C1 IMAS converter: `"standalone"` or `"imas-python"`. |
| `imas`                      | `{}`                                 | Options of the IMAS conversion (`dd`, `pulse`, `run`, `dbpath`, `backend`, `dd_version`, `nimrod2imas_path`, `python`, `setup_cmd`, `run_mode`, ...). |

### Export sections

The quantities to export are configured in three sections. Each section has the same three
lists:

```python
"equilibrium":   {"flux_averages": [...], "2d_fields": [...], "3d_fields": [...]},
"total_fields":  {"flux_averages": [...], "2d_fields": [...], "3d_fields": [...]},
"perturbations": {"flux_averages": [...], "2d_fields": [...], "3d_fields": [...],
                  "spectrum": {...}},
```

| Section         | Meaning |
|-----------------|---------|
| `equilibrium`   | The equilibrium (time = −1). |
| `total_fields`  | The full fields at every entry of `time_slices`; arrays get a leading time axis `[N_t, ...]`. |
| `perturbations` | The mode structure of every mode directory / toroidal mode (see *Mode perturbations*). |

Each list is a list of quantity names or `"all"`. `"q_profile"` is accepted as an alias
for `"q"`. Unknown names are passed to the adapter with a warning; quantities a code cannot
provide are skipped with a warning.

**A poloidal spectrum is computed for every scalar field listed under `perturbations`**
(in any of the three lists). There is no separate list of spectrum quantities. Vector
fields (`B`, `v`, `j`, `E`, ...) and flux-surface-only quantities (`q`, `rho`, `V`, ...)
get no spectrum; this is reported in the start-up summary.

#### Available quantity names

| Kind                     | Names |
|--------------------------|-------|
| Fields (2D, 3D, flux averages) | `j`, `ni`, `ne`, `v`, `B`, `p`, `pi`, `pe`, `ti`, `te`, `A`, `gradA`, `E`, `alpha`, `eta`, `eta_spitzer`, `shear`, `psi`, `ne/ng`, `S`, `lundquist` |
| Flux averages only       | `q`, `rho`, `flux_t`, `flux_p`, `Ip`, `V`, `f`, `ffprime` |

Vector fields are written as `<name>_R`, `<name>_phi`, `<name>_Z`.

Not every code provides every quantity:

| Code   | 2D / 3D fields | Flux averages |
|--------|----------------|---------------|
| M3D-C1 | everything `m3dc1.eval_field()` supports | everything `m3dc1.flux_average()` supports |
| NIMROD | `p`, `pe`, `pi`, `te`, `ti`, `ne`, `ni`, `psi`, `B`, `v`, `j`*, `E` | `p`, `pe`, `pi`, `ne`, `ni`, `te`, `ti`, `psi`, `B`, `v`, `j`*, `f`, `ffprime`, `Ip`, `V`, `q`, `rho`, `flux_p`, `flux_t` |

\* NIMROD: `j` requires `dump_ja=T` in the run (see *NIMROD notes*).

### Spectrum settings

Numerical settings of the poloidal spectra, in `perturbations["spectrum"]`:

| Key                | Default                  | Description |
|--------------------|--------------------------|-------------|
| `fcoords`          | top-level `fcoords`, else `"pest"` | Flux coordinate system (see below). |
| `points`           | `200`                    | Number of θ **and** ψ points (like `points` in `eigenfunction()`). |
| `phi`              | `0.0`                    | Toroidal angle at which the eigenfunction is evaluated. |
| `store_fluxgrid`   | `False`                  | Also store the eigenfunction and the (R, Z) flux grid. |
| `include_omega`    | `False`                  | NIMROD only: for Boozer/Hamada evaluate at constant toroidal flux-coordinate angle ζ instead of geometric φ. |
| `theta_oversample` | `1`                      | NIMROD only: θ oversampling of the flux-coordinate construction (1 = M3D-C1 algorithm). |

The old flat keys `spectrum_<name>` (e.g. `spectrum_points`) are still read as a fallback.

### Grid specifications

`edges_ml.GridSpec(type, resolution)`:

| `type`         | `resolution`                       | Notes |
|----------------|------------------------------------|-------|
| `"rectangular"`| `(nR, nZ)` for 2D, `(nR, nPhi, nZ)` for 3D | Regular grid over the bounding box of the domain (M3D-C1: inner-wall points; NIMROD: mesh nodes). |
| `"inside_wall"`| `None`                             | Scattered points inside the wall (M3D-C1: `get_points_inside_inner_wall`; NIMROD: mesh nodes inside `contours.h5` / `sol.grn`). 3D: φ = 0 only. |
| `"native"`     | `None`                             | NIMROD: finite-element nodes. **M3D-C1: not implemented** (placeholder that returns zeros). |

All 2D (3D) fields of all three sections are evaluated on the same `2d_grid` (`3d_grid`),
which is written once to `/grids/2d_grid` (`/grids/3d_grid`).

### NIMROD-specific keys

| Key                       | Default            | Description |
|---------------------------|--------------------|-------------|
| `nimrod_include_n0_mode`  | `False`            | Write an `n00` perturbation entry in multi-n runs. |
| `nimrod_store_complex`    | `False`            | Also write `<field>_re` / `<field>_im` with the complex Fourier coefficient. |
| `nimrod_dump_ja`          | from `nimrod.in`   | Override the detection of `dump_ja` (current density in the dumps). |
| `nimrod_eval_e`           | `False`            | Add `e` to the nimpy field list even if `E` is not requested. |
| `nimrod_safe_probe`       | `True`             | Probe every nimpy field once in a forked child process (crash protection). |
| `nimrod_probe_timeout`    | `600`              | Timeout of one probe in seconds; on timeout probing is switched off. |
| `nimrod_verbose`          | `False`            | Print a marker before every nimpy stage (locates remaining crashes). |
| `nimrod_use_eval_grid`    | `True`             | Reuse nimpy `EvalGrid` logical mappings for repeated point sets. |
| `nimrod_enable_fsa`       | `True`             | Enable the flux-surface averages (`nimpy.fsa.FSA`). |
| `nimrod_fsa_nsurf`        | `min(resolutions["1d"], 60)` | Number of FSA surfaces (interpolated onto the 1D grid afterwards). |
| `nimrod_compute_globals`  | `True`             | Compute the global parameters. |
| `nimrod_psi_ray_samples`  | `4 × spectrum points` (at least 50) | Samples per ray for the ψ reconstruction of the spectrum grid. |
| `nimrod_xpoint_guess`     | `None`             | `[R, Z]` or list of `[R, Z]` X-point guesses. |
| `nimrod_psi_lcfs`         | `None`             | Override for ψ_LCFS − ψ_axis (Wb/rad, magnitude used). |

### Legacy (flat) keys

Configurations written for older versions still run, with a deprecation warning. The
flat keys are mapped onto the sections as follows (if the section is given, the flat key
is ignored):

| Old key               | New location                     |
|-----------------------|----------------------------------|
| `flux_averages`       | `equilibrium["flux_averages"]`   |
| `2d_fields`           | `equilibrium["2d_fields"]`       |
| `3d_fields`           | `equilibrium["3d_fields"]`       |
| `total_flux_averages` | `total_fields["flux_averages"]`  |
| `total_2d_fields`     | `total_fields["2d_fields"]`      |
| `total_3d_fields`     | `total_fields["3d_fields"]`      |
| `mode_1d_profiles`    | `perturbations["flux_averages"]` |
| `mode_2d_fields`      | `perturbations["2d_fields"]`     |
| `mode_3d_fields`      | `perturbations["3d_fields"]`     |
| `spectrum_<name>`     | `perturbations["spectrum"][<name>]` |

The keys `mode_spectra`, `perturbation_2d_grid` and `perturbation_3d_grid` have no effect
any more (a warning is printed).

### Time slices

| Code   | `time = -1`               | Positive `time` | Mode simulations |
|--------|---------------------------|-----------------|------------------|
| M3D-C1 | `equilibrium.h5`          | `time_NNN.h5`   | last time slice (`time='last'`) |
| NIMROD | dump index 0 (step 0, or the first all-zero dump) | sequential dump index 1, 2, ... (sorted by step). If an index does not exist in all `nXX` directories, the next smaller common index is used (with a warning). | last dump of the directory |

For NIMROD, the equilibrium and the time slices are read from the **first** `nXX`
directory of a model.

---

## Output Schema (`reduced_h5` / `reduced_bp`)

```
/metadata/                               code, source, units, fcoords, output_format,
                                         perturbation_quantities, spectrum_quantities, ...
/metadata/file_hashes/<mode_dir>/        SHA-256 of the input files (campaign mode: dataset UUIDs)
/inputs/                                 (include_input_namelist)
/global_parameters/
/time_traces/
/grids/time/                             time_slice, simulation_time_step, simulation_time
/grids/1d_grid
/grids/2d_grid
/grids/3d_grid
/equilibrium/flux_averages/
/equilibrium/2d_fields/
/equilibrium/3d_fields/
/total_fields/flux_averages/             (if time_slices) arrays [N_t, ...]
/total_fields/2d_fields/
/total_fields/3d_fields/
/perturbations/<entry>/mode_information/ growth_rate, frequency, mode_type
/perturbations/<entry>/flux_averages/
/perturbations/<entry>/2d_fields/
/perturbations/<entry>/3d_fields/
/perturbations/<entry>/spectra/<field>/  (scalar fields listed under 'perturbations')
```

Datasets carry `unit` and `description` attributes. With `units = "codeunits"` the `unit`
attribute is currently a placeholder (`dummy_conversion_formula`).

Perturbation entries are named `nXX`; on a name collision (for example the same n in
several directories) `<dir>_nXX` is used and a warning is printed.

A total-field flux average that is not available for any time slice is skipped (with a
warning). If it is missing only for some time slices, those rows are filled with NaN.

For information on the IMAS schema please see
https://imas-data-dictionary.readthedocs.io/en/latest/reference_ids.html.
For `output_format = "imas_h5"` the IMAS converters (`m3dc1.convert2imas*`,
`nimrod2imas`) decide which quantities are written; the export sections have no effect.

---

## Mode Perturbations and Poloidal Spectra

### Perturbation entries

Each mode directory is expanded into one or more entries under `/perturbations/`:

| Code   | Entries                                    | 2D / 3D perturbation | Flux-average perturbation |
|--------|--------------------------------------------|----------------------|---------------------------|
| M3D-C1 | one per `nXX` directory                    | `field(mode run, last time slice) − field(equilibrium)` | `⟨field⟩(mode run) − ⟨field⟩(equilibrium)`, both on the equilibrium flux surfaces |
| NIMROD | one per toroidal mode in the last dump (`keff`) | `2 Re[c_n e^{i n φ}]` (n = 0: `Re[c_0]`) from `EvalNimrod.eval_comp_field()` | `⟨field⟩(last dump, eq + n=0) − ⟨field⟩(equilibrium)`; identical for all entries of a directory, because n > 0 components average to zero |

For NIMROD this works the same way for single-n and multi-n runs. No full fields are
subtracted, so there is no cancellation against the equilibrium. φ is the toroidal angle of
the output grid (φ = 0 for 2D grids).

### Poloidal spectra

For every scalar perturbation quantity, the poloidal Fourier spectrum of each mode is
computed the same way as M3D-C1's `eigenfunction(fourier=True)`:

1. Build a flux-coordinate grid of the **equilibrium** with `points` poloidal and radial
   points: uniform in normalised poloidal flux ψ_N ∈ [1e-4, 1] and uniform in the
   poloidal angle θ.
2. Evaluate the eigenfunction on that grid at constant geometric toroidal angle φ = `phi`.
3. Compute `amplitude = |rfft(eigenfunction, axis=θ, norm='forward')|` for every surface.

**The spectrum depends on the choice of poloidal angle.** Use a flux-aligned
(straight-field-line) coordinate system:

| `fcoords`            | Poloidal angle |
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

`nimrod_psi_lcfs` overrides the result. The heuristics used for the X-point search are
listed in [`docs/design_notes.md`](docs/design_notes.md) (section 4.6).

### Spectrum output

Datasets in `/perturbations/<entry>/spectra/<field>/`:

| Dataset         | Shape          | Description |
|-----------------|----------------|-------------|
| `psi_norm`      | `[n_ψ]`        | Normalised poloidal flux of the surfaces |
| `m`             | `[n_m]`        | Poloidal mode numbers |
| `amplitude`     | `[n_m, n_ψ]`   | Spectrum amplitude |
| `eigenfunction` | `[n_θ, n_ψ]`   | Eigenfunction on the flux grid (`store_fluxgrid`) |
| `R`, `Z`        | `[n_θ, n_ψ]`   | Flux grid coordinates (`store_fluxgrid`) |
| `theta`         | `[n_θ]`        | Poloidal angle (`store_fluxgrid`) |
| `omega`         | `[n_θ, n_ψ]`   | NIMROD only: ζ − φ for Boozer/Hamada (`store_fluxgrid`) |

The safety factor is **not** part of the spectrum group; use the equilibrium flux average
`q` on `/grids/1d_grid`.

Group attributes: `code`, `field`, `n`, `fcoords`, `phi`, `toroidal_angle`, `points`,
`normalization`, `theta_convention`. NIMROD additionally stores `psi_lcfs`, `psi_lcfs_source`,
`n_clipped` and `n_unreached`, which help check the quality of the flux grid.

### Plotting

`examples/plot_perturbation.py` plots a perturbation in the R–Z plane and its spectrum
(similar to `eigenfunction.py`). It only needs `h5py`, `numpy` and `matplotlib`:

```bash
python examples/plot_perturbation.py datasets/file.h5 --list
python examples/plot_perturbation.py datasets/file.h5 --entry n05 --field p --show-res
python examples/plot_perturbation.py datasets/file.h5 --norm --fluxgrid --save plots/
```

`--fluxgrid` needs `store_fluxgrid = True`; `--show-res` needs `q` in the equilibrium flux
averages.

---

## NIMROD Notes

### Dump files

edges-ml reads the HDF5 dumps `dumpgll.NNNNN.h5`. Binary dumps without an `.h5`
counterpart are converted automatically with `dump2h5`. **This step modifies `nimrod.in`
in place** (`nimuw_dump_compat=T`, `h5dump=T` in `&output_input`) and copies the `dump2h5`
executable from a fixed installation path on Perlmutter into the run directory.

### Field selectors

nimpy's `eq` argument is used as follows:

| `eq` | Field                              | Used for |
|------|------------------------------------|----------|
| 1    | equilibrium only                   | equilibrium flux averages, global parameters, spectrum flux grid |
| 2    | equilibrium + full perturbation    | 2D / 3D equilibrium and total fields |
| 3    | equilibrium + n = 0 perturbation   | total-field flux averages, perturbation flux averages |

### Current density (`dump_ja`)

The current density `j` is only in the dumps if the run used `dump_ja=T` in
`&output_input`. Evaluating `j` otherwise makes nimpy stop the whole Python process
(`NIM_STOP => Evaluation of j requires dump_ja=T`), which cannot be caught. edges-ml
therefore reads `dump_ja` from `nimrod.in` (override: `nimrod_dump_ja`) and, if it is
`F`, skips the quantities that evaluate `j` directly (the `j` field and its flux average)
with a warning.

The flux-surface averages themselves do not need `j` (field-line tracing only needs `B`).
The plasma current is computed from **B** with Ampère's law,
`I(ψ) = (1/μ0) ∮ B_pol · dl` along each flux-surface contour, so `Ip` (flux average and
global parameter) and `beta_n` are available without `dump_ja=T`.

### Crash-safe probing

Some nimpy fields can crash the Fortran library (segmentation fault or `NIM_STOP`) for a
given run. edges-ml first evaluates every field list and every field **once in a forked
child process** at a mesh node of the dump. Fields whose probe crashes or fails are not
used, and every quantity that needs them is skipped with a warning naming the field and the
signal. Probing is enabled by default (`nimrod_safe_probe`), requires `os.fork` (Linux /
macOS) and is cached per dump.

If a crash still happens, set `nimrod_verbose = True`; the last `NIMROD [stage]` line before
the crash shows the failing step.

### Global parameters

`R_axis`, `Z_axis`, `R_geo`, `a_minor`, `aspect_ratio`, `kappa`, `delta(_upper/_lower)`,
`q0`, `qa`, `q_min`, `q_max`, `q95`, `psi_axis`, `psi_lcfs`, `poloidal_flux`,
`toroidal_flux`, `volume`, `Ip`, `B0`, `p0`, `ne0`, `ni0`, `Te0`, `Ti0`, `p_average`,
`beta_t`, `beta_n`, `beta_p`, `li`. The shaping parameters use the outermost FSA surface.

### Time traces

Mapped from `discharge.bin` / `energy.bin`: `time`, `E_P`, `E_PE`, `toroidal_current`,
`loop_voltage`, `toroidal_flux`, `W_M` (summed over toroidal modes). Other names are
skipped with a warning.

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

- A collapsible **directory tree** that mirrors the on-disk layout: root directories →
  intermediate folders → **cases** (the parent directory of the simulations, i.e. the
  "model" directory of `build_dataset()`) → simulations. Double-click a row, or click its
  arrow or icon, to expand or collapse it.
- A **search box** filtering by name, path or physics-model summary, plus *Expand all* /
  *Collapse all* controls and a light/dark theme toggle (remembered by the browser).
- For every entry a detail pane with up to four tabs:
  - **Overview** – metadata (code, path, last modification time, toroidal mode number,
    number of time slices / dumps, presence of `equilibrium.h5`, `energy.bin`, …), the
    derived **physical model** table, clickable child entries and the directory listing.
    Cases and folders show the physical model of their first simulation.
  - **Inputs** – the code **input namelists** (`C1input`, `nimrod.in`, `nimeq.in`,
    `fluxgrid.in`, `oculus.in`, `README`, …). Each file can be viewed either as a
    filterable parameter table grouped by `&namelist`, or as raw text.
  - **Growth rates** – the M3D-C1 growth-rate (`gamma`) table as a sortable-looking data
    table or as raw text. Only shown when such a file exists; a simulation inherits the
    table of its case directory.
  - **Plots** – a gallery placeholder. Plot support is already wired up but inactive by
    default (see below).
- Selecting an entry updates the URL hash, so individual simulations can be bookmarked and
  shared as links.

The **physical model** is derived from the primary input file of each simulation:

| Code   | Parameters used                                                                 |
|--------|---------------------------------------------------------------------------------|
| M3D-C1 | `linear`, `numvar`, `itwo_fluid` / `db_fac`, `eqsubtract`, `icomplex`, `ntor`, …  |
| NIMROD | `nonlinear`, `ohms`, `continuity`, `separate_pe`, `p_model`, …                   |

### Keeping the page small

The embedded data is packed before it is written: namelists and growth-rate tables are
shipped as raw text and only parsed in the browser when they are opened, identical file
contents are stored once, long numbered file series (`dumpgll.00001`, …) are collapsed into
a single row, and the JSON payload is **gzip + base64** encoded. The page inflates it with
the browser's `DecompressionStream`, which all current browsers provide; use
`--no-compress` to embed plain JSON for older browsers.

### How simulations are discovered

The browser applies the same rules as `group_simulation_directories()`:

| Code   | A directory is a simulation if it contains |
|--------|--------------------------------------------|
| M3D-C1 | `C1.h5`                                    |
| NIMROD | `nimhist.bin` **and** `nimrod.in`, and the directory name starts with `nNN` or `nln` |

Any path containing a `base_*` component is skipped. In addition, the browser skips hidden
directories (names starting with `.`). The **parent** of a simulation directory becomes the
*case* node (its own input files and growth-rate table are attached there as well), and all
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

# Smaller page: no per-directory file listings, shorter input previews
python make_browser.py /scratch/mastu_runs --max-files 0 --max-doc-bytes 32768
```

A typical run prints a summary such as (numbers are illustrative):

```
edges-ml browser: 1 root directory  •  12 case(s)  •  37 simulation(s)  •  25 M3D-C1  •  12 NIMROD
Payload: 1.8 MB packed JSON -> 412.3 KB embedded (gzip+base64)
Wrote /path/to/simulations.html (430.6 KB)
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
| `--max-doc-bytes N`    | `262144`             | Maximum characters embedded per input / growth-rate file           |
| `--max-files N`        | `200`                | Maximum entries in the per-directory file listing (0 disables it)  |
| `--no-compress`        | off                  | Embed plain JSON instead of gzip+base64 (bigger file, works without `DecompressionStream`) |
| `--json FILE`          | –                    | Also dump the embedded (packed) data model as JSON                 |
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
paths (both codes are then searched), or `(code, directory)` tuples. It also takes the
keyword arguments `scan_plots`, `max_doc_bytes`, `max_files`, `json_output`, `compress` and
`quiet`, which correspond to the command-line options. Related helpers:

```python
from edges_ml import discover_simulations, describe_simulation, parse_namelist
```

### Plots

The *Plots* tab is present but empty by default and shows a placeholder note. Pass
`--scan-plots` to link image files (`*.png`, `*.jpg`, `*.jpeg`, `*.svg`, `*.gif`, `*.webp`,
`*.pdf`) found in the simulation directory itself or in a `plots/`, `figures/`, `figs/` or
`images/` subdirectory:

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

If the page only shows an error about `DecompressionStream`, regenerate it with
`--no-compress` or open it in a current browser.

---

## Known Limitations

- Poloidal spectra are only computed for scalar fields; vector fields (B, v, …) are planned.
- M3D-C1: the `native` grid type is not implemented (placeholder returning zeros).
- M3D-C1: the `units` setting is not forwarded to `m3dc1.eval_field()` for 2D/3D fields;
  these are written in `eval_field`'s default unit system.
- M3D-C1: `include_omega` is not implemented for spectra. Spectra are evaluated at constant
  geometric φ, as in `eigenfunction()`.
- M3D-C1 campaign-archive mode: spectra depend on `flux_coordinates()` working with the
  streamed data. If it fails, spectra are skipped for that model with an error message.
  IMAS output is not supported for remote data.
- NIMROD: the flux average of `j` and the `j` field need `dump_ja=T`.
- NIMROD: the crash-safe probe evaluates each field at a single point. A field that only
  crashes at other points still terminates the process. Probing needs `os.fork`.
- NIMROD: the mode axis returned by `eval_comp_field()` is assumed to follow the order of
  `keff` in the dump. A warning is printed if the mode counts disagree.
- NIMROD: the FSA-based flux averages and global parameters normalise ψ_N to the domain
  boundary in diverted cases. The spectrum grid uses the separatrix.
- NIMROD: total 2D/3D fields at a time slice build their grid points from that time
  slice's dump. This equals `/grids/2d_grid` because the mesh of a run does not change
  between dumps.
- `units = "codeunits"`: the `unit` attribute in the output is a placeholder.

---

## Adding a New Code

1. Create `edges_ml/mycode_adapter.py` with a class subclassing
   `edges_ml.base.SimulationAdapter`. Implement the `get_*` methods you need; unimplemented
   methods are skipped with a one-time warning. Read the quantities from
   `utils.export_selection(config)`, never from flat config keys.
2. Register it in `_make_adapter()` and add its discovery rule to
   `group_simulation_directories()` in `edges_ml/edges_ml.py`.
3. Export it in `edges_ml/__init__.py` (inside a `try` block, like the existing adapters).
4. Optionally teach the HTML browser about it by adding entries to `CODE_LABELS`,
   `CODE_ORDER`, `SIM_MARKERS`, `DOC_PATTERNS` and `PRIMARY_INPUT` in
   `edges_ml/browser.py`, plus a `_describe_mycode()` helper for the physical-model summary
   (called from `describe_simulation()`).

See [`docs/design_notes.md`](docs/design_notes.md) for the adapter contract and the
conventions every adapter must follow.
