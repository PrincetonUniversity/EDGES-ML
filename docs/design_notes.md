# edges-ml design notes

This document records how edges-ml is structured, which conventions all adapters follow,
and *why* several non-obvious decisions were made. It complements the user-facing
[README](../README.md). Keep it up to date when you change behaviour.

---

## 1. Architecture and data flow

```
build_dataset(sources, output_directory, config)            edges_ml/edges_ml.py
 ├─ normalize_export_config(config)                          edges_ml/utils.py
 │    resolves the three export sections once, validates them,
 │    prints the "Export selection" summary and caches the result
 │    in config["_export_selection"]
 ├─ group_simulation_directories(sources)                    model dir -> [mode dirs]
 └─ per model:
      adapter = _make_adapter(code, model_dir, mode_dirs)
      adapter.extract_grids(config)        -> /grids
      adapter.extract_equilibrium(config)  -> /equilibrium, /global_parameters, /time_traces
      adapter.extract_total_fields(config) -> /total_fields
      for mode_dir: adapter.extract_mode_entries(mode_dir, config) -> /perturbations/<entry>
      adapter.close()
```

* `edges_ml/base.py` – `SimulationAdapter`. Implements the extraction loops
  (`extract_*`) on top of small per-code primitives (`get_2d_field`, `get_flux_average`,
  `get_mode_spectrum`, ...). Unimplemented primitives warn once and return empty arrays.
* `edges_ml/m3dc1_adapter.py`, `edges_ml/nimrod_adapter.py` – the code adapters.
* `edges_ml/nimrod_fluxcoords.py` – ψ reconstruction and straight-field-line coordinates
  for NIMROD (port of M3D-C1's `flux_coordinates()`).
* `edges_ml/campaign.py` – ADIOS2 campaign archives (remote M3D-C1 data).
* `edges_ml/browser.py` – static HTML simulation browser (standard library only;
  `make_browser.py` is a thin launcher).

### The adapter contract

* Read the quantities to export **only** through `utils.export_selection(config)`
  (and `utils.spectrum_settings(config)` for spectra). Never read flat config keys.
* Evaluate all 2D/3D fields on `SimulationAdapter._grid_spec(config, 2|3)`, i.e. on
  `config["2d_grid"]` / `config["3d_grid"]`. There are no section-specific grids.
* Return an empty array (or `{}`) for anything that is unavailable, and warn once.
  Never return zeros as a stand-in for missing data.
* Vector fields are returned as a dict `{name_R, name_phi, name_Z}`.
* A spectrum is a dict `{psi_norm, m, amplitude, [flux grid datasets], _attrs}`.
  The `_attrs` entry is written as HDF5 / ADIOS2 attributes of the spectrum group.

---

## 2. Export selection

### Why three sections with identical lists

Earlier versions had independent flat keys (`2d_fields`, `total_2d_fields`,
`mode_2d_fields`, `mode_spectra`, ...). That made it possible to export the 2D perturbation
of `B` but the spectrum of `p`, which is inconsistent and was observed in practice.

The configuration now has three sections — `equilibrium`, `total_fields`,
`perturbations` — each with `flux_averages`, `2d_fields`, `3d_fields`. A poloidal spectrum
is computed for **every scalar field listed under `perturbations`**; there is no separate
spectrum list. `perturbations["spectrum"]` only holds numerical settings.

### Resolution rules (`utils._build_export_selection`)

* `"all"` (or a list containing it) expands to the full registry of that kind.
* `q_profile` is an alias for `q`.
* If a section is given, the corresponding legacy flat key is ignored (warning).
  If it is not given, the legacy key is used (deprecation warning).
* `mode_spectra`, `perturbation_2d_grid`, `perturbation_3d_grid` have no effect any more.
* `spectra` = perturbation quantities that are in `FIELD_LABELS` and not in
  `VECTOR_FIELDS`.
* The resolved selection is stored in `metadata/perturbation_quantities` and
  `metadata/spectrum_quantities`.

---

## 3. Conventions

| Item                     | Convention |
|--------------------------|------------|
| Toroidal angle φ         | M3D-C1 direction. NIMROD vector φ-components are multiplied by `PHI_SIGN_FLIP = -1`; NIMROD Fourier coefficients are complex-conjugated for spectra. |
| Poloidal angle θ         | θ = 0 on the outboard midplane, increasing clockwise in (R, Z) (M3D-C1 `flux_coordinates()`). |
| Spectrum normalisation   | `|rfft(ef, axis=θ, norm='forward')|`, as in `m3dc1.eigenfunction()`. |
| Mode number m            | 0 … n_θ/2; only magnitudes are stored, so the helicity sign is not resolved. |
| Perturbation (M3D-C1)    | `field(mode run, time='last') − field(equilibrium)`. |
| Perturbation (NIMROD)    | `2 Re[c_n e^{i n φ}]` from `eval_comp_field()` (n = 0: `Re[c_0]`). |
| Flux-average perturbation| Mode-run average minus equilibrium average, on the equilibrium 1D grid. For NIMROD this is the n = 0 profile change of the run (n > 0 parts average to zero) and is shared by all entries of a directory. |

---

## 4. NIMROD details

### 4.1 Field selectors

| `eq` | Field                            | Used for |
|------|----------------------------------|----------|
| 1    | equilibrium only                 | equilibrium FSA, global parameters, spectrum flux grid, magnetic axis |
| 2    | equilibrium + full perturbation  | 2D/3D equilibrium and total fields |
| 3    | equilibrium + n = 0 perturbation | total-field and perturbation flux averages |

### 4.2 The current density and the plasma current

`j` is only in the dump if the run used `dump_ja=T`. Evaluating it otherwise ends the
process with `NIM_STOP`, which cannot be caught from Python. This first appeared when
running `reduced_h5` for a run that had previously only been converted with `imas_h5`
(which never calls the FSA code).

* `_current_available()` reads `dump_ja` from `nimrod.in` (f90nml, regex fallback; missing
  means `F`). `config["nimrod_dump_ja"]` overrides it.
* With `dump_ja=F`: `j` is removed from the `EvalNimrod` field list, the integrands
  `j_R, j_Z, j_phi, jphi_over_R` are not computed, and `j` fields / spectra are skipped.

The FSA itself only needs `B`. The plasma current used to be computed by integrating
`⟨j_φ/R⟩`, which needlessly required `j`. It is now computed with Ampère's law on the FSA
contours (`_enclosed_current()`):

```
I(ψ) = (1/μ0) ∮ B_pol · dl
```

B is evaluated at the contour segment midpoints (one nimpy call for all surfaces). Each
contour is oriented counter-clockwise in (R, Z) using its signed area; in NIMROD's
right-handed (R, Z, φ) system this gives the current along +φ_NIMROD, which is then
converted with `PHI_SIGN_FLIP`. This is independent of nimpy's internal FSA normalisation.
`Ip` (flux average and global parameter) and `beta_n` therefore work without `dump_ja`.

Computing ⟨j⟩ from ∇×B would also be possible with `eval_field(..., dmode=1)`, but the
derivative layout of nimpy was not verified; it is not implemented.

### 4.3 Crash-safe probing

After removing `j`, the FSA started evaluating fields that had never been reached before
(`pe`, `psi`, `te`, `ti`, `ni`, `v`) and the field list contained `e`; one of these
segfaulted inside nimpy. Segfaults cannot be caught either, so every risky call is first
tried in a throwaway process:

* `_run_isolated(func)` forks. The child silences stdout/stderr, runs `func()` and sends the
  pickled result through a pipe. The parent waits with a timeout (`select`).
  * Normal return → `(True, result)`.
  * Exception, signal (`SIGSEGV`, ...) or exit without a payload (`NIM_STOP`) →
    `(False, message)`.
  * Timeout → the child is killed, probing is disabled for the rest of the run.
* `_probe_eval_init()` probes each `EvalNimrod(dump, fieldlist)`; failing field lists are
  skipped and the next candidate is tried.
* `_field_ok(dump, field, eq)` probes one evaluation at the mesh node closest to the centre
  of the domain. Results are cached per (dump, field list, field, eq).
* Every consumer (FSA integrands, 2D/3D fields, per-mode coefficients, spectra, on-axis
  global parameters) checks `_field_ok()` and skips the quantity with a warning if it fails.
  If `b` fails, flux averages and spectra are skipped entirely.

The default field list is `nvptb` (as in the standard nimpy FSA scripts); `j` is appended
only for `dump_ja=T`, `e` only if `E` is requested or `nimrod_eval_e=True`.

Limitations: a single probe point; requires `os.fork`; the child inherits the parent's
memory copy-on-write.

### 4.4 FSA records

`_get_fsa_record()` runs one FSA pass per (dump, nsurf, eq, integrand set) and caches it.
`dvar`, the integrand values **and the contours** are sorted by ascending ψ_N. Sorting the
contours as well fixed a bug where `contours[..., -1]` (used as the LCFS for the shaping
parameters) could be the innermost surface if nimpy returned surfaces outside-in.

### 4.5 Dumps and time slices

Dumps of each `nXX` directory are indexed sequentially: index 0 is step 0 or the first
all-zero dump, 1, 2, … follow by step. `time=-1` maps to index 0. A requested index missing
in some directories falls back to the next smaller common index. Binary dumps are converted
with `dump2h5`, which **edits `nimrod.in` in place** (`nimuw_dump_compat=T`, `h5dump=T`).

### 4.6 Flux coordinates for spectra (`nimrod_fluxcoords.py`)

NIMROD provides no poloidal flux function, so the spectrum grid is built from the
equilibrium field (`eq=1`) alone. The module is independent of nimpy (B is passed in as a
callback), so it can be tested with analytic fields.

1. **ψ along rays.** `m` rays from the magnetic axis (θ clockwise, θ = 0 outboard) up to
   99.9 % of the distance to the domain bounding box, each with `ray_samples` points
   (default 4 × `points`, at least 50). `dψ/dρ = R (B_Z cos − B_R sin)` (R → 1 for `lin`)
   is integrated with the trapezoidal rule. A ray ends at the first run of more than two
   consecutive NaN samples (it left the domain); isolated NaNs are interpolated.
2. **ψ_LCFS.** Candidates, the smallest |ψ| wins:
   * the smallest over all rays of the largest |ψ| reached along each ray (limiter value,
     or an estimate of the separatrix from above in diverted plasmas);
   * the flux at X-points found by `find_pf_null` from user guesses
     (`nimrod_xpoint_guess`) and automatic guesses. Automatic guesses are the |B_p| minima
     in the upper and lower half, between 35 % and 97 % of the ray length, that are below
     25 % of the median |B_p|. A null is accepted if it lies farther than 10 % of the median
     ray length from the axis, |B_p| there is below 5 % of the median, ψ has the right sign
     and |ψ_X| does not exceed the ray estimate.

   `nimrod_psi_lcfs` overrides the result.
3. **Inversion.** ψ_N targets `linspace(1e-4, 1, points)`; on each ray the first crossing
   is found with the quadratic that is exactly consistent with the trapezoidal ψ
   integration (no extra field evaluations). Unreachable targets are clipped to the last
   valid sample (`n_unreached`).
4. **Straight-field-line angle.** Faithful port of M3D-C1 (`dθ/dl` = |I|/(R²B_p) for PEST,
   B²/B_p for Boozer, 1/B_p for Hamada), including its two numerical quirks (open
   cumulative sum vs. closed normalisation; linear resampling with clipping, counted in
   `n_clipped`). For Boozer/Hamada, ω = ζ − φ = q (θ_f − θ_PEST) is available.

---

## 5. Total fields

`SimulationAdapter._stack_flux_averages()` stacks per-time-slice flux averages into
`[N_t, N_ψ]`. A quantity that is unavailable for every time slice is skipped (it used to be
written as zeros, which looked like real data); partially missing slices are filled with
NaN.

---

## 6. Simulation browser

`edges_ml/browser.py` crawls the same layout as `group_simulation_directories()` (plus it
skips hidden directories) and writes one self-contained HTML file. The payload is packed
(raw texts deduplicated in a pool, labels indexed, nodes stored as a flat array with parent
indices, numbered file series collapsed) and gzip+base64 encoded; namelists and growth-rate
tables are parsed in the page on demand. Tabs: Overview, Inputs, Growth rates (M3D-C1
`gamma` file, inherited from the case directory), Plots (`--scan-plots`).

---

## 7. Change history

The following commits form the series that introduced the behaviour described above
(newest last):

| Commit    | Summary |
|-----------|---------|
| (earlier) | `q` removed from the spectrum groups (it is a flux average); flux-grid datasets kept behind `store_fluxgrid`. |
| `b6492a7` | Export configuration restructured into `equilibrium` / `total_fields` / `perturbations` sections; spectra computed for every scalar perturbation quantity; perturbation flux averages (`perturbations/<entry>/flux_averages`, replacing `1d_profiles`); perturbations always on `2d_grid` / `3d_grid`; legacy keys mapped with deprecation warnings. |
| `9268dfe` | Examples updated to the sectioned configuration. |
| `24cff09` | `examples/plot_perturbation.py` (R–Z plot and spectrum from `reduced_h5`). |
| `7fc2f52` | NIMROD IMAS example with a working source path. |
| `0d44fa7` | NIMROD `reduced_h5` example aligned with the IMAS example. |
| `029ea72` | Total-field flux averages that are unavailable are skipped instead of written as zeros; partial gaps filled with NaN. |
| `0d38fd2` | NIMROD: `dump_ja` detection; quantities that need `j` are skipped for `dump_ja=F`. |
| `04a1ac4` | NIMROD: plasma current from B via Ampère's law; FSA contours sorted with ψ_N. |
| `9f1a655` | NIMROD: crash-safe probing of nimpy field lists and fields in forked subprocesses; default field list `nvptb`; `nimrod_verbose` stage markers. |
| (docs)    | Documentation pass: README rewritten, these design notes added, outdated comments corrected; browser section of the README and `nimrod_fluxcoords.py` docstrings checked against the code. |
