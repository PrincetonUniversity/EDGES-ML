"""
Abstract base class for all simulation code adapters in edges-ml.

The quantities to export are taken from the resolved export selection
(utils.export_selection), which has the same structure for every section:

    equilibrium   -> flux_averages / 2d_fields / 3d_fields
    total_fields  -> flux_averages / 2d_fields / 3d_fields   (per time slice)
    perturbations -> flux_averages / 2d_fields / 3d_fields   (+ poloidal spectra
                                                              of the same quantities)

All 2D / 3D fields of all sections are evaluated on the same grids
('2d_grid' / '3d_grid').
"""

import numpy as np
from pathlib import Path
from typing import Union

from .utils import (
    printwarn, printerr,
    GridSpec,
    brief_error,
    expand_request,
    export_selection,
    unique_names,
    spectrum_settings,
    parse_toroidal_mode_number,
    _subtract_field_results,
    AVAILABLE_FLUX_AVERAGES,
    AVAILABLE_TIME_TRACES,
)


DEFAULT_GRID_2D = GridSpec("rectangular", (128, 128))
DEFAULT_GRID_3D = GridSpec("rectangular", (64, 16, 64))


def _has_data(value):
    """True if a flux-average / field result contains any data."""
    if value is None:
        return False
    if isinstance(value, dict):
        return any(np.size(v) > 0 for v in value.values())
    return np.size(value) > 0


class SimulationAdapter:

    CODE_NAME = "UNKNOWN"

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.shared_inner_wall_points = None
        self._warned_methods = set()

    def _warn_unimplemented(self, method_name):
        if method_name not in self._warned_methods:
            printwarn(f"Warning: {method_name}() is not yet implemented for {self.CODE_NAME}. Skipping.")
            self._warned_methods.add(method_name)

    @staticmethod
    def _grid_spec(config, dim):
        """Grid used for ALL 2D (dim=2) or 3D (dim=3) fields of every section."""
        cfg = config or {}
        if dim == 2:
            return cfg.get("2d_grid", DEFAULT_GRID_2D)
        return cfg.get("3d_grid", DEFAULT_GRID_3D)

    # --- Abstract Methods (Safely stubbed to skip when not implemented by subclass) ---
    def get_metadata(self, mode_dirs):
        self._warn_unimplemented('get_metadata')
        return {
            "code": self.CODE_NAME,
            "source_directory": str(self.model_dir.resolve()),
            "release_version": "Unknown",
            "build_date": "Unknown"
        }

    def get_all_input_parameters(self):
        self._warn_unimplemented('get_all_input_parameters')
        return {}

    def get_global_parameters(self):
        self._warn_unimplemented('get_global_parameters')
        return {}

    def get_reproducibility_data(self, mode_dir):
        self._warn_unimplemented('get_reproducibility_data')
        return {}

    def get_1d_mesh(self, resolution, units, fcoords):
        self._warn_unimplemented('get_1d_mesh')
        return np.array([])

    def get_2d_mesh(self, grid_spec):
        self._warn_unimplemented('get_2d_mesh')
        return np.array([])

    def get_3d_mesh(self, grid_spec):
        self._warn_unimplemented('get_3d_mesh')
        return np.array([])

    def get_2d_field(self, name, grid_spec, units, time=-1):
        self._warn_unimplemented('get_2d_field')
        return np.array([])

    def get_3d_field(self, name, grid_spec, units, time=-1):
        self._warn_unimplemented('get_3d_field')
        return np.array([])

    def get_flux_average(self, name, resolution, units, fcoords, time=-1, use_eq_fs=True):
        self._warn_unimplemented('get_flux_average')
        return np.array([]), np.array([])

    def get_time_trace(self, name, units):
        self._warn_unimplemented('get_time_trace')
        return np.array([]), np.array([])

    def get_mode_metadata(self, mode_dir):
        self._warn_unimplemented('get_mode_metadata')
        return {"growth_rate": 0.0, "frequency": 0.0, "mode_type": -100}

    def get_mode_flux_average(self, mode_dir, name, resolution, units, fcoords):
        """
        Flux average of the perturbation of *name* on the equilibrium 1D grid:
            <field>(mode simulation) - <field>(equilibrium)
        Returns an array (scalars), a {component: array} dict (vectors) or an
        empty array if not available.
        """
        self._warn_unimplemented('get_mode_flux_average')
        return np.array([])

    def get_mode_2d_mesh(self, mode_dir, grid_spec):
        self._warn_unimplemented('get_mode_2d_mesh')
        return np.array([])

    def get_mode_3d_mesh(self, mode_dir, grid_spec):
        self._warn_unimplemented('get_mode_3d_mesh')
        return np.array([])

    def get_mode_2d_field(self, mode_dir, name, grid_spec, units):
        self._warn_unimplemented('get_mode_2d_field')
        return np.array([])

    def get_mode_3d_field(self, mode_dir, name, grid_spec, units):
        self._warn_unimplemented('get_mode_3d_field')
        return np.array([])

    def get_mode_spectrum(self, mode_dir, field, settings, n=None):
        """
        Poloidal Fourier spectrum of the mode perturbation of *field* on
        flux-aligned coordinates.

        Parameters
        ----------
        mode_dir : Path
            Mode directory (or campaign simulation path).
        field : str
            edges-ml field name (one of the perturbation quantities).
        settings : dict
            Output of utils.spectrum_settings(config).
        n : int, optional
            Toroidal mode number.

        Returns
        -------
        dict
            {
              "psi_norm":  [n_psi],
              "m":         [n_m],
              "amplitude": [n_m, n_psi],
              (optional, store_fluxgrid) "eigenfunction", "R", "Z", "theta", ...,
              "_attrs":    {...}   # written as group attributes
            }
            or {} if not available.
        """
        self._warn_unimplemented('get_mode_spectrum')
        return {}

    def get_time_metadata(self, time_slices):
        self._warn_unimplemented('get_time_metadata')
        return {
            "time_slice": np.array(time_slices, dtype=int),
            "simulation_time_step": np.zeros(len(time_slices), dtype=int),
            "simulation_time": np.zeros(len(time_slices), dtype=float)
        }

    def convert_to_imas(self, mode_dirs, output_directory, config, run=None, out_file=None):
        self._warn_unimplemented('convert_to_imas')
        return None

    def get_all_mode_entries(self, mode_dir, config):
        """
        Returns a list of (group_name, mode_metadata) tuples to be written under
        the 'perturbations' group for a given mode directory.
        """
        return [(mode_dir.name, self.get_mode_metadata(mode_dir))]

    def extract_mode_entries(self, mode_dir, config):
        """
        Returns a list of (entry_name, entry_metadata, mode_data) tuples for a
        mode directory.

        The default implementation extracts the mode data once and shares it
        between every entry returned by get_all_mode_entries().  Adapters that
        resolve several toroidal modes per directory (e.g. NIMROD multi-n runs)
        override this method to return per-mode data.
        """
        mode_data = self.extract_mode(mode_dir, config)
        entries   = self.get_all_mode_entries(mode_dir, config)
        return [(name, meta, mode_data) for name, meta in entries]

    def close(self):
        pass

    # --- Core Extraction Logic ---
    def extract_grids(self, config):
        """
        Extracts all static grids and the time coordinate mapping into one structure.
        """
        grids = {}
        sel     = export_selection(config)
        res_1d  = config.get("resolutions", {}).get("1d", 200)
        units   = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")
        time_slices = config.get("time_slices", [])

        if time_slices:
            grids["time"] = self.get_time_metadata(time_slices)

        fluxes_to_try = unique_names(
            sel["equilibrium"]["flux_averages"]
            + sel["total_fields"]["flux_averages"]
            + sel["perturbations"]["flux_averages"]
        )
        if not fluxes_to_try:
            fluxes_to_try = AVAILABLE_FLUX_AVERAGES

        grids["1d_grid"] = np.array([])
        for field in fluxes_to_try:
            f_mesh, _ = self.get_flux_average(field, res_1d, units, fcoords, time=-1)
            if len(f_mesh) > 0:
                grids["1d_grid"] = f_mesh
                break

        grids["2d_grid"] = self.get_2d_mesh(self._grid_spec(config, 2))
        grids["3d_grid"] = self.get_3d_mesh(self._grid_spec(config, 3))

        return grids

    def extract_equilibrium(self, config):
        """
        Extracts the equilibrium quantities listed under config['equilibrium'],
        plus the global parameters and the time traces.
        """
        sel = export_selection(config)["equilibrium"]
        data = {
            "global_parameters": self.get_global_parameters(),
            "2d_fields": {}, "3d_fields": {},
            "flux_averages": {}, "time_traces": {}
        }

        res_1d  = config.get("resolutions", {}).get("1d", 200)
        units   = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")

        grid_2d = self._grid_spec(config, 2)
        for field in sel["2d_fields"]:
            val = self.get_2d_field(field, grid_2d, units, time=-1)
            if isinstance(val, dict):
                data["2d_fields"].update(val)
            else:
                data["2d_fields"][field] = val

        grid_3d = self._grid_spec(config, 3)
        for field in sel["3d_fields"]:
            val = self.get_3d_field(field, grid_3d, units, time=-1)
            if isinstance(val, dict):
                data["3d_fields"].update(val)
            else:
                data["3d_fields"][field] = val

        for name in sel["flux_averages"]:
            _, f_vals = self.get_flux_average(name, res_1d, units, fcoords, time=-1)
            if isinstance(f_vals, dict):
                data["flux_averages"].update(f_vals)
            elif len(f_vals) > 0:
                data["flux_averages"][name] = f_vals

        traces     = expand_request(config.get("time_traces"), AVAILABLE_TIME_TRACES)
        time_saved = False
        for name in traces:
            t_arr, v_arr = self.get_time_trace(name, units)
            if not time_saved and len(t_arr) > 0:
                data["time_traces"]["time"] = t_arr
                time_saved = True
            if len(v_arr) > 0:
                data["time_traces"][name] = v_arr

        return data

    @staticmethod
    def _stack_flux_averages(field, results, time_slices):
        """
        Stacks per-time-slice flux averages into [N_t, N_psi] arrays.

        Time slices for which the quantity is not available are filled with
        NaN.  Returns {} (with a warning) if the quantity is unavailable for
        every time slice, so that no fake all-zero data is written.
        """
        valid = [r for r in results if _has_data(r)]
        if not valid:
            printwarn(f"Total-field flux average '{field}' is not available for any "
                      f"of the time slices {list(time_slices)}; skipping.")
            return {}

        missing = [ts for ts, r in zip(time_slices, results) if not _has_data(r)]
        if missing:
            printwarn(f"Total-field flux average '{field}' is not available for time "
                      f"slice(s) {missing}; these are filled with NaN.")

        ref = valid[0]
        out = {}
        if isinstance(ref, dict):
            for comp, ref_arr in ref.items():
                ref_arr = np.asarray(ref_arr, dtype=float)
                rows = []
                for r in results:
                    if isinstance(r, dict) and comp in r and np.size(r[comp]) == ref_arr.size:
                        rows.append(np.asarray(r[comp], dtype=float))
                    else:
                        rows.append(np.full_like(ref_arr, np.nan))
                out[comp] = np.stack(rows, axis=0)
        else:
            ref_arr = np.asarray(ref, dtype=float)
            rows = []
            for r in results:
                if not isinstance(r, dict) and np.size(r) == ref_arr.size:
                    rows.append(np.asarray(r, dtype=float))
                else:
                    rows.append(np.full_like(ref_arr, np.nan))
            out[field] = np.stack(rows, axis=0)
        return out

    def extract_total_fields(self, config):
        """
        Extracts the quantities listed under config['total_fields'] for every
        selected time slice.  Produces arrays of shape [N_t, ...].
        """
        time_slices = config.get("time_slices", [])
        if not time_slices:
            return None

        sel = export_selection(config)["total_fields"]
        data = {
            "flux_averages": {},
            "2d_fields":     {},
            "3d_fields":     {}
        }

        res_1d  = config.get("resolutions", {}).get("1d", 200)
        units   = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")

        for field in sel["flux_averages"]:
            results = []
            for ts in time_slices:
                _, f_vals = self.get_flux_average(field, res_1d, units, fcoords, time=ts, use_eq_fs=True)
                results.append(f_vals)
            data["flux_averages"].update(self._stack_flux_averages(field, results, time_slices))

        grid_2d = self._grid_spec(config, 2)
        for field in sel["2d_fields"]:
            results = []
            for ts in time_slices:
                val = self.get_2d_field(field, grid_2d, units, time=ts)
                results.append(val)
            if results and isinstance(results[0], dict):
                for comp in results[0].keys():
                    data["2d_fields"][comp] = np.stack([r[comp] for r in results], axis=0)
            else:
                data["2d_fields"][field] = np.stack(results, axis=0) if results else np.array([])

        grid_3d = self._grid_spec(config, 3)
        for field in sel["3d_fields"]:
            results = []
            for ts in time_slices:
                val = self.get_3d_field(field, grid_3d, units, time=ts)
                results.append(val)
            if results and isinstance(results[0], dict):
                for comp in results[0].keys():
                    data["3d_fields"][comp] = np.stack([r[comp] for r in results], axis=0)
            else:
                data["3d_fields"][field] = np.stack(results, axis=0) if results else np.array([])

        return data

    def _extract_mode_spectra(self, mode_dir, config, n=None):
        """
        Computes the poloidal spectra of all perturbation quantities
        (scalar fields listed under config['perturbations']).

        Returns {field: spectrum_dict} or {} if no such quantity is listed.
        """
        settings = spectrum_settings(config)
        if not settings["fields"]:
            return {}

        if n is None:
            n = parse_toroidal_mode_number(Path(mode_dir).name)

        spectra = {}
        for field in settings["fields"]:
            try:
                res = self.get_mode_spectrum(mode_dir, field, settings, n=n)
            except Exception as exc:
                printerr(f"Poloidal spectrum of '{field}' failed for {mode_dir}: "
                         f"{brief_error(exc)}")
                res = {}
            if res:
                spectra[field] = res
        return spectra

    def _extract_mode_flux_averages(self, mode_dir, config):
        """Perturbation flux averages listed under config['perturbations']."""
        sel     = export_selection(config)["perturbations"]
        res_1d  = config.get("resolutions", {}).get("1d", 200)
        units   = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")

        out = {}
        for name in sel["flux_averages"]:
            try:
                val = self.get_mode_flux_average(mode_dir, name, res_1d, units, fcoords)
            except Exception as exc:
                printerr(f"Perturbation flux average '{name}' failed for {mode_dir}: "
                         f"{brief_error(exc)}")
                continue
            if isinstance(val, dict):
                out.update({k: v for k, v in val.items() if np.size(v) > 0})
            elif np.size(val) > 0:
                out[name] = val
        return out

    def extract_mode(self, mode_dir, config):
        """
        Extracts the perturbation quantities listed under config['perturbations']:
            perturbation = field(finite time, mode sim) - field(equilibrium, time=-1)
        as flux averages, 2D fields and 3D fields, plus the poloidal spectra
        of the same quantities.
        """
        sel   = export_selection(config)["perturbations"]
        units = config.get("units", "codeunits")
        data  = {
            "mode_information": self.get_mode_metadata(mode_dir),
            "flux_averages": self._extract_mode_flux_averages(mode_dir, config),
            "2d_fields": {}, "3d_fields": {}
        }

        grid_2d = self._grid_spec(config, 2)
        for field in sel["2d_fields"]:
            val_finite = self.get_mode_2d_field(mode_dir, field, grid_2d, units)
            val_eq     = self.get_2d_field(field, grid_2d, units, time=-1)
            val        = _subtract_field_results(val_finite, val_eq)
            if isinstance(val, dict):
                data["2d_fields"].update(val)
            else:
                data["2d_fields"][field] = val

        grid_3d = self._grid_spec(config, 3)
        for field in sel["3d_fields"]:
            val_finite = self.get_mode_3d_field(mode_dir, field, grid_3d, units)
            val_eq     = self.get_3d_field(field, grid_3d, units, time=-1)
            val        = _subtract_field_results(val_finite, val_eq)
            if isinstance(val, dict):
                data["3d_fields"].update(val)
            else:
                data["3d_fields"][field] = val

        spectra = self._extract_mode_spectra(mode_dir, config)
        if spectra:
            data["spectra"] = spectra

        return data
