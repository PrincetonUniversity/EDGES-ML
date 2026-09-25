"""
Abstract base class for all simulation code adapters in edges-ml.
"""

import numpy as np
from pathlib import Path
from typing import Union

from .utils import (
    printwarn, printerr,
    GridSpec,
    expand_request,
    _subtract_field_results,
    AVAILABLE_1D_PROFILES,
    AVAILABLE_2D_FIELDS,
    AVAILABLE_3D_FIELDS,
    AVAILABLE_FLUX_AVERAGES,
    AVAILABLE_TIME_TRACES,
)


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

    def get_mode_1d_profile(self, mode_dir, name, resolution, units):
        self._warn_unimplemented('get_mode_1d_profile')
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

    def close(self):
        pass

    # --- Core Extraction Logic ---
    def extract_grids(self, config):
        """
        Extracts all static grids and the time coordinate mapping into one structure.
        """
        grids = {}
        res_1d  = config.get("resolutions", {}).get("1d", 200)
        units   = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")
        time_slices = config.get("time_slices", [])

        if time_slices:
            grids["time"] = self.get_time_metadata(time_slices)

        fluxes_to_try = expand_request(config.get("flux_averages"), AVAILABLE_FLUX_AVERAGES)
        if not fluxes_to_try:
            fluxes_to_try = AVAILABLE_FLUX_AVERAGES

        grids["1d_grid"] = np.array([])
        for field in fluxes_to_try:
            f_mesh, _ = self.get_flux_average(field, res_1d, units, fcoords, time=-1)
            if len(f_mesh) > 0:
                grids["1d_grid"] = f_mesh
                break

        grid_2d = config.get("2d_grid", GridSpec("rectangular", (128, 128)))
        grids["2d_grid"] = self.get_2d_mesh(grid_2d)

        grid_3d = config.get("3d_grid", GridSpec("rectangular", (64, 16, 64)))
        grids["3d_grid"] = self.get_3d_mesh(grid_3d)

        return grids

    def extract_equilibrium(self, config):
        """
        Parses the config dictionary and extracts all the shared,
        time-independent or base-state equilibrium quantities.
        """
        data = {
            "global_parameters": self.get_global_parameters(),
            "2d_fields": {}, "3d_fields": {},
            "flux_averages": {}, "time_traces": {}
        }

        res_1d  = config.get("resolutions", {}).get("1d", 200)
        units   = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")

        fields_2d = expand_request(config.get("2d_fields"), AVAILABLE_2D_FIELDS)
        grid_2d   = config.get("2d_grid", GridSpec("rectangular", (128, 128)))
        for field in fields_2d:
            val = self.get_2d_field(field, grid_2d, units, time=-1)
            if isinstance(val, dict):
                data["2d_fields"].update(val)
            else:
                data["2d_fields"][field] = val

        fields_3d = expand_request(config.get("3d_fields"), AVAILABLE_3D_FIELDS)
        grid_3d   = config.get("3d_grid", GridSpec("rectangular", (64, 16, 64)))
        for field in fields_3d:
            val = self.get_3d_field(field, grid_3d, units, time=-1)
            if isinstance(val, dict):
                data["3d_fields"].update(val)
            else:
                data["3d_fields"][field] = val

        fluxes = expand_request(config.get("flux_averages"), AVAILABLE_FLUX_AVERAGES)
        for name in fluxes:
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

    def extract_total_fields(self, config):
        """
        Parses the config dictionary and extracts total fields across selected time slices.
        Produces arrays of shape [N_t, ...]
        """
        time_slices = config.get("time_slices", [])
        if not time_slices:
            return None

        data = {
            "flux_averages": {},
            "2d_fields":     {},
            "3d_fields":     {}
        }

        res_1d  = config.get("resolutions", {}).get("1d", 200)
        units   = config.get("units", "codeunits")
        fcoords = config.get("fcoords", "pest")

        fluxes = expand_request(config.get("total_flux_averages"), AVAILABLE_FLUX_AVERAGES)
        for field in fluxes:
            results = []
            for ts in time_slices:
                _, f_vals = self.get_flux_average(field, res_1d, units, fcoords, time=ts, use_eq_fs=True)
                if not isinstance(f_vals, dict) and len(f_vals) == 0:
                    f_vals = np.zeros(res_1d)
                results.append(f_vals)
            if results and isinstance(results[0], dict):
                for comp in results[0].keys():
                    data["flux_averages"][comp] = np.stack([r[comp] for r in results], axis=0)
            else:
                data["flux_averages"][field] = np.stack(results, axis=0) if results else np.array([])

        fields_2d = expand_request(config.get("total_2d_fields"), AVAILABLE_2D_FIELDS)
        grid_2d   = config.get("2d_grid", GridSpec("rectangular", (128, 128)))
        for field in fields_2d:
            results = []
            for ts in time_slices:
                val = self.get_2d_field(field, grid_2d, units, time=ts)
                results.append(val)
            if results and isinstance(results[0], dict):
                for comp in results[0].keys():
                    data["2d_fields"][comp] = np.stack([r[comp] for r in results], axis=0)
            else:
                data["2d_fields"][field] = np.stack(results, axis=0) if results else np.array([])

        fields_3d = expand_request(config.get("total_3d_fields"), AVAILABLE_3D_FIELDS)
        grid_3d   = config.get("3d_grid", GridSpec("rectangular", (64, 16, 64)))
        for field in fields_3d:
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

    def extract_mode(self, mode_dir, config):
        """
        Extracts the mode-specific perturbation data (eigenfunction).
        eigenfunction = field(finite_time, mode sim) - field(equilibrium, time=-1)
        """
        res_1d = config.get("resolutions", {}).get("1d", 200)
        units  = config.get("units", "codeunits")
        data   = {
            "mode_information": self.get_mode_metadata(mode_dir),
            "1d_profiles": {}, "2d_fields": {}, "3d_fields": {}
        }

        fields_1d = expand_request(config.get("mode_1d_profiles"), AVAILABLE_1D_PROFILES)
        for field in fields_1d:
            data["1d_profiles"][field] = self.get_mode_1d_profile(mode_dir, field, res_1d, units)

        grid_2d   = config.get("perturbation_2d_grid", config.get("2d_grid", GridSpec("rectangular", (128, 128))))
        fields_2d = expand_request(config.get("mode_2d_fields"), AVAILABLE_2D_FIELDS)
        for field in fields_2d:
            val_finite = self.get_mode_2d_field(mode_dir, field, grid_2d, units)
            val_eq     = self.get_2d_field(field, grid_2d, units, time=-1)
            val        = _subtract_field_results(val_finite, val_eq)
            if isinstance(val, dict):
                data["2d_fields"].update(val)
            else:
                data["2d_fields"][field] = val

        grid_3d   = config.get("perturbation_3d_grid", config.get("3d_grid", GridSpec("rectangular", (64, 16, 64))))
        fields_3d = expand_request(config.get("mode_3d_fields"), AVAILABLE_3D_FIELDS)
        for field in fields_3d:
            val_finite = self.get_mode_3d_field(mode_dir, field, grid_3d, units)
            val_eq     = self.get_3d_field(field, grid_3d, units, time=-1)
            val        = _subtract_field_results(val_finite, val_eq)
            if isinstance(val, dict):
                data["3d_fields"].update(val)
            else:
                data["3d_fields"][field] = val

        return data
