"""
edges-ml  –  Multi-code MHD simulation dataset builder for machine learning.

This is the main entry point / workflow script. Adapter classes for individual
simulation codes live in their own modules:

    edges-ml/m3dc1_adapter.py   ->  M3DC1Adapter
    edges-ml/nimrod_adapter.py  ->  NIMRODAdapter

Shared utilities, label registries, and IMAS helpers live in:

    edges-ml/utils.py
    edges-ml/base.py
"""

import os
import re
import sys
import json
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
from termcolor import colored

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from .utils import (
    printwarn, printerr, printnote,
    expand_request, GridSpec,
    generate_unique_filename,
    FIELD_LABELS, FLUX_AVERAGE_LABELS, FLUX_AVERAGE_ONLY_LABELS,
    TIME_TRACE_LABELS,
    FIELD_UNITS_CODE, FLUX_AVERAGE_ONLY_UNITS_CODE, TIME_TRACE_UNITS_CODE,
    AVAILABLE_1D_PROFILES, AVAILABLE_2D_FIELDS, AVAILABLE_3D_FIELDS,
    AVAILABLE_FLUX_AVERAGES, AVAILABLE_TIME_TRACES, AVAILABLE_INPUTS,
    _record_imas_manifest,
)
from .base import SimulationAdapter
from .m3dc1_adapter import M3DC1Adapter
from .nimrod_adapter import NIMRODAdapter


# ===========================================================================
# SERIALIZATION HELPERS
# ===========================================================================

def write_dict_as_attributes(group, data):
    """Saves flat python dictionaries as HDF5 group attributes."""
    for k, v in data.items():
        if isinstance(v, (str, int, float, np.integer, np.floating)):
            group.attrs[k] = v


def write_dict_as_datasets(group, data):
    """Saves flat python dictionaries as individual HDF5 datasets."""
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, str):
            group.create_dataset(k, data=np.bytes_(v))
        else:
            group.create_dataset(k, data=v)


def write_array_group(parent, name, data_dict, unit_mapping=None, desc_mapping=None):
    """Creates an HDF5 group and saves all arrays inside a dictionary as datasets."""
    grp = parent.create_group(name)
    for key, value in data_dict.items():
        if value is None:
            continue

        if isinstance(value, str):
            dset = grp.create_dataset(key, data=np.bytes_(value))
        else:
            dset = grp.create_dataset(key, data=value)

        if unit_mapping or desc_mapping:
            base_key = key
            if base_key.endswith('_R') or base_key.endswith('_phi') or base_key.endswith('_Z'):
                base_key = base_key.rsplit('_', 1)[0]

            if unit_mapping and base_key in unit_mapping:
                dset.attrs['unit'] = unit_mapping[base_key]
            if desc_mapping and base_key in desc_mapping:
                dset.attrs['description'] = desc_mapping[base_key]

    return grp


# ===========================================================================
# DIRECTORY DISCOVERY
# ===========================================================================

def group_simulation_directories(sources):
    """
    Crawls the file system based on the defined sources to find valid data.
    Groups the discovered toroidal modes under their parent model directories.
    Returns: { (model_dir, code): [mode_dir1, mode_dir2, ...] }
    """
    grouped = defaultdict(list)

    for source in sources:
        code = source.get("code", "unknown").lower()

        target_models = source.get("model", None)
        if isinstance(target_models, str):
            target_models = [target_models]

        for root in source.get("directories", []):
            if code == "m3dc1":
                for c1_file in Path(root).rglob("C1.h5"):
                    if any(part.startswith("base_") for part in c1_file.parts):
                        continue

                    mode_dir  = c1_file.parent
                    model_dir = mode_dir.parent

                    if target_models and model_dir.name not in target_models:
                        continue

                    grouped[(model_dir, code)].append(mode_dir)

            elif code == "nimrod":
                for hist_file in Path(root).rglob("nimhist.bin"):
                    if any(part.startswith("base_") for part in hist_file.parts):
                        continue

                    mode_dir = hist_file.parent

                    if not (mode_dir / "nimrod.in").exists():
                        printwarn(f"Skipping {mode_dir}: 'nimhist.bin' found but 'nimrod.in' is missing (prevents segfault).")
                        continue

                    if re.match(r'^n(\d+|ln)', mode_dir.name):
                        model_dir = mode_dir.parent

                        if target_models and model_dir.name not in target_models:
                            continue

                        if mode_dir not in grouped[(model_dir, code)]:
                            grouped[(model_dir, code)].append(mode_dir)

    return grouped


# ===========================================================================
# MAIN WORKFLOW
# ===========================================================================

def build_dataset(sources, output_directory, config):

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    include_inputs = config.get("include_input_namelist", False)
    output_format  = config.get("output_format", "reduced_h5")

    req_units = config.get("units", "codeunits").lower()
    master_unit_mapping = {}
    master_desc_mapping = {}

    for k, v in TIME_TRACE_LABELS.items():
        master_desc_mapping[k] = v[0]
    for k, v in FIELD_LABELS.items():
        master_desc_mapping[k] = v[0]
    for k, v in FLUX_AVERAGE_ONLY_LABELS.items():
        master_desc_mapping[k] = v[0]
    master_desc_mapping["time"] = "Time"

    if req_units == 'codeunits':
        master_unit_mapping.update(TIME_TRACE_UNITS_CODE)
        master_unit_mapping.update(FIELD_UNITS_CODE)
        master_unit_mapping.update(FLUX_AVERAGE_ONLY_UNITS_CODE)
        master_unit_mapping["time"] = "dummy_conversion_formula"
    else:
        for k, v in TIME_TRACE_LABELS.items():
            master_unit_mapping[k] = v[1]
        for k, v in FIELD_LABELS.items():
            master_unit_mapping[k] = v[1]
        for k, v in FLUX_AVERAGE_ONLY_LABELS.items():
            master_unit_mapping[k] = v[1]
        master_unit_mapping["time"] = "s"

    grouped = group_simulation_directories(sources)

    if not grouped:
        printwarn("Warning: No valid simulation directories were found. Please check your source paths!")
        return

    shared_grid_source       = config.get("shared_grid_source", None)
    shared_inner_wall_points = None

    if shared_grid_source and grouped:
        if shared_grid_source == "first":
            for (m_dir, m_code), m_dirs in grouped.items():
                if m_code == "m3dc1":
                    tmp_adapter = M3DC1Adapter(m_dir)
                    shared_inner_wall_points = tmp_adapter._get_inner_wall_points()
                    tmp_adapter.close()
                    break
        elif os.path.exists(shared_grid_source):
            tmp_adapter = M3DC1Adapter(shared_grid_source)
            shared_inner_wall_points = tmp_adapter._get_inner_wall_points()
            tmp_adapter.close()
        else:
            printwarn(f"Shared grid source '{shared_grid_source}' not found or invalid. Defaulting to local grids.")

    for (model_dir, code), mode_dirs in grouped.items():
        print(model_dir)

        out_name = generate_unique_filename(model_dir, code)
        out_file = output_directory / out_name

        if output_format == "reduced_bp":
            out_file = out_file.with_suffix('.bp')

        base_stem = out_file.stem
        ext       = out_file.suffix
        counter   = 1
        while out_file.exists():
            out_file = output_directory / f"{base_stem}_{counter}{ext}"
            counter += 1

        if output_format == "reduced_h5":
            if code == "m3dc1":
                adapter = M3DC1Adapter(model_dir)
            elif code == "nimrod":
                adapter = NIMRODAdapter(model_dir)
                adapter.set_mode_dirs(mode_dirs)
            else:
                print(f"Skipping {model_dir}: Unsupported code '{code}'")
                continue

            if shared_inner_wall_points is not None:
                adapter.shared_inner_wall_points = shared_inner_wall_points

            metadata = adapter.get_metadata(mode_dirs)
            metadata["units"]         = config.get("units", "codeunits")
            metadata["fcoords"]       = config.get("fcoords", "pest")
            metadata["output_format"] = output_format

            grids        = adapter.extract_grids(config)
            equilibrium  = adapter.extract_equilibrium(config)
            total_fields = adapter.extract_total_fields(config)
            if include_inputs:
                inputs = adapter.get_all_input_parameters()

            with h5py.File(out_file, "w") as h5:

                grp_meta   = h5.create_group("metadata")
                write_dict_as_datasets(grp_meta, metadata)
                grp_hashes = grp_meta.create_group("file_hashes")

                if include_inputs:
                    grp_inputs = h5.create_group("inputs")
                    write_dict_as_datasets(grp_inputs, inputs)

                grp_globals = h5.create_group("global_parameters")
                write_dict_as_datasets(grp_globals, equilibrium["global_parameters"])

                write_array_group(h5, "time_traces", equilibrium["time_traces"],
                                  unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                grp_grids = h5.create_group("grids")
                if "time" in grids:
                    grp_time = grp_grids.create_group("time")
                    write_dict_as_datasets(grp_time, grids["time"])
                if "1d_grid" in grids:
                    grp_grids.create_dataset("1d_grid", data=grids["1d_grid"])
                if "2d_grid" in grids:
                    grp_grids.create_dataset("2d_grid", data=grids["2d_grid"])
                if "3d_grid" in grids:
                    grp_grids.create_dataset("3d_grid", data=grids["3d_grid"])

                grp_eq = h5.create_group("equilibrium")
                write_array_group(grp_eq, "flux_averages", equilibrium["flux_averages"],
                                  unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                write_array_group(grp_eq, "2d_fields", equilibrium["2d_fields"],
                                  unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                write_array_group(grp_eq, "3d_fields", equilibrium["3d_fields"],
                                  unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                if total_fields is not None:
                    grp_tot = h5.create_group("total_fields")
                    write_array_group(grp_tot, "flux_averages", total_fields["flux_averages"],
                                      unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                    write_array_group(grp_tot, "2d_fields", total_fields["2d_fields"],
                                      unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                    write_array_group(grp_tot, "3d_fields", total_fields["3d_fields"],
                                      unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                grp_modes = h5.create_group("perturbations")
                for mode_dir in mode_dirs:
                    mode_name = mode_dir.name

                    repro_hashes  = adapter.get_reproducibility_data(mode_dir)
                    grp_hash_mode = grp_hashes.create_group(mode_name)
                    write_dict_as_datasets(grp_hash_mode, repro_hashes)

                    mode_data    = adapter.extract_mode(mode_dir, config)
                    mode_entries = adapter.get_all_mode_entries(mode_dir, config)
                    for entry_name, entry_meta in mode_entries:
                        if entry_name in grp_modes:
                            printwarn(f"Perturbation group '{entry_name}' already exists; skipping duplicate from {mode_dir}.")
                            continue
                        grp_mode      = grp_modes.create_group(entry_name)
                        grp_mode_meta = grp_mode.create_group("mode_information")
                        write_dict_as_datasets(grp_mode_meta, entry_meta)
                        write_array_group(grp_mode, "1d_profiles", mode_data["1d_profiles"],
                                          unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                        write_array_group(grp_mode, "2d_fields", mode_data["2d_fields"],
                                          unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                        write_array_group(grp_mode, "3d_fields", mode_data["3d_fields"],
                                          unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

        elif output_format == "reduced_bp":
            try:
                import adios2
            except ImportError:
                printwarn("ADIOS2 is not installed. Cannot write .bp files.")
                continue

            if code == "m3dc1":
                adapter = M3DC1Adapter(model_dir)
            elif code == "nimrod":
                adapter = NIMRODAdapter(model_dir)
                adapter.set_mode_dirs(mode_dirs)
            else:
                print(f"Skipping {model_dir}: Unsupported code '{code}'")
                continue

            if shared_inner_wall_points is not None:
                adapter.shared_inner_wall_points = shared_inner_wall_points

            metadata = adapter.get_metadata(mode_dirs)
            metadata["units"]         = config.get("units", "codeunits")
            metadata["fcoords"]       = config.get("fcoords", "pest")
            metadata["output_format"] = output_format

            grids        = adapter.extract_grids(config)
            equilibrium  = adapter.extract_equilibrium(config)
            total_fields = adapter.extract_total_fields(config)
            if include_inputs:
                inputs = adapter.get_all_input_parameters()

            def write_adios_recursive(fh, data_dict, base_path="", unit_mapping=None, desc_mapping=None):
                for key, val in data_dict.items():
                    if val is None:
                        continue
                    path = f"{base_path}/{key}" if base_path else key
                    if isinstance(val, dict):
                        write_adios_recursive(fh, val, path, unit_mapping, desc_mapping)
                    else:
                        if isinstance(val, str):
                            fh.write(path, val)
                        elif np.isscalar(val):
                            fh.write(path, np.array([val]))
                        else:
                            fh.write(path, np.asarray(val))

                        if unit_mapping or desc_mapping:
                            base_key = key
                            if base_key.endswith('_R') or base_key.endswith('_phi') or base_key.endswith('_Z'):
                                base_key = base_key.rsplit('_', 1)[0]

                            if unit_mapping and base_key in unit_mapping:
                                fh.write_attribute('unit', str(unit_mapping[base_key]), variable_name=path)
                            if desc_mapping and base_key in desc_mapping:
                                fh.write_attribute('description', str(desc_mapping[base_key]), variable_name=path)

            with adios2.Stream(str(out_file), "w") as fh:

                write_adios_recursive(fh, metadata, "metadata")

                if include_inputs:
                    write_adios_recursive(fh, inputs, "inputs")

                write_adios_recursive(fh, equilibrium["global_parameters"], "global_parameters")
                write_adios_recursive(fh, equilibrium["time_traces"], "time_traces",
                                      unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                if "time" in grids:
                    write_adios_recursive(fh, grids["time"], "grids/time")
                if "1d_grid" in grids:
                    fh.write("grids/1d_grid", np.asarray(grids["1d_grid"]))
                if "2d_grid" in grids:
                    fh.write("grids/2d_grid", np.asarray(grids["2d_grid"]))
                if "3d_grid" in grids:
                    fh.write("grids/3d_grid", np.asarray(grids["3d_grid"]))

                write_adios_recursive(fh, equilibrium["flux_averages"], "equilibrium/flux_averages",
                                      unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                write_adios_recursive(fh, equilibrium["2d_fields"], "equilibrium/2d_fields",
                                      unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                write_adios_recursive(fh, equilibrium["3d_fields"], "equilibrium/3d_fields",
                                      unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                if total_fields is not None:
                    write_adios_recursive(fh, total_fields["flux_averages"], "total_fields/flux_averages",
                                          unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                    write_adios_recursive(fh, total_fields["2d_fields"], "total_fields/2d_fields",
                                          unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                    write_adios_recursive(fh, total_fields["3d_fields"], "total_fields/3d_fields",
                                          unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

                seen_bp_entries = set()
                for mode_dir in mode_dirs:
                    mode_name = mode_dir.name

                    repro_hashes = adapter.get_reproducibility_data(mode_dir)
                    write_adios_recursive(fh, repro_hashes, f"metadata/file_hashes/{mode_name}")

                    mode_data    = adapter.extract_mode(mode_dir, config)
                    mode_entries = adapter.get_all_mode_entries(mode_dir, config)
                    for entry_name, entry_meta in mode_entries:
                        if entry_name in seen_bp_entries:
                            printwarn(f"Perturbation entry '{entry_name}' already written; skipping duplicate from {mode_dir}.")
                            continue
                        seen_bp_entries.add(entry_name)
                        write_adios_recursive(fh, entry_meta, f"perturbations/{entry_name}/mode_information")
                        write_adios_recursive(fh, mode_data["1d_profiles"], f"perturbations/{entry_name}/1d_profiles",
                                              unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                        write_adios_recursive(fh, mode_data["2d_fields"], f"perturbations/{entry_name}/2d_fields",
                                              unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)
                        write_adios_recursive(fh, mode_data["3d_fields"], f"perturbations/{entry_name}/3d_fields",
                                              unit_mapping=master_unit_mapping, desc_mapping=master_desc_mapping)

        elif output_format == "imas_h5":
            if code == "m3dc1":
                adapter = M3DC1Adapter(model_dir)
                if shared_inner_wall_points is not None:
                    adapter.shared_inner_wall_points = shared_inner_wall_points
                info = adapter.convert_to_imas(mode_dirs, output_directory, config, out_file=out_file)
                adapter.close()
                if info:
                    _record_imas_manifest(output_directory, info)

            elif code == "nimrod":
                adapter = NIMRODAdapter(model_dir)
                adapter.set_mode_dirs(mode_dirs)
                info = adapter.convert_to_imas(mode_dirs, output_directory, config)
                adapter.close()
                if info:
                    _record_imas_manifest(output_directory, info)
                    printnote(f"IMAS entry written: {info.get('entry_directory')}")
                else:
                    printwarn(f"No IMAS output produced for {model_dir}.")

            else:
                printwarn(f"IMAS format not implemented for {code}")
