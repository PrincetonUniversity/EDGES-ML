"""
Example: build an edges-ml dataset from REMOTE M3D-C1 data that is registered
in an ADIOS2 campaign archive (.aca).

Nothing is copied up front: fusion-io opens a single persistent SSH connection
to the remote host and streams only the HDF5 variables that are actually
needed, caching them under 'cache_dir'.

Before running, inspect what the archive contains:

    import edges_ml
    edges_ml.list_campaign_simulations("mastu_45272_vped477.aca")
    # ['99/1f_eqrotnc-C_eta_x1/n40', '99/1f_eqrotnc-C_eta_x1/n30', ...]

Notes
-----
* output_format='imas_h5' is NOT supported for remote sources; run the IMAS
  conversion on the remote host instead.
* Slurm logs, C1input namelists and gamma files are not part of the campaign
  archive, so 'inputs' stays empty and the growth rates default to 0.
"""

import edges_ml

config = {
    "units": "mks", # Options are 'mks', 'cgs', or 'codeunits'
    "fcoords": "pest", # Options are 'pest', 'boozer', 'hamada', 'canonical', 'geometric'
    
    # Determine the time slices for which data is exported (e.g., [0, 1, 20])
    "time_slices": [2],

    "resolutions": {
        "1d": 200
    },

    # Choose equilibrium quantities to export
    "2d_fields": ["B", "p"],
    "3d_fields": [],
    "flux_averages": "all",

    # Choose time traces (1D scalars) to export
    "time_traces": ["all"],

    # Choose total fields on spatial grid to export
    "total_flux_averages": ["p", "q", "j"],
    "total_2d_fields": ["B", "p"],
    "total_3d_fields": [],

    # Choose perturbed quantities to export (mode structure)
    "mode_1d_profiles": [],
    "mode_2d_fields": ["p"],
    "mode_3d_fields": [],

    # Grid Specifications
    "shared_grid_source": "first", # Options: 'first', '/path/to/simulation', or None
    "2d_grid": edges_ml.GridSpec("inside_wall", None),
    "3d_grid": edges_ml.GridSpec("rectangular", (64, 16, 64)),

    "include_input_namelist": True,
    "output_format": 'reduced_h5', # Options are 'reduced_h5', 'reduced_bp', 'imas_h5'
}

# Source directories for each code
data_sources = [
    {
        "code": "m3dc1",

        # --- remote access -------------------------------------------------
        "campaign_archive": "mastu_45272_vped477.aca",
        "login": "username@host",

        # Optional: restrict the crawl to these prefixes INSIDE the archive.
        # Omit (or set to None) to use every registered simulation.
        "directories": ["99"],

        # Optional: keep only these model directories.
        "model": "1f_eqrotnc-C_eta_x1",

        # Alternatively, bypass discovery completely:
        # "simulations": ["99/1f_eqrotnc-C_eta_x1/n40"],

        # Optional fusion-io streaming options.
        "cache_dir":     ".cache/aca",
        "remote_python": "python3",
        "remote_setup":  "module load python >/dev/null 2>&1 || true",
        "verbose":       True,
    },
]

# Run the workflow
edges_ml.build_dataset(
    sources=data_sources,
    output_directory="./datasets_remote",
    config=config
)
