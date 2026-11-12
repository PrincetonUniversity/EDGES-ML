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
* A poloidal spectrum is computed for every scalar field listed under
  'perturbations'.  This relies on m3dc1's flux_coordinates() working with the
  streamed data.  If it fails for a model, the spectra of that model are
  skipped with an error message; all other output is still written.
  The toroidal mode number is taken from the simulation path (e.g. '.../n40').
"""

import edges_ml

config = {
    "units": "mks",          # 'mks', 'cgs' or 'codeunits'
    "fcoords": "pest",       # 'pest', 'boozer', 'hamada', 'canonical', 'geometric'

    # Time slices for which the total fields are exported (e.g. [0, 1, 20])
    "time_slices": [2],

    "resolutions": {
        "1d": 200
    },

    # Time traces (1D scalars) to export
    "time_traces": ["all"],

    # Equilibrium quantities
    "equilibrium": {
        "flux_averages": "all",
        "2d_fields": ["B", "p"],
        "3d_fields": [],
    },

    # Total fields, evaluated at every entry of 'time_slices'
    "total_fields": {
        "flux_averages": ["p", "q", "j"],
        "2d_fields": ["B", "p"],
        "3d_fields": [],
    },

    # Perturbations (mode structure); spectra are computed for 'p'.
    # A lower spectrum resolution reduces the amount of streamed data.
    "perturbations": {
        "flux_averages": [],
        "2d_fields": ["p"],
        "3d_fields": [],
        "spectrum": {
            "fcoords": "pest",         # 'pest' (default), 'boozer', 'hamada'; '' = geometric
            "points": 128,             # number of theta AND psi points
            "phi": 0.0,                # toroidal angle of the eigenfunction
            "store_fluxgrid": False,   # True: also store eigenfunction[theta, psi], R, Z, theta
        },
    },

    # Grid specifications (shared by ALL 2D / 3D fields of every section)
    "shared_grid_source": "first",     # 'first', '/path/to/simulation' or None
    "2d_grid": edges_ml.GridSpec("inside_wall", None),
    "3d_grid": edges_ml.GridSpec("rectangular", (64, 16, 64)),

    "include_input_namelist": True,
    "output_format": "reduced_h5",     # 'reduced_h5', 'reduced_bp' ('imas_h5' not supported remotely)
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

if __name__ == "__main__":
    edges_ml.build_dataset(
        sources=data_sources,
        output_directory="./datasets_remote",
        config=config
    )
