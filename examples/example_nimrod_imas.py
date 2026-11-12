"""
Example: convert NIMROD runs into IMAS with the nimrod2imas tools.

Note: for output_format='imas_h5' the nimrod2imas tools (input2imas,
dump2imas, gamma2imas) decide which quantities are written.  The
'equilibrium' / 'total_fields' / 'perturbations' sections below are kept so
that the same configuration can be used with 'reduced_h5' / 'reduced_bp'.
"""

import edges_ml

config = {
    "units": "mks",          # 'mks', 'cgs' or 'codeunits'
    "fcoords": "pest",       # 'pest', 'boozer', 'hamada', 'canonical', 'geometric'

    # Time slices to export (also used as the default dump selection for dump2imas)
    "time_slices": [2],

    "resolutions": {
        "1d": 600
    },

    # Time traces (1D scalars) to export
    "time_traces": [],

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
        "3d_fields": ["p"],
    },

    # Perturbations (mode structure); spectra are computed for 'p'
    "perturbations": {
        "flux_averages": [],
        "2d_fields": ["p"],
        "3d_fields": [],
    },

    # Grid specifications (shared by ALL 2D / 3D fields of every section)
    "shared_grid_source": "first",     # 'first', '/path/to/simulation' or None
    "2d_grid": edges_ml.GridSpec("inside_wall", None),
    "3d_grid": edges_ml.GridSpec("rectangular", (64, 2, 64)),

    "include_input_namelist": True,
    "output_format": "imas_h5",        # 'reduced_h5', 'reduced_bp', 'imas_h5'
}

# Explicit mapping of source directories to their underlying codes
data_sources = [
    {
        "code": "nimrod",
        "directories": [
            "nimrod_runs/nstx_132543/v04",
        ],
        "model": None
    }
]

if __name__ == "__main__":
    edges_ml.build_dataset(
        sources=data_sources,
        output_directory="./datasets",
        config=config
    )
