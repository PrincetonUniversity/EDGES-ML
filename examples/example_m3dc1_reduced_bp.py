"""
Example: export M3D-C1 data into the reduced ADIOS2 (.bp) format.

Every section (equilibrium, total_fields, perturbations) has the same three
lists: 'flux_averages', '2d_fields' and '3d_fields'.  A poloidal spectrum is
computed for every scalar field listed under 'perturbations'.
"""

import edges_ml

config = {
    "units": "mks",          # 'mks', 'cgs' or 'codeunits'
    "fcoords": "pest",       # 'pest', 'boozer', 'hamada', 'canonical', 'geometric'

    # Time slices for which the total fields are exported (e.g. [0, 1, 20])
    "time_slices": [2],

    "resolutions": {
        "1d": 600
    },

    # Time traces (1D scalars) to export
    "time_traces": [],

    # Equilibrium quantities
    "equilibrium": {
        "flux_averages": "all",
        "2d_fields": [],
        "3d_fields": [],
    },

    # Total fields, evaluated at every entry of 'time_slices'
    "total_fields": {
        "flux_averages": ["p", "q", "j"],
        "2d_fields": ["B", "p"],
        "3d_fields": [],
    },

    # Perturbations (mode structure); spectra are computed for 'p'
    "perturbations": {
        "flux_averages": [],
        "2d_fields": ["p"],
        "3d_fields": [],
        "spectrum": {
            "fcoords": "pest",
            "points": 200,
            "phi": 0.0,
            "store_fluxgrid": False,
        },
    },

    # Grid specifications (shared by ALL 2D / 3D fields of every section)
    "shared_grid_source": "first",     # 'first', '/path/to/simulation' or None
    "2d_grid": edges_ml.GridSpec("inside_wall", None),
    "3d_grid": edges_ml.GridSpec("rectangular", (64, 16, 64)),

    "include_input_namelist": True,
    "output_format": "reduced_bp",     # 'reduced_h5', 'reduced_bp', 'imas_h5'
}

# Explicit mapping of source directories to their underlying codes
data_sources = [
    {
        "code": "m3dc1",
        "directories": [
            "nstx_132543/varyped/varyped132543_700_kEFIT_Walter/70/",
            "nstx_132543/varyped/varyped132543_700_kEFIT_Walter/109/",
        ],
        "model": "1f_eqrotnc-ion_eta_x1"
    }
]

if __name__ == "__main__":
    edges_ml.build_dataset(
        sources=data_sources,
        output_directory="./datasets",
        config=config
    )
