"""
Example: export equilibrium, total and perturbed quantities from M3D-C1 runs.

Every section (equilibrium, total_fields, perturbations) has the same three
lists: 'flux_averages', '2d_fields' and '3d_fields'.  A poloidal spectrum is
computed for every scalar field listed under 'perturbations' (in any of the
three lists); there is no separate list of spectrum quantities.
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
        "2d_fields": ["te", "ne"],
        "3d_fields": [],
    },

    # Total fields, evaluated at every entry of 'time_slices'
    "total_fields": {
        "flux_averages": [],
        "2d_fields": ["te", "ne"],
        "3d_fields": [],
    },

    # Perturbations (mode structure).  The poloidal spectrum is computed for
    # every scalar quantity listed here.
    "perturbations": {
        "flux_averages": [],
        "2d_fields": ["p"],
        "3d_fields": [],
        "spectrum": {                  # numerical settings only, no quantity list
            "fcoords": "pest",         # 'pest' (default), 'boozer', 'hamada'; '' = geometric
            "points": 200,             # number of theta AND psi points
            "phi": 0.0,                # toroidal angle of the eigenfunction
            "store_fluxgrid": True,    # also store eigenfunction[theta, psi], R, Z, theta
        },
    },

    # Grid specifications (shared by ALL 2D / 3D fields of every section)
    "shared_grid_source": "first",     # 'first', '/path/to/simulation' or None
    "2d_grid": edges_ml.GridSpec("inside_wall", None),
    "3d_grid": edges_ml.GridSpec("rectangular", (64, 16, 64)),

    "include_input_namelist": True,
    "output_format": "reduced_h5",     # 'reduced_h5', 'reduced_bp', 'imas_h5'
}

data_sources = [
    {
        "code": "m3dc1",
        "directories": [
            "mastu_45272/varyped45272_477_MAST_VPED/116",
            "mastu_45272/varyped45272_477_MAST_VPED/102",
            "mastu_45272/varyped45272_477_MAST_VPED/240",
            "mastu_45272/varyped45272_477_MAST_VPED/174",
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
