"""
Example: export NIMROD runs into the reduced HDF5 format.

Every section (equilibrium, total_fields, perturbations) has the same three
lists: 'flux_averages', '2d_fields' and '3d_fields'.

NIMROD writes one /perturbations/nXX entry per toroidal mode stored in the
dump (single-n and multi-n runs).  The 2D / 3D perturbation fields are the
physical perturbation of that mode, 2 Re[c_n exp(i n phi)], evaluated per mode
(no subtraction).  A poloidal spectrum is computed for every scalar field
listed under 'perturbations' on flux-aligned coordinates of the equilibrium
(port of M3D-C1's flux_coordinates()); vector fields such as 'B' are written
as fields but get no spectrum.

The output can be inspected with examples/plot_perturbation.py, e.g.
    python examples/plot_perturbation.py datasets_nimrod/<file>.h5 --list
    python examples/plot_perturbation.py datasets_nimrod/<file>.h5 --field p --fluxgrid
"""

import edges_ml

config = {
    "units": "mks",          # 'mks', 'cgs' or 'codeunits'
    "fcoords": "pest",       # 'pest', 'boozer', 'hamada', 'canonical', 'geometric'

    # Time slices for which the total fields are exported (e.g. [0, 10, 20])
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

    # Perturbations (mode structure).  The poloidal spectrum is computed for
    # every scalar quantity listed here (here: 'p').
    "perturbations": {
        "flux_averages": [],
        "2d_fields": ["p"],
        "3d_fields": [],
        "spectrum": {                  # numerical settings only, no quantity list
            "fcoords": "pest",         # 'pest' (default), 'boozer', 'hamada'; '' = geometric
            "points": 200,             # number of theta AND psi points
            "phi": 0.0,                # toroidal angle (M3D-C1 convention)
            "store_fluxgrid": True,    # also store eigenfunction[theta, psi], R, Z, theta
            "include_omega": False,    # Boozer/Hamada only: constant zeta instead of phi
            "theta_oversample": 1,     # 1 = faithful M3D-C1 algorithm
        },
    },

    # NIMROD-specific perturbation options
    "nimrod_include_n0_mode": False,   # write an n00 entry in multi-n runs
    "nimrod_store_complex": False,     # also write <field>_re / <field>_im

    # NIMROD flux-coordinate construction (psi is reconstructed from B along rays)
    "nimrod_psi_ray_samples": 800,     # samples per ray (default 4 * spectrum points)
    # "nimrod_xpoint_guess": [[1.3, -1.1], [1.3, 1.1]],  # optional X-point guesses [R, Z]
    # "nimrod_psi_lcfs": 0.25,         # optional override for psi_LCFS - psi_axis (Wb/rad)

    # Grid specifications (shared by ALL 2D / 3D fields of every section)
    "shared_grid_source": "first",     # 'first', '/path/to/simulation' or None
    "2d_grid": edges_ml.GridSpec("inside_wall", None),
    "3d_grid": edges_ml.GridSpec("rectangular", (64, 2, 64)),

    "include_input_namelist": False,
    "output_format": "reduced_h5",     # 'reduced_h5', 'reduced_bp', 'imas_h5'
}

# Explicit mapping of source directories to their underlying codes
data_sources = [
    {
        "code": "nimrod",
        "directories": ["nimrod_runs/mastu_45272/run001", "nimrod_runs/mastu_45272/run002"],
        "model": None
    }
]

if __name__ == "__main__":
    edges_ml.build_dataset(
        sources=data_sources,
        output_directory="./datasets_nimrod",
        config=config
    )
