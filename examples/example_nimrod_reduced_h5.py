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
    "2d_fields": ["B", "ne", "te"],
    "3d_fields": [],
    "flux_averages": "all",
    
    # Choose time traces (1D scalars) to export
    "time_traces": ["Ave_P", "E_K3"],
    
    # Choose total fields on spatial grid to export
    "total_flux_averages": ['p', 'q'],
    "total_2d_fields": ["B", "ne", "te"],
    "total_3d_fields": [],

    # Choose perturbed quantities to export (mode structure).
    # NIMROD writes one /perturbations/nXX entry per toroidal mode stored in the
    # dump (single-n and multi-n runs). The fields are the physical perturbation
    # of that mode, 2 Re[c_n exp(i n phi)], evaluated per mode (no subtraction).
    "mode_1d_profiles": [],
    "mode_2d_fields": ["B", "p"],
    "mode_3d_fields": [],
    "nimrod_include_n0_mode": False,   # write an n00 entry in multi-n runs
    "nimrod_store_complex": False,     # also write <field>_re / <field>_im

    # Poloidal Fourier spectra of the mode eigenfunctions on flux-aligned
    # coordinates of the equilibrium (port of M3D-C1's flux_coordinates()).
    # Written to /perturbations/<nXX>/spectra/<field>/. Omit 'mode_spectra' to disable.
    "mode_spectra": ["p"],             # currently only 'p' is supported
    "spectrum_fcoords": "pest",        # 'pest' (default), 'boozer', 'hamada'; '' = geometric (fallback only)
    "spectrum_points": 200,            # number of theta AND psi points
    "spectrum_phi": 0.0,               # toroidal angle (M3D-C1 convention)
    "spectrum_store_fluxgrid": True,   # also store eigenfunction[theta, psi], R, Z, theta
    "spectrum_include_omega": False,   # Boozer/Hamada only: evaluate at constant zeta instead of phi
    "spectrum_theta_oversample": 1,    # 1 = faithful M3D-C1 algorithm

    # NIMROD flux-coordinate construction (psi is reconstructed from B along rays)
    "nimrod_psi_ray_samples": 800,     # samples per ray (default 4 * spectrum_points)
    # "nimrod_xpoint_guess": [[1.3, -1.1], [1.3, 1.1]],  # optional X-point guesses [R, Z]
    # "nimrod_psi_lcfs": 0.25,         # optional override for psi_LCFS - psi_axis (Wb/rad)
    
    # Grid Specifications
    "shared_grid_source": "first", # Options: 'first', '/path/to/simulation', or None
    "2d_grid": edges_ml.GridSpec("inside_wall", None),
    "3d_grid": edges_ml.GridSpec("rectangular", (128, 16, 128)),
    
    "include_input_namelist": True,
    "output_format": 'reduced_h5', # Options are 'reduced_h5', 'reduced_bp', 'imas_h5'
}

# Explicit mapping of source directories to their underlying codes
data_sources = [
    {
        "code": "nimrod",
        "directories": [
            "nimrod_runs/mastu_45272/run001",
            "nimrod_runs/mastu_45272/run002",
        ],
        "model": None
    }
]

# Run the workflow
edges_ml.build_dataset(
    sources=data_sources,
    output_directory="./datasets",
    config=config
)
