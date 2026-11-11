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
    "2d_fields": ["te", "ne"],
    "3d_fields": [],
    "flux_averages": "all",
    
    # Choose time traces (1D scalars) to export
    "time_traces": ["all"],
    
    # Choose total fields on spatial grid to export
    "total_flux_averages": [],
    "total_2d_fields": ["te", "ne"],
    "total_3d_fields": [],

    # Choose perturbed quantities to export (mode structure)
    "mode_1d_profiles": [],
    "mode_2d_fields": ["B"],
    "mode_3d_fields": [],

    # Poloidal Fourier spectra of the mode eigenfunctions
    # (same algorithm as m3dc1.eigenfunction.eigenfunction(fourier=True)).
    # Written to /perturbations/<nXX>/spectra/<field>/. Omit 'mode_spectra' to disable.
    "mode_spectra": ["p"],             # currently only 'p' is supported
    "spectrum_fcoords": "pest",        # 'pest' (default), 'boozer', 'hamada'; '' = geometric (fallback only)
    "spectrum_points": 200,            # number of theta AND psi points
    "spectrum_phi": 0.0,               # toroidal angle at which the eigenfunction is evaluated
    "spectrum_store_fluxgrid": True,   # also store eigenfunction[theta, psi], R, Z, theta
    
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
        "directories": [
            "mastu_45272/varyped45272_477_MAST_VPED/116",
            "mastu_45272/varyped45272_477_MAST_VPED/102",
            "mastu_45272/varyped45272_477_MAST_VPED/240",
            "mastu_45272/varyped45272_477_MAST_VPED/174",
        ],
        "model": "1f_eqrotnc-ion_eta_x1"
    }
]

# Run the workflow
edges_ml.build_dataset(
    sources=data_sources,
    output_directory="./datasets",
    config=config
)
