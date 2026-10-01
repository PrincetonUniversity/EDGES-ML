import edges_ml

config = {
    "units": "mks", # Options are 'mks', 'cgs', or 'codeunits'
    "fcoords": "pest", # Options are 'pest', 'boozer', 'hamada', 'canonical', 'geometric'
    
    # Determine the time slices for which data is exported (e.g., [0, 1, 20])
    "time_slices": [2],

    "resolutions": {
        "1d": 600
    },
    
    # Choose equilibrium quantities to export
    "2d_fields": [],
    "3d_fields": [],
    "flux_averages": "all",
    
    # Choose time traces (1D scalars) to export
    "time_traces": [],
    
    # Choose total fields on spatial grid to export
    "total_flux_averages": ['p', 'q', 'j'],
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
    "output_format": 'reduced_bp', # Options are 'reduced_h5', 'reduced_bp', 'imas_h5'
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

# Run the workflow
edges_ml.build_dataset(
    sources=data_sources,
    output_directory="./datasets",
    config=config
)
