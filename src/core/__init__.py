"""Core algorithms: TDOA localization and signal processing."""
from .gcc import gcc_phat, compute_tdoa_from_gcc
from .iq_handler import (
    generate_signal,
    generate_chirp,
    load_iq_file,
    load_iq_dataset,
    parse_metadata,
    save_iq_file,
    write_metadata,
)
from .tdoa import solve_tdoa_pairwise, tdoa_pairwise_residual
from .utils import (
    generate_pairwise_tdoa,
    get_bounds,
    hyperbola_field_2d,
    plot_hyperbola_2d,
    match_fig_aspect,
    lonlat_to_xy_m,
    xy_m_to_lonlat,
)

__all__ = [
    # GCC functions
    "gcc_phat",
    "compute_tdoa_from_gcc",
    # IQ signal functions
    "generate_signal",
    "generate_chirp",
    "load_iq_file",
    "load_iq_dataset",
    "parse_metadata",
    "save_iq_file",
    "write_metadata",
    # TDoA localization
    "solve_tdoa_pairwise",
    "tdoa_pairwise_residual",
    "generate_pairwise_tdoa",
    # Utilities
    "get_bounds",
    "hyperbola_field_2d",
    "plot_hyperbola_2d",
    "match_fig_aspect",
    "lonlat_to_xy_m",
    "xy_m_to_lonlat",
]
