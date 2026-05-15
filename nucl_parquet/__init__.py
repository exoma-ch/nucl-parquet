"""nucl-parquet: Nuclear data as Parquet — queryable with DuckDB."""

from .download import compute_data_sha256, data_dir, data_sha256, data_version, download
from .loader import (
    COINCIDENCE_SQL,
    DECAY_CHAIN_SQL,
    GAMMA_LINES_ALL_SQL,
    GAMMA_LINES_SQL,
    IDENTIFY_GAMMA_ALL_SQL,
    IDENTIFY_GAMMA_SQL,
    coincidences,
    compound_dedx,
    connect,
    elemental_dedx,
    gamma_lines,
    identify_gamma,
    linear_dedx,
)

__all__ = [
    "COINCIDENCE_SQL",
    "DECAY_CHAIN_SQL",
    "GAMMA_LINES_ALL_SQL",
    "GAMMA_LINES_SQL",
    "IDENTIFY_GAMMA_ALL_SQL",
    "IDENTIFY_GAMMA_SQL",
    "coincidences",
    "compound_dedx",
    "compute_data_sha256",
    "connect",
    "data_dir",
    "data_sha256",
    "data_version",
    "download",
    "elemental_dedx",
    "gamma_lines",
    "identify_gamma",
    "linear_dedx",
]
