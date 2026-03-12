"""nucl-parquet: Nuclear data as Parquet — queryable with DuckDB."""

from .download import data_dir, download
from .loader import (
    COINCIDENCE_SQL,
    DECAY_CHAIN_SQL,
    GAMMA_LINES_SQL,
    IDENTIFY_GAMMA_SQL,
    compound_dedx,
    connect,
    elemental_dedx,
    linear_dedx,
)

__all__ = [
    "COINCIDENCE_SQL",
    "DECAY_CHAIN_SQL",
    "GAMMA_LINES_SQL",
    "IDENTIFY_GAMMA_SQL",
    "compound_dedx",
    "connect",
    "data_dir",
    "download",
    "elemental_dedx",
    "linear_dedx",
]
