"""Expected Parquet column schemas for validation."""

from __future__ import annotations

XS_SCHEMA = {
    "target_A": "Int32",
    "residual_Z": "Int32",
    "residual_A": "Int32",
    "state": "Utf8",
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
}

STOPPING_SCHEMA = {
    "source": "Utf8",
    "target_Z": "Int32",
    "energy_MeV": "Float64",
    "dedx": "Float64",
}

ABUNDANCES_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "symbol": "Utf8",
    "abundance": "Float64",
    "atomic_mass": "Float64",
}

DECAY_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "state": "Utf8",
    "half_life_s": "Float64",
    "decay_mode": "Utf8",
    "daughter_Z": "Int32",
    "daughter_A": "Int32",
    "daughter_state": "Utf8",
    "branching": "Float64",
}

ELEMENTS_SCHEMA = {
    "Z": "Int32",
    "symbol": "Utf8",
}

EXFOR_SCHEMA = {
    "exfor_entry": "Utf8",
    "target_Z": "Int32",
    "target_A": "Int32",
    "residual_Z": "Int32",
    "residual_A": "Int32",
    "state": "Utf8",
    "energy_MeV": "Float64",
    "energy_err_MeV": "Float64",
    "xs_mb": "Float64",
    "xs_err_mb": "Float64",
    "author": "Utf8",
    "year": "Int32",
}
