"""Normalization helpers for cross-language golden-file diffs.

See `docs/NULL_CONVENTIONS.md` for the rules these encode.
"""

from __future__ import annotations

import math
from typing import Any

_FLOAT_PRECISION = 6


def _round_float(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            raise ValueError("NaN encountered in golden — shipped data must not contain NaN")
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return round(v, _FLOAT_PRECISION)
    return v


def _empty_str_or_null(v: Any) -> Any:
    """Empty strings stay; nulls stay; everything else passes through."""
    return v


def _normalize_row(row: dict[str, Any], float_cols: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in float_cols:
            out[k] = _round_float(v)
        else:
            out[k] = _empty_str_or_null(v)
    return out


_COINC_FLOAT_COLS = {
    "daughter_ex_keV",
    "emission1_energy_keV",
    "emission1_intensity",
    "emission2_energy_keV",
    "emission2_intensity",
    "pair_intensity",
}
_COINC_SORT_KEY = (
    "Z",
    "A",
    "parent_state",
    "parent_decay_mode",
    "emission1_rad_type",
    "emission1_energy_keV",
    "emission2_rad_type",
    "emission2_energy_keV",
    # Tie-breakers: same parent channel + same emission energies can repeat
    # across distinct cascade rows (different daughter_ex_keV / shells).
    "daughter_ex_keV",
    "emission1_shell",
    "emission2_shell",
    "pair_intensity",
)


def normalize_coincidences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort coincidence rows by canonical key; round floats."""
    normed = [_normalize_row(r, _COINC_FLOAT_COLS) for r in rows]
    return sorted(normed, key=lambda r: tuple(_sortable(r.get(k)) for k in _COINC_SORT_KEY))


_EMISSION_FLOAT_COLS = {"energy_keV", "intensity_pct", "icc_total"}
_EMISSION_SORT_KEY = ("Z", "A", "state", "rad_type", "energy_keV")


def normalize_emissions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normed = [_normalize_row(r, _EMISSION_FLOAT_COLS) for r in rows]
    return sorted(normed, key=lambda r: tuple(_sortable(r.get(k)) for k in _EMISSION_SORT_KEY))


_GAMMA_FLOAT_COLS = {"E_gamma_keV", "delta_keV", "intensity_pct", "energy_keV", "half_life_s"}


def normalize_gamma_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normed = [_normalize_row(r, _GAMMA_FLOAT_COLS) for r in rows]

    # Per docs/NULL_CONVENTIONS.md identify_gamma sort: (|delta_keV|, -intensity_pct).
    # `delta_keV` is the COINCIDENCE_SQL-style column; some queries return E_gamma_keV
    # without a delta. Fall back to E_gamma_keV ↔ a virtual energy diff of 0.
    def _key(r: dict[str, Any]) -> tuple[float, float]:
        delta = r.get("delta_keV")
        if delta is None:
            delta = 0.0
        if isinstance(delta, str):  # e.g. "Infinity"
            delta = float("inf") if delta == "Infinity" else 0.0
        intensity = r.get("intensity_pct") or 0.0
        if isinstance(intensity, str):
            intensity = 0.0
        return (abs(delta), -intensity)

    return sorted(normed, key=_key)


def _sortable(v: Any) -> tuple[int, Any]:
    """Stable sort that handles None / mixed types."""
    if v is None:
        return (0, "")
    if isinstance(v, str):
        return (1, v)
    return (2, v)
