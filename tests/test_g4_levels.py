"""Tests for the G4 PhotonEvaporation → meta/ensdf/levels/ converter (issue #70).

Unit tests cover the pure transform helpers (``encode_jp``, ``convert_half_life``);
``@pytest.mark.data`` spot checks confirm acceptance-criterion isotopes
(Eu-152m2, Hf-178m2, Tc-99 g.s./Tc-99m, He-4/Pb-208 stable) round-trip correctly
through the per-element parquet output.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.photon_evap_levels import (
    HALF_LIFE_STABLE_SENTINEL,
    JPI_UNKNOWN,
    Z_TO_SYMBOL,
    convert_half_life,
    encode_jp,
)

_REPO_ROOT = Path(__file__).parent.parent
_LEVELS_DIR = _REPO_ROOT / "data" / "meta" / "ensdf" / "levels"


# --------------------------------------------------------------------------- unit tests


class TestEncodeJp:
    """``encode_jp`` decodes G4's J × sign(parity) into ENSDF-style strings."""

    def test_integer_spin_positive_parity(self) -> None:
        assert encode_jp(3.0) == "3+"
        assert encode_jp(16.0) == "16+"

    def test_integer_spin_negative_parity(self) -> None:
        # Eu-152m2 is "8-" (jpi = -8.0) — acceptance-criterion case.
        assert encode_jp(-8.0) == "8-"
        assert encode_jp(-3.0) == "3-"

    def test_half_integer_spins(self) -> None:
        # Tc-99 g.s. = 9/2+ (jpi=4.5); Tc-99m = 1/2- (jpi=-0.5).
        assert encode_jp(4.5) == "9/2+"
        assert encode_jp(-0.5) == "1/2-"
        assert encode_jp(1.5) == "3/2+"
        assert encode_jp(-7.5) == "15/2-"

    def test_zero_spin_signed(self) -> None:
        # IEEE-754 signed zero distinguishes 0+ from 0-; Pb-208 ground = 0+.
        assert encode_jp(0.0) == "0+"
        assert encode_jp(-0.0) == "0-"

    def test_unknown_sentinel(self) -> None:
        assert encode_jp(JPI_UNKNOWN) is None
        assert encode_jp(99.0) is None

    def test_nan_and_none(self) -> None:
        assert encode_jp(float("nan")) is None
        assert encode_jp(None) is None  # type: ignore[arg-type]


class TestConvertHalfLife:
    """``convert_half_life`` only rewrites the ``-1`` stable sentinel; pass-through otherwise."""

    def test_stable_sentinel_maps_to_positive_infinity(self) -> None:
        result = convert_half_life(HALF_LIFE_STABLE_SENTINEL)
        assert math.isinf(result) and result > 0

    def test_finite_half_life_passes_through(self) -> None:
        # PhotonEvaporation already stores half-life in seconds — no rescaling.
        assert convert_half_life(5760.0) == 5760.0
        assert convert_half_life(9.78286e8) == 9.78286e8

    def test_zero_passes_through(self) -> None:
        # G4 prompt sentinel — keep as 0.0 (per ADR-0002).
        assert convert_half_life(0.0) == 0.0

    def test_does_not_corrupt_minus_one_via_arithmetic(self) -> None:
        # Boundary: a naive `mean_life * ln(2) * 1e-9` would yield ~-6.93e-10.
        # We must intercept the sentinel first, returning exactly +inf.
        assert convert_half_life(-1.0) == math.inf


class TestZToSymbol:
    """Z→symbol mapping covers every element strata can emit (1..118)."""

    def test_covers_full_periodic_table(self) -> None:
        assert set(Z_TO_SYMBOL.keys()) == set(range(1, 119))

    def test_capitalization_matches_iupac(self) -> None:
        # Single-letter elements: capital only; two-letter: capital + lowercase.
        for sym in Z_TO_SYMBOL.values():
            assert sym[0].isupper(), f"symbol {sym!r} must start uppercase"
            if len(sym) == 2:
                assert sym[1].islower(), f"symbol {sym!r} must have lowercase tail"

    def test_known_symbols(self) -> None:
        assert Z_TO_SYMBOL[1] == "H"
        assert Z_TO_SYMBOL[7] == "N"
        assert Z_TO_SYMBOL[63] == "Eu"
        assert Z_TO_SYMBOL[72] == "Hf"
        assert Z_TO_SYMBOL[82] == "Pb"


# --------------------------------------------------------------------------- @data spot checks


def _read_levels(symbol: str) -> pl.DataFrame:
    path = _LEVELS_DIR / f"{symbol}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built")
    return pl.read_parquet(path)


@pytest.mark.data
def test_eu152m2_long_lived_isomer() -> None:
    """Eu-152m2 must appear at 147.86 keV with t1/2 = 5760 s, jp 8-, jpi -8.0."""
    eu = _read_levels("Eu")
    rows = eu.filter((pl.col("A") == 152) & (pl.col("energy_keV") == 147.86))
    assert rows.height == 1
    row = rows.row(0, named=True)
    assert row["half_life_s"] == 5760.0
    assert row["jp"] == "8-"
    assert row["jpi"] == pytest.approx(-8.0)


@pytest.mark.data
def test_hf178m2_high_spin_isomer() -> None:
    """Hf-178m2 must appear at ~2446 keV with t1/2 ≈ 9.78e8 s (~31 yr).

    Hf-178 ground state is also stable (+inf), so we exclude it explicitly
    rather than filtering on raw half-life range.
    """
    hf = _read_levels("Hf")
    rows = hf.filter(
        (pl.col("A") == 178)
        & (pl.col("level_idx") > 0)
        & pl.col("half_life_s").is_finite()
        & (pl.col("half_life_s") > 1e6)
    )
    assert rows.height == 1, "expected exactly one Hf-178 finite long-lived level"
    row = rows.row(0, named=True)
    assert row["energy_keV"] == pytest.approx(2446.09, abs=0.05)
    assert row["half_life_s"] == pytest.approx(9.78e8, rel=1e-3)
    # jp encoding for K-isomer 16+.
    assert row["jp"] == "16+"


@pytest.mark.data
def test_stable_ground_states_use_positive_infinity() -> None:
    """He-4 and Pb-208 are stable: half_life_s must be +inf, NOT NULL."""
    for symbol, a in [("He", 4), ("Pb", 208)]:
        df = _read_levels(symbol)
        ground = df.filter((pl.col("A") == a) & (pl.col("level_idx") == 0))
        assert ground.height == 1
        hl = ground["half_life_s"].item()
        assert hl is not None
        assert math.isinf(hl) and hl > 0, f"{symbol}-{a} ground must be +inf, got {hl!r}"


@pytest.mark.data
def test_unknown_jpi_decodes_to_null_jp() -> None:
    """Every level with jpi == 99 must have jp == NULL (and 99 must round-trip in raw column)."""
    df = pl.read_parquet(_LEVELS_DIR / "*.parquet")
    unknown = df.filter(pl.col("jpi") == 99.0)
    assert unknown.height > 0  # sanity — strata input has 51k such rows
    assert unknown["jp"].null_count() == unknown.height


@pytest.mark.data
def test_per_element_partitioning_and_filenames() -> None:
    """Output is one parquet per element; filenames match capitalized IUPAC symbol."""
    files = sorted(_LEVELS_DIR.glob("*.parquet"))
    # Strata input covers Z=1..117; expect at least ~99 files (acceptance criterion).
    assert len(files) >= 99
    # Filenames must be valid symbols and unique by Z.
    seen_z: set[int] = set()
    sym_to_z = {sym: z for z, sym in Z_TO_SYMBOL.items()}
    for path in files:
        symbol = path.stem
        assert symbol in sym_to_z, f"unexpected file {path.name}"
        z = sym_to_z[symbol]
        assert z not in seen_z, f"duplicate Z={z} ({symbol})"
        seen_z.add(z)
        df = pl.read_parquet(path)
        # Every row in {Symbol}.parquet must have matching Z.
        unique_z = df["Z"].unique().to_list()
        assert unique_z == [z], f"{path.name}: expected Z={z}, got {unique_z}"
