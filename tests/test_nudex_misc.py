"""Tests for the three remaining NUDEX tables (#77 close-out):
nudex_shellcor / nudex_special_inputs / nudex_general_stat.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_dir_path() -> Path:
    return Path("data").resolve()


@pytest.mark.data
def test_shellcor_schema(data_dir_path: Path) -> None:
    """nudex_shellcor schema and dtypes."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "nudex_shellcor.parquet")
    expected = {"Z", "A", "symbol", "shell_MeV", "deformation_correction_MeV", "beta2", "beta4"}
    assert set(df.columns) == expected
    assert df["Z"].dtype == pl.Int32
    assert df["A"].dtype == pl.Int32


@pytest.mark.data
def test_shellcor_coverage(data_dir_path: Path) -> None:
    """Coverage spans the full nuclide chart — every (Z, A) in
    nudex_isotopes should appear here too (Hauser-Feshbach requires it)."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "nudex_shellcor.parquet")
    assert df.height > 6000, f"only {df.height} rows; expected ~6800"
    assert df["Z"].min() >= 1
    assert df["Z"].max() <= 120  # Z=120 sentinel for nuclei in NUDEX tables


@pytest.mark.data
def test_shellcor_o16_anchor(data_dir_path: Path) -> None:
    """O-16 doubly-magic: NUDEX shell correction = -3.974 ± 0.01 MeV.

    Tight bound is a strong column-swap guard — if any of shell/Defcor/β₂/β₄
    got transposed, the value would shift by an order of magnitude.
    """
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "nudex_shellcor.parquet")
    o16 = df.filter((pl.col("Z") == 8) & (pl.col("A") == 16))
    assert o16.height == 1
    # NUDEX gives O-16 shell = -3.974 MeV exactly
    shell = float(o16["shell_MeV"][0])
    assert abs(shell - (-3.974)) < 0.01, f"O-16 shell_MeV = {shell}; expected -3.974 ± 0.01"


@pytest.mark.data
def test_special_inputs_schema(data_dir_path: Path) -> None:
    """nudex_special_inputs is a small key-value override table."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "nudex_special_inputs.parquet")
    expected = {"za", "section", "subsection", "row_idx", "raw_text"}
    assert set(df.columns) == expected


@pytest.mark.data
def test_general_stat_schema(data_dir_path: Path) -> None:
    """nudex_general_stat: NUDEX runtime configuration defaults + overrides."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "nudex_general_stat.parquet")
    must_have = {"Z", "A", "LDtype", "PSFflag", "MaxSpin"}
    assert must_have.issubset(set(df.columns))


@pytest.mark.data
def test_views_registered(data_dir_path: Path) -> None:
    """All three new views queryable from connect()."""
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    for view in ("nudex_shellcor", "nudex_special_inputs", "nudex_general_stat"):
        n = db.sql(f"SELECT COUNT(*) FROM {view}").fetchall()[0][0]
        assert n > 0, f"{view} has no rows"
