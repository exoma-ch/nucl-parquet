"""Isotope-aware tests for the per-isotope catima stopping table (#169).

Verifies:
- Schema includes `proj_A` column
- Inventory covers every Z in 1..92 and includes the documented allowlist isotopes
- Different isotopes of the same Z give *different* dedx at low energy (real
  reduced-mass effect from nuclear stopping)
- Different isotopes of the same Z converge to <0.1% spread at high energy
- Querying an (Z, A) that isn't tabulated raises a clear KeyError
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nucl_parquet import loader as np_lib


@pytest.fixture(scope="module")
def data_dir_path() -> Path:
    return Path("data").resolve()


@pytest.mark.data
def test_catima_schema_has_proj_a(data_dir_path: Path) -> None:
    """catima_stopping must expose a proj_A:int32 column."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "catima" / "catima.parquet")
    assert "proj_A" in df.columns
    assert df["proj_A"].dtype == pl.Int32


@pytest.mark.data
def test_catima_covers_every_z(data_dir_path: Path) -> None:
    """Every Z in 1..92 must have at least one tabulated isotope."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "catima" / "catima.parquet")
    z_present = set(df["proj_Z"].unique().to_list())
    missing = sorted(set(range(1, 93)) - z_present)
    assert not missing, f"Z values without any tabulated isotope: {missing}"


@pytest.mark.data
def test_catima_includes_beam_allowlist(data_dir_path: Path) -> None:
    """The build's beam/medical allowlist must appear in the on-disk table."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "catima" / "catima.parquet")
    pairs = set((int(z), int(a)) for z, a in zip(df["proj_Z"].to_list(), df["proj_A"].to_list()))
    must_have = [
        (1, 3),  # T
        (6, 11),  # C-11
        (6, 14),  # C-14
        (9, 18),  # F-18
        (27, 60),  # Co-60
        (88, 226),  # Ra-226 (canonical for radioactive-only Ra)
        (92, 238),  # U-238
    ]
    missing = [p for p in must_have if p not in pairs]
    assert not missing, f"Allowlist isotopes missing from table: {missing}"


@pytest.mark.data
def test_isotopes_differ_at_low_energy(data_dir_path: Path) -> None:
    """H-1 vs T at 1 keV/u on Cu must differ by > 5% (real nuclear-stopping effect)."""
    db = np_lib.connect(data_dir_path)
    # p (PSTAR) and t (PSTAR), both route to PSTAR — not a catima isotope test.
    # Use the catima path explicitly via _get_catima_table.
    import numpy as np

    from nucl_parquet.loader import _get_catima_table, _interp_loglog

    log_E_p, log_S_p = _get_catima_table(db, proj_Z=1, proj_A=1, target_Z=29)
    log_E_t, log_S_t = _get_catima_table(db, proj_Z=1, proj_A=3, target_Z=29)
    e_per_u = np.array([0.001])
    p_dedx = float(_interp_loglog(log_E_p, log_S_p, e_per_u)[0])
    t_dedx = float(_interp_loglog(log_E_t, log_S_t, e_per_u)[0])
    rel = abs(p_dedx - t_dedx) / max(p_dedx, t_dedx) * 100.0
    assert rel > 5.0, (
        f"H-1 vs H-3 at 0.001 MeV/u on Cu: dedx(p)={p_dedx:.3g}, dedx(t)={t_dedx:.3g}, "
        f"diff={rel:.2f}% — expected >5% from reduced-mass nuclear stopping"
    )


@pytest.mark.data
def test_isotopes_converge_at_high_energy(data_dir_path: Path) -> None:
    """C-12 vs C-14 at 10 MeV/u on Cu must agree to <0.1% (Bethe regime)."""
    db = np_lib.connect(data_dir_path)
    import numpy as np

    from nucl_parquet.loader import _get_catima_table, _interp_loglog

    log_E_12, log_S_12 = _get_catima_table(db, proj_Z=6, proj_A=12, target_Z=29)
    log_E_14, log_S_14 = _get_catima_table(db, proj_Z=6, proj_A=14, target_Z=29)
    e_per_u = np.array([10.0])
    c12 = float(_interp_loglog(log_E_12, log_S_12, e_per_u)[0])
    c14 = float(_interp_loglog(log_E_14, log_S_14, e_per_u)[0])
    rel = abs(c12 - c14) / c12 * 100.0
    assert rel < 0.1, (
        f"C-12 vs C-14 at 10 MeV/u on Cu: dedx(C-12)={c12:.5g}, dedx(C-14)={c14:.5g}, "
        f"diff={rel:.4f}% — expected <0.1% in Bethe regime"
    )


@pytest.mark.data
def test_unknown_isotope_raises_with_clear_message(data_dir_path: Path) -> None:
    """Querying an (Z, A) outside the inventory must raise KeyError listing what *is* available."""
    db = np_lib.connect(data_dir_path)
    # He-5 is unbound — doesn't exist as a nucleus, so it can't be in any table.
    with pytest.raises(KeyError) as exc:
        np_lib.elemental_dedx(db, "he5", 29, 100.0)
    msg = str(exc.value)
    assert "Z=2" in msg
    assert "A=5" in msg
    assert "Available isotopes" in msg
    # He-4 (stable) must be advertised as available
    assert "4" in msg
