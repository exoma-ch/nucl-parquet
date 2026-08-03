"""Unit tests for nucl_parquet.loader — no data files required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nucl_parquet.loader import (
    COINCIDENCE_SQL,
    DECAY_CHAIN_SQL,
    GAMMA_LINES_SQL,
    IDENTIFY_GAMMA_SQL,
    _interp_loglog,
    connect,
    elemental_dedx,
)


@pytest.fixture()
def mini_data(tmp_path: Path) -> Path:
    """Create minimal parquet files for loader tests."""
    import json

    # catalog.json
    catalog = {
        "version": 1,
        "libraries": {
            "test-lib": {
                "name": "Test",
                "projectiles": ["p"],
                "data_type": "cross_sections",
                "path": "test-lib/xs/",
            },
        },
        "views": {
            "abundances": {"path": "meta/abundances.parquet", "type": "file"},
            "decay": {"path": "meta/decay.parquet", "type": "file"},
            "elements": {"path": "meta/elements.parquet", "type": "file"},
            "stopping": {"path": "stopping", "type": "glob"},
        },
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))

    # meta/
    meta = tmp_path / "meta"
    meta.mkdir()
    pl.DataFrame(
        {"Z": [29], "A": [63], "symbol": ["Cu"], "abundance": [0.6915], "atomic_mass": [62.93]},
        schema={"Z": pl.Int32, "A": pl.Int32, "symbol": pl.Utf8, "abundance": pl.Float64, "atomic_mass": pl.Float64},
    ).write_parquet(meta / "abundances.parquet")

    pl.DataFrame(
        {
            "Z": [30],
            "A": [65],
            "state": [""],
            "half_life_s": [244.0],
            "decay_mode": ["EC"],
            "daughter_Z": [29],
            "daughter_A": [65],
            "daughter_state": [""],
            "branching": [1.0],
        },
        schema={
            "Z": pl.Int32,
            "A": pl.Int32,
            "state": pl.Utf8,
            "half_life_s": pl.Float64,
            "decay_mode": pl.Utf8,
            "daughter_Z": pl.Int32,
            "daughter_A": pl.Int32,
            "daughter_state": pl.Utf8,
            "branching": pl.Float64,
        },
    ).write_parquet(meta / "decay.parquet")

    pl.DataFrame(
        {"Z": [29, 30], "symbol": ["Cu", "Zn"]},
        schema={"Z": pl.Int32, "symbol": pl.Utf8},
    ).write_parquet(meta / "elements.parquet")

    # stopping/
    stopping = tmp_path / "stopping"
    stopping.mkdir()
    pl.DataFrame(
        {
            "source": ["PSTAR"] * 4,
            "target_Z": [29] * 4,
            "energy_MeV": [0.1, 1.0, 10.0, 100.0],
            "dedx": [200.0, 50.0, 20.0, 10.0],
        },
        schema={"source": pl.Utf8, "target_Z": pl.Int32, "energy_MeV": pl.Float64, "dedx": pl.Float64},
    ).write_parquet(stopping / "stopping.parquet")

    # xs library
    xs_dir = tmp_path / "test-lib" / "xs"
    xs_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "target_A": [63, 63],
            "residual_Z": [30, 30],
            "residual_A": [63, 63],
            "state": ["", ""],
            "energy_MeV": [10.0, 20.0],
            "xs_mb": [100.0, 200.0],
        },
        schema={
            "target_A": pl.Int32,
            "residual_Z": pl.Int32,
            "residual_A": pl.Int32,
            "state": pl.Utf8,
            "energy_MeV": pl.Float64,
            "xs_mb": pl.Float64,
        },
    ).write_parquet(xs_dir / "p_Cu.parquet")

    return tmp_path


@pytest.fixture()
def canonical_data(tmp_path: Path) -> Path:
    """A library in canonical form: two projectiles, two elements, one file each.

    Deliberately mirrors the real layout, where a library is many
    `<projectile>_<Element>.parquet` files that the `xs` view globs together.
    """
    import json

    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "version": 1,
                "libraries": {
                    "lib-a": {
                        "name": "A",
                        "projectiles": ["n", "p"],
                        "data_type": "cross_sections",
                        "path": "lib-a/xs/",
                    }
                },
                "views": {},
            }
        )
    )
    xs_dir = tmp_path / "lib-a" / "xs"
    xs_dir.mkdir(parents=True)

    schema = {
        "library": pl.Utf8,
        "kind": pl.Utf8,
        "projectile": pl.Utf8,
        "proj_Z": pl.Int32,
        "proj_A": pl.Int32,
        "target_Z": pl.Int32,
        "target_A": pl.Int32,
        "MT": pl.Int32,
        "residual_Z": pl.Int32,
        "residual_A": pl.Int32,
        "energy_MeV": pl.Float64,
        "xs_mb": pl.Float64,
    }
    # Same target_A (56) on two different elements, and the same nuclide hit by
    # two different beams — the two ways rows get conflated when identity is
    # taken from the file path instead of the row.
    rows = [
        ("n", 0, 1, 26, 56, 100.0),
        ("p", 1, 1, 26, 56, 200.0),
        ("n", 0, 1, 25, 56, 300.0),  # isobar: same A, different Z
    ]
    for proj, pz, pa, tz, ta, xs in rows:
        sym = {26: "Fe", 25: "Mn"}[tz]
        df = pl.DataFrame(
            {
                "library": ["lib-a"],
                "kind": ["production"],
                "projectile": [proj],
                "proj_Z": [pz],
                "proj_A": [pa],
                "target_Z": [tz],
                "target_A": [ta],
                "MT": [None],
                "residual_Z": [27],
                "residual_A": [56],
                "energy_MeV": [10.0],
                "xs_mb": [xs],
            },
            schema=schema,
        )
        path = xs_dir / f"{proj}_{sym}.parquet"
        if path.exists():
            df = pl.concat([pl.read_parquet(path), df])
        df.write_parquet(path)
    return tmp_path


def test_xs_view_does_not_merge_projectiles(canonical_data: Path) -> None:
    """Filtering the unified view by target must not pull in other beams.

    `projectile` lived only in the filename, so `WHERE target_Z=26 AND target_A=56`
    returned the neutron *and* proton rows as one result set, with nothing in the
    payload to tell them apart. Principle 5: rows are self-describing.
    """
    db = connect(canonical_data)
    got = db.sql("SELECT projectile, xs_mb FROM xs WHERE target_Z=26 AND target_A=56 ORDER BY projectile").fetchall()
    assert got == [("n", 100.0), ("p", 200.0)]
    only_n = db.sql("SELECT xs_mb FROM xs WHERE target_Z=26 AND target_A=56 AND projectile='n'").fetchall()
    assert only_n == [(100.0,)]


def test_xs_view_separates_isobars(canonical_data: Path) -> None:
    """target_A alone does not identify a target — Fe-56 and Mn-56 share it (#273)."""
    db = connect(canonical_data)
    got = db.sql("SELECT target_Z, xs_mb FROM xs WHERE target_A=56 AND projectile='n' ORDER BY target_Z").fetchall()
    assert got == [(25, 300.0), (26, 100.0)]


def test_xs_view_unions_files_with_differing_column_order(canonical_data: Path) -> None:
    """A library is globbed from many files; positional union silently misaligns.

    The view must read by name, so a file written with a different column order
    still lands in the right columns rather than shifting energy into xs.
    """
    xs_dir = canonical_data / "lib-a" / "xs"
    df = pl.read_parquet(xs_dir / "n_Fe.parquet")
    reordered = df.select(sorted(df.columns))  # alphabetical, not schema order
    reordered.with_columns(pl.lit(92).cast(pl.Int32).alias("target_Z")).write_parquet(xs_dir / "n_U.parquet")

    db = connect(canonical_data)
    row = db.sql("SELECT projectile, target_A, energy_MeV, xs_mb FROM xs WHERE target_Z=92").fetchone()
    assert row == ("n", 56, 10.0, 100.0)


def test_connect_creates_views(mini_data: Path) -> None:
    db = connect(mini_data)
    views = {
        r[0] for r in db.sql("SELECT table_name FROM information_schema.tables WHERE table_type='VIEW'").fetchall()
    }
    assert "test_lib" in views
    assert "xs" in views
    assert "abundances" in views
    assert "decay" in views
    assert "elements" in views
    assert "stopping" in views


def test_xs_query(mini_data: Path) -> None:
    db = connect(mini_data)
    result = db.sql("SELECT * FROM xs WHERE target_A=63 AND residual_Z=30").fetchall()
    assert len(result) == 2


def test_unified_xs_has_library_column(mini_data: Path) -> None:
    db = connect(mini_data)
    row = db.sql("SELECT library FROM xs LIMIT 1").fetchone()
    assert row[0] == "test-lib"


def test_abundances_query(mini_data: Path) -> None:
    db = connect(mini_data)
    result = db.sql("SELECT * FROM abundances WHERE Z=29").fetchone()
    assert result is not None
    assert abs(result[3] - 0.6915) < 1e-6  # abundance column


def test_elemental_dedx(mini_data: Path) -> None:
    from nucl_parquet.loader import _stopping_cache

    _stopping_cache.clear()

    db = connect(mini_data)
    dedx = elemental_dedx(db, "p", 29, 1.0)
    assert dedx.shape == (1,)
    assert dedx[0] == pytest.approx(50.0, rel=0.01)


def test_elemental_dedx_array(mini_data: Path) -> None:
    from nucl_parquet.loader import _stopping_cache

    _stopping_cache.clear()

    db = connect(mini_data)
    E = np.array([1.0, 10.0])
    dedx = elemental_dedx(db, "p", 29, E)
    assert dedx.shape == (2,)
    assert dedx[0] == pytest.approx(50.0, rel=0.01)
    assert dedx[1] == pytest.approx(20.0, rel=0.01)


def test_interp_loglog() -> None:
    log_E = np.log([1.0, 10.0, 100.0])
    log_S = np.log([100.0, 10.0, 1.0])
    result = _interp_loglog(log_E, log_S, np.array([1.0, 100.0]))
    assert result[0] == pytest.approx(100.0, rel=1e-6)
    assert result[1] == pytest.approx(1.0, rel=1e-6)


def test_connect_empty_dir(tmp_path: Path) -> None:
    """connect() works even with no catalog or data."""
    db = connect(tmp_path)
    views = db.sql("SELECT table_name FROM information_schema.tables WHERE table_type='VIEW'").fetchall()
    assert len(views) == 0


def test_sql_constants_are_strings() -> None:
    """SQL constants should be non-empty strings."""
    for sql in (DECAY_CHAIN_SQL, GAMMA_LINES_SQL, IDENTIFY_GAMMA_SQL, COINCIDENCE_SQL):
        assert isinstance(sql, str)
        assert len(sql) > 50
