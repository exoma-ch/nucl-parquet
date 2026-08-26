"""Unit tests for `scripts/migrate_xs_schema.py`.

The migration lifts legacy per-library tables into `CANONICAL_XS_SCHEMA`. It ran
once over 35.4M rows, and every defect it had was a silent one: a stem it could
not parse meant a file skipped, and a column it did not carry forward meant data
deleted. Both happened. These pin the two failure modes.

Gates the legacy -> canonical lift, including that a file the migration cannot
handle now *raises* rather than being tallied into a counter nobody branched on
(#361). This file was absent from `ci.sh`'s allowlist, so neither those gates nor
the pre-existing `parse_stem` cases ran in CI at all — an allowlist gap on a file
whose whole subject is silent non-execution. Fixed at the root in #355; the note
moved here from `ci.sh` because this is where the next reader is looking.

Pure unit tests: no data tree, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_xs_schema import (  # noqa: E402
    _RENAMES,
    SUCCESS_STATUSES,
    UnmigratableFile,
    build_parser,
    migrate_file,
    parse_stem,
)


def _legacy_xs(path: Path) -> Path:
    """Write a minimal *legacy* 6-column table — the shape the migration lifts."""
    import polars as pl

    pl.DataFrame(
        {
            "target_A": [63, 63],
            "residual_Z": [30, 0],
            "residual_A": [63, 0],
            "state": ["", ""],
            "energy_MeV": [1.0, 2.0],
            "xs_mb": [10.0, 20.0],
        },
        schema={
            "target_A": pl.Int32,
            "residual_Z": pl.Int32,
            "residual_A": pl.Int32,
            "state": pl.Utf8,
            "energy_MeV": pl.Float64,
            "xs_mb": pl.Float64,
        },
    ).write_parquet(path)
    return path


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("p_Cu", ("p", 1, 1, 29)),
        ("n_Fe", ("n", 0, 1, 26)),
        ("a_U", ("a", 2, 4, 92)),
        # Heavy ions carry their own mass in the projectile token. He-3 goes down
        # the same path as a heavy ion rather than the light-ion table, and must
        # still land on (Z=2, A=3) rather than being read as helium-natural.
        ("he3_C", ("he3", 2, 3, 6)),
        ("ar40_Ac", ("ar40", 18, 40, 89)),
        ("c12_Au", ("c12", 6, 12, 79)),
        # Elements the builders have no symbol for are written Z<number>.
        # 183 files used this form and were skipped by the first regex.
        ("p_Z61", ("p", 1, 1, 61)),
        ("d_Z105", ("d", 1, 2, 105)),
        ("ar40_Z99", ("ar40", 18, 40, 99)),
        # Unparseable must be None, never a partial identity.
        ("garbage", None),
        ("p_Xx", None),
        ("", None),
    ],
)
def test_parse_stem(stem: str, expected: tuple | None) -> None:
    """Every stem a builder emits must parse.

    A stem that returns None is a file the migration skips, and skipping is
    invisible: the file stays in its legacy shape and only surfaces when a
    consumer unions it.
    """
    assert parse_stem(stem) == expected


def test_legacy_provenance_column_is_registered_for_rename() -> None:
    """`exfor_entry` must be renamed, not dropped.

    An early migration selected only the canonical column names, which silently
    deleted the provenance chain back to the measurement for every EXFOR row --
    real data loss, recovered only because the legacy tree was still in git.

    This asserts the rename is *registered*, not that a migration run applies it;
    the end state is covered from the other side by
    `test_canonical_schema.py::test_all_xs_tables_share_the_canonical_schema`,
    which fails if any shipped table is missing `source_entry`.
    """
    assert _RENAMES["exfor_entry"] == "source_entry"


# -- Failure must be loud (#361) ----------------------------------------------


def test_unparseable_stem_raises_instead_of_silently_no_opping(tmp_path: Path) -> None:
    """The exact trap that produced two false verdicts reviewing #349.

    A checker copied each library file to a temp path named `f.parquet` before
    migrating it. `parse_stem("f")` is None, so `migrate_file` returned
    `"unparseable-stem"` and left the file **byte-identical**. The checker then
    compared that untouched legacy file against its migrated counterpart and
    reported hundreds of files "differing" — a confident wrong answer from a
    proof that had silently never run.

    The asymmetry is the point: that disagreement pointed the safe way, but the
    same silence would equally have let a bad round-trip *pass*.
    """
    import polars as pl

    bad = _legacy_xs(tmp_path / "f.parquet")
    before = bad.read_bytes()

    with pytest.raises(UnmigratableFile) as excinfo:
        migrate_file(bad, "tendl-2025", "production", dry_run=False)

    assert excinfo.value.status == "unparseable-stem"
    assert "f" in str(excinfo.value)
    # The no-op itself is unchanged — that is *why* it has to raise.
    assert bad.read_bytes() == before
    assert "library" not in pl.read_parquet(bad).columns, "the file is still legacy, as the exception says"


def test_unreadable_file_raises(tmp_path: Path) -> None:
    """A shard that cannot be read is a failure, not a statistic.

    Named with a parseable stem so this exercises the read failure specifically
    rather than tripping the stem check first.
    """
    corrupt = tmp_path / "p_Cu.parquet"
    corrupt.write_bytes(b"this is not a parquet file")

    with pytest.raises(UnmigratableFile) as excinfo:
        migrate_file(corrupt, "tendl-2025", "production", dry_run=False)
    assert excinfo.value.status == "unreadable"
    assert excinfo.value.path == corrupt


def test_a_migratable_file_still_migrates(tmp_path: Path) -> None:
    """The strict default must not break the working path.

    Asserts a positive result — canonical columns present, rows preserved — so
    this cannot pass by the migration refusing to do anything.
    """
    import polars as pl

    good = _legacy_xs(tmp_path / "p_Cu.parquet")
    rows, status = migrate_file(good, "tendl-2025", "production", dry_run=False)

    assert status == "migrated"
    assert status in SUCCESS_STATUSES
    assert rows == 2
    out = pl.read_parquet(good)
    assert out.height == 2
    assert {"library", "projectile", "target_Z", "MT"} <= set(out.columns)
    assert out["library"].unique().to_list() == ["tendl-2025"]
    assert out["target_Z"].unique().to_list() == [29], "target_Z comes from the stem 'p_Cu'"
    # The 0/0 residual sentinel becomes a null, per the migration's own contract.
    assert out["residual_Z"].null_count() == 1


def test_already_canonical_is_a_success_not_a_skip(tmp_path: Path) -> None:
    """Re-running the migration must stay idempotent and must not raise.

    `already-canonical` is the one non-`migrated` status that means the file is
    fine, so it belongs in SUCCESS_STATUSES; classing it as a failure would make
    every second run exit non-zero.
    """
    good = _legacy_xs(tmp_path / "p_Cu.parquet")
    migrate_file(good, "tendl-2025", "production", dry_run=False)
    _, status = migrate_file(good, "tendl-2025", "production", dry_run=False)
    assert status == "already-canonical"
    assert status in SUCCESS_STATUSES


def test_tolerant_mode_is_opt_in_and_names_every_path(tmp_path: Path, caplog) -> None:
    """`strict=False` may return a status, but must log the path, not a tally.

    The old behaviour was this minus the log line and minus the opt-in: "3
    unparseable-stem" in a summary is not something a reader can act on.
    """
    bad = _legacy_xs(tmp_path / "f.parquet")
    with caplog.at_level("WARNING"):
        rows, status = migrate_file(bad, "tendl-2025", "production", dry_run=False, strict=False)

    assert (rows, status) == (0, "unparseable-stem")
    assert status not in SUCCESS_STATUSES
    assert "f.parquet" in caplog.text, "tolerant mode must name the file it skipped"


def test_main_exits_non_zero_when_a_file_cannot_be_migrated(tmp_path: Path, monkeypatch) -> None:
    """A run that leaves files in the legacy schema must not exit 0.

    This is the end-to-end shape of #361: the rebuild commands in the manifests
    chain `migrate_xs_schema.py`, so an exit code that lies lets a stale library
    through a rebuild.
    """
    import migrate_xs_schema

    lib = tmp_path / "tendl-2025" / "xs"
    lib.mkdir(parents=True)
    _legacy_xs(lib / "p_Cu.parquet")  # fine
    _legacy_xs(lib / "f.parquet")  # unmigratable
    (tmp_path / "catalog.json").write_text(
        json.dumps({"libraries": {"tendl-2025": {"data_type": "cross_sections", "path": "tendl-2025/xs/"}}})
    )

    monkeypatch.setattr(sys, "argv", ["migrate_xs_schema.py", "--data-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as excinfo:
        migrate_xs_schema.main()
    assert excinfo.value.code == 1


def test_main_survey_mode_also_exits_non_zero(tmp_path: Path, monkeypatch) -> None:
    """`--skip-unmigratable` surveys the damage; it does not accept it.

    A tolerant flag that also exited 0 would just relocate the original bug
    behind a nicer name.
    """
    import migrate_xs_schema
    import polars as pl

    lib = tmp_path / "tendl-2025" / "xs"
    lib.mkdir(parents=True)
    good = _legacy_xs(lib / "p_Cu.parquet")
    _legacy_xs(lib / "f.parquet")
    (tmp_path / "catalog.json").write_text(
        json.dumps({"libraries": {"tendl-2025": {"data_type": "cross_sections", "path": "tendl-2025/xs/"}}})
    )

    monkeypatch.setattr(sys, "argv", ["migrate_xs_schema.py", "--data-dir", str(tmp_path), "--skip-unmigratable"])
    with pytest.raises(SystemExit) as excinfo:
        migrate_xs_schema.main()
    assert excinfo.value.code == 1
    # ...and unlike strict mode, it got through the rest of the library first.
    assert "library" in pl.read_parquet(good).columns


def test_skip_unmigratable_defaults_to_off() -> None:
    """Strict is the default; tolerance must be asked for explicitly."""
    args = build_parser().parse_args([])
    assert args.skip_unmigratable is False
