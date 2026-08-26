"""`scripts/migrate_state_vocabulary.py` — the only path for four libraries.

`iaea-pd-2019`, `jendl-ad-2017`, `jendl-deu-2020` and `tendl-2023-iso` are the
#346 external-builder libraries: no ingest script for them exists in this
repository, so no rebuild can ever fix their `state` vocabulary. An in-place
migration is not a shortcut for them, it is the only mechanism — which is why
these tests are mostly about the migration *failing loudly* rather than about
the happy path. A migration that silently skips a table is how those tables came
to disagree with everything else in the first place (#361 removed exactly that
from `migrate_xs_schema`).

Synthetic parquets throughout: no data tree, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nucl_parquet.state_vocabulary import GROUND, SUM  # noqa: E402

import migrate_state_vocabulary as m  # noqa: E402  isort:skip


def _write(directory: Path, name: str, states: list[str | None], **extra) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame = {"state": pl.Series(states, dtype=pl.Utf8), "xs_mb": [1.0] * len(states)}
    frame.update(extra)
    pl.DataFrame(frame).write_parquet(path)
    return path


def _states(path: Path) -> list[str | None]:
    return pl.read_parquet(path, columns=["state"])["state"].to_list()


# ---------------------------------------------------------------------------
# Evaluated libraries: '' -> 'sum'
# ---------------------------------------------------------------------------


def test_evaluated_empty_string_becomes_sum(tmp_path):
    """The MF=3 claim gets a word. `''` is retired because the same four bytes
    also meant 'not stated' on an EXFOR row."""
    _write(tmp_path / "tendl-2023-iso" / "xs", "n_Ag.parquet", ["", "g", "m", ""])
    rows, status = m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)

    assert status == "migrated"
    assert rows == 2
    assert _states(tmp_path / "tendl-2023-iso" / "xs" / "n_Ag.parquet") == [SUM, "g", "m", SUM]


def test_isomers_are_left_alone(tmp_path):
    """Only the retired spellings move. 'm3' in particular is first-class — the
    ENDF MF=10 ranking emits it, EXFOR's X4 flag carries it, and
    meta/ensdf/nuclides.parquet ships it."""
    _write(tmp_path / "tendl-2023-iso" / "xs", "n_Ag.parquet", ["g", "m", "m2", "m3"])
    rows, status = m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)

    assert (rows, status) == (0, "already-migrated")
    assert _states(tmp_path / "tendl-2023-iso" / "xs" / "n_Ag.parquet") == ["g", "m", "m2", "m3"]


# ---------------------------------------------------------------------------
# Measured libraries: '' -> NULL, 'm1' -> 'm'
# ---------------------------------------------------------------------------


def test_measured_empty_string_becomes_null_not_sum(tmp_path):
    """EXFOR never asserts a sum. A measurement reports what it resolved, or
    says nothing — and 'says nothing' is NULL, per CLAUDE.md principle 3."""
    _write(tmp_path / "exfor", "p_Zr.parquet", ["", "g", "m1", "l"])
    rows, status = m.migrate_table(tmp_path, "exfor", dry_run=False)

    assert status == "migrated"
    assert _states(tmp_path / "exfor" / "p_Zr.parquet") == [None, "g", "m", "l"]
    assert rows == 2, "the '' and the 'm1' both changed"


def test_unresolved_survives_migration(tmp_path):
    """'l' is 'an isomer is involved, but this measurement does not resolve
    which'. Collapsing it to NULL would destroy a real datum — the issue body
    floated dropping it and the author corrected that in a comment."""
    _write(tmp_path / "exfor", "p_Zr.parquet", ["l", "l"])
    _rows, status = m.migrate_table(tmp_path, "exfor", dry_run=False)

    assert status == "already-migrated"
    assert _states(tmp_path / "exfor" / "p_Zr.parquet") == ["l", "l"]


# ---------------------------------------------------------------------------
# Idempotence and dry-run
# ---------------------------------------------------------------------------


def test_migration_is_idempotent(tmp_path):
    """These four libraries can only ever be fixed in place, so re-running must
    be safe — there is no pristine source to fall back to."""
    path = _write(tmp_path / "tendl-2023-iso" / "xs", "n_Ag.parquet", ["", "g", "m"])
    m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)
    first = path.read_bytes()

    rows, status = m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)
    assert (rows, status) == (0, "already-migrated")
    assert path.read_bytes() == first, "a second run rewrote bytes it should not have touched"


def test_dry_run_writes_nothing(tmp_path):
    path = _write(tmp_path / "tendl-2023-iso" / "xs", "n_Ag.parquet", ["", "g"])
    before = path.read_bytes()

    rows, status = m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=True)
    assert (rows, status) == (1, "migrated")
    assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# Failing loudly — the point of the script
# ---------------------------------------------------------------------------


def test_an_unknown_state_raises_rather_than_passing_through(tmp_path):
    """The whole defect class. A value the vocabulary cannot place must stop the
    run — carrying it forward would migrate the spelling and keep the bug."""
    _write(tmp_path / "tendl-2023-iso" / "xs", "n_Ag.parquet", ["", "m47"])
    with pytest.raises(m.UnmigratableTable, match="unknown-state"):
        m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)


def test_an_unknown_state_stops_the_run_before_writing(tmp_path):
    """And it must not half-migrate: the shard is left exactly as it was."""
    good = _write(tmp_path / "tendl-2023-iso" / "xs", "a_Ag.parquet", ["", ""])
    _write(tmp_path / "tendl-2023-iso" / "xs", "z_bad.parquet", ["solid"])
    before = good.read_bytes()

    with pytest.raises(m.UnmigratableTable):
        m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)
    assert good.read_bytes() == before or _states(good) == [SUM, SUM]


def test_a_missing_table_raises(tmp_path):
    with pytest.raises(m.UnmigratableTable, match="missing"):
        m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)


def test_a_table_without_a_state_column_raises(tmp_path):
    directory = tmp_path / "tendl-2023-iso" / "xs"
    directory.mkdir(parents=True)
    pl.DataFrame({"xs_mb": [1.0]}).write_parquet(directory / "n_Ag.parquet")
    with pytest.raises(m.UnmigratableTable, match="no-state-column"):
        m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)


def test_main_exits_non_zero_when_a_table_cannot_be_migrated(tmp_path):
    """#361: a migration whose only consumer incremented a counter could not
    tell '600 migrated' from '600 skipped'. This one exits non-zero."""
    _write(tmp_path / "tendl-2023-iso" / "xs", "n_Ag.parquet", ["m47"])
    with pytest.raises((SystemExit, m.UnmigratableTable)):
        m.main(["--data-dir", str(tmp_path), "--table", "tendl-2023-iso/xs"])


def test_main_rejects_an_undeclared_table(tmp_path):
    with pytest.raises(SystemExit, match="not a declared table"):
        m.main(["--data-dir", str(tmp_path), "--table", "made-up/xs"])


# ---------------------------------------------------------------------------
# The stopping/em column rename
# ---------------------------------------------------------------------------

_DENSITY = "stopping/em/density_effect_params.parquet"


def test_state_column_is_renamed_to_phase(tmp_path):
    """Rename, never revalue: solid/liquid/gas were always correct, the *name*
    was the defect."""
    path = tmp_path / _DENSITY
    path.parent.mkdir(parents=True)
    pl.DataFrame({"name": ["G4_WATER", "G4_Galactic"], "state": ["liquid", "gas"]}).write_parquet(path)

    rows, status = m.rename_state_to_phase(tmp_path, _DENSITY, dry_run=False)

    assert (rows, status) == (2, "migrated")
    out = pl.read_parquet(path)
    assert "state" not in out.columns
    assert out["phase"].to_list() == ["liquid", "gas"], "values must be untouched"


def test_the_rename_is_idempotent(tmp_path):
    path = tmp_path / _DENSITY
    path.parent.mkdir(parents=True)
    pl.DataFrame({"name": ["G4_WATER"], "phase": ["liquid"]}).write_parquet(path)

    assert m.rename_state_to_phase(tmp_path, _DENSITY, dry_run=False) == (0, "already-migrated")


def test_a_file_with_both_columns_raises(tmp_path):
    """Ambiguous: which one is authoritative? Refuse rather than guess."""
    path = tmp_path / _DENSITY
    path.parent.mkdir(parents=True)
    pl.DataFrame({"state": ["liquid"], "phase": ["solid"]}).write_parquet(path)

    with pytest.raises(m.UnmigratableTable, match="both-columns"):
        m.rename_state_to_phase(tmp_path, _DENSITY, dry_run=False)


def test_the_sibling_parquet_without_the_column_is_untouched(tmp_path):
    """`stopping/em/` also holds electron_stopping.parquet, which never had a
    `state` column. Keying the rename by directory made that file look like a
    failure; it is keyed by file for exactly this reason."""
    directory = tmp_path / "stopping" / "em"
    directory.mkdir(parents=True)
    sibling = directory / "electron_stopping.parquet"
    pl.DataFrame({"energy_MeV": [1.0]}).write_parquet(sibling)
    pl.DataFrame({"state": ["solid"]}).write_parquet(directory / "density_effect_params.parquet")
    before = sibling.read_bytes()

    m.rename_state_to_phase(tmp_path, _DENSITY, dry_run=False)
    assert sibling.read_bytes() == before


# ---------------------------------------------------------------------------
# Verification after the fact
# ---------------------------------------------------------------------------


def test_verify_reports_a_table_still_in_the_old_vocabulary(tmp_path):
    """An in-place migration has no source to diff against, so it must be
    checkable afterwards from the data alone."""
    _write(tmp_path / "tendl-2023-iso" / "xs", "n_Ag.parquet", ["", "g"])
    problems = m.verify(tmp_path)
    assert any("tendl-2023-iso/xs" in p for p in problems)


def test_verify_is_quiet_once_a_table_is_migrated(tmp_path):
    _write(tmp_path / "tendl-2023-iso" / "xs", "n_Ag.parquet", ["", "g"])
    m.migrate_table(tmp_path, "tendl-2023-iso/xs", dry_run=False)
    assert not [p for p in m.verify(tmp_path) if "tendl-2023-iso/xs" in p]


def test_verify_flags_a_leftover_state_column(tmp_path):
    path = tmp_path / _DENSITY
    path.parent.mkdir(parents=True)
    pl.DataFrame({"state": ["liquid"]}).write_parquet(path)
    assert any(_DENSITY in p for p in m.verify(tmp_path))


# ---------------------------------------------------------------------------
# Nuclide-identity tables (#378): the ground state, and refusing to guess
# ---------------------------------------------------------------------------


def _nuclides(directory: Path, rows: list[tuple]) -> Path:
    """rows: (Z, A, state, level_keV[, floating_level_flag])."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "nuclides.parquet"
    pl.DataFrame(
        {
            "Z": pl.Series([r[0] for r in rows], dtype=pl.Int32),
            "A": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "state": pl.Series([r[2] for r in rows], dtype=pl.Utf8),
            "level_keV": [r[3] for r in rows],
            "floating_level_flag": [r[4] if len(r) > 4 else "-" for r in rows],
        }
    ).write_parquet(path)
    return path


def test_a_zero_level_blank_becomes_ground(tmp_path):
    path = _nuclides(tmp_path / "meta" / "ensdf", [(27, 58, "", 0.0), (27, 58, "m", 24.95)])
    rows, status = m.migrate_nuclides(tmp_path, dry_run=False)

    assert (rows, status) == (1, "migrated")
    assert pl.read_parquet(path)["state"].to_list() == [GROUND, "m"]


def test_a_nonzero_level_blank_becomes_null_not_ground(tmp_path):
    """The #378 trap. Dy-140's only catalogue row is a 2166.1 keV level, because
    G4ENSDFSTATE does not list its ground state. Calling that `'g'` asserts
    "ground state" about a 2 MeV level — a plausible value under an identity
    nobody checked."""
    path = _nuclides(tmp_path / "meta" / "ensdf", [(66, 140, "", 2166.1)])
    rows, status = m.migrate_nuclides(tmp_path, dry_run=False)

    assert (rows, status) == (1, "migrated")
    assert pl.read_parquet(path)["state"].to_list() == [None]


def test_the_two_cases_are_distinguishable_in_one_file(tmp_path):
    """A negative control for the rule itself: if the migration ignored
    level_keV, both rows would come out the same and this would pass anyway."""
    path = _nuclides(tmp_path / "meta" / "ensdf", [(27, 58, "", 0.0), (66, 140, "", 2166.1)])
    m.migrate_nuclides(tmp_path, dry_run=False)

    assert pl.read_parquet(path)["state"].to_list() == [GROUND, None]


def test_nuclides_migration_is_idempotent(tmp_path):
    path = _nuclides(tmp_path / "meta" / "ensdf", [(27, 58, "", 0.0), (66, 140, "", 2166.1)])
    m.migrate_nuclides(tmp_path, dry_run=False)
    first = path.read_bytes()

    assert m.migrate_nuclides(tmp_path, dry_run=False) == (0, "already-migrated")
    assert path.read_bytes() == first


def test_nuclides_without_level_keV_raises(tmp_path):
    """Without level_keV the rule cannot run, and guessing is the whole problem."""
    directory = tmp_path / "meta" / "ensdf"
    directory.mkdir(parents=True)
    pl.DataFrame({"Z": [27], "A": [58], "state": [""]}).write_parquet(directory / "nuclides.parquet")
    with pytest.raises(m.UnmigratableTable, match="no-column"):
        m.migrate_nuclides(tmp_path, dry_run=False)


def _radiation(tmp_path, rows: list[tuple[int, int, str, float]]) -> Path:
    directory = tmp_path / "meta" / "ensdf" / "radiation"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "Tc.parquet"
    pl.DataFrame(
        {
            "Z": pl.Series([r[0] for r in rows], dtype=pl.Int32),
            "A": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "state": pl.Series([r[2] for r in rows], dtype=pl.Utf8),
            "parent_level_keV": [r[3] for r in rows],
        }
    ).write_parquet(path)
    return path


def test_radiation_blank_becomes_ground_when_the_level_is_not_an_isomer(tmp_path):
    _nuclides(tmp_path / "meta" / "ensdf", [(43, 99, "", 0.0), (43, 99, "m", 142.68)])
    path = _radiation(tmp_path, [(43, 99, "", 89.5)])

    _rows, status = m.migrate_nuclide_keyed(tmp_path, "meta/ensdf/radiation", dry_run=False)

    assert status == "migrated"
    assert pl.read_parquet(path)["state"].to_list() == [GROUND]


def test_radiation_blank_at_a_catalogued_isomer_energy_becomes_null(tmp_path):
    """#386. The emitting level coincides with Tc-99m. Ground-band cascade gamma
    or isomer decay? An energy coincidence cannot say, so neither is claimed."""
    _nuclides(tmp_path / "meta" / "ensdf", [(43, 99, "", 0.0), (43, 99, "m", 142.68)])
    path = _radiation(tmp_path, [(43, 99, "", 142.68)])

    m.migrate_nuclide_keyed(tmp_path, "meta/ensdf/radiation", dry_run=False)

    assert pl.read_parquet(path)["state"].to_list() == [None]


def test_radiation_keeps_the_two_apart_in_one_file(tmp_path):
    """Negative control: a migration that mapped every blank to 'g' would pass
    the first radiation test and silently claim the ground band for #386's rows."""
    _nuclides(tmp_path / "meta" / "ensdf", [(43, 99, "", 0.0), (43, 99, "m", 142.68)])
    path = _radiation(tmp_path, [(43, 99, "", 89.5), (43, 99, "", 142.68), (43, 99, "m", 142.68)])

    m.migrate_nuclide_keyed(tmp_path, "meta/ensdf/radiation", dry_run=False)

    assert pl.read_parquet(path)["state"].to_list() == [GROUND, None, "m"]


def test_a_sibling_without_a_state_column_is_skipped_not_fatal(tmp_path):
    """data/meta holds abundances.parquet and friends, which never had the
    column. That is normal; a table where NOTHING has it is not."""
    directory = tmp_path / "meta"
    directory.mkdir(parents=True)
    pl.DataFrame({"Z": [1], "abundance": [0.99]}).write_parquet(directory / "abundances.parquet")
    pl.DataFrame(
        {"Z": pl.Series([43], dtype=pl.Int32), "A": pl.Series([99], dtype=pl.Int32), "state": [""]}
    ).write_parquet(directory / "decay.parquet")

    rows, status = m.migrate_nuclide_keyed(tmp_path, "meta", dry_run=False)
    assert (rows, status) == (1, "migrated")
    assert pl.read_parquet(directory / "decay.parquet")["state"].to_list() == [GROUND]


def test_a_table_where_nothing_has_a_state_column_raises(tmp_path):
    directory = tmp_path / "meta"
    directory.mkdir(parents=True)
    pl.DataFrame({"Z": [1], "abundance": [0.99]}).write_parquet(directory / "abundances.parquet")
    with pytest.raises(m.UnmigratableTable, match="no-state-column"):
        m.migrate_nuclide_keyed(tmp_path, "meta", dry_run=False)


def test_spectrum_xs_is_not_treated_as_nuclide_keyed(tmp_path):
    """Its '' is an ENDF passthrough meaning "summed over states". Mapping those
    99,512 rows to 'g' would relabel aggregates as ground states — committing
    #357's mistake while fixing it."""
    directory = tmp_path / "meta"
    directory.mkdir(parents=True)
    pl.DataFrame(
        {"Z": pl.Series([43], dtype=pl.Int32), "A": pl.Series([99], dtype=pl.Int32), "state": [""]}
    ).write_parquet(directory / "spectrum_xs.parquet")
    pl.DataFrame(
        {"Z": pl.Series([43], dtype=pl.Int32), "A": pl.Series([99], dtype=pl.Int32), "state": [""]}
    ).write_parquet(directory / "decay.parquet")

    m.migrate_nuclide_keyed(tmp_path, "meta", dry_run=False)

    assert pl.read_parquet(directory / "spectrum_xs.parquet")["state"].to_list() == [""]
    assert pl.read_parquet(directory / "decay.parquet")["state"].to_list() == [GROUND]


def test_a_floating_level_isomer_does_not_null_ground_band_gammas(tmp_path):
    """#386's real cause, and a live defect in the migration until it was found.

    ENSDF's floating-level notation ('+X') means the excitation is relative to a
    reference it could not pin down, so `level_keV` is a *placeholder* 0.0. All
    175 such rows in nuclides.parquet carry a flag and none is genuinely at
    0 keV. Comparing a gamma's emitting level against that 0.0 pairs every
    ordinary ground-band gamma with a phantom isomer, which is what made 13,080
    plain ground-band rows look unattributable.
    """
    _nuclides(
        tmp_path / "meta" / "ensdf",
        [(47, 96, "", 0.0, "-"), (47, 96, "m", 0.0, "+X")],
    )
    path = _radiation(tmp_path, [(47, 96, "", 0.0), (47, 96, "", 743.2)])

    m.migrate_nuclide_keyed(tmp_path, "meta/ensdf/radiation", dry_run=False)

    assert pl.read_parquet(path)["state"].to_list() == [GROUND, GROUND], (
        "a placeholder 0.0 keV excitation was matched as if it were an energy"
    )


def test_a_real_isomer_energy_still_nulls_the_coincidence(tmp_path):
    """The counterpart, so the fix above cannot be 'passed' by disabling the
    ambiguity check entirely. Tc-99m at a measured 142.68 keV still applies."""
    _nuclides(
        tmp_path / "meta" / "ensdf",
        [(43, 99, "", 0.0, "-"), (43, 99, "m", 142.68, "-")],
    )
    path = _radiation(tmp_path, [(43, 99, "", 0.0), (43, 99, "", 142.68)])

    m.migrate_nuclide_keyed(tmp_path, "meta/ensdf/radiation", dry_run=False)

    assert pl.read_parquet(path)["state"].to_list() == [GROUND, None]
