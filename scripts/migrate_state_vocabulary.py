"""Rewrite shipped parquets into the `state` vocabulary of #357/#367.

The builders now emit the vocabulary directly, so most tables are fixed by a
re-ingest. Four are not, and this script is the only thing that can fix them:

    iaea-pd-2019, jendl-ad-2017, jendl-deu-2020, tendl-2023-iso

Those are the #346 external-builder libraries — `data/builder_stamp_exemptions.json`
records that no ingest script for them exists in this repository. There is
nothing to re-run, so an in-place migration is not a shortcut here, it is the
only path. That is also why this script raises rather than counting: a
migration that silently skips a table is how those tables came to disagree with
everything else in the first place.

What it does, per table kind (see `nucl_parquet.state_vocabulary`):

    evaluated xs   ''   -> 'sum'   (the MF=3 "summed over states" claim)
    measured xs    ''   -> NULL    ("the measurement did not say")
                   'm1' -> 'm'     (X4 synonym; one spelling reaches disk)
    stopping/em    column `state` -> `phase`  (phase of matter, never a state)

The nuclide-identity tables under `meta/` are deliberately **not** handled. Their
`''` means "the ground state" for 3,148 of 3,161 rows and something unresolved
for the other 13 — levels between 124.5 and 2166.1 keV that carry no isomer
flag. Mapping those to `'g'` would assert "ground state" about a 2 MeV level,
which is the same defect class this migration exists to remove. See the
follow-up issue named in `PENDING_MIGRATION`.

Idempotent: a table already in the new vocabulary is reported `already-migrated`
and its bytes are left untouched.

Usage:
    nix develop -c uv run python scripts/migrate_state_vocabulary.py --dry-run
    nix develop -c uv run python scripts/migrate_state_vocabulary.py --table tendl-2023-iso/xs
    nix develop -c uv run python scripts/migrate_state_vocabulary.py
    nix develop -c uv run python scripts/migrate_state_vocabulary.py --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))

from nucl_parquet.state_vocabulary import (  # noqa: E402
    GROUND,
    LEGACY_UNSPECIFIED,
    MEASURED_XS_STATES,
    NUCLIDE_STATES,
    PENDING_COLUMN_RENAME,
    PENDING_DAUGHTER_STATE_MIGRATION,
    PENDING_MIGRATION,
    PENDING_PARENT_STATE_MIGRATION,
    PHASE_NOT_STATE,
    SUM,
    TABLE_DAUGHTER_STATES,
    TABLE_PARENT_STATES,
    TABLE_STATES,
    allowed_daughter_states,
    allowed_parent_states,
    allowed_states,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

COMPRESSION = "zstd"

#: Statuses that mean the table is now in the new vocabulary. Anything else is
#: not, and must not be reported as a successful run (#361).
SUCCESS_STATUSES = frozenset({"migrated", "already-migrated"})


class UnmigratableTable(RuntimeError):
    """A table the migration could not process — it is still in its old shape.

    Raised rather than counted, deliberately. Every failure mode here is a
    *silent no-op* — the file is left exactly as it was — so a caller that only
    tallied statuses could not tell "17 tables migrated" from "17 tables
    skipped", which is precisely what #361 removed from `migrate_xs_schema`.
    """

    def __init__(self, table: str, status: str, detail: str = "") -> None:
        self.table = table
        self.status = status
        self.detail = detail
        super().__init__(f"{table}: {status}" + (f" — {detail}" if detail else ""))


def shards(data_dir: Path, table: str) -> list[Path]:
    """Every parquet shard of `table`, which is a directory under `data/`."""
    directory = data_dir / table
    if not directory.is_dir():
        raise UnmigratableTable(table, "missing", f"{directory} is not a directory")
    return sorted(directory.glob("*.parquet"))


def legacy_mapping(table: str) -> dict[str, str | None]:
    """The retired spellings this migration converts for `table`, and to what.

    Deliberately independent of `PENDING_MIGRATION`. That ledger records which
    tables *still ship* a retired value and empties as they are rebuilt, which
    is the right contract for a ledger and the wrong one for a converter: a
    migration that loses the ability to migrate the moment the tree is clean
    cannot be re-run on a fresh ingest, and cannot be tested at all. Reading
    `legacy` off the ledger made emptying it break six of this file's tests.

    `''` meant "summed over states" in an evaluated table and "not stated" in a
    measured one — one token, two meanings, which is the whole reason #380
    retired it — so the target depends on the table.
    """
    measured = TABLE_STATES[table] == MEASURED_XS_STATES
    mapping: dict[str, str | None] = {LEGACY_UNSPECIFIED: None if measured else SUM}
    if measured:
        # EXFOR writes the first isomer as `m1`; the vocabulary spells it `m`.
        mapping["m1"] = "m"
    return mapping


def migrated_state_column(table: str, states: pl.Series) -> pl.Series:
    """Map one table's `state` column onto the new vocabulary.

    Raises for a value the vocabulary cannot place, rather than passing it
    through or defaulting it. An unknown state is the thing this whole change is
    about; quietly carrying it forward would migrate the spelling and keep the
    defect.
    """
    target = TABLE_STATES[table]
    mapping = legacy_mapping(table)

    unknown = sorted(v for v in states.unique().to_list() if v is not None and v not in target and v not in mapping)
    if unknown:
        raise UnmigratableTable(table, "unknown-state", f"values the vocabulary cannot place: {unknown}")

    return states.replace_strict(mapping, default=pl.first(), return_dtype=pl.Utf8)


#: Tables whose `state` names which state of the *parent nuclide* a row is
#: about, so `''` meant the ground state. `meta/ensdf` is handled separately
#: because it needs `level_keV` to tell a real ground state from an excited
#: level that inherited the label.
_NUCLIDE_KEYED = frozenset({"meta/ensdf/radiation", "meta/ensdf/beta_spectra", "meta"})

#: Files inside a `_NUCLIDE_KEYED` directory that are NOT nuclide-keyed and must
#: not take the `'' -> 'g'` rule.
#:
#: `meta/spectrum_xs.parquet` is a roll-up of the ENDF cross-section tables and
#: its `state` is a passthrough, so its 99,512 `''` rows carry ENDF's "summed
#: over states" meaning. Mapping them to `'g'` would relabel a hundred thousand
#: aggregates as ground-state rows — the same mistake as #357, committed while
#: fixing #357. It is rebuilt from the xs tables and will inherit `'sum'`.
_NOT_NUCLIDE_KEYED = frozenset({"spectrum_xs.parquet"})

#: Isomer energies per (Z, A), read from nuclides.parquet, for deciding whether
#: a radiation row's emitting level coincides with a catalogued isomer.
_ISOMER_TOLERANCE_KEV = 1.0


def _catalogued_isomer_levels(data_dir: Path) -> dict[tuple[int, int], list[float]]:
    nuclides = data_dir / "meta" / "ensdf" / "nuclides.parquet"
    if not nuclides.is_file():
        raise UnmigratableTable("meta/ensdf", "missing", f"{nuclides} is needed to classify radiation rows")
    df = pl.read_parquet(nuclides, columns=["Z", "A", "state", "level_keV", "floating_level_flag"])
    levels: dict[tuple[int, int], list[float]] = {}
    for z, a, state, level, floating in df.iter_rows():
        # Pre-migration the isomers are already spelled 'm'/'m2'/'m3'; post-
        # migration they still are. Only the ground label moves, so this reads
        # correctly either way.
        if state in (LEGACY_UNSPECIFIED, GROUND, None):
            continue
        # ENSDF's floating-level notation ('+X', '+Y', …) means the excitation
        # is relative to a reference it could not pin down, so `level_keV` is a
        # *placeholder* 0.0 rather than a measured energy — all 175 such rows in
        # nuclides.parquet carry a flag, and none is genuinely at 0 keV.
        #
        # Comparing a gamma's emitting level against that 0.0 pairs every
        # ordinary ground-band gamma with a phantom isomer at 0 keV. That is
        # what made #386 look like 13,106 unattributable rows; 13,080 of them
        # were this, and are plain ground-band gammas. A placeholder is not an
        # energy, and must not be matched as one.
        if (floating or "-") != "-":
            continue
        levels.setdefault((int(z), int(a)), []).append(float(level))
    return levels


def migrate_nuclides(data_dir: Path, *, dry_run: bool) -> tuple[int, str]:
    """`meta/ensdf/nuclides.parquet`: level_keV == 0 -> 'g', else NULL.

    `assign_state_labels` calls the lowest *listed* level per (Z, A) the ground
    state. For 3,148 of 3,161 rows that level is at 0.0 keV and the label is
    right. For 13 it is not: G4ENSDFSTATE does not list those nuclides' ground
    states, so an excited level inherited the label — every one of them matches
    a level at index 1-5 in `meta/ensdf/levels/`, and four carry ENSDF's `+X`
    floating flag, meaning the excitation is relative to an unknown offset.

    Those 13 become NULL rather than `'g'`. Calling a 2166 keV level "the ground
    state" would be a plausible value under an identity nobody checked, which is
    the defect this migration exists to remove (#378).
    """
    path = data_dir / "meta" / "ensdf" / "nuclides.parquet"
    if not path.is_file():
        raise UnmigratableTable("meta/ensdf", "missing", str(path))
    df = pl.read_parquet(path)
    for column in ("state", "level_keV"):
        if column not in df.columns:
            raise UnmigratableTable("meta/ensdf", "no-column", f"{path.name} has no `{column}`")

    present = {v for v in df["state"].unique().to_list() if v is not None}
    stray = sorted(present - NUCLIDE_STATES - {LEGACY_UNSPECIFIED})
    if stray:
        raise UnmigratableTable("meta/ensdf", "unknown-state", f"{stray}")
    if LEGACY_UNSPECIFIED not in present:
        return 0, "already-migrated"

    new_state = (
        pl.when(pl.col("state") != LEGACY_UNSPECIFIED)
        .then(pl.col("state"))
        .when(pl.col("level_keV") == 0.0)
        .then(pl.lit(GROUND))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )
    out = df.with_columns(new_state.alias("state"))
    changed = int((df["state"] == LEGACY_UNSPECIFIED).sum())
    if not dry_run:
        out.write_parquet(path, compression=COMPRESSION)
    return changed, "migrated"


def migrate_nuclide_keyed(data_dir: Path, table: str, *, dry_run: bool) -> tuple[int, str]:
    """`''` -> `'g'` in tables whose `state` names which state of the parent decays.

    `radiation` is the hard one. Its `''` is documented as "from the ground-band
    decay chain", so the decaying species is the ground state and `'g'` is
    right — except for 13,106 rows across 45 nuclides whose emitting level
    coincides with a catalogued isomer of the same nuclide. Whether those are
    ground-band cascade gammas or isomer decays cannot be settled from an energy
    coincidence, so they become NULL (#386): 26 rows across 13 nuclides, once
    ENSDF's floating-level placeholders are excluded from the comparison.
    Resolving those needs the decay dataset's own parent attribution, not a
    better energy tolerance.
    """
    paths = shards(data_dir, table)
    if not paths:
        raise UnmigratableTable(table, "no-shards", "declared but ships no parquet")

    ambiguous_levels = _catalogued_isomer_levels(data_dir) if table.endswith("radiation") else {}
    changed = 0
    # "No shard in this table has a `state` column" is a real error; "this one
    # sibling never had one" is normal in data/meta. Keeping the two apart is
    # the difference between a skip and a silent skip.
    seen_state_column = False
    for path in paths:
        if path.name in _NOT_NUCLIDE_KEYED:
            continue  # different meaning of '' — see _NOT_NUCLIDE_KEYED
        if "state" not in pl.read_parquet_schema(path):
            continue  # a sibling that never had the column (data/meta is mixed)
        seen_state_column = True
        df = pl.read_parquet(path)
        present = {v for v in df["state"].unique().to_list() if v is not None}
        stray = sorted(present - NUCLIDE_STATES - {LEGACY_UNSPECIFIED})
        if stray:
            raise UnmigratableTable(table, "unknown-state", f"{path.name}: {stray}")
        if LEGACY_UNSPECIFIED not in present:
            continue

        blank = pl.col("state") == LEGACY_UNSPECIFIED
        if ambiguous_levels and "parent_level_keV" in df.columns:
            ambiguous = pl.Series(
                [
                    row_blank
                    and row_level is not None
                    and any(
                        abs(row_level - lvl) < _ISOMER_TOLERANCE_KEV for lvl in ambiguous_levels.get((row_z, row_a), ())
                    )
                    for row_blank, row_level, row_z, row_a in zip(
                        (df["state"] == LEGACY_UNSPECIFIED).to_list(),
                        df["parent_level_keV"].to_list(),
                        df["Z"].to_list(),
                        df["A"].to_list(),
                    )
                ],
                dtype=pl.Boolean,
            )
        else:
            ambiguous = pl.Series([False] * df.height, dtype=pl.Boolean)

        new_state = (
            pl.when(~blank)
            .then(pl.col("state"))
            .when(pl.lit(ambiguous))
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(pl.lit(GROUND))
        )
        changed += int((df["state"] == LEGACY_UNSPECIFIED).sum())
        if not dry_run:
            df.with_columns(new_state.alias("state")).write_parquet(path, compression=COMPRESSION)

    if not seen_state_column:
        raise UnmigratableTable(table, "no-state-column", "no shard in this table has a `state` column")
    return changed, "already-migrated" if changed == 0 else "migrated"


def migrate_table(data_dir: Path, table: str, *, dry_run: bool) -> tuple[int, str]:
    """Rewrite every shard of `table`. Returns (rows_changed, status)."""
    paths = shards(data_dir, table)
    if not paths:
        raise UnmigratableTable(table, "no-shards", "declared in TABLE_STATES but ships no parquet")

    # What the converter handles, not what the tree happens to still ship — see
    # `legacy_mapping`. `allowed_states` is still consulted so an undeclared
    # table raises, and it contributes any legacy value the ledger is tracking.
    legacy = frozenset(legacy_mapping(table))
    allowed_now = allowed_states(table) | legacy

    changed = 0
    for path in paths:
        df = pl.read_parquet(path)
        if "state" not in df.columns:
            raise UnmigratableTable(table, "no-state-column", str(path))

        present = {v for v in df["state"].unique().to_list() if v is not None}
        stray = sorted(present - allowed_now)
        if stray:
            raise UnmigratableTable(table, "unknown-state", f"{path.name}: {stray}")
        if not (present & legacy):
            continue  # this shard is already in the new vocabulary

        new_state = migrated_state_column(table, df["state"])
        n = int((df["state"] != new_state).sum() + (new_state.is_null() & df["state"].is_not_null()).sum())
        changed += n
        if not dry_run:
            df.with_columns(new_state.alias("state")).write_parquet(path, compression=COMPRESSION)

    if changed == 0:
        return 0, "already-migrated"
    return changed, "migrated"


#: Retired-spelling → new-spelling map for `parent_state` on the g4 tables.
#:
#: `''` → `'g'` when the row's parent decay is known (any non-null
#: `parent_decay_mode`): the g4 builders label `parent_ex_kev == 0` as `''` and
#: the source distinguishes the ground state from every catalogued isomer, so
#: `''` unambiguously means ground *for those rows*.
#:
#: `''` → NULL when `parent_decay_mode` is null. Those are γ-γ cascade pairs
#: `annotate_gamma_gamma` could not attribute to a parent channel — the
#: `parent_state.fill_null("")` was a default, not a determination. Claiming
#: `'g'` there would invent an attribution the builder never made, exactly the
#: defect class #326/#351/#377 removed elsewhere.
#:
#: The choice hinges on `parent_decay_mode` because it is the only field in the
#: shipped parquet that records whether the parent was actually resolved. The
#: ambiguity checked for `radiation` in #386 does not apply the same way here
#: (see the docstring of `migrate_parent_state_g4`).


#: The three tables whose `parent_state` this migration rewrites.
_PARENT_STATE_TABLES = frozenset(
    {
        "meta/ensdf/coincidences",
        "meta/ensdf/emissions",
        "meta/ensdf/summing_partners",
    }
)


def _parent_low_isomer_zA(data_dir: Path) -> set[tuple[int, int]]:
    """Parent (Z, A) whose ground state and a catalogued isomer coincide within 1 keV.

    The g4 builders label `parent_ex_kev` around zero with `_label_parent_state`,
    which uses a 1.0 keV tolerance — so a parent nuclide with a catalogued
    m-state at `level_keV < 1` would collapse ground and isomer under the same
    `''` label. In the 2026.8.5 catalog exactly two parents qualify: Ga-73m
    (level 0.3 keV) and U-235m (0.076 keV). Whether a row's `''` is actually
    ambiguous also depends on `decay_mode` — see `_ambiguous_parent_decay_modes`.
    """
    nuclides = data_dir / "meta" / "ensdf" / "nuclides.parquet"
    if not nuclides.is_file():
        raise UnmigratableTable("meta/ensdf", "missing", f"{nuclides} is needed to classify parents")
    df = pl.read_parquet(nuclides, columns=["Z", "A", "state", "level_keV", "floating_level_flag"])
    ambig: set[tuple[int, int]] = set()
    for z, a, state, level, floating in df.iter_rows():
        if state in (LEGACY_UNSPECIFIED, GROUND, None):
            continue
        if level is None:
            continue
        # Floating-flag levels (`+X`/`+Y`) mean the excitation is relative to an
        # unknown offset — the 0.0 keV they carry is a placeholder, not a real
        # coincidence with the ground state.
        if (floating or "-") != "-":
            continue
        if level < 1.0:
            ambig.add((int(z), int(a)))
    return ambig


def _ambiguous_parent_decay_modes(data_dir: Path) -> set[tuple[int, int, str]]:
    """`(Z, A, decay_mode)` triples where a `''` parent_state truly cannot be
    disambiguated.

    A (Z, A) with both a ground state and a low-lying isomer is "geometrically"
    ambiguous — but `decay_mode` often narrows it. Ga-73g decays by β⁻; Ga-73m
    decays by IT. A β⁻ row from Ga-73 must be Ga-73g. Only rows where *both*
    states support the same decay mode are genuinely ambiguous and become NULL;
    rows where the decay mode uniquely selects a state become `'g'`.

    Consulted against `decay.parquet`, which lists per (Z, A, state) the modes
    each supports. After the 2026.8.5 rebuild the ground/isomer supports for
    Ga-73 and U-235 are disjoint, so this returns the empty set. Kept because
    the invariant is worth naming even when it currently holds trivially — a
    future ENSDF revision could add an m-decay mode that collides.
    """
    low_isomer_parents = _parent_low_isomer_zA(data_dir)
    if not low_isomer_parents:
        return set()

    decay = data_dir / "meta" / "decay.parquet"
    if not decay.is_file():
        # No decay table means we cannot disambiguate — treat every (Z, A) as
        # ambiguous for every mode, so the migration errs on the safe side.
        raise UnmigratableTable("meta", "missing", f"{decay} is needed to disambiguate parent decays")

    df = pl.read_parquet(decay, columns=["Z", "A", "state", "decay_mode"]).unique()
    per_parent_state: dict[tuple[int, int], dict[str, set[str]]] = {}
    for z, a, state, mode in df.iter_rows():
        key = (int(z), int(a))
        per_parent_state.setdefault(key, {}).setdefault(state, set()).add(mode)

    ambig: set[tuple[int, int, str]] = set()
    for za in low_isomer_parents:
        modes_by_state = per_parent_state.get(za, {})
        ground_modes = modes_by_state.get(GROUND, set())
        isomer_modes = modes_by_state.get("m", set())
        for mode in ground_modes & isomer_modes:
            ambig.add((za[0], za[1], mode))
    return ambig


# G4 decay-mode string -> (delta_Z, delta_A) shift daughter -> parent. `None`
# means "cannot compute a parent (Z, A) from the daughter" — SF and unknown
# modes fall through, and the migration then plays it safe.
_MODE_TO_DA: dict[str, tuple[int, int]] = {
    "IT": (0, 0),
    "beta-": (-1, 0),
    "beta+": (1, 0),
    "KshellEC": (1, 0),
    "LshellEC": (1, 0),
    "MshellEC": (1, 0),
    "NshellEC": (1, 0),
    "alpha": (2, 4),
    "p": (1, 1),
    "n": (0, 1),
    "t": (1, 3),
}


def _parent_state_for_row(
    daughter_z: int | None,
    daughter_a: int | None,
    parent_decay_mode: str | None,
    ambiguous_parent_modes: set[tuple[int, int, str]],
    parent_z: int | None = None,
    parent_a: int | None = None,
) -> str | None:
    """Migrated `parent_state` for one row that shipped `''`.

    * `parent_decay_mode` null -> NULL (parent was not identified at all).
    * Parent (Z, A) is `daughter + shift` for a known decay mode, or the
      explicit `parent_z`/`parent_a` for the emissions table (where the row
      already carries them). If `(parent, mode)` is in `ambiguous_parent_modes`,
      NULL — the g4 builder's coarse tolerance cannot tell that (Z, A)'s
      ground from its low-lying isomer, and both states support this mode.
    * Otherwise `'g'`.
    """
    if parent_decay_mode is None:
        return None
    if parent_z is not None and parent_a is not None:
        parent_zA = (int(parent_z), int(parent_a))
    else:
        shift = _MODE_TO_DA.get(parent_decay_mode)
        if shift is None or daughter_z is None or daughter_a is None:
            return None  # cannot compute parent -> honest NULL
        parent_zA = (int(daughter_z) + shift[0], int(daughter_a) + shift[1])
    if (parent_zA[0], parent_zA[1], parent_decay_mode) in ambiguous_parent_modes:
        return None
    return GROUND


def migrate_parent_state_g4(data_dir: Path, table: str, *, dry_run: bool) -> tuple[int, str]:
    """Rewrite `parent_state == ''` in a g4 nuclide-keyed cascade table.

    The three tables this handles (coincidences, emissions, summing_partners)
    all key on `parent_state` naming the state of the decaying nuclide.

    * emissions is built from `decay_summary.state`, which uses
      `radioactive_decay._label_state` — an *exact* per-(Z, A, parent_ex_kev)
      lookup. `''` there unambiguously means `parent_ex_kev == 0`, i.e. ground,
      because separate rows exist for the isomers. `_parent_ambiguous_zA` is
      never triggered.
    * coincidences (via `mixed_coincidences._label_parent_state`) uses a coarse
      `parent_ex_kev < 1 keV → ''` rule. Ga-73m at 0.3 keV and U-235m at
      0.076 keV would collide with ground — but the current data ships zero
      IT-decay coincidence rows for those two parents (they are one-step
      decays with no cascade), and the other decay modes narrow the parent to
      ground unambiguously (Ga-73g β⁻ → Ge, U-235g α → Th).
    * summing_partners inherits from coincidences (`fill_null("")`), so the
      same rule applies.

    Rows with `parent_decay_mode` null are γ-γ pairs whose parent channel
    `annotate_gamma_gamma` could not identify; those become NULL because the
    `''` there was a stand-in default, not a determination.
    """
    paths = shards(data_dir, table)
    if not paths:
        raise UnmigratableTable(table, "no-shards", "declared but ships no parquet")

    ambiguous = _ambiguous_parent_decay_modes(data_dir)
    allowed_now = allowed_parent_states(table)

    changed = 0
    for path in paths:
        df = pl.read_parquet(path)
        if "parent_state" not in df.columns:
            raise UnmigratableTable(table, "no-parent-state-column", str(path))
        present = {v for v in df["parent_state"].unique().to_list() if v is not None}
        stray = sorted(present - allowed_now)
        if stray:
            raise UnmigratableTable(table, "unknown-parent-state", f"{path.name}: {stray}")
        if LEGACY_UNSPECIFIED not in present:
            continue

        blank = pl.col("parent_state") == LEGACY_UNSPECIFIED

        # Emissions carries parent_Z / parent_A explicitly; coincidences and
        # summing_partners carry only daughter (Z, A) and derive the parent.
        if table == "meta/ensdf/emissions":
            new_parent_state = pl.Series(
                [
                    _parent_state_for_row(
                        None,
                        None,
                        pdm,
                        ambiguous,
                        parent_z=pz,
                        parent_a=pa,
                    )
                    if is_blank
                    else current
                    for is_blank, current, pdm, pz, pa in zip(
                        (df["parent_state"] == LEGACY_UNSPECIFIED).to_list(),
                        df["parent_state"].to_list(),
                        df["decay_mode"].to_list(),
                        df["parent_Z"].to_list(),
                        df["parent_A"].to_list(),
                    )
                ],
                dtype=pl.Utf8,
            )
        else:
            new_parent_state = pl.Series(
                [
                    _parent_state_for_row(dz, da, pdm, ambiguous) if is_blank else current
                    for is_blank, current, dz, da, pdm in zip(
                        (df["parent_state"] == LEGACY_UNSPECIFIED).to_list(),
                        df["parent_state"].to_list(),
                        df["Z"].to_list(),
                        df["A"].to_list(),
                        df["parent_decay_mode"].to_list(),
                    )
                ],
                dtype=pl.Utf8,
            )

        # A row changed if the string differs OR the nullness flipped. Polars
        # `!=` between a string Series and a nullable one propagates null, so
        # `'' != None` returns null, not True — using `ne_missing` treats null
        # as a peer value and produces the boolean we want. This is the same
        # subtlety the `xs`-side count formula worked around with an over-count.
        old = df["parent_state"]
        differs = old.ne_missing(new_parent_state)
        _ = blank  # anchor for the reader: only blanks are ever rewritten
        changed += int(differs.sum())
        if not dry_run:
            df.with_columns(new_parent_state.alias("parent_state")).write_parquet(path, compression=COMPRESSION)

    return (changed, "already-migrated" if changed == 0 else "migrated")


def migrate_daughter_state_decay(data_dir: Path, *, dry_run: bool) -> tuple[int, str]:
    """Rewrite `daughter_state == ''` in `meta/decay.parquet` to NULL.

    Every one of the 6,431 shipped rows carries `''` — the builder
    (`radioactive_decay.py::build_decay_table`) computes a daughter label per
    `daughter_ex_kev` and *falls back to* `''` when the daughter doesn't match
    a catalogued m-state. A summary row can populate many daughter levels, so
    "the daughter state" is not a well-defined single value at this granularity
    in the first place. The right shape is NULL: "we cannot name a single
    daughter state from this summary row".

    Not `'g'`: many summary rows never populate the daughter's ground state at
    all (2,283 of 5,855 rows with detail have `min_daughter_ex_kev > 0`),
    so relabelling `''` as `'g'` would join to a state the decay may not even
    reach. NULL joins nothing, which is right.
    """
    path = data_dir / "meta" / "decay.parquet"
    if not path.is_file():
        raise UnmigratableTable("meta", "missing", str(path))
    df = pl.read_parquet(path)
    if "daughter_state" not in df.columns:
        raise UnmigratableTable("meta", "no-daughter-state-column", str(path))

    allowed_now = allowed_daughter_states("meta")
    present = {v for v in df["daughter_state"].unique().to_list() if v is not None}
    stray = sorted(present - allowed_now)
    if stray:
        raise UnmigratableTable("meta", "unknown-daughter-state", f"{path.name}: {stray}")
    if LEGACY_UNSPECIFIED not in present:
        return 0, "already-migrated"

    changed = int((df["daughter_state"] == LEGACY_UNSPECIFIED).sum())
    new = (
        pl.when(pl.col("daughter_state") == LEGACY_UNSPECIFIED)
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col("daughter_state"))
    )
    if not dry_run:
        df.with_columns(new.alias("daughter_state")).write_parquet(path, compression=COMPRESSION)
    return changed, "migrated"


def rename_state_to_phase(data_dir: Path, target: str, *, dry_run: bool) -> tuple[int, str]:
    """`density_effect_params`' `state` column holds solid/liquid/gas.

    Rename, never revalue: the values were always correct, the *name* was the
    defect. `target` is a file path relative to `data/`, because the directory
    also holds a parquet that never had the column.
    """
    path = data_dir / target
    if not path.is_file():
        raise UnmigratableTable(target, "missing", f"{path} is not a file")

    df = pl.read_parquet(path)
    if "phase" in df.columns and "state" not in df.columns:
        return 0, "already-migrated"
    if "state" not in df.columns:
        raise UnmigratableTable(target, "no-state-column", str(path))
    if "phase" in df.columns:
        raise UnmigratableTable(target, "both-columns", "has `state` and `phase`")

    if not dry_run:
        df.rename({"state": "phase"}).write_parquet(path, compression=COMPRESSION)
    return df.height, "migrated"


def verify(data_dir: Path) -> list[str]:
    """Every complaint the new vocabulary has about the tree. Empty is success.

    Consults `allowed_*` (which include a table's `PENDING_*` debt) so a table
    still mid-migration is not double-reported by both the ledger and verify.
    Once the ledger empties, `allowed_*` shrinks and any lingering retired
    spelling becomes a real complaint here.
    """
    problems: list[str] = []
    for table in sorted(TABLE_STATES):
        directory = data_dir / table
        if not directory.is_dir():
            problems.append(f"{table}: declared but absent")
            continue
        seen: set[str | None] = set()
        for path in sorted(directory.glob("*.parquet")):
            if "state" not in pl.read_parquet_schema(path):
                continue  # sibling with no state column (data/meta is mixed)
            seen.update(pl.read_parquet(path, columns=["state"])["state"].unique().to_list())
        allowed = allowed_states(table)
        outside = sorted(v for v in seen if v is not None and v not in allowed)
        if outside:
            problems.append(f"{table}: state {outside} outside its vocabulary")
    for table in sorted(TABLE_PARENT_STATES):
        directory = data_dir / table
        if not directory.is_dir():
            problems.append(f"{table}: declared for parent_state but absent")
            continue
        seen = set()
        for path in sorted(directory.glob("*.parquet")):
            seen.update(pl.read_parquet(path, columns=["parent_state"])["parent_state"].unique().to_list())
        allowed = allowed_parent_states(table)
        outside = sorted(v for v in seen if v is not None and v not in allowed)
        if outside:
            problems.append(f"{table}: parent_state {outside} outside its vocabulary")
    for table in sorted(TABLE_DAUGHTER_STATES):
        directory = data_dir / table
        if not directory.is_dir():
            problems.append(f"{table}: declared for daughter_state but absent")
            continue
        seen = set()
        for path in sorted(directory.glob("*.parquet")):
            if "daughter_state" not in pl.read_parquet_schema(path):
                continue
            seen.update(pl.read_parquet(path, columns=["daughter_state"])["daughter_state"].unique().to_list())
        allowed = allowed_daughter_states(table)
        outside = sorted(v for v in seen if v is not None and v not in allowed)
        if outside:
            problems.append(f"{table}: daughter_state {outside} outside its vocabulary")
    # `PHASE_NOT_STATE`, not `PENDING_COLUMN_RENAME`: the debt ledger empties
    # when the rename lands, but "this file must never call a phase of matter a
    # `state`" is permanent, and a verify that stops checking it the moment it
    # passes would not notice the column coming back.
    for target in sorted(PHASE_NOT_STATE):
        path = data_dir / target
        if path.is_file() and "state" in pl.read_parquet_schema(path):
            problems.append(f"{target}: still has a `state` column")
    return problems


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Migrate shipped parquets into the #357 `state` vocabulary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--table", help="migrate only this table (e.g. tendl-2023-iso/xs)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check the tree against the vocabulary and exit; writes nothing",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.verify:
        problems = verify(args.data_dir)
        for problem in problems:
            logger.error("  %s", problem)
        if problems:
            raise SystemExit(f"{len(problems)} table(s) do not match the vocabulary")
        logger.info("every table matches the #357 vocabulary")
        return

    # `parent_state` and `daughter_state` are tracked as `<table>:<column>` so
    # a single table can appear once per column and each migration is dispatched
    # independently. `state` migrations stay as bare `<table>` keys.
    if args.table:
        wanted: list[str] = [args.table]
    else:
        wanted = [
            *sorted(PENDING_MIGRATION),
            *(f"{t}:parent_state" for t in sorted(PENDING_PARENT_STATE_MIGRATION)),
            *(f"{t}:daughter_state" for t in sorted(PENDING_DAUGHTER_STATE_MIGRATION)),
            *sorted(PENDING_COLUMN_RENAME),
        ]
    unknown = [
        t
        for t in wanted
        if (
            (":" not in t and t not in TABLE_STATES and t not in PHASE_NOT_STATE)
            or (t.endswith(":parent_state") and t.split(":")[0] not in TABLE_PARENT_STATES)
            or (t.endswith(":daughter_state") and t.split(":")[0] not in TABLE_DAUGHTER_STATES)
        )
    ]
    if unknown:
        raise SystemExit(f"not a declared table: {unknown}")

    total = 0
    statuses: dict[str, str] = {}
    for spec in wanted:
        if spec.endswith(":parent_state"):
            table = spec.split(":")[0]
            rows, status = migrate_parent_state_g4(args.data_dir, table, dry_run=args.dry_run)
        elif spec.endswith(":daughter_state"):
            rows, status = migrate_daughter_state_decay(args.data_dir, dry_run=args.dry_run)
        elif spec == "meta/ensdf":
            rows, status = migrate_nuclides(args.data_dir, dry_run=args.dry_run)
        elif spec in _NUCLIDE_KEYED:
            rows, status = migrate_nuclide_keyed(args.data_dir, spec, dry_run=args.dry_run)
        elif spec in PHASE_NOT_STATE:
            rows, status = rename_state_to_phase(args.data_dir, spec, dry_run=args.dry_run)
        else:
            rows, status = migrate_table(args.data_dir, spec, dry_run=args.dry_run)
        statuses[spec] = status
        total += rows
        logger.info("  %-40s %-16s %8d row(s)", spec, status, rows)

    bad = {t: s for t, s in statuses.items() if s not in SUCCESS_STATUSES}
    if bad:
        raise SystemExit(f"tables not in the new vocabulary: {bad}")

    verb = "would change" if args.dry_run else "changed"
    logger.info("%s %d row(s) across %d table(s)", verb, total, len(statuses))
    if not args.dry_run:
        logger.info("re-run with --verify to confirm, then recompute catalog.json::data_sha256")


if __name__ == "__main__":
    main()
