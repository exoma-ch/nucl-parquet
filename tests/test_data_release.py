"""Gates against drift between `data/**/*.parquet` and `data/catalog.json`.

The PR that changes data IS the data-release PR — these checks make sure
the version identifier and the actual data move together. Failure modes
this catches:

1. Silent drift — parquets changed, `data_version` did not (consumers fetch
   the same tag and get different bytes).
2. Cosmetic bump — `data_version` changed, parquets did not (breaks
   reproducibility; consumers re-download for nothing).
3. Version regression — `data_version` not valid CalVer, or moves backwards.

The first two collapse into a single check thanks to `data_sha256`: if the
on-disk SHA-256 tree hash disagrees with the catalog's claim, something is
out of sync regardless of which side moved.

Tests are *not* marked `@pytest.mark.data` — they must run in every PR's
CI regardless of whether `--no-data` is passed.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path

import jsonschema
import pytest

import nucl_parquet

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_CATALOG = _DATA_DIR / "catalog.json"
_SCHEMA = _DATA_DIR / "catalog.schema.json"
_CALVER_RE = re.compile(r"^[0-9]{4}\.[0-9]+\.[0-9]+$")


# -- Layer 1.a: catalog.json conforms to its own schema ----------------------


def test_catalog_validates_against_schema() -> None:
    """catalog.json must satisfy catalog.schema.json.

    Catches: missing required fields, malformed `data_version`, bad
    `data_sha256` shape, schema-breaking edits that ship without
    catalog update.
    """
    catalog = json.loads(_CATALOG.read_text())
    schema = json.loads(_SCHEMA.read_text())
    jsonschema.validate(catalog, schema)


# -- Layer 1.b: data_version is valid CalVer ---------------------------------


def test_data_version_is_calver() -> None:
    """`data_version` must match YYYY.MM.MICRO (e.g. `2026.5.0`)."""
    version = nucl_parquet.data_version(_DATA_DIR)
    assert _CALVER_RE.match(version), (
        f"data_version={version!r} not YYYY.MM.MICRO; bumps should look like '2026.5.0' or '2026.5.1'"
    )


# -- Layer 1.c: SHA anchor matches the on-disk tree --------------------------


def test_data_sha256_matches_on_disk_tree() -> None:
    """The catalog's `data_sha256` must match the recomputed tree hash.

    This is the load-bearing gate. It catches:
      - silent drift: parquet edited, sha not updated
      - cosmetic bump: sha edited without a real data change
    If you intentionally changed parquets, also bump `data_version` AND
    update `data_sha256` to the new tree hash. Recompute via:
        uv run python -c 'import nucl_parquet; print(nucl_parquet.compute_data_sha256())'
    """
    declared = nucl_parquet.data_sha256(_DATA_DIR)
    actual = nucl_parquet.compute_data_sha256(_DATA_DIR)
    assert declared == actual, (
        f"data_sha256 mismatch:\n  declared (catalog.json): {declared}\n"
        f"  actual   (on-disk tree): {actual}\n\n"
        "Either:\n"
        "  (a) you changed parquet files — bump data_version AND set data_sha256 to the actual value above\n"
        "  (b) you set data_sha256 by hand — recompute with:\n"
        "      uv run python -c 'import nucl_parquet; print(nucl_parquet.compute_data_sha256())'"
    )


# -- Layer 1.d: version and sha co-change in PR diffs ------------------------


def _git_changed_files(base: str) -> list[str]:
    """Return paths that differ between `base` and HEAD. Tolerates the
    base being absent locally (returns [] so the test no-ops outside CI)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _git_value_at(ref: str, path: str, jsonpath_key: str) -> str | None:
    """Read a top-level key from `path` as it existed at git ref `ref`.

    Returns None if the file or key wasn't present at `ref` (e.g. first
    introduction of the field).
    """
    try:
        blob = subprocess.check_output(
            ["git", "show", f"{ref}:{path}"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    try:
        return json.loads(blob).get(jsonpath_key)
    except json.JSONDecodeError:
        return None


def test_version_and_sha_co_change_in_pr() -> None:
    """If parquets changed in this PR, `data_version` AND `data_sha256` must change too.

    Symmetric: if `data_version` OR `data_sha256` changed but no parquet
    files did, that's a cosmetic bump — also a fail.

    Skips outside a PR context (no `origin/main` to diff against), and
    silently passes if the SHA anchor itself wasn't yet introduced on
    `origin/main` (the very first PR adding it).
    """
    # Determine the merge base. In CI under PR builds, GITHUB_BASE_REF is
    # the target branch (e.g. "main"); locally we fall back to `origin/main`.
    import os

    base_ref = os.environ.get("GITHUB_BASE_REF") or "main"
    base = f"origin/{base_ref}"

    changed = _git_changed_files(base)
    if not changed:
        pytest.skip(f"No diff against {base} (running outside PR context, or branch is up-to-date).")

    parquet_changed = any(f.startswith("data/") and f.endswith(".parquet") for f in changed)
    catalog_changed = "data/catalog.json" in changed

    if not (parquet_changed or catalog_changed):
        pytest.skip("PR does not touch data/ or catalog.json — gate not applicable.")

    base_version = _git_value_at(base, "data/catalog.json", "data_version")
    base_sha = _git_value_at(base, "data/catalog.json", "data_sha256")
    current_version = nucl_parquet.data_version(_DATA_DIR)
    current_sha = nucl_parquet.data_sha256(_DATA_DIR)

    version_moved = base_version is not None and base_version != current_version
    sha_moved = base_sha is not None and base_sha != current_sha

    # Bootstrap exception: the PR that first introduces `data_sha256` is
    # allowed to update `data_version` without a parquet change — it's the
    # one-time format migration / field introduction, not a cosmetic bump.
    # After this lands, every future PR sees `data_sha256` on base and the
    # check applies normally.
    sha_bootstrapped = base_sha is None and current_sha is not None

    if parquet_changed:
        # Real data change: both must move.
        assert version_moved, (
            f"Parquet files changed but data_version did not (still {current_version!r}). "
            "Bump it to today's CalVer (e.g. 2026.5.X) in the same PR."
        )
        assert sha_moved or sha_bootstrapped, (
            "Parquet files changed but data_sha256 did not. "
            "Recompute the tree hash and update catalog.json::data_sha256."
        )
    elif (version_moved or sha_moved) and not sha_bootstrapped:
        # Cosmetic bump: neither parquets nor anything else justifying the move.
        pytest.fail(
            f"data_version / data_sha256 changed but no parquet files did. "
            f"version: {base_version!r} -> {current_version!r}, "
            f"sha: {base_sha!r} -> {current_sha!r}. "
            "Either revert the catalog edit, or include the data change that motivates it."
        )


# -- Layer 1.e: monotonicity vs main -----------------------------------------


def test_data_version_monotonic_vs_main() -> None:
    """`data_version` must be >= the value on the base branch.

    Catches: cherry-picking an old data revision back into a PR, or hand-
    editing the version backwards. CalVer YYYY.MM.MICRO compared by
    numeric tuple `(year, month, micro)` so `2026.5.10 > 2026.5.9` works.

    Bootstrap exception: when the SHA anchor (`data_sha256`) was absent
    on base and is introduced by this PR, we're doing a format migration
    or first-time field introduction; the version may be re-baselined to
    start a new MICRO sequence (e.g. `2026.05.11` legacy YYYY.MM.DD → new
    `2026.5.0`). After this lands, future PRs see the anchor on base and
    the monotonic check applies normally.
    """
    import os

    base_ref = os.environ.get("GITHUB_BASE_REF") or "main"
    base = f"origin/{base_ref}"

    base_version = _git_value_at(base, "data/catalog.json", "data_version")
    base_sha = _git_value_at(base, "data/catalog.json", "data_sha256")
    current_sha = nucl_parquet.data_sha256(_DATA_DIR)
    if base_version is None:
        pytest.skip(f"No catalog at {base} (first introduction).")
    if base_sha is None and current_sha is not None:
        pytest.skip(
            "Bootstrap commit introducing data_sha256 — version may re-baseline. "
            "Future PRs will have the anchor on base and this check will apply normally."
        )

    current_version = nucl_parquet.data_version(_DATA_DIR)

    def _tup(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in v.split("."))

    assert _tup(current_version) >= _tup(base_version), (
        f"data_version regression: {current_version!r} < {base_version!r} on {base}. CalVer must move forward."
    )


# -- Layer 1.f: the catalog describes what actually ships --------------------


def test_declared_projectiles_match_the_files_on_disk() -> None:
    """Every projectile a library claims must have at least one file.

    `iaea-medical` declared `["p", "d", "h", "a"]` and shipped only p and d
    (#310). A consumer enumerating the catalog to build a UI or a coverage
    matrix gets two entries that resolve to nothing — and finds out at query
    time, not at load time.

    This is the catalog's whole job. It is the single source of truth for what
    exists, so a claim it cannot back is worse than an omission: an omission is
    visibly incomplete, a false claim looks authoritative.

    Checked in both directions — an undeclared projectile shipping files is the
    same defect seen from the other side.
    """
    import collections
    import glob
    import os

    catalog = json.loads(_CATALOG.read_text())
    problems = []
    for name, lib in sorted(catalog["libraries"].items()):
        claimed, path = lib.get("projectiles"), lib.get("path")
        if not claimed or not path:
            continue
        on_disk = collections.Counter(
            os.path.basename(f).split("_")[0] for f in glob.glob(str(_DATA_DIR / path) + "*.parquet")
        )
        if not on_disk:  # library not present in this checkout
            continue
        missing = [p for p in claimed if on_disk.get(p, 0) == 0]
        undeclared = [p for p in on_disk if p not in claimed]
        if missing:
            problems.append(f"{name}: claims {missing} but ships no such files")
        if undeclared:
            problems.append(f"{name}: ships {undeclared} but does not declare them")
    assert not problems, "catalog.json disagrees with the data tree:\n  " + "\n  ".join(problems)


# -- Layer 1.g: physical sanity on the published data ------------------------


def test_no_negative_energies_anywhere() -> None:
    """An incident energy below zero is nonphysical.

    Three reached the published data (#330). The builders guard the cross
    section (`if xs <= 0: continue`) but never guarded the energy, so a sign
    error in a transcribed EXFOR entry passed straight into a column that every
    interpolation routine trusts without checking.

    Fixed in the 2026.8.3 re-ingest — the guard removed exactly those three
    rows and nothing else (exfor-channels 4,162,404 -> 4,162,401). The gate
    stays so a future sign error cannot ship.
    """
    import duckdb

    # Iterate the catalog's declared library paths rather than globbing all of
    # data/. A bare `**/*.parquet` sweeps in data/g4_raw/ — a gitignored build
    # cache holding a truncated file — and the meta tables, which have no
    # energy column at all. Checking exactly what the catalog declares is both
    # narrower and more meaningful.
    catalog = json.loads(_CATALOG.read_text())
    offenders = {}
    for name, lib in sorted(catalog["libraries"].items()):
        path = lib.get("path")
        if not path:
            continue
        glob = f"{_DATA_DIR}/{path}*.parquet"
        try:
            n = duckdb.sql(
                f"SELECT count(*) FROM read_parquet('{glob}', union_by_name=true) WHERE energy_MeV < 0"
            ).fetchone()[0]
        except duckdb.Error:
            continue  # library absent from this checkout, or carries no energy column
        if n:
            offenders[name] = n
    assert not offenders, f"rows with negative energy_MeV: {offenders} — see #330"


@dataclasses.dataclass(frozen=True)
class Anchor:
    """One well-known reaction, checked across every library that ships it.

    Two halves, and both are load-bearing:

    `libraries` is the set that **must** produce a value. Without it a spot-check
    can match nothing and report success — which is not hypothetical. The first
    version of the Au-197 capture gate asserted `xs_mb == 0`, found no rows, and
    passed while `cendl-3.2` was wrong by nineteen orders of magnitude. It is a
    lower bound, not an equality: a library added later that does not tabulate
    this point is not a failure, but one that stops tabulating it is (IRDFF's
    Li/B/F had simply never been ingested, and nothing noticed — #334).

    `witness` is a value the band must reject, so the assertion is proved to have
    teeth rather than assumed to. Where a real regression exists, `witness` is
    the number the shipped data actually had.
    """

    name: str
    element: str
    #: SQL predicate selecting the channel. Identity is (target, residual) —
    #: `MT` is present in the canonical schema but null throughout the evaluated
    #: libraries (#347), so it cannot be used to select here.
    where: str
    expected_mb: float
    #: Multiplicative half-width. A magnitude band, never equality: evaluations
    #: legitimately disagree at the few-percent-to-tens-of-percent level.
    factor: float
    libraries: frozenset[str]
    witness: float
    witness_note: str

    def in_band(self, xs: float) -> bool:
        return self.expected_mb / self.factor < xs < self.expected_mb * self.factor


_ALL_ENDF = frozenset(
    {
        "brond-3.1",
        "cendl-3.2",
        "endfb-8.0",
        "fendl-3.2",
        "irdff-2",
        "jeff-4.0",
        "jendl-5",
        "jendl-ad-2017",
        "tendl-2025",
    }
)

#: The spot-checks. Cheap — eight grouped scans over the committed parquets —
#: and between them they reject every published defect in this repository's
#: history that a single number could have caught.
PHYSICAL_ANCHORS: tuple[Anchor, ...] = (
    Anchor(
        name="Au-197(n,g) thermal",
        element="Au",
        where="target_Z=79 AND target_A=197 AND residual_Z=79 AND residual_A=198 "
        "AND energy_MeV BETWEEN 2.0e-8 AND 3.0e-8",
        expected_mb=98_700.0,  # the reference dosimetry standard, 98.7 b
        factor=5.0,
        # Most evaluated libraries do not tabulate down to thermal at all; not
        # sampling thermal is a scope choice, and failing it here would conflate
        # absence with wrongness. Only the three that reach it are required.
        libraries=frozenset({"endfb-8.0", "irdff-2", "jendl-ad-2017"}),
        witness=1.0002e-14,
        witness_note="what cendl-3.2 actually shipped (#328, #287) — MT=2 elastic mislabelled "
        "as (n,g), and CENDL's real capture tabulation does not reach thermal",
    ),
    Anchor(
        name="Au-197(n,g) @ 1 MeV",
        element="Au",
        where="target_Z=79 AND target_A=197 AND residual_Z=79 AND residual_A=198 AND energy_MeV BETWEEN 0.9 AND 1.1",
        expected_mb=82.7,
        factor=3.0,
        # Every neutron library tabulates 1 MeV, which is what makes this the
        # broadest of the capture anchors.
        libraries=_ALL_ENDF,
        witness=3952.6,
        witness_note="what brond-3.1, fendl-3.2 and jeff-4.0 all shipped for thirteen months "
        "before #334 — potential scattering (~barns) swamping real capture (~mb)",
    ),
    Anchor(
        name="Fe-56(n,g)Fe-57 @ 1 MeV",
        element="Fe",
        where="target_Z=26 AND target_A=56 AND residual_Z=26 AND residual_A=57 AND energy_MeV BETWEEN 0.9 AND 1.1",
        expected_mb=2.9,
        factor=3.0,
        libraries=_ALL_ENDF - {"irdff-2"},
        witness=2056.9,
        witness_note="what fendl-3.2 and jeff-4.0 shipped before #334 — a factor of 700, the "
        "sharpest of the elastic-as-capture signatures",
    ),
    Anchor(
        name="H-1(n,g)D thermal",
        element="H",
        where="target_Z=1 AND target_A=1 AND residual_Z=1 AND residual_A=2 AND energy_MeV BETWEEN 2.0e-8 AND 3.0e-8",
        expected_mb=332.0,
        factor=2.0,
        libraries=frozenset({"brond-3.1", "cendl-3.2", "endfb-8.0", "fendl-3.2", "jeff-4.0", "jendl-5"}),
        witness=20_436.33,
        witness_note="what brond-3.1 and fendl-3.2 shipped before #334 — H-1's free-atom "
        "elastic cross section, 20.4 b, in the capture channel",
    ),
    Anchor(
        name="Fe-56(n,p)Mn-56 @ 14 MeV",
        element="Fe",
        where="target_Z=26 AND target_A=56 AND residual_Z=25 AND residual_A=56 AND energy_MeV BETWEEN 13.5 AND 14.5",
        expected_mb=113.0,
        factor=2.0,
        libraries=_ALL_ENDF,
        witness=0.1143,
        witness_note="what fendl-3.2 shipped (#326) — a `unique(subset=…)` collapsing the "
        "MT-partials that share a residual down to a single one",
    ),
    Anchor(
        name="Au-197(n,2n)Au-196 @ 14 MeV",
        element="Au",
        where="target_Z=79 AND target_A=197 AND residual_Z=79 AND residual_A=196 AND energy_MeV BETWEEN 13.5 AND 14.5",
        expected_mb=2120.0,
        factor=2.0,
        libraries=_ALL_ENDF,
        witness=212.0,
        witness_note="no historical regression — a 10x error, the shape a mb/b confusion takes",
    ),
    Anchor(
        name="Ni-58(n,p)Co-58 @ 14.5 MeV",
        element="Ni",
        where="target_Z=28 AND target_A=58 AND residual_Z=27 AND residual_A=58 AND energy_MeV BETWEEN 14.2 AND 14.8",
        expected_mb=310.0,
        factor=2.0,
        libraries=_ALL_ENDF,
        witness=31.0,
        witness_note="no historical regression — a 10x error",
    ),
    Anchor(
        name="Al-27(n,a)Na-24 @ 14.5 MeV",
        element="Al",
        where="target_Z=13 AND target_A=27 AND residual_Z=11 AND residual_A=24 AND energy_MeV BETWEEN 14.2 AND 14.8",
        expected_mb=116.0,
        factor=2.0,
        libraries=frozenset({"cendl-3.2", "irdff-2", "jeff-4.0", "jendl-5", "jendl-ad-2017", "tendl-2025"}),
        witness=1160.0,
        witness_note="no historical regression — a 10x error",
    ),
    # U-235(n,f) thermal (585 b) and U-235 total thermal are the obvious
    # additions and cannot be written: `mt_to_residual` returns None for
    # fission, total, elastic and inelastic, and the caller drops the row rather
    # than emitting it with a null residual. Across 17.3M evaluated rows there
    # are zero null-residual rows and zero MT values. Tracked in #347; add both
    # anchors here when the ingest carries those channels.
)


def _anchor_rows(anchor: Anchor) -> list[tuple[str, float]]:
    import duckdb

    return duckdb.sql(f"""
        SELECT library, avg(xs_mb)
        FROM read_parquet('{_DATA_DIR}/*/xs/n_{anchor.element}.parquet', union_by_name=true)
        WHERE {anchor.where}
        GROUP BY library
    """).fetchall()


@pytest.mark.parametrize("anchor", PHYSICAL_ANCHORS, ids=lambda a: a.name)
def test_well_known_reactions_are_physically_plausible(anchor: Anchor) -> None:
    """Every library shipping this reaction must agree with the textbook value.

    This is the cheap half of #342: a full rebuild-and-compare is a multi-GB
    download per library and cannot run in PR CI, but a handful of reactions
    whose magnitude is not in dispute can be checked on the committed parquets
    in under a second — and *would* have caught the thirteen-month gap on day
    one. See `Anchor` for why both halves of each entry matter.
    """
    rows = _anchor_rows(anchor)
    present = {lib for lib, xs in rows if xs is not None}

    missing = anchor.libraries - present
    assert not missing, (
        f"{anchor.name}: no rows from {sorted(missing)}, which ship this reaction. "
        f"Found {sorted(present)}. A library that stops tabulating a channel it used to "
        "carry is a silent ingest failure (#334) — and a spot-check that matches nothing "
        "and passes is the failure mode #342 exists to prevent."
    )

    bad = {lib: xs for lib, xs in rows if xs is not None and not anchor.in_band(xs)}
    assert not bad, (
        f"{anchor.name} is implausible in {bad} — expected ~{anchor.expected_mb:,.4g} mb "
        f"within a factor of {anchor.factor:g} (libraries checked: {sorted(present)})."
    )


@pytest.mark.parametrize("anchor", PHYSICAL_ANCHORS, ids=lambda a: a.name)
def test_every_anchor_rejects_the_value_it_was_written_against(anchor: Anchor) -> None:
    """Prove the bands have teeth instead of assuming it.

    #342's whole premise is that a check which quietly matches nothing reports
    success. The same is true of a band so wide it accepts the bug: the first
    Au-197 gate asserted `xs_mb == 0` and passed against data that was wrong by
    1e19. So each anchor carries a `witness` — for five of the eight, the number
    the published data actually had — and this asserts the band rejects it.

    If widening a `factor` ever makes this fail, the band has stopped being a
    check and the anchor should be dropped rather than kept as decoration.
    """
    assert not anchor.in_band(anchor.witness), (
        f"{anchor.name}: factor {anchor.factor:g} around {anchor.expected_mb:,.4g} mb still "
        f"accepts {anchor.witness:,.4g} mb ({anchor.witness_note}). The band is too wide to "
        "catch the defect it was written for."
    )
