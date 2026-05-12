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
