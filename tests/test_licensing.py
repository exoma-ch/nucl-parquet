"""Licence-manifest gate — every tracked file under `data/` must be recorded.

Three classes of failure this catches (#232):

1. **Unrecorded data** — a Parquet tree lands under `data/` without an entry in
   `data/licenses.toml`, so it ships with no provenance and no citation. The
   gate fires from BOTH sides: an uncovered file, AND a manifest glob that
   matches nothing (the entry moved or is dead).

2. **Ambiguity** — two entries claim the same file, so the effective licence
   is undefined.

3. **Generated-artifact drift** — the ATTRIBUTION.md AUTO section or a
   per-library `LICENSE.txt` sidecar stops matching the manifest.

Fix for 1: add/adjust the entry's `paths`. Fix for 2: narrow the overlapping
globs. Fix for 3: `python scripts/build_notices.py --write`.

Needs a git checkout only — no data tree files opened, no network.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nucl_parquet import licensing

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def manifest() -> dict:
    return licensing.load_manifest()


@pytest.fixture(scope="module")
def tracked(manifest: dict) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *manifest["manifest"]["data_roots"]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


# ---------------------------------------------------------------------------
# 1. Coverage on the real repo
# ---------------------------------------------------------------------------


def test_every_data_file_has_recorded_provenance(manifest, tracked):
    coverage = licensing.check_coverage(manifest, tracked)
    assert not coverage.uncovered, (
        f"{len(coverage.uncovered)} bundled data file(s) have no entry in "
        f"data/licenses.toml, so they ship with no licence or citation: "
        f"{coverage.uncovered[:10]}"
    )


def test_no_stale_path_globs(manifest, tracked):
    """A glob matching nothing means the tree moved or the entry is dead."""
    coverage = licensing.check_coverage(manifest, tracked)
    assert not coverage.unused_patterns, f"manifest globs match nothing (stale entry?): {coverage.unused_patterns}"


def test_no_file_claimed_by_two_entries(manifest, tracked):
    """Overlapping globs make the effective licence of a file ambiguous."""
    coverage = licensing.check_coverage(manifest, tracked)
    ambiguous = {p: keys for p, keys in coverage.covered.items() if len(keys) > 1}
    assert not ambiguous, f"files claimed by multiple manifest entries: {dict(list(ambiguous.items())[:5])}"


def test_nothing_non_redistributable_is_bundled(manifest, tracked):
    """`redistributable = false` means the tree must not exist in the repo."""
    covered = licensing.check_coverage(manifest, tracked).covered
    ents = licensing.entries(manifest)
    offenders = {path: key for path, keys in covered.items() for key in keys if ents[key]["redistributable"] is False}
    assert not offenders, f"non-redistributable data is bundled: {offenders}"


# ---------------------------------------------------------------------------
# 2. Manifest structure
# ---------------------------------------------------------------------------


def test_every_manifest_entry_is_complete(manifest):
    """Missing a citation or terms URL silently produces a defective sidecar."""
    required = (
        "name",
        "custodian",
        "terms_url",
        "license",
        "redistributable",
        "risk",
        "citation",
        "paths",
    )
    for key, entry in licensing.entries(manifest).items():
        missing = [field for field in required if field not in entry]
        assert not missing, f"{key} is missing required field(s): {missing}"


def test_copyright_line_matches_the_license_file(manifest):
    """`[manifest].copyright` must be the string LICENSE actually stamps.

    Held in the manifest so the pending flip to "ETH Zürich" (issue #239) is a
    one-field edit that flows into every generated sidecar. If LICENSE moves
    the string, this fails until the manifest catches up.
    """
    line = licensing.copyright_line(manifest)
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert line in license_text, (
        f"[manifest].copyright = {line!r} does not appear in LICENSE — the "
        "single-field indirection has drifted from the source."
    )


# ---------------------------------------------------------------------------
# 3. Generated-artifact drift
# ---------------------------------------------------------------------------


def test_generated_artifacts_are_current():
    """The CI gate: `scripts/build_notices.py` in check mode must exit 0."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_notices.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        "Licence artifacts have drifted from data/licenses.toml.\n"
        "Run: python scripts/build_notices.py --write\n"
        f"{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 4. The gate must actually fail — red-before / green-after
#
# The three checks above pass on a clean tree by finding nothing wrong. That
# is the same shape as `test_no_library_shaped_directory_at_repo_root` in
# `test_repo_layout.py`: a matcher that never fires and a broken matcher look
# identical. Below we synthesise the two failure modes on a tmp manifest so
# the gate itself is exercised.
# ---------------------------------------------------------------------------


_MINIMAL_MANIFEST = """
schema_version = 2

[manifest]
maintainer = "test"
copyright = "Copyright (c) 2024-2026 eXoma"
data_roots = ["data"]
project_metadata = []

[libraries.example]
name = "Example"
custodian = "test"
terms_url = "https://example.invalid"
license = "test"
redistributable = true
risk = "green"
citation = "test"
paths = ["data/example/**"]
"""


def test_gate_fails_on_an_unclaimed_tracked_file(tmp_path):
    """An unclaimed file must be reported as uncovered."""
    m_path = tmp_path / "licenses.toml"
    m_path.write_text(textwrap.dedent(_MINIMAL_MANIFEST), encoding="utf-8")
    manifest = licensing.load_manifest(m_path)
    tracked = ["data/example/covered.parquet", "data/orphan/unclaimed.parquet"]
    coverage = licensing.check_coverage(manifest, tracked)
    assert coverage.uncovered == ["data/orphan/unclaimed.parquet"]
    assert coverage.covered == {"data/example/covered.parquet": ["libraries.example"]}
    assert not coverage.ok


def test_gate_fails_on_an_unused_glob(tmp_path):
    """A manifest pattern that matches nothing must be reported as unused."""
    m_path = tmp_path / "licenses.toml"
    m_path.write_text(
        textwrap.dedent(_MINIMAL_MANIFEST)
        + '\n[libraries.dead]\nname = "d"\ncustodian = "d"\nterms_url = "d"\n'
        + 'license = "d"\nredistributable = true\nrisk = "green"\ncitation = "d"\n'
        + 'paths = ["data/nothing_here/**"]\n',
        encoding="utf-8",
    )
    manifest = licensing.load_manifest(m_path)
    tracked = ["data/example/covered.parquet"]
    coverage = licensing.check_coverage(manifest, tracked)
    assert coverage.unused_patterns == ["libraries.dead: data/nothing_here/**"]
    assert not coverage.ok


def test_gate_reports_ambiguity_when_two_entries_overlap(tmp_path):
    """Overlapping globs must be visible to the check, not silently merged."""
    m_path = tmp_path / "licenses.toml"
    m_path.write_text(
        textwrap.dedent(_MINIMAL_MANIFEST)
        + '\n[libraries.other]\nname = "o"\ncustodian = "o"\nterms_url = "o"\n'
        + 'license = "o"\nredistributable = true\nrisk = "green"\ncitation = "o"\n'
        + 'paths = ["data/example/covered.parquet"]\n',
        encoding="utf-8",
    )
    manifest = licensing.load_manifest(m_path)
    tracked = ["data/example/covered.parquet"]
    coverage = licensing.check_coverage(manifest, tracked)
    assert set(coverage.covered["data/example/covered.parquet"]) == {
        "libraries.example",
        "libraries.other",
    }


def test_glob_star_does_not_cross_a_slash():
    """`data/foo/*.pq` must not match `data/foo/bar/baz.pq`.

    A segment-wise fnmatch keeps a typo local — `data/foo/*.pq` covers one
    level and cannot silently claim a whole subtree the author forgot about.
    """
    assert licensing.match("data/foo/*.pq", "data/foo/bar.pq")
    assert not licensing.match("data/foo/*.pq", "data/foo/sub/bar.pq")
    assert licensing.match("data/foo/**", "data/foo/sub/bar.pq")


def test_sidecars_are_excluded_from_coverage(tmp_path):
    """A generated LICENSE.txt must not be treated as uncovered data."""
    m_path = tmp_path / "licenses.toml"
    m_path.write_text(textwrap.dedent(_MINIMAL_MANIFEST), encoding="utf-8")
    manifest = licensing.load_manifest(m_path)
    tracked = [
        "data/example/covered.parquet",
        "data/example/LICENSE.txt",  # generated sidecar — must not count
    ]
    coverage = licensing.check_coverage(manifest, tracked)
    assert coverage.uncovered == []
    assert coverage.ok
