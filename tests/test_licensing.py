# SPDX-FileCopyrightText: 2024-2026 ETH Zürich (eXoma — Exotic Matter Applications)
# SPDX-FileContributor: Lars Gerchow
# SPDX-License-Identifier: MIT
"""Licence-compliance gates.

Three classes of failure this catches:

1. **Unrecorded data** — a Parquet tree lands under `data/` without an entry in
   `data/licenses.toml`, so it ships with no provenance and no citation.
2. **Drift** — `NOTICE`, `ATTRIBUTION.md`, the per-directory `LICENSE.txt`
   sidecars or the SPDX headers stop matching the manifest.
3. **Copyleft leaking into the runtime** — an AGPL/GPL package reaching
   `[project].dependencies`, which would be incompatible with the MIT wheel
   under RSETHZ 440.4 Art. 27(1)(c).

Fix for 1: add the entry. Fix for 2: `python scripts/build_notices.py --write`.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
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
# 1. Coverage
# ---------------------------------------------------------------------------


def test_every_data_file_has_recorded_provenance(manifest, tracked):
    coverage = licensing.check_coverage(manifest, tracked)
    assert not coverage.uncovered, (
        f"{len(coverage.uncovered)} bundled data file(s) have no entry in data/licenses.toml, "
        f"so they ship with no licence or citation: {coverage.uncovered[:10]}"
    )


def test_no_stale_path_globs(manifest, tracked):
    """A glob matching nothing means the tree moved or the entry is dead."""
    coverage = licensing.check_coverage(manifest, tracked)
    assert not coverage.unused_patterns, f"manifest globs match nothing: {coverage.unused_patterns}"


def test_no_file_claimed_by_two_entries(manifest, tracked):
    """Overlapping globs make the effective licence of a file ambiguous."""
    coverage = licensing.check_coverage(manifest, tracked)
    ambiguous = {p: keys for p, keys in coverage.covered.items() if len(keys) > 1}
    assert not ambiguous, f"files claimed by multiple manifest entries: {dict(list(ambiguous.items())[:5])}"


def test_nothing_non_redistributable_is_bundled(manifest, tracked):
    """`redistributable = false` means the tree must not exist in the repo."""
    covered = licensing.check_coverage(manifest, tracked).covered
    entries = licensing.entries(manifest)
    offenders = {
        path: key for path, keys in covered.items() for key in keys if entries[key]["redistributable"] is False
    }
    assert not offenders, f"non-redistributable data is bundled: {offenders}"


# ---------------------------------------------------------------------------
# 2. Generated-artifact drift
# ---------------------------------------------------------------------------


def test_generated_artifacts_are_current():
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


def test_every_manifest_entry_is_complete(manifest):
    """Missing a citation or terms URL silently produces a defective NOTICE."""
    required = ("name", "custodian", "terms_url", "license", "redistributable", "risk", "citation", "paths")
    for key, entry in licensing.entries(manifest).items():
        missing = [field for field in required if field not in entry]
        assert not missing, f"{key} is missing required field(s): {missing}"


def test_notice_references_every_mandatory_custodian(manifest):
    """The custodians whose terms make a notice mandatory must appear in NOTICE."""
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for required in ("NIST", "Geant4", "IAEA", "Japan Atomic Energy Agency", "Creative Commons", "ETH Zürich"):
        assert required in notice, f"NOTICE is missing the {required} notice"


def test_notice_targets_of_each_notice_block_exist(manifest):
    """`applies_to` must reference real entries, or the notice covers nothing."""
    keys = set(licensing.entries(manifest))
    for notice in manifest["notices"]:
        unknown = [k for k in notice["applies_to"] if k not in keys]
        assert not unknown, f"notice {notice['title']!r} references unknown entries: {unknown}"


# ---------------------------------------------------------------------------
# 3. ETH Zürich RSETHZ 440.4
# ---------------------------------------------------------------------------


def test_eth_is_the_named_copyright_holder(manifest):
    """Art. 27(1)(e): ETH Zürich as holder, creators named, in the OSS artifacts."""
    eth = manifest["eth"]
    assert eth["copyright_holder"] == "ETH Zürich"
    assert eth["creators"], "Art. 27(1)(e) requires the creators to be named"

    for filename in ("LICENSE", "NOTICE"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "ETH Zürich" in text, f"{filename} must name ETH Zürich as copyright holder"
        for creator in eth["creators"]:
            assert creator in text, f"{filename} must name the creator {creator}"


def test_distribution_carries_the_licence_files():
    """Art. 27(1)(e): the built artifact, not just the repo, must carry them."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    license_files = pyproject["project"].get("license-files", [])
    assert "LICENSE" in license_files and "NOTICE" in license_files, (
        f"pyproject must ship LICENSE and NOTICE in the wheel; got {license_files}"
    )


def test_source_files_carry_spdx_headers(manifest):
    header_first_line = licensing.spdx_header(manifest).split("\n")[0]
    sources = subprocess.run(
        ["git", "ls-files", "--", "nucl_parquet/*.py", "nucl_parquet/**/*.py", "scripts/*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert sources, "expected to find Python sources to check"
    missing = [s for s in sources if header_first_line not in (ROOT / s).read_text(encoding="utf-8").split("\n")[:4]]
    assert not missing, f"missing SPDX header (run scripts/build_notices.py --write): {missing}"


# ---------------------------------------------------------------------------
# 4. Copyleft containment — Art. 27(1)(c)
# ---------------------------------------------------------------------------


def test_no_copyleft_in_runtime_dependencies(manifest):
    """MIT out means no GPL/AGPL/copyleft package in [project].dependencies.

    The manifest records which code dependencies are copyleft and asserts they
    are build-time only; this checks pyproject actually honours that.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = pyproject["project"]["dependencies"]
    runtime_names = {req.split(">")[0].split("=")[0].split("[")[0].strip().lower() for req in runtime}

    for key, dep in manifest["code_dependencies"].items():
        if dep.get("runtime_dependency"):
            continue
        # Manifest names may be "pycatima / catima"; check each alias.
        aliases = {a.strip().lower() for a in dep["name"].split("/")}
        leaked = aliases & runtime_names
        assert not leaked, (
            f"{key} is {dep['spdx']} and recorded as build-time only, but {leaked} "
            f"is in [project].dependencies — incompatible with the MIT wheel "
            f"(RSETHZ 440.4 Art. 27(1)(c)). Move it to the `build` extra."
        )


def test_agpl_tooling_is_not_imported_at_runtime(manifest):
    """catima may be used to *generate* data, never imported by the loader."""
    dep = manifest["code_dependencies"]["pycatima"]
    assert dep["runtime_dependency"] is False
    assert dep["imported_by"] == ["nucl_parquet/build_heavy_ions.py"]

    loader = (ROOT / "nucl_parquet" / "loader.py").read_text(encoding="utf-8")
    assert "import pycatima" not in loader, (
        "the loader must read the pre-built catima_*.parquet shards, not import the AGPL library"
    )
