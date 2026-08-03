"""Every cross-section library ships an accurate `manifest.json`.

`endfb-8.0`, `hi-xs` and `hi-xs-prod` shipped without one for several releases —
each was added by a builder that did not write manifests, and nothing checked.
A manifest is how a consumer sees the shape of a library without opening 600
parquet files, and how a rebuild that silently drops half a library shows up in
a diff.

`scripts/build_manifests.py --check` is the same code path that writes them, so
these tests fail exactly when a regeneration would change something.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not (DATA_DIR / "catalog.json").exists(), reason="no data tree")


def _dirs():
    from build_manifests import library_dirs

    return library_dirs(DATA_DIR)


@pytest.mark.data
def test_every_xs_library_has_a_manifest() -> None:
    missing = [key for key, _, manifest in _dirs() if not manifest.exists()]
    assert not missing, "cross-section libraries without manifest.json: " + ", ".join(missing)


@pytest.mark.data
def test_manifests_match_the_data_on_disk() -> None:
    """Row and file counts must reflect what actually shipped."""
    from build_manifests import build_manifest

    problems: list[str] = []
    for key, pq_dir, manifest_path in _dirs():
        if not manifest_path.exists():
            continue
        recorded = json.loads(manifest_path.read_text())
        fresh = build_manifest(key, pq_dir)
        for field in ("files", "total_rows"):
            if recorded.get(field) != fresh[field]:
                problems.append(f"{key}.{field}: manifest says {recorded.get(field)}, disk has {fresh[field]}")
    assert not problems, "manifest drift — run scripts/build_manifests.py:\n  " + "\n  ".join(problems)
