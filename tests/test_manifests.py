"""Every cross-section library ships an accurate `manifest.json`.

`endfb-8.0`, `hi-xs` and `hi-xs-prod` shipped without one for several releases —
each was added by a builder that did not write manifests, and nothing checked.
A manifest is how a consumer sees the shape of a library without opening 600
parquet files, and how a rebuild that silently drops half a library shows up in
a diff.

`scripts/build_manifests.py --check` is the same code path that writes them, so
these tests fail exactly when a regeneration would change something.

Reads the committed parquets' footers, so it needs the data tree but no network.
The `data` marker is for the case where that tree is absent; it is not a reason
to skip these in CI, and while it was used as one `exfor-channels` shipped a
manifest that disagreed with its own parquets from #334 until #358 (#355).
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


def test_manifest_paths_are_unique_per_library() -> None:
    """Two libraries must never resolve to the same manifest.json.

    The first builder walked up from the parquet directory unconditionally, so
    `exfor/` and `exfor-channels/` both landed on `data/manifest.json`, then both
    `endfb-8.0/xs/` and `endfb-8.0/channels/` both landed on
    `data/endfb-8.0/manifest.json`. Each pair silently overwrote the other, and
    the surviving file described whichever library happened to be written last.

    This runs without the `data` marker: it is a property of the path rule, and
    the only thing it needs from disk is the catalog.
    """
    seen: dict[Path, str] = {}
    collisions: list[str] = []
    for key, _pq_dir, manifest in _dirs():
        if manifest in seen:
            collisions.append(f"{seen[manifest]} and {key} both write {manifest}")
        seen[manifest] = key
    assert not collisions, "manifest path collisions:\n  " + "\n  ".join(collisions)


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


@pytest.mark.data
def test_manifests_describe_every_projectile_they_ship() -> None:
    """`projectiles` is the manifest's answer to "what is in this library".

    It has to describe all of it. `tendl-2025` ships a/d/h/n/p/t; its manifest
    carried `"sublibrary": "a"` — one projectile out of six — because each
    `--sublibrary` run rewrote the file wholesale and the field kept whichever
    ran last. The counts beside it were regenerated from disk and correct, which
    is the worst combination: the file reads authoritative (#369).

    Checked against the parquet `projectile` column rather than against
    `catalog.json`, so this cannot be satisfied by two files agreeing with each
    other while both disagree with the data.
    """
    from build_manifests import build_manifest

    problems: list[str] = []
    checked = 0
    for key, pq_dir, manifest_path in _dirs():
        if not manifest_path.exists():
            continue
        recorded = set(json.loads(manifest_path.read_text()).get("projectiles", []))
        on_disk = set(build_manifest(key, pq_dir)["projectiles"])
        if not on_disk:
            continue  # meta-style library with no projectile column
        checked += 1
        if recorded != on_disk:
            problems.append(
                f"{key}: manifest says {sorted(recorded)}, the parquets carry {sorted(on_disk)}"
                f" (missing {sorted(on_disk - recorded)}, spurious {sorted(recorded - on_disk)})"
            )
    assert checked >= 15, f"only {checked} libraries had a projectile column — is the check still reaching the data?"
    assert not problems, "manifest projectiles disagree with the shipped files:\n  " + "\n  ".join(problems)


@pytest.mark.data
def test_no_manifest_carries_a_retired_per_run_field() -> None:
    """Retired keys must not come back, in a manifest or in a builder.

    Each held one `--sublibrary` run's value under a whole-library name.
    `sublibrary` was the visible one; the ten diagnostic counters added in #354
    and #376 have the same shape and had not reached committed data yet only
    because no multi-projectile library has been re-ingested since. They live
    under `ingest.<sublibrary>` now (#369).
    """
    from nucl_parquet.builder_stamp import RETIRED_MANIFEST_KEYS

    offenders: dict[str, list[str]] = {}
    for key, _pq_dir, manifest_path in _dirs():
        if not manifest_path.exists():
            continue
        stale = sorted(set(json.loads(manifest_path.read_text())) & RETIRED_MANIFEST_KEYS)
        if stale:
            offenders[key] = stale
    assert not offenders, (
        "manifests still carry per-run fields under whole-library names — "
        "run scripts/build_manifests.py:\n  " + "\n  ".join(f"{k}: {v}" for k, v in offenders.items())
    )


@pytest.mark.data
def test_ingest_records_are_keyed_by_a_projectile_the_library_ships() -> None:
    """`ingest` keys name sublibraries, so they must be sublibraries.

    No committed manifest carries `ingest` yet — it arrives with the first
    re-ingest after #369 (#345 is the issue that does them). The assertion is
    written now so the first one to land is checked rather than trusted, and it
    asserts a positive on the shape when the block is present.
    """
    problems: list[str] = []
    for key, _pq_dir, manifest_path in _dirs():
        if not manifest_path.exists():
            continue
        doc = json.loads(manifest_path.read_text())
        ingest = doc.get("ingest")
        if ingest is None:
            continue
        assert isinstance(ingest, dict), f"{key}: `ingest` must be an object keyed by sublibrary, got {type(ingest)}"
        shipped = set(doc.get("projectiles", []))
        for sub, record in ingest.items():
            if sub not in shipped:
                problems.append(f"{key}.ingest[{sub!r}]: not among the shipped projectiles {sorted(shipped)}")
            if not isinstance(record, dict) or "source_files" not in record:
                problems.append(f"{key}.ingest[{sub!r}]: expected the per-run counters, got {record!r}")
    assert not problems, "malformed `ingest` records:\n  " + "\n  ".join(problems)
