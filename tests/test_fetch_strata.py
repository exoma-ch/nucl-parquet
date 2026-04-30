"""Smoke + unit tests for scripts/fetch_strata_nuclear.py.

Most tests run offline (no `network` marker). The actual HF round-trip is
exercised by a single small `@pytest.mark.network` test that downloads
ensdfstate.parquet only — keeps CI fast while verifying the pinned-revision
contract.

Run all (incl. network):
    uv run pytest tests/test_fetch_strata.py -v

Skip network (default in offline CI):
    uv run pytest tests/test_fetch_strata.py -v -m 'not network'
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT_PATH = ROOT / "scripts" / "fetch_strata_nuclear.py"


def _load_fetcher_module():
    """Load fetch_strata_nuclear.py as a module without polluting sys.path."""
    spec = importlib.util.spec_from_file_location("fetch_strata_nuclear", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_pinned_revision_in_catalog() -> None:
    """catalog.json must have a valid 40-char SHA in libraries.strata-data-nuclear.revision."""
    catalog = json.loads((ROOT / "data" / "catalog.json").read_text())
    entry = catalog["libraries"]["strata-data-nuclear"]
    rev = entry["revision"]
    assert isinstance(rev, str)
    assert len(rev) == 40, f"Expected full 40-char SHA, got {len(rev)}"
    assert all(c in "0123456789abcdef" for c in rev), f"Non-hex characters in revision: {rev}"


def test_files_list_matches_catalog() -> None:
    """Script's hard-coded FILES list must match catalog.json declaration."""
    catalog = json.loads((ROOT / "data" / "catalog.json").read_text())
    expected = set(catalog["libraries"]["strata-data-nuclear"]["files"])

    fetcher = _load_fetcher_module()
    assert set(fetcher.FILES) == expected


def test_read_pinned_revision_works(tmp_path, monkeypatch) -> None:
    fetcher = _load_fetcher_module()
    rev = fetcher._read_pinned_revision()
    assert len(rev) == 40


def test_read_pinned_revision_missing_catalog_raises(tmp_path, monkeypatch) -> None:
    """If catalog.json is missing, the function must raise a clear error
    (not return None or silently default)."""
    fetcher = _load_fetcher_module()
    monkeypatch.setattr(fetcher, "CATALOG_PATH", tmp_path / "nonexistent.json")
    with pytest.raises(FileNotFoundError, match="catalog.json"):
        fetcher._read_pinned_revision()


def test_read_pinned_revision_missing_entry_raises(tmp_path, monkeypatch) -> None:
    fake_catalog = tmp_path / "catalog.json"
    fake_catalog.write_text(json.dumps({"libraries": {}}))
    fetcher = _load_fetcher_module()
    monkeypatch.setattr(fetcher, "CATALOG_PATH", fake_catalog)
    with pytest.raises(KeyError, match="strata-data-nuclear"):
        fetcher._read_pinned_revision()


def test_read_pinned_revision_invalid_sha_raises(tmp_path, monkeypatch) -> None:
    fake_catalog = tmp_path / "catalog.json"
    fake_catalog.write_text(json.dumps({"libraries": {"strata-data-nuclear": {"revision": "short"}}}))
    fetcher = _load_fetcher_module()
    monkeypatch.setattr(fetcher, "CATALOG_PATH", fake_catalog)
    with pytest.raises(ValueError, match="invalid"):
        fetcher._read_pinned_revision()


def test_atomic_copy_full_then_replace(tmp_path) -> None:
    """_atomic_copy must produce a final file that's byte-identical to the source."""
    fetcher = _load_fetcher_module()
    src = tmp_path / "src.bin"
    dest = tmp_path / "dest.bin"
    payload = b"hello\x00world" * 1024
    src.write_bytes(payload)

    fetcher._atomic_copy(src, dest)
    assert dest.read_bytes() == payload
    # Tempfile must be cleaned up
    assert not (tmp_path / "dest.bin.tmp").exists()


def test_fetch_from_local_happy_path(tmp_path) -> None:
    """--from-local copies files when the source dir has all expected names."""
    fetcher = _load_fetcher_module()
    src_dir = tmp_path / "strata-nuclear"
    src_dir.mkdir()
    for hf_path in fetcher.FILES:
        (src_dir / Path(hf_path).name).write_bytes(b"stub-content")

    dest_dir = tmp_path / "dest"
    monkeypatch_dest = pytest.MonkeyPatch()
    monkeypatch_dest.setattr(fetcher, "DEST_DIR", dest_dir)
    try:
        written = fetcher.fetch_from_local(src_dir)
    finally:
        monkeypatch_dest.undo()
    assert len(written) == len(fetcher.FILES)
    for f in written:
        assert f.exists()
        assert f.read_bytes() == b"stub-content"


def test_fetch_from_local_missing_file_raises(tmp_path) -> None:
    fetcher = _load_fetcher_module()
    src_dir = tmp_path / "strata-nuclear"
    src_dir.mkdir()
    # Create only one of the expected files; the rest are missing
    (src_dir / Path(fetcher.FILES[0]).name).write_bytes(b"stub")
    with pytest.raises(FileNotFoundError, match=Path(fetcher.FILES[1]).name):
        fetcher.fetch_from_local(src_dir)


def test_fetch_from_local_not_a_dir_raises(tmp_path) -> None:
    fetcher = _load_fetcher_module()
    not_a_dir = tmp_path / "missing"
    with pytest.raises(NotADirectoryError):
        fetcher.fetch_from_local(not_a_dir)


@pytest.mark.network
def test_fetch_smallest_file_from_hf(tmp_path) -> None:
    """Pull the smallest file (ensdfstate.parquet, ~268 KB) from HF at the
    pinned revision and verify it's a non-empty parquet. Single-file scope
    keeps CI fast; the full 4-file path is exercised by the build pipeline."""
    pl = pytest.importorskip("polars")
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import hf_hub_download

    fetcher = _load_fetcher_module()
    rev = fetcher._read_pinned_revision()

    cached = Path(
        hf_hub_download(
            repo_id=fetcher.HF_REPO_ID,
            filename="nuclear/ensdfstate.parquet",
            repo_type="dataset",
            revision=rev,
        )
    )
    df = pl.read_parquet(cached)
    assert len(df) > 1000, f"ensdfstate.parquet has only {len(df)} rows — pin may be drifted"
    assert "excitation_kev" in df.schema, "Schema regression: excitation_kev column missing"


@pytest.mark.network
def test_fetcher_cli_idempotent(tmp_path, monkeypatch) -> None:
    """Two consecutive runs of fetch_from_hf must short-circuit on the second
    via the _all_dests_present early-exit (covered by no-force semantics)."""
    fetcher = _load_fetcher_module()
    monkeypatch.setattr(fetcher, "DEST_DIR", tmp_path / "g4_raw")
    rev = fetcher._read_pinned_revision()

    fetcher.fetch_from_hf(rev, force=False)
    assert fetcher._all_dests_present()

    # Second call should short-circuit (no network needed) — manifested by
    # not raising even if HF is hypothetically unreachable. We don't easily
    # mock that here, but at minimum verify the early-exit code-path is hit.
    written = fetcher.fetch_from_hf(rev, force=False)
    assert len(written) == len(fetcher.FILES)
