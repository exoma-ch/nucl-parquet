"""End-to-end flow of `sync_huggingface.main()`, against a recording stub.

Every piece is unit-tested -- the splitter, the layout guard, the card -- but the
orchestration was not, and orchestration is where this script's remaining risk
lives: it pushes to a public dataset, and it only ever runs unattended, on a
release tag, with a credential no test has. The first real invocation should not
be the first time the sequence executes.

Three properties matter and none are visible from the parts:
  * shards upload BEFORE the card, because the card describes them;
  * the guard aborts before anything is written, not after;
  * card-only never touches the shards.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "endfb-8.0" / "xs").exists(),
    reason="no endfb-8.0 shards",
)


class _Recorder:
    """Stands in for HfApi, recording the call sequence."""

    def __init__(self, published: list[str]) -> None:
        self.calls: list[tuple] = []
        self._published = published

    def list_repo_tree(self, repo_id, path, repo_type=None):  # noqa: ARG002
        return [types.SimpleNamespace(path=p) for p in self._published]

    def upload_folder(self, *, folder_path, path_in_repo, repo_id, repo_type, commit_message):  # noqa: ARG002
        names = sorted(p.name for p in Path(folder_path).glob("*.parquet"))
        self.calls.append(("folder", path_in_repo, names))

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):  # noqa: ARG002
        self.calls.append(("file", path_in_repo, path_or_fileobj.decode()))


@pytest.fixture()
def run(monkeypatch):
    """Invoke main() with a stubbed hub client and a token in the environment."""

    def _run(argv: list[str], published: list[str]):
        rec = _Recorder(published)
        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            types.SimpleNamespace(HfApi=lambda token=None: rec),
        )
        monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
        monkeypatch.setattr(sys, "argv", ["sync_huggingface.py", *argv])
        import sync_huggingface

        sync_huggingface.main()
        return rec

    return _run


def _one_element_tree(tmp_path: Path) -> Path:
    """A data dir holding a single element shard, for speed."""
    xs = tmp_path / "endfb-8.0" / "xs"
    xs.mkdir(parents=True)
    src = DATA_DIR / "endfb-8.0" / "xs" / "n_Nd.parquet"
    (xs / src.name).write_bytes(src.read_bytes())
    for f in ("licenses.toml", "catalog.json"):
        (tmp_path / f).write_bytes((DATA_DIR / f).read_bytes())
    return tmp_path


def test_shards_upload_before_the_card(run, tmp_path: Path) -> None:
    """The card describes the shards, so it must not land first.

    Card-first would leave the mirror advertising a schema it does not have if
    the shard upload then failed.
    """
    data = _one_element_tree(tmp_path)
    rec = run(["--data-dir", str(data)], ["neutron/n_Nd143.parquet"])

    kinds = [c[0] for c in rec.calls]
    assert kinds == ["folder", "file"], f"expected shards then card, got {kinds}"

    _, path_in_repo, names = rec.calls[0]
    assert path_in_repo == "neutron"
    assert "n_Nd143.parquet" in names, "published names must be reproduced"
    assert "n_Nd.parquet" not in names, "element shards must not reach the mirror"


def test_synced_card_names_the_canonical_schema(run, tmp_path: Path) -> None:
    """A run that pushes shards may claim the schema those shards carry."""
    from nucl_parquet._schemas import CANONICAL_XS_SCHEMA

    data = _one_element_tree(tmp_path)
    rec = run(["--data-dir", str(data)], ["neutron/n_Nd143.parquet"])

    _, path_in_repo, card = rec.calls[-1]
    assert path_in_repo == "README.md"
    assert ", ".join(CANONICAL_XS_SCHEMA) in card
    assert "lag the main repository" not in card


def test_card_only_never_touches_the_shards(run, tmp_path: Path) -> None:
    """And says so: the mirror still holds the pre-migration snapshot."""
    data = _one_element_tree(tmp_path)
    rec = run(["--data-dir", str(data), "--card-only"], ["neutron/n_Nd143.parquet"])

    assert [c[0] for c in rec.calls] == ["file"]
    card = rec.calls[0][2]
    assert "lag the main repository" in card


def test_layout_mismatch_aborts_before_anything_is_written(run, tmp_path: Path) -> None:
    """The guard must fire before the first byte, not after a partial upload.

    If the published mirror used some third scheme, uploading and *then* noticing
    would have already interleaved the directory.
    """
    data = _one_element_tree(tmp_path)
    with pytest.raises(SystemExit) as e:
        run(["--data-dir", str(data)], ["neutron/something_else.parquet"])
    assert "different sharding schemes" in str(e.value)


def test_missing_token_pushes_nothing(run, tmp_path: Path, monkeypatch) -> None:
    """The credential check precedes every network call."""
    data = _one_element_tree(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    import sync_huggingface

    monkeypatch.setattr(sys, "argv", ["sync_huggingface.py", "--data-dir", str(data)])
    with pytest.raises(SystemExit):
        sync_huggingface.main()
