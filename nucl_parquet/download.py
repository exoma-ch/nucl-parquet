"""Data directory resolution and GitHub Release download.

Data and code release on separate cadences (#150 tracks the analogous code-side
split). Data tarballs are CalVer-tagged `data-YYYY.MM.MICRO` (e.g. `data-2026.5.0`,
`data-2026.5.1` for an in-month iteration); the version a given checkout pins
lives at `data/catalog.json::data_version`. A deterministic SHA-256 tree hash
of the parquets ships alongside in `data/catalog.json::data_sha256` so PR CI
can gate on "data changed ⇔ version changed".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import tempfile
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

import zstandard

_GITHUB_REPO = "exoma-ch/nucl-parquet"
_DATA_TAG_RE = re.compile(r"^data-\d{4}\.\d+\.\d+$")
_DATA_VERSION_RE = re.compile(r"^\d{4}\.\d+\.\d+$")


def data_dir() -> Path:
    """Resolve the nucl-parquet data directory.

    Resolution order:
        1. $NUCL_PARQUET_DATA environment variable (should point to the data/ subdir)
        2. Repo checkout — data/ detected by presence of data/catalog.json
        3. ~/.nucl-parquet/

    Returns:
        Path to data directory (may not exist yet).

    Raises:
        FileNotFoundError: If no data directory is found.
    """
    env = os.environ.get("NUCL_PARQUET_DATA")
    if env:
        p = Path(env)
        if p.is_dir():
            return p

    repo_root = Path(__file__).parent.parent
    if (repo_root / "data" / "catalog.json").exists():
        return repo_root / "data"

    home = Path.home() / ".nucl-parquet"
    if home.is_dir():
        return home

    raise FileNotFoundError(
        "nucl-parquet data not found. Set $NUCL_PARQUET_DATA, clone the repo, "
        "or run nucl_parquet.download.download() to fetch data."
    )


def data_version(data_dir_path: Path | str | None = None) -> str:
    """Return the CalVer identifier of the data shipping with this checkout.

    Reads `data/catalog.json::data_version` (e.g. `"2026.5.0"`). The matching
    GitHub Release tag is `data-{data_version}`.
    """
    root = Path(data_dir_path) if data_dir_path else data_dir()
    catalog = json.loads((root / "catalog.json").read_text())
    version = catalog.get("data_version")
    if not version:
        raise RuntimeError(f"catalog.json at {root} is missing `data_version`. Pre-CalVer catalog or corrupted file.")
    return version


_HASH_EXCLUDE_DIRS = frozenset(
    {
        "g4_raw",  # build-time cache (gitignored)
    }
)


def compute_data_sha256(data_dir_path: Path | str | None = None) -> str:
    """Deterministic SHA-256 tree hash of every `data/**/*.parquet` file.

    The hash digests, in sorted POSIX-relpath order, lines of the form
    `<relpath>\\0<per-file sha256>\\n`. Reproducible across machines and
    independent of filesystem mtimes / inode order. PR CI compares this
    to `data/catalog.json::data_sha256` to detect:
      - parquets changed but `data_version` did not (silent drift)
      - `data_version` changed but parquets did not (cosmetic bump)

    The hash deliberately ignores:
      - non-parquet files (manifests, schemas, catalog itself) — descriptors,
        not data; changes don't require a data release
      - any path under a top-level directory in `_HASH_EXCLUDE_DIRS` (e.g.
        `data/g4_raw/`, which is a gitignored build cache populated by
        `scripts/fetch_strata_nuclear.py`)
    """
    root = Path(data_dir_path) if data_dir_path else data_dir()
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.parquet")):
        rel_parts = path.relative_to(root).parts
        if rel_parts and rel_parts[0] in _HASH_EXCLUDE_DIRS:
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        fh = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda f=f: f.read(1 << 20), b""):
                fh.update(chunk)
        h.update(rel + b"\0" + fh.hexdigest().encode("ascii") + b"\n")
    return h.hexdigest()


def data_sha256(data_dir_path: Path | str | None = None) -> str:
    """Return the cached SHA-256 declared in `catalog.json::data_sha256`.

    Use `compute_data_sha256()` to recompute from the on-disk tree.
    Use this helper to read the value the catalog *claims* matches the
    tree. PR CI compares the two.
    """
    root = Path(data_dir_path) if data_dir_path else data_dir()
    catalog = json.loads((root / "catalog.json").read_text())
    declared = catalog.get("data_sha256")
    if not declared:
        raise RuntimeError(f"catalog.json at {root} is missing `data_sha256` field.")
    return declared


def _resolve_latest_data_tag() -> str:
    """Return the most recent `data-YYYY.MM.MICRO` release tag on GitHub.

    Scans the most-recent 100 releases (single page, max page_size). Code
    releases share this listing, so a high code-release cadence between
    data refreshes could mask the data tag; raise loudly if none found.
    """
    url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases?per_page=100"
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:  # noqa: S310
        payload = json.load(resp)
    for release in payload:
        tag = release.get("tag_name", "")
        if _DATA_TAG_RE.match(tag):
            return tag
    raise RuntimeError(f"No data-YYYY.MM.MICRO release found in the latest 100 releases at {url}")


_CODE_VERSION_RE = re.compile(r"^v?\d+\.\d+\.\d+")


def download(
    dest: Path | str | None = None,
    data_version: str = "latest",
    *,
    tag: str | None = None,
) -> Path:
    """Download nucl-parquet data from GitHub Releases.

    Args:
        dest: Destination directory. Defaults to ~/.nucl-parquet/.
        data_version: CalVer string (`"2026.5.0"`) OR `"latest"` to resolve
            the most recent `data-*` GitHub release. Pre-CalVer code-version
            strings (e.g. `"v0.13.0"`) are detected and resolve to latest data
            with a DeprecationWarning.
        tag: Deprecated keyword alias of pre-CalVer call sites. Ignored with
            a DeprecationWarning; resolves to latest data. New callers should
            use `data_version=`.

    Returns:
        Path to the downloaded data directory.
    """
    if tag is not None:
        warnings.warn(
            "`tag=` is deprecated in nucl_parquet.download(); pass `data_version=` "
            "with a CalVer identifier (e.g. `'2026.5.0'`) or `'latest'`. "
            "Resolving to latest data release.",
            DeprecationWarning,
            stacklevel=2,
        )
        data_version = "latest"
    elif _CODE_VERSION_RE.match(data_version):
        # Pre-CalVer positional call: download(dest, "v0.13.0"). Same
        # remediation as the keyword path — warn and fall through to latest.
        warnings.warn(
            f"`{data_version!r}` looks like a code-version tag; "
            "nucl_parquet.download() now takes CalVer data identifiers "
            "(e.g. `'2026.5.0'`) or `'latest'`. Resolving to latest data.",
            DeprecationWarning,
            stacklevel=2,
        )
        data_version = "latest"

    dest = Path(dest) if dest else Path.home() / ".nucl-parquet"
    dest.mkdir(parents=True, exist_ok=True)

    if data_version == "latest":
        data_tag = _resolve_latest_data_tag()
    elif _DATA_TAG_RE.match(data_version):
        data_tag = data_version  # caller passed full `data-YYYY.MM.DD`
    else:
        data_tag = f"data-{data_version}"

    cal = data_tag.removeprefix("data-")
    asset = f"nucl-parquet-data-{cal}.tar.zst"
    url = f"https://github.com/{_GITHUB_REPO}/releases/download/{data_tag}/{asset}"

    print(f"Downloading nucl-parquet data from {url} ...")

    with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urlopen(url) as resp:  # noqa: S310
            while chunk := resp.read(1 << 20):
                tmp.write(chunk)

    try:
        dctx = zstandard.ZstdDecompressor()
        with open(tmp_path, "rb") as compressed, dctx.stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(dest, filter="data")  # noqa: S202
    finally:
        tmp_path.unlink()

    print(f"Data extracted to {dest}")
    return dest
