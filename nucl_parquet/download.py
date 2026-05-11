"""Data directory resolution and GitHub Release download.

Data and code release on separate cadences (#150 tracks the analogous code-side
split). Data tarballs are CalVer-tagged `data-YYYY.MM.DD`; the version a given
checkout pins lives at `data/catalog.json::data_version`.
"""

from __future__ import annotations

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
_DATA_TAG_RE = re.compile(r"^data-\d{4}\.\d{2}\.\d{2}$")


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

    Reads `data/catalog.json::data_version` (e.g. `"2026.05.11"`). The matching
    GitHub Release tag is `data-{data_version}`.
    """
    root = Path(data_dir_path) if data_dir_path else data_dir()
    catalog = json.loads((root / "catalog.json").read_text())
    version = catalog.get("data_version")
    if not version:
        raise RuntimeError(f"catalog.json at {root} is missing `data_version`. Pre-CalVer catalog or corrupted file.")
    return version


def _resolve_latest_data_tag() -> str:
    """Return the most recent `data-YYYY.MM.DD` release tag on GitHub."""
    url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases?per_page=30"
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:  # noqa: S310
        payload = json.load(resp)
    for release in payload:
        tag = release.get("tag_name", "")
        if _DATA_TAG_RE.match(tag):
            return tag
    raise RuntimeError(f"No data-YYYY.MM.DD release found in the latest 30 releases at {url}")


def download(
    dest: Path | str | None = None,
    data_version: str = "latest",
    *,
    tag: str | None = None,
) -> Path:
    """Download nucl-parquet data from GitHub Releases.

    Args:
        dest: Destination directory. Defaults to ~/.nucl-parquet/.
        data_version: CalVer string (`"2026.05.11"`) OR `"latest"` to resolve
            the most recent `data-*` GitHub release.
        tag: Deprecated. Pre-CalVer callers passed a code-version tag
            (e.g. `"v0.13.0"`); this is now ignored with a DeprecationWarning
            and resolves to the latest data release instead. New callers
            should use `data_version=`.

    Returns:
        Path to the downloaded data directory.
    """
    if tag is not None:
        warnings.warn(
            "`tag=` is deprecated in nucl_parquet.download(); pass `data_version=` "
            "with a CalVer identifier (e.g. `'2026.05.11'`) or `'latest'`. "
            "Resolving to latest data release.",
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
