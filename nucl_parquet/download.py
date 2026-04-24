"""Data directory resolution and GitHub Release download."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import zstandard

_GITHUB_REPO = "exoma-ch/nucl-parquet"


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


def _resolve_latest_tag() -> str:
    url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:  # noqa: S310
        payload = json.load(resp)
    tag = payload.get("tag_name")
    if not tag:
        raise RuntimeError(f"Could not resolve latest release tag from {url}")
    return tag


def download(
    dest: Path | str | None = None,
    tag: str = "latest",
) -> Path:
    """Download nucl-parquet data from GitHub Releases.

    Args:
        dest: Destination directory. Defaults to ~/.nucl-parquet/.
        tag: Git tag to download (default: latest release).

    Returns:
        Path to the downloaded data directory.
    """
    dest = Path(dest) if dest else Path.home() / ".nucl-parquet"
    dest.mkdir(parents=True, exist_ok=True)

    if tag == "latest":
        tag = _resolve_latest_tag()
    version = tag.removeprefix("v")
    asset = f"nucl-parquet-data-v{version}.tar.zst"
    url = f"https://github.com/{_GITHUB_REPO}/releases/download/{tag}/{asset}"

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
