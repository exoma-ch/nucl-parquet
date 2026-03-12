"""Data directory resolution and GitHub Release download."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

_GITHUB_REPO = "exoma-ch/nucl-parquet"
_DEFAULT_TAG = "latest"


def data_dir() -> Path:
    """Resolve the nucl-parquet data directory.

    Resolution order:
        1. $NUCL_PARQUET_DATA environment variable
        2. Sibling repo checkout (../nucl-parquet relative to this package)
        3. ~/.nucl-parquet/

    Returns:
        Path to data directory (may not exist yet).

    Raises:
        FileNotFoundError: If no data directory is found.
    """
    # 1. Environment variable
    env = os.environ.get("NUCL_PARQUET_DATA")
    if env:
        p = Path(env)
        if p.is_dir():
            return p

    # 2. Repo root (when installed as editable or running from checkout)
    repo_root = Path(__file__).parent.parent
    if (repo_root / "catalog.json").exists():
        return repo_root

    # 3. Home directory
    home = Path.home() / ".nucl-parquet"
    if home.is_dir():
        return home

    raise FileNotFoundError(
        "nucl-parquet data not found. Set $NUCL_PARQUET_DATA, clone the repo, "
        "or run nucl_parquet.download.download() to fetch data."
    )


def download(
    dest: Path | str | None = None,
    tag: str = _DEFAULT_TAG,
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
        url = f"https://github.com/{_GITHUB_REPO}/releases/latest/download/nucl-parquet-data.tar.gz"
    else:
        url = f"https://github.com/{_GITHUB_REPO}/releases/download/{tag}/nucl-parquet-data.tar.gz"

    print(f"Downloading nucl-parquet data from {url} ...")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urlopen(url) as resp:  # noqa: S310
            while chunk := resp.read(1 << 20):
                tmp.write(chunk)

    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(dest, filter="data")  # noqa: S202
    finally:
        tmp_path.unlink()

    print(f"Data extracted to {dest}")
    return dest
