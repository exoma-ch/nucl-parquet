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
from collections.abc import Iterator
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


def iter_file_digests(
    data_dir_path: Path | str | None = None,
    *,
    parquet_only: bool = True,
) -> Iterator[tuple[str, str, int]]:
    """Yield `(posix_relpath, sha256_hex, size_bytes)` in sorted relpath order.

    The single walk behind both the tree hash and the signed release manifest
    (#296). Deriving them from one traversal is what makes it impossible for
    `catalog.json::data_sha256` and `manifest.json` to disagree about the same
    tree — two independently-written walks would eventually drift on some
    exclusion rule or path-separator detail, and the disagreement would surface
    as an unverifiable release rather than a test failure.

    `parquet_only` is the difference in scope between the two callers, and it is
    deliberate:

      * the **tree hash** covers only `*.parquet`, because it answers "did the
        *data* change" — a catalog edit must not look like a data change;
      * the **manifest** covers everything the tarball carries, because it
        answers "are these the bytes we published". `catalog.json` and
        `licenses.toml` ride inside the archive, and they are the files most
        worth tampering with: #234 is a live example of a wrong licence claim
        shipping on a published artefact.
    """
    root = Path(data_dir_path) if data_dir_path else data_dir()
    pattern = "*.parquet" if parquet_only else "*"
    for path in sorted(root.rglob(pattern)):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if rel_parts and rel_parts[0] in _HASH_EXCLUDE_DIRS:
            continue
        fh = hashlib.sha256()
        size = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda f=f: f.read(1 << 20), b""):
                fh.update(chunk)
                size += len(chunk)
        yield path.relative_to(root).as_posix(), fh.hexdigest(), size


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
    h = hashlib.sha256()
    for rel, digest, _size in iter_file_digests(data_dir_path, parquet_only=True):
        h.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return h.hexdigest()


def build_release_manifest(
    data_dir_path: Path | str | None = None,
    *,
    tag: str,
    tarball_sha256: str | None = None,
) -> dict:
    """Build the signed content manifest for a data release (#296).

    A signature over the tarball proves the *archive bytes* are ours. That stops
    being verifiable the moment anything legitimately rewrites the archive —
    which is routine on the way into an isolated network: Content Disarm &
    Reconstruction gateways (OPSWAT MetaDefender, Deep CDR) open a `.tar.zst`,
    scan each entry and repack it. The nuclear data arrives intact; the
    signature does not survive. Per exoma-ch/hyrr#614 that is roughly a fifth of
    realistic deployments, and they are the sites with the strictest
    verification requirements.

    A manifest of per-file digests, signed with the same offline key, survives
    anything that preserves file *contents* while changing archive *framing*.
    It also makes a partial transfer verifiable: a consumer who moved only
    `tendl-2023-iso/` across a data diode can check what they have, instead of
    being told to carry all 785 MB because only the whole archive is signed.
    This is Debian's `Release`/`InRelease` model.

    Both controls stay. The archive signature is cheaper and stronger when the
    bytes survive; the manifest is what remains when they do not.

    The `data_version` / `tag` fields are not decoration — they bind the
    manifest to a release. Without them a genuine manifest for release A
    verifies happily against release B's extracted files, and every digest that
    happens to be unchanged between the two agrees. That is the same replay gap
    the tarball signature closes via its signed trusted comment.
    """
    root = Path(data_dir_path) if data_dir_path else data_dir()
    files = {rel: {"sha256": digest, "size": size} for rel, digest, size in iter_file_digests(root, parquet_only=False)}
    manifest = {
        "manifest_version": 1,
        "data_version": data_version(root),
        "tag": tag,
        "data_sha256": compute_data_sha256(root),
        "file_count": len(files),
        "files": files,
    }
    if tarball_sha256:
        # Lets a consumer on the intact-bytes path confirm the two routes
        # describe the same release, rather than treating them as unrelated.
        manifest["tarball_sha256"] = tarball_sha256
    return manifest


def dump_release_manifest(manifest: dict) -> str:
    """Serialise a manifest deterministically.

    Sorted keys, no incidental whitespace, trailing newline. Two builds of the
    same tree must produce byte-identical output or diffing releases becomes
    noise — and, more practically, a signature over a non-deterministic
    serialisation is unreproducible by anyone trying to audit it.
    """
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def verify_against_manifest(
    manifest: dict,
    data_dir_path: Path | str | None = None,
) -> list[str]:
    """Check an extracted tree against a manifest. Returns a list of problems.

    Reports missing files, digest mismatches and unexpected extra files
    separately, because they mean different things: a missing file may be a
    deliberate partial transfer, while a mismatch is corruption or tampering.
    Callers decide which are fatal — a subset transfer legitimately has
    missing entries.
    """
    root = Path(data_dir_path) if data_dir_path else data_dir()
    expected: dict[str, dict] = manifest["files"]
    problems: list[str] = []
    seen: set[str] = set()

    for rel, digest, size in iter_file_digests(root, parquet_only=False):
        seen.add(rel)
        want = expected.get(rel)
        if want is None:
            problems.append(f"EXTRA    {rel} (not in manifest)")
        elif want["sha256"] != digest:
            problems.append(f"MODIFIED {rel}\n           expected {want['sha256']}\n           actual   {digest}")
        elif want["size"] != size:
            problems.append(f"SIZE     {rel} expected {want['size']} bytes, got {size}")

    for rel in sorted(set(expected) - seen):
        problems.append(f"MISSING  {rel}")
    return problems


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


def _resolve_base_url(catalog: dict, data_tag: str) -> str:
    """Build per-file base URL from catalog template or hardcoded fallback."""
    template = catalog.get("base_url", "")
    cal = data_tag.removeprefix("data-")
    if template and "{version}" in template:
        return template.replace("{version}", cal)
    return f"https://raw.githubusercontent.com/{_GITHUB_REPO}/{data_tag}/data"


def _fetch_one(url: str, dest: Path) -> None:
    """Download a single file from *url* into *dest* (atomic)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urlopen(url) as resp:  # noqa: S310
            with open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
        tmp.rename(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def ensure(
    dest: Path | str | None = None,
    data_version: str = "latest",
    *,
    lazy: bool = False,
) -> Path:
    """Ensure nucl-parquet data is available, downloading if needed.

    With ``lazy=False`` (default), behaves identically to ``download()`` —
    fetches the full tarball.

    With ``lazy=True``, fetches only ``catalog.json`` up front. Individual
    Parquet files are downloaded on first access by the loader. This is
    ideal for consumers that touch <10% of the dataset per session.

    Args:
        dest: Cache directory. Defaults to ``~/.nucl-parquet/``.
        data_version: CalVer string or ``"latest"``.
        lazy: If True, defer per-file downloads until first access.

    Returns:
        Path to the data directory (may be partially populated in lazy mode).
    """
    # Try to find existing data first
    try:
        existing = data_dir()
        if existing.is_dir() and (existing / "catalog.json").exists():
            return existing
    except FileNotFoundError:
        pass

    dest = Path(dest) if dest else Path.home() / ".nucl-parquet"
    dest.mkdir(parents=True, exist_ok=True)

    if not lazy:
        return download(dest=dest, data_version=data_version)

    # Lazy mode: fetch only catalog.json, resolve base_url for later per-file fetches.
    if data_version == "latest":
        data_tag = _resolve_latest_data_tag()
    elif _DATA_TAG_RE.match(data_version):
        data_tag = data_version
    else:
        data_tag = f"data-{data_version}"

    # Fetch catalog.json
    base = f"https://raw.githubusercontent.com/{_GITHUB_REPO}/{data_tag}/data"
    catalog_url = f"{base}/catalog.json"
    catalog_dest = dest / "catalog.json"

    if not catalog_dest.exists():
        print(f"Fetching catalog from {catalog_url} ...")
        _fetch_one(catalog_url, catalog_dest)

    # Write a marker so fetch_file() knows the base URL for lazy fetches
    marker = dest / ".lazy_base_url"
    catalog = json.loads(catalog_dest.read_text())
    resolved_base = _resolve_base_url(catalog, data_tag)
    marker.write_text(resolved_base)

    print(f"Lazy mode: catalog cached at {dest}, files fetched on demand from {resolved_base}")
    return dest


def fetch_file(data_root: Path, rel_path: str) -> Path:
    """Fetch a single data file if not already cached.

    Called by the loader when a view's backing file is missing. Reads the
    base URL from the ``.lazy_base_url`` marker written by ``ensure(lazy=True)``.

    Args:
        data_root: The data directory (from ``data_dir()`` or ``ensure()``).
        rel_path: Relative path within the data dir (e.g. ``meta/abundances.parquet``).

    Returns:
        Absolute path to the (now-local) file.

    Raises:
        FileNotFoundError: If no lazy base URL is configured (tarball mode).
    """
    dest = data_root / rel_path
    if dest.exists():
        return dest

    marker = data_root / ".lazy_base_url"
    if not marker.exists():
        raise FileNotFoundError(
            f"File not found: {dest}. No lazy fetch configured — "
            "run nucl_parquet.ensure(lazy=True) or nucl_parquet.download()."
        )

    base_url = marker.read_text().strip()
    url = f"{base_url}/{rel_path}"
    print(f"  Fetching {rel_path} ...")
    _fetch_one(url, dest)
    return dest
