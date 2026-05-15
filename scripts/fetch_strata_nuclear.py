"""Fetch Geant4-derived nuclear-structure parquet files from the strata HF dataset.

Pulls the four nuclear-structure files (ensdfstate, photon_evap_levels,
photon_evap_gammas, radioactive_decay) from the public Hugging Face dataset
`gerchowl/strata-data`, pinned to a specific revision SHA for build
reproducibility.

The strata project parses the canonical Geant4 ASCII data files
(G4ENSDFSTATE3.0, PhotonEvaporation6.1.2, RadioactiveDecay6.1.2 — distributed
by CERN with each Geant4 release) and republishes the resulting parquet on
HF. Reusing their published parquet means nucl-parquet's converters (#69–#72)
can stay focused on schema mapping and stay decoupled from G4-format parsing.

The pinned revision must change in lockstep with any nucl-parquet release that
embeds new data — `data/catalog.json` records the active SHA. Bumping the pin
is a deliberate two-step: re-run this fetcher with `--revision <new SHA>`,
re-run the converter pipeline + diff harness (#75), update catalog.json.

Output: data/g4_raw/strata-nuclear/*.parquet (gitignored — fetched on demand).

Usage:
    # Default: fetch the catalog-pinned revision from HF
    uv run python scripts/fetch_strata_nuclear.py

    # Override the revision (for testing a future bump)
    uv run python scripts/fetch_strata_nuclear.py --revision <sha>

    # Skip HF and copy from a local strata clone (offline development)
    uv run python scripts/fetch_strata_nuclear.py \
        --from-local ~/Projects/strata/strata-data/nuclear

References: ADR-0002 (schema decision), epic #66, this issue #68.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
HF_REPO_ID = "gerchowl/strata-data"
DEST_DIR = ROOT / "data" / "g4_raw" / "strata-nuclear"
CATALOG_PATH = ROOT / "data" / "catalog.json"
CATALOG_KEY = "strata-data-nuclear"

# Strata-data files we depend on, pinned to the same revision via
# catalog.json::libraries.strata-data-nuclear. The em/ files were added in
# v0.13 for #121 (ESTAR) and #118 (Seltzer-Berger DCS). Other strata content
# (hadronic/, optical/) is still out of scope; extend this list and bump the
# revision pin together.
FILES = [
    "nuclear/ensdfstate.parquet",
    "nuclear/photon_evap_levels.parquet",
    "nuclear/photon_evap_gammas.parquet",
    "nuclear/radioactive_decay.parquet",
    "em/estar_basic.parquet",
    "em/estar_long.parquet",
    "em/density_effect.parquet",
    "em/brem_sb_dcs.parquet",
]


def _read_pinned_revision() -> str:
    """Read the active pinned revision from data/catalog.json.

    The pin lives in catalog.json so that downstream consumers (Rust crate,
    docs, audit trail) can introspect 'which strata revision did this build
    use' without parsing Python source. Single source of truth. Path is
    resolved from this script's location, so the working directory doesn't
    matter.
    """
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"catalog.json not found at {CATALOG_PATH} — repo layout changed unexpectedly.")
    catalog = json.loads(CATALOG_PATH.read_text())
    entry = catalog.get("libraries", {}).get(CATALOG_KEY)
    if not entry:
        raise KeyError(
            f"catalog.json missing 'libraries.{CATALOG_KEY}' entry — "
            "this fetcher requires the entry created by issue #68."
        )
    revision = entry.get("revision")
    if not revision or not isinstance(revision, str) or len(revision) < 7:
        raise ValueError(f"catalog.json 'libraries.{CATALOG_KEY}.revision' is missing or invalid: {revision!r}")
    return revision


def _atomic_copy(src: Path, dest: Path) -> None:
    """Copy with rename-on-finish so a disk-full mid-copy can't leave a
    half-written file at `dest` that the next run silently treats as cached."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dest)


def _all_dests_present() -> bool:
    """True iff every expected output file already exists in DEST_DIR."""
    return all((DEST_DIR / Path(p).name).exists() for p in FILES)


def fetch_from_hf(revision: str, force: bool = False) -> list[Path]:
    """Download the four files from HF, pinned to the given revision SHA.

    Uses huggingface_hub's local cache (HF_HOME or ~/.cache/huggingface/) so
    repeat runs are free; copies into DEST_DIR for visibility (the cache uses
    blob hashes, not filenames).

    Short-circuits the HF round-trips when DEST_DIR is already populated and
    --force isn't set, so re-runs work fully offline once primed.
    """
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if not force and _all_dests_present():
        logger.info("All %d files already present in %s (use --force to re-fetch).", len(FILES), DEST_DIR)
        return [DEST_DIR / Path(p).name for p in FILES]

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
    except ImportError as e:
        raise ImportError(
            "huggingface_hub not installed. Run `uv sync` (or use `--from-local <path>` for offline development)."
        ) from e

    written: list[Path] = []
    for hf_path in FILES:
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=hf_path,
                    repo_type="dataset",
                    revision=revision,
                    force_download=force,
                )
            )
        except RevisionNotFoundError as e:
            raise RuntimeError(
                f"Revision {revision!r} not found in {HF_REPO_ID}. "
                "Check data/catalog.json or pass --revision <sha> with a valid SHA."
            ) from e
        except (EntryNotFoundError, RepositoryNotFoundError) as e:
            raise RuntimeError(
                f"Couldn't fetch {hf_path} from {HF_REPO_ID}@{revision[:8]}: {e}. "
                "If you're offline, copy files locally and use --from-local <strata-data/nuclear path>."
            ) from e

        dest = DEST_DIR / Path(hf_path).name
        if not dest.exists() or dest.stat().st_size != cached.stat().st_size or force:
            _atomic_copy(cached, dest)
        written.append(dest)
        logger.info("  %s: %d KB", dest.name, dest.stat().st_size // 1024)
    return written


def fetch_from_local(local_dir: Path) -> list[Path]:
    """Copy the four files from a local strata-data/nuclear directory.

    Useful for offline development on a machine that already has strata
    cloned. Skips network and revision verification — caller is on the
    hook for keeping the local clone in sync with the catalog pin.
    """
    if not local_dir.is_dir():
        raise NotADirectoryError(f"--from-local path is not a directory: {local_dir}")

    pinned = None
    try:
        pinned = _read_pinned_revision()
    except (FileNotFoundError, KeyError, ValueError):
        pass  # catalog.json not readable; warn-and-continue is fine for --from-local
    logger.warning(
        "--from-local skips revision verification; ensure your local clone matches catalog pin %s",
        (pinned[:8] + "...") if pinned else "(unknown)",
    )

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for hf_path in FILES:
        src = local_dir / Path(hf_path).name
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {src} (relative to --from-local). "
                "Pass the strata-data/nuclear directory, not the dataset root."
            )
        dest = DEST_DIR / src.name
        _atomic_copy(src, dest)
        written.append(dest)
        logger.info("  %s: %d KB (from local)", dest.name, dest.stat().st_size // 1024)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--revision",
        default=None,
        help="Override the catalog-pinned revision. Defaults to the SHA in data/catalog.json.",
    )
    parser.add_argument(
        "--from-local",
        type=Path,
        default=None,
        help="Skip HF download; copy from this local strata-data/nuclear directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download from HF even if the file is already cached.",
    )
    args = parser.parse_args()

    if args.from_local and args.revision:
        parser.error("--from-local and --revision are mutually exclusive.")

    if args.from_local:
        logger.info("Copying from local: %s", args.from_local)
        written = fetch_from_local(args.from_local)
    else:
        revision = args.revision or _read_pinned_revision()
        logger.info("Fetching from HF: %s @ %s", HF_REPO_ID, revision)
        written = fetch_from_hf(revision, force=args.force)

    logger.info("Wrote %d files to %s", len(written), DEST_DIR)


if __name__ == "__main__":
    main()
