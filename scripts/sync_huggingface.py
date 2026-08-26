"""Mirror the per-isotope shards and the dataset card to Hugging Face.

`catalog.json` has claimed since the endfb-8.0 release that the shards are
"also mirrored on Hugging Face (gerchowl/nucl-parquet-data)", but nothing in the
repo ever pushed them — the mirror was maintained by hand. That is how it came
to carry `license: mit` over ENDF/B-VIII.0, a US Government work nobody here is
in a position to relicense (#234).

This makes the mirror a build artifact: the card is generated from
`data/licenses.toml` so the published licence can no longer drift from the
compliance record, and `release-data.yml` calls it on every data tag.

Requires HF_TOKEN with write access to the dataset repo.

Usage:
    nix develop -c .venv/bin/python scripts/sync_huggingface.py --dry-run
    HF_TOKEN=... nix develop -c .venv/bin/python scripts/sync_huggingface.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))  # so `nucl_parquet` imports from the checkout

from nucl_parquet._schemas import CANONICAL_XS_SCHEMA  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "gerchowl/nucl-parquet-data"

# Which library's shards the mirror carries, and therefore whose licence governs
# the dataset card.
MIRRORED_LIBRARY = "endfb-8_0"

# The columns the *published* shards actually have. Deliberately not
# `CANONICAL_XS_SCHEMA`: the mirror is a pre-migration snapshot, and the card
# must describe what a consumer will find when they download the file, not what
# the repository holds today. Update this only when the shards are re-synced.
PUBLISHED_SCHEMA = "target_Z, target_A, residual_Z, residual_A, state, energy_MeV, xs_mb"

# Emitted only when this run is *not* pushing shards, i.e. the mirror still holds
# the pre-migration snapshot. Once a sync runs, the divergence is gone and so is
# this section.
_STALE_SHARDS_NOTE = """
## Status — the shards here lag the main repository

They are a snapshot taken before the canonical-schema migration: correct as far
as they go, but carrying fewer columns than the repository now does. **For
current data, use the release tarball** (see Usage below). This card is
regenerated on every data release, so the licence metadata above is current even
while the shards are not.
"""


def isotope_count(data_dir: Path) -> int:
    """Distinct isotopes in the mirrored library, counted from the data.

    Derived rather than written down: the card states coverage, and a hardcoded
    number is the same drift problem as a hardcoded licence.
    """
    import duckdb

    shard_glob = data_dir / "endfb-8.0" / "xs" / "*.parquet"
    con = duckdb.connect()
    return con.sql(f"SELECT count(DISTINCT (target_Z * 1000 + target_A)) FROM read_parquet('{shard_glob}')").fetchone()[
        0
    ]


def split_by_isotope(shard_dir: Path, out_dir: Path) -> list[Path]:
    """Rewrite per-element shards as per-isotope shards, matching the mirror.

    The repository shards per element (`n_Nd.parquet`) because that is the shape
    the builders produce and the shape the release tarball wants. The mirror
    shards per isotope (`n_Nd143.parquet`) because its whole purpose is fetching
    one isotope without pulling its neighbours.

    Those are both right for their context, so the sync converts rather than
    forcing one on the other -- which also keeps every already-published URL
    working. The element symbol comes from the source filename, so this needs no
    Z-to-symbol table of its own.
    """
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for src in sorted(shard_dir.glob("*.parquet")):
        # 'n_Nd' -> projectile 'n', symbol 'Nd'
        proj, _, sym = src.stem.partition("_")
        if not sym:
            logger.warning("skipping %s: cannot read an element symbol from the name", src.name)
            continue
        df = pl.read_parquet(src)
        for (target_a,), part in df.group_by(["target_A"], maintain_order=True):
            dest = out_dir / f"{proj}_{sym}{target_a}.parquet"
            part.write_parquet(dest, compression="zstd")
            written.append(dest)
    return written


def check_shard_layout(local: set[str], published: set[str]) -> None:
    """Refuse to upload a shard set that does not overlap what is published.

    The published mirror shards per *isotope* (`n_Nd143.parquet`); the local tree
    shards per *element* (`n_Nd.parquet`). Uploading one into the other does not
    overwrite anything — it interleaves two incompatible layouts in one
    directory, leaving the same data under two namings with nothing to say which
    is authoritative. Refuse rather than corrupt.

    An empty `published` is not a mismatch: that is a first sync into an empty
    directory, which is exactly when uploading is correct.
    """
    if not published or local & published:
        return
    raise SystemExit(
        f"refusing to upload: {len(local)} local shards ({sorted(local)[0]}, …) share no name "
        f"with the {len(published)} already published ({sorted(published)[0]}, …). "
        "These are different sharding schemes; reconcile them before syncing."
    )


def build_card(data_dir: Path, *, synced: bool) -> str:
    """Generate the dataset card, with licence taken from licenses.toml.

    Deriving it rather than hand-writing it is the point: a wrong licence on a
    public mirror is a compliance problem, and hand-maintained metadata drifts.

    The technical sections are generated too, because this card *overwrites* the
    published one. An earlier version emitted only the licence block, which would
    have silently deleted the coverage, schema and log-log interpolation guidance
    a consumer needs to use the shards correctly. Anything the published card
    must keep has to be produced here.
    """
    licenses = tomllib.loads((data_dir / "licenses.toml").read_text())
    lic = licenses["libraries"][MIRRORED_LIBRARY]
    catalog = json.loads((data_dir / "catalog.json").read_text())
    n_isotopes = isotope_count(data_dir)

    # The card must describe the shards a consumer will actually download. When
    # this run is pushing shards, that is the canonical schema; when it is
    # card-only, the mirror still holds the pre-migration snapshot and saying
    # otherwise would send people looking for columns that are not there.
    schema = ", ".join(CANONICAL_XS_SCHEMA) if synced else PUBLISHED_SCHEMA
    status = "" if synced else _STALE_SHARDS_NOTE

    return f"""---
license: other
license_name: us-government-work
license_link: {lic["terms_url"]}
tags:
  - nuclear-data
  - cross-sections
  - neutron
  - endf
pretty_name: nucl-parquet data (NJOY-processed neutron cross sections)
---

> **Licensing.** **{lic["license"]}**
>
> {lic["notes"]}
>
> MIT covers the nucl-parquet *code and conversion*, not the bundled evaluated
> data. Per-library terms are recorded in
> [`data/licenses.toml`](https://github.com/exoma-ch/nucl-parquet/blob/main/data/licenses.toml).
>
> Cite: {lic["citation"]}
>
> Custodian: {lic["custodian"]}

# nucl-parquet-data — NJOY-processed neutron cross sections

Per-isotope neutron cross-section shards for
[nucl-parquet](https://github.com/exoma-ch/nucl-parquet), hosted here so consumers
**fetch one isotope at a time** rather than download a monolith.

Data version: `{catalog["data_version"]}`

## `neutron/` — ENDF/B-VIII.0, NJOY-processed

- **Source:** OpenMC-processed ENDF/B-VIII.0 HDF5
  ([openmc-data-storage/ENDF-B-VIII.0-NNDC](https://github.com/openmc-data-storage/ENDF-B-VIII.0-NNDC))
  — the pointwise data OpenMC / MCNP / Serpent transport on.
- **Temperature:** 294 K (Doppler-broadened).
- **Coverage:** {n_isotopes} target isotopes.
- **Channels:** transmutation only — capture (MT 102) plus threshold reactions
  (n,2n / n,3n / n,p / n,α / …). Elastic, inelastic-to-levels and fission are
  excluded; those live in the `endfb-8.0-channels` transport product in the main
  repository.
- **Schema:** `{schema}`.
- **Grid:** dense NJOY pointwise grid thinned to ≤1 % log-log accuracy.
  **Interpolate in log-log** — the 1/v region is sparse, so nearest-point reads
  will over-read.
{status}
Unlike a raw MF=3 ingestion, these carry the full resolved/unresolved resonance
region and the thermal point (e.g. Nd-143(n,γ) reaches 1e-5 eV, not 0.225 MeV).

## Usage

The shards are plain Parquet — read them directly:

```python
from huggingface_hub import hf_hub_download
import duckdb

path = hf_hub_download(
    "gerchowl/nucl-parquet-data", "neutron/n_Nd143.parquet", repo_type="dataset"
)
duckdb.sql(f"SELECT energy_MeV, xs_mb FROM '{{path}}' WHERE residual_A = 144")
```

For the full dataset — every library, not just neutron — install the package and
use the release tarball instead:

```bash
pip install nucl-parquet && python -c "import nucl_parquet; nucl_parquet.download()"
```

---

*This card is generated by `scripts/sync_huggingface.py` from
`data/licenses.toml`. Edit it there, not here — a hand edit will be overwritten
on the next data release.*
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--card-only",
        action="store_true",
        help="push the dataset card but not the shards",
    )
    args = ap.parse_args()

    shard_dir = args.data_dir / "endfb-8.0" / "xs"
    will_sync_shards = not args.card_only and shard_dir.exists()

    if args.dry_run:
        print(build_card(args.data_dir, synced=will_sync_shards))
        logger.info("dry run — nothing uploaded")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN is not set")
        raise SystemExit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Shards first, card second. The card *describes* the shards — their schema
    # and sharding — so pushing it first would leave the mirror advertising a
    # layout it does not have if the upload then failed.
    if will_sync_shards:
        with tempfile.TemporaryDirectory() as tmp:
            staged = split_by_isotope(shard_dir, Path(tmp))
            published = {
                f.rsplit("/", 1)[-1]
                for f in (s.path for s in api.list_repo_tree(args.repo_id, "neutron", repo_type="dataset"))
                if f.endswith(".parquet")
            }
            check_shard_layout({p.name for p in staged}, published)
            api.upload_folder(
                folder_path=tmp,
                path_in_repo="neutron",
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="chore: sync neutron shards",
            )
        logger.info("%d isotope shards pushed from %s", len(staged), shard_dir)
    elif not args.card_only:
        logger.warning("no shards at %s — card only", shard_dir)

    api.upload_file(
        path_or_fileobj=build_card(args.data_dir, synced=will_sync_shards).encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="chore: regenerate dataset card from licenses.toml",
    )
    logger.info("card pushed to %s", args.repo_id)


if __name__ == "__main__":
    main()
