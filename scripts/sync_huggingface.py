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
import tomllib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REPO_ID = "gerchowl/nucl-parquet-data"

# Which library's shards the mirror carries, and therefore whose licence governs
# the dataset card.
MIRRORED_LIBRARY = "endfb-8_0"

# The columns the *published* shards actually have. Deliberately not
# `CANONICAL_XS_SCHEMA`: the mirror is a pre-migration snapshot, and the card
# must describe what a consumer will find when they download the file, not what
# the repository holds today. Update this only when the shards are re-synced.
PUBLISHED_SCHEMA = "target_Z, target_A, residual_Z, residual_A, state, energy_MeV, xs_mb"


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


def build_card(data_dir: Path) -> str:
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
- **Schema:** `{PUBLISHED_SCHEMA}`.
- **Grid:** dense NJOY pointwise grid thinned to ≤1 % log-log accuracy.
  **Interpolate in log-log** — the 1/v region is sparse, so nearest-point reads
  will over-read.

## Status — this mirror lags the main repository

These shards are a snapshot taken before the canonical-schema migration. They
are correct as far as they go, but they are **not** the current shape of the
data:

- they carry {len(PUBLISHED_SCHEMA.split(", "))} columns, where the repository now
  carries the canonical cross-section schema (adding `library`, `kind`,
  `projectile`, `proj_Z`, `proj_A`, `MT`, uncertainties and provenance);
- they shard per isotope (`n_Nd143.parquet`), where the repository shards per
  element (`n_Nd.parquet`).

Re-syncing the shards is pending reconciliation of those two layouts. **For
current data, use the release tarball** (see Usage below) rather than these
files. This card is regenerated on every data release, so the licence metadata
above is always current even while the shards are not.

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
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--card-only",
        action="store_true",
        help="push the dataset card but not the shards",
    )
    args = ap.parse_args()

    card = build_card(args.data_dir)
    if args.dry_run:
        print(card)
        logger.info("dry run — nothing uploaded")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN is not set")
        raise SystemExit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="chore: regenerate dataset card from licenses.toml",
    )
    logger.info("card pushed to %s", args.repo_id)

    if args.card_only:
        return

    shard_dir = args.data_dir / "endfb-8.0" / "xs"
    if not shard_dir.exists():
        logger.warning("no shards at %s — card only", shard_dir)
        return

    # The published mirror shards per *isotope* (`n_Nd143.parquet`); the local
    # tree shards per *element* (`n_Nd.parquet`). Uploading one into the other
    # does not overwrite — it interleaves two incompatible layouts in the same
    # directory, leaving the same data under two namings and no way for a
    # consumer to tell which is authoritative. Refuse rather than corrupt.
    published = {
        f.rsplit("/", 1)[-1]
        for f in (s.path for s in api.list_repo_tree(args.repo_id, "neutron", repo_type="dataset"))
        if f.endswith(".parquet")
    }
    local = {p.name for p in shard_dir.glob("*.parquet")}
    if published and not local & published:
        raise SystemExit(
            f"refusing to upload: {len(local)} local shards ({sorted(local)[0]}, …) share no name "
            f"with the {len(published)} already published ({sorted(published)[0]}, …). "
            "These are different sharding schemes; reconcile them before syncing."
        )

    api.upload_folder(
        folder_path=str(shard_dir),
        path_in_repo="neutron",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="chore: sync neutron shards",
    )
    logger.info("shards pushed from %s", shard_dir)


if __name__ == "__main__":
    main()
