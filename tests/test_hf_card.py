"""The generated Hugging Face dataset card must stay complete and correct.

`scripts/sync_huggingface.py` *overwrites* the published card on every data
release. That makes the generator the single source of truth for what the public
mirror says -- and it means anything the generator forgets to emit is silently
deleted from a public page on the next release.

It had already forgotten most of it. An earlier version emitted the licence block
and nothing else, which would have wiped the coverage, schema and the log-log
interpolation warning a consumer needs to read the shards correctly.

Two things are therefore pinned here: the licence really is derived from
`data/licenses.toml` (the #234 compliance guarantee), and the technical sections
are actually present.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "licenses.toml").exists(),
    reason="no data tree",
)


def _card() -> str:
    from sync_huggingface import build_card

    return build_card(DATA_DIR, synced=True)


@pytest.mark.data
def test_licence_frontmatter_is_not_mit() -> None:
    """The mirror published `license: mit` over a US Government work (#234).

    An MIT grant is not ours to make over ENDF/B-VIII.0. This is the regression
    that must never come back.
    """
    card = _card()
    head = card.split("---")[1]
    assert "license: other" in head
    assert "license_name: us-government-work" in head
    assert "mit" not in head.lower(), "the mirror must not claim MIT over evaluated data"


@pytest.mark.data
def test_licence_text_is_derived_from_licenses_toml() -> None:
    """Not merely correct today -- structurally unable to drift.

    If the card restated the licence in prose, `licenses.toml` could be corrected
    and the public mirror would keep the old claim.
    """
    lic = tomllib.loads((DATA_DIR / "licenses.toml").read_text())["libraries"]["endfb-8_0"]
    card = _card()
    for field in ("license", "notes", "citation", "custodian"):
        value = lic[field]
        assert value in card, f"card does not carry licenses.toml::{field} verbatim"


@pytest.mark.data
def test_card_keeps_the_documentation_a_consumer_needs() -> None:
    """The card overwrites a published page; omissions are deletions.

    Each of these was present on the live card and absent from the generator --
    they would have been destroyed on the next release with a token configured.
    """
    card = _card()
    required = {
        "log-log": "how to interpolate; nearest-point reads over-read the 1/v region",
        "294 K": "the temperature the data is broadened to",
        "Coverage:": "how many isotopes are here",
        "Schema:": "the column layout",
        "openmc-data-storage": "provenance of the processed form",
        "Usage": "a worked example",
    }
    missing = [f"{k} ({why})" for k, why in required.items() if k not in card]
    assert not missing, "generated card is missing:\n  " + "\n  ".join(missing)


@pytest.mark.data
def test_card_describes_the_published_schema_not_the_local_one() -> None:
    """The card documents what a consumer downloads, which is not what we hold.

    The mirror is a pre-migration snapshot: 7 columns, sharded per isotope. The
    repository is on the 18-column canonical schema, sharded per element. Naming
    the local schema on the card would send consumers looking for `MT` and
    `projectile` columns that are not in the file they just fetched.
    """
    from sync_huggingface import PUBLISHED_SCHEMA, build_card

    from nucl_parquet._schemas import CANONICAL_XS_SCHEMA

    published_cols = {c.strip() for c in PUBLISHED_SCHEMA.split(",")}
    assert published_cols < set(CANONICAL_XS_SCHEMA), "published schema should be a subset of canonical"

    # Card-only run: the mirror still holds the pre-migration snapshot, so the
    # card must name the old columns and disclose the divergence.
    stale = build_card(DATA_DIR, synced=False)
    assert PUBLISHED_SCHEMA in stale
    assert "lag the main repository" in stale

    # Shard-syncing run: the mirror is being brought up to date in the same
    # invocation, so the card names the canonical schema and drops the notice.
    fresh = build_card(DATA_DIR, synced=True)
    assert ", ".join(CANONICAL_XS_SCHEMA) in fresh
    assert "lag the main repository" not in fresh


@pytest.mark.data
def test_coverage_count_is_derived_not_hardcoded() -> None:
    """A written-down isotope count is the same drift problem as a written-down licence."""
    from sync_huggingface import isotope_count

    n = isotope_count(DATA_DIR)
    assert n > 0
    assert f"Coverage:** {n} target isotopes" in _card()


def test_shard_layout_guard_blocks_incompatible_shardings() -> None:
    """The guard is the only thing standing between a sync and a corrupted mirror.

    Uploading element shards into a directory of isotope shards does not
    overwrite -- it interleaves, leaving `n_Nd.parquet` beside `n_Nd143.parquet`
    with the same data under two namings. This is the live situation: 97 local
    element files against 533 published isotope files.
    """
    from sync_huggingface import check_shard_layout

    with pytest.raises(SystemExit) as e:
        check_shard_layout({"n_Nd.parquet", "n_Fe.parquet"}, {"n_Nd143.parquet", "n_Fe56.parquet"})
    assert "different sharding schemes" in str(e.value)


def test_shard_layout_guard_allows_a_matching_sync() -> None:
    """It must not block the case it exists to permit: same scheme, newer data."""
    from sync_huggingface import check_shard_layout

    check_shard_layout({"n_Nd143.parquet", "n_Fe56.parquet"}, {"n_Nd143.parquet"})


def test_shard_layout_guard_allows_a_first_sync() -> None:
    """An empty published set is not a mismatch -- it is the initial upload."""
    from sync_huggingface import check_shard_layout

    check_shard_layout({"n_Nd.parquet"}, set())


@pytest.mark.data
def test_card_advertises_no_api_that_does_not_exist() -> None:
    """The published card demonstrated `nucl_parquet.neutron_xs(...)`, which is
    not a function this package has ever defined. A usage example that does not
    run is worse than none -- it sends consumers to a traceback."""
    import nucl_parquet

    card = _card()
    for line in card.splitlines():
        if "nucl_parquet." not in line:
            continue
        attr = line.split("nucl_parquet.", 1)[1].split("(")[0].strip()
        if attr and attr.isidentifier():
            assert hasattr(nucl_parquet, attr), f"card references nucl_parquet.{attr}, which does not exist"


@pytest.mark.data
def test_split_by_isotope_is_lossless_and_reproduces_published_names(tmp_path: Path) -> None:
    """The mirror shards per isotope; the repository shards per element.

    The sync converts rather than forcing one shape on the other, so that every
    already-published `n_Nd143.parquet` URL keeps resolving. Two things have to
    hold: no row may be lost, and the names must come out exactly as published.
    """
    import polars as pl
    from sync_huggingface import split_by_isotope

    src = DATA_DIR / "endfb-8.0" / "xs" / "n_Nd.parquet"
    if not src.exists():
        pytest.skip("endfb-8.0 shards not present")

    staging = tmp_path / "in"
    staging.mkdir()
    (staging / src.name).write_bytes(src.read_bytes())

    produced = split_by_isotope(staging, tmp_path / "out")

    before = pl.read_parquet(src)
    after = pl.concat([pl.read_parquet(p) for p in produced])
    assert after.height == before.height, "rows lost in the split"
    assert after.columns == before.columns, "the split must not reshape the schema"

    # Names match what the mirror already serves.
    assert {p.name for p in produced} >= {"n_Nd143.parquet", "n_Nd142.parquet"}
    # And each shard really is one isotope.
    assert all(pl.read_parquet(p)["target_A"].n_unique() == 1 for p in produced)
