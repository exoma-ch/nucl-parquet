"""Backfill target nuclides that a library's original ingest silently dropped.

Why this exists (issue #335)
----------------------------
`tendl-2023-iso` was cut from TENDL's *residual-production* tables — TALYS's
per-residual output (`tables/residual/rp*.tot`), on TALYS's standard energy grid.
Those files only exist where TALYS actually ran.

For a handful of light nuclides TENDL-2023 does not run TALYS at all: it adopts
an older external evaluation wholesale. Every nuclide below carries a LANL
evaluation from 1988–2004 (Young, Hale, Page, Arthur) rather than a 2023
IAEA/Koning TALYS run, and PSI publishes no `tables/residual/` directory for
any of them. Confirmed directly from the Wayback archive of the PSI tree:
`triton_file/Li/Li006/` holds only the ENDF tape, while its sibling `Li007/`
(a 2023 Koning TALYS evaluation) holds all eight `rp*.tot` files.

So the ingest asked for tables that were never published, got nothing, and a
`if not rows: continue` dropped the nuclide without a word — taking the whole
element shard with it when *every* isotope of that element was affected
(`p_Li`, `d_Li` were never written).

What those tapes carry instead varies, and it bounds what can be recovered:

* Seven of the eight express reactions as ordinary **exclusive ENDF channels**
  (MT=102, or the discrete-level bands MT=50–91 / 600–849) and carry no MT=5 at
  all. Summing those channels per residual reconstructs their production data
  in full.
* **p+Be-9 is different.** It has MT=5 (peaking at 700 mb, 0–113 MeV) *and* an
  MF=6/MT=5 block — but that block lists only light ejectiles (n, p, d, t, α, γ)
  and names no heavy product. So most of its non-elastic strength cannot be
  attributed to a residual from this tape at all. What is recovered here is the
  (p,n) discrete-level band summed into B-9 production — which is the channel a
  Be converter foil actually needs — and `channel_rows` warns loudly about the
  rest. Consumers wanting the full Be-9 residual set should use `tendl-2025`,
  whose TALYS evaluation tabulates ten residuals.

This script rebuilds the missing nuclides from the authoritative ENDF-6 tapes on
the IAEA NDS mirror, summing exclusive channels per residual so the output keeps
the library's existing contract: `kind='production'`, one row per
(target, residual, state, energy), `MT` null because a production row is a sum
over channels rather than a single one.

Residual convention
-------------------
A "residual" is any product that is not one of the six standard ENDF ejectiles
(n, ¹H, ²H, ³H, ³He, ⁴He) or a photon. This is TALYS's own convention and it is
what the shipped library already follows — it contains He-6 residuals but no
He-4/He-3, and no Z<2 products at all. Keeping it means backfilled rows are
queryable on the same terms as every other row in the library.

State vocabulary
----------------
Backfilled rows spell their state `'sum'` (`state_vocabulary.SUM`) — the total
over isomeric states, which is what an MF=3 channel sum is. The rest of this
library still ships the retired `''` for the same claim and is listed in
`state_vocabulary.PENDING_MIGRATION` until `scripts/migrate_state_vocabulary.py`
rewrites it. So a backfilled shard carries both spellings for a while.

That is deliberate. The alternative — writing `''` to match the neighbouring
rows — would add new rows to a debt the repository has already decided to
retire (#357, #380), and `''` is not a value the vocabulary admits. Mixed is
visible and converges; uniformly-wrong is invisible and does not. The migration
is idempotent, so it leaves these rows alone when it runs.

Usage:
    # Show what would change, touching nothing:
    nix develop -c uv run python scripts/backfill_xs_nuclides.py --dry-run

    # Apply to the shipped data directory:
    nix develop -c uv run python scripts/backfill_xs_nuclides.py

    # A single nuclide:
    nix develop -c uv run python scripts/backfill_xs_nuclides.py --only p:4:9
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _canonical import LIGHT_ION, Z_TO_SYMBOL, canonical_frame, element_stem  # noqa: E402
from _paths import DATA_DIR, ROOT  # noqa: E402
from fetch_endf_libs import IAEA_MIRROR, is_signed_section, mt_to_residual  # noqa: E402

sys.path.insert(0, str(ROOT))  # so `nucl_parquet` imports from the checkout

from nucl_parquet.endf_interp import LIN_LIN, laws_per_point  # noqa: E402
from nucl_parquet.state_vocabulary import SUM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COMPRESSION = "zstd"

# The six standard ENDF ejectiles plus the photon. A product that is one of
# these is an emitted particle, not the residual nucleus we tabulate.
EJECTILES: frozenset[tuple[int, int]] = frozenset({(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 3), (2, 4)})

# IAEA sub-directory name per projectile code.
SUBLIB_DIR: dict[str, str] = {"n": "n", "p": "p", "d": "d", "t": "t", "h": "he3", "a": "he4"}

# Nuclides dropped by the tendl-2023-iso ingest, as (projectile, Z, A).
# Established in #335 by diffing natural-isotope coverage against tendl-2025 and
# confirming each one exists on the IAEA NDS TENDL-2023 mirror.
TENDL_2023_GAPS: tuple[tuple[str, int, int], ...] = (
    ("p", 4, 9),  # Be-9  — 100 % of natural Be; root cause of hyrr#668
    ("p", 6, 13),  # C-13
    ("p", 3, 6),  # Li-6
    ("p", 3, 7),  # Li-7
    ("d", 3, 6),  # Li-6
    ("d", 3, 7),  # Li-7
    ("t", 3, 6),  # Li-6
    ("h", 3, 6),  # Li-6
)


def find_tape(session: requests.Session, library_path: str, projectile: str, z: int, a: int) -> str | None:
    """Locate a nuclide's tape by listing the sub-library directory.

    The MAT number is part of the filename and is not always the ENDF-standard
    value (TENDL-2023 ships p+Be-9 as MAT 0409, not the canonical 0425), so the
    filename must be discovered rather than computed.
    """
    sub = SUBLIB_DIR[projectile]
    url = f"{IAEA_MIRROR}/{library_path}/{sub}/"
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    sym = Z_TO_SYMBOL[z]
    pattern = re.compile(rf"{sub}_{z:03d}-{sym}-{a}_(\d+)\.zip")
    hits = sorted(set(pattern.findall(resp.text)))
    if not hits:
        return None
    return f"{url}{sub}_{z:03d}-{sym}-{a}_{hits[0]}.zip"


def fetch_tape_text(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"empty zip: {url}")
        return zf.read(names[0]).decode("ascii", errors="replace")


def _lin_lin(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear-linear interpolation, zero outside the tabulated range.

    ENDF interpolation law 2, and used only on the summation path, which
    `channel_rows` refuses to take for any law but 2 — so nothing is ever
    *resampled* under the wrong law.

    A residual fed by a single channel is emitted on the tape's own grid
    verbatim, law and all, and several of the backfilled tapes declare laws 5/6
    (log-log, charged-particle threshold). A consumer interpolating those points
    linearly is wrong between them, most visibly near threshold.

    That used to be silent debt, shared with every library this repository ships.
    It is not any more: #338 added `interp_law`, and `channel_rows` now carries
    the evaluator's own law onto every row it emits, so a consumer can see which
    of its points `np.interp` reads wrongly instead of having to know.
    """
    out = np.interp(grid, x, y, left=0.0, right=0.0)
    out[(grid < x[0]) | (grid > x[-1])] = 0.0
    return out


# A summed MT and the discrete-level band that decomposes it. Both describe the
# same reaction, so a tape carrying the total *and* its levels must contribute
# only once — otherwise the residual is counted twice. Where both are present the
# total wins, being the evaluator's own sum.
#
# The neutron band is 51–91 here, per ENDF-102's definition of MT=4, even though
# mt_to_residual maps the wider 50–91 (MT=50 is a real (z,n₀) channel for charged
# projectiles). Whether MT=4 is meant to include n₀ is exactly the ambiguity
# _suppressed_levels refuses to guess at — see below.
REDUNDANT_BANDS: tuple[tuple[int, tuple[int, int]], ...] = (
    (4, (51, 91)),  # (z,n) total  vs n levels
    (16, (875, 891)),  # (z,2n) total vs 2n levels
    (103, (600, 649)),  # (z,p) total  vs p levels
    (104, (650, 699)),  # (z,d) total  vs d levels
    (105, (700, 749)),  # (z,t) total  vs t levels
    (106, (750, 799)),  # (z,³He) total vs ³He levels
    (107, (800, 849)),  # (z,α) total  vs α levels
)


def _suppressed_levels(present: set[int]) -> set[int]:
    """MT numbers to ignore because their summed equivalent is also tabulated.

    Raises when a tape carries both MT=4 and MT=50, because the two readings of
    ENDF-102 disagree there and both failure modes are silent: if MT=4 already
    contains n₀ then keeping MT=50 double-counts it, and if it does not then
    dropping MT=50 loses the ground-state transition — the very bug this branch
    fixed elsewhere. No TENDL-2023 tape observed so far does this (TALYS writes
    either the total or the band, never both), so rather than guess at a
    structure nobody has validated, stop and make a human look.
    """
    drop: set[int] = set()
    if 4 in present and 50 in present:
        raise ValueError(
            "tape carries both MT=4 and MT=50; whether the (z,n) total includes "
            "the n0 ground-state transition is ambiguous, and guessing either way "
            "fails silently. Verify against the evaluation before ingesting it."
        )
    for total, (lo, hi) in REDUNDANT_BANDS:
        if total in present:
            drop |= {mt for mt in present if lo <= mt <= hi}
    return drop


def channel_rows(
    endf_text: str, projectile: str, target_z: int, target_a: int
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Residual-production curves from a tape's exclusive MF=3 channels.

    Returns {(residual_Z, residual_A): (energy_eV, xs_barns, interp_laws)}, where
    `interp_laws[i]` is the ENDF `INT` for the interval starting at
    `energy_eV[i]` (#338). Channels sharing a residual are summed onto the union
    of their energy grids.
    """
    import endf

    material = endf.Material(io.StringIO(endf_text))
    proj_z, proj_a = LIGHT_ION[projectile]

    mf3_mts = {mt for mf, mt in material.section_data if mf == 3}
    skip = _suppressed_levels(mf3_mts)

    # MT=5 ("anything else") names no single residual, so it cannot become
    # production rows on its own. Some tapes push most of their non-elastic
    # strength through it and describe the outcome only as light-ejectile yields
    # in MF=6/MT=5 — no heavy product ZAPs, nothing to attribute. That strength
    # is real and simply not recoverable in residual form from this tape, which
    # would leave the nuclide quietly under-covered. Say so, with the magnitude,
    # so "restored" is never mistaken for "complete".
    if (3, 5) in material.section_data:
        sigma5 = material.section_data[(3, 5)].get("sigma")
        heavy = [
            (int(p.get("ZAP", 0)) // 1000, int(p.get("ZAP", 0)) % 1000)
            for p in material.section_data.get((6, 5), {}).get("products", [])
        ]
        recoverable = [za for za in heavy if za not in EJECTILES]
        if sigma5 is not None and not recoverable:
            logger.warning(
                "  Z=%d A=%d %s: MF=3/MT=5 carries up to %.1f mb (%.2f–%.0f MeV) that "
                "names no residual — MF=6/MT=5 lists only ejectiles, so this strength "
                "is NOT represented in the backfilled rows",
                target_z,
                target_a,
                projectile,
                max(sigma5.y) * 1e3,
                sigma5.x[0] * 1e-6,
                sigma5.x[-1] * 1e-6,
            )

    # residual -> list of (Tabulated1D, mt)
    per_residual: dict[tuple[int, int], list] = {}
    for (mf, mt), section in material.section_data.items():
        if mf != 3 or mt in skip:
            continue
        residual = mt_to_residual(mt, target_z, target_a, proj_z, proj_a)
        if residual is None or residual in EJECTILES:
            continue
        tab = section.get("sigma")
        if tab is None or len(tab.x) == 0:
            continue
        # Same rule as the main ingest (#377/#379): a section carrying any
        # negative value is not a cross-section, so its positive part is not one
        # either and the whole section goes. `build_frame`'s point-wise `xs > 0`
        # filter is exactly the half-fix #379 rejected — it would keep the
        # positive lobe of a signed curve and write it as `xs_mb`.
        #
        # No backfilled tape trips this today: across all eight, the only signed
        # MF=3 section is p+Be-9 MT=2, which never reaches here because MT=2 is
        # in NO_RESIDUAL_MTS. The check is here so the class stays closed for the
        # next tape, not because it currently fires.
        if is_signed_section(np.asarray(tab.y, dtype=float)):
            logger.warning(
                "  Z=%d A=%d %s: MF=3/MT=%d carries negative values and is not a "
                "cross-section — dropping the section whole (#377)",
                target_z,
                target_a,
                projectile,
                mt,
            )
            continue
        per_residual.setdefault(residual, []).append((tab, mt))

    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for residual, tabs in per_residual.items():
        if len(tabs) == 1:
            tab, _ = tabs[0]
            # Single channel — emit its own grid verbatim, no interpolation, so
            # the evaluator's own per-region laws carry over exactly (#338).
            out[residual] = (
                np.asarray(tab.x, float),
                np.asarray(tab.y, float),
                laws_per_point(tab.breakpoints, tab.interpolation, len(tab.x)),
            )
            continue

        laws = {int(v) for tab, _ in tabs for v in tab.interpolation}
        if laws != {2}:
            mts = sorted(mt for _, mt in tabs)
            raise ValueError(
                f"Z={target_z} A={target_a} {projectile}: residual {residual} needs "
                f"summing over MTs {mts} but uses interpolation laws {sorted(laws)}; "
                "only lin-lin (law 2) summation is implemented."
            )
        grid = np.unique(np.concatenate([np.asarray(tab.x, float) for tab, _ in tabs]))
        total = np.zeros_like(grid)
        for tab, _ in tabs:
            total += _lin_lin(np.asarray(tab.x, float), np.asarray(tab.y, float), grid)
        # Every contribution is lin-lin — the guard above refuses anything else —
        # so resampling onto the union grid is exact and the sum really is law 2.
        # This is the same rule `fetch_endf_libs.sum_on_union_grid` applies; the
        # difference is only that this script raises where that one emits NULL,
        # because a backfill rewrites shipped data and should stop for a human.
        out[residual] = (grid, total, np.full(grid.shape, LIN_LIN, dtype=np.int64))

    return out


#: The columns a shard is sorted by. Identical to the order
#: `fetch_endf_libs.fetch_library` writes, so a backfilled shard and a rebuilt
#: one are byte-comparable rather than merely equivalent.
SORT_KEY = ("target_A", "kind", "MT", "residual_Z", "residual_A", "state", "energy_MeV")


def build_frame(library: str, projectile: str, target_z: int, target_a: int, curves: dict) -> pl.DataFrame:
    """Canonical-schema rows for one target nuclide."""
    proj_z, proj_a = LIGHT_ION[projectile]
    recs: list[dict] = []
    for (res_z, res_a), (energies_ev, xs_barns, interp_laws) in sorted(curves.items()):
        for e_ev, xs_b, law in zip(energies_ev, xs_barns, interp_laws):
            # Same guard the main converter uses: drop non-positive points and
            # TALYS overflow sentinels (~1.99e35 b).
            if xs_b <= 0 or xs_b > 1e30:
                continue
            recs.append(
                {
                    "target_A": target_a,
                    # A production row is a sum over channels, so it names no
                    # single MT. Null, not 0 — see CLAUDE.md principle 3.
                    "MT": None,
                    "residual_Z": res_z,
                    "residual_A": res_a,
                    # MF=3 carries no isomeric split, so summing its channels
                    # gives the total over states. That is a claim, and 'sum' is
                    # how the vocabulary spells it (#357/#380) — see the module
                    # docstring for why these rows do not copy their
                    # neighbours' retired ''.
                    "state": SUM,
                    "energy_MeV": float(e_ev) * 1e-6,
                    "xs_mb": float(xs_b) * 1e3,
                    # The evaluator's own interpolation law for the interval
                    # starting here. These tapes are exactly the ones that
                    # motivated #338: several declare laws 5 and 6, where a
                    # lin-lin reading is wrong by up to 30% and by multiples
                    # respectively.
                    "interp_law": int(law),
                }
            )
    # Identity columns come from `canonical_frame`, the one place that knows how
    # to spell a canonical row (#359) — rather than a second hand-written copy
    # of the eighteen-column literal, which is how the builders and the migration
    # drifted apart in the first place.
    frame = pl.DataFrame(
        recs,
        schema={
            "target_A": pl.Int32,
            "MT": pl.Int32,
            "residual_Z": pl.Int32,
            "residual_A": pl.Int32,
            "state": pl.Utf8,
            "energy_MeV": pl.Float64,
            "xs_mb": pl.Float64,
            "interp_law": pl.Int32,
        },
    )
    return canonical_frame(
        frame,
        library=library,
        kind="production",
        projectile=projectile,
        proj_z=proj_z,
        proj_a=proj_a,
        target_z=target_z,
    )


def merge_into_shard(shard: Path, new: pl.DataFrame, dry_run: bool) -> tuple[int, int, str]:
    """Add `new` rows to a shard, replacing any existing rows for those targets."""
    targets = set(new["target_A"].to_list())
    if shard.exists():
        existing = pl.read_parquet(shard)
        kept = existing.filter(~pl.col("target_A").is_in(list(targets)))
        action = "updated" if len(kept) != len(existing) else "extended"

        # Widen the *old* rows to the new schema, never narrow the new ones to
        # the old. A parquet file has one schema, so appending a row that knows
        # its interpolation law to a shard that has no such column would silently
        # drop the law — data we just went to the mirror to fetch, discarded to
        # match a file that predates the column (#338). The other direction is
        # honest: the pre-existing rows genuinely have no law on record, and a
        # typed NULL says exactly that.
        missing = [c for c in new.columns if c not in kept.columns]
        if missing:
            logger.info(
                "  %s predates %s; adding it as NULL to its %d existing row(s), which is what "
                "'the source never stated one' looks like",
                shard.name,
                ", ".join(missing),
                len(kept),
            )
            kept = kept.with_columns([pl.lit(None, dtype=new.schema[c]).alias(c) for c in missing])
        combined = pl.concat([kept.select(new.columns), new], how="vertical")
        before = len(existing)
    else:
        action = "created"
        combined = new
        before = 0

    # Sort only, never `unique(subset=…)`. Two MT partials can legitimately reach
    # the same residual at the same energy — deduping on the identity columns
    # collapsed Fe-56(n,p) to 0.1 mb in #326 — and the target's own rows have
    # already been replaced wholesale by the filter above, so there is nothing
    # for a dedupe to do here but damage.
    combined = combined.sort(list(SORT_KEY))

    if not dry_run:
        shard.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(shard, compression=COMPRESSION)
    return before, len(combined), action


def build_parser() -> argparse.ArgumentParser:
    """The CLI, extractable without running anything (#363).

    This script fetches from the IAEA mirror and overwrites *tracked* parquets,
    so importing it must reveal where it writes without writing there.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"Data directory (default: {DATA_DIR.name}/)")
    ap.add_argument("--library", default="tendl-2023-iso")
    ap.add_argument("--library-path", default="TENDL-2023", help="directory name on the IAEA mirror")
    ap.add_argument("--only", help="restrict to one gap, formatted proj:Z:A (e.g. p:4:9)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    gaps = TENDL_2023_GAPS
    if args.only:
        proj, z, a = args.only.split(":")
        gaps = ((proj, int(z), int(a)),)

    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (nuclear data research)"

    xs_dir = args.data_dir / args.library / "xs"
    empty: list[str] = []

    for projectile, z, a in gaps:
        sym = Z_TO_SYMBOL[z]
        label = f"{projectile}+{sym}-{a}"
        url = find_tape(session, args.library_path, projectile, z, a)
        if url is None:
            logger.error("%s: no tape on the mirror — upstream really lacks it", label)
            continue

        curves = channel_rows(fetch_tape_text(session, url), projectile, z, a)
        frame = build_frame(args.library, projectile, z, a, curves)
        if frame.is_empty():
            # Not a silent skip: say so, loudly, and keep going.
            empty.append(label)
            logger.warning(
                "%s: tape has no channel with a residual outside the standard "
                "ejectiles — nothing to tabulate under this library's convention",
                label,
            )
            continue

        # `element_stem`, not the symbol directly: it is the builder's own
        # spelling and falls back to 'Z61' for elements with no symbol, so a
        # backfilled shard always lands on the file a re-ingest would write.
        shard = xs_dir / f"{projectile}_{element_stem(z)}.parquet"
        before, after, action = merge_into_shard(shard, frame, args.dry_run)
        residuals = ", ".join(f"{Z_TO_SYMBOL[rz]}-{ra}" for rz, ra in sorted(curves) if (rz, ra) not in EJECTILES)
        logger.info(
            "%-10s %-16s %s: %d -> %d rows (+%d for %s-%d; residuals: %s)",
            label,
            shard.name,
            action,
            before,
            after,
            len(frame),
            sym,
            a,
            residuals,
        )

    if empty:
        logger.warning("No residual-production data available for: %s", ", ".join(empty))
    if args.dry_run:
        logger.info("dry run — nothing written")


if __name__ == "__main__":
    main()
