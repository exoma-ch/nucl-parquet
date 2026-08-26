"""Compare an evaluation's MF=3 channel totals against its MF=10 isomeric split (#368).

Where MF=3 gives a channel total and MF=10 gives the ground/metastable split for
the same product, `sigma(state='sum')` ought to equal `sigma('g') + sigma('m') +
…`. It usually does. Nine products across two independent samples did not, and
the reasons turned out to be three different things wearing one symptom — which
is the case for having this as a tool rather than a paragraph.

## What it measures, per product

Three things, because the verdict needs all three and any one alone misleads:

1. **The MF=3 MT set mapped to the product.** This is what a `kind='production'`
   row actually sums — every MT that `mt_to_residual` sends to that residual,
   minus the partials `redundant_partial_mts` skips. Imported from the ingest
   rather than restated, so a verdict is about the data and not about two
   opinions on redundancy.
2. **The MF=10 (MT, LFS) set.** Which MTs the evaluator chose to split, and at
   which levels. An evaluation splits the MTs it wants to; it is under no
   obligation to cover the same set MF=3 does.
3. **The restricted ratio curve.** MF=10's level sum against MF=3 *restricted to
   the MTs MF=10 covers*, across the overlapping energy range. This is the
   measurement that separates the causes.

## Why the per-MT comparison settles it

The product-level comparison conflates two aggregations: a `kind='production'`
row sums every MT mapped to the residual, while MF=10 splits only the MTs the
evaluator chose to. The pairing that means something is **MF=10 MT=X against
MF=3 MT=X**, each on its own overlap window, and that is what decides the
verdict. Pooling a product's levels into one grid truncates to the narrowest of
them — which is how the first draft of this script missed Er-147 entirely.

Run over the seven evaluations named in #368, it reproduces all nine outliers
and finds four more the sampling missed:

    jeff-4.0    In-113  -> Cd-111   mt-coverage          MF=3 also sums MT=105
    jeff-4.0    Cr-52   -> Ti-49    mt-coverage          MF=10 splits MT=5, unmapped
    endfb-8.1   Pm-145  -> Nd-142   mt-coverage          MF=3 also sums MT=107
    endfb-8.1   Mo-93   -> Zr-90    mt-coverage          MF=3 also sums MT=107
    endfb-8.1   Pt-193  -> Ir-190   mt-coverage          MF=10 splits MT=42, MF=3 has MT=33
    endfb-8.1   W-183   -> Ta-180   mt-coverage
    endfb-8.1   Pt-193  -> Ir-191   upstream             median ratio 432
    endfb-8.1   Pt-193  -> Ir-192   upstream             median ratio 806
    endfb-8.1   Pt-193  -> Os-190   upstream             median ratio 5.68
    endfb-8.1   W-183   -> Ta-182   upstream             median ratio 1.058
    tendl-2025  Er-147  -> Dy-145   split-without-total  109.1 b where MF=3 is 0
    tendl-2025  Er-147  -> Tb-143   outlier-point
    tendl-2025  Er-147  -> Dy-146   outlier-point

`mt-coverage` is not a failure: it is the sum rule being asked of quantities
that are not the same sum, and the restricted ratio is 1.000000 for every one of
them. `upstream` is — a split cannot exceed the total it splits.

Two verdicts exist because a ratio alone is blind twice over. `outlier-point`
catches a curve that tracks its MT in the median but not at its worst point.
`split-without-total` catches a split that is positive where its MT is zero,
which no ratio can express: Er-147's MF=10 MT=115 reads 109.1 barns at its
0.7235 MeV threshold where MF=3 MT=115 reads exactly zero.

`lfs_to_state` mis-ranking, the leading hypothesis when #368 was filed, explains
none of them and structurally cannot: renaming a level from `'m'` to `'m2'`
changes the key, not the value, so the total over states is identical either way.
The tool does not test for it because there is nothing to test.

## Why a script and not a test

It needs the network, and a network check in PR CI fails on a train and every
time an upstream institution has a bad afternoon — the same argument
`tests/test_library_registry.py` and `scripts/check_source_urls.py` make. The
offline invariant worth gating on is the *restricted* rule, and that belongs on
committed data once the rebuild has run; the unrestricted one would report a
false failure for every product in the first table above.

Usage:
    nix develop -c uv run python scripts/check_isomeric_sum_rule.py --library jeff-4.0
    nix develop -c uv run python scripts/check_isomeric_sum_rule.py \\
        --library endfb-8.1 --nuclide Pt-193 --verbose
    nix develop -c uv run python scripts/check_isomeric_sum_rule.py --library tendl-2025 --limit 40

Exit status is 1 if any product is classified `upstream`, 0 otherwise. A
`mt-coverage` verdict is not a failure: it is the sum rule being asked where its
precondition does not hold.
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_endf_libs import (  # noqa: E402
    LIBRARIES,
    fetch_endf_text,
    list_endf_files,
    mt_to_residual,
    parse_endf_filename,
    redundant_partial_mts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#: Ratio outside this band counts as disagreement. Deliberately loose: the
#: question is "are these the same quantity", and the failures worth reporting
#: are factors of 7 to 4e7, not percents. W-183's systematic 1.051 is the
#: closest real call and sits outside it on purpose.
TOLERANCE = 0.05

#: Verdicts worth printing without --verbose.
FLAGGED = frozenset({"upstream", "mt-coverage", "outlier-point", "split-without-total"})

#: A max ratio this far from 1 is reported even when the median agrees. TENDL's
#: Er-147 tracks its MT exactly except for one point 4.6e22 times too big; a
#: median-only check calls that healthy, which is how the first draft of this
#: script missed it.
SPIKE_FACTOR = 9.0

#: A split-without-total point below this (barns) is threshold noise, not a
#: finding. Er-147's is 109 b, five orders above it.
ORPHAN_FLOOR = 1e-3

#: A product needs at least this many overlapping points before a median ratio
#: means anything. Two points can agree by accident.
MIN_POINTS = 3


@dataclass
class ProductVerdict:
    """One product's three measurements and what they imply."""

    library: str
    filename: str
    target_z: int
    target_a: int
    product: tuple[int, int]
    mf3_mts: list[int]
    mf10_levels: list[tuple[int, int]]
    n_points: int = 0
    ratio_unrestricted: float = float("nan")
    ratio_restricted: float = float("nan")
    ratio_restricted_max: float = float("nan")
    orphan_points: int = 0
    orphan_max: float = 0.0
    worst_mt: int | None = None
    verdict: str = "ok"
    note: str = ""

    @property
    def mf10_mts(self) -> list[int]:
        return sorted({mt for mt, _lfs in self.mf10_levels})

    @property
    def only_in_mf3(self) -> list[int]:
        return sorted(set(self.mf3_mts) - set(self.mf10_mts))

    @property
    def ground_level_present(self) -> bool:
        return any(lfs == 0 for _mt, lfs in self.mf10_levels)


def _curve(tab) -> tuple[np.ndarray, np.ndarray] | None:  # noqa: ANN001 — endf Tabulated1D
    """Finite, strictly positive points of a TAB1, or None if there are none."""
    if tab is None:
        return None
    x = np.asarray(tab.x, dtype=float)
    y = np.asarray(tab.y, dtype=float)
    good = np.isfinite(y) & (y > 0)
    return (x[good], y[good]) if good.any() else None


def _ratio(grid: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> tuple[float, float]:
    """(median, max) of numerator/denominator wherever the denominator is positive."""
    live = denominator > 0
    if live.sum() < MIN_POINTS:
        return float("nan"), float("nan")
    r = numerator[live] / denominator[live]
    return float(np.median(r)), float(r.max())


def _compare(
    split: list[tuple[np.ndarray, np.ndarray]], total: tuple[np.ndarray, np.ndarray]
) -> tuple[int, float, float]:
    """(points, median ratio, max ratio) of a level sum against the curve it splits.

    Each comparison gets its **own** overlap window. Pooling every level of a
    product into one grid truncates to the narrowest of them, which is how the
    first version of this script missed TENDL Er-147's spike entirely: the bad
    point sits near the MT=115 threshold, below where MT=44's levels begin.
    """
    lo = max(max(c[0][0] for c in split), total[0][0])
    hi = min(min(c[0][-1] for c in split), total[0][-1])
    grid = np.unique(np.concatenate([c[0] for c in split] + [total[0]]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    if grid.size < MIN_POINTS:
        return 0, float("nan"), float("nan")
    num = sum(np.interp(grid, c[0], c[1], left=0.0, right=0.0) for c in split)
    den = np.interp(grid, total[0], total[1], left=0.0, right=0.0)
    live = den > 0
    if live.sum() < MIN_POINTS:
        return 0, float("nan"), float("nan")
    r = np.asarray(num)[live] / den[live]
    return int(live.sum()), float(np.median(r)), float(r.max())


def split_without_total(
    split: list[tuple[np.ndarray, np.ndarray]], total: tuple[np.ndarray, np.ndarray]
) -> tuple[int, float]:
    """(points, largest value) where the split is positive and its total is not.

    A ratio cannot see this — the denominator is zero, so the point is excluded
    from every median and every max. TENDL-2025's Er-147 MF=10 MT=115 reads
    **109.1 barns** at its 0.7235 MeV threshold while MF=3 MT=115 reads exactly
    zero there, and the next point down is 7.4e-20 b. A split that exists where
    the thing it splits does not is an inconsistency however small the ratio
    machinery says it is.
    """
    worst, count = 0.0, 0
    for e, y in split:
        below = e < total[0][0]
        above = e > total[0][-1]
        interp = np.interp(e, total[0], total[1], left=0.0, right=0.0)
        missing = (below | above | (interp <= 0)) & (y > 0)
        if missing.any():
            count += int(missing.sum())
            worst = max(worst, float(y[missing].max()))
    return count, worst


def analyse_material(material, target_z: int, target_a: int, projectile: str) -> list[ProductVerdict]:  # noqa: ANN001
    """Every product this evaluation describes in both MF=3 and MF=10.

    Importable so the MF=10-versus-MF=3 detector #368 asks for can reuse the
    measurement rather than growing a second one.
    """
    from fetch_endf_libs import PROJECTILE_ZA

    proj_z, proj_a = PROJECTILE_ZA[projectile]
    mf3_mts = {mt for (mf, mt) in material.section_data if mf == 3}
    skip = redundant_partial_mts(mf3_mts)

    # (1) which MF=3 MTs a production row for each residual draws on
    mf3_by_product: dict[tuple[int, int], dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for (mf, mt), section in material.section_data.items():
        if mf != 3 or mt in skip:
            continue
        residual = mt_to_residual(mt, target_z, target_a, proj_z, proj_a)
        if residual is None:
            continue
        curve = _curve(section.get("sigma"))
        if curve is not None:
            mf3_by_product.setdefault(residual, {})[mt] = curve

    # (2) which (MT, LFS) MF=10 splits, per product
    mf10_by_product: dict[tuple[int, int], dict[int, list[tuple[int, tuple[np.ndarray, np.ndarray]]]]] = {}
    for (mf, mt), section in material.section_data.items():
        if mf != 10:
            continue
        for level in section["levels"]:
            izap = int(level["IZAP"])
            if izap <= 0:
                continue
            curve = _curve(level.get("sigma"))
            if curve is not None:
                product = (izap // 1000, izap % 1000)
                mf10_by_product.setdefault(product, {}).setdefault(mt, []).append((int(level["LFS"]), curve))

    out: list[ProductVerdict] = []
    for product, by_mt in sorted(mf10_by_product.items()):
        contributions = mf3_by_product.get(product, {})
        v = ProductVerdict(
            library="",
            filename="",
            target_z=target_z,
            target_a=target_a,
            product=product,
            mf3_mts=sorted(contributions),
            mf10_levels=sorted((mt, lfs) for mt, lv in by_mt.items() for lfs, _c in lv),
        )
        if not contributions:
            v.verdict = "mf10-only"
            v.note = "MF=3 maps no MT to this product, so there is no total to compare against"
            out.append(v)
            continue

        # (3) the restricted comparison, per MT — MF=10 MT=X splits MF=3 MT=X,
        # and that pairing is the only one whose disagreement means anything.
        shared = sorted(set(by_mt) & set(contributions))
        worst_median, worst_max, points, worst_mt = 1.0, 1.0, 0, None
        for mt in shared:
            n, med, mx = _compare([c for _lfs, c in by_mt[mt]], contributions[mt])
            if not n:
                continue
            points += n
            if abs(med - 1.0) > abs(worst_median - 1.0):
                worst_median, worst_mt = med, mt
            worst_max = max(worst_max, mx)
        orphan_points, orphan_max = 0, 0.0
        for mt in shared:
            n, mx = split_without_total([c for _lfs, c in by_mt[mt]], contributions[mt])
            orphan_points += n
            orphan_max = max(orphan_max, mx)
        v.orphan_points, v.orphan_max = orphan_points, orphan_max

        v.n_points = points
        v.ratio_restricted, v.ratio_restricted_max = (
            (worst_median, worst_max) if points else (float("nan"), float("nan"))
        )
        v.worst_mt = worst_mt

        # The product-level comparison the sum rule is usually stated as: every
        # MF=10 level against every MF=3 MT mapped to the product.
        all_levels = [c for lv in by_mt.values() for _lfs, c in lv]
        n_all, med_all, _mx_all = _compare(all_levels, _pool(contributions.values()))
        v.ratio_unrestricted = med_all if n_all else float("nan")

        restricted_ok = points and abs(v.ratio_restricted - 1.0) <= TOLERANCE
        spike = points and abs(v.ratio_restricted_max - 1.0) > SPIKE_FACTOR
        if not points:
            v.verdict = "mt-coverage"
            v.note = (
                f"MF=10 splits MT {sorted(by_mt)}, which MF=3 never maps to this product, "
                "so there is no shared MT to compare"
            )
        elif not restricted_ok:
            v.verdict = "upstream"
            v.note = (
                f"MT={v.worst_mt}: the split disagrees with the very MT it splits "
                f"(median ratio {v.ratio_restricted:.4g})"
            )
        elif orphan_points and orphan_max > ORPHAN_FLOOR:
            v.verdict = "split-without-total"
            v.note = (
                f"{orphan_points} point(s) where the split is positive and the MT it splits "
                f"is not, the largest {orphan_max * 1e3:.4g} mb — a ratio cannot see these"
            )
        elif spike:
            v.verdict = "outlier-point"
            v.note = (
                f"the split tracks its MT (median {v.ratio_restricted:.6g}) but reaches "
                f"{v.ratio_restricted_max:.4g} somewhere — an isolated bad datum, not a "
                "systematic disagreement"
            )
        elif not np.isnan(v.ratio_unrestricted) and abs(v.ratio_unrestricted - 1.0) > TOLERANCE:
            v.verdict = "mt-coverage"
            v.note = (
                f"MF=3 also sums MT {v.only_in_mf3}, which MF=10 does not split; "
                "restricted to the shared MTs the rule holds"
            )
        else:
            v.verdict = "ok"
        out.append(v)
    return out


def _pool(curves) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """Sum several curves onto the union of their grids."""
    curves = list(curves)
    grid = np.unique(np.concatenate([c[0] for c in curves]))
    return grid, sum(np.interp(grid, c[0], c[1], left=0.0, right=0.0) for c in curves)


def analyse_file(lib_key: str, sublib: str, filename: str, session: requests.Session) -> list[ProductVerdict]:
    import endf

    parsed = parse_endf_filename(filename)
    if parsed is None:
        logger.warning("cannot parse filename: %s", filename)
        return []
    target_z, target_a, _marker = parsed
    text = fetch_endf_text(LIBRARIES[lib_key], sublib, filename, session)
    if text is None:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            material = endf.Material(io.StringIO(text))
        except Exception as e:  # noqa: BLE001 — a bad tape must not stop the sweep
            logger.warning("cannot parse %s: %s", filename, e)
            return []
    verdicts = analyse_material(material, target_z, target_a, sublib)
    for v in verdicts:
        v.library, v.filename = lib_key, filename
    return verdicts


def _report(v: ProductVerdict, verbose: bool) -> None:
    z, a = v.product
    print(f"  {v.filename:28s} product Z={z:<3d} A={a:<4d}  {v.verdict}")
    if v.note:
        print(f"      {v.note}")
    if verbose or v.verdict in FLAGGED:
        print(f"      MF=3  MTs : {v.mf3_mts}")
        print(f"      MF=10     : {v.mf10_levels}  ground level present: {v.ground_level_present}")
        if v.n_points:
            print(
                f"      ratio     : unrestricted {v.ratio_unrestricted:.6g} · "
                f"restricted median {v.ratio_restricted:.6g} max {v.ratio_restricted_max:.6g} "
                f"over {v.n_points} points"
            )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--library", required=True, choices=sorted(LIBRARIES), help="library to check")
    ap.add_argument("--sublibrary", default="n", help="sublibrary code (default: n)")
    ap.add_argument("--nuclide", help="only files whose name contains this, e.g. 'Pt-193'")
    ap.add_argument("--limit", type=int, help="stop after this many source files")
    ap.add_argument("--verbose", action="store_true", help="print the measurements for every product")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    lib = LIBRARIES[args.library]
    if args.sublibrary not in lib.sublibraries:
        ap.error(f"{args.library} has no sublibrary {args.sublibrary!r}")

    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (isomeric sum-rule check)"
    filenames = list_endf_files(lib, args.sublibrary, session)
    if args.nuclide:
        pattern = re.compile(re.escape(args.nuclide), re.IGNORECASE)
        filenames = [f for f in filenames if pattern.search(f)]
    if args.limit:
        filenames = filenames[: args.limit]
    if not filenames:
        ap.error("no source files matched")

    logger.info("checking %d file(s) from %s/%s", len(filenames), args.library, args.sublibrary)
    tally: dict[str, int] = {}
    flagged: list[ProductVerdict] = []
    for i, filename in enumerate(filenames, 1):
        if i % 50 == 0:
            logger.info("  %d/%d ...", i, len(filenames))
        for v in analyse_file(args.library, args.sublibrary, filename, session):
            tally[v.verdict] = tally.get(v.verdict, 0) + 1
            if v.verdict in FLAGGED or args.verbose:
                flagged.append(v)

    if flagged:
        print(f"\n{args.library}/{args.sublibrary}:")
        for v in sorted(flagged, key=lambda v: (v.verdict != "upstream", v.filename)):
            _report(v, args.verbose)

    total = sum(tally.values())
    print(f"\n{total} product(s) described in both MF=3 and MF=10:")
    for verdict in ("ok", "mt-coverage", "outlier-point", "split-without-total", "upstream", "mf10-only", "no-overlap"):
        if verdict in tally:
            print(f"  {verdict:12s} {tally[verdict]:>5d}")
    upstream = tally.get("upstream", 0)
    if upstream:
        print(
            f"\n{upstream} product(s) whose split disagrees with the MT it splits. "
            "That is an inconsistency in the evaluation, not in this repository — "
            "record it (#368) rather than correcting it here."
        )
    sys.exit(1 if upstream else 0)


if __name__ == "__main__":
    main()
