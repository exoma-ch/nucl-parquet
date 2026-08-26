"""Regression: naturally-occurring targets we ship must not silently vanish.

Issue #335: the tendl-2023-iso rebuild silently lost several natural targets
(p+Li-6, p+Li-7, p+Be-9, p+C-13, d+Li-6, d+Li-7, t+Li-6, h+Li-6). Nothing in
the suite noticed — the surviving files were internally consistent and the
existing coverage test only checks that each *projectile* has *some* file, not
that every natural isotope is represented.

The class of bug this guards against is a re-ingest, re-cut, or upstream drop
that quietly removes evaluations for isotopes we previously shipped. The check
is intentionally driven by iterating (natural isotope × claimed projectile) and
LOOKING FOR the file — never by globbing what's on disk. A globbing check would
silently skip a whole missing element file (e.g. p_Li.parquet absent entirely),
which is exactly the blind spot that let #335 hide.

Scope note: only libraries that empirically claim broad natural coverage
(>=~95% of natural isotopes for the named projectile) are checked strictly.
Partial/special-purpose libraries — dosimetry, activation, medical isotopes,
photonuclear, deuteron-only pilots, ENDF/B-VIII.1 charged-particle subset —
are excluded by design: running a natural-coverage check on them would produce
hundreds of "gaps" that are simply not in the library's scope. See
COMPREHENSIVE_SUBLIBRARIES below.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pytest

# ---------------------------------------------------------------------------
# Scope: which (library, projectile) sublibraries claim near-full natural
# coverage. Determined empirically from the current data tree — each entry
# below covers >=97% of the 287 natural isotopes for the named projectile.
#
# Every other cross-section (library, projectile) ships a deliberately partial
# subset for which a natural-coverage check would be meaningless:
#   * cendl-3.2 / fendl-3.2       — regional / fusion-focused subset (~65%)
#   * brond-3.1                   — Russian evaluations, ~90% but with real
#                                   gaps around As, Ne, Os, Pt, Hg, Tl that
#                                   are upstream reality, not our bug
#   * iaea-medical                — 4% coverage, medical isotopes only
#   * iaea-pd-2019                — photonuclear subset (~67%)
#   * irdff-2                     — dosimetry file set (~14%)
#   * jendl-ad-2017               — activation/dosimetry (~70%)
#   * jendl-deu-2020              — deuteron-only pilot (~2%)
#   * endfb-8.1 charged particle  — very small d/t/h/a set (2%), p at 17%
#   * jendl-5 p/d/a               — partial charged-particle coverage
#   * jeff-4.0 p                  — partial proton coverage (35%)
# hi-xs / hi-xs-prod are heavy-ion libraries with data_type != "cross_sections"
# so they never enter this check.
COMPREHENSIVE_SUBLIBRARIES: frozenset[tuple[str, str]] = frozenset(
    {
        ("tendl-2023-iso", "p"),
        ("tendl-2023-iso", "d"),
        ("tendl-2023-iso", "t"),
        ("tendl-2023-iso", "h"),
        ("tendl-2023-iso", "a"),
        ("tendl-2025", "n"),
        ("tendl-2025", "p"),
        ("tendl-2025", "d"),
        ("tendl-2025", "t"),
        ("tendl-2025", "h"),
        ("tendl-2025", "a"),
        ("endfb-8.0", "n"),
        ("jeff-4.0", "n"),
        ("jendl-5", "n"),
    }
)


# ---------------------------------------------------------------------------
# KNOWN_GAPS — natural targets that a comprehensive sublibrary does NOT ship
# and where absence is upstream reality, not a nucl-parquet ingest bug.
#
# Each entry is (projectile, symbol, A) -> short reason. Adding an entry here
# silently exempts a real drop, so every reason must be *verified* against the
# upstream directory — not "probably fine".
#
# Seven of the eight tendl-2023-iso dropouts found in #335 — p+Be-9, p+C-13,
# p+Li-7, d+Li-6, d+Li-7, t+Li-6, h+Li-6 — are FIXED (backfilled from the IAEA
# NDS TENDL-2023 tapes by scripts/backfill_xs_nuclides.py) and so are absent
# from this table: they are covered now, and must stay covered.
#
# The eighth, p+Li-6, is listed below. It is not an ingest failure but a real
# property of the evaluation — see its reason string.
#
# The table is deliberately short. When this check first ran against the tree it
# was written on it also flagged jeff-4.0 n+Ta-180 and jendl-5 n+Ce-140 /
# n+Ho-165; all three have since been restored by #334 and #336 and their
# entries are gone, because an exemption for something that is no longer absent
# is a licence for the next regression to hide behind.
_TENDL_H_HE = [("H", 1), ("H", 2), ("He", 3), ("He", 4)]

_TENDL_HHE_REASON = (
    "TENDL charged-particle sublibraries do not evaluate H/He targets (upstream reality; applies to every projectile)"
)


def _tendl_hhe_gaps(projectiles: list[str]) -> dict[tuple[str, str, int], str]:
    return {(proj, sym, a): _TENDL_HHE_REASON for proj in projectiles for sym, a in _TENDL_H_HE}


# n + He-4 is an inert scatterer in JEFF-4.0 and JENDL-5, and this was verified
# against the upstream tapes rather than assumed. `n_002-He-4_0228` in both IAEA
# directories carries exactly two MF=3 sections — MT=1 (total) and MT=2
# (elastic), both peaking at ~7.63 b — and no MF=10 at all. Neither MT names a
# residual, so there is no transmutation product to tabulate and zero production
# rows is the right answer, not a dropout.
#
# Worth spelling out because these shards *used* to carry He-4 rows and their
# disappearance is the #334 fix, not a #335-style silent drop: 152 rows in
# jeff-4.0 and 398 in jendl-5, every one of them a single group at residual
# (Z=2, A=5) peaking at 7635 mb. He-5 is unbound. That was MT=2 elastic filed as
# (n,gamma) — precisely the bug #334 removed.
#
# These entries clean themselves up: once the re-ingest lands (#345), MT=1 and
# MT=2 become `kind='channel'` rows under #347, target_A=4 is present again, and
# `test_known_gaps_still_correspond_to_real_absences` says so.
_HE4_INERT = (
    "verified against n_002-He-4_0228: upstream tape has only MT=1/MT=2, no residual and no MF=10. "
    "The rows that used to be here were the #334 elastic-as-capture artefact at unbound He-5"
)


KNOWN_GAPS: dict[str, dict[tuple[str, str, int], str]] = {
    "tendl-2023-iso": {
        **_tendl_hhe_gaps(["p", "d", "t", "h", "a"]),
        # Verified: IAEA NDS TENDL-2023 t/ directory contains no
        # t_033-As-* files — the triton sublibrary has no As evaluation.
        ("t", "As", 75): "IAEA NDS TENDL-2023 t/ dir ships no t_033-As-* files (upstream gap)",
        # Verified against the tape p_003-Li-6_0325: TENDL-2023's p+Li-6
        # evaluation has exactly two MF=3 sections, MT=2 (elastic) and MT=750,
        # i.e. ⁶Li(p,³He)⁴He. Both products of that channel are standard
        # ejectiles, so the evaluation contains no residual nucleus to tabulate
        # under this library's residual-production convention. Nothing was
        # dropped — there is genuinely nothing to carry.
        ("p", "Li", 6): "TENDL-2023 p+Li-6 has only elastic + (p,3He)a; no heavy residual exists",
    },
    "tendl-2025": {
        **_tendl_hhe_gaps(["n", "p", "d", "t", "h", "a"]),
    },
    "endfb-8.0": {
        # Same physical fact as the two entries below, reached by a different
        # builder: our endfb-8.0 shard is transmutation-only (capture + threshold
        # reactions; elastic/inelastic/fission excluded), and the
        # OpenMC-processed ENDF/B-VIII.0 HDF5 ships no threshold transmutation
        # product for He-4 — (n,g)He-5 is unbound. Upstream reality.
        ("n", "He", 4): "endfb-8.0 (transmutation-only) has no He-4 channels in the OpenMC source",
    },
    "jeff-4.0": {
        ("n", "He", 4): _HE4_INERT,
    },
    "jendl-5": {
        ("n", "He", 4): _HE4_INERT,
    },
}


# ---------------------------------------------------------------------------
# Coverage helper — one query per file, cached; ~500 files total, sub-second.
_CON = duckdb.connect()
_TARGET_A_CACHE: dict[Path, frozenset[int] | None] = {}


def _target_As(path: Path) -> frozenset[int] | None:
    """Return {target_A} for a per-element xs shard, or None if the file is absent."""
    if path not in _TARGET_A_CACHE:
        if not path.exists():
            _TARGET_A_CACHE[path] = None
        else:
            rows = _CON.sql(f"SELECT DISTINCT target_A FROM read_parquet('{path}')").fetchall()
            _TARGET_A_CACHE[path] = frozenset(r[0] for r in rows)
    return _TARGET_A_CACHE[path]


def _natural_isotopes(data_dir: Path) -> list[tuple[int, int, str, float]]:
    """Return (Z, A, symbol, abundance) for every natural isotope."""
    return _CON.sql(
        f"SELECT Z, A, symbol, abundance "
        f"FROM read_parquet('{data_dir}/meta/abundances.parquet') "
        f"WHERE abundance > 0 "
        f"ORDER BY Z, A"
    ).fetchall()


def natural_target_gaps(data_dir: Path) -> list[tuple[str, str, str, int, int, float, str]]:
    """Walk (comprehensive sublibrary × natural isotope) and return uncovered rows.

    A row is uncovered if the expected per-element file is missing (cause
    "no-file") or exists but ships no rows for that target_A ("no-rows").
    Never driven by globbing — the file lookup is the whole point.

    Returns (library, projectile, symbol, Z, A, abundance, cause) tuples.
    """
    catalog = json.loads((data_dir / "catalog.json").read_text())
    naturals = _natural_isotopes(data_dir)
    gaps: list[tuple[str, str, str, int, int, float, str]] = []
    for lib_key, info in catalog["libraries"].items():
        if info.get("data_type") != "cross_sections":
            continue
        rel_path = info.get("path", "")
        # Layout must be per-element shards {proj}_{Sym}.parquet under <lib>/xs/.
        if not rel_path.endswith("/xs/"):
            continue
        xs_dir = data_dir / rel_path
        if not xs_dir.exists():
            continue
        for proj in info.get("projectiles", []):
            if (lib_key, proj) not in COMPREHENSIVE_SUBLIBRARIES:
                continue
            for z, a, sym, ab in naturals:
                fp = xs_dir / f"{proj}_{sym}.parquet"
                avail = _target_As(fp)
                if avail is None:
                    gaps.append((lib_key, proj, sym, z, a, ab, "no-file"))
                elif a not in avail:
                    gaps.append((lib_key, proj, sym, z, a, ab, "no-rows"))
    return gaps


# ---------------------------------------------------------------------------
# The fix itself, asserted positively
#
# `test_no_undocumented_natural_target_gaps` is a negative check: it passes when
# it finds nothing. If `natural_target_gaps` ever stops looking — a renamed
# column, a changed layout, an empty `naturals` list — it passes for the wrong
# reason and #335 comes back unobserved. So the seven restored targets are also
# named here, with row counts and one physical anchor.

#: (projectile, symbol, A, minimum rows) restored from the IAEA NDS TENDL-2023
#: tapes by scripts/backfill_xs_nuclides.py. Row counts are floors, not equalities:
#: a re-run against a refreshed upstream tape may legitimately carry more points,
#: and this test's subject is "did the target come back", not "is the grid frozen".
BACKFILLED_TENDL_2023: tuple[tuple[str, str, int, int], ...] = (
    ("p", "Be", 9, 32),  # 100 % of natural Be — the hyrr#668 root cause
    ("p", "C", 13, 95),
    ("p", "Li", 7, 203),  # p_Li.parquet did not exist at all
    ("d", "Li", 6, 122),  # d_Li.parquet did not exist at all
    ("d", "Li", 7, 13),
    ("t", "Li", 6, 27),
    ("h", "Li", 6, 57),
)


@pytest.mark.data
def test_the_backfilled_targets_are_present_with_data(data_dir_path: Path) -> None:
    """The seven #335 dropouts must ship rows, not merely a row count of zero.

    Two of them (`p_Li`, `d_Li`) had no shard on disk at all, which is why the
    original repro in #335 — which globbed the directory — could not see them.
    """
    xs = data_dir_path / "tendl-2023-iso" / "xs"
    problems: list[str] = []
    for proj, sym, a, floor in BACKFILLED_TENDL_2023:
        path = xs / f"{proj}_{sym}.parquet"
        if not path.exists():
            problems.append(f"{proj}+{sym}-{a}: {path.name} does not exist")
            continue
        rows = _CON.sql(
            f"SELECT count(*), max(xs_mb) FROM read_parquet('{path}') WHERE target_A = {a} AND xs_mb > 0"
        ).fetchone()
        n, peak = rows
        if n < floor:
            problems.append(f"{proj}+{sym}-{a}: {n} rows with xs_mb>0, expected at least {floor}")
        elif peak is None or peak <= 0:
            problems.append(f"{proj}+{sym}-{a}: {n} rows but no positive cross section")
    assert not problems, "backfilled targets are missing or empty (#335):\n  " + "\n  ".join(problems)


@pytest.mark.data
def test_li7_pn_opens_at_its_textbook_threshold(data_dir_path: Path) -> None:
    """⁷Li(p,n)⁷Be must start at 1.88 MeV — the number that proves the physics.

    A row count says the target came back; it does not say the *right* rows came
    back. This one does, against a value derived outside this repository: the
    AME2020 mass excesses give Q = -1.6442 MeV and hence a lab threshold of
    1.8804 MeV, and the shipped curve's first point is 1.881 MeV.

    It is also the check that catches the MT=50 half of this fix regressing. For
    a charged projectile MT=50 is (z,n₀), the ground-state transition and the
    dominant (p,n) channel at threshold; with the band starting at 51 the curve
    would open ~250 keV too high, still look like a plausible cross section, and
    make a Be or Li converter foil wrong in exactly the region it is used in.
    """
    path = data_dir_path / "tendl-2023-iso" / "xs" / "p_Li.parquet"
    assert path.exists(), "p_Li.parquet is missing — the #335 backfill did not run"
    first, peak = _CON.sql(
        f"SELECT min(energy_MeV), max(xs_mb) FROM read_parquet('{path}') "
        f"WHERE target_A = 7 AND residual_Z = 4 AND residual_A = 7 AND xs_mb > 0"
    ).fetchone()
    assert first is not None, "no ⁷Li(p,n)⁷Be rows at all"
    assert 1.87 <= first <= 1.90, f"⁷Li(p,n)⁷Be opens at {first} MeV, expected ~1.881 (AME2020 gives 1.8804)"
    assert peak > 100, f"⁷Li(p,n)⁷Be peaks at only {peak} mb — the curve is there but the magnitude is wrong"


# ---------------------------------------------------------------------------
# Tests
@pytest.mark.data
def test_no_undocumented_natural_target_gaps(data_dir_path: Path) -> None:
    """Every natural isotope must be a target of every comprehensive sublibrary,
    or be explicitly listed in KNOWN_GAPS with a verified reason.

    Failure mode this catches: an ingest / re-cut / upstream refresh silently
    drops one or more natural targets from a broadly-scoped library (issue #335).
    """
    t0 = time.perf_counter()
    gaps = natural_target_gaps(data_dir_path)
    undocumented: list[str] = []
    for lib, proj, sym, _z, a, ab, cause in gaps:
        if (proj, sym, a) not in KNOWN_GAPS.get(lib, {}):
            undocumented.append(f"{lib} / {proj} / {sym}-{a} (abundance={ab:.4g}) [{cause}]")
    elapsed = time.perf_counter() - t0
    assert not undocumented, (
        f"Natural targets are missing without a KNOWN_GAPS entry (see issue #335).\n"
        f"Either restore the ingested files or add an entry to KNOWN_GAPS with a\n"
        f"verified upstream reason. Ran in {elapsed:.2f}s.\n  " + "\n  ".join(undocumented)
    )


@pytest.mark.data
def test_known_gaps_still_correspond_to_real_absences(data_dir_path: Path) -> None:
    """Every KNOWN_GAPS entry must still describe an actual absent isotope.

    Once an ingest bug is fixed and the isotope reappears, its KNOWN_GAPS entry
    would silently mask a future re-regression. This test flags stale entries
    so they get removed at the same time as the fix.
    """
    catalog = json.loads((data_dir_path / "catalog.json").read_text())
    stale: list[str] = []
    for lib_key, entries in KNOWN_GAPS.items():
        lib_info = catalog["libraries"].get(lib_key)
        if lib_info is None:
            stale.append(f"{lib_key}: library not in catalog")
            continue
        xs_dir = data_dir_path / lib_info["path"]
        for (proj, sym, a), _reason in entries.items():
            if (lib_key, proj) not in COMPREHENSIVE_SUBLIBRARIES:
                stale.append(f"{lib_key} / {proj} / {sym}-{a}: sublibrary not in COMPREHENSIVE_SUBLIBRARIES")
                continue
            fp = xs_dir / f"{proj}_{sym}.parquet"
            avail = _target_As(fp)
            if avail is not None and a in avail:
                stale.append(f"{lib_key} / {proj} / {sym}-{a}: isotope IS present — remove the KNOWN_GAPS entry")
    assert not stale, "stale KNOWN_GAPS entries:\n  " + "\n  ".join(stale)


@pytest.mark.data
def test_the_gap_walk_can_actually_find_both_kinds_of_gap(data_dir_path: Path, tmp_path: Path) -> None:
    """The guard on the guard: reproduce #335 synthetically and demand it fires.

    `test_no_undocumented_natural_target_gaps` passes by finding nothing, so a
    walker that stopped looking is indistinguishable from a clean tree. Build a
    tree with the two failure shapes #335 actually had — a whole element shard
    missing (`no-file`, which is how `p_Li` hid from the globbing repro) and a
    shard present but silent on one isotope (`no-rows`, how `p+Be-9` hid) — and
    assert both are reported.
    """
    lib, proj = "tendl-2023-iso", "p"
    xs_src = data_dir_path / lib / "xs"
    xs_dst = tmp_path / lib / "xs"
    xs_dst.mkdir(parents=True)
    for shard in xs_src.glob(f"{proj}_*.parquet"):
        # Li is simply never linked -> 'no-file'. Be is linked but rewritten
        # without Be-9 -> 'no-rows'.
        if shard.stem == f"{proj}_Li":
            continue
        if shard.stem == f"{proj}_Be":
            _CON.sql(
                f"COPY (SELECT * FROM read_parquet('{shard}') WHERE target_A <> 9) "
                f"TO '{xs_dst / shard.name}' (FORMAT PARQUET)"
            )
            continue
        (xs_dst / shard.name).symlink_to(shard)

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "abundances.parquet").symlink_to(data_dir_path / "meta" / "abundances.parquet")
    catalog = json.loads((data_dir_path / "catalog.json").read_text())
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "libraries": {
                    lib: {**catalog["libraries"][lib], "projectiles": [proj], "path": f"{lib}/xs/"},
                }
            }
        )
    )

    _TARGET_A_CACHE.clear()
    try:
        found = {(sym, a, cause) for _lib, _proj, sym, _z, a, _ab, cause in natural_target_gaps(tmp_path)}
    finally:
        _TARGET_A_CACHE.clear()

    assert ("Be", 9, "no-rows") in found, "a shard that ships no rows for one isotope was not reported"
    assert ("Li", 7, "no-file") in found, "a missing element shard was not reported — the #335 blind spot is back"
    assert ("C", 13, "no-rows") not in found, "C-13 is present in the copied tree and must not be reported"


@pytest.mark.data
def test_comprehensive_sublibraries_are_actually_comprehensive(data_dir_path: Path) -> None:
    """Every entry in COMPREHENSIVE_SUBLIBRARIES must genuinely cover a large
    fraction of natural isotopes. If someone adds a partial sublibrary here by
    mistake, the strict natural-target check would flood with false positives
    (or, worse, get silenced with a bulk KNOWN_GAPS dump). Guard the invariant.
    """
    catalog = json.loads((data_dir_path / "catalog.json").read_text())
    naturals = _natural_isotopes(data_dir_path)
    total = len(naturals)
    problems: list[str] = []
    for lib_key, proj in COMPREHENSIVE_SUBLIBRARIES:
        info = catalog["libraries"].get(lib_key)
        assert info is not None, f"{lib_key} not in catalog"
        assert proj in info["projectiles"], f"{lib_key} does not claim projectile {proj}"
        xs_dir = data_dir_path / info["path"]
        covered = 0
        for _z, a, sym, _ab in naturals:
            avail = _target_As(xs_dir / f"{proj}_{sym}.parquet")
            if avail is not None and a in avail:
                covered += 1
        pct = 100.0 * covered / total
        if pct < 90.0:
            problems.append(f"{lib_key} / {proj}: only {covered}/{total} ({pct:.1f}%) natural targets covered")
    assert not problems, "COMPREHENSIVE_SUBLIBRARIES contains a partial sublibrary:\n  " + "\n  ".join(problems)
