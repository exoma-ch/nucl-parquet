"""Download and parse evaluated nuclear data libraries into Parquet.

Fetches ENDF-6 format files from the IAEA NDS mirror, parses cross-sections
using the `endf` package, and converts to nucl-parquet Parquet format.

Supports all major evaluated libraries: ENDF/B-VIII.1, JEFF-4.0, JENDL-5,
TENDL-2025, CENDL-3.2, BROND-3.1, FENDL-3.2.

Usage:
    # Fetch a single library (neutron sub-library):
    python scripts/fetch_endf_libs.py --library endfb-8.1 --sublibrary n

    # Fetch all neutron libraries:
    python scripts/fetch_endf_libs.py --sublibrary n --all

    # Fetch proton sub-library for a specific library:
    python scripts/fetch_endf_libs.py --library jendl-5 --sublibrary p

    # List available libraries:
    python scripts/fetch_endf_libs.py --list
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _canonical import canonical_frame, element_stem, parse_stem  # noqa: E402
from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))  # so `nucl_parquet` imports from the checkout

from nucl_parquet.builder_stamp import write_builder_stamp  # noqa: E402
from nucl_parquet.state_vocabulary import (  # noqa: E402
    ENDF_TARGET_MARKERS,
    SUM,
    isomer_state,
    target_state_for_natural_element,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

IAEA_MIRROR = "https://nds.iaea.org/public/download-endf"
COMPRESSION = "zstd"

#: How many tape names the "yielded no rows" report spells out before it stops.
#: A sublibrary is up to ~2300 tapes and a wholesale failure would otherwise
#: emit one unreadable multi-kilobyte line. The *count* is always exact and the
#: remainder is stated — a truncation nobody is told about is the same class of
#: bug as the silent skip this report exists to expose.
_MAX_LISTED_TAPES = 40

# ---------------------------------------------------------------------------
# Library registry
# ---------------------------------------------------------------------------


@dataclass
class LibraryDef:
    """Definition of an evaluated nuclear data library."""

    key: str  # Our short identifier (used in directory names)
    name: str  # Display name
    iaea_path: str  # Path on IAEA mirror
    description: str
    source_url: str
    sublibraries: dict[str, str]  # sublibrary code -> IAEA subdirectory name


LIBRARIES: dict[str, LibraryDef] = {
    "endfb-8.1": LibraryDef(
        key="endfb-8.1",
        name="ENDF/B-VIII.1",
        iaea_path="ENDF-B-VIII.1",
        description="US Evaluated Nuclear Data File (NNDC/BNL)",
        source_url="https://www.nndc.bnl.gov/endf-b8.1/",
        sublibraries={"n": "n", "p": "p", "d": "d", "t": "t", "h": "he3", "a": "he4"},
    ),
    "jeff-4.0": LibraryDef(
        key="jeff-4.0",
        name="JEFF-4.0",
        iaea_path="JEFF-4.0",
        description="Joint Evaluated Fission and Fusion File (NEA)",
        source_url="https://www.oecd-nea.org/dbdata/jeff/",
        sublibraries={"n": "n", "p": "p"},
    ),
    "jendl-5": LibraryDef(
        key="jendl-5",
        name="JENDL-5",
        iaea_path="JENDL-5",
        description="Japanese Evaluated Nuclear Data Library (JAEA)",
        source_url="https://wwwndc.jaea.go.jp/jendl/j5/j5.html",
        sublibraries={"n": "n", "p": "p", "d": "d", "a": "he4"},
    ),
    "tendl-2025": LibraryDef(
        key="tendl-2025",
        name="TENDL-2025",
        iaea_path="TENDL-2025",
        description="TALYS Evaluated Nuclear Data Library (PSI)",
        source_url="https://tendl.web.psi.ch/",
        sublibraries={"n": "n", "p": "p", "d": "d", "t": "t", "h": "he3", "a": "he4"},
    ),
    "cendl-3.2": LibraryDef(
        key="cendl-3.2",
        name="CENDL-3.2",
        iaea_path="CENDL-3.2",
        description="Chinese Evaluated Nuclear Data Library (CIAE)",
        source_url="http://www.nuclear.csdb.cn/",
        sublibraries={"n": "n"},
    ),
    "brond-3.1": LibraryDef(
        key="brond-3.1",
        name="BROND-3.1",
        iaea_path="BROND-3.1",
        description="Russian Evaluated Nuclear Data Library (IPPE)",
        source_url="https://vant.ippe.ru/",
        sublibraries={"n": "n"},
    ),
    "fendl-3.2": LibraryDef(
        key="fendl-3.2",
        name="FENDL-3.2",
        iaea_path="FENDL-3.2c",
        description="Fusion Evaluated Nuclear Data Library (IAEA)",
        source_url="https://www-nds.iaea.org/fendl/",
        sublibraries={"n": "n"},
    ),
    "irdff-2": LibraryDef(
        key="irdff-2",
        name="IRDFF-II",
        iaea_path="IRDFF-II",
        description="International Reactor Dosimetry and Fusion File (IAEA)",
        source_url="https://www-nds.iaea.org/IRDFF/",
        sublibraries={"n": "n"},
    ),
    "iaea-medical": LibraryDef(
        key="iaea-medical",
        name="IAEA-Medical",
        iaea_path="IAEA-Medical",
        description="Medical isotope production cross-sections (IAEA)",
        source_url="https://www-nds.iaea.org/medical/",
        # No neutron sublibrary: IAEA-Medical/n/ is a 404 on the mirror, and
        # never was anything else. `data/catalog.json` has said `['p','d']`
        # since #321 and `data/iaea-medical/xs/` holds only d_* and p_* — the
        # registry was the last thing still claiming neutrons (#356).
        sublibraries={"p": "p", "d": "d", "h": "he3", "a": "he4"},
    ),
}

#: Sublibraries the registry can fetch but that this repo does not ship, and
#: why. Every entry in `LIBRARIES` must either appear in
#: `catalog.json::projectiles` or be listed here — `tests/test_library_registry.py`
#: enforces that, offline.
#:
#: The point is that "declared, and deliberately not shipped" and "declared by
#: mistake" look identical in a diff. iaea-medical's neutron sublibrary sat in
#: the registry for as long as it did because nothing could tell them apart.
UNSHIPPED_SUBLIBRARIES: dict[tuple[str, str], str] = {
    ("endfb-8.1", "n"): (
        "Retired in #263. The raw MF=3 tabulation omits the resonance region "
        "entirely, so neutron ships as endfb-8.0 (NJOY-processed pointwise) "
        "instead. Still served by the mirror and still fetchable on request."
    ),
    ("iaea-medical", "h"): "Served by the mirror (2 evaluations), never ingested.",
    ("iaea-medical", "a"): "Served by the mirror (3 evaluations), never ingested.",
}


# ---------------------------------------------------------------------------
# Element data
# ---------------------------------------------------------------------------

# The 99-entry element-symbol table that used to live here is gone: file stems
# come from `_canonical.element_stem`, which is the same table
# `migrate_xs_schema.py` parses them back with. Two copies meant two chances to
# disagree about what `Z61` is called.

# Light-particle (Z, A), for mass/charge balance.
#
# Serves both roles the balance needs: the sublibrary code names the *incoming*
# projectile, and the same symbols name the particles that *leave* in
# MT_EMITTED_PARTICLES below. One table, so "d" cannot be a deuteron on the way
# in and something else on the way out.
PARTICLE_ZA: dict[str, tuple[int, int]] = {
    "g": (0, 0),  # γ — carries neither charge nor mass number
    "n": (0, 1),
    "p": (1, 1),
    "d": (1, 2),
    "t": (1, 3),
    "h": (2, 3),  # ³He
    "a": (2, 4),  # α
}

#: Sublibrary code -> projectile (Z, A). A view of PARTICLE_ZA restricted to the
#: symbols that can be an incident particle (γ is not a sublibrary here).
PROJECTILE_ZA: dict[str, tuple[int, int]] = {code: PARTICLE_ZA[code] for code in ("n", "p", "d", "t", "h", "a")}


# ---------------------------------------------------------------------------
# MT number -> residual product mapping
# ---------------------------------------------------------------------------

# What leaves, named as particles rather than as a (Z, A) pair.
#
# The residual is what is left over:
#     residual_Z = target_Z + proj_Z - emitted_Z
#     residual_A = target_A + proj_A - emitted_A
#
# This table used to hold the (Z, A) sums directly, hand-written, with the
# particle list beside them in a comment. Thirteen of ~30 entries had the wrong
# sum, and every affected row was filed under the wrong nuclide — not dropped,
# *misattributed*, so the row looked fine and the number was plausible (#351).
# MT=44 was commented `(x,n2p): n + 2p`, which is (2, 3), and coded `(2, 6)`;
# MT=111 `(x,2p)` was coded `(1, 2)`, which is a deuteron; MT=109 `(x,3α)` was
# coded `(4, 11)` where three alphas are (6, 12). The comment and the arithmetic
# were two spellings of one fact, and they disagreed for years because nothing
# could compare them.
#
# So there is one spelling now. The particle list *is* the entry, and the (Z, A)
# sum below is derived from it — a comment can no longer contradict the code
# because the comment is gone. `tests/test_mt_residuals.py` checks these lists
# two ways: against `endf.reaction.REACTION_NAME`, a third-party transcription
# of the ENDF-102 reaction names, and against MF=10's `IZAP` product identifier
# recorded from real evaluations.
#
# MT numbers with no single residual product are deliberately absent — see
# NO_RESIDUAL_MTS. Note in particular that MT=5 ("anything") names a *different*
# product in every evaluation that uses it and can never be tabulated here.
MT_EMITTED_PARTICLES: dict[int, tuple[str, ...]] = {
    4: ("n",),
    11: ("n", "n", "d"),
    16: ("n",) * 2,
    17: ("n",) * 3,
    22: ("n", "a"),
    23: ("n", "a", "a", "a"),
    24: ("n", "n", "a"),
    25: ("n", "n", "n", "a"),
    28: ("n", "p"),
    29: ("n", "a", "a"),
    30: ("n", "n", "a", "a"),
    32: ("n", "d"),
    33: ("n", "t"),
    34: ("n", "h"),
    35: ("n", "d", "a", "a"),
    36: ("n", "t", "a", "a"),
    37: ("n",) * 4,
    41: ("n", "n", "p"),
    42: ("n", "n", "n", "p"),
    44: ("n", "p", "p"),
    45: ("n", "p", "a"),
    102: ("g",),
    103: ("p",),
    104: ("d",),
    105: ("t",),
    106: ("h",),
    107: ("a",),
    108: ("a", "a"),
    109: ("a", "a", "a"),
    111: ("p", "p"),
    112: ("p", "a"),
    113: ("t", "a", "a"),
    114: ("d", "a", "a"),
    115: ("p", "d"),
    116: ("p", "t"),
    117: ("d", "a"),
    # ENDF-102's high-multiplicity channels, MT 152-200. TALYS-derived
    # evaluations do ship them (MT=155, 191 and 197 all appear in the sample
    # this table was verified against), and every one of them was previously
    # absent, so `mt_to_residual` returned None and the channel was skipped
    # outright — the same silent data loss as MT=11, 30 and 114 above.
    152: ("n",) * 5,
    153: ("n",) * 6,
    154: ("n", "n", "t"),
    155: ("t", "a"),
    156: ("n",) * 4 + ("p",),
    157: ("n",) * 3 + ("d",),
    158: ("n", "d", "a"),
    159: ("n", "n", "p", "a"),
    160: ("n",) * 7,
    161: ("n",) * 8,
    162: ("n",) * 5 + ("p",),
    163: ("n",) * 6 + ("p",),
    164: ("n",) * 7 + ("p",),
    165: ("n",) * 4 + ("a",),
    166: ("n",) * 5 + ("a",),
    167: ("n",) * 6 + ("a",),
    168: ("n",) * 7 + ("a",),
    169: ("n",) * 4 + ("d",),
    170: ("n",) * 5 + ("d",),
    171: ("n",) * 6 + ("d",),
    172: ("n",) * 3 + ("t",),
    173: ("n",) * 4 + ("t",),
    174: ("n",) * 5 + ("t",),
    175: ("n",) * 6 + ("t",),
    176: ("n",) * 2 + ("h",),
    177: ("n",) * 3 + ("h",),
    178: ("n",) * 4 + ("h",),
    179: ("n",) * 3 + ("p", "p"),
    180: ("n",) * 3 + ("a", "a"),
    181: ("n",) * 3 + ("p", "a"),
    182: ("d", "t"),
    183: ("n", "p", "d"),
    184: ("n", "p", "t"),
    185: ("n", "d", "t"),
    186: ("n", "p", "h"),
    187: ("n", "d", "h"),
    188: ("n", "t", "h"),
    189: ("n", "t", "a"),
    190: ("n", "n", "p", "p"),
    191: ("p", "h"),
    192: ("d", "h"),
    193: ("h", "a"),
    194: ("n",) * 4 + ("p", "p"),
    195: ("n",) * 4 + ("a", "a"),
    196: ("n",) * 4 + ("p", "a"),
    197: ("p", "p", "p"),
    198: ("n", "p", "p", "p"),
    199: ("n",) * 3 + ("p", "p", "a"),
    200: ("n",) * 5 + ("p", "p"),
}

# Discrete-level and continuum partials: one range of MTs, one emitted particle.
#   MT 50-91:   (x,n) to specific levels, plus the n continuum at 91
#   MT 600-649: (x,p) levels, 650-699 (x,d), 700-749 (x,t),
#   MT 750-799: (x,³He), 800-849 (x,α), 875-891 (x,2n)
#
# The neutron band starts at 50, not 51. For an incident *neutron* MT=50 is
# (n,n₀) — the same thing as elastic — and the residual==target check in
# `mt_to_residual` discards it, so widening the band is a no-op for every
# neutron library. For an incident *charged particle* MT=50 is (z,n₀), the
# ground-state transition of a genuine transmutation, and dropping it loses real
# cross section near threshold: ⁹Be(p,n₀)⁹B opens at 2.06 MeV and is the
# dominant (p,n) channel there.
#
# This only bites tapes that decompose (z,n) into the discrete-level band with
# no MT=4 total to fall back on — in practice the older adopted evaluations
# (LANL) rather than TALYS output, which writes MT=4. That is the same family of
# tapes as the #335 dropouts, and ⁷Li(p,n)⁷Be lands on its textbook 1.88 MeV
# threshold once MT=50 is included.
LEVEL_RANGE_PARTICLES: dict[tuple[int, int], tuple[str, ...]] = {
    (50, 91): ("n",),
    (600, 649): ("p",),
    (650, 699): ("d",),
    (700, 749): ("t",),
    (750, 799): ("h",),
    (800, 849): ("a",),
    (875, 891): ("n", "n"),
}

#: MT numbers that describe no single residual product, and so must never be
#: given an entry above. Listed explicitly rather than left to fall through the
#: table's `in` check, because "absent because it has no residual" and "absent
#: because somebody forgot it" are different states and only one of them is a
#: bug. `mt_to_residual` consults this first.
#:
#:   1 total, 3 nonelastic, 27 absorption, 101 disappearance — sums of other
#:     channels; filing them as a product would double-count every one.
#:   2 elastic — the projectile re-emerges and no isotope is produced. It also
#:     collides with the (n,γ) residual (Z, A+1), where potential scattering
#:     (~barns) would swamp the real capture (~mb).
#:   5 "anything" — a catch-all whose product differs per evaluation. MF=10
#:     names that product directly via IZAP; MT alone cannot.
#:   18-21, 38 fission — many products, not one.
#:   201-207 (x,Xn), (x,Xγ), (x,Xp) … (x,Xα) — particle-production yields
#:     ("how many protons come out", summed over every channel), not channels.
#:     201 and 202 are listed for the same reason as 203-207, but note that
#:     `endf.reaction.REACTION_NAME` does not carry them, so the completeness
#:     check in tests/test_mt_residuals.py cannot see them either way. They are
#:     here because a reader needs the set to be the whole answer.
#:   301, 444 heating and damage energy — not cross-sections of a channel.
NO_RESIDUAL_MTS: frozenset[int] = frozenset(
    {1, 2, 3, 5, 18, 19, 20, 21, 27, 38, 101, 201, 202, 203, 204, 205, 206, 207, 301, 444}
)


def emitted_za(particles: tuple[str, ...]) -> tuple[int, int]:
    """Sum a list of emitted particles into the (Z, A) that leaves."""
    return (
        sum(PARTICLE_ZA[p][0] for p in particles),
        sum(PARTICLE_ZA[p][1] for p in particles),
    )


#: MT -> (emitted_Z, emitted_A). Derived, never hand-written — see #351.
MT_TO_EMISSION: dict[int, tuple[int, int]] = {
    mt: emitted_za(particles) for mt, particles in MT_EMITTED_PARTICLES.items()
}

#: (mt_lo, mt_hi) -> (emitted_Z, emitted_A), derived the same way.
LEVEL_RANGES: dict[tuple[int, int], tuple[int, int]] = {
    mt_range: emitted_za(particles) for mt_range, particles in LEVEL_RANGE_PARTICLES.items()
}


def mt_to_residual(
    mt: int,
    target_z: int,
    target_a: int,
    proj_z: int,
    proj_a: int,
) -> tuple[int, int] | None:
    """Compute residual (Z, A) from MT number and target+projectile.

    Returns None for reactions that don't produce a single residual (fission,
    the summed and catch-all MTs — see NO_RESIDUAL_MTS) or that don't transmute
    the nucleus: elastic (MT=2) and inelastic scattering leave the target
    isotope unchanged (an excited state decays back to it), so they must NOT
    populate a product channel. Elastic in particular collides with the (n,γ)
    residual (Z, A+1) and would swamp the real capture with potential
    scattering (~barns vs ~mb). Metastable products from inelastic are carried
    by the MF=10 isomeric section instead — see `parse_mf10_rows`, which names
    the product from ENDF's IZAP rather than deriving it from MT.

    Also returns None for any MT with no entry in `MT_EMITTED_PARTICLES` or
    `LEVEL_RANGE_PARTICLES`. That is a *skip*, so a missing entry is silent data
    loss rather than a visible error — which is how MT=11, 30 and 114 went
    unnoticed until #351. Add the channel rather than letting it fall through.
    """
    if mt in NO_RESIDUAL_MTS:
        return None

    emit: tuple[int, int] | None = MT_TO_EMISSION.get(mt)
    if emit is None:
        for (mt_lo, mt_hi), e in LEVEL_RANGES.items():
            if mt_lo <= mt <= mt_hi:
                emit = e
                break
    if emit is None:
        return None

    res_z = target_z + proj_z - emit[0]
    res_a = target_a + proj_a - emit[1]
    if res_z <= 0 or res_a <= 0:
        return None
    if (res_z, res_a) == (target_z, target_a):
        # No transmutation — the residual is the target, which decays back to
        # it. MT=4 and MT=51-91 are what motivate this: they legitimately emit
        # one neutron and legitimately produce nothing new.
        #
        # It is also belt-and-suspenders. A future entry whose emission summed
        # to the projectile's own (Z, A) would be silently swallowed here
        # rather than filed wrongly — quieter than it should be, but the MF=10
        # IZAP oracle in tests/test_mt_residuals.py compares against the
        # evaluator's own product and would fail on it.
        return None
    return (res_z, res_a)


# ---------------------------------------------------------------------------
# ENDF-6 file parsing
# ---------------------------------------------------------------------------

# ENDF filenames are not one pattern — the mirrors disagree, and every
# disagreement so far has failed *silently*: an unmatched name is skipped with a
# warning and the run still exits 0. Three separate shapes have been observed,
# and each was found only by reading the log of a run that reported success:
#
#   n_029-Cu-63_2925.zip      Z zero-padded to 3   (most libraries)
#   n_79-Au-197_7925.zip      Z to 2               (IRDFF-II)
#   n_3-Li-6_0325.zip         Z to 1               (IRDFF-II, Z < 10)
#   n_9640_96-Cm-245.zip      MAT first, then Z    (BROND-3.1 — this one
#                                                   produced 0 elements,
#                                                   0 rows, exit code 0)
#   n_095-Am-242M_9547.zip    isomeric state suffix on A (JEFF/JENDL/TENDL)
#   he4_002-HE-4_0228.zip     projectile code carries a digit, element
#                             UPPERCASE (ENDF/B-VIII.1 he3/he4 — this one
#                                        matched *nothing*, so five shipped
#                                        sublibraries could not be rebuilt)
#
# Two alternatives rather than one increasingly baroque pattern, and an
# explicit isomer group so metastable targets stop being discarded.
#
# The prefix is the sublibrary *directory* name on the mirror, which is the
# projectile code: n, p, d, t, he3, he4. `[a-z]+` could not match the last two,
# so `parse_endf_filename` returned None for every file in them (#372).
#
# `[a-z]+\d*_` and not `[a-z0-9]+_`. Both accept every real filename — the
# difference is what they accept *besides*. `[a-z0-9]+` cannot tell a projectile
# code from a MAT number, so it reads `9640_029-Cu-63_2925.zip` as Z=29, A=63,
# attributing to a target a file whose leading field is a MAT. `[a-z]+\d*`
# requires the shape a projectile code actually has — letters, then an optional
# mass number — and rejects `9640_`, `42_` and `0_`. Accepting a malformed name
# and inventing an attribution for it is the failure direction this file has
# been bitten by four times; rejecting it reaches the empty-ingest guard, which
# is loud.
_FN_ZFIRST = re.compile(r"[a-z]+\d*_(\d{1,3})-([A-Za-z]+)-(\d+)([A-Za-z]\d?)?_(\d+)\.zip")
_FN_MATFIRST = re.compile(r"[a-z]+\d*_(\d+)_(\d{1,3})-([A-Za-z]+)-(\d+)([A-Za-z]\d?)?\.zip")


def parse_endf_filename(filename: str) -> tuple[int, int, str] | None:
    """Return (target_Z, target_A, isomer) or None if unrecognised.

    `isomer` is '' for a ground-state target, else the lowercased suffix
    ('m', 'm1', 'n'). Callers must treat None as an error worth surfacing —
    silently skipping is how BROND-3.1 ingested nothing while reporting success.
    """
    m = _FN_ZFIRST.match(filename)
    if m:
        return int(m.group(1)), int(m.group(3)), (m.group(4) or "").lower()
    m = _FN_MATFIRST.match(filename)
    if m:
        return int(m.group(2)), int(m.group(4)), (m.group(5) or "").lower()
    return None


# Cross-sections below/above these bounds are TALYS overflow sentinels
# (~1.99e35 b) or non-physical, and are dropped on both the MF=3 and MF=10 paths.
_XS_MAX_BARNS = 1e30


def is_signed_section(xs_barns: np.ndarray) -> bool:
    """True if this MF=3 curve carries a negative value, and so is not a sigma.

    A signed quantity is not a cross-section, so none of it is usable — not even
    the positive part (#377).

    For charged-particle projectiles, MF=3 MT=2 does not hold elastic sigma.
    Charged-particle elastic lives in MF=6 MT=2 under LAW=5 and diverges
    (Rutherford), so there is no finite total to tabulate; MF=3 MT=2 holds the
    *nuclear-interference term* relative to it, which is signed by construction.

    Filtering to `xs > 0` handles the all-negative sections correctly by
    accident — the whole section vanishes. The mixed-sign ones are the defect:
    p+Cu-65 has 44 points, 25 of them positive, and keeping those writes a
    172 mb → 12.6 b curve over 1–30 MeV as `kind='channel'`, `MT=2`, `xs_mb`.
    That is a different physical quantity wearing sigma's units, and it looks
    entirely reasonable — the #326 and #351 failure shape, where the number is
    plausible so nothing questions it.

    The rule is stated on the data, not on (projectile, MT): any negative value
    means the section is not sigma. An audit of 165 evaluations across five
    charged-particle sublibraries and three neutron ones found MT=2 to be the
    *only* signed MF=3 section, and no negatives at all under neutrons — so this
    is not broader than the class it describes, and it stays right if another MT
    joins it.

    A function rather than an inline test because `scripts/backfill_xs_nuclides.py`
    reads MF=3 sections too, straight off a tape, and had its own point-wise
    `xs > 0` filter — i.e. the half-fix this rule exists to reject. One spelling,
    imported (CLAUDE.md principle 1).
    """
    finite = np.isfinite(xs_barns)
    return bool(finite.any() and (xs_barns[finite] < 0).any())


@dataclass
class ParsedFile:
    """Rows from one evaluation, plus the counters the ingest guards need.

    The `*_sections` counters record what was *present in the source*, whether
    or not any row survived; the `*_rows` counters record what came out. Keeping
    the two apart is the whole point: #340 was a parser that saw the sections
    and emitted zero rows while the run exited 0, and a guard can only catch
    that if it can see both numbers.

    MF=3 is counted for the same reason MF=10 is. Before #340 the MF=10 path
    produced nothing, so "MF=3 died" and "the whole library died" were the same
    event and the empty-ingest guard caught it. Now that MF=10 emits rows, a
    library could keep a healthy element count on MF=10 alone while every MF=3
    cross-section vanished — `parse_mf3` returns its curve under the key
    `'sigma'`, and that lookup is exactly as version-fragile as the one that
    caused #340.
    """

    rows: list[dict]
    mf3_sections: int = 0
    mf3_rows: int = 0
    mf10_sections: int = 0
    mf10_rows: int = 0
    #: MF=3 sections carrying at least one finite, positive, in-range point.
    #: A section of all-negative values yields no row and is not a failure —
    #: charged-particle MF=3 MT=2 is the nuclear-interference term against a
    #: divergent Rutherford cross-section and is negative by construction.
    mf3_usable_sections: int = 0
    #: Of those, the ones whose MT names a single residual. Only these can
    #: produce a `kind='production'` row, so only these belong in that guard's
    #: denominator — jendl-5's deuteron sublibrary is entirely MT=2 and MT=5,
    #: neither of which names one, and zero production rows there is correct.
    mf3_residual_sections: int = 0
    #: MF=10 sections carrying at least one level with a nameable product.
    #: MT=18 uses IZAP=-1 for "fission products, unspecified"; a section of
    #: nothing but those yields no row and is not a failure either.
    mf10_product_sections: int = 0
    #: MT -> number of MF=3 sections dropped whole for carrying a negative
    #: value, i.e. for not being a cross-section (#377). Reported rather than
    #: merely skipped: "this library represents no charged-particle elastic"
    #: and "this library was never asked" must stay distinguishable.
    signed_sections: dict[int, int] = field(default_factory=dict)
    #: `kind='channel'` rows — one per MF=3 MT, carrying MT itself (#347).
    channel_rows: int = 0
    #: Channel rows whose MT names no single product: total, elastic,
    #: inelastic, fission. Zero of these is the #347 signature.
    null_residual_rows: int = 0


def sum_on_union_grid(
    contribs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Sum cross-section contributions onto the union of their energy grids.

    Several ENDF reactions routinely reach the same product: MT=600-649 all make
    (Z-1, A) in MF=3, and JEFF's Sr-86 reaches Rb-84m through MT=32, 41 and 105
    in MF=10. Emitting one row set per reaction would leave a consumer to either
    double-count or arbitrarily pick one, and deduplicating after the fact is
    what collapsed Fe-56(n,p) from 114 mb to 0.1 mb (#326). Interpolate each
    contribution onto the union grid instead and add.

    Linear interpolation is the ENDF default (INT=2); outside a contribution's
    own threshold-to-max range it contributes zero.
    """
    if len(contribs) == 1:
        return contribs[0]
    e_union = np.unique(np.concatenate([e for e, _ in contribs]))
    xs_total = np.zeros_like(e_union)
    for e, s in contribs:
        xs_total += np.interp(e_union, e, s, left=0.0, right=0.0)
    positive = xs_total > 0
    return e_union[positive], xs_total[positive]


def lfs_to_state(lfs: int, product_lfs_levels: set[int]) -> str:
    """Map an ENDF MF=9/MF=10 `LFS` onto this repo's `state` vocabulary.

    `LFS` is the *level number* of the produced nuclide, not an isomer index.
    Real evaluations use 0, 1, 2, 3 — but also 14, 22, 47 and 72, because
    TALYS-derived files (TENDL, JEFF, parts of ENDF/B) put the isomer wherever
    it happens to sit in the level scheme. Fe-53m really is level 22. Spelling
    that literally as `'m22'` would mint a state no other table in this repo can
    join against: `meta/ensdf/nuclides.parquet` knows only `''`, `'m'`, `'m2'`,
    `'m3'`, and `tendl-2023-iso/xs` only `''`, `'g'`, `'m'`.

    So rank rather than transcribe. `LFS=0` becomes `'g'` — the evaluation
    asserts *ground state*, which is a different and stronger claim than MF=3's
    `''` ("summed over whatever states this channel populates"), and keeping the
    two spellings apart is what stops Al-27(n,2n) from filing its 177 mb total
    and its 114 mb ground-state part under one key. The non-zero levels for that
    product, in ascending order, become `'m'`, `'m2'`, `'m3'`, …

    `product_lfs_levels` must be every LFS the material carries for this product
    across *all* its MF=10 sections, not just the section in hand: the same
    product routinely appears under several MTs carrying different subsets of
    its levels, and ranking per-section would spell one nuclide two ways in one
    file. Known limitation: if an evaluation ships only the second isomer and
    omits the first, we call it `'m'`, because MF=10 alone cannot tell us how
    many isomers sit below the level it named.
    """
    if lfs <= 0:
        return "g"
    excited = sorted(level for level in product_lfs_levels if level > 0)
    rank = excited.index(lfs) + 1
    return "m" if rank == 1 else f"m{rank}"


class TargetStateConflict(RuntimeError):
    """The filename and the evaluation disagree about the target's isomeric state.

    One of the two is wrong about which nuclide this file is, and there is no
    safe way to pick. Raised rather than preferred, per #353 — a silent
    preference here writes a metastable evaluation under a ground-state key,
    which is the exact defect the column was added to end.
    """


def target_state_from_material(
    material,  # noqa: ANN001 — endf.Material
    marker: str,
    target_a: int,
    filename: str,
) -> str | None:
    """Resolve `target_state` for one evaluation, and check the two sources agree.

    `LISO` from MF=1/451 is authoritative: it is the *isomeric state number* of
    the target, stated by the evaluation itself, so LISO=1 is the first isomer
    with no decoding required.

    The filename marker is used only to detect disagreement, never as the rank.
    That inverts #353's suggestion of cross-checking `LIS`, which is a different
    field — `LIS` counts excited *levels*, `LISO` counts *isomeric states*, and
    they routinely differ. The issue's own example shows it: `n_035-Br-80M`
    carries `LIS=2` while being the *first* isomer, so an equality check against
    the marker's rank would have failed on a correct file.

    Using LISO for the rank also means an unrecognised marker costs nothing:
    `'n'` appears in some mirror listings and this repository does not know what
    it means, so it is not in `ENDF_TARGET_MARKERS` and is not guessed at — LISO
    already said the answer.

    Returns NULL for a natural-element target: an isotopic mixture has no
    isomeric state (see `state_vocabulary.target_state_for_natural_element`).
    """
    if target_a == 0:
        return target_state_for_natural_element()

    info = material.section_data.get((1, 451), {})
    liso = info.get("LISO")

    marker_rank = ENDF_TARGET_MARKERS.get(marker)

    if liso is None:
        # No MF=1/451 to consult. Fall back to the marker, and refuse an
        # unrecognised one rather than filing it under the ground state — which
        # is what discarding the marker entirely used to do, for every file.
        if marker_rank is None:
            raise TargetStateConflict(
                f"{filename}: isomer marker {marker!r} is not one this repository can "
                f"spell ({sorted(ENDF_TARGET_MARKERS)}), and the evaluation has no "
                "MF=1/451 LISO to resolve it. Refusing to guess the target's state."
            )
        return isomer_state(marker_rank)

    liso = int(liso)
    if marker_rank is not None and marker_rank != liso:
        raise TargetStateConflict(
            f"{filename}: the filename says isomer rank {marker_rank} (marker {marker!r}) "
            f"but MF=1/451 says LISO={liso}. One of them is wrong about which nuclide "
            "this evaluation is; refusing to pick."
        )
    if marker_rank is None and liso == 0:
        # An unknown marker on a file the evaluation calls ground state. The
        # marker means *something*, and 'ground state with a suffix' is not a
        # combination we can explain, so it is not one to write.
        raise TargetStateConflict(
            f"{filename}: isomer marker {marker!r} is unrecognised, but MF=1/451 says "
            "LISO=0 (ground state). The marker contradicts the evaluation."
        )
    return isomer_state(liso)


def parse_mf10_rows(
    material,  # noqa: ANN001 — endf.Material, imported lazily by the caller
    target_a: int,
) -> tuple[list[dict], int, int]:
    """Extract MF=10 isomeric-production rows.

    Returns `(rows, sections_seen, sections_naming_a_product)`.

    The two counts differ, and the difference is the guard's denominator. A
    section whose every level carries `IZAP <= 0` names no product to file a row
    under — MT=18 uses `IZAP = -1` for "fission products, unspecified" — so
    emitting nothing for it is correct, not a failure. jeff-4.0's proton
    sublibrary is *entirely* such sections, and counting raw sections made the
    #340 guard fire on a library that was being read perfectly (#372 follow-up).

    Shape note (#340). The pinned `endf` package parses MF=9 and MF=10 with
    `endf.mf9.parse_mf9_mf10`, which returns::

        {'ZA':…, 'AWR':…, 'LIS':…, 'NS':…,
         'levels': [{'QM':…, 'QI':…, 'IZAP':…, 'LFS':…, 'sigma': Tabulated1D}]}

    The previous implementation read `section.get("subsections", [])` and
    `sub.get("ZAPS", 0)`. Neither key has ever existed, so `.get()` returned its
    default on every section, the loop body never ran, and every isomeric
    production section in every library was discarded while the ingest exited 0.
    `tests/test_fetch_endf_libs.py` pins this shape so an `endf` version bump
    that moves it fails a test instead of silently dropping the data again.

    Note MF=10 names its product directly via `IZAP` (Z*1000 + A), so unlike the
    MF=3 path there is nothing to derive from MT and no reaction we have to know
    about in advance — MT=5 ("anything") is carried as faithfully as MT=16.
    """
    sections = [(mt, sec) for (mf, mt), sec in material.section_data.items() if mf == 10]
    if not sections:
        return [], 0, 0

    # Sections carrying at least one level with a nameable product. This, not
    # `len(sections)`, is what "MF=10 should have produced rows" means.
    product_sections = sum(1 for _mt, sec in sections if any(int(level["IZAP"]) > 0 for level in sec["levels"]))

    # Which levels does this material carry for each product, across every
    # section? lfs_to_state needs the whole set to rank consistently.
    product_lfs_levels: dict[int, set[int]] = {}
    for _mt, section in sections:
        for level in section["levels"]:
            izap = int(level["IZAP"])
            if izap > 0:
                product_lfs_levels.setdefault(izap, set()).add(int(level["LFS"]))

    # (product_Z, product_A, state) -> contributions to sum on the union grid.
    by_product: dict[tuple[int, int, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for mt, section in sections:
        for level in section["levels"]:
            izap = int(level["IZAP"])
            if izap <= 0:
                # MT=18 carries IZAP=-1 in JEFF/TENDL: fission names no single
                # product, so there is nothing to file a production row under.
                # Expected for fission, worth hearing about anywhere else.
                log = logger.debug if mt == 18 else logger.warning
                log("  MF=10 MT=%d: IZAP=%d names no product, skipping", mt, izap)
                continue
            tab = level.get("sigma")
            if tab is None:
                logger.warning("  MF=10 MT=%d IZAP=%d: no 'sigma' TAB1, skipping", mt, izap)
                continue

            energies_ev = np.asarray(tab.x, dtype=float)
            xs_barns = np.asarray(tab.y, dtype=float)
            good = np.isfinite(xs_barns) & (xs_barns > 0) & (xs_barns <= _XS_MAX_BARNS)
            if not good.any():
                continue

            state = lfs_to_state(int(level["LFS"]), product_lfs_levels[izap])
            key = (izap // 1000, izap % 1000, state)
            by_product.setdefault(key, []).append((energies_ev[good], xs_barns[good]))

    rows: list[dict] = []
    for (prod_z, prod_a, state), contribs in by_product.items():
        e_union, xs_total = sum_on_union_grid(contribs)
        rows.extend(
            {
                "target_A": target_a,
                # Summed across every MT reaching this product, so no single MT
                # identifies it — a production row, exactly like the MF=3 sums.
                "kind": "production",
                "MT": None,
                "residual_Z": prod_z,
                "residual_A": prod_a,
                "state": state,
                "energy_MeV": float(e_ev) * 1e-6,
                "xs_mb": float(xs_b) * 1e3,
            }
            for e_ev, xs_b in zip(e_union, xs_total)
        )
    return rows, len(sections), product_sections


def parse_endf_file(
    endf_text: str,
    target_z: int,
    target_a: int,
    projectile: str,
    marker: str = "",
) -> ParsedFile:
    """Parse an ENDF-6 format text file and extract cross-section data.

    Returns a `ParsedFile`: row dicts keyed to the Parquet schema, plus the
    MF=10 counters `fetch_library` guards on.

    Discrete-level partials (e.g. MT=600-649 for (n,p), 800-849 for (n,α)) all
    map to the same residual (Z-1,A) or (Z-2,A-3). When the evaluation *also*
    ships the summed MT (103, 107, ...), the summed MT already carries the full
    channel; when it does not (FENDL Fe-56 ships only MT=600-649), the partials
    must be summed by us. Earlier revisions deduplicated by
    (target_A, residual, energy) after appending every MT, which silently kept
    exactly one MT's contribution per (residual, energy) — a summed row where the
    library was helpful, a single-level partial where it was not (Fe-56(n,p)
    read 0.1 mb instead of 114 mb; #326). We now accumulate per residual on the
    union energy grid via linear interpolation and sum the MT contributions
    before emitting rows.
    """
    import endf

    proj_z, proj_a = PROJECTILE_ZA[projectile]

    try:
        material = endf.Material(io.StringIO(endf_text))
    except Exception as e:
        logger.warning("Failed to parse ENDF material Z=%d A=%d: %s", target_z, target_a, e)
        return ParsedFile(rows=[])

    rows: list[dict] = []

    # Extract MF=3 (cross-section) data, accumulating per residual so that
    # discrete-level partial MTs sharing one residual (e.g. MT=600-649 → (Z-1,A))
    # are summed on the union grid rather than mutually clobbered by dedup.
    #
    # Key policy: prefer the summed MT (103, 107, …) when present, and skip the
    # partials it sums. Otherwise sum the partials. This mirrors the redundant-MT
    # handling in build_neutron_njoy.py — never double-count a channel that the
    # evaluator has already summed for us.
    _SUMMED_TO_PARTIAL_RANGE: dict[int, range] = {
        103: range(600, 650),  # (n,p) — summed by MT=103, partials MT=600–649
        104: range(650, 700),  # (n,d)
        105: range(700, 750),  # (n,t)
        106: range(750, 800),  # (n,³He)
        107: range(800, 850),  # (n,α)
        16: range(875, 892),  # (n,2n) — summed by MT=16, partials MT=875–891
    }
    mf3_mts = {mt for (mf, mt) in material.section_data if mf == 3}
    skip_partials: set[int] = set()
    for summed_mt, partial_range in _SUMMED_TO_PARTIAL_RANGE.items():
        if summed_mt in mf3_mts:
            skip_partials.update(mt for mt in partial_range if mt in mf3_mts)

    # (residual_Z, residual_A) -> list[(energies_ev: np.ndarray, xs_barns: np.ndarray)]
    by_residual: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    channel_rows: list[dict] = []
    null_residual_rows = 0
    mf3_usable_sections = 0
    mf3_residual_sections = 0
    signed_sections: dict[int, int] = {}

    for (mf, mt), section in material.section_data.items():
        if mf != 3:
            continue

        try:
            tab = section.get("sigma")
            if tab is None:
                continue
            energies_ev = np.asarray(tab.x, dtype=float)
            xs_barns = np.asarray(tab.y, dtype=float)
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("  Skipping MF=%d MT=%d: %s", mf, mt, e)
            continue

        # Drop TALYS overflow sentinels (~1.99e35 b) and non-positive values,
        # and drop a signed section whole — see `is_signed_section` (#377).
        if is_signed_section(xs_barns):
            signed_sections[mt] = signed_sections.get(mt, 0) + 1
            continue

        good = np.isfinite(xs_barns) & (xs_barns > 0) & (xs_barns <= _XS_MAX_BARNS)
        if not good.any():
            logger.debug("  MF=3 MT=%d: no positive in-range points, skipping", mt)
            continue
        mf3_usable_sections += 1
        e_good, xs_good = energies_ev[good], xs_barns[good]

        residual = mt_to_residual(mt, target_z, target_a, proj_z, proj_a)
        if residual is not None:
            mf3_residual_sections += 1

        # --- kind='channel': the ENDF datum as ENDF states it -----------------
        # One row set per MT, carrying MT itself. `residual` is null wherever the
        # channel names no single product — total, elastic, inelastic, fission.
        # Those four used to be dropped outright, because the caller treated
        # `mt_to_residual() is None` as "skip" rather than "no residual", so the
        # evaluated half of the repository shipped only transmutation channels
        # and U-235(n,f) could not be queried at all (#347). MT is the primitive:
        # MT -> residual is derivable, residual -> MT is not.
        res_z, res_a = residual if residual is not None else (None, None)
        if residual is None:
            null_residual_rows += len(e_good)
        channel_rows.extend(
            {
                "target_A": target_a,
                "kind": "channel",
                "MT": mt,
                "residual_Z": res_z,
                "residual_A": res_a,
                # MF=3 gives the channel total over every state it populates.
                # SUM says that; it is a claim, not the absence of one (#357).
                "state": SUM,
                "energy_MeV": float(e_ev) * 1e-6,
                "xs_mb": float(xs_b) * 1e3,
            }
            for e_ev, xs_b in zip(e_good, xs_good)
        )

        # --- kind='production': our sum over the channels reaching a residual --
        if residual is None or mt in skip_partials:
            continue
        by_residual.setdefault(residual, []).append((e_good, xs_good))

    for (res_z, res_a), contribs in by_residual.items():
        e_union, xs_total = sum_on_union_grid(contribs)
        rows.extend(
            {
                "target_A": target_a,
                # A sum over several MTs names no single one, so MT stays null —
                # that is what tells the two kinds apart in a union.
                "kind": "production",
                "MT": None,
                "residual_Z": res_z,
                "residual_A": res_a,
                # SUM is "summed over whatever states this channel
                # populates", which MF=10 spells 'g'/'m'/'m2' — see
                # lfs_to_state. It was spelled '' until #357, where the same
                # four bytes also meant "not stated" on an EXFOR row and "the
                # ground state" in meta/ensdf.
                "state": SUM,
                "energy_MeV": float(e_ev) * 1e-6,
                "xs_mb": float(xs_b) * 1e3,
            }
            for e_ev, xs_b in zip(e_union, xs_total)
        )

    mf3_rows = len(rows)
    mf10_rows, mf10_sections, mf10_product_sections = parse_mf10_rows(material, target_a)
    rows.extend(mf10_rows)
    rows.extend(channel_rows)

    # Stamped once here rather than in each of the three row builders: it is a
    # property of the *file*, identical for every row in it, and three copies of
    # that is three places for them to disagree (#353).
    target_state = target_state_from_material(material, marker, target_a, f"Z={target_z} A={target_a}")
    for row in rows:
        row["target_state"] = target_state

    return ParsedFile(
        rows=rows,
        channel_rows=len(channel_rows),
        null_residual_rows=null_residual_rows,
        mf3_sections=len(mf3_mts),
        mf3_usable_sections=mf3_usable_sections,
        mf3_residual_sections=mf3_residual_sections,
        signed_sections=signed_sections,
        mf10_product_sections=mf10_product_sections,
        mf3_rows=mf3_rows,
        mf10_sections=mf10_sections,
        mf10_rows=len(mf10_rows),
    )


# ---------------------------------------------------------------------------
# Download + process
# ---------------------------------------------------------------------------


def list_endf_files(
    lib: LibraryDef,
    sublib_code: str,
    session: requests.Session,
) -> list[str]:
    """Get the zip filenames the mirror serves for one sublibrary.

    Raises rather than returning an empty list. A declared sublibrary that
    lists no files is a defect in the registry or a change on the mirror, not
    a condition to log: the previous `logger.warning(...); return` meant a
    re-ingest of `iaea-medical/n` walked a 404, wrote a line nobody reads in
    the middle of a multi-hour run, and exited 0 (#356). That is #334's
    zero-file ingest again — BROND-3.1 ingested nothing, reported success, and
    would have deleted an 84-element library had the diff not caught it.

    Raised here rather than in `fetch_library` so nothing downstream has to
    decide what an empty listing means.
    """
    sublib_dir = lib.sublibraries.get(sublib_code)
    if sublib_dir is None:
        raise KeyError(f"{lib.key} declares no '{sublib_code}' sublibrary; have {sorted(lib.sublibraries)}")

    url = f"{IAEA_MIRROR}/{lib.iaea_path}/{sublib_dir}/"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"{lib.key}/{sublib_code}: cannot list {url}: {e}. The registry declares this "
            "sublibrary, so either the mirror moved or LIBRARIES is wrong. Refusing to "
            "continue and report success on a library that fetched nothing."
        ) from e

    # Parse HTML directory listing for .zip filenames
    filenames = re.findall(r'href="([^"]+\.zip)"', resp.text)
    if not filenames:
        raise RuntimeError(
            f"{lib.key}/{sublib_code}: {url} returned HTTP {resp.status_code} but listed no "
            ".zip files. A declared sublibrary with zero files is a defect — correct "
            "LIBRARIES, or the mirror's directory layout has changed."
        )
    return filenames


def download_and_parse(
    lib: LibraryDef,
    sublib_code: str,
    filename: str,
    session: requests.Session,
) -> ParsedFile:
    """Download a single ENDF zip file and parse it."""
    sublib_dir = lib.sublibraries[sublib_code]
    url = f"{IAEA_MIRROR}/{lib.iaea_path}/{sublib_dir}/{filename}"

    # Parse target Z, A from filename
    parsed = parse_endf_filename(filename)
    if parsed is None:
        logger.warning("Cannot parse filename: %s", filename)
        return ParsedFile(rows=[])

    target_z, target_a, marker = parsed

    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Download failed %s: %s", filename, e)
        return ParsedFile(rows=[])

    # Extract ENDF text from zip
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if not names:
                return ParsedFile(rows=[])
            endf_text = zf.read(names[0]).decode("ascii", errors="replace")
    except (zipfile.BadZipFile, KeyError) as e:
        logger.warning("Bad zip %s: %s", filename, e)
        return ParsedFile(rows=[])

    return parse_endf_file(endf_text, target_z, target_a, sublib_code, marker)


def fetch_library(
    lib_key: str,
    sublib_code: str,
    output_dir: Path,
    session: requests.Session,
) -> None:
    """Fetch and convert an entire sub-library to Parquet."""
    lib = LIBRARIES[lib_key]
    logger.info("Fetching %s / %s ...", lib.name, sublib_code)

    # No empty-listing check here: list_endf_files raises rather than returning
    # [], so there is one place that decides what zero files means (#356).
    filenames = list_endf_files(lib, sublib_code, session)
    logger.info("  Found %d ENDF files", len(filenames))

    # Group rows by element
    element_rows: dict[str, list[dict]] = {}
    total_files = 0
    total_rows = 0
    # Counted across every source file, including files that yielded no rows —
    # "sections seen but nothing emitted" is exactly the state we must detect.
    mf3_sections = 0
    mf3_rows = 0
    mf10_sections = 0
    mf10_rows = 0
    channel_rows = 0
    null_residual_rows = 0
    mf3_usable_sections = 0
    mf3_residual_sections = 0
    mf10_product_sections = 0
    signed_sections: dict[int, int] = {}
    # Every upstream tape that yielded nothing. Skipping one silently is how a
    # single natural target (p+Be-9, n+Ho-165) disappears from a release with no
    # diff to notice and ~100% coverage everywhere else — see #335 / #336.
    #
    # The aggregate guards below cannot see this: they ask whether a *sublibrary*
    # went dark, and a sublibrary where 2296 of 2298 tapes convert perfectly is
    # healthy by every one of those measures. Which two are missing is the whole
    # question, and only a per-tape account answers it. The skip is legitimate
    # (some evaluations really do carry no production data), the silence is not.
    yielded_nothing: list[str] = []

    for i, fname in enumerate(filenames):
        if (i + 1) % 50 == 0:
            logger.info("  Processing %d/%d ...", i + 1, len(filenames))

        parsed_file = download_and_parse(lib, sublib_code, fname, session)
        mf3_sections += parsed_file.mf3_sections
        mf3_rows += parsed_file.mf3_rows
        channel_rows += parsed_file.channel_rows
        null_residual_rows += parsed_file.null_residual_rows
        mf3_usable_sections += parsed_file.mf3_usable_sections
        mf3_residual_sections += parsed_file.mf3_residual_sections
        mf10_product_sections += parsed_file.mf10_product_sections
        for mt, n in parsed_file.signed_sections.items():
            signed_sections[mt] = signed_sections.get(mt, 0) + n
        mf10_sections += parsed_file.mf10_sections
        mf10_rows += parsed_file.mf10_rows
        if not parsed_file.rows:
            yielded_nothing.append(fname)
            continue

        parsed = parse_endf_filename(fname)
        if parsed is None:
            continue

        target_z = parsed[0]
        elem = element_stem(target_z)
        element_rows.setdefault(elem, []).extend(parsed_file.rows)
        total_files += 1
        total_rows += len(parsed_file.rows)

    if yielded_nothing:
        # Loud by default. A run that drops nuclides is not necessarily wrong,
        # but it must never again be indistinguishable from one that doesn't.
        # Emitted before the guards below so the account survives even when the
        # run goes on to raise — the names are what a human needs either way.
        shown = yielded_nothing[:_MAX_LISTED_TAPES]
        rest = len(yielded_nothing) - len(shown)
        logger.warning(
            "  %d/%d tapes yielded no rows for %s/%s and are ABSENT from the output: %s%s",
            len(yielded_nothing),
            len(filenames),
            lib.name,
            sublib_code,
            ", ".join(sorted(shown)),
            f", … and {rest} more" if rest else "",
        )

    # A library that ingests nothing must not report success. BROND-3.1 did
    # exactly that — every filename used a shape the parser did not recognise,
    # each was skipped with a warning, and the run finished with 0 elements and
    # exit code 0. Had that output been copied over data/, the library would
    # have been silently deleted rather than rebuilt. Raised before anything is
    # written, so no artefact claiming success survives.
    #
    # #334 added this check but left an earlier `logger.warning(...); return` on
    # the same condition above it, so the raise was unreachable and the guard
    # never fired. Warn-and-return *is* the silent success it was written to
    # stop; there is one check now, and it raises.
    #
    # #342 found the same dead guard independently and proposed the same fix.
    # Both placements were compared rather than deferred to: identical position,
    # and this one additionally guards MF=3 and MF=10 going dark on their own,
    # so #342 keeps none of its own. The half it does keep is in
    # `tests/test_builder_staleness.py` — that an empty ingest leaves *no
    # artefact*, which is what a provenance stamp rests on and what
    # `test_empty_ingest_guard_raises_rather_than_returning` does not assert.
    if not element_rows:
        raise RuntimeError(
            f"{lib_key}/{sublib_code} produced 0 elements from {len(filenames)} source files. "
            "Every file was skipped — check the 'Cannot parse filename' warnings above. "
            "Refusing to report success on an empty ingest."
        )

    # Per-file guards. The empty-ingest check above only sees a library that
    # produced *nothing*; each ENDF file type can go dark on its own while the
    # other keeps the element count healthy, and neither failure is visible in
    # an exit code. All three raise before anything is written.
    #
    # Each one asks "sections that *could* have produced rows produced none".
    # The denominator matters as much as the numerator: counting raw sections
    # made two of these fire on charged-particle sublibraries that were being
    # read perfectly, because a section can legitimately have nothing to emit.
    # A guard that cries wolf gets lowered to a warning, and then it is not a
    # guard — so the denominators are narrowed here rather than the guards.
    #
    # MF=10 (#340): reading it through an object shape the `endf` package never
    # returned discarded every isomeric section in every library, and the #334
    # guard could not see it because MF=3 kept producing plenty of rows.
    if mf10_product_sections and not mf10_rows:
        raise RuntimeError(
            f"{lib_key}/{sublib_code}: {mf10_product_sections} of {mf10_sections} MF=10 "
            "isomeric-production sections name a product, and they produced 0 rows. "
            "That is the #340 signature — the `endf` package's MF=10 shape has "
            "most likely moved again (see parse_mf10_rows). Refusing to report "
            "success while silently dropping every isomeric product."
        )
    if mf10_sections and not mf10_product_sections:
        # Not a failure: MT=18 carries IZAP=-1, "fission products, unspecified".
        # jeff-4.0/p is entirely such sections. Said out loud so the difference
        # between "nothing to emit" and "emitted nothing" stays visible.
        logger.info(
            "  %s/%s: all %d MF=10 sections name no product (IZAP<=0, e.g. MT=18 fission); no isomeric rows to emit.",
            lib_key,
            sublib_code,
            mf10_sections,
        )

    # MF=3 production rows. Only a section whose MT names a single residual can
    # make one, so only those belong in the denominator: jendl-5's deuteron
    # sublibrary is entirely MT=2 and MT=5, neither of which names a residual,
    # and zero production rows there is the right answer rather than a fault.
    if mf3_residual_sections and not mf3_rows:
        raise RuntimeError(
            f"{lib_key}/{sublib_code}: {mf3_residual_sections} of {mf3_sections} MF=3 "
            "sections name a residual, and they produced 0 production rows. Either "
            "the `endf` package no longer returns the curve under 'sigma', or "
            "mt_to_residual now rejects every MT."
        )
    if mf3_usable_sections and not mf3_residual_sections:
        logger.info(
            "  %s/%s: none of the %d usable MF=3 sections name a residual "
            "(all MT=2/MT=5 or similar); channel rows only, no production sums.",
            lib_key,
            sublib_code,
            mf3_usable_sections,
        )

    # Channel rows (#347). MF=3 sections that name no residual — total, elastic,
    # inelastic, fission — were dropped by the caller for the whole life of this
    # script, and neither guard above can see it: production rows keep flowing
    # from the transmutation channels. The denominator is sections with at least
    # one usable point, because a section of all-negative values (charged-particle
    # MF=3 MT=2) has nothing to emit either.
    if mf3_usable_sections and not channel_rows:
        raise RuntimeError(
            f"{lib_key}/{sublib_code}: {mf3_usable_sections} of {mf3_sections} MF=3 "
            "sections carry usable points, and they produced 0 kind='channel' rows. "
            "Every evaluated row would then carry a null MT, which is the #347 "
            "state this ingest exists to leave behind."
        )
    if signed_sections:
        # Loud, and once per library rather than once per file. Dropping data is
        # exactly the kind of thing this script has done silently before.
        logger.info(
            "  %s/%s: dropped %d MF=3 section(s) whole for carrying negative values "
            "(by MT: %s). A signed quantity is not a cross-section, so its positive "
            "part is not one either — for charged particles MT=2 is the Rutherford "
            "interference term and the elastic distribution is in MF=6 LAW=5, which "
            "this ingest does not read (#377).",
            lib_key,
            sublib_code,
            sum(signed_sections.values()),
            dict(sorted(signed_sections.items())),
        )

    if mf3_sections and not mf3_usable_sections:
        logger.info(
            "  %s/%s: none of the %d MF=3 sections carry a positive in-range point. "
            "Expected where MF=3 MT=2 is the charged-particle interference term "
            "(the elastic distribution is in MF=6 LAW=5).",
            lib_key,
            sublib_code,
            mf3_sections,
        )

    state_counts: dict[str, int] = {}
    for rows in element_rows.values():
        for row in rows:
            state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1

    # Write Parquet files per element
    # For neutron data: lib_key/xs/n_Fe.parquet
    # For charged particles: lib_key/xs/p_Fe.parquet (same as TENDL layout)
    xs_dir = output_dir / lib_key / "xs"
    xs_dir.mkdir(parents=True, exist_ok=True)

    for elem, rows in element_rows.items():
        stem = f"{sublib_code}_{elem}"
        parsed_stem = parse_stem(stem)
        if parsed_stem is None:  # pragma: no cover — element_stem builds it
            raise RuntimeError(f"{lib_key}: built an unparseable file stem {stem!r}")
        projectile, proj_z, proj_a, target_z = parsed_stem

        df = pl.DataFrame(
            rows,
            schema={
                "target_A": pl.Int32,
                # Per-file, stamped by parse_endf_file. Declared here so a shard
                # whose evaluations are all ground-state still gets a Utf8
                # column rather than an inferred Null one (#353).
                "target_state": pl.Utf8,
                "kind": pl.Utf8,
                "MT": pl.Int32,
                "residual_Z": pl.Int32,
                "residual_A": pl.Int32,
                "state": pl.Utf8,
                "energy_MeV": pl.Float64,
                "xs_mb": pl.Float64,
            },
        )
        # Write CANONICAL_XS_SCHEMA directly (#359). This script used to emit the
        # 6-column legacy form and rely on migrate_xs_schema.py being run
        # afterwards, so a plain re-ingest silently dropped twelve of eighteen
        # columns — library, kind, projectile, proj_Z/A, target_Z, MT and the
        # error/provenance columns — and put identity back in the file path that
        # CLAUDE.md principle 5 exists to get it out of. `kind` and `MT` come
        # from the rows; the rest is per-file identity from the stem.
        df = canonical_frame(
            df,
            library=lib_key,
            kind="production",
            projectile=projectile,
            proj_z=proj_z,
            proj_a=proj_a,
            target_z=target_z,
        )
        # parse_endf_file has already summed contributions per (residual, state)
        # on the union energy grid — for MF=3 partial MTs and, since #340, for
        # the several MTs that reach one MF=10 product. MF=10 rows cannot
        # collide with the MF=3 rows for the same residual either, because MF=3
        # spells its state '' and MF=10 spells it 'g'/'m'/'m2' (lfs_to_state):
        # Al-27(n,2n) files its 177 mb total and its 114 mb ground-state part
        # under different keys, which is what they are. A bare unique(subset=…)
        # here would silently drop MT-partials that share a residual (Fe-56(n,p)
        # collapsed to 0.1 mb; #326) — sort only.
        # target_state sorts beside target_A: the two together name the target
        # nuclide, and leaving it out makes the row order of a shard holding both
        # Br-80 and Br-80m depend on dict iteration (#353).
        df = df.sort("target_A", "target_state", "kind", "MT", "residual_Z", "residual_A", "state", "energy_MeV")

        out_path = xs_dir / f"{stem}.parquet"
        df.write_parquet(out_path, compression=COMPRESSION)

    # Write manifest. `mf10_sections` / `states` are recorded so the next reader
    # can tell "this library ships no isomeric data" from "this library's
    # isomeric data was dropped on the floor" without re-running the ingest.
    #
    # Then stamp it with this script's digest (#342). Without the stamp, a
    # correctness fix here and the parquets it should have regenerated can
    # diverge indefinitely with CI green — which is exactly what happened
    # between #260 and #334, for thirteen months. Routed through
    # `write_builder_stamp` rather than inlining the stamp, so every builder in
    # the repo passes the one guard that refuses to stamp a run that wrote
    # nothing.
    manifest = {
        "library": lib_key,
        "sublibrary": sublib_code,
        "files": len(element_rows),
        "total_rows": total_rows,
        "source_files": total_files,
        "projectiles": [sublib_code],
        "elements": sorted(element_rows.keys()),
        "mf3_sections": mf3_sections,
        # Sections dropped for being signed, and what that means is absent as a
        # result. "not represented" and "not measured" must stay apart: without
        # this a reader cannot tell a library that ships no charged-particle
        # elastic from one nobody asked (#377).
        "signed_sections_dropped": dict(sorted(signed_sections.items())),
        "charged_particle_elastic": (
            "not represented — MF=3 MT=2 is the Rutherford interference term, not "
            "sigma; the elastic distribution is in MF=6 LAW=5, which this ingest "
            "does not read (#377)"
        )
        if signed_sections
        else None,
        "mf3_usable_sections": mf3_usable_sections,
        "mf3_residual_sections": mf3_residual_sections,
        "mf3_rows": mf3_rows,
        "mf10_sections": mf10_sections,
        "mf10_product_sections": mf10_product_sections,
        "mf10_rows": mf10_rows,
        "channel_rows": channel_rows,
        "null_residual_rows": null_residual_rows,
        "states": dict(sorted(state_counts.items())),
    }
    manifest_path = output_dir / lib_key / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_builder_stamp(manifest_path, Path(__file__), files_written=len(element_rows))

    logger.info(
        "  Done: %d elements, %d source files, %d total rows "
        "(%d channel rows, %d of them null-residual; %d MF=10 sections → %d isomeric rows; states %s) → %s/",
        len(element_rows),
        total_files,
        total_rows,
        channel_rows,
        null_residual_rows,
        mf10_sections,
        mf10_rows,
        dict(sorted(state_counts.items())),
        lib_key,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Separate from `main` so tests can assert on the argument defaults — notably
    `--output`, whose old repo-root value scattered ingests across tracked
    top-level directories (#341).
    """
    parser = argparse.ArgumentParser(
        description="Fetch evaluated nuclear data libraries and convert to Parquet.",
    )
    parser.add_argument(
        "--library",
        choices=list(LIBRARIES.keys()),
        help="Library to fetch",
    )
    parser.add_argument(
        "--sublibrary",
        default="n",
        choices=["n", "p", "d", "t", "h", "a"],
        help="Sub-library / projectile type (default: n)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all libraries for the specified sub-library",
    )
    parser.add_argument(
        "--all-sublibs",
        action="store_true",
        help="Fetch all sub-libraries for the specified library(ies)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR,
        # Writes <output>/<library>/xs/. The repo root was the old default, which
        # put a fresh ingest in a tracked top-level directory instead of data/ (#341).
        # Help text derived from the default so the two cannot drift apart — the
        # old help said "repo root" and was accurate, which is how the surprise
        # stayed documented but unfixed.
        help=f"Output directory (default: {DATA_DIR.name}/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available libraries and their sub-libraries",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        for key, lib in LIBRARIES.items():
            sublibs = ", ".join(sorted(lib.sublibraries.keys()))
            print(f"  {key:20s}  {lib.name:20s}  sub-libs: {sublibs}")
        return

    if not args.library and not args.all:
        parser.error("Specify --library or --all")

    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (nuclear data research)"

    libs = list(LIBRARIES.keys()) if args.all else [args.library]

    for lib_key in libs:
        lib = LIBRARIES[lib_key]
        if args.all_sublibs:
            sublibs = sorted(lib.sublibraries.keys())
        else:
            sublibs = [args.sublibrary]

        for sublib in sublibs:
            if sublib not in lib.sublibraries:
                logger.info("Skipping %s/%s (not available)", lib.name, sublib)
                continue
            # A sweep must not ingest what the repo deliberately does not ship.
            # `--sublibrary n --all` would otherwise write endfb-8.1 neutron
            # parquets, retired in #263, into data/endfb-8.1/xs/ — a projectile
            # catalog.json does not list, which is the drift #356's test
            # forbids. Still reachable by naming it: --library endfb-8.1.
            #
            # `--all-sublibs` is a sweep too, and #360 guarded only `--all`
            # (#372). Every `rebuild_command` in catalog.json uses the
            # per-library form, so the rebuild drove the one path without the
            # skip and attempted iaea-medical/a — recorded right here as never
            # ingested. A sweep is "I did not name this sublibrary", however the
            # sweep is spelled.
            swept = args.all or args.all_sublibs
            if swept and (lib_key, sublib) in UNSHIPPED_SUBLIBRARIES:
                logger.info(
                    "Skipping %s/%s in a sweep: %s Fetch it explicitly with --library %s if you mean to.",
                    lib.name,
                    sublib,
                    UNSHIPPED_SUBLIBRARIES[lib_key, sublib],
                    lib_key,
                )
                continue
            fetch_library(lib_key, sublib, args.output, session)


if __name__ == "__main__":
    main()
