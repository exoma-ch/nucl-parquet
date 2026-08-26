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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))  # so `nucl_parquet` imports from the checkout

from nucl_parquet.builder_stamp import write_builder_stamp  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

IAEA_MIRROR = "https://nds.iaea.org/public/download-endf"
COMPRESSION = "zstd"

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
        sublibraries={"n": "n", "p": "p", "d": "d", "h": "he3", "a": "he4"},
    ),
}


# ---------------------------------------------------------------------------
# Element data
# ---------------------------------------------------------------------------

_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
}

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
#   MT 51-91:   (x,n') to specific levels, plus the n continuum at 91
#   MT 600-649: (x,p) levels, 650-699 (x,d), 700-749 (x,t),
#   MT 750-799: (x,³He), 800-849 (x,α), 875-891 (x,2n)
LEVEL_RANGE_PARTICLES: dict[tuple[int, int], tuple[str, ...]] = {
    (51, 91): ("n",),
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
#
# Two alternatives rather than one increasingly baroque pattern, and an
# explicit isomer group so metastable targets stop being discarded.
_FN_ZFIRST = re.compile(r"[a-z]+_(\d{1,3})-([A-Za-z]+)-(\d+)([A-Za-z]\d?)?_(\d+)\.zip")
_FN_MATFIRST = re.compile(r"[a-z]+_(\d+)_(\d{1,3})-([A-Za-z]+)-(\d+)([A-Za-z]\d?)?\.zip")


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


def parse_mf10_rows(
    material,  # noqa: ANN001 — endf.Material, imported lazily by the caller
    target_a: int,
) -> tuple[list[dict], int]:
    """Extract MF=10 isomeric-production rows. Returns (rows, sections_seen).

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
        return [], 0

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
                "residual_Z": prod_z,
                "residual_A": prod_a,
                "state": state,
                "energy_MeV": float(e_ev) * 1e-6,
                "xs_mb": float(xs_b) * 1e3,
            }
            for e_ev, xs_b in zip(e_union, xs_total)
        )
    return rows, len(sections)


def parse_endf_file(
    endf_text: str,
    target_z: int,
    target_a: int,
    projectile: str,
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

    for (mf, mt), section in material.section_data.items():
        if mf != 3:
            continue
        if mt in skip_partials:
            continue

        residual = mt_to_residual(mt, target_z, target_a, proj_z, proj_a)
        if residual is None:
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

        # Drop TALYS overflow sentinels (~1.99e35 b) and non-positive values.
        good = np.isfinite(xs_barns) & (xs_barns > 0) & (xs_barns <= _XS_MAX_BARNS)
        if not good.any():
            continue
        by_residual.setdefault(residual, []).append((energies_ev[good], xs_barns[good]))

    for (res_z, res_a), contribs in by_residual.items():
        e_union, xs_total = sum_on_union_grid(contribs)
        for e_ev, xs_b in zip(e_union, xs_total):
            rows.append(
                {
                    "target_A": target_a,
                    "residual_Z": res_z,
                    "residual_A": res_a,
                    # '' is "summed over whatever states this channel populates",
                    # which MF=10 spells 'g'/'m'/'m2' — see lfs_to_state.
                    "state": "",
                    "energy_MeV": float(e_ev) * 1e-6,
                    "xs_mb": float(xs_b) * 1e3,
                }
            )

    mf3_rows = len(rows)
    mf10_rows, mf10_sections = parse_mf10_rows(material, target_a)
    rows.extend(mf10_rows)

    return ParsedFile(
        rows=rows,
        mf3_sections=len(mf3_mts),
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
    """Get list of zip filenames from the IAEA mirror directory."""
    sublib_dir = lib.sublibraries.get(sublib_code)
    if sublib_dir is None:
        return []

    url = f"{IAEA_MIRROR}/{lib.iaea_path}/{sublib_dir}/"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to list %s: %s", url, e)
        return []

    # Parse HTML directory listing for .zip filenames
    filenames = re.findall(r'href="([^"]+\.zip)"', resp.text)
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

    target_z, target_a, _isomer = parsed

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

    return parse_endf_file(endf_text, target_z, target_a, sublib_code)


def fetch_library(
    lib_key: str,
    sublib_code: str,
    output_dir: Path,
    session: requests.Session,
) -> None:
    """Fetch and convert an entire sub-library to Parquet."""
    lib = LIBRARIES[lib_key]

    if sublib_code not in lib.sublibraries:
        logger.error("%s does not have sub-library '%s'", lib.name, sublib_code)
        return

    logger.info("Fetching %s / %s ...", lib.name, sublib_code)

    filenames = list_endf_files(lib, sublib_code, session)
    if not filenames:
        logger.warning("No files found for %s/%s", lib.name, sublib_code)
        return

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

    for i, fname in enumerate(filenames):
        if (i + 1) % 50 == 0:
            logger.info("  Processing %d/%d ...", i + 1, len(filenames))

        parsed_file = download_and_parse(lib, sublib_code, fname, session)
        mf3_sections += parsed_file.mf3_sections
        mf3_rows += parsed_file.mf3_rows
        mf10_sections += parsed_file.mf10_sections
        mf10_rows += parsed_file.mf10_rows
        if not parsed_file.rows:
            continue

        parsed = parse_endf_filename(fname)
        if parsed is None:
            continue

        target_z = parsed[0]
        elem = _ELEMENT_SYMBOLS.get(target_z, f"Z{target_z}")
        element_rows.setdefault(elem, []).extend(parsed_file.rows)
        total_files += 1
        total_rows += len(parsed_file.rows)

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
    # an exit code. Both raise before anything is written.
    #
    # MF=10 (#340): reading it through an object shape the `endf` package never
    # returned discarded every isomeric section in every library, and the #334
    # guard could not see it because MF=3 kept producing plenty of rows.
    if mf10_sections and not mf10_rows:
        raise RuntimeError(
            f"{lib_key}/{sublib_code}: {mf10_sections} MF=10 isomeric-production "
            f"sections were read from the source files and produced 0 rows. "
            "That is the #340 signature — the `endf` package's MF=10 shape has "
            "most likely moved again (see parse_mf10_rows). Refusing to report "
            "success while silently dropping every isomeric product."
        )

    # MF=3, symmetrically. This became possible to miss only once MF=10 started
    # emitting rows: before #340 a dead MF=3 read meant a dead library and the
    # empty-ingest guard caught it, whereas now MF=10 alone could keep the
    # element count up while every channel cross-section vanished.
    if mf3_sections and not mf3_rows:
        raise RuntimeError(
            f"{lib_key}/{sublib_code}: {mf3_sections} MF=3 cross-section sections "
            f"were read from the source files and produced 0 rows. Either the "
            "`endf` package no longer returns the curve under 'sigma', or "
            "mt_to_residual now rejects every MT. Refusing to report success on "
            "a library with no channel cross-sections."
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
        df = pl.DataFrame(
            rows,
            schema={
                "target_A": pl.Int32,
                "residual_Z": pl.Int32,
                "residual_A": pl.Int32,
                "state": pl.Utf8,
                "energy_MeV": pl.Float64,
                "xs_mb": pl.Float64,
            },
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
        df = df.sort("target_A", "residual_Z", "residual_A", "state", "energy_MeV")

        out_path = xs_dir / f"{sublib_code}_{elem}.parquet"
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
        "mf3_rows": mf3_rows,
        "mf10_sections": mf10_sections,
        "mf10_rows": mf10_rows,
        "states": dict(sorted(state_counts.items())),
    }
    manifest_path = output_dir / lib_key / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_builder_stamp(manifest_path, Path(__file__), files_written=len(element_rows))

    logger.info(
        "  Done: %d elements, %d source files, %d total rows (%d MF=10 sections → %d isomeric rows; states %s) → %s/",
        len(element_rows),
        total_files,
        total_rows,
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
            fetch_library(lib_key, sublib, args.output, session)


if __name__ == "__main__":
    main()
