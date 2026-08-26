"""What each ENDF MT number means, and which MTs are sums of which (#347).

Once `kind='channel'` rows exist, a cross-section table contains both MT=1
(total) and MT=2 (elastic) for the same target. `SUM(xs_mb) GROUP BY target` is
then nonsense — MT=1 already contains MT=2 — and nothing in the row says so.
This is the same class of trap as `state=''` (#340): a plausible number from a
reasonable query.

The fix is *not* a `redundant` column. That fact is a property of the MT number,
not of the datum: 17 million rows would each carry a copy of a forty-row table,
and `CLAUDE.md` principle 4 ("long, never wide") argues against exactly that. So
the redundancy relation is reference data, keyed by MT, joinable:

    SELECT * FROM xs JOIN endf_mt USING (MT) WHERE NOT endf_mt.redundant

`nucl_parquet.loader.connect()` registers this as the `endf_mt` view, built from
the table below — no parquet file, so it is queryable from a plain checkout.

Redundancy is ENDF-102's own, from Appendix B ("Summary of ENDF/B Reaction
Types") and section 3.2 on redundant reactions. `sums_over` names the MTs a
redundant MT aggregates; an evaluation need not ship all of them, so a JOIN
tells you what *could* double-count, not what does.
"""

from __future__ import annotations

# --- Redundant MTs: MT -> the MTs it sums over --------------------------------
#
# Only the relations ENDF-102 defines by construction. MT=103–107 and MT=16 are
# conditionally redundant — redundant exactly when the evaluation also ships
# their discrete-level partials — and are listed because a table carrying both
# double-counts just as badly.
_LEVELS = tuple(range(51, 92))  # MT=50..91 discrete inelastic; 91 is continuum

REDUNDANT_MT: dict[int, tuple[int, ...]] = {
    1: (2, 3),  # total = elastic + nonelastic
    3: (4, 5, 11, 16, 17, 18, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 41, 42, 44, 45, 101),
    4: _LEVELS,  # inelastic = sum over discrete levels + continuum
    18: (19, 20, 21, 38),  # fission = 1st + 2nd + 3rd + 4th chance
    27: (18, 101),  # absorption = fission + disappearance
    101: (102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117),
    16: tuple(range(875, 892)),  # (n,2n) = sum over its discrete levels
    103: tuple(range(600, 650)),  # (n,p)
    104: tuple(range(650, 700)),  # (n,d)
    105: tuple(range(700, 750)),  # (n,t)
    106: tuple(range(750, 800)),  # (n,3He)
    107: tuple(range(800, 850)),  # (n,a)
}

# --- Particle production: not reaction cross-sections at all -------------------
#
# MT=201–207 are multiplicity-weighted particle *yields* (sigma x number of that
# particle emitted), so a single reaction contributes more than once and the row
# does not count reactions. Summing them with reaction MTs is a category error,
# not merely double-counting.
PARTICLE_PRODUCTION_MT: dict[int, str] = {
    201: "(z,Xn) neutron production",
    202: "(z,Xg) photon production",
    203: "(z,Xp) proton production",
    204: "(z,Xd) deuteron production",
    205: "(z,Xt) triton production",
    206: "(z,X3He) helion production",
    207: "(z,Xa) alpha production",
}

# --- Names ---------------------------------------------------------------------
_NAMES: dict[int, str] = {
    1: "(n,total)",
    2: "(z,elastic)",
    3: "(z,nonelastic)",
    4: "(z,n')",
    5: "(z,anything)",
    11: "(z,2nd)",
    16: "(z,2n)",
    17: "(z,3n)",
    18: "(z,fission)",
    19: "(z,f) first chance",
    20: "(z,nf) second chance",
    21: "(z,2nf) third chance",
    22: "(z,na)",
    23: "(z,n3a)",
    24: "(z,2na)",
    25: "(z,3na)",
    27: "(z,absorption)",
    28: "(z,np)",
    29: "(z,n2a)",
    30: "(z,2n2a)",
    32: "(z,nd)",
    33: "(z,nt)",
    34: "(z,nh)",
    35: "(z,nd2a)",
    36: "(z,nt2a)",
    37: "(z,4n)",
    38: "(z,3nf) fourth chance",
    41: "(z,2np)",
    42: "(z,3np)",
    44: "(z,n2p)",
    45: "(z,npa)",
    101: "(z,disappearance)",
    102: "(z,gamma)",
    103: "(z,p)",
    104: "(z,d)",
    105: "(z,t)",
    106: "(z,3He)",
    107: "(z,a)",
    108: "(z,2a)",
    109: "(z,3a)",
    111: "(z,2p)",
    112: "(z,pa)",
    113: "(z,t2a)",
    114: "(z,d2a)",
    115: "(z,pd)",
    116: "(z,pt)",
    117: "(z,da)",
    # The 152-200 block, which real evaluations do use — IRDFF-II ships MT=152
    # and MT=153 on its dosimetry targets.
    152: "(z,5n)",
    153: "(z,6n)",
    154: "(z,2nt)",
    155: "(z,ta)",
    156: "(z,4np)",
    157: "(z,3nd)",
    158: "(z,nda)",
    159: "(z,2npa)",
    160: "(z,7n)",
    161: "(z,8n)",
    162: "(z,5np)",
    163: "(z,6np)",
    164: "(z,7np)",
    165: "(z,4na)",
    166: "(z,5na)",
    167: "(z,6na)",
    168: "(z,7na)",
    169: "(z,4nd)",
    170: "(z,5nd)",
    171: "(z,6nd)",
    172: "(z,3nt)",
    173: "(z,4nt)",
    174: "(z,5nt)",
    175: "(z,6nt)",
    176: "(z,2nh)",
    177: "(z,3nh)",
    178: "(z,4nh)",
    179: "(z,3n2p)",
    180: "(z,3n2a)",
    181: "(z,3npa)",
    182: "(z,dt)",
    183: "(z,npd)",
    184: "(z,npt)",
    185: "(z,ndt)",
    186: "(z,nph)",
    187: "(z,ndh)",
    188: "(z,nth)",
    189: "(z,nta)",
    190: "(z,2n2p)",
    191: "(z,ph)",
    192: "(z,dh)",
    193: "(z,ha)",
    194: "(z,4n2p)",
    195: "(z,4n2a)",
    196: "(z,4npa)",
    197: "(z,3p)",
    198: "(z,n3p)",
    199: "(z,3n2pa)",
    200: "(z,5n2p)",
}

# Discrete-level ranges: (lo, hi, emitted-particle label). `hi` is the continuum
# MT, which ENDF reserves as the last of each block.
_LEVEL_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (50, 91, "n"),
    (600, 649, "p"),
    (650, 699, "d"),
    (700, 749, "t"),
    (750, 799, "3He"),
    (800, 849, "a"),
    (875, 891, "2n"),
)


def mt_name(mt: int) -> str:
    """Human-readable reaction name, including the discrete-level blocks."""
    if mt in _NAMES:
        return _NAMES[mt]
    if mt in PARTICLE_PRODUCTION_MT:
        return PARTICLE_PRODUCTION_MT[mt]
    for lo, hi, particle in _LEVEL_BLOCKS:
        if lo <= mt <= hi:
            if mt == hi:
                return f"(z,{particle}) continuum"
            return f"(z,{particle}) level {mt - lo}"
    return f"MT={mt}"


def mt_table() -> list[dict]:
    """Every MT this repo knows about, as rows for the `endf_mt` view.

    Columns: MT, name, redundant, sums_over, particle_production.
    """
    known: set[int] = set(_NAMES) | set(PARTICLE_PRODUCTION_MT) | set(REDUNDANT_MT)
    for lo, hi, _p in _LEVEL_BLOCKS:
        known.update(range(lo, hi + 1))

    return [
        {
            "MT": mt,
            "name": mt_name(mt),
            "redundant": mt in REDUNDANT_MT,
            "sums_over": list(REDUNDANT_MT.get(mt, ())),
            "particle_production": mt in PARTICLE_PRODUCTION_MT,
        }
        for mt in sorted(known)
    ]


def exclusive_mts(present: set[int]) -> set[int]:
    """Of `present`, the MTs safe to sum: neither redundant over another member
    nor a particle-production yield.

    An MT is dropped only when at least one MT it sums over is *also* present —
    an evaluation that ships MT=103 but none of MT=600-649 is not double-counting
    anything, and its (n,p) channel must still be summable.
    """
    out = set()
    for mt in present:
        if mt in PARTICLE_PRODUCTION_MT:
            continue
        if any(c in present for c in REDUNDANT_MT.get(mt, ())):
            continue
        out.add(mt)
    return out
