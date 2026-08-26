"""ENDF's interpolation laws — what `interp_law` means, and how to honour it (#338).

ENDF-6 does not tabulate a curve as points. It tabulates points **plus a law
saying how to get between them** (ENDF-102 `INT`). Keeping only the points and
discarding the law is the same defect this repository keeps finding: the
evaluator *stated* something and we threw the statement away, leaving the reader
to assume a convention. Every evaluated row we shipped before #338 silently
asserted "interpolate these however you like", and every consumer reaches for
`np.interp`.

## Why this is a column and not a view

`endf_mt` (#347) is reference data: "MT=1 sums MT=2 and MT=3" is an ENDF-102
*relation*, true of every evaluation ever written and derivable from the MT
number alone. Forty rows of it as a code-backed view, joined on MT, beats
copying a `redundant` flag onto seventeen million rows.

The interpolation law is not like that. It is **per-evaluation datum**, and it
varies *within a single MF=3 section*:

    TENDL-2023 p+Li-6, MF=3 MT=750:  breakpoints=[23, 134]  interpolation=[6, 5]

Points 1-23 are law 6, points 23-134 are law 5, in one section of one tape.
Nothing about MT=750 predicts that; the same MT is plain lin-lin in other
evaluations. So there is nothing to key a reference table on, and a per-region
table would need a region identity this schema does not have — and could still
not describe a row that is our own sum over several MTs.

Hence `interp_law` on the row: the ENDF `INT` code governing the interval that
*starts* at that row's energy. This module is the vocabulary that gives it
meaning, registered by `nucl_parquet.loader.connect()` as the `endf_interp`
view, exactly as `endf_mt` is:

    SELECT * FROM xs JOIN endf_interp USING (interp_law)
    WHERE kind='channel' AND NOT endf_interp.is_linear

## How wrong is ignoring it?

Measured against the tapes #335 restored, comparing `np.interp` to the stated
law at the midpoint of every tabulated interval:

| law | region | median error | worst |
|---|---|---|---|
| 5 (log-log) | h+Li-6 MT=112 | 0.02% | **30.2%** at 7.5 MeV |
| 6 (charged-particle threshold) | t+Li-6 MT=22 | **550%** | — |

Law 5 is a sparse-grid problem: almost always fine, occasionally 30% out. Law 6
is not — it is the near-threshold Coulomb-penetrability regime, where a linear
reading is wrong by *multiples*, and it is exactly the regime a (p,n) converter
foil operates in.

## NULL means "not stated", never "law 2"

A row whose source never carried a law gets NULL, not 2. The distinction is the
whole point: `endfb-8.0` comes from NJOY-reconstructed pointwise data where the
ENDF regions were already collapsed upstream, so no law survives to record.
Writing 2 there would manufacture an evaluator's statement that nobody made.
(That data is separately safe — `build_neutron_njoy.thin_pointwise` thins so the
grid reproduces the curve under both lin-lin and log-log — but "safe to read
linearly" and "the evaluator said linear" are different claims.)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

#: Histogram: y is constant across the interval.
HISTOGRAM = 1
#: Linear in x, linear in y. ENDF's default and the overwhelming majority.
LIN_LIN = 2
#: Linear in y, logarithmic in x.
LIN_LOG = 3
#: Logarithmic in y, linear in x.
LOG_LIN = 4
#: Logarithmic in both. The common non-default, especially in resonance regions.
LOG_LOG = 5
#: ENDF-102's special charged-particle threshold law — see `_interp_law6`.
CHARGED_PARTICLE_THRESHOLD = 6


class InterpLaw(NamedTuple):
    """One ENDF-102 `INT` value, and what a consumer must do about it."""

    #: The ENDF `INT` code. The primitive — stored, never derived.
    code: int
    #: Short name, as ENDF-102 spells the scheme.
    name: str
    #: True if x is interpolated logarithmically.
    log_x: bool
    #: True if y is interpolated logarithmically.
    log_y: bool
    #: True when plain `np.interp` is the correct reading. Only law 2 is.
    #: Law 1 is *not*: a histogram read linearly is wrong between every pair.
    is_linear: bool
    description: str


#: Every INT value ENDF-102 defines for one-dimensional interpolation.
#:
#: Laws 11-15 and 21-25 also exist in ENDF, but they are *two*-dimensional
#: (corresponding-point / unit-base interpolation between incident energies in
#: MF=4/5/6) and never appear on an MF=3 cross-section. They are deliberately
#: absent rather than mapped onto their one-dimensional cousins: a row claiming
#: INT=22 would be a row from a file this column does not describe.
INTERP_LAWS: dict[int, InterpLaw] = {
    HISTOGRAM: InterpLaw(
        HISTOGRAM,
        "histogram",
        log_x=False,
        log_y=False,
        is_linear=False,
        description="y is constant from this point to the next; y(x) = y_i",
    ),
    LIN_LIN: InterpLaw(
        LIN_LIN,
        "lin-lin",
        log_x=False,
        log_y=False,
        is_linear=True,
        description="y linear in x — the one law np.interp reads correctly",
    ),
    LIN_LOG: InterpLaw(
        LIN_LOG,
        "lin-log",
        log_x=True,
        log_y=False,
        is_linear=False,
        description="y linear in ln(x)",
    ),
    LOG_LIN: InterpLaw(
        LOG_LIN,
        "log-lin",
        log_x=False,
        log_y=True,
        is_linear=False,
        description="ln(y) linear in x",
    ),
    LOG_LOG: InterpLaw(
        LOG_LOG,
        "log-log",
        log_x=True,
        log_y=True,
        is_linear=False,
        description="ln(y) linear in ln(x)",
    ),
    CHARGED_PARTICLE_THRESHOLD: InterpLaw(
        CHARGED_PARTICLE_THRESHOLD,
        "charged-particle threshold",
        log_x=False,
        log_y=True,
        is_linear=False,
        description=(
            "ln(x*y) linear in 1/sqrt(x - T), T the kinematic threshold — the "
            "Coulomb-penetrability form; reading it linearly errs by multiples"
        ),
    ),
}


def interp_table() -> list[dict]:
    """Rows for the `endf_interp` view: one per law, keyed by `interp_law`."""
    return [
        {
            "interp_law": law.code,
            "name": law.name,
            "log_x": law.log_x,
            "log_y": law.log_y,
            "is_linear": law.is_linear,
            "description": law.description,
        }
        for law in sorted(INTERP_LAWS.values())
    ]


def is_valid_law(value: int | None) -> bool:
    """True if `value` may appear in an `interp_law` column. NULL always may."""
    return value is None or value in INTERP_LAWS


def laws_per_point(
    breakpoints: np.ndarray | list[int],
    interpolation: np.ndarray | list[int],
    n_points: int,
) -> np.ndarray:
    """Expand ENDF's (NBT, INT) region arrays to one law per tabulated point.

    The returned law at index `i` governs the interval **starting** at point `i`
    — which is the only reading that makes the value useful on a row, since a
    row is one point and what a consumer needs to know is how to get to the next
    one. The final point has no following interval and carries the last region's
    law, so that a reversed or clipped scan still finds a value.

    ## The off-by-one this function exists to get right

    `NBT(k)` is the 1-based index of the **last** point of region k, so
    consecutive regions *share* a point. The obvious implementation —

        out[start:nbt] = law   # then start = nbt

    — gives that shared point to the **lower** region, and is wrong on every
    multi-region section. Checked against the `endf` package's own region
    selection on six real two-region sections (TENDL-2023 p+Li-6 MT=750,
    t+Li-6 MT=22/650, h+Li-6 MT=112/650, d+Li-7 MT=22), it disagrees at exactly
    one index each time — the boundary — and gets law 6 where the answer is 5.

    That is the worst possible place to be wrong. The boundary is where the
    evaluator changed law *because the curve changes shape there*.

    The correct rule, which `endf.function.Tabulated1D._interpolate_scalar`
    also implements, is: the interval starting at 0-based `i` belongs to the
    first region `k` with `i < NBT(k) - 1`.
    """
    nbt = np.asarray(breakpoints, dtype=np.int64)
    ints = np.asarray(interpolation, dtype=np.int64)
    if nbt.size == 0 or ints.size == 0:
        raise ValueError("a TAB1 record always declares at least one region")
    if nbt.size != ints.size:
        raise ValueError(f"NBT has {nbt.size} regions but INT has {ints.size}")

    # `searchsorted(nbt - 1, i, 'right')` is the number of breakpoints whose
    # (0-based) last-interval index is <= i, i.e. the index of the first region
    # with i < NBT(k) - 1. Exactly the loop above, vectorised.
    idx = np.searchsorted(nbt - 1, np.arange(n_points), side="right")
    return ints[np.clip(idx, 0, ints.size - 1)]


def _interp_law6(x: float, x1: float, x2: float, y1: float, y2: float, threshold: float) -> float:
    """ENDF-102's charged-particle threshold law: ln(x*y) linear in 1/sqrt(x-T).

    Derived from the limiting form of the Coulomb penetrability, so it is
    concave-upward near T in a way no neutron-shaped law reproduces. ENDF-102
    warns it is only meant for x close to T; above that, evaluations switch
    region (in practice to law 5, which is what every observed tape does).

    **Validated empirically, not recalled.** Leave-one-out over the law-6 region
    of TENDL-2023 p+Li-6 MT=750 (23 points): predicting each interior point from
    its two neighbours gives a median error of **0.02%** under this formula,
    against 1.68% for log-log and 13.78% for lin-lin. On a region that
    well-sampled, 0.02% means this *is* the law the evaluator used. The `endf`
    package does not implement law 6 at all — `_interpolate_scalar` falls off
    the end of its if-chain and returns `None` — so there was no reference to
    copy and the formula had to be earned against real data.
    """
    if x <= threshold or x1 <= threshold or x2 <= threshold or y1 <= 0 or y2 <= 0:
        raise ValueError(
            f"law 6 is undefined at x={x} for the interval [{x1}, {x2}] with threshold T={threshold}: "
            "it needs x > T on both endpoints and strictly positive y"
        )
    b1, b2 = 1.0 / math.sqrt(x1 - threshold), 1.0 / math.sqrt(x2 - threshold)
    if b1 == b2:
        return y1
    a1, a2 = math.log(x1 * y1), math.log(x2 * y2)
    b = 1.0 / math.sqrt(x - threshold)
    return math.exp(a1 + (a2 - a1) * (b - b1) / (b2 - b1)) / x


def interpolate_one(law: int, x: float, x1: float, x2: float, y1: float, y2: float, threshold: float = 0.0) -> float:
    """Evaluate the curve at `x` inside the interval [`x1`, `x2`] under `law`.

    `threshold` is only consulted for law 6, where it is ENDF's `T` — 0 for an
    exothermic reaction and the kinematic threshold otherwise. Callers reading a
    tape should pass the first energy of the law-6 region, which is where the
    evaluator anchors it.

    Raises on an unknown law rather than falling through to linear. Silently
    defaulting to lin-lin is the behaviour #338 exists to remove, and putting it
    back inside the fix would be a joke at the reader's expense.
    """
    if law not in INTERP_LAWS:
        raise ValueError(f"unknown ENDF interpolation law {law!r}; known laws are {sorted(INTERP_LAWS)}")
    if law == HISTOGRAM:
        return y1
    if law == LIN_LIN:
        return y1 + (x - x1) / (x2 - x1) * (y2 - y1)
    if law == LIN_LOG:
        return y1 + math.log(x / x1) / math.log(x2 / x1) * (y2 - y1)
    if law == LOG_LIN:
        return y1 * math.exp((x - x1) / (x2 - x1) * math.log(y2 / y1))
    if law == LOG_LOG:
        return y1 * math.exp(math.log(x / x1) / math.log(x2 / x1) * math.log(y2 / y1))
    return _interp_law6(x, x1, x2, y1, y2, threshold)


def interpolate(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    laws: np.ndarray | list[int],
    at: np.ndarray | list[float],
    threshold: float = 0.0,
) -> np.ndarray:
    """Evaluate a tabulated curve at `at`, honouring each interval's own law.

    `laws[i]` governs the interval starting at `x[i]` — the same convention
    `laws_per_point` produces and the same one the `interp_law` column carries,
    so a consumer can read a shard and call this directly:

        df = pl.read_parquet(shard).filter(...).sort("energy_MeV")
        sigma = interpolate(df["energy_MeV"], df["xs_mb"], df["interp_law"], grid)

    Outside the tabulated range the result is 0, matching how ENDF treats a
    cross-section below threshold or above the evaluation's top energy.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    ls = np.asarray(laws)
    query = np.asarray(at, dtype=float)
    if xs.size != ys.size or xs.size != ls.size:
        raise ValueError(f"x, y and laws must be the same length, got {xs.size}, {ys.size}, {ls.size}")
    if xs.size == 0:
        return np.zeros_like(query)

    out = np.zeros_like(query)
    inside = (query >= xs[0]) & (query <= xs[-1])
    idx = np.clip(np.searchsorted(xs, query, side="right") - 1, 0, xs.size - 2)
    for k in np.nonzero(inside)[0]:
        i = int(idx[k])
        law = ls[i]
        if law is None or (isinstance(law, float) and math.isnan(law)):
            raise ValueError(
                f"no interpolation law for the interval starting at x={xs[i]}. "
                "A NULL interp_law means the source never stated one — decide what to "
                "assume rather than having this function assume for you."
            )
        out[k] = interpolate_one(
            int(law), float(query[k]), float(xs[i]), float(xs[i + 1]), float(ys[i]), float(ys[i + 1]), threshold
        )
    return out
