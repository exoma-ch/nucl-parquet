"""Synthesize X-ray and Auger emission rows from G4EMLOW × per-shell EC/IC.

This module convolves three independent inputs into a per-(Z, A, state)
``rad_type ∈ {'xray', 'auger'}`` table appended (post-process) to
``meta/ensdf/radiation/{Symbol}.parquet``:

1. **EADL atomic-relaxation tables** (``data/meta/eadl/{Symbol}.parquet``):
   per-element radiative + non-radiative transition probabilities, normalized
   so ``Σ probability per vacancy = 1``.

2. **Per-shell EC fractions** (``data/meta/decay_detailed.parquet``, output of
   :mod:`nucl_parquet.g4.radioactive_decay`): ``decay_mode`` ∈
   ``{KshellEC, LshellEC, MshellEC, NshellEC}`` carrying per-shell ``branching``
   summed across daughter levels.

3. **Internal-conversion totals** (``data/g4_raw/strata-nuclear/photon_evap_gammas.parquet``):
   ``icc_total`` per gamma transition; we approximate per-shell IC vacancies
   as **K-shell only** (per-shell partial ICCs absent from strata).

Output schema (matches ``radiation/`` v0.10.x columns + new optional ones):
    Z, A, state, rad_type, energy_keV, intensity_pct,
    decay_mode, parent_level_keV, rad_subtype

Per HANDOFF.md guidance, the synthesized rows are written to a sibling
``data/meta/ensdf/radiation_atomic/{Symbol}.parquet`` directory; a follow-up
PR (#75 validation harness) will merge them into the canonical ``radiation/``
files alongside #72's gamma rows.

Design memo: ``docs/g4-xray-auger-design.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# v0.10.x state-fuzzy-match tolerance (matches radioactive_decay.py).
_STATE_KEV_TOL: Final[float] = 1.0

# Energy-conservation slack for the per-vacancy intensity-sum invariant.
_ENERGY_SLACK_PCT: Final[float] = 5.0

# EADL maps: shell-letter sub-shells to roll-up labels.
_SHELL_LETTER: Final[dict[str, str]] = {
    "K": "K",
    "L1": "L",
    "L2": "L",
    "L3": "L",
    "M1": "M",
    "M2": "M",
    "M3": "M",
    "M4": "M",
    "M5": "M",
    "N1": "N",
    "N2": "N",
    "N3": "N",
    "N4": "N",
    "N5": "N",
    "N6": "N",
    "N7": "N",
    "O1": "O",
    "O2": "O",
    "O3": "O",
    "O4": "O",
    "O5": "O",
    "O6": "O",
    "O7": "O",
    "P1": "P",
    "P2": "P",
    "P3": "P",
    "P4": "P",
    "P5": "P",
    "Q1": "Q",
}

# Canonical Siegbahn / IUPAC X-ray line labels by (vacancy, filling).
# Source: IUPAC nomenclature for X-ray spectroscopy, J. M. Hollas (mnemonics);
# matched against EADL energies for Fe/Tc/Te to verify line ordering.
# Where EADL exposes a sub-shell pair we don't have a canonical Siegbahn name
# for (e.g. exotic Lγ lines), we fall back to "{vacancy}-{filling}" notation.
_XRAY_LABEL: Final[dict[tuple[str, str], str]] = {
    # K-series
    ("K", "L2"): "Kα2",
    ("K", "L3"): "Kα1",
    ("K", "M2"): "Kβ3",
    ("K", "M3"): "Kβ1",
    ("K", "M4"): "Kβ5",
    ("K", "M5"): "Kβ5",
    ("K", "N2"): "Kβ2",
    ("K", "N3"): "Kβ2",
    ("K", "N4"): "Kβ4",
    ("K", "N5"): "Kβ4",
    ("K", "O2"): "Kβ",
    ("K", "O3"): "Kβ",
    # L1-series (Lβ3, Lβ4, Lγ2, Lγ3)
    ("L1", "M2"): "Lβ4",
    ("L1", "M3"): "Lβ3",
    ("L1", "M4"): "Lβ10",
    ("L1", "M5"): "Lβ9",
    ("L1", "N2"): "Lγ2",
    ("L1", "N3"): "Lγ3",
    ("L1", "O2"): "Lγ4",
    ("L1", "O3"): "Lγ4'",
    # L2-series
    ("L2", "M1"): "Lη",
    ("L2", "M4"): "Lβ1",
    ("L2", "N1"): "Lγ5",
    ("L2", "N4"): "Lγ1",
    ("L2", "O1"): "Lγ8",
    ("L2", "O4"): "Lγ6",
    # L3-series
    ("L3", "M1"): "Lℓ",
    ("L3", "M2"): "Lt",
    ("L3", "M4"): "Lα2",
    ("L3", "M5"): "Lα1",
    ("L3", "N1"): "Lβ6",
    ("L3", "N4"): "Lβ15",
    ("L3", "N5"): "Lβ2",
    ("L3", "O1"): "Lβ7",
    ("L3", "O4"): "Lβ5",
    ("L3", "O5"): "Lβ5",
    # M-series (rough — IUPAC names get sparse here)
    ("M3", "N5"): "Mγ",
    ("M4", "N6"): "Mβ",
    ("M5", "N6"): "Mα2",
    ("M5", "N7"): "Mα1",
}

# v0.11 approximation: EC L/M/N totals attributed to L1/M1/N1 vacancies for
# atomic-relaxation lookup. (Sub-shell branchings absent from strata input.)
_SHELL_TOTAL_TO_VACANCY: Final[dict[str, str]] = {
    "K": "K",
    "L": "L1",
    "M": "M1",
    "N": "N1",
}


# -----------------------------------------------------------------------------
# Pure helpers
# -----------------------------------------------------------------------------


def xray_subtype(vacancy: str, filling: str) -> str:
    """Return canonical Siegbahn label for an X-ray (vacancy → filling).

    Falls back to ``"{vacancy}-{filling}"`` for obscure transitions not in the
    Siegbahn table.
    """
    return _XRAY_LABEL.get((vacancy, filling), f"{vacancy}-{filling}")


def auger_subtype(vacancy: str, filling: str) -> str:
    """Return three-letter Auger label aggregating to shell-level.

    EADL stores Auger transitions tagged by (vacancy, filling) only; the third
    shell (the one the ejected electron originates from) varies row-by-row
    distinguishable only by ``energy_keV``. We label by the two-letter
    ``vacancy_letter + filling_letter`` and append the *most likely* third
    shell, which is the same as the filling shell for KLL, KMM, LMM family
    and the higher-letter shell for cross-family (KLM, LMM…).

    For the v0.11 granularity we collapse to the two-letter prefix +
    filling-letter as the typical third shell. Not the full atomic-physics
    accounting, but matches v0.10.x's ``"Auger K"`` / ``"Auger L"`` granularity
    and is consistent with the EADL encoding.
    """
    vl = _SHELL_LETTER.get(vacancy, vacancy)
    fl = _SHELL_LETTER.get(filling, filling)
    # Heuristic: third electron ejected from the same shell-letter as filling
    # (most KLL transitions involve two L electrons; KLM = L electron fills,
    # M electron ejected). When vl != fl we use vl + fl + fl as the dominant
    # group (e.g. K + L + L = KLL), upgraded to vl + fl + (next shell) when
    # filling is already the highest in scope.
    return f"{vl}{fl}{fl}"


def map_decay_mode_to_shell(decay_mode: str) -> str | None:
    """KshellEC → 'K', LshellEC → 'L', MshellEC → 'M', NshellEC → 'N'."""
    if not decay_mode.endswith("shellEC"):
        return None
    return decay_mode[: -len("shellEC")]


def assign_state(
    z: int,
    a: int,
    parent_ex_kev: float,
    nuclides: pl.DataFrame,
) -> str | None:
    """Map (Z, A, parent_ex_kev) → v0.10.x state label using ±1 keV fuzzy match.

    Returns:
        - ``""`` if ``parent_ex_kev`` is 0
        - ``"m"`` / ``"m2"`` / … if a catalog isomer matches within ±1 keV
        - The single m-isomer's label if the catalog has an m-row with
          ``level_keV`` NULL and ``parent_ex_kev > 0`` (Ba-137m fallback —
          nuclides.parquet from G4ENSDFSTATE doesn't always carry the
          m-state energy for IT-only isomers)
        - ``None`` if excited but no catalog match (skip such parents)
    """
    if parent_ex_kev == 0.0:
        return ""
    za_isomers = nuclides.filter((pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") != ""))
    if za_isomers.is_empty():
        return None
    candidates = za_isomers.filter(
        pl.col("level_keV").is_not_null() & ((pl.col("level_keV") - parent_ex_kev).abs() <= _STATE_KEV_TOL)
    )
    if not candidates.is_empty():
        # Closest match wins (matches radioactive_decay.py convention).
        closest = (
            candidates.with_columns((pl.col("level_keV") - parent_ex_kev).abs().alias("__delta"))
            .sort("__delta")
            .row(0, named=True)
        )
        return closest["state"]
    # Fallback for catalog-known isomers with missing level_keV: if there
    # is exactly one m-row and its level_keV is NULL, accept any positive
    # excitation as that isomer. (Ba-137m is the canonical case.)
    null_iso = za_isomers.filter(pl.col("level_keV").is_null())
    if null_iso.height == 1:
        return null_iso.row(0, named=True)["state"]
    return None


# -----------------------------------------------------------------------------
# Vacancy rate computation
# -----------------------------------------------------------------------------


def compute_ec_vacancy_rates(
    decay_detailed: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate per-shell EC branchings into vacancy rates per parent.

    Returns DataFrame: (Z, A, parent_ex_kev, daughter_Z, shell, vacancy_rate).
    ``shell`` ∈ {K, L, M, N}; ``daughter_Z = Z - 1`` (the atom in which the
    vacancy is created).
    """
    ec = decay_detailed.filter(pl.col("decay_mode").str.ends_with("shellEC"))
    if ec.is_empty():
        return pl.DataFrame(
            schema={
                "Z": pl.Int32,
                "A": pl.Int32,
                "parent_ex_kev": pl.Float64,
                "daughter_Z": pl.Int32,
                "shell": pl.Utf8,
                "vacancy_rate": pl.Float64,
            }
        )
    return (
        ec.with_columns(pl.col("decay_mode").str.replace("shellEC", "").alias("shell"))
        .group_by(["Z", "A", "parent_ex_kev", "daughter_Z", "shell"])
        .agg(pl.col("branching").sum().alias("vacancy_rate"))
        .sort(["Z", "A", "parent_ex_kev", "shell"])
    )


def _propagate_cascade(
    z: int,
    a: int,
    isomer_level_idx: int,
    gammas_for_za: pl.DataFrame,
) -> dict[int, float]:
    """Iterate cascade populations downward starting from isomer_level_idx.

    Returns {level_idx → cumulative population}. Each level's population is
    the fraction of isomer-decay events that pass through that level.

    Algorithm: BFS-like. Start with pop[isomer]=1. At each step, find the
    highest-populated source level not yet processed, distribute its
    population across daughter levels weighted by transition branch
    fraction (intensity × (1+icc) renormalized within the source level),
    and add to daughter populations. Repeat until no level above ground
    has unprocessed population.
    """
    if gammas_for_za.is_empty():
        return {isomer_level_idx: 1.0}

    # Pre-compute transition branch fractions per source level.
    g = gammas_for_za.with_columns(
        (pl.col("intensity") * (1.0 + pl.col("icc_total"))).alias("__trans_rate"),
    ).with_columns(
        (pl.col("__trans_rate") / pl.col("__trans_rate").sum().over(["level_idx"])).alias("transition_frac"),
    )

    # Group transitions by source level for fast lookup.
    by_source: dict[int, list[tuple[int, float]]] = {}
    for r in g.iter_rows(named=True):
        by_source.setdefault(int(r["level_idx"]), []).append((int(r["daughter_level"]), float(r["transition_frac"])))

    pop: dict[int, float] = {isomer_level_idx: 1.0}
    processed: set[int] = set()
    # Process levels in descending order of level_idx (assumed monotonic
    # with excitation in strata's encoding). If the assumption breaks for
    # some odd file the loop still terminates because we mark each level
    # processed.
    while True:
        candidates = [lvl for lvl in pop if lvl not in processed and lvl > 0]
        if not candidates:
            break
        src = max(candidates)
        processed.add(src)
        src_pop = pop[src]
        if src_pop <= 0:
            continue
        for dst, frac in by_source.get(src, []):
            pop[dst] = pop.get(dst, 0.0) + src_pop * frac
    return pop


def compute_ic_vacancy_rates(
    photon_evap_gammas: pl.DataFrame,
    photon_evap_levels: pl.DataFrame,
    parent_states: pl.DataFrame,
    k_edges: dict[int, float] | None = None,
) -> pl.DataFrame:
    """Approximate per-(Z, A, state) IC vacancy rates (K-shell only).

    Restricted to **the depopulating gammas of the isomer level itself** —
    we don't propagate cascade populations through subsequent levels. This
    is a v0.11.x simplification (see design memo §"Known gaps"):

    - Avoids over-counting K vacancies when the same lower level appears in
      many cascade paths under the isomer.
    - Captures the dominant IC for canonical isomers when the level the
      strata catalog matches as ``parent_ex_kev`` is the depopulating one.
    - For ground-state EC parents, IC of post-EC daughter gammas is not
      yet folded in (see design memo §"Known gaps", item 4).

    The yield is computed as
    ``Σ_g (intensity_g / 100) × icc_g / (1 + icc_g)``
    over all gammas whose ``parent_level`` equals the isomer's level_idx
    (resolved by ±1 keV match against ``photon_evap_levels``). Cascade
    propagation (a daughter level emitting its own IC-converted gamma) is
    intentionally skipped — that path is a v0.12 follow-up.

    Args:
        photon_evap_gammas: Strata's gamma table (z, a, parent_level,
            daughter_level, gamma_energy_kev, intensity, icc_total, …).
        photon_evap_levels: Strata's level table (z, a, level_idx,
            excitation_kev, half_life_s, …).
        parent_states: DataFrame of (Z, A, state, parent_ex_kev) for the
            parents we want IC vacancies for.

    Returns:
        DataFrame: (Z, A, parent_ex_kev, daughter_Z, shell='K', vacancy_rate).
        Daughter Z = parent Z (IC happens in same atom).
    """
    empty_schema = {
        "Z": pl.Int32,
        "A": pl.Int32,
        "parent_ex_kev": pl.Float64,
        "daughter_Z": pl.Int32,
        "shell": pl.Utf8,
        "vacancy_rate": pl.Float64,
    }
    if parent_states.is_empty():
        return pl.DataFrame(schema=empty_schema)

    # We're interested in isomer parents (parent_ex_kev > 0) — IC vacancies
    # arise from their IT cascade. For ground-state parents, IC of post-EC
    # daughter gammas is a v0.12 enrichment (see design memo).
    isomer_parents = parent_states.filter(pl.col("parent_ex_kev") > 0).select(["Z", "A", "parent_ex_kev"]).unique()
    if isomer_parents.is_empty():
        return pl.DataFrame(schema=empty_schema)

    # Resolve isomer parent_ex_kev to level_idx using ±1 keV match.
    levels = photon_evap_levels.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("level_idx"),
        pl.col("excitation_kev"),
    )
    isomer_levels = isomer_parents.join(levels, on=["Z", "A"], how="inner").filter(
        (pl.col("excitation_kev") - pl.col("parent_ex_kev")).abs() <= _STATE_KEV_TOL
    )
    if isomer_levels.is_empty():
        return pl.DataFrame(schema=empty_schema)

    # Pick the closest level_idx for each (Z, A, parent_ex_kev).
    isomer_levels = (
        isomer_levels.with_columns((pl.col("excitation_kev") - pl.col("parent_ex_kev")).abs().alias("__delta"))
        .sort(["Z", "A", "parent_ex_kev", "__delta"])
        .group_by(["Z", "A", "parent_ex_kev"])
        .agg(pl.col("level_idx").first())
    )

    gammas = photon_evap_gammas.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("parent_level").alias("level_idx"),
        pl.col("daughter_level").cast(pl.Int32),
        pl.col("gamma_energy_kev").cast(pl.Float64),
        pl.col("intensity").cast(pl.Float64),
        pl.col("icc_total").cast(pl.Float64),
    )

    # For each isomer parent, propagate cascade populations from the isomer
    # level down through the gamma scheme, summing K-shell IC vacancies
    # over each transition. This captures the canonical Tc-99m physics:
    # the *level-1 → ground* 140.5 keV transition (populated via the
    # 2.17 keV depopulating gamma of the isomer) is the dominant
    # K-vacancy source, not the depopulating gamma itself.
    out_rows: list[dict] = []
    for rec in isomer_levels.iter_rows(named=True):
        z = int(rec["Z"])
        a = int(rec["A"])
        parent_ex_kev = float(rec["parent_ex_kev"])
        isomer_idx = int(rec["level_idx"])
        k_edge = (k_edges or {}).get(z)

        gammas_za = gammas.filter((pl.col("Z") == z) & (pl.col("A") == a))
        if gammas_za.is_empty():
            continue

        populations = _propagate_cascade(z, a, isomer_idx, gammas_za)
        if not populations:
            continue

        # Pre-compute transition_frac per row.
        gammas_za = gammas_za.with_columns(
            (pl.col("intensity") * (1.0 + pl.col("icc_total"))).alias("__trans_rate"),
        ).with_columns(
            (pl.col("__trans_rate") / pl.col("__trans_rate").sum().over(["level_idx"])).alias("transition_frac"),
        )

        v_k = 0.0
        for g in gammas_za.iter_rows(named=True):
            src = int(g["level_idx"])
            src_pop = populations.get(src, 0.0)
            if src_pop <= 0.0:
                continue
            energy = float(g["gamma_energy_kev"])
            if k_edge is not None and energy < k_edge:
                continue
            icc = float(g["icc_total"])
            t_frac = float(g["transition_frac"])
            ic_frac = icc / (1.0 + icc) if (1.0 + icc) > 0 else 0.0
            v_k += src_pop * t_frac * ic_frac

        if v_k <= 0.0:
            continue
        # Cap at 1.0: physical bound is one K vacancy per parent decay.
        out_rows.append(
            {
                "Z": z,
                "A": a,
                "parent_ex_kev": parent_ex_kev,
                "daughter_Z": z,
                "shell": "K",
                "vacancy_rate": min(v_k, 1.0),
            }
        )

    if not out_rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.DataFrame(out_rows, schema=empty_schema)


# -----------------------------------------------------------------------------
# EADL convolution
# -----------------------------------------------------------------------------


def _z_to_symbol(nuclides: pl.DataFrame) -> dict[int, str]:
    """Build Z → element symbol map (e.g. 26 → 'Fe')."""
    pairs = (
        nuclides.select(["Z", "symbol"])
        .filter(pl.col("symbol").is_not_null() & (pl.col("symbol") != ""))
        .unique()
        .iter_rows()
    )
    return {int(z): sym for z, sym in pairs}


def synthesize_emissions(
    vacancy_rates: pl.DataFrame,
    decay_mode_label: str,
    eadl_dir: Path,
    z_to_symbol: dict[int, str],
) -> pl.DataFrame:
    """Convolve vacancy rates with EADL atomic-relaxation tables.

    Args:
        vacancy_rates: rows of (Z, A, parent_ex_kev, daughter_Z, shell,
            vacancy_rate).
        decay_mode_label: stored in the output ``decay_mode`` column,
            distinguishing EC vs IC origin (e.g. ``"EC"``, ``"IT"``).
        eadl_dir: Directory holding ``{Symbol}.parquet`` per-element tables.
        z_to_symbol: mapping from Z to element symbol for EADL file lookup.

    Returns:
        DataFrame with columns (Z, A, parent_ex_kev, rad_type, energy_keV,
        intensity_pct, decay_mode, rad_subtype). Caller assigns ``state``.
    """
    if vacancy_rates.is_empty():
        return _empty_emissions()

    rows: list[dict] = []
    # Cache EADL files by daughter_Z to avoid re-reading.
    eadl_cache: dict[int, pl.DataFrame] = {}

    for rec in vacancy_rates.iter_rows(named=True):
        d_z = int(rec["daughter_Z"])
        sym = z_to_symbol.get(d_z)
        if sym is None:
            logger.debug("no symbol for daughter Z=%d, skipping", d_z)
            continue

        if d_z not in eadl_cache:
            eadl_path = eadl_dir / f"{sym}.parquet"
            if not eadl_path.exists():
                logger.debug("no EADL table for %s (Z=%d)", sym, d_z)
                eadl_cache[d_z] = pl.DataFrame()
                continue
            eadl_cache[d_z] = pl.read_parquet(eadl_path)

        eadl = eadl_cache[d_z]
        if eadl.is_empty():
            continue

        vacancy_shell = _SHELL_TOTAL_TO_VACANCY.get(rec["shell"])
        if vacancy_shell is None:
            continue

        # Pull all relaxation transitions starting at this vacancy.
        sub = eadl.filter(pl.col("vacancy_shell") == vacancy_shell)
        if sub.is_empty():
            continue

        rate = float(rec["vacancy_rate"])

        for transition in sub.iter_rows(named=True):
            t_type = transition["transition_type"]
            filling = transition["filling_shell"]
            energy = float(transition["energy_keV"])
            prob = float(transition["probability"])
            intensity_pct = rate * prob * 100.0

            if t_type == "radiative":
                rad_type = "xray"
                subtype = xray_subtype(vacancy_shell, filling)
            elif t_type == "auger":
                rad_type = "auger"
                subtype = auger_subtype(vacancy_shell, filling)
            else:
                continue

            rows.append(
                {
                    "Z": int(rec["Z"]),
                    "A": int(rec["A"]),
                    "parent_ex_kev": float(rec["parent_ex_kev"]),
                    "rad_type": rad_type,
                    "energy_keV": energy,
                    "intensity_pct": intensity_pct,
                    "decay_mode": decay_mode_label,
                    "rad_subtype": subtype,
                }
            )

    if not rows:
        return _empty_emissions()
    return pl.DataFrame(rows)


def _empty_emissions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Z": pl.Int32,
            "A": pl.Int32,
            "parent_ex_kev": pl.Float64,
            "rad_type": pl.Utf8,
            "energy_keV": pl.Float64,
            "intensity_pct": pl.Float64,
            "decay_mode": pl.Utf8,
            "rad_subtype": pl.Utf8,
        }
    )


# -----------------------------------------------------------------------------
# State assignment + output schema
# -----------------------------------------------------------------------------


def attach_states(
    emissions: pl.DataFrame,
    nuclides: pl.DataFrame,
) -> pl.DataFrame:
    """Add ``state`` and ``parent_level_keV`` columns; drop unmappable rows.

    Per ADR-0001 / radioactive_decay.py convention:
        - parent_ex_kev = 0  → state = ""
        - parent_ex_kev > 0 within ±1 keV of a catalog isomer → state = that isomer's label
        - parent_ex_kev > 0 with no catalog match → drop (not representable in v0.10.x state enum)

    The NULL-level isomer fallback (Ba-137m case) is restricted to only
    the *lowest-excitation* parent for each (Z, A) when multiple candidate
    excitations exist — this prevents short-lived non-isomeric excited
    states from being mislabelled as "m".
    """
    if emissions.is_empty():
        return emissions.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("state"),
            pl.lit(None, dtype=pl.Float64).alias("parent_level_keV"),
        ).select(_OUTPUT_COLUMNS)

    # Determine the lowest excitation per (Z, A) where the catalog has a
    # NULL-level isomer (i.e. fallback territory). Other excitations under
    # the same (Z, A) won't be granted the fallback "m".
    null_level_pairs = set(
        nuclides.filter((pl.col("state") != "") & pl.col("level_keV").is_null())
        .group_by(["Z", "A"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") == 1)
        .select(["Z", "A"])
        .iter_rows()
    )
    # For each (Z, A) in the fallback set, find the lowest parent_ex_kev
    # appearing in the emissions table.
    fallback_lowest: dict[tuple[int, int], float] = {}
    for rec in (
        emissions.filter(pl.col("parent_ex_kev") > 0)
        .group_by(["Z", "A"])
        .agg(pl.col("parent_ex_kev").min().alias("min_ex"))
        .iter_rows(named=True)
    ):
        key = (int(rec["Z"]), int(rec["A"]))
        if key in null_level_pairs:
            fallback_lowest[key] = float(rec["min_ex"])

    out_rows: list[dict] = []
    cache: dict[tuple[int, int, float], str | None] = {}

    for rec in emissions.iter_rows(named=True):
        z = int(rec["Z"])
        a = int(rec["A"])
        ex = float(rec["parent_ex_kev"])
        key = (z, a, ex)
        if key not in cache:
            state = assign_state(z, a, ex, nuclides)
            # Restrict the NULL-level fallback to the lowest-ex parent.
            if state is not None and ex > 0 and (z, a) in fallback_lowest:
                if abs(ex - fallback_lowest[(z, a)]) > _STATE_KEV_TOL:
                    # This is a higher-excitation parent in a (Z,A) where
                    # the fallback was activated — refuse to label it.
                    state = None
            cache[key] = state
        state = cache[key]
        if state is None:
            continue
        out_rows.append(
            {
                "Z": z,
                "A": a,
                "state": state,
                "rad_type": rec["rad_type"],
                "energy_keV": float(rec["energy_keV"]),
                "intensity_pct": float(rec["intensity_pct"]),
                "decay_mode": rec["decay_mode"],
                "parent_level_keV": ex,
                "rad_subtype": rec["rad_subtype"],
            }
        )

    if not out_rows:
        empty = pl.DataFrame(schema={c: t for c, t in _OUTPUT_SCHEMA.items()})
        return empty
    return pl.DataFrame(out_rows).select(_OUTPUT_COLUMNS)


_OUTPUT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "Z": pl.Int32,
    "A": pl.Int32,
    "state": pl.Utf8,
    "rad_type": pl.Utf8,
    "energy_keV": pl.Float64,
    "intensity_pct": pl.Float64,
    "decay_mode": pl.Utf8,
    "parent_level_keV": pl.Float64,
    "rad_subtype": pl.Utf8,
}
_OUTPUT_COLUMNS: Final[list[str]] = list(_OUTPUT_SCHEMA.keys())


def _enforce_schema(df: pl.DataFrame) -> pl.DataFrame:
    casts = []
    for col, dtype in _OUTPUT_SCHEMA.items():
        if col in df.columns:
            casts.append(pl.col(col).cast(dtype))
    return df.with_columns(casts).select(_OUTPUT_COLUMNS)


# -----------------------------------------------------------------------------
# Top-level orchestration
# -----------------------------------------------------------------------------


def load_k_edges(eadl_dir: Path) -> dict[int, float]:
    """Build a {Z: K-edge_keV} map by scanning EADL files."""
    edges: dict[int, float] = {}
    if not eadl_dir.exists():
        return edges
    for parquet in eadl_dir.glob("*.parquet"):
        try:
            df = pl.read_parquet(parquet, columns=["Z", "vacancy_shell", "edge_keV"])
        except Exception:
            continue
        k_rows = df.filter(pl.col("vacancy_shell") == "K").select(["Z", "edge_keV"]).unique()
        for row in k_rows.iter_rows(named=True):
            edges[int(row["Z"])] = float(row["edge_keV"])
    return edges


def synthesize_xray_auger(
    decay_detailed: pl.DataFrame,
    photon_evap_gammas: pl.DataFrame,
    photon_evap_levels: pl.DataFrame,
    nuclides: pl.DataFrame,
    eadl_dir: Path,
) -> pl.DataFrame:
    """Compute X-ray and Auger emission rows for all parents in scope.

    Returns the consolidated DataFrame; caller writes per-element parquet.
    """
    z_to_symbol = _z_to_symbol(nuclides)
    k_edges = load_k_edges(eadl_dir)

    # Step 1: EC vacancies in daughter (Z-1).
    ec_vacancies = compute_ec_vacancy_rates(decay_detailed)
    logger.info("EC vacancy rows: %d", ec_vacancies.height)

    # Step 2: build the parent-state list for IC lookup. Sourced from BOTH
    # decay_detailed (covers EC isomers) AND photon_evap_levels (covers
    # IT-only isomers like Ba-137m where strata's radioactive_decay only
    # ships a summary row, not detail, so the level isn't in
    # decay_detailed). Ba-137m is the canonical case: Cs-137 → Ba-137m →
    # Ba-137 via 661.66 keV M4 IC; without level-sourced parent states the
    # synthesizer would emit no rows for it.
    #
    # Half-life threshold is 100 ns: captures all classical metastable
    # isomers (Tc-99m at 6h, Ba-137m at 153s, In-115m at 4.5h, …) while
    # excluding short-lived excited states that have positive half-life
    # but aren't independently treated as parents (Ba-137 levels at 2349
    # keV and 2623 keV with t½ ~ 30 ns — these are cascade-transit levels,
    # not metastable parents).
    detail_parents = decay_detailed.select(["Z", "A", "parent_ex_kev"]).unique()
    isomer_levels_seed = (
        photon_evap_levels.filter((pl.col("excitation_kev") > 0) & (pl.col("half_life_s") >= 1e-7))
        .select(
            pl.col("z").cast(pl.Int32).alias("Z"),
            pl.col("a").cast(pl.Int32).alias("A"),
            pl.col("excitation_kev").alias("parent_ex_kev"),
        )
        .unique()
    )
    parent_states = pl.concat([detail_parents, isomer_levels_seed], how="vertical_relaxed").unique()
    ic_vacancies = compute_ic_vacancy_rates(photon_evap_gammas, photon_evap_levels, parent_states, k_edges=k_edges)
    logger.info("IC vacancy rows: %d", ic_vacancies.height)

    # Step 3: convolve with EADL.
    ec_emissions = synthesize_emissions(
        ec_vacancies,
        decay_mode_label="EC",
        eadl_dir=eadl_dir,
        z_to_symbol=z_to_symbol,
    )
    ic_emissions = synthesize_emissions(
        ic_vacancies,
        decay_mode_label="IT",
        eadl_dir=eadl_dir,
        z_to_symbol=z_to_symbol,
    )
    logger.info(
        "synthesized: %d EC-emission rows, %d IC-emission rows",
        ec_emissions.height,
        ic_emissions.height,
    )

    combined = pl.concat([ec_emissions, ic_emissions], how="vertical_relaxed")
    if combined.is_empty():
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    # Step 4: assign v0.10.x state labels.
    out = attach_states(combined, nuclides)
    logger.info("final emission rows after state assignment: %d", out.height)
    return _enforce_schema(out)


def write_radiation_atomic(
    emissions: pl.DataFrame,
    nuclides: pl.DataFrame,
    out_dir: Path,
) -> list[Path]:
    """Split synthesized emissions per element-symbol and write parquet files.

    Returns list of paths written.
    """
    if emissions.is_empty():
        logger.warning("no emissions to write")
        return []

    z_to_symbol = _z_to_symbol(nuclides)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for z, group in emissions.group_by("Z"):
        z_int = int(z[0])
        sym = z_to_symbol.get(z_int)
        if sym is None:
            logger.warning("no symbol for Z=%d, skipping write", z_int)
            continue
        path = out_dir / f"{sym}.parquet"
        # Sort for deterministic output.
        group.sort(["A", "state", "rad_type", "energy_keV"]).write_parquet(path)
        written.append(path)
    logger.info("wrote %d per-element files to %s", len(written), out_dir)
    return written


# -----------------------------------------------------------------------------
# Energy-conservation invariant
# -----------------------------------------------------------------------------


def check_energy_conservation(
    emissions: pl.DataFrame,
    vacancy_rates: pl.DataFrame,
    slack_pct: float = _ENERGY_SLACK_PCT,
) -> pl.DataFrame:
    """For each (Z, A, parent_ex_kev, shell), verify
    ``Σ intensity_pct ≈ vacancy_rate × 100`` within ``slack_pct``.

    Returns a DataFrame of *violations* (empty if all pass).
    """
    if emissions.is_empty() or vacancy_rates.is_empty():
        return pl.DataFrame(
            schema={
                "Z": pl.Int32,
                "A": pl.Int32,
                "parent_ex_kev": pl.Float64,
                "shell": pl.Utf8,
                "expected_pct": pl.Float64,
                "actual_pct": pl.Float64,
                "delta_pct": pl.Float64,
            }
        )

    # Map emissions → shell letter from rad_subtype.
    em = emissions.with_columns(pl.col("rad_subtype").str.slice(0, 1).alias("shell"))
    actual = (
        em.group_by(["Z", "A", "parent_level_keV", "shell"])
        .agg(pl.col("intensity_pct").sum().alias("actual_pct"))
        .rename({"parent_level_keV": "parent_ex_kev"})
    )
    expected = vacancy_rates.with_columns((pl.col("vacancy_rate") * 100.0).alias("expected_pct")).select(
        ["Z", "A", "parent_ex_kev", "shell", "expected_pct"]
    )

    joined = (
        expected.join(actual, on=["Z", "A", "parent_ex_kev", "shell"], how="left")
        .with_columns(
            pl.col("actual_pct").fill_null(0.0),
        )
        .with_columns(
            (pl.col("actual_pct") - pl.col("expected_pct")).abs().alias("delta_pct"),
        )
    )
    violations = joined.filter(pl.col("delta_pct") > slack_pct)
    return violations


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--decay-detailed",
        type=Path,
        default=Path("data/meta/decay_detailed.parquet"),
    )
    parser.add_argument(
        "--gammas",
        type=Path,
        default=Path("data/g4_raw/strata-nuclear/photon_evap_gammas.parquet"),
    )
    parser.add_argument(
        "--levels",
        type=Path,
        default=Path("data/g4_raw/strata-nuclear/photon_evap_levels.parquet"),
    )
    parser.add_argument(
        "--nuclides",
        type=Path,
        default=Path("data/meta/ensdf/nuclides.parquet"),
    )
    parser.add_argument(
        "--eadl-dir",
        type=Path,
        default=Path("data/meta/eadl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/meta/ensdf/radiation_atomic"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    decay_detailed = pl.read_parquet(args.decay_detailed)
    gammas = pl.read_parquet(args.gammas)
    levels = pl.read_parquet(args.levels)
    nuclides = pl.read_parquet(args.nuclides)

    emissions = synthesize_xray_auger(decay_detailed, gammas, levels, nuclides, args.eadl_dir)
    write_radiation_atomic(emissions, nuclides, args.out_dir)


if __name__ == "__main__":
    _main()
