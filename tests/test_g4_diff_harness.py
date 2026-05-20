"""Canonical-isotope diff harness for the v0.11 G4 migration (issue #75).

Asserts that the rebuilt dataset (produced by ``nucl_parquet.g4.build_all``)
matches the v0.11 acceptance criteria for every isotope on the canonical
list. This is the migration's *acceptance gate* — under-specifying it lets
bugs ship; over-specifying brittles the suite against legitimate physics
updates.

Conventions:

- Each test loads from the on-disk parquet (data/meta/...). Skip the test
  if the file is missing — the harness assumes ``build_all`` has already
  been run.
- Tests use ``pytest.approx`` with relative tolerance for half-lives and
  level energies; the values come from NUBASE2020 / ICRP-107 canonical
  references.
- A separate sweep test asserts the v0.11 invariants: no corrupted
  ``parent_level_keV``, no IT-on-ground in ``decay.parquet``, no phantom
  isomers in ``nuclides.parquet``, ``intensity_pct >= 0`` everywhere,
  ``half_life_s = +inf`` for stable nuclides (NOT NULL).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
RAD_DIR = DATA / "meta" / "ensdf" / "radiation"
NUC_PATH = DATA / "meta" / "ensdf" / "nuclides.parquet"
DECAY_PATH = DATA / "meta" / "decay.parquet"
DECAY_DETAILED_PATH = DATA / "meta" / "decay_detailed.parquet"
COINC_DIR = DATA / "meta" / "ensdf" / "coincidences"
LEVELS_DIR = DATA / "meta" / "ensdf" / "levels"

_built = pytest.mark.skipif(
    not NUC_PATH.exists(),
    reason="nuclides.parquet not built — run `python -m nucl_parquet.g4.build_all` first",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nuclides() -> pl.DataFrame:
    return pl.read_parquet(NUC_PATH)


@pytest.fixture(scope="module")
def decay() -> pl.DataFrame:
    return pl.read_parquet(DECAY_PATH)


@pytest.fixture(scope="module")
def radiation() -> pl.DataFrame:
    return pl.read_parquet(str(RAD_DIR / "*.parquet"))


# ---------------------------------------------------------------------------
# Per-isotope acceptance tests
# ---------------------------------------------------------------------------


@_built
@pytest.mark.parametrize(
    ("z", "a", "state", "expected_level_keV", "expected_half_life_s", "label"),
    [
        # Existing canonical _validate spot checks
        (43, 99, "m", 142.6836, 21630.0, "Tc-99m"),
        (21, 44, "m", 271.241, 210996.0, "Sc-44m"),  # ~58.6 hr (G4 ENSDFSTATE3.0)
        (49, 113, "m", 391.7, 5976.0, "In-113m"),  # 99.5 min
        (47, 108, "m", 109.5, 1.32e10, "Ag-108m"),  # ~418 yr
        (63, 152, "m", 45.6, 33390.0, "Eu-152m"),  # 9.31 hr
        (56, 137, "m", 661.66, 153.1, "Ba-137m"),  # 2.55 min — Cs-137 daughter
        # The canonical IAEA-fetcher hole — was missing from v0.10.x
        (63, 152, "m2", 147.86, 5760.0, "Eu-152m2"),  # 96 min
        # The notorious K-isomer
        (72, 178, "m2", 2446.0, 9.78e8, "Hf-178m2"),  # 31 yr
    ],
)
def test_canonical_isomer_present(
    nuclides: pl.DataFrame,
    z: int,
    a: int,
    state: str,
    expected_level_keV: float,
    expected_half_life_s: float,
    label: str,
) -> None:
    rows = nuclides.filter((pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") == state))
    assert rows.height == 1, f"{label}: expected exactly 1 row, got {rows.height}"
    row = rows.row(0, named=True)
    assert row["level_keV"] == pytest.approx(expected_level_keV, rel=0.001), (
        f"{label}: level_keV {row['level_keV']} != {expected_level_keV}"
    )
    assert row["half_life_s"] == pytest.approx(expected_half_life_s, rel=0.05), (
        f"{label}: half_life_s {row['half_life_s']} != {expected_half_life_s}"
    )


@_built
@pytest.mark.parametrize(
    ("z", "a", "label"),
    [
        (1, 1, "H-1"),
        (2, 4, "He-4"),
        (26, 56, "Fe-56"),
        (82, 208, "Pb-208"),
    ],
)
def test_stable_isotope_is_positive_infinity(nuclides: pl.DataFrame, z: int, a: int, label: str) -> None:
    """Per ADR-0002: stable nuclides ship `half_life_s = +inf`, NOT NULL.
    The NULL conflation was a v0.10.x DQ flaw; v0.11 distinguishes
    'stable per physical observation' from 'unknown/unmeasured'."""
    rows = nuclides.filter((pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") == ""))
    assert rows.height == 1, f"{label}: missing ground-state row"
    hl = rows["half_life_s"][0]
    assert hl is not None, f"{label}: half_life_s is NULL (should be +inf for stable)"
    assert hl == float("inf"), f"{label}: half_life_s={hl} (expected +inf for stable)"


@_built
def test_eu152m2_cascade_in_radiation(radiation: pl.DataFrame) -> None:
    """Eu-152m2 IT cascade gammas — these are the rows that the v0.10.x
    IAEA pipeline silently lost when the get_state_map fabrication bug
    relabeled them. v0.11 must show them as state='m2' or as Eu-152m2's
    daughter cascade lines (depends on filing convention)."""
    # Lines from PhotonEvaporation rooted in the 147.86 keV level
    eu152_at_m2 = radiation.filter(
        (pl.col("Z") == 63)
        & (pl.col("A") == 152)
        & (pl.col("rad_type") == "gamma")
        & (pl.col("parent_level_keV") >= 147.0)
        & (pl.col("parent_level_keV") <= 148.5)
    )
    assert eu152_at_m2.height >= 1, (
        "Eu-152m2 cascade lines missing from radiation — the 147.86 keV isomer should emit gammas to lower levels"
    )


@_built
def test_tc99m_142_keV_gamma_in_radiation(radiation: pl.DataFrame) -> None:
    """The canonical medical-isotope line: Tc-99m emits a 142.6836 keV
    gamma (ENSDF level 1) — often informally quoted as "140.5 keV" in
    older medical-physics literature, but the precise ENSDF value is
    142.68. Tc-99m is the most common gamma spectroscopy benchmark;
    missing it shipping v0.11 is a P0 regression."""
    rows = radiation.filter(
        (pl.col("Z") == 43)
        & (pl.col("A") == 99)
        & (pl.col("state") == "m")
        & (pl.col("rad_type") == "gamma")
        & (pl.col("energy_keV") > 142.0)
        & (pl.col("energy_keV") < 143.0)
    )
    assert rows.height >= 1, "Tc-99m 142.68 keV gamma missing from radiation"


@_built
def test_no_corrupted_parent_level_kev(radiation: pl.DataFrame) -> None:
    """v0.10.x had `parent_level_keV` values up to 2e+215 from IAEA-fetcher
    text-as-number parsing artifacts. G4 source is fixed-width numeric;
    v0.11 must show no values above the largest physically plausible
    nuclear level (~30 MeV; ceiling 1e8 keV is generous)."""
    max_pl = radiation.select(pl.col("parent_level_keV").max()).item()
    assert max_pl is None or max_pl < 1e8, (
        f"radiation parent_level_keV ceiling breach: max={max_pl} keV (v0.10.x 2e+215 corruption regression?)"
    )


@_built
def test_no_it_on_ground_in_decay(decay: pl.DataFrame) -> None:
    """v0.10.x had ~49 IT-on-ground rows from IAEA fetcher. IT decays
    require a metastable parent by definition; ground-state IT is
    unphysical. v0.11 must show zero."""
    offenders = decay.filter((pl.col("state") == "") & (pl.col("decay_mode").str.to_uppercase() == "IT"))
    assert offenders.height == 0, (
        f"IT-on-ground regression: {offenders.height} rows "
        f"(IAEA-fetcher bug class re-emerged?): {offenders.head(5).to_dicts()}"
    )


@_built
def test_no_phantom_isomers_in_nuclides(nuclides: pl.DataFrame) -> None:
    """v0.10.x had ~146 phantom isomer entries fabricated by get_state_map.
    Every isomer in v0.11 must be backed by either a decay-mode entry
    (G4 RadioactiveDecay) or a level-scheme entry (G4 PhotonEvaporation),
    surfaced via non-NULL `level_keV`."""
    phantom = nuclides.filter((pl.col("state") != "") & pl.col("level_keV").is_null())
    # A few legitimate isomer rows from RadioactiveDecay don't have a
    # level_keV (G4 omits the energy for some short-lived states); allow
    # a small bound rather than zero. Zero would over-specify.
    assert phantom.height < 50, (
        f"phantom isomers in nuclides (state != '' AND level_keV IS NULL): "
        f"{phantom.height} (v0.10.x baseline was 146; v0.11 should be much lower)"
    )


@_built
def test_radiation_intensity_pct_non_negative(radiation: pl.DataFrame) -> None:
    """v0.10.x had negative intensity_pct values from IAEA Auger/CE rows.
    G4 is per-cascade-relative and never negative."""
    neg = radiation.filter(pl.col("intensity_pct") < 0)
    assert neg.height == 0, f"negative intensity_pct: {neg.head(5).to_dicts()}"


@_built
@pytest.mark.parametrize(
    ("symbol", "z_daughter", "a_daughter", "label"),
    [
        # Daughter-keyed filing per the existing convention (G4 indexes by
        # the de-exciting nucleus, not the decaying parent). Co-60 cascade
        # lives in Ni.parquet (Z=28); Cs-137 → Ba-137m → Ba-137 single
        # transition (no cascade pair).
        ("Ni", 28, 60, "Co-60 → Ni-60* 1173/1332 keV cascade"),
        ("Sm", 62, 152, "Eu-152 → Sm-152* multi-line cascade"),
    ],
)
def test_canonical_coincidence_pair_in_daughter_file(symbol: str, z_daughter: int, a_daughter: int, label: str) -> None:
    coinc_path = COINC_DIR / f"{symbol}.parquet"
    if not coinc_path.exists():
        pytest.skip(f"{symbol}.parquet not built")
    df = pl.read_parquet(coinc_path)
    pairs = df.filter((pl.col("Z") == z_daughter) & (pl.col("A") == a_daughter))
    assert pairs.height >= 1, f"{label}: no cascade pairs in {symbol}.parquet"


# ---------------------------------------------------------------------------
# Cross-check tests (multi-file invariants)
# ---------------------------------------------------------------------------


@_built
def test_radiation_atomic_dir_removed() -> None:
    """The merge step in build_all.merge_radiation_atomic removes the
    transient radiation_atomic/ directory after merging into radiation/.
    Asserts the canonical layout is single-source."""
    atomic_dir = DATA / "meta" / "ensdf" / "radiation_atomic"
    assert not atomic_dir.exists(), "radiation_atomic/ was not cleaned up after the merge step"


@_built
def test_radiation_carries_all_three_rad_types(radiation: pl.DataFrame) -> None:
    """Post-merge, radiation/ must contain rows of all three rad_types
    we synthesize in v0.11: gamma (from photon_evap), xray + auger (from
    G4EMLOW × per-shell EC/IC). Missing a class indicates the merge
    regressed."""
    types = set(radiation["rad_type"].unique().to_list())
    assert {"gamma", "xray", "auger"} <= types, (
        f"radiation missing rad_types: have {types}, expected ≥ {{gamma, xray, auger}}"
    )


@_built
def test_radiation_schema_is_union_of_sources(radiation: pl.DataFrame) -> None:
    """Every radiation row carries the union of v0.10.x compat columns
    + bonus columns from both #72 (gammas) and #74 (xray/auger).
    Source-specific columns are NULL on rows from the other source."""
    expected = {
        "Z",
        "A",
        "state",
        "rad_type",
        "energy_keV",
        "intensity_pct",
        "decay_mode",
        "rad_subtype",
        "dose_MeV_per_Bq_s",
        "parent_level_keV",
        # bonus from #72
        "daughter_level_keV",
        "multipolarity",
        "mixing_ratio",
        "icc_total",
        # bonus from #74
        "vacancy_shell",
    }
    assert set(radiation.columns) == expected, (
        f"radiation schema diverges from union: {set(radiation.columns) ^ expected}"
    )


@_built
def test_decay_nuclides_halflife_agreement(decay: pl.DataFrame, nuclides: pl.DataFrame) -> None:
    """Sweep invariant (#203): every (Z, A, state) parent in decay.parquet
    must have a half_life_s within 10% of the catalog value in
    nuclides.parquet.

    This catches the G4 RadioactiveDecay half-life swap bug class —
    where ground and metastable half-lives are swapped at the source —
    permanently, regardless of which layer introduces the error.

    Known upstream G4 disagreements are accepted via a small budget.
    The pre-fix dataset had ~60 mismatches from the swap bug; the
    budget of 5 catches any regression while tolerating unfixed
    upstream cases (Li-4 resonance width, Ir-192m unfixed swap —
    gerchowl/strata#711)."""
    decay_hl = decay.select("Z", "A", "state", "half_life_s").unique(subset=["Z", "A", "state"])
    nuclides_hl = nuclides.filter(
        pl.col("half_life_s").is_not_null() & pl.col("half_life_s").is_finite() & (pl.col("half_life_s") > 0)
    ).select(
        pl.col("Z"),
        pl.col("A"),
        pl.col("state"),
        pl.col("half_life_s").alias("nuc_hl"),
    )
    joined = decay_hl.join(nuclides_hl, on=["Z", "A", "state"], how="inner").filter(
        pl.col("half_life_s").is_not_null() & pl.col("half_life_s").is_finite() & (pl.col("half_life_s") > 0)
    )
    mismatches = joined.filter(((pl.col("half_life_s") - pl.col("nuc_hl")).abs() / pl.col("nuc_hl")) > 0.10)
    # Budget: the pre-fix dataset had ~60 mismatches from the swap bug.
    # Post-fix, 2 remain as unfixed upstream G4 disagreements:
    #   - Li-4 (Z=3,A=4): unbound resonance, different width evaluations
    #   - Ir-192m (Z=77,A=192): unfixed swap — decay carries ground half-life
    #     (6.38e6 s) on state='m'; nuclides correctly has 87 s (gerchowl/strata#711)
    # Budget of 5 catches any regression while tolerating known upstream quirks.
    assert mismatches.height <= 5, (
        f"decay ↔ nuclides half-life mismatch (>10%): {mismatches.height} rows "
        f"(budget: 5, pre-fix baseline: ~60); "
        f"sample: {mismatches.head(10).to_dicts()}"
    )
    if mismatches.height > 0:
        import warnings

        warnings.warn(
            f"decay ↔ nuclides: {mismatches.height} half-life disagreements >10% "
            f"(within budget of 5): {mismatches.select('Z', 'A', 'state').to_dicts()}",
            stacklevel=1,
        )
