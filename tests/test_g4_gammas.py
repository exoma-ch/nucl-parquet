"""Tests for the G4 photon_evap_gammas → meta/ensdf/radiation/ converter.

Two layers:

1. **Pure transform unit tests** — exercise ``build_isomer_state_map`` and
   ``transform`` on synthetic frames covering the ADR-0002 boundaries (the
   0.1 keV state-match tolerance, the 1 ms long-lived threshold, the
   stable-sentinel ``-1``, the per-row LEFT JOIN against levels). No I/O.
2. **Integration spot checks** — build the parquet from the shipped strata
   inputs and verify canonical isotopes (Eu-152 m2 cascade, Tc-99 m gammas,
   parent_level_keV ceiling, schema compat). Marked ``@pytest.mark.data`` so
   they auto-skip when raw data is absent.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.photon_evap_gammas import (
    HALF_LIFE_STABLE_SENTINEL,
    LEVEL_MATCH_TOLERANCE_KEV,
    LONG_LIVED_HALF_LIFE_S_THRESHOLD,
    PARENT_LEVEL_KEV_CEILING,
    build,
    build_isomer_state_map,
    transform,
)

# ---------------------------------------------------------------------------
# Synthetic-frame fixtures.
# ---------------------------------------------------------------------------


def _mk_levels(rows: list[dict]) -> pl.DataFrame:
    """Tiny strata-shaped levels frame builder."""
    return pl.DataFrame(
        rows,
        schema={
            "z": pl.UInt8,
            "a": pl.UInt16,
            "level_idx": pl.UInt16,
            "floating_flag": pl.String,
            "excitation_kev": pl.Float64,
            "half_life_s": pl.Float64,
            "jpi": pl.Float32,
            "n_gammas": pl.UInt16,
        },
    )


def _mk_gammas(rows: list[dict]) -> pl.DataFrame:
    """Tiny strata-shaped gammas frame builder."""
    return pl.DataFrame(
        rows,
        schema={
            "z": pl.UInt8,
            "a": pl.UInt16,
            "parent_level": pl.UInt16,
            "daughter_level": pl.UInt16,
            "gamma_energy_kev": pl.Float64,
            "intensity": pl.Float32,
            "multipolarity": pl.UInt16,
            "mixing_ratio": pl.Float32,
            "icc_total": pl.Float32,
        },
    )


# ---------------------------------------------------------------------------
# Pure transform primitives — no I/O.
# ---------------------------------------------------------------------------


class TestBuildIsomerStateMap:
    """ADR-0002 long-lived isomer threshold + ground/m/m2 labelling."""

    def test_ground_is_empty_string(self) -> None:
        """The lowest excitation per (Z, A) gets state=''."""
        levels = _mk_levels(
            [
                {
                    "z": 43,
                    "a": 99,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": 6.66e12,
                    "jpi": 4.5,
                    "n_gammas": 0,
                },
            ]
        )
        sm = build_isomer_state_map(levels)
        assert sm.height == 1
        row = sm.to_dicts()[0]
        assert row["state"] == ""
        assert row["level_keV"] == 0.0

    def test_long_lived_isomer_above_ground_gets_m(self) -> None:
        """Tc-99 m-state at 142.68 keV (T½ = 21626 s ≫ 1 ms) → state='m'."""
        levels = _mk_levels(
            [
                {
                    "z": 43,
                    "a": 99,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": 6.66e12,
                    "jpi": 4.5,
                    "n_gammas": 0,
                },
                {
                    "z": 43,
                    "a": 99,
                    "level_idx": 2,
                    "floating_flag": "-",
                    "excitation_kev": 142.6836,
                    "half_life_s": 21625.9,
                    "jpi": -0.5,
                    "n_gammas": 2,
                },
            ]
        )
        sm = build_isomer_state_map(levels)
        # Map by state for assertion clarity.
        by_state = {r["state"]: r for r in sm.to_dicts()}
        assert by_state[""]["level_keV"] == 0.0
        assert by_state["m"]["level_keV"] == pytest.approx(142.6836)

    def test_short_lived_excited_excluded(self) -> None:
        """Excited levels under the 1 ms threshold do NOT get an m label —
        they fall through to ``state=""`` via the assignment join (a gamma
        from a 1 ns excited parent isn't a v0.10.x 'm' isomer)."""
        levels = _mk_levels(
            [
                {
                    "z": 1,
                    "a": 1,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": -1.0,  # stable
                    "jpi": 0.5,
                    "n_gammas": 0,
                },
                {
                    "z": 1,
                    "a": 1,
                    "level_idx": 1,
                    "floating_flag": "-",
                    "excitation_kev": 100.0,
                    "half_life_s": 1e-9,  # 1 ns ≪ 1 ms
                    "jpi": 0.5,
                    "n_gammas": 0,
                },
            ]
        )
        sm = build_isomer_state_map(levels)
        assert set(sm["state"].to_list()) == {""}, (
            "Short-lived excited level should not appear as an isomer in the state map."
        )

    def test_g4_stable_sentinel_qualifies_as_long_lived(self) -> None:
        """The ``-1`` sentinel on an excited level (rare but exists) must be
        treated as long-lived per ADR-0002 stable-by-observation semantics."""
        levels = _mk_levels(
            [
                {
                    "z": 50,
                    "a": 100,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": -1.0,
                    "jpi": 0.0,
                    "n_gammas": 0,
                },
                {
                    "z": 50,
                    "a": 100,
                    "level_idx": 5,
                    "floating_flag": "-",
                    "excitation_kev": 200.0,
                    "half_life_s": HALF_LIFE_STABLE_SENTINEL,
                    "jpi": 4.0,
                    "n_gammas": 1,
                },
            ]
        )
        sm = build_isomer_state_map(levels)
        states = {r["level_keV"]: r["state"] for r in sm.to_dicts()}
        assert states[200.0] == "m"

    def test_multiple_isomers_get_m_m2_m3(self) -> None:
        """Eu-152 has m at 45.6 keV and m2 at 147.86 keV (both ≫ 1 ms)."""
        levels = _mk_levels(
            [
                {
                    "z": 63,
                    "a": 152,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": 4.27e8,
                    "jpi": -3.0,
                    "n_gammas": 0,
                },
                {
                    "z": 63,
                    "a": 152,
                    "level_idx": 1,
                    "floating_flag": "-",
                    "excitation_kev": 45.5998,
                    "half_life_s": 33521.8,
                    "jpi": -0.0,
                    "n_gammas": 0,
                },
                {
                    "z": 63,
                    "a": 152,
                    "level_idx": 16,
                    "floating_flag": "-",
                    "excitation_kev": 147.86,
                    "half_life_s": 5760.0,
                    "jpi": -8.0,
                    "n_gammas": 1,
                },
            ]
        )
        sm = build_isomer_state_map(levels)
        states = {round(r["level_keV"], 2): r["state"] for r in sm.to_dicts()}
        assert states[0.0] == ""
        assert states[45.6] == "m"
        assert states[147.86] == "m2"


class TestTransform:
    """End-to-end pure transform on synthetic input (no I/O)."""

    def test_basic_schema_and_join(self) -> None:
        """Output must contain the v0.10.x columns + bonus columns + the
        joined parent/daughter level energies."""
        levels = _mk_levels(
            [
                {
                    "z": 43,
                    "a": 99,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": 6.66e12,
                    "jpi": 4.5,
                    "n_gammas": 0,
                },
                {
                    "z": 43,
                    "a": 99,
                    "level_idx": 1,
                    "floating_flag": "-",
                    "excitation_kev": 140.511,
                    "half_life_s": 1e-9,
                    "jpi": 0.5,
                    "n_gammas": 1,
                },
                {
                    "z": 43,
                    "a": 99,
                    "level_idx": 2,
                    "floating_flag": "-",
                    "excitation_kev": 142.6836,
                    "half_life_s": 21625.9,
                    "jpi": -0.5,
                    "n_gammas": 2,
                },
            ]
        )
        gammas = _mk_gammas(
            [
                # Tc-99 142 keV (m → ground), should be labelled state='m'.
                {
                    "z": 43,
                    "a": 99,
                    "parent_level": 2,
                    "daughter_level": 0,
                    "gamma_energy_kev": 142.6836,
                    "intensity": 0.02,
                    "multipolarity": 9,
                    "mixing_ratio": 0.0,
                    "icc_total": 40.3,
                },
                # 140.5 keV (level 1 → ground), short-lived parent → state=''.
                {
                    "z": 43,
                    "a": 99,
                    "parent_level": 1,
                    "daughter_level": 0,
                    "gamma_energy_kev": 140.511,
                    "intensity": 100.0,
                    "multipolarity": 304,
                    "mixing_ratio": 0.13,
                    "icc_total": 0.113,
                },
            ]
        )

        out = transform(gammas, levels)
        assert set(out.columns) == {
            "Z",
            "A",
            "state",
            "rad_type",
            "energy_keV",
            "intensity_pct",
            "decay_mode",
            "rad_subtype",
            "parent_level_keV",
            "daughter_level_keV",
            "multipolarity",
            "mixing_ratio",
            "icc_total",
        }
        # Constant rad_type.
        assert set(out["rad_type"].to_list()) == {"gamma"}
        # decay_mode and rad_subtype are NULL on this branch (#71/#74 fill them).
        assert out["decay_mode"].null_count() == out.height
        assert out["rad_subtype"].null_count() == out.height

        # Look up by gamma energy for stable assertions.
        rows = {round(r["energy_keV"], 4): r for r in out.to_dicts()}
        # 142 keV gamma — parent is the 142.68 keV m-state.
        r142 = rows[142.6836]
        assert r142["state"] == "m"
        assert r142["parent_level_keV"] == pytest.approx(142.6836)
        assert r142["daughter_level_keV"] == pytest.approx(0.0)
        assert r142["multipolarity"] == 9
        assert r142["icc_total"] == pytest.approx(40.3, rel=1e-4)
        # 140.5 keV gamma — parent at 140.511 is short-lived, falls to ''.
        r140 = rows[140.511]
        assert r140["state"] == ""
        assert r140["parent_level_keV"] == pytest.approx(140.511)
        assert r140["multipolarity"] == 304  # M1+E2 mixing — preserved as-is.

    def test_state_match_tolerance_just_inside(self) -> None:
        """A parent_level_keV within 0.1 keV of an isomer level matches."""
        levels = _mk_levels(
            [
                {
                    "z": 50,
                    "a": 100,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": -1.0,
                    "jpi": 0.0,
                    "n_gammas": 0,
                },
                {
                    "z": 50,
                    "a": 100,
                    "level_idx": 1,
                    "floating_flag": "-",
                    "excitation_kev": 200.0,
                    "half_life_s": 100.0,  # 100 s ≫ 1 ms
                    "jpi": 4.0,
                    "n_gammas": 1,
                },
            ]
        )
        gammas = _mk_gammas(
            [
                # parent_level idx=1 → level_keV = 200.0 exactly, well within tol.
                {
                    "z": 50,
                    "a": 100,
                    "parent_level": 1,
                    "daughter_level": 0,
                    "gamma_energy_kev": 200.0,
                    "intensity": 100.0,
                    "multipolarity": 4,
                    "mixing_ratio": 0.0,
                    "icc_total": 0.01,
                },
            ]
        )
        out = transform(gammas, levels)
        assert out["state"].to_list() == ["m"]

    def test_constants_match_adr(self) -> None:
        """Pin the threshold/tolerance constants — these are load-bearing
        for ADR-0002 and the v0.11 release notes."""
        assert LEVEL_MATCH_TOLERANCE_KEV == 0.1
        assert LONG_LIVED_HALF_LIFE_S_THRESHOLD == 1e-3
        assert HALF_LIFE_STABLE_SENTINEL == -1.0
        assert PARENT_LEVEL_KEV_CEILING == 1e8

    def test_intensity_passes_through_as_pct(self) -> None:
        """``intensity`` field is renamed (no rescale) into ``intensity_pct``."""
        levels = _mk_levels(
            [
                {
                    "z": 1,
                    "a": 2,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": -1.0,
                    "jpi": 1.0,
                    "n_gammas": 0,
                },
                {
                    "z": 1,
                    "a": 2,
                    "level_idx": 1,
                    "floating_flag": "-",
                    "excitation_kev": 50.0,
                    "half_life_s": 1e-9,
                    "jpi": 0.0,
                    "n_gammas": 1,
                },
            ]
        )
        gammas = _mk_gammas(
            [
                {
                    "z": 1,
                    "a": 2,
                    "parent_level": 1,
                    "daughter_level": 0,
                    "gamma_energy_kev": 50.0,
                    "intensity": 73.5,
                    "multipolarity": 2,
                    "mixing_ratio": 0.0,
                    "icc_total": 0.0,
                },
            ]
        )
        out = transform(gammas, levels)
        assert out["intensity_pct"].to_list() == pytest.approx([73.5])

    def test_join_failure_raises(self) -> None:
        """A gamma row whose parent_level idx isn't in the levels table is
        an upstream-data integrity failure; converter must raise rather than
        silently emit a NULL parent_level_keV."""
        levels = _mk_levels(
            [
                {
                    "z": 1,
                    "a": 1,
                    "level_idx": 0,
                    "floating_flag": "-",
                    "excitation_kev": 0.0,
                    "half_life_s": -1.0,
                    "jpi": 0.5,
                    "n_gammas": 0,
                },
            ]
        )
        gammas = _mk_gammas(
            [
                # parent_level=42 not present in levels.
                {
                    "z": 1,
                    "a": 1,
                    "parent_level": 42,
                    "daughter_level": 0,
                    "gamma_energy_kev": 100.0,
                    "intensity": 1.0,
                    "multipolarity": 1,
                    "mixing_ratio": 0.0,
                    "icc_total": 0.0,
                },
            ]
        )
        with pytest.raises(ValueError, match="parent_level_keV join"):
            transform(gammas, levels)


# ---------------------------------------------------------------------------
# Integration spot checks (require shipped data; auto-skip otherwise).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_radiation_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build per-element radiation files into a tmp dir from shipped strata
    inputs. Module-scoped so the 297k-row build runs once across data tests.

    Reads ``conftest.DATA_DIR`` directly (rather than the function-scoped
    ``data_dir_path`` fixture) to keep this fixture module-scoped.
    """
    from conftest import DATA_DIR  # type: ignore[import-not-found]

    if DATA_DIR is None:
        pytest.skip("nucl-parquet data not available")
    out = tmp_path_factory.mktemp("radiation_g4")
    build(data_dir=DATA_DIR, out_dir=out)
    return out


@pytest.mark.data
def test_data_max_parent_level_keV_below_ceiling(built_radiation_dir: Path) -> None:
    """Sweep test pinned by ADR-0002: ``parent_level_keV.max() < 1e8 keV``.

    Would have caught the v0.10.x IAEA-fetcher 2e+215 corruption. G4 source
    values stay well under 50 MeV; this is a defensive ceiling.
    """
    glob = str(built_radiation_dir / "*.parquet")
    df = pl.read_parquet(glob)
    max_parent = df.select(pl.col("parent_level_keV").max()).item()
    assert max_parent is not None
    assert max_parent < PARENT_LEVEL_KEV_CEILING, (
        f"parent_level_keV.max()={max_parent} keV exceeds ceiling {PARENT_LEVEL_KEV_CEILING:g} keV — DQ regression."
    )
    # Sanity: G4 values are in the keV-MeV range.
    assert max_parent < 1.0e6, (
        f"parent_level_keV.max()={max_parent} unexpectedly above 1 GeV; investigate strata input."
    )


@pytest.mark.data
def test_data_eu152_m2_cascade_present(built_radiation_dir: Path) -> None:
    """Eu-152 has exactly one long-lived m2 isomer at 147.86 keV (T½=96 min)
    that emits a single cascade gamma in the G4 PhotonEvaporation data.

    This is the only Eu-152 *photon-evap* row that gets state='m2'; it
    pins both the state assignment (147.86 keV → m2 via 0.1 keV match) and
    the per-element partition (Z=63 → Eu.parquet).
    """
    df = pl.read_parquet(built_radiation_dir / "Eu.parquet")
    m2 = df.filter(
        (pl.col("Z") == 63)
        & (pl.col("A") == 152)
        & (pl.col("state") == "m2")
        & ((pl.col("parent_level_keV") - 147.86).abs() < 0.5)
    )
    assert m2.height >= 1, "Eu-152 m2 cascade gamma (parent ≈ 147.86 keV) missing"


@pytest.mark.data
def test_data_tc99m_isomer_gammas_present(built_radiation_dir: Path) -> None:
    """Tc-99m at 142.68 keV decays via two gammas (the 2.17 keV M4 isomeric
    transition and the 142.68 keV cross-over). Both must inherit state='m'
    from the parent_level_keV match against the 142.6836 keV isomer level.
    """
    df = pl.read_parquet(built_radiation_dir / "Tc.parquet")
    m_rows = df.filter((pl.col("Z") == 43) & (pl.col("A") == 99) & (pl.col("state") == "m"))
    assert m_rows.height >= 2, f"Expected ≥2 Tc-99m gamma rows (cross-over + IT), got {m_rows.height}"
    parent_keVs = m_rows["parent_level_keV"].to_list()
    assert all(abs(p - 142.6836) < 0.1 for p in parent_keVs), (
        f"Tc-99m gammas must originate at the 142.68 keV isomer; got {parent_keVs}"
    )


@pytest.mark.data
def test_data_schema_compat_existing_consumers(built_radiation_dir: Path) -> None:
    """Every column the v0.10.x Rust crate / MCP / DuckDB views project from
    radiation/*.parquet must still exist with compatible dtype.

    Pinned columns: Z, A, state, rad_type, energy_keV, intensity_pct,
    decay_mode (nullable), rad_subtype (nullable), parent_level_keV.
    """
    df = pl.read_parquet(built_radiation_dir / "Eu.parquet")
    schema = df.schema
    # Identifiers.
    assert schema["Z"] in (pl.Int32, pl.Int64)
    assert schema["A"] in (pl.Int32, pl.Int64)
    # Strings.
    assert schema["state"] == pl.String
    assert schema["rad_type"] == pl.String
    assert schema["decay_mode"] == pl.String
    assert schema["rad_subtype"] == pl.String
    # Floats.
    assert schema["energy_keV"] == pl.Float64
    assert schema["intensity_pct"] == pl.Float64
    assert schema["parent_level_keV"] == pl.Float64
    # Bonus columns must exist (additive per ADR-0002).
    assert "multipolarity" in schema
    assert "mixing_ratio" in schema
    assert "icc_total" in schema
    assert "daughter_level_keV" in schema


@pytest.mark.data
def test_data_bonus_columns_populated(built_radiation_dir: Path) -> None:
    """Additive bonus columns must carry data, not be all-NULL — verifies the
    LEFT JOIN against levels (daughter_level_keV) and the strata pass-through
    (multipolarity, mixing_ratio, icc_total) actually work end-to-end.
    """
    glob = str(built_radiation_dir / "*.parquet")
    df = pl.read_parquet(glob)
    assert df.height > 100_000, f"Expected ≥100 k gamma rows; got {df.height}"
    # Most rows should have a non-NULL multipolarity (G4 always emits one
    # even if the value is 0=unknown).
    assert df["multipolarity"].null_count() == 0, "multipolarity is non-nullable in strata input; should never be NULL"
    # daughter_level_keV is the LEFT-JOINed lookup; 100% hit rate expected.
    assert df["daughter_level_keV"].null_count() == 0, (
        "daughter_level_keV LEFT JOIN should be 100% covered by strata data"
    )
    # icc_total non-zero for at least some rows (otherwise the pass-through is
    # a silent zero-fill).
    nonzero_icc = df.filter(pl.col("icc_total") > 0.0).height
    assert nonzero_icc > 1000, f"Only {nonzero_icc} rows have non-zero icc_total; pass-through broken?"
