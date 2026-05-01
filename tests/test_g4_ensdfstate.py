"""Tests for the G4 ensdfstate -> nuclides + ground_states converter.

Two layers:

1. **Pure transform unit tests** — exercise ``convert_mean_life`` and
   ``encode_jp`` directly, including the ADR-0002 boundary conditions
   (the ``-1`` G4 stable sentinel must yield ``+inf``, never a negative
   half-life).
2. **Integration spot checks** — build the parquet from the shipped strata
   inputs and verify the canonical isotope set from issue #69. Marked
   ``@pytest.mark.data`` so they auto-skip when raw data is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.ensdfstate import (
    G4_STABLE_MEAN_LIFE_NS,
    G4_UNKNOWN_SPIN_X2,
    LN2,
    LONG_LIVED_MEAN_LIFE_NS_THRESHOLD,
    assign_state_labels,
    build,
    convert_mean_life,
    encode_jp,
)

# ---------------------------------------------------------------------------
# Pure transform primitives — no I/O.
# ---------------------------------------------------------------------------


class TestConvertMeanLife:
    """ADR-0002 transform spec: mean_life_ns -> half_life_s."""

    def test_finite_positive(self) -> None:
        # 1 ns mean life -> ln(2) * 1e-9 s half life.
        assert convert_mean_life(1.0) == pytest.approx(LN2 * 1e-9)

    def test_tc99m_canonical(self) -> None:
        # G4 ENSDFSTATE Tc-99m mean_life_ns == 3.12e13 -> ~21630 s.
        assert convert_mean_life(3.12e13) == pytest.approx(21626, rel=1e-3)

    def test_g4_stable_sentinel_yields_positive_infinity(self) -> None:
        """The most load-bearing test in this module.

        The naive product ``-1 * ln(2) * 1e-9 ≈ -6.93e-10`` is *negative*
        and would silently corrupt downstream queries. The converter MUST
        intercept -1 *before* multiplying.
        """
        result = convert_mean_life(G4_STABLE_MEAN_LIFE_NS)
        assert result == math.inf
        assert result > 0  # paranoia — guard against any sign flip

    def test_zero_is_prompt(self) -> None:
        assert convert_mean_life(0.0) == 0.0

    def test_none_passes_through(self) -> None:
        assert convert_mean_life(None) is None

    def test_no_negative_half_lives_for_any_canonical_input(self) -> None:
        """Sweep ADR-0002's table values — none may produce a negative result."""
        for value in (-1.0, 0.0, 1.0, 1e6, 3.12e13, 1.4114e18):
            result = convert_mean_life(value)
            assert result is None or result >= 0, f"{value!r} -> {result!r}"


class TestEncodeJp:
    """G4ENSDFSTATE has no parity column; jp is J-only."""

    def test_integer_spin(self) -> None:
        assert encode_jp(12) == "6"

    def test_half_integer_spin(self) -> None:
        assert encode_jp(3) == "3/2"

    def test_zero_spin(self) -> None:
        assert encode_jp(0) == "0"

    def test_g4_unknown_sentinel(self) -> None:
        assert encode_jp(G4_UNKNOWN_SPIN_X2) is None

    def test_floating_level_flag_dash_dropped(self) -> None:
        # '-' = "not floating", drop.
        assert encode_jp(3, "-") == "3/2"

    def test_floating_level_flag_token_preserved(self) -> None:
        assert encode_jp(3, "+X") == "3/2(+X)"

    def test_none_spin(self) -> None:
        assert encode_jp(None) is None


class TestAssignStateLabels:
    """State-label synthesis: '' / 'm' / 'm2' ... in ascending level_keV."""

    def _frame(self, rows: list[tuple[int, int, float, float]]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Z": [r[0] for r in rows],
                "A": [r[1] for r in rows],
                "level_keV": [r[2] for r in rows],
                "mean_life_ns": [r[3] for r in rows],
            }
        )

    def test_ground_only(self) -> None:
        # Z=99, A=199, single ground (mean_life > threshold) -> state ''.
        df = self._frame([(99, 199, 0.0, 1e10)])
        out = assign_state_labels(df)
        assert out["state"].to_list() == [""]

    def test_short_lived_excited_filtered_out(self) -> None:
        # Excited level below 1 ms threshold should NOT appear.
        df = self._frame([(99, 199, 0.0, 1e10), (99, 199, 100.0, 1.0)])
        out = assign_state_labels(df)
        assert out["state"].to_list() == [""]
        assert len(out) == 1

    def test_eu152_pattern(self) -> None:
        # ground + m + m2 — the canonical multi-isomer case.
        df = self._frame(
            [
                (63, 152, 0.0, 6.154e17),  # ground (stable mean-life)
                (63, 152, 45.6, 4.836e13),  # m
                (63, 152, 147.86, 8.31e12),  # m2
            ]
        )
        out = assign_state_labels(df).sort("level_keV")
        assert out["state"].to_list() == ["", "m", "m2"]

    def test_stable_sentinel_kept_as_ground(self) -> None:
        # mean_life_ns = -1 (stable) must be retained: ground state of stable
        # isotopes such as H-1 / Fe-56 / Pb-208 must appear.
        df = self._frame([(1, 1, 0.0, G4_STABLE_MEAN_LIFE_NS)])
        out = assign_state_labels(df)
        assert out["state"].to_list() == [""]

    def test_threshold_boundary(self) -> None:
        # Level exactly at the long-lived threshold is excluded (strict >).
        df = self._frame(
            [
                (1, 100, 0.0, 1e20),
                (1, 100, 50.0, LONG_LIVED_MEAN_LIFE_NS_THRESHOLD),
            ]
        )
        out = assign_state_labels(df)
        assert out["state"].to_list() == [""]


# ---------------------------------------------------------------------------
# Integration spot checks against the shipped strata + AME + IUPAC inputs.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def built_artefacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build the converter once per test session, reuse across tests.

    Snapshots the raw strata input into a session-scoped tempdir up front
    so that *other* tests in the suite which exercise the fetcher and
    accidentally pollute ``data/g4_raw/strata-nuclear/`` can't corrupt our
    fixture mid-session (cf. ``test_fetch_from_local_missing_file_raises``).
    """
    repo_data = Path(__file__).parent.parent / "data"
    raw_strata = repo_data / "g4_raw" / "strata-nuclear" / "ensdfstate.parquet"
    if not raw_strata.exists() or raw_strata.stat().st_size < 1024:
        pytest.skip("strata raw data not available — run scripts/fetch_strata_nuclear.py")
    if not (repo_data / "auxiliary" / "ame2020.parquet").exists():
        pytest.skip("auxiliary AME2020 not available")

    import shutil

    tmp = tmp_path_factory.mktemp("g4_ensdfstate")
    snapshot = tmp / "ensdfstate.parquet"
    shutil.copy2(raw_strata, snapshot)

    nuc_out = tmp / "nuclides.parquet"
    gs_out = tmp / "ground_states.parquet"
    build(
        data_dir=repo_data,
        strata_path=snapshot,
        nuclides_out=nuc_out,
        ground_states_out=gs_out,
    )
    return nuc_out, gs_out


@pytest.mark.data
class TestNuclidesSpotChecks:
    """Issue #69 acceptance criteria — canonical _validate() spot checks."""

    @pytest.fixture()
    def nuclides(self, built_artefacts: tuple[Path, Path]) -> pl.DataFrame:
        return pl.read_parquet(built_artefacts[0])

    @pytest.mark.parametrize(
        ("z", "a", "state", "expected_hl_s", "expected_level_keV"),
        [
            # Issue #69 mandatory canonical set.
            (43, 99, "m", 21630.0, 142.6836),  # Tc-99m
            (21, 44, "m", 210600.0, 271.241),  # Sc-44m
            (49, 113, "m", 5976.0, 391.699),  # In-113m
            (47, 108, "m", 1.32e10, 109.466),  # Ag-108m
            (63, 152, "m", 33390.0, 45.5998),  # Eu-152m
            (56, 137, "m", 153.1, 661.659),  # Ba-137m
            # Bonus cases.
            (63, 152, "m2", 5760.0, 147.86),  # Eu-152m2
            (72, 178, "m2", 9.78e8, 2446.09),  # Hf-178m2
        ],
    )
    def test_isomer_half_life_and_level(
        self,
        nuclides: pl.DataFrame,
        z: int,
        a: int,
        state: str,
        expected_hl_s: float,
        expected_level_keV: float,
    ) -> None:
        row = nuclides.filter((pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") == state))
        assert len(row) == 1, f"(Z={z},A={a},state={state!r}) not found / not unique"
        assert row["half_life_s"][0] == pytest.approx(expected_hl_s, rel=0.05)
        assert row["level_keV"][0] == pytest.approx(expected_level_keV, abs=0.01)

    @pytest.mark.parametrize(
        ("z", "a"),
        [(1, 1), (2, 4), (26, 56), (82, 208)],
    )
    def test_stable_isotope_is_positive_infinity(self, nuclides: pl.DataFrame, z: int, a: int) -> None:
        """Stable isotopes must have half_life_s = +inf, NOT NULL."""
        row = nuclides.filter((pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") == ""))
        assert len(row) == 1
        hl = row["half_life_s"][0]
        assert hl is not None, f"({z},{a}) ground has NULL half_life_s — must be +inf"
        assert math.isinf(hl) and hl > 0, f"({z},{a}) hl={hl!r}, expected +inf"

    def test_no_negative_half_lives(self, nuclides: pl.DataFrame) -> None:
        """The -1 G4 sentinel must never multiply through to a negative value."""
        negs = nuclides.filter(pl.col("half_life_s") < 0)
        assert len(negs) == 0, f"found {len(negs)} rows with negative half_life_s"

    def test_schema_v0_10_x_compatible(self, nuclides: pl.DataFrame) -> None:
        """v0.10.x consumers project these column names — additions OK, removals not."""
        required = {
            "Z",
            "A",
            "state",
            "symbol",
            "jp",
            "half_life_s",
            "level_keV",
            "decay_1",
            "decay_1_pct",
            "decay_2",
            "decay_2_pct",
        }
        missing = required - set(nuclides.columns)
        assert not missing, f"v0.10.x columns missing: {missing}"

    def test_dual_carry_columns_present(self, nuclides: pl.DataFrame) -> None:
        """ADR-0002 requires spin_x2 + floating_level_flag + magnetic_moment_jt."""
        for col in ("spin_x2", "floating_level_flag", "magnetic_moment_jt"):
            assert col in nuclides.columns, f"dual-carry column {col!r} missing"


@pytest.mark.data
class TestGroundStatesAuxJoins:
    """ground_states.parquet must populate AME mass-excess + IUPAC abundance."""

    @pytest.fixture()
    def gs(self, built_artefacts: tuple[Path, Path]) -> pl.DataFrame:
        return pl.read_parquet(built_artefacts[1])

    @pytest.mark.parametrize(
        ("z", "a", "expected_mass_excess_keV"),
        [
            (1, 1, 7288.971),  # H-1
            (6, 12, 0.0),  # C-12 (definition of u)
            (26, 56, -60605.4),  # Fe-56
        ],
    )
    def test_ame_mass_excess_populated(self, gs: pl.DataFrame, z: int, a: int, expected_mass_excess_keV: float) -> None:
        row = gs.filter((pl.col("Z") == z) & (pl.col("A") == a))
        assert len(row) == 1
        me = row["mass_excess_keV"][0]
        assert me is not None, f"({z},{a}) AME mass-excess not joined"
        assert me == pytest.approx(expected_mass_excess_keV, abs=10.0)

    @pytest.mark.parametrize(
        ("z", "a", "expected_abundance"),
        [
            # H-1 and C-12 carry abundance directly; Fe ships only the
            # element-level standard_atomic_weight (per-isotope abundance is
            # NULL in the upstream IUPAC table). We assert that JOIN brings
            # the value through wherever the source has one.
            (1, 1, 0.999885),  # H-1
            (6, 12, 0.9893),  # C-12
        ],
    )
    def test_iupac_abundance_populated(self, gs: pl.DataFrame, z: int, a: int, expected_abundance: float) -> None:
        row = gs.filter((pl.col("Z") == z) & (pl.col("A") == a))
        assert len(row) == 1
        ab = row["abundance"][0]
        assert ab is not None, f"({z},{a}) IUPAC abundance not joined"
        assert ab == pytest.approx(expected_abundance, rel=0.05)

    def test_iupac_standard_atomic_weight_populated(self, gs: pl.DataFrame) -> None:
        """Fe-56's per-isotope abundance is NULL in IUPAC, but element-level
        standard_atomic_weight should be present."""
        row = gs.filter((pl.col("Z") == 26) & (pl.col("A") == 56))
        assert len(row) == 1
        saw = row["standard_atomic_weight"][0]
        assert saw == pytest.approx(55.845, rel=0.01)

    def test_left_join_preserves_unmatched_rows(self, gs: pl.DataFrame) -> None:
        """A non-IUPAC isotope (e.g. Tc-99) must still have a ground-state row,
        even though no abundance is available."""
        row = gs.filter((pl.col("Z") == 43) & (pl.col("A") == 99))
        assert len(row) == 1
        assert row["abundance"][0] is None

    def test_one_ground_per_za(self, gs: pl.DataFrame) -> None:
        """ground_states must be keyed exactly on (Z, A)."""
        dup = gs.group_by(["Z", "A"]).len().filter(pl.col("len") > 1)
        assert len(dup) == 0, f"duplicate (Z,A) rows: {dup}"
