"""Tests for the radioactive_decay G4 → v0.10.x converter (issue #71).

Two layers, mirroring `tests/test_state.py`:

- Synthetic unit tests on small hand-built DataFrames covering the transform's
  surface (mode mapping, half-life sentinel, state synthesis, IT-on-ground
  filter, duplicate handling, renormalization, schema enforcement).
- `@pytest.mark.data` integrity tests over the real strata source +
  catalog, asserting the issue-#71 acceptance criteria (Eu-152 splits,
  Tc-99m IT, no IT-on-ground, branching sums ∈ [0.95, 1.05]).

Notes
-----

Synthetic tests do NOT touch parquet files on disk; they call the transform
functions directly on in-memory DataFrames. The data tests run only when
`data/g4_raw/strata-nuclear/radioactive_decay.parquet` and
`data/meta/ensdf/nuclides.parquet` are both present, otherwise they skip.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.radioactive_decay import (
    _M_STATE_KEV_TOLERANCE,
    G4_TO_V010_DECAY_MODE,
    _convert_half_life,
    _label_state,
    build_decay_detailed_table,
    build_decay_table,
    write_decay_parquets,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_PATH = REPO_ROOT / "data" / "g4_raw" / "strata-nuclear" / "radioactive_decay.parquet"
NUCLIDES_PATH = REPO_ROOT / "data" / "meta" / "ensdf" / "nuclides.parquet"

# Skip-if-missing markers. Synthetic tests don't need either file.
_real_data = pytest.mark.skipif(
    not (SRC_PATH.exists() and NUCLIDES_PATH.exists()),
    reason=f"strata source or nuclides catalog missing ({SRC_PATH}, {NUCLIDES_PATH})",
)


# ---------------------------------------------------------------------------
# Synthetic fixture: a tiny strata-shaped DataFrame.
# ---------------------------------------------------------------------------


def _make_src_row(
    parent_z: int,
    parent_a: int,
    parent_ex_kev: float,
    decay_mode: str,
    branching: float,
    *,
    daughter_z: int | None = None,
    daughter_a: int | None = None,
    daughter_ex_kev: float = 0.0,
    half_life_s: float = 100.0,
    is_summary: bool = True,
    parent_level_flag: str = "-",
    daughter_level_flag: str = "-",
    q_value_kev: float = float("nan"),
    forbiddenness: str = "",
) -> dict:
    """Build a single strata-shape row dict; inferred daughter via mode if not given."""
    if daughter_z is None or daughter_a is None:
        # Cheap derivation matching strata's table.
        deltas = {
            "BetaMinus": (1, 0),
            "BetaPlus": (-1, 0),
            "KshellEC": (-1, 0),
            "LshellEC": (-1, 0),
            "MshellEC": (-1, 0),
            "NshellEC": (-1, 0),
            "Alpha": (-2, -4),
            "IT": (0, 0),
            "SpFission": (0, 0),
            "Proton": (-1, -1),
            "Neutron": (0, -1),
            "Triton": (-1, -3),
        }
        dz, da = deltas[decay_mode]
        daughter_z = parent_z + dz
        daughter_a = parent_a + da
        if decay_mode == "SpFission":
            daughter_z, daughter_a = -1, -1
    return dict(
        parent_z=parent_z,
        parent_a=parent_a,
        parent_ex_kev=parent_ex_kev,
        parent_level_flag=parent_level_flag,
        half_life_s=half_life_s,
        decay_mode=decay_mode,
        daughter_z=daughter_z,
        daughter_a=daughter_a,
        daughter_ex_kev=daughter_ex_kev,
        daughter_level_flag=daughter_level_flag,
        branching_ratio=branching,
        q_value_kev=q_value_kev,
        is_summary=is_summary,
        forbiddenness=forbiddenness,
    )


def _src_df(rows: list[dict]) -> pl.DataFrame:
    """Build a strata-shaped DataFrame with the right dtypes."""
    schema = {
        "parent_z": pl.UInt8,
        "parent_a": pl.UInt16,
        "parent_ex_kev": pl.Float64,
        "parent_level_flag": pl.String,
        "half_life_s": pl.Float64,
        "decay_mode": pl.String,
        "daughter_z": pl.Int16,
        "daughter_a": pl.Int16,
        "daughter_ex_kev": pl.Float64,
        "daughter_level_flag": pl.String,
        "branching_ratio": pl.Float64,
        "q_value_kev": pl.Float64,
        "is_summary": pl.Boolean,
        "forbiddenness": pl.String,
    }
    return pl.DataFrame(rows, schema=schema)


def _nuclides_df(m_states: list[tuple[int, int, float]]) -> pl.DataFrame:
    """Build a nuclides.parquet-shaped frame with a list of (Z, A, m_keV) m-rows."""
    rows = []
    for z, a, level in m_states:
        rows.append(
            dict(
                Z=z,
                A=a,
                state="m",
                symbol="X",
                jp="",
                half_life_s=None,
                level_keV=level,
                decay_1="",
                decay_1_pct=None,
                decay_2="",
                decay_2_pct=None,
            )
        )
    schema = {
        "Z": pl.Int32,
        "A": pl.Int32,
        "state": pl.String,
        "symbol": pl.String,
        "jp": pl.String,
        "half_life_s": pl.Float64,
        "level_keV": pl.Float64,
        "decay_1": pl.String,
        "decay_1_pct": pl.Float64,
        "decay_2": pl.String,
        "decay_2_pct": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# Synthetic unit tests
# ---------------------------------------------------------------------------


class TestModeMapping:
    def test_all_g4_tokens_have_mappings(self) -> None:
        # Sanity: the mapping table covers every token strata's converter
        # emits. If strata adds a new token (e.g. a heavy-ion mode) this
        # test fails loud, prompting the maintainer to extend the table.
        assert set(G4_TO_V010_DECAY_MODE) == {
            "BetaMinus",
            "BetaPlus",
            "KshellEC",
            "LshellEC",
            "MshellEC",
            "NshellEC",
            "IT",
            "Alpha",
            "SpFission",
            "Proton",
            "Neutron",
            "Triton",
        }

    def test_unknown_mode_raises(self, tmp_path: Path) -> None:
        # Bypass the helper's daughter-derivation table to inject a bogus mode.
        good = _make_src_row(1, 3, 0.0, "BetaMinus", 1.0)
        bogus = dict(good)
        bogus["parent_z"] = 2
        bogus["parent_a"] = 4
        bogus["decay_mode"] = "Bogus"
        src = _src_df([good, bogus])
        nu = _nuclides_df([])
        nu_path = tmp_path / "nuclides.parquet"
        nu.write_parquet(nu_path)
        with pytest.raises(ValueError, match="unmapped G4 decay_mode"):
            build_decay_table(src, nu_path)

    def test_per_shell_ec_preserved_not_lumped(self, tmp_path: Path) -> None:
        # K/L/M-shell EC remain as separate rows with their own decay_mode.
        src = _src_df(
            [
                _make_src_row(63, 152, 0.0, "KshellEC", 0.6),
                _make_src_row(63, 152, 0.0, "LshellEC", 0.1),
                _make_src_row(63, 152, 0.0, "MshellEC", 0.03),
                _make_src_row(63, 152, 0.0, "BetaMinus", 0.27),
            ]
        )
        nu_path = tmp_path / "nuclides.parquet"
        _nuclides_df([]).write_parquet(nu_path)
        out = build_decay_table(src, nu_path)
        modes = set(out["decay_mode"].to_list())
        assert "KshellEC" in modes
        assert "LshellEC" in modes
        assert "MshellEC" in modes
        assert "EC" not in modes  # never collapsed


class TestHalfLifeConversion:
    def test_stable_sentinel_to_inf(self) -> None:
        df = pl.DataFrame({"hl": [-1.0, 100.0, 1e-9, float("nan")]})
        out = df.select(_convert_half_life(pl.col("hl")).alias("hl_out"))
        vals = out["hl_out"].to_list()
        assert vals[0] == math.inf
        assert vals[1] == 100.0
        assert vals[2] == 1e-9
        assert vals[3] is None

    def test_negative_sentinel_does_not_silently_propagate(self) -> None:
        # Boundary check: a -1 must NOT come through as -1 or as a negative
        # half-life from naive arithmetic.
        df = pl.DataFrame({"hl": [-1.0]})
        out = df.select(_convert_half_life(pl.col("hl")).alias("hl_out"))
        v = out["hl_out"].item()
        assert v == math.inf
        assert v > 0  # tautological after the first assert, kept for documentation


class TestStateLabelling:
    def test_ground_gets_empty_string(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "parent_z": [43],
                "parent_a": [99],
                "parent_ex_kev": [0.0],
            }
        )
        m_levels = pl.DataFrame(
            {"Z": [43], "A": [99], "m_keV": [142.6836]},
            schema={"Z": pl.Int32, "A": pl.Int32, "m_keV": pl.Float64},
        )
        out = _label_state(
            df,
            z_col="parent_z",
            a_col="parent_a",
            ex_col="parent_ex_kev",
            out_col="state",
            m_levels=m_levels,
        )
        assert out["state"].to_list() == [""]

    def test_m_state_within_tolerance(self, tmp_path: Path) -> None:
        # Eu-152 G4 emits 45.5998; catalog says 45.6. Diff = 0.0002 < 1.0.
        df = pl.DataFrame(
            {
                "parent_z": [63],
                "parent_a": [152],
                "parent_ex_kev": [45.5998],
            }
        )
        m_levels = pl.DataFrame(
            {"Z": [63], "A": [152], "m_keV": [45.6]},
            schema={"Z": pl.Int32, "A": pl.Int32, "m_keV": pl.Float64},
        )
        out = _label_state(
            df,
            z_col="parent_z",
            a_col="parent_a",
            ex_col="parent_ex_kev",
            out_col="state",
            m_levels=m_levels,
        )
        assert out["state"].to_list() == ["m"]

    def test_only_closest_match_gets_m(self) -> None:
        # Ba-127 has G4 levels at 80.32 *and* 81.29 keV, both within ±1 keV
        # of the catalog's m at 80.3. Only the closer one (80.32) gets `m`.
        df = pl.DataFrame(
            {
                "parent_z": [56, 56],
                "parent_a": [127, 127],
                "parent_ex_kev": [80.32, 81.29],
            }
        )
        m_levels = pl.DataFrame(
            {"Z": [56], "A": [127], "m_keV": [80.3]},
            schema={"Z": pl.Int32, "A": pl.Int32, "m_keV": pl.Float64},
        )
        out = _label_state(
            df,
            z_col="parent_z",
            a_col="parent_a",
            ex_col="parent_ex_kev",
            out_col="state",
            m_levels=m_levels,
        )
        states = out.sort("parent_ex_kev")["state"].to_list()
        assert states == ["m", None]

    def test_excited_with_no_catalog_match_is_null(self) -> None:
        df = pl.DataFrame(
            {
                "parent_z": [27],
                "parent_a": [60],
                "parent_ex_kev": [1500.0],  # nowhere near any m-state
            }
        )
        m_levels = pl.DataFrame(
            {"Z": [27], "A": [60], "m_keV": [58.6]},
            schema={"Z": pl.Int32, "A": pl.Int32, "m_keV": pl.Float64},
        )
        out = _label_state(
            df,
            z_col="parent_z",
            a_col="parent_a",
            ex_col="parent_ex_kev",
            out_col="state",
            m_levels=m_levels,
        )
        assert out["state"].to_list() == [None]

    def test_tolerance_constant_is_one_keV(self) -> None:
        # If someone tightens the tolerance (e.g. to 0.1 keV), Eu-152m's
        # 0.0002 keV gap still passes — but Pm-148's ENSDF rounding can be
        # ~0.5 keV, hence we deliberately keep 1 keV. This test pins it.
        assert _M_STATE_KEV_TOLERANCE == 1.0


class TestITonGroundFilter:
    def test_filter_strips_offending_rows_and_renormalizes(self, tmp_path: Path) -> None:
        # Br-92-like: G4 source carries IT 0.5 + B- 0.5 on ground. We strip
        # the IT row and renormalize B- to 1.0.
        src = _src_df(
            [
                _make_src_row(35, 92, 0.0, "IT", 0.5),
                _make_src_row(35, 92, 0.0, "BetaMinus", 0.5),
            ]
        )
        nu_path = tmp_path / "nuclides.parquet"
        _nuclides_df([]).write_parquet(nu_path)
        out = build_decay_table(src, nu_path)
        # No IT-on-ground row.
        assert out.filter((pl.col("state") == "") & (pl.col("decay_mode") == "IT")).height == 0
        # B- renormalized to 1.0.
        b = out.filter(pl.col("decay_mode") == "beta-")
        assert b.height == 1
        assert b["branching"].item() == pytest.approx(1.0)

    def test_isomer_IT_is_preserved(self, tmp_path: Path) -> None:
        # Tc-99m IT to ground stays — only ground-state IT is stripped.
        src = _src_df(
            [
                _make_src_row(43, 99, 142.6836, "IT", 0.99996),
                _make_src_row(43, 99, 142.6836, "BetaMinus", 0.000037),
            ]
        )
        nu_path = tmp_path / "nuclides.parquet"
        _nuclides_df([(43, 99, 142.6836)]).write_parquet(nu_path)
        out = build_decay_table(src, nu_path)
        it_rows = out.filter(pl.col("decay_mode") == "IT")
        assert it_rows.height == 1
        assert it_rows["state"].item() == "m"


class TestFloatingLevelFilter:
    def test_only_canonical_parent_level_flag_kept(self, tmp_path: Path) -> None:
        # Ir-171-like: G4 ships ground entries with both `-` and `+X` flags.
        # Legacy schema can only carry one — we keep `-`.
        src = _src_df(
            [
                _make_src_row(77, 171, 0.0, "Alpha", 0.15, parent_level_flag="-"),
                _make_src_row(77, 171, 0.0, "Alpha", 0.85, parent_level_flag="+X"),
            ]
        )
        nu_path = tmp_path / "nuclides.parquet"
        _nuclides_df([]).write_parquet(nu_path)
        out = build_decay_table(src, nu_path)
        assert out.filter(pl.col("decay_mode") == "alpha").height == 1
        assert out["branching"].sum() == pytest.approx(0.15)


class TestSpFissionDaughter:
    def test_negative_one_daughter_becomes_null(self, tmp_path: Path) -> None:
        src = _src_df(
            [
                _make_src_row(98, 252, 0.0, "Alpha", 0.96),
                _make_src_row(98, 252, 0.0, "SpFission", 0.04, daughter_z=-1, daughter_a=-1),
            ]
        )
        nu_path = tmp_path / "nuclides.parquet"
        _nuclides_df([]).write_parquet(nu_path)
        out = build_decay_table(src, nu_path)
        sf = out.filter(pl.col("decay_mode") == "SF")
        assert sf.height == 1
        assert sf["daughter_Z"].item() is None
        assert sf["daughter_A"].item() is None


class TestSchemaEnforcement:
    def test_decay_parquet_has_legacy_schema(self, tmp_path: Path) -> None:
        src = _src_df([_make_src_row(1, 3, 0.0, "BetaMinus", 1.0)])
        nu_path = tmp_path / "nuclides.parquet"
        _nuclides_df([]).write_parquet(nu_path)
        decay_path, detailed_path = write_decay_parquets(_write_temp_src(src, tmp_path), nu_path, tmp_path)
        df = pl.read_parquet(decay_path)
        assert df.schema == {
            "Z": pl.Int32,
            "A": pl.Int32,
            "state": pl.String,
            "half_life_s": pl.Float64,
            "decay_mode": pl.String,
            "daughter_Z": pl.Int32,
            "daughter_A": pl.Int32,
            "daughter_state": pl.String,
            "branching": pl.Float64,
        }

    def test_decay_detailed_has_full_schema(self, tmp_path: Path) -> None:
        src = _src_df(
            [
                _make_src_row(
                    1,
                    3,
                    0.0,
                    "BetaMinus",
                    1.0,
                    is_summary=False,
                    q_value_kev=18.591,
                    forbiddenness="firstForbidden",
                )
            ]
        )
        nu_path = tmp_path / "nuclides.parquet"
        _nuclides_df([]).write_parquet(nu_path)
        detailed = build_decay_detailed_table(src)
        assert "q_value_kev" in detailed.columns
        assert "forbiddenness" in detailed.columns
        assert "parent_ex_kev" in detailed.columns
        assert detailed["q_value_kev"].item() == pytest.approx(18.591)
        assert detailed["forbiddenness"].item() == "firstForbidden"


def _write_temp_src(df: pl.DataFrame, tmp_path: Path) -> Path:
    p = tmp_path / "src.parquet"
    df.write_parquet(p)
    return p


# ---------------------------------------------------------------------------
# Real-data integrity tests (issue #71 acceptance criteria)
# ---------------------------------------------------------------------------


@pytest.mark.data
@_real_data
class TestRealDataAcceptance:
    @pytest.fixture(scope="class")
    def decay(self) -> pl.DataFrame:
        src = pl.read_parquet(SRC_PATH)
        return build_decay_table(src, NUCLIDES_PATH)

    @pytest.fixture(scope="class")
    def detailed(self) -> pl.DataFrame:
        src = pl.read_parquet(SRC_PATH)
        return build_decay_detailed_table(src)

    def test_eu152_ground_branchings(self, decay: pl.DataFrame) -> None:
        eu = decay.filter((pl.col("Z") == 63) & (pl.col("A") == 152) & (pl.col("state") == ""))
        b_minus = eu.filter(pl.col("decay_mode") == "beta-")["branching"].sum()
        ec_b_plus = eu.filter(pl.col("decay_mode").is_in(["KshellEC", "LshellEC", "MshellEC", "NshellEC", "beta+"]))[
            "branching"
        ].sum()
        # Canonical: B- 27.92 %, EC+B+ 72.08 %.
        assert b_minus == pytest.approx(0.2792, abs=1e-3)
        assert ec_b_plus == pytest.approx(0.7208, abs=1e-3)

    def test_tc99m_IT_present(self, decay: pl.DataFrame) -> None:
        tc = decay.filter(
            (pl.col("Z") == 43) & (pl.col("A") == 99) & (pl.col("state") == "m") & (pl.col("decay_mode") == "IT")
        )
        assert tc.height == 1
        assert tc["branching"].item() == pytest.approx(0.99996, abs=1e-4)

    def test_no_IT_on_ground_anywhere(self, decay: pl.DataFrame) -> None:
        # The v0.10.x bug-class canary. Must be zero. Case-insensitive
        # to future-proof against a downstream lower-caser.
        offenders = decay.filter((pl.col("state") == "") & (pl.col("decay_mode").str.to_uppercase() == "IT"))
        assert offenders.height == 0, (
            f"IT-on-ground regression — {offenders.height} rows: {offenders.select('Z', 'A').to_dicts()[:5]}"
        )

    def test_no_IT_on_ground_in_detailed_table(self, detailed: pl.DataFrame) -> None:
        """Same canary as ``test_no_IT_on_ground_anywhere`` but applied to the
        per-transition detail table. PR #83 review (#71) flagged that the
        original test only scanned the summary table; an IT-on-ground row
        could in principle land in detail without showing up in summary.
        Pin both tables together so the bug class can't shape-shift."""
        offenders = detailed.filter(
            (pl.col("parent_ex_kev") == 0.0) & (pl.col("decay_mode").str.to_uppercase() == "IT")
        )
        assert offenders.height == 0, (
            f"IT-on-ground regression in decay_detailed — {offenders.height} rows: "
            f"{offenders.select('parent_z', 'parent_a').to_dicts()[:5]}"
        )

    def test_branching_sums_within_tolerance(self, decay: pl.DataFrame) -> None:
        sums = decay.group_by(["Z", "A", "state"]).agg(pl.col("branching").sum().alias("br"))
        outliers = sums.filter((pl.col("br") < 0.95) | (pl.col("br") > 1.05))
        assert outliers.height == 0, f"branching-sum outliers: {outliers.to_dicts()}"

    def test_no_duplicate_rows(self, decay: pl.DataFrame) -> None:
        keys = [
            "Z",
            "A",
            "state",
            "decay_mode",
            "daughter_Z",
            "daughter_A",
            "daughter_state",
        ]
        dup = decay.group_by(keys).len().filter(pl.col("len") > 1)
        assert dup.height == 0, f"duplicate keys: {dup.head(5).to_dicts()}"

    def test_state_enum_is_legacy(self, decay: pl.DataFrame) -> None:
        # v0.10.x compat: only "" and "m". Anything else is a regression.
        assert set(decay["state"].unique().to_list()) <= {"", "m"}

    def test_detailed_has_per_shell_EC(self, detailed: pl.DataFrame) -> None:
        # Issue #74 (X-ray/Auger) consumes per-shell EC fractions from the
        # detailed table. Pin presence here so they don't silently disappear.
        ec_rows = detailed.filter(pl.col("decay_mode").is_in(["KshellEC", "LshellEC", "MshellEC"]))
        assert ec_rows.height > 1000  # plenty of EC parents in the catalog

    def test_detailed_carries_q_values(self, detailed: pl.DataFrame) -> None:
        # The detail table must expose Q-values for spectroscopy consumers.
        assert detailed["q_value_kev"].is_not_null().any()
