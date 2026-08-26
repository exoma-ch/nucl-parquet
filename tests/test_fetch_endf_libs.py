"""Tests for the ENDF-6 ingest (`scripts/fetch_endf_libs.py`), centred on MF=10.

MF=10 is where the ground/metastable production split lives. It was read through
an object shape the `endf` package has never returned — `section["subsections"]`
and `sub["ZAPS"]` — so `.get()` handed back its default on every section, the
loop body never ran, and every isomeric-production section in every library was
discarded while the ingest exited 0 (#340).

That is the fourth silent-success defect in this one file, so these tests are
written to fail loudly on the *specific* thing that broke:

* `test_endf_package_mf10_shape_is_pinned` asserts the exact keys the pinned
  `endf` package returns. Bump `endf` to a version that moves them and this
  fails, instead of the data quietly emptying out again.
* `test_dead_shape_from_340_is_really_absent` asserts the *old* keys are absent,
  so nobody reintroduces the dead lookup believing it once worked.
* the guard tests assert the ingest *raises* rather than reporting success.

Everything asserts a positive value — a row count, a cross-section, a state
string. A test that passes because it found nothing is what let #340 ship.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _mod():
    import fetch_endf_libs as m

    return m


# ---------------------------------------------------------------------------
# A synthetic ENDF-6 material — no network, no fixture blob to go stale
# ---------------------------------------------------------------------------
#
# Modelled on IRDFF-II's Al-27 evaluation, which really does carry MF=3 MT=16
# (total (n,2n) → Al-26) alongside MF=10 MT=16 splitting the same product into
# LFS=0 and LFS=1. Values are round numbers so the assertions read as physics.

MAT = 1325


def _endf_float(value: float) -> str:
    """Format a float in ENDF-6's 11-column 'e-less' notation ('1.234560+6')."""
    mantissa, exponent = f"{value:.6E}".split("E")
    exp = int(exponent)
    text = f"{mantissa}{exp:+d}".replace("+0", "+").replace("-0", "-")
    if len(text) > 11:  # 3-digit exponent: borrow a column from the mantissa
        mantissa = mantissa[: len(mantissa) - (len(text) - 11)]
        text = f"{mantissa}{exp:+d}"
    return f"{text:>11}"


def _endf_int(value: int) -> str:
    return f"{value:>11d}"


def _line(fields: str, mf: int, mt: int, seq: int) -> str:
    return f"{fields:<66}{MAT:>4d}{mf:>2d}{mt:>3d}{seq:>5d}\n"


def _tab1(params: list, xy: list[tuple[float, float]], mf: int, mt: int, seq: int) -> tuple[str, int]:
    """One TAB1 record: 6 head fields, one NBT/INT pair, then the x/y pairs."""
    head = "".join(params) + _endf_int(1) + _endf_int(len(xy))
    out = _line(head, mf, mt, seq)
    seq += 1
    out += _line(_endf_int(len(xy)) + _endf_int(2), mf, mt, seq)  # lin-lin
    seq += 1
    flat = [v for pair in xy for v in pair]
    for i in range(0, len(flat), 6):
        out += _line("".join(_endf_float(v) for v in flat[i : i + 6]), mf, mt, seq)
        seq += 1
    return out, seq


def mf3_section(za: float, mt: int, xy: list[tuple[float, float]]) -> str:
    head = _endf_float(za) + _endf_float(26.75) + _endf_int(0) * 4
    out = _line(head, 3, mt, 1)
    body, _ = _tab1([_endf_float(0.0), _endf_float(0.0), _endf_int(0), _endf_int(0)], xy, 3, mt, 2)
    return out + body + _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 3, 0, 99999)


def mf10_section(za: float, mt: int, levels: list[tuple[int, int, list]]) -> str:
    """levels: [(IZAP, LFS, [(E_eV, sigma_barns), ...]), ...]"""
    head = (
        _endf_float(za)
        + _endf_float(26.75)
        + _endf_int(0)  # LIS — ground-state target
        + _endf_int(0)
        + _endf_int(len(levels))  # NS
        + _endf_int(0)
    )
    out = _line(head, 10, mt, 1)
    seq = 2
    for izap, lfs, xy in levels:
        body, seq = _tab1(
            [_endf_float(0.0), _endf_float(0.0), _endf_int(izap), _endf_int(lfs)],
            xy,
            10,
            mt,
            seq,
        )
        out += body
    return out + _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 10, 0, 99999)


# Al-26 from (n,2n): a 100 mb ground part and a 40 mb metastable part.
AL26_G = [(1.4e7, 0.060), (2.0e7, 0.100), (3.0e7, 0.080)]
AL26_M = [(1.4e7, 0.020), (2.0e7, 0.040), (3.0e7, 0.030)]
# The MF=3 total for the same product is, as in real evaluations, the sum.
AL26_TOTAL = [(e, g + m) for (e, g), (_, m) in zip(AL26_G, AL26_M)]

# Na-24 reached by two different MTs — must be summed, not emitted twice.
NA24_A = [(5.0e6, 0.050), (1.4e7, 0.120)]
NA24_B = [(5.0e6, 0.010), (1.4e7, 0.030)]

# Two isomers above ground, to exercise 'm' / 'm2' ranking, plus a TALYS
# overflow sentinel and a zero that must both be dropped.
MG27 = [(1.0e7, 0.010), (2.0e7, 1.99e35), (3.0e7, 0.0), (4.0e7, 0.020)]


def synthetic_material() -> str:
    za = 13027.0
    parts = [
        "".ljust(66) + "TPID\n",
        mf3_section(za, 16, AL26_TOTAL),
        mf10_section(za, 16, [(13026, 0, AL26_G), (13026, 1, AL26_M)]),
        mf10_section(za, 5, [(11024, 0, NA24_A)]),
        mf10_section(za, 22, [(11024, 0, NA24_B)]),
        mf10_section(
            za,
            103,
            [(12027, 0, MG27), (12027, 3, MG27[:1]), (12027, 7, MG27[:1])],
        ),
        _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),  # FEND
        f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",  # MEND
    ]
    return "".join(parts)


@pytest.fixture(scope="module")
def material():
    import endf

    return endf.Material(io.StringIO(synthetic_material()))


@pytest.fixture(scope="module")
def parsed():
    return _mod().parse_endf_file(synthetic_material(), 13, 27, "n")


def _rows(parsed, z, a, state):
    return sorted(
        (r["energy_MeV"], r["xs_mb"])
        for r in parsed.rows
        if (r["residual_Z"], r["residual_A"], r["state"]) == (z, a, state)
    )


# ---------------------------------------------------------------------------
# The version pin — what shape does `endf` actually return for MF=10?
# ---------------------------------------------------------------------------


def test_endf_package_mf10_shape_is_pinned(material):
    """MF=10 sections are {'ZA','AWR','LIS','NS','levels':[{...,'sigma'}]}.

    This is the contract `parse_mf10_rows` reads. If an `endf` upgrade renames
    or nests these, fail here — do not let the isomeric data go quiet again.
    """
    section = material.section_data[10, 16]
    assert set(section) == {"ZA", "AWR", "LIS", "NS", "levels"}
    assert section["ZA"] == 13027
    assert section["LIS"] == 0  # target's own isomeric state
    assert section["NS"] == 2  # two levels of the product

    assert len(section["levels"]) == 2
    for level in section["levels"]:
        assert set(level) == {"QM", "QI", "IZAP", "LFS", "sigma"}
        assert level["IZAP"] == 13026  # Al-26, named directly — not derived
        assert len(level["sigma"].x) == 3

    assert [level["LFS"] for level in section["levels"]] == [0, 1]
    ground, meta = section["levels"]
    assert ground["sigma"].y[1] == pytest.approx(0.100)
    assert meta["sigma"].y[1] == pytest.approx(0.040)


def test_dead_shape_from_340_is_really_absent(material):
    """The keys the old code read have never existed. Pin that, so the dead
    lookup is not reintroduced by someone assuming it used to work."""
    section = material.section_data[10, 16]
    assert "subsections" not in section
    assert section.get("subsections", []) == []  # the exact silent no-op of #340
    for level in section["levels"]:
        assert "ZAPS" not in level


# ---------------------------------------------------------------------------
# LFS -> state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lfs", "levels", "expected"),
    [
        (0, {0}, "g"),  # evaluation asserts ground state
        (0, {0, 1}, "g"),
        (1, {0, 1}, "m"),
        (2, {0, 2}, "m"),  # LFS is a level index: level 2 is still the 1st isomer
        (22, {0, 22}, "m"),  # Fe-53m really is level 22 in ENDF/B-VIII.1
        (72, {0, 72}, "m"),  # TENDL goes this high
        (1, {0, 1, 2}, "m"),
        (2, {0, 1, 2}, "m2"),
        (3, {0, 1, 2, 3}, "m3"),
        (7, {0, 3, 7}, "m2"),  # rank, don't transcribe
    ],
)
def test_lfs_to_state(lfs, levels, expected):
    assert _mod().lfs_to_state(lfs, levels) == expected


def test_lfs_to_state_uses_repo_state_vocabulary():
    """Only '', 'g', 'm', 'm2', 'm3' … join against nuclides.parquet and
    tendl-2023-iso. A literal 'm22' would be an orphan."""
    m = _mod()
    for lfs in (1, 2, 3, 14, 22, 47, 72):
        state = m.lfs_to_state(lfs, {0, lfs})
        assert state == "m", f"LFS={lfs} must rank to 'm', got {state!r}"


# ---------------------------------------------------------------------------
# Rows actually come out, with the right products, states and magnitudes
# ---------------------------------------------------------------------------


def test_mf10_produces_isomeric_rows(parsed):
    """The regression: this was 0 for every library, in every release."""
    assert parsed.mf10_sections == 4
    assert parsed.mf10_rows > 0
    states = {r["state"] for r in parsed.rows}
    assert "m" in states, "no metastable rows — MF=10 is dead again"
    assert "g" in states


def test_counters_report_both_file_types(parsed):
    """The guards run on these numbers, so they must reflect the source, not
    just what happened to survive."""
    assert parsed.mf3_sections == 1  # the synthetic material carries MF=3 MT=16
    assert parsed.mf3_rows == 3
    assert parsed.mf10_sections == 4
    assert parsed.mf3_rows + parsed.mf10_rows == len(parsed.rows)


def test_mf10_ground_and_metastable_carry_their_own_cross_sections(parsed):
    """Al-26 g and m are separate products with separate magnitudes."""
    ground = _rows(parsed, 13, 26, "g")
    meta = _rows(parsed, 13, 26, "m")
    assert [xs for _e, xs in ground] == pytest.approx([60.0, 100.0, 80.0])
    assert [xs for _e, xs in meta] == pytest.approx([20.0, 40.0, 30.0])
    assert [e for e, _xs in ground] == pytest.approx([14.0, 20.0, 30.0])


def test_mf3_total_and_mf10_split_do_not_collide(parsed):
    """MF=3 spells its state '' and MF=10 spells it 'g'/'m', so the 140 mb total
    and its 100 mb ground-state part are different rows, not a silent overwrite
    of one by the other."""
    total = _rows(parsed, 13, 26, "")
    assert [xs for _e, xs in total] == pytest.approx([80.0, 140.0, 110.0])
    for (_e, t), (_e2, g), (_e3, m) in zip(total, _rows(parsed, 13, 26, "g"), _rows(parsed, 13, 26, "m")):
        assert t == pytest.approx(g + m), "sum rule broken"


def test_two_mts_reaching_one_product_are_summed(parsed):
    """Na-24 comes from MT=5 and MT=22 here (JEFF's Sr-86 reaches Rb-84m through
    three MTs). One summed row set per (product, state) — not two overlapping
    ones for a consumer to double-count or arbitrarily pick between."""
    na24 = _rows(parsed, 11, 24, "g")
    assert [e for e, _xs in na24] == pytest.approx([5.0, 14.0])
    assert [xs for _e, xs in na24] == pytest.approx([60.0, 150.0])


def test_second_isomer_is_spelled_m2(parsed):
    """Mg-27 is shipped at levels 0, 3 and 7 → 'g', 'm', 'm2'."""
    assert _rows(parsed, 12, 27, "g")
    assert _rows(parsed, 12, 27, "m")
    assert _rows(parsed, 12, 27, "m2")


def test_overflow_sentinels_and_zeros_are_dropped(parsed):
    """TALYS writes ~1.99e35 b on overflow; that must never reach a row."""
    ground = _rows(parsed, 12, 27, "g")
    assert [e for e, _xs in ground] == pytest.approx([10.0, 40.0])
    assert max(r["xs_mb"] for r in parsed.rows) < 1e6


def test_every_row_matches_the_parquet_schema(parsed):
    expected = {"target_A", "residual_Z", "residual_A", "state", "energy_MeV", "xs_mb"}
    assert parsed.rows
    for row in parsed.rows:
        assert set(row) == expected
        assert row["target_A"] == 27
        assert row["state"] in ("", "g", "m", "m2", "m3")
        assert row["xs_mb"] > 0


# ---------------------------------------------------------------------------
# sum_on_union_grid
# ---------------------------------------------------------------------------


def test_sum_on_union_grid_interpolates_onto_the_union():
    import numpy as np

    m = _mod()
    a = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
    b = (np.array([2.0, 4.0]), np.array([5.0, 15.0]))
    e, xs = m.sum_on_union_grid([a, b])
    assert list(e) == [1.0, 2.0, 3.0, 4.0]
    # b contributes 0 below its threshold and above its max; a likewise.
    assert list(xs) == pytest.approx([10.0, 25.0, 40.0, 15.0])


def test_sum_on_union_grid_passes_a_lone_contribution_through():
    import numpy as np

    m = _mod()
    e_in, xs_in = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    e, xs = m.sum_on_union_grid([(e_in, xs_in)])
    assert list(e) == [1.0, 2.0]
    assert list(xs) == [3.0, 4.0]


# ---------------------------------------------------------------------------
# The guards — an ingest that drops data must not exit 0
# ---------------------------------------------------------------------------


def _stub_library(monkeypatch, parsed_files: dict):
    """Run fetch_library against canned per-file parse results, no network."""
    m = _mod()
    monkeypatch.setattr(m, "list_endf_files", lambda *a, **k: list(parsed_files))
    monkeypatch.setattr(m, "download_and_parse", lambda lib, sub, fname, sess: parsed_files[fname])
    return m


# Al-27(n,p)Mg-27 as MF=3 would file it, and an MF=10 metastable row.
_MF3_ROW = {
    "target_A": 27,
    "residual_Z": 12,
    "residual_A": 27,
    "state": "",
    "energy_MeV": 14.0,
    "xs_mb": 80.0,
}
_MF10_ROW = {**_MF3_ROW, "residual_Z": 13, "residual_A": 26, "state": "m", "xs_mb": 40.0}


def test_guard_fires_when_mf10_sections_yield_no_rows(monkeypatch, tmp_path):
    """The #340 signature: MF=3 still produces plenty of rows, so the #334
    empty-ingest guard sees a healthy run. This one must not."""
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW],
                mf3_sections=12,
                mf3_rows=1,
                mf10_sections=4,
                mf10_rows=0,
            )
        },
    )
    with pytest.raises(RuntimeError, match="MF=10"):
        m.fetch_library("irdff-2", "n", tmp_path, session=None)
    assert not (tmp_path / "irdff-2" / "manifest.json").exists()


def test_guard_fires_when_mf3_sections_yield_no_rows(monkeypatch, tmp_path):
    """Symmetric to the MF=10 guard, and only reachable since #340: MF=10 rows
    can now keep the element count healthy while every channel cross-section
    disappears, which the empty-ingest guard would read as a successful run."""
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF10_ROW],
                mf3_sections=12,
                mf3_rows=0,
                mf10_sections=4,
                mf10_rows=1,
            )
        },
    )
    with pytest.raises(RuntimeError, match="MF=3"):
        m.fetch_library("irdff-2", "n", tmp_path, session=None)
    assert not (tmp_path / "irdff-2" / "manifest.json").exists()


def test_guard_is_quiet_when_the_library_genuinely_has_no_mf10(monkeypatch, tmp_path):
    """CENDL-3.2 and BROND-3.1 ship no MF=10 at all. Zero isomeric rows is the
    correct outcome there and must not raise."""
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW],
                mf3_sections=12,
                mf3_rows=1,
                mf10_sections=0,
                mf10_rows=0,
            )
        },
    )
    m.fetch_library("cendl-3.2", "n", tmp_path, session=None)
    manifest = json.loads((tmp_path / "cendl-3.2" / "manifest.json").read_text())
    assert manifest["mf10_sections"] == 0
    assert manifest["mf3_rows"] == 1
    assert manifest["states"] == {"": 1}


def test_empty_ingest_guard_raises_rather_than_returning(monkeypatch, tmp_path):
    """#334 added this guard but left an earlier `warning(); return` on the same
    condition above it, so the raise was unreachable and BROND-3.1-style empty
    ingests would still have exited 0."""
    m = _stub_library(monkeypatch, {"weird-name.zip": _mod().ParsedFile(rows=[])})
    with pytest.raises(RuntimeError, match="0 elements"):
        m.fetch_library("brond-3.1", "n", tmp_path, session=None)


def test_manifest_records_the_isomeric_yield(monkeypatch, tmp_path):
    """So a reader can tell 'ships no isomeric data' from 'isomeric data was
    dropped' without re-running a multi-GB ingest."""
    rows = _mod().parse_endf_file(synthetic_material(), 13, 27, "n")
    m = _stub_library(monkeypatch, {"n_013-Al-27_1325.zip": rows})
    m.fetch_library("irdff-2", "n", tmp_path, session=None)

    manifest = json.loads((tmp_path / "irdff-2" / "manifest.json").read_text())
    assert manifest["mf10_sections"] == 4
    assert manifest["mf10_rows"] == rows.mf10_rows > 0
    assert manifest["states"]["m"] > 0
    assert manifest["states"]["g"] > 0
    assert sum(manifest["states"].values()) == manifest["total_rows"]


def test_written_parquet_carries_the_states(monkeypatch, tmp_path):
    import polars as pl

    rows = _mod().parse_endf_file(synthetic_material(), 13, 27, "n")
    m = _stub_library(monkeypatch, {"n_013-Al-27_1325.zip": rows})
    m.fetch_library("irdff-2", "n", tmp_path, session=None)

    df = pl.read_parquet(tmp_path / "irdff-2" / "xs" / "n_Al.parquet")
    assert set(df["state"].unique()) == {"", "g", "m", "m2"}
    al26m = df.filter((pl.col("residual_Z") == 13) & (pl.col("residual_A") == 26) & (pl.col("state") == "m"))
    assert al26m["xs_mb"].max() == pytest.approx(40.0)
