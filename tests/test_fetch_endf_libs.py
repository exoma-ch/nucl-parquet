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

# The four channels #347 is about. None of them names a single product, so
# `mt_to_residual` returns None for all four and every one used to be dropped:
# MT=1 total, MT=2 elastic, MT=4 inelastic, MT=18 fission. MT=102 capture is
# here as the control — it does name a residual, and always survived.
TOTAL = [(1.0e5, 5.000), (1.0e6, 4.000), (1.4e7, 3.000)]
ELASTIC = [(1.0e5, 3.000), (1.0e6, 2.500), (1.4e7, 1.500)]
INELASTIC = [(1.0e6, 0.500), (1.4e7, 0.400)]
FISSION = [(1.0e5, 1.200), (1.0e6, 1.100), (1.4e7, 2.079)]
CAPTURE = [(1.0e5, 0.300), (1.0e6, 0.020), (1.4e7, 0.001)]


def synthetic_material() -> str:
    za = 13027.0
    parts = [
        "".ljust(66) + "TPID\n",
        mf3_section(za, 1, TOTAL),
        mf3_section(za, 2, ELASTIC),
        mf3_section(za, 4, INELASTIC),
        mf3_section(za, 18, FISSION),
        mf3_section(za, 102, CAPTURE),
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


def _rows(parsed, z, a, state, kind="production"):
    """Production rows for one (residual, state). `kind` matters since #347:
    the same residual now also appears on per-MT `channel` rows."""
    return sorted(
        (r["energy_MeV"], r["xs_mb"])
        for r in parsed.rows
        if (r["residual_Z"], r["residual_A"], r["state"], r["kind"]) == (z, a, state, kind)
    )


def _channel(parsed, mt):
    """(energy, xs) for one MF=3 MT's `channel` rows."""
    return sorted((r["energy_MeV"], r["xs_mb"]) for r in parsed.rows if r["kind"] == "channel" and r["MT"] == mt)


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
    # MF=3 MT = 1, 2, 4, 16, 18, 102
    assert parsed.mf3_sections == 6
    # Production rows come only from the MTs that name a residual: MT=16 -> Al-26
    # and MT=102 -> Al-28. The other four are channel rows only.
    assert parsed.mf3_rows == len(AL26_TOTAL) + len(CAPTURE)
    assert parsed.mf10_sections == 4
    assert parsed.mf3_rows + parsed.mf10_rows + parsed.channel_rows == len(parsed.rows)


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
    expected = {"target_A", "kind", "MT", "residual_Z", "residual_A", "state", "energy_MeV", "xs_mb"}
    assert parsed.rows
    for row in parsed.rows:
        assert set(row) == expected
        assert row["target_A"] == 27
        assert row["state"] in ("", "g", "m", "m2", "m3")
        assert row["xs_mb"] > 0


# ---------------------------------------------------------------------------
# #347 — MT and null residuals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mt", "curve", "name"),
    [(1, TOTAL, "total"), (2, ELASTIC, "elastic"), (4, INELASTIC, "inelastic"), (18, FISSION, "fission")],
)
def test_channels_that_name_no_residual_are_emitted_not_dropped(parsed, mt, curve, name):
    """The #347 regression, one channel at a time.

    `mt_to_residual` correctly returns None for these — they genuinely name no
    single product. The caller then treated None as "skip", so 17.3M evaluated
    rows carried zero fission, total, elastic or inelastic, and U-235(n,f) — the
    most-cited number in neutron physics — could not be queried at all.
    """
    rows = _channel(parsed, mt)
    assert rows, f"MT={mt} ({name}) produced no rows — #347 is back"
    assert [xs for _e, xs in rows] == pytest.approx([xs * 1e3 for _e, xs in curve])
    assert [e for e, _xs in rows] == pytest.approx([e * 1e-6 for e, _xs in curve])


@pytest.mark.parametrize("mt", [1, 2, 4, 18])
def test_those_channels_carry_a_null_residual_not_a_sentinel(parsed, mt):
    """Null, not 0/0. `residual_Z = residual_A = 0` collides with a real Z=0
    product and is what made (n,tot), (n,el) and (n,f) indistinguishable."""
    rows = [r for r in parsed.rows if r["kind"] == "channel" and r["MT"] == mt]
    assert rows
    for r in rows:
        assert r["residual_Z"] is None, f"MT={mt} residual_Z={r['residual_Z']!r}, expected None"
        assert r["residual_A"] is None


def test_capture_still_names_its_residual(parsed):
    """The control: MT=102 does fix a product, so it must keep naming one.
    A fix that nulled every residual would pass the tests above and be wrong."""
    rows = [r for r in parsed.rows if r["kind"] == "channel" and r["MT"] == 102]
    assert rows
    for r in rows:
        assert (r["residual_Z"], r["residual_A"]) == (13, 28)  # Al-27 + n


def test_kind_tells_channels_and_production_sums_apart(parsed):
    """`kind` is the marker that stops a union double-counting: a production row
    is our sum over every channel reaching a residual, a channel row is one MT.
    Adding them together is wrong, so the distinction has to be queryable."""
    assert {r["kind"] for r in parsed.rows} == {"channel", "production"}
    for r in parsed.rows:
        if r["kind"] == "channel":
            assert r["MT"] is not None, "channel row without an MT"
        else:
            assert r["MT"] is None, f"production row carrying MT={r['MT']}"


def test_production_rows_are_unchanged_by_the_channel_addition(parsed):
    """#347 adds rows; it must not alter the ones already shipped. Al-26 is
    still the MF=3 MT=16 sum and the MF=10 g/m split, exactly as in #340."""
    assert [xs for _e, xs in _rows(parsed, 13, 26, "")] == pytest.approx([80.0, 140.0, 110.0])
    assert [xs for _e, xs in _rows(parsed, 13, 26, "g")] == pytest.approx([60.0, 100.0, 80.0])
    assert [xs for _e, xs in _rows(parsed, 13, 26, "m")] == pytest.approx([20.0, 40.0, 30.0])


def test_channel_and_production_rows_coexist_for_one_residual(parsed):
    """Al-26 is reachable as both: MT=16's own channel row and the production
    sum over every MT reaching (13,26). Same numbers here because MT=16 is the
    only contributor — but they are different claims and different rows."""
    assert _channel(parsed, 16)
    assert _rows(parsed, 13, 26, "")
    assert _channel(parsed, 16) == _rows(parsed, 13, 26, "")


def test_counters_report_the_channel_rows(parsed):
    assert parsed.channel_rows == len([r for r in parsed.rows if r["kind"] == "channel"])
    assert parsed.null_residual_rows == len(TOTAL) + len(ELASTIC) + len(INELASTIC) + len(FISSION)
    assert parsed.null_residual_rows > 0


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
    "kind": "production",
    "MT": None,
    "residual_Z": 12,
    "residual_A": 27,
    "state": "",
    "energy_MeV": 14.0,
    "xs_mb": 80.0,
}
_MF10_ROW = {**_MF3_ROW, "residual_Z": 13, "residual_A": 26, "state": "m", "xs_mb": 40.0}
# A null-residual channel row: MT=18 fission names no single product (#347).
_CHANNEL_ROW = {**_MF3_ROW, "kind": "channel", "MT": 18, "residual_Z": None, "residual_A": None}


def test_guard_fires_when_mf10_sections_yield_no_rows(monkeypatch, tmp_path):
    """The #340 signature: MF=3 still produces plenty of rows, so the #334
    empty-ingest guard sees a healthy run. This one must not."""
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW],
                mf3_sections=12,
                mf3_usable_sections=12,
                mf3_residual_sections=12,
                mf3_rows=1,
                mf10_sections=4,
                # The section names a product and still emitted nothing — that
                # is the #340 signature. A section naming none is a different
                # thing and is covered by the IZAP=-1 tests below.
                mf10_product_sections=4,
                mf10_rows=0,
            )
        },
    )
    with pytest.raises(RuntimeError, match="MF=10"):
        m.fetch_library("irdff-2", "n", tmp_path, session=None)
    assert not (tmp_path / "irdff-2" / "manifest.json").exists()


def test_guard_fires_when_mf3_sections_yield_no_rows(monkeypatch, tmp_path):
    """Symmetric to the MF=10 guard, and only reachable since #340: MF=10 rows
    can now keep the element count healthy while every production sum
    disappears, which the empty-ingest guard would read as a successful run.

    `channel_rows` is deliberately non-zero. Without it the *channel* guard
    fires first and this passes without ever reaching the one it names — which
    it did, silently, until a mutation run showed that disarming the production
    guard broke no test.
    """
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF10_ROW, _CHANNEL_ROW],
                mf3_sections=12,
                mf3_usable_sections=12,
                mf3_residual_sections=12,
                mf3_rows=0,
                channel_rows=1,
                mf10_sections=4,
                mf10_product_sections=4,
                mf10_rows=1,
            )
        },
    )
    with pytest.raises(RuntimeError, match="name a residual"):
        m.fetch_library("irdff-2", "n", tmp_path, session=None)
    assert not (tmp_path / "irdff-2" / "manifest.json").exists()


def test_guard_is_quiet_when_the_library_genuinely_has_no_mf10(monkeypatch, tmp_path):
    """CENDL-3.2 and BROND-3.1 ship no MF=10 at all. Zero isomeric rows is the
    correct outcome there and must not raise."""
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW, _CHANNEL_ROW],
                mf3_sections=12,
                mf3_usable_sections=12,
                mf3_residual_sections=12,
                mf3_rows=1,
                mf10_sections=0,
                mf10_product_sections=0,
                mf10_rows=0,
                channel_rows=1,
                null_residual_rows=1,
            )
        },
    )
    m.fetch_library("cendl-3.2", "n", tmp_path, session=None)
    manifest = json.loads((tmp_path / "cendl-3.2" / "manifest.json").read_text())
    assert manifest["mf10_sections"] == 0
    assert manifest["mf3_rows"] == 1
    assert manifest["states"] == {"": 2}  # the production row and the channel row
    assert manifest["null_residual_rows"] == 1


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


# ---------------------------------------------------------------------------
# #359 — the ingest writes CANONICAL_XS_SCHEMA, not the legacy 6 columns
# ---------------------------------------------------------------------------


def test_written_file_is_exactly_the_canonical_schema(monkeypatch, tmp_path):
    """A fresh ingest must be canonical on the way out.

    This script wrote the 6-column legacy form and relied on
    `migrate_xs_schema.py` being run afterwards, so the documented maintenance
    operation — re-ingest a library — silently dropped twelve of eighteen
    columns and put identity back in the file path (#359). Nothing chained the
    two steps and no test noticed.
    """
    import polars as pl

    from nucl_parquet._schemas import CANONICAL_XS_SCHEMA

    rows = _mod().parse_endf_file(synthetic_material(), 13, 27, "n")
    m = _stub_library(monkeypatch, {"n_013-Al-27_1325.zip": rows})
    m.fetch_library("irdff-2", "n", tmp_path, session=None)

    df = pl.read_parquet(tmp_path / "irdff-2" / "xs" / "n_Al.parquet")
    assert df.columns == list(CANONICAL_XS_SCHEMA), "fresh ingest is not canonical"
    for col, dtype in CANONICAL_XS_SCHEMA.items():
        assert df.schema[col] == getattr(pl, dtype), f"{col}: {df.schema[col]} != {dtype}"


def test_identity_lives_in_the_row_not_the_path(monkeypatch, tmp_path):
    """Principle 5. `library`, `projectile`, `proj_Z/A` and `target_Z` used to be
    recoverable only by regexing `data/<library>/xs/<proj>_<El>.parquet`."""
    import polars as pl

    rows = _mod().parse_endf_file(synthetic_material(), 13, 27, "n")
    m = _stub_library(monkeypatch, {"n_013-Al-27_1325.zip": rows})
    m.fetch_library("irdff-2", "n", tmp_path, session=None)

    df = pl.read_parquet(tmp_path / "irdff-2" / "xs" / "n_Al.parquet")
    assert df["library"].unique().to_list() == ["irdff-2"]
    assert df["projectile"].unique().to_list() == ["n"]
    assert df["proj_Z"].unique().to_list() == [0]
    assert df["proj_A"].unique().to_list() == [1]
    assert df["target_Z"].unique().to_list() == [13]
    assert df["MT"].null_count() < df.height, "MT is all-null — the #347 state"


def test_migration_finds_nothing_to_do_on_a_fresh_ingest(monkeypatch, tmp_path):
    """A migration that still has work after its cause is fixed is a signal.

    `migrate_file` reports `already-canonical` for any file carrying `library`,
    so this asserts the ingest and the migration agree on the target shape.
    """
    import migrate_xs_schema

    rows = _mod().parse_endf_file(synthetic_material(), 13, 27, "n")
    m = _stub_library(monkeypatch, {"n_013-Al-27_1325.zip": rows})
    m.fetch_library("irdff-2", "n", tmp_path, session=None)

    path = tmp_path / "irdff-2" / "xs" / "n_Al.parquet"
    n, status = migrate_xs_schema.migrate_file(path, "irdff-2", "production", dry_run=True)
    assert status == "already-canonical", f"migration still wants to rewrite a fresh ingest: {status}"
    assert n > 0


def test_guard_fires_when_no_channel_rows_are_emitted(monkeypatch, tmp_path):
    """#347's signature: production rows keep flowing from the transmutation
    channels, so neither the empty-ingest guard nor #340's MF=10 guard sees that
    total/elastic/inelastic/fission have gone missing again."""
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW],
                mf3_sections=12,
                mf3_usable_sections=12,
                mf3_residual_sections=12,
                mf3_rows=1,
                channel_rows=0,
                null_residual_rows=0,
            )
        },
    )
    with pytest.raises(RuntimeError, match="channel"):
        m.fetch_library("irdff-2", "n", tmp_path, session=None)


# ---------------------------------------------------------------------------
# The endf_mt reference view — how a consumer avoids double-counting
# ---------------------------------------------------------------------------


def test_endf_mt_marks_the_redundant_totals():
    from nucl_parquet.endf_mt import REDUNDANT_MT, mt_name

    assert 2 in REDUNDANT_MT[1] and 3 in REDUNDANT_MT[1]  # total = elastic + nonelastic
    assert REDUNDANT_MT[18] == (19, 20, 21, 38)
    assert mt_name(18) == "(z,fission)"
    assert mt_name(601) == "(z,p) level 1"
    assert mt_name(649) == "(z,p) continuum"


def test_exclusive_mts_drops_only_actual_double_counts():
    """MT=103 is redundant *only* when its MT=600-649 partials also ship.
    An evaluation carrying MT=103 alone must keep it — that was #326's bug from
    the other side, where dropping a summed MT lost the whole channel."""
    from nucl_parquet.endf_mt import exclusive_mts

    assert exclusive_mts({103}) == {103}
    assert exclusive_mts({103, 600, 601}) == {600, 601}
    assert exclusive_mts({1, 2, 3}) == {2, 3}
    assert 205 not in exclusive_mts({2, 205}), "particle production is not a reaction channel"


def test_endf_mt_view_is_queryable_from_a_plain_checkout():
    """Reference data as a view, not a column repeated on 17M rows, and not a
    parquet that would need a data release to correct a reaction name."""
    import nucl_parquet.loader as loader

    db = loader.connect()
    row = db.sql("SELECT name, redundant, sums_over FROM endf_mt WHERE MT = 1").fetchone()
    assert row[0] == "(n,total)"
    assert row[1] is True
    assert sorted(row[2]) == [2, 3]
    n = db.sql("SELECT count(*) FROM endf_mt WHERE redundant").fetchone()[0]
    assert n >= 10


# ---------------------------------------------------------------------------
# End to end against the real mirror. Network-marked, so CI skips it; run with
#   nix develop -c uv run pytest tests/test_fetch_endf_libs.py -m network
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_real_u235_evaluation_reproduces_the_thermal_anchors(tmp_path):
    """Ingest IRDFF-II's U-235 evaluation and check the two textbook numbers.

    This is the anchor `tests/test_data_release.py` cannot yet make of the
    evaluated libraries, because those are only rebuilt after this lands. Run
    against the builder instead, it is provable today: IRDFF-II ships U-235 as
    pointwise data from 1e-11 MeV up, so thermal is in MF=3 and needs no
    resonance reconstruction.

    ENDF/B-VIII.1 and JEFF-4.0 would *not* work here — their MF=3 MT=18 is
    identically zero below 2250 eV because the whole thermal cross-section lives
    in MF=2 resonance parameters, which this script does not reconstruct.
    """
    import numpy as np
    import requests

    m = _mod()
    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (test)"
    parsed = m.download_and_parse(m.LIBRARIES["irdff-2"], "n", "n_92-U-235_9228.zip", session)
    assert parsed.rows, "download or parse failed"

    def sigma_at(mt, energy_MeV):
        pts = sorted(
            (r["energy_MeV"], r["xs_mb"])
            for r in parsed.rows
            if r["kind"] == "channel" and r["MT"] == mt and r["xs_mb"] > 0
        )
        assert pts, f"MT={mt} produced no channel rows"
        e = np.array([p[0] for p in pts])
        s = np.array([p[1] for p in pts])
        return float(np.exp(np.interp(np.log(energy_MeV), np.log(e), np.log(s))))

    fission_thermal = sigma_at(18, 2.53e-8)
    total_thermal = sigma_at(1, 2.53e-8)
    fission_14mev = sigma_at(18, 14.0)

    assert 555_000 < fission_thermal < 615_000, f"U-235(n,f) thermal = {fission_thermal:,.0f} mb, expected ~585,000"
    assert 665_000 < total_thermal < 735_000, f"U-235 total thermal = {total_thermal:,.0f} mb, expected ~700,000"
    assert 1_900 < fission_14mev < 2_300, f"U-235(n,f) at 14 MeV = {fission_14mev:,.0f} mb, expected ~2,079"

    # And the whole point: these rows name no residual.
    for mt in (1, 18):
        assert all(r["residual_Z"] is None for r in parsed.rows if r["kind"] == "channel" and r["MT"] == mt)


# ---------------------------------------------------------------------------
# Guard denominators — sections that *could* have produced rows (#372 follow-up)
# ---------------------------------------------------------------------------
#
# Two guards fired on charged-particle sublibraries that were being read
# perfectly. Both counted raw sections, and a section can legitimately have
# nothing to emit:
#
#   jeff-4.0/p   36 MF=10 sections, every one MT=18 with IZAP=-1
#                ("fission products, unspecified" — names no product)
#   jendl-5/d    18 MF=3 sections, every one MT=2 or MT=5
#                (neither names a residual, so no production row is possible)
#
# The charged-particle sublibraries have never been re-ingested since #334, so
# nothing had ever exercised these guards against them. These fixtures are those
# two shapes, so the denominators cannot regress.

# jeff-4.0/p's Hf-180: MF=3 MT=2 negative throughout (the nuclear-interference
# term against a divergent Rutherford cross-section — the elastic distribution
# lives in MF=6 with LAW=5), MT=5 positive, and one MF=10 MT=18 with IZAP=-1.
CP_ELASTIC_NEGATIVE = [(2.5e5, -3e-06), (4.0e6, -3e-06), (5.0e6, -9.5e-05)]
CP_ANYTHING = [(2.5e5, 0.5), (4.0e6, 1.2), (5.0e6, 1.859)]
CP_FISSION = [(2.5e5, 0.01), (4.0e6, 0.02), (5.0e6, 0.03)]


def charged_particle_material() -> str:
    """The jeff-4.0/p shape: MF=10 that names no product, MF=3 MT=2 all negative."""
    za = 72180.0
    return "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 2, CP_ELASTIC_NEGATIVE),
            mf3_section(za, 5, CP_ANYTHING),
            mf10_section(za, 18, [(-1, 0, CP_FISSION)]),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )


def deuteron_material() -> str:
    """The jendl-5/d shape: only MT=2 and MT=5, neither naming a residual."""
    za = 3006.0
    return "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 2, [(2.0e5, -0.004948), (4.0e5, -0.013521), (6.0e7, 0.030008)]),
            mf3_section(za, 5, [(2.0e5, 0.0301), (4.0e5, 0.5), (6.0e7, 0.9789)]),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )


def test_mf10_sections_naming_no_product_are_counted_separately():
    """IZAP=-1 is "fission products, unspecified". A section of nothing but those
    has nothing to emit, so it must not sit in the guard's denominator."""
    parsed = _mod().parse_endf_file(charged_particle_material(), 72, 180, "p")
    assert parsed.mf10_sections == 1, "the section is still counted and reported"
    assert parsed.mf10_product_sections == 0, "IZAP=-1 names no product"
    assert parsed.mf10_rows == 0, "and correctly produces no rows"


def test_a_library_whose_only_mf10_is_izap_minus_one_does_not_trip_the_guard(monkeypatch, tmp_path):
    """The jeff-4.0/p false positive, end to end. 36 sections, all IZAP=-1."""
    m = _stub_library(
        monkeypatch,
        {"p_072-Hf-180_7243.zip": _mod().parse_endf_file(charged_particle_material(), 72, 180, "p")},
    )
    m.fetch_library("jeff-4.0", "p", tmp_path, session=None)  # must not raise

    manifest = json.loads((tmp_path / "jeff-4.0" / "manifest.json").read_text())
    assert manifest["mf10_sections"] == 1
    assert manifest["mf10_product_sections"] == 0
    assert manifest["mf10_rows"] == 0
    assert manifest["channel_rows"] > 0, "the MF=3 side still produced rows"


def test_mf3_sections_naming_no_residual_are_counted_separately():
    """MT=2 and MT=5 name no residual, so no production row is possible and
    zero of them is the right answer — the jendl-5/d false positive."""
    parsed = _mod().parse_endf_file(deuteron_material(), 3, 6, "d")
    assert parsed.mf3_sections == 2
    assert parsed.mf3_usable_sections == 2, "both have positive points"
    assert parsed.mf3_residual_sections == 0, "neither MT names a residual"
    assert parsed.mf3_rows == 0, "so there are no production rows"
    assert parsed.channel_rows > 0, "but the channel rows are there"
    assert parsed.null_residual_rows == parsed.channel_rows


def test_a_library_with_no_residual_naming_mts_does_not_trip_the_guard(monkeypatch, tmp_path):
    """The jendl-5/d false positive, end to end."""
    m = _stub_library(
        monkeypatch,
        {"d_003-Li-6_0325.zip": _mod().parse_endf_file(deuteron_material(), 3, 6, "d")},
    )
    m.fetch_library("jendl-5", "d", tmp_path, session=None)  # must not raise

    manifest = json.loads((tmp_path / "jendl-5" / "manifest.json").read_text())
    assert manifest["mf3_sections"] == 2
    assert manifest["mf3_residual_sections"] == 0
    assert manifest["mf3_rows"] == 0
    assert manifest["channel_rows"] > 0


def test_an_all_negative_mf3_section_is_not_counted_as_usable():
    """Charged-particle MF=3 MT=2 is negative by construction. Counting it as a
    section that should have produced rows would fire the channel guard on a
    library being read correctly."""
    parsed = _mod().parse_endf_file(charged_particle_material(), 72, 180, "p")
    assert parsed.mf3_sections == 2, "MT=2 and MT=5 are both present"
    assert parsed.mf3_usable_sections == 1, "only MT=5 has a positive point"
    assert not _channel(parsed, 2), "the all-negative section emits nothing"
    assert _channel(parsed, 5), "the positive one still does"
