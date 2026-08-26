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

Builds its own ENDF-6 material in-process, so it needs no data tree and no
network — the point is that a version bump of the `endf` package fails *here*
rather than silently emptying every ground/metastable split again.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import pytest

from nucl_parquet.builder_stamp import RETIRED_MANIFEST_KEYS

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from nucl_parquet.state_vocabulary import (  # noqa: E402
    GROUND,
    STATES,
    SUM,
    TARGET_STATES,
    parse_x4_state,
)


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


def mf9_section(za: float, mt: int, levels: list[tuple[int, int, list]]) -> str:
    """MF=9 isomeric *yields*. Same record layout as MF=10 — the `endf` package
    parses both with `parse_mf9_mf10` — but each level's TAB1 is Y(E), a
    fraction, rather than sigma(E). `levels` is [(IZAP, LFS, [(E_eV, Y), ...])].
    """
    head = (
        _endf_float(za)
        + _endf_float(26.75)
        + _endf_int(0)  # LIS
        + _endf_int(0)
        + _endf_int(len(levels))  # NS
        + _endf_int(0)
    )
    out = _line(head, 9, mt, 1)
    seq = 2
    for izap, lfs, xy in levels:
        body, seq = _tab1([_endf_float(0.0), _endf_float(0.0), _endf_int(izap), _endf_int(lfs)], xy, 9, mt, seq)
        out += body
    return out + _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 9, 0, 99999)


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
    """MF=3 spells its state 'sum' and MF=10 spells it 'g'/'m', so the 140 mb total
    and its 100 mb ground-state part are different rows, not a silent overwrite
    of one by the other."""
    total = _rows(parsed, 13, 26, SUM)
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
    expected = {
        "target_A",
        # The target's own isomeric state (#353). Present on every row because
        # `fetch_library` declares it in the write schema; a row missing it would
        # make polars infer a Null column for a shard of ground-state targets.
        "target_state",
        "kind",
        "MT",
        "residual_Z",
        "residual_A",
        "state",
        "energy_MeV",
        "xs_mb",
        # ENDF's interpolation law for the interval starting at this row (#338).
        "interp_law",
    }
    assert parsed.rows
    for row in parsed.rows:
        assert set(row) == expected
        assert row["target_A"] == 27
        assert row["state"] is None or row["state"] in STATES
        assert row["target_state"] in TARGET_STATES
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
    assert [xs for _e, xs in _rows(parsed, 13, 26, SUM)] == pytest.approx([80.0, 140.0, 110.0])
    assert [xs for _e, xs in _rows(parsed, 13, 26, "g")] == pytest.approx([60.0, 100.0, 80.0])
    assert [xs for _e, xs in _rows(parsed, 13, 26, "m")] == pytest.approx([20.0, 40.0, 30.0])


def test_channel_and_production_rows_coexist_for_one_residual(parsed):
    """Al-26 is reachable as both: MT=16's own channel row and the production
    sum over every MT reaching (13,26). Same numbers here because MT=16 is the
    only contributor — but they are different claims and different rows."""
    assert _channel(parsed, 16)
    assert _rows(parsed, 13, 26, SUM)
    assert _channel(parsed, 16) == _rows(parsed, 13, 26, SUM)


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
    lin = lambda n: np.full(n, 2, dtype=np.int64)  # noqa: E731 — all lin-lin
    a = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]), lin(3))
    b = (np.array([2.0, 4.0]), np.array([5.0, 15.0]), lin(2))
    e, xs, laws = m.sum_on_union_grid([a, b])
    assert list(e) == [1.0, 2.0, 3.0, 4.0]
    # b contributes 0 below its threshold and above its max; a likewise.
    assert list(xs) == pytest.approx([10.0, 25.0, 40.0, 15.0])
    # Every contribution was lin-lin, so resampling was exact and the sum can
    # honestly claim law 2 (#338).
    assert list(laws) == [2, 2, 2, 2]


def test_sum_on_union_grid_passes_a_lone_contribution_through():
    import numpy as np

    m = _mod()
    e_in, xs_in = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    laws_in = np.array([5, 5], dtype=np.int64)
    e, xs, laws = m.sum_on_union_grid([(e_in, xs_in, laws_in)])
    assert list(e) == [1.0, 2.0]
    assert list(xs) == [3.0, 4.0]
    # Nothing was resampled, so the evaluator's own law survives untouched —
    # even a non-default one.
    assert list(laws) == [5, 5]


def test_a_sum_over_disagreeing_laws_reports_no_law():
    """The #338 branch that must not quietly assert lin-lin.

    `sum_on_union_grid` resamples with `np.interp`. When a contribution says the
    curve is log-log, that resampling *approximated* it, so no single law
    describes the result and NULL is the only honest answer. Emitting 2 here
    would be the ingest inventing an evaluator's statement — the exact defect
    #338 exists to remove, reintroduced inside its own fix.
    """
    import numpy as np

    m = _mod()
    a = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]), np.array([2, 2, 2], dtype=np.int64))
    b = (np.array([2.0, 4.0]), np.array([5.0, 15.0]), np.array([5, 5], dtype=np.int64))
    e, xs, laws = m.sum_on_union_grid([a, b])
    assert list(e) == [1.0, 2.0, 3.0, 4.0], "the sum itself must be unaffected"
    assert laws is None, "a sum over mixed laws has no law, and must say so"


def test_a_lone_contribution_keeps_a_law_that_varies_along_it():
    """A single section can change law partway — TENDL-2023 p+Li-6 MT=750 is
    law 6 for 23 points then law 5 for 111. Passing it through must preserve
    that, not collapse it to whichever law came first."""
    import numpy as np

    m = _mod()
    e_in = np.array([1.0, 2.0, 3.0, 4.0])
    laws_in = np.array([6, 6, 5, 5], dtype=np.int64)
    _e, _xs, laws = m.sum_on_union_grid([(e_in, np.array([1.0, 2.0, 3.0, 4.0]), laws_in)])
    assert list(laws) == [6, 6, 5, 5]


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
    "state": SUM,
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
    assert manifest["ingest"]["n"]["mf10_sections"] == 0
    assert manifest["ingest"]["n"]["mf3_rows"] == 1
    assert manifest["ingest"]["n"]["states"] == {SUM: 2}  # the production row and the channel row
    assert manifest["ingest"]["n"]["null_residual_rows"] == 1


def test_a_tape_that_yields_nothing_is_named_in_the_log(monkeypatch, tmp_path, caplog):
    """#335 / #336: dropping a nuclide is allowed; doing it silently is not.

    None of the guards above can see this. They ask whether a *sublibrary* went
    dark, and a run where 2 of 2298 tapes convert to nothing is healthy by every
    one of those measures — the element count is fine, MF=3 and MF=10 are both
    producing, and the exit code is 0. That is exactly the state in which
    tendl-2023-iso lost p+Be-9 and jendl-5 lost n+Ho-165, and the only thing that
    would have surfaced either is being told *which* tapes produced nothing.

    So the assertion is on the tape's name, not on a count: a count says
    something went missing, a name says what.
    """
    healthy = _mod().ParsedFile(
        rows=[_MF3_ROW, _CHANNEL_ROW],
        mf3_sections=12,
        mf3_usable_sections=12,
        mf3_residual_sections=12,
        mf3_rows=1,
        channel_rows=1,
        null_residual_rows=1,
    )
    # Parsed fine, named nothing: an evaluation with no production data. Its
    # counters stay zero so no aggregate guard fires on it.
    silent = _mod().ParsedFile(rows=[])
    m = _stub_library(
        monkeypatch,
        {"n_013-Al-27_1325.zip": healthy, "n_004-Be-9_0425.zip": silent},
    )
    with caplog.at_level(logging.WARNING):
        m.fetch_library("cendl-3.2", "n", tmp_path, session=None)

    report = "\n".join(r.getMessage() for r in caplog.records)
    assert "n_004-Be-9_0425.zip" in report, "the dropped tape was not named — this is the #335 silence"
    assert "1/2" in report, "the report must say how many tapes of how many"
    assert "n_013-Al-27_1325.zip" not in report, "a tape that produced rows must not be reported as absent"
    # And the run still succeeds: a legitimate drop is not a failure.
    assert (tmp_path / "cendl-3.2" / "manifest.json").exists()


def test_no_report_when_every_tape_produced_rows(monkeypatch, tmp_path, caplog):
    """The other half. A warning that fires on a clean run is noise, and noise
    is how the next real one gets scrolled past."""
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW, _CHANNEL_ROW],
                mf3_sections=12,
                mf3_usable_sections=12,
                mf3_residual_sections=12,
                mf3_rows=1,
                channel_rows=1,
                null_residual_rows=1,
            )
        },
    )
    with caplog.at_level(logging.WARNING):
        m.fetch_library("cendl-3.2", "n", tmp_path, session=None)
    assert "ABSENT from the output" not in caplog.text


def test_a_long_report_states_what_it_truncated(monkeypatch, tmp_path, caplog):
    """A sublibrary is up to ~2300 tapes, so the list has to stop somewhere.

    It stops *out loud*. The count is always exact and the remainder is spelled
    out, because a cap nobody is told about reads as "that was all of them" —
    the same shape of silence as the skip this report exists to expose.
    """
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW, _CHANNEL_ROW],
                mf3_sections=12,
                mf3_usable_sections=12,
                mf3_residual_sections=12,
                mf3_rows=1,
                channel_rows=1,
                null_residual_rows=1,
            ),
            **{f"n_0{i:02d}-X-{i}_000{i}.zip": _mod().ParsedFile(rows=[]) for i in range(1, 6)},
        },
    )
    monkeypatch.setattr(m, "_MAX_LISTED_TAPES", 2)
    with caplog.at_level(logging.WARNING):
        m.fetch_library("cendl-3.2", "n", tmp_path, session=None)

    report = "\n".join(r.getMessage() for r in caplog.records)
    assert "5/6 tapes" in report, f"the count must be exact even when the list is not: {report}"
    assert "… and 3 more" in report, f"the truncation must be stated: {report}"


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
    assert manifest["ingest"]["n"]["mf10_sections"] == 4
    assert manifest["ingest"]["n"]["mf10_rows"] == rows.mf10_rows > 0
    assert manifest["ingest"]["n"]["states"]["m"] > 0
    assert manifest["ingest"]["n"]["states"]["g"] > 0
    assert sum(manifest["ingest"]["n"]["states"].values()) == manifest["total_rows"]


def test_written_parquet_carries_the_states(monkeypatch, tmp_path):
    import polars as pl

    rows = _mod().parse_endf_file(synthetic_material(), 13, 27, "n")
    m = _stub_library(monkeypatch, {"n_013-Al-27_1325.zip": rows})
    m.fetch_library("irdff-2", "n", tmp_path, session=None)

    df = pl.read_parquet(tmp_path / "irdff-2" / "xs" / "n_Al.parquet")
    assert set(df["state"].unique()) == {SUM, "g", "m", "m2"}
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
# Mixed sign, modelled on ENDF/B-VIII.1 p+Cu-65: 44 points, 25 of them positive,
# which the old `xs > 0` filter kept and wrote as a 172 mb -> 12.6 b curve
# labelled (z,elastic). THE load-bearing fixture — an all-negative section
# vanishes under the old code too, so a test built only on one would pass while
# the defect survived.
CP_ELASTIC_MIXED = [(1.0e6, -0.2574), (5.0e6, 0.172235), (1.5e7, 4.0), (3.0e7, 12.6045)]
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
            # The real d+Li-6 carries MF=6 MT=2 LAW=5 — that is what makes its
            # MF=3 MT=2 the interference term rather than a blemished elastic
            # curve, and it is 78% positive so nothing about its sign says so.
            mf6_law5_section(za, 2),
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
    assert manifest["ingest"]["p"]["mf10_sections"] == 1
    assert manifest["ingest"]["p"]["mf10_product_sections"] == 0
    assert manifest["ingest"]["p"]["mf10_rows"] == 0
    assert manifest["ingest"]["p"]["channel_rows"] > 0, "the MF=3 side still produced rows"


def test_mf3_sections_naming_no_residual_are_counted_separately():
    """MT=2 and MT=5 name no residual, so no production row is possible and
    zero of them is the right answer — the jendl-5/d false positive."""
    parsed = _mod().parse_endf_file(deuteron_material(), 3, 6, "d")
    assert parsed.mf3_sections == 2
    # MT=2 carries MF=6 LAW=5 here, as it really does in jendl-5/d, so it is the
    # interference term and is dropped whole (#377/#394); only MT=5 is usable.
    assert parsed.signed_sections == {2: 1}
    assert parsed.mf3_usable_sections == 1, "only MT=5 survives"
    assert parsed.mf3_residual_sections == 0, "and MT=5 names no residual"
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
    assert manifest["ingest"]["d"]["mf3_sections"] == 2
    assert manifest["ingest"]["d"]["mf3_residual_sections"] == 0
    assert manifest["ingest"]["d"]["mf3_rows"] == 0
    assert manifest["ingest"]["d"]["channel_rows"] > 0


def test_an_all_negative_mf3_section_is_not_counted_as_usable():
    """Charged-particle MF=3 MT=2 is negative by construction. Counting it as a
    section that should have produced rows would fire the channel guard on a
    library being read correctly."""
    parsed = _mod().parse_endf_file(charged_particle_material(), 72, 180, "p")
    assert parsed.mf3_sections == 2, "MT=2 and MT=5 are both present"
    assert parsed.mf3_usable_sections == 1, "only MT=5 has a positive point"
    assert not _channel(parsed, 2), "the all-negative section emits nothing"
    assert _channel(parsed, 5), "the positive one still does"


# ---------------------------------------------------------------------------
# Structurally signed vs incidentally negative MF=3 sections (#377, #394)
# ---------------------------------------------------------------------------
#
# #379 dropped any MF=3 section carrying a negative value. That is the wrong
# predicate: it discarded 69 real cross-sections across the corpus, including
# JEFF-4.0's Au-197 capture, whose MT=102 is 4.6% negative and sound otherwise.
#
# Measured over 620 evaluations, the negative *fraction* of a structurally
# signed section runs 0.0294..1.0000, and of an ordinary section with bad points
# 0.0008..1.0000. The ranges overlap almost entirely, so no threshold on sign can
# separate them and none is used. The signal is structural instead: MF=6 MT=X
# carrying LAW=5 means MF=3 MT=X is the Rutherford interference term.


def _tab2(params: list, mf: int, mt: int, seq: int) -> tuple[str, int]:
    """A TAB2 record: 6 head fields then one NBT/INT pair."""
    out = _line("".join(params) + _endf_int(1) + _endf_int(1), mf, mt, seq)
    seq += 1
    return out + _line(_endf_int(1) + _endf_int(2), mf, mt, seq), seq + 1


def _list_record(params: list, values: list[float], mf: int, mt: int, seq: int) -> tuple[str, int]:
    out = _line("".join(params) + _endf_int(len(values)) + _endf_int(0), mf, mt, seq)
    seq += 1
    for i in range(0, len(values), 6):
        out += _line("".join(_endf_float(v) for v in values[i : i + 6]), mf, mt, seq)
        seq += 1
    return out, seq


def mf6_law5_section(za: float, mt: int) -> str:
    """A minimal MF=6 section whose product carries LAW=5.

    LAW=5 is ENDF's charged-particle elastic law. Its presence at MT=X is the
    structural statement that MF=3 MT=X holds the interference term — the signal
    #394 uses in place of #379's sign test.
    """
    head = _endf_float(za) + _endf_float(26.75) + _endf_int(0) + _endf_int(1) + _endf_int(1) + _endf_int(0)
    out = _line(head, 6, mt, 1)
    body, seq = _tab1(
        [_endf_float(1001.0), _endf_float(1.0), _endf_int(0), _endf_int(5)],
        [(1.0e5, 1.0), (2.0e7, 1.0)],
        6,
        mt,
        2,
    )
    out += body
    body, seq = _tab2([_endf_float(0.5), _endf_float(0.0), _endf_int(1), _endf_int(0)], 6, mt, seq)
    out += body
    body, seq = _list_record([_endf_float(0.0), _endf_float(1.0e5), _endf_int(1), _endf_int(0)], [0.0, 0.0], 6, mt, seq)
    return out + body + _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 6, 0, 99999)


# The p+Cu-65 shape from #377: mixed sign, and structurally signed.
CP_ELASTIC_MIXED = [(1.0e6, -0.2574), (5.0e6, 0.172235), (1.5e7, 4.0), (3.0e7, 12.6045)]
CP_ANYTHING = [(2.5e5, 0.5), (4.0e6, 1.2), (5.0e6, 1.859)]
# The JEFF-4.0 Au-197 shape from #394: isolated negatives in a real capture curve.
AU_CAPTURE_WITH_BLEMISH = [(1.0e5, 0.32), (5.0e5, -0.011), (1.0e6, 0.083), (1.4e7, 0.0012)]


def structurally_signed_material() -> str:
    """MF=3 MT=2 mixed-sign *and* MF=6 MT=2 LAW=5 — the #377 case."""
    za = 29065.0
    return "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 2, CP_ELASTIC_MIXED),
            mf3_section(za, 5, CP_ANYTHING),
            mf6_law5_section(za, 2),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )


def blemished_capture_material() -> str:
    """MF=3 MT=102 with one negative point and no LAW=5 anywhere — the #394 case."""
    za = 79197.0
    return "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 102, AU_CAPTURE_WITH_BLEMISH),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )


def test_law5_marks_the_structurally_signed_section():
    """The signal, read straight off the material."""
    import endf

    m = _mod()
    assert m.charged_particle_elastic_mts(endf.Material(io.StringIO(structurally_signed_material()))) == {2}
    assert m.charged_particle_elastic_mts(endf.Material(io.StringIO(blemished_capture_material()))) == set()


def test_a_law5_marked_section_is_dropped_whole_even_when_mostly_positive():
    """#377 must keep working. This curve is 75% positive and still not a sigma,
    so a sign-based test would keep three of its four points."""
    parsed = _mod().parse_endf_file(structurally_signed_material(), 29, 65, "p")
    assert not _channel(parsed, 2), "the interference term must not reach the data"
    assert parsed.signed_sections == {2: 1}
    positives = {xs * 1e3 for _e, xs in CP_ELASTIC_MIXED if xs > 0}
    assert not (positives & {r["xs_mb"] for r in parsed.rows}), "positive lobe leaked through"
    assert _channel(parsed, 5), "the ordinary section in the same file survives"


def test_a_blemished_curve_is_repaired_not_discarded():
    """The #394 regression, in the shape that caused it.

    JEFF-4.0's Au-197 MT=102 is 4.6% negative and lost all 350 of its rows.
    Keep the section, drop the bad point, and record how many.
    """
    parsed = _mod().parse_endf_file(blemished_capture_material(), 79, 197, "n")
    kept = _channel(parsed, 102)
    assert kept, "a capture curve with one bad point is still a capture curve"
    assert [xs for _e, xs in kept] == pytest.approx([320.0, 83.0, 1.2])
    assert parsed.signed_sections == {}, "not a structural drop"
    assert parsed.negative_points_dropped == {102: 1}, "the repair must be recorded"
    assert _rows(parsed, 79, 198, SUM), "and the production row is back"


def test_a_wholly_negative_section_needs_no_special_case():
    """It has no good points, so it emits nothing anyway — but is counted as
    signed, so 'not a cross-section' stays distinct from 'a curve we repaired'."""
    za = 68156.0
    material = "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 2, [(1.0e6, -0.4756), (2.0e7, -7.074e-07)]),
            mf3_section(za, 102, CAPTURE),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )
    parsed = _mod().parse_endf_file(material, 68, 156, "h")
    assert not _channel(parsed, 2)
    assert parsed.signed_sections == {2: 1}
    assert _channel(parsed, 102), "the sound section is untouched"


@pytest.mark.parametrize("mt", [1, 18, 102, 91, 600, 649])
def test_negatives_outside_mt_2_are_repaired_not_discarded(mt):
    """The 69 sections #379 destroyed spanned MT=1, 3, 18, 91, 102, 600 and 649.
    None carried LAW=5; all are real cross-sections with a defect in them."""
    za = 79197.0
    material = "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, mt, AU_CAPTURE_WITH_BLEMISH),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )
    parsed = _mod().parse_endf_file(material, 79, 197, "n")
    assert _channel(parsed, mt), f"MT={mt} was discarded for a single bad point"
    assert parsed.negative_points_dropped == {mt: 1}
    assert parsed.signed_sections == {}


def test_the_rule_is_not_projectile_conditional():
    """LAW=5 decides, not the sublibrary code — a neutron file carrying it would
    still be structurally signed, and a charged-particle file without it repaired."""
    parsed = _mod().parse_endf_file(structurally_signed_material(), 29, 65, "n")
    assert parsed.signed_sections == {2: 1}
    parsed = _mod().parse_endf_file(blemished_capture_material(), 79, 197, "a")
    assert parsed.signed_sections == {}
    assert parsed.negative_points_dropped == {102: 1}


def test_both_mf3_readers_share_the_structural_rule():
    """`scripts/backfill_xs_nuclides.py` parses MF=3 off a tape too. One spelling
    of "this section is not a sigma", imported rather than re-implemented."""
    import backfill_xs_nuclides as bf

    m = _mod()
    assert bf.charged_particle_elastic_mts is m.charged_particle_elastic_mts
    assert "is_signed_section" not in bf.__dict__, "the backfill must not carry the retired rule"


def test_manifest_separates_structural_drops_from_repairs(monkeypatch, tmp_path):
    """Two different decisions, two different fields. #379 blurred them."""
    m = _stub_library(
        monkeypatch,
        {"p_029-Cu-65_2931.zip": _mod().parse_endf_file(structurally_signed_material(), 29, 65, "p")},
    )
    m.fetch_library("jeff-4.0", "p", tmp_path, session=None)
    record = json.loads((tmp_path / "jeff-4.0" / "manifest.json").read_text())["ingest"]["p"]
    assert record["signed_sections_dropped"] == {"2": 1}
    assert record["negative_points_dropped"] == {}
    assert "not represented" in record["charged_particle_elastic"]


# ---------------------------------------------------------------------------
# The target's own isomeric state (#353)
# ---------------------------------------------------------------------------
#
# `parse_endf_filename` has extracted the target's isomer marker since #334, and
# the caller assigned it to `_isomer` and dropped it. With no `target_state`
# column, `n_035-Br-80` and `n_035-Br-80M` — two nuclides, different half-lives,
# different cross-sections — both landed as `target_A = 80` in one shard:
# 1,490 rows under 757 distinct keys in the shipped tendl-2025/xs/n_Br.parquet.


def mf1_451_section(za: float, lis: int, liso: int) -> str:
    """A minimal MF=1/451 descriptive section carrying LIS and LISO."""
    seq = 1
    out = _line(_endf_float(za) + _endf_float(26.75) + _endf_int(0) * 4, 1, 451, seq)
    seq += 1
    # ELIS, STA, LIS, LISO, -, NFOR
    out += _line(
        _endf_float(0.0) + _endf_float(0.0) + _endf_int(lis) + _endf_int(liso) + _endf_int(0) + _endf_int(6),
        1,
        451,
        seq,
    )
    seq += 1
    # AWI, EMAX, LREL, -, NSUB, NVER
    out += _line(
        _endf_float(1.0) + _endf_float(2.0e7) + _endf_int(0) + _endf_int(0) + _endf_int(10) + _endf_int(8),
        1,
        451,
        seq,
    )
    seq += 1
    # TEMP, -, LDRV, -, NWD, NXC
    out += _line(
        _endf_float(0.0) + _endf_float(0.0) + _endf_int(0) + _endf_int(0) + _endf_int(1) + _endf_int(0), 1, 451, seq
    )
    seq += 1
    out += _line("synthetic material for tests".ljust(66), 1, 451, seq)
    seq += 1
    return out + _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 1, 0, 99999)


def material_with_liso(lis: int, liso: int):
    """A material carrying only MF=1/451, for target-state resolution."""
    import endf

    text = (
        "".ljust(66)
        + "TPID\n"
        + mf1_451_section(13027.0, lis, liso)
        + _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0)
        + f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n"
    )
    return endf.Material(io.StringIO(text))


def test_endf_package_exposes_liso_and_lis_separately():
    """Pin the field the target state is read from, as #340 pinned MF=10's shape.

    ENDF-6 has two nearby fields and they are not interchangeable: `LIS` is the
    target's excited *level* number, `LISO` its *isomeric state* number. The
    values below are the real Br-80m shape — a first isomer sitting at the second
    excited level — so a package version that swapped or dropped one fails here
    rather than silently ranking every metastable target one step too high.
    """
    info = material_with_liso(lis=2, liso=1).section_data[1, 451]
    assert info["LIS"] == 2, "LIS counts excited levels"
    assert info["LISO"] == 1, "LISO counts isomeric states — this is the first isomer"


def test_target_state_comes_from_liso_not_lis():
    """The rank must be LISO's, which is why #353's suggested LIS check is wrong.

    Cross-checking the filename marker against `LIS` would have compared rank 1
    ('M') against 2 and failed on a perfectly correct evaluation — the issue's
    own Br-80m example.
    """
    m = _mod()
    assert m.target_state_from_material(material_with_liso(2, 1), "m", 80, "n_035-Br-80M.zip") == "m"
    assert m.target_state_from_material(material_with_liso(0, 0), "", 80, "n_035-Br-80.zip") == GROUND
    assert m.target_state_from_material(material_with_liso(4, 2), "m2", 108, "n_047-Ag-108M2.zip") == "m2"


def test_an_unmarked_endf_filename_is_a_ground_state_claim():
    """No marker means the ground state — a claim, not an absence.

    ENDF names metastable targets explicitly and states LISO=0 for the rest, so
    absence of a marker carries information here. Contrast EXFOR, where an absent
    suffix means the record did not say and `parse_x4_state` returns None.
    """
    m = _mod()
    assert m.target_state_from_material(material_with_liso(0, 0), "", 27, "n_013-Al-27.zip") == GROUND
    assert parse_x4_state(None) is None


def test_a_natural_element_target_gets_no_state():
    """`target_A = 0` is an isotopic mixture, so NULL — checked before LISO."""
    m = _mod()
    assert m.target_state_from_material(material_with_liso(0, 0), "", 0, "n_017-Cl-0.zip") is None


def test_filename_and_evaluation_disagreeing_is_a_hard_failure():
    """A ground-state marker on an isomeric evaluation must not be resolved silently.

    Preferring either source writes one nuclide's cross-sections under the
    other's key — the defect this column exists to end, re-created by the fix for
    it. #353 asks for a hard failure and this is it.
    """
    m = _mod()
    with pytest.raises(m.TargetStateConflict, match="LISO=1"):
        m.target_state_from_material(material_with_liso(2, 1), "", 80, "n_035-Br-80.zip")
    with pytest.raises(m.TargetStateConflict, match="LISO=0"):
        m.target_state_from_material(material_with_liso(0, 0), "zz", 80, "n_035-Br-80ZZ.zip")


def test_an_unspellable_marker_without_liso_raises_rather_than_guessing():
    """No MF=1/451 and a marker we cannot spell: refuse.

    Filing it under the ground state is what discarding the marker did for every
    file, which is the bug. 'n' appears in some mirror listings and this
    repository does not know what it means — so it does not pretend to.
    """
    m = _mod()
    material = _endf_material_without_mf1()
    with pytest.raises(m.TargetStateConflict, match="not one this repository can spell"):
        m.target_state_from_material(material, "n", 80, "n_035-Br-80N.zip")
    # A marker it *can* spell still resolves from the filename alone.
    assert m.target_state_from_material(material, "m", 80, "n_035-Br-80M.zip") == "m"
    assert m.target_state_from_material(material, "", 27, "n_013-Al-27.zip") == GROUND


def _endf_material_without_mf1():
    import endf

    return endf.Material(io.StringIO(synthetic_material()))


def test_every_row_of_a_file_carries_the_targets_state(parsed):
    """The stamp is a property of the file, so it must be on every row and equal.

    Asserts a positive value rather than "no row is missing it": the synthetic
    material has no MF=1/451 and no marker, so every row must say `'g'`.
    """
    assert parsed.rows, "the synthetic material must produce rows"
    states = {row["target_state"] for row in parsed.rows}
    assert states == {GROUND}, f"expected every row stamped 'g', got {sorted(states)}"


def test_a_metastable_target_stamps_every_row_m():
    """The #353 fix, end to end: the marker reaches the rows instead of `_isomer`."""
    rows = _mod().parse_endf_file(synthetic_material(), 13, 27, "n", "m").rows
    assert rows
    assert {row["target_state"] for row in rows} == {"m"}


def test_target_state_values_are_in_the_vocabulary():
    """Whatever the ingest stamps must be spellable by state_vocabulary (#380)."""
    for marker in ("", "m", "m2"):
        value = _mod().parse_endf_file(synthetic_material(), 13, 27, "n", marker).rows[0]["target_state"]
        assert value in TARGET_STATES, f"marker {marker!r} produced {value!r}, outside {sorted(TARGET_STATES)}"


def test_a_second_sublibrary_adds_to_the_manifest_instead_of_replacing_it(monkeypatch, tmp_path):
    """Two `--sublibrary` runs must leave a manifest describing both.

    The manifest used to be written flat and overwritten wholesale, so a library
    built `--all-sublibs` kept only whichever projectile went last: `tendl-2025`
    ships a/d/h/n/p/t and its manifest said `"sublibrary": "a"` with a
    `source_files` count for that one run, sitting next to `files` and
    `total_rows` that `build_manifests.py` had correctly regenerated from all six
    (#369). The ten MF/state counters added in #354 and #376 had the same shape
    and had simply not reached committed data yet.

    Runs the real `fetch_library` twice against stubbed parse results — the only
    way to exercise the merge without a multi-GB download.
    """
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW, _CHANNEL_ROW],
                mf3_sections=3,
                mf3_usable_sections=3,
                mf3_residual_sections=3,
                mf3_rows=1,
                channel_rows=1,
            )
        },
    )
    m.fetch_library("jendl-5", "n", tmp_path, session=None)
    manifest_path = tmp_path / "jendl-5" / "manifest.json"
    after_first = json.loads(manifest_path.read_text())
    assert set(after_first["ingest"]) == {"n"}
    assert after_first["ingest"]["n"]["mf3_rows"] == 1

    m = _stub_library(
        monkeypatch,
        {
            "p_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW, _MF10_ROW, _CHANNEL_ROW],
                mf3_sections=7,
                mf3_usable_sections=7,
                mf3_residual_sections=7,
                mf3_rows=1,
                mf10_sections=2,
                mf10_product_sections=2,
                mf10_rows=1,
                channel_rows=1,
            )
        },
    )
    m.fetch_library("jendl-5", "p", tmp_path, session=None)
    after_second = json.loads(manifest_path.read_text())

    # The neutron run survives the proton run, and each keeps its own counts.
    assert set(after_second["ingest"]) == {"n", "p"}
    assert after_second["ingest"]["n"] == after_first["ingest"]["n"]
    assert after_second["ingest"]["p"]["mf3_sections"] == 7
    assert after_second["ingest"]["p"]["mf10_rows"] == 1
    # And no flat spelling of any of it came back.
    assert not set(after_second) & RETIRED_MANIFEST_KEYS, sorted(set(after_second) & RETIRED_MANIFEST_KEYS)


def test_the_ingest_record_survives_a_manifest_regeneration(monkeypatch, tmp_path):
    """`build_manifests.py` must preserve what only a real ingest can know.

    It derives `files`/`total_rows`/`projectiles`/`elements` from disk and knows
    nothing about MF sections or source-file counts, so losing `ingest` on
    regeneration would silently discard the diagnostics #354 and #376 added to
    detect dropped data.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from build_manifests import build_manifest

    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW, _CHANNEL_ROW],
                mf3_sections=3,
                mf3_usable_sections=3,
                mf3_residual_sections=3,
                mf3_rows=1,
                channel_rows=1,
            )
        },
    )
    m.fetch_library("jendl-5", "n", tmp_path, session=None)
    manifest_path = tmp_path / "jendl-5" / "manifest.json"
    existing = json.loads(manifest_path.read_text())

    # The exact merge build_manifests.py performs.
    fresh = build_manifest("jendl-5", tmp_path / "jendl-5" / "xs")
    merged = {k: v for k, v in existing.items() if k not in RETIRED_MANIFEST_KEYS}
    merged.update(fresh)

    assert merged["ingest"] == existing["ingest"]
    assert merged["builder"] == existing["builder"]
    assert merged["projectiles"] == ["n"]
    assert merged["files"] == 1


# MF=9 — isomeric yields (#352)
# ---------------------------------------------------------------------------
#
# Same shape as MF=10, different currency: each level carries Y(E), a fraction of
# the MT reaction, so the cross-section is sigma_MF3(MT, E) * Y(E). Getting the
# pairing wrong fabricates cross-sections, which is worse than the absence it
# replaces — so the fixtures pin the pairing, not just the parse.

# MT=102 capture on Al-27 -> Al-28, split 75/25 between ground and the isomer.
CAPTURE_Y_G = [(1.0e5, 0.75), (1.0e6, 0.75), (1.4e7, 0.75)]
CAPTURE_Y_M = [(1.0e5, 0.25), (1.0e6, 0.25), (1.4e7, 0.25)]


def yield_material() -> str:
    """MF=3 MT=102 with an MF=9 MT=102 splitting it — and no MF=10 at all,
    which is ENDF/B-VIII.1 Am-241's situation."""
    za = 13027.0
    return "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 102, CAPTURE),
            mf9_section(za, 102, [(13028, 0, CAPTURE_Y_G), (13028, 1, CAPTURE_Y_M)]),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )


@pytest.fixture(scope="module")
def yields():
    return _mod().parse_endf_file(yield_material(), 13, 27, "n")


def test_endf_package_mf9_shape_is_pinned():
    """MF=9 levels carry 'Y', where MF=10 carries 'sigma'.

    `endf.mf9.parse_mf9_mf10` serves both files and switches on MF for that one
    key. Pin it, so a version bump that renames it fails here rather than
    silently reinstating the absence #352 describes.
    """
    import endf

    material = endf.Material(io.StringIO(yield_material()))
    section = material.section_data[9, 102]
    assert set(section) == {"ZA", "AWR", "LIS", "NS", "levels"}
    assert section["NS"] == 2
    for level in section["levels"]:
        assert set(level) == {"QM", "QI", "IZAP", "LFS", "Y"}
        assert "sigma" not in level, "MF=9 carries a yield, not a cross-section"
        assert level["IZAP"] == 13028
    assert [lv["LFS"] for lv in section["levels"]] == [0, 1]
    assert section["levels"][0]["Y"].y[0] == pytest.approx(0.75)


def test_mf9_yields_are_multiplied_by_their_mf3_cross_section(yields):
    """The pairing, asserted on the numbers rather than assumed.

    MF=9 MT=102 x MF=3 MT=102: capture is 300 / 20 / 1 mb, split 75/25, so the
    ground rows must be 225 / 15 / 0.75 mb and the isomer 75 / 5 / 0.25 mb.
    A row set that merely *exists* would pass a weaker test while multiplying by
    the wrong section.
    """
    ground = _rows(yields, 13, 28, "g")
    meta = _rows(yields, 13, 28, "m")
    sigma_mb = [xs * 1e3 for _e, xs in CAPTURE]
    assert [xs for _e, xs in ground] == pytest.approx([s * 0.75 for s in sigma_mb])
    assert [xs for _e, xs in meta] == pytest.approx([s * 0.25 for s in sigma_mb])


def test_mf9_reconstructs_the_channel_exactly(yields):
    """Y is a normalised fraction, so summing the states must give back the MF=3
    channel. Measured to hold on 53 of 54 real products; exact by construction
    here."""
    ground = dict(_rows(yields, 13, 28, "g"))
    meta = dict(_rows(yields, 13, 28, "m"))
    total = dict(_rows(yields, 13, 28, SUM))
    assert total, "the MF=3 production row must still be there"
    for energy, sigma in total.items():
        assert ground[energy] + meta[energy] == pytest.approx(sigma)


def test_mf9_rows_are_production_rows_with_no_mt(yields):
    """Summed across whatever MTs reach the product, exactly like MF=10's."""
    rows = [r for r in yields.rows if r["state"] in ("g", "m") and r["kind"] == "production"]
    assert rows
    for r in rows:
        assert r["MT"] is None
        assert (r["residual_Z"], r["residual_A"]) == (13, 28)


def test_mf9_rows_carry_no_interpolation_law(yields):
    """An MF=9 row is sigma(E) x Y(E), and no ENDF law describes a product of two
    curves that are not both logarithmic in y (#338/#390).

    Laws 4 and 5 survive multiplication — ln(sigma*Y) = ln(sigma) + ln(Y) — and
    law 1 does, being constant. Laws 2 and 3 do not: linear times linear is
    quadratic. Since `parse_mf9_rows` resamples both curves with `np.interp`, the
    only case where that resampling is exact is both-law-2, which is exactly the
    case whose product is unrepresentable. So there is no branch on which such a
    row could honestly name a law, and inheriting one would claim lin-lin while
    being wrong by 30% on an ordinary interval — see
    `tests/test_endf_interp.py::test_inheriting_lin_lin_through_a_product_would_be_wrong_by_a_third`.

    Asserted positively — the rows exist and every one of them is NULL — so this
    cannot pass by finding no MF=9 rows at all.
    """
    mf9 = [r for r in yields.rows if r["state"] in ("g", "m") and r["kind"] == "production"]
    assert mf9, "no MF=9 rows to check — the fixture stopped producing them"
    assert all(r["interp_law"] is None for r in mf9)
    # And the MF=3 channel rows in the same file still carry theirs, so this is a
    # statement about products rather than the column having quietly gone dark.
    channels = [r for r in yields.rows if r["kind"] == "channel"]
    assert channels and all(r["interp_law"] is not None for r in channels)


def test_mf9_counters_and_guard_denominator(yields):
    assert yields.mf9_sections == 1
    assert yields.mf9_product_sections == 1
    assert yields.mf9_rows == len(CAPTURE) * 2


def test_a_yield_with_no_mf3_section_produces_nothing_and_is_not_counted():
    """A yield alone is not a cross-section. The section must not sit in the
    guard's denominator either, or the guard fires on a file being read right."""
    za = 13027.0
    material = "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 16, AL26_TOTAL),  # some other MT
            mf9_section(za, 102, [(13028, 0, CAPTURE_Y_G)]),  # no MF=3 MT=102
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )
    parsed = _mod().parse_endf_file(material, 13, 27, "n")
    assert parsed.mf9_sections == 1
    assert parsed.mf9_product_sections == 0, "no MF=3 MT=102 to multiply by"
    assert parsed.mf9_rows == 0
    assert not _rows(parsed, 13, 28, "g")


def test_mf9_does_not_double_count_a_product_mf10_already_carries():
    """ENDF routes a product to one file via MF=8's LMF, and no overlap was found
    in 120 sampled evaluations. If one ever appears, MF=10 wins and MF=9 is
    skipped — summing both would report the same production twice."""
    za = 13027.0
    material = "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 102, CAPTURE),
            mf10_section(za, 102, [(13028, 0, AL26_G), (13028, 1, AL26_M)]),
            mf9_section(za, 102, [(13028, 0, CAPTURE_Y_G), (13028, 1, CAPTURE_Y_M)]),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )
    parsed = _mod().parse_endf_file(material, 13, 27, "n")
    assert parsed.mf10_rows > 0, "MF=10 keeps the product"
    assert parsed.mf9_rows == 0, "MF=9 must not add it a second time"
    # The surviving values are MF=10's, not sigma x Y.
    assert [xs for _e, xs in _rows(parsed, 13, 28, "g")] == pytest.approx([60.0, 100.0, 80.0])


def test_yields_summing_above_one_are_carried_but_reported():
    """ENDF/B-VIII.1's Pt-196 MF=9 MT=102 reaches a combined yield of 4.887.
    That is the evaluation's normalisation, so the rows are carried — but a level
    out-producing its own channel is not something to ship silently."""
    za = 13027.0
    material = "".join(
        [
            "".ljust(66) + "TPID\n",
            mf3_section(za, 102, CAPTURE),
            mf9_section(
                za,
                102,
                [(13028, 0, [(1.0e5, 2.0), (1.4e7, 2.0)]), (13028, 1, [(1.0e5, 1.5), (1.4e7, 1.5)])],
            ),
            _line(_endf_float(0.0) * 2 + _endf_int(0) * 4, 0, 0, 0),
            f"{'':<66}{0:>4d}{0:>2d}{0:>3d}{0:>5d}\n",
        ]
    )
    parsed = _mod().parse_endf_file(material, 13, 27, "n")
    assert parsed.mf9_rows > 0, "carried, not dropped"
    assert parsed.mf9_yield_overshoots == {(13, 28): pytest.approx(3.5)}


def test_guard_fires_when_mf9_sections_yield_no_rows(monkeypatch, tmp_path):
    """Same family as #340's MF=10 guard and #376's denominators."""
    m = _stub_library(
        monkeypatch,
        {
            "n_013-Al-27_1325.zip": _mod().ParsedFile(
                rows=[_MF3_ROW, _CHANNEL_ROW],
                mf3_sections=12,
                mf3_usable_sections=12,
                mf3_residual_sections=12,
                mf3_rows=1,
                channel_rows=1,
                mf9_sections=4,
                mf9_product_sections=4,
                mf9_rows=0,
            )
        },
    )
    with pytest.raises(RuntimeError, match="MF=9"):
        m.fetch_library("jendl-5", "n", tmp_path, session=None)


def test_manifest_records_the_yield_rows(monkeypatch, tmp_path):
    m = _stub_library(monkeypatch, {"n_013-Al-27_1325.zip": _mod().parse_endf_file(yield_material(), 13, 27, "n")})
    m.fetch_library("jendl-5", "n", tmp_path, session=None)
    manifest = json.loads((tmp_path / "jendl-5" / "manifest.json").read_text())
    # Per-sublibrary since #383: one --sublibrary run sees one projectile, so a
    # flat count is a fact about whichever run went last, not about the library.
    record = manifest["ingest"]["n"]
    assert record["mf9_sections"] == 1
    assert record["mf9_product_sections"] == 1
    assert record["mf9_rows"] == len(CAPTURE) * 2
    assert record["mf9_yield_overshoots"] == {}


def test_a_run_that_discards_more_sections_than_the_last_one_raises(monkeypatch, tmp_path):
    """The general form of #394 (#394).

    The anchors caught Au-197 because someone had listed that reaction; the other
    68 sections were invisible because nobody had. This asks "did this run throw
    away sections the last one kept?" and so covers every MT.
    """
    m = _mod()
    parsed = m.parse_endf_file(blemished_capture_material(), 79, 197, "n")
    stub = _stub_library(monkeypatch, {"n_079-Au-197_7925.zip": parsed})
    stub.fetch_library("jeff-4.0", "n", tmp_path, session=None)

    record = json.loads((tmp_path / "jeff-4.0" / "manifest.json").read_text())["ingest"]["n"]
    assert record["signed_sections_dropped"] == {}, "baseline: nothing discarded whole"

    # Now a run that discards the section instead of repairing it — the regression.
    regressed = m.ParsedFile(
        rows=list(parsed.rows),
        mf3_sections=1,
        mf3_usable_sections=1,
        mf3_residual_sections=1,
        mf3_rows=parsed.mf3_rows,
        channel_rows=parsed.channel_rows,
        signed_sections={102: 1},
    )
    stub = _stub_library(monkeypatch, {"n_079-Au-197_7925.zip": regressed})
    with pytest.raises(RuntimeError, match="discards 1 MF=3 section"):
        stub.fetch_library("jeff-4.0", "n", tmp_path, session=None)


def test_dropping_fewer_sections_than_before_is_fine(monkeypatch, tmp_path):
    """A fix looks like a decrease. Only an increase is alarming."""
    m = _mod()
    before = m.ParsedFile(
        rows=[_MF3_ROW, _CHANNEL_ROW],
        mf3_sections=3,
        mf3_usable_sections=3,
        mf3_residual_sections=3,
        mf3_rows=1,
        channel_rows=1,
        signed_sections={102: 3},
    )
    stub = _stub_library(monkeypatch, {"n_079-Au-197_7925.zip": before})
    stub.fetch_library("jeff-4.0", "n", tmp_path, session=None)

    after = m.parse_endf_file(blemished_capture_material(), 79, 197, "n")
    stub = _stub_library(monkeypatch, {"n_079-Au-197_7925.zip": after})
    stub.fetch_library("jeff-4.0", "n", tmp_path, session=None)  # must not raise
    record = json.loads((tmp_path / "jeff-4.0" / "manifest.json").read_text())["ingest"]["n"]
    assert record["signed_sections_dropped"] == {}
    assert record["negative_points_dropped"] == {"102": 1}
