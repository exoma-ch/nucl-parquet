"""Unit tests for the X4 parser in `scripts/fetch_exfor_master.py`.

Every test here pins a bug that was found by *reading 4.5M output rows*, which is
the wrong place to find them: the whole EXFOR tree has to be re-parsed before the
defect is visible, so a regression ships and is only noticed downstream. These
run against inline fixtures in a few milliseconds and need no data tree.

Fixture format follows real X4 records: 11-character fixed-width fields, keyword
in columns 0-9, pointer in column 10, body from column 11.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_exfor_master import (  # noqa: E402
    MT_BY_PROCESS,
    Stats,
    convert_entry,
    parse_nuclide,
    parse_number,
    split_top,
)

FIELD_W = 11


def _f(*vals: str) -> str:
    """One X4 record line: left-aligned 11-character fields."""
    return "".join(f"{v:<{FIELD_W}}" for v in vals)


def _col(name: str, pointer: str = "") -> str:
    """A DATA/COMMON header field: name in columns 0-9, pointer in column 10."""
    return f"{name:<10}{pointer or ' '}"


def _header(*cols: str) -> str:
    return "".join(cols)


def _entry(subentries: list[str], entry_id: str = "10186") -> str:
    """Wrap subentry bodies in the ENTRY/SUBENT scaffolding the parser walks."""
    head = "\n".join(
        [
            f"SUBENT        {entry_id}001   20050926",
            "BIB                  2          2",
            "AUTHOR     (J.A.Smith)",
            "REFERENCE  (J,NSE,60,1,1976)",
            "ENDBIB               2",
            "NOCOMMON             0          0",
            "ENDSUBENT           10",
        ]
    )
    body = []
    for n, sub in enumerate(subentries, start=2):
        body.append(f"SUBENT        {entry_id}{n:03d}   20050926")
        body.append(sub)
        body.append("ENDSUBENT           99")
    return f"ENTRY            {entry_id}   20050926\n" + head + "\n" + "\n".join(body) + "\nENDENTRY\n"


def _rows(tmp_path: Path, *subentries: str, entry_id: str = "10186") -> list[dict]:
    path = tmp_path / f"{entry_id}.x4"
    path.write_text(_entry(list(subentries), entry_id))
    return convert_entry(path, Stats())


# --------------------------------------------------------------- field parsing


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("14.5", 14.5),
        ("1.23+5", 123000.0),  # ENDF-style implicit exponent
        ("1.23-5", 1.23e-5),
        ("-2.5+3", -2500.0),
        ("1.23E+5", 123000.0),
        ("", None),
        ("   ", None),
        ("ABC", None),
    ],
)
def test_parse_number_handles_implicit_exponents(token: str, expected: float | None) -> None:
    """X4 writes 1.23+5 for 1.23e5. Reading that as `None` silently drops points."""
    assert parse_number(token) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("13-AL-27", (13, 27, "")),
        ("11-NA-24-M", (11, 24, "m")),
        ("29-CU-0", (29, 0, "")),  # A=0 means natural element
        ("0-G-0", (0, 0, "")),
        ("ELEM/MASS", None),
        ("", None),
    ],
)
def test_parse_nuclide(token: str, expected: tuple | None) -> None:
    assert parse_nuclide(token) == expected


def test_split_top_ignores_commas_inside_parens() -> None:
    """SF1 contains the (projectile,process) parens; splitting naively shreds it."""
    assert split_top("13-AL-27(N,A)11-NA-24,,SIG)") == ["13-AL-27(N,A)11-NA-24", "", "SIG"]


# ------------------------------------------------------------ residual nulling


def test_no_residual_is_null_not_zero(tmp_path: Path) -> None:
    """(n,tot) names no residual. NULL, never (0, 0) — principle 3."""
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (13-AL-27(N,TOT),,SIG)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 2          1",
                _header(_col("EN"), _col("DATA")),
                _f("MEV", "B"),
                _f("14.5", "1.75"),
                "ENDDATA              3",
            ]
        ),
    )
    assert len(rows) == 1
    assert rows[0]["residual_Z"] is None
    assert rows[0]["residual_A"] is None
    assert rows[0]["MT"] == 1


def test_emitted_particle_is_not_stored_as_a_zero_residual(tmp_path: Path) -> None:
    """`0-G-0` is EXFOR naming an emitted gamma, not a residual nuclide.

    Parsing it literally yields (0, 0) — the exact sentinel the canonical schema
    exists to remove, and it collides with "this channel names no product".
    1,065 such rows reached a build before the guard was added.
    """
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (6-C-0(A,X)0-G-0,PAR,SIG)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 2          1",
                _header(_col("EN"), _col("DATA")),
                _f("MEV", "MB"),
                _f("20.0", "12.5"),
                "ENDDATA              3",
            ]
        ),
    )
    assert not [r for r in rows if r["residual_Z"] == 0 and r["residual_A"] == 0]


def test_elem_mass_zero_residual_is_not_stored_either(tmp_path: Path) -> None:
    """The per-row ELEM/MASS path needs the same guard as the fixed-residual path.

    Fixing only one of the two leaves the sentinel reachable.
    """
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (29-CU-0(N,X)ELEM/MASS,CUM,SIG)",
                "ENDBIB               1",
                "COMMON               1          1",
                _header(_col("EN")),
                _f("MEV"),
                _f("14.5"),
                "ENDCOMMON            3",
                "DATA                 4          2",
                _header(_col("ELEMENT"), _col("MASS"), _col("DATA"), _col("DATA-ERR")),
                _f("NO-DIM", "NO-DIM", "MB", "MB"),
                _f("11.", "22.", "2.62", "0.29"),
                _f("0.", "0.", "1.00", "0.10"),
                "ENDDATA              4",
            ]
        ),
    )
    assert not [r for r in rows if r["residual_Z"] == 0 and r["residual_A"] == 0]
    assert [(r["residual_Z"], r["residual_A"]) for r in rows] == [(11, 22)]


def test_elem_mass_residuals_vary_per_row(tmp_path: Path) -> None:
    """ELEM/MASS means the residual is a data column, not a fixed subentry property.

    Treating it as fixed drops the whole subentry — 6,500 rows in one build.
    """
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (29-CU-0(N,X)ELEM/MASS,CUM,SIG)",
                "ENDBIB               1",
                "COMMON               1          1",
                _header(_col("EN")),
                _f("MEV"),
                _f("14.5"),
                "ENDCOMMON            3",
                "DATA                 5          3",
                _header(_col("ELEMENT"), _col("MASS"), _col("ISOMER"), _col("DATA"), _col("DATA-ERR")),
                _f("NO-DIM", "NO-DIM", "NO-DIM", "MB", "MB"),
                _f("11.", "22.", "", "2.62", "0.29"),
                _f("17.", "34.", "1.", "1.35", "0.18"),
                _f("30.", "65.", "0.", "3.84", "0.42"),
                "ENDDATA              5",
            ]
        ),
    )
    assert [(r["residual_Z"], r["residual_A"], r["state"]) for r in rows] == [
        (11, 22, ""),
        (17, 34, "m"),
        (30, 65, "g"),
    ]


# ------------------------------------------------------------------- pointers


def test_pointered_reaction_continuations_are_kept(tmp_path: Path) -> None:
    """A pointered REACTION continuation has a blank keyword field.

        REACTION  1(13-AL-27(N,A)11-NA-24,,SIG)
                  2(13-AL-27(N,P)12-MG-27,,SIG)

    Requiring a non-blank keyword drops every reaction after the first in a
    pointered subentry — 52,000 rows in one build.
    """
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          2",
                "REACTION  1(13-AL-27(N,A)11-NA-24,,SIG)",
                "          2(13-AL-27(N,P)12-MG-27,,SIG)",
                "ENDBIB               2",
                "NOCOMMON             0          0",
                "DATA                 3          1",
                _header(_col("EN"), _col("DATA", "1"), _col("DATA", "2")),
                _f("MEV", "MB", "MB"),
                _f("14.5", "120.", "70."),
                "ENDDATA              3",
            ]
        ),
    )
    by_residual = {(r["residual_Z"], r["residual_A"]): r["xs_mb"] for r in rows}
    assert by_residual == {(11, 24): 120.0, (12, 27): 70.0}


def test_pointered_columns_do_not_bleed_between_reactions(tmp_path: Path) -> None:
    """Each pointer must resolve to *its* DATA/DATA-ERR column, not the first one."""
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          2",
                "REACTION  1(13-AL-27(N,A)11-NA-24,,SIG)",
                "          2(13-AL-27(N,P)12-MG-27,,SIG)",
                "ENDBIB               2",
                "NOCOMMON             0          0",
                "DATA                 5          1",
                _header(
                    _col("EN"),
                    _col("DATA", "1"),
                    _col("DATA-ERR", "1"),
                    _col("DATA", "2"),
                    _col("DATA-ERR", "2"),
                ),
                _f("MEV", "MB", "MB", "MB", "MB"),
                _f("14.5", "120.", "5.", "70.", "3."),
                "ENDDATA              3",
            ]
        ),
    )
    got = {(r["residual_Z"], r["residual_A"]): (r["xs_mb"], r["xs_err_mb"]) for r in rows}
    assert got == {(11, 24): (120.0, 5.0), (12, 27): (70.0, 3.0)}


# ---------------------------------------------------------------- units, errors


def test_units_are_converted_to_mev_and_mb(tmp_path: Path) -> None:
    """Units live in the record, not the column name. Principle 6 says the output
    column name carries them instead — which only works if conversion happens."""
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (13-AL-27(N,TOT),,SIG)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 2          2",
                _header(_col("EN"), _col("DATA")),
                _f("KEV", "B"),
                _f("14500.", "1.75"),
                _f("1.45+4", "1.75"),
                "ENDDATA              4",
            ]
        ),
    )
    assert [r["energy_MeV"] for r in rows] == pytest.approx([14.5, 14.5])
    assert [r["xs_mb"] for r in rows] == pytest.approx([1750.0, 1750.0])


def test_per_cent_uncertainty_is_relative_to_the_datum(tmp_path: Path) -> None:
    """PER-CENT is the one unit that is not a fixed scale factor.

    Treating 5 PER-CENT as 5 mb understates a 120 mb datum by 20x.
    """
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (13-AL-27(N,A)11-NA-24,,SIG)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 3          1",
                _header(_col("EN"), _col("DATA"), _col("DATA-ERR")),
                _f("MEV", "MB", "PER-CENT"),
                _f("14.5", "120.", "5."),
                "ENDDATA              3",
            ]
        ),
    )
    assert rows[0]["xs_err_mb"] == pytest.approx(6.0)


def test_energy_from_common_when_data_has_no_energy_column(tmp_path: Path) -> None:
    """A single-point measurement puts EN in COMMON. Requiring it in DATA drops it."""
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (13-AL-27(N,A)11-NA-24,,SIG)",
                "ENDBIB               1",
                "COMMON               1          1",
                _header(_col("EN")),
                _f("MEV"),
                _f("14.5"),
                "ENDCOMMON            3",
                "DATA                 2          1",
                _header(_col("DATA"), _col("DATA-ERR")),
                _f("MB", "MB"),
                _f("120.", "5."),
                "ENDDATA              3",
            ]
        ),
    )
    assert len(rows) == 1
    assert rows[0]["energy_MeV"] == pytest.approx(14.5)


# ------------------------------------------------------------ kind / MT routing


def test_channel_rows_carry_an_mt_and_production_rows_do_not(tmp_path: Path) -> None:
    """`kind` records how the datum was formed, which governs whether rows may be
    added together. (n,a) is one channel; (n,X) is a sum over channels."""
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          2",
                "REACTION  1(13-AL-27(N,A)11-NA-24,,SIG)",
                "          2(13-AL-27(N,X)11-NA-24,,SIG)",
                "ENDBIB               2",
                "NOCOMMON             0          0",
                "DATA                 3          1",
                _header(_col("EN"), _col("DATA", "1"), _col("DATA", "2")),
                _f("MEV", "MB", "MB"),
                _f("14.5", "120.", "130."),
                "ENDDATA              3",
            ]
        ),
    )
    by_kind = {r["kind"]: r for r in rows}
    assert by_kind["channel"]["MT"] == MT_BY_PROCESS["A"]
    assert by_kind["production"]["MT"] is None


def test_fission_product_yield_is_not_the_fission_channel(tmp_path: Path) -> None:
    """`(N,F)56-BA-140` is a fission *product* cross section, not sigma_f.

    ENDF keeps fission yields in MF=8/MT=454 precisely because they are not the
    MF=3/MT=18 channel. Tagging them MT=18 makes the obvious consumer query --
    `WHERE MT = 18` -- silently sum sigma_f with a dozen fragment curves.
    Ba-140 peaks near 100 mb where U-235 sigma_f is thousands of barns, so the
    contamination is invisible in magnitude but wrong in every row it adds.
    """
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          2",
                "REACTION  1(92-U-235(N,F),,SIG)",
                "          2(92-U-235(N,F)56-BA-140,CUM,SIG)",
                "ENDBIB               2",
                "NOCOMMON             0          0",
                "DATA                 3          1",
                _header(_col("EN"), _col("DATA", "1"), _col("DATA", "2")),
                _f("MEV", "B", "MB"),
                _f("14.5", "2.1", "95."),
                "ENDDATA              3",
            ]
        ),
    )
    sigma_f = [r for r in rows if r["MT"] == 18]
    assert len(sigma_f) == 1, "MT=18 must identify sigma_f alone, not fission products"
    assert sigma_f[0]["residual_Z"] is None
    assert sigma_f[0]["xs_mb"] == pytest.approx(2100.0)

    fpy = [r for r in rows if (r["residual_Z"], r["residual_A"]) == (56, 140)]
    assert len(fpy) == 1
    assert fpy[0]["kind"] == "production", "a named fission product is a production datum"
    assert fpy[0]["MT"] is None


def test_elastic_and_inelastic_residuals_are_the_target_itself(tmp_path: Path) -> None:
    """MT=2/4 leave the nucleus unchanged, so a named residual must equal the target.

    Anything else means the SF4 token was misread onto a scattering channel.
    """
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (26-FE-56(N,INL)26-FE-56,,SIG)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 2          1",
                _header(_col("EN"), _col("DATA")),
                _f("MEV", "MB"),
                _f("14.5", "760."),
                "ENDDATA              3",
            ]
        ),
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["MT"] == 4
    assert (r["residual_Z"], r["residual_A"]) == (r["target_Z"], r["target_A"])


def test_rows_with_neither_mt_nor_residual_are_dropped(tmp_path: Path) -> None:
    """A row that identifies no reaction is not a datum — storing it as (0, 0)
    is what made (n,tot), (n,el) and (n,f) indistinguishable (#279)."""
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (13-AL-27(N,X),,SIG)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 2          1",
                _header(_col("EN"), _col("DATA")),
                _f("MEV", "MB"),
                _f("14.5", "120."),
                "ENDDATA              3",
            ]
        ),
    )
    assert rows == []


# ------------------------------------------------------------------ provenance


def test_rows_are_self_describing(tmp_path: Path) -> None:
    """Principle 5: a glob read must not need the path regexed to attribute a row.

    Library, projectile (label and Z/A) and provenance all live in the payload.
    """
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (13-AL-27(N,A)11-NA-24,,SIG)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 2          1",
                _header(_col("EN"), _col("DATA")),
                _f("MEV", "MB"),
                _f("14.5", "120."),
                "ENDDATA              3",
            ]
        ),
    )
    r = rows[0]
    assert r["library"] == "exfor"
    assert (r["projectile"], r["proj_Z"], r["proj_A"]) == ("n", 0, 1)
    assert (r["target_Z"], r["target_A"]) == (13, 27)
    assert r["source_entry"] == "10186-002-0"
    assert r["author"] == "J.A.Smith"
    assert r["year"] == 1976


def test_author_and_year_fall_back_to_the_entry_header(tmp_path: Path) -> None:
    """Only SUBENT 001 carries AUTHOR/REFERENCE; data subentries inherit them."""
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (13-AL-27(N,TOT),,SIG)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 2          1",
                _header(_col("EN"), _col("DATA")),
                _f("MEV", "B"),
                _f("14.5", "1.75"),
                "ENDDATA              3",
            ]
        ),
    )
    assert rows[0]["author"] == "J.A.Smith"
    assert rows[0]["year"] == 1976


def test_non_sig_quantities_are_rejected(tmp_path: Path) -> None:
    """SF6 says what was measured. Only SIG is a cross section; DE is a spectrum,
    RI a resonance integral. Admitting them puts other observables in an xs table."""
    rows = _rows(
        tmp_path,
        "\n".join(
            [
                "BIB                  1          1",
                "REACTION   (13-AL-27(N,X)0-G-0,,DE,,,EVAL)",
                "ENDBIB               1",
                "NOCOMMON             0          0",
                "DATA                 2          1",
                _header(_col("EN"), _col("DATA")),
                _f("MEV", "MB"),
                _f("14.5", "120."),
                "ENDDATA              3",
            ]
        ),
    )
    assert rows == []
