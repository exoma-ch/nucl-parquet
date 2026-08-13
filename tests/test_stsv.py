"""Gates for the Swiss StSV Annex 3 ingest (#294).

The parser's failure modes are all *silent* ones — it is easy to produce 792
plausible rows that are wrong in ways nothing downstream would notice:

  * treating the footnote columns as layout padding drops 500+ provenance
    markers while the row count stays right;
  * keying on the nuclide instead of the full label silently merges H-3's three
    chemical forms, which carry different limits;
  * parsing ``<0.001`` as 0.001 turns an upper bound into a measurement;
  * accepting Fedlex's HTML shell (served with HTTP 200) yields zero rows from
    a "successful" fetch;
  * an identity that does not identify — two element-wide rows collapsing to the
    same all-null key, or an isomeric state filed as a chemical form.

Each of those gets a test, and the ones asserting on the published parquet are
deliberately *not* marked ``data`` — they are where the real invariants live, so
letting CI's ``-m "not data"`` deselect them would defeat the point. Only the
live Fedlex fetch is marked ``network``, to keep CI hermetic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_DATA = _REPO_ROOT / "data" / "stsv"
_LIMITS = _DATA / "limits" / "stsv_limits.parquet"
_DOSE = _DATA / "dose_coefficients" / "stsv_dose_coefficients.parquet"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_stsv", _REPO_ROOT / "scripts" / "fetch_stsv.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which is None for an unregistered module and
    # fails with a confusing AttributeError inside dataclasses itself.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    stsv = _load_module()
except ImportError as exc:  # lxml missing
    pytest.skip(f"fetch_stsv unavailable: {exc}", allow_module_level=True)


# -- Pure parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "number", "upper"),
    [
        ("4.10 E-11", 4.10e-11, False),
        ("1.E+02", 1.0e2, False),
        ("<0.001", 0.001, True),
        ("<1", 1.0, True),
        ("<0.1", 0.1, True),
        ("1.E+03 [1]", 1.0e3, False),
        ("", None, False),
        ("1000", 1000.0, False),
    ],
)
def test_parse_value(raw: str, number: float | None, upper: bool) -> None:
    """Exponent spellings and upper bounds both survive parsing.

    The source writes exponents three ways (``4.10 E-11``, ``1.E+02``, plain)
    and marks 233 cells as upper bounds. Collapsing ``<0.001`` to 0.001 without
    the flag would assert a precision the ordinance explicitly declines to.
    """
    v = stsv.parse_value(raw)
    assert v.number == pytest.approx(number) if number is not None else v.number is None
    assert v.is_upper_bound is upper


def test_upper_bound_keeps_the_number_and_the_flag() -> None:
    """A bound is a number *plus* a claim about it — never a sentinel."""
    v = stsv.parse_value("<0.001")
    assert v.number == pytest.approx(0.001)
    assert v.is_upper_bound
    assert v.raw == "<0.001", "the verbatim cell must survive for audit"


@pytest.mark.parametrize(
    ("label", "expect"),
    [
        ("H-3, OBT", (1, 3, None, "OBT", False)),
        ("H-3, HTO", (1, 3, None, "HTO", False)),
        ("Co-60", (27, 60, None, None, False)),
        ("Ag-104m", (47, 104, "m", None, False)),
        ("Cs-137 / Ba-137m", (55, 137, None, None, False)),
        ("Bi-212 / Po-212, Tl-208", (83, 212, None, None, False)),
        # Successive isomeric states run m, n, p, q. Accepting only "m" filed
        # the "n" as a chemical form, so it sat beside HTO and a query for
        # isomer='n' matched nothing.
        ("Sb-124n", (51, 124, "n", None, False)),
        ("Ir-192n", (77, 192, "n", None, False)),
        # Element-wide entries. Returning an all-null identity for both made
        # natural thorium and natural uranium share a key across all nine
        # quantities — Z is recoverable, only A is genuinely unknown.
        ("Th (+ Töchter)", (90, None, None, None, True)),
        ("U (+ Töchter)", (92, None, None, None, True)),
        # "+ daughters" is a statement about scope, not a chemical form.
        ("Sr-90 (+ Töchter)", (38, 90, None, None, True)),
    ],
)
def test_parse_identity(label: str, expect: tuple) -> None:
    """Z/A/isomer/form/daughter-scope recovered from the source label.

    Decay-chain rows are identified by their parent; the verbatim label is kept
    alongside so the daughters are not lost.
    """
    ident = stsv.parse_identity(label)
    assert (ident.Z, ident.A, ident.isomer, ident.chemical_form, ident.includes_daughters) == expect


def test_value_parser_refuses_to_guess_a_separator() -> None:
    """ "1,000" is 1.0 in one convention and 1000 in another, and the cell does not say.

    The source uses "." as its decimal separator and no thousands separator, so
    a comma means the format changed. Guessing silently would put a value off by
    1000x into a legal limit; build_records raises on any such cell instead.
    """
    assert stsv.parse_value("1,000").number is None


def test_unparsed_value_cell_stops_the_build() -> None:
    """A cell that looks numeric but is not understood must not become null.

    Emitting null keeps the row count right and the table wrong, which is the
    hardest kind of error to notice downstream.
    """
    row = ["Co-60", "5.27 a", "b", "1", "1", "1", "1", "1", "12,5", "", "3.E+05", "5.E+02", "", "3", ""]
    with pytest.raises(RuntimeError, match="did not parse as numbers"):
        stsv.build_records([row], [row[:]], "20260701")


def test_footnote_columns_are_not_treated_as_padding() -> None:
    """LL, CA and CS each have a dedicated footnote column beside them.

    Those columns are empty on most rows, which makes them look like layout
    spacers. Reading them as such silently discards the markers recording that
    a clearance limit came from the IAEA BSS small-quantity table ([1]) or
    folded in daughter nuclides ([2]) — 400+ of them, with the row count
    unchanged, so nothing downstream would notice.
    """
    assert stsv.QUANTITIES["LL"].note_idx == 9
    assert stsv.QUANTITIES["CA"].note_idx == 12
    assert stsv.QUANTITIES["CS"].note_idx == 14


def test_quantities_are_split_across_two_tables() -> None:
    """Dose coefficients and legal limits must not share a table.

    One is jurisdiction-neutral physics, the other is Swiss law with its own
    revision cadence. Merging them makes a second country's table a schema
    change instead of new rows.
    """
    tables = {q.table for q in stsv.QUANTITIES.values()}
    assert tables == {"dose_coefficients", "limits"}
    assert stsv.QUANTITIES["e_inh"].table == "dose_coefficients"
    assert stsv.QUANTITIES["LL"].table == "limits"


# -- Fetch validation -------------------------------------------------------


def test_fetch_rejects_the_html_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fedlex answers an unknown version with HTTP 200 and an HTML page.

    ``curl -f`` succeeds, so a fetcher that trusts the status code parses zero
    rows from a "successful" request. This is #283's failure mode — a build
    reporting green while producing nothing.
    """
    shell = b"<!DOCTYPE html><html data-beasties-container><head><title>Casemates</title>" + b" " * 8000
    monkeypatch.setattr(stsv, "_http_get", lambda *a, **k: shell)
    with pytest.raises(RuntimeError, match="HTML shell|did not parse"):
        stsv.fetch_akn("https://example.invalid/whatever.xml")


def test_fetch_rejects_non_akoma_ntoso(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid XML that is not the ordinance must not be accepted either."""
    body = b"<?xml version='1.0'?><root>" + b"<pad/>" * 30000 + b"</root>"
    monkeypatch.setattr(stsv, "_http_get", lambda *a, **k: body)
    with pytest.raises(RuntimeError, match="not Akoma Ntoso"):
        stsv.fetch_akn("https://example.invalid/whatever.xml")


def test_de_en_numeric_divergence_is_fatal() -> None:
    """The English text is a non-binding translation and must never move a number.

    Fedlex publishes it marked "not an official language of the Swiss
    Confederation ... no legal force". Labels may differ; values may not.
    """
    de = [["Co-60", "5.27 a", "b", "1", "1", "1", "1", "1", "1.E+01", "", "3.E+05", "5.E+02", "", "3", ""]]
    en = [r[:] for r in de]
    en[0][8] = "9.E+09"  # a translated LL that disagrees
    with pytest.raises(RuntimeError, match="non-binding translation"):
        stsv.build_records(de, en, "20260701")


def test_label_divergence_is_allowed() -> None:
    """Spelling differences between the language versions are expected.

    The German text writes 'C-11 monoxyde' where the English writes
    'C-11 monoxide'. That must not fail the build.
    """
    de = [["C-11 monoxyde", "20 m", "b", "1", "1", "1", "1", "1", "1.E+01", "", "3.E+05", "5.E+02", "", "3", ""]]
    en = [["C-11 monoxide", "20 m", "b", "1", "1", "1", "1", "1", "1.E+01", "", "3.E+05", "5.E+02", "", "3", ""]]
    recs = stsv.build_records(de, en, "20260701")
    assert recs, "label-only differences must not abort the build"
    assert recs[0]["nuclide_label_de"] == "C-11 monoxyde"
    assert recs[0]["nuclide_label_en"] == "C-11 monoxide"


# -- The published tables ---------------------------------------------------
#
# Deliberately NOT marked `data`, following tests/test_data_release.py: these
# parquets are committed to the repo, and they are where the real assertions
# live (the H-3 collision, the exact footnote counts, the 233 upper bounds).
# Marking them `data` would let CI's `-m "not data"` deselect precisely the
# checks that make this ingest trustworthy.


def test_published_tables_exist_and_are_long() -> None:
    import polars as pl

    for path, n in ((_LIMITS, 3168), (_DOSE, 3960)):
        assert path.exists(), f"missing {path}"
        df = pl.read_parquet(path)
        assert df.height == n, f"{path.name}: expected {n} rows, got {df.height}"
        # long, not wide: the quantity is a value, not a column name
        assert {"quantity", "value", "unit"} <= set(df.columns)


def test_identity_key_is_unique_per_quantity() -> None:
    """The identity tuple must actually identify a row.

    This is the invariant the scalar count tests cannot express, and the one
    that would have caught both identity bugs found in review: `Th (+ Töchter)`
    and `U (+ Töchter)` collapsing to an all-null key, and `Sb-124n` /`Ir-192n`
    losing their isomer. Neither moved any row count, so every other assertion
    in this file still passed while natural thorium and natural uranium shared
    a key across all nine quantities.

    Any group-by or join on this key must be safe; if a future row cannot be
    told apart from another, that is a representation defect, not a query bug.
    """
    import polars as pl

    key = ["nuclide_Z", "nuclide_A", "isomer", "chemical_form", "includes_daughters", "quantity"]
    for path in (_LIMITS, _DOSE):
        df = pl.read_parquet(path)
        dupes = df.group_by(key).len().filter(pl.col("len") > 1)
        assert dupes.height == 0, (
            f"{path.name}: {dupes.height} identity keys match more than one row.\n"
            f"{dupes.head(5)}\n"
            "Two rows sharing an identity means a consumer grouping on it silently "
            "merges distinct regulatory values."
        )


def test_chemical_form_holds_only_chemical_forms() -> None:
    """The column must not become a dumping ground for unparsed label fragments.

    It previously held 'n' (an isomeric state) and '(+ Töchter)' (a statement
    that the value covers the decay chain) alongside HTO and monoxyde. Each of
    those is a different kind of claim and needs its own column.
    """
    import polars as pl

    forms = set(pl.concat([pl.read_parquet(_LIMITS), pl.read_parquet(_DOSE)])["chemical_form"].drop_nulls().to_list())
    assert "n" not in forms, "an isomeric state leaked into chemical_form"
    assert not any("Töchter" in f or "daughter" in f for f in forms), (
        "decay-chain scope leaked into chemical_form; it belongs in includes_daughters"
    )


def test_daughter_inclusive_entries_are_flagged() -> None:
    """'(+ Töchter)' rows are marked, not silently indistinguishable.

    A clearance limit that covers a whole decay chain is a different claim from
    one that covers a single nuclide.
    """
    import polars as pl

    lim = pl.read_parquet(_LIMITS)
    assert lim.filter("includes_daughters").height > 0, "no daughter-inclusive rows found — parsing regressed"
    th = lim.filter((pl.col("nuclide_Z") == 90) & pl.col("nuclide_A").is_null() & (pl.col("quantity") == "LL"))
    u = lim.filter((pl.col("nuclide_Z") == 92) & pl.col("nuclide_A").is_null() & (pl.col("quantity") == "LL"))
    assert th.height == 1 and u.height == 1, "elemental Th and U must be separately identifiable"
    assert th["includes_daughters"][0] and u["includes_daughters"][0]


def test_h3_chemical_forms_stay_distinct() -> None:
    """H-3 appears three times with three different licensing limits.

    OBT, HTO and gaz are legally distinct. A schema keyed on (Z, A) alone would
    silently collapse them and hand a consumer one arbitrary limit of the three.
    """
    import polars as pl

    df = pl.read_parquet(_LIMITS).filter(
        (pl.col("nuclide_Z") == 1) & (pl.col("nuclide_A") == 3) & (pl.col("quantity") == "LA")
    )
    assert df.height == 3, "H-3's three chemical forms must remain separate rows"
    assert df["value"].n_unique() == 3, "the three forms carry three different LA values"
    assert set(df["chemical_form"].to_list()) == {"OBT", "HTO", "gaz"}


def test_footnote_provenance_survived_the_build() -> None:
    """The counts are exact, so a regression in footnote handling is visible."""
    import polars as pl

    lim = pl.read_parquet(_LIMITS)
    ll = lim.filter(pl.col("quantity") == "LL")
    assert ll.filter(pl.col("source_note") == "1").height == 303, "IAEA small-quantity basis markers"
    assert ll.filter(pl.col("source_note") == "2").height == 106, "daughter-inclusive markers"


def test_upper_bounds_are_flagged_not_silently_numeric() -> None:
    import polars as pl

    both = pl.concat([pl.read_parquet(_LIMITS), pl.read_parquet(_DOSE)])
    flagged = both.filter(pl.col("value_is_upper_bound"))
    assert flagged.height == 233
    assert all(r.startswith("<") for r in flagged["value_raw"].to_list())


def test_every_row_records_its_consolidation() -> None:
    """A regulatory value without its version is unattributable."""
    import polars as pl

    for path in (_LIMITS, _DOSE):
        df = pl.read_parquet(path)
        assert df["consolidation_date"].null_count() == 0
        assert df["consolidation_date"].n_unique() == 1


def test_catalog_and_licence_entries_exist() -> None:
    """Rows must be attributable from the catalog, not from a path regex."""
    import json
    import tomllib

    cat = json.loads((_REPO_ROOT / "data" / "catalog.json").read_text())
    for key in ("stsv-2026-limits", "stsv-2026-dose-coefficients"):
        assert key in cat["libraries"], f"catalog.json missing {key}"
        assert cat["libraries"][key]["version"] == "2026-07-01"

    lic = tomllib.loads((_REPO_ROOT / "data" / "licenses.toml").read_text())
    entry = lic["libraries"]["stsv-2026"]
    assert entry["redistributable"] is True
    assert entry["risk"] == "green"
    assert "Art. 5" in entry["license"], "the URG basis must be recorded, not just 'public domain'"


@pytest.mark.network
def test_pinned_consolidation_is_still_current() -> None:
    """Warns when Fedlex publishes a newer consolidation.

    Not a hard failure on its own — Annex 3 was unchanged between 2022-01-01
    and the pinned version even though the ordinance was revised — but a new
    consolidation is a prompt to diff the annex and re-pin deliberately.
    """
    current = stsv.current_consolidation()
    assert current == stsv.PINNED_CONSOLIDATION, (
        f"Fedlex now publishes consolidation {current}, but the build is pinned to "
        f"{stsv.PINNED_CONSOLIDATION}. Diff Annex 3 and re-pin (see #294)."
    )
