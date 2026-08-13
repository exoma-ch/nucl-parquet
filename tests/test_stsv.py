"""Gates for the Swiss StSV Annex 3 ingest (#294).

The parser's failure modes are all *silent* ones — it is easy to produce 792
plausible rows that are wrong in ways nothing downstream would notice:

  * treating the footnote columns as layout padding drops 500+ provenance
    markers while the row count stays right;
  * keying on the nuclide instead of the full label silently merges H-3's three
    chemical forms, which carry different limits;
  * parsing ``<0.001`` as 0.001 turns an upper bound into a measurement;
  * accepting Fedlex's HTML shell (served with HTTP 200) yields zero rows from
    a "successful" fetch.

Each of those gets a test. The pure-parsing ones run everywhere; anything that
touches the published parquet is marked ``data``, and the live fetch is marked
``network`` so CI stays hermetic.
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
        ("H-3, OBT", (1, 3, None, "OBT")),
        ("H-3, HTO", (1, 3, None, "HTO")),
        ("Co-60", (27, 60, None, None)),
        ("Ag-104m", (47, 104, "m", None)),
        ("Cs-137 / Ba-137m", (55, 137, None, None)),
        ("Bi-212 / Po-212, Tl-208", (83, 212, None, None)),
    ],
)
def test_parse_identity(label: str, expect: tuple) -> None:
    """Z/A/isomer/form recovered from the source label.

    Decay-chain rows are identified by their parent; the verbatim label is kept
    alongside so the daughters are not lost.
    """
    assert stsv.parse_identity(label) == expect


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
