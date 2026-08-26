#!/usr/bin/env python3
"""Fetch Swiss StSV Annex 3 and emit dose coefficients + regulatory limits (#294).

The Strahlenschutzverordnung (StSV, SR 814.501) Annex 3 is a 792-row table of
per-radionuclide dose coefficients and regulatory thresholds. It is the Swiss
answer to "below what activity is this material no longer supervised" — and it
ships as a PDF annex to an ordinance, which is exactly the legacy-format problem
this repository exists to fix.

Licence: none needed. Swiss URG (SR 231.1) Art. 5(1)(a) excludes "Gesetze,
Verordnungen, völkerrechtliche Verträge und andere amtliche Erlasse" from
copyright protection entirely. The StSV is a Verordnung, so Annex 3 is not
copyrightable subject matter. See data/licenses.toml.

Source selection
----------------
Fedlex publishes each consolidation as Akoma Ntoso XML (OASIS LegalDocML),
alongside PDF and HTML. We take the XML:

  * the PDF needs column positions recovered from whitespace, and page furniture
    corrupts wrapped rows;
  * fedlex.admin.ch's HTML is a JavaScript app whose DOM is generated from this
    same XML.

The XML has real <table>/<tr>/<th>/<td>, so the table parses exactly.

Two traps this module exists to handle
--------------------------------------
1. **The file URL cannot be constructed.** The trailing artifact number is
   per-consolidation: the 2022-01-01 XML is `…-de-xml-8.xml`, the 2026-07-01 one
   is `…-de-xml-3.xml`. So we discover it from the Fedlex SPARQL endpoint
   instead of guessing.

2. **A wrong URL returns HTTP 200, not 404** — roughly 9 KB of Fedlex
   single-page-app HTML. `curl -f` succeeds and a lenient parser yields zero
   rows. Every fetch is therefore structurally validated: right root element,
   a table present, and a plausible row count. Trusting the status code is how
   a build reports success while producing nothing (cf. #283).

Language
--------
Values are taken from the German consolidation, which is authoritative; labels
come from the official English translation, which Fedlex publishes but marks
"not an official language of the Swiss Confederation … no legal force". The two
are cross-checked cell by cell, and the build fails on any numeric divergence —
a non-binding translation must never be able to change a number.

Usage:
    python scripts/fetch_stsv.py --out data/stsv
    python scripts/fetch_stsv.py --check      # verify the pinned consolidation is current
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

# The consolidation this build is pinned to. Annex 3 was unchanged between
# 2022-01-01 and this one, but the ordinance around it was revised — pin the
# version actually fetched, not the oldest one that happens to agree.
PINNED_CONSOLIDATION = "20260701"

SR_NUMBER = "814.501"
ELI_BASE = "https://fedlex.data.admin.ch/eli/cc/2017/502"
SPARQL_ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"
AKN_NS = {"a": "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"}

LANG_URI = {
    "de": "http://publications.europa.eu/resource/authority/language/DEU",
    "en": "http://publications.europa.eu/resource/authority/language/ENG",
}

# Annex 3 body rows carry 15 cells. Four of them are not values but *footnote
# columns* sitting immediately right of the quantity they annotate — 9 for LL,
# 12 for CA, 14 for CS. They look like layout padding (they are empty on most
# rows) and reading them as such silently discards 400+ provenance markers:
# [1] records that the clearance limit came from the IAEA BSS small-quantity
# table rather than the Brenk calculation, and [2] that daughter nuclides were
# folded into it. The ordinance treats both as material.
COL = {"label": 0, "half_life": 1, "decay": 2}
RAW_CELL_COUNT = 15


@dataclass(frozen=True)
class Quantity:
    value_idx: int
    note_idx: int | None  # dedicated footnote column, where the source has one
    unit: str
    table: str
    description: str


# Splitting dose coefficients from limits keeps the jurisdiction-neutral physics
# reusable and confines the Swiss-specific thresholds, which matters the moment
# a second country's table arrives.
QUANTITIES: dict[str, Quantity] = {
    "e_inh": Quantity(3, None, "Sv/Bq", "dose_coefficients", "Committed effective dose per unit intake by inhalation"),
    "e_ing": Quantity(4, None, "Sv/Bq", "dose_coefficients", "Committed effective dose per unit intake by ingestion"),
    "h_10": Quantity(
        5, None, "(mSv/h)/GBq", "dose_coefficients", "Ambient dose equivalent rate at 1 m from a 1 GBq source"
    ),
    "h_007": Quantity(
        6, None, "(mSv/h)/GBq", "dose_coefficients", "Directional dose equivalent rate at 10 cm from a 1 GBq source"
    ),
    "h_c007": Quantity(7, None, "(mSv/h)/(kBq/cm2)", "dose_coefficients", "Dose rate from skin contamination"),
    "LL": Quantity(8, 9, "Bq/g", "limits", "Clearance limit (Befreiungsgrenze) for specific activity"),
    "LA": Quantity(10, None, "Bq", "limits", "Licensing limit (Bewilligungsgrenze)"),
    "CA": Quantity(11, 12, "Bq/m3", "limits", "Guideline value for air contamination"),
    "CS": Quantity(13, 14, "Bq/cm2", "limits", "Guideline value for surface contamination"),
}

# Isomer suffixes run m, n, p, q for successive metastable states — Sb-124n and
# Ir-192n are real entries here. Matching only `m` silently files the rest as a
# chemical form, which puts "n" in the same column as "HTO" and makes a filter
# on isomer='n' return nothing.
_NUCLIDE_RE = re.compile(r"^([A-Z][a-z]?)-(\d{1,3})([mnpq]\d?)?")
_ELEMENT_ONLY_RE = re.compile(r"^([A-Z][a-z]?)(?![a-z-])")
_FOOTNOTE_RE = re.compile(r"\[(\d+)\]")
# "(+ Töchter)" / "(+ daughters)" says the value covers the decay chain. That is
# a statement about scope, not a chemical form, and it must not share a column
# with HTO and monoxyde.
_DAUGHTERS_RE = re.compile(r"\(\s*\+\s*(Töchter|daughters|filles|figlie)\s*\)", re.I)

_ELEMENTS = (
    "n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr"
).split()
SYMBOL_TO_Z = {s: i for i, s in enumerate(_ELEMENTS)}


@dataclass(frozen=True)
class Value:
    """A parsed cell.

    `is_upper_bound` carries the 233 cells written `<0.001`, `<1`, `<0.1`. Those
    are upper bounds, not measurements. Storing 0.001 would assert a precision
    the ordinance does not, and storing 0 would be a magic value — so the bound
    travels beside the number as a flag (representation principle 3).
    """

    number: float | None
    is_upper_bound: bool
    raw: str


def _http_get(url: str, *, accept: str | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nucl-parquet/stsv (+https://github.com/exoma-ch/nucl-parquet)"}
    )
    if accept:
        req.add_header("Accept", accept)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        return resp.read()


def sparql(query: str) -> dict:
    import json

    url = f"{SPARQL_ENDPOINT}?{urllib.parse.urlencode({'query': query})}"
    return json.loads(_http_get(url, accept="application/sparql-results+json"))


def current_consolidation() -> str:
    """Return the newest consolidation date (YYYYMMDD) of SR 814.501."""
    q = f"""
    PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
    SELECT ?c ?d WHERE {{
      ?c jolux:isMemberOf <{ELI_BASE}> ; jolux:dateApplicability ?d .
    }} ORDER BY DESC(?d) LIMIT 1
    """
    rows = sparql(q)["results"]["bindings"]
    if not rows:
        raise RuntimeError(f"SPARQL returned no consolidations for SR {SR_NUMBER}")
    return rows[0]["d"]["value"].replace("-", "")


def discover_xml_url(consolidation: str, lang: str) -> str:
    """Resolve the XML file URL for a consolidation + language.

    Constructing this by hand is what breaks: the trailing artifact number is
    per-consolidation (`-8` for 2022-01-01, `-3` for 2026-07-01), and a wrong
    guess returns HTTP 200 with an HTML page rather than a 404.
    """
    q = f"""
    PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
    SELECT ?f WHERE {{
      <{ELI_BASE}/{consolidation}> jolux:isRealizedBy ?e .
      ?e jolux:language <{LANG_URI[lang]}> ; jolux:isEmbodiedBy ?m .
      ?m jolux:userFormat ?fmt ; jolux:isExemplifiedBy ?f .
      FILTER(CONTAINS(STR(?fmt), "xml"))
    }}
    """
    rows = sparql(q)["results"]["bindings"]
    urls = [r["f"]["value"] for r in rows if r["f"]["value"].endswith(".xml")]
    if not urls:
        raise RuntimeError(f"no XML manifestation for {consolidation}/{lang} — check the consolidation exists")
    return urls[0]


def fetch_akn(url: str) -> etree._ElementTree:
    """Fetch and structurally validate an Akoma Ntoso document.

    Validation is not defensive padding. Fedlex answers an unknown version with
    HTTP 200 and ~9 KB of single-page-app HTML, so 'the request succeeded' says
    nothing about whether we got the ordinance.
    """
    blob = _http_get(url)
    if len(blob) < 100_000:
        head = blob[:120].decode("utf-8", "replace")
        raise RuntimeError(
            f"{url}\nreturned only {len(blob)} bytes — this is Fedlex's HTML shell, not the ordinance.\n"
            f"HTTP 200 here does NOT mean the document exists. First bytes: {head!r}"
        )
    try:
        tree = etree.fromstring(blob).getroottree()
    except etree.XMLSyntaxError as exc:
        raise RuntimeError(f"{url} did not parse as XML ({exc}); Fedlex likely served an HTML page") from exc
    if etree.QName(tree.getroot()).localname != "akomaNtoso":
        raise RuntimeError(f"{url} is not Akoma Ntoso (root={tree.getroot().tag})")
    return tree


def annex3_rows(tree: etree._ElementTree) -> list[list[str]]:
    """Return Annex 3's body rows as raw cell lists.

    Annex 3 is the first and by far the largest table in the document; it is
    selected by shape (the 15-cell body rows) rather than by position, so a
    reordering upstream fails loudly instead of silently reading another table.
    """
    best: list[list[str]] = []
    for table in tree.getroot().findall(".//a:table", AKN_NS):
        body: list[list[str]] = []
        for tr in table.findall(".//a:tr", AKN_NS):
            cells = [c for c in tr if etree.QName(c).localname in ("td", "th")]
            # Header and body are cleanly separated by cell type: Annex 3 has
            # exactly four <th> rows (spanned group titles, names, column
            # numbers, a spacer) followed by 792 <td> rows. Selecting on the
            # tag is structural, so it cannot mistake a translated header
            # ("Radionuklid" vs "Radionuclide") for data the way a text
            # heuristic would.
            if not cells or any(etree.QName(c).localname == "th" for c in cells):
                continue
            row = [" ".join(c.itertext()).strip() for c in cells]
            if len(row) == RAW_CELL_COUNT and row[COL["label"]]:
                body.append(row)
        if len(body) > len(best):
            best = body
    if len(best) < 500:
        raise RuntimeError(f"found only {len(best)} Annex 3 rows; expected ~792 — the table layout changed")
    return best


def parse_value(raw: str) -> Value:
    """Parse a numeric cell, preserving upper-bound semantics."""
    s = _FOOTNOTE_RE.sub("", raw).replace("\xa0", " ").strip()
    if not s or s in {"-", "–", "—"}:
        return Value(None, False, raw)
    upper = s.startswith("<")
    s = s.lstrip("<").strip()
    # Source writes exponents as "4.10 E-11" and "1.E+02".
    # No comma handling on purpose. "1,000" is 1.0 under a European reading and
    # 1000 under an English one, and nothing in the cell says which. This source
    # uses "." as the decimal separator and no thousands separator, so a comma
    # means the format changed — and build_records raises on any cell that fails
    # to parse rather than letting a guess through.
    s = s.replace(" ", "")
    s = re.sub(r"(?<=[\d.])E", "e", s)
    try:
        return Value(float(s), upper, raw)
    except ValueError:
        return Value(None, upper, raw)


@dataclass(frozen=True)
class Identity:
    """The structured half of a source label.

    Every field here is an assertion, so each has to be independently true —
    lumping them together is what produced two separate collisions:

    * `Th (+ Töchter)` and `U (+ Töchter)` are elements, not nuclides. Returning
      an all-null identity for both made them share a key across all nine
      quantities, so any group-by silently merged natural thorium's limits with
      natural uranium's. Null Z was being used to mean "elemental", which is an
      assertion, not absence of data — the sentinel trap in a different coat.
      Z is recoverable from the symbol; only A is genuinely unknown.

    * `Sb-124n`, `Ir-192n` and friends carry the second isomeric state. Filing
      that "n" as a chemical form put it beside HTO and monoxyde, and made
      `isomer = 'n'` match nothing.
    """

    Z: int | None
    A: int | None
    isomer: str | None
    chemical_form: str | None
    includes_daughters: bool


def parse_identity(label: str) -> Identity:
    """Split a source label into its structured parts.

    The label is not a bare nuclide. `H-3` appears three times — OBT, HTO and
    gaz — with different e_inh and different LA, so keying on the nuclide alone
    merges three legally distinct rows. Some rows are decay chains
    (`Bi-212 / Po-212, Tl-208`); the parent identifies those and the verbatim
    label is kept alongside.
    """
    cleaned = _FOOTNOTE_RE.sub("", label).strip()
    includes_daughters = bool(_DAUGHTERS_RE.search(cleaned))
    cleaned = _DAUGHTERS_RE.sub("", cleaned).strip()

    head = cleaned.split("/")[0].strip()
    parts = [p.strip() for p in head.split(",") if p.strip()]
    if not parts:
        return Identity(None, None, None, None, includes_daughters)

    form = ", ".join(parts[1:]).strip() or None
    m = _NUCLIDE_RE.match(parts[0])
    if m:
        symbol, mass, isomer = m.group(1), int(m.group(2)), m.group(3)
        trailing = parts[0][m.end() :].strip()
        return Identity(SYMBOL_TO_Z.get(symbol), mass, isomer, form or trailing or None, includes_daughters)

    # No mass number: an element-wide entry such as "Th" or "U". Z is known,
    # A genuinely is not.
    em = _ELEMENT_ONLY_RE.match(parts[0])
    z = SYMBOL_TO_Z.get(em.group(1)) if em else None
    trailing = parts[0][em.end() :].strip() if em else parts[0]
    return Identity(z, None, None, form or trailing or None, includes_daughters)


def build_records(de_rows: list[list[str]], en_rows: list[list[str]], consolidation: str) -> list[dict]:
    """Cross-check DE against EN, then emit one record per (identity, quantity)."""
    if len(de_rows) != len(en_rows):
        raise RuntimeError(f"DE has {len(de_rows)} rows, EN has {len(en_rows)} — cannot align")

    for i, (de, en) in enumerate(zip(de_rows, en_rows, strict=True)):
        for q in QUANTITIES.values():
            c = q.value_idx
            if de[c].replace(" ", "") != en[c].replace(" ", ""):
                raise RuntimeError(
                    f"row {i}: DE and EN disagree on a value (col {c}): {de[c]!r} vs {en[c]!r}.\n"
                    "The English text is a non-binding translation and must never change a number."
                )

    unparsed: list[tuple[str, str, str]] = []
    iso_date = f"{consolidation[:4]}-{consolidation[4:6]}-{consolidation[6:]}"
    records: list[dict] = []
    for de, en in zip(de_rows, en_rows, strict=True):
        ident = parse_identity(de[COL["label"]])
        for qty, q in QUANTITIES.items():
            cell = de[q.value_idx]
            val = parse_value(cell)
            if val.number is None and _FOOTNOTE_RE.sub("", cell).strip(" -–—\xa0"):
                unparsed.append((de[COL["label"]], qty, cell))
            # Markers can sit inline in the value cell or in the dedicated
            # footnote column beside it; collect both.
            notes = _FOOTNOTE_RE.findall(cell)
            if q.note_idx is not None:
                notes += _FOOTNOTE_RE.findall(de[q.note_idx])
            records.append(
                {
                    "nuclide_Z": ident.Z,
                    "nuclide_A": ident.A,
                    "isomer": ident.isomer,
                    "chemical_form": ident.chemical_form,
                    "includes_daughters": ident.includes_daughters,
                    "nuclide_label_en": en[COL["label"]],
                    "nuclide_label_de": de[COL["label"]],
                    "half_life": de[COL["half_life"]] or None,
                    "decay_mode": de[COL["decay"]] or None,
                    "table": q.table,
                    "quantity": qty,
                    "quantity_description": q.description,
                    "value": val.number,
                    "unit": q.unit,
                    "value_is_upper_bound": val.is_upper_bound,
                    "value_raw": val.raw or None,
                    "source_note": ",".join(dict.fromkeys(notes)) or None,
                    "consolidation_date": iso_date,
                }
            )
    if unparsed:
        shown = "\n".join(f"  {lbl!r} {q}: {cell!r}" for lbl, q, cell in unparsed[:10])
        raise RuntimeError(
            f"{len(unparsed)} value cells did not parse as numbers:\n{shown}\n"
            "A cell that looks numeric but is not understood must stop the build — "
            "silently emitting null is how a wrong table ships with the right row count."
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it (#363).

    `--out` has no default on purpose: with no `--out` this script parses and
    reports but writes nothing, so the default behaviour is already read-only.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="output directory for parquet files")
    ap.add_argument("--consolidation", default=PINNED_CONSOLIDATION)
    ap.add_argument("--check", action="store_true", help="only report whether the pin is current")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    current = current_consolidation()
    if args.check:
        status = "current" if current == args.consolidation else f"STALE (current is {current})"
        print(f"pinned consolidation {args.consolidation}: {status}")
        return 0 if current == args.consolidation else 1
    if current != args.consolidation:
        print(
            f"note: pinned to {args.consolidation}, but {current} is now current. "
            "Re-pin deliberately after diffing Annex 3 — see #294.",
            file=sys.stderr,
        )

    de = annex3_rows(fetch_akn(discover_xml_url(args.consolidation, "de")))
    en = annex3_rows(fetch_akn(discover_xml_url(args.consolidation, "en")))
    records = build_records(de, en, args.consolidation)
    print(f"parsed {len(de)} nuclide rows -> {len(records)} records", file=sys.stderr)

    if not args.out:
        return 0

    import polars as pl

    # Declared, not inferred. `isomer` is null for hundreds of rows before the
    # first "m" appears, so inference from a leading sample picks Null and then
    # fails on the first isomeric nuclide — and an inferred schema is not a
    # schema anyone can rely on.
    schema = {
        "nuclide_Z": pl.Int32,
        "nuclide_A": pl.Int32,
        "isomer": pl.Utf8,
        "chemical_form": pl.Utf8,
        "includes_daughters": pl.Boolean,
        "nuclide_label_en": pl.Utf8,
        "nuclide_label_de": pl.Utf8,
        "half_life": pl.Utf8,
        "decay_mode": pl.Utf8,
        "table": pl.Utf8,
        "quantity": pl.Utf8,
        "quantity_description": pl.Utf8,
        "value": pl.Float64,
        "unit": pl.Utf8,
        "value_is_upper_bound": pl.Boolean,
        "value_raw": pl.Utf8,
        "source_note": pl.Utf8,
        "consolidation_date": pl.Utf8,
    }
    df = pl.DataFrame(records, schema=schema).with_columns(pl.col("consolidation_date").str.to_date("%Y-%m-%d"))
    args.out.mkdir(parents=True, exist_ok=True)
    for table in sorted({q.table for q in QUANTITIES.values()}):
        sub = df.filter(pl.col("table") == table).drop("table")
        # One directory per table so each can carry its own catalog entry and
        # its own data_type: the dose coefficients are jurisdiction-neutral
        # physics, the limits are Swiss law. Filing both under one label would
        # misdescribe half the rows.
        subdir = args.out / table
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"stsv_{table}.parquet"
        sub.write_parquet(path)
        print(f"  wrote {path} ({sub.height} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
