"""Fetch IUPAC/NIST atomic weights and isotopic compositions, convert to Parquet.

Source: NIST Physics Reference Data hosting CIAAW (IUPAC Commission on Isotopic
Abundances and Atomic Weights). Single CGI endpoint returns a fixed-width
ASCII table wrapped in HTML; we extract the <pre>-block and parse it.

Plugs the second G4 gap: G4 data files don't include natural isotopic
abundances or standard atomic weights. CIAAW is the canonical compilation.

Output: data/auxiliary/iupac_compositions.parquet (gitignored — regenerate via this script).

Usage:
    python scripts/fetch_iupac_compositions.py
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
SRC_URL = "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii&isotype=some"
RAW_PATH = ROOT / "data" / "g4_raw" / "iupac" / "compositions.txt"
OUT_PATH = ROOT / "data" / "auxiliary" / "iupac_compositions.parquet"

# `1.00782503223(9)`: leading float, then optional uncertainty in parens.
_VAL_RE = re.compile(r"^([\d.]+)\s*(?:\(([^)]+)\))?$")
# `[1.00784,1.00811]`: range bracket.
_RANGE_RE = re.compile(r"^\[([\d.]+),\s*([\d.]+)\]$")


def _fetch() -> Path:
    if RAW_PATH.exists():
        logger.info("IUPAC compositions cached at %s", RAW_PATH)
        return RAW_PATH
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", SRC_URL)
    resp = requests.get(SRC_URL, timeout=30)
    resp.raise_for_status()
    RAW_PATH.write_bytes(resp.content)
    return RAW_PATH


def _parse_value(s: str) -> tuple[float | None, float | None]:
    """Parse `1.00782503223(9)` → (value, uncertainty in last-digit units).

    The uncertainty `(9)` means ±9 in the last quoted digit. We return the
    raw integer; consumers can compute the absolute uncertainty if they
    care to scale by the last-digit place.
    """
    s = s.strip()
    if not s:
        return None, None
    m = _VAL_RE.match(s)
    if not m:
        return None, None
    val = float(m.group(1))
    unc_str = m.group(2)
    if unc_str is None:
        return val, None
    try:
        return val, float(unc_str)
    except ValueError:
        return val, None


def _parse_atomic_weight(s: str) -> tuple[float | None, float | None, float | None]:
    """Parse standard-atomic-weight column.

    Three forms:
      - `9.0121831(5)` → (9.0121831, 5, None) — single-value
      - `[1.00784,1.00811]` → (None, None, midpoint or low/high) — interval
      - empty → all None
    """
    s = s.strip()
    if not s:
        return None, None, None
    m = _RANGE_RE.match(s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        return None, None, (lo + hi) / 2  # midpoint of conventional range
    val, unc = _parse_value(s)
    return val, unc, None


def _parse(path: Path) -> pl.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")

    # Extract the <pre>...</pre> block (the only one with our table)
    pre_match = re.search(r"<pre>(.*?)</pre>", text, re.DOTALL)
    if not pre_match:
        raise ValueError("Couldn't find <pre>...</pre> data block in HTML")
    body = pre_match.group(1)

    # Remove HTML entities / tags inside (mostly &nbsp; and trailing notes anchors)
    body = re.sub(r"&nbsp;", " ", body)
    body = re.sub(r"<[^>]+>", "", body)

    rows: list[dict] = []
    current_z = None
    current_symbol = None
    current_atomic_weight = (None, None, None)
    current_notes = ""

    # Each isotope row: cols are Z(7), Symbol(4), A(5), Mass(20), Composition(15), AW(20), Notes
    # Z + Symbol + AW + Notes only on the first row of each element block.
    for line in body.splitlines():
        if line.startswith("_") or line.startswith("=") or not line.strip():
            continue
        if "Isotope" in line and "Atomic Mass" in line:
            continue
        if "Relative" in line or "Composition" in line:
            continue
        if "Notes" in line and "appendix" in line.lower():
            break  # footer / notes appendix

        # Anchor on whether the line starts with whitespace or a digit:
        # Z/symbol present on lines starting with a digit.
        if re.match(r"^\s*\d+\s+\S+\s+\d+\s", line):
            # parse Z, symbol, A, then continue with isotope fields
            parts = line.split(None, 3)
            try:
                current_z = int(parts[0])
                current_symbol = parts[1]
                a = int(parts[2])
                rest = parts[3]
            except (ValueError, IndexError):
                continue
        elif re.match(r"^\s+(\S+\s+)?\d+\s", line):
            # continuation row (no Z/symbol; possibly an alternate symbol like D for ²H)
            parts = line.split(None, 2)
            # Two cases:
            #   "    D   2    2.014..."  → D + 2 + rest
            #   "        4    4.0026..." → 4 + rest
            if len(parts) >= 3 and not parts[0].isdigit():
                a_str = parts[1]
                rest = parts[2]
            else:
                a_str = parts[0]
                rest = parts[1] if len(parts) > 1 else ""
            try:
                a = int(a_str)
            except ValueError:
                continue
        else:
            continue

        if current_z is None:
            continue

        # Split `rest` by whitespace runs of 2+ to keep multi-token cells together.
        cols = re.split(r"\s{2,}", rest.strip())
        # Pad to expected 4 cells: mass, composition, atomic_weight, notes
        while len(cols) < 4:
            cols.append("")

        rel_atomic_mass, rel_atomic_mass_unc = _parse_value(cols[0])
        composition, composition_unc = _parse_value(cols[1])
        # Standard atomic weight only on the first row of an element block.
        if cols[2].strip():
            current_atomic_weight = _parse_atomic_weight(cols[2])
            current_notes = cols[3].strip() if len(cols) > 3 else ""

        std_aw, std_aw_unc, std_aw_range = current_atomic_weight

        rows.append(
            {
                "Z": current_z,
                "A": a,
                "symbol": current_symbol,
                "relative_atomic_mass_u": rel_atomic_mass,
                "relative_atomic_mass_unc_last_digit": rel_atomic_mass_unc,
                "isotopic_composition": composition,
                "isotopic_composition_unc_last_digit": composition_unc,
                "standard_atomic_weight": std_aw,
                "standard_atomic_weight_unc_last_digit": std_aw_unc,
                "standard_atomic_weight_range_midpoint": std_aw_range,
                "notes": current_notes,
            }
        )

    return pl.DataFrame(rows).sort("Z", "A")


def main() -> None:
    raw = _fetch()
    df = _parse(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH, compression="zstd")
    logger.info("Wrote %d isotope rows to %s", len(df), OUT_PATH)
    n_with_composition = df.filter(pl.col("isotopic_composition").is_not_null()).height
    logger.info("  with measured composition: %d", n_with_composition)


if __name__ == "__main__":
    main()
