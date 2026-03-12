"""Fetch EXFOR experimental cross-section data from the IAEA DataExplorer API.

Downloads measured cross-sections for charged-particle reactions and stores
them as Parquet files matching the nucl-parquet layout.

Usage:
    # Fetch all proton data for copper:
    python scripts/fetch_exfor.py --projectile p --element Cu

    # Fetch all proton data for all elements in TENDL:
    python scripts/fetch_exfor.py --projectile p --all

    # Fetch all projectiles for all elements:
    python scripts/fetch_exfor.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
API_BASE = "https://nds.iaea.org/dataexplorer/api/reactions/xs"
COMPRESSION = "zstd"
RATE_LIMIT_S = 0.5

# Projectile codes: nucl-parquet shorthand -> IAEA API inc_pt
PROJECTILE_MAP = {
    "n": "N",
    "p": "P",
    "d": "D",
    "t": "T",
    "h": "HE3",
    "a": "A",
}

# Parse EXFOR sf4 product notation: "29-CU-63" -> (29, 63) or "ELEM/MASS" -> None
SF4_PATTERN = re.compile(r"^(\d+)-[A-Z]{1,2}-(\d+)([GM]?)$")


def _parse_sf4(sf4: str) -> tuple[int, int, str] | None:
    """Parse EXFOR sf4 field like '29-CU-63' -> (Z=29, A=63, state='').

    Returns None for unparseable entries like 'ELEM/MASS'.
    """
    if not sf4:
        return None
    m = SF4_PATTERN.match(sf4.strip())
    if not m:
        return None
    z, a, state = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    return (z, a, state)


def _parse_residual(residual_str: str) -> tuple[int, int, str] | None:
    """Parse residual field like 'Zn-63' or 'Co-58-M' -> (Z, A, state).

    Uses element symbol to Z mapping.
    """
    if not residual_str or residual_str == "None":
        return None

    # Try EXFOR notation first: "30-ZN-63"
    m = SF4_PATTERN.match(residual_str.strip().upper())
    if m:
        return (int(m.group(1)), int(m.group(2)), m.group(3).lower())

    # Try symbol notation: "Zn-63" or "Zn-63-M"
    parts = residual_str.strip().split("-")
    if len(parts) < 2:
        return None

    sym = parts[0].capitalize()
    z = _SYMBOL_TO_Z.get(sym)
    if z is None:
        return None

    try:
        a = int(parts[1])
    except ValueError:
        return None

    state = parts[2].lower() if len(parts) > 2 else ""
    if state not in ("", "g", "m", "m1", "m2"):
        state = "m" if "m" in state else ""

    return (z, a, state)


# Element symbols for reverse lookup
_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
    79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
    86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U",
}
_SYMBOL_TO_Z: dict[str, int] = {sym: z for z, sym in _ELEMENT_SYMBOLS.items()}


def fetch_exfor_for_target(
    element: str,
    target_mass: int,
    projectile: str,
    session: requests.Session,
) -> list[dict]:
    """Fetch all EXFOR data for a specific target + projectile from IAEA API.

    Returns list of row dicts ready for DataFrame construction.
    """
    inc_pt = PROJECTILE_MAP[projectile]
    target_z = _SYMBOL_TO_Z.get(element)
    if target_z is None:
        logger.warning("Unknown element: %s", element)
        return []

    rows: list[dict] = []
    page = 1

    while True:
        params = {
            "target_elem": element,
            "target_mass": str(target_mass),
            "inc_pt": inc_pt,
            "table": "true",
            "page": str(page),
        }

        try:
            resp = session.get(API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(
                "API error for %s-%d(%s,x) page %d: %s",
                element, target_mass, projectile, page, e,
            )
            break

        hits = data.get("hits", 0)
        aggregations = data.get("aggregations", {})

        if not aggregations:
            break

        for entry_id, entry in aggregations.items():
            # Only want absolute cross-sections
            sf6 = entry.get("sf6", "")
            if sf6 != "SIG":
                continue

            author = entry.get("author", "")
            year = entry.get("year", 0)
            x4_code = entry.get("x4_code", "")

            # Parse product from sf4
            sf4 = entry.get("sf4", "")
            datatable = entry.get("datatable")
            if not datatable:
                continue

            en_inc = datatable.get("en_inc", [])
            xs_data = datatable.get("data", [])
            ddata = datatable.get("ddata", [])
            den_inc = datatable.get("den_inc", [])
            residuals = datatable.get("residual", [])

            if not en_inc or not xs_data:
                continue

            # Handle ELEM/MASS entries (multiple products in one dataset)
            if sf4 == "ELEM/MASS" and residuals:
                for i, (e, xs) in enumerate(zip(en_inc, xs_data)):
                    if e is None or xs is None:
                        continue
                    res_str = residuals[i] if i < len(residuals) else None
                    if res_str is None or res_str == "None":
                        continue
                    parsed = _parse_residual(str(res_str))
                    if parsed is None:
                        continue
                    res_z, res_a, state = parsed

                    err = ddata[i] if i < len(ddata) and ddata[i] is not None else None
                    en_err = den_inc[i] if i < len(den_inc) and den_inc[i] is not None else None

                    # Convert barns to millibarns
                    try:
                        e_val = float(e)
                        xs_val = float(xs) * 1000.0
                        err_val = float(err) * 1000.0 if err is not None and str(err) != "None" else None
                        en_err_val = float(en_err) if en_err is not None and str(en_err) != "None" else None
                    except (ValueError, TypeError):
                        continue

                    if xs_val <= 0:
                        continue

                    rows.append({
                        "exfor_entry": entry_id,
                        "target_Z": target_z,
                        "target_A": target_mass,
                        "residual_Z": res_z,
                        "residual_A": res_a,
                        "state": state,
                        "energy_MeV": e_val,
                        "energy_err_MeV": en_err_val,
                        "xs_mb": xs_val,
                        "xs_err_mb": err_val,
                        "author": author,
                        "year": year,
                    })
            else:
                # Single product from sf4
                parsed = _parse_sf4(sf4)
                if parsed is None:
                    continue
                res_z, res_a, state = parsed

                for i, (e, xs) in enumerate(zip(en_inc, xs_data)):
                    if e is None or xs is None:
                        continue

                    err = ddata[i] if i < len(ddata) and ddata[i] is not None else None
                    en_err = den_inc[i] if i < len(den_inc) and den_inc[i] is not None else None

                    try:
                        e_val = float(e)
                        xs_val = float(xs) * 1000.0
                        err_val = float(err) * 1000.0 if err is not None and str(err) != "None" else None
                        en_err_val = float(en_err) if en_err is not None and str(en_err) != "None" else None
                    except (ValueError, TypeError):
                        continue

                    if xs_val <= 0:
                        continue

                    rows.append({
                        "exfor_entry": entry_id,
                        "target_Z": target_z,
                        "target_A": target_mass,
                        "residual_Z": res_z,
                        "residual_A": res_a,
                        "state": state,
                        "energy_MeV": e_val,
                        "energy_err_MeV": en_err_val,
                        "xs_mb": xs_val,
                        "xs_err_mb": err_val,
                        "author": author,
                        "year": year,
                    })

        # Pagination: 20 entries per page (API default)
        if len(aggregations) < 20:
            break
        page += 1
        time.sleep(RATE_LIMIT_S)

    return rows


def fetch_element(
    element: str,
    projectile: str,
    output_dir: Path,
    session: requests.Session,
) -> int:
    """Fetch all EXFOR data for an element + projectile, write Parquet."""
    # Get target masses from TENDL to know which isotopes to query
    tendl_path = ROOT / "tendl-2024" / "xs" / f"{projectile}_{element}.parquet"
    target_masses: list[int] = [0]  # Always query natural element (mass=0)

    if tendl_path.exists():
        tendl_df = pl.read_parquet(tendl_path)
        isotope_masses = tendl_df["target_A"].unique().sort().to_list()
        target_masses.extend(isotope_masses)

    all_rows: list[dict] = []

    for mass in target_masses:
        logger.info("  Fetching EXFOR: %s-%d(%s,x)", element, mass, projectile)
        rows = fetch_exfor_for_target(element, mass, projectile, session)
        all_rows.extend(rows)
        time.sleep(RATE_LIMIT_S)

    if not all_rows:
        logger.info("  No EXFOR data for %s(%s,x)", element, projectile)
        return 0

    df = pl.DataFrame(
        all_rows,
        schema={
            "exfor_entry": pl.Utf8,
            "target_Z": pl.Int32,
            "target_A": pl.Int32,
            "residual_Z": pl.Int32,
            "residual_A": pl.Int32,
            "state": pl.Utf8,
            "energy_MeV": pl.Float64,
            "energy_err_MeV": pl.Float64,
            "xs_mb": pl.Float64,
            "xs_err_mb": pl.Float64,
            "author": pl.Utf8,
            "year": pl.Int32,
        },
    )

    # Deduplicate (same entry + energy + product)
    df = df.unique(
        subset=["exfor_entry", "target_A", "residual_Z", "residual_A", "state", "energy_MeV"],
    ).sort("residual_Z", "residual_A", "energy_MeV")

    exfor_dir = output_dir / "exfor"
    exfor_dir.mkdir(parents=True, exist_ok=True)
    out_path = exfor_dir / f"{projectile}_{element}.parquet"
    df.write_parquet(out_path, compression=COMPRESSION)

    logger.info(
        "  Wrote %s (%d rows, %d datasets, %d KB)",
        out_path, len(df),
        df["exfor_entry"].n_unique(),
        out_path.stat().st_size // 1024,
    )
    return len(df)


def get_available_elements(projectile: str) -> list[str]:
    """Get list of elements available across all libraries for a projectile."""
    seen: set[str] = set()
    # Scan all library xs/ directories
    for lib_dir in ROOT.iterdir():
        xs_dir = lib_dir / "xs"
        if not xs_dir.is_dir():
            continue
        for f in xs_dir.glob(f"{projectile}_*.parquet"):
            elem = f.stem.split("_", 1)[1]
            if not elem.startswith("Z"):
                seen.add(elem)

    # Fallback: use the elements table for neutron (all elements are valid targets)
    if not seen and projectile == "n":
        elements_path = ROOT / "meta" / "elements.parquet"
        if elements_path.exists():
            df = pl.read_parquet(elements_path)
            seen = set(df["symbol"].to_list())

    return sorted(seen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EXFOR experimental data from IAEA DataExplorer API.",
    )
    parser.add_argument(
        "--projectile", choices=["n", "p", "d", "t", "h", "a"],
        help="Projectile type (default: all if --all)",
    )
    parser.add_argument(
        "--element", help="Element symbol (e.g., Cu, Fe)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Fetch all elements available in TENDL",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT,
        help="Output directory (default: repo root)",
    )
    args = parser.parse_args()

    if not args.element and not args.all:
        parser.error("Specify --element or --all")

    projectiles = [args.projectile] if args.projectile else list(PROJECTILE_MAP.keys())

    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (nuclear data research)"

    total_rows = 0

    for proj in projectiles:
        if args.all:
            elements = get_available_elements(proj)
        else:
            elements = [args.element]

        logger.info("Fetching EXFOR for projectile=%s, %d elements", proj, len(elements))

        for elem in elements:
            rows = fetch_element(elem, proj, args.output, session)
            total_rows += rows

    logger.info("Done. Total EXFOR rows: %d", total_rows)


if __name__ == "__main__":
    main()
