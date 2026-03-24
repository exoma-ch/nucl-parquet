"""Fetch EXFOR experimental cross-section data from the IAEA DataExplorer API.

Downloads measured cross-sections for all projectile+target combinations present
in any evaluated library (TENDL, ENDF, JENDL, etc.) and stores them as Parquet
files matching the nucl-parquet layout under data/exfor/.

Schema per file ({proj}_{Element}.parquet):
    exfor_entry     str     EXFOR entry+subentry ID (e.g. "C1042-010-0")
    target_Z        int32
    target_A        int32   0 = natural element
    residual_Z      int32
    residual_A      int32
    state           str     "", "g", "m", "m1", "m2"
    energy_MeV      f64
    energy_err_MeV  f64     NULL if not reported
    xs_mb           f64     millibarns
    xs_err_mb       f64     NULL if not reported
    author          str     first author
    year            int32

Usage:
    # Fill gaps (skip already-downloaded files):
    uv run python scripts/fetch_exfor.py --all --skip-existing

    # Force re-fetch specific combo:
    uv run python scripts/fetch_exfor.py --projectile p --element Cu

    # Single projectile, all elements:
    uv run python scripts/fetch_exfor.py --projectile n --all

    # All projectiles, all elements (slow — ~6 h with rate limiting):
    uv run python scripts/fetch_exfor.py --all
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
DATA_DIR = ROOT / "data"
API_BASE = "https://nds.iaea.org/dataexplorer/api/reactions/xs"
COMPRESSION = "zstd"
RATE_LIMIT_S = 0.5  # seconds between API calls — IAEA asks for polite access

# Projectile codes: nucl-parquet shorthand -> IAEA API inc_pt field
PROJECTILE_MAP = {
    "n": "N",
    "p": "P",
    "d": "D",
    "t": "T",
    "h": "HE3",
    "a": "A",
}

# EXFOR sf4 product notation: "29-CU-63" -> (29, 63, "")
SF4_PATTERN = re.compile(r"^(\d+)-[A-Z]{1,2}-(\d+)([GM]?)$")

# Element tables
_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",  8: "O",
    9: "F",  10: "Ne",11: "Na",12: "Mg",13: "Al",14: "Si",15: "P", 16: "S",
    17: "Cl",18: "Ar",19: "K", 20: "Ca",21: "Sc",22: "Ti",23: "V", 24: "Cr",
    25: "Mn",26: "Fe",27: "Co",28: "Ni",29: "Cu",30: "Zn",31: "Ga",32: "Ge",
    33: "As",34: "Se",35: "Br",36: "Kr",37: "Rb",38: "Sr",39: "Y", 40: "Zr",
    41: "Nb",42: "Mo",43: "Tc",44: "Ru",45: "Rh",46: "Pd",47: "Ag",48: "Cd",
    49: "In",50: "Sn",51: "Sb",52: "Te",53: "I", 54: "Xe",55: "Cs",56: "Ba",
    57: "La",58: "Ce",59: "Pr",60: "Nd",61: "Pm",62: "Sm",63: "Eu",64: "Gd",
    65: "Tb",66: "Dy",67: "Ho",68: "Er",69: "Tm",70: "Yb",71: "Lu",72: "Hf",
    73: "Ta",74: "W", 75: "Re",76: "Os",77: "Ir",78: "Pt",79: "Au",80: "Hg",
    81: "Tl",82: "Pb",83: "Bi",84: "Po",85: "At",86: "Rn",87: "Fr",88: "Ra",
    89: "Ac",90: "Th",91: "Pa",92: "U",
}
_SYMBOL_TO_Z: dict[str, int] = {sym: z for z, sym in _ELEMENT_SYMBOLS.items()}


def _parse_sf4(sf4: str) -> tuple[int, int, str] | None:
    """Parse EXFOR sf4 like '29-CU-63' → (Z=29, A=63, state='')."""
    if not sf4:
        return None
    m = SF4_PATTERN.match(sf4.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3).lower()


def _parse_residual(residual_str: str) -> tuple[int, int, str] | None:
    """Parse residual like 'Zn-63' or '30-ZN-63-M' → (Z, A, state)."""
    if not residual_str or residual_str == "None":
        return None
    # Try EXFOR Z-SYM-A notation
    m = SF4_PATTERN.match(residual_str.strip().upper())
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3).lower()
    # Try Sym-A[-state] notation
    parts = residual_str.strip().split("-")
    if len(parts) < 2:
        return None
    z = _SYMBOL_TO_Z.get(parts[0].capitalize())
    if z is None:
        return None
    try:
        a = int(parts[1])
    except ValueError:
        return None
    state = parts[2].lower() if len(parts) > 2 else ""
    if state not in ("", "g", "m", "m1", "m2"):
        state = "m" if "m" in state else ""
    return z, a, state


def _get_target_masses(element: str, projectile: str) -> list[int]:
    """Collect all isotope masses for element+projectile across all evaluated libs.

    Returns sorted list including 0 (natural element query). Deduplicates across
    TENDL, ENDF, JENDL, etc. so EXFOR coverage matches our full lib coverage.
    """
    masses: set[int] = {0}  # always query natural element
    for lib_dir in DATA_DIR.iterdir():
        if not lib_dir.is_dir():
            continue
        xs_file = lib_dir / "xs" / f"{projectile}_{element}.parquet"
        if xs_file.exists():
            try:
                df = pl.read_parquet(xs_file, columns=["target_A"])
                masses.update(df["target_A"].unique().to_list())
            except Exception:  # noqa: BLE001
                pass
    return sorted(masses)


def fetch_exfor_for_target(
    element: str,
    target_mass: int,
    projectile: str,
    session: requests.Session,
) -> list[dict]:
    """Fetch all EXFOR rows for a specific target isotope + projectile.

    Handles pagination (20 entries/page), unit conversion (b → mb),
    and both single-product (sf4) and ELEM/MASS multi-product datasets.
    """
    inc_pt = PROJECTILE_MAP[projectile]
    target_z = _SYMBOL_TO_Z.get(element)
    if target_z is None:
        logger.warning("Unknown element symbol: %s", element)
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
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 500:
                # IAEA API returns 500 (not 404) when no data exists for a target
                logger.debug("No data: %s-%d(%s,x)", element, target_mass, projectile)
            else:
                logger.warning("API error %s-%d(%s,x) page %d: %s",
                               element, target_mass, projectile, page, exc)
            break
        except (requests.RequestException, json.JSONDecodeError) as exc:
            logger.warning(
                "API error %s-%d(%s,x) page %d: %s",
                element, target_mass, projectile, page, exc,
            )
            break

        aggregations = data.get("aggregations", {})
        if not aggregations:
            break

        for entry_id, entry in aggregations.items():
            # Only absolute cross-sections (not ratios, spectra, etc.)
            if entry.get("sf6", "") != "SIG":
                continue

            author = entry.get("author", "")
            year = entry.get("year", 0)
            sf4 = entry.get("sf4", "")
            datatable = entry.get("datatable")
            if not datatable:
                continue

            en_inc  = datatable.get("en_inc",  [])
            xs_data = datatable.get("data",    [])
            ddata   = datatable.get("ddata",   [])
            den_inc = datatable.get("den_inc", [])
            residuals = datatable.get("residual", [])

            if not en_inc or not xs_data:
                continue

            def _append_row(i: int, res_z: int, res_a: int, state: str) -> None:
                e   = en_inc[i]
                xs  = xs_data[i]
                if e is None or xs is None:
                    return
                err    = ddata[i]   if i < len(ddata)   and ddata[i]   is not None else None
                en_err = den_inc[i] if i < len(den_inc) and den_inc[i] is not None else None
                try:
                    e_val    = float(e)
                    xs_val   = float(xs) * 1000.0
                    err_val  = float(err) * 1000.0 if err   is not None and str(err)   != "None" else None
                    en_err_v = float(en_err)        if en_err is not None and str(en_err) != "None" else None
                except (ValueError, TypeError):
                    return
                if xs_val <= 0:
                    return
                rows.append({
                    "exfor_entry":    entry_id,
                    "target_Z":       target_z,
                    "target_A":       target_mass,
                    "residual_Z":     res_z,
                    "residual_A":     res_a,
                    "state":          state,
                    "energy_MeV":     e_val,
                    "energy_err_MeV": en_err_v,
                    "xs_mb":          xs_val,
                    "xs_err_mb":      err_val,
                    "author":         author,
                    "year":           year,
                })

            if sf4 == "ELEM/MASS" and residuals:
                for i in range(len(en_inc)):
                    res_str = residuals[i] if i < len(residuals) else None
                    if res_str is None or str(res_str) == "None":
                        continue
                    parsed = _parse_residual(str(res_str))
                    if parsed:
                        _append_row(i, *parsed)
            else:
                parsed = _parse_sf4(sf4)
                if parsed is None:
                    continue
                res_z, res_a, state = parsed
                for i in range(len(en_inc)):
                    _append_row(i, res_z, res_a, state)

        if len(aggregations) < 20:
            break
        page += 1
        time.sleep(RATE_LIMIT_S)

    return rows


def fetch_element(
    element: str,
    projectile: str,
    session: requests.Session,
    *,
    skip_existing: bool = False,
) -> int:
    """Fetch all EXFOR data for element + projectile and write to data/exfor/."""
    out_path = DATA_DIR / "exfor" / f"{projectile}_{element}.parquet"

    if skip_existing and out_path.exists():
        logger.info("  Skipping %s (already exists)", out_path.name)
        return 0

    target_masses = _get_target_masses(element, projectile)
    all_rows: list[dict] = []

    for mass in target_masses:
        logger.info("  Fetching %s-%d(%s,x)", element, mass, projectile)
        rows = fetch_exfor_for_target(element, mass, projectile, session)
        all_rows.extend(rows)
        time.sleep(RATE_LIMIT_S)

    if not all_rows:
        logger.info("  No EXFOR data found for %s(%s,x)", element, projectile)
        return 0

    df = pl.DataFrame(
        all_rows,
        schema={
            "exfor_entry":    pl.Utf8,
            "target_Z":       pl.Int32,
            "target_A":       pl.Int32,
            "residual_Z":     pl.Int32,
            "residual_A":     pl.Int32,
            "state":          pl.Utf8,
            "energy_MeV":     pl.Float64,
            "energy_err_MeV": pl.Float64,
            "xs_mb":          pl.Float64,
            "xs_err_mb":      pl.Float64,
            "author":         pl.Utf8,
            "year":           pl.Int32,
        },
    ).unique(
        subset=["exfor_entry", "target_A", "residual_Z", "residual_A", "state", "energy_MeV"],
    ).sort("residual_Z", "residual_A", "energy_MeV")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression=COMPRESSION)

    logger.info(
        "  Wrote %s: %d rows, %d datasets, %d KB",
        out_path.name, len(df),
        df["exfor_entry"].n_unique(),
        out_path.stat().st_size // 1024,
    )
    return len(df)


def get_elements_for_projectile(projectile: str) -> list[str]:
    """Union of all elements covered by any evaluated library for this projectile."""
    seen: set[str] = set()
    for lib_dir in DATA_DIR.iterdir():
        if not lib_dir.is_dir():
            continue
        xs_dir = lib_dir / "xs"
        if not xs_dir.is_dir():
            continue
        for f in xs_dir.glob(f"{projectile}_*.parquet"):
            elem = f.stem.split("_", 1)[1]
            if not elem.startswith("Z"):  # skip Z96-style placeholders
                seen.add(elem)
    return sorted(seen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EXFOR experimental data from IAEA DataExplorer API.",
    )
    parser.add_argument(
        "--projectile", choices=list(PROJECTILE_MAP),
        help="Projectile (n/p/d/t/h/a). Omit with --all to run all.",
    )
    parser.add_argument(
        "--element",
        help="Single element symbol (e.g. Cu, Li). Requires --projectile.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Fetch all elements covered by any evaluated library.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip elements where data/exfor/{proj}_{elem}.parquet already exists.",
    )
    args = parser.parse_args()

    if not args.element and not args.all:
        parser.error("Specify --element ELEM or --all")
    if args.element and not args.projectile:
        parser.error("--element requires --projectile")

    projectiles = [args.projectile] if args.projectile else list(PROJECTILE_MAP)

    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/fetch_exfor (nuclear data research)"

    total_rows = 0
    for proj in projectiles:
        elements = [args.element] if args.element else get_elements_for_projectile(proj)
        logger.info(
            "Projectile=%s: %d elements to process", proj, len(elements),
        )
        for elem in elements:
            total_rows += fetch_element(
                elem, proj, session, skip_existing=args.skip_existing,
            )

    logger.info("Done. Total EXFOR rows written: %d", total_rows)


if __name__ == "__main__":
    main()
