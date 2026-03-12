"""Fetch ENSDF nuclear structure data from the IAEA LiveChart API.

Downloads gamma transitions, energy levels, and ground state properties
for all nuclides and stores them as Parquet files.

Data includes:
- Gamma-ray energies and intensities (for gamma spectroscopy)
- Energy level schemes (for coincidence analysis)
- Ground state properties (masses, Q-values, abundances)

Usage:
    # Fetch all data:
    python scripts/fetch_ensdf.py --all

    # Fetch specific nuclide:
    python scripts/fetch_ensdf.py --nuclide Co-60

    # Fetch only gammas:
    python scripts/fetch_ensdf.py --all --gammas-only
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
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
API_BASE = "https://nds.iaea.org/relnsd/v1/data"
COMPRESSION = "zstd"
RATE_LIMIT_S = 0.3

# Element symbols for building nuclide strings
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
    93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es",
    100: "Fm", 101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db",
    106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds", 111: "Rg",
    112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts",
    118: "Og",
}


def _safe_float(val: str) -> float | None:
    """Convert string to float, returning None for empty/invalid."""
    if not val or val.strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: str) -> int | None:
    """Convert string to int, returning None for empty/invalid."""
    if not val or val.strip() == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def fetch_ground_states(session: requests.Session, output_dir: Path) -> int:
    """Fetch ground state properties for all nuclides.

    Includes: mass, abundance, half-life, decay modes, Q-values,
    separation energies, magnetic/quadrupole moments.
    """
    logger.info("Fetching ground states for all nuclides...")

    resp = session.get(API_BASE, params={"fields": "ground_states", "nuclides": "all"}, timeout=120)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    rows: list[dict] = []

    for row in reader:
        z = _safe_int(row.get("z"))
        n = _safe_int(row.get("n"))
        if z is None or n is None:
            continue

        rows.append({
            "Z": z,
            "N": n,
            "A": z + n,
            "symbol": row.get("symbol", "").strip(),
            "jp": row.get("jp", "").strip(),
            "half_life_s": _safe_float(row.get("half_life_sec")),
            "half_life_unit": row.get("unit_hl", "").strip(),
            "abundance": _safe_float(row.get("abundance")),
            "unc_abundance": _safe_float(row.get("unc_a")),
            "decay_1": row.get("decay_1", "").strip(),
            "decay_1_pct": _safe_float(row.get("decay_1_%")),
            "decay_2": row.get("decay_2", "").strip(),
            "decay_2_pct": _safe_float(row.get("decay_2_%")),
            "decay_3": row.get("decay_3", "").strip(),
            "decay_3_pct": _safe_float(row.get("decay_3_%")),
            "qbm_keV": _safe_float(row.get("qbm")),
            "qa_keV": _safe_float(row.get("qa")),
            "qec_keV": _safe_float(row.get("qec")),
            "sn_keV": _safe_float(row.get("sn")),
            "sp_keV": _safe_float(row.get("sp")),
            "binding_keV": _safe_float(row.get("binding")),
            "atomic_mass_uAMU": _safe_float(row.get("atomic_mass")),
            "mass_excess_keV": _safe_float(row.get("massexcess")),
            "magnetic_dipole": _safe_float(row.get("magnetic_dipole")),
            "electric_quadrupole": _safe_float(row.get("electric_quadrupole")),
            "radius_fm": _safe_float(row.get("radius")),
        })

    if not rows:
        logger.warning("No ground state data returned")
        return 0

    df = pl.DataFrame(rows)
    out_path = output_dir / "ground_states.parquet"
    df.write_parquet(out_path, compression=COMPRESSION)
    logger.info("Wrote %s (%d nuclides, %d KB)", out_path, len(df), out_path.stat().st_size // 1024)
    return len(df)


def fetch_gammas_for_nuclide(
    symbol: str, a: int, session: requests.Session,
) -> list[dict]:
    """Fetch gamma transitions for a single nuclide."""
    nuclide = f"{symbol}-{a}"
    params = {"fields": "gammas", "nuclides": nuclide}

    try:
        resp = session.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.debug("No gamma data for %s: %s", nuclide, e)
        return []

    text = resp.text.strip()
    if not text or text.isdigit() or "," not in text:
        return []

    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []

    for row in reader:
        z = _safe_int(row.get("z"))
        n = _safe_int(row.get("n"))
        if z is None or n is None:
            continue

        energy = _safe_float(row.get("energy"))
        if energy is None:
            continue

        rows.append({
            "Z": z,
            "A": z + n,
            "start_level_idx": _safe_int(row.get("start_level_idx")),
            "start_level_energy_keV": _safe_float(row.get("start_level_energy")),
            "start_level_jp": row.get("start_level_jp", "").strip(),
            "end_level_idx": _safe_int(row.get("end_level_idx")),
            "end_level_energy_keV": _safe_float(row.get("end_level_energy")),
            "end_level_jp": row.get("end_level_jp", "").strip(),
            "energy_keV": energy,
            "unc_energy_keV": _safe_float(row.get("unc_en")),
            "relative_intensity": _safe_float(row.get("relative_intensity")),
            "unc_intensity": _safe_float(row.get("unc_ri")),
            "multipolarity": row.get("multipolarity", "").strip(),
            "mixing_ratio": _safe_float(row.get("mixing_ratio")),
            "tot_conv_coeff": _safe_float(row.get("tot_conv_coeff")),
        })

    return rows


def fetch_levels_for_nuclide(
    symbol: str, a: int, session: requests.Session,
) -> list[dict]:
    """Fetch energy levels for a single nuclide."""
    nuclide = f"{symbol}-{a}"
    params = {"fields": "levels", "nuclides": nuclide}

    try:
        resp = session.get(API_BASE, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.debug("No level data for %s: %s", nuclide, e)
        return []

    text = resp.text.strip()
    if not text or text.isdigit() or "," not in text:
        return []

    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict] = []

    for row in reader:
        z = _safe_int(row.get("z"))
        n = _safe_int(row.get("n"))
        if z is None or n is None:
            continue

        rows.append({
            "Z": z,
            "A": z + n,
            "level_idx": _safe_int(row.get("idx")),
            "energy_keV": _safe_float(row.get("energy")),
            "unc_energy_keV": _safe_float(row.get("unc_e")),
            "jp": row.get("jp", "").strip(),
            "half_life_s": _safe_float(row.get("half_life_sec")),
            "half_life_unit": row.get("unit_hl", "").strip(),
            "decay_1": row.get("decay_1", "").strip(),
            "decay_1_pct": _safe_float(row.get("decay_1_%")),
            "decay_2": row.get("decay_2", "").strip(),
            "decay_2_pct": _safe_float(row.get("decay_2_%")),
            "decay_3": row.get("decay_3", "").strip(),
            "decay_3_pct": _safe_float(row.get("decay_3_%")),
            "isospin": _safe_float(row.get("isospin")),
            "magnetic_dipole": _safe_float(row.get("magnetic_dipole")),
            "electric_quadrupole": _safe_float(row.get("electric_quadrupole")),
        })

    return rows


def fetch_all_structure(
    output_dir: Path,
    session: requests.Session,
    gammas_only: bool = False,
) -> None:
    """Fetch gammas and levels for all nuclides, write per-element Parquet."""
    # First get the ground states to know which nuclides exist
    gs_path = output_dir / "ground_states.parquet"
    if not gs_path.exists():
        fetch_ground_states(session, output_dir)

    gs_df = pl.read_parquet(gs_path)

    # Group by element
    elements = gs_df.group_by("Z", "symbol").agg(pl.col("A")).sort("Z")

    gamma_dir = output_dir / "gammas"
    gamma_dir.mkdir(parents=True, exist_ok=True)

    level_dir = output_dir / "levels"
    if not gammas_only:
        level_dir.mkdir(parents=True, exist_ok=True)

    total_gammas = 0
    total_levels = 0

    for row in elements.iter_rows(named=True):
        z = row["Z"]
        symbol = row["symbol"]
        masses = sorted(row["A"])

        all_gamma_rows: list[dict] = []
        all_level_rows: list[dict] = []

        for a in masses:
            gamma_rows = fetch_gammas_for_nuclide(symbol, a, session)
            all_gamma_rows.extend(gamma_rows)

            if not gammas_only:
                level_rows = fetch_levels_for_nuclide(symbol, a, session)
                all_level_rows.extend(level_rows)

            time.sleep(RATE_LIMIT_S)

        # Write gammas
        if all_gamma_rows:
            gdf = pl.DataFrame(all_gamma_rows, infer_schema_length=None)
            gpath = gamma_dir / f"{symbol}.parquet"
            gdf.write_parquet(gpath, compression=COMPRESSION)
            total_gammas += len(gdf)

        # Write levels
        if all_level_rows:
            ldf = pl.DataFrame(all_level_rows, infer_schema_length=None)
            lpath = level_dir / f"{symbol}.parquet"
            ldf.write_parquet(lpath, compression=COMPRESSION)
            total_levels += len(ldf)

        n_g = len(all_gamma_rows)
        n_l = len(all_level_rows)
        if n_g > 0 or n_l > 0:
            logger.info("  %s (Z=%d): %d gammas, %d levels from %d isotopes",
                        symbol, z, n_g, n_l, len(masses))

    logger.info("Done. Total: %d gamma transitions, %d levels", total_gammas, total_levels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ENSDF nuclear structure data from IAEA LiveChart API.",
    )
    parser.add_argument("--nuclide", help="Specific nuclide (e.g., Co-60)")
    parser.add_argument("--all", action="store_true", help="Fetch all nuclides")
    parser.add_argument("--ground-states-only", action="store_true")
    parser.add_argument("--gammas-only", action="store_true",
                        help="Skip level data (faster)")
    parser.add_argument("--output", type=Path, default=ROOT / "meta" / "ensdf")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (nuclear data research)"

    if args.ground_states_only:
        fetch_ground_states(session, args.output)
        return

    if args.nuclide:
        # Parse nuclide like "Co-60"
        parts = args.nuclide.split("-")
        symbol = parts[0]
        a = int(parts[1])

        gamma_rows = fetch_gammas_for_nuclide(symbol, a, session)
        if gamma_rows:
            gdf = pl.DataFrame(gamma_rows, infer_schema_length=None)
            gpath = args.output / "gammas" / f"{symbol}.parquet"
            gpath.parent.mkdir(parents=True, exist_ok=True)
            gdf.write_parquet(gpath, compression=COMPRESSION)
            logger.info("Wrote %s (%d transitions)", gpath, len(gdf))

        if not args.gammas_only:
            level_rows = fetch_levels_for_nuclide(symbol, a, session)
            if level_rows:
                ldf = pl.DataFrame(level_rows, infer_schema_length=None)
                lpath = args.output / "levels" / f"{symbol}.parquet"
                lpath.parent.mkdir(parents=True, exist_ok=True)
                ldf.write_parquet(lpath, compression=COMPRESSION)
                logger.info("Wrote %s (%d levels)", lpath, len(ldf))

    elif args.all:
        # First fetch ground states
        fetch_ground_states(session, args.output)
        # Then fetch gammas + levels per element
        fetch_all_structure(args.output, session, gammas_only=args.gammas_only)
    else:
        parser.error("Specify --nuclide or --all")


if __name__ == "__main__":
    main()
