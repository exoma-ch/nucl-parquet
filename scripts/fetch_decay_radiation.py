"""Fetch comprehensive decay radiation data from NNDC NuDat3.

Downloads ALL radiation types for radioactive nuclides:
- Alpha particles (energy, intensity)
- Beta- (average energy, endpoint energy, intensity)
- Beta+/EC (positron energy, EC fraction)
- Gamma rays (energy, intensity, dose)
- X-rays (K-alpha, K-beta, L lines)
- Conversion electrons (CE K/L/M/N shells)
- Auger electrons (K/L shells)
- Gamma coincidence data

Source: NNDC/BNL NuDat3 (ENSDF evaluated data)
https://www.nndc.bnl.gov/nudat3/

Usage:
    # Fetch single nuclide:
    python scripts/fetch_decay_radiation.py --nuclide Co-60

    # Fetch all nuclides (uses ground_states list):
    python scripts/fetch_decay_radiation.py --all
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
NUDAT_URL = "https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp"
COMPRESSION = "zstd"
RATE_LIMIT_S = 0.5

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
}
_SYMBOL_TO_Z: dict[str, int] = {v: k for k, v in _ELEMENT_SYMBOLS.items()}


def _clean_text(s: str) -> str:
    """Clean HTML entities and whitespace from text."""
    s = s.replace("&nbsp;", " ").replace("&alpha;", "a").replace("&beta;", "b")
    s = s.replace("&gamma;", "g").replace("&pi;", "pi").replace("&sup;", "")
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()


def _parse_value(s: str) -> float | None:
    """Parse a numeric value, handling ENSDF uncertainty notation like '1.23 <i>4</i>'."""
    s = re.sub(r"<[^>]+>", "", s)  # strip HTML tags
    s = s.replace("&nbsp;", " ").replace("%", "").strip()
    if not s:
        return None
    # Remove uncertainty in parens or italics: "1.23 4" -> "1.23"
    parts = s.split()
    if not parts:
        return None
    try:
        return float(parts[0].replace("E", "e"))
    except (ValueError, TypeError):
        return None


def _parse_nudat_html(html: str, z: int, a: int) -> dict:
    """Parse NuDat3 decay radiation HTML into structured data.

    Returns dict with keys: radiation, coincidences, datasets
    """
    radiation_rows: list[dict] = []
    coincidence_rows: list[dict] = []

    # Split into datasets
    datasets = re.split(r"<u>Dataset #(\d+):", html)

    for i in range(1, len(datasets), 2):
        dataset_num = int(datasets[i])
        content = datasets[i + 1] if i + 1 < len(datasets) else ""

        # Extract parent info from bgcolor=white cells in the parent table
        # Cells: [0]=nucleus, [1]=E(level), [2]=Jπ, [3]=T1/2, [4]=decay mode, [5]=Q-value, [6]=daughter
        decay_mode = ""
        q_value = None
        parent_level = 0.0

        first_section = re.search(r"<u>(Alpha|Beta|Electron|Gamma)", content)
        parent_area = content[:first_section.start()] if first_section else content[:3000]
        white_cells = re.findall(r'bgcolor=white>(.*?)</td>', parent_area, re.DOTALL)

        if len(white_cells) >= 5:
            parent_level = _parse_value(white_cells[1]) or 0.0
            dm_text = _clean_text(white_cells[4])
            # Normalize decay mode: "β-: 100 %" -> "B-", "IT" -> "IT", "a: 100 %" -> "A"
            dm_text = dm_text.replace("b-", "B-").replace("b+", "B+")
            dm_text = dm_text.replace("a:", "A:").replace("EC", "EC")
            decay_mode = dm_text.split(":")[0].strip()
        if len(white_cells) >= 6:
            q_value = _parse_value(white_cells[5])

        # --- Parse radiation tables ---
        # Find all table sections
        sections = re.split(r"<u>([^<]+)</u>:", content)

        current_section = ""
        for j in range(1, len(sections), 2):
            section_name = sections[j].strip()
            section_content = sections[j + 1] if j + 1 < len(sections) else ""

            # Parse table rows
            table_rows = re.findall(r"<tr>(.*?)</tr>", section_content, re.DOTALL)

            if "Alpha" in section_name:
                for tr in table_rows:
                    cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
                    if len(cells) >= 3:
                        energy = _parse_value(cells[0])
                        intensity = _parse_value(cells[1]) if len(cells) > 1 else None
                        dose = _parse_value(cells[2]) if len(cells) > 2 else None
                        if energy is not None:
                            radiation_rows.append({
                                "Z": z, "A": a,
                                "dataset": dataset_num,
                                "parent_level_keV": parent_level,
                                "decay_mode": decay_mode,
                                "rad_type": "alpha",
                                "rad_subtype": "",
                                "energy_keV": energy,
                                "end_point_keV": None,
                                "intensity_pct": intensity,
                                "dose_MeV_per_Bq_s": dose,
                            })

            elif "Beta-" in section_name:
                for tr in table_rows:
                    cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
                    if len(cells) >= 3:
                        avg_energy = _parse_value(cells[0])
                        endpoint = _parse_value(cells[1]) if len(cells) > 1 else None
                        intensity = _parse_value(cells[2]) if len(cells) > 2 else None
                        dose = _parse_value(cells[3]) if len(cells) > 3 else None
                        if avg_energy is not None:
                            radiation_rows.append({
                                "Z": z, "A": a,
                                "dataset": dataset_num,
                                "parent_level_keV": parent_level,
                                "decay_mode": decay_mode,
                                "rad_type": "beta-",
                                "rad_subtype": "",
                                "energy_keV": avg_energy,
                                "end_point_keV": endpoint,
                                "intensity_pct": intensity,
                                "dose_MeV_per_Bq_s": dose,
                            })

            elif "Beta+" in section_name or "EC" in section_name:
                for tr in table_rows:
                    cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
                    if len(cells) >= 3:
                        avg_energy = _parse_value(cells[0])
                        endpoint = _parse_value(cells[1]) if len(cells) > 1 else None
                        intensity = _parse_value(cells[2]) if len(cells) > 2 else None
                        dose = _parse_value(cells[3]) if len(cells) > 3 else None
                        if avg_energy is not None:
                            radiation_rows.append({
                                "Z": z, "A": a,
                                "dataset": dataset_num,
                                "parent_level_keV": parent_level,
                                "decay_mode": decay_mode,
                                "rad_type": "beta+/EC",
                                "rad_subtype": "",
                                "energy_keV": avg_energy,
                                "end_point_keV": endpoint,
                                "intensity_pct": intensity,
                                "dose_MeV_per_Bq_s": dose,
                            })

            elif "Electron" in section_name:
                for tr in table_rows:
                    cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
                    if len(cells) >= 3:
                        # First cell(s) may be type (Auger L, CE K, etc.) and energy
                        subtype_text = _clean_text(cells[0])
                        energy = _parse_value(cells[1]) if len(cells) > 1 else None
                        intensity = _parse_value(cells[2]) if len(cells) > 2 else None
                        dose = _parse_value(cells[3]) if len(cells) > 3 else None

                        # Classify subtype
                        subtype = subtype_text.strip()
                        if not subtype and energy is None:
                            continue

                        rad_type = "electron"
                        if "Auger" in subtype:
                            rad_type = "auger"
                        elif "CE" in subtype:
                            rad_type = "ce"  # conversion electron

                        if energy is not None:
                            radiation_rows.append({
                                "Z": z, "A": a,
                                "dataset": dataset_num,
                                "parent_level_keV": parent_level,
                                "decay_mode": decay_mode,
                                "rad_type": rad_type,
                                "rad_subtype": subtype,
                                "energy_keV": energy,
                                "end_point_keV": None,
                                "intensity_pct": intensity,
                                "dose_MeV_per_Bq_s": dose,
                            })

            elif "Gamma" in section_name and "Coincidence" not in section_name:
                for tr in table_rows:
                    cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
                    if len(cells) >= 3:
                        # May have XR type prefix in first cell
                        subtype = _clean_text(cells[0]) if len(cells) > 3 else ""
                        energy_idx = 1 if len(cells) > 3 else 0
                        energy = _parse_value(cells[energy_idx])
                        intensity = _parse_value(cells[energy_idx + 1]) if energy_idx + 1 < len(cells) else None
                        dose = _parse_value(cells[energy_idx + 2]) if energy_idx + 2 < len(cells) else None

                        rad_type = "gamma"
                        if "XR" in subtype:
                            rad_type = "xray"

                        if energy is not None:
                            radiation_rows.append({
                                "Z": z, "A": a,
                                "dataset": dataset_num,
                                "parent_level_keV": parent_level,
                                "decay_mode": decay_mode,
                                "rad_type": rad_type,
                                "rad_subtype": subtype,
                                "energy_keV": energy,
                                "end_point_keV": None,
                                "intensity_pct": intensity,
                                "dose_MeV_per_Bq_s": dose,
                            })

            elif "Coincidence" in section_name:
                coinc_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", section_content, re.DOTALL)
                for tr in coinc_rows:
                    cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
                    if len(cells) >= 2:
                        gamma_energy = _parse_value(cells[0])
                        coinc_text = _clean_text(cells[1])
                        if gamma_energy is not None and coinc_text:
                            # Parse coincident gamma energies
                            coinc_energies = []
                            for part in coinc_text.split(","):
                                e = _parse_value(part.strip())
                                if e is not None:
                                    coinc_energies.append(e)

                            for ce in coinc_energies:
                                coincidence_rows.append({
                                    "Z": z, "A": a,
                                    "dataset": dataset_num,
                                    "gamma_energy_keV": gamma_energy,
                                    "coinc_energy_keV": ce,
                                })

    return {"radiation": radiation_rows, "coincidences": coincidence_rows}


def fetch_nuclide(
    symbol: str, a: int, session: requests.Session,
) -> dict:
    """Fetch all decay radiation for a nuclide from NuDat3."""
    z = _SYMBOL_TO_Z.get(symbol, 0)
    nuc_str = f"{symbol}-{a}"

    try:
        resp = session.get(NUDAT_URL, params={"nuc": nuc_str, "unc": "NDS"}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.debug("Failed to fetch %s: %s", nuc_str, e)
        return {"radiation": [], "coincidences": []}

    html = resp.text
    if "decay possibilities were found" not in html and "Dataset #" not in html:
        return {"radiation": [], "coincidences": []}

    return _parse_nudat_html(html, z, a)


def fetch_all(output_dir: Path, session: requests.Session) -> None:
    """Fetch decay radiation for all known radioactive nuclides."""
    # Load ground states to get nuclide list
    gs_path = ROOT / "meta" / "ensdf" / "ground_states.parquet"
    if not gs_path.exists():
        gs_path = ROOT / "meta" / "decay.parquet"

    if gs_path.name == "ground_states.parquet":
        gs = pl.read_parquet(gs_path)
        # Only radioactive nuclides
        nuclides = gs.filter(
            pl.col("half_life_s").is_not_null()
            & (pl.col("half_life_s") > 0)
            & (pl.col("half_life_s") < 1e18)  # exclude stable
        ).select("Z", "A", "symbol").unique().sort("Z", "A")
    else:
        gs = pl.read_parquet(gs_path)
        nuclides = gs.filter(
            pl.col("decay_mode") != "stable"
        ).select("Z", "A").unique().sort("Z", "A")
        # Add symbols
        sym_map = {z: s for z, s in _ELEMENT_SYMBOLS.items()}
        nuclides = nuclides.with_columns(
            pl.col("Z").replace_strict(sym_map, default="?").alias("symbol")
        )

    rad_dir = output_dir / "radiation"
    coinc_dir = output_dir / "coincidences"
    rad_dir.mkdir(parents=True, exist_ok=True)
    coinc_dir.mkdir(parents=True, exist_ok=True)

    # Group by element
    by_element = nuclides.group_by("Z", "symbol").agg(pl.col("A")).sort("Z")

    total_rad = 0
    total_coinc = 0

    for row in by_element.iter_rows(named=True):
        z = row["Z"]
        symbol = row["symbol"]
        masses = sorted(row["A"])

        all_rad: list[dict] = []
        all_coinc: list[dict] = []

        for a in masses:
            result = fetch_nuclide(symbol, a, session)
            all_rad.extend(result["radiation"])
            all_coinc.extend(result["coincidences"])
            time.sleep(RATE_LIMIT_S)

        if all_rad:
            rdf = pl.DataFrame(all_rad, infer_schema_length=None)
            rdf.write_parquet(rad_dir / f"{symbol}.parquet", compression=COMPRESSION)
            total_rad += len(rdf)

        if all_coinc:
            cdf = pl.DataFrame(all_coinc, infer_schema_length=None)
            cdf.write_parquet(coinc_dir / f"{symbol}.parquet", compression=COMPRESSION)
            total_coinc += len(cdf)

        n_r = len(all_rad)
        n_c = len(all_coinc)
        if n_r > 0:
            logger.info("  %s (Z=%d): %d radiation entries, %d coincidences from %d isotopes",
                        symbol, z, n_r, n_c, len(masses))

    logger.info("Done. Total: %d radiation entries, %d coincidences", total_rad, total_coinc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch decay radiation data from NNDC NuDat3.",
    )
    parser.add_argument("--nuclide", help="Specific nuclide (e.g., Co-60)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "meta" / "ensdf")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "nucl-parquet/0.1 (nuclear data research)"

    if args.nuclide:
        parts = args.nuclide.split("-")
        symbol = parts[0]
        a = int(parts[1])
        z = _SYMBOL_TO_Z.get(symbol, 0)

        result = fetch_nuclide(symbol, a, session)
        if result["radiation"]:
            rdf = pl.DataFrame(result["radiation"], infer_schema_length=None)
            out = args.output / "radiation" / f"{symbol}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            rdf.write_parquet(out, compression=COMPRESSION)
            logger.info("Wrote %s (%d entries)", out, len(rdf))
            print(rdf)
        if result["coincidences"]:
            cdf = pl.DataFrame(result["coincidences"], infer_schema_length=None)
            out = args.output / "coincidences" / f"{symbol}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            cdf.write_parquet(out, compression=COMPRESSION)
            logger.info("Wrote %s (%d entries)", out, len(cdf))
            print(cdf)

        if not result["radiation"] and not result["coincidences"]:
            logger.info("No decay radiation data for %s-%d", symbol, a)

    elif args.all:
        fetch_all(args.output, session)
    else:
        parser.error("Specify --nuclide or --all")


if __name__ == "__main__":
    main()
