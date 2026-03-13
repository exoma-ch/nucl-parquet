"""Fetch NIST XCOM mass attenuation coefficients and save as Parquet.

Fetches µ/ρ and µ_en/ρ for all 92 elements (Z=1-92) and key compounds
from the NIST X-Ray Mass Attenuation Coefficients database.

Output: meta/xcom_elements.parquet, meta/xcom_compounds.parquet

Usage:
    uv run python -m nucl_parquet.build_xcom
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.request import urlopen

from .download import data_dir as _resolve_data_dir

_BASE_URL = "https://physics.nist.gov/PhysRefData/XrayMassCoef"

# Compound URLs (table 4) — slug : display name
# slug (filename without .html) -> display name
_COMPOUNDS = {
    "air": "Air, Dry",
    "water": "Water, Liquid",
    "bone": "Bone, Cortical (ICRU-44)",
    "tissue": "Tissue, Soft (ICRU-44)",
    "tissue4": "Tissue, Soft (ICRU Four-Component)",
    "muscle": "Muscle, Skeletal (ICRU-44)",
    "adipose": "Adipose Tissue (ICRU-44)",
    "blood": "Blood, Whole (ICRU-44)",
    "brain": "Brain, Grey/White Matter (ICRU-44)",
    "breast": "Breast Tissue (ICRU-44)",
    "lung": "Lung Tissue (ICRU-44)",
    "ovary": "Ovary (ICRU-44)",
    "testis": "Testis (ICRU-44)",
    "eye": "Eye Lens (ICRU-44)",
    "concrete": "Concrete, Ordinary",
    "concreteba": "Concrete, Barite",
    "pyrex": "Glass, Borosilicate (Pyrex)",
    "glass": "Glass, Lead",
    "cesium": "Cesium Iodide",
    "telluride": "Cadmium Telluride",
    "gallium": "Gallium Arsenide",
    "polyethylene": "Polyethylene",
    "pmma": "Polymethyl Methacrylate (PMMA)",
    "polystyrene": "Polystyrene",
    "teflon": "Polytetrafluoroethylene (Teflon)",
    "polyvinyl": "Polyvinyl Chloride (PVC)",
    "a150": "A-150 Tissue-Equivalent Plastic",
    "b100": "B-100 Bone-Equivalent Plastic",
    "c552": "C-552 Air-Equivalent Plastic",
    "lithiumflu": "Lithium Fluoride",
    "lithium": "Lithium Tetraborate",
    "fluoride": "Calcium Fluoride",
    "calcium": "Calcium Sulfate",
}


def _parse_nist_table(html: str) -> list[tuple[float, float, float]]:
    """Extract (energy_MeV, mu_rho, mu_en_rho) rows from NIST HTML page.

    The data is in an ASCII-formatted table within the page, with rows like:
        1.00000E-03  1.057E+04  1.049E+04
    """
    rows = []
    # Match lines with 3 scientific-notation numbers
    pattern = re.compile(r"(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)\s+(\d+\.\d+E[+-]\d+)")
    for match in pattern.finditer(html):
        e = float(match.group(1))
        mu = float(match.group(2))
        mu_en = float(match.group(3))
        rows.append((e, mu, mu_en))
    return rows


def _fetch_url(url: str) -> str:
    """Fetch URL content as string with polite delay."""
    from urllib.request import Request

    req = Request(url, headers={"User-Agent": "nucl-parquet/0.2 (nuclear data research)"})  # noqa: S310
    with urlopen(req) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def build(data_dir: Path | None = None) -> None:
    """Fetch XCOM data and write to Parquet."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    # --- Elements (Z=1 to Z=92) ---
    print("Fetching XCOM element data...")
    elem_Z = []
    elem_E = []
    elem_mu = []
    elem_mu_en = []

    for z in range(1, 93):
        url = f"{_BASE_URL}/ElemTab/z{z:02d}.html"
        try:
            html = _fetch_url(url)
            rows = _parse_nist_table(html)
            for e, mu, mu_en in rows:
                elem_Z.append(z)
                elem_E.append(e)
                elem_mu.append(mu)
                elem_mu_en.append(mu_en)
            print(f"  Z={z:3d}: {len(rows)} points")
        except Exception as exc:
            print(f"  Z={z:3d}: FAILED ({exc})")
        time.sleep(0.3)  # polite rate limiting

    df_elem = pl.DataFrame(
        {
            "Z": pl.Series(elem_Z, dtype=pl.Int32),
            "energy_MeV": pl.Series(elem_E, dtype=pl.Float64),
            "mu_rho_cm2_g": pl.Series(elem_mu, dtype=pl.Float64),
            "mu_en_rho_cm2_g": pl.Series(elem_mu_en, dtype=pl.Float64),
        }
    ).sort("Z", "energy_MeV")

    out_elem = data_dir / "meta" / "xcom_elements.parquet"
    df_elem.write_parquet(out_elem, compression="zstd")
    print(f"\nWrote {len(df_elem)} rows to {out_elem}")

    # --- Compounds ---
    print("\nFetching XCOM compound data...")
    comp_name = []
    comp_E = []
    comp_mu = []
    comp_mu_en = []

    for slug, display_name in _COMPOUNDS.items():
        url = f"{_BASE_URL}/ComTab/{slug}.html"
        try:
            html = _fetch_url(url)
            rows = _parse_nist_table(html)
            for e, mu, mu_en in rows:
                comp_name.append(slug)
                comp_E.append(e)
                comp_mu.append(mu)
                comp_mu_en.append(mu_en)
            print(f"  {slug}: {len(rows)} points")
        except Exception as exc:
            print(f"  {slug}: FAILED ({exc})")
        time.sleep(0.3)

    df_comp = pl.DataFrame(
        {
            "material": pl.Series(comp_name, dtype=pl.Utf8),
            "energy_MeV": pl.Series(comp_E, dtype=pl.Float64),
            "mu_rho_cm2_g": pl.Series(comp_mu, dtype=pl.Float64),
            "mu_en_rho_cm2_g": pl.Series(comp_mu_en, dtype=pl.Float64),
        }
    ).sort("material", "energy_MeV")

    out_comp = data_dir / "meta" / "xcom_compounds.parquet"
    df_comp.write_parquet(out_comp, compression="zstd")
    print(f"Wrote {len(df_comp)} rows to {out_comp}")


if __name__ == "__main__":
    build()
