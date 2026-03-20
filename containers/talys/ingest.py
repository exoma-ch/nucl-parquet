"""Ingest TALYS heavy-ion output parquets into the nucl-parquet library layout.

Run after the container has written output to hi-xs/:
    python3 containers/talys/ingest.py --src hi-xs/ --data-dir .

Creates: hi-xs/{proj}_{target}.parquet per target element, matching the
existing XS file layout (one file per projectile+target combination).
Also registers the library in catalog.json.

Usage:
    uv run python3 containers/talys/ingest.py --src /path/to/hi-xs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

ELEMENT_SYMBOLS = [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",
]
Z_TO_SYMBOL = {i + 1: s for i, s in enumerate(ELEMENT_SYMBOLS)}


def ingest(src_dir: Path, data_dir: Path) -> None:
    for src_file in sorted(src_dir.glob("*.parquet")):
        proj_key = src_file.stem  # e.g. "c12"
        df = pl.read_parquet(src_file)
        print(f"{proj_key}: {len(df):,} rows")

        # Split by target_Z → one file per target element (matches light-ion layout)
        out_dir = data_dir / "hi-xs" / "xs"
        out_dir.mkdir(parents=True, exist_ok=True)

        for target_Z in sorted(df["target_Z"].unique().to_list()):
            target_sym = Z_TO_SYMBOL.get(target_Z, f"Z{target_Z}")
            chunk = df.filter(pl.col("target_Z") == target_Z)
            out_path = out_dir / f"{proj_key}_{target_sym}.parquet"
            chunk.write_parquet(out_path, compression="zstd")

        print(f"  → {out_dir}/")

    # Update catalog.json
    catalog_path = data_dir / "catalog.json"
    catalog = json.loads(catalog_path.read_text())

    if "hi-xs" not in catalog["libraries"]:
        catalog["libraries"]["hi-xs"] = {
            "name": "HI-XS (TALYS-2.0)",
            "description": "Heavy-ion reaction cross-sections computed with TALYS-2.0",
            "source_url": "https://tendl.web.psi.ch/tendl_2025/talys.html",
            "projectiles": sorted({f.stem for f in src_dir.glob("*.parquet")}),
            "data_type": "cross_sections",
            "version": "2.0",
            "path": "hi-xs/xs/",
        }
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
        print("Updated catalog.json with hi-xs library")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",      default="hi-xs",  help="Directory with TALYS output parquets")
    parser.add_argument("--data-dir", default=".",      help="nucl-parquet repo root")
    args = parser.parse_args()

    ingest(Path(args.src), Path(args.data_dir))
