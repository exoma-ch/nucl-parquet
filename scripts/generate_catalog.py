"""Generate a Markdown catalog of cross-section plots for GitHub Pages.

Produces one MD page per element with embedded SVG plots comparing
evaluated (TENDL) and experimental (EXFOR) cross-section data.

Usage:
    # Generate catalog for a single element:
    python scripts/generate_catalog.py --projectile p --element Cu

    # Generate full catalog:
    python scripts/generate_catalog.py --all

    # Generate full catalog with plot generation:
    python scripts/generate_catalog.py --all --plot
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

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

ELEMENT_NAMES: dict[int, str] = {
    1: "Hydrogen", 2: "Helium", 3: "Lithium", 4: "Beryllium", 5: "Boron",
    6: "Carbon", 7: "Nitrogen", 8: "Oxygen", 9: "Fluorine", 10: "Neon",
    11: "Sodium", 12: "Magnesium", 13: "Aluminium", 14: "Silicon", 15: "Phosphorus",
    16: "Sulfur", 17: "Chlorine", 18: "Argon", 19: "Potassium", 20: "Calcium",
    21: "Scandium", 22: "Titanium", 23: "Vanadium", 24: "Chromium", 25: "Manganese",
    26: "Iron", 27: "Cobalt", 28: "Nickel", 29: "Copper", 30: "Zinc",
    31: "Gallium", 32: "Germanium", 33: "Arsenic", 34: "Selenium", 35: "Bromine",
    36: "Krypton", 37: "Rubidium", 38: "Strontium", 39: "Yttrium", 40: "Zirconium",
    41: "Niobium", 42: "Molybdenum", 43: "Technetium", 44: "Ruthenium",
    45: "Rhodium", 46: "Palladium", 47: "Silver", 48: "Cadmium", 49: "Indium",
    50: "Tin", 51: "Antimony", 52: "Tellurium", 53: "Iodine", 54: "Xenon",
    55: "Caesium", 56: "Barium", 57: "Lanthanum", 58: "Cerium", 59: "Praseodymium",
    60: "Neodymium", 61: "Promethium", 62: "Samarium", 63: "Europium",
    64: "Gadolinium", 65: "Terbium", 66: "Dysprosium", 67: "Holmium",
    68: "Erbium", 69: "Thulium", 70: "Ytterbium", 71: "Lutetium",
    72: "Hafnium", 73: "Tantalum", 74: "Tungsten", 75: "Rhenium", 76: "Osmium",
    77: "Iridium", 78: "Platinum", 79: "Gold", 80: "Mercury", 81: "Thallium",
    82: "Lead", 83: "Bismuth", 84: "Polonium", 85: "Astatine", 86: "Radon",
    87: "Francium", 88: "Radium", 89: "Actinium", 90: "Thorium",
    91: "Protactinium", 92: "Uranium",
}

PROJECTILE_LABELS = {"p": "Proton", "d": "Deuteron", "t": "Triton", "h": "³He", "a": "Alpha"}


def _sym(z: int) -> str:
    return _ELEMENT_SYMBOLS.get(z, f"Z{z}")


def _reaction_str(
    target_a: int, target_sym: str, proj: str,
    res_z: int, res_a: int, state: str,
) -> str:
    res_sym = _sym(res_z)
    s = f"{'m' if state == 'm' else 'g' if state == 'g' else ''}"
    return f"<sup>{target_a}</sup>{target_sym}({proj},x)<sup>{res_a}{s}</sup>{res_sym}"


def _svg_path(
    proj: str, elem: str, target_a: int,
    res_z: int, res_a: int, state: str,
) -> str:
    """Relative SVG path from docs/elements/ to plots/."""
    state_suffix = f"-{state}" if state else ""
    filename = f"{target_a}{elem}_{proj}_{res_z}-{_sym(res_z)}-{res_a}{state_suffix}.svg"
    return f"../../plots/{proj}/{elem}/{target_a}/{filename}"


def generate_element_page(
    element: str,
    projectiles: list[str],
    output_dir: Path,
) -> Path | None:
    """Generate a markdown page for one element across all projectiles."""
    z = _SYMBOL_TO_Z.get(element)
    if z is None:
        return None

    name = ELEMENT_NAMES.get(z, element)
    lines: list[str] = [
        f"# {name} ({element}, Z={z})",
        "",
    ]

    total_plots = 0

    for proj in projectiles:
        tendl_path = ROOT / "tendl-2024" / "xs" / f"{proj}_{element}.parquet"
        if not tendl_path.exists():
            continue

        df = pl.read_parquet(tendl_path)
        reactions = (
            df.select("target_A", "residual_Z", "residual_A", "state")
            .unique()
            .sort("target_A", "residual_Z", "residual_A", "state")
        )

        if reactions.is_empty():
            continue

        # Count EXFOR datasets per reaction
        exfor_path = ROOT / "exfor" / f"{proj}_{element}.parquet"
        exfor_df = pl.read_parquet(exfor_path) if exfor_path.exists() else None

        proj_label = PROJECTILE_LABELS.get(proj, proj)
        lines.append(f"## {proj_label} reactions")
        lines.append("")

        # Group by target isotope
        target_masses = reactions["target_A"].unique().sort().to_list()

        for target_a in target_masses:
            target_reactions = reactions.filter(pl.col("target_A") == target_a)
            lines.append(f"### <sup>{target_a}</sup>{element}")
            lines.append("")

            for row in target_reactions.iter_rows(named=True):
                res_z = row["residual_Z"]
                res_a = row["residual_A"]
                state = row["state"]

                svg_rel = _svg_path(proj, element, target_a, res_z, res_a, state)
                svg_abs = output_dir / svg_rel.replace("../../", "")

                reaction = _reaction_str(target_a, element, proj, res_z, res_a, state)

                # Count EXFOR points
                exfor_count = 0
                if exfor_df is not None:
                    exfor_count = len(exfor_df.filter(
                        (pl.col("target_A").is_in([target_a, 0]))
                        & (pl.col("residual_Z") == res_z)
                        & (pl.col("residual_A") == res_a)
                        & (pl.col("state") == state)
                    ))

                if svg_abs.exists():
                    exfor_badge = f" ({exfor_count} EXFOR pts)" if exfor_count > 0 else ""
                    lines.append(f"#### {reaction}{exfor_badge}")
                    lines.append("")
                    lines.append(f"![{reaction}]({svg_rel})")
                    lines.append("")
                    total_plots += 1

        lines.append("")

    if total_plots == 0:
        return None

    # Write
    docs_dir = output_dir / "docs" / "elements"
    docs_dir.mkdir(parents=True, exist_ok=True)
    page_path = docs_dir / f"{element}.md"
    page_path.write_text("\n".join(lines))
    logger.info("Wrote %s (%d plots)", page_path, total_plots)
    return page_path


def generate_index(
    elements: list[str],
    output_dir: Path,
) -> None:
    """Generate the index page with element listing."""
    lines = [
        "# nucl-parquet — Cross-Section Catalog",
        "",
        "Comparison of evaluated nuclear data (TENDL-2024) with EXFOR experimental measurements.",
        "",
        "## Elements",
        "",
        "| Z | Element | Symbol | Page |",
        "|---|---------|--------|------|",
    ]

    for elem in sorted(elements, key=lambda e: _SYMBOL_TO_Z.get(e, 999)):
        z = _SYMBOL_TO_Z.get(elem, 0)
        name = ELEMENT_NAMES.get(z, elem)
        lines.append(f"| {z} | {name} | {elem} | [{elem}](elements/{elem}.md) |")

    lines.extend([
        "",
        "---",
        "",
        "Data sources: [TENDL-2024](https://tendl.web.psi.ch/tendl_2024/tendl2024.html), "
        "[EXFOR](https://www-nds.iaea.org/exfor/)",
        "",
        "Generated by [nucl-parquet](https://github.com/eXoma/nucl-parquet)",
    ])

    index_path = output_dir / "docs" / "index.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines))
    logger.info("Wrote %s", index_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown cross-section catalog.",
    )
    parser.add_argument("--projectile", choices=["p", "d", "t", "h", "a"])
    parser.add_argument("--element", help="Element symbol")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Generate plots before catalog")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()

    projectiles = [args.projectile] if args.projectile else ["p", "d", "t", "h", "a"]

    # Optionally generate plots first
    if args.plot:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "plot_xs", ROOT / "scripts" / "plot_xs.py",
        )
        plot_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot_mod)

        if args.element:
            for proj in projectiles:
                plot_mod.plot_element(proj, args.element, args.output)
        elif args.all:
            for proj in projectiles:
                for elem in plot_mod.get_tendl_elements(proj):
                    plot_mod.plot_element(proj, elem, args.output)

    # Generate element pages
    if args.element:
        elements = [args.element]
    elif args.all:
        # Find all elements across all projectiles
        seen: set[str] = set()
        xs_dir = ROOT / "tendl-2024" / "xs"
        for f in xs_dir.glob("*.parquet"):
            elem = f.stem.split("_", 1)[1]
            if not elem.startswith("Z"):
                seen.add(elem)
        elements = sorted(seen)
    else:
        parser.error("Specify --element or --all")
        return

    generated = []
    for elem in elements:
        path = generate_element_page(elem, projectiles, args.output)
        if path:
            generated.append(elem)

    generate_index(generated, args.output)
    logger.info("Done. %d element pages.", len(generated))


if __name__ == "__main__":
    main()
