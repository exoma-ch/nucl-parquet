"""Target material feasibility assessment for isotope production routes.

Provides natural abundance lookup and practical availability classification
for target isotopes used in cyclotron-based radionuclide production.
"""

from __future__ import annotations


def load_abundance_map(db) -> dict[tuple[int, int], float]:
    """Build (Z, A) -> abundance % map from ground_states table."""
    rows = db.execute("""
        SELECT Z, A, abundance FROM ground_states
        WHERE abundance IS NOT NULL AND abundance > 0
    """).fetchall()
    return {(int(Z), int(A)): float(ab) for Z, A, ab in rows}


def load_element_z(db) -> dict[str, int]:
    """Build symbol -> Z map from elements table."""
    return {sym: int(Z) for Z, sym in db.execute("SELECT Z, symbol FROM elements").fetchall()}


# Noble gases — gaseous targets require special handling (gas cells, ice targets)
_NOBLE_GASES = {2, 10, 18, 36, 54, 86}  # He, Ne, Ar, Kr, Xe, Rn

# Elements that are toxic, volatile, or otherwise difficult to handle
_DIFFICULT_ELEMENTS = {
    80: "toxic (Hg)",
    84: "radioactive",
    85: "no stable isotopes",
    86: "noble gas, radioactive",
    87: "no stable isotopes",
    88: "radioactive",
    89: "radioactive",
    90: "radioactive",
    91: "radioactive",
    92: "radioactive, regulated",
}

# Practically available enriched isotopes (well-established supply chain)
# These are commonly enriched and commercially available
_COMMONLY_ENRICHED = {
    # Light elements
    (8, 18),   # O-18 (for F-18 production) — very well established
    (7, 15),   # N-15
    (6, 13),   # C-13
    # Transition metals — EMIS enrichment well established
    (26, 54), (26, 56), (26, 57), (26, 58),  # Fe isotopes
    (28, 58), (28, 60), (28, 61), (28, 62), (28, 64),  # Ni isotopes
    (29, 63), (29, 65),  # Cu isotopes
    (30, 64), (30, 66), (30, 67), (30, 68), (30, 70),  # Zn isotopes
    (42, 92), (42, 94), (42, 95), (42, 96), (42, 97), (42, 98), (42, 100),  # Mo
    (48, 106), (48, 108), (48, 110), (48, 111), (48, 112), (48, 113), (48, 114), (48, 116),  # Cd
    (50, 112), (50, 114), (50, 115), (50, 116), (50, 117), (50, 118), (50, 119), (50, 120), (50, 122), (50, 124),  # Sn
    (52, 120), (52, 122), (52, 123), (52, 124), (52, 125), (52, 126), (52, 128), (52, 130),  # Te
    (22, 46), (22, 47), (22, 48), (22, 49), (22, 50),  # Ti
    (20, 40), (20, 42), (20, 43), (20, 44), (20, 46), (20, 48),  # Ca
    (24, 50), (24, 52), (24, 53), (24, 54),  # Cr
    # Noble gases — enrichable via centrifuge
    (36, 78), (36, 80), (36, 82), (36, 83), (36, 84), (36, 86),  # Kr
    (54, 124), (54, 126), (54, 128), (54, 129), (54, 130), (54, 131), (54, 132), (54, 134), (54, 136),  # Xe
    # Lanthanides — EMIS available
    (64, 152), (64, 154), (64, 155), (64, 156), (64, 157), (64, 158), (64, 160),  # Gd
    (66, 156), (66, 158), (66, 160), (66, 161), (66, 162), (66, 163), (66, 164),  # Dy
    (68, 162), (68, 164), (68, 166), (68, 167), (68, 168), (68, 170),  # Er
    (70, 168), (70, 170), (70, 171), (70, 172), (70, 173), (70, 174), (70, 176),  # Yb
    (72, 174), (72, 176), (72, 177), (72, 178), (72, 179), (72, 180),  # Hf
}


# Element names for Wikipedia links
_ELEMENT_NAMES: dict[int, str] = {
    1: "Hydrogen", 2: "Helium", 3: "Lithium", 4: "Beryllium", 5: "Boron",
    6: "Carbon", 7: "Nitrogen", 8: "Oxygen", 9: "Fluorine", 10: "Neon",
    11: "Sodium", 12: "Magnesium", 13: "Aluminium", 14: "Silicon", 15: "Phosphorus",
    16: "Sulfur", 17: "Chlorine", 18: "Argon", 19: "Potassium", 20: "Calcium",
    21: "Scandium", 22: "Titanium", 23: "Vanadium", 24: "Chromium", 25: "Manganese",
    26: "Iron", 27: "Cobalt", 28: "Nickel", 29: "Copper", 30: "Zinc",
    31: "Gallium", 32: "Germanium", 33: "Arsenic", 34: "Selenium", 35: "Bromine",
    36: "Krypton", 37: "Rubidium", 38: "Strontium", 39: "Yttrium", 40: "Zirconium",
    41: "Niobium", 42: "Molybdenum", 43: "Technetium", 44: "Ruthenium", 45: "Rhodium",
    46: "Palladium", 47: "Silver", 48: "Cadmium", 49: "Indium", 50: "Tin",
    51: "Antimony", 52: "Tellurium", 53: "Iodine", 54: "Xenon", 55: "Caesium",
    56: "Barium", 57: "Lanthanum", 58: "Cerium", 59: "Praseodymium", 60: "Neodymium",
    61: "Promethium", 62: "Samarium", 63: "Europium", 64: "Gadolinium", 65: "Terbium",
    66: "Dysprosium", 67: "Holmium", 68: "Erbium", 69: "Thulium", 70: "Ytterbium",
    71: "Lutetium", 72: "Hafnium", 73: "Tantalum", 74: "Tungsten", 75: "Rhenium",
    76: "Osmium", 77: "Iridium", 78: "Platinum", 79: "Gold", 80: "Mercury",
    81: "Thallium", 82: "Lead", 83: "Bismuth", 84: "Polonium", 85: "Astatine",
    86: "Radon", 87: "Francium", 88: "Radium", 89: "Actinium", 90: "Thorium",
    91: "Protactinium", 92: "Uranium",
}


def wikipedia_url(target_Z: int) -> str:
    """Return Wikipedia URL for the target element."""
    name = _ELEMENT_NAMES.get(target_Z, "")
    if name:
        return f"https://en.wikipedia.org/wiki/{name}"
    return ""


def assess_target(
    target_Z: int,
    target_A: int,
    abundance_pct: float | None,
    element_symbol: str = "",
) -> dict:
    """Assess practical feasibility of a target isotope.

    Returns dict with:
        abundance_pct: natural abundance (None if radioactive)
        availability: one of "natural", "enriched (standard)", "enriched (costly)",
                      "enriched (very rare)", "radioactive target", "gas target", ...
        feasibility: "excellent", "good", "moderate", "difficult", "impractical"
        note: brief explanation
    """
    result = {
        "abundance_pct": round(abundance_pct, 4) if abundance_pct else None,
        "availability": "",
        "feasibility": "",
        "note": "",
    }

    # Radioactive target (no natural abundance)
    if abundance_pct is None or abundance_pct <= 0:
        result["availability"] = "radioactive target"
        result["feasibility"] = "impractical"
        result["note"] = "no stable isotope — requires secondary production"
        return result

    # Check for difficult elements
    if target_Z in _DIFFICULT_ELEMENTS:
        result["availability"] = _DIFFICULT_ELEMENTS[target_Z]
        result["feasibility"] = "impractical"
        result["note"] = _DIFFICULT_ELEMENTS[target_Z]
        return result

    is_gas = target_Z in _NOBLE_GASES
    is_commonly_enriched = (target_Z, target_A) in _COMMONLY_ENRICHED

    if abundance_pct >= 99.0:
        # Mono-isotopic or near-mono — natural element works
        result["availability"] = "natural (mono-isotopic)"
        result["feasibility"] = "excellent"
        result["note"] = f"{abundance_pct:.1f}% — use natural element"
    elif abundance_pct >= 50.0:
        result["availability"] = "natural (dominant)"
        result["feasibility"] = "excellent"
        result["note"] = f"{abundance_pct:.1f}% — natural element viable, enrichment optional"
    elif abundance_pct >= 10.0:
        if is_commonly_enriched:
            result["availability"] = "enriched (standard)"
            result["feasibility"] = "good"
            result["note"] = f"{abundance_pct:.2f}% nat. — commercially enriched"
        else:
            result["availability"] = "enriched (available)"
            result["feasibility"] = "good"
            result["note"] = f"{abundance_pct:.2f}% nat. — enrichment straightforward"
        if is_gas:
            result["note"] += " (gas target)"
    elif abundance_pct >= 1.0:
        if is_commonly_enriched:
            result["availability"] = "enriched (standard)"
            result["feasibility"] = "good"
            result["note"] = f"{abundance_pct:.2f}% nat. — commercially enriched"
        else:
            result["availability"] = "enriched (moderate cost)"
            result["feasibility"] = "moderate"
            result["note"] = f"{abundance_pct:.2f}% nat. — enrichment needed"
        if is_gas:
            result["note"] += " (gas target)"
    elif abundance_pct >= 0.05:
        if is_commonly_enriched:
            result["availability"] = "enriched (costly)"
            result["feasibility"] = "moderate"
            result["note"] = f"{abundance_pct:.3f}% nat. — available but expensive"
        else:
            result["availability"] = "enriched (very costly)"
            result["feasibility"] = "difficult"
            result["note"] = f"{abundance_pct:.3f}% nat. — calutron/AVLIS, very expensive"
        if is_gas:
            result["note"] += " (gas target)"
    else:
        result["availability"] = "ultra-rare"
        result["feasibility"] = "impractical"
        result["note"] = f"{abundance_pct:.4f}% nat. — not practically enrichable"

    return result
