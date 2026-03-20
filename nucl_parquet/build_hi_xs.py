"""Compute heavy-ion total reaction cross-sections using the Tripathi (1997) formula.

The Tripathi semi-empirical parameterization covers all projectile/target
combinations (Z=1-92) and is the standard formula in the medical physics and
cosmic-ray shielding communities.  It is isotope-dependent (requires A for both
projectile and target).

Reference:
    Tripathi, R.K., Cucinotta, F.A., Wilson, J.W. (1997).
    "Accurate universal parameterization of absorption cross sections."
    Nucl. Instrum. Methods B, 117, 347-349.
    doi:10.1016/S0168-583X(96)00331-X

Output schema  (hi-xs/xs/{proj}_{target}.parquet):
    proj_A       Int32   Projectile mass number
    target_Z     Int32   Target atomic number
    target_A     Int32   Target mass number (most-abundant stable isotope)
    energy_MeV   Float64 Total projectile kinetic energy (MeV)
    xs_mb        Float64 Total reaction cross-section (mb)

Usage:
    uv run python -m nucl_parquet.build_hi_xs
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Element data
# ---------------------------------------------------------------------------

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

# Most abundant stable isotope per element (A_ref[Z-1] = A for Z).
# Used as the representative target isotope; for radioactive elements a
# long-lived isotope is chosen.
MOST_ABUNDANT_A = [
    1,  4,  7,  9, 11, 12, 14, 16, 19, 20,   # Z 1-10
   23, 24, 27, 28, 31, 32, 35, 40, 39, 40,   # Z 11-20
   45, 48, 51, 52, 55, 56, 59, 58, 63, 64,   # Z 21-30
   69, 74, 75, 80, 79, 84, 85, 88, 89, 90,   # Z 31-40
   93, 98, 98, 102, 103, 106, 107, 114, 115, 120,  # Z 41-50
   121, 130, 127, 132, 133, 138, 139, 140, 141, 144,  # Z 51-60
   145, 152, 153, 158, 159, 164, 165, 166, 169, 174,  # Z 61-70
   175, 180, 181, 184, 187, 192, 193, 195, 197, 202,  # Z 71-80
   205, 208, 209, 210, 210, 222, 223, 226, 227, 232,  # Z 81-90
   231, 238,                                          # Z 91-92
]

# Projectile definitions: label → (symbol, proj_Z, proj_A)
PROJECTILES = {
    "c12":  ("C",   6, 12),
    "o16":  ("O",   8, 16),
    "ne20": ("Ne", 10, 20),
    "si28": ("Si", 14, 28),
    "ar40": ("Ar", 18, 40),
    "fe56": ("Fe", 26, 56),
    "he4":  ("He",  2,  4),
    "p":    ("H",   1,  1),
    "ca40": ("Ca", 20, 40),
    "ni58": ("Ni", 28, 58),
    "pb208":("Pb", 82, 208),
    "xe132":("Xe", 54, 132),
}

# Energy grid: 1–1000 MeV/u, 60 log-spaced points (stored as total MeV)
_N_ENERGIES = 60
_E_PER_U_MIN = 1.0     # MeV/u
_E_PER_U_MAX = 1000.0  # MeV/u

# ---------------------------------------------------------------------------
# Tripathi (1997) total reaction cross-section formula
# ---------------------------------------------------------------------------

def _radius(a: float) -> float:
    """Nuclear radius in fm, r = 1.29 A^(1/3)."""
    return 1.29 * a ** (1.0 / 3.0)


def tripathi_xs(z1: int, a1: int, z2: int, a2: int, e_mev_per_u: float) -> float:
    """Total reaction cross-section [mb] via Tripathi et al. (1997).

    Parameters
    ----------
    z1, a1 : int
        Projectile Z and A.
    z2, a2 : int
        Target Z and A.
    e_mev_per_u : float
        Kinetic energy *per nucleon* [MeV/u].

    Returns
    -------
    float
        Total reaction cross-section in mb.  Returns 0 if below Coulomb
        barrier or if the formula returns a non-physical value.
    """
    pi = math.pi
    onethird = 1.0 / 3.0
    fourthird = 4.0 / 3.0

    # Ensure projectile is the lighter nucleus (Tripathi convention)
    if a2 < a1:
        z1, a1, z2, a2 = z2, a2, z1, a1

    rp = _radius(float(a1))
    rt = _radius(float(a2))
    vp = fourthird * pi * rp ** 3
    vt = fourthird * pi * rt ** 3
    dens = 0.5 * ((a1 / vp) + (a2 / vt))

    const = 1.75 * dens / 8.824728e-2
    const1 = 0.0  # only used for alpha

    if a1 == 1:
        const = 2.05
    if z1 == 2 and a1 == 4:
        const1 = 2.77 - a2 * 8.0e-3 + (a2 * a2) * 1.8e-5
    if z1 == 3:
        const *= onethird

    t1 = 40.0
    if z1 == 0:
        if 11 <= a2 < 40:
            t1 = 30.0
        if z2 == 14:
            t1 = 35.0
        if z2 == 26:
            t1 = 30.0

    # e is energy per nucleon of projectile [MeV/u]
    e = e_mev_per_u

    # Relativistic kinematics
    mp = 938.0  # nucleon mass MeV/c²
    gcm = (a1 * (1.0 + e / mp) + a2) / math.sqrt(
        a1 ** 2 + a2 ** 2 + 2.0 * a1 * (e + mp) * a2 / mp
    )
    if gcm <= 1.0:
        return 0.0

    bcm = math.sqrt(1.0 - 1.0 / gcm ** 2)
    plab = a1 * math.sqrt(2.0 * mp * e + e * e)
    ecmp = gcm * (e + mp) * a1 - bcm * gcm * plab - a1 * mp
    ecmt = gcm * mp * a2 - a2 * mp
    ecm = ecmp + ecmt

    bigr = rp + rt + 1.2 * (a1 ** onethird + a2 ** onethird) / (ecm ** onethird)
    bigb = 1.44 * z1 * z2 / bigr

    # Special Coulomb-barrier corrections from Tripathi
    if z1 == 1 and a2 > 56:
        bigb *= 0.90
    if a1 > 56 and z2 == 1:
        bigb *= 0.90
    if a1 == 1 and a2 == 12:
        bigb *= 3.5
    if a1 == 1:
        if 13 <= a2 <= 16:
            bigb *= a2 / 7.0
        if z2 == 12:
            bigb *= 1.8
        if z2 == 14:
            bigb *= 1.4
        if z2 == 20:
            bigb *= 1.3
    if a1 == 1 and a2 < 4:
        bigb *= 21.0
    if a1 < 4 and a2 == 1:
        bigb *= 21.0
    if a1 == 1 and a2 == 4:
        bigb *= 27.0
    if a1 == 4 and a2 == 1:
        bigb *= 27.0
    if z1 == 0 or z2 == 0:
        bigb = 0.0

    # X_m factor (neutron only)
    xm = 1.0
    if z1 == 0:
        if a2 < 200.0:
            x1 = 2.83 - 3.1e-2 * a2 + 1.7e-4 * a2 * a2
            x1 = max(x1, 1.0)
            sl = 1.0
            if a2 == 12:
                sl = 1.6
            if a2 < 12:
                sl = 0.6
            xm = 1.0 - x1 * math.exp(-e / (sl * x1))
        else:
            xm = (1.0 - 0.3 * math.exp(-(e - 1.0) / 15.0)) * (1.0 - math.exp(-(e - 0.9)))

    if z1 == 2 and a1 == 4:
        const = const1 - 0.8 / (1.0 + math.exp((250.0 - e) / 75.0))

    expo = min((e - 20.0) / 10.0, 80.0)
    if z1 == 1 and a1 == 1:
        if a2 > 45:
            t1 = 40.0 + a2 / 3.0
        if a2 < 4:
            t1 = 55.0
        const = 2.05 - 0.05 / (1.0 + math.exp((250.0 - e) / 75.0))
        if a2 < 4:
            const = 1.7
        if z2 == 12:
            t1 = 40.0
            const = 2.05 - 3.0 / (1.0 + math.exp(expo))
        if z2 == 14:
            t1 = 40.0
            const = 2.05 - 1.75 / (1.0 + math.exp(expo))
        if z2 == 18:
            t1 = 40.0
            const = 2.05 - 2.0 / (1.0 + math.exp(expo))
        if z2 == 20:
            t1 = 40.0
            expo1 = min((e - 40.0) / 10.0, 80.0)
            const = 2.05 - 1.0 / (1.0 + math.exp(expo1))

    if z1 == 0 and a1 == 1:
        const = 2.0 * (0.134457 / dens)
        if 140 < a2 < 200:
            const -= 1.5 * (a2 - 2.0 * z2) / a2
        if a2 < 60:
            const -= 1.5 * (a2 - 2.0 * z2) / a2
        if a2 <= 40:
            const += 0.25 / (1.0 + math.exp(-(170.0 - e) / 100.0))
        if z2 > 82:
            const -= float(z2) / (a2 - z2)
        if z2 >= 82:
            const -= 2.0 / (1.0 + math.exp(expo))
        if 10 <= z2 <= 20:
            const -= 1.0 / (1.0 + math.exp(expo))

    ce = const * (1.0 - math.exp(-e / t1)) - 0.292 * math.exp(-e / 792.0) * math.cos(
        0.229 * e ** 0.453
    )
    term1 = (a2 * a1) ** onethird / (a2 ** onethird + a1 ** onethird)
    delta = 1.615 * term1 - 0.873 * ce
    delta += 0.140 * term1 / ecm ** onethird
    delta += 0.794 * (a2 - 2.0 * z2) * z1 / (a2 * a1)
    delta = -delta

    twxsec = 10.0 * pi * 1.26 * 1.26 * (0.873 * a1 ** onethird + 0.873 * a2 ** onethird - delta) ** 2
    xs = twxsec * (1.0 - bigb / ecm) * xm
    return max(xs, 0.0)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def build(data_dir: Path) -> None:
    import numpy as np

    out_dir = data_dir / "hi-xs" / "xs"
    out_dir.mkdir(parents=True, exist_ok=True)

    energies_per_u = np.geomspace(_E_PER_U_MIN, _E_PER_U_MAX, _N_ENERGIES)

    for proj_key, (proj_sym, proj_Z, proj_A) in PROJECTILES.items():
        energies_MeV = energies_per_u * proj_A
        print(f"\n=== {proj_key.upper()} (Z={proj_Z}, A={proj_A}) ===")
        rows: list[dict] = []

        for target_Z in range(1, 93):
            target_A = MOST_ABUNDANT_A[target_Z - 1]

            for e_mev, e_per_u in zip(energies_MeV, energies_per_u):
                xs = tripathi_xs(proj_Z, proj_A, target_Z, target_A, float(e_per_u))
                if xs > 0:
                    rows.append({
                        "target_Z":   target_Z,
                        "target_A":   target_A,
                        "energy_MeV": float(e_mev),
                        "xs_mb":      xs,
                    })

        if not rows:
            print(f"  No data, skipping")
            continue

        df = pl.DataFrame({
            "target_Z":   pl.Series([r["target_Z"]   for r in rows], dtype=pl.Int32),
            "target_A":   pl.Series([r["target_A"]   for r in rows], dtype=pl.Int32),
            "energy_MeV": pl.Series([r["energy_MeV"] for r in rows], dtype=pl.Float64),
            "xs_mb":      pl.Series([r["xs_mb"]      for r in rows], dtype=pl.Float64),
        }).sort("target_Z", "target_A", "energy_MeV")

        # Write per-target parquets (matches light-ion layout)
        for target_Z in sorted(df["target_Z"].unique().to_list()):
            target_sym = Z_TO_SYMBOL[target_Z]
            chunk = df.filter(pl.col("target_Z") == target_Z)
            out_path = out_dir / f"{proj_key}_{target_sym}.parquet"
            chunk.write_parquet(out_path, compression="zstd")

        print(f"  → {len(df):,} rows across {df['target_Z'].n_unique()} targets")

    _update_catalog(data_dir)
    print("\nDone.")


def _update_catalog(data_dir: Path) -> None:
    catalog_path = data_dir / "catalog.json"
    catalog = json.loads(catalog_path.read_text())

    if "hi-xs" not in catalog["libraries"]:
        catalog["libraries"]["hi-xs"] = {
            "name": "HI-XS (Tripathi 1997)",
            "description": (
                "Heavy-ion total reaction cross-sections computed with the "
                "Tripathi (1997) semi-empirical parameterization.  "
                "Covers all projectile/target combinations Z=1-92."
            ),
            "source_url": "https://doi.org/10.1016/S0168-583X(96)00331-X",
            "projectiles": sorted(PROJECTILES.keys()),
            "data_type": "total_reaction_cross_sections",
            "version": "tripathi-1997",
            "path": "hi-xs/xs/",
        }
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
        print("Updated catalog.json with hi-xs library")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HI total reaction XS via Tripathi (1997)"
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="nucl-parquet repo root (default: current directory)",
    )
    args = parser.parse_args()
    build(Path(args.data_dir))
