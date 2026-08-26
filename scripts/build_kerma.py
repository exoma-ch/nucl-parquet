"""Compute neutron KERMA coefficients from evaluated cross-section data.

KERMA (Kinetic Energy Released in MAtter) gives the energy transferred to
charged particles per unit neutron fluence per target atom. It is the neutron
analogue of stopping power for charged particles.

Computes KERMA from our existing XS parquet files + Q-values from nuclear
masses. For each element:

    K(E) = K_elastic(E) + K_capture(E) + Σ_charged K_j(E)

where:
    K_elastic = σ_el(E) × E × 2A/(A+1)²               [recoil]
    K_capture = σ_γ(E)  × E × A/(A+1)²/(2A) ≈ 0       [negligible]
    K_charged = σ_j(E)  × (|Q_j| + E × M_heavy/M_total) [charged products]

Elemental KERMA is additive for compounds (Bragg rule):
    K_compound(E) = Σᵢ wᵢ · Kᵢ(E)

Output: data/meta/kerma/{Element}.parquet
Schema: Z int32, A int32, energy_MeV f64, kerma_eV_barn f64

Usage:
    uv run python scripts/build_kerma.py
    uv run python scripts/build_kerma.py --library endfb-8.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# AMU to MeV/c²
AMU = 931.494

# Neutron mass in AMU
M_N = 1.008665

ELEMENT_SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
]
SYMBOL_TO_Z = {sym: z for z, sym in enumerate(ELEMENT_SYMBOLS, start=1)}


def elastic_kerma_fraction(A: int) -> float:
    """Average fraction of neutron energy transferred to recoiling nucleus.

    For isotropic CM elastic scattering: <E_recoil>/E = 2A/(A+1)².
    """
    return 2.0 * A / (A + 1.0) ** 2


def charged_product_energy(
    E_n: np.ndarray,
    target_A: int,
    residual_A: int,
    emitted_A: int,
    Q_MeV: float,
) -> np.ndarray:
    """Estimate kinetic energy available to charged products [MeV].

    Two-body kinematics approximation: the charged product receives
    the Q-value plus a fraction of the incident energy proportional to
    the heavy fragment mass ratio.

    Returns energy per reaction in MeV, clipped to 0 for sub-threshold.
    """
    # fraction of CM energy going to the lighter (emitted) particle
    if residual_A <= 0:
        return np.zeros_like(E_n)
    f = residual_A / (emitted_A + residual_A)
    E_available = Q_MeV + E_n * target_A / (target_A + 1.0)
    return np.maximum(E_available * f, 0.0)


# Reaction types that produce charged particles
# (residual_Z - target_Z) tells us the emitted particle
# (n,p): emits proton (Z=1, A=1)
# (n,d): emits deuteron (Z=1, A=2)
# (n,t): emits triton (Z=1, A=3)
# (n,α): emits alpha (Z=2, A=4)
# (n,2n): no charged product — skip
# (n,γ): capture, negligible charged KERMA

# Q-values from mass excess tables (approximate, for most abundant isotopes)
# We'll compute Q from the mass excess data in our decay/abundances tables


def compute_element_kerma(
    data_dir: Path,
    element_sym: str,
    library: str = "endfb-8.1",
) -> pl.DataFrame | None:
    """Compute KERMA for one element from its XS parquet file.

    Returns DataFrame with (Z, A, energy_MeV, kerma_eV_barn) or None.
    """
    target_Z = SYMBOL_TO_Z.get(element_sym)
    if target_Z is None:
        return None

    xs_path = data_dir / library / "xs" / f"n_{element_sym}.parquet"
    if not xs_path.exists():
        return None

    df = pl.read_parquet(xs_path)
    if df.is_empty():
        return None

    results = []

    # KERMA is computed per residual, from the Q-value implied by
    # (target -> residual). Since #347 these tables also carry `kind='channel'`
    # rows, and the four that name no residual — total, elastic, inelastic,
    # fission — arrive with `residual_Z = residual_A = NULL`. Those have no
    # (delta_Z, delta_A) to derive a Q-value from: `target_Z - res_Z` below is
    # `int - None` and raises. Take the production sums, which is what this
    # calculation always meant.
    if "kind" in df.columns:
        df = df.filter(pl.col("kind") == "production")
    df = df.filter(pl.col("residual_Z").is_not_null() & pl.col("residual_A").is_not_null())

    for target_A in df["target_A"].unique().sort().to_list():
        iso_df = df.filter(pl.col("target_A") == target_A)

        # Get all unique reactions for this isotope
        reactions = iso_df.select("residual_Z", "residual_A").unique().to_dicts()

        # Collect energy grids from all reactions, build a common grid
        all_energies = iso_df["energy_MeV"].unique().sort().to_numpy()

        kerma_total = np.zeros(len(all_energies))

        for rxn in reactions:
            res_Z = rxn["residual_Z"]
            res_A = rxn["residual_A"]

            rxn_df = iso_df.filter((pl.col("residual_Z") == res_Z) & (pl.col("residual_A") == res_A)).sort("energy_MeV")

            E = rxn_df["energy_MeV"].to_numpy()
            xs_mb = rxn_df["xs_mb"].to_numpy()

            # Interpolate to common grid
            xs_interp = np.interp(all_energies, E, xs_mb, left=0, right=0)
            # Convert mb to barns for KERMA units
            xs_b = xs_interp * 1e-3

            # Determine reaction type from (target_Z, target_A) → (res_Z, res_A)
            delta_Z = target_Z + 0 - res_Z  # neutron has Z=0
            delta_A = target_A + 1 - res_A  # neutron has A=1

            if delta_Z == 0 and delta_A == 0:
                # (n,γ) capture: residual = (Z, A+1)
                # Negligible KERMA — recoil only
                f_recoil = elastic_kerma_fraction(target_A)
                kerma_total += xs_b * all_energies * f_recoil * 1e6  # MeV → eV

            elif delta_Z == 0 and delta_A == 1 and res_Z == target_Z:
                # Elastic or (n,n') inelastic
                # Both have nuclear recoil as primary KERMA mechanism
                f_recoil = elastic_kerma_fraction(target_A)
                kerma_total += xs_b * all_energies * f_recoil * 1e6

            elif delta_Z == 1 and delta_A == 1:
                # (n,p): emits proton
                Q = _estimate_Q(target_Z, target_A, res_Z, res_A, 1, 1)
                E_charged = charged_product_energy(all_energies, target_A, res_A, 1, Q)
                kerma_total += xs_b * E_charged * 1e6

            elif delta_Z == 1 and delta_A == 2:
                # (n,d): emits deuteron
                Q = _estimate_Q(target_Z, target_A, res_Z, res_A, 1, 2)
                E_charged = charged_product_energy(all_energies, target_A, res_A, 2, Q)
                kerma_total += xs_b * E_charged * 1e6

            elif delta_Z == 1 and delta_A == 3:
                # (n,t): emits triton
                Q = _estimate_Q(target_Z, target_A, res_Z, res_A, 1, 3)
                E_charged = charged_product_energy(all_energies, target_A, res_A, 3, Q)
                kerma_total += xs_b * E_charged * 1e6

            elif delta_Z == 2 and delta_A == 4:
                # (n,α): emits alpha
                Q = _estimate_Q(target_Z, target_A, res_Z, res_A, 2, 4)
                E_charged = charged_product_energy(all_energies, target_A, res_A, 4, Q)
                kerma_total += xs_b * E_charged * 1e6

            elif delta_Z == 2 and delta_A == 3:
                # (n,³He): emits helion
                Q = _estimate_Q(target_Z, target_A, res_Z, res_A, 2, 3)
                E_charged = charged_product_energy(all_energies, target_A, res_A, 3, Q)
                kerma_total += xs_b * E_charged * 1e6

            elif delta_Z == 0 and delta_A == 2:
                # (n,2n): no charged products — only neutrons out
                # Recoil KERMA only
                f_recoil = elastic_kerma_fraction(target_A)
                kerma_total += xs_b * all_energies * f_recoil * 1e6

            # else: complex multi-particle emission — skip (minor contribution)

        # Filter out zero KERMA points
        mask = kerma_total > 0
        if not np.any(mask):
            continue

        n = int(np.sum(mask))
        results.append(
            pl.DataFrame(
                {
                    "Z": pl.Series([target_Z] * n, dtype=pl.Int32),
                    "A": pl.Series([target_A] * n, dtype=pl.Int32),
                    "energy_MeV": pl.Series(all_energies[mask], dtype=pl.Float64),
                    "kerma_eV_barn": pl.Series(kerma_total[mask], dtype=pl.Float64),
                }
            )
        )

    if not results:
        return None

    return pl.concat(results).sort("A", "energy_MeV")


def _estimate_Q(
    target_Z: int,
    target_A: int,
    residual_Z: int,
    residual_A: int,
    emitted_Z: int,
    emitted_A: int,
) -> float:
    """Estimate Q-value [MeV] from semi-empirical mass formula.

    Q = [M(target) + M(n) - M(residual) - M(emitted)] × c²

    Uses the Weizsäcker mass formula for binding energies.
    """

    def binding_energy(Z: int, A: int) -> float:
        """Weizsäcker semi-empirical binding energy [MeV]."""
        if A <= 0 or Z < 0:
            return 0.0
        N = A - Z
        a_v = 15.56
        a_s = 17.23
        a_c = 0.697
        a_a = 23.285
        if A % 2 != 0:
            delta = 0.0
        elif Z % 2 == 0:
            delta = 12.0 / A**0.5
        else:
            delta = -12.0 / A**0.5

        BE = a_v * A - a_s * A ** (2.0 / 3.0) - a_c * Z * (Z - 1) / A ** (1.0 / 3.0) - a_a * (N - Z) ** 2 / A + delta
        return max(BE, 0.0)

    # Q = BE(residual) + BE(emitted) - BE(target) - BE(neutron)
    # neutron BE = 0 (free neutron)
    BE_target = binding_energy(target_Z, target_A)
    BE_residual = binding_energy(residual_Z, residual_A)
    BE_emitted = binding_energy(emitted_Z, emitted_A)

    return BE_residual + BE_emitted - BE_target


def build(
    data_dir: Path | None = None,
    library: str = "endfb-8.1",
) -> None:
    if data_dir is None:
        data_dir = DATA_DIR

    out_dir = data_dir / "meta" / "kerma"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = data_dir / "catalog.json"
    if not catalog_path.exists():
        logger.error("catalog.json not found at %s", catalog_path)
        return

    catalog = json.loads(catalog_path.read_text())
    lib_info = catalog.get("libraries", {}).get(library)
    if lib_info is None:
        logger.error("Library %s not in catalog", library)
        return

    if "n" not in lib_info.get("projectiles", []):
        logger.error("Library %s has no neutron data", library)
        return

    xs_dir = data_dir / lib_info["path"]
    neutron_files = sorted(xs_dir.glob("n_*.parquet"))
    logger.info("Computing KERMA from %s: %d element files", library, len(neutron_files))

    total_rows = 0
    total_elements = 0

    for pq_path in neutron_files:
        sym = pq_path.stem.split("_", 1)[1]  # n_Cu → Cu

        df = compute_element_kerma(data_dir, sym, library)
        if df is None or df.is_empty():
            continue

        out_path = out_dir / f"{sym}.parquet"
        df.write_parquet(out_path, compression="zstd")
        total_elements += 1
        total_rows += len(df)
        logger.info("  %s: %d rows", sym, len(df))

    logger.info("Done: %d elements, %d total KERMA data points", total_elements, total_rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it (#363).

    The parser used to be constructed inside `if __name__ == "__main__"`, so it
    was unreachable by anything importing this module — including a test asking
    where the script writes.
    """
    parser = argparse.ArgumentParser(description="Compute neutron KERMA coefficients")
    parser.add_argument("--library", default="endfb-8.1", help="XS library to use")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Data directory (default: {DATA_DIR.name}/)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build(data_dir=args.data_dir, library=args.library)


if __name__ == "__main__":
    main()
