# /// script
# dependencies = [
#   "nucl-parquet @ /Users/larsgerchow/Projects/eXoma/nucl-parquet",
#   "scipy",
#   "matplotlib",
#   "pandas",
#   "numpy",
# ]
# ///
"""Material constants for PET scanner efficiency analysis.

Loads LYSO and water attenuation data from the nucl-parquet library and computes:
- Linear attenuation coefficients at 511 keV and 1157 keV
- EPDL97 process fractions (Compton / photoelectric) for LYSO
- Klein-Nishina fraction of Compton events depositing > E_thresh
- Effective 1157 keV prompt-gamma usable fraction

Run with:
    uv run estimates/pet_efficiency/material.py
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad

import nucl_parquet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LYSO_COMPOSITION = {
    71: ("Lu", 0.7145),  # (Z, mass_fraction)
    39: ("Y",  0.0403),
    14: ("Si", 0.0637),
     8: ("O",  0.1815),
}
LYSO_DENSITY_G_CM3 = 7.1  # g/cm³

ENERGIES_KEV = [511.0, 1157.0]
ENERGIES_MEV = [e / 1000.0 for e in ENERGIES_KEV]

# Reference values for validation (back-of-envelope)
REF = {
    "mu_lyso_511":   0.86,   # cm⁻¹
    "mu_lyso_1157":  0.46,   # cm⁻¹
    "P_KN_600_1157": 0.475,  # fraction
}
VALIDATION_TOL = 0.15  # 15 % tolerance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def loglog_interp_array(
    energies_tab: np.ndarray,
    xs_tab: np.ndarray,
    e_query: np.ndarray,
) -> np.ndarray:
    """Log-log interpolation over an array of query points."""
    return np.exp(np.interp(np.log(e_query), np.log(energies_tab), np.log(xs_tab)))


def loglog_interp(energies_tab: np.ndarray, xs_tab: np.ndarray, e_query: float) -> float:
    """Log-log interpolation at a single query point."""
    return float(loglog_interp_array(energies_tab, xs_tab, np.array([e_query]))[0])


def _fetch_xcom_element(db, z: int) -> tuple[np.ndarray, np.ndarray]:
    """Fetch XCOM μ/ρ table for element Z in 0.4–1.5 MeV range."""
    df = db.sql(
        "SELECT energy_MeV, mu_rho_cm2_g FROM xcom_elements "
        "WHERE Z = $z AND energy_MeV BETWEEN 0.4 AND 1.5 "
        "ORDER BY energy_MeV",
        params={"z": z},
    ).fetchdf()
    return df["energy_MeV"].to_numpy(), df["mu_rho_cm2_g"].to_numpy()


def _fetch_epdl_element(db, z: int) -> pd.DataFrame:
    """Fetch EPDL97 cross-section table for element Z in 0.4–1.5 MeV range."""
    return db.sql(
        "SELECT energy_MeV, process, xs_barns FROM epdl_photon_xs "
        "WHERE Z = $z AND process IN ('incoherent', 'photoelectric', 'total') "
        "AND energy_MeV BETWEEN 0.4 AND 1.5 "
        "ORDER BY energy_MeV",
        params={"z": z},
    ).fetchdf()


# ---------------------------------------------------------------------------
# 1. LYSO linear attenuation coefficients
# ---------------------------------------------------------------------------

ElemTables = dict[int, tuple[np.ndarray, np.ndarray]]


def fetch_xcom_tables(
    db,
    composition: dict[int, tuple[str, float]],
) -> ElemTables:
    """Fetch XCOM μ/ρ tables for all elements in a composition (one query per element).

    Returns:
        Dict Z -> (energy_MeV array, mu_rho_cm2_g array).
    """
    return {z: _fetch_xcom_element(db, z) for z in composition}


def compute_lyso_mu(
    energies_mev: list[float],
    composition: dict[int, tuple[str, float]],
    density: float,
    elem_tables: ElemTables,
) -> dict[float, float]:
    """Compute LYSO linear attenuation [cm⁻¹] at given energies via Bragg mixing.

    Args:
        energies_mev: List of photon energies [MeV] to evaluate.
        composition: Dict of Z -> (symbol, mass_fraction).
        density: Material density [g/cm³].
        elem_tables: Pre-fetched XCOM tables from fetch_xcom_tables().

    Returns:
        Dict mapping energy_MeV -> mu [cm⁻¹].
    """
    mu_rho: dict[float, float] = {e: 0.0 for e in energies_mev}
    for z, (sym, w) in composition.items():
        e_tab, mu_rho_tab = elem_tables[z]
        for e in energies_mev:
            mu_rho[e] += w * loglog_interp(e_tab, mu_rho_tab, e)
    return {e: v * density for e, v in mu_rho.items()}


def get_lyso_mu_curve(
    composition: dict[int, tuple[str, float]],
    density: float,
    elem_tables: ElemTables,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Return energy grid and LYSO μ curve for plotting, plus per-element contributions.

    Reuses pre-fetched elem_tables to avoid duplicate DB queries.

    Returns:
        (energy_MeV, mu_cm1_lyso, {Z: mu_rho_element}) on a common grid.
    """
    all_energies: set[float] = set()
    for e_tab, _ in elem_tables.values():
        all_energies.update(e_tab.tolist())
    e_grid = np.array(sorted(all_energies))

    elem_mu_rho: dict[int, np.ndarray] = {}
    mu_rho_lyso = np.zeros_like(e_grid)

    for z, (sym, w) in composition.items():
        e_tab, mu_rho_tab = elem_tables[z]
        mu_rho_z = loglog_interp_array(e_tab, mu_rho_tab, e_grid)
        elem_mu_rho[z] = mu_rho_z
        mu_rho_lyso += w * mu_rho_z

    return e_grid, mu_rho_lyso * density, elem_mu_rho


# ---------------------------------------------------------------------------
# 2. Water linear attenuation coefficients
# ---------------------------------------------------------------------------

def compute_water_mu(db, energies_mev: list[float]) -> dict[float, float]:
    """Compute water linear attenuation [cm⁻¹] at given energies.

    Uses μ/ρ from xcom_compounds with water density 1.0 g/cm³.

    Args:
        db: nucl-parquet DuckDB connection.
        energies_mev: List of photon energies [MeV].

    Returns:
        Dict mapping energy_MeV -> mu [cm⁻¹].
    """
    df = db.sql(
        "SELECT energy_MeV, mu_rho_cm2_g FROM xcom_compounds "
        "WHERE material = 'water' AND energy_MeV BETWEEN 0.4 AND 1.5 "
        "ORDER BY energy_MeV",
    ).fetchdf()

    e_tab = df["energy_MeV"].to_numpy()
    mu_rho_tab = df["mu_rho_cm2_g"].to_numpy()

    return {e: loglog_interp(e_tab, mu_rho_tab, e) for e in energies_mev}


# ---------------------------------------------------------------------------
# 3. EPDL97 process fractions for LYSO
# ---------------------------------------------------------------------------

def compute_process_fractions(
    db,
    energies_mev: list[float],
    composition: dict[int, tuple[str, float]],
) -> dict[float, dict[str, float]]:
    """Compute Compton/photoelectric fractions for LYSO via EPDL97.

    Fetches each element's cross-sections once, then interpolates at all energies.

    Args:
        db: nucl-parquet DuckDB connection.
        energies_mev: Energies [MeV] at which to evaluate.
        composition: Dict Z -> (symbol, mass_fraction).

    Returns:
        Dict energy_MeV -> {"f_compton": float, "f_pe": float}.
    """
    # Accumulate weighted cross-sections per energy
    sigma: dict[float, dict[str, float]] = {
        e: {"total": 0.0, "incoherent": 0.0, "photoelectric": 0.0}
        for e in energies_mev
    }

    for z, (sym, w) in composition.items():
        df = _fetch_epdl_element(db, z)
        for proc in ("total", "incoherent", "photoelectric"):
            proc_df = df[df["process"] == proc].sort_values("energy_MeV")
            e_tab  = proc_df["energy_MeV"].to_numpy()
            xs_tab = proc_df["xs_barns"].to_numpy()
            for e in energies_mev:
                sigma[e][proc] += w * loglog_interp(e_tab, xs_tab, e)

    return {
        e: {
            "f_compton": sigma[e]["incoherent"]   / sigma[e]["total"],
            "f_pe":      sigma[e]["photoelectric"] / sigma[e]["total"],
        }
        for e in energies_mev
    }


# ---------------------------------------------------------------------------
# 4. Klein-Nishina deposited-energy fraction
# ---------------------------------------------------------------------------

def _kn_dsigma_dOmega(cos_theta: float, alpha: float) -> float:
    """Klein-Nishina dσ/dΩ (unnormalized).

    Args:
        cos_theta: cos(θ) of Compton scatter angle.
        alpha: E_gamma / (m_e c²) = E_gamma [MeV] / 0.511 MeV.
    """
    r = 1.0 / (1.0 + alpha * (1.0 - cos_theta))  # E'/E
    return r * r * (r + 1.0 / r - (1.0 - cos_theta * cos_theta))


def _kn_integrand(theta: float, alpha: float) -> float:
    return _kn_dsigma_dOmega(np.cos(theta), alpha) * np.sin(theta)


def kn_fraction_above_threshold(
    e_gamma_mev: float,
    e_thresh_mev: float,
    _kn_denominator: float | None = None,
) -> float:
    """Fraction of Compton events depositing ΔE > e_thresh via Klein-Nishina.

    P_KN = ∫_{θ_min}^{π} (dσ/dΩ) sinθ dθ / ∫₀^{π} (dσ/dΩ) sinθ dθ

    Args:
        e_gamma_mev: Incident photon energy [MeV].
        e_thresh_mev: Minimum deposited energy threshold [MeV].
        _kn_denominator: Pre-computed total KN integral (avoids redundant quad
            calls when sweeping over many thresholds at fixed e_gamma_mev).

    Returns:
        Probability in [0, 1].  Returns 0.0 if threshold exceeds Compton edge.
    """
    alpha = e_gamma_mev / 0.511
    e_compton_edge = e_gamma_mev * (2.0 * alpha) / (1.0 + 2.0 * alpha)

    if e_thresh_mev >= e_compton_edge:
        return 0.0

    cos_theta_min = np.clip(1.0 - (e_gamma_mev / e_thresh_mev - 1.0) / alpha, -1.0, 1.0)
    theta_min = np.arccos(cos_theta_min)

    numerator, _   = quad(_kn_integrand, theta_min, np.pi, args=(alpha,))
    denominator = _kn_denominator
    if denominator is None:
        denominator, _ = quad(_kn_integrand, 0.0, np.pi, args=(alpha,))

    return numerator / denominator if denominator else 0.0


def kn_fraction_curve(
    e_gamma_mev: float,
    thresh_kev: np.ndarray,
) -> np.ndarray:
    """Evaluate P_KN(ΔE > E_thresh) across an array of thresholds.

    Computes the KN denominator once and reuses it across all thresholds.

    Args:
        e_gamma_mev: Incident photon energy [MeV].
        thresh_kev: Array of thresholds [keV].

    Returns:
        Array of fractions in [0, 1].
    """
    alpha = e_gamma_mev / 0.511
    denominator, _ = quad(_kn_integrand, 0.0, np.pi, args=(alpha,))
    return np.array([
        kn_fraction_above_threshold(e_gamma_mev, t / 1000.0, _kn_denominator=denominator)
        for t in thresh_kev
    ])


# ---------------------------------------------------------------------------
# 5. Effective usable fraction for 1157 keV
# ---------------------------------------------------------------------------

def effective_usable_fraction(f_pe: float, f_compton: float, p_kn: float) -> float:
    """Effective fraction of 1157 keV events usable above energy threshold.

    f_useful = f_PE + f_Compton × P_KN(ΔE > E_thresh)
    """
    return f_pe + f_compton * p_kn


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_results(mu_lyso: dict[float, float], p_kn_600_1157: float) -> None:
    """Print comparison table and warn if values differ >15% from reference."""
    checks = [
        ("μ_LYSO(511 keV) [cm⁻¹]",  "mu_lyso_511",   mu_lyso[0.511], "0.85"),
        ("μ_LYSO(1157 keV) [cm⁻¹]", "mu_lyso_1157",  mu_lyso[1.157], "0.45"),
        ("P_KN(ΔE>600 keV, 1157)",   "P_KN_600_1157", p_kn_600_1157,  "0.45–0.50"),
    ]

    print()
    print("=" * 60)
    print(f"{'':30s}{'Back-of-envelope':>16s}{'XCOM-derived':>12s}")
    print("-" * 60)
    for label, key, val, ref_str in checks:
        print(f"  {label:28s}  {ref_str:>14s}  {val:>10.4f}")
        if abs(val - REF[key]) / REF[key] > VALIDATION_TOL:
            warnings.warn(
                f"Validation: {label} = {val:.4f} deviates >15% from reference {REF[key]:.4f}",
                stacklevel=2,
            )
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mu_curve(
    e_grid: np.ndarray,
    mu_lyso: np.ndarray,
    elem_mu_rho: dict[int, np.ndarray],
    composition: dict[int, tuple[str, float]],
    output_path: str,
) -> None:
    """Log-log plot of LYSO μ vs energy with per-element contributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(e_grid * 1000, mu_lyso, "k-", linewidth=2, label="LYSO (Bragg mix)")

    for z, (sym, w) in composition.items():
        contribution = w * elem_mu_rho[z] * LYSO_DENSITY_G_CM3
        ax.loglog(e_grid * 1000, contribution, "--", linewidth=1,
                  label=f"{sym} (Z={z}) × {w:.4f}")

    ax.axvline(511,  color="gray", linestyle=":",  alpha=0.7, label="511 keV")
    ax.axvline(1157, color="gray", linestyle="-.", alpha=0.7, label="1157 keV")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Linear attenuation μ [cm⁻¹]")
    ax.set_title("LYSO linear attenuation coefficient (Bragg mixture)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_process_fractions(
    fractions: dict[float, dict[str, float]],
    output_path: str,
) -> None:
    """Stacked bar chart of PE/Compton/other fractions at 511 and 1157 keV."""
    labels   = ["511 keV", "1157 keV"]
    energies = [0.511, 1.157]

    f_pe      = np.array([fractions[e]["f_pe"]      for e in energies])
    f_compton = np.array([fractions[e]["f_compton"] for e in energies])
    f_other   = np.clip(1.0 - f_pe - f_compton, 0.0, None)

    x     = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, f_pe,      width, label="Photoelectric")
    ax.bar(x, f_compton, width, bottom=f_pe,            label="Compton (incoherent)")
    ax.bar(x, f_other,   width, bottom=f_pe + f_compton, label="Other / pair")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction")
    ax.set_title("LYSO photon interaction fractions (EPDL97)")
    ax.set_ylim(0, 1.05)
    ax.legend()

    for i, e in enumerate(energies):
        ax.text(i, f_pe[i] / 2, f"{f_pe[i]:.3f}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(i, f_pe[i] + f_compton[i] / 2, f"{f_compton[i]:.3f}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_kn_fraction(
    thresh_kev: np.ndarray,
    p_kn_1157: np.ndarray,
    p_kn_600: float,
    output_path: str,
) -> None:
    """P_KN(ΔE > E_thresh) vs E_thresh for E_gamma=1157 keV."""
    e_edge_511 = 511.0 * 2.0 / (1.0 + 2.0)  # alpha=1 → 340.7 keV

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresh_kev, p_kn_1157, "b-", linewidth=2,
            label="$P_{KN}(ΔE > E_{thresh})$, $E_γ = 1157$ keV")
    ax.axvline(600, color="red", linestyle="--", label="$E_{thresh} = 600$ keV")
    ax.axhline(p_kn_600, color="red", linestyle=":", alpha=0.6,
               label=f"$P_{{KN}}$ = {p_kn_600:.3f}")
    ax.axvline(e_edge_511, color="orange", linestyle="--",
               label=f"511 keV Compton edge ≈ {e_edge_511:.0f} keV")

    ax.set_xlabel("Energy threshold $E_{thresh}$ [keV]")
    ax.set_ylabel("$P_{KN}(ΔE > E_{thresh})$")
    ax.set_title("Klein-Nishina fraction above energy threshold ($E_γ = 1157$ keV)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(thresh_kev[0], thresh_kev[-1])
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute and report all material constants for PET scanner efficiency analysis."""
    os.makedirs("estimates/pet_efficiency", exist_ok=True)
    os.makedirs("estimates/results", exist_ok=True)

    print("Connecting to nucl-parquet...")
    db = nucl_parquet.connect("/Users/larsgerchow/Projects/eXoma/nucl-parquet")

    # Fetch XCOM tables once; reused by both compute_lyso_mu and get_lyso_mu_curve
    xcom_tables = fetch_xcom_tables(db, LYSO_COMPOSITION)

    print("Computing LYSO attenuation coefficients...")
    mu_lyso = compute_lyso_mu(ENERGIES_MEV, LYSO_COMPOSITION, LYSO_DENSITY_G_CM3, xcom_tables)
    print(f"  μ_LYSO(511 keV)  = {mu_lyso[0.511]:.4f} cm⁻¹")
    print(f"  μ_LYSO(1157 keV) = {mu_lyso[1.157]:.4f} cm⁻¹")

    print("Computing water attenuation coefficients...")
    mu_water = compute_water_mu(db, ENERGIES_MEV)
    print(f"  μ_water(511 keV)  = {mu_water[0.511]:.4f} cm⁻¹")
    print(f"  μ_water(1157 keV) = {mu_water[1.157]:.4f} cm⁻¹")

    print("Computing EPDL97 process fractions for LYSO...")
    fractions = compute_process_fractions(db, ENERGIES_MEV, LYSO_COMPOSITION)
    for e_mev in ENERGIES_MEV:
        e_kev = int(e_mev * 1000)
        f = fractions[e_mev]
        print(f"  {e_kev} keV: f_Compton = {f['f_compton']:.4f}, f_PE = {f['f_pe']:.4f}")

    print("Computing Klein-Nishina fractions...")
    thresh_kev = np.linspace(200, 800, 121)
    p_kn_1157  = kn_fraction_curve(1.157, thresh_kev)
    p_kn_600   = kn_fraction_above_threshold(1.157, 0.600)
    print(f"  P_KN(ΔE > 600 keV, E=1157 keV) = {p_kn_600:.4f}")

    # Verify 511 keV Compton edge is below 600 keV threshold
    e_edge_511 = 511.0 * 2.0 / (1.0 + 2.0)
    print(f"  511 keV Compton edge = {e_edge_511:.1f} keV  (< 600 keV → P_KN=0 for 511 keV)")
    assert kn_fraction_above_threshold(0.511, 0.600) == 0.0

    f_pe_1157      = fractions[1.157]["f_pe"]
    f_compton_1157 = fractions[1.157]["f_compton"]
    f_useful_1157  = effective_usable_fraction(f_pe_1157, f_compton_1157, p_kn_600)
    print(f"  f_useful_1157 (E_thresh=600 keV) = {f_useful_1157:.4f}")

    validate_results(mu_lyso, p_kn_600)

    print("Generating plots...")
    e_grid, mu_curve, elem_mu_rho = get_lyso_mu_curve(
        LYSO_COMPOSITION, LYSO_DENSITY_G_CM3, xcom_tables
    )
    plot_mu_curve(e_grid, mu_curve, elem_mu_rho, LYSO_COMPOSITION,
                  "estimates/results/material_mu.png")
    plot_process_fractions(fractions, "estimates/results/material_process_fractions.png")
    plot_kn_fraction(thresh_kev, p_kn_1157, p_kn_600,
                     "estimates/results/material_kn_fraction.png")
    print("  Saved: estimates/results/material_mu.png")
    print("  Saved: estimates/results/material_process_fractions.png")
    print("  Saved: estimates/results/material_kn_fraction.png")

    rows = []
    for e_mev, e_kev in zip(ENERGIES_MEV, ENERGIES_KEV):
        f = fractions[e_mev]
        is_1157 = e_kev == 1157.0
        rows.append({
            "energy_keV":            int(e_kev),
            "mu_lyso_cm1":           mu_lyso[e_mev],
            "mu_water_cm1":          mu_water[e_mev],
            "f_compton_lyso":        f["f_compton"],
            "f_pe_lyso":             f["f_pe"],
            "P_KN_600keV_threshold": p_kn_600 if is_1157 else 0.0,
            "f_useful_1157":         f_useful_1157 if is_1157 else 1.0,
        })

    csv_path = "estimates/results/material_constants.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
