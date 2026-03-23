# /// script
# dependencies = [
#   "nucl-parquet @ /Users/larsgerchow/Projects/eXoma/nucl-parquet",
#   "scipy",
#   "matplotlib",
#   "pandas",
#   "numpy",
# ]
# ///
"""Triple-coincidence efficiency for a centered line source in the PET4PETs scanner.

Sweeps over FOV and crystal depth configurations. Outputs CSV and PNG to
estimates/results/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import nucl_parquet

# ---------------------------------------------------------------------------
# Scanner / source parameters
# ---------------------------------------------------------------------------
R_IN_MM = 350.0
CRYSTAL_DEPTHS_MM = [25, 40]
FOV_VALUES_MM = [3, 6, 12, 24, 48, 96, 192]
LINE_SOURCE_LENGTH_MM = 1000.0  # 100 cm
NUCL_PARQUET_DIR = "/Users/larsgerchow/Projects/eXoma/nucl-parquet"
LYSO_COMPOSITION = {71: 0.7145, 39: 0.0403, 14: 0.0637, 8: 0.1815}  # Z: mass_fraction
LYSO_DENSITY_G_CM3 = 7.1

# ---------------------------------------------------------------------------
# LYSO attenuation coefficient via XCOM + Bragg mixing
# ---------------------------------------------------------------------------
db = nucl_parquet.connect(NUCL_PARQUET_DIR)


def _loglog_interp(sub_df, energy_MeV: float, col: str) -> float:
    """Log-log interpolate `col` in a sorted DataFrame at `energy_MeV`."""
    return np.exp(np.interp(
        np.log(energy_MeV),
        np.log(sub_df["energy_MeV"].values),
        np.log(sub_df[col].values),
    ))


def get_lyso_mu(energy_MeV: float) -> float:
    """Linear attenuation coefficient [cm⁻¹] for LYSO at given energy."""
    df = db.sql("""
        SELECT Z, energy_MeV, mu_rho_cm2_g FROM xcom_elements
        WHERE Z IN (71, 39, 14, 8) AND energy_MeV BETWEEN 0.3 AND 2.0
        ORDER BY Z, energy_MeV
    """).fetchdf()
    mu_rho = 0.0
    for Z, w in LYSO_COMPOSITION.items():
        sub = df[df["Z"] == Z].sort_values("energy_MeV")
        mu_rho += w * _loglog_interp(sub, energy_MeV, "mu_rho_cm2_g")
    return mu_rho * LYSO_DENSITY_G_CM3


mu_511 = get_lyso_mu(0.511)   # cm⁻¹
mu_1157 = get_lyso_mu(1.157)  # cm⁻¹

# ---------------------------------------------------------------------------
# f_useful for 1157 keV photon — Klein-Nishina + EPDL97 process fractions
# ---------------------------------------------------------------------------

def kn_fraction_above_threshold(E_keV: float, threshold_keV: float) -> float:
    """Fraction of KN Compton events depositing > threshold_keV."""
    alpha = E_keV / 511.0

    def dkn(costh):
        Ep = E_keV / (1 + alpha * (1 - costh))
        r = Ep / E_keV
        return r ** 2 * (r + 1 / r - (1 - costh ** 2))

    total, _ = integrate.quad(lambda c: dkn(c), -1, 1)
    # deposit = E - E' > threshold  ↔  costh < 1 - (E/(E-threshold) - 1)/α
    costh_max = 1.0 - (E_keV / (E_keV - threshold_keV) - 1.0) / alpha
    if costh_max <= -1:
        return 0.0
    above, _ = integrate.quad(lambda c: dkn(c), -1, min(costh_max, 1.0))
    return above / total


def get_process_fractions(energy_MeV: float):
    """Return (f_compton, f_pe) weighted by Bragg mix at given energy."""
    df = db.sql("""
        SELECT Z, energy_MeV, process, xs_barns FROM epdl_photon_xs
        WHERE Z IN (71, 39, 14, 8)
          AND process IN ('incoherent', 'photoelectric', 'total')
          AND energy_MeV BETWEEN 0.3 AND 2.0
        ORDER BY Z, energy_MeV, process
    """).fetchdf()

    f_compton_total, f_pe_total, weight_total = 0.0, 0.0, 0.0
    for Z, w in LYSO_COMPOSITION.items():
        sub = df[df["Z"] == Z]

        def interp_proc(proc, _sub=sub):
            s = _sub[_sub["process"] == proc].sort_values("energy_MeV")
            return _loglog_interp(s, energy_MeV, "xs_barns")

        xs_total = interp_proc("total")
        xs_compton = interp_proc("incoherent")
        xs_pe = interp_proc("photoelectric")
        f_compton_total += w * xs_compton
        f_pe_total += w * xs_pe
        weight_total += w * xs_total

    return f_compton_total / weight_total, f_pe_total / weight_total


f_compton_1157, f_pe_1157 = get_process_fractions(1.157)
P_KN_600 = kn_fraction_above_threshold(1157.0, 600.0)
f_useful_1157 = f_pe_1157 + f_compton_1157 * P_KN_600

# ---------------------------------------------------------------------------
# Point-source efficiency for a source at (0, 0, z_s)
# ---------------------------------------------------------------------------

def point_source_efficiency_at_zs(
    z_s_mm: float,
    dr_mm: float,
    fov_mm: float,
    mu_511_cm1: float,
    mu_1157_cm1: float,
    f_useful: float,
) -> float:
    """Triple-coincidence efficiency for a point source at axial position z_s_mm.

    Returns eps_pair * eps_gamma * f_useful where each factor is normalised to 4π.
    """

    def integrand_pair(theta):
        sin_th = np.sin(theta)
        if sin_th < 1e-10:
            return 0.0
        cot_th = np.cos(theta) / sin_th
        z1 = z_s_mm + R_IN_MM * cot_th
        z2 = z_s_mm - R_IN_MM * cot_th
        if abs(z1) > fov_mm / 2 or abs(z2) > fov_mm / 2:
            return 0.0
        L_eff_cm = dr_mm / sin_th / 10.0  # mm → cm
        P511 = 1.0 - np.exp(-mu_511_cm1 * L_eff_cm)
        return P511 ** 2 * sin_th

    def integrand_gamma(theta):
        sin_th = np.sin(theta)
        if sin_th < 1e-10:
            return 0.0
        cot_th = np.cos(theta) / sin_th
        z_hit = z_s_mm + R_IN_MM * cot_th
        if abs(z_hit) > fov_mm / 2:
            return 0.0
        L_eff_cm = dr_mm / sin_th / 10.0  # mm → cm
        P1157 = 1.0 - np.exp(-mu_1157_cm1 * L_eff_cm)
        return P1157 * sin_th

    eps_pair, _ = integrate.quad(integrand_pair, 1e-6, np.pi - 1e-6, limit=200)
    eps_pair *= 0.5  # normalise solid angle to 4π

    eps_gamma, _ = integrate.quad(integrand_gamma, 1e-6, np.pi - 1e-6, limit=200)
    eps_gamma *= 0.5

    return eps_pair * eps_gamma * f_useful


# ---------------------------------------------------------------------------
# Line-source efficiency (average over 200 axial positions)
# ---------------------------------------------------------------------------

Z_POSITIONS = np.linspace(-LINE_SOURCE_LENGTH_MM / 2, LINE_SOURCE_LENGTH_MM / 2, 200)

rows = []
for dr_mm in CRYSTAL_DEPTHS_MM:
    for fov_mm in FOV_VALUES_MM:
        effs = [
            point_source_efficiency_at_zs(
                zs, dr_mm, fov_mm, mu_511, mu_1157, f_useful_1157
            )
            for zs in Z_POSITIONS
        ]
        eps_triple_useful = float(np.mean(effs))
        # f_useful is a scalar factor applied uniformly — derive raw from useful
        eps_triple_raw = eps_triple_useful / f_useful_1157 if f_useful_1157 > 0 else 0.0

        rows.append({
            "dr_mm": dr_mm,
            "fov_mm": fov_mm,
            "eps_triple_useful": eps_triple_useful,
            "eps_triple_raw": eps_triple_raw,
        })
        print(f"  dr={dr_mm:3d} mm  FOV={fov_mm:4d} mm  "
              f"ε_useful={eps_triple_useful:.3e}  ε_raw={eps_triple_raw:.3e}")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
df_out = pd.DataFrame(rows)
csv_path = "estimates/results/line_source.csv"
df_out.to_csv(csv_path, index=False)
print(f"\nSaved {len(df_out)} rows → {csv_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

markers = {25: "o", 40: "s"}
colors = {25: "#1f77b4", 40: "#d62728"}

for dr_mm in CRYSTAL_DEPTHS_MM:
    sub = df_out[df_out["dr_mm"] == dr_mm].sort_values("fov_mm")
    ax.plot(
        sub["fov_mm"],
        sub["eps_triple_useful"],
        marker=markers[dr_mm],
        color=colors[dr_mm],
        label=f"Crystal depth {dr_mm} mm",
        linewidth=1.8,
        markersize=6,
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("FOV length [mm]", fontsize=12)
ax.set_ylabel(r"$\varepsilon_{\rm triple,useful}$ (line source avg.)", fontsize=12)
ax.set_title("PET4PETs triple-coincidence efficiency — 100 cm line source", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", alpha=0.5)

# Annotate key parameters
ax.text(
    0.02, 0.03,
    f"R_in = {R_IN_MM:.0f} mm, LYSO\n"
    f"f_useful(1157 keV) = {f_useful_1157:.3f}\n"
    f"μ(511 keV) = {mu_511:.3f} cm⁻¹,  μ(1157 keV) = {mu_1157:.3f} cm⁻¹",
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
)

fig.tight_layout()
png_path = "estimates/results/line_source_efficiency.png"
fig.savefig(png_path, dpi=150)
print(f"Saved plot → {png_path}")
