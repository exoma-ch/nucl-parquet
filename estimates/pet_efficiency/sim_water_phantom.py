# /// script
# dependencies = [
#   "nucl-parquet @ /Users/larsgerchow/Projects/eXoma/nucl-parquet",
#   "scipy",
#   "matplotlib",
#   "pandas",
#   "numpy",
# ]
# ///
"""Triple-coincidence efficiency for a NEMA-style water phantom (20 cm diam, 100 cm long)
with a centred line source, including photon attenuation through water before the LYSO crystal.

PET4PETs scanner geometry, ⁴⁴Sc (β⁺γ).
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import nucl_parquet
import pandas as pd
from scipy import integrate

# ---------------------------------------------------------------------------
# Geometry / scanner parameters
# ---------------------------------------------------------------------------
R_IN_MM = 350.0
CRYSTAL_DEPTHS_MM = [25, 40]
FOV_VALUES_MM = [3, 6, 12, 24, 48, 96, 192]
PHANTOM_RADIUS_MM = 100.0   # 10 cm radius → 20 cm NEMA diameter
PHANTOM_LENGTH_MM = 1000.0  # 100 cm
LINE_SOURCE_LENGTH_MM = 1000.0
N_Z_POSITIONS = 200

NUCL_PARQUET_DIR = "/Users/larsgerchow/Projects/eXoma/nucl-parquet"

LYSO_COMPOSITION = {71: 0.7145, 39: 0.0403, 14: 0.0637, 8: 0.1815}
LYSO_DENSITY_G_CM3 = 7.1

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Material data
# ---------------------------------------------------------------------------
db = nucl_parquet.connect(NUCL_PARQUET_DIR)


def _log_interp(x_new, x_arr, y_arr):
    """Log-log interpolation."""
    return np.exp(np.interp(np.log(x_new), np.log(x_arr), np.log(y_arr)))


def get_mu_lyso(E_MeV: float) -> float:
    """Total linear attenuation coefficient for LYSO at energy E_MeV [cm⁻¹]."""
    df = db.sql(
        "SELECT Z, energy_MeV, mu_rho_cm2_g FROM xcom_elements "
        "WHERE Z IN (71,39,14,8) AND energy_MeV BETWEEN 0.3 AND 2.0 "
        "ORDER BY Z, energy_MeV"
    ).fetchdf()
    mu_rho = sum(
        w * _log_interp(E_MeV, sub["energy_MeV"].values, sub["mu_rho_cm2_g"].values)
        for Z, w in LYSO_COMPOSITION.items()
        for sub in [df[df["Z"] == Z].sort_values("energy_MeV")]
    )
    return mu_rho * LYSO_DENSITY_G_CM3


def get_mu_water(E_MeV: float) -> float:
    """Total linear attenuation coefficient for water at energy E_MeV [cm⁻¹].

    Water density = 1.0 g/cm³, so mu = mu/rho.
    """
    df = db.sql(
        "SELECT energy_MeV, mu_rho_cm2_g FROM xcom_compounds "
        "WHERE material = 'water' ORDER BY energy_MeV"
    ).fetchdf()
    return _log_interp(E_MeV, df["energy_MeV"].values, df["mu_rho_cm2_g"].values)


def kn_fraction_above(E_keV: float, thresh_keV: float) -> float:
    """Fraction of Klein–Nishina cross section where scattered photon deposits > thresh_keV."""
    alpha = E_keV / 511.0

    def dkn(c):
        Ep = E_keV / (1 + alpha * (1 - c))
        r = Ep / E_keV
        return r**2 * (r + 1 / r - (1 - c**2))

    total, _ = integrate.quad(dkn, -1, 1)
    # Threshold: scattered photon has E' < E - thresh → c < c_max
    c_max = 1.0 - (E_keV / (E_keV - thresh_keV) - 1.0) / alpha
    if c_max <= -1:
        return 0.0
    above, _ = integrate.quad(dkn, -1, min(c_max, 1.0))
    return above / total


def get_epdl_fractions(E_MeV: float):
    """Return (f_compton, f_photoelectric) relative to total cross section from EPDL97."""
    df = db.sql(
        "SELECT Z, energy_MeV, process, xs_barns FROM epdl_photon_xs "
        "WHERE Z IN (71,39,14,8) "
        "AND process IN ('incoherent','photoelectric','total') "
        "AND energy_MeV BETWEEN 0.3 AND 2.0 "
        "ORDER BY Z, energy_MeV"
    ).fetchdf()
    fc, fpe, wtot = 0.0, 0.0, 0.0
    for Z, w in LYSO_COMPOSITION.items():
        s = df[df["Z"] == Z]

        def iv(proc, s=s):
            ss = s[s["process"] == proc].sort_values("energy_MeV")
            return _log_interp(E_MeV, ss["energy_MeV"].values, ss["xs_barns"].values)

        xt = iv("total")
        xc = iv("incoherent")
        xpe = iv("photoelectric")
        fc += w * xc
        fpe += w * xpe
        wtot += w * xt
    return fc / wtot, fpe / wtot


# ---------------------------------------------------------------------------
# Pre-compute attenuation coefficients and detection fractions
# ---------------------------------------------------------------------------
mu_lyso_511 = get_mu_lyso(0.511)
mu_lyso_1157 = get_mu_lyso(1.157)
mu_water_511 = get_mu_water(0.511)
mu_water_1157 = get_mu_water(1.157)

fc_1157, fpe_1157 = get_epdl_fractions(1.157)
P_KN_600 = kn_fraction_above(1157.0, 600.0)
f_useful_1157 = fpe_1157 + fc_1157 * P_KN_600

print(f"mu_lyso_511  = {mu_lyso_511:.4f} cm⁻¹")
print(f"mu_lyso_1157 = {mu_lyso_1157:.4f} cm⁻¹")
print(f"mu_water_511 = {mu_water_511:.4f} cm⁻¹  (expect ~0.096)")
print(f"mu_water_1157= {mu_water_1157:.4f} cm⁻¹  (expect ~0.059)")
print(f"f_useful_1157= {f_useful_1157:.4f}")


# ---------------------------------------------------------------------------
# Water path length through the phantom cylinder
# ---------------------------------------------------------------------------
def water_path_mm(theta: float, z_s: float) -> float:
    """Shortest path [mm] from the line source at z_s to the phantom boundary.

    The photon travels at polar angle theta from the z-axis.
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    d_side = PHANTOM_RADIUS_MM / sin_t if sin_t > 1e-9 else np.inf
    d_cap = (PHANTOM_LENGTH_MM / 2 - abs(z_s)) / max(abs(cos_t), 1e-9)
    return min(d_side, d_cap)


# ---------------------------------------------------------------------------
# Per-z_s efficiency integrand
# ---------------------------------------------------------------------------
def eps_at_z(z_s: float, dr: float, fov: float) -> tuple[float, float]:
    """Triple-coincidence efficiency at a single z_s position.

    Returns (eps_with_water, eps_no_water) in a single pass to avoid
    duplicate quadrature evaluations.
    """

    def integrand_pair(theta, use_water: bool):
        sin_t = np.sin(theta)
        if sin_t < 1e-9:
            return 0.0
        cot_t = np.cos(theta) / sin_t
        z1 = z_s + R_IN_MM * cot_t
        z2 = z_s - R_IN_MM * cot_t
        if abs(z1) > fov / 2 or abs(z2) > fov / 2:
            return 0.0
        d_w = water_path_mm(theta, z_s)
        Pw = np.exp(-mu_water_511 * d_w / 10) if use_water else 1.0
        Pc = 1 - np.exp(-mu_lyso_511 * dr / sin_t / 10)
        return (Pw * Pc) ** 2 * sin_t

    def integrand_gamma(theta, use_water: bool):
        sin_t = np.sin(theta)
        if sin_t < 1e-9:
            return 0.0
        z_hit = z_s + R_IN_MM * np.cos(theta) / sin_t
        if abs(z_hit) > fov / 2:
            return 0.0
        d_w = water_path_mm(theta, z_s)
        Pw = np.exp(-mu_water_1157 * d_w / 10) if use_water else 1.0
        Pc = 1 - np.exp(-mu_lyso_1157 * dr / sin_t / 10)
        return Pw * Pc * sin_t

    results = []
    for use_water in (True, False):
        ep, _ = integrate.quad(lambda t: integrand_pair(t, use_water), 1e-6, np.pi - 1e-6, limit=200)
        eg, _ = integrate.quad(lambda t: integrand_gamma(t, use_water), 1e-6, np.pi - 1e-6, limit=200)
        # 0.5 factors: solid-angle normalisation (∫₀^π sinθ dθ = 2)
        results.append(0.5 * ep * 0.5 * eg * f_useful_1157)
    return results[0], results[1]


# ---------------------------------------------------------------------------
# Line source averaging
# ---------------------------------------------------------------------------
def eps_line_source(dr: float, fov: float) -> tuple[float, float]:
    """Average triple-coincidence efficiency over the line source.

    Returns (eps_with_water, eps_no_water).
    """
    z_positions = np.linspace(
        -LINE_SOURCE_LENGTH_MM / 2, LINE_SOURCE_LENGTH_MM / 2, N_Z_POSITIONS
    )
    pairs = [eps_at_z(z, dr, fov) for z in z_positions]
    w, nw = zip(*pairs)
    return float(np.mean(w)), float(np.mean(nw))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
rows = []
for dr in CRYSTAL_DEPTHS_MM:
    for fov in FOV_VALUES_MM:
        print(f"  dr={dr:3d} mm  fov={fov:4d} mm", end="", flush=True)
        eps_w, eps_nw = eps_line_source(dr, fov)
        print(f"  → eps={eps_w:.4e}  eps_no_water={eps_nw:.4e}")
        rows.append(
            {
                "dr_mm": dr,
                "fov_mm": fov,
                "eps_triple_useful": eps_w,
                "eps_triple_no_water_atten": eps_nw,
            }
        )

df_out = pd.DataFrame(rows)
csv_path = RESULTS_DIR / "water_phantom.csv"
df_out.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")
print(df_out.to_string(index=False))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
colors = {25: "steelblue", 40: "darkorange"}

for dr in CRYSTAL_DEPTHS_MM:
    sub = df_out[df_out["dr_mm"] == dr]
    ax.plot(
        sub["fov_mm"], sub["eps_triple_useful"],
        "-o", color=colors[dr], label=f"dr={dr} mm (water atten)"
    )
    ax.plot(
        sub["fov_mm"], sub["eps_triple_no_water_atten"],
        "--o", color=colors[dr], alpha=0.5, label=f"dr={dr} mm (no water atten)"
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("FOV length [mm]")
ax.set_ylabel("Triple-coincidence efficiency")
ax.set_title("PET4PETs water phantom — ⁴⁴Sc triple-coincidence efficiency")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
ax.set_xticks(FOV_VALUES_MM)
ax.set_xticklabels([str(v) for v in FOV_VALUES_MM])

png_path = RESULTS_DIR / "water_phantom_efficiency.png"
fig.tight_layout()
fig.savefig(png_path, dpi=150)
print(f"Saved {png_path}")
