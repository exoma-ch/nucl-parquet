# /// script
# dependencies = [
#   "nucl-parquet @ /Users/larsgerchow/Projects/eXoma/nucl-parquet",
#   "scipy",
#   "matplotlib",
#   "pandas",
#   "numpy",
# ]
# ///
"""
Triple-coincidence efficiency for a homogeneous cylinder source in the PET4PETs scanner.
Source: ⁴⁴Sc (β⁺γ, 1157 keV). Models a filled phantom or patient body.
Two cases: centered (offset=0) and off-center (offset=175 mm = R_in/2).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nucl_parquet
from scipy import integrate

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
R_IN_MM = 350.0
CRYSTAL_DEPTHS_MM = [25, 40]
FOV_VALUES_MM = [3, 6, 12, 24, 48, 96, 192]
SOURCE_RADIUS_MM = 100.0
SOURCE_LENGTH_MM = 1000.0
SOURCE_OFFSETS_MM = [0.0, 175.0]
N_MC = 300_000

NUCL_PARQUET_DIR = "/Users/larsgerchow/Projects/eXoma/nucl-parquet"

LYSO_COMPOSITION = {71: 0.7145, 39: 0.0403, 14: 0.0637, 8: 0.1815}
LYSO_DENSITY_G_CM3 = 7.1

# ---------------------------------------------------------------------------
# Attenuation coefficients for LYSO
# ---------------------------------------------------------------------------
db = nucl_parquet.connect(NUCL_PARQUET_DIR)


def _loglog_interp(E, xs, ys):
    """Log-log interpolation at scalar E given tabulated (xs, ys)."""
    return np.exp(np.interp(np.log(E), np.log(xs), np.log(ys)))


def get_mu_lyso(E_MeV, df=None):
    if df is None:
        df = db.sql(
            "SELECT Z, energy_MeV, mu_rho_cm2_g FROM xcom_elements "
            "WHERE Z IN (71,39,14,8) AND energy_MeV BETWEEN 0.3 AND 2.0 "
            "ORDER BY Z, energy_MeV"
        ).fetchdf()
    mu_rho = 0.0
    for Z, w in LYSO_COMPOSITION.items():
        sub = df[df["Z"] == Z].sort_values("energy_MeV")
        mu_rho += w * _loglog_interp(E_MeV, sub["energy_MeV"].values, sub["mu_rho_cm2_g"].values)
    return mu_rho * LYSO_DENSITY_G_CM3


_xcom_df = db.sql(
    "SELECT Z, energy_MeV, mu_rho_cm2_g FROM xcom_elements "
    "WHERE Z IN (71,39,14,8) AND energy_MeV BETWEEN 0.3 AND 2.0 "
    "ORDER BY Z, energy_MeV"
).fetchdf()
mu_511 = get_mu_lyso(0.511, _xcom_df)
mu_1157 = get_mu_lyso(1.157, _xcom_df)


# ---------------------------------------------------------------------------
# Useful fraction for 1157 keV gamma (Klein-Nishina + EPDL97)
# ---------------------------------------------------------------------------
def kn_fraction_above(E_keV, thresh_keV):
    """Fraction of Compton-scattered photons that deposit above thresh_keV."""
    alpha = E_keV / 511.0

    def dkn(c):
        Ep = E_keV / (1 + alpha * (1 - c))
        r = Ep / E_keV
        return r**2 * (r + 1 / r - (1 - c**2))

    total, _ = integrate.quad(dkn, -1, 1)
    c_max = 1.0 - (E_keV / (E_keV - thresh_keV) - 1.0) / alpha
    if c_max <= -1:
        return 0.0
    above, _ = integrate.quad(dkn, -1, min(c_max, 1.0))
    return above / total


def get_epdl_fractions(E_MeV):
    df = db.sql(
        "SELECT Z, energy_MeV, process, xs_barns FROM epdl_photon_xs "
        "WHERE Z IN (71,39,14,8) AND process IN ('incoherent','photoelectric','total') "
        "AND energy_MeV BETWEEN 0.3 AND 2.0"
    ).fetchdf()
    fc, fpe, wtot = 0.0, 0.0, 0.0
    for Z, w in LYSO_COMPOSITION.items():
        s = df[df["Z"] == Z]

        def iv(proc, s=s):
            ss = s[s["process"] == proc].sort_values("energy_MeV")
            return _loglog_interp(E_MeV, ss["energy_MeV"].values, ss["xs_barns"].values)

        xt = iv("total")
        xc = iv("incoherent")
        xpe = iv("photoelectric")
        fc += w * xc
        fpe += w * xpe
        wtot += w * xt
    return fc / wtot, fpe / wtot


fc_1157, fpe_1157 = get_epdl_fractions(1.157)
f_useful_1157 = fpe_1157 + fc_1157 * kn_fraction_above(1157.0, 600.0)

print(f"mu_511  = {mu_511:.4f} cm⁻¹")
print(f"mu_1157 = {mu_1157:.4f} cm⁻¹")
print(f"f_useful_1157 = {f_useful_1157:.4f}")


# ---------------------------------------------------------------------------
# Monte Carlo efficiency
# ---------------------------------------------------------------------------
def barrel_intersect(sx, sy, sz, dx, dy, dz):
    """
    Forward intersection of rays from interior sources with the cylindrical barrel r=R_IN_MM.
    Returns (z_hit, cos_inc, valid): axial hit position, cosine of incidence on crystal face,
    and a boolean mask for geometrically valid hits.
    Sources are always inside the bore, so the forward root (larger t) is always positive.
    """
    A = dx**2 + dy**2
    B = 2 * (sx * dx + sy * dy)
    C = sx**2 + sy**2 - R_IN_MM**2
    disc = B**2 - 4 * A * C
    valid = (disc >= 0) & (A > 1e-12)
    sqrt_disc = np.sqrt(np.maximum(disc, 0))
    t = np.where(valid, (-B + sqrt_disc) / (2 * A), np.inf)

    x_hit = sx + t * dx
    y_hit = sy + t * dy
    z_hit = sz + t * dz

    r_hit = np.sqrt(x_hit**2 + y_hit**2)
    r_hat_x = x_hit / np.maximum(r_hit, 1e-9)
    r_hat_y = y_hit / np.maximum(r_hit, 1e-9)
    cos_inc = np.abs(dx * r_hat_x + dy * r_hat_y)
    return z_hit, cos_inc, valid


def mc_efficiency(dr_mm, fov_mm, offset_mm, N=N_MC):
    rng = np.random.default_rng(42)

    # Sample source positions uniformly in cylinder (axis at x=offset_mm, y=0)
    r_src = np.sqrt(rng.uniform(0, SOURCE_RADIUS_MM**2, N))
    phi_src = rng.uniform(0, 2 * np.pi, N)
    x_src = offset_mm + r_src * np.cos(phi_src)
    y_src = r_src * np.sin(phi_src)
    z_src = rng.uniform(-SOURCE_LENGTH_MM / 2, SOURCE_LENGTH_MM / 2, N)

    # 511 keV back-to-back pair direction
    cos_th1 = rng.uniform(-1, 1, N)
    sin_th1 = np.sqrt(1 - cos_th1**2)
    phi1 = rng.uniform(0, 2 * np.pi, N)
    d1x = sin_th1 * np.cos(phi1)
    d1y = sin_th1 * np.sin(phi1)
    d1z = cos_th1

    # 1157 keV gamma direction (independent)
    cos_tg = rng.uniform(-1, 1, N)
    sin_tg = np.sqrt(1 - cos_tg**2)
    phi_g = rng.uniform(0, 2 * np.pi, N)
    dgx = sin_tg * np.cos(phi_g)
    dgy = sin_tg * np.sin(phi_g)
    dgz = cos_tg

    z1, cos1, v1 = barrel_intersect(x_src, y_src, z_src, d1x, d1y, d1z)
    z2, cos2, v2 = barrel_intersect(x_src, y_src, z_src, -d1x, -d1y, -d1z)
    zg, cosg, vg = barrel_intersect(x_src, y_src, z_src, dgx, dgy, dgz)

    half_fov = fov_mm / 2
    in_fov1 = v1 & (np.abs(z1) <= half_fov)
    in_fov2 = v2 & (np.abs(z2) <= half_fov)
    in_fovg = vg & (np.abs(zg) <= half_fov)

    # Effective path length through crystal (mm → cm conversion inside exponent)
    L1 = dr_mm / np.maximum(cos1, 0.01)
    L2 = dr_mm / np.maximum(cos2, 0.01)
    Lg = dr_mm / np.maximum(cosg, 0.01)

    P1 = np.where(in_fov1, 1 - np.exp(-mu_511 * L1 / 10), 0.0)
    P2 = np.where(in_fov2, 1 - np.exp(-mu_511 * L2 / 10), 0.0)
    Pg = np.where(in_fovg, 1 - np.exp(-mu_1157 * Lg / 10), 0.0)

    eps_triple = np.mean(P1 * P2 * Pg) * f_useful_1157
    eps_pair = np.mean(P1 * P2)
    eps_gamma = np.mean(Pg) * f_useful_1157
    return eps_triple, eps_pair, eps_gamma


# ---------------------------------------------------------------------------
# Run all configurations
# ---------------------------------------------------------------------------
configs = [
    (offset_mm, dr_mm, fov_mm)
    for offset_mm in SOURCE_OFFSETS_MM
    for dr_mm in CRYSTAL_DEPTHS_MM
    for fov_mm in FOV_VALUES_MM
]
records = []
for i, (offset_mm, dr_mm, fov_mm) in enumerate(configs, 1):
    label = "homogeneous_centered" if offset_mm == 0.0 else "homogeneous_offcenter"
    print(
        f"[{i}/{len(configs)}] offset={offset_mm:.0f}mm  dr={dr_mm}mm  fov={fov_mm}mm",
        flush=True,
    )
    eps_t, eps_p, eps_g = mc_efficiency(dr_mm, fov_mm, offset_mm)
    records.append(
        {
            "source_type": label,
            "offset_mm": offset_mm,
            "dr_mm": dr_mm,
            "fov_mm": fov_mm,
            "eps_triple_useful": eps_t,
            "eps_pair": eps_p,
            "eps_gamma": eps_g,
        }
    )

df = pd.DataFrame(records)

os.makedirs("estimates/results", exist_ok=True)
df.to_csv("estimates/results/homogeneous.csv", index=False)
print(f"\nSaved {len(df)} rows → estimates/results/homogeneous.csv")
print(df.to_string())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plot(df_sub, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {25: "steelblue", 40: "tomato"}
    for dr in CRYSTAL_DEPTHS_MM:
        sub = df_sub[df_sub["dr_mm"] == dr].sort_values("fov_mm")
        ax.plot(
            sub["fov_mm"],
            sub["eps_triple_useful"],
            marker="o",
            color=colors[dr],
            label=f"crystal depth {dr} mm",
        )
    ax.set_xscale("log")
    ax.set_xlabel("FOV length (mm)")
    ax.set_ylabel("Triple-coincidence efficiency ε_triple")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out_path}")


make_plot(
    df[df["source_type"] == "homogeneous_centered"],
    "PET4PETs – Homogeneous cylinder source (centered)\n"
    r"$^{44}$Sc, R$_{src}$=100 mm, L=1000 mm",
    "estimates/results/homogeneous_centered_efficiency.png",
)

make_plot(
    df[df["source_type"] == "homogeneous_offcenter"],
    "PET4PETs – Homogeneous cylinder source (off-centre, 175 mm)\n"
    r"$^{44}$Sc, R$_{src}$=100 mm, L=1000 mm",
    "estimates/results/homogeneous_offcenter_efficiency.png",
)
