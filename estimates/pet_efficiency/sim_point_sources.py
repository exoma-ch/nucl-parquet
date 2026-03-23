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
sim_point_sources.py
====================
Compute triple-coincidence efficiency for point sources (centered and off-center)
in a finite-tube LYSO PET scanner, sweeping over FOV and crystal depth.

Scanner: PET4PETs for 44Sc (β+γ) triple coincidence.
  - 2× 511 keV annihilation photons (back-to-back)
  - 1× 1157 keV prompt gamma (independent)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad

import os

# Ensure the data directory is resolvable when running via `uv run`
# (the installed copy of nucl_parquet can't find the data next to it)
_NUCL_PARQUET_DIR = "/Users/larsgerchow/Projects/eXoma/nucl-parquet"
os.environ.setdefault("NUCL_PARQUET_DATA", _NUCL_PARQUET_DIR)

import nucl_parquet  # noqa: E402

# ── Scanner / source parameters ──────────────────────────────────────────────
R_IN_MM = 350.0
CRYSTAL_DEPTHS_MM = [25, 40]
FOV_VALUES_MM = [3, 6, 12, 24, 48, 96, 192]
CRYSTAL_PITCH_MM = 3.0
OFF_CENTER_R_MM = 175.0  # R_in / 2

N_MC = 500_000  # Monte-Carlo events per (dr, FOV)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LYSO material constants ───────────────────────────────────────────────────
LYSO_COMPOSITION = {
    71: ("Lu", 0.7145),
    39: ("Y",  0.0403),
    14: ("Si", 0.0637),
    8:  ("O",  0.1815),
}
LYSO_DENSITY_G_CM3 = 7.1


# ── Shared log-log interpolation helper ──────────────────────────────────────
def _loglog_interp(x_arr: np.ndarray, y_arr: np.ndarray, x: float) -> float:
    """Scalar log-log interpolation (clamps to endpoints)."""
    return math.exp(float(np.interp(math.log(x), np.log(x_arr), np.log(y_arr))))


# ── Load attenuation data from XCOM ──────────────────────────────────────────
def build_lyso_mu_fn(db):
    """Return callable (energy_MeV → μ_LYSO [cm⁻¹]) via log-log interpolation."""
    zs = list(LYSO_COMPOSITION.keys())
    z_str = ",".join(str(z) for z in zs)
    rows = db.sql(
        f"SELECT Z, energy_MeV, mu_rho_cm2_g FROM xcom_elements "
        f"WHERE Z IN ({z_str}) AND energy_MeV BETWEEN 0.4 AND 1.5 "
        f"ORDER BY Z, energy_MeV"
    ).fetchall()

    by_z: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    tmp: dict[int, tuple[list, list]] = {}
    for z, e, mu_rho in rows:
        tmp.setdefault(z, ([], []))
        tmp[z][0].append(e)
        tmp[z][1].append(mu_rho)
    for z, (es, mus) in tmp.items():
        by_z[z] = (np.array(es), np.array(mus))

    # Precompute per-element weight × density factor to avoid repeated lookups
    weights = {z: w for z, (_, w) in LYSO_COMPOSITION.items()}

    def mu_lyso(energy_MeV: float) -> float:
        mu_rho_mix = sum(
            weights[z] * _loglog_interp(es, mus, energy_MeV)
            for z, (es, mus) in by_z.items()
        )
        return mu_rho_mix * LYSO_DENSITY_G_CM3

    return mu_lyso


# ── Klein-Nishina fraction for 1157 keV ──────────────────────────────────────
def compute_p_kn_600() -> float:
    """Fraction of 1157 keV Compton scatters depositing > 600 keV in crystal."""
    alpha = 1157.0 / 511.0

    def dkn_sin(theta: float) -> float:
        ep = 1157.0 / (1.0 + alpha * (1.0 - math.cos(theta)))
        r = ep / 1157.0
        return r**2 * (r + 1.0 / r - math.sin(theta) ** 2) * math.sin(theta)

    cos_theta_min = 1.0 - (1157.0 / 557.0 - 1.0) / alpha
    theta_min = math.acos(min(1.0, max(-1.0, cos_theta_min)))

    num, _ = quad(dkn_sin, theta_min, math.pi)
    denom, _ = quad(dkn_sin, 0.0, math.pi)
    return num / denom if denom > 0 else 0.0


# ── Process fractions from EPDL97 at 1157 keV ────────────────────────────────
def compute_f_useful_1157(db, p_kn_600: float) -> float:
    """f_useful_1157 = f_PE + f_Compton × P_KN_600."""
    zs = list(LYSO_COMPOSITION.keys())
    z_str = ",".join(str(z) for z in zs)
    processes = ("incoherent", "photoelectric", "total")
    proc_str = ",".join(f"'{p}'" for p in processes)

    rows = db.sql(
        f"SELECT Z, energy_MeV, process, xs_barns FROM epdl_photon_xs "
        f"WHERE Z IN ({z_str}) AND process IN ({proc_str}) "
        f"AND energy_MeV BETWEEN 0.9 AND 1.5 "
        f"ORDER BY Z, process, energy_MeV"
    ).fetchall()

    tmp: dict[tuple, tuple[list, list]] = {}
    for z, e, proc, xs in rows:
        key = (z, proc)
        tmp.setdefault(key, ([], []))
        tmp[key][0].append(e)
        tmp[key][1].append(xs)
    by_zp: dict[tuple, tuple[np.ndarray, np.ndarray]] = {
        k: (np.array(es), np.array(xs)) for k, (es, xs) in tmp.items()
    }

    def interp_xs(z: int, process: str, e_mev: float) -> float:
        es, xs_arr = by_zp[(z, process)]
        return _loglog_interp(es, xs_arr, e_mev)

    e_target = 1.157
    f_compton = f_pe = total_weight = 0.0
    for z, (_, w) in LYSO_COMPOSITION.items():
        xs_total = interp_xs(z, "total", e_target)
        f_compton    += w * interp_xs(z, "incoherent", e_target)
        f_pe         += w * interp_xs(z, "photoelectric", e_target)
        total_weight += w * xs_total

    # Normalise by total (mass-fraction × total-XS) weight
    norm = sum(
        w * _loglog_interp(by_zp[(z, "total")][0], by_zp[(z, "total")][1], e_target)
        for z, (_, w) in LYSO_COMPOSITION.items()
    )
    f_compton /= norm
    f_pe      /= norm
    return f_pe + f_compton * p_kn_600


# ── Core physics helper ───────────────────────────────────────────────────────
def p_abs(energy_keV: float, theta: float, dr_mm: float, mu_fn) -> float:
    """Absorption probability for a photon at polar angle θ from z-axis."""
    sin_t = math.sin(theta)
    if sin_t < 1e-9:
        return 1.0
    return 1.0 - math.exp(-mu_fn(energy_keV / 1000.0) * dr_mm / sin_t / 10.0)


# ── Central source: analytical integrals ─────────────────────────────────────
def central_efficiency(
    dr_mm: float, fov_mm: float, mu_fn, f_useful_1157: float
) -> tuple[float, float, float]:
    """ε_pair, ε_gamma, ε_triple for central source, normalised to 4π."""
    theta_c = math.atan2(2.0 * R_IN_MM, fov_mm)

    eps_pair_raw,  _ = quad(
        lambda t: p_abs(511.0,  t, dr_mm, mu_fn) ** 2 * math.sin(t),
        theta_c, math.pi / 2,
    )
    eps_gamma_raw, _ = quad(
        lambda t: p_abs(1157.0, t, dr_mm, mu_fn) * math.sin(t),
        theta_c, math.pi / 2,
    )

    eps_pair  = 0.5 * eps_pair_raw
    eps_gamma = 0.5 * eps_gamma_raw
    return eps_pair, eps_gamma, eps_pair * eps_gamma * f_useful_1157


# ── Off-center source: Monte Carlo ───────────────────────────────────────────
def _check_barrel_vec(
    src: np.ndarray,
    directions: np.ndarray,
    r_in: float,
    fov: float,
    dr_mm: float,
    mu_cm: float,
) -> np.ndarray:
    """Vectorised barrel intersection + crystal absorption probability."""
    sx, sy = src[0], src[1]
    dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]

    A = dx**2 + dy**2
    B = 2.0 * (sx * dx + sy * dy)
    disc = B**2 - 4.0 * A * (sx**2 + sy**2 - r_in**2)

    valid = (disc >= 0.0) & (A > 1e-12)
    t = np.where(valid, (-B + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A), np.inf)

    in_fov = np.abs(src[2] + t * dz) <= fov / 2.0

    x_hit = sx + t * dx
    y_hit = sy + t * dy
    r_hit = np.maximum(np.sqrt(x_hit**2 + y_hit**2), 1e-9)
    d_r = np.maximum(np.abs(dx * x_hit / r_hit + dy * y_hit / r_hit), 0.01)

    p = 1.0 - np.exp(-mu_cm * dr_mm / d_r / 10.0)
    return np.where(in_fov & valid, p, 0.0)


def _sample_isotropic(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n isotropic unit directions uniformly on the sphere."""
    cos_t = rng.uniform(-1.0, 1.0, n)
    sin_t = np.sqrt(1.0 - cos_t**2)
    phi = rng.uniform(0.0, 2.0 * math.pi, n)
    return np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])


def offcenter_efficiency(
    dr_mm: float,
    fov_mm: float,
    mu511: float,
    mu1157: float,
    f_useful_1157: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Monte-Carlo triple-coincidence efficiency for off-center source."""
    d1 = _sample_isotropic(rng, N_MC)
    d3 = _sample_isotropic(rng, N_MC)
    src = np.array([OFF_CENTER_R_MM, 0.0, 0.0])

    p1 = _check_barrel_vec(src, d1,  R_IN_MM, fov_mm, dr_mm, mu511)
    p2 = _check_barrel_vec(src, -d1, R_IN_MM, fov_mm, dr_mm, mu511)
    p3 = _check_barrel_vec(src, d3,  R_IN_MM, fov_mm, dr_mm, mu1157)

    eps_pair  = float(np.mean(p1 * p2))
    eps_gamma = float(np.mean(p3))
    return eps_pair, eps_gamma, float(np.mean(p1 * p2 * p3)) * f_useful_1157


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Building LYSO attenuation function …")
    db = nucl_parquet.connect()
    mu_fn = build_lyso_mu_fn(db)

    print("Computing Klein-Nishina fraction (ΔE > 600 keV) …")
    p_kn_600 = compute_p_kn_600()
    print(f"  P_KN_600 = {p_kn_600:.4f}")

    print("Computing useful fraction at 1157 keV …")
    f_useful = compute_f_useful_1157(db, p_kn_600)
    print(f"  f_useful_1157 = {f_useful:.4f}")

    # Pre-evaluate attenuation coefficients used repeatedly in the MC loop
    mu511  = mu_fn(0.511)
    mu1157 = mu_fn(1.157)

    rng = np.random.default_rng(42)
    records = []

    print("\nCentral source (analytical) …")
    for dr in CRYSTAL_DEPTHS_MM:
        for fov in FOV_VALUES_MM:
            eps_pair, eps_gamma, eps_triple = central_efficiency(
                dr, fov, mu_fn, f_useful
            )
            records.append(dict(
                source_type="center", dr_mm=dr, fov_mm=fov,
                eps_pair_511=eps_pair, eps_gamma_1157=eps_gamma,
                eps_triple_raw=eps_pair * eps_gamma,
                eps_triple_useful=eps_triple,
            ))
            print(
                f"  dr={dr:2d} mm  FOV={fov:4d} mm  "
                f"ε_pair={eps_pair:.4e}  ε_γ={eps_gamma:.4e}  "
                f"ε_triple={eps_triple:.4e}"
            )

    print(f"\nOff-center source (MC, N={N_MC:,}) …")
    for dr in CRYSTAL_DEPTHS_MM:
        for fov in FOV_VALUES_MM:
            eps_pair, eps_gamma, eps_triple = offcenter_efficiency(
                dr, fov, mu511, mu1157, f_useful, rng
            )
            records.append(dict(
                source_type="offcenter", dr_mm=dr, fov_mm=fov,
                eps_pair_511=eps_pair, eps_gamma_1157=eps_gamma,
                eps_triple_raw=eps_pair * eps_gamma,
                eps_triple_useful=eps_triple,
            ))
            print(
                f"  dr={dr:2d} mm  FOV={fov:4d} mm  "
                f"ε_pair={eps_pair:.4e}  ε_γ={eps_gamma:.4e}  "
                f"ε_triple={eps_triple:.4e}"
            )

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "point_sources.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows → {csv_path}")

    _plot_efficiency(df, "center")
    _plot_efficiency(df, "offcenter")
    print("Done.")


def _plot_efficiency(df: pd.DataFrame, source_type: str) -> None:
    sub = df[df["source_type"] == source_type]
    colors = {25: "steelblue", 40: "darkorange"}
    lines: dict[int, pd.DataFrame] = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    for dr in CRYSTAL_DEPTHS_MM:
        s = sub[sub["dr_mm"] == dr].sort_values("fov_mm")
        ax.plot(s["fov_mm"], s["eps_triple_useful"],
                marker="o", label=f"dr={dr} mm", color=colors[dr])
        lines[dr] = s

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Axial FOV [mm]", fontsize=12)
    ax.set_ylabel("Triple-coincidence efficiency", fontsize=12)
    src_label = "Center" if source_type == "center" else "Off-center (r=175 mm)"
    ax.set_title(
        f"PET4PETs triple-coincidence efficiency\nPoint source: {src_label}",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    ax2 = ax.inset_axes([0.55, 0.08, 0.40, 0.35])
    s25 = lines[25].sort_values("fov_mm")
    s40 = lines[40].sort_values("fov_mm")
    ax2.plot(
        s25["fov_mm"].values,
        s40["eps_triple_useful"].values / s25["eps_triple_useful"].values,
        marker="s", color="purple",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("FOV [mm]", fontsize=8)
    ax2.set_ylabel("40/25 mm ratio", fontsize=8)
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    out = RESULTS_DIR / f"point_source_{source_type}_efficiency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {out}")


if __name__ == "__main__":
    main()
