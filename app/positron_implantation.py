"""
Na-22 positron escape fraction from one face of an Al slab.

Geometry: uniform Na-22 source in Al disc; cylindrical symmetry → 1D slab problem.
Physics:
  - ESTAR mass stopping power (MeV cm²/g) for electrons in Al → CSDA range
  - Positrons use the same stopping power as electrons to good approximation
  - Isotropic emission; straight-line CSDA path-length criterion for escape
  - Na-22 β+ spectrum (Emax = 545.4 keV), allowed transition with Fermi function

Escape probability for a positron at depth z from the exit face with range R:
  P(z, R) = (1/2)(1 - z/R)  for z ≤ R
           = 0               for z > R

Mean escape fraction over uniform slab of thickness t:
  F(E, t) = (1/t) ∫₀ᵗ P(z, R(E)) dz
           = 1/2 - t/(4R)   if R ≥ t
           = R/(4t)         if R < t

Total fraction: F(t) = ∫ F(E,t) · S(E) dE
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import pyarrow.parquet as pq

# ── material ──────────────────────────────────────────────────────────────────
RHO_AL = 2.699  # g/cm³

# ── load ESTAR for Al (Z=13) ─────────────────────────────────────────────────
tbl = pq.read_table("stopping/stopping.parquet").to_pydict()
mask = [s == "ESTAR" and z == 13 for s, z in zip(tbl["source"], tbl["target_Z"])]
E_tab = np.array([e for e, m in zip(tbl["energy_MeV"], mask) if m])
sp_tab = np.array([x for x, m in zip(tbl["dedx"], mask) if m])  # MeV cm²/g
idx = np.argsort(E_tab)
E_tab, sp_tab = E_tab[idx], sp_tab[idx]

# linear stopping power [MeV/cm]
sp_lin = sp_tab * RHO_AL

# CSDA range [cm] via cumulative trapezoid of 1/(dE/dx)
csda_cm = cumulative_trapezoid(1.0 / sp_lin, E_tab, initial=0.0)
csda_mm = csda_cm * 10.0  # → mm

R_of_E = interp1d(E_tab, csda_mm, kind="cubic", bounds_error=False, fill_value=(0.0, csda_mm[-1]))

# ── Na-22 β+ spectrum ─────────────────────────────────────────────────────────
ME = 0.511          # MeV, electron/positron rest mass
EMAX = 0.5454       # MeV, Na-22 β+ endpoint (main branch 89.84 % → Ne-22* 1274 keV)
Z_DAUGHTER = 10     # Ne-22

N_E = 2000
E_b = np.linspace(1e-4, EMAX * 0.9999, N_E)

E_tot = E_b + ME                      # total energy [MeV]
p_b   = np.sqrt(E_tot**2 - ME**2)     # momentum [MeV/c]

# Approximate Fermi function for positrons (Z_d = 10):
# F(Z, W) ≈ 2πη / (e^{2πη} - 1)  where η = −α Z_d W / p  (negative for β+)
ALPHA = 1 / 137.036
eta   = -ALPHA * Z_DAUGHTER * E_tot / p_b   # < 0 for positrons (repulsion)
two_pi_eta = 2 * np.pi * eta
fermi = two_pi_eta / (np.exp(two_pi_eta) - 1)   # <1 for positrons (suppressed)

spectrum = fermi * p_b * E_tot * (EMAX - E_b) ** 2
spectrum /= np.trapezoid(spectrum, E_b)   # normalise

# ── escape fraction vs slab thickness ────────────────────────────────────────
thicknesses_mm = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

R_b = R_of_E(E_b)   # CSDA range in Al [mm] for each β energy
R_b = np.maximum(R_b, 1e-12)   # avoid /0 at lowest energies

print(f"{'thickness (mm)':>16}  {'escape fraction':>16}")
print("-" * 38)

results = []
for t_mm in thicknesses_mm:
    # per-energy escape fraction
    F_E = np.where(R_b >= t_mm,
                   0.5 - t_mm / (4.0 * R_b),
                   R_b / (4.0 * t_mm))
    F_total = np.trapezoid(F_E * spectrum, E_b)
    results.append(F_total)
    print(f"{t_mm:>16.2f}  {F_total:>16.4f}")

# also report mean range for context
mean_R = np.trapezoid(R_b * spectrum, E_b)
print(f"\n  Mean positron CSDA range in Al: {mean_R:.3f} mm")
print(f"  (endpoint range at 545 keV:     {float(R_of_E(EMAX)):.3f} mm)")
print(f"\nNote: CSDA straight-line path overestimates projected depth range.")
print(f"      Actual mean depth ≈ 0.56–0.70 × CSDA range for e⁺/e⁻ in matter.")
