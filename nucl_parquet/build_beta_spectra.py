"""Pre-tabulate β-decay continuous spectra (#78).

For each β-minus and β-plus transition in strata's radioactive_decay.parquet,
compute the kinetic-energy spectrum N(E) using the Fermi function plus the
appropriate shape factor for the transition's forbiddenness, then ship the
binned spectrum so consumers don't have to evaluate the Fermi function at
runtime.

Output: `data/meta/ensdf/beta_spectra/{Symbol}.parquet`
Schema:
    Z                  int32   (parent atomic number)
    A                  int32   (parent mass number)
    state              string  ("" for ground, "m" for metastable)
    transition_idx     int32   (0-based, orders transitions per (Z,A,state))
    decay_mode         string  ("BetaMinus" | "BetaPlus")
    forbiddenness      string  ("" allowed | "uniqueFirstForbidden" | …)
    endpoint_keV       float64 (β endpoint kinetic energy, Q for that transition)
    branching          float64 (transition's branching ratio within decay mode)
    energy_keV         float64 (kinetic-energy bin center)
    dN_dE              float64 (probability density, integrates to 1.0)
    cumulative         float64 (CDF — useful for inverse-CDF sampling)

Physics:
- Allowed: N(E) ∝ F(±Z_daughter, E) · p · E_total · (E_max - E)²
- 1st-forbidden unique:    × S(E) = p² + q²       (units of m_e c)
- 2nd-forbidden unique:    × S(E) = q⁴ + (10/3) p² q² + p⁴
- Non-unique forbidden:    × S(E) = 1  (approximate; documented caveat)

F is the relativistic Fermi function evaluated via |Γ(γ₀ + iν)|² /
Γ(2γ₀+1)². No finite-nuclear-size correction (Bühring) is applied — within
~0.5% of the bare Fermi result at MeV-scale endpoints.

References:
- ICRP-107 (2008) — Nuclear Decay Data for Dosimetric Calculations
- Wilkinson, "Evaluation of beta-decay" (DSir Press, 1993)
- Endpoint values + forbiddenness from strata's radioactive_decay.parquet

Usage:
    uv run python -m nucl_parquet.build_beta_spectra
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .download import data_dir as _resolve_data_dir

# Constants (natural units where useful)
_M_E_KEV = 510.998_950  # electron rest mass in keV
_ALPHA = 7.297_352_5693e-3  # fine-structure constant

# Energy grid: 200 linear bins from 0 to endpoint. Linear keeps the
# (E_max - E)² near-endpoint cutoff well-resolved; 200 is plenty for
# detector-resolution-limited downstream use.
_N_BINS = 200

# Z range: cap at 99 (Es). Beyond that the Fermi-function αZ approaches the
# relativistic-limit corner and the input isotopes are short-lived; not useful.
_Z_MAX = 99

# Element symbol table (Z=1..118). We only need Z up to ~99 in practice.
_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
    "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es"
).split()


def _symbol_for(z: int) -> str:
    return _SYMBOLS[z - 1]


def _fermi_function(z_daughter_signed: int, e_kev: np.ndarray) -> np.ndarray:
    """Relativistic Fermi function F(Z, E) for β-decay shape correction.

    z_daughter_signed: +Z_daughter for β-minus, -Z_daughter for β-plus.
        (Coulomb attraction for emitted electron in β-; repulsion for β+.)
    e_kev: array of kinetic energies in keV (must be > 0).

    F is normalized in the convention that F(0, E) = 1 (no Coulomb correction).
    Computed as: F = 2(1 + γ₀) (2 p R)^(2(γ₀-1)) |Γ(γ₀ + iν)|² / Γ(2γ₀+1)² · e^(π ν)
    where p, ν are in natural units (m_e c = 1; energies / m_e c²).

    The (2 p R)^(2(γ₀-1)) factor introduces a nuclear radius R dependence, but
    since R cancels in the normalization to ∫ N(E) dE = 1 and the (γ₀-1) factor
    is small for Z ≲ 30, we use R = 1 (drops out of normalized spectra).
    """
    from scipy.special import loggamma

    e = e_kev / _M_E_KEV  # kinetic energy in m_e c² units
    w = e + 1.0  # total energy (γ in natural units)
    p = np.sqrt(np.maximum(w * w - 1.0, 1e-30))  # momentum

    alpha_z = _ALPHA * z_daughter_signed
    gamma_0 = math.sqrt(max(1.0 - alpha_z * alpha_z, 1e-30))  # √(1 - (αZ)²)
    nu = alpha_z * w / p  # Sommerfeld parameter

    # |Γ(γ₀ + iν)|² = exp(2 Re(loggamma(γ₀ + iν)))
    log_gamma_complex = loggamma(gamma_0 + 1j * nu)
    log_gamma_real = loggamma(2.0 * gamma_0 + 1.0)
    log_F = (
        math.log(2.0 * (1.0 + gamma_0))
        + 2.0 * (gamma_0 - 1.0) * np.log(2.0 * p)
        + 2.0 * np.real(log_gamma_complex)
        - 2.0 * log_gamma_real
        + math.pi * nu
    )
    return np.exp(log_F)


def _shape_factor(forbiddenness: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Multiplicative shape factor S(E) by forbiddenness class.

    Args:
        p: electron momentum (in m_e c units)
        q: neutrino momentum = E_max - E (in m_e c units)
    """
    if forbiddenness in ("", None) or forbiddenness == "allowed":
        return np.ones_like(p)
    if forbiddenness == "uniqueFirstForbidden":
        return p * p + q * q
    if forbiddenness == "uniqueSecondForbidden":
        return q**4 + (10.0 / 3.0) * (p * q) ** 2 + p**4
    # Non-unique 1st/2nd forbidden: small correction, treat as pure Fermi
    # within the tabulation tolerance. Documented in the catalog.
    return np.ones_like(p)


def _spectrum(
    parent_z: int,
    decay_mode: str,
    endpoint_kev: float,
    forbiddenness: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (energy_keV, dN/dE, cumulative) for one transition.

    Spectrum is normalized so that ∫ dN/dE · dE = 1.
    """
    if not math.isfinite(endpoint_kev) or endpoint_kev <= 0:
        return np.array([]), np.array([]), np.array([])

    # β-minus: emitted electron sees daughter Z = parent_Z + 1 (attractive)
    # β-plus: emitted positron sees daughter Z = parent_Z - 1 (repulsive)
    if decay_mode == "BetaMinus":
        z_signed = parent_z + 1
    elif decay_mode == "BetaPlus":
        z_signed = -(parent_z - 1)
    else:
        return np.array([]), np.array([]), np.array([])

    # Linear energy grid from ε > 0 to endpoint
    eps = max(endpoint_kev * 1e-4, 0.1)  # avoid the E=0 singularity in F(Z,E)
    energy_kev = np.linspace(eps, endpoint_kev, _N_BINS)

    e = energy_kev / _M_E_KEV
    w = e + 1.0
    p = np.sqrt(np.maximum(w * w - 1.0, 1e-30))
    q = (endpoint_kev - energy_kev) / _M_E_KEV  # neutrino momentum (massless ν)

    fermi = _fermi_function(z_signed, energy_kev)
    shape = _shape_factor(forbiddenness, p, q)
    dN_dE = fermi * p * w * (endpoint_kev - energy_kev) ** 2 * shape

    # Normalize to integrate to 1 over [0, endpoint_kev]
    integral = np.trapezoid(dN_dE, energy_kev)
    if integral <= 0 or not math.isfinite(integral):
        return np.array([]), np.array([]), np.array([])
    dN_dE = dN_dE / integral

    cumulative = np.zeros_like(dN_dE)
    cumulative[1:] = np.cumsum(0.5 * (dN_dE[1:] + dN_dE[:-1]) * np.diff(energy_kev))
    return energy_kev, dN_dE, cumulative


def build(data_dir: Path | None = None) -> None:
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    src = data_dir / "g4_raw" / "strata-nuclear" / "radioactive_decay.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Missing strata source {src}. Run `python scripts/fetch_strata_nuclear.py` first.")

    raw = pl.read_parquet(src)

    # Keep only beta transitions with a real endpoint Q-value, Z ≤ Z_MAX.
    beta = raw.filter(
        pl.col("decay_mode").is_in(["BetaMinus", "BetaPlus"])
        & (~pl.col("is_summary"))
        & pl.col("q_value_kev").is_not_null()
        & pl.col("q_value_kev").is_finite()
        & (pl.col("q_value_kev") > 0)
        & (pl.col("parent_z") <= _Z_MAX)
    )
    print(f"Building β-spectra for {beta.height:,} transitions across {beta['parent_z'].n_unique()} elements")

    # Map parent_level_flag to project-conventional state column
    def _state(level_flag: str | None) -> str:
        return "m" if (level_flag and level_flag.upper().startswith("M")) else ""

    # Group transitions by parent symbol → write one parquet per symbol
    out_root = data_dir / "meta" / "ensdf" / "beta_spectra"
    out_root.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_transitions = 0
    for parent_z in sorted(beta["parent_z"].unique().to_list()):
        symbol = _symbol_for(parent_z)
        slice_z = beta.filter(pl.col("parent_z") == parent_z).sort("parent_a", "parent_ex_kev", "daughter_ex_kev")

        rows_z: list[int] = []
        rows_a: list[int] = []
        rows_state: list[str] = []
        rows_tidx: list[int] = []
        rows_mode: list[str] = []
        rows_forb: list[str] = []
        rows_endpoint: list[float] = []
        rows_branching: list[float] = []
        rows_energy: list[float] = []
        rows_dnde: list[float] = []
        rows_cum: list[float] = []

        # Counter resets per (A, state) — each combination has its own transition index
        prev_key: tuple[int, str] | None = None
        tidx = 0

        for row in slice_z.iter_rows(named=True):
            state = _state(row.get("parent_level_flag"))
            key = (int(row["parent_a"]), state)
            if key != prev_key:
                tidx = 0
                prev_key = key
            else:
                tidx += 1

            energy, dnde, cum = _spectrum(
                parent_z=parent_z,
                decay_mode=row["decay_mode"],
                endpoint_kev=float(row["q_value_kev"]),
                forbiddenness=row.get("forbiddenness") or "",
            )
            if len(energy) == 0:
                continue

            for e_v, d_v, c_v in zip(energy.tolist(), dnde.tolist(), cum.tolist()):
                rows_z.append(int(parent_z))
                rows_a.append(int(row["parent_a"]))
                rows_state.append(state)
                rows_tidx.append(tidx)
                rows_mode.append(row["decay_mode"])
                rows_forb.append(row.get("forbiddenness") or "")
                rows_endpoint.append(float(row["q_value_kev"]))
                rows_branching.append(float(row["branching_ratio"]))
                rows_energy.append(e_v)
                rows_dnde.append(d_v)
                rows_cum.append(c_v)
            n_transitions += 1

        if not rows_z:
            continue

        df = pl.DataFrame(
            {
                "Z": pl.Series(rows_z, dtype=pl.Int32),
                "A": pl.Series(rows_a, dtype=pl.Int32),
                "state": pl.Series(rows_state, dtype=pl.Utf8),
                "transition_idx": pl.Series(rows_tidx, dtype=pl.Int32),
                "decay_mode": pl.Series(rows_mode, dtype=pl.Utf8),
                "forbiddenness": pl.Series(rows_forb, dtype=pl.Utf8),
                "endpoint_keV": pl.Series(rows_endpoint, dtype=pl.Float64),
                "branching": pl.Series(rows_branching, dtype=pl.Float64),
                # Float32 for the per-bin arrays — detector resolution is
                # always > 10⁻⁵ relative anyway, and halves the file size.
                "energy_keV": pl.Series(rows_energy, dtype=pl.Float32),
                "dN_dE": pl.Series(rows_dnde, dtype=pl.Float32),
                "cumulative": pl.Series(rows_cum, dtype=pl.Float32),
            }
        ).sort("A", "state", "transition_idx", "energy_keV")

        out_path = out_root / f"{symbol}.parquet"
        df.write_parquet(out_path, compression="zstd")
        n_written += 1

    print(f"Wrote {n_written} per-symbol files; {n_transitions:,} transitions total.")


if __name__ == "__main__":
    build()
