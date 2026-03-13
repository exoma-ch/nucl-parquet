"""Build gamma dose rate constants from ENSDF radiation data.

Computes the gamma dose rate constant k [µSv·m²/(MBq·h)] for every
radionuclide in the ENSDF radiation dataset. Uses the air-kerma approach
with NIST XCOM µ_en/ρ values for dry air.

Method:
    k = Σ_i [ dose_i × (µ_en/ρ)_air(E_i) ] × 0.1 × 1.602e-13 / (4π) × 3.6e15

where dose_i = E_i(MeV) × Y_i (fractional yield) is already provided as
dose_MeV_per_Bq_s in the radiation data.

Output: meta/dose_constants.parquet

Usage:
    uv run python -m nucl_parquet.build_dose_constants
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np

from .download import data_dir as _resolve_data_dir

# ---------------------------------------------------------------------------
# NIST XCOM µ_en/ρ for dry air (cm²/g)
# Source: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html
# ---------------------------------------------------------------------------
_MU_EN_E_KEV = np.array(
    [
        10,
        15,
        20,
        30,
        40,
        50,
        60,
        80,
        100,
        150,
        200,
        300,
        400,
        500,
        600,
        800,
        1000,
        1250,
        1500,
        2000,
        3000,
        4000,
        5000,
        6000,
        8000,
        10000,
    ]
)
_MU_EN = np.array(
    [
        4.742,
        1.334,
        0.5389,
        0.1537,
        0.06833,
        0.04098,
        0.03041,
        0.02407,
        0.02325,
        0.02496,
        0.02672,
        0.02872,
        0.02949,
        0.02966,
        0.02953,
        0.02882,
        0.02789,
        0.02666,
        0.02547,
        0.02345,
        0.02057,
        0.01870,
        0.01740,
        0.01647,
        0.01525,
        0.01450,
    ]
)
_LOG_E = np.log(_MU_EN_E_KEV)
_LOG_MU = np.log(_MU_EN)

# Unit conversion factor:
# dose [MeV/Bq·s] × µ_en/ρ [m²/kg] × 1.602e-13 [J/MeV] / (4π) → Gy·m²/(Bq·s)
# × 3600 [s/h] × 1e6 [µSv/Sv ≈ µGy/Gy for photons] × 1e6 [Bq/MBq]
_UNIT_FACTOR = 0.1 * 1.602e-13 / (4 * np.pi) * 3.6e15


def _mu_en_air(E_keV: np.ndarray) -> np.ndarray:
    """Interpolate µ_en/ρ for dry air [cm²/g] at given energies."""
    return np.exp(np.interp(np.log(np.clip(E_keV, 10, 10000)), _LOG_E, _LOG_MU))


def build(data_dir: Path | None = None) -> None:
    """Build meta/dose_constants.parquet from ENSDF radiation data."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    rad_dir = data_dir / "meta" / "ensdf" / "radiation"
    if not rad_dir.exists():
        raise FileNotFoundError(f"Radiation data not found: {rad_dir}")

    db = duckdb.connect()

    # Load all photon emissions
    rows = db.sql(f"""
        SELECT Z, A, dataset, parent_level_keV, decay_mode,
               rad_type, rad_subtype, energy_keV, intensity_pct, dose_MeV_per_Bq_s
        FROM read_parquet('{rad_dir}/*.parquet')
        WHERE rad_type IN ('gamma', 'xray') AND Z > 0
        ORDER BY Z, A, dataset
    """).fetchnumpy()

    Z = np.array(rows["Z"], dtype=np.int32)
    A = np.array(rows["A"], dtype=np.int32)
    parent_level = np.array(rows["parent_level_keV"], dtype=np.float64)
    E_keV = np.array(rows["energy_keV"], dtype=np.float64)
    I_pct = np.array(rows["intensity_pct"], dtype=np.float64)

    # Handle masked/None dose values (annihilation radiation)
    raw_dose = rows["dose_MeV_per_Bq_s"]
    dose = np.zeros(len(E_keV), dtype=np.float64)
    for i in range(len(dose)):
        d = raw_dose[i]
        if d is not None and d is not np.ma.masked:
            try:
                dose[i] = float(d)
            except (TypeError, ValueError):
                dose[i] = 0.0
        else:
            # Annihilation radiation: compute dose = E(MeV) * Y
            dose[i] = (E_keV[i] / 1000.0) * (I_pct[i] / 100.0)

    # Group by (Z, A, state) where state is derived from parent level.
    # Multiple datasets from the same excited level (e.g. IT + B- branches
    # of a metastable state) are merged into one entry.
    #
    # State assignment: ground (parent_level ≈ 0) → "", excited → "m".
    # If all datasets share the same parent level, state = "".

    # Step 1: collect indices per (Z, A, parent_level_rounded)
    state_groups = {}  # (Z, A, state_str) -> [indices]
    za_levels = {}  # (Z, A) -> set of rounded parent levels

    for i in range(len(Z)):
        z_i, a_i = int(Z[i]), int(A[i])
        pl_i = float(parent_level[i])
        key = (z_i, a_i)
        if key not in za_levels:
            za_levels[key] = set()
        za_levels[key].add(round(pl_i, 1))

    # Determine which levels are ground vs metastable
    for i in range(len(Z)):
        z_i, a_i = int(Z[i]), int(A[i])
        pl_i = round(float(parent_level[i]), 1)
        levels = za_levels[(z_i, a_i)]
        if len(levels) == 1 or pl_i == 0.0:
            state = ""
        else:
            state = "m"
        group_key = (z_i, a_i, state)
        if group_key not in state_groups:
            state_groups[group_key] = []
        state_groups[group_key].append(i)

    # Build results
    out_Z = []
    out_A = []
    out_state = []
    out_k = []
    out_dominant_keV = []
    out_n_lines = []

    for (z, a, state), indices in sorted(state_groups.items()):
        idx = np.array(indices)

        # Filter photon lines: E >= 10 keV, positive dose
        e = E_keV[idx]
        d = dose[idx]
        mask = (e >= 10) & (d > 0)

        if mask.sum() == 0:
            out_Z.append(z)
            out_A.append(a)
            out_state.append(state)
            out_k.append(0.0)
            out_dominant_keV.append(0.0)
            out_n_lines.append(0)
            continue

        e_filt = e[mask]
        d_filt = d[mask]
        i_filt = I_pct[idx][mask]

        mu = _mu_en_air(e_filt)
        k = float(np.sum(d_filt * mu) * _UNIT_FACTOR)

        # Dominant gamma: highest intensity line
        dom_idx = np.argmax(i_filt)
        dominant = float(e_filt[dom_idx])

        out_Z.append(z)
        out_A.append(a)
        out_state.append(state)
        out_k.append(round(k, 6))
        out_dominant_keV.append(round(dominant, 3))
        out_n_lines.append(int(mask.sum()))

    # Also add pure-beta emitters with k=0 from ground_states
    gs_path = data_dir / "meta" / "ensdf" / "ground_states.parquet"
    if gs_path.exists():
        gs = db.sql(f"""
            SELECT DISTINCT Z, A FROM read_parquet('{gs_path}')
            WHERE half_life_s IS NOT NULL AND half_life_s > 0 AND Z > 0
        """).fetchnumpy()
        existing = {(z, a, s) for z, a, s in zip(out_Z, out_A, out_state)}
        for z, a in zip(gs["Z"], gs["A"]):
            z, a = int(z), int(a)
            if (z, a, "") not in existing:
                out_Z.append(z)
                out_A.append(a)
                out_state.append("")
                out_k.append(0.0)
                out_dominant_keV.append(0.0)
                out_n_lines.append(0)

    # Write parquet
    import polars as pl

    df = pl.DataFrame(
        {
            "Z": pl.Series(out_Z, dtype=pl.Int32),
            "A": pl.Series(out_A, dtype=pl.Int32),
            "state": pl.Series(out_state, dtype=pl.Utf8),
            "k_uSv_m2_MBq_h": pl.Series(out_k, dtype=pl.Float64),
            "dominant_gamma_keV": pl.Series(out_dominant_keV, dtype=pl.Float64),
            "n_photon_lines": pl.Series(out_n_lines, dtype=pl.Int32),
        }
    ).sort("Z", "A", "state")

    out_path = data_dir / "meta" / "dose_constants.parquet"
    df.write_parquet(out_path, compression="zstd")

    print(f"Wrote {len(df)} nuclides to {out_path}")
    print(f"  {(df['k_uSv_m2_MBq_h'] > 0).sum()} with k > 0")
    print(f"  {(df['k_uSv_m2_MBq_h'] == 0).sum()} with k = 0 (pure beta/alpha)")

    # Validation
    _validate(df)


def _validate(df) -> None:
    """Cross-check against known RADAR dose rate constants."""
    import polars as pl

    radar = {
        (43, 99, "m"): 0.0141,  # Tc-99m (gamma-only ~0.0142, with xray ~0.018)
        (9, 18, ""): 0.143,  # F-18
        (27, 60, ""): 0.306,  # Co-60
        (55, 137, ""): 0.077,  # Cs-137
        (53, 131, ""): 0.055,  # I-131
        (31, 68, ""): 0.130,  # Ga-68
        (11, 22, ""): 0.271,  # Na-22
    }

    print("\nValidation against RADAR:")
    print(f"  {'Isotope':<10} {'Computed':>10} {'RADAR':>10} {'Error':>8}")
    print(f"  {'-' * 42}")

    for (z, a, state), radar_k in radar.items():
        row = df.filter((pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") == state))
        if len(row) == 0:
            print(f"  Z={z} A={a} state='{state}': NOT FOUND")
            continue
        k = row["k_uSv_m2_MBq_h"][0]
        err = (k / radar_k - 1) * 100
        name = f"{'m' if state else ''}{['', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs'][z]}-{a}"
        status = "OK" if abs(err) < 20 else "CHECK"
        print(f"  {name:<10} {k:10.4f} {radar_k:10.4f} {err:+7.1f}% {status}")


if __name__ == "__main__":
    build()
