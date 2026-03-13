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
    out_source = []

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
            out_source.append("ensdf")
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
        out_source.append("ensdf")

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
                out_source.append("zero")

    # Backfill IT metastable states missing from radiation data.
    # These decay by isomeric transition (IT) emitting a single gamma at
    # ~level energy. We approximate the photon fraction using a simple
    # ICC model: fraction ≈ 0.5 for E < 200 keV, 0.9 for higher.
    existing = {(z, a, s) for z, a, s in zip(out_Z, out_A, out_state)}
    levels_dir = data_dir / "meta" / "ensdf" / "levels"
    if levels_dir.exists():
        it_levels = db.sql(f"""
            SELECT DISTINCT Z, A, energy_keV, half_life_s
            FROM read_parquet('{levels_dir}/*.parquet')
            WHERE energy_keV > 0
              AND half_life_s > 0.001
              AND (decay_1 = 'IT' OR decay_2 = 'IT' OR decay_3 = 'IT')
            ORDER BY Z, A
        """).fetchnumpy()

        n_it = 0
        for i in range(len(it_levels["Z"])):
            z, a = int(it_levels["Z"][i]), int(it_levels["A"][i])
            if (z, a, "m") in existing or (z, a, "") in existing:
                continue
            e_level = float(it_levels["energy_keV"][i])
            if e_level < 10:
                continue  # below µ_en/ρ table range

            # Approximate photon fraction from ICC
            photon_fraction = 0.5 if e_level < 200 else 0.9
            # dose = E(MeV) × Y, where Y = photon_fraction (single line, 100% IT)
            dose_it = (e_level / 1000.0) * photon_fraction
            mu = float(_mu_en_air(np.array([e_level]))[0])
            k = round(float(dose_it * mu * _UNIT_FACTOR), 6)

            out_Z.append(z)
            out_A.append(a)
            out_state.append("m")
            out_k.append(k)
            out_dominant_keV.append(round(e_level, 3))
            out_n_lines.append(1)
            out_source.append("it-approx")
            existing.add((z, a, "m"))
            n_it += 1

        print(f"  {n_it} IT metastable states backfilled")

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
            "source": pl.Series(out_source, dtype=pl.Utf8),
        }
    ).sort("Z", "A", "state")

    out_path = data_dir / "meta" / "dose_constants.parquet"
    df.write_parquet(out_path, compression="zstd")

    print(f"Wrote {len(df)} nuclides to {out_path}")
    print(f"  {(df['k_uSv_m2_MBq_h'] > 0).sum()} with k > 0")
    print(f"  {(df['k_uSv_m2_MBq_h'] == 0).sum()} with k = 0 (pure beta/alpha)")
    for src in ["ensdf", "it-approx", "zero"]:
        n = (df["source"] == src).sum()
        print(f"  source={src}: {n}")

    # Validation
    _validate(df)


_SYMBOLS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am",
]  # fmt: skip


def _validate(df) -> None:
    """Cross-check against published air kerma rate constants.

    References:
    - Ninkovic & Adrovic 2012, Table 1 (air kerma rate constants, δ=20 keV)
    - Smith & Stabin 2012 (exposure rate constants)
    Units: µSv·m²/(MBq·h)
    """
    import polars as pl

    # Reference: Ninkovic & Adrovic 2012, Table 1
    # Units: µSv·m²/(MBq·h)
    reference = {
        # --- PET β+ emitters ---
        (6, 11, ""): 0.1393,  # C-11
        (7, 13, ""): 0.1394,  # N-13
        (8, 15, ""): 0.1395,  # O-15
        (9, 18, ""): 0.1351,  # F-18
        (31, 68, ""): 0.1290,  # Ga-68
        (11, 22, ""): 0.271,  # Na-22 (RADAR)
        # --- SPECT ---
        (43, 99, "m"): 0.01410,  # Tc-99m
        (49, 111, ""): 0.08313,  # In-111
        (49, 113, "m"): 0.04400,  # In-113m (IT validation target)
        (53, 123, ""): 0.0361,  # I-123
        (81, 201, ""): 0.01022,  # Tl-201
        (31, 67, ""): 0.01945,  # Ga-67
        # --- Therapy ---
        (53, 131, ""): 0.05220,  # I-131
        (53, 125, ""): 0.03773,  # I-125
        # --- Calibration / shielding ---
        (27, 60, ""): 0.3090,  # Co-60
        (27, 57, ""): 0.01411,  # Co-57
        (27, 58, ""): 0.1290,  # Co-58
        (55, 137, ""): 0.08210,  # Cs-137 (secular eq. with Ba-137m)
        (24, 51, ""): 0.00422,  # Cr-51
        (77, 192, ""): 0.1091,  # Ir-192
        (34, 75, ""): 0.04825,  # Se-75
        (63, 152, ""): 0.1489,  # Eu-152
        (63, 154, ""): 0.1592,  # Eu-154
        (95, 241, ""): 0.00397,  # Am-241
        # --- Other common ---
        (42, 99, ""): 0.01977,  # Mo-99
        (26, 59, ""): 0.1459,  # Fe-59
        (11, 24, ""): 0.4367,  # Na-24
        (79, 198, ""): 0.05454,  # Au-198
        (19, 42, ""): 0.0328,  # K-42
    }

    print("\nValidation against Ninkovic & Adrovic 2012:")
    print(f"  {'Isotope':<12} {'Computed':>10} {'Ref':>10} {'Error':>8} {'Source'}")
    print(f"  {'-' * 55}")

    n_ok, n_check, n_missing = 0, 0, 0
    for (z, a, state), ref_k in reference.items():
        row = df.filter((pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") == state))
        sym = _SYMBOLS[z] if z < len(_SYMBOLS) else f"Z{z}"
        name = f"{'m' if state else ''}{sym}-{a}"
        if len(row) == 0:
            print(f"  {name:<12} {'—':>10} {ref_k:10.4f} {'N/A':>8}   MISSING")
            n_missing += 1
            continue
        k = row["k_uSv_m2_MBq_h"][0]
        src = row["source"][0]
        err = (k / ref_k - 1) * 100
        status = "OK" if abs(err) < 20 else "CHECK"
        if status == "OK":
            n_ok += 1
        else:
            n_check += 1
        print(f"  {name:<12} {k:10.4f} {ref_k:10.4f} {err:+7.1f}% {status:<6} {src}")

    print(f"\n  Summary: {n_ok} OK, {n_check} CHECK, {n_missing} MISSING (of {len(reference)})")


if __name__ == "__main__":
    build()
