"""Generate heavy-ion stopping power tables via pycatima and save as Parquet.

Pre-computes mass stopping power [MeV cm2/g] for an enumerated set of
projectile isotopes (Z, A) against all elements Z=1-92 as targets.

The energy axis is stored in MeV/u. At fixed MeV/u above ~15 keV/u, dedx is
isotope-agnostic to <1% for medium-to-heavy projectiles and within catima's
quoted uncertainty. Below that threshold, nuclear stopping dominates and the
reduced-mass A_p·A_t/(A_p+A_t) introduces real isotope dependence (up to
~15% for light projectiles). Per-isotope tabulation lets consumers query the
correct row at any energy, eliminating the silent-error path that existed
when only one row per Z was shipped.

Bohr energy straggling variance dΩ²/d(ρx) [MeV² cm²/g] is the high-energy
limit (Bohr 1948); it overestimates at very low β but is acceptable for
therapy-range ions.

Projectile inventory (≈391 isotopes):
  - All stable + primordial isotopes (abundance > 0 in
    data/meta/abundances.parquet) — 287 nuclides
  - All ground-state nuclides with T½ > 1 year in data/meta/decay.parquet —
    includes transuranics (Np, Pu, Am, Cm, Bk, Cf, Es) up to Z=99 and the
    long-lived isotopes of every Z below 92
  - Beam/medical allowlist for short-lived but commonly-used isotopes:
    T (³H), C-11, N-13, O-15, F-18, P-32 (others already covered above)

Targets are restricted to Z=1..92 where pycatima's get_material() returns
a valid molar mass; transuranic projectiles still hit Z≤92 targets fine.

Output: stopping/catima/catima.parquet
        (schema: proj_Z, proj_A, target_Z, energy_MeV_u, dedx, straggling)

Usage:
    uv run python -m nucl_parquet.build_heavy_ions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .download import data_dir as _resolve_data_dir

# Energy grid in MeV/u: 0.001–300, 200 log-spaced points
_ENERGIES_MEV_U: np.ndarray = np.geomspace(0.001, 300.0, 200)

# Bohr energy straggling constant [MeV² cm²/g]:
#   4π N_A r_e² m_e c² = 0.15737 MeV² cm²/mol
# dΩ²/d(ρx) = BOHR_CONST × Z_p² × Z_t / A_t
_BOHR_CONST: float = 0.15737

# Threshold for "long-lived" — any ground state with T½ > 1 year ships.
# Covers transuranics (Pu-239 24kyr, Am-241 432yr, Cf-252 2.6yr, etc.),
# every long-lived radio-anchor (Tc-97, Pm-145, Pa-231, ...) and standard
# nucleosynthesis chain members. Catches Es-254 (276d) by not requiring stability.
_LONGLIVED_THRESHOLD_S: float = 365.0 * 86400.0  # 1 year

# Beam/medical isotopes worth shipping even if their T½ is < 1 year.
_BEAM_ALLOWLIST: list[tuple[int, int]] = [
    (1, 3),  # T (tritium beams, fusion plasma)  T½ 12.3 yr — already long-lived
    (6, 11),  # C-11 (PET)                        T½ 20 min
    (7, 13),  # N-13 (PET)                        T½ 9.97 min
    (8, 15),  # O-15 (PET)                        T½ 2.04 min
    (9, 18),  # F-18 (PET / FDG)                  T½ 110 min
    (15, 32),  # P-32 (therapy)                    T½ 14.3 d
]


def _resolve_projectile_isotopes(data_dir: Path) -> list[tuple[int, int]]:
    """Return sorted list of (proj_Z, proj_A) pairs to bake.

    Union of:
      - stable + primordial isotopes (abundance > 0 in abundances.parquet)
      - all ground-state nuclides with T½ > _LONGLIVED_THRESHOLD_S in decay.parquet
      - short-lived beam/medical allowlist

    Capped to Z=1..99 (catima Bethe-Bloch accepts arbitrary projectile Z, but
    we stop at Es since beyond that no isotope has T½ > 1 yr).
    """
    import polars as pl

    abund = pl.read_parquet(data_dir / "meta" / "abundances.parquet")
    stable = abund.filter(pl.col("abundance") > 0).select("Z", "A").unique()

    decay = pl.read_parquet(data_dir / "meta" / "decay.parquet")
    longlived = (
        decay.filter((pl.col("state") == "") & (pl.col("half_life_s") > _LONGLIVED_THRESHOLD_S))
        .select("Z", "A")
        .unique()
    )

    rows: set[tuple[int, int]] = set()
    rows.update((int(z), int(a)) for z, a in zip(stable["Z"].to_list(), stable["A"].to_list()))
    rows.update((int(z), int(a)) for z, a in zip(longlived["Z"].to_list(), longlived["A"].to_list()))
    rows.update(_BEAM_ALLOWLIST)

    # Fallback: for any Z=1..92 still missing (At, Rn, Fr — every isotope has
    # T½ < 1 yr), pick the longest-lived ground state so the table covers every
    # element. Better a short-lived canonical than a NaN.
    covered_zs = {z for z, _ in rows}
    for z in range(1, 93):
        if z in covered_zs:
            continue
        best = (
            decay.filter((pl.col("Z") == z) & (pl.col("state") == ""))
            .sort("half_life_s", descending=True, nulls_last=True)
            .select("A")
            .head(1)
        )
        if best.height == 0:
            raise RuntimeError(f"No ground-state decay row for Z={z}; cannot pick canonical isotope")
        rows.add((z, int(best["A"][0])))

    return sorted(p for p in rows if 1 <= p[0] <= 99)


def build(data_dir: Path | None = None) -> None:
    """Generate catima stopping tables for the enumerated isotope set."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl
    import pycatima as catima

    isotopes = _resolve_projectile_isotopes(data_dir)
    n_target = 92
    n_energy = len(_ENERGIES_MEV_U)
    total = len(isotopes) * n_target * n_energy
    print(
        f"Building catima tables: {len(isotopes)} isotopes × {n_target} targets "
        f"× {n_energy} energies = {total:,} rows\n"
    )

    # Verify every Z=1..92 has at least one isotope (sanity check).
    z_covered = {z for z, _ in isotopes}
    missing_z = sorted(set(range(1, 93)) - z_covered)
    if missing_z:
        raise RuntimeError(f"No isotope for Z in {missing_z}")

    out_path = data_dir / "stopping" / "catima" / "catima.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_proj_z: list[int] = []
    all_proj_a: list[int] = []
    all_target_z: list[int] = []
    all_energies: list[float] = []
    all_dedxs: list[float] = []
    all_strag: list[float] = []

    # Pre-compute target atomic weights from pycatima for Bohr straggling.
    target_A: dict[int, float] = {}
    for z in range(1, 93):
        target_A[z] = catima.get_material(z).molar_mass()

    last_z = None
    for proj_Z, proj_A in isotopes:
        proj = catima.Projectile(proj_A, proj_Z)
        for target_Z in range(1, 93):
            mat = catima.get_material(target_Z)
            # Bohr straggling: dΩ²/d(ρx) = const × Z_p² × Z_t / A_t
            strag = _BOHR_CONST * proj_Z**2 * target_Z / target_A[target_Z]
            for e in _ENERGIES_MEV_U:
                proj.T(float(e))
                all_proj_z.append(proj_Z)
                all_proj_a.append(proj_A)
                all_target_z.append(target_Z)
                all_energies.append(float(e))
                all_dedxs.append(catima.dedx(proj, mat))
                all_strag.append(strag)
        if proj_Z != last_z:
            print(f"  Z={proj_Z:2d}: A={proj_A} done", flush=True)
            last_z = proj_Z
        else:
            print(f"          A={proj_A} done", flush=True)

    df = pl.DataFrame(
        {
            "proj_Z": pl.Series(all_proj_z, dtype=pl.Int32),
            "proj_A": pl.Series(all_proj_a, dtype=pl.Int32),
            "target_Z": pl.Series(all_target_z, dtype=pl.Int32),
            "energy_MeV_u": pl.Series(all_energies, dtype=pl.Float64),
            "dedx": pl.Series(all_dedxs, dtype=pl.Float64),
            "straggling": pl.Series(all_strag, dtype=pl.Float64),
        }
    ).sort("proj_Z", "proj_A", "target_Z", "energy_MeV_u")

    df.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    build()
