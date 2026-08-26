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

Output: stopping/catima_<Sym><A>.parquet — one federated shard per projectile
        isotope (e.g. catima_C12.parquet), build-generated (not hand-committed).
        Schema: source, proj_Z, proj_A, target_Z, energy_MeV_u, energy_MeV, dedx,
        straggling. Replaces the former single 92×92 monolith (#252).

Usage:
    uv run python -m nucl_parquet.build_heavy_ions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .download import writable_data_dir as _resolve_data_dir

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


# Element symbols Z=1..99 — projectiles range up to Es (Z=99, the heaviest
# isotope with T½ > 1 yr). build_hi_xs.Z_TO_SYMBOL stops at 92, so define the
# full range here.
_ELEMENT_SYMBOLS: list[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
]  # fmt: skip
_Z_TO_SYMBOL: dict[int, str] = {i + 1: s for i, s in enumerate(_ELEMENT_SYMBOLS)}


def shard_name(proj_Z: int, proj_A: int) -> str:
    """Federated shard filename stem for a projectile isotope, e.g. ``catima_C12``.

    Used as both the file stem (``<stem>.parquet``) and the ``source`` column
    value, so the shard is reachable via ``StoppingDb::nist_table(source, …)``.
    """
    return f"catima_{_Z_TO_SYMBOL[proj_Z]}{proj_A}"


def write_isotope_shard(
    stopping_dir: Path,
    proj_Z: int,
    proj_A: int,
    target_z: list[int],
    energy_MeV_u: list[float],
    dedx: list[float],
    straggling: list[float],
) -> Path:
    """Write one federated per-isotope catima shard and return its path.

    Schema: ``source, proj_Z, proj_A, target_Z, energy_MeV_u, energy_MeV, dedx,
    straggling``. Both energy columns are stored — ``energy_MeV_u`` (catima's
    native MeV/u axis, used by the loaders' isotope-keyed path) and
    ``energy_MeV`` (total kinetic energy = MeV/u × A, used when the shard is read
    as a NIST-style source) — so every consumer reads its native unit with no
    conversion. They are perfectly correlated, so zstd stores the pair cheaply.
    """
    import polars as pl

    source = shard_name(proj_Z, proj_A)
    n = len(target_z)
    df = pl.DataFrame(
        {
            "source": pl.Series([source] * n),
            "proj_Z": pl.Series([proj_Z] * n, dtype=pl.Int32),
            "proj_A": pl.Series([proj_A] * n, dtype=pl.Int32),
            "target_Z": pl.Series(target_z, dtype=pl.Int32),
            "energy_MeV_u": pl.Series(energy_MeV_u, dtype=pl.Float64),
            "energy_MeV": pl.Series([e * proj_A for e in energy_MeV_u], dtype=pl.Float64),
            "dedx": pl.Series(dedx, dtype=pl.Float64),
            "straggling": pl.Series(straggling, dtype=pl.Float64),
        }
    ).sort("target_Z", "energy_MeV_u")
    out_path = stopping_dir / f"{source}.parquet"
    df.write_parquet(out_path, compression="zstd")
    return out_path


def build(data_dir: Path | None = None) -> None:
    """Generate federated per-isotope catima stopping shards.

    Writes one ``stopping/catima_<Sym><A>.parquet`` per projectile isotope (build
    output, not hand-committed). Replaces the former single 92×92 monolith — see
    #252; the shards are the only catima representation now.
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import pycatima as catima

    isotopes = _resolve_projectile_isotopes(data_dir)
    n_target = 92
    n_energy = len(_ENERGIES_MEV_U)
    total = len(isotopes) * n_target * n_energy
    print(
        f"Building catima shards: {len(isotopes)} isotopes × {n_target} targets "
        f"× {n_energy} energies = {total:,} rows ({len(isotopes)} files)\n"
    )

    # Verify every Z=1..92 has at least one isotope (sanity check).
    z_covered = {z for z, _ in isotopes}
    missing_z = sorted(set(range(1, 93)) - z_covered)
    if missing_z:
        raise RuntimeError(f"No isotope for Z in {missing_z}")

    stopping_dir = data_dir / "stopping"
    stopping_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute target atomic weights from pycatima for Bohr straggling.
    target_A: dict[int, float] = {z: catima.get_material(z).molar_mass() for z in range(1, 93)}

    last_z = None
    for proj_Z, proj_A in isotopes:
        proj = catima.Projectile(proj_A, proj_Z)
        target_z: list[int] = []
        energies: list[float] = []
        dedxs: list[float] = []
        strag: list[float] = []
        for target_Z in range(1, 93):
            mat = catima.get_material(target_Z)
            # Bohr straggling: dΩ²/d(ρx) = const × Z_p² × Z_t / A_t
            s = _BOHR_CONST * proj_Z**2 * target_Z / target_A[target_Z]
            for e in _ENERGIES_MEV_U:
                proj.T(float(e))
                target_z.append(target_Z)
                energies.append(float(e))
                dedxs.append(catima.dedx(proj, mat))
                strag.append(s)
        write_isotope_shard(stopping_dir, proj_Z, proj_A, target_z, energies, dedxs, strag)
        if proj_Z != last_z:
            print(f"  Z={proj_Z:2d}: A={proj_A} done", flush=True)
            last_z = proj_Z
        else:
            print(f"          A={proj_A} done", flush=True)

    print(f"\nWrote {len(isotopes)} catima shards to {stopping_dir}")


if __name__ == "__main__":
    build()
