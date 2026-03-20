"""Generate heavy-ion stopping power tables via pycatima and save as Parquet.

Pre-computes mass stopping power [MeV cm2/g] for all elements Z=1-92 as
projectiles against all elements Z=1-92 as targets.

Stopping power at a given MeV/u depends only on projectile Z and velocity,
not on the specific isotope — so energy is stored in MeV/u. Any isotope of
element Z can be looked up by converting total MeV → MeV/u (divide by A).

Output: stopping/catima.parquet  (schema: proj_Z, target_Z, energy_MeV_u, dedx)

Usage:
    uv run python -m nucl_parquet.build_heavy_ions
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .download import data_dir as _resolve_data_dir

# Energy grid in MeV/u: 0.001–300, 200 log-spaced points
_ENERGIES_MEV_U: np.ndarray = np.geomspace(0.001, 300.0, 200)


def build(data_dir: Path | None = None) -> None:
    """Generate catima stopping tables for all 92 projectile elements."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import pycatima as catima
    import polars as pl

    n_proj = 92
    n_target = 92
    n_energy = len(_ENERGIES_MEV_U)
    total = n_proj * n_target * n_energy
    print(f"Building catima tables: {n_proj} × {n_target} × {n_energy} = {total:,} rows\n")

    out_path = data_dir / "stopping" / "catima.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_proj_zs: list[int] = []
    all_target_zs: list[int] = []
    all_energies: list[float] = []
    all_dedxs: list[float] = []

    # A only affects velocity at a given total energy — at fixed MeV/u, dedx is
    # isotope-independent. We use a representative A (≈2Z) purely to initialise
    # the catima Projectile object; it does not affect the stored dedx values.
    for proj_Z in range(1, 93):
        proj_A = max(1, round(proj_Z * 2.0))
        proj = catima.Projectile(proj_A, proj_Z)

        for target_Z in range(1, 93):
            mat = catima.get_material(target_Z)
            for e in _ENERGIES_MEV_U:
                proj.T(float(e))
                all_proj_zs.append(proj_Z)
                all_target_zs.append(target_Z)
                all_energies.append(float(e))
                all_dedxs.append(catima.dedx(proj, mat))

        print(f"  Z={proj_Z:2d}: done")

    df = pl.DataFrame(
        {
            "proj_Z":       pl.Series(all_proj_zs,  dtype=pl.Int32),
            "target_Z":     pl.Series(all_target_zs, dtype=pl.Int32),
            "energy_MeV_u": pl.Series(all_energies,  dtype=pl.Float64),
            "dedx":         pl.Series(all_dedxs,     dtype=pl.Float64),
        }
    ).sort("proj_Z", "target_Z", "energy_MeV_u")

    df.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    build()
