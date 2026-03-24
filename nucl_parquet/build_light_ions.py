"""Derive stopping power for d, t, ³He from PSTAR/ASTAR via velocity scaling.

Electronic stopping depends only on projectile Z and velocity (MeV/u), not on A.
Therefore, at the same MeV/u:
  - p, d, t (Z=1) all have the same stopping as PSTAR
  - ³He, α (Z=2) all have the same stopping as ASTAR

For a given *total* energy E of the heavy isotope, the equivalent proton/alpha
energy at the same velocity is:

    E_p  = E_d / 2       (deuteron, A=2)
    E_p  = E_t / 3       (triton, A=3)
    E_α  = E_He3 × 4/3   (helion ³He, A=3; α has A=4)

So we re-label the PSTAR/ASTAR energy axis by the scaling factor — no
interpolation is required; each existing (target_Z, dedx) row is preserved
exactly, with only the energy_MeV column multiplied.

Output: stopping/dSTAR.parquet, stopping/tSTAR.parquet, stopping/He3STAR.parquet
        (one file per derived source; idempotent)

Usage:
    uv run python -m nucl_parquet.build_light_ions
"""

from __future__ import annotations

from pathlib import Path

from .download import data_dir as _resolve_data_dir

# (source_name, base_source, energy_scale_factor)
# energy_MeV_new = energy_MeV_base × scale
_LIGHT_IONS: list[tuple[str, str, float]] = [
    ("dSTAR",   "PSTAR", 2.0),         # deuteron: E_d = E_p × 2
    ("tSTAR",   "PSTAR", 3.0),         # triton:   E_t = E_p × 3
    ("He3STAR", "ASTAR", 3.0 / 4.0),   # ³He:      E_He3 = E_α × 3/4
]


def build(data_dir: Path | None = None) -> None:
    """Derive d/t/³He stopping files from PSTAR/ASTAR in stopping/."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    stopping_dir = data_dir / "stopping"

    for name, base_source, scale in _LIGHT_IONS:
        base_path = stopping_dir / f"{base_source}.parquet"
        if not base_path.exists():
            print(f"  {name}: base file {base_path.name} not found — skipped")
            continue
        src_df = pl.read_parquet(base_path)
        derived = src_df.with_columns([
            pl.lit(name).alias("source"),
            (pl.col("energy_MeV") * scale).alias("energy_MeV"),
        ]).sort("target_Z", "energy_MeV")
        out_path = stopping_dir / f"{name}.parquet"
        derived.write_parquet(out_path, compression="zstd")
        print(f"  {name}: {len(derived):,} rows (scaled from {base_source} × {scale:.4g}) → {out_path.name}")


if __name__ == "__main__":
    build()
