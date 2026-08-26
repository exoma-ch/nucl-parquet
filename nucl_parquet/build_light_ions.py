"""Derive stopping power for d, t from PSTAR via velocity scaling.

Electronic stopping depends only on projectile Z and velocity (MeV/u), not on A.
Therefore, at the same MeV/u, p / d / t (Z=1) all share the PSTAR curve.

For a given *total* energy E of the heavy isotope, the equivalent proton
energy at the same velocity is:

    E_p = E_d / 2       (deuteron, A=2)
    E_p = E_t / 3       (triton, A=3)

So we re-label the PSTAR energy axis by the scaling factor — no interpolation
is required; each existing (target_Z, dedx) row is preserved exactly, with
only the energy_MeV column multiplied.

³He no longer derives from ASTAR — the prior He3STAR.parquet inherited a
Z²-at-wrong-axis bug from the broken ASTAR.parquet (#137). ³He now routes
directly through catima in the loader (no NIST ³He table exists). α stopping
comes from NIST ASTAR via build_stopping.py.

Output: stopping/dSTAR.parquet, stopping/tSTAR.parquet
        (one file per derived source; idempotent)

Usage:
    uv run python -m nucl_parquet.build_light_ions
"""

from __future__ import annotations

from pathlib import Path

from .download import writable_data_dir as _resolve_data_dir

# (source_name, base_source, energy_scale_factor)
# energy_MeV_new = energy_MeV_base × scale
_LIGHT_IONS: list[tuple[str, str, float]] = [
    ("dSTAR", "PSTAR", 2.0),  # deuteron: E_d = E_p × 2
    ("tSTAR", "PSTAR", 3.0),  # triton:   E_t = E_p × 3
]


def build(data_dir: Path | None = None) -> None:
    """Derive d/t stopping files from PSTAR in stopping/."""
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
        derived = src_df.with_columns(
            [
                pl.lit(name).alias("source"),
                (pl.col("energy_MeV") * scale).alias("energy_MeV"),
            ]
        ).sort("target_Z", "energy_MeV")
        out_path = stopping_dir / f"{name}.parquet"
        derived.write_parquet(out_path, compression="zstd")
        print(f"  {name}: {len(derived):,} rows (scaled from {base_source} × {scale:.4g}) → {out_path.name}")


if __name__ == "__main__":
    build()
