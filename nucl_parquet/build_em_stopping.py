"""Build electron-stopping tables from strata-data EM parquets.

Reads the three strata em/*.parquet files fetched by
`scripts/fetch_strata_nuclear.py` (gerchowl/strata-data, em/ subdir) and emits:

1. `data/stopping/ESTAR.parquet` — legacy schema rebuild
   `{source, target_Z, energy_MeV, dedx}`, elements only. Drop-in replacement
   for the v0.5 NIST scrape — every `WHERE source='ESTAR'` query still works.
   `dedx` is total mass stopping power (collision + radiative).

2. `data/stopping/em/electron_stopping.parquet` — rich schema, NEW
   `{name, is_element, target_Z, energy_MeV, collision_sp_mev_cm2_g,
     radiative_sp_mev_cm2_g, total_sp_mev_cm2_g, density_effect_delta}`.
   Covers all 98 elements AND ~180 compounds (water, air, soft tissue,
   bone, common detectors, etc.). target_Z is NULL for compounds.

3. `data/stopping/em/density_effect_params.parquet` — NEW
   Sternheimer parameterization per material
   `{name, I_eV, C, X0, X1, a, k, delta0, density_gcm3, state, Zeff, nElements}`.
   Names use the G4_<symbol>/G4_<COMPOUND> convention from PhysicsList; this
   does NOT 1:1 align with electron_stopping.name yet (which uses bare element-Z
   strings and uppercase ICRU compound names). Cross-references are by Zeff
   for elements (Zeff == Z) and by-name for compounds — TODO cross-walk in a
   followup.

Energy axis is the strata `estar_long.parquet` grid (energies 0.001–10000 MeV).

Usage:
    uv run python -m nucl_parquet.build_em_stopping
"""

from __future__ import annotations

from pathlib import Path

from .download import data_dir as _resolve_data_dir

_STRATA_DIR_REL = ("g4_raw", "strata-nuclear")


def _strata_dir(data_dir: Path) -> Path:
    return data_dir.joinpath(*_STRATA_DIR_REL)


def build(data_dir: Path | None = None) -> None:
    """Build all three electron-stopping outputs from strata sources."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    src = _strata_dir(data_dir)
    estar_long = src / "estar_long.parquet"
    density_effect = src / "density_effect.parquet"
    for p in (estar_long, density_effect):
        if not p.exists():
            raise FileNotFoundError(f"Missing strata source {p}. Run `python scripts/fetch_strata_nuclear.py` first.")

    estar = pl.read_parquet(estar_long)
    de = pl.read_parquet(density_effect)

    # --- Output 1: legacy ESTAR.parquet ---
    # Elements only, four columns matching the v0.5 schema. dedx is total
    # mass stopping power (NIST tabulation also gives total).
    legacy_estar = (
        estar.filter(pl.col("is_element") & (pl.col("z") >= 1))
        .select(
            pl.lit("ESTAR").alias("source"),
            pl.col("z").cast(pl.Int32).alias("target_Z"),
            pl.col("energy_mev").alias("energy_MeV"),
            pl.col("total_sp_mev_cm2_g").alias("dedx"),
        )
        .sort("target_Z", "energy_MeV")
    )
    legacy_out = data_dir / "stopping" / "ESTAR.parquet"
    legacy_out.parent.mkdir(parents=True, exist_ok=True)
    legacy_estar.write_parquet(legacy_out, compression="zstd")
    print(f"  ESTAR.parquet (legacy): {len(legacy_estar):,} rows → {legacy_out}")

    # --- Output 2: electron_stopping.parquet ---
    # Full strata schema — elements + compounds, collision/radiative split.
    # target_Z populated for elements, NULL for compounds.
    rich = estar.select(
        pl.col("name"),
        pl.col("is_element"),
        pl.when(pl.col("is_element")).then(pl.col("z").cast(pl.Int32)).otherwise(None).alias("target_Z"),
        pl.col("energy_mev").alias("energy_MeV"),
        pl.col("collision_sp_mev_cm2_g"),
        pl.col("radiative_sp_mev_cm2_g"),
        pl.col("total_sp_mev_cm2_g"),
        pl.col("density_effect_delta"),
    ).sort("is_element", "target_Z", "name", "energy_MeV")
    rich_out = data_dir / "stopping" / "em" / "electron_stopping.parquet"
    rich_out.parent.mkdir(parents=True, exist_ok=True)
    rich.write_parquet(rich_out, compression="zstd")
    n_elem = rich.filter(pl.col("is_element"))["name"].n_unique()
    n_comp = rich.filter(~pl.col("is_element"))["name"].n_unique()
    print(f"  electron_stopping.parquet: {len(rich):,} rows ({n_elem} elements + {n_comp} compounds) → {rich_out}")

    # --- Output 3: density_effect_params.parquet ---
    # Drop strata's row `index` (not load-bearing — sort key on output).
    # Rename to project conventions where useful.
    de_out_df = de.select(
        pl.col("name"),
        pl.col("I_eV"),
        pl.col("C"),
        pl.col("X0"),
        pl.col("X1"),
        pl.col("a"),
        pl.col("k"),
        pl.col("delta0"),
        pl.col("density_gcm3"),
        pl.col("state"),
        pl.col("Zeff"),
        pl.col("nElements"),
    ).sort("Zeff", "name")
    de_path = data_dir / "stopping" / "em" / "density_effect_params.parquet"
    de_out_df.write_parquet(de_path, compression="zstd")
    print(f"  density_effect_params.parquet: {len(de_out_df):,} rows → {de_path}")


if __name__ == "__main__":
    build()
