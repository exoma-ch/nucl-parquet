"""Build the Seltzer-Berger bremsstrahlung DCS table from strata-data (#118).

Geant4 ships the Seltzer-Berger (1985) differential cross section dσ/dκ for
electron-bremsstrahlung photon emission, tabulated on a grid of
(target_Z, electron_kinetic_energy, κ = E_γ / T_electron). The DCS is the
load-bearing input for Monte Carlo electron transport: it lets you sample the
emitted photon energy at any electron energy, not just total σ.

strata republishes this from G4EMLOW's `brem_SB.dat` as
`em/brem_sb_dcs.parquet`. We mirror to `data/em/electron_brem_sb_dcs.parquet`
with project-conventional column names.

Output schema:
    target_Z          int32  (1..92 per G4EMLOW)
    electron_kev      float64 (incident electron kinetic energy in keV)
    kappa             float64 (E_γ / T_electron, dimensionless, in (0, 1])
    dcs               float64 (SCALED differential cross section per Berger &
                               Seltzer (1985): dcs = (dσ/dκ) × β² / Z². The
                               Z² is factored out so values are O(1-10) across
                               all Z. To recover the raw dσ/dκ, multiply by Z²/β².)

Energy grid: 32 incident energies (1 keV .. 10 GeV log-spaced); 32 κ values.
Total rows: Z=1..99 × ~32 × ~32 ≈ 100k.

Companion to:
- `data/em/electron_brem.parquet` (integrated σ(Z, T_electron); already shipped)

Issue: https://github.com/exoma-ch/nucl-parquet/issues/118

Usage:
    uv run python -m nucl_parquet.build_em_brem_dcs
"""

from __future__ import annotations

from pathlib import Path

from .download import data_dir as _resolve_data_dir

_STRATA_DIR_REL = ("g4_raw", "strata-nuclear")
_SRC_BASENAME = "brem_sb_dcs.parquet"


def build(data_dir: Path | None = None) -> None:
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    src = data_dir.joinpath(*_STRATA_DIR_REL, _SRC_BASENAME)
    if not src.exists():
        raise FileNotFoundError(f"Missing strata source {src}. Run `python scripts/fetch_strata_nuclear.py` first.")

    raw = pl.read_parquet(src)

    out = raw.select(
        pl.col("z").cast(pl.Int32).alias("target_Z"),
        pl.col("incident_energy_kev").alias("electron_kev"),
        pl.col("kappa"),
        pl.col("dcs"),
    ).sort("target_Z", "electron_kev", "kappa")

    out_path = data_dir / "em" / "electron_brem_sb_dcs.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path, compression="zstd")

    n_z = out["target_Z"].n_unique()
    n_energy = out["electron_kev"].n_unique()
    n_kappa = out["kappa"].n_unique()
    print(
        f"  electron_brem_sb_dcs.parquet: {len(out):,} rows "
        f"({n_z} elements × {n_energy} energies × {n_kappa} κ values) → {out_path}"
    )


if __name__ == "__main__":
    build()
