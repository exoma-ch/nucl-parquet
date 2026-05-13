"""Port the last three NUDEX tables from strata-data (#77 close-out).

15 of the 18 NUDEX-derived tables listed in #77 are already shipped under
data/meta/ (capture_gammas{,_summary}, icc_factors, level_density_bfm/ctm/
params, nudex_isotopes, nudex_level_gammas, nudex_levels, psf_e1, psf_gdr_*,
psf_photonuclear). The remaining three:

1. `nudex_shellcor_ms.parquet` — microscopic shell corrections + ground-state
   deformation parameters (β₂, β₄) per (Z, A). Used by Hauser-Feshbach codes.
2. `nudex_special_inputs.parquet` — per-isotope NUDEX configuration overrides.
3. `nudex_general_stat.parquet` — NUDEX runtime statistics defaults/overrides.

Output:
- `data/meta/nudex_shellcor.parquet` (renamed from strata's `_ms` suffix)
- `data/meta/nudex_special_inputs.parquet`
- `data/meta/nudex_general_stat.parquet`

Issue: https://github.com/exoma-ch/nucl-parquet/issues/77

Usage:
    uv run python -m nucl_parquet.build_nudex_misc
"""

from __future__ import annotations

from pathlib import Path

from .download import data_dir as _resolve_data_dir

_STRATA_DIR_REL = ("g4_raw", "strata-nuclear")
_SOURCES: list[tuple[str, str]] = [
    # (strata basename, project basename under data/meta/)
    ("nudex_shellcor_ms.parquet", "nudex_shellcor.parquet"),
    ("nudex_special_inputs.parquet", "nudex_special_inputs.parquet"),
    ("nudex_general_stat.parquet", "nudex_general_stat.parquet"),
]


def build(data_dir: Path | None = None) -> None:
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    import polars as pl

    src_dir = data_dir.joinpath(*_STRATA_DIR_REL)
    out_dir = data_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    for src_name, out_name in _SOURCES:
        src = src_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing strata source {src}. Run `python scripts/fetch_strata_nuclear.py` first.")
        df = pl.read_parquet(src)

        if src_name == "nudex_shellcor_ms.parquet":
            # Rename lowercase strata columns to project conventions.
            df = (
                df.rename(
                    {
                        "z": "Z",
                        "a": "A",
                        "El": "symbol",
                        "Shell_mev": "shell_MeV",
                        "Defcor_mev": "deformation_correction_MeV",
                        "bet2": "beta2",
                        "bet4": "beta4",
                    }
                )
                .with_columns(pl.col("Z").cast(pl.Int32), pl.col("A").cast(pl.Int32))
                .sort("Z", "A")
            )
        elif src_name == "nudex_general_stat.parquet":
            # Consistent (Z, A) capitalization with the rest of the meta tables.
            df = df.rename({"z": "Z", "a": "A"}).with_columns(pl.col("Z").cast(pl.Int32), pl.col("A").cast(pl.Int32))

        out_path = out_dir / out_name
        df.write_parquet(out_path, compression="zstd")
        print(f"  {out_name}: {len(df):,} rows → {out_path}")


if __name__ == "__main__":
    build()
