"""Compute spectrum-averaged neutron cross-sections for standard neutron spectra.

Reads all neutron XS parquet files from every evaluated library listed in
catalog.json and computes flux-weighted averages over three standard spectra:

  thermal    - 1/v weighting φ(E) = 1/√E,  E < 0.1 MeV
  epithermal - 1/E slowing-down φ(E) = 1/E
  fast       - Watt fission φ(E) = exp(-E/0.988)·sinh(√(2.249·E))

Output: data/meta/spectrum_xs.parquet

Schema: target_Z int32, target_A int32, residual_Z int32, residual_A int32,
        state str, spectrum str, xs_avg_mb f64, library str

Usage:
    uv run python -m nucl_parquet.build_spectrum_xs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .download import data_dir as _resolve_data_dir
from .loader import _SYMBOL_TO_Z as _SYMBOL_TO_Z_LOWER

# Re-key the shared map with capitalized symbols (filenames use "n_Cu.parquet" style)
_SYMBOL_TO_Z: dict[str, int] = {sym.capitalize(): z for sym, z in _SYMBOL_TO_Z_LOWER.items()}

# --- Spectrum definitions ---
# All energies in MeV.  Each entry: (name, E_lo, E_hi, flux_fn)


def _flux_thermal(e: np.ndarray) -> np.ndarray:
    # phi ∝ 1/v ∝ 1/√E: a strict Maxwell-Boltzmann (kT≈0.025 eV) would be
    # numerically zero at the minimum energies stored in evaluated libraries
    # (~10-100 keV), so 1/v weighting is used instead — it correctly
    # down-weights higher-energy contributions within the sub-100 keV window.
    return 1.0 / np.sqrt(e)


def _flux_epithermal(e: np.ndarray) -> np.ndarray:
    return 1.0 / e


def _flux_fast(e: np.ndarray) -> np.ndarray:
    # Watt fission spectrum (U-235): a=0.988 MeV, b=2.249 MeV^-1
    return np.exp(-e / 0.988) * np.sinh(np.sqrt(2.249 * e))


_SPECTRA: list[tuple[str, float, float, object]] = [
    ("thermal", 0.0, 0.1, _flux_thermal),
    ("epithermal", 5e-7, 1e-4, _flux_epithermal),
    ("fast", 0.1, 20.0, _flux_fast),
]

_MIN_POINTS = 3  # minimum energy grid points inside window to emit a row


def _spectrum_average(
    energy: np.ndarray,
    xs: np.ndarray,
    e_lo: float,
    e_hi: float,
    flux_fn: object,
) -> float | None:
    """Return flux-weighted average XS or None if too few grid points."""
    mask = (energy >= e_lo) & (energy <= e_hi)
    e = energy[mask]
    s = xs[mask]
    if len(e) < _MIN_POINTS:
        return None
    phi = flux_fn(e)  # type: ignore[operator]
    denom = np.trapezoid(phi, e)
    if denom == 0.0:
        return None
    return float(np.trapezoid(s * phi, e) / denom)


def build(data_dir: Path | None = None) -> None:
    """Read all neutron XS parquet files and write data/meta/spectrum_xs.parquet."""
    import polars as pl

    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    catalog_path = data_dir / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    libraries = catalog.get("libraries", catalog)

    # Collect rows across all libraries and files
    rows: list[dict] = []

    for lib_key, lib_info in libraries.items():
        if not isinstance(lib_info, dict):
            continue
        if "n" not in lib_info.get("projectiles", []):
            continue
        # Skip experimental data — only evaluated libraries
        if lib_info.get("data_type") == "experimental_cross_sections":
            continue

        xs_dir = data_dir / lib_info["path"]
        if not xs_dir.is_dir():
            continue

        neutron_files = sorted(xs_dir.glob("n_*.parquet"))
        if not neutron_files:
            continue

        print(f"  {lib_key}: {len(neutron_files)} neutron files")

        for pq_path in neutron_files:
            # Extract target symbol from filename: n_Cu.parquet → Cu
            stem = pq_path.stem  # "n_Cu"
            symbol = stem[2:]  # "Cu"
            target_Z = _SYMBOL_TO_Z.get(symbol)
            if target_Z is None:
                print(f"    WARNING: unknown symbol '{symbol}' in {pq_path.name}, skipping")
                continue

            df = pl.read_parquet(pq_path)

            # Spectrum-averaging is per production channel. Since #347 these
            # tables also carry `kind='channel'` rows, and every one that names
            # no residual — MT=1, 2, 4, 18, 19, 20, 21, 38 — shares the group key
            # (target_A, NULL, NULL, ''), because polars treats null as a
            # distinct-but-equal key. Without this filter the total, elastic,
            # inelastic and fission curves are interleaved, sorted by energy and
            # integrated as though they were one reaction, and the result is
            # written out as a row naming no residual — wrong, and invisible to
            # any downstream query that filters on residual.
            if "kind" in df.columns:
                df = df.filter(pl.col("kind") == "production")

            reaction_cols = ["target_A", "residual_Z", "residual_A", "state"]
            for group_keys, group_df in df.group_by(reaction_cols):
                target_A, residual_Z, residual_A, state = group_keys

                sub = group_df.sort("energy_MeV")
                energy = sub["energy_MeV"].to_numpy()
                xs_mb = sub["xs_mb"].to_numpy()

                for spec_name, e_lo, e_hi, flux_fn in _SPECTRA:
                    avg = _spectrum_average(energy, xs_mb, e_lo, e_hi, flux_fn)
                    if avg is None:
                        continue
                    rows.append(
                        {
                            "target_Z": target_Z,
                            "target_A": target_A,
                            "residual_Z": residual_Z,
                            "residual_A": residual_A,
                            # Passed straight through from the source xs row.
                            # This used to coerce null to '', minting a fourth
                            # meaning for those four bytes in a table that is
                            # otherwise a faithful roll-up (#357). Null means
                            # the source did not say; say the same thing.
                            "state": state,
                            "spectrum": spec_name,
                            "xs_avg_mb": avg,
                            "library": lib_key,
                        }
                    )

    if not rows:
        print("No data produced — check that neutron XS parquet files are present.")
        return

    out_df = pl.DataFrame(
        rows,
        schema={
            "target_Z": pl.Int32,
            "target_A": pl.Int32,
            "residual_Z": pl.Int32,
            "residual_A": pl.Int32,
            "state": pl.Utf8,
            "spectrum": pl.Utf8,
            "xs_avg_mb": pl.Float64,
            "library": pl.Utf8,
        },
    ).sort("library", "target_Z", "target_A", "residual_Z", "residual_A", "spectrum")

    out_path = data_dir / "meta" / "spectrum_xs.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path, compression="zstd")
    print(f"\nWrote {len(out_df):,} rows to {out_path}")


if __name__ == "__main__":
    build()
