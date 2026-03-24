"""Build spectrum-averaged neutron cross-sections.

Computes ⟨σ⟩ = ∫σ(E)·φ(E)dE / ∫φ(E)dE for three standard neutron spectra:

    thermal     Maxwellian, kT = 0.0253 eV     ~1e-11 – 5e-7 MeV
    epithermal  1/E (slowing-down)               5e-7  – 0.1 MeV
    fast        Watt fission (U-235 params)       0.1   – 20 MeV

Output: data/meta/spectrum_xs.parquet
Schema: target_Z, target_A, residual_Z, residual_A, state, spectrum,
        xs_avg_mb, library
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from .download import data_dir as _resolve_data_dir

# Thermal: Maxwellian kT = 0.0253 eV = 2.53e-8 MeV
_kT = 0.0253e-6  # MeV

# Watt fission spectrum parameters (U-235)
_WATT_A = 0.988   # MeV
_WATT_B = 2.249   # MeV^-1

# Integration energy bounds [MeV]
_BOUNDS = {
    "thermal":    (1e-11, 5e-7),    # 0 – ~20×kT
    "epithermal": (5e-7,  0.1),     # 0.5 eV – 100 keV
    "fast":       (0.1,   20.0),    # fission spectrum peak region
}

_N_FINE = 2000  # points in interpolation grid per decade


def _phi_thermal(E: np.ndarray) -> np.ndarray:
    return E * np.exp(-E / _kT)


def _phi_epithermal(E: np.ndarray) -> np.ndarray:
    return 1.0 / E


def _phi_fast(E: np.ndarray) -> np.ndarray:
    sqrt_arg = np.maximum(_WATT_B * E, 0.0)
    return np.exp(-E / _WATT_A) * np.sinh(np.sqrt(sqrt_arg))


_PHI = {
    "thermal":    _phi_thermal,
    "epithermal": _phi_epithermal,
    "fast":       _phi_fast,
}

# Element symbol → Z (H=1 … U=92)
_SYMBOL_TO_Z: dict[str, int] = {
    sym.lower(): z for z, sym in enumerate([
        "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
        "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U",
    ], start=1)
}


def _spectrum_avg(
    E_data: np.ndarray,
    xs_data: np.ndarray,
    spectrum: str,
) -> float:
    """Compute ⟨σ⟩ = ∫σ(E)φ(E)dE / ∫φ(E)dE over the spectrum's energy range."""
    lo, hi = _BOUNDS[spectrum]
    phi_fn = _PHI[spectrum]

    # Narrow data to the integration window
    mask = (E_data >= lo) & (E_data <= hi) & np.isfinite(xs_data) & (xs_data >= 0)
    E = E_data[mask]
    xs = xs_data[mask]
    if len(E) < 2:
        return float("nan")

    # Log-spaced grid spanning the data range within [lo, hi]
    n_decades = np.log10(E[-1] / E[0])
    n_pts = max(200, int(_N_FINE * n_decades))
    E_fine = np.geomspace(E[0], E[-1], n_pts)

    # Interpolate XS log-log onto fine grid (linear outside range → zeros)
    xs_fine = np.exp(np.interp(np.log(E_fine), np.log(E), np.log(np.maximum(xs, 1e-300))))

    phi_fine = phi_fn(E_fine)
    denom = np.trapezoid(phi_fine, E_fine)
    if denom == 0:
        return float("nan")
    return float(np.trapezoid(xs_fine * phi_fine, E_fine) / denom)


def _process_file(path: Path, target_Z: int, library: str) -> pl.DataFrame | None:
    """Compute spectrum averages for all reactions in one XS parquet file."""
    df = pl.read_parquet(path)
    # Keep only rows with valid energy and xs
    df = df.filter(pl.col("xs_mb").is_finite() & (pl.col("xs_mb") >= 0) & (pl.col("xs_mb") < 1e30))

    groups = df.group_by(["target_A", "residual_Z", "residual_A", "state"])
    rows: list[dict] = []

    for keys, grp in groups:
        target_A, residual_Z, residual_A, state = keys
        grp_sorted = grp.sort("energy_MeV")
        E = grp_sorted["energy_MeV"].to_numpy()
        xs = grp_sorted["xs_mb"].to_numpy()

        for spectrum in ("thermal", "epithermal", "fast"):
            avg = _spectrum_avg(E, xs, spectrum)
            if not np.isnan(avg):
                rows.append({
                    "target_Z":    target_Z,
                    "target_A":    int(target_A),
                    "residual_Z":  int(residual_Z),
                    "residual_A":  int(residual_A),
                    "state":       state or "",
                    "spectrum":    spectrum,
                    "xs_avg_mb":   avg,
                    "library":     library,
                })

    if not rows:
        return None
    return pl.DataFrame(rows, schema={
        "target_Z":   pl.Int32,
        "target_A":   pl.Int32,
        "residual_Z": pl.Int32,
        "residual_A": pl.Int32,
        "state":      pl.Utf8,
        "spectrum":   pl.Utf8,
        "xs_avg_mb":  pl.Float64,
        "library":    pl.Utf8,
    })


def build(data_dir: Path | None = None) -> None:
    """Build data/meta/spectrum_xs.parquet from all neutron XS libraries."""
    data_dir = Path(data_dir) if data_dir else _resolve_data_dir()
    catalog = json.loads((data_dir / "catalog.json").read_text())

    out_path = data_dir / "meta" / "spectrum_xs.parquet"
    _file_re = re.compile(r"^n_([A-Za-z]+)\.parquet$")

    frames: list[pl.DataFrame] = []
    for lib_key, lib_info in catalog.get("libraries", {}).items():
        if "n" not in lib_info.get("projectiles", []):
            continue
        xs_dir = data_dir / lib_info["path"]
        if not xs_dir.exists():
            continue

        neutron_files = [f for f in xs_dir.glob("n_*.parquet") if _file_re.match(f.name)]
        if not neutron_files:
            continue

        print(f"  {lib_key}: {len(neutron_files)} target files")
        for f in sorted(neutron_files):
            m = _file_re.match(f.name)
            sym = m.group(1).lower()
            target_Z = _SYMBOL_TO_Z.get(sym)
            if target_Z is None:
                continue
            df = _process_file(f, target_Z, lib_key)
            if df is not None:
                frames.append(df)

    if not frames:
        raise RuntimeError("No neutron XS data found.")

    result = pl.concat(frames).sort(["library", "target_Z", "target_A", "residual_Z", "residual_A", "spectrum"])
    result.write_parquet(out_path, compression="zstd")
    print(f"Wrote {len(result):,} rows → {out_path}")


if __name__ == "__main__":
    print("Building spectrum-averaged neutron XS ...")
    build()
