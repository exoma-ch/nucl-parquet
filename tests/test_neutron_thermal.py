"""Neutron thermal (n,γ) capture — resonance-reconstructed data regression.

The resolved-resonance region (thermal 1/v + resonances) is reconstructed from
the ENDF MF=2 parameters and spliced onto the fast MF=3 data (see
scripts/reconstruct_resonances.py). This locks in the reconstructed thermal
capture cross-section at 0.0253 eV against literature 2200 m/s values.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "endfb-8.1", "xs")

# (element, Z, target_A, literature σ_th [b], tolerance fraction)
CASES = [
    ("Co", 27, 59, 37.18, 0.10),
    ("Au", 79, 197, 98.65, 0.10),
    ("Mn", 25, 55, 13.36, 0.10),
    ("In", 49, 115, 202.0, 0.10),
    ("Na", 11, 23, 0.53, 0.15),
    ("Ag", 47, 109, 91.0, 0.15),
    ("Sm", 62, 152, 206.0, 0.15),
    ("Gd", 64, 157, 254000.0, 0.20),  # the giant absorber — exercises dynamic range
    # LRF=7 (R-Matrix Limited) — reconstructed via the multi-channel Reich-Moore path
    ("Cu", 29, 63, 4.5, 0.12),
    ("V", 23, 51, 4.9, 0.12),
    ("W", 74, 186, 38.1, 0.12),
    ("Fe", 26, 54, 2.25, 0.15),
    # P-31 has no resolved resonances (LRU=0) — its capture comes straight from
    # MF=3, so it only matches literature if elastic (MT=2) is excluded from the
    # (n,γ) residual channel (the contamination fix).
    ("P", 15, 31, 0.166, 0.10),
]


def test_capture_channel_not_elastic_contaminated():
    """The (n,γ) product channel (Z, A+1) must be pure capture, not swamped by
    elastic scattering (MT=2, ~barns of potential scattering). Fast capture on Co
    is ~milli-barns; a value near the elastic ~few-barn level means MT=2 leaked in."""
    path = os.path.join(DATA, "n_Co.parquet")
    if not os.path.exists(path):
        pytest.skip("data not present")
    df = pl.read_parquet(path)
    cap = df.filter(
        (pl.col("target_A") == 59)
        & (pl.col("residual_Z") == 27)
        & (pl.col("residual_A") == 60)
        & (pl.col("state") == "")
    ).sort("energy_MeV")
    e = cap["energy_MeV"].to_numpy()
    xs = cap["xs_mb"].to_numpy() / 1e3
    sigma_1mev = float(np.interp(1.0, e, xs))
    assert sigma_1mev < 0.1, (
        f"⁵⁹Co(n,γ) at 1 MeV = {sigma_1mev:.3g} b — expected ~0.006 b; a value near "
        f"the ~4 b elastic level means MT=2 leaked into the capture channel"
    )


@pytest.mark.parametrize("elem,z,a,ref,tol", CASES)
def test_thermal_capture_matches_literature(elem, z, a, ref, tol):
    path = os.path.join(DATA, f"n_{elem}.parquet")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    df = pl.read_parquet(path)
    cap = df.filter(
        (pl.col("target_A") == a)
        & (pl.col("residual_Z") == z)
        & (pl.col("residual_A") == a + 1)
        & (pl.col("state") == "")
    ).sort("energy_MeV")
    assert cap.height > 0, f"no (n,γ) capture channel for {elem}-{a}"
    e = cap["energy_MeV"].to_numpy()
    xs = cap["xs_mb"].to_numpy() / 1e3  # mb → b
    assert e.min() < 1e-6, f"{elem}-{a} capture has no thermal data (E_min={e.min():.1e} MeV)"
    sigma_th = float(np.interp(2.53e-8, e, xs))
    assert abs(sigma_th - ref) / ref < tol, (
        f"{elem}-{a} σ_th={sigma_th:.4g} b vs literature {ref} b (ratio {sigma_th / ref:.2f}, tol {tol})"
    )
