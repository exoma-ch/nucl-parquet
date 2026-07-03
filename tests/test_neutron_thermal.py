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
]


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
