"""Neutron (n,γ) capture — broad validation of the reconstructed resonance region.

The resolved-resonance region (thermal 1/v + resonances) is reconstructed from the
ENDF MF=2 parameters and spliced onto the fast MF=3 data
(scripts/reconstruct_resonances.py). This is the load-bearing check that we don't
ship bad cross sections: the two standard integral quantities that fully
characterise that region — the 2200 m/s thermal cross section σ_th and the
infinite-dilution resonance integral I₀ — are validated against published values
across a broad, periodic-table-spanning set of isotopes and all four resonance
formalisms (SLBW / MLBW / Reich-Moore / R-Matrix Limited). The fast region above
the resonance range is the evaluation's own MF=3 pointwise table (authoritative by
construction; not reconstructed here).

Reference values are recommended/measured literature σ_th and I₀ (Mughabghab
Atlas / activation standards) — independent of both the evaluation processing and
this code. Tolerances are generous (0.30) because the job of this fixture is to
catch *regressions* (a broken reconstruction is off by ≥2×, not 30%); the actual
agreement is far tighter (σ_th median 1.00, I₀ median 1.03).
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "endfb-8.1", "xs")

# (element, Z, target_A): 2200 m/s radiative-capture σ_th [b] — literature.
SIGMA_TH = {
    ("Na", 11, 23): 0.53,
    ("Al", 13, 27): 0.231,
    ("P", 15, 31): 0.166,
    ("Cl", 17, 35): 43.6,
    ("Sc", 21, 45): 27.2,
    ("V", 23, 51): 4.9,
    ("Cr", 24, 50): 15.9,
    ("Mn", 25, 55): 13.36,
    ("Fe", 26, 58): 1.32,
    ("Co", 27, 59): 37.18,
    ("Cu", 29, 63): 4.5,
    ("Cu", 29, 65): 2.17,
    ("Zn", 30, 64): 0.76,
    ("As", 33, 75): 4.1,
    ("Y", 39, 89): 1.28,
    ("Nb", 41, 93): 1.15,
    ("Mo", 42, 98): 0.13,
    ("Rh", 45, 103): 145.0,
    ("Ag", 47, 107): 37.6,
    ("Ag", 47, 109): 91.0,
    ("In", 49, 115): 202.0,
    ("Sb", 51, 121): 6.0,
    ("I", 53, 127): 6.15,
    ("Cs", 55, 133): 29.0,
    ("La", 57, 139): 9.04,
    ("Pr", 59, 141): 11.5,
    ("Nd", 60, 143): 325.0,
    ("Sm", 62, 152): 206.0,
    ("Eu", 63, 151): 9200.0,
    ("Gd", 64, 157): 254000.0,
    ("Dy", 66, 164): 2650.0,
    ("Ho", 67, 165): 64.7,
    ("Tm", 69, 169): 100.0,
    ("Lu", 71, 176): 2065.0,
    ("Ta", 73, 181): 20.5,
    ("W", 74, 184): 1.7,
    ("W", 74, 186): 37.9,
    ("Re", 75, 185): 112.0,
    ("Ir", 77, 191): 954.0,
    ("Au", 79, 197): 98.65,
}

# (element, Z, target_A): infinite-dilution resonance integral I₀ [b] — gold-standard
# (activation/dosimetry references). Epithermal check, orthogonal to σ_th.
RESONANCE_INTEGRAL = {
    ("Na", 11, 23): 0.31,
    ("Sc", 21, 45): 12.0,
    ("Cu", 29, 63): 5.0,
    ("Mn", 25, 55): 14.0,
    ("Co", 27, 59): 74.0,
    ("Rh", 45, 103): 1040.0,
    ("Ag", 47, 109): 1470.0,
    ("In", 49, 115): 3300.0,
    ("I", 53, 127): 149.0,
    ("Cs", 55, 133): 415.0,
    ("Sm", 62, 152): 2970.0,
    ("Ta", 73, 181): 655.0,
    ("W", 74, 186): 490.0,
    ("Au", 79, 197): 1550.0,
}


def _capture_curve(elem, z, a):
    path = os.path.join(DATA, f"n_{elem}.parquet")
    if not os.path.exists(path):
        return None
    df = (
        pl.read_parquet(path)
        .filter(
            (pl.col("target_A") == a)
            & (pl.col("residual_Z") == z)
            & (pl.col("residual_A") == a + 1)
            & (pl.col("state") == "")
        )
        .unique("energy_MeV")
        .sort("energy_MeV")
    )
    if df.height < 5:
        return None
    return df["energy_MeV"].to_numpy(), df["xs_mb"].to_numpy() / 1e3  # b


@pytest.mark.parametrize("elem,z,a,ref", [(e, z, a, v) for (e, z, a), v in SIGMA_TH.items()])
def test_thermal_capture_matches_literature(elem, z, a, ref):
    c = _capture_curve(elem, z, a)
    if c is None:
        pytest.skip(f"{elem}-{a} not present")
    e, xs = c
    assert e.min() < 1e-6, f"{elem}-{a} capture has no thermal data (E_min={e.min():.1e} MeV)"
    sigma_th = float(np.interp(2.53e-8, e, xs))
    assert abs(sigma_th - ref) / ref < 0.30, (
        f"{elem}-{a} σ_th={sigma_th:.4g} b vs literature {ref} b (ratio {sigma_th / ref:.2f})"
    )


@pytest.mark.parametrize("elem,z,a,ref", [(e, z, a, v) for (e, z, a), v in RESONANCE_INTEGRAL.items()])
def test_resonance_integral_matches_literature(elem, z, a, ref):
    """Infinite-dilution I₀ = ∫_{0.5 eV}^{2 MeV} σ_γ(E)/E dE — the epithermal check."""
    c = _capture_curve(elem, z, a)
    if c is None:
        pytest.skip(f"{elem}-{a} not present")
    e, xs = c
    m = (e >= 0.5e-6) & (e <= 2.0)
    assert m.sum() >= 3, f"{elem}-{a} no epithermal data"
    i0 = float(np.trapezoid(xs[m] / e[m], e[m]))
    assert abs(i0 - ref) / ref < 0.30, f"{elem}-{a} I₀={i0:.4g} b vs literature {ref} b (ratio {i0 / ref:.2f})"


def test_capture_channel_not_elastic_contaminated():
    """The (n,γ) product channel (Z, A+1) must be pure capture, not swamped by
    elastic scattering (MT=2, ~barns of potential scattering). Fast capture on Co
    is ~milli-barns; a value near the elastic ~few-barn level means MT=2 leaked in."""
    c = _capture_curve("Co", 27, 59)
    if c is None:
        pytest.skip("data not present")
    e, xs = c
    sigma_1mev = float(np.interp(1.0, e, xs))
    assert sigma_1mev < 0.1, (
        f"⁵⁹Co(n,γ) at 1 MeV = {sigma_1mev:.3g} b — expected ~0.006 b; a value near "
        f"the ~4 b elastic level means MT=2 leaked into the capture channel"
    )
