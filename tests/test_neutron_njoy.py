"""Tests for the NJOY-processed neutron library (endfb-8.0) and the retirement of
in-repo resonance reconstruction (#263).

Design: the NJOY shards are a normal in-repo cross-section library — same 6-column
schema as jendl/tendl/endfb-8.1 — so they auto-wire into the loader's `xs` view with
no client code. These tests assert that wiring, the physics, and that the superseded
reconstruction (and endfb-8.1 neutron) are gone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _builder():
    import build_neutron_njoy as m

    return m


# ---------------------------------------------------------------------------
# Thinning — pure, no data
# ---------------------------------------------------------------------------


def test_thin_loglog_respects_tolerance():
    m = _builder()
    e = np.geomspace(1e-5, 2e7, 4000)
    xs = 100.0 / np.sqrt(e)
    for c, amp in ((10.0, 400.0), (1e3, 250.0)):
        xs = xs + amp * np.exp(-((np.log(e) - np.log(c)) ** 2) / 0.02)
    te, ts = m.thin_loglog(e, xs, tol=0.01)
    assert 2 < len(te) < len(e)
    approx = np.exp(np.interp(np.log(e), np.log(te), np.log(ts)))
    assert (np.abs(approx - xs) / xs).max() <= 0.0101


def test_thin_loglog_preserves_resonance_peak():
    m = _builder()
    e = np.geomspace(1.0, 100.0, 501)
    xs = np.ones_like(e)
    xs[250] = 1000.0
    te, ts = m.thin_loglog(e, xs, tol=0.01)
    assert e[250] in te


def test_parse_nuclide_stem_skips_metastable_targets():
    m = _builder()
    assert m.parse_nuclide_stem("Nd143") == (60, 143, "Nd")
    assert m.parse_nuclide_stem("U238") == (92, 238, "U")
    assert m.parse_nuclide_stem("Ag110_m1") is None


def test_shard_schema_matches_other_xs_libraries():
    """The NJOY shards must carry the exact 6-column xs schema so they auto-union."""
    import polars as pl

    xs_dir = ROOT / "data" / "endfb-8.0" / "xs"
    if not xs_dir.exists():
        pytest.skip("endfb-8.0 data not present")
    ref = pl.read_parquet_schema(ROOT / "data" / "jendl-5" / "xs" / "n_Cu.parquet")
    got = pl.read_parquet_schema(next(xs_dir.glob("n_*.parquet")))
    assert list(got.keys()) == list(ref.keys()), f"schema {list(got.keys())} != xs schema {list(ref.keys())}"
    assert [str(v) for v in got.values()] == [str(v) for v in ref.values()]


# ---------------------------------------------------------------------------
# Auto-wiring + physics via the loader `xs` sugar
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_endfb_8_0_autowires_into_xs():
    """endfb-8.0 must register as a view AND fold into the unified `xs` union."""
    import nucl_parquet

    db = nucl_parquet.connect()
    assert db.sql("SELECT COUNT(*) FROM endfb_8_0").fetchone()[0] > 0
    libs = {r[0] for r in db.sql("SELECT DISTINCT library FROM xs").fetchall()}
    assert "endfb-8.0" in libs


def _thermal(db, target_a, res_z, res_a):
    d = db.sql(
        "SELECT energy_MeV, xs_mb FROM xs "
        f"WHERE library='endfb-8.0' AND target_A={target_a} AND residual_Z={res_z} AND residual_A={res_a} "
        "ORDER BY energy_MeV"
    ).fetchnumpy()
    e, s = d["energy_MeV"], d["xs_mb"] / 1000.0
    return len(e), float(np.exp(np.interp(np.log(2.53e-8), np.log(e), np.log(s))))


@pytest.mark.data
@pytest.mark.parametrize(
    "target_a,res_z,res_a,lo,hi,lit",
    [
        (143, 60, 144, 300.0, 350.0, 325.0),  # Nd-143(n,γ)
        (56, 26, 57, 2.3, 2.9, 2.59),  # Fe-56(n,γ)
        (238, 92, 239, 2.5, 2.9, 2.68),  # U-238(n,γ)
        (63, 29, 64, 4.2, 4.8, 4.5),  # Cu-63(n,γ)
    ],
)
def test_capture_thermal_values(target_a, res_z, res_a, lo, hi, lit):
    """(n,γ) thermal σ via the xs sugar (log-log interp) matches literature."""
    import nucl_parquet

    db = nucl_parquet.connect()
    n, sth = _thermal(db, target_a, res_z, res_a)
    assert n > 100, f"expected a dense resonance grid, got {n} points"
    assert lo < sth < hi, f"σ_th={sth:.3f} b outside [{lo}, {hi}] (lit ~{lit})"


@pytest.mark.data
def test_resonance_region_present():
    """The processed data must reach thermal (raw MF=3 started ~0.2 MeV)."""
    import nucl_parquet

    db = nucl_parquet.connect()
    d = db.sql(
        "SELECT energy_MeV FROM xs WHERE library='endfb-8.0' AND target_A=143 AND residual_Z=60 AND residual_A=144"
    ).fetchnumpy()
    e = d["energy_MeV"]
    assert e.min() < 1e-8, f"expected thermal coverage, Emin={e.min():.2e} MeV"
    assert ((e >= 1e-6) & (e <= 1e-4)).sum() > 10, "resolved-resonance region missing"


# ---------------------------------------------------------------------------
# Retirement guards
# ---------------------------------------------------------------------------


def test_endfb_8_1_neutron_retired():
    """endfb-8.1 must no longer ship neutron: no n_* files, no 'n' projectile."""
    import json

    xs_dir = ROOT / "data" / "endfb-8.1" / "xs"
    if xs_dir.exists():
        assert not list(xs_dir.glob("n_*.parquet")), "endfb-8.1 neutron files must be retired"
    catalog = json.loads((ROOT / "data" / "catalog.json").read_text())
    assert "n" not in catalog["libraries"]["endfb-8.1"]["projectiles"]
    assert catalog["libraries"]["endfb-8.0"]["data_type"] == "cross_sections"


def test_reconstruction_engine_retired():
    """The home-grown resonance reconstructor must be gone (superseded by NJOY)."""
    assert not (ROOT / "scripts" / "reconstruct_resonances.py").exists()
    assert not (ROOT / "scripts" / "validate_vs_openmc_hdf5.py").exists()
    fetch = (ROOT / "scripts" / "fetch_endf_libs.py").read_text()
    assert "reconstruct_resonances" not in fetch
    assert "_splice_reconstructed_capture" not in fetch
