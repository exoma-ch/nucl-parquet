"""Tests for the NJOY-processed neutron library (endfb-8.0) and the retirement of
in-repo resonance reconstruction (#263).

Design: the NJOY shards are a normal in-repo cross-section library — same 6-column
schema as jendl/tendl/endfb-8.1 — so they auto-wire into the loader's `xs` view with
no client code. These tests assert that wiring, the physics, and that the superseded
reconstruction (and endfb-8.1 neutron) are gone.

The thinning and shard-schema tests are pure unit tests over synthetic curves;
the physics spot checks read the committed shards. No network either way.
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


def test_thin_pointwise_respects_tolerance_both_laws():
    """The thinned grid must reconstruct within tol under BOTH lin-lin and log-log."""
    m = _builder()
    e = np.geomspace(1e-5, 2e7, 4000)
    xs = 100.0 / np.sqrt(e)
    for c, amp in ((10.0, 400.0), (1e3, 250.0)):
        xs = xs + amp * np.exp(-((np.log(e) - np.log(c)) ** 2) / 0.02)
    te, ts = m.thin_pointwise(e, xs, tol=0.01)
    assert 2 < len(te) < len(e)
    approx_log = np.exp(np.interp(np.log(e), np.log(te), np.log(ts)))
    approx_lin = np.interp(e, te, ts)
    assert (np.abs(approx_log - xs) / xs).max() <= 0.0101
    assert (np.abs(approx_lin - xs) / xs).max() <= 0.0101


def test_thin_pointwise_preserves_1_over_v_for_linear_readers():
    """A pure 1/v curve is a log-log straight line; a log-log-only thin would drop
    every interior point and wreck linear interpolation (the #271 footgun). The
    dual-metric thin must keep enough points for lin-lin to stay within tol."""
    m = _builder()
    e = np.geomspace(1e-5, 1.0, 2000)  # eV: thermal 1/v region
    xs = 1000.0 / np.sqrt(e)
    te, ts = m.thin_pointwise(e, xs, tol=0.01)
    assert len(te) > 20, f"1/v thermal region collapsed to {len(te)} points"
    approx_lin = np.interp(e, te, ts)
    assert (np.abs(approx_lin - xs) / xs).max() <= 0.0101


def test_thin_pointwise_preserves_resonance_peak():
    m = _builder()
    e = np.geomspace(1.0, 100.0, 501)
    xs = np.ones_like(e)
    xs[250] = 1000.0
    te, ts = m.thin_pointwise(e, xs, tol=0.01)
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
@pytest.mark.parametrize(
    "target_a,res_z,res_a",
    [
        (59, 27, 60),  # Co-59(n,γ) — canonical 1/v thermal, the #271 evidence
        (63, 29, 64),  # Cu-63(n,γ)
        (56, 26, 57),  # Fe-56(n,γ)
    ],
)
def test_thermal_linear_interp_matches_loglog(target_a, res_z, res_a):
    """Regression guard for #271: a *linear* reader of the shipped thermal grid
    must agree with log-log to a few %. A log-log-only thin left the 1/v thermal
    region so sparse that np.interp read ~36× high; the dual-metric thin fixes it."""
    import nucl_parquet

    db = nucl_parquet.connect()
    d = db.sql(
        "SELECT energy_MeV, xs_mb FROM xs "
        f"WHERE library='endfb-8.0' AND target_A={target_a} AND residual_Z={res_z} AND residual_A={res_a} "
        "ORDER BY energy_MeV"
    ).fetchnumpy()
    e, s = d["energy_MeV"], d["xs_mb"] / 1000.0
    e_th = 2.53e-8  # 0.0253 eV, Maxwellian peak — squarely in the old bare gap
    lin = float(np.interp(e_th, e, s))
    log = float(np.exp(np.interp(np.log(e_th), np.log(e), np.log(s))))
    assert abs(lin / log - 1.0) < 0.03, (
        f"lin={lin:.3f} vs loglog={log:.3f} b — thermal grid too sparse for linear readers"
    )


@pytest.mark.data
def test_no_duplicate_energy_points_per_channel():
    """Within one element file, every (target_A, residual) channel must be a single-
    valued curve. Discrete-level partials sharing a residual (e.g. MT800/MT801 →
    Li-7) must be *summed*, not concatenated — concatenation left duplicate
    energies no reader can interpolate (B-10(n,α) read ~12000 b instead of 3844 b).

    Grouping includes the source file: the shared 6-col schema drops target_Z, so
    isobars in different element files (Ar-40 vs Ca-40) legitimately share
    (target_A, residual_Z, residual_A) in the unified view — that is not a dup."""
    import duckdb

    xs_dir = ROOT / "data" / "endfb-8.0" / "xs"
    if not xs_dir.exists():
        pytest.skip("endfb-8.0 data not present")
    con = duckdb.connect()
    dups = con.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT filename, target_A, residual_Z, residual_A, energy_MeV, COUNT(*) c "
        "  FROM read_parquet(?, filename=true) "
        "  GROUP BY 1,2,3,4,5 HAVING COUNT(*) > 1)",
        [str(xs_dir / "n_*.parquet")],
    ).fetchone()[0]
    assert dups == 0, f"{dups} duplicate-energy points within a channel — partials concatenated instead of summed"


@pytest.mark.data
@pytest.mark.parametrize(
    "sym,target_a,res_z,res_a,lo,hi,lit",
    [
        ("B", 10, 3, 7, 3700.0, 4000.0, 3840.0),  # B-10(n,α)→Li-7, MT800+MT801 summed
        ("Li", 6, 2, 4, 900.0, 980.0, 940.0),  # Li-6(n,t)→He-4
    ],
)
def test_charged_particle_out_thermal_summed(sym, target_a, res_z, res_a, lo, hi, lit):
    """(n,α)/(n,t) residual production sums its level partials to the right thermal σ."""
    import nucl_parquet

    db = nucl_parquet.connect()
    n, sth = _thermal(db, target_a, res_z, res_a)
    assert lo < sth < hi, f"{sym}-{target_a} σ_th={sth:.1f} b outside [{lo}, {hi}] (lit ~{lit})"


@pytest.mark.data
def test_xs_view_target_Z_disambiguates_isobars():
    """The unified `xs` view must carry a `target_Z` (derived from the element file)
    so isobaric targets reaching the same absolute residual via different reactions
    are separable — without it the target_Z-less view interleaves them (#273)."""
    import nucl_parquet

    db = nucl_parquet.connect()
    # Nd-145, Pm-145, Sm-145 all reach Nd-143 (residual_Z=60) at 14–20 MeV via
    # different multi-particle channels — they collide on (target_A, residual).
    zs = [
        r[0]
        for r in db.sql(
            "SELECT DISTINCT target_Z FROM xs "
            "WHERE library='endfb-8.0' AND target_A=145 AND residual_Z=60 AND residual_A=143 "
            "ORDER BY target_Z"
        ).fetchall()
    ]
    assert zs == [60, 61, 62], f"expected Nd/Pm/Sm target_Z, got {zs}"
    # With target_Z pinned the channel is single-valued again (no interleaving).
    dup = db.sql(
        "SELECT COUNT(*) FROM ("
        "  SELECT energy_MeV, COUNT(*) c FROM xs "
        "  WHERE library='endfb-8.0' AND target_Z=60 AND target_A=145 AND residual_Z=60 AND residual_A=143 "
        "  GROUP BY energy_MeV HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert dup == 0, f"{dup} interleaved points even with target_Z pinned"
    # target_Z is correct for a plain (n,γ) channel too (Co-59 → Z=27).
    co = db.sql(
        "SELECT DISTINCT target_Z FROM xs WHERE library='endfb-8.0' AND target_A=59 AND residual_Z=27 AND residual_A=60"
    ).fetchall()
    assert [r[0] for r in co] == [27]


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
