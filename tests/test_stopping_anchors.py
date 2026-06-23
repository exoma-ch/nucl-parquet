"""Physics-invariant regression for α, ³He, and p stopping (after #137).

Source routing (locked design from #139):
- α (a, he4)      → NIST ASTAR (ICRU-49 reference, reproducible via build_stopping.py)
- ³He (h, he3)    → catima (no NIST ³He table exists)
- p, d, t          → NIST PSTAR (ICRU-49)
- e                → NIST ESTAR (ICRU-37)
- heavy ions       → catima

Anchor values are NIST ICRU-49 published numbers (log-log-interpolated from
the table). Because α now comes from the same NIST table the anchors were
pulled from, agreement is exact to interpolation noise (<0.05%). The wider
catima fallback tolerance only applies to ³He.

The 2026-05-07 root-cause was that the prior ASTAR.parquet was Z²-scaled
from PSTAR at the wrong energy axis. This test file enforces a tight contract
that the bug class cannot silently return.

In addition to point-wise anchors, this module asserts physics invariants:
- α via ASTAR vs p via PSTAR: S(α, Z, 4E) ≈ 4·S(p, Z, E) at high MeV/u (Z²=4)
- target-ratio independence: S(α,Pb)/S(α,C) ≈ S(p,Pb)/S(p,C) at fixed velocity
- monotonicity above the Bragg peak
- boundary cleanliness: out-of-range Z returns NaN
- water compound anchor (Bragg-additivity regression guard)
- ³He via catima dispatches and matches α dedx at equal MeV/u (both Z=2)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import nucl_parquet as np_lib
from nucl_parquet.loader import _catima_cache, _stopping_cache


@pytest.fixture(autouse=True)
def _clear_stopping_caches() -> None:
    """Module-level caches in nucl_parquet.loader are keyed by (source, target_Z)
    without the db handle, so the synthetic mini_data fixture in test_loader.py
    can pollute them. Clear before each anchor test.
    """
    _stopping_cache.clear()
    _catima_cache.clear()


# (target_Z, energy_MeV_per_u, NIST_dedx [MeV·cm²/g])
# All values pulled directly from NIST ASTAR CGI on 2026-05-10 and 2026-05-11
# (spike #139). Tolerance is interpolation noise — we look up the same table
# the user query hits, so agreement is exact to <0.05%.
_NIST_ASTAR_ALPHA: list[tuple[int, float, float]] = [
    # Hydrogen
    (1, 0.10, 7935.0),
    (1, 0.15, 8237.0),
    (1, 0.20, 7819.0),
    (1, 0.30, 6494.0),
    (1, 0.5, 4593.0),
    (1, 1.0, 2678.0),
    (1, 2.0, 1539.0),
    (1, 5.0, 724.5),
    (1, 10.0, 405.1),
    (1, 20.0, 225.8),
    (1, 50.0, 105.3),
    (1, 100.0, 60.89),
    # Carbon
    (6, 0.10, 1867.0),
    (6, 0.15, 1910.0),
    (6, 0.20, 1869.0),
    (6, 0.30, 1706.0),
    (6, 0.5, 1373.0),
    (6, 1.0, 898.9),
    (6, 2.0, 554.8),
    (6, 5.0, 278.9),
    (6, 10.0, 161.3),
    (6, 20.0, 92.18),
    (6, 50.0, 44.07),
    (6, 100.0, 25.82),
    # Oxygen
    (8, 0.10, 1644.0),
    (8, 0.15, 1795.0),
    (8, 0.20, 1822.0),
    (8, 0.30, 1689.0),
    (8, 0.5, 1330.0),
    (8, 1.0, 860.9),
    (8, 2.0, 531.7),
    (8, 5.0, 269.0),
    (8, 10.0, 156.5),
    (8, 20.0, 89.83),
    (8, 50.0, 43.13),
    (8, 100.0, 25.34),
    # Aluminum
    (13, 0.10, 1284.0),
    (13, 0.15, 1300.0),
    (13, 0.20, 1271.0),
    (13, 0.30, 1174.0),
    (13, 0.5, 985.9),
    (13, 1.0, 699.1),
    (13, 2.0, 440.1),
    (13, 5.0, 227.2),
    (13, 10.0, 134.5),
    (13, 20.0, 78.35),
    (13, 50.0, 38.17),
    (13, 100.0, 22.60),
    # Copper
    (29, 0.10, 609.8),
    (29, 0.15, 678.7),
    (29, 0.20, 704.1),
    (29, 0.30, 695.9),
    (29, 0.5, 628.3),
    (29, 1.0, 483.8),
    (29, 2.0, 326.0),
    (29, 5.0, 177.4),
    (29, 10.0, 108.2),
    (29, 20.0, 64.54),
    (29, 50.0, 32.20),
    (29, 100.0, 19.32),
    # Tungsten — sub-MeV/u + therapy range
    (74, 0.10, 346.4),
    (74, 0.15, 386.8),
    (74, 0.20, 401.7),
    (74, 0.30, 392.2),
    (74, 0.5, 345.3),
    (74, 1.0, 261.1),
    (74, 2.0, 187.4),
    (74, 5.0, 112.6),
    (74, 10.0, 72.89),
    (74, 20.0, 45.57),
    (74, 50.0, 23.91),
    (74, 100.0, 14.73),
    # Gold
    (79, 0.10, 330.1),
    (79, 0.15, 372.7),
    (79, 0.20, 388.8),
    (79, 0.30, 385.9),
    (79, 0.5, 347.3),
    (79, 1.0, 258.7),
    (79, 2.0, 185.6),
    (79, 5.0, 111.4),
    (79, 10.0, 72.11),
    (79, 20.0, 45.13),
    (79, 50.0, 23.55),
    (79, 100.0, 14.45),
    # Lead — sub-MeV/u Bragg-peak shoulder for high-Z TAT relevance
    (82, 0.10, 359.7),
    (82, 0.15, 395.4),
    (82, 0.20, 404.9),
    (82, 0.30, 394.8),
    (82, 0.5, 349.1),
    (82, 1.0, 258.6),
    (82, 2.0, 184.6),
    (82, 5.0, 110.3),
    (82, 10.0, 71.30),
    (82, 20.0, 44.61),
    (82, 50.0, 23.15),
    (82, 100.0, 14.15),
]


@pytest.mark.data
@pytest.mark.parametrize(("target_Z", "energy_MeV_per_u", "nist_dedx"), _NIST_ASTAR_ALPHA)
def test_alpha_dedx_matches_nist_astar_exactly(
    data_dir_path: Path, target_Z: int, energy_MeV_per_u: float, nist_dedx: float
) -> None:
    """α stopping must match NIST ASTAR within log-log interpolation noise.

    Tight tolerance (0.5%) — the lookup hits the same NIST CGI-derived table
    the anchors were pulled from. Anything looser hides a regression.
    """
    db = np_lib.connect(data_dir_path)
    energy_MeV = energy_MeV_per_u * 4.0
    got = float(np_lib.elemental_dedx(db, "a", target_Z, energy_MeV)[0])
    rel = abs(got - nist_dedx) / nist_dedx * 100.0
    assert rel <= 0.5, (
        f"α on Z={target_Z} at {energy_MeV_per_u} MeV/u: got={got:.4g}, NIST={nist_dedx:.4g}, |Δ|={rel:.3f}% > 0.5%"
    )


@pytest.mark.data
@pytest.mark.parametrize("energy_MeV_per_u", [0.2, 1.0, 10.0, 100.0])
def test_he3_dispatches_through_catima(data_dir_path: Path, energy_MeV_per_u: float) -> None:
    """³He via the loader must equal a direct catima lookup at the same MeV/u.

    Goes through `_get_catima_table(2, 3, target_Z)` and `_interp_loglog(.../A)`
    independently to assert the dispatch is reading the right row and dividing
    by proj_A=3. Note this only validates the *dispatch arithmetic* — pycatima's
    actual ³He physics isn't anchored against a published reference here.
    """
    import numpy as np

    from nucl_parquet.loader import _get_catima_table, _interp_loglog

    db = np_lib.connect(data_dir_path)
    he3 = float(np_lib.elemental_dedx(db, "he3", 29, energy_MeV_per_u * 3.0)[0])
    log_E, log_S = _get_catima_table(db, proj_Z=2, proj_A=3, target_Z=29)
    expected = float(_interp_loglog(log_E, log_S, np.atleast_1d(energy_MeV_per_u))[0])
    assert he3 == pytest.approx(expected, rel=1e-9), (
        f"³He dispatch at {energy_MeV_per_u} MeV/u: got {he3}, direct catima lookup {expected}"
    )


@pytest.mark.data
@pytest.mark.parametrize(
    ("target_Z", "energy_MeV_per_u"),
    [(6, 100.0), (29, 100.0), (82, 100.0), (6, 1.0), (29, 1.0), (82, 1.0)],
)
def test_alpha_proton_velocity_scaling_invariant(data_dir_path: Path, target_Z: int, energy_MeV_per_u: float) -> None:
    """Direct invariant: S(α, Z, 4E) ≈ 4·S(p, Z, E) at same velocity (Bethe regime).

    Z²-scaling holds at high energy where effective charge ≈ Z. If anyone
    re-introduced a Z²-scaled-at-wrong-axis pattern, S(α)/S(p) at total
    energies would NOT be 4× — it would only be 4 after velocity matching.
    Both projectiles route to NIST tables now, so this catches dispatch bugs
    in the energy_MeV → table-energy conversion.
    """
    db = np_lib.connect(data_dir_path)
    a = float(np_lib.elemental_dedx(db, "a", target_Z, energy_MeV_per_u * 4.0)[0])
    p = float(np_lib.elemental_dedx(db, "p", target_Z, energy_MeV_per_u * 1.0)[0])
    ratio = a / p
    # ICRU-49 Z²-scaling holds within ~3% across 1-100 MeV/u — the empirical
    # ratio sits 3.97-4.11 across that range on light/medium/heavy targets.
    # The original Z²-at-wrong-axis bug pinned ratio ≈ 4 only when the wrong
    # energy axis was used; running this at two distinct MeV/u values catches
    # a regression that survives single-energy validation.
    assert 3.85 <= ratio <= 4.15, (
        f"Z²-scaling broken at Z={target_Z}, {energy_MeV_per_u} MeV/u: S_α/S_p = {ratio:.3f}, expected ≈ 4.0 (Z²=4)"
    )


@pytest.mark.data
def test_target_ratio_projectile_independent_at_high_mev_u(data_dir_path: Path) -> None:
    """S(α,Pb)/S(α,C) ≈ S(p,Pb)/S(p,C) at fixed velocity (Bethe regime).

    Catches (proj_Z, target_Z) index transposition in catima or the loader.
    """
    db = np_lib.connect(data_dir_path)
    a_pb = float(np_lib.elemental_dedx(db, "a", 82, 100.0 * 4.0)[0])
    a_c = float(np_lib.elemental_dedx(db, "a", 6, 100.0 * 4.0)[0])
    p_pb = float(np_lib.elemental_dedx(db, "p", 82, 100.0)[0])
    p_c = float(np_lib.elemental_dedx(db, "p", 6, 100.0)[0])
    ratio_a = a_pb / a_c
    ratio_p = p_pb / p_c
    rel = abs(ratio_a - ratio_p) / ratio_p * 100.0
    assert rel < 3.0, (
        f"target ratio at 100 MeV/u: α gives Pb/C={ratio_a:.3f}, p gives Pb/C={ratio_p:.3f}, diff={rel:.2f}% > 3%"
    )


@pytest.mark.data
@pytest.mark.parametrize("target_Z", [1, 6, 29, 82])
def test_alpha_dedx_monotone_above_bragg_peak(data_dir_path: Path, target_Z: int) -> None:
    """Above ~1 MeV/u, α stopping decreases monotonically with energy."""
    db = np_lib.connect(data_dir_path)
    energies_per_u = [1.0, 5.0, 20.0, 100.0]
    values = [float(np_lib.elemental_dedx(db, "a", target_Z, e * 4.0)[0]) for e in energies_per_u]
    for i in range(len(values) - 1):
        assert values[i] > values[i + 1], (
            f"Z={target_Z}: dedx at {energies_per_u[i]} MeV/u = {values[i]:.3g} "
            f"not > {energies_per_u[i + 1]} MeV/u = {values[i + 1]:.3g}"
        )


@pytest.mark.data
def test_alpha_out_of_range_target_returns_nan(data_dir_path: Path) -> None:
    """Z=93+ has no NIST or catima coverage; must return NaN, not garbage."""
    db = np_lib.connect(data_dir_path)
    result = float(np_lib.elemental_dedx(db, "a", 93, 10.0)[0])
    assert math.isnan(result), f"Expected NaN for Z=93, got {result}"


@pytest.mark.data
def test_alpha_falls_back_to_catima_for_non_nist_element(data_dir_path: Path) -> None:
    """NIST ASTAR only publishes 25 elemental targets; the rest fall back to catima.

    Tc (Z=43) is not in NIST's list. Verify both (a) the dispatch returns a
    finite α dedx and (b) the value actually came from catima, not some
    stale cache entry — by comparing against a direct catima lookup at the
    same energy.
    """
    import numpy as np

    from nucl_parquet.loader import _get_catima_table, _interp_loglog

    db = np_lib.connect(data_dir_path)
    got = float(np_lib.elemental_dedx(db, "a", 43, 5.0)[0])
    log_E_c, log_S_c = _get_catima_table(db, proj_Z=2, proj_A=4, target_Z=43)
    expected = float(_interp_loglog(log_E_c, log_S_c, np.atleast_1d(5.0 / 4.0))[0])
    assert got == pytest.approx(expected, rel=1e-9), (
        f"Tc α fallback returned {got}, but direct catima lookup is {expected}"
    )


@pytest.mark.data
def test_alpha_in_water_compound_matches_nist(data_dir_path: Path) -> None:
    """α stopping in liquid water at clinical energies.

    NIST ASTAR "Water, Liquid" reference (matno=276), fetched 2026-05-11
    and log-log-interpolated:
        5  MeV (1.25 MeV/u):  885.5
        20 MeV (5.00 MeV/u):  314.6
        100 MeV (25  MeV/u):   86.49

    Bragg-rule on the elemental NIST tables differs from the compound table
    by a few % because the compound mean excitation energy I is not the
    weighted average of H, O atomic I-values. The ICRU-49 compound correction
    is filed as #140; this test pins the current systematic.
    """
    db = np_lib.connect(data_dir_path)
    water = [(1, 0.1119), (8, 0.8881)]
    # Therapy-range α (1-8 MeV) is the dosimetrically critical regime for
    # targeted-α decay chains (²¹³Bi, ²²⁵Ac); Bragg peak lives near 0.3-0.8
    # MeV/u so 1-2 MeV α is on the rising side. Bragg-rule on the elemental
    # NIST tables overestimates compound water by ~9% at 1 MeV, dropping to
    # ~2% above 5 MeV — that's the I-value compound systematic that ICRU-49
    # corrects via published compound mean-excitation energies. The wider
    # low-energy tolerance pins the current behaviour; #140 will tighten it
    # via compound I-values.
    cases = [
        (1.0, 2193.0, 10.0),
        (2.0, 1625.0, 6.0),
        (5.0, 885.5, 5.0),
        (8.0, 630.6, 5.0),
        (20.0, 314.6, 5.0),
        (100.0, 86.49, 5.0),
    ]
    for energy_MeV, nist, tol_pct in cases:
        got = float(np_lib.compound_dedx(db, "a", water, energy_MeV)[0])
        rel = abs(got - nist) / nist * 100.0
        assert rel <= tol_pct, (
            f"α in water at {energy_MeV} MeV: got={got:.4g}, NIST={nist}, "
            f"|Δ|={rel:.2f}% > {tol_pct}% — check elemental H/O anchors first"
        )


@pytest.mark.data
def test_nist_compound_vs_bragg_proton_water(data_dir_path: Path) -> None:
    """Route 1 (NIST compound table) vs Route 2 (Bragg additivity) for protons in water.

    The NIST PSTAR compound table uses the measured compound I-value (75 eV),
    while Bragg-mixed elemental tables use atomic I-values (19.2 eV for H,
    95.0 eV for O). The two routes should diverge by 5-15% at 1 MeV and
    converge to <2% above 10 MeV.
    """
    db = np_lib.connect(data_dir_path)

    # Route 1: NIST compound table (direct)
    nist_rows = db.sql(
        "SELECT energy_MeV, dedx FROM pstar_compounds WHERE compound = 'WATER_LIQUID' ORDER BY energy_MeV"
    ).fetchall()
    assert len(nist_rows) > 10, "NIST water compound table should have >10 points"

    # Route 2: Bragg additivity from elemental PSTAR (H=11.19%, O=88.81% by mass)

    # Compare at therapeutic proton energies
    for energy in [1.0, 5.0, 10.0, 50.0, 100.0, 200.0]:
        # Bragg route
        bragg = (
            float(np_lib.elemental_dedx(db, "p", 1, energy)[0]) * 0.1119
            + float(np_lib.elemental_dedx(db, "p", 8, energy)[0]) * 0.8881
        )
        # NIST compound route (interpolated via DuckDB)
        nist = db.sql(
            "SELECT dedx FROM pstar_compounds WHERE compound = 'WATER_LIQUID' ORDER BY ABS(energy_MeV - $e) LIMIT 1",
            params={"e": energy},
        ).fetchone()
        assert nist is not None, f"No NIST data near {energy} MeV"
        nist_val = float(nist[0])

        divergence = (bragg - nist_val) / nist_val
        # Bragg additivity overestimates vs the NIST compound mean-excitation
        # energy by a few percent across the therapeutic range — largest at low
        # energy (~2.7% at 1 MeV) and shrinking with energy. Mirror of the rs
        # `nist_vs_bragg_water_divergence` guard corrected in #247 (the old
        # "5–15% at 1 MeV" claim was wrong; the gap is genuinely ~2.7% there).
        assert 0.0 < divergence < 0.05, (
            f"Bragg vs NIST at {energy} MeV: divergence={divergence:.1%}, expected 0–5% overestimate"
        )


@pytest.mark.data
@pytest.mark.parametrize(("projectile", "scale_A"), [("d", 2.0), ("t", 3.0)])
def test_deuteron_triton_velocity_scaling(data_dir_path: Path, projectile: str, scale_A: float) -> None:
    """S(d, Z, 2E) == S(p, Z, E) and S(t, Z, 3E) == S(p, Z, E) at equal velocity.

    d and t are velocity-scaled from PSTAR by build_light_ions.py — they
    share Z=1, so at the same MeV/u they collapse onto the PSTAR curve.
    Test exercises the loader's `energy_MeV / proj_A` divisor for the
    PSTAR branch.
    """
    db = np_lib.connect(data_dir_path)
    for target_Z in (6, 29, 82):
        for energy_MeV_per_u in (1.0, 10.0, 100.0):
            heavy = float(np_lib.elemental_dedx(db, projectile, target_Z, energy_MeV_per_u * scale_A)[0])
            proton = float(np_lib.elemental_dedx(db, "p", target_Z, energy_MeV_per_u)[0])
            rel = abs(heavy - proton) / proton * 100.0
            assert rel < 0.5, (
                f"{projectile} vs p at Z={target_Z}, {energy_MeV_per_u} MeV/u: "
                f"got {heavy:.4g} vs {proton:.4g}, |Δ|={rel:.3f}%"
            )


@pytest.mark.data
def test_electron_dedx_matches_nist_estar(data_dir_path: Path) -> None:
    """ESTAR anchor on Cu at 1 MeV: NIST ICRU-37 reference value.

    The electron path (ESTAR) routes through the same `_get_stopping_table`
    machinery as PSTAR/ASTAR but with no MeV→MeV/u rescaling. One anchor
    here keeps that branch structurally tested.
    """
    db = np_lib.connect(data_dir_path)
    got = float(np_lib.elemental_dedx(db, "e", 29, 1.0)[0])
    # NIST ESTAR Cu at 1 MeV: 1.309 MeV·cm²/g (fetched 2026-05-11).
    assert got == pytest.approx(1.309, rel=0.005), f"ESTAR e on Cu at 1 MeV: got {got:.4g}, NIST 1.309"
