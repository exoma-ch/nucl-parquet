"""The interpolation-law vocabulary, and the region arithmetic behind it (#338).

ENDF tabulates sigma(E) as points *plus a law*. We kept the points and threw the
law away, so every evaluated row silently asserted "read these however you like"
and every consumer read them linearly.

Two things are worth testing here and they are not the same thing:

* the **vocabulary** — that `interp_law` values mean something fixed, and that
  the one law `np.interp` reads correctly is distinguished from the five it does
  not;
* the **region arithmetic** — `laws_per_point`, which converts ENDF's
  index-based (NBT, INT) regions into a value a row can carry. That conversion
  has an off-by-one at every region boundary and the obvious implementation gets
  it wrong, so it is pinned against the reference implementation's own rule.

Everything asserts a positive value. A test that passes because it found no
laws would be indistinguishable from the bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nucl_parquet.endf_interp import (
    CHARGED_PARTICLE_THRESHOLD,
    HISTOGRAM,
    INTERP_LAWS,
    LIN_LIN,
    LOG_LOG,
    interp_table,
    interpolate,
    interpolate_one,
    is_valid_law,
    laws_per_point,
)

# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


def test_only_lin_lin_is_safe_to_read_with_np_interp():
    """The whole point of the column. `is_linear` is what a consumer filters on
    to find the rows their default interpolation gets wrong."""
    linear = {code for code, law in INTERP_LAWS.items() if law.is_linear}
    assert linear == {LIN_LIN}, f"exactly one law is np.interp-safe, got {sorted(linear)}"


def test_histogram_is_not_linear():
    """Law 1 holds y constant to the next point. Reading it linearly is wrong
    between every single pair — it is the *least* linear law, not a near-miss,
    and an `is_linear` that lumped it in with law 2 would be worse than useless."""
    assert not INTERP_LAWS[HISTOGRAM].is_linear
    assert interpolate_one(HISTOGRAM, 1.5, 1.0, 2.0, 10.0, 20.0) == 10.0


def test_every_law_declares_its_log_axes():
    """`log_x`/`log_y` are what let a consumer pick a plotting or resampling
    strategy without hardcoding ENDF's numbering."""
    assert (INTERP_LAWS[LOG_LOG].log_x, INTERP_LAWS[LOG_LOG].log_y) == (True, True)
    assert (INTERP_LAWS[LIN_LIN].log_x, INTERP_LAWS[LIN_LIN].log_y) == (False, False)
    assert INTERP_LAWS[3].log_x and not INTERP_LAWS[3].log_y
    assert INTERP_LAWS[4].log_y and not INTERP_LAWS[4].log_x


def test_two_dimensional_laws_are_absent_on_purpose():
    """ENDF also defines INT=11-15 and 21-25, but those are two-dimensional
    (unit-base interpolation between incident energies in MF=4/5/6) and never
    describe an MF=3 cross-section. A row claiming INT=22 would be a row from a
    file this column does not describe, so they must not silently validate."""
    assert set(INTERP_LAWS) == {1, 2, 3, 4, 5, 6}
    assert not is_valid_law(22)
    assert not is_valid_law(0)


def test_null_is_always_allowed_and_means_not_stated():
    """NULL is how a row says its source never carried a law — NJOY-reconstructed
    pointwise data, EXFOR measurements. It is not a synonym for law 2."""
    assert is_valid_law(None)
    assert None not in INTERP_LAWS


def test_the_view_table_is_keyed_by_the_column_name():
    """`endf_interp` joins as `USING (interp_law)`, so the key must be spelled
    exactly like the column — an `int` or `code` key would need an alias and the
    documented query would not work verbatim."""
    rows = interp_table()
    assert {r["interp_law"] for r in rows} == set(INTERP_LAWS)
    assert all(r["description"] for r in rows), "every law must explain itself"
    assert sum(r["is_linear"] for r in rows) == 1


# ---------------------------------------------------------------------------
# laws_per_point — the region arithmetic
# ---------------------------------------------------------------------------


def _reference(breakpoints, interpolation, n):
    """The rule `endf.function.Tabulated1D._interpolate_scalar` implements:
    the interval starting at 0-based `i` belongs to the first region with
    `i < NBT(k) - 1`. Written out longhand as an independent oracle."""
    out = []
    for i in range(n):
        law = interpolation[-1]
        for b, p in zip(breakpoints, interpolation):
            if i < b - 1:
                law = p
                break
        out.append(law)
    return out


def test_a_single_region_gives_every_point_that_law():
    assert list(laws_per_point([5], [LOG_LOG], 5)) == [LOG_LOG] * 5


@pytest.mark.parametrize(
    ("breakpoints", "interpolation", "n"),
    [
        ([23, 134], [6, 5], 134),  # TENDL-2023 p+Li-6 MT=750
        ([7, 27], [6, 5], 27),  # TENDL-2023 t+Li-6 MT=22 and MT=650
        ([18, 57], [6, 5], 57),  # TENDL-2023 h+Li-6 MT=112 and MT=650
        ([9, 153, 207], [5, 2, 5], 207),  # JEFF-4.0 n+H-1 MT=1
        ([128, 130], [5, 2], 130),  # JENDL-5 n+Fe-55 MT=102
        ([4, 41], [2, 5], 41),  # JENDL-5 n+Fe-55 MT=5
    ],
)
def test_matches_the_reference_rule_on_real_region_layouts(breakpoints, interpolation, n):
    """Every layout here was observed in a real evaluation during the #338
    survey, so this is not a synthetic shape that happens to work."""
    assert list(laws_per_point(breakpoints, interpolation, n)) == _reference(breakpoints, interpolation, n)


def test_the_shared_boundary_point_belongs_to_the_UPPER_region():
    """The off-by-one this function exists to prevent.

    `NBT(k)` is the 1-based index of the *last* point of region k, so regions
    share a point. The obvious implementation —

        out[start:nbt] = law; start = nbt

    — gives that point to the lower region. Checked against six real two-region
    sections it is wrong at exactly one index each time, and it is the boundary:
    the place the evaluator changed law *because the curve changes shape there*,
    which is the worst possible index to get wrong.
    """
    laws = laws_per_point([23, 134], [6, 5], 134)
    assert laws[21] == 6, "well inside the first region"
    assert laws[22] == 5, "the shared point starts an interval governed by region 2"
    assert laws[23] == 5, "well inside the second region"

    naive = np.empty(134, dtype=int)
    naive[:23] = 6
    naive[23:] = 5
    assert naive[22] != laws[22], "the naive slice must actually differ, or this test proves nothing"


def test_the_last_point_carries_the_last_region():
    """It starts no interval, so any value is defensible — but it must be a
    value, so a reversed or clipped scan still finds one rather than an index
    error or a null in the middle of an otherwise complete column."""
    assert laws_per_point([3, 6], [2, 5], 6)[-1] == 5


def test_mismatched_region_arrays_raise():
    """NBT and INT come in pairs. A TAB1 where they disagree is a corrupt record,
    and guessing which array to trust would file rows under a law nobody stated."""
    with pytest.raises(ValueError, match="NBT has 2 regions but INT has 1"):
        laws_per_point([3, 6], [2], 6)
    with pytest.raises(ValueError, match="at least one region"):
        laws_per_point([], [], 0)


# ---------------------------------------------------------------------------
# The laws themselves
# ---------------------------------------------------------------------------


def test_lin_lin_matches_np_interp():
    """The control. If law 2 disagreed with `np.interp` the whole premise —
    "consumers reaching for np.interp are right only for law 2" — would be
    wrong, and every other number in this file meaningless."""
    x, y = np.array([1.0, 2.0, 4.0]), np.array([10.0, 20.0, 80.0])
    grid = np.array([1.5, 2.0, 3.0])
    got = interpolate(x, y, [LIN_LIN] * 3, grid)
    assert got == pytest.approx(np.interp(grid, x, y))


def test_log_log_is_a_power_law():
    """y = x^2 through (1,1) and (4,16): log-log must land exactly on 4 at x=2,
    where lin-lin reads 6 — a 50% error on a curve this ordinary."""
    assert interpolate_one(LOG_LOG, 2.0, 1.0, 4.0, 1.0, 16.0) == pytest.approx(4.0)
    assert interpolate_one(LIN_LIN, 2.0, 1.0, 4.0, 1.0, 16.0) == pytest.approx(6.0)


def test_log_lin_and_lin_log_are_not_the_same_thing():
    """They are trivially transposable and were transposed in the first draft of
    the table; ENDF-102 numbers them 3 and 4 in the order lin-log, log-lin."""
    assert interpolate_one(3, 2.0, 1.0, 4.0, 10.0, 20.0) == pytest.approx(10.0 + math.log(2.0) / math.log(4.0) * 10.0)
    assert interpolate_one(4, 2.0, 1.0, 4.0, 10.0, 20.0) == pytest.approx(10.0 * math.exp(math.log(2.0) / 3.0))


def test_law_6_is_concave_upward_near_threshold():
    """ENDF-102's charged-particle law comes from the Coulomb penetrability, so
    just above T it must sit *below* the linear chord — that concavity is the
    whole reason it exists and is why reading it linearly overestimates."""
    T = 1.0
    lin = interpolate_one(LIN_LIN, 1.5, 1.2, 5.0, 1e-3, 1.0)
    law6 = interpolate_one(CHARGED_PARTICLE_THRESHOLD, 1.5, 1.2, 5.0, 1e-3, 1.0, threshold=T)
    assert law6 < lin, f"law 6 must undercut the chord near T, got {law6} vs {lin}"
    assert law6 > 0


def test_law_6_reproduces_a_curve_built_from_its_own_form():
    """Round-trip: build points from ln(x*y) linear in 1/sqrt(x-T), then ask the
    implementation for a point in between. It must land on the curve, not near
    it — this is the formula the `endf` package does not implement, so there is
    no upstream oracle and it has to be pinned here."""
    T, a1, slope = 2.0, 1.0, -3.0

    def curve(x):
        return math.exp(a1 + slope / math.sqrt(x - T)) / x

    x1, x2, xm = 3.0, 9.0, 5.0
    got = interpolate_one(CHARGED_PARTICLE_THRESHOLD, xm, x1, x2, curve(x1), curve(x2), threshold=T)
    assert got == pytest.approx(curve(xm), rel=1e-12)


def test_law_6_refuses_a_point_at_or_below_the_threshold():
    """1/sqrt(x-T) is undefined there. Returning a number anyway would put a
    finite cross-section below threshold."""
    with pytest.raises(ValueError, match="law 6 is undefined"):
        interpolate_one(CHARGED_PARTICLE_THRESHOLD, 1.0, 1.0, 5.0, 1.0, 2.0, threshold=1.0)


def test_an_unknown_law_raises_rather_than_falling_back_to_linear():
    """Defaulting to lin-lin is the behaviour #338 exists to delete. Putting it
    back inside the fix, as a fallthrough, would be the same bug wearing a
    different hat."""
    with pytest.raises(ValueError, match="unknown ENDF interpolation law"):
        interpolate_one(22, 1.5, 1.0, 2.0, 1.0, 2.0)


def test_interpolate_honours_a_law_that_changes_partway():
    """The case that forces a per-row column: one curve, two laws. Points 0-1 are
    log-log and 2-3 lin-lin, and the result must follow each in its own span."""
    x = np.array([1.0, 4.0, 8.0, 16.0])
    y = np.array([1.0, 16.0, 32.0, 64.0])
    laws = np.array([LOG_LOG, LIN_LIN, LIN_LIN, LIN_LIN])
    got = interpolate(x, y, laws, np.array([2.0, 12.0]))
    assert got[0] == pytest.approx(4.0), "first span is x^2"
    assert got[1] == pytest.approx(48.0), "second span is the linear chord"


def test_outside_the_tabulated_range_is_zero():
    """ENDF has no cross-section below threshold or above the evaluation's top
    energy, and extrapolating a log-log curve outward diverges fast."""
    got = interpolate([1.0, 2.0], [10.0, 20.0], [LIN_LIN, LIN_LIN], [0.5, 3.0])
    assert list(got) == [0.0, 0.0]


def test_a_null_law_refuses_rather_than_assuming():
    """A NULL means the source never said. The helper must not decide on the
    caller's behalf — that choice belongs to whoever knows the provenance."""
    with pytest.raises(ValueError, match="no interpolation law"):
        interpolate([1.0, 2.0], [10.0, 20.0], [None, None], [1.5])


def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        interpolate([1.0, 2.0], [10.0], [LIN_LIN, LIN_LIN], [1.5])


# ---------------------------------------------------------------------------
# Closure: which laws survive the operations the ingest performs
# ---------------------------------------------------------------------------
#
# Two builders combine curves, and each has to decide what law the result
# carries. The answer is not a matter of taste — it is whether the operation is
# exact under that law — so it is computed here rather than asserted in prose.


@pytest.mark.parametrize("law", [1, 2, 3, 4, 5])
def test_which_laws_are_closed_under_multiplication(law):
    """MF=9 emits sigma(E) x Y(E) (#390). Can the product inherit an input law?

    Only if the law is closed under multiplication. Laws 4 and 5 are, because
    they are logarithmic in y and ln(sigma*Y) = ln(sigma) + ln(Y) stays linear
    in the law's own x-axis. Law 1 is, being constant. Laws 2 and 3 are not: a
    linear times a linear is a quadratic, which no ENDF law spells.

    This is why `parse_mf9_rows` writes NULL rather than inheriting — see the
    companion test below for what inheriting would cost.
    """
    x1, x2 = 1.0e6, 2.0e6
    s1, s2 = 0.100, 0.400  # sigma, barns
    y1, y2 = 0.90, 0.30  # yield
    xm = math.sqrt(x1 * x2) if law in (3, 5) else 0.5 * (x1 + x2)

    product_of_curves = interpolate_one(law, xm, x1, x2, s1, s2) * interpolate_one(law, xm, x1, x2, y1, y2)
    curve_of_products = interpolate_one(law, xm, x1, x2, s1 * y1, s2 * y2)

    closed_under_multiplication = law in {1, 4, 5}
    if closed_under_multiplication:
        assert curve_of_products == pytest.approx(product_of_curves, rel=1e-12)
    else:
        assert curve_of_products != pytest.approx(product_of_curves, rel=1e-6)


def test_inheriting_lin_lin_through_a_product_would_be_wrong_by_a_third():
    """The cost of the tempting choice, in a number.

    Law 2 is 92% of all regions, so "just inherit" would in practice mean
    "claim lin-lin". On an ordinary interval — sigma rising 100 -> 400 mb while
    the yield falls 0.90 -> 0.30 — the true product at the midpoint is 150 mb
    and the lin-lin chord through the endpoint products says 105 mb.
    """
    x1, x2 = 1.0e6, 2.0e6
    s1, s2, y1, y2 = 0.100, 0.400, 0.90, 0.30
    xm = 0.5 * (x1 + x2)

    truth = interpolate_one(LIN_LIN, xm, x1, x2, s1, s2) * interpolate_one(LIN_LIN, xm, x1, x2, y1, y2)
    claimed = interpolate_one(LIN_LIN, xm, x1, x2, s1 * y1, s2 * y2)

    assert truth == pytest.approx(0.150)
    assert claimed == pytest.approx(0.105)
    assert abs(claimed - truth) / truth == pytest.approx(0.30, abs=1e-9)


@pytest.mark.parametrize("law", [1, 2, 3, 4, 5])
def test_lin_lin_is_the_law_closed_under_ADDITION(law):
    """The mirror image, and why the two builders answer differently.

    `sum_on_union_grid` adds curves, and addition is exact under law 2 — a sum
    of linears is linear — which is what lets a summed production row claim
    law 2 when every contribution is law 2. Multiplication is exact under the
    logarithmic laws instead. Same question, opposite answer, so the two paths
    must not share a rule.
    """
    x1, x2 = 1.0e6, 2.0e6
    a1, a2, b1, b2 = 0.100, 0.400, 0.90, 0.30
    xm = math.sqrt(x1 * x2) if law in (3, 5) else 0.5 * (x1 + x2)

    sum_of_curves = interpolate_one(law, xm, x1, x2, a1, a2) + interpolate_one(law, xm, x1, x2, b1, b2)
    curve_of_sums = interpolate_one(law, xm, x1, x2, a1 + b1, a2 + b2)

    if law in {1, 2, 3}:
        assert curve_of_sums == pytest.approx(sum_of_curves, rel=1e-12)
    else:
        assert curve_of_sums != pytest.approx(sum_of_curves, rel=1e-6)
