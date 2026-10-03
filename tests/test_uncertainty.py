"""Independent algebra, distribution and enclosure checks for fixed-level inference."""

from fractions import Fraction
from math import comb, lcm

import numpy as np
import pytest
from scipy.fft import dct
from scipy.stats import f, kstest, t

from strategy_inference.parametric import known_phi_gls_t
from strategy_inference.uncertainty import (
    _bernstein,
    _beta_bracket,
    _beta_numerator,
    _certify_candidate,
    _confidence_intervals,
    _critical_squared,
    _f_cutoffs,
    _fraction_inverse,
    _gls_polynomials,
    _innovation_quadratics,
    _integer_column,
    _Projection,
    _projection,
    _student_tail_at_u,
    uncertainty_test,
)


def _evaluate(coefficients, value):
    return sum(coefficient * value**i for i, coefficient in enumerate(coefficients))


def test_integer_column_preserves_binary64_ratios_exactly():
    values = np.array([0.125, -1.5, np.nextafter(2.0, 3.0), 0.0, 4.0])
    integers = _integer_column(values)
    scale = Fraction(integers[0]) / Fraction(float(values[0]))
    assert scale > 0
    assert all(
        Fraction(integer) == scale * Fraction(float(value))
        for integer, value in zip(integers, values, strict=True)
    )


@pytest.mark.parametrize("coefficients", [(0,), (7,), (1, -4, 4), (2, -7, 3, 9)])
def test_bernstein_bounds_enclose_exact_polynomial(coefficients):
    left, right, depth = 3, 13, 4
    bounds = _bernstein(coefficients, left, right, depth)
    degree = len(coefficients) - 1
    common = 1
    for i in range(degree + 1):
        common = lcm(common, comb(degree, i))
    denominator = common * (1 << depth) ** degree
    for numerator in range(41):
        value = Fraction(left, 1 << depth) + Fraction(numerator, 40) * Fraction(
            right - left, 1 << depth
        )
        evaluated = _evaluate(coefficients, value)
        assert Fraction(min(bounds), denominator) <= evaluated <= Fraction(max(bounds), denominator)


@pytest.mark.parametrize("a,b", [(1, 1), (2, 3), (4, 7), (5, 50)])
def test_integer_beta_cdf_and_outer_bisection(a, b):
    numerator, denominator = 17, 64
    degree = a + b - 1
    expected = sum(
        Fraction(comb(degree, j))
        * Fraction(numerator, denominator) ** j
        * Fraction(denominator - numerator, denominator) ** (degree - j)
        for j in range(a, degree + 1)
    )
    assert Fraction(_beta_numerator(numerator, denominator, a, b), denominator**degree) == expected
    target = Fraction(1, 400)
    low, high = _beta_bracket(a, b, target, 32)
    assert high - low == Fraction(1, 1 << 32)
    for value, side in [(low, "low"), (high, "high")]:
        cdf = Fraction(
            _beta_numerator(value.numerator, value.denominator, a, b), value.denominator**degree
        )
        assert cdf <= target if side == "low" else cdf > target


@pytest.mark.parametrize("n_obs", [8, 128, 200, 512, 1024])
def test_F_cutoffs_are_outside_analytic_quantiles(n_obs):
    projection = _projection(n_obs)
    low, high = _f_cutoffs(projection.q, projection.s, Fraction(1, 200), 40)
    expected = f.ppf([0.0025, 0.9975], projection.q, projection.s)
    assert float(low) <= expected[0] + 2e-13
    assert float(high) >= expected[1] - 2e-13
    assert np.allclose([float(low), float(high)], expected, rtol=2e-8, atol=2e-10)
    for value, target, lower in [(low, Fraction(1, 400), True), (high, Fraction(399, 400), False)]:
        u = projection.q * value / (projection.s + projection.q * value)
        degree = (projection.q + projection.s) // 2 - 1
        cdf = Fraction(
            _beta_numerator(u.numerator, u.denominator, projection.q // 2, projection.s // 2),
            u.denominator**degree,
        )
        assert cdf <= target if lower else cdf >= target


@pytest.mark.parametrize("even_df", [2, 4, 6, 126, 198, 510, 1022])
def test_exact_Student_polynomial_matches_independent_cdf(even_df):
    u = Fraction(3, 32)
    exact = _student_tail_at_u(u.numerator, u.denominator, even_df)
    c = np.sqrt(even_df * float(u * u / (1 - u * u)))
    assert float(exact) == pytest.approx(t.sf(c, even_df), abs=3e-15)
    if even_df == 2:
        assert exact == (1 - u) / 2
    if even_df == 4:
        assert exact == Fraction(1, 2) - Fraction(3, 4) * (u - u**3 / 3)


@pytest.mark.parametrize("n_obs,k", [(8, 1), (128, 20), (200, 1), (512, 100), (1024, 20)])
def test_critical_value_is_conservative_for_actual_GLS_df(n_obs, k):
    budget = Fraction(9, 200)
    squared = _critical_squared(n_obs, k, budget, 40)
    even_df = 2 * ((n_obs - 1) // 2)
    u_squared = squared / (even_df + squared)
    denominator = 1 << 40
    numerator = round(float(u_squared) ** 0.5 * denominator)
    assert Fraction(numerator, denominator) ** 2 == u_squared
    assert _student_tail_at_u(numerator, denominator, even_df) <= budget / k
    assert _student_tail_at_u(numerator - 1, denominator, even_df) > budget / k
    assert t.sf(float(squared) ** 0.5, n_obs - 1) <= float(budget / k)


@pytest.mark.parametrize(
    "n_obs,n,q",
    [(8, 7, 2), (28, 27, 4), (128, 127, 6), (129, 127, 6), (344, 343, 8), (512, 511, 8)],
)
def test_projection_rank_rule_and_exact_zero_mean(n_obs, n, q):
    projection = _projection(n_obs)
    assert (projection.n, projection.q, projection.s) == (n, q, n - 1 - q)
    assert q % 2 == projection.s % 2 == 0
    assert all(sum(row) == 0 for row in projection.basis)
    gram = [
        [sum(x * y for x, y in zip(left, right, strict=True)) for right in projection.basis]
        for left in projection.basis
    ]
    for i in range(q):
        for j in range(q):
            product = sum(gram[i][k] * projection.inverse_numerator[k][j] for k in range(q))
            assert product == (projection.inverse_denominator if i == j else 0)


def test_rational_surrogate_preserves_the_low_DCT_subspace():
    projection = _projection(128)
    basis = np.asarray(projection.basis, dtype=float)
    q, _ = np.linalg.qr(basis.T)
    exact_dct = dct(np.eye(projection.n), norm="ortho", axis=0)[1 : projection.q + 1].T
    assert np.linalg.norm(q @ q.T - exact_dct @ exact_dct.T, ord=2) < 2e-5


def test_rank_deficient_projection_aborts():
    with pytest.raises(ValueError, match="rank deficient"):
        _fraction_inverse(((1, 2), (2, 4)))


def test_innovation_quadratics_equal_a_direct_rational_projection():
    column = (3, -1, 2, 5, 0, -2, 4, 1)
    projection = _projection(len(column))
    low, high = _innovation_quadratics(column, projection)
    phi = Fraction(2, 3)
    z = [Fraction(column[i + 1]) - phi * column[i] for i in range(projection.n)]
    coordinates = [
        sum(value * weight for value, weight in zip(z, row, strict=True))
        for row in projection.basis
    ]
    direct_low = (
        sum(
            coordinates[i] * projection.inverse_numerator[i][j] * coordinates[j]
            for i in range(projection.q)
            for j in range(projection.q)
        )
        / projection.inverse_denominator
    )
    direct_high = sum(value * value for value in z) - sum(z) ** 2 / projection.n - direct_low
    assert _evaluate(low, phi) / _evaluate(high, phi) == direct_low / direct_high
    assert direct_low > 0 and direct_high > 0


@pytest.mark.parametrize("phi", [0.0, 0.7, 0.99])
def test_true_AR_innovations_have_the_predicted_F_distribution(phi):
    # This checks the actual AR transform, rather than assuming fixed lagged
    # regressors. It is a deterministic development check, not a level proof.
    rng = np.random.default_rng(719488219)
    count, n_obs = 6000, 64
    paths = rng.normal(size=(count, n_obs))
    for i in range(1, n_obs):
        paths[:, i] = phi * paths[:, i - 1] + np.sqrt(1 - phi * phi) * paths[:, i]
    paths = 3.0 + 2.5 * paths
    projection = _projection(n_obs)
    z = paths[:, 1:] - phi * paths[:, :-1]
    basis = np.asarray(projection.basis, dtype=float)
    coordinates = z @ basis.T
    inverse = np.asarray(
        [
            [float(Fraction(value, projection.inverse_denominator)) for value in row]
            for row in projection.inverse_numerator
        ]
    )
    low = np.einsum("bi,ij,bj->b", coordinates, inverse, coordinates)
    high = np.sum((z - z.mean(axis=1, keepdims=True)) ** 2, axis=1) - low
    ratio = (low / projection.q) / (high / projection.s)
    assert kstest(ratio, f(projection.q, projection.s).cdf).statistic < 0.025


def test_disconnected_confidence_set_is_enclosed_without_a_grid_maximum():
    projection = _Projection(7, 2, 4, (), (), 1)
    intervals, unresolved = _confidence_intervals(
        (1, -4, 4), (1, 0, 0), projection, (Fraction(1, 10), Fraction(3, 10)), 16
    )
    expected = [
        ((1 - np.sqrt(0.15)) / 2, (1 - np.sqrt(0.05)) / 2),
        ((1 + np.sqrt(0.05)) / 2, (1 + np.sqrt(0.15)) / 2),
    ]
    assert len(intervals) == 2
    assert unresolved > 0
    for (left, right), (truth_left, truth_right) in zip(intervals, expected, strict=True):
        assert float(left) <= truth_left <= truth_right <= float(right)
        assert truth_left - float(left) < 2**-16
        assert float(right) - truth_right < 2**-16


def test_confidence_enclosure_retains_uncertain_cells_and_empty_sets():
    projection = _Projection(7, 2, 4, (), (), 1)
    cutoffs = Fraction(1, 10), Fraction(3, 10)
    whole, unresolved = _confidence_intervals((1, -4, 4), (1, 0, 0), projection, cutoffs, 0)
    assert whole == ((Fraction(0), Fraction(1)),)
    assert unresolved == 1
    empty, unresolved = _confidence_intervals((1, 0, 0), (1, 0, 0), projection, cutoffs, 8)
    assert empty == ()
    assert unresolved == 0


@pytest.mark.parametrize("phi", [-0.9, 0.0, 0.7, 0.99, 0.999999])
def test_GLS_polynomial_statistic_matches_independent_whitening(phi):
    column = tuple(np.random.default_rng(93141121).integers(-40, 41, size=31).tolist())
    a, h, _ = _gls_polynomials(column, Fraction(7))
    statistic = (
        np.sqrt(len(column) - 1) * _evaluate(a, phi) * np.sqrt(1 - phi) / np.sqrt(_evaluate(h, phi))
    )
    expected = known_phi_gls_t(np.array(column), phi)[0]
    assert statistic == pytest.approx(expected, rel=4e-11, abs=1e-11)


def test_GLS_critical_cubic_is_exact_and_unit_boundary_is_nonpositive():
    column = (3, -1, 2, 5, 0, -2, 4, 1)
    critical = Fraction(13, 2)
    a, h, r = _gls_polynomials(column, critical)
    for phi in [Fraction(0), Fraction(1, 4), Fraction(3, 4), Fraction(1)]:
        raw = (len(column) - 1) * (1 - phi) * _evaluate(a, phi) ** 2 - critical * _evaluate(h, phi)
        assert np.sign(_evaluate(r, phi)) == np.sign(raw)
    assert _evaluate(h, Fraction(1)) == 2 * sum(
        (y - x) ** 2 for x, y in zip(column[:-1], column[1:], strict=True)
    )
    assert _evaluate(r, Fraction(1)) < 0


def test_certificate_catches_an_interior_failure_despite_positive_endpoints():
    intervals = ((Fraction(0), Fraction(1)),)
    decision, unresolved, nodes = _certify_candidate((1,), (1, -8, 8), intervals, 20, 4096)
    assert not decision and not unresolved
    assert nodes > 1


def test_certificate_budget_can_only_withhold_a_positive_decision():
    intervals = ((Fraction(0), Fraction(1)),)
    a, r = (1,), (1, -3, 3)
    assert _certify_candidate(a, r, intervals, 20, 4096) == (True, False, 3)
    assert _certify_candidate(a, r, intervals, 0, 4096) == (False, True, 1)
    assert _certify_candidate(a, r, intervals, 20, 1) == (False, True, 1)
    assert _certify_candidate(a, r, (), 20, 4096) == (False, False, 0)


def test_mean_and_scale_invariant_confidence_set_with_power_translation():
    data = np.random.default_rng(12).integers(-20, 21, size=(64, 2)).astype(float) / 16
    original = uncertainty_test(data)
    shifted = uncertainty_test(data + np.array([100.0, -100.0]))
    scaled = uncertainty_test(data * np.array([8.0, 0.25]))
    assert original.intervals == shifted.intervals == scaled.intervals
    np.testing.assert_array_equal(original.decisions, scaled.decisions)
    assert shifted.decisions.tolist() == [True, False]
    assert shifted.global_reject
    assert not shifted.phi1_retained
    assert shifted.n_innovations == 63 and shifted.low_df == 4 and shifted.high_df == 58


def test_retaining_phi_one_blocks_rejection_even_with_a_large_signal():
    data = np.random.default_rng(11).integers(-20, 21, size=(64, 2)).astype(float) / 16
    result = uncertainty_test(data + np.array([10000.0, 0.0]))
    assert result.phi1_retained
    assert not result.global_reject
    assert not result.certificate_unresolved.any()


def test_coarse_CI_is_conservative_and_results_are_read_only():
    data = np.random.default_rng(12).integers(-20, 21, size=(64, 2)).astype(float) / 16 + 100
    fine = uncertainty_test(data)
    coarse = uncertainty_test(data, ci_depth=0)
    assert fine.global_reject
    assert coarse.intervals == ((Fraction(0), Fraction(1)),)
    assert not coarse.global_reject
    assert fine.ci_width <= coarse.ci_width
    for array in [fine.decisions, fine.certificate_unresolved, fine.certificate_nodes]:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.5},
        {"beta": 0.05},
        {"beta": 0},
        {"alpha": "0.05"},
        {"reference": -1},
        {"reference": 2},
        {"reference": True},
        {"ci_depth": -1},
        {"certificate_depth": -1},
        {"max_nodes": 0},
        {"critical_bits": 7},
        {"ci_depth": 65},
        {"critical_bits": 129},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    data = np.random.default_rng(39).normal(size=(16, 2))
    with pytest.raises(ValueError):
        uncertainty_test(data, **kwargs)


@pytest.mark.parametrize("data", [np.ones(16), np.arange(7), [[1, np.nan]] * 8, ["one"] * 8])
def test_invalid_data_are_rejected_without_dropping_observations(data):
    with pytest.raises(ValueError):
        uncertainty_test(data)
