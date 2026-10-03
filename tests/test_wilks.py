"""Independent determinant, distributional-bound and enclosure checks."""

from fractions import Fraction
from itertools import permutations
from math import prod

import numpy as np
import pytest
from scipy.stats import beta as beta_distribution

import strategy_inference.wilks as wilks
from strategy_inference import WilksResult, WilksScale, wilks_uncertainty_test
from strategy_inference.parametric import known_phi_gls_t


def _evaluate(coefficients, x):
    return sum(value * x**i for i, value in enumerate(coefficients))


def _permutation_determinant(matrix):
    total = 0
    for permutation in permutations(range(len(matrix))):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(len(matrix))
            for j in range(i + 1, len(matrix))
        )
        total += (-1) ** inversions * prod(matrix[i][j] for i, j in enumerate(permutation))
    return total


def _data(n=128, k=3):
    return np.random.default_rng(20261119).integers(-20, 21, size=(n, k)).astype(float)


def _contains(intervals, point):
    return any(left <= point <= right for left, right in intervals)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5])
def test_bareiss_matches_independent_permutation_determinants(size):
    rng = np.random.default_rng(947)
    for _ in range(8):
        matrix = tuple(tuple(map(int, row)) for row in rng.integers(-8, 9, size=(size, size)))
        assert wilks._determinant(matrix) == _permutation_determinant(matrix)


def test_bareiss_handles_row_swap_and_singular_inputs():
    for matrix in (((0, 2), (3, 1)), ((1, 2, 3), (2, 4, 6), (1, 0, 1))):
        assert wilks._determinant(matrix) == _permutation_determinant(matrix)


def test_determinant_polynomial_is_exact_at_points_not_used_for_interpolation():
    rng = np.random.default_rng(153)
    matrix = tuple(
        tuple(tuple(map(int, value)) for value in row)
        for row in rng.integers(-4, 5, size=(3, 3, 3))
    )
    polynomial = wilks._determinant_polynomial(matrix)
    for x in (Fraction(-3, 2), Fraction(1, 3), Fraction(7, 9), Fraction(19, 2)):
        evaluated = tuple(tuple(a + x * b + x * x * c for a, b, c in row) for row in matrix)
        assert _evaluate(polynomial, x) == _permutation_determinant(evaluated)


def test_interpolation_requires_integer_power_coefficients():
    assert wilks._interpolate_integer((3, 2, 7, 24)) == (3, -2, 0, 1)
    with pytest.raises(ArithmeticError, match="noninteger"):
        wilks._interpolate_integer((0, 0, 1))


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_scatter_ratio_matches_independent_exact_gram_calculation(dimension):
    columns = tuple(tuple(map(int, column)) for column in _data(64, dimension).T)
    n, length = 60, 4
    within, total = wilks._scatter_polynomials(columns, length, n)
    for phi in (Fraction(), Fraction(1, 4), Fraction(9, 10), Fraction(1), Fraction(3, 2)):
        z = [[column[t + 1] - phi * column[t] for column in columns] for t in range(n)]
        sums = [sum(row[j] for row in z) for j in range(dimension)]
        grouped = [
            [sum(row[j] for row in z[start : start + length]) for j in range(dimension)]
            for start in range(0, n, length)
        ]
        a = tuple(
            tuple(
                sum(row[i] * row[j] for row in z) - sum(row[i] * row[j] for row in grouped) / length
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        b = tuple(
            tuple(
                sum(row[i] * row[j] for row in z) - sums[i] * sums[j] / n for j in range(dimension)
            )
            for i in range(dimension)
        )
        ratio = _permutation_determinant(a) / _permutation_determinant(b)
        assert Fraction(_evaluate(within, phi), _evaluate(total, phi)) == ratio
        assert 0 <= ratio <= 1


@pytest.mark.parametrize(
    "dimension,q,s,order", [(1, 2, 30, 3), (8, 2, 40, 12), (3, 30, 100, 64), (8, 6, 441, 256)]
)
def test_moments_match_independent_beta_rising_factorial_formula(dimension, q, s, order):
    expected = Fraction(1)
    for i in range(1, dimension + 1):
        a, b = Fraction(s - i + 1, 2), Fraction(q, 2)
        expected *= prod(a + u for u in range(order)) / prod(a + b + u for u in range(order))
    assert wilks._moment(dimension, q, s, order) == expected
    if 2 * order < s - dimension + 1:
        expected = Fraction(1)
        for i in range(1, dimension + 1):
            a, b = Fraction(s - i + 1, 2), Fraction(q, 2)
            expected *= prod(a + b - u for u in range(1, order + 1)) / prod(
                a - u for u in range(1, order + 1)
            )
        assert wilks._moment(dimension, q, s, -order) == expected


def test_negative_moment_boundary_is_strict_and_q_may_be_smaller_than_dimension():
    assert wilks._moment(8, 2, 40, -16) > 1
    with pytest.raises(ValueError, match="does not exist"):
        wilks._moment(8, 2, 40, -17)


@pytest.mark.parametrize(
    "target,order,bits",
    [
        (Fraction(1, 2), 1, 40),
        (Fraction(1, 8), 3, 40),
        (Fraction(123, 321), 64, 40),
        (Fraction(1, 10**300), 4096, 40),
        (Fraction(999, 1000), 7, 128),
    ],
)
def test_root_brackets_are_verified_by_exact_integer_powers(target, order, bits):
    left, right = wilks._root_bracket(target, order, bits)
    assert left**order <= target <= right**order
    assert right - left <= Fraction(1, 1 << bits)


def test_inaccurate_floating_root_seed_does_not_change_certificate(monkeypatch):
    target = Fraction(79, 101)
    expected = wilks._root_bracket(target, 3, 20)
    monkeypatch.setattr(wilks, "exp", lambda _: 0.0)
    assert wilks._root_bracket(target, 3, 20) == expected


def test_positive_order_search_passes_64_and_reaches_the_prespecified_limit():
    orders = wilks._positive_orders(448)
    assert orders[:8] == (1, 2, 3, 4, 6, 8, 12, 16)
    assert 256 in orders
    assert orders[-1] == 2048


@pytest.mark.parametrize("dimension,q,s", [(1, 2, 30), (8, 6, 441), (8, 126, 381)])
def test_selected_cutoffs_satisfy_exact_chernoff_probability_bounds(dimension, q, s):
    beta = Fraction(1, 600)
    (lower, upper), low_h, high_h = wilks._wilks_cutoffs(dimension, q, s, beta, 40)
    assert 0 <= lower < upper <= 1
    assert wilks._moment(dimension, q, s, -low_h) * lower**low_h <= beta / 2
    if upper < 1:
        assert wilks._moment(dimension, q, s, high_h) <= beta * upper**high_h / 2
    if dimension == 1:
        miss = beta_distribution.cdf(float(lower), s / 2, q / 2) + beta_distribution.sf(
            float(upper), s / 2, q / 2
        )
        assert miss <= float(beta) + 1e-13


def test_multiscale_enclosure_preserves_disconnected_acceptance_sets():
    intervals, unresolved = wilks._confidence_intervals(((3, -16, 16),), 10)
    assert len(intervals) == 2
    assert unresolved > 0
    for i in range(1001):
        point = Fraction(i, 1000)
        if _evaluate((3, -16, 16), point) >= 0:
            assert _contains(intervals, point)
    assert not _contains(intervals, Fraction(1, 2))
    assert wilks._confidence_intervals((), 0) == (((Fraction(), Fraction(1)),), 0)
    assert wilks._confidence_intervals(((-1,),), 16) == ((), 0)


def test_joint_shape_set_is_exactly_invariant_to_column_mixing_scales_and_means():
    data = _data(128, 3)
    mixed = data @ np.array([[1, 2, 0], [0, 1, 1], [0, 0, 2]]) + [100, -400, 800]
    original = wilks.wilks_uncertainty_test(data, max_dimension=3, block_lengths=(4, 16))
    transformed = wilks.wilks_uncertainty_test(mixed, max_dimension=3, block_lengths=(4, 16))
    assert original.intervals == transformed.intervals
    reordered = wilks.wilks_uncertainty_test(data, max_dimension=3, block_lengths=(16, 4))
    assert original.intervals == reordered.intervals


def test_singular_covariance_fallback_retains_the_full_parameter_domain():
    first = _data(128, 1)[:, 0]
    data = np.column_stack((first, first * 2 + 1000, first * -3 - 1000))
    result = wilks.wilks_uncertainty_test(data, max_dimension=3)
    assert result.singular_fallback
    assert result.intervals == ((Fraction(), Fraction(1)),)
    assert result.phi1_retained and not result.global_reject
    assert all(
        scale.det_total == (0,) and scale.det_within == (0,)
        for scale in result.scales
        if scale.usable
    )
    scalar = wilks.wilks_uncertainty_test(data, max_dimension=1)
    assert not scalar.singular_fallback
    assert scalar.ci_width < 1


def test_insufficient_observations_keep_all_scales_without_reallocating_beta():
    short = wilks.wilks_uncertainty_test(_data(8, 3))
    assert not any(scale.usable for scale in short.scales)
    assert short.intervals == ((Fraction(), Fraction(1)),)
    assert not short.global_reject
    result = wilks.wilks_uncertainty_test(_data(128, 3))
    assert [scale.usable for scale in result.scales] == [True, True, False]
    for scale in result.scales[:2]:
        expected, _, _ = wilks._wilks_cutoffs(
            result.dimension,
            scale.low_df,
            scale.high_df,
            Fraction(result.beta) / 3,
            result.critical_bits,
        )
        assert scale.cutoffs == expected


def test_signal_rejection_and_ci_coarsening_have_the_expected_conservative_direction():
    data = _data(256, 3)
    data[:, 0] += 1000
    fine = wilks.wilks_uncertainty_test(data, max_dimension=3, block_lengths=(4,))
    coarse = wilks.wilks_uncertainty_test(data, max_dimension=3, block_lengths=(4,), ci_depth=0)
    assert fine.decisions[0]
    assert coarse.phi1_retained and not coarse.global_reject
    for left, right in fine.intervals:
        assert _contains(coarse.intervals, left) and _contains(coarse.intervals, right)
        for phi in np.linspace(float(left), float(right), 17):
            if phi < 1:
                t = known_phi_gls_t(data, phi)
                assert np.all(t[fine.decisions] ** 2 > float(fine.critical_squared))
                assert np.all(t[fine.decisions] > 0)


def test_result_reports_predeclared_dimensions_odd_blocks_and_read_only_arrays():
    result = wilks.wilks_uncertainty_test(_data(256, 3), max_dimension=8)
    assert result.dimension == 3 and result.max_dimension == 8
    assert result.block_lengths == (4, 16, 64)
    assert all(scale.low_df % 2 == 0 for scale in result.scales)
    assert all(scale.n_innovations // scale.block_length % 2 == 1 for scale in result.scales)
    assert result.global_reject == bool(result.decisions.any())
    assert result.empty_ci == (not result.intervals)
    assert result.interval_bounds == tuple((float(a), float(b)) for a, b in result.intervals)
    for array in (result.decisions, result.certificate_unresolved, result.certificate_nodes):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 0


def test_public_exports_are_the_new_fixed_level_interface():
    assert wilks_uncertainty_test is wilks.wilks_uncertainty_test
    assert WilksResult is wilks.WilksResult
    assert WilksScale is wilks.WilksScale


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_dimension": 0},
        {"max_dimension": 9},
        {"max_dimension": True},
        {"block_lengths": ()},
        {"block_lengths": (4, 4)},
        {"block_lengths": (1,)},
        {"block_lengths": "4"},
        {"block_lengths": 4},
        {"block_lengths": (4.0,)},
        {"beta": 0.05},
        {"alpha": 0.6},
        {"ci_depth": 65},
        {"certificate_depth": -1},
        {"max_nodes": 0},
        {"critical_bits": 7},
        {"critical_bits": 129},
    ],
)
def test_invalid_new_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        wilks.wilks_uncertainty_test(_data(), **kwargs)
