"""Exact equivalence checks for mechanical certificate optimizations."""

from fractions import Fraction
from math import gcd

import numpy as np
import pytest

import strategy_inference.uncertainty as uncertainty
import strategy_inference.wilks as wilks


def _ratio_column(values):
    """The pre-optimization conversion, independently using float ratios."""
    ratios = tuple(float(value).as_integer_ratio() for value in values)
    exponent = max(denominator.bit_length() - 1 for _, denominator in ratios)
    integers = tuple(
        numerator << (exponent - denominator.bit_length() + 1) for numerator, denominator in ratios
    )
    divisor = gcd(*integers)
    return tuple(value // divisor for value in integers) if divisor else integers


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_binary64_decode_matches_independent_ratios(seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, np.iinfo(np.uint64).max, size=300, dtype=np.uint64)
    bits &= np.uint64(0xFFEFFFFFFFFFFFFF)  # Exclude NaN and infinity bit patterns.
    values = bits.view(np.float64)
    assert uncertainty._integer_column(values) == _ratio_column(values)
    assert uncertainty._integer_column(values[::3]) == _ratio_column(values[::3])


@pytest.mark.parametrize(
    "values",
    [
        [0.0, -0.0, 0.0],
        [np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0), 0.0],
        [np.finfo(float).max, np.finfo(float).tiny, np.nextafter(0.0, 1.0), -0.0],
        [1.0, -1.5, 0.125, -16.0],
    ],
)
def test_binary64_decode_extremes_and_signed_zero(values):
    native = np.array(values)
    foreign_endian = native.astype(">f8")
    expected = _ratio_column(native)
    assert uncertainty._integer_column(native) == expected
    assert uncertainty._integer_column(foreign_endian) == expected


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 8, 16])
def test_de_casteljau_children_equal_direct_integer_power_conversion(degree):
    rng = np.random.default_rng(112 + degree)
    coefficients = tuple(map(int, rng.integers(-1000, 1001, size=degree + 1)))
    for cell in ((0, 1, 0), (3, 9, 4), (111, 335, 10), (0, 1, 64)):
        bounds = uncertainty._bernstein(coefficients, *cell)
        left, right = uncertainty._split(cell)
        children = uncertainty._split_bernstein(bounds)
        assert children == (
            uncertainty._bernstein(coefficients, *left),
            uncertainty._bernstein(coefficients, *right),
        )
        # A second split checks that accumulating the positive scaling is exact.
        for child, child_bounds in zip((left, right), children, strict=True):
            grandchildren = uncertainty._split(child)
            assert uncertainty._split_bernstein(child_bounds) == tuple(
                uncertainty._bernstein(coefficients, *x) for x in grandchildren
            )


def _direct_intervals(constraints, max_depth):
    """Old traversal, converting power coefficients separately at every node."""
    stack, kept, unresolved = [(0, 1, 0)], [], 0
    while stack:
        cell = stack.pop()
        bounds = tuple(uncertainty._bernstein(polynomial, *cell) for polynomial in constraints)
        if any(max(bound) < 0 for bound in bounds):
            continue
        if all(min(bound) >= 0 for bound in bounds):
            kept.append(cell)
        elif cell[2] >= max_depth:
            kept.append(cell)
            unresolved += 1
        else:
            left, right = uncertainty._split(cell)
            stack.extend((right, left))
    merged = []
    for left, right, depth in kept:
        interval = Fraction(left, 1 << depth), Fraction(right, 1 << depth)
        if merged and merged[-1][1] == interval[0]:
            merged[-1] = merged[-1][0], interval[1]
        else:
            merged.append(interval)
    return tuple(merged), unresolved


@pytest.mark.parametrize("depth", [0, 3, 16])
@pytest.mark.parametrize(
    "constraints",
    [(), ((0,),), ((3, -16, 16),), ((-1, 6, -6), (1, -1)), ((1, -7, 0, 21, -15),)],
)
def test_reused_bounds_preserve_exact_outer_sets_and_unresolved_counts(constraints, depth):
    assert wilks._confidence_intervals(constraints, depth) == _direct_intervals(constraints, depth)


def _direct_candidate(a, r, intervals, max_depth, max_nodes):
    """Old certificate traversal; node counts and budget exhaustion are observable."""
    if not intervals:
        return False, False, 0
    stack = [(uncertainty._interval_cell(x), 0) for x in intervals[::-1]]
    nodes = 0
    while stack:
        cell, refinements = stack.pop()
        if nodes >= max_nodes:
            return False, True, nodes
        nodes += 1
        bounds_a = uncertainty._bernstein(a, *cell)
        bounds_r = uncertainty._bernstein(r, *cell)
        if min(bounds_a) > 0 and min(bounds_r) > 0:
            continue
        if min(bounds_a[0], bounds_a[-1], bounds_r[0], bounds_r[-1]) <= 0:
            return False, False, nodes
        if refinements >= max_depth:
            return False, True, nodes
        left, right = uncertainty._split(cell)
        stack.extend(((right, refinements + 1), (left, refinements + 1)))
    return True, False, nodes


@pytest.mark.parametrize("depth,budget", [(0, 1), (1, 2), (20, 1), (20, 4096)])
def test_reused_candidate_bounds_preserve_decisions_and_budget_diagnostics(depth, budget):
    intervals = ((Fraction(1, 8), Fraction(7, 8)),)
    for a, r in (((1,), (1, -4, 4)), ((1, -1), (3, -10, 10)), ((1,), (-1, 8, -8))):
        assert uncertainty._certify_candidate(a, r, intervals, depth, budget) == _direct_candidate(
            a, r, intervals, depth, budget
        )


def _direct_critical(n_obs, k, budget, bits):
    """Full-range exact bisection; it uses no floating quantile or initial guess."""
    df, denominator = 2 * ((n_obs - 1) // 2), 1 << bits
    left, right = 0, denominator
    while right - left > 1:
        middle = (left + right) // 2
        if uncertainty._student_tail_at_u(middle, denominator, df) <= budget / k:
            right = middle
        else:
            left = middle
    if right == denominator:
        raise ValueError("insufficient bits")
    u = Fraction(right, denominator)
    return df * u * u / (1 - u * u)


@pytest.mark.parametrize(
    "n_obs,k,budget,bits",
    [
        (8, 1, Fraction(45, 1000), 8),
        (33, 20, Fraction(45, 1000), 40),
        (128, 100, Fraction(1, 1000), 40),
        (512, 20, Fraction(45, 1000), 40),
        (64, 5, Fraction(1, 37), 128),
    ],
)
def test_student_seed_selects_identical_exact_dyadic_cutoff(n_obs, k, budget, bits):
    assert uncertainty._critical_squared(n_obs, k, budget, bits) == _direct_critical(
        n_obs, k, budget, bits
    )


@pytest.mark.parametrize("guess", [-1.0, 0.0, 20.0])
def test_bad_student_seed_is_corrected_by_exact_bracket_expansion(monkeypatch, guess):
    class BadNormal:
        def inv_cdf(self, _):
            return guess

    expected = _direct_critical(33, 7, Fraction(3, 100), 16)
    monkeypatch.setattr(uncertainty, "NormalDist", BadNormal)
    uncertainty._critical_squared.cache_clear()
    try:
        assert uncertainty._critical_squared(33, 7, Fraction(3, 100), 16) == expected
    finally:
        uncertainty._critical_squared.cache_clear()


def test_student_float_underflow_preserves_conservative_failure():
    with pytest.raises(ValueError, match="insufficient"):
        uncertainty._critical_squared(8, 1, Fraction(1, 10**400), 8)


@pytest.mark.parametrize("dimension", [1, 3, 8])
def test_shared_prefix_grams_match_separate_exact_scatter_polynomials(dimension):
    rng = np.random.default_rng(319 + dimension)
    columns = tuple(tuple(map(int, row)) for row in rng.integers(-99, 100, size=(dimension, 257)))
    plans = ((4, 252), (16, 240), (64, 192))
    grams = wilks._scatter_grams(columns, (240, 192, 252, 240))
    assert set(grams) == {n for _, n in plans}
    for length, n in plans:
        assert wilks._scatter_polynomials(columns, length, n, gram_matrix=grams[n]) == (
            wilks._scatter_polynomials(columns, length, n)
        )
    assert wilks._scatter_grams(columns, ()) == {}
