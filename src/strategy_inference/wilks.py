"""Multiscale Gaussian AR shape sets and certified simultaneous GLS decisions.

Fixed columns and fixed block contrasts provide independent Wishart scatters
at the true common coefficient. Exact determinant moments give conservative
Wilks cutoffs; rational polynomial certificates enclose the continuous set.
The construction assumes stationary marginal Gaussian AR covariance and an
arbitrary positive semidefinite cross-column covariance. A singular selected
covariance supplies no shape information and conservatively retains [0,1].
Certificates concern the supplied floats, not the rounding of ideal Gaussian
measurements. This research interface returns fixed-level decisions, not p-values.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import exp, factorial, gcd, log, prod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_returns, positive_integer, probability
from .uncertainty import (
    _bernstein,
    _certify_candidate,
    _critical_squared,
    _gls_polynomials,
    _integer_column,
    _primitive,
    _split,
)


def _determinant(matrix: tuple[tuple[int, ...], ...]) -> int:
    """Fraction-free integer determinant, including singular matrices."""
    rows = [list(row) for row in matrix]
    size = len(rows)
    if not size:
        return 1
    sign, previous = 1, 1
    for column in range(size - 1):
        pivot_row = next((i for i in range(column, size) if rows[i][column]), None)
        if pivot_row is None:
            return 0
        if pivot_row != column:
            rows[column], rows[pivot_row] = rows[pivot_row], rows[column]
            sign = -sign
        pivot = rows[column][column]
        for i in range(column + 1, size):
            for j in range(column + 1, size):
                value = rows[i][j] * pivot - rows[i][column] * rows[column][j]
                quotient, remainder = divmod(value, previous)
                if remainder:
                    raise ArithmeticError("Bareiss determinant division was not exact.")
                rows[i][j] = quotient
            rows[i][column] = 0
        previous = pivot
    return sign * rows[-1][-1]


def _interpolate_integer(values: tuple[int, ...]) -> tuple[int, ...]:
    """Power coefficients from exact values at 0,1,...,degree."""
    differences = list(values)
    leading = []
    while differences:
        leading.append(differences[0])
        differences = [y - x for x, y in zip(differences[:-1], differences[1:], strict=True)]
    coefficients = [Fraction() for _ in values]
    basis = [1]
    for degree, difference in enumerate(leading):
        multiplier = Fraction(difference, factorial(degree))
        for i, value in enumerate(basis):
            coefficients[i] += multiplier * value
        next_basis = [0] * (len(basis) + 1)
        for i, value in enumerate(basis):
            next_basis[i] -= degree * value
            next_basis[i + 1] += value
        basis = next_basis
    if any(value.denominator != 1 for value in coefficients):
        raise ArithmeticError("The determinant interpolation has noninteger coefficients.")
    result = tuple(value.numerator for value in coefficients)
    while len(result) > 1 and result[-1] == 0:
        result = result[:-1]
    return result


def _determinant_polynomial(
    matrix: tuple[tuple[tuple[int, int, int], ...], ...],
) -> tuple[int, ...]:
    dimension = len(matrix)
    values = tuple(
        _determinant(tuple(tuple(a + x * (b + x * c) for a, b, c in row) for row in matrix))
        for x in range(2 * dimension + 1)
    )
    return _interpolate_integer(values)


def _scatter_polynomials(
    columns: tuple[tuple[int, ...], ...], block_length: int, n: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Common-scaled determinants of within-block and total centered scatters."""
    future = tuple(column[1 : n + 1] for column in columns)
    past = tuple(column[:n] for column in columns)
    sums = tuple((sum(f), sum(p)) for f, p in zip(future, past, strict=True))
    blocks = tuple(
        tuple(
            (sum(f[start : start + block_length]), sum(p[start : start + block_length]))
            for start in range(0, n, block_length)
        )
        for f, p in zip(future, past, strict=True)
    )
    dimension = len(columns)
    within = [[(0, 0, 0) for _ in columns] for _ in columns]
    total = [[(0, 0, 0) for _ in columns] for _ in columns]
    for i in range(dimension):
        for j in range(i, dimension):
            gram = (
                sum(x * y for x, y in zip(future[i], future[j], strict=True)),
                -sum(x * y for x, y in zip(future[i], past[j], strict=True))
                - sum(x * y for x, y in zip(past[i], future[j], strict=True)),
                sum(x * y for x, y in zip(past[i], past[j], strict=True)),
            )
            grouped = (
                sum(x[0] * y[0] for x, y in zip(blocks[i], blocks[j], strict=True)),
                -sum(x[0] * y[1] + x[1] * y[0] for x, y in zip(blocks[i], blocks[j], strict=True)),
                sum(x[1] * y[1] for x, y in zip(blocks[i], blocks[j], strict=True)),
            )
            fi, pi = sums[i]
            fj, pj = sums[j]
            centered = (fi * fj, -fi * pj - pi * fj, pi * pj)
            a = tuple(n * (block_length * g - b) for g, b in zip(gram, grouped, strict=True))
            b = tuple(block_length * (n * g - t) for g, t in zip(gram, centered, strict=True))
            within[i][j] = within[j][i] = a
            total[i][j] = total[j][i] = b
    det_within = _determinant_polynomial(tuple(tuple(row) for row in within))
    det_total = _determinant_polynomial(tuple(tuple(row) for row in total))
    divisor = gcd(*det_within, *det_total)
    if divisor:
        det_within = tuple(value // divisor for value in det_within)
        det_total = tuple(value // divisor for value in det_total)
    return det_within, det_total


def _moment(dimension: int, low_df: int, high_df: int, order: int) -> Fraction:
    """Exact E[Lambda**order]; low_df is even and order may be negative."""
    b, h = low_df // 2, abs(order)
    numerator, denominator = [], []
    for i in range(1, dimension + 1):
        a = high_df - i + 1  # twice the first Beta shape
        if order < 0 and 2 * h >= a:
            raise ValueError("The requested negative determinant moment does not exist.")
        if h <= b:
            if order >= 0:
                numerator.extend(a + 2 * u for u in range(h))
                denominator.extend(a + low_df + 2 * u for u in range(h))
            else:
                numerator.extend(a + low_df - 2 * u for u in range(1, h + 1))
                denominator.extend(a - 2 * u for u in range(1, h + 1))
        else:
            numerator.extend(a + 2 * v for v in range(b))
            denominator.extend(a + 2 * v + 2 * order for v in range(b))
    return Fraction(prod(numerator), prod(denominator))


def _root_bracket(target: Fraction, order: int, bits: int) -> tuple[Fraction, Fraction]:
    """An exact dyadic enclosure of target**(1/order), for 0<=target<=1.

    A floating estimate only starts the search. Integer power comparisons
    establish both endpoints and every subsequent refinement.
    """
    if target <= 0:
        return Fraction(), Fraction()
    if target >= 1:
        return Fraction(1), Fraction(1)
    denominator = 1 << bits
    threshold = target.numerator << (bits * order)
    estimate = exp((log(target.numerator) - log(target.denominator)) / order)
    guess = min(denominator, max(0, int(denominator * estimate)))
    step = 2
    left, right = max(0, guess - step), min(denominator, guess + step)

    def compare(value: int) -> int:
        difference = value**order * target.denominator - threshold
        return (difference > 0) - (difference < 0)

    while compare(left) > 0 or compare(right) < 0:
        step *= 2
        left, right = max(0, guess - step), min(denominator, guess + step)
    while right - left > 1:
        middle = (left + right) // 2
        if compare(middle) <= 0:
            left = middle
        else:
            right = middle
    if compare(left) == 0:
        return Fraction(left, denominator), Fraction(left, denominator)
    return Fraction(left, denominator), Fraction(right, denominator)


def _positive_orders(n: int) -> tuple[int, ...]:
    limit = 1 << (4 * n - 1).bit_length()
    values = {1, 2, 3, 4}
    power = 4
    while power <= limit:
        values.add(power)
        if 3 * power // 2 <= limit:
            values.add(3 * power // 2)
        power *= 2
    return tuple(sorted(values))


@lru_cache(maxsize=64)
def _wilks_cutoffs(
    dimension: int, low_df: int, high_df: int, beta: Fraction, bits: int
) -> tuple[tuple[Fraction, Fraction], int, int]:
    lower, upper = Fraction(), Fraction(1)
    lower_order = upper_order = 0
    for order in range(1, min(64, (high_df - dimension) // 2) + 1):
        candidate, _ = _root_bracket(
            beta / (2 * _moment(dimension, low_df, high_df, -order)), order, bits
        )
        if candidate > lower:
            lower, lower_order = candidate, order
    for order in _positive_orders(low_df + high_df + 1):
        target = 2 * _moment(dimension, low_df, high_df, order) / beta
        if target >= 1:
            continue  # Lambda<=1, so 1 is already a valid upper cutoff.
        _, candidate = _root_bracket(target, order, bits)
        if candidate < upper:
            upper, upper_order = candidate, order
    return (lower, upper), lower_order, upper_order


@dataclass(frozen=True)
class WilksScale:
    """One prespecified block contrast and its exact numerical evidence."""

    block_length: int
    n_innovations: int
    low_df: int
    high_df: int
    cutoffs: tuple[Fraction, Fraction]
    lower_moment: int
    upper_moment: int
    usable: bool
    singular_fallback: bool
    det_within: tuple[int, ...]
    det_total: tuple[int, ...]


def _constraints(scale: WilksScale) -> tuple[tuple[int, ...], ...]:
    if not scale.usable or scale.singular_fallback:
        return ()
    size = max(len(scale.det_within), len(scale.det_total))
    within = scale.det_within + (0,) * (size - len(scale.det_within))
    total = scale.det_total + (0,) * (size - len(scale.det_total))
    lower, upper = scale.cutoffs
    return (
        _primitive(
            tuple(
                a * lower.denominator - b * lower.numerator
                for a, b in zip(within, total, strict=True)
            )
        ),
        _primitive(
            tuple(
                b * upper.numerator - a * upper.denominator
                for a, b in zip(within, total, strict=True)
            )
        ),
    )


def _confidence_intervals(
    constraints: tuple[tuple[int, ...], ...], max_depth: int
) -> tuple[tuple[tuple[Fraction, Fraction], ...], int]:
    """Retain every unresolved dyadic interval; never assume connectedness."""
    stack, kept = [(0, 1, 0)], []
    unresolved = 0
    while stack:
        cell = stack.pop()
        bounds = [_bernstein(polynomial, *cell) for polynomial in constraints]
        if any(max(bound) < 0 for bound in bounds):
            continue
        if all(min(bound) >= 0 for bound in bounds):
            kept.append(cell)
        elif cell[2] >= max_depth:
            kept.append(cell)
            unresolved += 1
        else:
            left, right = _split(cell)
            stack.extend((right, left))
    merged = []
    for left, right, depth in kept:
        interval = Fraction(left, 1 << depth), Fraction(right, 1 << depth)
        if merged and merged[-1][1] == interval[0]:
            merged[-1] = merged[-1][0], interval[1]
        else:
            merged.append(interval)
    return tuple(merged), unresolved


@dataclass(frozen=True)
class WilksResult:
    """Fixed-level decisions, rational shape enclosure, and per-scale evidence."""

    decisions: NDArray[np.bool_]
    intervals: tuple[tuple[Fraction, Fraction], ...]
    alpha: float
    beta: float
    n_obs: int
    k: int
    dimension: int
    max_dimension: int
    block_lengths: tuple[int, ...]
    scales: tuple[WilksScale, ...]
    critical_squared: Fraction
    ci_unresolved_cells: int
    certificate_unresolved: NDArray[np.bool_]
    certificate_nodes: NDArray[np.int64]
    ci_depth: int
    certificate_depth: int
    max_nodes: int
    critical_bits: int

    @property
    def global_reject(self) -> bool:
        return bool(self.decisions.any())

    @property
    def interval_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((float(left), float(right)) for left, right in self.intervals)

    @property
    def empty_ci(self) -> bool:
        return not self.intervals

    @property
    def phi1_retained(self) -> bool:
        return bool(self.intervals and self.intervals[-1][1] == 1)

    @property
    def ci_width(self) -> float:
        return float(sum((right - left for left, right in self.intervals), Fraction()))

    @property
    def singular_fallback(self) -> bool:
        return any(scale.singular_fallback for scale in self.scales)


def wilks_uncertainty_test(
    data: ArrayLike,
    *,
    alpha: float = 0.05,
    beta: float = 0.005,
    max_dimension: int = 8,
    block_lengths: tuple[int, ...] = (4, 16, 64),
    ci_depth: int = 16,
    certificate_depth: int = 20,
    max_nodes: int = 4096,
    critical_bits: int = 40,
) -> WilksResult:
    """Strong-FWER decisions with a multiscale joint Gaussian AR shape set.

    The first min(K,max_dimension) columns are fixed before observing data.
    Each scale uses the largest odd block count, and its allotted miss
    probability is beta/len(block_lengths), including unusable scales. At
    the true phi independent Gaussian contrast scatters have a Wilks
    determinant ratio. Integer moments and outward dyadic roots control
    both tails without Monte Carlo critical values. Their continuous
    polynomial acceptance sets are intersected and enclosed by Bernstein
    inequalities. No independence between scales or columns is required.

    A singular selected covariance or insufficient observations can only
    remove a source of parameter information. No rank is estimated. On
    coverage, the same known-phi GLS and shared-budget argument as the
    reference F procedure bounds strong FWER by alpha. Input rounding lies
    outside the ideal Gaussian distributional theorem.
    """
    values = as_returns(data)
    alpha, beta = probability(alpha, "alpha"), probability(beta, "beta")
    if not beta < alpha < 0.5:
        raise ValueError("Require 0 < beta < alpha < 0.5 for a positive one-sided cutoff.")
    max_dimension = positive_integer(max_dimension, "max_dimension")
    if max_dimension > 8:
        raise ValueError("max_dimension must be <= 8 for the degree-16 certificate interface.")
    if isinstance(block_lengths, (str, bytes)):
        raise ValueError("block_lengths must be a nonempty sequence of distinct integers >= 2.")
    try:
        lengths = tuple(positive_integer(value, "block_length", 2) for value in block_lengths)
    except TypeError as exc:
        raise ValueError(
            "block_lengths must be a nonempty sequence of distinct integers >= 2."
        ) from exc
    if not lengths or len(lengths) != len(set(lengths)):
        raise ValueError("block_lengths must be a nonempty sequence of distinct integers >= 2.")
    ci_depth = positive_integer(ci_depth, "ci_depth", 0)
    certificate_depth = positive_integer(certificate_depth, "certificate_depth", 0)
    max_nodes = positive_integer(max_nodes, "max_nodes")
    critical_bits = positive_integer(critical_bits, "critical_bits", 8)
    if ci_depth > 64 or certificate_depth > 64 or critical_bits > 128:
        raise ValueError("Depths must be <= 64 and critical_bits must be <= 128.")
    n_obs, k = values.shape
    dimension = min(k, max_dimension)
    exact_beta = Fraction(beta)
    critical = _critical_squared(n_obs, k, Fraction(alpha) - exact_beta, critical_bits)
    columns = tuple(_integer_column(values[:, j]) for j in range(k))
    scales = []
    for length in lengths:
        blocks = (n_obs - 1) // length
        if blocks % 2 == 0:
            blocks -= 1
        blocks = max(0, blocks)
        n = blocks * length
        q, s = max(0, blocks - 1), n - blocks
        # At least one finite negative moment is needed for the two-sided cut.
        usable = blocks >= 3 and s >= dimension + 2
        if not usable:
            scales.append(
                WilksScale(
                    length, n, q, s, (Fraction(), Fraction(1)), 0, 0, False, False, (0,), (0,)
                )
            )
            continue
        cutoffs, lower_order, upper_order = _wilks_cutoffs(
            dimension, q, s, exact_beta / len(lengths), critical_bits
        )
        within, total = _scatter_polynomials(columns[:dimension], length, n)
        singular = not any(total)
        if singular and any(within):
            raise ArithmeticError("A singular total scatter has a nonzero within determinant.")
        scales.append(
            WilksScale(
                length, n, q, s, cutoffs, lower_order, upper_order, True, singular, within, total
            )
        )
    constraints = tuple(polynomial for scale in scales for polynomial in _constraints(scale))
    intervals, ci_unresolved = _confidence_intervals(constraints, ci_depth)
    decisions, unresolved, counts = [], [], []
    for column in columns:
        a, _, r = _gls_polynomials(column, critical)
        reject, uncertain, nodes = _certify_candidate(a, r, intervals, certificate_depth, max_nodes)
        decisions.append(reject)
        unresolved.append(uncertain)
        counts.append(nodes)
    decision_array = np.asarray(decisions, dtype=bool)
    unresolved_array = np.asarray(unresolved, dtype=bool)
    count_array = np.asarray(counts, dtype=np.int64)
    for array in (decision_array, unresolved_array, count_array):
        array.flags.writeable = False
    return WilksResult(
        decision_array,
        intervals,
        alpha,
        beta,
        n_obs,
        k,
        dimension,
        max_dimension,
        lengths,
        tuple(scales),
        critical,
        ci_unresolved,
        unresolved_array,
        count_array,
        ci_depth,
        certificate_depth,
        max_nodes,
        critical_bits,
    )
