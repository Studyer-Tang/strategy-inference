"""Certified fixed-level GLS decisions with an unknown common Gaussian AR(1).

The reference column and projection are fixed before inspecting the data.
An exact innovation F confidence set is enclosed by rational intervals. Each
rejection is certified over that enclosure by integer Bernstein bounds.
Unresolved intervals are retained; unresolved decisions do not reject.
This research interface changes the HAC statistic and assumes common phi in
[0, 1), stationary marginal Gaussian AR covariance, and arbitrary cross-column
dependence. It returns decisions at one level, not continuous p-values.
The numeric certificates concern the supplied floats; rounding a continuous
Gaussian sample does not itself preserve an exactly Gaussian distribution.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb, cos, gcd, lcm, pi

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_returns, positive_integer, probability


def _primitive(coefficients: tuple[int, ...]) -> tuple[int, ...]:
    values = list(coefficients)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    divisor = gcd(*values)
    return tuple(value // divisor for value in values) if divisor else tuple(values)


def _integer_column(values: NDArray[np.float64]) -> tuple[int, ...]:
    """Scale a binary64 column exactly by a positive power of two."""
    ratios = [float(value).as_integer_ratio() for value in values]
    exponent = max(denominator.bit_length() - 1 for _, denominator in ratios)
    integers = tuple(
        numerator << (exponent - (denominator.bit_length() - 1))
        for numerator, denominator in ratios
    )
    divisor = gcd(*integers)
    return tuple(value // divisor for value in integers) if divisor else integers


def _bernstein(coefficients: tuple[int, ...], left: int, right: int, depth: int) -> tuple[int, ...]:
    """Same-sign-scaled exact Bernstein coefficients on [left,right]/2**depth."""
    degree = len(coefficients) - 1
    denominator = 1 << depth
    width = right - left
    powers = [
        sum(
            coefficients[j] * comb(j, i) * left ** (j - i) * denominator ** (degree - j)
            for j in range(i, degree + 1)
        )
        * width**i
        for i in range(degree + 1)
    ]
    common = lcm(*(comb(degree, i) for i in range(degree + 1)))
    return tuple(
        sum(powers[i] * comb(k, i) * (common // comb(degree, i)) for i in range(k + 1))
        for k in range(degree + 1)
    )


def _split(cell: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    left, right, depth = cell
    middle = left + right
    return ((2 * left, middle, depth + 1), (middle, 2 * right, depth + 1))


def _beta_numerator(numerator: int, denominator: int, a: int, b: int) -> int:
    """I_(numerator/denominator)(a,b) times denominator**(a+b-1)."""
    degree = a + b - 1
    complement = denominator - numerator
    if numerator == 0:
        return 0
    if complement == 0:
        return denominator**degree
    # Every term is positive; the integer recurrence has no rounding error.
    term = comb(degree, a) * numerator**a * complement ** (degree - a)
    total = term
    for j in range(a, degree):
        term = term * (degree - j) * numerator // ((j + 1) * complement)
        total += term
    return total


def _beta_bracket(a: int, b: int, target: Fraction, bits: int) -> tuple[Fraction, Fraction]:
    denominator = 1 << bits
    normalizer = denominator ** (a + b - 1)
    left, right = 0, denominator
    while right - left > 1:
        middle = (left + right) // 2
        value = _beta_numerator(middle, denominator, a, b)
        if value * target.denominator <= target.numerator * normalizer:
            left = middle
        else:
            right = middle
    return Fraction(left, denominator), Fraction(right, denominator)


@lru_cache(maxsize=64)
def _f_cutoffs(low_df: int, high_df: int, beta: Fraction, bits: int) -> tuple[Fraction, Fraction]:
    low, _ = _beta_bracket(low_df // 2, high_df // 2, beta / 2, bits)
    _, high = _beta_bracket(low_df // 2, high_df // 2, 1 - beta / 2, bits)
    if high == 1:
        raise ValueError("critical_bits is insufficient to represent a finite F cutoff.")
    return high_df * low / (low_df * (1 - low)), high_df * high / (low_df * (1 - high))


@lru_cache(maxsize=32)
def _student_polynomial(m: int) -> tuple[tuple[int, ...], int]:
    common = lcm(*(2 * k + 1 for k in range(m)))
    coefficients = tuple((-1) ** k * comb(m - 1, k) * (common // (2 * k + 1)) for k in range(m))
    return coefficients, common


def _student_tail_at_u(numerator: int, denominator: int, even_df: int) -> Fraction:
    """Exact t_(even_df) survival at c=sqrt(df)*u/sqrt(1-u*u)."""
    m = even_df // 2
    coefficients, common = _student_polynomial(m)
    squared_denominator = denominator * denominator
    squared_numerator = numerator * numerator
    value = coefficients[-1]
    power = squared_denominator
    for coefficient in coefficients[-2::-1]:
        value = value * squared_numerator + coefficient * power
        power *= squared_denominator
    center = Fraction(
        m * comb(2 * m, m) * numerator * value,
        (1 << (2 * m)) * common * denominator ** (2 * m - 1),
    )
    return Fraction(1, 2) - center


@lru_cache(maxsize=64)
def _critical_squared(n_obs: int, k: int, budget: Fraction, bits: int) -> Fraction:
    even_df = 2 * ((n_obs - 1) // 2)
    target = budget / k
    denominator = 1 << bits
    left, right = 0, denominator
    while right - left > 1:
        middle = (left + right) // 2
        if _student_tail_at_u(middle, denominator, even_df) <= target:
            right = middle
        else:
            left = middle
    if right == denominator:
        raise ValueError("critical_bits is insufficient to represent a finite critical value.")
    u = Fraction(right, denominator)
    return even_df * u * u / (1 - u * u)


@dataclass(frozen=True)
class _Projection:
    n: int
    q: int
    s: int
    basis: tuple[tuple[int, ...], ...]
    inverse_numerator: tuple[tuple[int, ...], ...]
    inverse_denominator: int


def _fraction_inverse(matrix: tuple[tuple[int, ...], ...]) -> list[list[Fraction]]:
    size = len(matrix)
    rows = [
        [Fraction(value) for value in row] + [Fraction(i == j) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next((i for i in range(column, size) if rows[i][column]), None)
        if pivot is None:
            raise ValueError("The fixed rational low-frequency basis is rank deficient.")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value / scale for value in rows[column]]
        for i in range(size):
            if i != column and rows[i][column]:
                factor = rows[i][column]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[column], strict=True)]
    return [row[size:] for row in rows]


@lru_cache(maxsize=16)
def _projection(n_obs: int) -> _Projection:
    n = n_obs - 1
    if n % 2 == 0:
        n -= 1
    # Odd cubes are the midpoints between neighboring even ranks. Integer
    # comparisons also implement upward ties without cube-root rounding.
    q = 2
    while (q + 1) ** 3 <= n:
        q += 2
    q = min(n - 3, q)
    basis = []
    for frequency in range(1, q + 1):
        raw = [round((1 << 16) * cos(pi * (i + 0.5) * frequency / n)) for i in range(n)]
        total = sum(raw)
        basis.append(tuple(n * value - total for value in raw))
    fixed = tuple(basis)
    gram = tuple(
        tuple(sum(x * y for x, y in zip(row, other, strict=True)) for other in fixed)
        for row in fixed
    )
    inverse = _fraction_inverse(gram)
    denominator = lcm(*(value.denominator for row in inverse for value in row))
    numerator = tuple(
        tuple(value.numerator * (denominator // value.denominator) for value in row)
        for row in inverse
    )
    return _Projection(n, q, n - 1 - q, fixed, numerator, denominator)


def _innovation_quadratics(
    column: tuple[int, ...], projection: _Projection
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Integer coefficients of a common-scaled low/high innovation energy."""
    n = projection.n
    future, past = column[1 : n + 1], column[:n]
    a = tuple(sum(x * y for x, y in zip(row, future, strict=True)) for row in projection.basis)
    b = tuple(sum(x * y for x, y in zip(row, past, strict=True)) for row in projection.basis)

    def bilinear(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        return sum(
            left[i] * sum(value * right[j] for j, value in enumerate(row))
            for i, row in enumerate(projection.inverse_numerator)
        )

    low = (n * bilinear(a, a), -2 * n * bilinear(a, b), n * bilinear(b, b))
    sf, sp = sum(future), sum(past)
    centered = (
        n * sum(x * x for x in future) - sf * sf,
        -2 * (n * sum(x * y for x, y in zip(future, past, strict=True)) - sf * sp),
        n * sum(x * x for x in past) - sp * sp,
    )
    high = tuple(
        projection.inverse_denominator * value - subtract
        for value, subtract in zip(centered, low, strict=True)
    )
    divisor = gcd(*low, *high)
    if divisor:
        low = tuple(value // divisor for value in low)
        high = tuple(value // divisor for value in high)
    return low, high


def _confidence_intervals(
    low: tuple[int, ...],
    high: tuple[int, ...],
    projection: _Projection,
    cutoffs: tuple[Fraction, Fraction],
    max_depth: int,
) -> tuple[tuple[tuple[Fraction, Fraction], ...], int]:
    lower, upper = cutoffs
    polynomials = (
        _primitive(
            tuple(
                projection.s * x * lower.denominator - projection.q * y * lower.numerator
                for x, y in zip(low, high, strict=True)
            )
        ),
        _primitive(
            tuple(
                projection.q * y * upper.numerator - projection.s * x * upper.denominator
                for x, y in zip(low, high, strict=True)
            )
        ),
    )
    stack = [(0, 1, 0)]
    kept = []
    unresolved = 0
    while stack:
        cell = stack.pop()
        bounds = [_bernstein(polynomial, *cell) for polynomial in polynomials]
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
    # Traversal is ordered. Merge adjacent closed dyadic intervals exactly.
    merged: list[tuple[Fraction, Fraction]] = []
    for left, right, depth in kept:
        interval = Fraction(left, 1 << depth), Fraction(right, 1 << depth)
        if merged and merged[-1][1] == interval[0]:
            merged[-1] = merged[-1][0], interval[1]
        else:
            merged.append(interval)
    return tuple(merged), unresolved


def _multiply(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * (len(left) + len(right) - 1)
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            result[i + j] += x * y
    return tuple(result)


def _gls_polynomials(
    column: tuple[int, ...], critical_squared: Fraction
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """a, H and critical inequality R, using exact integer sample summaries."""
    n = len(column)
    interior_sum = sum(column[1:-1])
    interior_squares = sum(value * value for value in column[1:-1])
    a = (sum(column), -interior_sum)
    d = (n, -(n - 2))
    q = (
        column[0] ** 2 + column[-1] ** 2 + interior_squares,
        -2 * sum(x * y for x, y in zip(column[:-1], column[1:], strict=True)),
        interior_squares,
    )
    dq = _multiply(d, q)
    numerator = _multiply((1, -1), _multiply(a, a))
    h = tuple(x - y for x, y in zip(dq, numerator, strict=True))
    r = tuple(
        (n - 1) * x * critical_squared.denominator - y * critical_squared.numerator
        for x, y in zip(numerator, h, strict=True)
    )
    return a, h, _primitive(r)


def _interval_cell(interval: tuple[Fraction, Fraction]) -> tuple[int, int, int]:
    left, right = interval
    denominator = max(left.denominator, right.denominator)
    return (
        left.numerator * (denominator // left.denominator),
        right.numerator * (denominator // right.denominator),
        denominator.bit_length() - 1,
    )


def _certify_candidate(
    a: tuple[int, ...],
    r: tuple[int, ...],
    intervals: tuple[tuple[Fraction, Fraction], ...],
    max_depth: int,
    max_nodes: int,
) -> tuple[bool, bool, int]:
    if not intervals:
        return False, False, 0
    stack = [(_interval_cell(interval), 0) for interval in intervals[::-1]]
    nodes = 0
    while stack:
        cell, refinements = stack.pop()
        if nodes >= max_nodes:
            return False, True, nodes
        nodes += 1
        bounds_a = _bernstein(a, *cell)
        bounds_r = _bernstein(r, *cell)
        if min(bounds_a) > 0 and min(bounds_r) > 0:
            continue
        # An actual endpoint disproves positivity on this *outer* set. It does
        # not assert that the un-enclosed F acceptance set contains that point.
        if min(bounds_a[0], bounds_a[-1], bounds_r[0], bounds_r[-1]) <= 0:
            return False, False, nodes
        if refinements >= max_depth:
            return False, True, nodes
        left, right = _split(cell)
        stack.extend(((right, refinements + 1), (left, refinements + 1)))
    return True, False, nodes


@dataclass(frozen=True)
class UncertaintyResult:
    """Fixed-level decisions and the rational certificates that support them.

    ``intervals`` encloses the nuisance acceptance set in the closed domain
    [0,1]; retaining 1 also covers coefficients arbitrarily close to one.
    ``certificate_unresolved`` flags budget/depth exhaustion, not every
    nonrejection. A nonpositive endpoint also yields a conservative nonrejection.
    """

    decisions: NDArray[np.bool_]
    intervals: tuple[tuple[Fraction, Fraction], ...]
    alpha: float
    beta: float
    reference: int
    n_obs: int
    k: int
    n_innovations: int
    low_df: int
    high_df: int
    critical_squared: Fraction
    f_cutoffs: tuple[Fraction, Fraction]
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


def uncertainty_test(
    data: ArrayLike,
    *,
    alpha: float = 0.05,
    beta: float = 0.005,
    reference: int = 0,
    ci_depth: int = 16,
    certificate_depth: int = 20,
    max_nodes: int = 4096,
    critical_bits: int = 40,
) -> UncertaintyResult:
    """Strong-FWER decisions under the stated common-phi marginal AR model.

    At the true phi the reference F set misses with probability at most beta.
    On its coverage event each rejected true-null column has a known-phi t
    exceeding an upper critical value for tail (alpha-beta)/K. Bonferroni then
    gives FWER <= alpha with no cross-column covariance estimate or independence
    between the confidence set and the t statistics. Stationarity is needed for
    GLS; the innovation F construction itself does not use the initial density.

    All interval decisions and critical values use exact integers/rationals on
    the supplied binary64 observations. The finite budgets can only suppress
    rejections. ``certificate_depth`` counts refinements of each merged CI
    interval, separately from the nuisance enclosure's ``ci_depth``.
    """
    values = as_returns(data)
    alpha, beta = probability(alpha, "alpha"), probability(beta, "beta")
    if not beta < alpha < 0.5:
        raise ValueError("Require 0 < beta < alpha < 0.5 for a positive one-sided cutoff.")
    reference = positive_integer(reference, "reference", 0)
    if reference >= values.shape[1]:
        raise ValueError("reference must be a predeclared valid candidate index.")
    ci_depth = positive_integer(ci_depth, "ci_depth", 0)
    certificate_depth = positive_integer(certificate_depth, "certificate_depth", 0)
    max_nodes = positive_integer(max_nodes, "max_nodes")
    critical_bits = positive_integer(critical_bits, "critical_bits", 8)
    if ci_depth > 64 or certificate_depth > 64 or critical_bits > 128:
        raise ValueError("Depths must be <= 64 and critical_bits must be <= 128.")
    n_obs, k = values.shape
    projection = _projection(n_obs)
    exact_beta = Fraction(beta)
    cutoffs = _f_cutoffs(projection.q, projection.s, exact_beta, critical_bits)
    critical = _critical_squared(n_obs, k, Fraction(alpha) - exact_beta, critical_bits)
    columns = tuple(_integer_column(values[:, j]) for j in range(k))
    low, high = _innovation_quadratics(columns[reference], projection)
    intervals, ci_unresolved = _confidence_intervals(low, high, projection, cutoffs, ci_depth)
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
    return UncertaintyResult(
        decision_array,
        intervals,
        alpha,
        beta,
        reference,
        n_obs,
        k,
        projection.n,
        projection.q,
        projection.s,
        critical,
        cutoffs,
        ci_unresolved,
        unresolved_array,
        count_array,
        ci_depth,
        certificate_depth,
        max_nodes,
        critical_bits,
    )
