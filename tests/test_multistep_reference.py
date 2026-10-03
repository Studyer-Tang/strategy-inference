"""Independent rational references for the delayed-controller coverage bounds.

These checks use no package implementation. They guard the indexing and finite
constants in docs/multistep-methods.md; they do not certify binary64 arithmetic
or replace the proof. Every interior threshold permits either binary error, so
the exhaustive paths include arbitrary bounded-score sequences.
"""

from fractions import Fraction

import pytest


def _schedule(kind, length):
    if kind == "constant":
        return (Fraction(3, 5),) * length
    if kind == "decreasing":
        return tuple(Fraction(3, 5 * n) for n in range(1, length + 1))
    values = (3, 1, 4, 2, 5, 1, 3, 2, 4, 1)
    return tuple(Fraction(value, 5) for value in values[:length])


def _admissible_errors(issued):
    if issued < 0:
        return (1,)
    if issued >= 1:
        return (0,)
    return (0, 1)


def _window_mass(steps, delay):
    return max(
        sum(steps[max(0, end - delay) : end], Fraction(0)) for end in range(1, len(steps) + 1)
    )


def _abel(states, steps):
    weights = tuple(1 / eta for eta in steps)
    return (
        states[-1] * weights[-1]
        - states[0] * weights[0]
        - sum(
            (states[n] * (weights[n] - weights[n - 1]) for n in range(1, len(steps))),
            Fraction(0),
        )
    )


@pytest.mark.parametrize("delay", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("kind", ["constant", "decreasing", "nonmonotone"])
def test_all_admissible_prefixes_obey_range_and_abel_bounds(delay, kind):
    alpha, initial = Fraction(2, 5), Fraction(1, 3)
    steps = _schedule(kind, 10)
    paths = [((initial,), ())]
    for n, eta in enumerate(steps, start=1):
        prefix = steps[:n]
        mass = _window_mass(prefix, delay)
        lower, upper = -alpha * mass, 1 + (1 - alpha) * mass
        width = upper - lower
        weights = tuple(1 / step for step in prefix)
        variation = weights[0] + sum(
            (abs(weights[j] - weights[j - 1]) for j in range(1, n)), Fraction(0)
        )
        nonincreasing = all(prefix[j] <= prefix[j - 1] for j in range(1, n))
        following = []
        for states, errors in paths:
            issued = states[max(n - delay, 0)]
            for error in _admissible_errors(issued):
                updated = states[-1] + eta * (error - alpha)
                next_states, next_errors = (*states, updated), (*errors, error)
                excess = sum(next_errors, Fraction(0)) - n * alpha
                assert lower <= updated <= upper
                assert _abel(next_states, prefix) == excess
                assert abs(excess) <= width * (weights[-1] + variation) / 2
                if nonincreasing:
                    assert mass == sum(prefix[:delay], Fraction(0))
                    assert -width / eta + (upper - initial) / prefix[0] <= excess
                    assert excess <= width / eta - (initial - lower) / prefix[0]
                    assert abs(excess) / n <= width / (n * eta)
                if kind == "constant":
                    assert excess == (updated - initial) / eta
                    assert (lower - initial) / eta <= excess <= (upper - initial) / eta
                following.append((next_states, next_errors))
        paths = following


@pytest.mark.parametrize("error,initial", [(1, Fraction(99, 100)), (0, Fraction(1, 100))])
def test_delay_counts_own_feedback_and_cannot_be_reduced_by_one(error, initial):
    delay, alpha, eta = 4, Fraction(2, 5), Fraction(1, 2)
    state = initial
    # Every first-h issuance precedes all feedback and retains the initial q.
    for _ in range(delay):
        assert error in _admissible_errors(initial)
        state += eta * (error - alpha)
    lower = -delay * alpha * eta
    upper = 1 + delay * (1 - alpha) * eta
    assert lower <= state <= upper
    if error:
        assert state > 1 + (delay - 1) * (1 - alpha) * eta
    else:
        assert state < -(delay - 1) * alpha * eta


def test_general_reciprocal_variation_constant_is_attainable_for_bounded_states():
    steps = tuple(1 / Fraction(w) for w in (5, 3, 7, 2))
    lower, upper = Fraction(-1), Fraction(1)
    states = (lower, upper, lower, upper, upper)
    variation = Fraction(5 + 2 + 4 + 5)
    assert _abel(states, steps) == (upper - lower) * (1 / steps[-1] + variation) / 2


@pytest.mark.parametrize("delay", [1, 2, 6, 12])
@pytest.mark.parametrize("kind", ["constant", "decreasing"])
def test_interlaced_bound_uses_each_nonempty_lane_feedback_clock(delay, kind):
    alpha, initial = Fraction(2, 5), Fraction(1, 3)
    states = [initial] * delay
    counts, errors = [0] * delay, [0] * delay
    steps = _schedule(kind, 30)
    for n in range(1, 31):
        lane = (n - 1) % delay
        score = Fraction((7 * n) % 17, 17)
        q = states[lane]
        error = int(q < 0 or (q < 1 and score > q))
        counts[lane] += 1
        errors[lane] += error
        eta = steps[counts[lane] - 1]
        states[lane] += eta * (error - alpha)
        excess = sum(errors) - n * alpha
        lane_bound = sum(
            ((1 + steps[0]) / steps[count - 1] for count in counts if count),
            Fraction(0),
        )
        assert abs(excess) <= lane_bound
        assert sum(counts) == n
        assert max(counts) - min(counts) <= 1
        for state in states:
            assert -alpha * steps[0] <= state <= 1 + (1 - alpha) * steps[0]


def test_observed_float_ledger_needs_actual_state_range_and_rounding_residual():
    alpha = 0.1
    states = [0.5]
    steps, errors = [], []
    for n in range(1, 41):
        eta = 0.3 / n
        error = int(n % 3 == 0)
        states.append(states[-1] + eta * (error - alpha))
        steps.append(Fraction(eta))
        errors.append(error)
    rational_states = tuple(Fraction(q) for q in states)
    excess = sum(errors) - len(errors) * Fraction(alpha)
    residuals = tuple(
        rational_states[n + 1] - rational_states[n] - steps[n] * (error - Fraction(alpha))
        for n, error in enumerate(errors)
    )
    assert any(residuals)
    correction = sum(
        (residual / eta for residual, eta in zip(residuals, steps, strict=True)), Fraction(0)
    )
    assert excess == _abel(rational_states, steps) - correction
    span = max(rational_states) - min(rational_states)
    assert abs(excess) <= span / steps[-1] + sum(
        (abs(residual) / eta for residual, eta in zip(residuals, steps, strict=True)), Fraction(0)
    )


@pytest.mark.parametrize("strategy", ["pooled", "interlaced"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("vector_rate", [False, True])
def test_core_matches_independent_frozen_interval_rational_reference(strategy, sparse, vector_rate):
    from strategy_inference.multistep import MultiStepConformal

    leads = (1, 3, 4)
    alpha, initial = Fraction(1, 4), Fraction(1, 2)
    rates = (
        (Fraction(1, 8), Fraction(1, 16), Fraction(1, 32)) if vector_rate else (Fraction(1, 8),) * 3
    )
    scales = (Fraction(1), Fraction(2), Fraction(4))
    tracker = MultiStepConformal(
        leads,
        alpha=float(alpha),
        step_size=[float(value) for value in rates] if vector_rate else float(rates[0]),
        decay=0,
        scale=[float(value) for value in scales],
        strategy=strategy,
    )
    labels = [(7 * time) % 11 - 5 for time in range(47)]
    states, counts, lane_errors, pending = {}, {}, {}, {}
    evaluated, misses = [0] * 3, [0] * 3
    for time, actual in enumerate(labels):
        due = sorted(pending.pop(time, ()), key=lambda record: record[1])
        updates = tracker.observe(time, actual)
        assert len(updates) == len(due)
        for update, (origin, lead, point, issued, scale) in zip(updates, due, strict=True):
            i = leads.index(lead)
            key = (i, origin % lead if strategy == "interlaced" else 0)
            previous = states.get(key, initial)
            residual = abs(Fraction(actual) - point)
            error = int(issued < 0 or (issued < 1 and residual > scale * issued / (1 - issued)))
            counts[key] = counts.get(key, 0) + 1
            lane_errors[key] = lane_errors.get(key, 0) + error
            states[key] = previous + rates[i] * (error - alpha)
            evaluated[i] += 1
            misses[i] += error
            assert (update.origin, update.target, update.lead_time) == (origin, time, lead)
            assert update.issued_quantile == float(issued)
            assert update.previous_quantile == float(previous)
            assert update.quantile == float(states[key])
            assert update.scale == float(scale)
            assert update.step_size == float(rates[i])
            assert update.miss == bool(error)
            assert update.score == pytest.approx(float(residual / (scale + residual)))
            assert tracker.lane_state(lead, key[1]) == dict(
                quantile=float(states[key]), n_updates=counts[key], misses=lane_errors[key]
            )
        if not sparse or time % 5 not in (1, 2):
            for i, interval in enumerate(tracker.predict([actual] * len(leads))):
                lead = leads[i]
                key = (i, time % lead if strategy == "interlaced" else 0)
                issued = states.get(key, initial)
                assert interval.quantile == float(issued)
                assert interval.scale == float(scales[i])
                assert (interval.origin, interval.target) == (time, time + lead)
                kind = "empty" if issued < 0 else "unbounded" if issued >= 1 else "finite"
                assert interval.kind == kind
                if kind == "finite":
                    radius = scales[i] * issued / (1 - issued)
                    assert interval.lower == pytest.approx(float(actual - radius))
                    assert interval.upper == pytest.approx(float(actual + radius))
                pending.setdefault(time + lead, []).append(
                    (time, lead, Fraction(actual), issued, scales[i])
                )
        assert tracker.n_updates == tuple(evaluated)
        for i, n in enumerate(evaluated):
            if not n:
                assert tracker.coverage[i] is tracker.coverage_bound[i] is None
                continue
            assert tracker.coverage[i] == 1 - misses[i] / n
            if strategy == "pooled":
                bound = (1 + min(leads[i], n) * rates[i]) / (n * rates[i])
            else:
                active = sum(index == i for index, _ in counts)
                bound = active * (1 + rates[i]) / (n * rates[i])
            assert tracker.coverage_bound[i] == pytest.approx(float(min(1, bound)))
    assert len(tracker.pending) == sum(map(len, pending.values()))
    assert all(interval.target >= len(labels) for interval in tracker.pending)
