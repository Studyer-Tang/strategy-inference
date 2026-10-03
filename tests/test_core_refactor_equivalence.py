"""Guard validated kernels, scalar summary arithmetic and lane allocation."""

from math import fsum, isfinite

import numpy as np
import pytest

import strategy_inference.audit as audit
import strategy_inference.bootstrap as bootstrap
import strategy_inference.inference as inference
import strategy_inference.multistep as multistep
from strategy_inference._validation import as_returns
from strategy_inference.evaluation import interval_score


@pytest.mark.parametrize("studentization", ["fixed", "resampled"])
def test_audit_validates_input_once_and_retains_public_calculations(monkeypatch, studentization):
    values = np.random.default_rng(219).normal(size=(71, 5)) + 0.02
    iid = inference.infer_mean(values, method="iid")
    hac = inference.infer_mean(values, lags=7)
    options = dict(n_resamples=39, block_length=6, seed=217, batch_size=7)
    if studentization == "fixed":
        draws = bootstrap.stationary_bootstrap_means(values, center=True, **options)
        draws /= hac.standard_error
    else:
        draws = bootstrap.stationary_bootstrap_statistics(values, lags=7, **options)
    calls = []

    def validate(data):
        calls.append(data.shape)
        return as_returns(data)

    for module in (audit, bootstrap, inference):
        monkeypatch.setattr(module, "as_returns", validate)
    result = audit.audit_returns(values, lags=7, studentization=studentization, **options)
    assert calls == [values.shape]
    np.testing.assert_array_equal(result.mean, hac.mean)
    np.testing.assert_array_equal(result.iid_pvalue, iid.pvalue)
    np.testing.assert_array_equal(result.hac_pvalue, hac.pvalue)
    np.testing.assert_array_equal(result.bootstrap_statistics, draws)
    expected = (1 + (draws.max(axis=1)[:, None] >= hac.statistic).sum(axis=0)) / 40
    np.testing.assert_array_equal(result.adjusted_pvalue, expected)


@pytest.mark.parametrize(
    "function,options",
    [
        (inference.infer_mean, {}),
        (bootstrap.stationary_bootstrap_means, {"n_resamples": 9}),
        (bootstrap.stationary_bootstrap_statistics, {"n_resamples": 9}),
    ],
)
def test_public_entry_points_still_validate_before_internal_kernels(function, options):
    with pytest.raises(ValueError, match="finite"):
        function(np.array([1, 2, 3, 4, 5, 6, 7, np.nan]), **options)
    with pytest.raises(ValueError, match="sample variance"):
        function(np.ones((8, 2)), **options)


def _legacy_summary(result, alpha):
    """The v0.8 scalar formula, independent of the shared NumPy kernel."""
    rows = []
    for h, lead in enumerate(result.lead_times):
        mask = result.evaluated[:, h]
        n = int(mask.sum())
        finite = mask & ~result.empty[:, h] & ~result.unbounded[:, h]
        widths = [
            float(result.upper[f, h]) - float(result.lower[f, h]) for f in np.flatnonzero(finite)
        ]
        width_ok = bool(widths) and all(isfinite(value) for value in widths)
        empty = int(np.count_nonzero(mask & result.empty[:, h]))
        full = int(np.count_nonzero(mask & result.unbounded[:, h]))
        status, score = "no_evaluated_intervals", None
        if n and (empty or full):
            status = "empty_or_unbounded"
        elif n:
            scores = [
                float(result.upper[f, h])
                - float(result.lower[f, h])
                + 2 * (max(float(result.lower[f, h]) - float(result.actual[f, h]), 0) / alpha)
                + 2 * (max(float(result.actual[f, h]) - float(result.upper[f, h]), 0) / alpha)
                for f in np.flatnonzero(mask)
            ]
            status = "finite" if all(isfinite(value) for value in scores) else "overflow"
            if status == "finite":
                maximum = max(scores)
                score = (
                    (fsum(value / maximum for value in scores) / len(scores)) * maximum
                    if maximum
                    else 0.0
                )
        maximum = max(widths) if width_ok else 0
        mean_width = (
            (fsum(value / maximum for value in widths) / len(widths)) * maximum if maximum else 0.0
        )
        rows.append(
            dict(
                lead_time=lead,
                n_issued=result.n_origins,
                n_evaluated=n,
                n_pending=result.n_origins - n,
                coverage=1 - int(np.count_nonzero(mask & result.misses[:, h])) / n if n else None,
                coverage_bound=result.coverage_bound[h],
                empty_rate=empty / n if n else None,
                unbounded_rate=full / n if n else None,
                finite_width_count=len(widths),
                mean_finite_width=mean_width if width_ok else None,
                finite_width_status="finite"
                if width_ok
                else "overflow"
                if widths
                else "no_finite_evaluations",
                mean_interval_score=score,
                interval_score_status=status,
            )
        )
    return rows


@pytest.mark.parametrize("alpha", [0.1, 1e-300])
def test_vector_summary_matches_scalar_v08_formula_at_numeric_and_pending_edges(alpha):
    largest = np.finfo(float).max
    actual = np.array([[0, 0, 0, 0], [2, 0, 0, 0], [np.nan, np.nan, np.nan, np.nan]])
    lower = np.array(
        [[0, -largest, -largest / 2, np.inf], [0, -largest, -largest / 2, -np.inf], [0, 0, 0, 0]]
    )
    upper = np.array(
        [[0, largest, largest / 2, -np.inf], [1, largest, largest / 2, np.inf], [0, 0, 0, 0]]
    )
    evaluated = np.ones((3, 4), dtype=bool)
    evaluated[-1] = False
    empty = np.zeros((3, 4), dtype=bool)
    full = empty.copy()
    empty[0, 3], full[1, 3] = True, True
    misses = empty | ((actual < lower) | (actual > upper))
    result = multistep.MultiStepResult(
        np.arange(3),
        np.arange(3)[:, None] + np.array([1, 6, 12, 24]),
        (1, 6, 12, 24),
        actual,
        np.zeros((3, 4)),
        lower,
        upper,
        np.full((3, 4), 0.5),
        np.ones((3, 4)),
        evaluated,
        empty,
        full,
        misses,
        np.full((3, 4), 0.01),
        (),
        {"alpha": alpha, "last_time": 2, "summary": [{"coverage_bound": 1.0}] * 4},
    )
    assert result.summary() == _legacy_summary(result, alpha)
    # The large finite mean remains representable; nonfinite groups are retained.
    assert result.summary()[2]["mean_interval_score"] == largest
    assert result.summary()[1]["interval_score_status"] == "overflow"
    assert result.summary()[3]["interval_score_status"] == "empty_or_unbounded"
    import json

    json.dumps(result.to_dict(), allow_nan=False)


def test_multistep_summary_scores_match_public_proper_score_exactly():
    rng = np.random.default_rng(41)
    actual = rng.normal(size=100)
    result = multistep.multistep_intervals(
        actual,
        np.zeros((76, 2)),
        origins=np.arange(76),
        lead_times=[1, 24],
        step_size=0.001,
        initial_quantile=0.8,
    )
    assert result.summary() == _legacy_summary(result, 0.1)
    for h, row in enumerate(result.summary()):
        scores = interval_score(result.actual[:, h], result.lower[:, h], result.upper[:, h])
        maximum = max(scores.tolist())
        expected = (fsum(value / maximum for value in scores.tolist()) / len(scores)) * maximum
        assert row["mean_interval_score"] == expected


def test_existing_lanes_do_not_allocate_discarded_default_states(monkeypatch):
    constructions = []
    original = multistep._Lane

    def lane(*args, **kwargs):
        constructions.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(multistep, "_Lane", lane)
    tracker = multistep.MultiStepConformal([1, 3])
    tracker.observe(0, 0)
    tracker.predict([0, 0])
    constructions.clear()
    tracker.observe(1, 0)
    assert len(constructions) == 1  # The newly updated lane, not an unused default.
    before = tracker.lane_state(1)
    interval = tracker.predict([0, 0])[0]
    assert interval.quantile == before["quantile"]
    assert len(constructions) == 1
