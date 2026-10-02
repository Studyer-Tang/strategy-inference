import json

import numpy as np
import pytest
from scipy import stats

from strategy_inference import audit_returns, infer_mean, stationary_bootstrap_means
from strategy_inference.bootstrap import stationary_bootstrap_statistics


@pytest.fixture
def returns():
    rng = np.random.default_rng(591)
    values = rng.normal(size=(128, 5))
    for t in range(1, len(values)):
        values[t] += 0.45 * values[t - 1]
    return values + np.array([0.08, 0, -0.15, 0.2, 0.04])


def test_joint_bootstrap_uses_centered_means_and_original_hac_scale(returns):
    audit = audit_returns(returns, n_resamples=173, block_length=8, lags=5, seed=54)
    means = stationary_bootstrap_means(
        returns, n_resamples=173, block_length=8, center=True, seed=54
    )
    hac = infer_mean(returns, method="hac", lags=5)
    np.testing.assert_allclose(audit.bootstrap_statistics, means / hac.standard_error)
    np.testing.assert_allclose(audit.statistic, hac.statistic)


def test_single_candidate_needs_no_selection_adjustment(returns):
    audit = audit_returns(returns[:, 0], n_resamples=173, seed=94)
    np.testing.assert_array_equal(audit.adjusted_pvalue, audit.marginal_bootstrap_pvalue)
    assert audit.global_pvalue == audit.adjusted_pvalue[0]


def test_selection_uses_statistic_rather_than_largest_mean():
    time = np.arange(128)
    values = np.column_stack([10 + 100 * np.sin(time), 1 + 0.2 * np.cos(time)])
    audit = audit_returns(values, n_resamples=73, names=["volatile", "stable"])
    assert audit.mean[0] > audit.mean[1]
    assert audit.selected_index == 1
    assert audit.selected_name == "stable"
    assert audit.global_pvalue == audit.adjusted_pvalue[1]


def test_family_adjustment_dominates_marginal_pvalues(returns):
    audit = audit_returns(returns, n_resamples=191, seed=77)
    assert np.all(audit.adjusted_pvalue >= audit.marginal_bootstrap_pvalue)
    assert audit.global_pvalue == audit.adjusted_pvalue[audit.selected_index]
    assert np.all(audit.adjusted_pvalue >= 1 / 192)
    assert np.all(audit.adjusted_pvalue <= 1)


def test_duplicate_candidates_do_not_inflate_effective_family(returns):
    kwargs = dict(n_resamples=179, block_length=7, seed=52)
    single = audit_returns(returns[:, 0], **kwargs)
    duplicates = audit_returns(np.repeat(returns[:, :1], 4, axis=1), **kwargs)
    np.testing.assert_allclose(duplicates.adjusted_pvalue, single.adjusted_pvalue[0])
    assert duplicates.global_pvalue == single.global_pvalue
    assert duplicates.max_abs_cutoff == pytest.approx(single.max_abs_cutoff)


def test_ties_are_counted_and_plus_one_prevents_zero(monkeypatch):
    values = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=float)
    scale = infer_mean(values, method="hac", lags=1).standard_error[0]
    fixed_means = np.array([[0], [-scale], [scale], [0]])
    monkeypatch.setattr(
        "strategy_inference.audit.stationary_bootstrap_means",
        lambda *_args, **_kwargs: fixed_means,
    )
    audit = audit_returns(values, n_resamples=4, lags=1)
    assert audit.marginal_bootstrap_pvalue[0] == 4 / 5
    assert audit.global_pvalue == 4 / 5

    all_below = audit_returns(values + 10, n_resamples=4, lags=1)
    assert all_below.global_pvalue == 1 / 5


def test_simultaneous_intervals_use_absolute_family_maximum(returns):
    audit = audit_returns(returns, n_resamples=179, alpha=0.1, seed=72)
    centered = audit.bootstrap_statistics
    lower = (audit.mean - audit.simultaneous_ci_low) / audit.standard_error
    upper = (audit.simultaneous_ci_high - audit.mean) / audit.standard_error
    np.testing.assert_allclose(lower, audit.max_abs_cutoff)
    np.testing.assert_allclose(upper, audit.max_abs_cutoff)
    assert np.mean(np.all(np.abs(centered) <= audit.max_abs_cutoff, axis=1)) >= 0.9
    assert np.mean(np.all(centered <= audit.max_cutoff, axis=1)) >= 0.9
    np.testing.assert_allclose(
        audit.one_sided_lower, audit.mean - audit.max_cutoff * audit.standard_error
    )
    assert audit.max_abs_cutoff >= audit.max_cutoff


def test_positive_units_and_column_permutation_preserve_family_inference(returns):
    kwargs = dict(n_resamples=179, block_length=7, lags=4, seed=52)
    audit = audit_returns(returns, **kwargs)
    scales = np.array([10, 0.001, 0.5, 7, 2])
    order = np.array([3, 1, 4, 0, 2])
    changed = audit_returns((returns * scales)[:, order], **kwargs)
    np.testing.assert_allclose(changed.statistic, audit.statistic[order])
    np.testing.assert_allclose(changed.bootstrap_statistics, audit.bootstrap_statistics[:, order])
    np.testing.assert_array_equal(changed.adjusted_pvalue, audit.adjusted_pvalue[order])
    assert changed.global_pvalue == audit.global_pvalue
    assert order[changed.selected_index] == audit.selected_index
    np.testing.assert_allclose(
        changed.simultaneous_ci_low, (audit.simultaneous_ci_low * scales)[order]
    )


def test_centered_reference_distribution_is_location_invariant(returns):
    kwargs = dict(n_resamples=179, block_length=7, lags=4, seed=52)
    audit = audit_returns(returns, **kwargs)
    shifted = audit_returns(returns + np.array([0.5, -0.2, 1, -0.1, 0.3]), **kwargs)
    np.testing.assert_allclose(shifted.bootstrap_statistics, audit.bootstrap_statistics, atol=1e-13)


def test_record_is_json_serializable_and_states_search_scope(returns):
    audit = audit_returns(returns, n_resamples=99, seed=74, search_complete=None)
    record = audit.to_dict()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert "bootstrap_statistics" not in record
    assert record["search_complete"] is None
    assert any("supplied columns only" in warning for warning in record["warnings"])
    confirmed = audit_returns(returns, n_resamples=99, seed=74, search_complete=True)
    assert not any("supplied columns only" in warning for warning in confirmed.warnings)


@pytest.mark.parametrize("names", ["abcde", ["x"] * 5, ["x", "y"], ["", "b", "c", "d", "e"]])
def test_ambiguous_candidate_names_are_rejected(returns, names):
    with pytest.raises(ValueError):
        audit_returns(returns, n_resamples=19, names=names)


def _audit_with_tail_count(monkeypatch, count, total=99, alpha=0.05):
    values = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=float)
    scale = infer_mean(values, method="hac", lags=1).standard_error[0]
    statistics = np.concatenate([np.ones(count), -np.ones(total - count)])
    monkeypatch.setattr(
        "strategy_inference.audit.stationary_bootstrap_means",
        lambda *_args, **_kwargs: statistics[:, None] * scale,
    )
    return audit_returns(values, n_resamples=total, lags=1, alpha=alpha)


@pytest.mark.parametrize(
    "count,total", [(0, 99), (1, 99), (5, 99), (50, 99), (99, 99), (0, 999), (999, 999)]
)
def test_conditional_tail_interval_inverts_binomial_score_equation(monkeypatch, count, total):
    audit = _audit_with_tail_count(monkeypatch, count, total)
    z_squared = stats.norm.ppf(0.975) ** 2
    # Roots of (B+z²) q² - (2k+z²) q + k²/B = 0.
    discriminant = z_squared * (z_squared + 4 * count * (1 - count / total))
    expected = np.array(
        [
            (2 * count + z_squared - np.sqrt(discriminant)) / (2 * (total + z_squared)),
            (2 * count + z_squared + np.sqrt(discriminant)) / (2 * (total + z_squared)),
        ]
    )
    low, high = audit.bootstrap_tail_interval
    np.testing.assert_allclose([low, high], expected, atol=1e-14)
    assert 0 <= low <= high <= 1
    assert audit.global_pvalue == (count + 1) / (total + 1)
    if count == 0:
        assert low == 0
    if count == total:
        assert high == 1


def test_tail_interval_is_conditional_simulation_uncertainty_not_mean_interval(monkeypatch):
    audit = _audit_with_tail_count(monkeypatch, 5, alpha=0.05)
    other_level = _audit_with_tail_count(monkeypatch, 5, alpha=0.1)
    assert audit.bootstrap_tail_interval == other_level.bootstrap_tail_interval
    record = audit.to_dict()
    assert record["conditional_bootstrap_tail_interval_95"] == list(audit.bootstrap_tail_interval)
    assert record["candidates"][0]["simultaneous_ci"] != list(audit.bootstrap_tail_interval)


@pytest.mark.parametrize("count, straddles", [(0, False), (2, True), (99, False)])
def test_boundary_warning_depends_on_conditional_tail_interval(monkeypatch, count, straddles):
    audit = _audit_with_tail_count(monkeypatch, count)
    assert any("straddles alpha" in warning for warning in audit.warnings) is straddles


@pytest.mark.parametrize("names", [[["a"], ["b"], ["c"], ["d"], ["e"]], 7])
def test_nonstring_or_noniterable_names_raise_value_error(returns, names):
    with pytest.raises(ValueError):
        audit_returns(returns, n_resamples=19, names=names)


def test_resampled_audit_uses_draw_scales_and_original_observed_hac_statistic(returns):
    kwargs = dict(n_resamples=103, block_length=7, lags=5, seed=451)
    audit = audit_returns(returns, studentization="resampled", **kwargs)
    expected = stationary_bootstrap_statistics(returns, **kwargs)
    observed = infer_mean(returns, method="hac", lags=5)
    np.testing.assert_allclose(audit.bootstrap_statistics, expected)
    np.testing.assert_allclose(audit.statistic, observed.statistic)
    expected_p = (1 + np.count_nonzero(expected.max(axis=1) >= observed.statistic.max())) / 104
    assert audit.global_pvalue == expected_p
    assert audit.studentization == "resampled"
    assert audit.to_dict()["studentization"] == "resampled"


def test_resampled_audit_family_intervals_use_studentized_maximum(returns):
    audit = audit_returns(returns, n_resamples=103, studentization="resampled", seed=711)
    absolute = np.abs(audit.bootstrap_statistics).max(axis=1)
    assert np.mean(absolute <= audit.max_abs_cutoff) >= 1 - audit.alpha
    np.testing.assert_allclose(
        audit.simultaneous_ci_low, audit.mean - audit.max_abs_cutoff * audit.standard_error
    )
    np.testing.assert_allclose(
        audit.simultaneous_ci_high, audit.mean + audit.max_abs_cutoff * audit.standard_error
    )
    assert np.all(audit.adjusted_pvalue >= audit.marginal_bootstrap_pvalue)


def test_resampled_audit_preserves_units_column_order_and_batch_partition(returns):
    kwargs = dict(n_resamples=103, block_length=7, lags=5, seed=451, studentization="resampled")
    audit = audit_returns(returns, batch_size=103, **kwargs)
    order = np.array([3, 1, 4, 0, 2])
    scales = np.array([0.01, 7, 2, 100, 0.5])
    changed = audit_returns((returns * scales)[:, order], batch_size=7, **kwargs)
    np.testing.assert_allclose(
        changed.bootstrap_statistics, audit.bootstrap_statistics[:, order], atol=1e-12
    )
    np.testing.assert_allclose(changed.statistic, audit.statistic[order])
    np.testing.assert_array_equal(changed.adjusted_pvalue, audit.adjusted_pvalue[order])
    assert changed.global_pvalue == audit.global_pvalue
    np.testing.assert_allclose(
        changed.simultaneous_ci_low, (audit.simultaneous_ci_low * scales)[order]
    )


@pytest.mark.parametrize(
    "studentization", ["invalid", None, 1, ["resampled"], np.array(["resampled"])]
)
def test_invalid_studentization_is_rejected(returns, studentization):
    with pytest.raises(ValueError):
        audit_returns(returns, n_resamples=19, studentization=studentization)
