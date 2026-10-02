import numpy as np
import pytest
from scipy import stats

from strategy_inference import infer_mean, long_run_variance


def _bartlett_variance(values, lags):
    """A direct scalar reference with autocovariances divided by T."""
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    size = len(values)
    variance = sum(value * value for value in centered) / size
    for lag in range(1, lags + 1):
        covariance = sum(centered[t] * centered[t - lag] for t in range(lag, size)) / size
        variance += 2 * (1 - lag / (lags + 1)) * covariance
    return variance


@pytest.fixture
def returns():
    rng = np.random.default_rng(711)
    innovations = rng.normal(size=(96, 3))
    values = innovations.copy()
    for t in range(1, len(values)):
        values[t] += np.array([0.6, -0.4, 0.2]) * values[t - 1]
    return values + np.array([0.05, -0.02, 0.1])


@pytest.mark.parametrize("lags", [0, 1, 7, 20])
def test_hac_matches_scalar_reference(returns, lags):
    expected = np.array(
        [_bartlett_variance(returns[:, column], lags) for column in range(returns.shape[1])]
    )
    np.testing.assert_allclose(long_run_variance(returns, lags=lags), expected)
    inference = infer_mean(returns, method="hac", lags=lags)
    np.testing.assert_allclose(inference.standard_error, np.sqrt(expected / len(returns)))


def test_iid_matches_scipy_student_test(returns):
    inference = infer_mean(returns, method="iid", confidence=0.9)
    reference = stats.ttest_1samp(returns, 0, axis=0, alternative="greater")
    standard_error = returns.std(axis=0, ddof=1) / np.sqrt(len(returns))
    critical = stats.t.ppf(0.95, len(returns) - 1)

    np.testing.assert_allclose(inference.standard_error, standard_error)
    np.testing.assert_allclose(inference.statistic, reference.statistic)
    np.testing.assert_allclose(inference.pvalue, reference.pvalue)
    np.testing.assert_allclose(inference.ci_low, returns.mean(axis=0) - critical * standard_error)
    np.testing.assert_allclose(inference.ci_high, returns.mean(axis=0) + critical * standard_error)


def test_hac_uses_normal_reference_distribution(returns):
    inference = infer_mean(returns, method="hac", lags=6, confidence=0.9)
    critical = stats.norm.ppf(0.95)
    np.testing.assert_allclose(inference.pvalue, stats.norm.sf(inference.statistic))
    np.testing.assert_allclose(
        inference.ci_low, inference.mean - critical * inference.standard_error
    )
    np.testing.assert_allclose(
        inference.ci_high, inference.mean + critical * inference.standard_error
    )


@pytest.mark.parametrize("lags", [0, 3, 12])
def test_hac_matches_optional_statsmodels_sandwich_reference(returns, lags):
    regression = pytest.importorskip("statsmodels.regression.linear_model")
    covariance = pytest.importorskip("statsmodels.stats.sandwich_covariance")
    intercept = np.ones((len(returns), 1))
    expected = []
    for column in range(returns.shape[1]):
        fit = regression.OLS(returns[:, column], intercept).fit()
        expected.append(covariance.cov_hac(fit, nlags=lags, use_correction=False)[0, 0])
    inference = infer_mean(returns, method="hac", lags=lags)
    np.testing.assert_allclose(inference.standard_error**2, expected, atol=1e-14)


def test_negative_serial_dependence_can_reduce_standard_error():
    values = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    hac = infer_mean(values, method="hac", lags=1)
    iid = infer_mean(values, method="iid")
    assert 0 < hac.standard_error[0] < iid.standard_error[0]


def test_one_dimensional_input_retains_strategy_axis(returns):
    inference = infer_mean(returns[:, 0])
    assert inference.sample_size == len(returns)
    assert inference.n_strategies == 1
    assert inference.mean.shape == (1,)
    assert inference.pvalue.shape == (1,)


@pytest.mark.parametrize("method", ["iid", "hac"])
def test_inference_is_invariant_to_positive_units_and_column_order(returns, method):
    permutation = np.array([2, 0, 1])
    scales = np.array([0.01, 10, 3])
    reference = infer_mean(returns, method=method)
    changed = infer_mean((returns * scales)[:, permutation], method=method)
    np.testing.assert_allclose(changed.statistic, reference.statistic[permutation])
    np.testing.assert_allclose(changed.pvalue, reference.pvalue[permutation])
    np.testing.assert_allclose(changed.ci_low, (reference.ci_low * scales)[permutation])


@pytest.mark.parametrize(
    "values",
    [
        np.ones(12),
        np.arange(7),
        np.arange(24).reshape(2, 3, 4),
        np.array([0, 1, 2, 3, 4, 5, 6, np.nan]),
        np.array([0, 1, 2, 3, 4, 5, 6, np.inf]),
    ],
)
def test_invalid_samples_are_rejected(values):
    with pytest.raises(ValueError):
        infer_mean(values)


def test_constant_strategy_invalidates_multistrategy_input(returns):
    with pytest.raises(ValueError):
        infer_mean(np.column_stack([returns, np.zeros(len(returns))]))


@pytest.mark.parametrize("method", ["iid", "hac"])
def test_constant_nonexact_decimal_is_not_mistaken_for_positive_variance(method):
    with pytest.raises(ValueError):
        infer_mean(np.full(67, 0.12345), method=method)


@pytest.mark.parametrize("imaginary", [0, 1])
def test_complex_returns_are_not_silently_cast_to_real(returns, imaginary):
    with pytest.raises(ValueError):
        infer_mean(returns.astype(complex) + imaginary * 1j)


@pytest.mark.parametrize("confidence", ["0.95", 0.95 + 0j, object(), [0.95]])
def test_nonnumeric_confidence_raises_value_error(returns, confidence):
    with pytest.raises(ValueError):
        infer_mean(returns, confidence=confidence)
