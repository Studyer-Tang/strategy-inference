"""Unsupported floating-point scales must fail before reporting inference."""

import numpy as np
import pytest

from strategy_inference import audit_returns, infer_mean, stationary_mean_variance
from strategy_inference.bootstrap import stationary_bootstrap_statistics


@pytest.fixture
def underflow_returns():
    # The input sample variance is representable, while its division by T is not.
    values = 5e-162 * np.cos(2 * np.pi * np.arange(512) / 512)
    assert np.mean((values - values.mean()) ** 2) > 0
    return values


def test_hac_standard_error_underflow_raises(underflow_returns):
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="positive|finite"):
        infer_mean(underflow_returns, method="hac")


def test_resampled_hac_standard_error_underflow_raises(underflow_returns):
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="positive|finite"):
        stationary_bootstrap_statistics(underflow_returns, n_resamples=7, seed=11)


@pytest.mark.parametrize("studentization", ["fixed", "resampled"])
def test_audit_rejects_underflow_before_reporting_pvalue(underflow_returns, studentization):
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="positive|finite"):
        audit_returns(
            underflow_returns, studentization=studentization, n_resamples=7, seed=11
        )


def test_conditional_bootstrap_variance_underflow_raises(underflow_returns):
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="positive|finite"):
        stationary_mean_variance(underflow_returns)


def test_conditional_bootstrap_fft_overflow_raises():
    # Ordinary HAC remains finite; squaring the concentrated FFT does not.
    values = 1e152 * np.cos(2 * np.pi * np.arange(512) / 512)
    assert np.isfinite(infer_mean(values, method="hac").standard_error).all()
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="positive|finite"):
        stationary_mean_variance(values)
