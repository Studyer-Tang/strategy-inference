from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import binom, norm, t

from strategy_inference import calibration
from strategy_inference.bootstrap import stationary_indices
from strategy_inference.inference import default_lags
from strategy_inference.simulations import simulate_returns


def _config(n_mc=6):
    return dict(
        n_obs=24,
        n_mc=n_mc,
        supplement_mc=n_mc,
        n_bootstrap=39,
        seed=20261004,
        sigma=0.03,
        cross_corr=0.0,
        burnin=32,
        alpha=0.05,
        confidence=0.95,
        n_assessment_cells=19,
    )


def _manual_statistics(values, lags):
    """Direct single-sample formulas, independent of the inference APIs."""
    mean = values.mean(axis=0)
    centered = values - mean
    n_obs = len(values)
    iid_se = np.sqrt(np.sum(centered**2, axis=0) / (n_obs - 1) / n_obs)
    variance = np.sum(centered**2, axis=0) / n_obs
    for lag in range(1, lags + 1):
        products = np.sum(centered[lag:] * centered[:-lag], axis=0) / n_obs
        variance += 2 * (1 - lag / (lags + 1)) * products
    return mean, iid_se, np.sqrt(variance / n_obs)


def _manual_counts(config, scenario, blocks, prefixes, deltas, study, address):
    keys = [
        (length, prefix, delta, method)
        for length in blocks
        for prefix in prefixes
        for delta in deltas
        for method in ("iid", "hac", "fixed", "resampled", "oracle")
    ]
    counts = dict.fromkeys(keys, 0)
    size = config["n_obs"]
    lags = default_lags(size)
    time = np.arange(size)
    covariance = config["sigma"] ** 2 * scenario["phi"] ** np.abs(
        time[:, None] - time[None, :]
    )
    true_variance = covariance.mean()
    mean_shift = config["sigma"] * np.sqrt(
        (1 + scenario["phi"]) / (1 - scenario["phi"]) / size
    )
    for replicate in range(config["n_mc"]):
        values = simulate_returns(
            size,
            scenario["n_strategies"],
            process=scenario["process"],
            phi=scenario["phi"],
            cross_corr=config["cross_corr"],
            sigma=config["sigma"],
            seed=calibration._stream_seed(config["seed"], study, address, replicate, 0),
            burnin=config["burnin"],
        )
        mean, iid_se, hac_se = _manual_statistics(values, lags)
        for length in blocks:
            indices = stationary_indices(
                size,
                config["n_bootstrap"],
                block_length=length,
                seed=calibration._stream_seed(config["seed"], study, address, replicate, 1),
            )
            samples = (values - mean)[indices]
            fixed = samples.mean(axis=1) / hac_se
            resampled = np.array(
                [
                    _manual_statistics(sample, lags)[0]
                    / _manual_statistics(sample, lags)[2]
                    for sample in samples
                ]
            )
            for prefix in prefixes:
                # At rho=0 the known Gaussian maximum has a closed-form quantile.
                oracle_cutoff = norm.ppf((1 - config["alpha"]) ** (1 / prefix))
                for delta in deltas:
                    shifted_mean = mean[:prefix].copy()
                    shifted_mean[0] += delta * mean_shift
                    observed = shifted_mean / hac_se[:prefix]
                    winner = np.argmax(observed)
                    fixed_p = (
                        1 + np.count_nonzero(fixed[:, :prefix].max(axis=1) >= observed[winner])
                    ) / (config["n_bootstrap"] + 1)
                    resampled_p = (
                        1
                        + np.count_nonzero(
                            resampled[:, :prefix].max(axis=1) >= observed[winner]
                        )
                    ) / (config["n_bootstrap"] + 1)
                    decisions = dict(
                        iid=shifted_mean[winner] / iid_se[winner]
                        >= t.ppf(1 - config["alpha"], size - 1),
                        hac=observed[winner] >= norm.ppf(1 - config["alpha"]),
                        fixed=fixed_p <= config["alpha"],
                        resampled=resampled_p <= config["alpha"],
                        oracle=shifted_mean.max() / np.sqrt(true_variance) >= oracle_cutoff,
                    )
                    for method, reject in decisions.items():
                        counts[length, prefix, delta, method] += int(reject)
    return counts


def test_cell_counts_match_independent_indexed_samples_and_known_gaussian_law(monkeypatch):
    config = _config()
    scenario = dict(process="gaussian_ar", phi=0.5, n_strategies=3)
    blocks, prefixes, deltas = [2.0, 6.0], [1, 3], [0, 2]
    calls = {"fixed": [], "resampled": []}
    means = calibration.stationary_bootstrap_means
    statistics = calibration.stationary_bootstrap_statistics

    def means_spy(values, **kwargs):
        calls["fixed"].append((values.copy(), kwargs.copy()))
        return means(values, **kwargs)

    def statistics_spy(values, **kwargs):
        calls["resampled"].append((values.copy(), kwargs.copy()))
        return statistics(values, **kwargs)

    monkeypatch.setattr(calibration, "stationary_bootstrap_means", means_spy)
    monkeypatch.setattr(calibration, "stationary_bootstrap_statistics", statistics_spy)
    rows, diagnostics = calibration._cell(
        config,
        scenario,
        study=12,
        address=3,
        stage="selection",
        counts_k=prefixes,
        deltas=deltas,
        blocks=blocks,
    )
    observed = {
        (row["block_length"], row["n_strategies"], row["delta"], row["method"]): row[
            "reject_count"
        ]
        for row in rows
    }
    assert observed == _manual_counts(config, scenario, blocks, prefixes, deltas, 12, 3)
    assert len(calls["fixed"]) == len(calls["resampled"]) == config["n_mc"] * len(blocks)
    for (fixed_data, fixed_kwargs), (resampled_data, resampled_kwargs) in zip(
        calls["fixed"], calls["resampled"], strict=True
    ):
        np.testing.assert_array_equal(fixed_data, resampled_data)
        assert fixed_kwargs["seed"] == resampled_kwargs["seed"]
        assert fixed_kwargs["block_length"] == resampled_kwargs["block_length"]
        assert fixed_kwargs["center"] is True
        assert resampled_kwargs["lags"] == default_lags(config["n_obs"])
    assert len(diagnostics) == len(blocks)
    for row in rows:
        assert row["rate"] == row["reject_count"] / config["n_mc"]
        assert bool(row["size_upper_simultaneous"] != "") == (
            row["method"] == "resampled" and row["delta"] == 0
        )


def test_naive_iid_uses_the_hac_winner_even_when_its_own_winner_differs(monkeypatch):
    config = _config(n_mc=1)
    scenario = dict(process="student_ar", phi=0.5, n_strategies=2)
    monkeypatch.setattr(calibration, "simulate_returns", lambda *_args, **_kwargs: np.eye(24, 2))

    def inference(_values, *, method, **_kwargs):
        # IID prefers column 0; HAC prefers column 1. Its IID p-value is not significant.
        return SimpleNamespace(
            mean=np.array([0.1, 0.2]),
            standard_error=np.array([0.025, 0.4])
            if method == "iid"
            else np.array([0.1, 0.1]),
            statistic=np.array([4.0, 0.5]) if method == "iid" else np.array([1.0, 2.0]),
        )

    monkeypatch.setattr(calibration, "infer_mean", inference)
    monkeypatch.setattr(
        calibration,
        "stationary_bootstrap_means",
        lambda _data, **kwargs: np.zeros((kwargs["n_resamples"], 2)),
    )
    monkeypatch.setattr(
        calibration,
        "stationary_bootstrap_statistics",
        lambda _data, **kwargs: np.zeros((kwargs["n_resamples"], 2)),
    )
    rows, _ = calibration._cell(config, scenario, study=12, address=0, stage="selection")
    counts = {row["method"]: row["reject_count"] for row in rows}
    assert counts == {"iid": 0, "hac": 1, "fixed": 1, "resampled": 1}


def test_extending_mc_preserves_existing_data_streams_and_rejections(monkeypatch):
    seen = []
    simulator = calibration.simulate_returns

    def record_simulation(*args, **kwargs):
        values = simulator(*args, **kwargs)
        seen.append((kwargs["seed"], values.copy()))
        return values

    monkeypatch.setattr(calibration, "simulate_returns", record_simulation)
    scenario = dict(process="gaussian_ar", phi=0.4, n_strategies=2)
    small, _ = calibration._cell(_config(3), scenario, study=14, address=2, stage="holdout")
    initial = seen.copy()
    seen.clear()
    extended, _ = calibration._cell(_config(8), scenario, study=14, address=2, stage="holdout")
    for (seed, values), (later_seed, later_values) in zip(initial, seen[:3], strict=True):
        assert seed == later_seed
        np.testing.assert_array_equal(values, later_values)
    for original, additional in zip(small, extended, strict=True):
        assert original["method"] == additional["method"]
        assert original["reject_count"] <= additional["reject_count"]
        assert additional["reject_count"] - original["reject_count"] <= 5


@pytest.mark.parametrize("count", [0, 5, 50, 999])
def test_simultaneous_upper_bound_inverts_binomial_tail_not_wilson(count):
    family_alpha, total, n_cells = 0.05, 1000, 19
    upper = calibration._upper(count, total, family_alpha=family_alpha, n_cells=n_cells)
    assert binom.cdf(count, total, upper) == pytest.approx(family_alpha / n_cells, rel=1e-10)
    _, _, wilson_upper = calibration._wilson(count, total, 0.95)
    assert upper > wilson_upper
    assert calibration._upper(total, total, family_alpha=family_alpha, n_cells=n_cells) == 1


def test_frozen_protocol_has_exactly_nineteen_distinct_null_assessment_cells():
    protocol, _, _ = calibration._protocol()
    assert protocol["seed"] == 20261004
    assert protocol["assessment"]["n_assessment_cells"] == (
        len(protocol["figure_1"]["phi"])
        + len(protocol["figure_2"]["strategy_counts"])
        + len(protocol["figure_3"]["processes"])
        + len(protocol["holdout"])
    ) == 19
    assert protocol["block_sensitivity"]["block_multipliers"] == [0.5, 1.0, 2.0, 4.0]
    assert protocol["assessment"]["method"] == "resampled"
    assert 0 in protocol["figure_3"]["delta"]
