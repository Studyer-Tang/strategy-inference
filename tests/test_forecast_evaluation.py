"""Independent loss formulas and fixed-lead bootstrap comparisons."""

import builtins
import json
import warnings
from dataclasses import replace

import numpy as np
import pytest

from strategy_inference.bootstrap import default_block_length
from strategy_inference.evaluation import (
    compare_forecasts,
    evaluate_forecasts,
    forecast_loss,
    interval_score,
    select_forecaster,
)
from strategy_inference.inference import default_lags
from strategy_inference.model_selection import backtest, drift_forecast, naive_forecast
from strategy_inference.testing import test_returns as run_test


def _scalar_loss(actual, predicted, loss, quantile=None):
    error = float(actual) - float(predicted)
    if loss == "squared":
        return error * error
    if loss == "absolute":
        return abs(error)
    return quantile * error if error >= 0 else (quantile - 1) * error


@pytest.fixture
def panel():
    innovations = np.random.default_rng(2341).normal(size=180)
    y = innovations.copy()
    for i in range(1, len(y)):
        y[i] += 0.35 * y[i - 1]

    def historical_mean(train, leads):
        return np.full(len(leads), train.mean())

    return backtest(
        y,
        {"mean": historical_mean, "baseline": naive_forecast, "drift": drift_forecast},
        initial_train_size=30,
        horizon=4,
        gap=3,
        step=2,
        window=25,
    )


@pytest.mark.parametrize(
    "loss,quantile", [("squared", None), ("absolute", None), ("pinball", 0.25)]
)
def test_losses_match_independent_scalar_calculation_with_model_axis(loss, quantile):
    actual = np.array([[1.0, 3.0], [-2.0, 4.0]])
    predicted = np.array([[[0.0, 2.0], [5.0, 1.0]], [[-1.0, -4.0], [4.0, 0.0]]])
    expected = np.empty((2, 2, 2))
    for f in range(2):
        for h in range(2):
            for m in range(2):
                expected[f, h, m] = _scalar_loss(actual[f, h], predicted[f, h, m], loss, quantile)
    actual_loss = forecast_loss(actual, predicted, loss=loss, quantile=quantile)
    np.testing.assert_array_equal(actual_loss, expected)
    np.testing.assert_array_equal(
        forecast_loss(actual, predicted[..., 0], loss=loss, quantile=quantile),
        expected[..., 0],
    )


def test_pinball_is_asymmetric_and_median_is_half_absolute_loss():
    np.testing.assert_array_equal(
        forecast_loss([2.0, -2.0], [0.0, 0.0], loss="pinball", quantile=0.25), [0.5, 1.5]
    )
    np.testing.assert_array_equal(
        forecast_loss([2.0, -2.0], [0.0, 0.0], loss="pinball", quantile=0.5), [1.0, 1.0]
    )


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([1, 2], [1]),
        ([1, 2], [[1, 2]]),
        ([[1, 2], [3, 4]], [1, 2]),
        ([[1, 2, 3], [4, 5, 6]], np.zeros((3, 2))),
    ],
)
def test_loss_shapes_do_not_use_general_numpy_broadcasting(actual, predicted):
    with pytest.raises(ValueError, match="match actual"):
        forecast_loss(actual, predicted)


@pytest.mark.parametrize(
    "values", [[], [np.nan], [np.inf], [1 + 1j], [True], ["1"], np.ma.array([1.0], mask=[True])]
)
@pytest.mark.parametrize("side", ["actual", "predicted"])
def test_loss_inputs_reject_missing_nonfinite_and_nonreal_values(values, side):
    operands = {"actual": [0.0], "predicted": [0.0]}
    operands[side] = values
    with pytest.raises(ValueError):
        forecast_loss(**operands)


@pytest.mark.parametrize("quantile", [None, 0, 1, -0.1, True, np.bool_(True), np.nan, "0.5"])
def test_pinball_requires_a_strict_probability(quantile):
    with pytest.raises(ValueError, match="quantile"):
        forecast_loss([1.0], [0.0], loss="pinball", quantile=quantile)


@pytest.mark.parametrize("loss", ["squared", "absolute"])
def test_quantile_is_not_silently_used_by_other_losses(loss):
    with pytest.raises(ValueError, match="only used"):
        forecast_loss([1.0], [0.0], loss=loss, quantile=0.5)


def test_loss_overflow_is_an_explicit_error():
    with pytest.raises(ValueError, match="finite float range"):
        forecast_loss([1e308], [-1e308], loss="absolute")
    with pytest.raises(ValueError, match="finite float range"):
        forecast_loss([1e200], [0.0], loss="squared")
    with pytest.raises(ValueError, match="loss must"):
        forecast_loss([1.0], [0.0], loss="unknown")


def test_central_interval_score_matches_width_and_both_tail_penalties():
    np.testing.assert_array_equal(
        interval_score([-1, 0, 1, 2, 3], [0] * 5, [2] * 5, alpha=0.5), [6, 2, 2, 2, 6]
    )
    actual = [0.0, 3.0]
    lower = np.array([[-1.0, 1.0], [0.0, 1.0]])
    upper = np.array([[1.0, 2.0], [2.0, 4.0]])
    np.testing.assert_array_equal(
        interval_score(actual, lower, upper, alpha=0.25), [[2.0, 9.0], [10.0, 3.0]]
    )


def test_outward_infinite_intervals_score_infinity_without_nan():
    np.testing.assert_array_equal(
        interval_score([0.0, 0.0, 0.0], [-np.inf, -np.inf, -1.0], [np.inf, 1.0, np.inf]),
        [np.inf] * 3,
    )


def test_tiny_alpha_does_not_turn_zero_miss_distances_into_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        score = interval_score(
            [0.0, 1.0, 3.0], [0.0, 0.0, 0.0], [0.0, 2.0, 2.0], alpha=np.nextafter(0.0, 1.0)
        )
    np.testing.assert_array_equal(score, [0.0, 2.0, np.inf])


@pytest.mark.parametrize(
    "lower,upper",
    [
        ([2], [1]),
        ([0], [np.nan]),
        ([np.nan], [1]),
        ([np.inf], [np.inf]),
        ([-np.inf], [-np.inf]),
        ([0], [1, 2]),
        ([[0], [0]], [[1], [1]]),
        (["0"], ["1"]),
        ([True], [False]),
        (np.ma.array([0.0], mask=[True]), [1.0]),
        ([0.0], np.ma.array([1.0], mask=[True])),
    ],
)
def test_interval_bounds_reject_missing_misordered_or_misaligned_values(lower, upper):
    with pytest.raises(ValueError):
        interval_score([0.0], lower, upper)


@pytest.mark.parametrize("alpha", [0, 1, np.nan, True, "0.1"])
def test_interval_alpha_is_validated(alpha):
    with pytest.raises(ValueError, match="alpha"):
        interval_score([0.0], [-1.0], [1.0], alpha=alpha)


def test_evaluation_preserves_physical_leads_origins_and_model_order(panel):
    evaluation = evaluate_forecasts(panel, loss="absolute")
    assert evaluation.names == ("mean", "baseline", "drift")
    assert evaluation.lead_times == (4, 5, 6, 7)
    np.testing.assert_array_equal(evaluation.origins, [split.origin for split in panel.splits])
    np.testing.assert_array_equal(evaluation.target_indices, panel.target_indices)
    expected = np.array(
        [
            [
                [
                    abs(float(panel.actuals[f, h]) - float(panel.forecasts[f, h, m]))
                    for m in range(3)
                ]
                for h in range(4)
            ]
            for f in range(panel.n_folds)
        ]
    )
    np.testing.assert_array_equal(evaluation.losses, expected)
    for h in range(4):
        for m in range(3):
            expected_mean = (
                sum(float(expected[f, h, m]) for f in range(panel.n_folds)) / panel.n_folds
            )
            assert evaluation.mean_loss[h, m] == pytest.approx(expected_mean, rel=2e-15)
    assert len(evaluation.records()) == 4 * 3
    assert [row["lead_time"] for row in evaluation.records()] == [4] * 3 + [5] * 3 + [6] * 3 + [
        7
    ] * 3


def test_evaluation_exports_are_finite_readonly_and_detached(panel):
    evaluation = evaluate_forecasts(panel, loss="pinball", quantile=np.float32(0.25))
    record = evaluation.to_dict()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert type(evaluation.quantile) is float
    assert record["n_origins"] == panel.n_folds
    for array in (
        evaluation.origins,
        evaluation.target_indices,
        evaluation.losses,
        evaluation.mean_loss,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    assert not np.shares_memory(evaluation.target_indices, panel.target_indices)
    record["scores"][0]["mean_loss"] = -999
    assert evaluation.mean_loss[0, 0] >= 0


def test_large_finite_loss_means_are_not_silently_exported_as_infinity():
    panel = backtest(
        np.full(12, 1e308),
        {"zero": lambda train, leads: np.zeros(len(leads))},
        initial_train_size=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        evaluation = evaluate_forecasts(panel, loss="absolute")
    np.testing.assert_allclose(evaluation.mean_loss, [[1e308]], rtol=2e-15)
    json.dumps(evaluation.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "loss,quantile", [("squared", None), ("absolute", None), ("pinball", 0.25)]
)
@pytest.mark.parametrize("studentization", ["fixed", "resampled"])
def test_comparison_matches_direct_shared_family_bootstrap_elementwise(
    panel, loss, quantile, studentization
):
    options = dict(n_resamples=73, seed=918, studentization=studentization, search_complete=True)
    actual = compare_forecasts(
        panel,
        baseline="baseline",
        lead_time=6,
        loss=loss,
        quantile=quantile,
        alpha=0.1,
        lags=2,
        block_length=4,
        **options,
    )
    # Construct the differences independently, preserving the prespecified
    # baseline in the middle of the input and the original candidate order.
    differences = []
    scores = []
    for f in range(panel.n_folds):
        row = [
            _scalar_loss(panel.actuals[f, 2], panel.forecasts[f, 2, m], loss, quantile)
            for m in range(3)
        ]
        scores.append(row)
        differences.append([row[1] - row[0], row[1] - row[2]])
    expected = run_test(
        differences, names=("mean", "drift"), alpha=0.1, lags=2, block_length=4, **options
    )
    assert actual.inference.names == ("mean", "drift")
    assert actual.baseline == "baseline" and actual.lead_time == 6 and actual.origin_step == 2
    assert actual.inference.method == "bootstrap"
    np.testing.assert_array_equal(actual.inference.mean, expected.mean)
    np.testing.assert_array_equal(actual.inference.decisions, expected.decisions)
    np.testing.assert_array_equal(actual.inference.adjusted_pvalue, expected.adjusted_pvalue)
    np.testing.assert_array_equal(
        actual.inference.details.bootstrap_statistics, expected.details.bootstrap_statistics
    )
    assert actual.inference.global_pvalue == expected.global_pvalue
    assert actual.baseline_loss == pytest.approx(np.mean(scores, axis=0)[1])
    np.testing.assert_allclose(actual.candidate_loss, np.mean(scores, axis=0)[[0, 2]], rtol=2e-15)


def test_single_column_horizon_with_gap_infers_the_physical_lead():
    y = np.random.default_rng(984).normal(size=160)
    panel = backtest(
        y,
        {"base": naive_forecast, "drift": drift_forecast},
        initial_train_size=20,
        horizon=1,
        gap=8,
        step=3,
    )
    result = compare_forecasts(
        panel, baseline="base", n_resamples=31, seed=8, studentization="fixed"
    )
    assert result.lead_time == 9 and result.origin_step == 3
    assert result.inference.sample_size == panel.n_folds
    assert result.inference.diagnostics["lags"] == max(default_lags(panel.n_folds), 2)
    assert result.inference.diagnostics["block_length"] == max(
        default_block_length(panel.n_folds), 3
    )


def test_overlap_defaults_use_physical_lead_in_origin_steps():
    y = np.random.default_rng(723).normal(size=160)
    panel = backtest(
        y,
        {"base": naive_forecast, "drift": drift_forecast},
        initial_train_size=20,
        horizon=3,
        gap=14,
        step=2,
    )
    result = compare_forecasts(
        panel, baseline="base", lead_time=17, n_resamples=31, seed=7, studentization="fixed"
    )
    assert panel.n_folds == 62
    assert result.inference.diagnostics["lags"] == 8
    assert result.inference.diagnostics["block_length"] == 9
    assert not result.warnings


def test_large_gap_defaults_are_capped_and_short_overlap_is_reported():
    y = np.random.default_rng(717).normal(size=80)
    panel = backtest(
        y, {"base": naive_forecast, "drift": drift_forecast}, initial_train_size=20, gap=40
    )
    result = compare_forecasts(
        panel, baseline="base", n_resamples=31, seed=9, studentization="fixed"
    )
    assert panel.n_folds == 20 and result.lead_time == 41
    assert result.inference.diagnostics["lags"] == 18
    assert result.inference.diagnostics["block_length"] == 20
    assert result.warnings and all(
        isinstance(message, str) and message for message in result.warnings
    )
    assert result.to_dict()["comparison_warnings"] == list(result.warnings)


def test_explicit_lag_block_settings_are_not_overridden_by_overlap():
    y = np.random.default_rng(717).normal(size=80)
    panel = backtest(
        y, {"base": naive_forecast, "drift": drift_forecast}, initial_train_size=20, gap=40
    )
    result = compare_forecasts(
        panel,
        baseline="base",
        lags=0,
        block_length=1,
        n_resamples=31,
        seed=9,
        studentization="fixed",
    )
    assert result.inference.diagnostics["lags"] == 0
    assert result.inference.diagnostics["block_length"] == 1
    with pytest.raises(ValueError, match="lags"):
        compare_forecasts(panel, baseline="base", lags=19, n_resamples=3)
    with pytest.raises(ValueError, match="block_length"):
        compare_forecasts(panel, baseline="base", block_length=21, n_resamples=3)


def test_multi_step_comparison_requires_one_explicit_lead_and_does_not_pool(panel):
    with pytest.raises(ValueError, match="lead_time.*multi-step"):
        compare_forecasts(panel, baseline="baseline")
    with pytest.raises(ValueError, match="evaluated lead"):
        compare_forecasts(panel, baseline="baseline", lead_time=1)
    for lead in [True, np.bool_(True), 6.0, 0]:
        with pytest.raises(ValueError, match="lead_time"):
            compare_forecasts(panel, baseline="baseline", lead_time=lead)


def test_missing_baseline_single_model_and_method_override_are_rejected(panel):
    with pytest.raises(ValueError, match="baseline"):
        compare_forecasts(panel, baseline="missing", lead_time=6)
    single = backtest(np.arange(20), {"last": naive_forecast}, initial_train_size=5)
    with pytest.raises(ValueError, match="at least two"):
        compare_forecasts(single, baseline="last")
    with pytest.raises(TypeError):
        compare_forecasts(panel, baseline="baseline", lead_time=6, method="gaussian_ar")


def test_constant_difference_is_not_silently_removed_from_the_family():
    panel = backtest(
        np.random.default_rng(182).normal(size=60),
        {"base": naive_forecast, "duplicate": naive_forecast, "valid": drift_forecast},
        initial_train_size=10,
    )
    with pytest.raises(ValueError, match="Constant.*duplicate"):
        compare_forecasts(panel, baseline="base", n_resamples=3)
    nonzero = backtest(
        np.zeros(20),
        {
            "base": lambda train, leads: np.zeros(len(leads)),
            "always_one": lambda train, leads: np.ones(len(leads)),
        },
        initial_train_size=5,
    )
    with pytest.raises(ValueError, match="Constant.*always_one"):
        compare_forecasts(nonzero, baseline="base", n_resamples=3)


def test_comparison_requires_at_least_eight_origins():
    panel = backtest(
        np.random.default_rng(26).normal(size=12),
        {"base": naive_forecast, "drift": drift_forecast},
        initial_train_size=5,
    )
    with pytest.raises(ValueError):
        compare_forecasts(panel, baseline="base", n_resamples=3)


def test_irregular_origins_or_different_leads_are_rejected(panel):
    targets = panel.target_indices.copy()
    targets[2] += 1
    with pytest.raises(ValueError, match="same positive lead"):
        evaluate_forecasts(replace(panel, target_indices=targets))
    splits = list(panel.splits)
    splits[2] = replace(splits[2], origin=splits[2].origin + 1)
    irregular = replace(panel, splits=tuple(splits), target_indices=targets)
    with pytest.raises(ValueError, match="regularly spaced"):
        compare_forecasts(irregular, baseline="baseline", lead_time=6)
    with pytest.raises(TypeError, match="BacktestResult"):
        evaluate_forecasts(np.zeros((20, 2)))


def test_comparison_exports_are_readonly_finite_and_keep_mean_improvement(panel):
    result = compare_forecasts(
        panel,
        baseline="baseline",
        lead_time=6,
        loss="pinball",
        quantile=np.float32(0.25),
        n_resamples=31,
        seed=5,
        studentization="fixed",
    )
    record = result.to_dict()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert type(result.quantile) is float
    assert not result.candidate_loss.flags.writeable
    with pytest.raises(ValueError):
        result.candidate_loss[0] = 0
    rows = result.records()
    for i, row in enumerate(rows):
        assert row["model"] == result.inference.names[i]
        assert row["mean_improvement"] == result.inference.mean[i]
        assert row["relative_improvement"] == row["mean_improvement"] / result.baseline_loss
        assert row["mean_loss"] == result.candidate_loss[i]
        assert row["baseline_loss"] == result.baseline_loss
        assert "mean" not in row and "name" not in row
    record["candidates"][0]["mean_improvement"] = 999
    assert result.inference.mean[0] != 999


def test_zero_baseline_loss_has_no_invented_relative_improvement():
    panel = backtest(
        np.zeros(20),
        {
            "zero": lambda train, leads: np.zeros(len(leads)),
            "changing": lambda train, leads: np.full(len(leads), len(train)),
        },
        initial_train_size=5,
    )
    result = compare_forecasts(
        panel, baseline="zero", n_resamples=31, seed=5, studentization="fixed"
    )
    assert result.baseline_loss == 0
    assert result.records()[0]["relative_improvement"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_optional_frames_have_explicit_lead_and_model_indexes(panel):
    pd = pytest.importorskip("pandas")
    evaluation = evaluate_forecasts(panel)
    expected = pd.DataFrame.from_records(evaluation.records()).set_index(["lead_time", "model"])
    pd.testing.assert_frame_equal(evaluation.to_frame(), expected)
    result = compare_forecasts(
        panel, baseline="baseline", lead_time=6, n_resamples=31, seed=4, studentization="fixed"
    )
    pd.testing.assert_frame_equal(
        result.to_frame(), pd.DataFrame.from_records(result.records()).set_index("model")
    )


def test_optional_frame_dependency_errors_are_actionable(panel, monkeypatch):
    evaluation = evaluate_forecasts(panel)
    comparison = compare_forecasts(
        panel, baseline="baseline", lead_time=6, n_resamples=31, seed=4, studentization="fixed"
    )
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("Unavailable in this environment")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    for result in [evaluation, comparison]:
        with pytest.raises(ImportError, match="requires pandas"):
            result.to_frame()


@pytest.mark.parametrize("loss,quantile", [("absolute", None), ("squared", None), ("pinball", 0.3)])
@pytest.mark.parametrize("lead_time,step,window", [(1, 1, None), (3, 2, None), (3, 1, 3)])
def test_selection_matches_independent_scalar_rolling_reference(
    loss, quantile, lead_time, step, window
):
    train, validation = [1.0, -1.0, 2.0, 0.0, 3.0], [4.0, -2.0, 1.0, 5.0, 6.0, 7.0]

    def mean_shift(history, leads):
        return [sum(history) / len(history) + 0.3 * int(lead) for lead in leads]

    models = {"last": naive_forecast, "mean": mean_shift}
    result = select_forecaster(
        train,
        validation,
        models,
        loss=loss,
        quantile=quantile,
        lead_time=lead_time,
        step=step,
        window=window,
    )
    values = train + validation
    targets, losses = [], []
    for origin in range(len(train) - 1, len(values) - lead_time, step):
        history = values[0 if window is None else max(0, origin + 1 - window) : origin + 1]
        target = origin + lead_time
        predictions = [history[-1], sum(history) / len(history) + 0.3 * lead_time]
        losses.append(
            [_scalar_loss(values[target], prediction, loss, quantile) for prediction in predictions]
        )
        targets.append(target)
    expected = [sum(row[i] for row in losses) / len(losses) for i in range(2)]
    np.testing.assert_allclose(result.mean_loss, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(result.target_indices, targets)
    winner = 0 if expected[0] <= expected[1] else 1
    assert result.selected_index == winner
    assert result.name == tuple(models)[winner]
    assert result.forecaster is models[result.name]
    assert result.names == tuple(models)
    assert result.lead_time == lead_time
    assert result.train_size == len(train)
    assert result.validation_size == len(validation)


def test_selection_callbacks_expose_only_prior_prefix_without_input_or_future_base():
    whole = np.arange(30.0)
    train, validation = whole[:6], whole[6:13]
    calls = []

    def model(history, leads):
        assert not history.flags.writeable and not leads.flags.writeable
        assert history.base is leads.base is None
        assert not np.shares_memory(history, whole)
        calls.append((history.copy(), leads.copy()))
        return [history[-1]]

    result = select_forecaster(train, validation, {"model": model}, lead_time=3, step=2, window=4)
    np.testing.assert_array_equal(result.target_indices, [8, 10, 12])
    for (history, leads), origin in zip(calls, [5, 7, 9], strict=True):
        np.testing.assert_array_equal(history, whole[origin - 3 : origin + 1])
        np.testing.assert_array_equal(leads, [3])
    np.testing.assert_array_equal(whole, np.arange(30.0))


def test_selection_mapping_snapshot_keeps_selected_callback_after_user_mapping_mutation():
    models = {}

    def original(history, leads):
        models["original"] = replacement
        models["new"] = replacement
        return [0.0]

    def replacement(history, leads):
        pytest.fail("User mutation must not change the supplied mapping snapshot")

    models["original"] = original
    result = select_forecaster([0.0, 0.0], [0.0, 0.0], models)
    assert result.names == ("original",)
    assert result.forecaster is original
    assert models["original"] is replacement


def test_exact_selection_ties_keep_supplied_order_and_exports_are_detached():
    def first(history, leads):
        return [0.0]

    def second(history, leads):
        return [0.0]

    result = select_forecaster([1.0, 2.0], [0.0, 0.0, 0.0, 0.0], {"z": first, "a": second}, step=2)
    assert result.name == "z"
    assert result.selected_index == 0
    assert result.forecaster is first
    assert not result.mean_loss.flags.writeable
    assert not result.target_indices.flags.writeable
    assert result.mean_loss.base is result.target_indices.base is None
    with pytest.raises(ValueError):
        result.mean_loss[0] = 9
    with pytest.raises(ValueError):
        result.target_indices[0] = 0
    record = result.to_dict()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["selected_name"] == "z"
    assert [row["model"] for row in record["scores"]] == ["z", "a"]
    assert record["target_range"] == [2, 5]
    assert record["n_origins"] == 2
    record["scores"][0]["mean_loss"] = 99
    record["target_range"][0] = -1
    assert result.mean_loss[0] == 0
    assert result.target_indices[0] == 2


@pytest.mark.parametrize(
    "options",
    [
        {"loss": "unknown"},
        {"loss": "absolute", "quantile": 0.5},
        {"loss": "pinball"},
        {"loss": "pinball", "quantile": True},
        {"loss": "pinball", "quantile": 0},
        {"loss": "pinball", "quantile": np.nan},
    ],
)
def test_selection_invalid_loss_fails_before_calling_models(options):
    def forbidden(history, leads):
        pytest.fail("Invalid loss options must be checked before fitting")

    with pytest.raises(ValueError):
        select_forecaster([1.0, 2.0], [3.0, 4.0], {"model": forbidden}, **options)


@pytest.mark.parametrize(
    "bad", [[], [[1.0]], [np.nan], [np.inf], [True], ["1"], [1j], np.ma.array([1.0], mask=True)]
)
@pytest.mark.parametrize("side", ["train", "validation"])
def test_selection_invalid_segments_fail_before_calling_models(bad, side):
    def forbidden(history, leads):
        pytest.fail("Invalid segments must be checked before fitting")

    values = {"train": [1.0, 2.0], "validation": [3.0, 4.0]}
    values[side] = bad
    with pytest.raises(ValueError, match=side):
        select_forecaster(**values, forecasters={"model": forbidden})


@pytest.mark.parametrize(
    "options",
    [
        {"lead_time": 0},
        {"lead_time": True},
        {"lead_time": 1.5},
        {"step": 0},
        {"step": True},
        {"window": 0},
        {"window": True},
        {"lead_time": 3},
    ],
)
def test_selection_invalid_boundaries_fail_before_calling_models(options):
    def forbidden(history, leads):
        pytest.fail("Invalid boundaries must be checked before fitting")

    with pytest.raises(ValueError):
        select_forecaster([1.0, 2.0], [3.0, 4.0], {"model": forbidden}, **options)


@pytest.mark.parametrize(
    "models", [[], [("a", naive_forecast)], {}, {"": naive_forecast}, {"a": 1}]
)
def test_selection_forecasters_reuse_mapping_and_callback_validation(models):
    with pytest.raises(ValueError):
        select_forecaster([1.0, 2.0], [3.0], models)


def test_selection_pinball_metadata_normalizes_numpy_scalar_quantile():
    result = select_forecaster(
        [0.0, 1.0],
        [1.0, 2.0],
        {"last": naive_forecast},
        loss="pinball",
        quantile=np.float32(0.25),
    )
    assert type(result.quantile) is float
    assert result.quantile == 0.25
    json.dumps(result.to_dict(), allow_nan=False)
