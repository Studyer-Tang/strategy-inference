"""Independent fold arithmetic, baseline forecasts and isolation checks."""

import builtins
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from strategy_inference.model_selection import (
    Autoregression,
    Differenced,
    RollingSplit,
    SeasonalNaive,
    backtest,
    drift_forecast,
    naive_forecast,
    rolling_splits,
)


def test_expanding_folds_match_hand_written_half_open_ranges():
    splits = rolling_splits(12, initial_train_size=4, horizon=3, step=2, gap=1)
    assert splits == (
        RollingSplit(0, 4, 5, 8, 3),
        RollingSplit(0, 6, 7, 10, 5),
        RollingSplit(0, 8, 9, 12, 7),
    )
    assert isinstance(splits, tuple)
    with pytest.raises(FrozenInstanceError):
        splits[0].train_stop = 100


def test_fixed_window_caps_initial_history_and_keeps_the_same_origins():
    assert rolling_splits(12, initial_train_size=4, horizon=3, step=2, gap=1, window=3) == (
        RollingSplit(1, 4, 5, 8, 3),
        RollingSplit(3, 6, 7, 10, 5),
        RollingSplit(5, 8, 9, 12, 7),
    )
    assert rolling_splits(6, initial_train_size=3) == (
        RollingSplit(0, 3, 3, 4, 2),
        RollingSplit(0, 4, 4, 5, 3),
        RollingSplit(0, 5, 5, 6, 4),
    )
    assert rolling_splits(6, initial_train_size=3, window=100) == rolling_splits(
        6,
        initial_train_size=3,
    )


def test_numpy_integer_settings_are_accepted_and_partial_test_is_not_added():
    splits = rolling_splits(
        np.int64(11),
        initial_train_size=np.int32(4),
        horizon=np.int64(3),
        step=np.int32(2),
        gap=np.int64(1),
        window=np.int32(5),
    )
    assert splits == (RollingSplit(0, 4, 5, 8, 3), RollingSplit(1, 6, 7, 10, 5))
    assert all(type(value) is int for split in splits for value in vars(split).values())


@pytest.mark.parametrize(
    "parameter", ["n_obs", "initial_train_size", "horizon", "step", "gap", "window"]
)
@pytest.mark.parametrize("value", [True, np.bool_(False), 1.0, "1", -1, None])
def test_split_settings_require_nonbool_integers(parameter, value):
    if parameter == "window" and value is None:
        return
    options = dict(n_obs=12, initial_train_size=4, horizon=2, step=1, gap=0, window=5)
    options[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        rolling_splits(**options)


@pytest.mark.parametrize("parameter", ["n_obs", "initial_train_size", "horizon", "step", "window"])
def test_zero_sizes_are_rejected(parameter):
    options = dict(n_obs=12, initial_train_size=4, horizon=2, step=1, gap=0, window=5)
    options[parameter] = 0
    with pytest.raises(ValueError, match=parameter):
        rolling_splits(**options)


@pytest.mark.parametrize(
    "options",
    [
        dict(initial_train_size=13),
        dict(initial_train_size=12),
        dict(initial_train_size=4, horizon=9),
        dict(initial_train_size=4, gap=8),
        dict(initial_train_size=4, horizon=4, gap=5),
    ],
)
def test_out_of_bounds_and_no_complete_test_are_errors(options):
    with pytest.raises(ValueError):
        rolling_splits(12, **options)


def test_backtest_arrays_and_gap_leads_match_manual_forecasts():
    y = np.arange(12, dtype=float)
    result = backtest(
        y,
        {"drift": drift_forecast, "last": naive_forecast},
        initial_train_size=4,
        horizon=3,
        step=2,
        gap=1,
    )
    assert result.names == ("drift", "last")
    assert result.n_obs == result.sample_size == 12
    assert result.n_folds == result.horizon == 3 and result.n_models == 2
    np.testing.assert_array_equal(result.target_indices, [[5, 6, 7], [7, 8, 9], [9, 10, 11]])
    np.testing.assert_array_equal(result.actuals, [[5, 6, 7], [7, 8, 9], [9, 10, 11]])
    np.testing.assert_array_equal(result.forecasts[:, :, 0], result.actuals)
    np.testing.assert_array_equal(result.forecasts[:, :, 1], [[3, 3, 3], [5, 5, 5], [7, 7, 7]])
    for array in (result.forecasts, result.actuals, result.target_indices):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


def test_callbacks_receive_only_their_fold_history_with_no_future_in_the_base():
    y = np.arange(12, dtype=float)
    seen = []

    def inspect(train, leads):
        assert train.ndim == leads.ndim == 1
        assert not train.flags.writeable and not leads.flags.writeable
        assert not np.shares_memory(train, y)
        base = train
        while isinstance(base.base, np.ndarray):
            base = base.base
        np.testing.assert_array_equal(base, train)
        with pytest.raises(ValueError):
            train[0] = 99
        with pytest.raises(ValueError):
            leads[0] = 99
        seen.append((train.copy(), leads.copy()))
        return train[-1] + leads

    result = backtest(
        y, {"inspect": inspect}, initial_train_size=4, horizon=3, step=2, gap=1, window=3
    )
    expected = [([1, 2, 3], [2, 3, 4]), ([3, 4, 5], [2, 3, 4]), ([5, 6, 7], [2, 3, 4])]
    for (train, leads), (expected_train, expected_leads) in zip(seen, expected, strict=True):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(leads, expected_leads)
    np.testing.assert_array_equal(result.forecasts[:, :, 0], result.actuals)


def test_even_reenabled_write_flags_cannot_change_other_models_or_input():
    y = np.arange(8, dtype=float)

    def mutate_own_copies(train, leads):
        train.flags.writeable = leads.flags.writeable = True
        train[:] = -100
        leads[:] = 999
        return np.full(2, -100.0)

    result = backtest(
        y,
        {"mutator": mutate_own_copies, "drift": drift_forecast},
        initial_train_size=3,
        horizon=2,
        step=2,
        gap=1,
    )
    np.testing.assert_array_equal(y, np.arange(8))
    np.testing.assert_array_equal(result.forecasts[:, :, 1], [[4, 5], [6, 7]])
    np.testing.assert_array_equal(result.actuals, [[4, 5], [6, 7]])


@pytest.mark.parametrize(
    "values",
    [
        [],
        None,
        [[1, 2], [3, 4]],
        [[1], [2, 3]],
        [1, np.nan, 3],
        [1, np.inf, 3],
        [1 + 1j, 2 + 1j],
        [True, False, True],
        ["1", "2", "3"],
        np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False]),
    ],
)
def test_bad_series_are_rejected_before_callbacks(values):
    def must_not_run(train, leads):
        raise AssertionError("Invalid series reached callback")

    with pytest.raises(ValueError, match="y"):
        backtest(values, {"never": must_not_run}, initial_train_size=1)


@pytest.mark.parametrize(
    "forecasters",
    [None, [], {}, {"": naive_forecast}, {" ": naive_forecast}, {1: naive_forecast}, {"bad": 5}],
)
def test_forecaster_mapping_must_have_names_and_callables(forecasters):
    with pytest.raises(ValueError):
        backtest([1, 2, 3], forecasters, initial_train_size=1)


@pytest.mark.parametrize(
    "prediction",
    [
        1.0,
        [1.0],
        [],
        [[1.0, 2.0]],
        [1.0, 2.0, 3.0],
        [np.nan, 2.0],
        [1.0, np.inf],
        [1 + 1j, 2 + 0j],
        [True, False],
        ["1", "2"],
    ],
)
def test_bad_prediction_shapes_or_values_abort_with_model_and_fold(prediction):
    calls = []

    def broken(train, leads):
        calls.append(len(train))
        return prediction

    with pytest.raises(ValueError, match=r"Forecaster 'broken' at fold 0 \(origin 2\)"):
        backtest(np.arange(9), {"broken": broken}, initial_train_size=3, horizon=2)
    assert calls == [3]


def test_callback_failure_does_not_skip_a_bad_fold():
    calls = []

    def fails_on_second_fold(train, leads):
        calls.append(len(train))
        if len(train) == 4:
            raise ArithmeticError("model did not converge")
        return naive_forecast(train, leads)

    with pytest.raises(RuntimeError, match=r"'unstable' at fold 1 \(origin 3\)") as error:
        backtest(np.arange(9), {"unstable": fails_on_second_fold}, initial_train_size=3)
    assert isinstance(error.value.__cause__, ArithmeticError)
    assert calls == [3, 4]


def test_constant_and_single_observation_history_are_valid_for_naive():
    result = backtest([2, 2, 2, 2], {"naive": naive_forecast}, initial_train_size=1)
    np.testing.assert_array_equal(result.forecasts[:, :, 0], [[2], [2], [2]])


def test_baselines_match_independent_multistep_and_seasonal_values():
    np.testing.assert_array_equal(naive_forecast([1, 3, 4], [1, 3, 6]), [4, 4, 4])
    np.testing.assert_array_equal(drift_forecast([1, 3, 4], [1, 3, 6]), [5.5, 8.5, 13])
    seasonal = SeasonalNaive(3)
    np.testing.assert_array_equal(
        seasonal([2, 4, 8, 16, 32], [1, 2, 3, 4, 6, 7]), [8, 16, 32, 8, 32, 8]
    )
    np.testing.assert_array_equal(SeasonalNaive(1)([1, 3, 4], [1, 3, 6]), [4, 4, 4])
    result = backtest(
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        {"seasonal": seasonal},
        initial_train_size=4,
        horizon=3,
        gap=2,
        step=2,
    )
    np.testing.assert_array_equal(result.forecasts[:, :, 0], [[1, 2, 3]])


@pytest.mark.parametrize("period", [0, -1, True, np.bool_(True), 2.0, "2", None])
def test_seasonal_period_requires_positive_nonbool_integer(period):
    with pytest.raises(ValueError, match="period"):
        SeasonalNaive(period)


def test_baselines_report_insufficient_history_and_numerical_overflow():
    with pytest.raises(ValueError, match="period training"):
        SeasonalNaive(4)([1, 2, 3], [1])
    with pytest.raises(ValueError, match="at least two"):
        drift_forecast([1], [1])
    with pytest.raises(ValueError, match="finite"):
        drift_forecast([-1e308, 1e308], [1])


@pytest.mark.parametrize(
    "leads", [[], [0], [-1], [1.0], [True], [[1]], [1 + 1j], np.array([2**64 - 1], dtype=np.uint64)]
)
@pytest.mark.parametrize("forecaster", [naive_forecast, drift_forecast, SeasonalNaive(2)])
def test_baselines_require_positive_integer_leads(forecaster, leads):
    with pytest.raises(ValueError, match="lead_times"):
        forecaster([1, 2, 3], leads)


def test_exports_keep_positions_model_order_and_overlap_without_aliasing():
    result = backtest(
        np.arange(7),
        {"last": naive_forecast, "drift": drift_forecast},
        initial_train_size=3,
        horizon=2,
    )
    record = result.to_dict()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["names"] == ["last", "drift"]
    rows = result.records()
    assert len(rows) == 3 * 2 * 2
    assert rows[0] == dict(
        fold=0, origin=2, lead_time=1, target_index=3, model="last", actual=3.0, forecast=2.0
    )
    assert rows[3]["target_index"] == rows[4]["target_index"] == 4
    assert rows[3]["origin"] != rows[4]["origin"]
    record["forecasts"][0][0][0] = 900
    record["splits"][0]["train_start"] = 900
    rows[0]["actual"] = 900
    assert result.forecasts[0, 0, 0] == 2 and result.actuals[0, 0] == 3
    assert result.splits[0].train_start == 0


def test_optional_frame_has_the_same_long_rows():
    pd = pytest.importorskip("pandas")
    result = backtest([1, 2, 3, 4], {"last": naive_forecast}, initial_train_size=2)
    pd.testing.assert_frame_equal(result.to_frame(), pd.DataFrame.from_records(result.records()))


def test_frame_missing_dependency_has_actionable_error(monkeypatch):
    result = backtest([1, 2, 3, 4], {"last": naive_forecast}, initial_train_size=2)
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("Unavailable in this environment")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ImportError, match="requires pandas"):
        result.to_frame()


def _ar_reference(train, lags, leads, ridge=0):
    """Independent row construction and intercept-preserving ridge system."""
    train = np.asarray(train, dtype=float)
    if ridge:
        train_center = float(np.mean(train))
        train_scale = float(np.max(np.abs(train - train_center)))
        work = (train - train_center) / train_scale
    else:
        train_center, train_scale, work = 0.0, 1.0, train
    offsets = tuple(range(1, lags + 1)) if isinstance(lags, int) else lags
    maximum, count = max(offsets), len(offsets)
    x = np.array([[work[t - lag] for lag in offsets] for t in range(maximum, len(work))])
    y = work[maximum:]
    if ridge:
        # This ordinary-unit design explicitly includes the unpenalized intercept.
        design = np.column_stack((np.ones(len(x)), x))
        penalty = np.zeros((count, count + 1))
        penalty[:, 1:] = np.sqrt(ridge) * np.eye(count)
        coefficient = np.linalg.lstsq(
            np.vstack((design, penalty)), np.concatenate((y, np.zeros(count))), rcond=None
        )[0]
    else:
        coefficient = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
    history = work.tolist()
    for _ in range(max(leads)):
        history.append(
            coefficient[0]
            + sum(coefficient[index + 1] * history[-lag] for index, lag in enumerate(offsets))
        )
    return np.array([history[len(work) + lead - 1] * train_scale + train_center for lead in leads])


@pytest.mark.parametrize("lags", [1, 3, 12])
@pytest.mark.parametrize("ridge", [0, 0.1, 2.5])
def test_autoregression_matches_independent_ols_and_augmented_ridge(lags, ridge):
    train = np.random.default_rng(801 + lags).normal(size=80).cumsum() + 40
    leads = [9, 1, 3, 9, 2]
    actual = Autoregression(lags=lags, ridge=ridge)(train, leads)
    expected = _ar_reference(train, lags, leads, ridge)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual.shape == (len(leads),) and np.isfinite(actual).all()


def test_autoregression_recursive_ar1_matches_hand_written_values():
    # y[t] = 2 + 0.5*y[t-1]; fitting must not substitute future observations.
    train = np.array([0, 2, 3, 3.5, 3.75, 3.875])
    np.testing.assert_allclose(Autoregression(1)(train, [1, 2, 4]), [3.9375, 3.96875, 3.9921875])


@pytest.mark.parametrize("ridge", [0.0, 0.7])
def test_autoregression_training_normalization_is_affine_equivariant(ridge):
    train = np.random.default_rng(906).normal(size=90).cumsum()
    ar = Autoregression(4, ridge)
    expected = ar(train, [1, 4, 8])
    for multiplier, offset in [(1e200, 3e200), (1e-200, 0.0), (-5, 1000)]:
        actual = ar(multiplier * train + offset, [1, 4, 8])
        np.testing.assert_allclose((actual - offset) / multiplier, expected, rtol=5e-12, atol=5e-12)


@pytest.mark.parametrize("value", [0.0, -2.0, np.finfo(float).max, -np.finfo(float).max])
@pytest.mark.parametrize("ridge", [0.0, 1.0])
def test_autoregression_constant_history_preserves_intercept_and_extreme_scale(value, ridge):
    np.testing.assert_array_equal(Autoregression(2, ridge)([value] * 7, [1, 1000]), [value, value])


def test_autoregression_rank_deficient_centered_system_has_valid_minimum_norm_fit():
    train = np.arange(20, dtype=float)
    np.testing.assert_allclose(Autoregression(4)(train, [1, 2, 10]), [20, 21, 29], atol=1e-12)
    # The minimum permitted history is accepted without requiring full rank.
    np.testing.assert_allclose(Autoregression(2)([1, 2, 3, 4, 5], [1, 3]), [6, 8], atol=1e-12)


def test_autoregression_is_frozen_stateless_and_causal_inside_backtest():
    y = np.random.default_rng(109).normal(size=45).cumsum()
    changed = y.copy()
    changed[30:] += 10000
    ar = Autoregression(3, 0.2)
    first = ar(y[:20], [1, 4])
    ar(y[10:35], [1, 4])
    np.testing.assert_array_equal(ar(y[:20], [1, 4]), first)
    assert vars(ar) == {"lags": 3, "ridge": 0.2}
    with pytest.raises(FrozenInstanceError):
        ar.lags = 1
    options = dict(initial_train_size=15, horizon=3, gap=2, step=3, window=12)
    run = backtest(y, {"ar": ar}, **options)
    perturbed = backtest(changed, {"ar": ar}, **options)
    for i, split in enumerate(run.splits):
        expected = _ar_reference(y[split.train_start : split.train_stop], 3, [3, 4, 5], 0.2)
        np.testing.assert_allclose(run.forecasts[i, :, 0], expected, rtol=1e-12, atol=1e-12)
        if split.train_stop <= 30:
            np.testing.assert_array_equal(run.forecasts[i], perturbed.forecasts[i])


@pytest.mark.parametrize("lags", [0, -1, True, np.bool_(True), 2.0, "2", None])
def test_autoregression_lags_require_a_positive_nonbool_integer(lags):
    with pytest.raises(ValueError, match="lags"):
        Autoregression(lags)


@pytest.mark.parametrize(
    "ridge",
    [
        -1,
        np.inf,
        np.nan,
        True,
        np.bool_(False),
        "1",
        1j,
        [0.1],
        np.array(0.1),
        np.ma.array(0.1, mask=True),
        10**500,
    ],
)
def test_autoregression_ridge_requires_a_finite_nonnegative_scalar(ridge):
    with pytest.raises(ValueError, match="ridge"):
        Autoregression(1, ridge)


def test_autoregression_numpy_scalar_parameters_and_training_errors():
    ar = Autoregression(np.int64(2), np.float64(0.5))
    assert type(ar.lags) is int and type(ar.ridge) is float
    with pytest.raises(ValueError, match="2 \\* lags \\+ 1"):
        ar([1, 2, 3, 4], [1])
    for train in (
        [1, 2, np.nan, 4, 5],
        [1, 2, np.inf, 4, 5],
        [True] * 5,
        np.ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 0]),
    ):
        with pytest.raises(ValueError, match="train"):
            ar(train, [1])


@pytest.mark.parametrize(
    "leads",
    [
        [],
        [0],
        [-1],
        [True],
        [1.0],
        [1j],
        [[1]],
        np.ma.array([1, 2], mask=[0, 1]),
        np.array([2**64 - 1], dtype=np.uint64),
    ],
)
def test_autoregression_requires_valid_unmasked_integer_leads(leads):
    with pytest.raises(ValueError, match="lead_times"):
        Autoregression(1)([1, 2, 3, 4, 5], leads)


@pytest.mark.parametrize("model", [naive_forecast, drift_forecast, SeasonalNaive(2)])
def test_baselines_share_the_unmasked_lead_contract(model):
    with pytest.raises(ValueError, match="lead_times"):
        model([1, 2, 3, 4, 5], np.ma.array([1, 2], mask=[False, True]))


def test_autoregression_overflow_is_reported_without_clipping_or_replacing_forecasts():
    maximum = np.finfo(float).max
    train = maximum * np.array([0.1, 0.2, 0.4, 0.8])
    with pytest.raises(ValueError, match="finite"):
        Autoregression(1)(train, [1])
    ar = Autoregression(1)
    with pytest.raises(ValueError, match="finite"):
        ar([1, 2, 4, 8], [1100])
    np.testing.assert_allclose(ar([1, 2, 4, 8], [1]), [16])


@pytest.mark.parametrize("lags", [(2,), (1, 3, 7), (1, 2, 24, 48)])
@pytest.mark.parametrize("ridge", [0.0, 0.2, 10.0])
def test_sparse_ar_matches_independently_indexed_ols_and_ridge(lags, ridge):
    train = np.random.default_rng(411).normal(size=120).cumsum() + 20
    leads = [8, 1, 3, 8, 2]
    expected = _ar_reference(train, lags, leads, ridge)
    np.testing.assert_allclose(
        Autoregression(lags, ridge)(train, leads), expected, rtol=2e-12, atol=2e-12
    )


def test_sparse_lags_are_canonical_and_contiguous_tuple_preserves_dense_predictions():
    ar = Autoregression((np.int64(7), 1, np.int32(3)), np.float64(0.2))
    assert ar.lags == (1, 3, 7) and all(type(lag) is int for lag in ar.lags)
    assert vars(ar) == {"lags": (1, 3, 7), "ridge": 0.2}
    train = np.random.default_rng(122).normal(size=50).cumsum()
    np.testing.assert_array_equal(ar(train, [1, 4]), Autoregression((1, 3, 7), 0.2)(train, [1, 4]))
    for ridge in (0, 0.2):
        np.testing.assert_allclose(
            Autoregression((1, 2, 3), ridge)(train, [1, 4, 9]),
            Autoregression(3, ridge)(train, [1, 4, 9]),
            rtol=2e-12,
            atol=2e-12,
        )
    assert Autoregression().lags == 12


@pytest.mark.parametrize(
    "lags",
    [
        (),
        (1, 1),
        (0, 2),
        (-1, 2),
        (True, 2),
        (np.bool_(True), 2),
        (1.0, 2),
        ("1", 2),
        (None, 2),
        [1, 2],
        np.array([1, 2]),
        np.ma.array([1, 2], mask=[0, 1]),
    ],
)
def test_sparse_ar_rejects_empty_duplicate_or_invalid_lags(lags):
    with pytest.raises(ValueError, match="lags"):
        Autoregression(lags)


def test_sparse_minimum_history_uses_maximum_lag_and_coefficient_count():
    ar = Autoregression((2, 5))  # max lag 5 + two coefficients + intercept = 8.
    with pytest.raises(ValueError, match=r"max\(lags\) \+ len\(lags\) \+ 1"):
        ar(np.arange(7), [1])
    np.testing.assert_allclose(ar(np.arange(8), [1, 4]), [8, 11], atol=1e-12)
    np.testing.assert_array_equal(Autoregression((100,))([3.0] * 102, [1, 5]), [3, 3])


@pytest.mark.parametrize("period", [1, 4])
@pytest.mark.parametrize("lags", [3, (1, 3, 7)])
@pytest.mark.parametrize("ridge", [0.0, 0.2])
def test_differenced_ar_composition_matches_independent_difference_fit_and_inverse(
    period, lags, ridge
):
    train = np.random.default_rng(923).normal(size=100).cumsum() + 30
    leads = [9, 1, 3, 9]
    differences = np.array([train[t] - train[t - period] for t in range(period, len(train))])
    changes = _ar_reference(differences, lags, list(range(1, max(leads) + 1)), ridge)
    restored = train.tolist()
    for change in changes:
        restored.append(change + restored[-period])
    expected = np.array([restored[len(train) + lead - 1] for lead in leads])
    result = Differenced(Autoregression(lags, ridge), period)(train, leads)
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)


def test_seasonal_difference_restores_observed_cycle_then_recursive_forecast_cycle():
    calls = []

    def forecast_changes(train, leads):
        calls.append((train.copy(), leads.copy()))
        return np.arange(1, len(leads) + 1, dtype=float)

    result = Differenced(forecast_changes, period=3)([2, 5, 11, 4, 10, 20], [8, 1, 4, 8])
    np.testing.assert_array_equal(result, [25, 5, 9, 25])
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [2, 5, 9])
    np.testing.assert_array_equal(calls[0][1], np.arange(1, 9))
    np.testing.assert_array_equal(
        Differenced(naive_forecast)([1, 3, 5, 7, 9], [3, 1, 6]), [15, 11, 21]
    )


def test_differenced_callback_has_detached_readonly_inputs_with_no_future_base():
    complete = np.arange(30, dtype=float) ** 2
    original = complete.copy()
    requested = np.array([4, 1, 4])

    def inspect(train, leads):
        assert not train.flags.writeable and not leads.flags.writeable
        assert not np.shares_memory(train, complete) and not np.shares_memory(leads, requested)
        base = train
        while isinstance(base.base, np.ndarray):
            base = base.base
        np.testing.assert_array_equal(base, original[3:12] - original[:9])
        np.testing.assert_array_equal(leads, [1, 2, 3, 4])
        with pytest.raises(ValueError):
            train[0] = 99
        with pytest.raises(ValueError):
            leads[0] = 99
        # Even forcing own arrays writable must not alter original history/leads.
        train.flags.writeable = leads.flags.writeable = True
        train[:] = -100
        leads[:] = 99
        return np.zeros(len(leads))

    wrapper = Differenced(inspect, period=3)
    np.testing.assert_array_equal(wrapper(complete[:12], requested), [81, 81, 81])
    np.testing.assert_array_equal(complete, original)
    np.testing.assert_array_equal(requested, [4, 1, 4])
    with pytest.raises(FrozenInstanceError):
        wrapper.period = 1


def test_sparse_differenced_backtest_gap_and_fixed_window_are_causal():
    y = np.random.default_rng(166).normal(size=65).cumsum()
    future_changed = y.copy()
    future_changed[50:] += 10000
    model = Differenced(Autoregression((1, 3, 7), 0.2), period=4)
    options = dict(initial_train_size=32, horizon=3, gap=3, step=5, window=24)
    original = backtest(y, {"sparse-diff": model}, **options)
    changed = backtest(future_changed, {"sparse-diff": model}, **options)
    for fold, split in enumerate(original.splits):
        history = y[split.train_start : split.train_stop].tolist()
        difference = [history[t] - history[t - 4] for t in range(4, len(history))]
        predictions = _ar_reference(difference, (1, 3, 7), [1, 2, 3, 4, 5, 6], 0.2)
        restored = history.copy()
        for prediction in predictions:
            restored.append(prediction + restored[-4])
        np.testing.assert_allclose(
            original.forecasts[fold, :, 0], restored[-3:], rtol=2e-12, atol=2e-12
        )
        if split.train_stop <= 50:
            np.testing.assert_array_equal(original.forecasts[fold], changed.forecasts[fold])


@pytest.mark.parametrize("period", [0, -1, True, np.bool_(True), 2.0, "2", None])
def test_differenced_period_requires_positive_nonbool_integer(period):
    with pytest.raises(ValueError, match="period"):
        Differenced(naive_forecast, period)


@pytest.mark.parametrize("callback", [None, 1, "naive", [naive_forecast]])
def test_differenced_requires_callable_model(callback):
    with pytest.raises(ValueError, match="forecaster"):
        Differenced(callback)


@pytest.mark.parametrize(
    "prediction",
    [
        [1],
        [[1, 2, 3]],
        [1, np.nan, 2],
        [1, np.inf, 2],
        [True, True, True],
        np.ma.array([1, 2, 3], mask=[0, 1, 0]),
    ],
)
def test_differenced_never_skips_invalid_intervening_forecasts(prediction):
    with pytest.raises(ValueError):
        Differenced(lambda train, leads: prediction)([1, 2, 3, 4], [3])


@pytest.mark.parametrize(
    "leads",
    [
        [],
        [0],
        [-1],
        [1.0],
        [True],
        [[1]],
        np.ma.array([1, 2], mask=[0, 1]),
        np.array([2**64 - 1], dtype=np.uint64),
    ],
)
def test_differenced_rejects_invalid_or_masked_leads_before_callback(leads):
    def must_not_run(*args):
        pytest.fail("Callback ran before input validation.")

    with pytest.raises(ValueError, match="lead_times"):
        Differenced(must_not_run)([1, 2, 3, 4], leads)


def test_differenced_minimum_training_is_checked_before_the_callback():
    def must_not_run(*args):
        pytest.fail("Callback ran before validating its training data.")

    with pytest.raises(ValueError, match="more than period"):
        Differenced(must_not_run, period=3)([1, 2, 3], [1])
    with pytest.raises(ValueError, match="train"):
        Differenced(must_not_run)([1, np.nan, 3], [1])
    with pytest.raises(ValueError, match="max\\(lags\\)"):
        Differenced(Autoregression((2, 5)), period=3)(np.arange(10), [1])


def test_differenced_training_overflow_fails_before_callback():
    def must_not_run(*args):
        pytest.fail("Overflowing differences reached the callback.")

    maximum = np.finfo(float).max
    with pytest.raises(ValueError, match="Training differences"):
        Differenced(must_not_run)([-maximum, maximum], [1])


@pytest.mark.parametrize("period", [1, 2])
def test_differenced_unrequested_intermediate_level_overflow_is_reported(period):
    maximum = np.finfo(float).max
    history = [0, maximum * 0.75] if period == 1 else [0, 0, maximum * 0.75, maximum * 0.5]
    changes = [maximum * 0.75, -maximum * 0.75] if period == 1 else [maximum * 0.75, 0]
    with pytest.raises(ValueError, match="Restored forecasts"):
        Differenced(lambda train, leads: changes, period)(history, [2])
