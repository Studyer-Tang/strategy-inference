"""Independent fold arithmetic, baseline forecasts and isolation checks."""

import builtins
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from strategy_inference.model_selection import (
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
