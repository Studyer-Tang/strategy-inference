"""Saved model predictions retain the same time semantics as a computed backtest."""

import json
from dataclasses import replace

import numpy as np
import pytest

from strategy_inference import (
    BacktestResult,
    SeasonalNaive,
    backtest,
    compare_forecasts,
    drift_forecast,
    evaluate_forecasts,
    naive_forecast,
    sequential_compare_forecasts,
)
from strategy_inference.model_selection import RollingSplit


@pytest.mark.parametrize("order", [("naive", "drift", "seasonal"), ("seasonal", "naive", "drift")])
def test_saved_forecasts_reproduce_scoring_and_both_comparison_workflows(order):
    y = np.random.default_rng(601).normal(size=96).cumsum()
    available = {"naive": naive_forecast, "drift": drift_forecast, "seasonal": SeasonalNaive(5)}
    run = backtest(
        y,
        {name: available[name] for name in order},
        initial_train_size=15,
        horizon=2,
        step=4,
        gap=1,
        window=12,
    )
    imported = BacktestResult.from_predictions(
        y,
        {name: run.forecasts[:, :, m] for m, name in enumerate(order)},
        splits=run.splits,
    )
    assert imported.to_dict() == run.to_dict()
    for loss in ("squared", "absolute"):
        expected = evaluate_forecasts(run, loss=loss)
        actual = evaluate_forecasts(imported, loss=loss)
        np.testing.assert_array_equal(actual.losses, expected.losses)
        assert actual.to_dict() == expected.to_dict()
    settings = dict(baseline="naive", lead_time=3, n_resamples=99, seed=98, search_complete=True)
    assert (
        compare_forecasts(imported, **settings).to_dict()
        == compare_forecasts(
            run,
            **settings,
        ).to_dict()
    )
    assert sequential_compare_forecasts(imported, lead_time=3).to_dict() == (
        sequential_compare_forecasts(run, lead_time=3).to_dict()
    )


def test_mapping_order_controls_model_axis_without_changing_predictions():
    y = [1, 3, 2, 5, 4, 7]
    splits = (RollingSplit(0, 3, 3, 5, 2), RollingSplit(1, 4, 4, 6, 3))
    first = np.array([[1, 2], [3, 4]])
    second = np.array([[11, 12], [13, 14]])
    a = BacktestResult.from_predictions(y, {"a": first, "b": second}, splits=splits)
    b = BacktestResult.from_predictions(y, {"b": second, "a": first}, splits=splits)
    assert b.names == ("b", "a")
    np.testing.assert_array_equal(b.forecasts, a.forecasts[:, :, ::-1])
    assert [row["model"] for row in b.records()[:2]] == ["b", "a"]
    assert evaluate_forecasts(b).to_dict()["scores"][0]["model"] == "b"


def test_hand_declared_gap_overlap_and_training_metadata_survive_import():
    splits = (
        RollingSplit(1, 4, 5, 8, 3),
        RollingSplit(3, 6, 7, 10, 5),
        RollingSplit(5, 8, 9, 12, 7),
    )
    result = BacktestResult.from_predictions(
        np.arange(12),
        {"external": [[20, 21, 22], [30, 31, 32], [40, 41, 42]]},
        splits=splits,
    )
    assert result.splits == splits
    assert result.sample_size == 12 and result.n_folds == 3
    assert result.horizon == 3 and result.n_models == 1
    np.testing.assert_array_equal(result.actuals, [[5, 6, 7], [7, 8, 9], [9, 10, 11]])
    assert evaluate_forecasts(result).lead_times == (2, 3, 4)
    rows = result.records()
    assert rows[2]["target_index"] == rows[3]["target_index"] == 7
    assert rows[2]["origin"] == 3 and rows[3]["origin"] == 5
    assert rows[2]["forecast"] == 22 and rows[3]["forecast"] == 30
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False)) == result.to_dict()
    with pytest.raises(ValueError, match="nonoverlapping feedback"):
        sequential_compare_forecasts(result, lead_time=4)


def test_irregular_origins_are_retained_and_inference_keeps_its_own_requirements():
    result = BacktestResult.from_predictions(
        np.arange(20),
        {"a": [[1, 2], [3, 4], [5, 6]], "b": [[2, 3], [4, 5], [6, 7]]},
        splits=(
            RollingSplit(0, 3, 4, 6, 2),
            RollingSplit(2, 7, 8, 10, 6),
            RollingSplit(5, 12, 13, 15, 11),
        ),
    )
    np.testing.assert_array_equal(evaluate_forecasts(result).origins, [2, 6, 11])
    assert sequential_compare_forecasts(result, lead_time=3).n_updates == 3
    with pytest.raises(ValueError, match="regularly spaced"):
        compare_forecasts(result, baseline="a", lead_time=3)


def test_imported_arrays_and_split_sequence_are_detached_readonly_snapshots():
    y = np.arange(7, dtype=float)
    predictions = np.arange(12, dtype=float).reshape(3, 4)
    splits = [RollingSplit(0, 3, 3, 5, 2), RollingSplit(1, 4, 4, 6, 3), RollingSplit(2, 5, 5, 7, 4)]
    result = BacktestResult.from_predictions(
        y,
        {"a": predictions[:, ::2], "b": predictions[:, 1::2]},
        splits=splits,
    )
    saved = result.to_dict()
    y[:] = -99
    predictions[:] = -100
    splits.clear()
    assert result.to_dict() == saved
    for array in (result.forecasts, result.actuals, result.target_indices):
        assert not array.flags.writeable
        assert not np.shares_memory(array, y) and not np.shares_memory(array, predictions)
        with pytest.raises(ValueError):
            array.flat[0] = 0


@pytest.mark.parametrize("predictions", [None, [], {}, {"": [[1]]}, {" ": [[1]]}, {1: [[1]]}])
def test_predictions_require_a_named_nonempty_mapping(predictions):
    with pytest.raises(ValueError, match="predictions"):
        BacktestResult.from_predictions([1, 2], predictions, splits=(RollingSplit(0, 1, 1, 2, 0),))


@pytest.mark.parametrize(
    "prediction",
    [
        [],
        [1, 2],
        [[1]],
        [[1, 2], [3, 4]],
        [[[1], [2]]],
        [[1, 2], [3]],
        [[np.nan, 2]],
        [[1, np.inf]],
        [[1 + 0j, 2 + 0j]],
        [[True, False]],
        [["1", "2"]],
        np.array([[1, 2]], dtype=object),
        np.ma.array([[1.0, 2.0]], mask=[[False, True]]),
        [[10**500, 2]],
    ],
)
def test_invalid_prediction_arrays_do_not_get_reshaped_dropped_or_coerced(prediction):
    with pytest.raises(ValueError, match="Predictions for 'saved'"):
        BacktestResult.from_predictions(
            [1, 2, 3],
            {"saved": prediction},
            splits=(RollingSplit(0, 1, 1, 3, 0),),
        )


@pytest.mark.parametrize(
    "y",
    [
        [],
        [[1, 2, 3]],
        [1, np.nan, 3],
        [1, np.inf, 3],
        [1 + 0j, 2, 3],
        np.ma.array([1, 2, 3], mask=[False, True, False]),
    ],
)
def test_invalid_target_series_are_rejected_without_dropping_positions(y):
    with pytest.raises(ValueError, match="y"):
        BacktestResult.from_predictions(
            y,
            {"saved": [[2, 3]]},
            splits=(RollingSplit(0, 1, 1, 3, 0),),
        )


@pytest.mark.parametrize("splits", [None, (), [], [None], [(0, 1, 1, 3, 0)]])
def test_splits_must_be_explicit_nonempty_rolling_split_objects(splits):
    with pytest.raises(ValueError, match="splits"):
        BacktestResult.from_predictions([1, 2, 3], {"saved": [[2, 3]]}, splits=splits)


@pytest.mark.parametrize(
    "field", ["train_start", "train_stop", "test_start", "test_stop", "origin"]
)
@pytest.mark.parametrize("value", [True, np.bool_(True), 1.0, -1, 2**100, np.uint64(2**64 - 1)])
def test_split_positions_reject_nonintegers_negative_values_and_integer_overflow(field, value):
    split = replace(RollingSplit(0, 2, 3, 5, 1), **{field: value})
    with pytest.raises(ValueError, match="splits"):
        BacktestResult.from_predictions(np.arange(6), {"saved": [[2, 3]]}, splits=(split,))


@pytest.mark.parametrize(
    "split",
    [
        RollingSplit(2, 2, 3, 5, 1),
        RollingSplit(0, 3, 2, 4, 2),
        RollingSplit(0, 2, 3, 3, 1),
        RollingSplit(0, 2, 3, 7, 1),
        RollingSplit(0, 2, 3, 5, 0),
    ],
)
def test_split_ranges_and_last_observed_origin_must_agree(split):
    with pytest.raises(ValueError, match="splits"):
        BacktestResult.from_predictions(np.arange(6), {"saved": [[2, 3]]}, splits=(split,))


@pytest.mark.parametrize(
    "second",
    [
        RollingSplit(0, 2, 3, 5, 1),
        RollingSplit(0, 1, 2, 4, 0),
        RollingSplit(0, 3, 4, 7, 2),
        RollingSplit(0, 3, 5, 7, 2),
    ],
)
def test_origins_cannot_repeat_or_reverse_and_leads_must_match(second):
    with pytest.raises(ValueError, match="origins|horizon|lead"):
        BacktestResult.from_predictions(
            np.arange(8),
            {"saved": [[2, 3], [4, 5]]},
            splits=(RollingSplit(0, 2, 3, 5, 1), second),
        )


def test_numpy_integer_splits_and_unmasked_numeric_arrays_are_accepted():
    split = RollingSplit(*(np.int64(value) for value in (0, 2, 3, 5, 1)))
    result = BacktestResult.from_predictions(
        np.ma.array(np.arange(6), mask=False),
        {"saved": np.ma.array([[7, 8]], mask=False)},
        splits=iter([split]),
    )
    assert all(type(value) is int for value in vars(result.splits[0]).values())
    assert result.forecasts.dtype == result.actuals.dtype == np.float64
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["splits"][0]["origin"] == 1
