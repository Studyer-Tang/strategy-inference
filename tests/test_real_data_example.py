"""Offline fixtures check the example's split, selection and scoring protocol."""

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from strategy_inference import TimeSeries, TimeSeriesDataset

_PATH = Path(__file__).resolve().parents[1] / "examples" / "real_data.py"
_SPEC = importlib.util.spec_from_file_location("real_data_example", _PATH)
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def _dataset(values, *, name="fred_md", series="T1", frequency="monthly"):
    values = np.array(values, dtype=float)
    values.flags.writeable = False
    item = TimeSeries(
        series,
        values,
        MappingProxyType(dict(series_name=series, start_timestamp=datetime(2000, 1, 1))),
    )
    return TimeSeriesDataset(
        name,
        (item,),
        MappingProxyType(dict(frequency=frequency)),
        "a" * 64,
        source="https://example.test/fixed-source",
        license="CC-BY-4.0",
        revision="b" * 40,
    )


def test_validation_excludes_test_and_uses_only_mature_history(monkeypatch):
    values = np.arange(60, dtype=float) + np.sin(np.arange(60))
    dataset = _dataset(values)
    monkeypatch.setattr(example, "load_dataset", lambda *args, **kwargs: dataset)
    seen = []

    def recording_backtest(y, models, **options):
        result = original(y, models, **options)
        seen.append((np.asarray(y).copy(), options, result))
        return result

    original = example.backtest
    monkeypatch.setattr(example, "backtest", recording_backtest)
    result = example.run_dataset("fred_md")
    assert len(seen) == 2
    validation_y, validation_options, validation = seen[0]
    np.testing.assert_array_equal(validation_y, values[:51])
    assert validation_options["initial_train_size"] == 42
    np.testing.assert_array_equal(validation.target_indices[:, 0], np.arange(42, 51))
    assert validation.splits[-1].train_stop == 50
    test_y, test_options, test = seen[1]
    np.testing.assert_array_equal(test_y, values)
    assert test_options["initial_train_size"] == 51
    np.testing.assert_array_equal(test.target_indices[:, 0], np.arange(51, 60))
    assert test.names == ("naive", "seasonal", "drift", "ar")
    assert result["sequential"]["n_updates"] == 9
    assert result["sequential"]["pending"] is None
    assert result["protocol"]["train"] == [0, 42]
    assert result["protocol"]["validation"] == [42, 51]
    assert result["protocol"]["test"] == [51, 60]


def test_mase_uses_initial_train_only_and_rmse_original_units(monkeypatch):
    values = np.arange(60, dtype=float)
    values[42:] += np.linspace(0, 1000, 18)  # Later variation cannot enter the scale.
    dataset = _dataset(values)
    monkeypatch.setattr(example, "load_dataset", lambda *args, **kwargs: dataset)
    result = example.run_dataset("fred_md")
    assert result["protocol"]["mase_train_denominator"] == 12.0
    naive_errors = values[51:60] - values[50:59]
    naive = result["scores"][0]
    assert naive["mae"] == pytest.approx(np.abs(naive_errors).mean())
    assert naive["mase"] == pytest.approx(np.abs(naive_errors).mean() / 12)
    assert naive["rmse"] == pytest.approx(np.sqrt((naive_errors**2).mean()))


def test_exact_validation_ties_choose_first_penalty_and_zero_mase_is_null(monkeypatch):
    monkeypatch.setattr(example, "load_dataset", lambda *args, **kwargs: _dataset(np.ones(60)))
    result = example.run_dataset("fred_md")
    assert result["protocol"]["validation_mae"] == [0.0, 0.0, 0.0]
    assert result["protocol"]["selected_ridge"] == 0.0
    assert result["protocol"]["mase_train_denominator"] == 0.0
    assert all(row["mase"] is None for row in result["scores"])
    json.dumps(result, allow_nan=False)


def test_bitcoin_leading_missing_is_trimmed_and_source_offset_preserved():
    values = np.r_[np.nan, np.nan, np.arange(60, dtype=float)]
    selected, provenance = example._prepare(
        _dataset(values, name="bitcoin", series="price", frequency="daily"), "bitcoin"
    )
    np.testing.assert_array_equal(selected, values[2:])
    assert provenance["source_index_range"] == [2, 62]
    assert provenance["original_n_observations"] == 62
    assert provenance["original_start_timestamp"] == "2000-01-01T00:00:00"


@pytest.mark.parametrize("values", [[np.nan, 1, np.nan, 3], [np.nan, 1, 2, np.nan], [np.inf, 1, 2]])
def test_bitcoin_internal_trailing_or_infinite_values_are_not_removed(values):
    dataset = _dataset(values, name="bitcoin", series="price", frequency="daily")
    with pytest.raises(ValueError, match="no observations are dropped"):
        example._prepare(dataset, "bitcoin")


def test_weather_is_fixed_final_8760_observations_without_calendar_relabeling():
    dataset = _dataset(np.arange(9000), name="oikolab_weather", frequency="hourly")
    selected, provenance = example._prepare(dataset, "oikolab_weather")
    np.testing.assert_array_equal(selected, np.arange(240, 9000))
    assert provenance["source_index_range"] == [240, 9000]
    assert provenance["original_start_timestamp"] == "2000-01-01T00:00:00"
    assert provenance["n_observations"] == 8760


def test_offline_and_cache_options_are_forwarded_and_json_stays_compact(monkeypatch, tmp_path):
    calls = []

    def loader(name, **options):
        calls.append((name, options))
        return _dataset(np.arange(60))

    monkeypatch.setattr(example, "load_dataset", loader)
    result = example.run("fred_md", cache_dir=tmp_path, offline=True)
    assert calls == [("fred_md", dict(cache_dir=tmp_path, offline=True))]
    encoded = json.dumps(result, allow_nan=False)
    assert "forecasts" not in result["datasets"][0]
    assert "actuals" not in result["datasets"][0]
    assert "strong conditional" in result["sequential_scope"]
    assert len(encoded) < 6000


def test_cli_prints_table_and_writes_new_json_without_overwriting(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(example, "load_dataset", lambda *args, **kwargs: _dataset(np.arange(60)))
    output = tmp_path / "scores.json"
    example.main(["--dataset", "fred_md", "--offline", "--output", str(output)])
    text = capsys.readouterr().out
    assert "MAE" in text and "MASE" in text and "RMSE" in text
    assert "SMCS retained:" in text
    assert not text.startswith("{")
    record = json.loads(output.read_text())
    assert record["datasets"][0]["dataset"] == "fred_md"
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        example.main(["--dataset", "fred_md", "--output", str(output)])
    assert output.read_bytes() == original
