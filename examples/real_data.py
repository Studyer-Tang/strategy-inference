"""Small historical-data backtests; from-scratch models, not a zero-shot benchmark.

Run ``python examples/real_data.py --dataset all`` after installing the library.
Use ``--offline`` to require previously downloaded, hash-verified archives.
The chronological 70/15/15 split below is this example's protocol, not the
published Monash/HF train/validation/test split.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from strategy_inference import (
    Autoregression,
    Differenced,
    SeasonalNaive,
    __version__,
    backtest,
    drift_forecast,
    evaluate_forecasts,
    load_dataset,
    naive_forecast,
    select_forecaster,
    sequential_compare_forecasts,
)

# Fixed candidate grid, common validation targets, and unchanged simple baselines.
# (series, frequency, seasonal period, training window, AR lags, final slice)
_SETTINGS = {
    "fred_md": ("T1", "monthly", 12, 120, 12, None),
    "bitcoin": ("price", "daily", 7, 365, 7, None),
    "oikolab_weather": ("T1", "hourly", 24, 336, 24, 365 * 24),
}
_RIDGE_GRID = (0.0, 1.0, 10.0)
_LAG_GRID = {
    "fred_md": (12, (1, 2, 3, 12)),
    "bitcoin": (7, (1, 2, 7)),
    "oikolab_weather": (24, (1, 2, 24, 48, 168)),
}
_SOURCE_NOTES = {
    "fred_md": "Frozen archive values and anonymous series names; no release-vintage information.",
    "bitcoin": "Historical scraped price series; not a current or point-in-time market feed.",
    "oikolab_weather": "Provider climate series; the archive does not specify a timezone.",
}
_SMCS_SCOPE = (
    "The strong conditional target requires a model to have no larger expected absolute loss "
    "than every original competitor at every update, conditional on information before issue. "
    "It can be empty; exclusions do not recover. This is not a set of test-average winners "
    "or a floating-point rounding certificate."
)


def _prepare(dataset, name):
    series_name, frequency, _, _, _, tail = _SETTINGS[name]
    if dataset.frequency != frequency:
        raise ValueError(f"Expected {frequency!r} data for {name}.")
    series = dataset[series_name]
    values = series.values
    start, stop = 0, len(values)
    if name == "bitcoin":
        observed = np.flatnonzero(~np.isnan(values))
        if not observed.size:
            raise ValueError("Bitcoin price has no observations after its leading missing values.")
        start = int(observed[0])
    if tail is not None:
        if stop < tail:
            raise ValueError(
                f"{name} requires at least {tail} observations for the fixed final slice."
            )
        start = stop - tail
    selected = values[start:stop]
    if not selected.size or not np.isfinite(selected).all():
        raise ValueError(
            f"{name}/{series_name} contains internal/trailing missing or nonfinite values; "
            "no observations are dropped or interpolated."
        )
    return selected, dict(
        series=series_name,
        attributes={
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in series.attributes.items()
        },
        original_start_timestamp=(
            None if series.start_timestamp is None else series.start_timestamp.isoformat()
        ),
        original_n_observations=stop,
        source_index_range=[start, stop],
        n_observations=len(selected),
    )


def _windowed_forecast(train, lead_times, *, model, window):
    return model(train[-window:], lead_times)


def run_dataset(name, *, cache_dir=None, offline=False):
    """Choose a configuration on earlier labels, then score every test label."""
    if name not in _SETTINGS:
        raise ValueError(f"Choose one of {tuple(_SETTINGS)}.")
    dataset = load_dataset(name, cache_dir=cache_dir, offline=offline)
    values, selection = _prepare(dataset, name)
    _, _, period, window, _, _ = _SETTINGS[name]
    n_obs = len(values)
    train_stop, validation_stop = 7 * n_obs // 10, 85 * n_obs // 100
    minimum_train = max(
        2 * lags + 2 if isinstance(lags, int) else max(lags) + len(lags) + 2
        for lags in _LAG_GRID[name]
    )
    if min(train_stop, window) < max(minimum_train, period):
        raise ValueError("Initial training data are too short for the fixed models.")
    if not train_stop < validation_stop < n_obs:
        raise ValueError("The chronological split requires nonempty validation and test ranges.")

    candidates, configurations = {}, {}
    for candidate_window in (window, 2 * window):
        for candidate_lags in _LAG_GRID[name]:
            for difference in (0, 1):
                for ridge in _RIDGE_GRID:
                    label = f"w={candidate_window}/lags={candidate_lags}/d={difference}/r={ridge:g}"
                    model = Autoregression(lags=candidate_lags, ridge=ridge)
                    if difference:
                        model = Differenced(model)
                    candidates[label] = partial(
                        _windowed_forecast, model=model, window=candidate_window
                    )
                    configurations[label] = dict(
                        window=candidate_window,
                        lags=list(candidate_lags) if isinstance(candidate_lags, tuple) else candidate_lags,
                        difference_period=difference,
                        ridge=ridge,
                    )
    # Only training and validation are passed to selection; every candidate
    # scores the same target indices, regardless of its lag set or window.
    selection_result = select_forecaster(
        values[:train_stop], values[train_stop:validation_stop], candidates,
        loss="absolute", window=2 * window,
    )
    configuration = configurations[selection_result.name]
    models = {
        "naive": naive_forecast,
        "seasonal": SeasonalNaive(period),
        "drift": partial(_windowed_forecast, model=drift_forecast, window=window),
        "ar": selection_result.forecaster,
    }
    test = backtest(
        values,
        models,
        initial_train_size=validation_stop,
        horizon=1,
        window=2 * window,
    )
    mae = evaluate_forecasts(test, loss="absolute").mean_loss[0]
    rmse = np.sqrt(evaluate_forecasts(test, loss="squared").mean_loss[0])
    # A common, fixed scaling denominator uses only the initial training segment.
    denominator = float(np.mean(np.abs(values[period:train_stop] - values[: train_stop - period])))
    if not np.isfinite(denominator):
        raise ValueError("The training seasonal scale is outside the finite float range.")
    sequential = sequential_compare_forecasts(test, lead_time=1, loss="absolute")
    return dict(
        dataset=name,
        source=dataset.source,
        license=dataset.license,
        revision=dataset.revision,
        sha256=dataset.sha256,
        frequency=dataset.frequency,
        source_note=_SOURCE_NOTES[name],
        selection=selection,
        protocol=dict(
            split="Example-specific chronological 70/15/15; half-open indices in selected series.",
            train=[0, train_stop],
            validation=[train_stop, validation_stop],
            test=[validation_stop, n_obs],
            lead_time=1,
            step=1,
            baseline_window=window,
            history_window=2 * window,
            selected_configuration=configuration,
            seasonal_period=period,
            ridge_grid=list(_RIDGE_GRID),
            candidate_configurations=configurations,
            validation_selection=selection_result.to_dict(),
            selection_rule="Minimum validation one-step absolute loss; exact ties choose first grid entry.",
            refit="Fit from scratch at each origin using only that origin's training window.",
            mase_train_denominator=denominator,
            mase_note="Undefined (null) when the initial training seasonal difference is zero.",
        ),
        scores=[
            dict(
                model=model,
                n_origins=test.n_folds,
                mae=float(mae[index]),
                mase=None if denominator == 0 else float(mae[index] / denominator),
                rmse=float(rmse[index]),
            )
            for index, model in enumerate(test.names)
        ],
        sequential=sequential.to_dict(),
    )


def run(dataset="all", *, cache_dir=None, offline=False):
    """Return compact provenance and descriptive scores, without forecast trajectories."""
    names = tuple(_SETTINGS) if dataset == "all" else (dataset,)
    return dict(
        schema_version=1,
        library_version=__version__,
        interpretation=(
            "Development comparisons on previously published fixed-series examples; "
            "configuration selection uses validation only. From-scratch models; "
            "not a foundation-model benchmark or evidence of broad model superiority."
        ),
        sequential_scope=_SMCS_SCOPE,
        datasets=[run_dataset(name, cache_dir=cache_dir, offline=offline) for name in names],
    )


def _print_table(record):
    for result in record["datasets"]:
        selected, protocol = result["selection"], result["protocol"]
        print(
            f"{result['dataset']}/{selected['series']}: {selected['n_observations']} observations, "
            f"{result['frequency']}"
        )
        config = protocol["selected_configuration"]
        print(
            f"AR: window={config['window']}, lags={config['lags']}, "
            f"difference={config['difference_period']}, ridge={config['ridge']:g}"
        )
        start, stop = selected["source_index_range"]
        print(f"Original positions [{start}, {stop}); test targets {protocol['test']}")
        if result["dataset"] == "bitcoin" and start:
            print(f"Omitted {start} leading missing positions; no internal gaps removed.")
        print(f"{'model':<10} {'MAE':>12} {'MASE':>12} {'RMSE':>12}")
        for row in result["scores"]:
            mase = "undefined" if row["mase"] is None else f"{row['mase']:.5g}"
            print(f"{row['model']:<10} {row['mae']:>12.5g} {mase:>12} {row['rmse']:>12.5g}")
        retained = ", ".join(result["sequential"]["confidence_set"]) or "(empty)"
        print(f"SMCS retained: {retained}\n")
    print(
        "SMCS uses the strong conditional target; an exclusion is permanent, not a test-mean ranking."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *_SETTINGS), default="all")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Save compact JSON to a new file; never overwrite."
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    record = run(args.dataset, cache_dir=args.cache_dir, offline=args.offline)
    if args.output is not None:
        contents = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(contents)
    _print_table(record)
    if args.output is not None:
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
