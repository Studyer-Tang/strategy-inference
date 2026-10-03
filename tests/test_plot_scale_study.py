"""Saved path ledger integrity and independently computed plotting statistics."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_scale_study", ROOT / "scripts/plot_scale_study.py"
)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


@pytest.fixture
def report():
    settings = dict(n_obs=100, initial_train_size=40, repetitions=2, seed_offset=0)
    scenarios = [dict(name="constant"), dict(name="scale_shift", middle_scale=3)]
    methods = ["fixed", "horizon", "shortest", "half", "mixture"]
    protocol = dict(
        seed=1000,
        profiles={"pilot": settings},
        scenarios=scenarios,
        methods=methods,
        lead_times=[1, 24],
        evaluation_warmup=5,
        local_window=20,
        alpha=0.1,
        change_fractions=[0.45, 0.75],
    )
    scores = {
        "fixed": [12, 113],
        "horizon": [9, 111],
        "shortest": [10, 110],
        "half": [8, 108],
        "mixture": [9, 109],
    }
    rows = [
        dict(
            scenario=scenario["name"],
            replicate=rep,
            seed=1000 + index * 10000 + rep,
            method=method,
            lead_time=h,
            n_evaluated=32,
            coverage=0.875,
            worst_local_coverage_error=[0.2, 0.4][rep],
            mean_interval_score=scores[method][rep],
            interval_score_status="finite",
            mean_finite_width=1.0,
            empty_count=0,
            unbounded_count=0,
        )
        for index, scenario in enumerate(scenarios)
        for rep in range(2)
        for method in methods
        for h in protocol["lead_times"]
    ]
    paths = [
        dict(
            scenario=scenario["name"],
            replicate=0,
            origins=list(range(39, 76)),
            weights=[[0.5, 0.5 + index / 100] for index in range(37)],
            issued_scales=[[1.0, 2.0] for _ in range(37)],
        )
        for scenario in scenarios
    ]
    return dict(
        schema_version=1,
        study="test-study",
        profile="pilot",
        profile_settings=settings,
        protocol=protocol,
        records=rows,
        representative_paths=paths,
        input_fingerprints=[
            dict(scenario=scenario["name"], replicate=rep, seed=1000 + index * 10000 + rep)
            for index, scenario in enumerate(scenarios)
            for rep in range(2)
        ],
    )


def test_score_ci_is_paired_by_path_and_in_score_units(report):
    computed, _ = plotter.statistics(report)
    row = next(
        row
        for row in computed["score_differences"]
        if row["scenario"] == "constant" and row["method"] == "horizon"
    )
    # Differences are [-1, 1] although marginal scores vary by about 100.
    # Student t with df=1 is Cauchy, whose quantile is independently explicit.
    critical = math.tan(math.pi * (0.975 - 0.5))
    assert row["estimate"]["mean"] == 0
    assert row["estimate"]["mcse"] == pytest.approx(1, abs=1e-15)
    assert row["estimate"]["n"] == row["n_pairs"] == 2
    assert row["estimate"]["lower"] == pytest.approx(-critical)
    assert row["estimate"]["upper"] == pytest.approx(critical)
    shuffled = copy.deepcopy(report)
    shuffled["records"].reverse()
    assert plotter.statistics(shuffled)[0] == computed


def test_local_error_ci_uses_path_maxima_and_is_not_clipped(report):
    computed, _ = plotter.statistics(report)
    estimate = computed["local_coverage_error"][0]["estimate"]
    assert estimate["mean"] == pytest.approx(0.3)
    assert estimate["mcse"] == pytest.approx(0.1)
    assert estimate["lower"] < 0 and estimate["upper"] > 1


@pytest.mark.parametrize("method", ["mixture", "shortest"])
def test_any_invalid_path_invalidates_entire_score_comparison(report, method):
    row = next(
        row
        for row in report["records"]
        if row["method"] == method
        and row["scenario"] == "constant"
        and row["replicate"] == 1
        and row["lead_time"] == 24
    )
    row.update(
        mean_interval_score=None, interval_score_status="unbounded_intervals", unbounded_count=1
    )
    computed, _ = plotter.statistics(report)
    selected = [
        row
        for row in computed["score_differences"]
        if row["scenario"] == "constant" and (method == "shortest" or row["method"] == "mixture")
    ]
    assert all(
        row["estimate"] is None and row["invalid_score_pairs"] == 1 and row["n_pairs"] == 2
        for row in selected
    )
    assert all(row["estimate"]["n"] == 2 for row in computed["local_coverage_error"])


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "seed", "count", "status", "nan", "fingerprint"]
)
def test_invalid_complete_ledger_is_rejected(report, problem):
    if problem == "missing":
        report["records"].pop()
    elif problem == "duplicate":
        report["records"].append(copy.deepcopy(report["records"][0]))
    elif problem == "fingerprint":
        report["input_fingerprints"][0]["seed"] += 1
    else:
        key, value = {
            "seed": ("seed", 7),
            "count": ("n_evaluated", 31),
            "status": ("interval_score_status", "empty_intervals"),
            "nan": ("mean_interval_score", float("nan")),
        }[problem]
        report["records"][0][key] = value
    with pytest.raises(ValueError):
        plotter.statistics(report)


@pytest.mark.parametrize("problem", ["replicate", "origin", "weight", "bool", "scale", "missing"])
def test_representative_paths_are_complete_and_preset(report, problem):
    path = report["representative_paths"][0]
    if problem == "replicate":
        path["replicate"] = 1
    elif problem == "origin":
        path["origins"][0] += 1
    elif problem in ("weight", "bool"):
        path["weights"][0][1] = 1.1 if problem == "weight" else True
    elif problem == "scale":
        path["issued_scales"][0][1] = 0
    else:
        report["representative_paths"].pop()
    with pytest.raises(ValueError):
        plotter.statistics(report)


def test_exports_are_reproducible_and_preserve_input(report, tmp_path):
    pytest.importorskip("matplotlib")
    source = tmp_path / "results.json"
    source.write_text(json.dumps(report, allow_nan=False))
    before = source.read_bytes()
    output = tmp_path / "figures"
    first = plotter.plot_study(source, output)
    before_outputs = {path.name: path.read_bytes() for path in first}
    second = plotter.plot_study(source, output)
    assert source.read_bytes() == before
    assert len(second) == 10
    assert all(path.read_bytes() == before_outputs[path.name] for path in second)
    for stem in plotter.STEMS:
        assert (output / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
        assert "<dc:date>" not in (output / f"{stem}.svg").read_text()
        assert b"/CreationDate" not in (output / f"{stem}.pdf").read_bytes()
    manifest = json.loads((output / "figure-data.json").read_text())
    assert manifest["statistics"] == plotter.statistics(report)[0]


@pytest.mark.parametrize("alias", [False, True])
def test_input_cannot_be_overwritten_by_a_managed_target(report, tmp_path, alias):
    pytest.importorskip("matplotlib")
    output = tmp_path / "figures"
    output.mkdir()
    target = output / "figure-data.json"
    target.write_text(json.dumps(report))
    source = tmp_path / "alias.json" if alias else target
    if alias:
        source.symlink_to(target)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="overlap"):
        plotter.plot_study(source, output)
    assert target.read_bytes() == before
    assert list(output.iterdir()) == [target]
