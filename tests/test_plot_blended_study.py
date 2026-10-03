"""Independent path statistics, full source inventory and fixed-weight labels."""

import copy
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "plot_blended_study", ROOT / "scripts/plot_blended_study.py"
)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


@pytest.fixture
def report():
    protocol = json.loads((ROOT / "experiments/blended-scale-protocol.json").read_text())
    settings = dict(n_obs=650, initial_train_size=200, repetitions=2, seed_offset=0)
    protocol["profiles"] = {"full": settings}
    scores = {
        "fixed": [12, 113],
        "horizon": [9, 111],
        "shortest": [10, 110],
        "blend_25": [8.5, 108.5],
        "blend_50": [9, 109],
        "blend_75": [9.5, 109.5],
    }
    rows = [
        dict(
            scenario=scenario["name"],
            replicate=rep,
            seed=protocol["seed"] + s * 10000 + rep,
            method=method,
            lead_time=h,
            n_evaluated=299,
            coverage=0.875,
            worst_local_coverage_error=[0.2, 0.4][rep],
            mean_interval_score=scores[method][rep],
            interval_score_status="finite",
            mean_finite_width=1.0,
            empty_count=0,
            unbounded_count=0,
        )
        for s, scenario in enumerate(protocol["scenarios"])
        for rep in range(2)
        for method in protocol["methods"]
        for h in protocol["lead_times"]
    ]
    return dict(
        schema_version=1,
        study=protocol["study"],
        profile="full",
        profile_settings=settings,
        protocol=protocol,
        records=rows,
        package_version="0.8.0",
        source_sha256=dict.fromkeys(plotter.REQUIRED_SOURCES, "0" * 64),
        input_fingerprints=[
            dict(
                scenario=scenario["name"],
                replicate=rep,
                seed=protocol["seed"] + s * 10000 + rep,
                actual_sha256=f"{s * 2 + rep:064x}",
                predicted_sha256=f"{s * 2 + rep + 1:064x}",
                initial_scales=[1, 2, 3, 4],
            )
            for s, scenario in enumerate(protocol["scenarios"])
            for rep in range(2)
        ],
    )


def test_all_five_sensitivity_points_are_raw_path_means(report):
    computed, _ = plotter.statistics(report)
    points = [row for row in computed["weight_sensitivity"] if row["scenario"] == "constant"]
    assert [row["short_source_weight"] for row in points] == [0, 0.25, 0.5, 0.75, 1]
    assert [row["estimate"]["mean"] for row in points] == [60, 58.5, 59, 59.5, 60]
    assert all(row["estimate"]["n"] == 2 for row in points)
    shuffled = copy.deepcopy(report)
    shuffled["records"].reverse()
    assert plotter.statistics(shuffled)[0] == computed


def test_paired_interval_uses_explicit_cauchy_quantile_not_marginal_variance(report):
    computed, _ = plotter.statistics(report)
    selected = next(
        row
        for row in computed["score_differences"]
        if row["scenario"] == "constant" and row["method"] == "horizon"
    )
    estimate = selected["estimate"]
    assert estimate["mean"] == 0
    assert estimate["mcse"] == pytest.approx(1)
    assert estimate["upper"] == pytest.approx(math.tan(math.pi * 0.475))
    assert {row["method"] for row in computed["local_coverage_error"]} == {
        "fixed",
        "horizon",
        "shortest",
        "blend_50",
    }


def test_invalid_path_remains_in_pair_count_and_sensitivity_point(report):
    row = next(
        row
        for row in report["records"]
        if row["scenario"] == "constant"
        and row["method"] == "blend_50"
        and row["replicate"] == 0
        and row["lead_time"] == 24
    )
    row.update(mean_interval_score=None, interval_score_status="empty_intervals", empty_count=1)
    computed, _ = plotter.statistics(report)
    selected = next(
        row
        for row in computed["score_differences"]
        if row["scenario"] == "constant" and row["method"] == "blend_50"
    )
    assert (
        selected["estimate"] is None
        and selected["n_pairs"] == 2
        and selected["invalid_score_pairs"] == 1
    )
    point = next(
        row
        for row in computed["weight_sensitivity"]
        if row["scenario"] == "constant" and row["method"] == "blend_50"
    )
    assert point["estimate"] is None and point["n_paths"] == 2 and point["invalid_score_runs"] == 1


@pytest.mark.parametrize("problem", ["source", "weight", "fingerprint", "missing", "seed"])
def test_incomplete_or_mislabeled_study_is_rejected(report, problem):
    if problem == "source":
        report["source_sha256"].pop("src/strategy_inference/audit.py")
    elif problem == "weight":
        report["protocol"]["method_parameters"]["blend_25"]["scale_share_weight"] = 0.75
    elif problem == "fingerprint":
        report["input_fingerprints"][0]["predicted_sha256"] = "unknown"
    elif problem == "missing":
        report["records"].pop()
    else:
        report["records"][0]["seed"] += 1
    with pytest.raises(ValueError):
        plotter.statistics(report)


def archive_for(report, tmp_path):
    path = tmp_path / "source.zip"
    content = {name: name.encode() for name in plotter.REQUIRED_SOURCES}
    content["experiments/blended-scale-protocol.json"] = json.dumps(report["protocol"]).encode()
    report["source_sha256"] = {
        name: hashlib.sha256(value).hexdigest() for name, value in content.items()
    }
    with zipfile.ZipFile(path, "w") as packed:
        for name, value in sorted(content.items()):
            packed.writestr(name, value)
    manifest = dict(
        schema_version=1,
        source_sha256=report["source_sha256"],
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (tmp_path / "source-manifest.json").write_text(json.dumps(manifest))
    return path


def test_source_archive_hash_inventory_and_protocol_are_bound(report, tmp_path):
    archive = archive_for(report, tmp_path)
    result = plotter.verify_archive(tmp_path / "results.json", report)
    assert result == dict(sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), n_files=31)
    changed = copy.deepcopy(report)
    changed["protocol"]["alpha"] = 0.2
    with pytest.raises(ValueError, match="protocol"):
        plotter.verify_archive(tmp_path / "results.json", changed)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="manifest"):
        plotter.verify_archive(tmp_path / "results.json", report)


def test_exports_have_fixed_hashes_and_preserve_input(report, tmp_path):
    pytest.importorskip("matplotlib")
    archive_for(report, tmp_path)
    source = tmp_path / "results.json"
    source.write_text(json.dumps(report, allow_nan=False))
    before = source.read_bytes()
    output = tmp_path / "figures"
    first = plotter.plot_study(source, output)
    outputs = {path.name: path.read_bytes() for path in first}
    assert len(first) == 10
    plotter.plot_study(source, output)
    assert all(path.read_bytes() == outputs[path.name] for path in first)
    assert source.read_bytes() == before
    manifest = json.loads((output / "figure-data.json").read_text())
    assert manifest["source_archive"]["n_files"] == 31
    assert manifest["statistics"] == plotter.statistics(report)[0]
    assert all("<dc:date>" not in (output / f"{stem}.svg").read_text() for stem in plotter.STEMS)


def test_managed_output_cannot_overwrite_input(report, tmp_path):
    pytest.importorskip("matplotlib")
    source = tmp_path / "figure-data.json"
    source.write_text(json.dumps(report))
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overlap"):
        plotter.plot_study(source, tmp_path)
    assert source.read_bytes() == before
