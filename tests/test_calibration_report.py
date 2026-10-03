import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from strategy_inference import calibration, cli, experiments
from strategy_inference.calibration_report import write_calibration_report
from strategy_inference.experiments import _wilson

FIGURE_STEMS = (
    "figure-1-calibration-dependence",
    "figure-2-calibration-selection",
    "figure-3-calibration-processes",
)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a" and "href" in attributes:
            self.targets.append(attributes["href"])
        if tag == "img":
            self.targets.append(attributes["src"])


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _saved_run(output, assessment_status="smoke_only", *, alpha=0.05, threshold=0.07):
    """Small saved tables with the actual frozen schema and exact count records."""
    output.mkdir(parents=True)
    protocol, _raw, _path = calibration._protocol()
    protocol["alpha"] = alpha
    protocol["assessment"]["maximum_false_positive_rate"] = threshold
    profile = "quick" if assessment_status == "smoke_only" else "full"
    config = {
        **protocol["profiles"][profile],
        "alpha": alpha,
        "confidence": protocol["confidence"],
        "cross_corr": protocol["cross_corr"],
        "sigma": protocol["sigma"],
        "burnin": protocol["burnin"],
        "n_assessment_cells": 19,
    }

    def row(stage, process, phi, strategies, method, *, delta=0, n_obs=None):
        total = config["supplement_mc"] if stage in ("holdout", "sensitivity") else config["n_mc"]
        rate = {"fixed": 0.10, "resampled": 0.04, "oracle": 0.05, "iid": 0.25, "hac": 0.20}[method]
        if (
            assessment_status == "failed"
            and stage == "dependence"
            and phi == 0.8
            and method == "resampled"
        ):
            rate = 0.10
        count = round(total * rate)
        rate, low, high = _wilson(count, total, config["confidence"])
        assessed = method == "resampled" and delta == 0 and stage != "sensitivity"
        return dict(
            stage=stage,
            process=process,
            n_obs=n_obs or config["n_obs"],
            n_strategies=strategies,
            phi=phi,
            cross_corr=config["cross_corr"],
            delta=delta,
            method=method,
            hac_lags=5,
            block_length=13,
            n_mc=total,
            n_bootstrap=config["n_bootstrap"],
            reject_count=count,
            rate=rate,
            ci_low=low,
            ci_high=high,
            alpha=alpha,
            size_upper_simultaneous=calibration._upper(
                count, total, family_alpha=1 - config["confidence"], n_cells=19
            )
            if assessed
            else "",
        )

    gaussian_methods = ("fixed", "resampled", "oracle", "iid", "hac")
    tables = {
        "figure-1-dependence.csv": [
            row("dependence", "gaussian_ar", phi, 1, method)
            for phi in protocol["figure_1"]["phi"]
            for method in gaussian_methods
        ],
        "figure-2-selection.csv": [
            row("selection", "gaussian_ar", 0.5, strategies, method)
            for strategies in protocol["figure_2"]["strategy_counts"]
            for method in gaussian_methods
        ],
        "figure-3-size.csv": [
            row("process", scenario["process"], scenario["phi"], 20, method)
            for scenario in protocol["figure_3"]["processes"]
            for method in gaussian_methods
            if method != "oracle" or scenario["process"] == "gaussian_ar"
        ],
        "figure-3-power.csv": [
            row("process", scenario["process"], scenario["phi"], 20, method, delta=delta)
            for scenario in protocol["figure_3"]["processes"]
            for delta in protocol["figure_3"]["delta"]
            for method in ("fixed", "resampled", "oracle")
            if method != "oracle" or scenario["process"] == "gaussian_ar"
        ],
        "holdout.csv": [
            row(
                "holdout", scenario["process"], scenario["phi"], 10, method, n_obs=scenario["n_obs"]
            )
            for scenario in protocol["holdout"]
            for method in gaussian_methods
            if method != "oracle" or scenario["process"] == "gaussian_ar"
        ],
        "block-sensitivity.csv": [row("sensitivity", "gaussian_ar", 0.8, 1, "resampled")],
        "variance-diagnostics.csv": [
            dict(stage="dependence", process="gaussian_ar", true_mean_variance=0.001)
        ],
    }
    for name, records in tables.items():
        _write_rows(output / name, records)
    failed = [
        record
        for name, records in tables.items()
        if name
        in ("figure-1-dependence.csv", "figure-2-selection.csv", "figure-3-size.csv", "holdout.csv")
        for record in records
        if record.get("size_upper_simultaneous", "") != ""
        and record["size_upper_simultaneous"] > threshold
    ]
    metadata = dict(
        study="calibration",
        status="complete",
        profile=profile,
        seed=20261004,
        protocol=protocol,
        resolved_protocol=config,
        elapsed_seconds=0.25,
        git_revision="abc123",
        outputs=list(tables),
        assessment=dict(
            status=assessment_status,
            n_cells=19,
            maximum_false_positive_rate=threshold,
            failed_cells=failed,
        ),
        file_sha256={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in tables
        },
    )
    _save_metadata(output, metadata)
    for stem in FIGURE_STEMS:
        for suffix in ("png", "pdf", "svg"):
            (output / f"{stem}.{suffix}").write_bytes(b"figure fixture")
    return metadata


def _save_metadata(output, metadata):
    (output / "run-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _change_table(output, name, change):
    with (output / name).open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    change(records)
    _write_rows(output / name, records)
    metadata = json.loads((output / "run-metadata.json").read_text(encoding="utf-8"))
    metadata["file_sha256"][name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    _save_metadata(output, metadata)


@pytest.mark.parametrize(
    "status, expected", [("smoke_only", "流程检查"), ("passed", "通过"), ("failed", "未通过")]
)
def test_report_distinguishes_smoke_pass_and_failure_without_general_calibration_claim(
    tmp_path, status, expected
):
    output = tmp_path / status
    _saved_run(output, status)
    path = write_calibration_report(output)
    report = path.read_text(encoding="utf-8")
    assert path == output / "report.html"
    assert expected in report
    if status == "smoke_only":
        assert "不足以作校准判断" in report
        assert "这次评价通过" not in report
    elif status == "passed":
        assert "不等于" in report and "真实金融数据" in report
    else:
        assert "1 / 19" in report
    assert "重尾与 GARCH 不套用高斯最大值分布" in report
    assert "不是正确识别信号列的概率" in report
    links = _Links()
    links.feed(report)
    local = [link for link in links.targets if "://" not in link and not link.startswith("#")]
    assert len([link for link in local if link.endswith(".png")]) == 3
    for link in local:
        assert (output / link).is_file(), f"Broken local report link: {link}"


def test_report_values_come_from_saved_counts_and_intervals(tmp_path):
    output = tmp_path / "saved"
    _saved_run(output, "passed")
    count, total = 246, 2000
    rate, low, high = _wilson(count, total, 0.95)

    def change(records):
        target = next(
            row for row in records if row["method"] == "fixed" and float(row["phi"]) == 0.8
        )
        target.update(reject_count=count, rate=rate, ci_low=low, ci_high=high)

    _change_table(output, "figure-1-dependence.csv", change)
    report = write_calibration_report(output).read_text(encoding="utf-8")
    assert f"{rate:.2%}" in report
    assert f"[{low:.2%}, {high:.2%}]" in report


def test_report_nominal_level_and_assessment_limit_follow_metadata(tmp_path):
    output = tmp_path / "levels"
    _saved_run(output, "passed", alpha=0.1, threshold=0.08)
    report = write_calibration_report(output).read_text(encoding="utf-8")
    assert "名义水平 10%" in report
    assert "不超过 8%" in report
    assert "不超过 7%" not in report


def test_csv_labels_and_git_revision_are_html_escaped(tmp_path):
    output = tmp_path / "escaped"
    metadata = _saved_run(output)
    label = '<img src=x onerror="alert(1)">'
    metadata["git_revision"] = label
    _save_metadata(output, metadata)
    _change_table(
        output,
        "holdout.csv",
        lambda rows: next(row for row in rows if row["method"] == "resampled").update(
            process=label
        ),
    )
    report = write_calibration_report(output).read_text(encoding="utf-8")
    assert label not in report
    assert "&lt;img src=x" in report


@pytest.mark.parametrize("status", ["running", "failed", "cancelled"])
def test_report_rejects_unfinished_run(tmp_path, status):
    output = tmp_path / "unfinished"
    metadata = _saved_run(output)
    metadata["status"] = status
    _save_metadata(output, metadata)
    with pytest.raises(ValueError):
        write_calibration_report(output)


@pytest.mark.parametrize(
    "change",
    [
        lambda record: record.update(study="baseline"),
        lambda record: record["assessment"].update(status="unrecognized"),
        lambda record: record["assessment"].update(status="passed"),
    ],
)
def test_report_rejects_wrong_study_or_inconsistent_assessment(tmp_path, change):
    output = tmp_path / "wrong"
    metadata = _saved_run(output)
    change(metadata)
    _save_metadata(output, metadata)
    with pytest.raises(ValueError):
        write_calibration_report(output)


def test_report_rejects_gaussian_oracle_row_in_heavy_tailed_process(tmp_path):
    output = tmp_path / "mixed"
    _saved_run(output)

    def change(rows):
        row = dict(next(row for row in rows if row["process"] == "student_ar"))
        row["method"] = "oracle"
        rows.append(row)

    _change_table(output, "figure-3-size.csv", change)
    with pytest.raises(ValueError):
        write_calibration_report(output)


def test_report_rejects_modified_csv_without_matching_recorded_hash(tmp_path):
    output = tmp_path / "modified"
    _saved_run(output)
    with (output / "holdout.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError):
        write_calibration_report(output)


def test_report_requires_hash_record_for_each_consumed_csv(tmp_path):
    output = tmp_path / "missing_hash"
    metadata = _saved_run(output)
    del metadata["file_sha256"]["holdout.csv"]
    _save_metadata(output, metadata)
    with pytest.raises(ValueError):
        write_calibration_report(output)


@pytest.mark.parametrize("name", ["protocol.json", "calibration-protocol.json"])
def test_packaged_protocol_copy_matches_canonical_bytes(name):
    package = Path(calibration.__file__).resolve().parent
    canonical = package.parents[1] / "experiments" / name
    assert (package / "protocols" / name).read_bytes() == canonical.read_bytes()


@pytest.mark.parametrize(
    "module, loader, name",
    [
        (experiments, "_load_protocol", "protocol.json"),
        (calibration, "_protocol", "calibration-protocol.json"),
    ],
)
def test_installed_protocol_loader_does_not_require_source_checkout_or_matching_prefix(
    tmp_path, monkeypatch, module, loader, name
):
    raw = (Path(module.__file__).resolve().parent / "protocols" / name).read_bytes()
    package = tmp_path / "target" / "lib" / "strategy_inference"
    (package / "protocols").mkdir(parents=True)
    resource = package / "protocols" / name
    resource.write_bytes(raw)
    monkeypatch.setattr(module, "__file__", str(package / Path(module.__file__).name))
    monkeypatch.setattr(module.sys, "prefix", str(tmp_path / "unrelated-prefix"))
    protocol, observed, path = getattr(module, loader)()
    assert observed == raw
    assert protocol == json.loads(raw)
    assert path == resource


def _stub_reproduction(monkeypatch):
    plotting = pytest.importorskip("strategy_inference.calibration_plotting")
    calls = []

    def baseline(output, *, profile, seed):
        calls.append(("baseline", Path(output), profile, seed))
        Path(output).mkdir(parents=True)
        return dict(status="complete", elapsed_seconds=0.1)

    def baseline_report(output):
        path = Path(output) / "report.html"
        path.write_text("baseline report", encoding="utf-8")
        return path

    def calibrated(output, *, profile, seed):
        calls.append(("calibration", Path(output), profile, seed))
        record = _saved_run(Path(output), "smoke_only" if profile == "quick" else "passed")
        if seed is not None:
            record["seed"] = seed
            _save_metadata(Path(output), record)
        return record

    def saved_figures(output, _metadata):
        return [
            Path(output) / f"{stem}.{suffix}"
            for stem in FIGURE_STEMS
            for suffix in ("png", "pdf", "svg")
        ]

    monkeypatch.setattr(experiments, "run_experiments", baseline)
    monkeypatch.setattr("strategy_inference.report.write_experiment_report", baseline_report)
    monkeypatch.setattr(calibration, "run_calibration", calibrated)
    monkeypatch.setattr(plotting, "plot_calibration", saved_figures)
    return calls


def test_cli_retains_fixed_audit_and_baseline_reproduction_defaults():
    audit = cli.parser().parse_args(["audit", "input.csv"])
    reproduce = cli.parser().parse_args(["reproduce"])
    assert audit.studentization == "fixed"
    assert reproduce.study == "baseline"
    assert reproduce.seed is None
    assert reproduce.profile == "full"
    assert reproduce.output is None


@pytest.mark.parametrize("study", [None, "baseline", "calibration"])
def test_cli_study_dispatch_uses_separate_default_output_and_seed(
    tmp_path, monkeypatch, capsys, study
):
    monkeypatch.chdir(tmp_path)
    calls = _stub_reproduction(monkeypatch)
    arguments = ["reproduce"]
    if study is not None:
        arguments += ["--study", study]
    assert cli.main(arguments) == 0
    expected_study = study or "baseline"
    expected_output = Path("results/calibration/full" if study == "calibration" else "results/full")
    assert calls == [
        (expected_study, expected_output, "full", None if study == "calibration" else 20261002)
    ]
    assert (expected_output / "report.html").is_file()
    if study == "calibration":
        record = json.loads((expected_output / "run-metadata.json").read_text(encoding="utf-8"))
        assert record["seed"] == calibration._protocol()[0]["seed"]
        assert "report.html" in record["outputs"]
        for name in record["outputs"]:
            assert (
                record["file_sha256"][name]
                == hashlib.sha256((expected_output / name).read_bytes()).hexdigest()
            )
    assert "Status: complete" in capsys.readouterr().out


@pytest.mark.parametrize("study", ["baseline", "calibration"])
def test_cli_explicit_profile_output_and_seed_override_defaults(
    tmp_path, monkeypatch, capsys, study
):
    monkeypatch.chdir(tmp_path)
    calls = _stub_reproduction(monkeypatch)
    output = tmp_path / "custom"
    assert (
        cli.main(
            [
                "reproduce",
                "--study",
                study,
                "--profile",
                "quick",
                "--output",
                str(output),
                "--seed",
                "87",
            ]
        )
        == 0
    )
    assert calls == [(study, output, "quick", 87)]
    assert (output / "report.html").is_file()
    if study == "calibration":
        record = json.loads((output / "run-metadata.json").read_text(encoding="utf-8"))
        assert record["seed"] == 87
        assert record["assessment"]["status"] == "smoke_only"
    capsys.readouterr()


def test_cli_unknown_study_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["reproduce", "--study", "unrecognized"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_calibration_cli_reports_missing_figure_dependency_as_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _stub_reproduction(monkeypatch)
    from strategy_inference import calibration_plotting

    def missing_dependency(*_args):
        raise ImportError("matplotlib is unavailable")

    monkeypatch.setattr(calibration_plotting, "plot_calibration", missing_dependency)
    assert cli.main(["reproduce", "--study", "calibration", "--profile", "quick"]) == 2
    assert not (tmp_path / "results/calibration/quick/report.html").exists()
    captured = capsys.readouterr()
    assert "Status: complete" not in captured.out
    assert "strategy-inference[figures]" in captured.err
