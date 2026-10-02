import csv
import hashlib
import json

import numpy as np
import pytest

from strategy_inference.cli import main
from strategy_inference.io import read_returns_csv


def _write_csv(path, names, values, dates=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((["date"] if dates is not None else []) + list(names))
        for index, row in enumerate(values):
            writer.writerow(([dates[index]] if dates is not None else []) + list(row))
    return path


def test_csv_preserves_observations_and_subtracts_benchmark(tmp_path):
    rng = np.random.default_rng(73)
    excess = rng.normal(size=(12, 2))
    benchmark = np.linspace(-0.2, 0.3, 12)
    values = np.column_stack([excess[:, 0] + benchmark, benchmark, excess[:, 1] + benchmark])
    dates = [f"2026-01-{day:02d}" for day in [1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16]]
    path = _write_csv(tmp_path / "returns.csv", ["first", "benchmark", "second"], values, dates)
    original = path.read_bytes()

    table = read_returns_csv(path, benchmark="benchmark")
    assert table.names == ("first", "second")
    assert table.dates == tuple(dates)
    np.testing.assert_allclose(table.values, excess)
    assert path.read_bytes() == original


def test_csv_without_dates_and_with_utf8_bom(tmp_path):
    path = tmp_path / "returns.csv"
    path.write_text(
        "\ufeff  alpha  , beta\n" + "\n".join(f"{i},{i * i}" for i in range(8)), encoding="utf-8"
    )
    table = read_returns_csv(path)
    assert table.names == ("alpha", "beta")
    assert table.dates is None
    np.testing.assert_array_equal(table.values[:, 0], np.arange(8))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "x,x\n1,2\n",
        "x,\n1,2\n",
        "date\n2026-01-01\n",
        "x,y\n1,2\n3\n",
        "x,y\n1,2\n3,4,5\n",
        "x\n1\nnot-a-return\n",
        "x,y\n1,2\n3,\n",
        "x\n0\n1\n2\n",
        "x\n0\n1\n\n2\n",
        "x\n" + "\n".join(["0", "1", "2", "3", "4", "5", "6", "nan"]),
        "x\n" + "\n".join(["0", "1", "2", "3", "4", "5", "6", "inf"]),
    ],
)
def test_malformed_csv_is_rejected_without_deleting_rows(tmp_path, text):
    path = tmp_path / "invalid.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_returns_csv(path)
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize(
    "dates",
    [
        ["2026-01-01", "2026-01-01"],
        ["2026-01-02", "2026-01-01"],
        ["2026-01-01", "2026-13-01"],
        ["2026-01-01", "2026-01-02T00:00:00+00:00"],
    ],
)
def test_dates_must_be_valid_consistent_and_strictly_increasing(tmp_path, dates):
    path = _write_csv(tmp_path / "dates.csv", ["x"], np.array([[1], [2]]), dates)
    with pytest.raises(ValueError):
        read_returns_csv(path)


@pytest.mark.parametrize("benchmark", ["missing", "only"])
def test_benchmark_needs_another_numeric_candidate(tmp_path, benchmark):
    path = _write_csv(tmp_path / "single.csv", ["only"], np.arange(8)[:, None])
    with pytest.raises(ValueError):
        read_returns_csv(path, benchmark=benchmark)


@pytest.mark.parametrize(
    "flag, expected", [(None, None), ("--complete-search", True), ("--incomplete-search", False)]
)
def test_cli_records_search_scope_hash_and_finite_numbers(tmp_path, capsys, flag, expected):
    values = np.random.default_rng(19).normal(size=(32, 2))
    values -= values.mean(axis=0)
    name = '<script>alert("candidate")</script>'
    path = _write_csv(tmp_path / "returns.csv", [name, "other"], values)
    output = tmp_path / "report"
    arguments = ["audit", str(path), "--output", str(output), "--n-resamples", "99", "--seed", "21"]
    if flag is not None:
        arguments.append(flag)

    assert main(arguments) == 0
    record = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert record["search_complete"] is expected
    assert record["provenance"]["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["provenance"]["date_column_present"] is False
    assert record["provenance"]["seed"] == 21
    assert record["global_pvalue"] > record["alpha"]
    json.dumps(record, allow_nan=False)

    report = (output / "audit.html").read_text(encoding="utf-8")
    assert name not in report
    assert "&lt;script&gt;" in report
    assert "尚不能拒绝全族原假设" in report
    assert "固定当前数据时" in report
    assert "只反映内层模拟误差" in report
    assert "不衡量方法假设是否成立" in report
    captured = capsys.readouterr()
    assert "family p" in captured.out
    assert ("supplied columns only" in captured.err) is (expected is not True)


def test_cli_benchmark_is_reflected_in_record(tmp_path, capsys):
    rng = np.random.default_rng(617)
    values = rng.normal(size=(32, 2))
    path = _write_csv(tmp_path / "returns.csv", ["strategy", "cash"], values)
    output = tmp_path / "report"
    assert (
        main(
            [
                "audit",
                str(path),
                "--benchmark",
                "cash",
                "--output",
                str(output),
                "--n-resamples",
                "99",
            ]
        )
        == 0
    )
    record = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert record["n_strategies"] == 1
    assert record["provenance"]["benchmark_column"] == "cash"
    assert record["candidates"][0]["mean"] == pytest.approx(np.mean(values[:, 0] - values[:, 1]))
    capsys.readouterr()


@pytest.mark.parametrize("bad_input", ["missing", "nonnumeric", "bad_resamples"])
def test_cli_errors_return_two_without_a_success_report(tmp_path, capsys, bad_input):
    path = tmp_path / "returns.csv"
    arguments = ["audit", str(path), "--output", str(tmp_path / "report"), "--n-resamples", "99"]
    if bad_input == "nonnumeric":
        path.write_text("x\n1\nnot-a-number\n", encoding="utf-8")
    elif bad_input == "bad_resamples":
        _write_csv(path, ["x"], np.arange(16)[:, None])
        arguments[-1] = "0"
    assert main(arguments) == 2
    assert not (tmp_path / "report" / "audit.html").exists()
    assert "Error:" in capsys.readouterr().err


def test_cli_rejects_unknown_option(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["audit", str(tmp_path / "returns.csv"), "--claim-complete-search"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
