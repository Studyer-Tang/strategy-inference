"""Compact CLI exports remain reproducible, parseable and safe for inputs."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from strategy_inference.cli import main


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "returns.csv"
    np.savetxt(path, np.random.default_rng(382).normal(size=(48, 2)),
        delimiter=",", header="value,carry", comments="")
    return path


@pytest.mark.parametrize("method", ["bootstrap", "gaussian_ar"])
def test_json_stdout_is_parseable_without_extra_progress_or_warning_text(source, capsys, method):
    options = ["--n-resamples", "37", "--seed", "12"] if method == "bootstrap" else []
    assert main(["test", str(source), "--method", method, *options]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == ""
    assert result["method"] == method
    assert result["provenance"]["seed"] == (12 if method == "bootstrap" else None)
    assert [row["name"] for row in result["candidates"]] == ["value", "carry"]
    if method == "gaussian_ar":
        assert result["global_pvalue"] is None
        assert all("adjusted_pvalue" not in row for row in result["candidates"])
    else:
        assert result["diagnostics"]["studentization"] == "resampled"
        assert result["diagnostics"]["warnings"]
        assert len(result["diagnostics"]["conditional_bootstrap_tail_interval_95"]) == 2


def test_csv_suffix_export_preserves_names_and_input(tmp_path, source, capsys):
    before = source.read_bytes()
    output = tmp_path / "nested" / "decisions.csv"
    assert main(["test", str(source), "--output", str(output), "--n-resamples", "37"]) == 0
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["name"] for row in rows] == ["value", "carry"]
    assert all(0 < float(row["adjusted_pvalue"]) <= 1 for row in rows)
    assert source.read_bytes() == before
    assert "Result:" in capsys.readouterr().out


@pytest.mark.parametrize("alias", ["same_path", "symlink", "hardlink"])
def test_input_and_inode_aliases_cannot_be_overwritten(tmp_path, source, capsys, alias):
    output = source if alias == "same_path" else tmp_path / "alias.json"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        output.hardlink_to(source)
    before = source.read_bytes()
    assert main(["test", str(source), "--output", str(output), "--n-resamples", "3"]) == 2
    assert source.read_bytes() == before
    assert "must differ" in capsys.readouterr().err


def test_cli_wrong_method_options_leave_no_result(tmp_path, source, capsys):
    output = tmp_path / "missing" / "result.json"
    assert main(["test", str(source), "--method", "gaussian_ar", "--seed", "12", "--output", str(output)]) == 2
    assert not output.exists() and not output.parent.exists()
    assert "Unsupported options" in capsys.readouterr().err


def test_version_and_help_do_not_import_numerical_dependencies():
    import subprocess
    import sys

    code = """
import sys
from strategy_inference.cli import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'numpy' not in sys.modules and 'scipy' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, stdout=subprocess.DEVNULL, cwd=Path.cwd())
