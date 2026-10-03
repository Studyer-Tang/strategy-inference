"""Numerical research helpers do not require optional figure dependencies."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCK_FIGURES = """
import importlib.abc
import sys

class NoMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise ModuleNotFoundError("matplotlib deliberately unavailable")

sys.meta_path.insert(0, NoMatplotlib())
"""


def _without_figures(code, *arguments):
    result = subprocess.run(
        [sys.executable, "-B", "-c", BLOCK_FIGURES + code, *map(str, arguments)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_helpers_calibration_and_formula_checks_import_without_matplotlib():
    _without_figures("""
import runpy
from strategy_inference import calibration, experiments

rate, low, high = experiments._wilson(2, 10, 0.95)
assert rate == 0.2 and 0 < low < rate < high < 1
assert experiments._bootstrap_pvalue(2, experiments.np.array([1, 2, 3])) == 0.75
assert experiments._stream_seed(17, 1, 0, 0, 0) == experiments._stream_seed(17, 1, 0, 0, 0)
assert "python" in experiments._versions()
assert callable(calibration.run_calibration)
sys.path.insert(0, "scripts")
for name in ("verify_joint_uncertainty.py", "verify_parameter_uncertainty.py"):
    runpy.run_path("scripts/" + name, run_name="import_smoke")
assert not any(name == "matplotlib" or name.startswith("matplotlib.") for name in sys.modules)
assert "strategy_inference.plotting" not in sys.modules
assert "strategy_inference.calibration_plotting" not in sys.modules
""")


def test_experiments_checks_plot_dependency_before_simulating_or_writing(tmp_path):
    output = tmp_path / "uncreated"
    _without_figures("""
from pathlib import Path
from strategy_inference import experiments

def forbidden(*args, **kwargs):
    raise AssertionError("Simulation started before checking plotting dependencies")

experiments._dependence = forbidden
try:
    experiments.run_experiments(sys.argv[1], profile="quick", seed=17)
except ModuleNotFoundError as exc:
    assert "matplotlib" in str(exc)
else:
    raise AssertionError("A figure-producing run must require matplotlib")
assert not Path(sys.argv[1]).exists()
""", output)
    assert not output.exists()


@pytest.mark.parametrize("study", ["baseline", "calibration"])
def test_reproduce_cli_preserves_optional_dependency_error_without_output(tmp_path, study):
    output = tmp_path / study
    result = _without_figures("""
from pathlib import Path
from strategy_inference.cli import main

assert main(["reproduce", "--study", sys.argv[1], "--profile", "quick",
             "--output", sys.argv[2]]) == 2
assert not Path(sys.argv[2]).exists()
""", study, output)
    assert "Missing optional dependency:" in result.stderr
    assert "strategy-inference[figures]" in result.stderr
    assert not output.exists()


def test_packaged_protocols_are_exactly_the_two_installed_cli_resources():
    path = ROOT / "scripts/sync_protocols.py"
    spec = importlib.util.spec_from_file_location("sync_protocols_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {"protocol.json", "calibration-protocol.json"}
    resources = ROOT / "src/strategy_inference/protocols"
    assert set(module.PACKAGED_PROTOCOLS) == expected
    assert {path.name for path in resources.glob("*.json")} == expected
    for name in expected:
        assert (resources / name).read_bytes() == (ROOT / "experiments" / name).read_bytes()
