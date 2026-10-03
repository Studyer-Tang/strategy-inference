"""Public exports preserve the underlying statistical results and input labels."""

import builtins
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from strategy_inference import audit_returns, read_returns_csv, wilks_uncertainty_test
from strategy_inference import test_returns as run_test


@pytest.fixture
def data():
    return np.random.default_rng(951).normal(size=(96, 3)) + [0.2, 0, -0.2]


def test_bootstrap_wrapper_preserves_draws_decisions_and_default_scale(data):
    actual = run_test(data, n_resamples=73, seed=42)
    expected = audit_returns(data, studentization="resampled", n_resamples=73, seed=42)
    np.testing.assert_array_equal(actual.details.bootstrap_statistics, expected.bootstrap_statistics)
    np.testing.assert_array_equal(actual.decisions, expected.adjusted_pvalue <= expected.alpha)
    np.testing.assert_array_equal(actual.adjusted_pvalue, expected.adjusted_pvalue)
    assert actual.global_pvalue == expected.global_pvalue
    assert actual.global_reject == bool(actual.decisions.any())
    assert actual.n_obs == 96 and actual.n_strategies == 3
    assert actual.diagnostics["studentization"] == "resampled"
    assert actual.parameter_intervals is None


def test_gaussian_wrapper_preserves_exact_certificate_and_has_no_pvalues(data):
    actual = run_test(data, method="gaussian_ar")
    expected = wilks_uncertainty_test(data)
    np.testing.assert_array_equal(actual.decisions, expected.decisions)
    assert actual.details.scales == expected.scales
    assert actual.details.intervals == expected.intervals
    assert actual.parameter_intervals == expected.interval_bounds
    assert actual.adjusted_pvalue is actual.global_pvalue is None
    assert "adjusted_pvalue" not in actual.records()[0]
    assert actual.to_dict()["global_pvalue"] is None


def test_json_exports_are_compact_finite_and_detached_from_arrays(data):
    result = run_test(data, names=["alpha", "β", "third"], n_resamples=39)
    record = result.to_dict()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert [row["name"] for row in record["candidates"]] == ["alpha", "β", "third"]
    assert "bootstrap_statistics" not in str(record)
    record["candidates"][0]["mean"] = 100
    record["diagnostics"]["n_resamples"] = -1
    assert result.mean[0] != 100 and result.diagnostics["n_resamples"] == 39
    with pytest.raises(ValueError):
        result.decisions[0] = False
    with pytest.raises(ValueError):
        result.mean[0] = 0
    with pytest.raises(TypeError):
        result.diagnostics["n_resamples"] = -1


def test_numeric_dataframe_and_nullable_dtypes_preserve_names(data):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(data, columns=["carry", "value", "trend"])
    numeric = run_test(frame, n_resamples=37, seed=83)
    nullable = run_test(frame.astype("Float64"), n_resamples=37, seed=83)
    np.testing.assert_array_equal(numeric.adjusted_pvalue, nullable.adjusted_pvalue)
    table = numeric.to_frame()
    assert table.index.tolist() == frame.columns.tolist()
    assert table.index.name == "strategy"
    assert table["reject"].tolist() == numeric.decisions.tolist()
    assert table["adjusted_pvalue"].tolist() == numeric.adjusted_pvalue.tolist()
    assert run_test(frame, names=["a", "b", "c"], n_resamples=3).names == ("a", "b", "c")


def test_dataframe_missing_values_and_duplicate_labels_are_not_removed(data):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(data, columns=["a", "b", "c"]).astype("Float64")
    frame.loc[0, "a"] = pd.NA
    with pytest.raises(ValueError, match="missing or non-finite"):
        run_test(frame, n_resamples=3)
    with pytest.raises(ValueError, match="distinct"):
        run_test(pd.DataFrame(data, columns=[1, "1", "c"]), n_resamples=3)


def test_csv_table_and_single_series_have_consistent_output(tmp_path, data):
    path = tmp_path / "returns.csv"
    np.savetxt(path, data, delimiter=",", header="a,b,c", comments="")
    result = run_test(read_returns_csv(path), n_resamples=31, seed=1)
    assert result.names == ("a", "b", "c")
    assert result.details.names == result.names
    single = run_test(data[:, 0], n_resamples=31, seed=1)
    assert single.names == ("strategy_1",) and single.mean.shape == (1,)


def test_to_frame_gives_an_actionable_optional_dependency_error(monkeypatch, data):
    result = run_test(data, n_resamples=3)
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("Unavailable in this environment")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(ImportError, match="requires pandas"):
        result.to_frame()


@pytest.mark.parametrize("names", ["abc", ["a", "a", "c"], ["a", "b"], [1, 2, 3], 5])
def test_invalid_labels_fail_before_computation(data, names):
    with pytest.raises(ValueError, match="names"):
        run_test(data, names=names, n_resamples=3)


@pytest.mark.parametrize("method,options", [
    ("unknown", {}), (None, {}), ([], {}),
    ("bootstrap", {"beta": 0.005}), ("gaussian_ar", {"seed": 1}),
    ("gaussian_ar", {"n_resamples": 99}), ("bootstrap", {"n_resample": 99}),
])
def test_wrong_method_and_inapplicable_options_are_rejected(data, method, options):
    with pytest.raises((ValueError, TypeError)):
        run_test(data, method=method, **options)


def test_package_import_is_lazy_and_public_exports_remain_available():
    code = """
import sys
import strategy_inference as si
assert 'numpy' not in sys.modules and 'scipy' not in sys.modules
assert 'test_returns' in dir(si)
for name in si.__all__:
    assert getattr(si, name) is getattr(si, name)
try:
    si.missing_interface
except AttributeError:
    pass
else:
    raise AssertionError('Missing attribute did not fail')
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
