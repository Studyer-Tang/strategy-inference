"""Causal forecast fitting and matched training scales for the new study."""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "scale_transfer_study", ROOT / "scripts/reproduce_scale_transfer.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
PROTOCOL = json.loads((ROOT / "experiments/scale-transfer-protocol.json").read_text())


def test_rolling_forecasts_match_independent_least_squares():
    y = np.random.default_rng(18).normal(size=800).cumsum()
    origins, predicted = study.forecasts(y, PROTOCOL, PROTOCOL["scenarios"][0])
    for row in (0, 19, 255, len(origins) - 1):
        origin = origins[row]
        train = y[max(0, origin - PROTOCOL["forecaster"]["window"] + 1) : origin + 1]
        design = np.column_stack((np.ones(len(train) - 1), train[:-1]))
        intercept, phi = np.linalg.lstsq(design, train[1:], rcond=None)[0]
        phi = np.clip(phi, -0.995, 0.995)
        # Re-estimate intercept after the specified stability projection.
        intercept = train[1:].mean() - phi * train[:-1].mean()
        expected, current = [], y[origin]
        for h in range(1, max(PROTOCOL["lead_times"]) + 1):
            current = intercept + phi * current
            if h in PROTOCOL["lead_times"]:
                expected.append(current)
        np.testing.assert_allclose(predicted[row], expected, rtol=1e-10, atol=1e-10)


def test_future_labels_do_not_change_past_forecasts_or_initial_scales():
    y = np.random.default_rng(19).normal(size=900).cumsum()
    altered = y.copy()
    altered[384:] += np.random.default_rng(20).normal(size=len(y) - 384) * 20
    origins, points = study.forecasts(y, PROTOCOL, PROTOCOL["scenarios"][0])
    origins2, points2 = study.forecasts(altered, PROTOCOL, PROTOCOL["scenarios"][0])
    np.testing.assert_array_equal(origins, origins2)
    np.testing.assert_array_equal(points[origins < 384], points2[origins < 384])
    a = study.initial_scales(y, origins, points, PROTOCOL["lead_times"], 384)
    b = study.initial_scales(altered, origins2, points2, PROTOCOL["lead_times"], 384)
    np.testing.assert_array_equal(a, b)


def test_initial_scales_use_identical_mature_training_targets():
    y = np.random.default_rng(21).normal(size=900)
    origins, points = study.forecasts(y, PROTOCOL, PROTOCOL["scenarios"][0])
    scales = study.initial_scales(y, origins, points, PROTOCOL["lead_times"], 384)
    for h, scale in zip(PROTOCOL["lead_times"], scales, strict=True):
        errors = [
            y[target] - points[target - h - origins[0], PROTOCOL["lead_times"].index(h)]
            for target in range(origins[0] + max(PROTOCOL["lead_times"]), 384)
        ]
        assert np.isclose(scale, np.sqrt(np.mean(np.square(errors))))


def test_deliberate_bias_leaves_shortest_forecasts_unchanged():
    y = np.random.default_rng(22).normal(size=900)
    origins, plain = study.forecasts(y, PROTOCOL, PROTOCOL["scenarios"][0])
    _, biased = study.forecasts(y, PROTOCOL, PROTOCOL["scenarios"][-1])
    np.testing.assert_array_equal(plain[:, 0], biased[:, 0])
    middle = (origins >= int(len(y) * 0.45)) & (origins < int(len(y) * 0.75))
    np.testing.assert_allclose(biased[middle, -1] - plain[middle, -1], 2.5)
    np.testing.assert_array_equal(plain[~middle], biased[~middle])
