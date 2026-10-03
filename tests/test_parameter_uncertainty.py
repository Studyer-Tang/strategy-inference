"""Check design, paired cost accounting and strong-FWER labels independently."""

import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import t

from strategy_inference.parametric import fit_equicorrelated_ar1, known_phi_gls_t
from strategy_inference.uncertainty import uncertainty_test

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    sys.modules.pop("parameter_uncertainty", None)
    return importlib.import_module("parameter_uncertainty")


@pytest.fixture
def protocol():
    return json.loads((ROOT / "experiments/parameter-uncertainty-protocol.json").read_text())


@pytest.fixture
def group():
    return {"id": 3, "n_obs": 16, "k": 4, "phi_factor": .4, "phi_idio": .4,
            "rho": .3, "signed_scaled": True, "in_scope": True}


def test_formal_phases_have_separate_new_seeds(protocol):
    seeds = [value for name, profile in protocol["profiles"].items()
             for key, value in profile.items() if key.endswith("seed")]
    assert len(seeds) == len(set(seeds))
    old = json.loads((ROOT / "experiments/parametric-replay-protocol.json").read_text())
    old_seeds = {value for profile in old["profiles"].values() for key, value in profile.items() if key.endswith("seed")}
    assert not old_seeds.intersection(seeds)


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_masks_and_partial_null_labels_match_direct_gls(runner, group, protocol, phase):
    seed, replicate, delta = 7750301, 5, 3.0 if phase != 1 else 0.0
    base, sd = runner._dataset(group, seed, phase, replicate)
    data = base.copy()
    signals = np.zeros(4, dtype=bool)
    if phase == 2:
        signals[0] = True
    if phase == 3:
        signals[::2] = True
    data[:, signals] += delta * sd[signals]
    result = uncertainty_test(data)
    oracle = known_phi_gls_t(data, group["phi_factor"])
    plugin = known_phi_gls_t(data, fit_equicorrelated_ar1(data).phi)
    expected = {"gls_known": 4 * t.sf(oracle, 15) <= .05,
                "gls_known_budget": oracle > np.nextafter(math.sqrt(float(result.critical_squared)), math.inf),
                "gls_fitted": 4 * t.sf(plugin, 15) <= .05, "uncertainty": result.decisions}
    rows, certificates = runner._records(group, seed, phase, replicate, [delta], protocol)
    for row in rows:
        mask = np.unpackbits(np.frombuffer(bytes.fromhex(row["decision_mask"]), dtype=np.uint8))[:4].astype(bool)
        np.testing.assert_array_equal(mask, expected[row["method"]])
        assert row["false_reject"] == int(mask[~signals].any())
        assert row["signal_reject"] == int(mask[signals].any())
        assert row["rejection_count"] == mask.sum()
    assert certificates[0]["critical_squared"] == str(result.critical_squared)


def test_no_oracle_claim_for_heterogeneous_persistence(runner, group, protocol):
    mismatch = dict(group, phi_factor=.95, phi_idio=.2, in_scope=False)
    rows, _ = runner._records(mismatch, 7770301, 2, 0, [2.0], protocol)
    assert {row["method"] for row in rows} == {"gls_fitted", "uncertainty"}


def test_paired_loss_accounts_for_discordant_decisions(runner, protocol):
    masks = {"gls_known": [1, 1, 1, 0], "gls_known_budget": [1, 1, 0, 0], "uncertainty": [0, 1, 0, 0]}
    rows = [dict(method=method, delta=3.0, reject=value, false_reject=0, signal_reject=value,
                 ci_contains_truth=1, phi1_retained=0, empty_ci=0)
            for method, decisions in masks.items() for value in decisions]
    _, paired = runner._summaries(rows, 3, 2, protocol)
    oracle = next(row for row in paired if row["method"] == "gls_known" and row["mode"] == "reject")
    assert oracle["method_only"] == 2 and oracle["reference_only"] == 0
    assert oracle["risk_difference"] == .5
    assert oracle["mcnemar_p"] == .5


def test_existing_evidence_is_not_overwritten(runner, tmp_path):
    existing = tmp_path / "evidence"
    existing.mkdir()
    saved = existing / "retained.txt"
    saved.write_text("frozen")
    with pytest.raises(ValueError, match="never overwritten"):
        runner.run("quick", existing)
    assert saved.read_text() == "frozen"


def test_small_run_audits_all_phases_and_binds_metadata(runner, protocol, tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    files = ("scripts/parameter_uncertainty.py", "scripts/parametric_replay.py",
             "scripts/verify_parameter_uncertainty.py", *(
                 f"src/strategy_inference/{name}" for name in
                 ("uncertainty.py", "parametric.py", "reference.py", "_validation.py", "experiments.py", "inference.py")))
    for name in files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    protocol["profiles"]["quick"] = dict(null_replicates=3, power_replicates=3, partial_replicates=3,
                                         null_seed=9940301, power_seed=9940302, partial_seed=9940303)
    protocol["groups"] = [dict(group, n_obs=16, k=4) for group in protocol["groups"] if group["id"] in (3, 8)]
    protocol["power"]["groups"] = [3, 8]
    protocol["power"]["standardized_mean_shifts"] = [1.0, 3.0]
    protocol["partial_null"]["groups"] = [3]
    saved_protocol = snapshot / "experiments/parameter-uncertainty-protocol.json"
    saved_protocol.parent.mkdir()
    saved_protocol.write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "ROOT", snapshot)
    monkeypatch.setattr(runner, "__file__", str(snapshot / "scripts/parameter_uncertainty.py"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="c" * 40))
    output = tmp_path / "evidence"
    runner.run("quick", output)
    sys.modules.pop("verify_parameter_uncertainty", None)
    auditor = importlib.import_module("verify_parameter_uncertainty")
    monkeypatch.setattr(auditor, "ROOT", snapshot)
    result = auditor.verify(output)
    assert result["status"] == "passed" and result["records_checked"] == 66
    assert result["metadata_sha256"] == runner._sha(output / "metadata.json")
    metadata = json.loads((output / "metadata.json").read_text())
    metadata["cells"].pop()
    (output / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Missing or duplicated"):
        auditor.verify(output)
