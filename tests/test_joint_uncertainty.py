"""Check new seeds, scientific targets and integrity of the paired pipeline."""

import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("joint_uncertainty")


@pytest.fixture
def protocol():
    value = json.loads((ROOT / "experiments/joint-uncertainty-protocol.json").read_text())
    value["procedure"]["block_lengths"] = [4]
    return value


def test_fresh_seeds_and_scope():
    protocol = json.loads((ROOT / "experiments/joint-uncertainty-protocol.json").read_text())
    seeds = {value for profile in protocol["profiles"].values() for key, value in profile.items() if key.endswith("seed")}
    assert len(seeds) == 6
    for path in (ROOT / "experiments").glob("*.json"):
        if path.name == "joint-uncertainty-protocol.json":
            continue
        old = json.loads(path.read_text())
        previous = {value for profile in old.get("profiles", {}).values() if isinstance(profile, dict)
                    for key, value in profile.items() if key.endswith("seed")}
        assert not seeds.intersection(previous)
    assert protocol["groups"][7]["in_scope"] is False
    assert protocol["groups"][8]["rho"] == .995
    assert protocol["groups"][9]["rho"] == 1


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_signal_and_false_rejection_are_separate(runner, protocol, phase):
    group = dict(protocol["groups"][2], n_obs=32, k=4)
    rows, _ = runner._records(group, 7910301, phase, 2, [3.0 if phase != 1 else 0.0], protocol)
    signals = np.zeros(4, dtype=bool)
    if phase == 2:
        signals[0] = True
    elif phase == 3:
        signals[::2] = True
    for row in rows:
        mask = np.unpackbits(np.frombuffer(bytes.fromhex(row["decision_mask"]), dtype=np.uint8))[:4].astype(bool)
        assert row["false_reject"] == int(mask[~signals].any())
        assert row["signal_reject"] == int(mask[signals].any())


def test_parallel_execution_preserves_pairing_and_certificates(runner, protocol):
    group = dict(protocol["groups"][2], n_obs=32, k=2)
    jobs = [(group, 7910302, 2, rep, [2., 3.], protocol) for rep in range(2)]
    expected = list(map(runner._job, jobs))
    with ProcessPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(runner._job, jobs))
    assert actual == expected


def test_existing_evidence_is_preserved(runner, tmp_path):
    (tmp_path / "retained.txt").write_text("frozen")
    with pytest.raises(ValueError, match="never overwritten"):
        runner.run("quick", tmp_path)
    assert (tmp_path / "retained.txt").read_text() == "frozen"


def test_small_complete_run_and_tampering(runner, protocol, tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    protocol["profiles"]["quick"] = dict(null_replicates=2, power_replicates=2, partial_replicates=2,
        null_seed=9910401, power_seed=9910402, partial_seed=9910403)
    protocol["groups"] = [dict(group, n_obs=32, k=2) for group in protocol["groups"] if group["id"] in (3, 8)]
    protocol["power"]["groups"] = [3, 8]
    protocol["power"]["standardized_mean_shifts"] = [2., 3.]
    protocol["partial_null"]["groups"] = [3]
    for name in (*runner.SOURCES, "scripts/verify_joint_uncertainty.py"):
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    (snapshot / runner.SOURCES[0]).write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, "ROOT", snapshot)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="e" * 40))
    output = tmp_path / "evidence"
    runner.run("quick", output)
    auditor = importlib.import_module("verify_joint_uncertainty")
    monkeypatch.setattr(auditor, "ROOT", snapshot)
    result = auditor.verify(output)
    assert result["status"] == "passed" and result["records_checked"] == 72
    assert result["shifted_datasets_regenerated"] == 14
    metadata = json.loads((output / "metadata.json").read_text())
    metadata["cells"].pop()
    (output / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Missing or duplicated"):
        auditor.verify(output)
