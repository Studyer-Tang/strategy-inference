"""Saved scientific evidence must bind the generated page, without unsafe writes."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "multistep_site", ROOT / "scripts/build_multistep_site.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / "repository"
    paths = [
        "experiments/multistep-protocol.json",
        "scripts/reproduce_multistep.py",
        "benchmarks/multistep.py",
        "benchmarks/results/multistep-0.7.json",
        "results/research/multistep/full/results.json",
    ]
    paths += ["results/research/multistep/full/" + name for name in mod.FIGURES]
    paths += [str(p.relative_to(ROOT)) for p in (ROOT / "src/strategy_inference").glob("*.py")]
    for name in paths:
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    old = root / "docs/library/v0.6.0/index.html"
    old.parent.mkdir(parents=True)
    old.write_text("Archived v0.6.\n")
    (root / "docs/library/index.html").write_text("Old current page.\n")
    older = root / "docs/library/v0.5.0/index.html"
    older.parent.mkdir(parents=True)
    older.write_text("Archived v0.5.\n")
    monkeypatch.setattr(mod, "ROOT", root)
    return mod


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_build_and_readonly_check_preserve_archives_and_raw_evidence(builder):
    archive = builder.ROOT / "docs/library/v0.6.0/index.html"
    older = builder.ROOT / "docs/library/v0.5.0/index.html"
    before = (archive.read_bytes(), archive.stat().st_mtime_ns)
    older_before = (older.read_bytes(), older.stat().st_mtime_ns)
    source_before = snapshot(builder.ROOT / "src")
    raw_before = snapshot(builder.ROOT / "results")
    builder.build()
    assert source_before == snapshot(builder.ROOT / "src")
    assert raw_before == snapshot(builder.ROOT / "results")
    assert before == (archive.read_bytes(), archive.stat().st_mtime_ns)
    assert older_before == (older.read_bytes(), older.stat().st_mtime_ns)
    prior = snapshot(builder.ROOT)
    builder.build(check=True)
    assert snapshot(builder.ROOT) == prior
    assert (builder.ROOT / "docs/research/multistep/results.json").read_bytes() == (
        builder.ROOT / "results/research/multistep/full/results.json"
    ).read_bytes()


@pytest.mark.parametrize(
    "name",
    [
        "src/strategy_inference/multistep.py",
        "scripts/reproduce_multistep.py",
        "benchmarks/multistep.py",
        "experiments/multistep-protocol.json",
        "results/research/multistep/full/01-efficiency.svg",
    ],
)
def test_source_protocol_runner_and_figure_changes_reject_without_repair(builder, name):
    path = builder.ROOT / name
    path.write_bytes(path.read_bytes() + b"\n ")
    before = snapshot(builder.ROOT)
    with pytest.raises(ValueError):
        builder.build()
    assert before == snapshot(builder.ROOT)


def test_missing_simulation_paths_cannot_be_presented_as_complete_study(builder):
    path = builder.ROOT / "results/research/multistep/full/results.json"
    raw = json.loads(path.read_bytes())
    raw["records"].pop()
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Every independent path"):
        builder.build()


@pytest.mark.parametrize("change", ["thread", "memory", "median", "size", "feedback"])
def test_invalid_performance_records_are_rejected(builder, change):
    path = builder.ROOT / "benchmarks/results/multistep-0.7.json"
    raw = json.loads(path.read_bytes())
    c = raw["cases"][0]
    if change == "thread":
        c["thread_environment"]["OMP_NUM_THREADS"] = "2"
    elif change == "memory":
        c["traced_peak_bytes"] = c["traced_current_bytes"] - 1
    elif change == "median":
        c["warm_median_seconds"] *= 2
    elif change == "size":
        c["settings"]["n_obs"] = 12001
    else:
        c["settings"]["future_pending_flushed"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        builder.build()


def test_all_targets_preflight_before_writing_any_page(builder, tmp_path):
    external = tmp_path / "external.svg"
    external.write_text("External file.\n")
    bad = builder.ROOT / "docs/research/multistep/03-adaptation.svg"
    bad.parent.mkdir(parents=True)
    bad.symlink_to(external)
    prior = (builder.ROOT / "docs/library/index.html").read_bytes()
    with pytest.raises(ValueError):
        builder.build()
    assert (builder.ROOT / "docs/library/index.html").read_bytes() == prior
    assert external.read_text() == "External file.\n"


def test_check_does_not_create_missing_current_assets(builder):
    before = snapshot(builder.ROOT)
    with pytest.raises(ValueError):
        builder.build(check=True)
    assert snapshot(builder.ROOT) == before


def test_environment_is_escaped_and_methods_are_explicit(builder):
    study, report, _, _ = builder.evidence()
    report["platform"] = '<img src=x onerror="alert(1)">'
    report["python"] = "<script>python</script> 3"
    report["numpy"] = "<script>numpy</script>"
    page = builder.render(study, report).decode()
    assert '<img src=x onerror="alert(1)">' not in page
    assert "&lt;img src=x" in page
    assert "<script>" not in page
    assert "&lt;script&gt;python" in page and "&lt;script&gt;numpy" in page
    assert "不是条件覆盖" in page and "未作首创性主张" in page


def _mutate(builder, name, path, change):
    target = builder.ROOT / name
    raw = json.loads(target.read_bytes())
    node = raw
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = change(node[path[-1]])
    target.write_text(json.dumps(raw))


STUDY = "results/research/multistep/full/results.json"
BENCHMARK = "benchmarks/results/multistep-0.7.json"


@pytest.mark.parametrize(
    "path,change",
    [
        (("records", 0, "seed"), lambda x: x + 1),
        (("records", 0, "replicate"), lambda _: True),
        (("records", 0, "lead_time"), lambda _: True),
        (("records", 0, "n_evaluated"), lambda x: x - 1),
        (("records", 0, "coverage"), lambda x: x + 0.0001),
        (("records", 0, "coverage"), lambda _: True),
        (("records", 0, "mean_interval_score"), lambda _: float("nan")),
        (("records", 0, "mean_interval_score"), lambda _: -1),
        (("records", 0, "mean_interval_score"), lambda _: 0),
        (("records", 0, "mean_interval_score"), lambda x: x + 1),
        (("records", 0, "mean_finite_width"), lambda _: -1),
        (("records", 0, "worst_local_coverage_error"), lambda _: float("inf")),
        (("records", 0, "empty_count"), lambda _: True),
        (("records", 0, "empty_count"), lambda _: 999999),
        (("records", 0, "empty_count"), lambda _: 1),
        (("records", 0, "unbounded_count"), lambda _: 1),
        (("records", 0, "interval_score_status"), lambda _: "empty_intervals"),
        (("records", 0, "interval_score_status"), lambda _: "unknown"),
        (("input_fingerprints", 0, "seed"), lambda x: x + 1),
        (("input_fingerprints", 0, "replicate"), lambda _: False),
        (("input_fingerprints", 0, "actual_sha256"), lambda _: "invalid"),
        (("input_fingerprints", 0, "initial_scales"), lambda _: [1]),
        (("input_fingerprints", 0, "initial_scales", 0), lambda _: 0),
        (("input_fingerprints", 0, "initial_scales", 0), lambda _: True),
        (("aggregate", 0, "coverage", "mean"), lambda x: x + 0.1),
        (("aggregate", 0, "n_runs"), lambda x: float(x)),
        (("aggregate", 0, "invalid_score_runs"), lambda _: False),
        (("paired_contrasts", 3, "score_difference", "mean"), lambda x: x + 2),
        (("paired_contrasts", 3, "score_difference", "lower"), lambda x: x - 2),
        (("paired_contrasts", 3, "invalid_score_pairs"), lambda _: "<script>bad</script>"),
        (("paired_contrasts", 3, "invalid_score_pairs"), lambda _: False),
        (("schema_version",), lambda _: True),
    ],
)
def test_raw_study_fields_and_derived_summaries_cannot_be_tampered(builder, path, change):
    _mutate(builder, STUDY, path, change)
    before = snapshot(builder.ROOT)
    with pytest.raises(ValueError):
        builder.build()
    assert snapshot(builder.ROOT) == before


@pytest.mark.parametrize("change", ["missing", "duplicate", "shared_actual", "shared_prediction"])
def test_fingerprints_retain_all_distinct_independent_paths(builder, change):
    target = builder.ROOT / STUDY
    raw = json.loads(target.read_bytes())
    fingerprints = raw["input_fingerprints"]
    if change == "missing":
        fingerprints.pop()
    elif change == "duplicate":
        fingerprints[-1] = fingerprints[0].copy()
    elif change == "shared_actual":
        fingerprints[1]["actual_sha256"] = fingerprints[0]["actual_sha256"]
    else:
        fingerprints[1]["predicted_sha256"] = fingerprints[0]["predicted_sha256"]
    target.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="fingerprint"):
        builder.evidence()


def test_duplicate_records_cannot_replace_a_missing_path(builder):
    _mutate(builder, STUDY, ("records",), lambda rows: rows[:-1] + [rows[0]])
    with pytest.raises(ValueError, match="Every independent path"):
        builder.evidence()


def test_reordered_raw_records_retain_the_same_valid_summary(builder):
    _mutate(builder, STUDY, ("records",), lambda rows: rows[::-1])
    builder.evidence()


def test_small_summary_rounding_is_allowed_but_structure_and_counts_stay_strict(builder):
    _mutate(builder, STUDY, ("aggregate", 0, "coverage", "mean"), lambda x: x + 1e-14)
    builder.evidence()
    _mutate(builder, STUDY, ("aggregate", 0, "coverage", "n"), lambda x: float(x))
    with pytest.raises(ValueError, match="aggregate"):
        builder.evidence()


@pytest.mark.parametrize(
    "path,change",
    [
        (("cases", 0, "settings", "n_origins"), lambda x: x + 1),
        (("cases", 0, "settings", "n_leads"), lambda _: 3),
        (("cases", 0, "settings", "seed"), lambda x: x + 1),
        (("cases", 0, "settings", "options", "alpha"), lambda _: 0.2),
        (("cases", 0, "settings", "options", "initial_quantile"), lambda _: 0.5),
        (("cases", 0, "settings", "options", "strategy"), lambda _: "interlaced"),
        (("cases", 3, "settings", "options", "strategy"), lambda _: "pooled"),
        (("cases", 0, "settings", "options", "scale_floor"), lambda _: True),
        (("cases", 0, "settings", "options", "scale", 1), lambda x: x * 2),
        (("cases", 0, "settings", "options"), lambda x: x | {"unknown_option": 1}),
        (("cases", 0, "settings", "inputs", "actual", "shape"), lambda _: [12001]),
        (("cases", 0, "settings", "inputs", "origins", "dtype"), lambda _: "float64"),
        (("cases", 0, "settings", "inputs", "actual", "sha256"), lambda _: "not-a-hash"),
        (
            (
                "cases",
                0,
                "settings",
                "consistency_check_excluded_from_timing",
                "same_input_batch_stream_terminal_state",
            ),
            lambda _: 1,
        ),
        (
            ("cases", 0, "settings", "consistency_check_excluded_from_timing", "n_evaluated", 0),
            lambda x: x - 1,
        ),
        (
            ("cases", 0, "settings", "consistency_check_excluded_from_timing", "n_pending"),
            lambda x: x - 1,
        ),
        (("cases", 0, "warm_seconds", 0), lambda _: False),
        (("cases", 0, "warm_median_seconds"), lambda _: True),
        (("cases", 0, "traced_peak_bytes"), lambda _: True),
        (("cases", 0, "terminal_state", "start_time"), lambda _: True),
        (("cases", 0, "terminal_state", "next_time"), lambda x: x + 1),
        (("cases", 0, "terminal_state", "step_size"), lambda _: [0.1]),
        (("cases", 0, "terminal_state", "states", 0, "n_updates"), lambda x: x - 1),
        (("cases", 0, "terminal_state", "states", 0, "misses"), lambda _: 999999),
        (("cases", 0, "terminal_state", "states", 0, "quantile"), lambda _: float("nan")),
        (("cases", 3, "terminal_state", "states"), lambda rows: rows[:-1]),
        (("cases", 0, "terminal_state", "summary", 0, "n_evaluated"), lambda x: x + 1),
        (("cases", 0, "terminal_state", "summary", 0, "coverage"), lambda x: x + 0.01),
        (("cases", 0, "terminal_state", "summary", 0, "coverage_bound"), lambda x: x * 2),
        (("cases", 0, "terminal_state", "pending", 0, "target"), lambda x: x + 1),
        (("cases", 0, "terminal_state", "pending", 0, "scale"), lambda _: 0),
        (("cases", 0, "terminal_state", "pending", 0, "lower"), lambda x: x - 1),
        (("cases", 0, "terminal_state", "pending"), lambda rows: rows[:-1]),
        (("cases", 0, "terminal_state_sha256"), lambda _: "0" * 64),
        (("cases", 1, "result_sha256"), lambda _: "0" * 64),
        (("cases", 0, "result_sha256"), lambda _: "0" * 64),
        (("streaming_memory_comparison", "long_peak_bytes"), lambda x: x + 1),
        (("schema_version",), lambda _: True),
        (("repeats",), lambda _: 5.0),
        (("warmup_calls",), lambda _: True),
    ],
)
def test_benchmark_settings_terminal_accounts_and_hashes_are_bound(builder, path, change):
    _mutate(builder, BENCHMARK, path, change)
    before = snapshot(builder.ROOT)
    with pytest.raises(ValueError):
        builder.build()
    assert snapshot(builder.ROOT) == before


def test_valid_hash_does_not_bypass_terminal_count_semantics(builder):
    target = builder.ROOT / BENCHMARK
    raw = json.loads(target.read_bytes())
    case = raw["cases"][1]
    case["terminal_state"]["states"][0]["n_updates"] -= 1
    digest = hashlib.sha256(
        json.dumps(case["terminal_state"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    case["terminal_state_sha256"] = case["result_sha256"] = digest
    case["settings"]["consistency_check_excluded_from_timing"]["terminal_state_sha256"] = digest
    target.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="feedback count"):
        builder.evidence()


def test_null_invalid_score_summary_cannot_be_replaced_with_a_finite_winner(builder):
    target = builder.ROOT / STUDY
    raw = json.loads(target.read_bytes())
    row = next(
        r for r in raw["records"] if r["method"] == "pooled_shortest" and r["lead_time"] == 24
    )
    row.update(empty_count=1, mean_interval_score=None, interval_score_status="empty_intervals")
    raw["aggregate"], raw["paired_contrasts"] = builder._aggregate(raw["records"], raw["protocol"])
    target.write_text(json.dumps(raw))
    builder.evidence()
    contrast = next(
        r
        for r in raw["paired_contrasts"]
        if r["scenario"] == row["scenario"] and r["lead_time"] == 24
    )
    assert contrast["score_difference"]["mean"] is None and contrast["invalid_score_pairs"] == 1
    contrast["score_difference"]["mean"] = -999
    target.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="paired contrasts"):
        builder.evidence()


def test_invalid_score_message_escapes_dynamic_text_even_when_rendered_directly(builder):
    study, report, _, _ = builder.evidence()
    contrast = next(row for row in study["paired_contrasts"] if row["lead_time"] == 24)
    contrast["score_difference"]["mean"] = None
    contrast["invalid_score_pairs"] = "<script>bad</script>"
    page = builder.render(study, report).decode()
    assert "<script>bad</script>" not in page
    assert "&lt;script&gt;bad&lt;/script&gt;" in page


def test_duplicate_json_fields_and_missing_required_fields_are_errors(builder):
    target = builder.ROOT / STUDY
    raw = target.read_text()
    target.write_text(
        raw.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1)
    )
    with pytest.raises(ValueError, match="duplicate"):
        builder.evidence()
    target.write_text(raw)
    decoded = json.loads(raw)
    del decoded["records"][0]["seed"]
    target.write_text(json.dumps(decoded))
    with pytest.raises(ValueError):
        builder.evidence()


def test_fresh_process_check_is_readonly_without_existing_local_bytecode(builder):
    builder.build()
    script = builder.ROOT / "scripts/build_multistep_site.py"
    shutil.copyfile(ROOT / "scripts/build_multistep_site.py", script)
    assert not list(builder.ROOT.rglob("__pycache__"))
    before = snapshot(builder.ROOT)
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    completed = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not list(builder.ROOT.rglob("__pycache__"))
    assert snapshot(builder.ROOT) == before


def test_summary_validation_preserves_process_import_settings_on_failure(builder):
    _mutate(builder, STUDY, ("aggregate", 0, "n_runs"), lambda n: n - 1)
    previous_path, previous_bytecode = sys.path[:], sys.dont_write_bytecode
    with pytest.raises(ValueError, match="aggregate"):
        builder.evidence()
    assert sys.path == previous_path and sys.dont_write_bytecode == previous_bytecode


@pytest.mark.parametrize("kind", ["broken_symlink", "hardlink", "directory"])
def test_unsafe_output_targets_reject_before_any_write(builder, tmp_path, kind):
    target = builder.ROOT / "docs/research/multistep/03-adaptation.svg"
    target.parent.mkdir(parents=True)
    external = tmp_path / "external.svg"
    external.write_text("External unchanged.\n")
    if kind == "broken_symlink":
        target.symlink_to(tmp_path / "missing.svg")
    elif kind == "hardlink":
        target.hardlink_to(external)
    else:
        target.mkdir()
    before = (builder.ROOT / "docs/library/index.html").read_bytes()
    with pytest.raises(ValueError, match="ordinary files"):
        builder.build()
    assert (builder.ROOT / "docs/library/index.html").read_bytes() == before
    assert external.read_text() == "External unchanged.\n"


@pytest.mark.parametrize(
    "directory", ["docs", "docs/library", "docs/research", "docs/research/multistep"]
)
def test_managed_parent_symlinks_reject_without_external_writes(builder, tmp_path, directory):
    parent = builder.ROOT / directory
    external = tmp_path / "external"
    external.mkdir()
    if parent.exists():
        parent.rename(tmp_path / "saved-original")
    else:
        parent.parent.mkdir(parents=True, exist_ok=True)
    parent.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="ordinary directories"):
        builder.build()
    assert list(external.iterdir()) == []
