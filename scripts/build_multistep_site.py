"""Render the frozen v0.7 overview from its release studies and timings."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.7.0"
RELEASE_COMMIT = "93e3ff427ab1a05b0b17c5c164585ab756645539"
FIGURES = ("01-efficiency.svg", "02-coverage.svg", "03-adaptation.svg")
STUDY_PATH = "results/research/multistep/full/results.json"
BENCHMARK_PATH = "benchmarks/results/multistep-0.7.json"
RUNNER_PATH = "scripts/reproduce_multistep.py"
PROTOCOL_PATH = "experiments/multistep-protocol.json"
FIXED_PATHS = (
    RUNNER_PATH,
    PROTOCOL_PATH,
    "benchmarks/multistep.py",
    STUDY_PATH,
    BENCHMARK_PATH,
    *(f"results/research/multistep/full/{name}" for name in FIGURES),
)
THREADS = {
    key: "1"
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
}


def _digest(contents):
    return hashlib.sha256(contents).hexdigest()


def _version(contents):
    version = re.search(r'^__version__ = "([^"]+)"$', contents.decode(), re.MULTILINE)
    if version is None or version[1] != VERSION:
        raise ValueError(f"Frozen multistep source must have package version {VERSION}.")


def _git_bytes(*arguments):
    """Read release blobs using the same checks as the frozen v0.6 builder."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *arguments], capture_output=True, check=False
        )
    except OSError as exc:
        raise ValueError(
            f"Cannot read frozen release {RELEASE_COMMIT}: Git is unavailable."
        ) from exc
    if result.returncode:
        raise ValueError(
            f"Cannot read frozen release {RELEASE_COMMIT}; fetch this commit and its Git objects."
        )
    return result.stdout


def _release_inputs():
    """Complete release source plus immutable runner, protocol and saved bytes.

    A non-Git fixture must itself contain v0.7 source. Never reinterpret a
    future checkout's source or data as the frozen release when Git is missing.
    """
    prefix = "src/strategy_inference/"
    if not (ROOT / ".git").exists() and not (ROOT / ".git").is_symlink():
        init = ROOT / prefix / "__init__.py"
        if init.is_symlink() or not init.is_file():
            raise ValueError("Frozen release __init__.py must be an ordinary file.")
        _version(init.read_bytes())
        paths = sorted((ROOT / prefix).glob("*.py")) + [ROOT / name for name in FIXED_PATHS]
        inputs = {}
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Frozen release inputs must be ordinary files.")
            inputs[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    else:
        if _git_bytes("cat-file", "-t", RELEASE_COMMIT).strip() != b"commit":
            raise ValueError(f"Frozen release {RELEASE_COMMIT} must identify a Git commit.")
        tree = _git_bytes(
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            RELEASE_COMMIT,
            "--",
            "src/strategy_inference",
            *FIXED_PATHS,
        )
        inputs = {}
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            metadata, encoded_path = entry.split(b"\t", 1)
            mode, kind, _ = metadata.split()
            path = encoded_path.decode()
            is_source = (
                path.startswith(prefix) and path.endswith(".py") and "/" not in path[len(prefix) :]
            )
            if not is_source and path not in FIXED_PATHS:
                continue
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                raise ValueError("Frozen release inputs must be ordinary Git files.")
            inputs[path] = _git_bytes("show", f"{RELEASE_COMMIT}:{path}")
    if not set(FIXED_PATHS).issubset(inputs) or prefix + "__init__.py" not in inputs:
        raise ValueError(f"Frozen release {RELEASE_COMMIT} lacks source or evidence objects.")
    _version(inputs[prefix + "__init__.py"])
    return inputs


def _integer(value, name, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in the declared range.")
    return value


def _number(value, name, *, minimum=None, maximum=None):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be a finite number in the declared range.")
    return value


def _fields(value, names, name):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(f"{name} has missing or unexpected fields.")


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest.")


def _same(actual, expected, name):
    """Strict structure/counts; tolerate only numerical library rounding."""
    if isinstance(expected, dict):
        _fields(actual, expected, name)
        for key, value in expected.items():
            _same(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(f"{name} differs from the declared evidence.")
        for i, value in enumerate(expected):
            _same(actual[i], value, f"{name}[{i}]")
    elif type(expected) is float:
        _number(actual, name)
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{name} differs from the recomputed value.")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} differs from the declared evidence.")


def _read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Evidence JSON must not contain duplicate fields.")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Evidence JSON must not contain {value}.")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def _aggregate(records, protocol, *, runner_source=None):
    """Import the source-bound runner and invoke only its summary calculation."""
    spec = importlib.util.spec_from_file_location("_multistep_summary_runner", ROOT / RUNNER_PATH)
    if runner_source is None:
        runner_source = _release_inputs()[RUNNER_PATH]
    runner = importlib.util.module_from_spec(spec)
    previous_path = sys.path[:]
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        # Execute the verified source directly so --check cannot create a
        # __pycache__ file in the evidence checkout.
        exec(compile(runner_source, spec.origin, "exec"), runner.__dict__)
    finally:
        sys.path[:] = previous_path  # The runner adds its checkout to sys.path.
        sys.dont_write_bytecode = previous_bytecode
    return runner.aggregate(records, protocol)


def _study_records(study, protocol, *, runner_source):
    settings = protocol["profiles"]["full"]
    repetitions = settings["repetitions"]
    scenarios = [row["name"] for row in protocol["scenarios"]]
    methods = [row["name"] for row in protocol["methods"]]
    leads = protocol["lead_times"]
    seeds = {
        (name, rep): protocol["seed"] + settings["seed_offset"] + i * 10000 + rep
        for i, name in enumerate(scenarios)
        for rep in range(repetitions)
    }
    expected = {
        (s, m, h, rep)
        for s in scenarios
        for m in methods
        for h in leads
        for rep in range(repetitions)
    }
    records = study["records"]
    keys = []
    n = (
        settings["n_obs"]
        - max(leads)
        - (settings["initial_train_size"] - 1 + protocol["evaluation_warmup"])
    )
    if n < protocol["local_window"]:
        raise ValueError("The study must retain complete local evaluation windows.")
    for row in records:
        _fields(
            row,
            (
                "scenario",
                "method",
                "lead_time",
                "replicate",
                "seed",
                "n_evaluated",
                "coverage",
                "worst_local_coverage_error",
                "mean_interval_score",
                "interval_score_status",
                "mean_finite_width",
                "empty_count",
                "unbounded_count",
            ),
            "study record",
        )
        rep = _integer(row["replicate"], "replicate", maximum=repetitions - 1)
        lead = _integer(row["lead_time"], "lead_time", minimum=1)
        key = (row["scenario"], row["method"], lead, rep)
        if key not in expected:
            raise ValueError("Every independent path, method and lead must be retained.")
        keys.append(key)
        _same(row["seed"], seeds[(row["scenario"], rep)], "study seed")
        _same(row["n_evaluated"], n, "study evaluated count")
        empty = _integer(row["empty_count"], "empty count", maximum=n)
        full = _integer(row["unbounded_count"], "unbounded count", maximum=n - empty)
        coverage = _number(row["coverage"], "coverage", minimum=0, maximum=1)
        misses = round((1 - coverage) * n)
        _same(coverage, 1 - misses / n, "coverage from integer miss count")
        if misses < empty or n - misses < full:
            raise ValueError("Empty/full counts contradict the observed coverage.")
        _number(
            row["worst_local_coverage_error"],
            "local coverage error",
            minimum=0,
            maximum=max(protocol["alpha"], 1 - protocol["alpha"]) + 1e-12,
        )
        finite = n - empty - full
        if finite:
            width = _number(row["mean_finite_width"], "finite width", minimum=0)
        elif row["mean_finite_width"] is not None:
            raise ValueError("No finite evaluation must have null finite width.")
        if empty or full:
            expected_status = "empty_intervals" if empty else "unbounded_intervals"
            if (
                row["interval_score_status"] != expected_status
                or row["mean_interval_score"] is not None
            ):
                raise ValueError("Empty/full failures must invalidate the entire score summary.")
        elif row["interval_score_status"] == "finite":
            score = _number(row["mean_interval_score"], "interval score", minimum=0)
            if score < width and not math.isclose(score, width, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Proper interval scores cannot be smaller than interval width.")
        elif (
            row["interval_score_status"] != "numeric_overflow"
            or row["mean_interval_score"] is not None
        ):
            raise ValueError("Undefined scores need an explicit failure status and null score.")
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Every independent path, method and lead must be retained.")
    fingerprints = study["input_fingerprints"]
    fingerprint_keys, actual_hashes, predicted_hashes = [], [], []
    for row in fingerprints:
        _fields(
            row,
            (
                "scenario",
                "replicate",
                "seed",
                "actual_sha256",
                "predicted_sha256",
                "initial_scales",
            ),
            "input fingerprint",
        )
        rep = _integer(row["replicate"], "fingerprint replicate", maximum=repetitions - 1)
        key = (row["scenario"], rep)
        if key not in seeds:
            raise ValueError("Input fingerprints must cover every independent path.")
        fingerprint_keys.append(key)
        _same(row["seed"], seeds[key], "fingerprint seed")
        for field, hashes in (
            ("actual_sha256", actual_hashes),
            ("predicted_sha256", predicted_hashes),
        ):
            _hash(row[field], field)
            hashes.append(row[field])
        if type(row["initial_scales"]) is not list or len(row["initial_scales"]) != len(leads):
            raise ValueError("Every path needs one initial scale per lead.")
        for value in row["initial_scales"]:
            if _number(value, "initial scale", minimum=0) == 0:
                raise ValueError("Initial scales must be strictly positive.")
    if (
        len(fingerprint_keys) != len(seeds)
        or set(fingerprint_keys) != set(seeds)
        or len(set(actual_hashes)) != len(seeds)
        or len(set(predicted_hashes)) != len(seeds)
    ):
        raise ValueError("Input fingerprints must retain distinct independent paths.")
    groups, paired = _aggregate(records, protocol, runner_source=runner_source)
    _same(study["aggregate"], groups, "study aggregate")
    _same(study["paired_contrasts"], paired, "study paired contrasts")


def _terminal(case, size, strategy, options):
    leads = [1, 6, 12, 24]
    state = case["terminal_state"]
    for field, expected in dict(
        schema_version=1,
        method="multistep_conformal",
        lead_times=leads,
        start_time=0,
        last_time=size - 1,
        next_time=size,
        alpha=options["alpha"],
        step_size=[options["step_size"]] * 4,
        decay=options["decay"],
        initial_quantile=options["initial_quantile"],
        strategy=strategy,
        initial_scale=options["scale"],
        scale_decay=options["scale_decay"],
        scale_source=options["scale_source"],
        scale_floor=[options["scale_floor"]] * 4,
        default_lane_state=dict(quantile=options["initial_quantile"], n_updates=0, misses=0),
        lane_counts=leads if strategy == "interlaced" else [1] * 4,
    ).items():
        _same(state[field], expected, f"terminal {field}")
    current = state["current_scales"]
    if type(current) is not list or len(current) != 4:
        raise ValueError("Terminal scales need one entry per lead.")
    for scale in current:
        if _number(scale, "terminal scale", minimum=0) == 0:
            raise ValueError("Terminal scales must be strictly positive.")
    _same(
        current,
        [
            max(options["scale_floor"], current[0] * value / options["scale"][0])
            for value in options["scale"]
        ],
        "terminal shortest-scale ratios",
    )
    expected_keys = {
        (h, lane) for h in leads for lane in range(h if strategy == "interlaced" else 1)
    }
    lanes = {}
    for row in state["states"]:
        _fields(row, ("lead_time", "lane", "quantile", "n_updates", "misses"), "terminal lane")
        key = (_integer(row["lead_time"], "lane lead", minimum=1), _integer(row["lane"], "lane"))
        if key not in expected_keys or key in lanes:
            raise ValueError("Terminal lanes must be unique and complete.")
        h, lane = key
        expected_n = size - h if strategy == "pooled" else max(0, 1 + (size - h - 1 - lane) // h)
        _same(row["n_updates"], expected_n, "lane feedback count")
        _integer(row["misses"], "lane miss count", maximum=expected_n)
        _number(row["quantile"], "lane quantile")
        lanes[key] = row
    if set(lanes) != expected_keys:
        raise ValueError("Terminal lanes must be unique and complete.")
    check = case["settings"]["consistency_check_excluded_from_timing"]
    _same(check["same_input_batch_stream_terminal_state"], True, "interface consistency flag")
    _same(
        check["fields_checked"],
        [
            "lane_quantiles",
            "lane_updates",
            "lane_misses",
            "per_lead_evaluated_counts",
            "array_miss_counts",
            "current_scales",
            "pending_intervals",
            "complete_terminal_record",
        ],
        "interface consistency fields",
    )
    _same(check["n_evaluated"], [size - h for h in leads], "interface feedback counts")
    totals = [sum(row["misses"] for (lead, _), row in lanes.items() if lead == h) for h in leads]
    _same(check["misses"], totals, "interface miss counts")
    _same(check["n_pending"], sum(leads), "interface pending count")
    expected_summary = []
    for h, misses in zip(leads, totals, strict=True):
        n, gamma, decay = size - h, options["step_size"], options["decay"]
        if strategy == "pooled":
            bound = (1 + math.fsum(gamma * j**-decay for j in range(1, min(h, n) + 1))) / (
                n * gamma * n**-decay
            )
        else:
            bound = (
                math.fsum(
                    (1 + gamma) / (gamma * row["n_updates"] ** -decay)
                    for (lead, _), row in lanes.items()
                    if lead == h
                )
                / n
            )
        expected_summary.append(
            dict(
                lead_time=h,
                n_issued=size,
                n_evaluated=n,
                n_pending=h,
                coverage=1 - misses / n,
                coverage_bound=min(1.0, bound),
            )
        )
    _same(state["summary"], expected_summary, "terminal summary")
    expected_pending = {(origin, h) for h in leads for origin in range(size - h, size)}
    pending_keys, order = [], []
    for row in state["pending"]:
        _fields(
            row,
            (
                "origin",
                "target",
                "lead_time",
                "prediction",
                "lower",
                "upper",
                "quantile",
                "scale",
                "kind",
            ),
            "pending interval",
        )
        origin = _integer(row["origin"], "pending origin", maximum=size - 1)
        lead = _integer(row["lead_time"], "pending lead", minimum=1)
        if (origin, lead) not in expected_pending:
            raise ValueError("Pending intervals must be precisely the unobserved tail.")
        pending_keys.append((origin, lead))
        _same(row["target"], origin + lead, "pending target")
        order.append((row["target"], origin))
        q, point = (
            _number(row["quantile"], "issued quantile"),
            _number(row["prediction"], "issued point"),
        )
        scale = _number(row["scale"], "issued scale", minimum=0)
        if not scale:
            raise ValueError("Issued scales must be strictly positive.")
        kind = "empty" if q < 0 else "unbounded" if q >= 1 else "finite"
        _same(row["kind"], kind, "pending interval kind")
        if kind == "finite":
            radius = scale * (q / (1 - q))
            _same(row["lower"], point - radius, "pending lower")
            _same(row["upper"], point + radius, "pending upper")
        elif row["lower"] is not None or row["upper"] is not None:
            raise ValueError("Empty/full interval boundaries must be null in JSON.")
        if strategy == "interlaced" or origin == size - 1:
            lane = origin % lead if strategy == "interlaced" else 0
            _same(q, lanes[(lead, lane)]["quantile"], "pending lane threshold")
        if origin == size - 1:
            _same(scale, current[leads.index(lead)], "latest issued scale")
    if (
        len(pending_keys) != len(expected_pending)
        or set(pending_keys) != expected_pending
        or order != sorted(order)
    ):
        raise ValueError("Pending intervals must retain the complete ordered unobserved tail.")
    terminal_hash = hashlib.sha256(
        json.dumps(state, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    _same(case["terminal_state_sha256"], terminal_hash, "terminal fingerprint")
    _same(check["terminal_state_sha256"], terminal_hash, "interface terminal fingerprint")
    _hash(case["result_sha256"], "result_sha256")
    _hash(check["batch_result_sha256"], "batch result fingerprint")
    _same(
        case["result_sha256"],
        check["batch_result_sha256"] if case["case"].startswith("batch_") else terminal_hash,
        "measured result fingerprint",
    )


def _performance(performance):
    expected_cases = (
        "batch_12000x4",
        "pooled_stream_50000x4",
        "pooled_stream_1000x4",
        "interlaced_stream_50000x4",
    )
    _same([c["case"] for c in performance["cases"]], list(expected_cases), "benchmark cases")
    _same(performance["repeats"], 5, "benchmark repeats")
    _same(performance["warmup_calls"], 1, "benchmark warmups")
    _same(performance["blas_thread_environment"], THREADS, "benchmark thread environment")
    leads, phi = [1, 6, 12, 24], 0.65
    scales = [math.sqrt(math.fsum(phi ** (2 * i) for i in range(h))) for h in leads]
    seeds = {12000: 20261230, 50000: 20261231, 1000: 20261232}
    for case, size in zip(performance["cases"], (12000, 50000, 1000, 50000), strict=True):
        settings = case["settings"]
        strategy = "interlaced" if case["case"].startswith("interlaced_") else "pooled"
        expected_options = dict(
            alpha=0.1,
            step_size=0.1,
            decay=0.2,
            scale=scales,
            initial_quantile=0.65,
            strategy=strategy,
            scale_decay=0.97,
            scale_source="shortest",
            scale_floor=1e-8,
        )
        _same(settings["options"], expected_options, "benchmark options")
        for field, value in dict(
            n_obs=size,
            n_origins=size,
            n_leads=4,
            lead_times=leads,
            seed=seeds[size],
            future_pending_flushed=False,
            feedback_order="At each t: observe(t, actual[t]), then predict(predicted[t]).",
            dgp=dict(
                model="Stationary Gaussian AR(1)",
                phi=phi,
                innovation_variance=1.0,
                initial_variance=1 / (1 - phi**2),
            ),
        ).items():
            _same(settings[field], value, f"benchmark {field}")
        _fields(settings["inputs"], ("actual", "predicted", "origins"), "benchmark inputs")
        for field, shape, dtype in (
            ("actual", [size], "float64"),
            ("predicted", [size, 4], "float64"),
            ("origins", [size], "int64"),
        ):
            record = settings["inputs"][field]
            _fields(record, ("shape", "dtype", "sha256"), "benchmark input fingerprint")
            _same(record["shape"], shape, "benchmark input shape")
            _same(record["dtype"], dtype, "benchmark input dtype")
            _hash(record["sha256"], "benchmark input fingerprint")
        samples = case["warm_seconds"]
        if type(samples) is not list or len(samples) != 5:
            raise ValueError("Every engineering measurement needs five warm samples.")
        for value in samples:
            if _number(value, "warm time", minimum=0) == 0:
                raise ValueError("Warm times must be strictly positive.")
        _same(case["warm_median_seconds"], statistics.median(samples), "warm median")
        peak = _integer(case["traced_peak_bytes"], "traced peak")
        _integer(case["traced_current_bytes"], "traced current allocation", maximum=peak)
        _same(case["package_version"], VERSION, "benchmark package version")
        _same(case["numpy_version"], performance["numpy"], "benchmark NumPy version")
        _same(case["thread_environment"], THREADS, "case thread environment")
        _terminal(case, size, strategy, expected_options)
    by_name = {c["case"]: c for c in performance["cases"]}
    short, long = by_name["pooled_stream_1000x4"], by_name["pooled_stream_50000x4"]
    comparison = performance["streaming_memory_comparison"]
    for field, expected in dict(
        short_case=short["case"],
        long_case=long["case"],
        short_n_obs=1000,
        long_n_obs=50000,
        short_peak_bytes=short["traced_peak_bytes"],
        long_peak_bytes=long["traced_peak_bytes"],
    ).items():
        _same(comparison[field], expected, f"memory comparison {field}")
    interlaced = by_name["interlaced_stream_50000x4"]
    _same(interlaced["settings"]["inputs"], long["settings"]["inputs"], "same-input long cases")


def evidence(*, _inputs=None):
    try:
        return _evidence(_release_inputs() if _inputs is None else _inputs)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Saved evidence is malformed or missing required fields.") from exc


def _evidence(inputs):
    prefix = "src/strategy_inference/"
    _version(inputs[prefix + "__init__.py"])
    source = {
        name[len(prefix) :]: _digest(contents)
        for name, contents in inputs.items()
        if name.startswith(prefix)
    }
    study_raw, benchmark_raw = inputs[STUDY_PATH], inputs[BENCHMARK_PATH]
    study, performance = _read_json(study_raw), _read_json(benchmark_raw)
    for record in (study, performance):
        if (
            type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or record["package_version"] != VERSION
            or record["candidate_source_sha256"] != source
        ):
            raise ValueError("Saved evidence must match the complete v0.7 source.")
    if (
        study["runner_sha256"] != _digest(inputs[RUNNER_PATH])
        or study["protocol_sha256"] != _digest(inputs[PROTOCOL_PATH])
        or performance["benchmark_source_sha256"] != _digest(inputs["benchmarks/multistep.py"])
    ):
        raise ValueError("Evidence differs from the saved protocol or runner.")
    protocol = _read_json(inputs[PROTOCOL_PATH])
    if (
        study["profile"] != "full"
        or study["protocol"] != protocol
        or study["profile_settings"] != protocol["profiles"]["full"]
        or protocol["protocol_version"] != 2
        or protocol["profiles"]["full"]["repetitions"] != 40
        or protocol["profiles"]["full"]["n_obs"] != 2400
        or protocol["lead_times"] != [1, 6, 12, 24]
        or protocol["step_size_rule"] != "base / sqrt(lead_time)"
    ):
        raise ValueError("The overview requires the complete declared simulation protocol.")
    _same(study["protocol"], protocol, "study protocol")
    _same(study["profile_settings"], protocol["profiles"]["full"], "study profile")
    _same(study["study"], protocol["study"], "study identity")
    _study_records(study, protocol, runner_source=inputs[RUNNER_PATH])
    _performance(performance)
    for name in FIGURES:
        if study["figure_sha256"][name] != _digest(
            inputs[f"results/research/multistep/full/{name}"]
        ):
            raise ValueError("A saved scientific figure differs from its generation record.")
    return study, performance, study_raw, benchmark_raw


def render(study, performance):
    tag = f"https://github.com/Studyer-Tang/strategy-inference/blob/v{VERSION}"
    install = f"python -m pip install https://github.com/Studyer-Tang/strategy-inference/releases/download/v{VERSION}/strategy_inference-{VERSION}-py3-none-any.whl"
    code = """import numpy as np
from strategy_inference import backtest, naive_forecast, multistep_intervals

y = np.random.default_rng(17).normal(size=512).cumsum()
run = backtest(y, {"naive": naive_forecast}, initial_train_size=128, horizon=12)
origins = np.array([split.origin for split in run.splits])
leads = run.target_indices[0] - origins[0]
intervals = multistep_intervals(
    y, run.forecasts[:, :, 0], origins=origins, lead_times=leads,
    scale=np.sqrt(leads), step_size=.1 / np.sqrt(leads),
    decay=.2, initial_quantile=.65,
    scale_decay=.97, scale_source="shortest",
)
print(intervals.summary())"""
    tasks = {
        "batch_12000x4": "批量 pooled · 12,000 × 4",
        "pooled_stream_50000x4": "流式 pooled · 50,000 × 4",
        "pooled_stream_1000x4": "流式 pooled · 1,000 × 4",
        "interlaced_stream_50000x4": "流式 interlaced · 50,000 × 4",
    }
    rows = "".join(
        f"<tr><td>{html.escape(tasks[c['case']])}</td><td>{c['warm_median_seconds'] * 1000:.2f}</td><td>{c['traced_peak_bytes'] / 1024:.2f}</td></tr>"
        for c in performance["cases"]
    )
    names = {
        "constant": "恒定方差",
        "persistent_scale": "持续波动",
        "fast_scale": "快速波动",
        "scale_shift": "方差突变",
    }
    contrasts = []
    for row in study["paired_contrasts"]:
        if row["lead_time"] != 24:
            continue
        value = row["score_difference"]
        if value["mean"] is None:
            display = f"{row['invalid_score_pairs']} 对含空集或无界区间，均值比较未定义"
        else:
            display = f"{value['mean']:.3f} [{value['lower']:.3f}, {value['upper']:.3f}]"
        contrasts.append(
            f"<tr><td>{names[row['scenario']]}</td><td>{html.escape(display)}</td></tr>"
        )
    figures = "".join(
        f'<figure><img src="research/{name}" alt="{caption}" loading="lazy"><figcaption>{caption}</figcaption></figure>'
        for name, caption in zip(
            FIGURES,
            (
                "区间评分的配对差：短步尺度减各自步长尺度，负值表示评分更低。",
                "四种方法的实际覆盖，阴影为独立路径间的 Monte Carlo 区间。",
                "事先指定的第一条方差突变路径；发行尺度与成熟目标覆盖采用各自时间轴。",
            ),
            strict=True,
        )
    )
    return f'''<!doctype html><html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>strategy-inference · 多步预测与成熟反馈</title>
<meta name="description" content="Python 时间序列评估与不确定性：滚动回测、统计比较、多步成熟反馈和在线区间。">
<style>
:root{{color-scheme:light;background:#fbf8f1;color:#29251f}}*{{box-sizing:border-box}}
body{{margin:auto;max-width:62rem;padding:2.5rem clamp(1rem,4vw,3rem) 3rem;font:17px/1.8 Georgia,"Songti SC",serif}}
h1,h2{{font-weight:normal;line-height:1.3}}h1{{font-size:clamp(2rem,6vw,3rem)}}h2{{margin-top:2.5rem;font-size:1.5rem}}
nav{{display:flex;flex-wrap:wrap;gap:.4rem 1rem;border-bottom:1px solid #d8cebd;padding-bottom:1rem}}
a{{color:#775126;text-underline-offset:.2em;overflow-wrap:anywhere}}p{{margin:.8rem 0 1rem}}
code{{font:.84em/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}}pre{{padding:1rem;background:#f0eade;border-left:2px solid #c8b89e;overflow-x:auto}}
pre code{{white-space:pre}}pre.install code{{white-space:pre-wrap;overflow-wrap:anywhere}}
.table-scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:590px;font-size:.9rem}}
th,td{{padding:.65rem;border-bottom:1px solid #d8cebd;text-align:left}}th{{font-weight:normal;color:#665c4e}}
.small,figcaption,footer{{font-size:.88rem;color:#665c4e}}figure{{margin:2rem 0}}img{{display:block;width:100%;height:auto;background:white}}
footer{{border-top:1px solid #d8cebd;margin-top:3rem;padding-top:1rem}}
</style></head><body>
<nav><a href="{tag}/README.md">源码与安装</a><a href="{tag}/docs/multistep-api.md">多步 API</a><a href="{tag}/docs/time-series.md">评估与比较</a><a href="{tag}/docs/multistep-methods.md">方法推导</a><a href="{tag}/docs/toolbox-roadmap.md">路线图</a></nav>
<main><p class="small">Python · v{VERSION}</p><h1>strategy-inference</h1>
<p>时间序列的预测评估、统计比较与在线不确定性。模型通过 callable 接入；输出保留预测起点、物理步长与目标位置。</p>
<h2>一条实际工作流</h2>
<p>滚动或扩展窗回测 → 点／分位数损失与区间评分 → 固定候选的时间依赖比较 → 单步或多步在线区间。NumPy、SciPy 为核心依赖；pandas 和 Matplotlib 可选。</p>
<p>v0.7 新增明确的反馈时钟：先观察当期数据，处理目标已经成熟的预测，再发出下一条路径。尾部未成熟预测保留为 pending；已发出的边界、阈值和尺度不会回改。</p>
<pre class="install"><code>{html.escape(install)}</code></pre><pre><code>{html.escape(code)}</code></pre>
<p class="small">这是已知创新方差为 1 的模拟 random walk，初始尺度为 √h。实际应用须从训练数据或模型给出尺度；提供给库的点预测也须避免未来信息。尚未发布到 PyPI。</p>
<h2>共享信息，但保持目标清楚</h2>
<p>每个步长维护一个阈值状态，也可以按起点相位分路更新。可选 EWMA 尺度只使用成熟残差：各步长独立估计，或用最短步长的成熟误差更新共同尺度，按事先指定的比例传到其他步长。连续发行且队列填满后，各步长在同一时点收到当前标签；共享尺度利用不同预测起点的残差信息，并不提前获得标签或消除长步长阈值的反馈延迟。</p>
<p>这里的保证是各步长已成熟预测的历史平均覆盖，在理想运算中有显式延迟与步长代价。它不是条件覆盖，也不是整条预测路径同时覆盖。普通浮点运算没有舍入认证；空集和无界区间完整保留。</p>
<p>基于 <a href="https://proceedings.mlr.press/v235/angelopoulos24a.html">ICML 2024 的分位数跟踪</a>，并对照 <a href="https://arxiv.org/html/2410.13115v2">2026 AcMCP 修订预印本</a>与 <a href="https://arxiv.org/html/2609.07251v1">2026 年 9 月的延迟反馈预印本</a>。短步尺度转移是本项目检验的实用扩展，未作首创性主张；完整 AcMCP、条件覆盖与序贯模型集合仍在路线图。</p>
<h2>尺度转移在什么情形有用</h2>
<p>四种预设过程，各 40 条独立路径，T=2400，步长 1、6、12、24。预测器使用已知条件均值以隔离校准问题；初始尺度仅估计于训练前缀。方法与参数在正式路径生成前固定，正式种子与工程试运行分开。</p>
<p>下表只展示 h=24 的评分差及逐点 95% Monte Carlo 区间。负值是短步共享尺度评分更低；含空集或无界区间的路径不会从比较中删除。区间不表示同时显著性，也不代表真实数据验证。</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="实验结果，可横向滚动"><table><thead><tr><th>过程</th><th>短步尺度减独立尺度，差值 [区间]</th></tr></thead><tbody>{"".join(contrasts)}</tbody></table></div>
<p>共享方案在这四种设定中优于自身步长 EWMA；固定尺度或相位交织在部分情形的评分更低，局部覆盖也未同步改善。完整比较、开发选择与解释范围见<a href="{tag}/docs/multistep-results.md">结果说明</a>。</p>
{figures}
<p><a href="research/results.json" download>完整原始记录</a> · <a href="research/protocol.json" download>固定协议</a> · <a href="{tag}/scripts/reproduce_multistep.py">一键复现脚本</a></p>
<h2 id="performance">本机工程计时</h2>
<p>四个步长，串行进程，BLAS 线程环境设为 1；一次预热后五次热调用的中位数。耗时为毫秒，跟踪分配峰值为 KiB。输入生成、导入、独立一致性检查和调用后的指纹不计时；API 内部输出构造计时。</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="性能表，可横向滚动"><table><thead><tr><th>任务</th><th>耗时 / ms</th><th>峰值 / KiB</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="small">{html.escape(performance["platform"])} · Python {html.escape(performance["python"].split()[0])} · NumPy {html.escape(performance["numpy"])}。峰值包含新输出与临时分配，排除已有输入及导入，不是 RSS。流式状态与 pending 队列空间取决于步长集合，批量结果另需 O(FH)。本机计时不能作为通用性能或统计有效性承诺。</p>
<p class="small">计时负载使用统一学习率 0.1；正式模拟使用 0.1/√h。两者分别用于工程成本与统计效果，详细设置见<a href="{tag}/docs/multistep-performance.md">性能说明</a>。</p>
<p><a href="multistep-benchmark.json" download>原始计时、任务设置与源码哈希</a> · <a href="../v0.6.0/">v0.6 归档</a> · <a href="../v0.5.0/">v0.5 归档</a></p>
</main><footer>BSD-3-Clause · <a href="{tag}/docs/multistep-api.md">参数、返回值、边界与失败处理</a></footer></body></html>'''.encode()


def build(*, check=False):
    inputs = _release_inputs()
    study, performance, study_raw, benchmark_raw = evidence(_inputs=inputs)
    archive = ROOT / f"docs/library/v{VERSION}"
    outputs = {
        archive / "index.html": render(study, performance),
        archive / "multistep-benchmark.json": benchmark_raw,
        archive / "research/results.json": study_raw,
        archive / "research/protocol.json": inputs[PROTOCOL_PATH],
    }
    for name in FIGURES:
        outputs[archive / "research" / name] = inputs[f"results/research/multistep/full/{name}"]
    for path in outputs:
        for parent in (
            ROOT,
            *[
                ROOT / Path(*path.relative_to(ROOT).parts[:n])
                for n in range(1, len(path.relative_to(ROOT).parts))
            ],
        ):
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError("Managed page directories must be ordinary directories.")
        if path.is_symlink() or (
            path.exists() and (not path.is_file() or path.stat().st_nlink != 1)
        ):
            raise ValueError("Managed page targets must be ordinary files with one link.")
    for path, content in outputs.items():
        if check:
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError("Current pages differ from their saved evidence.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(check=args.check)
    print("Multistep pages match saved evidence." if args.check else "Multistep pages built.")


if __name__ == "__main__":
    main()
