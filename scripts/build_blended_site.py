"""Package the v0.8 book page from saved, verified research and timing evidence.

No model fitting, simulation, benchmark or source import is performed. Only
source-bound ledger validators and path-level summary calculations are used.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.8.0"
GITHUB = "https://github.com/Studyer-Tang/strategy-inference/blob/main"
STUDY = "results/research/blended-scales/full"
BENCHMARK = "benchmarks/results/blended-scales-0.8.json"
PROTOCOL = "experiments/blended-scale-protocol.json"
FIGURES = ("score-difference", "local-coverage-error", "weight-sensitivity")
CASES = (
    "batch_12000x4",
    "horizon_stream_50000x4",
    "shortest_stream_50000x4",
    "blended_stream_50000x4",
    "blended_stream_1000x4",
)
THREADS = {
    name: "1"
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
}
SCENARIOS = {
    "constant": "常数尺度",
    "persistent_scale": "持续随机尺度",
    "heavy_tail": "重尾创新",
    "persistence_shift": "持久性切换",
    "scale_shift": "尺度切换",
    "lead_bias": "步长特有偏差",
}


def _sha(contents):
    return hashlib.sha256(contents).hexdigest()


def _read(path):
    if not path.is_relative_to(ROOT):
        raise ValueError("Evidence must remain inside the repository.")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Evidence paths must not contain symlinks.")
        if parent == ROOT:
            break
    if not path.is_file():
        raise ValueError(f"Missing ordinary evidence file: {path.relative_to(ROOT)}")
    return path.read_bytes()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON fields are not allowed.")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _same(actual, expected, label):
    if isinstance(expected, dict):
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError(f"Incorrect {label} fields.")
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(f"Incorrect {label} length.")
        for i, value in enumerate(expected):
            _same(actual[i], value, f"{label}[{i}]")
    elif type(expected) is float:
        if (
            type(actual) not in (int, float)
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError(f"Incorrect recomputed {label}.")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Incorrect {label}.")


def _number(value, label, *, positive=False, integer=False):
    if (type(value) is not int if integer else type(value) not in (int, float)) or (
        not math.isfinite(value) or value < 0 or (positive and value == 0)
    ):
        raise ValueError(f"{label} must be finite, nonboolean and nonnegative.")
    return value


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a SHA-256 digest.")


def _plotters():
    """Load only the two existing read-only validation/summary modules."""
    previous = sys.modules.get("plot_scale_study")
    loaded = []
    try:
        for name in ("plot_scale_study", "plot_blended_study"):
            path = ROOT / "scripts" / f"{name}.py"
            source = _read(path)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            exec(compile(source, str(path), "exec"), module.__dict__)
            loaded.append(module)
            if name == "plot_scale_study":
                sys.modules[name] = module
    finally:
        if previous is None:
            sys.modules.pop("plot_scale_study", None)
        else:
            sys.modules["plot_scale_study"] = previous
    return loaded


def _summaries(study, common):
    protocol = study["protocol"]
    reps = study["profile_settings"]["repetitions"]
    rows = {
        (r["scenario"], r["replicate"], r["method"], r["lead_time"]): r for r in study["records"]
    }
    aggregate, paired = [], []

    def ci(values):
        return (
            common.mean_ci(values)
            if values
            else dict(mean=None, mcse=None, lower=None, upper=None, n=0)
        )

    for scenario in protocol["scenarios"]:
        name = scenario["name"]
        for lead in protocol["lead_times"]:
            groups = {
                m: [rows[name, rep, m, lead] for rep in range(reps)] for m in protocol["methods"]
            }
            for method, group in groups.items():
                invalid = sum(r["mean_interval_score"] is None for r in group)
                aggregate.append(
                    dict(
                        scenario=name,
                        lead_time=lead,
                        method=method,
                        n_runs=reps,
                        coverage=ci([r["coverage"] for r in group]),
                        mean_interval_score=ci(
                            [] if invalid else [r["mean_interval_score"] for r in group]
                        ),
                        worst_local_coverage_error=ci(
                            [r["worst_local_coverage_error"] for r in group]
                        ),
                        invalid_score_runs=invalid,
                        empty_count=sum(r["empty_count"] for r in group),
                        unbounded_count=sum(r["unbounded_count"] for r in group),
                    )
                )
            for baseline in ("horizon", "shortest"):
                pairs = list(zip(groups["blend_50"], groups[baseline], strict=True))
                invalid = sum(
                    a["mean_interval_score"] is None or b["mean_interval_score"] is None
                    for a, b in pairs
                )
                paired.append(
                    dict(
                        scenario=name,
                        lead_time=lead,
                        baseline=baseline,
                        candidate="blend_50",
                        invalid_score_pairs=invalid,
                        score_difference=ci(
                            []
                            if invalid
                            else [
                                a["mean_interval_score"] - b["mean_interval_score"]
                                for a, b in pairs
                            ]
                        ),
                        coverage_difference=ci([a["coverage"] - b["coverage"] for a, b in pairs]),
                    )
                )
    _same(study["aggregate"], aggregate, "aggregate")
    _same(study["paired_contrasts"], paired, "paired contrasts")


def _terminal(case, options, size):
    state, check = (
        case["terminal_state"],
        case["settings"]["consistency_check_excluded_from_timing"],
    )
    leads = [1, 6, 12, 24]
    expected = dict(
        schema_version=1,
        method="multistep_conformal",
        lead_times=leads,
        start_time=0,
        last_time=size - 1,
        next_time=size,
        alpha=0.1,
        step_size=[0.1] * 4,
        decay=0.2,
        initial_quantile=0.65,
        strategy="pooled",
        initial_scale=options["scale"],
        scale_decay=0.97,
        scale_source=options["scale_source"],
        scale_floor=[1e-8] * 4,
        default_lane_state=dict(quantile=0.65, n_updates=0, misses=0),
        lane_counts=[1] * 4,
    )
    for field, value in expected.items():
        _same(state[field], value, f"terminal {field}")
    for value in state["current_scales"]:
        _number(value, "current scale", positive=True)
    if len(state["current_scales"]) != 4:
        raise ValueError("Four terminal scales are required.")
    if options["scale_source"] == "blended":
        _same(state["scale_share_weight"], [0.5] * 4, "terminal weights")
        for field in ("own_scales", "shared_scales"):
            if len(state[field]) != 4:
                raise ValueError("Both scale sources must be complete.")
            for value in state[field]:
                _number(value, field, positive=True)
        mixed = [
            (a / max(a, b) + b / max(a, b)) * 0.5 * max(a, b)
            for a, b in zip(state["own_scales"], state["shared_scales"], strict=True)
        ]
        _same(state["current_scales"], mixed, "terminal blend")
        _same(
            state["shared_scales"],
            [
                max(1e-8, state["shared_scales"][0] * s / options["scale"][0])
                for s in options["scale"]
            ],
            "shared ratios",
        )
    elif options["scale_source"] == "shortest":
        _same(
            state["current_scales"],
            [
                max(1e-8, state["current_scales"][0] * s / options["scale"][0])
                for s in options["scale"]
            ],
            "shortest ratios",
        )
    lanes = state["states"]
    if [r["lead_time"] for r in lanes] != leads:
        raise ValueError("All four pooled lanes must be retained.")
    for h, row in zip(leads, lanes, strict=True):
        _same(row["lane"], 0, "lane id")
        _same(row["n_updates"], size - h, "lane updates")
        if _number(row["misses"], "lane misses", integer=True) > size - h:
            raise ValueError("Lane misses exceed mature feedback.")
        if type(row["quantile"]) not in (int, float) or not math.isfinite(row["quantile"]):
            raise ValueError("Lane quantiles must be finite.")
    expected_summary = []
    for h, row in zip(leads, lanes, strict=True):
        n = size - h
        eta = 0.1 * n**-0.2
        step_sum = math.fsum(j**-0.2 for j in range(1, h + 1))
        bound = min(1.0, (1 / eta + step_sum * (0.1 / eta)) / n)
        expected_summary.append(
            dict(
                lead_time=h,
                n_issued=size,
                n_evaluated=n,
                n_pending=h,
                coverage=1 - row["misses"] / n,
                coverage_bound=bound,
            )
        )
    _same(state["summary"], expected_summary, "terminal summary")
    expected_pending = {(origin, origin + h, h) for h in leads for origin in range(size - h, size)}
    pending = state["pending"]
    found = {(r["origin"], r["target"], r["lead_time"]) for r in pending}
    if len(pending) != 43 or found != expected_pending:
        raise ValueError("All 43 future-tail forecasts must remain pending.")
    if [(r["origin"], r["target"], r["lead_time"]) for r in pending] != sorted(
        expected_pending, key=lambda key: (key[1], key[0])
    ):
        raise ValueError("The pending tail must retain its public ordering.")
    for row in pending:
        _number(row["scale"], "issued scale", positive=True)
        if row["kind"] not in ("finite", "empty", "unbounded"):
            raise ValueError("Invalid pending interval kind.")
        for field in ("prediction", "quantile"):
            if type(row[field]) not in (int, float) or not math.isfinite(row[field]):
                raise ValueError("Pending forecast fields must be finite.")
        if row["kind"] == "finite":
            if (
                not all(
                    type(row[k]) in (int, float) and math.isfinite(row[k])
                    for k in ("lower", "upper")
                )
                or row["lower"] > row["upper"]
            ):
                raise ValueError("Finite pending bounds must be ordered.")
        elif row["lower"] is not None or row["upper"] is not None:
            raise ValueError("Nonfinite set bounds must use JSON null.")
    _same(check["same_input_batch_stream_terminal_state"], True, "interface consistency")
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
        "interface fields",
    )
    _same(check["n_evaluated"], [size - h for h in leads], "interface counts")
    _same(check["misses"], [r["misses"] for r in lanes], "interface misses")
    _same(check["n_pending"], 43, "interface pending count")
    terminal_hash = _sha(json.dumps(state, sort_keys=True, allow_nan=False).encode())
    _same(case["terminal_state_sha256"], terminal_hash, "terminal hash")
    _same(check["terminal_state_sha256"], terminal_hash, "interface terminal hash")
    _digest(check["batch_result_sha256"])
    _same(
        case["result_sha256"],
        check["batch_result_sha256"] if case["case"].startswith("batch_") else terminal_hash,
        "result hash",
    )


def _benchmark(performance, study):
    _same(performance["schema_version"], 1, "benchmark schema")
    _same(performance["package_version"], VERSION, "benchmark version")
    sources = {Path(k).name: v for k, v in study["source_sha256"].items() if k.startswith("src/")}
    _same(performance["candidate_source_sha256"], sources, "benchmark source inventory")
    _same(
        performance["benchmark_source_sha256"],
        _sha(_read(ROOT / "benchmarks/blended_scales.py")),
        "benchmark runner source",
    )
    _same(performance["repeats"], 5, "timing repeats")
    _same(performance["warmup_calls"], 1, "timing warmups")
    _same(performance["blas_thread_environment"], THREADS, "BLAS thread settings")
    _same([c["case"] for c in performance["cases"]], list(CASES), "benchmark cases")
    scales = [math.sqrt(math.fsum(0.65 ** (2 * i) for i in range(h))) for h in (1, 6, 12, 24)]
    for case in performance["cases"]:
        name = case["case"]
        size = (
            12000
            if name.startswith("batch_")
            else 1000
            if name == "blended_stream_1000x4"
            else 50000
        )
        source = (
            "horizon"
            if name.startswith("horizon_")
            else "shortest"
            if name.startswith("shortest_")
            else "blended"
        )
        options = dict(
            alpha=0.1,
            step_size=0.1,
            decay=0.2,
            scale=scales,
            initial_quantile=0.65,
            strategy="pooled",
            scale_decay=0.97,
            scale_source=source,
            scale_floor=1e-8,
        )
        if source == "blended":
            options["scale_share_weight"] = 0.5
        settings = case["settings"]
        _same(settings["options"], options, "benchmark options")
        _same(
            settings["dgp"],
            dict(
                model="Stationary Gaussian AR(1)",
                phi=0.65,
                innovation_variance=1.0,
                initial_variance=1 / (1 - 0.65**2),
            ),
            "benchmark DGP",
        )
        for field, value in dict(
            n_obs=size,
            n_origins=size,
            n_leads=4,
            lead_times=[1, 6, 12, 24],
            seed={12000: 20261230, 50000: 20261231, 1000: 20261232}[size],
            future_pending_flushed=False,
            feedback_order="At each t: observe(t, actual[t]), then predict(predicted[t]).",
        ).items():
            _same(settings[field], value, f"benchmark {field}")
        _same(case["package_version"], VERSION, "case version")
        _same(case["numpy_version"], performance["numpy"], "case NumPy version")
        _same(case["thread_environment"], THREADS, "case thread environment")
        if not isinstance(case["numpy_runtime"], str) or not case["numpy_runtime"].strip():
            raise ValueError("NumPy runtime information is required.")
        for field, shape, dtype in (
            ("actual", [size], "float64"),
            ("predicted", [size, 4], "float64"),
            ("origins", [size], "int64"),
        ):
            _same(settings["inputs"][field]["shape"], shape, "input shape")
            _same(settings["inputs"][field]["dtype"], dtype, "input dtype")
            _digest(settings["inputs"][field]["sha256"])
        samples = case["warm_seconds"]
        if len(samples) != 5:
            raise ValueError("Every timing repeat must be retained.")
        for value in samples:
            _number(value, "warm seconds", positive=True)
        _same(case["warm_median_seconds"], statistics.median(samples), "timing median")
        for field in ("traced_current_bytes", "traced_peak_bytes"):
            _number(case[field], field, integer=True)
        if case["traced_current_bytes"] > case["traced_peak_bytes"]:
            raise ValueError("Current allocation exceeds its peak.")
        _terminal(case, options, size)
    by_name = {c["case"]: c for c in performance["cases"]}
    _same(
        by_name[CASES[1]]["settings"]["inputs"],
        by_name[CASES[2]]["settings"]["inputs"],
        "same-input stream baseline",
    )
    _same(
        by_name[CASES[3]]["settings"]["inputs"],
        by_name[CASES[2]]["settings"]["inputs"],
        "same-input stream blend",
    )
    long, short = by_name[CASES[3]], by_name[CASES[4]]
    _same(
        performance["streaming_memory_comparison"],
        dict(
            short_case=CASES[4],
            long_case=CASES[3],
            short_n_obs=1000,
            long_n_obs=50000,
            short_peak_bytes=short["traced_peak_bytes"],
            long_peak_bytes=long["traced_peak_bytes"],
            scope="Different seeded inputs with independent fingerprints. Two measured lengths supplement implementation inspection; they do not prove an asymptotic memory law.",
        ),
        "stream memory comparison",
    )


def evidence():
    raw = _read(ROOT / STUDY / "results.json")
    study = _json(raw)
    _same(study["package_version"], VERSION, "study version")
    _same(study["profile"], "full", "study profile")
    _read(ROOT / STUDY / "source.zip")
    _json(_read(ROOT / STUDY / "source-manifest.json"))
    common, plotter = _plotters()
    computed, _ = plotter.statistics(study)
    if len(study["records"]) != 5760 or len(study["input_fingerprints"]) != 240:
        raise ValueError("The public page requires all 240 full paths and 5760 records.")
    for path, digest in study["source_sha256"].items():
        _same(_sha(_read(ROOT / path)), digest, "current study source")
    _same(_json(_read(ROOT / PROTOCOL)), study["protocol"], "protocol")
    init = _read(ROOT / "src/strategy_inference/__init__.py").decode()
    if re.search(rf'^__version__ = "{re.escape(VERSION)}"$', init, re.MULTILINE) is None:
        raise ValueError("Current package version must match the evidence.")
    _summaries(study, common)
    archive = plotter.verify_archive(ROOT / STUDY / "results.json", study)
    if archive is None:
        raise ValueError("The frozen experiment source snapshot is required.")
    figure_raw = _read(ROOT / STUDY / "figures/figure-data.json")
    figures = _json(figure_raw)
    for field, value in dict(
        schema_version=1,
        input_sha256=_sha(raw),
        study=study["study"],
        profile="full",
        lead_time=24,
        protocol=study["protocol"],
        study_source_sha256=study["source_sha256"],
        source_archive=archive,
        input_fingerprints=study["input_fingerprints"],
        statistics=computed,
    ).items():
        _same(figures[field], value, f"figure {field}")
    _same(
        set(figures["figure_sha256"]),
        {f"{s}.{e}" for s in FIGURES for e in ("svg", "png", "pdf")},
        "figure inventory",
    )
    _same(
        set(figures["plotter_sha256"]),
        {"scripts/plot_scale_study.py", "scripts/plot_blended_study.py"},
        "plotter inventory",
    )
    for path, digest in figures["plotter_sha256"].items():
        _same(_sha(_read(ROOT / path)), digest, "plotter source")
    for name, digest in figures["figure_sha256"].items():
        _same(_sha(_read(ROOT / STUDY / "figures" / name)), digest, "figure bytes")
    environment = _json(_read(ROOT / STUDY / "environment.json"))
    _same(
        environment["source_manifest_sha256"],
        _sha(_read(ROOT / STUDY / "source-manifest.json")),
        "environment manifest binding",
    )
    for field in ("python", "numpy"):
        _same(environment[field], study[field], f"study environment {field}")
    _same(environment["matplotlib"], figures["matplotlib"], "plot environment")
    benchmark_raw = _read(ROOT / BENCHMARK)
    performance = _json(benchmark_raw)
    _benchmark(performance, study)
    return study, performance


CSS = """
:root{color-scheme:light;background:#fbf8f1;color:#29251f}*{box-sizing:border-box}
body{margin:auto;max-width:62rem;padding:2.5rem clamp(1rem,4vw,3rem) 3rem;font:17px/1.8 Georgia,"Songti SC",serif}
h1,h2{font-weight:normal;line-height:1.3}h1{font-size:clamp(2rem,6vw,3rem)}h2{margin-top:2.5rem;font-size:1.5rem}
nav{display:flex;flex-wrap:wrap;gap:.4rem 1rem;border-bottom:1px solid #d8cebd;padding-bottom:1rem}
a{color:#775126;text-underline-offset:.2em;overflow-wrap:anywhere}p{margin:.8rem 0 1rem}
code{font:.84em/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}pre{padding:1rem;background:#f0eade;border-left:2px solid #c8b89e;overflow-x:auto}
pre code{white-space:pre}pre.install code{white-space:pre-wrap;overflow-wrap:anywhere}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:590px;font-size:.9rem}
th,td{padding:.65rem;border-bottom:1px solid #d8cebd;text-align:left}th{font-weight:normal;color:#665c4e}
.small,figcaption,footer{font-size:.88rem;color:#665c4e}figure{margin:2rem 0}img{display:block;width:100%;height:auto;background:white}
footer{border-top:1px solid #d8cebd;margin-top:3rem;padding-top:1rem}
"""


def render(study, performance):
    names = [r["name"] for r in study["protocol"]["scenarios"]]
    group = {(r["scenario"], r["method"]): r for r in study["aggregate"] if r["lead_time"] == 24}
    methods = ("fixed", "horizon", "shortest", "blend_50")
    score_rows, coverage_rows = [], []
    for name in names:
        scores, coverage = [], []
        for method in methods:
            row = group[name, method]
            mean = row["mean_interval_score"]["mean"]
            scores.append("不可汇总" if mean is None else f"{mean:.3f}")
            coverage.append(
                f"{100 * row['coverage']['mean']:.2f}% / {100 * row['worst_local_coverage_error']['mean']:.2f} pp"
            )
        label = html.escape(SCENARIOS.get(name, name))
        score_rows.append(
            f"<tr><td>{label}</td>" + "".join(f"<td>{value}</td>" for value in scores) + "</tr>"
        )
        coverage_rows.append(
            f"<tr><td>{label}</td>" + "".join(f"<td>{value}</td>" for value in coverage) + "</tr>"
        )
    best_fixed = sum(
        group[name, "fixed"]["mean_interval_score"]["mean"] is not None
        and group[name, "fixed"]["mean_interval_score"]["mean"]
        == min(
            r["mean_interval_score"]["mean"]
            for r in study["aggregate"]
            if r["scenario"] == name
            and r["lead_time"] == 24
            and r["mean_interval_score"]["mean"] is not None
        )
        for name in names
    )
    by_case = {c["case"]: c for c in performance["cases"]}
    stream, shortest = by_case[CASES[3]], by_case[CASES[2]]
    overhead = 100 * (stream["warm_median_seconds"] / shortest["warm_median_seconds"] - 1)
    task_labels = {
        CASES[0]: "批量 · 12,000 × 4",
        CASES[1]: "自身 RMS 流式 · 50,000 × 4",
        CASES[2]: "最短共享流式 · 50,000 × 4",
        CASES[3]: "固定融合流式 · 50,000 × 4",
        CASES[4]: "固定融合流式 · 1,000 × 4",
    }
    timing_rows = "".join(
        f"<tr><td>{task_labels[c['case']]}</td><td>{1000 * c['warm_median_seconds']:.2f}</td><td>{c['traced_peak_bytes'] / 1024:.2f}</td></tr>"
        for c in performance["cases"]
    )
    headings = "<tr><th>情形 · h=24</th><th>固定尺度</th><th>自身 RMS</th><th>最短共享</th><th>固定融合 · ½</th></tr>"
    figure_captions = (
        "相对最短共享的路径配对评分差。负值较低；误差条为逐项名义 95% MC t 汇总。",
        "每条路径最差 200 点窗口覆盖误差，再跨路径汇总；它不是理论前缀界。",
        "预先声明的固定权重敏感性。完整展示五个权重，不从正式结果选择最优权重。",
    )
    figures = "".join(
        f'<figure><img src="research/blended/{stem}.svg" alt="{html.escape(caption)}" loading="lazy"><figcaption>{html.escape(caption)}</figcaption></figure>'
        for stem, caption in zip(FIGURES, figure_captions, strict=True)
    )
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>strategy-inference · 时间序列工具库</title><meta name="description" content="时间序列评价、预测比较与在线区间。Python API、完整研究记录和本机性能证据。">
<style>{CSS}</style></head><body>
<nav aria-label="目录"><a href="#start">开始使用</a><a href="#tools">功能</a><a href="#research">研究记录</a><a href="#performance">工程计时</a><a href="v0.7.0/">v0.7 归档</a><a href="https://github.com/Studyer-Tang/strategy-inference">GitHub</a></nav>
<main><p class="small">strategy-inference · v{VERSION} · Python ≥ 3.10</p>
<h1>从时序评价，到在线不确定性</h1>
<p>一个可安装的 Python 工具库：整理预测损失、做因果回测、比较预测器，并在标签成熟后更新单步或多步区间。统计假设、数值边界与复现记录随代码一起保留。</p>
<h2 id="start">开始使用</h2>
<pre class="install"><code>python -m pip install https://github.com/Studyer-Tang/strategy-inference/releases/download/v{VERSION}/strategy_inference-{VERSION}-py3-none-any.whl</code></pre>
<pre><code>import numpy as np
from strategy_inference import MultiStepConformal

tracker = MultiStepConformal(
    [1, 6, 12, 24], scale=[1.0, 1.3, 1.3, 1.3],
    step_size=0.1 / np.sqrt([1, 6, 12, 24]), decay=0.2,
    scale_decay=0.97, scale_source="blended", scale_share_weight=0.5,
)
for t, value in enumerate(observations):
    updates = tracker.observe(t, float(value))
    intervals = tracker.predict([value] * 4)  # naive 基线；只用当前标签</code></pre>
<p class="small">示例尺度须在使用前由训练数据确定；上面只示范接口。允许跳过发行，观测时钟连续。每个区间保持发行时的阈值和尺度，未成熟标签留在 pending。</p>
<h2 id="tools">可用功能</h2>
<ul><li><a href="{GITHUB}/docs/time-series.md">预测评价与回测</a>：损失矩阵、滚动窗口、预先固定的 naive / drift / seasonal 基线。</li>
<li><a href="{GITHUB}/docs/time-series.md">预测器比较</a>：时间依赖下的路径损失差与重采样。</li>
<li><a href="{GITHUB}/docs/multistep-api.md">在线区间</a>：单步、多步成熟反馈、pool / interlace 与固定尺度融合。</li>
<li><a href="{GITHUB}/docs/api.md">金融策略推断</a>：HAC、筛选重放及有模型条件的参数不确定性检验。</li></ul>
<p><a href="{GITHUB}/examples/multistep.py">完整例子</a> · <a href="{GITHUB}/docs/multistep-methods.md">递推与证明</a> · <a href="{GITHUB}/docs/toolbox-roadmap.md">发展路线</a></p>
<h2 id="research">固定尺度融合：保留折中，保留对照</h2>
<p>v0.8 可按预先固定的权重，融合自身步长成熟残差 RMS 与最短步长共享 RMS。权重为 0、1 时，在两来源均可表示的范围内复现两个原模式；默认半权重不表示最优。融合保持每步长阈值独立，既不提前获得长步长标签，也不改变理想平均覆盖账本。</p>
<p>六个情形，每个 40 条独立路径；每条 3,000 个观测。滚动 AR(1) 点预测只用起点已有标签；训练前缀为 600，评价排除前 128 个发行起点。表中是本次模拟的 interval score 观测均值，越低越好。</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="评分表，可横向滚动"><table><thead>{headings}</thead><tbody>{"".join(score_rows)}</tbody></table></div>
<p>h=24 时，固定尺度在六个情形中的 {best_fixed} 个评分均值最低。半权重融合保留两种尺度来源，在部分偏差情形的观测评分有所改善，但没有普遍领先。它是可选配置，不是新的默认方法。</p>
<p class="small">本次记录没有空集或全域；这不能排除其尾事件。阈值不裁剪，若全域区间在模型下有正概率，无条件 raw interval score 的总体期望可能无穷。表格与 t 误差条是观测结果和名义 MC 汇总，不是总体风险或联合显著性的证书。</p>
<details><summary>同时查看覆盖率与最差局部误差</summary>
<p class="small">每格为整体覆盖率 / 路径内最差 200 点窗口误差的跨路径均值（百分点）。历史前缀界不能直接用于这个去 warmup 的后缀或局部窗口。</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="覆盖表，可横向滚动"><table><thead>{headings}</thead><tbody>{"".join(coverage_rows)}</tbody></table></div></details>
{figures}
<p><a href="research/blended/results.json" download>六方法 × 四步长完整路径账本</a> · <a href="research/blended/protocol.json" download>冻结协议</a> · <a href="research/blended/source.zip" download>实验源码快照</a> · <a href="research/blended/source-manifest.json" download>源码 SHA-256</a> · <a href="research/blended/environment.json" download>运行环境</a> · <a href="research/blended/figure-data.json" download>图表数据与哈希</a></p>
<p class="small">source.zip 保存实验运行时的源码，是复核快照；不是 Python 安装包。完整统计口径与工程选择见<a href="{GITHUB}/docs/blended-scales-results.md">结果说明</a>，重放见<a href="{GITHUB}/scripts/reproduce_blended_scales.py">复现程序</a>。</p>
<h2 id="performance">本机工程计时</h2>
<p>固定融合流式处理 50,000 × 4：{stream["warm_median_seconds"]:.3f} 秒，跟踪分配峰值 {stream["traced_peak_bytes"] / 1024:.1f} KiB；在同机同输入下，比最短共享增加 {overhead:.1f}% 耗时。独立保存两个 RMS 来源只增加 O(H) 状态；pending 取决于步长集合，批量结果另需 O(FH)。</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="本机性能表，可横向滚动"><table><thead><tr><th>任务</th><th>热调用 / ms</th><th>分配峰值 / KiB</th></tr></thead><tbody>{timing_rows}</tbody></table></div>
<p class="small">{html.escape(performance["platform"])} · Python {html.escape(performance["python"].split()[0])} · NumPy {html.escape(performance["numpy"])}。串行进程，BLAS 线程环境设为 1，一次预热后五次热调用中位数。导入、输入生成、一致性检查和调用后指纹不计时；API 内部输出构造计时。tracemalloc 单独一次，排除已有输入，包含新增分配，不是 RSS。</p>
<p class="small">工程负载统一学习率 0.1；统计实验采用 0.1/√h。这些 source-bound 本机测量不构成通用性能或统计保证。</p>
<p><a href="research/blended/benchmark.json" download>全部计时、设置与源码哈希</a> · <a href="v0.7.0/">v0.7 页面与计时归档</a> · <a href="v0.6.0/">v0.6 归档</a> · <a href="v0.5.0/">v0.5 归档</a></p>
</main><footer>BSD-3-Clause · <a href="{GITHUB}/docs/multistep-api.md">参数、数值边界与失败处理</a> · <a href="research/blended/site-manifest.json" download>本页证据索引</a></footer></body></html>'''.encode()


def build(*, check=False, destination=None):
    study, performance = evidence()
    destination = ROOT / "docs/library" if destination is None else Path(destination)
    resources = {
        "results.json": ROOT / STUDY / "results.json",
        "protocol.json": ROOT / PROTOCOL,
        "source.zip": ROOT / STUDY / "source.zip",
        "source-manifest.json": ROOT / STUDY / "source-manifest.json",
        "environment.json": ROOT / STUDY / "environment.json",
        "benchmark.json": ROOT / BENCHMARK,
        "figure-data.json": ROOT / STUDY / "figures/figure-data.json",
        **{f"{name}.svg": ROOT / STUDY / "figures" / f"{name}.svg" for name in FIGURES},
    }
    outputs = {"index.html": render(study, performance)}
    outputs.update({f"research/blended/{name}": _read(path) for name, path in resources.items()})
    manifest = dict(
        schema_version=1,
        package_version=VERSION,
        builder_sha256=_sha(_read(ROOT / "scripts/build_blended_site.py")),
        input_sha256={
            path.relative_to(ROOT).as_posix(): _sha(_read(path)) for path in resources.values()
        },
        output_sha256={name: _sha(raw) for name, raw in outputs.items()},
        scope="Saved evidence only; managed current-page files, leaving old archives and assets untouched.",
    )
    outputs["research/blended/site-manifest.json"] = (
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    ).encode()
    targets = {destination / name: raw for name, raw in outputs.items()}
    for path in targets:
        for parent in path.parents:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError("Managed output directories must be ordinary directories.")
        if path.is_symlink() or (
            path.exists() and (not path.is_file() or path.stat().st_nlink != 1)
        ):
            raise ValueError("Managed outputs must be ordinary unlinked files.")
        if path.resolve() in {source.resolve() for source in resources.values()}:
            raise ValueError("Managed output must not overwrite evidence inputs.")
    if check:
        for path, raw in targets.items():
            if not path.is_file() or path.read_bytes() != raw:
                raise ValueError(f"Managed page is stale or missing: {path}")
    else:
        for path, raw in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    return tuple(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/library")
    args = parser.parse_args()
    build(check=args.check, destination=args.output)


if __name__ == "__main__":
    main()
