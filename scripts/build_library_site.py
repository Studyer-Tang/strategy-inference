"""Build a small library page from saved, source-bound benchmark evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.5.0"
TAG_ROOT = f"https://github.com/Studyer-Tang/strategy-inference/blob/v{VERSION}"
FILES = {"index.html", "benchmark.json"}


def _sha(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _literal(source: str, name: str):
    matches = [
        node.value for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(matches) != 1:
        raise ValueError(f"One literal {name} assignment is required.")
    return ast.literal_eval(matches[0])


def _source_hashes() -> dict[str, str]:
    package = ROOT / "src/strategy_inference"
    paths = sorted(package.glob("*.py"))
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Library source must contain ordinary Python files.")
    return {path.name: _sha(path.read_bytes()) for path in paths}


def _number(value, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0 or (positive and value == 0)
    ):
        raise ValueError("Benchmark numbers must be finite and nonnegative; timings must be positive.")
    return float(value)


def _equal_number(actual, expected, label: str):
    if not math.isclose(_number(actual), expected, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f"Inconsistent benchmark {label}.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _validate_timing(cell: dict, repeats: int):
    samples = cell.get("warm_call_ms")
    if not isinstance(samples, list) or len(samples) != repeats:
        raise ValueError("Every benchmark cell must retain all timing repeats.")
    values = [_number(value, positive=True) for value in samples]
    _equal_number(cell.get("warm_median_ms"), statistics.median(values), "warm median")
    components = [_number(cell.get(name), positive=name == "cold_call_ms")
                  for name in ("import_ms", "api_load_ms", "cold_call_ms")]
    _equal_number(cell.get("first_use_ms"), sum(components), "first-use sum")
    for name in ("process_peak_rss_mib", "traced_call_peak_mib"):
        _number(cell.get(name))
    wrapper = cell.get("unified_api_warm_median_ms")
    if wrapper is not None:
        _number(wrapper, positive=True)


def _validate_pair(before: dict, after: dict, comparison: dict):
    if comparison.get("status") != "passed":
        raise ValueError("Every saved result comparison must have passed.")
    _equal_number(comparison.get("warm_speedup"),
                  before["warm_median_ms"] / after["warm_median_ms"], "speedup")
    a, b = before.get("fingerprint"), after.get("fingerprint")
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("Both benchmark result fingerprints are required.")
    mode, k = before["mode"], before["k"]
    expected_decisions = True if mode in ("fixed", "resampled") else None
    expected_certificates = True if mode == "gaussian_ar" else None
    if (
        "exact_decisions_and_pvalues" not in comparison
        or comparison["exact_decisions_and_pvalues"] is not expected_decisions
        or "exact_certificates" not in comparison
        or comparison["exact_certificates"] is not expected_certificates
    ):
        raise ValueError("Exact result comparison flags do not match the benchmark method.")
    if mode == "gaussian_ar":
        digest = a.get("exact_evidence_sha256")
        if (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or digest != b.get("exact_evidence_sha256")
        ):
            raise ValueError("Gaussian certificate fingerprints differ.")
        if comparison.get("maximum_numeric_errors") != {}:
            raise ValueError("Gaussian exact certificates must not claim numeric comparisons.")
    else:
        if mode != "hac":
            for key in ("decisions", "adjusted_pvalue", "global_pvalue"):
                if key not in a or a[key] != b.get(key):
                    raise ValueError(f"Bootstrap fingerprints differ: {key}.")
            if (
                not isinstance(a["decisions"], list) or len(a["decisions"]) != k
                or any(not isinstance(value, bool) for value in a["decisions"])
                or not isinstance(a["adjusted_pvalue"], list) or len(a["adjusted_pvalue"]) != k
                or any(not 0 <= _number(value) <= 1 for value in a["adjusted_pvalue"])
                or not 0 <= _number(a["global_pvalue"]) <= 1
            ):
                raise ValueError("Bootstrap decision/p-value fingerprints are malformed.")
        expected = {"long_run_variance"} if mode == "hac" else {
            "mean", "standard_error", "statistic", "simultaneous_ci_low", "simultaneous_ci_high",
        }
        numeric_a, numeric_b = a.get("numeric"), b.get("numeric")
        if (
            not isinstance(numeric_a, dict) or not isinstance(numeric_b, dict)
            or set(numeric_a) != expected or set(numeric_b) != expected
        ):
            raise ValueError("The numeric result fingerprint is incomplete.")
        errors = comparison.get("maximum_numeric_errors")
        if not isinstance(errors, dict) or set(errors) != expected:
            raise ValueError("The numeric-error comparison is incomplete.")
        for key in expected:
            left, right = numeric_a[key], numeric_b[key]
            if not isinstance(left, list) or not isinstance(right, list) or len(left) != k or len(right) != k:
                raise ValueError("Numeric fingerprints must contain one value per candidate.")
            differences = []
            for x, y in zip(left, right, strict=True):
                if (
                    isinstance(x, bool) or isinstance(y, bool)
                    or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                    or not math.isfinite(x) or not math.isfinite(y)
                    or abs(x-y) > 1e-12 + 1e-11 * abs(x)
                ):
                    raise ValueError("Numeric result fingerprints differ.")
                differences.append(abs(x-y))
            _equal_number(errors[key], max(differences), "numeric error")


def _snapshot(source: Path):
    if source.is_symlink() or not source.is_file():
        raise ValueError("The benchmark report must be an ordinary file.")
    raw = source.read_bytes()
    report = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("Benchmark schema version 1 is required.")
    hashes = _source_hashes()
    if report.get("candidate_source_sha256") != hashes:
        raise ValueError("Candidate source SHA-256 differs from the saved benchmark.")
    version = _literal((ROOT / "src/strategy_inference/__init__.py").read_text(), "__version__")
    if version != VERSION:
        raise ValueError(f"Current library version must be {VERSION}.")
    runner = ROOT / "benchmarks/run.py"
    if runner.is_symlink() or report.get("benchmark_source_sha256") != _sha(runner.read_bytes()):
        raise ValueError("Benchmark runner SHA-256 differs from the saved benchmark.")
    cases = _literal(runner.read_text(), "CASES")
    if not isinstance(cases, dict) or len(cases) != 6:
        raise ValueError("The benchmark manifest must contain six cases.")
    baseline, candidate, comparisons = (report.get(name) for name in ("baseline", "candidate", "comparisons"))
    if any(not isinstance(items, list) or len(items) != 6 for items in (baseline, candidate, comparisons)):
        raise ValueError("Six baseline, candidate and comparison records are required.")
    repeats = report.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("A positive integer timing-repeat count is required.")
    for index, (case, manifest) in enumerate(cases.items()):
        if not isinstance(manifest, tuple) or len(manifest) != 4:
            raise ValueError("The benchmark case manifest is malformed.")
        mode, t, k, b = manifest
        if mode not in ("hac", "fixed", "resampled", "gaussian_ar"):
            raise ValueError("Unknown benchmark method.")
        for side, version in ((baseline, "0.4.0"), (candidate, VERSION)):
            cell = side[index]
            if not isinstance(cell, dict) or (
                cell.get("case"), cell.get("mode"), cell.get("n_obs"),
                cell.get("k"), cell.get("n_resamples"),
            ) != (case, mode, t, k, b):
                raise ValueError("Benchmark cells must match the ordered case manifest.")
            if cell.get("code_version") != version:
                raise ValueError("Benchmark code_version does not match its release.")
            _validate_timing(cell, repeats)
        if not isinstance(comparisons[index], dict):
            raise ValueError("A comparison record must be a JSON object.")
        _validate_pair(baseline[index], candidate[index], comparisons[index])
    return raw, report, hashes


def _render(report: dict, digest: str) -> bytes:
    escape = html.escape
    rows = []
    labels = {"hac": "HAC", "fixed": "Bootstrap · fixed",
              "resampled": "Bootstrap · resampled", "gaussian_ar": "Gaussian AR"}
    for before, after, comparison in zip(report["baseline"], report["candidate"],
                                          report["comparisons"], strict=True):
        wrapper = after["unified_api_warm_median_ms"]
        wrapper_text = "—" if wrapper is None else f"{wrapper:.2f}"
        rows.append(
            "<tr>"
            f"<td>{labels[after['mode']]}</td><td>{after['n_obs']} × {after['k']}</td>"
            f"<td>{after['n_resamples'] or '—'}</td>"
            f"<td>{before['warm_median_ms']:.2f} → {after['warm_median_ms']:.2f}</td>"
            f"<td>{comparison['warm_speedup']:.2f}×</td><td>{wrapper_text}</td>"
            f"<td>{before['first_use_ms']:.2f} → {after['first_use_ms']:.2f}</td></tr>"
        )
    example = '''import numpy as np
from strategy_inference import test_returns

rng = np.random.default_rng(17)
returns = rng.normal(0.0, 0.01, size=(512, 3))
returns[:, 0] += 0.0005

result = test_returns(
    returns, method="bootstrap",
    names=["strategy_a", "strategy_b", "strategy_c"],
    n_resamples=1999, seed=17, search_complete=True,
)
print(result.global_pvalue)
print(result.adjusted_pvalue)
print(result.decisions)'''
    install = (
        "python -m pip install "
        "https://github.com/Studyer-Tang/strategy-inference/releases/download/"
        "v0.5.0/strategy_inference-0.5.0-py3-none-any.whl"
    )
    source_install = "python -m pip install 'git+https://github.com/Studyer-Tang/strategy-inference.git@v0.5.0'"
    frame = '''import pandas as pd  # Optional: python -m pip install pandas
frame = pd.DataFrame(returns, columns=["carry", "value", "trend"])
result = test_returns(frame, n_resamples=1999, seed=17)'''
    csv_example = '''from strategy_inference import read_returns_csv
table = read_returns_csv("returns.csv", benchmark="benchmark")
result = test_returns(table, n_resamples=1999, seed=17)'''
    export = '''import json
print(json.dumps(result.to_dict(), ensure_ascii=False))
rows = result.records()   # Candidate rows; no optional dependency.
frame = result.to_frame() # Candidate table; requires pandas.'''
    page = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>strategy-inference — Python 均值推断</title>
<meta name="description" content="时间依赖与候选筛选的 Python 均值推断库：安装、API 与可核验的本机性能数据。">
<style>
:root {{ color-scheme: light; background:#fbf8f1; color:#29251f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0 auto; max-width:58rem; padding:2.5rem clamp(1rem,4vw,3rem) 3rem;
font:17px/1.8 Georgia,"Noto Serif CJK SC","Songti SC",serif; }}
main,section {{ min-width:0; }}
h1,h2,h3 {{ line-height:1.3; font-weight:normal; }}
h1 {{ font-size:clamp(2rem,6vw,3rem); margin:.6rem 0 1rem; }}
h2 {{ font-size:1.55rem; margin:2.6rem 0 1rem; }}
h3 {{ font-size:1.13rem; margin:1.7rem 0 .7rem; }}
p {{ margin:.7rem 0 1rem; }}
nav {{ display:flex; flex-wrap:wrap; gap:.4rem 1.1rem; padding:0 0 1rem;
border-bottom:1px solid #d8cebd; }}
a {{ color:#775126; text-underline-offset:.2em; overflow-wrap:anywhere; }}
code {{ font: .87em/1.6 ui-monospace,SFMono-Regular,Consolas,monospace; }}
pre {{ margin:1rem 0; padding:1rem 1.1rem; background:#f0eade;
border-left:2px solid #c8b89e; max-width:100%; overflow-x:auto; }}
pre code {{ display:block; white-space:pre; }}
.table-scroll {{ max-width:100%; overflow-x:auto; }}
table {{ border-collapse:collapse; min-width:850px; width:100%; font-size:.85rem; }}
th,td {{ padding:.55rem .65rem; border-bottom:1px solid #d8cebd; text-align:left; white-space:nowrap; }}
th {{ font-weight:normal; color:#665c4e; }}
.small,footer {{ font-size:.88rem; color:#665c4e; }}
footer {{ border-top:1px solid #d8cebd; margin-top:2.8rem; padding-top:1rem; }}
</style>
</head>
<body>
<nav aria-label="文档导航">
<a href="{TAG_ROOT}/README.md">源码与首页</a>
<a href="{TAG_ROOT}/docs/api.md">API</a>
<a href="{TAG_ROOT}/docs/performance.md">性能说明</a>
<a href="{TAG_ROOT}/docs/methods.md">方法</a>
<a href="{TAG_ROOT}/docs/research.md">研究证据</a>
</nav>
<main>
<header>
<p class="small">Python library · v{VERSION}</p>
<h1>strategy-inference</h1>
<p>时间依赖与候选策略筛选之后的均值推断。输入同期收益矩阵，返回全族检验、列级决定和方法诊断。</p>
</header>
<section aria-labelledby="install">
<h2 id="install">安装与首次调用</h2>
<p>Python 3.10+，核心依赖 NumPy 与 SciPy。直接安装 GitHub v0.5.0 release 的 wheel，无需 clone；尚未发布到 PyPI。</p>
<pre><code>{escape(install)}</code></pre>
<p>也可从对应标签的源码安装：</p>
<pre><code>{escape(source_install)}</code></pre>
<p>下面的模拟数据可直接运行，候选集在生成数据前确定。收益为单期小数，结果不自动年化。</p>
<pre><code>{escape(example)}</code></pre>
</section>
<section aria-labelledby="input">
<h2 id="input">NumPy、DataFrame 与 CSV</h2>
<p>形状为 <code>(T, K)</code>，行是时间、列是候选，一维输入作为单列。要求 T ≥ 8、有限数值和正样本方差。观测须等间隔、同期对齐；基准与成本由使用者扣除。缺失值报错，不自动删行。</p>
<p>DataFrame 自动保留字符串化列名，需安装可选 pandas。输入的行顺序就是时间顺序，不根据索引自动重排。</p>
<pre><code>{escape(frame)}</code></pre>
<p>CSV 可含严格递增的 ISO <code>date</code> 列，其他列为数值。指定基准列会从候选收益中减去并移除；日期检查不推断交易频率。</p>
<pre><code>{escape(csv_example)}</code></pre>
</section>
<section aria-labelledby="methods">
<h2 id="methods">选择方法</h2>
<h3><code>method="bootstrap"</code></h3>
<p>共享时间索引的 stationary bootstrap，默认每轮重新估计 HAC，返回全族和列级调整 p 值。近似需要平稳、弱时间依赖、适当矩及一致尺度估计；渐近讨论固定候选数。短样本或高持久性可能失准。</p>
<h3><code>method="gaussian_ar"</code></h3>
<p>在预先固定的候选集、共同平稳 Gaussian AR(1)、0 ≤ φ &lt; 1、正边际方差下，用参数集合与 GLS 认证提供保守强 FWER 决定。返回参数区间和固定水平决定，不提供连续 p 值。参数信息列与块尺度须事先确定；认证预算不足或信息退化时保守不拒绝，功效可能较低。</p>
<p>调整覆盖输入候选集。隐藏搜索、自适应生成、反复监测后停止及未来市场变化，不能由一张收益矩阵自动解决。普通样本均值只是结果摘要，Gaussian AR 决策采用 GLS。</p>
</section>
<section aria-labelledby="export">
<h2 id="export">结果导出</h2>
<pre><code>{escape(export)}</code></pre>
<p>JSON 保留方法、水平、全族结果、参数集合和诊断；候选表只含名称、均值、拒绝决定及存在时的调整 p 值。完整底层结果在 <code>result.details</code>，参数和字段含义见 <a href="{TAG_ROOT}/docs/api.md">API 说明</a>。</p>
<pre><code>strategy-inference test returns.csv --method bootstrap --seed 17 --output result.json
strategy-inference test returns.csv --method gaussian_ar --output gaussian-ar.json</code></pre>
</section>
<section aria-labelledby="performance">
<h2 id="performance">保存的本机性能</h2>
<p>v0.4.0 与 v0.5.0 的六格串行对照，暖核心时间为 {report['repeats']} 次调用的中位数。时间单位毫秒。统一入口时间单独列出，不能与底层核心时间混用。</p>
<p class="small">窄屏可横向滚动表格查看完整数据；长代码也可在块内横向滚动。</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="可横向滚动的性能表">
<table>
<thead><tr><th>任务</th><th>T × K</th><th>B</th><th>暖核心 0.4 → 0.5</th><th>比值</th><th>新入口暖调用</th><th>首次使用 0.4 → 0.5</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
<p>首次使用是一次“包导入＋接口加载＋首次核心调用”。NumPy 已用于构造输入，因此该列不含 Python/NumPy 启动及数据生成；也不代表完整新进程总耗时。暖调用不含导入和首次证书缓存成本。</p>
<p>环境：{escape(str(report['platform']))}；Python {escape(str(report['python']).split()[0])}；NumPy {escape(str(report['numpy']))}；SciPy {escape(str(report['scipy']))}；BLAS 线程 {escape(str(report['blas_threads']))}。</p>
<p>这些是指定输入、本机串行数据，不是普遍速度或统计校准保证。指纹对照保留 p 值、决定与 Gaussian 证书的一致性检查；浮点区间比较允许已记录的舍入误差。进程峰值 RSS 包含解释器、导入、输入与输出；调用分配峰值也不是同一对象，详见 <a href="{TAG_ROOT}/docs/performance.md">性能说明</a>。</p>
<p><a href="benchmark.json" download>下载完整基准记录</a>。生成前已核对当前 Python 源码与基准脚本哈希；来源绑定不等于对计时数据的可信签名。</p>
</section>
</main>
<footer>
<p>BSD-3-Clause · <a href="{TAG_ROOT}/docs/research.md">历史研究与失败边界</a> · <a href="{TAG_ROOT}/docs/references.bib">方法来源</a></p>
<p>基准记录 SHA-256：<span style="overflow-wrap:anywhere">{digest}</span></p>
</footer>
</body>
</html>
'''
    return page.encode("utf-8")


def build_site(source: Path, destination: Path, *, check: bool = False):
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("Source and destination symlinks are not managed.")
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source.parent) or source.parent.is_relative_to(destination):
        raise ValueError("Benchmark source and destination directories must not overlap.")
    raw, report, hashes = _snapshot(source)
    expected = {"index.html": _render(report, _sha(raw)), "benchmark.json": raw}
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("The site destination must be a directory.")
        for path in destination.iterdir():
            if path.name not in FILES:
                raise ValueError("The managed site contains unexpected files.")
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
                raise ValueError("Managed site files must not be symlinks, hardlinks or directories.")
    if source.read_bytes() != raw or _source_hashes() != hashes:
        raise ValueError("Benchmark evidence or source changed during generation.")
    if report["benchmark_source_sha256"] != _sha((ROOT / "benchmarks/run.py").read_bytes()):
        raise ValueError("Benchmark runner changed during generation.")
    if check:
        if any(
            not (destination / name).is_file() or (destination / name).read_bytes() != contents
            for name, contents in expected.items()
        ):
            raise ValueError("Generated site differs from verified benchmark evidence.")
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in expected.items():
        (destination / name).write_bytes(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "benchmarks/results/library-0.5.json")
    parser.add_argument("--destination", type=Path, default=ROOT / "docs/library")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        build_site(args.source, args.destination, check=args.check)
    except (ValueError, OSError, KeyError, TypeError, SyntaxError) as exc:
        parser.exit(1, f"Library site {'check' if args.check else 'build'} failed: {exc}\n")
    print(f"Library site {'matches saved evidence' if args.check else 'built'}: {args.destination.resolve()}")


if __name__ == "__main__":
    main()
