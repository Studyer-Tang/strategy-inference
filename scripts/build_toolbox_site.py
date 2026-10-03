"""Render the frozen v0.6 toolbox overview from its saved measurements."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.6.0"
RELEASE_COMMIT = "ffcd83b7043056182fdf7a6e36f1011e11889a34"
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(contents):
    match = re.search(r'^__version__ = "([^"]+)"$', contents.decode("utf-8"), re.MULTILINE)
    if match is None or match[1] != VERSION:
        raise ValueError(f"The frozen toolbox source must have package version {VERSION}.")


def _git_bytes(*arguments):
    """Read release blobs without importing or executing historical code."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *arguments], capture_output=True, check=False
        )
    except OSError as exc:
        raise ValueError(f"Cannot read frozen release {RELEASE_COMMIT}: Git is unavailable.") from exc
    if result.returncode:
        raise ValueError(
            f"Cannot read frozen release {RELEASE_COMMIT}; fetch this commit and its Git objects."
        )
    return result.stdout


def _release_sources():
    """Bind saved evidence to Git release blobs, or a non-Git v0.6 fixture."""
    prefix = "src/strategy_inference/"
    runner_path = "benchmarks/time_series.py"
    if not (ROOT / ".git").exists():
        package = ROOT / prefix
        init = package / "__init__.py"
        if init.is_symlink() or not init.is_file():
            raise ValueError("Release __init__.py must be an ordinary Python file.")
        init_contents = init.read_bytes()
        _version(init_contents)
        paths = sorted(package.glob("*.py"))
        runner = ROOT / runner_path
        if (
            not paths
            or any(path.is_symlink() or not path.is_file() for path in paths)
            or runner.is_symlink()
            or not runner.is_file()
        ):
            raise ValueError("Release source and benchmark runner must be ordinary files.")
        return {path.name: _sha(path) for path in paths}, init_contents, runner.read_bytes()

    if _git_bytes("cat-file", "-t", RELEASE_COMMIT).strip() != b"commit":
        raise ValueError(f"Frozen release {RELEASE_COMMIT} must identify a Git commit.")
    tree = _git_bytes(
        "ls-tree", "-r", "-z", "--full-tree", RELEASE_COMMIT, "--",
        "src/strategy_inference", runner_path,
    )
    hashes, init_contents, runner_contents = {}, None, None
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, encoded_path = entry.split(b"\t", 1)
        mode, kind, _ = metadata.split()
        path = encoded_path.decode("utf-8")
        is_source = (
            path.startswith(prefix) and path.endswith(".py")
            and "/" not in path[len(prefix):]
        )
        if not is_source and path != runner_path:
            continue
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError("Frozen release source and runner must be ordinary Git files.")
        contents = _git_bytes("show", f"{RELEASE_COMMIT}:{path}")
        if is_source:
            hashes[path[len(prefix):]] = hashlib.sha256(contents).hexdigest()
            if path == prefix + "__init__.py":
                init_contents = contents
        else:
            runner_contents = contents
    if not hashes or init_contents is None or runner_contents is None:
        raise ValueError(f"Frozen release {RELEASE_COMMIT} lacks source or benchmark objects.")
    return hashes, init_contents, runner_contents


def _read_performance():
    path = ROOT / "benchmarks/results/time-series-0.6.json"
    raw = path.read_bytes()
    report = json.loads(raw)
    hashes, init_contents, runner_contents = _release_sources()
    if (
        report["schema_version"] != 1
        or report["package_version"] != VERSION
        or report["candidate_source_sha256"] != hashes
        or report["benchmark_source_sha256"] != hashlib.sha256(runner_contents).hexdigest()
    ):
        raise ValueError(
            "Performance record must match the frozen release source and benchmark runner."
        )
    _version(init_contents)
    cases = report["cases"]
    expected = (
        "squared_loss_10000x20",
        "backtest_4096",
        "compare_forecasts_958x2",
        "adaptive_intervals_100000",
        "adaptive_stream_100000",
        "adaptive_stream_1000",
    )
    if (
        tuple(case["case"] for case in cases) != expected
        or report["repeats"] != 5
        or report["warmup_calls"] != 1
        or report["blas_thread_environment"] != THREAD_ENV
    ):
        raise ValueError("The six complete five-repeat benchmark cases are required.")
    for case in cases:
        samples = case["warm_seconds"]
        if (
            len(samples) != 5
            or any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in samples)
            or isinstance(case["warm_median_seconds"], bool)
            or not math.isclose(
                statistics.median(samples), case["warm_median_seconds"], rel_tol=1e-12
            )
            or type(case["traced_peak_bytes"]) is not int
            or case["traced_peak_bytes"] < 0
            or type(case["traced_current_bytes"]) is not int
            or case["traced_current_bytes"] < 0
            or case["traced_current_bytes"] > case["traced_peak_bytes"]
            or case["package_version"] != VERSION
            or case["numpy_version"] != report["numpy"]
            or case["thread_environment"] != THREAD_ENV
        ):
            raise ValueError("Benchmark timings or allocation peaks are inconsistent.")
        settings = case["settings"]
        name = case["case"]
        if name == "squared_loss_10000x20":
            valid = (
                settings["n_obs"] == 10000
                and settings["n_models"] == 20
                and settings["loss"] == "squared"
            )
        elif name in ("backtest_4096", "compare_forecasts_958x2"):
            valid = (
                settings["n_obs"] == 4096
                and settings["models"] == ["naive", "drift", "seasonal_naive_12"]
                and settings["backtest_options"]
                == {
                    "initial_train_size": 256,
                    "window": 256,
                    "horizon": 12,
                    "step": 4,
                }
            )
            if name.startswith("compare"):
                valid = (
                    valid
                    and settings["n_origins"] == 958
                    and settings["n_candidates"] == 2
                    and settings["comparison_options"]
                    == {
                        "baseline": "naive",
                        "lead_time": 1,
                        "n_resamples": 999,
                        "seed": 17,
                    }
                    and settings["existing_backtest_excluded"] is True
                )
        else:
            valid = settings["n_obs"] == (1000 if name.endswith("_1000") else 100000) and settings[
                "options"
            ] == {
                "alpha": 0.1,
                "step_size": 0.1,
                "decay": 0.6,
                "scale": 1.0,
                "initial_quantile": 0.5,
            }
        if not valid:
            raise ValueError("Benchmark settings disagree with the displayed workload.")
    return raw, report


def render(report):
    tag = f"https://github.com/Studyer-Tang/strategy-inference/blob/v{VERSION}"
    install = (
        "python -m pip install https://github.com/Studyer-Tang/strategy-inference/"
        f"releases/download/v{VERSION}/strategy_inference-{VERSION}-py3-none-any.whl"
    )
    example = """import numpy as np
from strategy_inference import (
    backtest, naive_forecast, drift_forecast,
    evaluate_forecasts, compare_forecasts, adaptive_intervals,
)

y = np.random.default_rng(17).normal(size=512).cumsum()
result = backtest(
    y, {"naive": naive_forecast, "drift": drift_forecast},
    initial_train_size=128, window=128, horizon=1,
)
scores = evaluate_forecasts(result, loss="squared")
comparison = compare_forecasts(
    result, baseline="naive", n_resamples=999, seed=17,
)
intervals = adaptive_intervals(
    result.actuals[:, 0], result.forecasts[:, 0, 0], scale=1.,
)
print(scores.records())
print(comparison.records())
print(intervals.coverage)"""
    labels = (
        "逐观测 squared loss · 10,000 × 20",
        "三基线滚动预测 · T=4096 / H=12",
        "模型比较 · 958 起点 / 2 候选 / B=999",
        "在线区间批量回放 · 100,000 步",
        "在线区间流式 · 100,000 步",
        "在线区间流式 · 1,000 步",
    )
    rows = "".join(
        f"<tr><td>{label}</td><td>{case['warm_median_seconds'] * 1000:.3f}</td>"
        f"<td>{case['traced_peak_bytes'] / 1024:.2f}</td></tr>"
        for label, case in zip(labels, report["cases"], strict=True)
    )
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>strategy-inference · 时间序列评估与不确定性</title>
<meta name="description" content="Python 时序工具库：滚动评估、预测评分、模型比较、在线区间与统计推断。">
<style>
:root{{color-scheme:light;background:#fbf8f1;color:#29251f}}*{{box-sizing:border-box}}
body{{margin:auto;max-width:58rem;padding:2.5rem clamp(1rem,4vw,3rem) 3rem;font:17px/1.8 Georgia,"Songti SC",serif}}
h1,h2{{line-height:1.3;font-weight:normal}}h1{{font-size:clamp(2rem,6vw,3rem);margin:.6rem 0 1rem}}h2{{font-size:1.5rem;margin:2.5rem 0 1rem}}
nav{{display:flex;flex-wrap:wrap;gap:.4rem 1.1rem;border-bottom:1px solid #d8cebd;padding-bottom:1rem}}
a{{color:#775126;text-underline-offset:.2em;overflow-wrap:anywhere}}p{{margin:.8rem 0 1rem}}
code{{font:.85em/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}}
pre{{padding:1rem 1.1rem;background:#f0eade;border-left:2px solid #c8b89e;overflow-x:auto;max-width:100%}}
pre code{{white-space:pre}}pre.install code{{white-space:pre-wrap;overflow-wrap:anywhere}}
.table-scroll{{overflow-x:auto;max-width:100%}}table{{border-collapse:collapse;width:100%;min-width:650px;font-size:.9rem}}
th,td{{padding:.65rem;border-bottom:1px solid #d8cebd;text-align:left}}th{{font-weight:normal;color:#665c4e}}
.small,footer{{font-size:.88rem;color:#665c4e}}footer{{border-top:1px solid #d8cebd;margin-top:2.8rem;padding-top:1rem}}
</style>
</head>
<body>
<nav><a href="{tag}/README.md">源码</a><a href="{tag}/docs/time-series.md">时序 API</a><a href="{tag}/docs/api.md">均值检验</a><a href="{tag}/docs/toolbox-roadmap.md">路线图与文献</a><a href="{tag}/docs/research.md">研究证据</a></nav>
<main>
<p class="small">Python · v{VERSION} · 第一阶段</p>
<h1>strategy-inference</h1>
<p>用于时间序列的滚动评估、模型比较与在线不确定性。模型通过简单 callable 接入，预测起点、步长、损失与检验结果一起保留。</p>
<h2>可以完成什么</h2>
<div class="table-scroll" tabindex="0" role="region" aria-label="功能表，可横向滚动"><table>
<thead><tr><th>环节</th><th>已有功能</th><th>输出</th></tr></thead><tbody>
<tr><td>切分与预测</td><td>扩展／滚动窗、gap、多步预测、三种基线</td><td>origin × lead × model 预测</td></tr>
<tr><td>评估</td><td>平方、绝对、pinball、中央区间 score</td><td>逐步长损失摘要</td></tr>
<tr><td>模型比较</td><td>固定基准与步长的共享 max bootstrap</td><td>平均改善、调整 p 值、诊断</td></tr>
<tr><td>在线区间</td><td>单步衰减步长 conformal 跟踪</td><td>先发出区间，再反馈更新</td></tr>
<tr><td>统计推断</td><td>HAC、stationary bootstrap、Gaussian AR 参数集合</td><td>均值与同时决定</td></tr>
</tbody></table></div>
<p class="small">窄屏可横向滚动表格与代码。模型训练、数据频率和适用条件由具体任务确定；不自动填补或删掉缺失观测。</p>
<h2>安装与一条完整流程</h2>
<p>Python 3.10+。核心依赖 NumPy 与 SciPy；表格导出可选 pandas，绘图可选 Matplotlib。GitHub 安装包无需克隆仓库，尚未发布到 PyPI。</p>
<pre class="install"><code>{html.escape(install)}</code></pre>
<pre><code>{html.escape(example)}</code></pre>
<p>例子是模拟 random walk，展示调用流程。平均损失与已实现覆盖不是有效性的独立验证，也不保证未来表现。完整模拟例子见 <a href="{tag}/examples/time_series.py">time_series.py</a>。</p>
<h2>方法条件直接写在接口里</h2>
<p>滚动预测只传入本折历史，但 callback 的外部状态仍须避免未来资料。多步目标可重叠，比较接口按一个事先固定的步长检验，绝不把重叠步长展平为更多样本。</p>
<p>Bootstrap 模型比较依赖平稳、弱依赖及合适矩条件。在线区间基于 <a href="https://proceedings.mlr.press/v235/angelopoulos24a.html">ICML 2024 的衰减步长跟踪器</a>，只支持单步顺序反馈；其理想递推保证是历史平均覆盖，不是每个时点或多步路径的同时覆盖。空集与无界区间明确保存，binary64 运算没有舍入认证。</p>
<h2>近期方法与下一阶段</h2>
<p>后续重点是成熟标签队列、多步误差相关性、条件覆盖与序贯模型集合。<a href="https://proceedings.mlr.press/v267/areces25a.html">ICML 2025 在线优化 conformal</a>在条件分位数具有线性结构等条件下研究条件覆盖；<a href="https://arxiv.org/abs/2410.13115v2">AcMCP 的 2026 修订稿</a>处理多步误差相关性及随步长增加的覆盖代价。这两项目前是路线图，未接入公共接口。</p>
<p>数据诊断、经典模型接入、panel、变点、序贯模型比较及跨步长联合推断的范围与验收要求见 <a href="{tag}/docs/toolbox-roadmap.md">路线图</a>。已有算法的复现与工程组织不作为研究创新宣称。</p>
<h2 id="performance">本机效率记录</h2>
<p>串行、BLAS 线程环境设为 1、五次热调用中位数；耗时为毫秒，跟踪分配峰值为 KiB。输入生成、导入与结果指纹不计时；模型比较的既有回测也不计时。峰值包含新输出、不包含已有输入和导入。</p>
<div class="table-scroll" tabindex="0" role="region" aria-label="性能表，可横向滚动"><table><thead><tr><th>任务</th><th>耗时 / ms</th><th>峰值 / KiB</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>环境：{html.escape(report["platform"])}；Python {html.escape(report["python"].split()[0])}；NumPy {html.escape(report["numpy"])}。流式 tracker 保存常数个状态；批量接口保留整段区间，空间为 O(T)。这些实测不能作为通用速度承诺或统计校准。</p>
<p><a href="benchmark-0.6.json" download>完整原始计时与源代码哈希</a> · <a href="../v0.5.0/">v0.5.0 冻结性能对照</a></p>
</main><footer>BSD-3-Clause · 完整条件、返回值与异常见 <a href="{tag}/docs/time-series.md">时序 API</a>。</footer>
</body></html>
'''.encode()


def build(*, check=False):
    raw, report = _read_performance()
    expected = {"index.html": render(report), "benchmark-0.6.json": raw}
    destination = ROOT / f"docs/library/v{VERSION}"
    if any(path.is_symlink() for path in (ROOT, ROOT / "docs", ROOT / "docs/library", destination)):
        raise ValueError("Overview directories must not be symlinks.")
    paths = {name: destination / name for name in expected}
    for path in paths.values():
        if path.is_symlink() or (
            path.exists() and (not path.is_file() or path.stat().st_nlink != 1)
        ):
            raise ValueError("Managed overview files must be ordinary files with one link.")
    for name, content in expected.items():
        path = paths[name]
        if check:
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError("The toolbox overview differs from its saved evidence.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(check=args.check)
    print("Toolbox overview matches saved evidence." if args.check else "Toolbox overview built.")


if __name__ == "__main__":
    main()
