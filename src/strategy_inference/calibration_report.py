"""Render saved calibration results as a short research note."""

import csv
import hashlib
import html
import json
from pathlib import Path

from .report import document

REPOSITORY = "https://github.com/Studyer-Tang/strategy-inference"


def write_calibration_report(output: str | Path) -> Path:
    output = Path(output)
    metadata = json.loads((output / "run-metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("The calibration run must be complete before rendering its report.")
    if metadata.get("study") != "calibration":
        raise ValueError("This renderer requires a calibration study.")
    config, assessment = metadata["resolved_protocol"], metadata["assessment"]
    if assessment["status"] not in ("smoke_only", "passed", "failed"):
        raise ValueError("Unknown calibration assessment status.")
    if (metadata["profile"] == "quick") != (assessment["status"] == "smoke_only"):
        raise ValueError("A quick run must be labelled smoke_only, never a calibration assessment.")
    n_cells = assessment["n_cells"]
    threshold = assessment["maximum_false_positive_rate"]
    alpha = config["alpha"]

    def read(name):
        actual_hash = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if metadata.get("file_sha256", {}).get(name) != actual_hash:
            raise ValueError(f"Saved table hash does not match its run record: {name}")
        with (output / name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if any(row["method"] == "oracle" and row["process"] != "gaussian_ar" for row in rows):
            raise ValueError("The Gaussian reference cannot be used for a non-Gaussian process.")
        return rows

    dependence, selection, size = [
        read(name)
        for name in ("figure-1-dependence.csv", "figure-2-selection.csv", "figure-3-size.csv")
    ]
    selected = (
        ("单策略，φ = 0.8", [row for row in dependence if float(row["phi"]) == 0.8]),
        ("50 条候选，φ = 0.5", [row for row in selection if int(row["n_strategies"]) == 50]),
        ("重尾 AR，20 条候选", [row for row in size if row["process"] == "student_ar"]),
        ("GARCH，20 条候选", [row for row in size if row["process"] == "garch"]),
    )
    summary = []
    for label, records in selected:
        cells = []
        for method in ("fixed", "resampled", "oracle"):
            row = next((row for row in records if row["method"] == method), None)
            cells.append(
                "<td>—</td>"
                if row is None
                else f"<td>{float(row['rate']):.2%}<br><span class='note'>[{float(row['ci_low']):.2%}, {float(row['ci_high']):.2%}]</span></td>"
            )
        summary.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    if assessment["status"] == "smoke_only":
        interpretation = "本页为 quick 流程检查，重复次数不足以作校准判断。"
    elif assessment["status"] == "passed":
        interpretation = f"这次评价通过了预设的有限基准门槛：{n_cells} 个场景的误报率同时上界均不超过 {threshold:.0%}。这不等于对真实金融数据普遍达到名义 {alpha:.0%}。"
    else:
        interpretation = f"这次评价未通过预设的证据门槛：{len(assessment['failed_cells'])} / {n_cells} 个场景的误报率同时上界超过 {threshold:.0%}。这不表示这些场景的真实误报率必然超过 {threshold:.0%}；新方法的有限样本偏差与模拟精度仍需考虑。研究版本不提供普适的 {alpha:.0%} 校准保证。"
    figure_specs = (
        (
            "figure-1-calibration-dependence",
            "一　时间依赖与尺度估计",
            "左图比较两种 Bootstrap 尺度与已知协方差参考；右图展示 IID 和 HAC 的误报。高斯模型参考使用真实有限样本均值方差，真实收益审计中没有这一信息。",
            ("figure-1-dependence.csv",),
        ),
        (
            "figure-2-calibration-selection",
            "二　筛选之后的全族检验",
            "候选真实均值全部为零。Bootstrap 检验整族；IID 和 HAC 读取同一个最大 HAC 统计量胜者的未调整结果。两个面板采用不同纵轴，保留全部计数与区间。",
            ("figure-2-selection.csv",),
        ),
        (
            "figure-3-calibration-processes",
            "三　误报与检出能力",
            "左图检验零均值；右图仅在第一列植入正均值。拒绝概率是全族检验的检出能力，不是正确识别信号列的概率。不同过程的实际误报率不同，功效曲线不能解释为同尺寸的公平排名。",
            ("figure-3-size.csv", "figure-3-power.csv"),
        ),
    )
    figures = []
    for stem, title, caption, csv_files in figure_specs:
        links = [f'<a href="{stem}.{suffix}">{suffix.upper()}</a>' for suffix in ("pdf", "svg")]
        links.extend(f'<a href="{name}">{name}</a>' for name in csv_files)
        figures.append(
            f'<h2>{title}</h2><figure><img src="{stem}.png" alt="{html.escape(title)}"><figcaption>{caption}<br>{" · ".join(links)}</figcaption></figure>'
        )
    holds = []
    for row in read("holdout.csv"):
        if row["method"] == "resampled":
            holds.append(
                f"<tr><td>{html.escape(row['process'])}</td><td>{html.escape(row['n_obs'])}</td><td>{html.escape(row['phi'])}</td><td>{html.escape(row['cross_corr'])}</td><td>{float(row['rate']):.2%}</td><td>{float(row['size_upper_simultaneous']):.2%}</td></tr>"
            )
    content = f"""
    <header><div class="eyebrow">STRATEGY INFERENCE · RESEARCH NOTE 02</div>
    <h1>时间依赖与策略筛选<br>如何影响显著性</h1>
    <p class="note">有限样本评价 · 固定尺度与逐次重新学生化</p></header>
    <p>同一个正均值结果，在独立观测、时间相关观测和筛选之后，统计证据的含义并不相同。
    本研究以已知真实均值的模拟数据检验这些差别，并检查联合重抽样自身的误报偏差。</p>
    <p>本轮只改变 Bootstrap 统计量的分母：旧方法固定使用原样本 HAC 标准误，新方法在每次重抽样中重新估计同一滞后阶的 HAC。
    两种方法使用相同数据、相同行索引和相同原样本统计量。块长、带宽、评价格点与验收规则均在正式评价前固定。</p>
    <p class="note">主实验 T = {config["n_obs"]}，每个场景 {config["n_mc"]:,} 次重复，每次 {config["n_bootstrap"]:,} 次重抽样；
    名义水平 {alpha:.0%}，正式评价种子 {html.escape(str(metadata["seed"]))}。开发试验与本轮使用不同随机流。</p>
    <h2>主要结果</h2><div class="scroll"><table><thead><tr><th>零均值场景</th><th>固定尺度</th><th>重新学生化</th><th>高斯模型参考</th></tr></thead>
    <tbody>{"".join(summary)}</tbody></table></div>
    <p class="note">方括号为点态 95% Wilson Monte Carlo 区间。高斯参考仅在已知生成参数的高斯场景存在；
    重尾与 GARCH 不套用高斯最大值分布。</p>
    <p><strong>{interpretation}</strong></p>
    {"".join(figures)}
    <h2>额外评价格点</h2>
    <p>下表每个场景有 10 条零均值候选，使用原样本确定的样本量规则与块长。最后一列是涵盖全部 {n_cells} 个主实验与额外零均值格点的
    Bonferroni 调整单侧 Clopper–Pearson 上界。它衡量模拟中的误报率，与前面的点态 Wilson 区间用途不同。</p>
    <div class="scroll"><table><thead><tr><th>过程</th><th>T</th><th>φ</th><th>创新相关</th><th>新方法误报率</th><th>同时上界</th></tr></thead><tbody>{"".join(holds)}</tbody></table></div>
    <h2>偏差诊断与解释边界</h2>
    <p>除拒绝计数外，本轮保存了高斯场景中 HAC 方差与真实有限样本均值方差的比值，以及 stationary-bootstrap 均值的精确条件方差比值。
    这些诊断帮助区分标准误估计偏差与重抽样分布偏差。条件方差的“精确”只针对当前数据和抽样机制，不是对未知总体方差的精确估计。</p>
    <p>块长敏感性以另外的随机流评价 0.5、1、2、4 倍默认块长，结果完整保存在附表。它未用于挑选主图设置。
    重新学生化不构成二阶正确性的证明，三个过程也不能代替对全部金融时间序列的有效性证明。</p>
    <p>审计输入应是同期、等间隔、已扣成本的基准调整收益。候选族必须由研究者定义。
    隐藏试验、自适应构造策略、数据时间错误、反复查看结果和交易成本遗漏，都不能从一张收益矩阵自动恢复。</p>
    <h2>复现与原始记录</h2>
    <p><code>strategy-inference reproduce --study calibration --profile {html.escape(metadata["profile"])} --output results/calibration/{html.escape(metadata["profile"])}</code></p>
    <p class="note">运行用时 {metadata["elapsed_seconds"]:.1f} 秒。计算源码版本 <code>{html.escape(str(metadata.get("git_revision") or "未记录"))}</code>。
    <a href="run-metadata.json">协议、种子、环境与哈希</a> · <a href="holdout.csv">额外格点计数</a> ·
    <a href="block-sensitivity.csv">块长敏感性</a> · <a href="variance-diagnostics.csv">方差诊断</a></p>
    <footer><a href="{REPOSITORY}">项目源码</a> · <a href="{REPOSITORY}/blob/main/docs/methods.md">方法说明</a> ·
    <a href="{REPOSITORY}/blob/main/docs/results.md">结果解读</a>。本项目是现有统计方法的实现与模拟研究，不声明原创定理。</footer>
    """
    path = output / "report.html"
    path.write_text(document("时间依赖与策略筛选 · 独立校准评价", content), encoding="utf-8")
    return path
