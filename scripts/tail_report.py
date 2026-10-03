"""Render the three diagnostic figures from hash-verified simulation evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import PercentFormatter

from strategy_inference.tail import ar1_tail_factor

METHODS = (
    "oracle", "bartlett", "tail_known_lrv", "tail_plugin_lrv",
    "tail_known_finite", "tail_plugin_finite", "mean_unbiased_oracle",
)
LABELS = (
    "Known variance", "Bartlett", r"Tail: known $\phi$, LRV", r"Tail: fitted $\phi$, LRV",
    r"Tail: known $\phi$, finite T", r"Tail: fitted $\phi$, finite T", "Mean-unbiased scale",
)
COLORS = ("#252525", "#999999", "#6c8c98", "#a75a39", "#35586a", "#5c7451", "#846586")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "grid.color": "#e7e4de", "grid.linewidth": .5, "svg.hashsalt": "tail-diagnostics",
        "savefig.facecolor": "white",
    })


def _save(figure, output: Path, name: str) -> None:
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def build(output: Path) -> None:
    metadata = json.loads((output / "metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("Only a completed study can be rendered.")
    for name, digest in metadata["output_hashes"].items():
        if _sha(output / name) != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")
    summary, moments = _read(output / "summary.csv"), _read(output / "moments.csv")
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.65))
    single = [row for row in moments if int(row["group"]) <= 4]
    phi = np.array([float(row["phi"]) for row in single])
    expectation = np.array([float(row["expectation"]) for row in single])
    target = np.array([float(row["target"]) for row in single])
    lags = [int(row["lags"]) for row in single]
    known_lrv = np.array([float(ar1_tail_factor(p, lag+1)) for p, lag in zip(phi, lags, strict=True)])
    known_finite = np.array([
        float(ar1_tail_factor(p, lag+1, target="finite_sample", n_obs=200))
        for p, lag in zip(phi, lags, strict=True)
    ])
    for ratios, label, color in (
        (expectation/target, "Bartlett", COLORS[1]),
        (expectation*known_lrv/target, "Known AR: LRV tail", COLORS[2]),
        (expectation*known_finite/target, "Known AR: finite-T tail", COLORS[4]),
        (np.ones_like(phi), "Mean-unbiased scale", COLORS[6]),
    ):
        axes[0].plot(phi, ratios, "o-", color=color, label=label, linewidth=1.3, markersize=4)
    axes[0].set(xlabel=r"AR coefficient $\phi$", ylabel=r"$E[\widehat v] / \{T\,Var(\bar X)\}$", title="(a) Exact scale expectation")
    axes[0].axhline(1, color="#333333", linewidth=.7, linestyle=":")
    axes[0].legend(frameon=False, fontsize=8.4, loc="lower left")
    axes[1].plot(phi, [math.sqrt(float(row["variance"]))/float(row["expectation"]) for row in single], "o-", color=COLORS[4])
    axes[1].set(xlabel=r"AR coefficient $\phi$", ylabel="Coefficient of variation", title="(b) Random scale remains variable")
    for axis in axes:
        axis.grid(axis="y")
        axis.set_xticks([0, .7, .9, .99], ["0", ".70", ".90", ".99"])
    figure.tight_layout(w_pad=2.5)
    _save(figure, output, "figure-1-tail-bias")

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), sharey=True)
    for axis, rho, panel in zip(axes, (0.0, .35), ("a", "b"), strict=True):
        for index, method in enumerate(METHODS):
            rows = [
                row for row in summary if int(row["group"]) == 5 and row["method"] == method
                and (float(row["rho"]) == rho or int(row["k"]) == 1)
            ]
            rows.sort(key=lambda row: int(row["k"]))
            y = np.array([float(row["rate"]) for row in rows])
            errors = np.array([
                [float(row["rate"])-float(row["ci_low"]) for row in rows],
                [float(row["ci_high"])-float(row["rate"]) for row in rows],
            ])
            axis.errorbar(np.arange(3)+.025*(index-3), y, yerr=errors,
                          marker="o", markersize=3.7, linewidth=1,
                          color=COLORS[index], label=LABELS[index], capsize=2)
        axis.axhline(.05, color="#333333", linestyle=":", linewidth=.9)
        axis.set(xlabel="Candidate count K", title=rf"({panel}) Common correlation $\rho={rho:g}$", ylim=(0, 1))
        axis.set_xticks([0, 1, 2], ["1", "20", "100"])
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(axis="y")
    axes[0].set_ylabel("Null rejection probability")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.8, frameon=False)
    figure.tight_layout(rect=(0, .19, 1, 1), w_pad=2.5)
    _save(figure, output, "figure-2-tail-selection")

    with gzip.open(output / "g05-k100-r35.csv.gz", "rt") as stream:
        records = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True, sharey=True)
    density_palette = LinearSegmentedColormap.from_list(
        "visible_greys", plt.get_cmap("Greys")(np.linspace(.22, .95, 128))
    )
    for axis, method, title in zip(
        axes, ("bartlett", "tail_plugin_finite"),
        ("(a) Bartlett winner", "(b) Fitted finite-T tail winner"), strict=True,
    ):
        ratios = np.array([float(row[f"{method}_variance_ratio"]) for row in records])
        true_z = np.array([float(row[f"{method}_statistic"]) for row in records]) * np.sqrt(ratios)
        plotted = axis.hexbin(true_z, ratios, gridsize=28, mincnt=1, cmap=density_palette, norm=LogNorm())
        axis.axhline(1, color=COLORS[3], linewidth=.8, linestyle=":")
        critical = float(records[0]["critical"])
        boundary = np.linspace(0, max(0, float(true_z.max())), 100)
        axis.plot(boundary, (boundary/critical)**2, color=COLORS[4], linestyle="--", linewidth=.8)
        axis.set(xlabel="Selected mean / true standard error", ylabel=r"$\widehat v_{\hat j}/\{T\,Var(\bar X_{\hat j})\}$", title=title)
        figure.colorbar(plotted, ax=axis, label="Replicates", shrink=.85, pad=.02)
    figure.tight_layout(w_pad=2.1)
    _save(figure, output, "figure-3-tail-winner")

    heading = "有限样本尺度、尾部修正与策略筛选"
    warning = (
        "流程检查：每格只有 64 轮，不用于研究结论。" if metadata["profile"] == "quick" else
        "固定协议的机制诊断：每格 5000 轮。全部方法使用已知相关矩阵的 Gaussian 最大值临界值；只有真实方差参照拥有相应的精确尺度。"
    )
    figures = (
        ("figure-1-tail-bias", "1. 偏差修正与随机尺度", "左图由 Gaussian 二次型恒等式精确计算期望；右图显示相对随机波动。确定性的乘法修正不会改变变异系数。使用 T=200、固定带宽。"),
        ("figure-2-tail-selection", "2. 策略筛选后的误报", "T=512、自相关系数 0.9。误差条为点态 95% Wilson 区间，虚线为 5%。K=1 的分布与共同相关参数无关，因此两栏共用该参照格点。"),
        ("figure-3-tail-winner", "3. 被选候选的尺度", "T=512、K=100、共同相关系数 0.35。每个方法按自己的统计量选择候选；横轴使用该候选的真实均值标准误，纵轴为方差估计与有限样本真值的比值。横轴为正且低于蓝色曲线的点对应拒绝；水平点线为真实方差。每轮数据可由随机地址重新生成。"),
    )
    sections = "".join(
        f'<section><h2>{title}</h2><figure><img src="{name}.svg" alt="{title}"><figcaption>{caption}</figcaption></figure><p class="downloads">'
        + " · ".join(f'<a href="{name}.{extension}">{extension.upper()}</a>' for extension in ("png", "svg", "pdf"))
        + "</p></section>" for name, title, caption in figures
    )
    keys = list(dict.fromkeys(row["cell"] for row in summary))
    table = ""
    for key in keys:
        rows = {row["method"]: row for row in summary if row["cell"] == key}
        first = rows["oracle"]
        table += f'<tr><th><a href="{key}.csv.gz">T={first["n_obs"]}, φ={float(first["phi"]):.4f}<br>K={first["k"]}, ρ={first["rho"]}</a></th>'
        for method in METHODS:
            row = rows[method]
            table += f'<td>{100*float(row["rate"]):.2f}%<small>{100*float(row["ci_low"]):.2f}–{100*float(row["ci_high"]):.2f}</small></td>'
        table += "</tr>"
    labels_html = ["真实方差", "Bartlett", "已知 φ·LRV", "估计 φ·LRV", "已知 φ·有限 T", "估计 φ·有限 T", "均值无偏参照"]
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{heading}</title>
<style>body{{margin:0;background:#f7f5ef;color:#282621;font:17px/1.8 Georgia,"Songti SC",serif}}main{{max-width:1030px;margin:38px auto;padding:42px 46px;background:#fffefb}}h1{{font-size:29px;font-weight:500;line-height:1.5}}h2{{font-size:22px;font-weight:500;margin-top:40px}}a{{color:#35586a;text-underline-offset:3px}}figure{{margin:20px 0}}img{{width:100%;height:auto}}figcaption,small,.downloads{{font-size:14px;color:#655f54}}small{{display:block;white-space:nowrap}}.table{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{padding:9px 8px;border-bottom:1px solid #ddd7ca;text-align:right}}th:first-child{{text-align:left;font-weight:400;white-space:nowrap}}footer{{border-top:1px solid #ccc4b4;margin-top:35px;padding-top:18px;font-size:14px}}@media(max-width:650px){{main{{margin:0;padding:24px 18px}}h1{{font-size:24px}}body{{font-size:16px}}}}</style></head><body><main><h1>{heading}</h1>
<p>{warning}</p><p>这里复现 Liu–Chan 的固定带宽 AR(1) 尾部修正公式，并将总体截断、有限样本中心化、参数拟合与候选筛选分开诊断。修正估计量的期望，不等于校准随机分母统计量的尾部。</p>
<p>本报告没有使用未知参数的可实施联合临界值，未复现作者的自动带宽实验，也未建立新方法的普遍有效性。已知参数与均值无偏参照用于辨认误差来源。</p>{sections}
<section><h2>完整零均值结果</h2><p>数值下方为点态 95% Monte Carlo 区间；行标题可下载该格点的逐轮记录。成对拒绝差异另存，未经多重比较调整，不用于确认性显著声明。</p><div class="table"><table><thead><tr><th>生成模型</th>{''.join(f'<th>{label}</th>' for label in labels_html)}</tr></thead><tbody>{table}</tbody></table></div></section>
<footer><p><a href="summary.csv">误报汇总</a> · <a href="paired.csv">配对比较</a> · <a href="moments.csv">精确矩</a> · <a href="metadata.json">运行与哈希记录</a></p><p>公式来源：<a href="https://arxiv.org/html/2605.15596v1">Liu–Chan，JASA 2026</a>。<a href="../../../../docs/tail-mechanism.md">机制命题与证明</a>。原 v0.2.0 结果继续保留。</p><p>冻结计算版本：<code>{html.escape(metadata['git_revision'][:12])}</code>；图由保存的原始记录生成。</p></footer></main></body></html>'''
    (output / "report.html").write_text(document)
    sources = {"metadata.json": _sha(output / "metadata.json"), "render_script": _sha(Path(__file__))}
    presentation = {
        "status": "complete", "input_hashes": sources,
        "output_hashes": {path.name: _sha(path) for path in sorted(output.iterdir()) if path.suffix in (".png", ".svg", ".pdf", ".html")},
    }
    (output / "presentation.json").write_text(json.dumps(presentation, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
