"""Render three scientific figures and a restrained research report from evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from parametric_replay import _sha

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("figure-1-joint-size", "figure-2-joint-information", "figure-3-joint-power")
METHODS = ("gls_known_budget", "uncertainty", "wilks_scalar", "wilks_joint")
LABELS = {"gls_known": r"Known $\phi$, 5%", "gls_known_budget": r"Known $\phi$, same cutoff",
          "gls_fitted": r"Fitted $\phi$, 5%", "uncertainty": "Single-column cosine F",
          "wilks_scalar": "Single-column multiscale", "wilks_joint": "Joint multiscale (up to 8)"}
COLORS = {"gls_known": "#262c30", "gls_known_budget": "#898071", "gls_fitted": "#a05c42",
          "uncertainty": "#ad875f", "wilks_scalar": "#77917a", "wilks_joint": "#315f75"}


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _index(rows):
    return {(int(row["phase"]), int(row["group"]), float(row["delta"]), row["method"], row["mode"]): row for row in rows}


def _error(rows):
    return np.maximum(0, np.array([[float(row["rate"]) - float(row["low"]) for row in rows],
                                  [float(row["high"]) - float(row["rate"]) for row in rows]]))


def _save(figure, output, name):
    for extension in ("png", "svg", "pdf"):
        metadata = {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight", metadata=metadata)
    plt.close(figure)


def _size(output, index, protocol):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.7, 1, 1]})
    panels = ((1, [g["id"] for g in protocol["groups"] if g["in_scope"]], 0, "reject", "Global null: common AR"),
              (3, protocol["partial_null"]["groups"], 3, "false_reject", "Partial null: strong FWER"),
              (1, [8], 0, "reject", "Heterogeneous AR: outside model"))
    methods = ("gls_fitted", "uncertainty", "wilks_scalar", "wilks_joint")
    for ax, (phase, groups, delta, mode, title) in zip(axes, panels, strict=True):
        for j, method in enumerate(methods):
            rows = [index[(phase, group, delta, method, mode)] for group in groups]
            y = np.arange(len(groups)) + (j - 1.5) * .16
            ax.errorbar([float(row["rate"]) for row in rows], y, xerr=_error(rows), fmt="o", markersize=4,
                        capsize=2, linewidth=.9, color=COLORS[method], label=LABELS[method])
        ax.axvline(.05, linestyle="--", color="#625e56", linewidth=.8)
        ax.set_yticks(range(len(groups)), [f"G{group}" for group in groups])
        ax.set_ylim(len(groups) - .5, -.5)
        maximum = max(float(index[(phase, group, delta, method, mode)]["high"]) for group in groups for method in methods)
        ax.set_xlim(0, max(.1, maximum * 1.08))
        ax.set_title(title, pad=14)
        ax.set_xlabel("Rejection probability")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    figure.subplots_adjust(left=.055, right=.985, bottom=.22, top=.87, wspace=.32)
    _save(figure, output, FIGURES[0])


def _information(output, index, geometry, protocol):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    groups = [g["id"] for g in protocol["groups"] if g["in_scope"]]
    methods = ("uncertainty", "wilks_scalar", "wilks_joint")
    for j, method in enumerate(methods):
        x = np.arange(len(groups)) + (j - 1) * .22
        for ax, mode in zip(axes[:2], ("ci_contains_truth", "phi1_retained"), strict=True):
            rows = [index[(1, group, 0, method, mode)] for group in groups]
            ax.errorbar(x, [float(row["rate"]) for row in rows], yerr=_error(rows), fmt="o", markersize=4,
                        color=COLORS[method], capsize=2, linewidth=.8, label=LABELS[method])
        rows = [geometry[(1, group, 0, method)] for group in groups]
        median = np.array([float(row["median_width"]) for row in rows])
        axes[2].errorbar(x, median,
            yerr=np.array([median - [float(row["q10_width"]) for row in rows],
                           [float(row["q90_width"]) for row in rows] - median]),
            fmt="o", markersize=4, capsize=2, linewidth=.8, color=COLORS[method], label=LABELS[method])
    axes[0].axhline(.995, color="#736b5e", linestyle="--", linewidth=.8)
    for ax, title in zip(axes, ("Coverage of the common coefficient", r"Outer set retains $\phi=1$",
                               "CI width: median and 10–90% range"), strict=True):
        ax.set_xticks(range(len(groups)), [f"G{group}" for group in groups])
        ax.set_ylim(-.025, 1.035)
        ax.set_title(title, pad=14)
        ax.grid(axis="y")
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    coverage_low = min(float(index[(1, group, 0, method, "ci_contains_truth")]["low"])
                       for group in groups for method in methods)
    axes[0].set_ylim(max(0, coverage_low - .001), 1.0015)
    axes[0].set_title("Common-coefficient coverage (zoomed axis)", pad=14)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    figure.subplots_adjust(left=.045, right=.985, bottom=.20, top=.87, wspace=.25)
    _save(figure, output, FIGURES[1])


def _power(output, index):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharex=True, sharey=True)
    titles = (r"G2: $\phi=.5,\ T=256$", r"G3: $\phi=.9,\ T=512$", r"G5: $\phi=.96875,\ T=1024$",
              r"G6: $\phi=.99,\ K=1$", r"G9: $\rho=.995,\ \phi=.9$", r"G10: $\rho=1$, singular")
    methods = ("gls_known", *METHODS)
    for ax, group, title in zip(axes.ravel(), (2, 3, 5, 6, 9, 10), titles, strict=True):
        for method in methods:
            rows = [index[(2, group, delta, method, "signal_reject")] for delta in (2, 3, 6)]
            ax.errorbar((2, 3, 6), [float(row["rate"]) for row in rows], yerr=_error(rows),
                        fmt="s-" if method == "wilks_joint" else "o--" if method == "gls_known_budget" else "o-",
                        markersize=4, capsize=2, linewidth=1.1, color=COLORS[method], label=LABELS[method])
        ax.set_title(title, pad=12)
        ax.set_ylim(-.025, 1.035)
        ax.set_xticks((2, 3, 6))
        ax.grid(axis="y")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for ax in axes[-1]:
        ax.set_xlabel(r"Mean / true SD of sample mean, $\delta$")
    for ax in axes[:, 0]:
        ax.set_ylabel("True signal detection probability")
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    figure.subplots_adjust(left=.06, right=.985, bottom=.17, top=.94, wspace=.18, hspace=.32)
    _save(figure, output, FIGURES[2])


def _pct(row):
    return f"{100 * float(row['rate']):.1f}% [{100 * float(row['low']):.1f}, {100 * float(row['high']):.1f}]"


def _html(output, index, differences, metadata):
    protocol = metadata["protocol"]
    rows = []
    for group in protocol["power"]["groups"]:
        old, new = (index[(2, group, 3, name, "signal_reject")] for name in ("uncertainty", "wilks_joint"))
        pair = differences[(2, group, 3, "uncertainty", "signal_reject")]
        gain = f"{-100 * float(pair['risk_difference']):.1f} [{-100 * float(pair['high']):.1f}, {-100 * float(pair['low']):.1f}]"
        oracle = _pct(index[(2, group, 3, "gls_known_budget", "signal_reject")]) if group != 8 else "无共同 φ"
        scalar = _pct(index[(2, group, 3, "wilks_scalar", "signal_reject")])
        rows.append(f"<tr><td>G{group}</td><td>{oracle}</td><td>{_pct(old)}</td><td>{scalar}</td><td>{_pct(new)}</td><td>{gain}</td></tr>")
    groups = "".join(f"<tr><td>G{g['id']}</td><td>{g['n_obs']}</td><td>{g['k']}</td><td>{g['phi_factor']:g}/{g['phi_idio']:g}</td><td>{g['rho']:g}</td><td>{'异质 AR，保证外' if not g['in_scope'] else '奇异、重复列' if g['rho']==1 else '正负相关、异方差' if g['signed_scaled'] else '共同 AR'}</td></tr>" for g in protocol["groups"])
    downloads = " · ".join(f'<a href="{name}">{name}</a>' for name in ("summary.csv", "paired.csv", "geometry.csv", "metadata.json", "certificates.json", "audit.json", "bounds-audit.json"))
    raw = " · ".join(f'<a href="{cell["key"]}.csv.gz">{cell["key"]}</a>' for cell in metadata["cells"])
    captions = (
        "全零均值、混合真假原假设与异质时间参数分别呈现。混合情形仅把拒绝零均值列计作误报。点态 95% Wilson 区间；理论保证不由这些误差线替代。",
        "共同参数覆盖率、边界保留率与区间宽度分别衡量可靠性、阻塞和信息利用。覆盖率纵轴放大靠近 100% 的范围，虚线为名义 99.5%。宽度误差棒是逐轮分布的 10–90% 范围，不是均值置信区间。G8 无共同 φ，故不画覆盖率；G10 的奇异回退完整保留。",
        "只统计真实信号列 0 的检出。所有方法共用同一噪声、同一 GLS 目标；新旧认证方法的 α、β 和临界值相同。保留近单位根单列及完全重复列的功效不足。",
    )
    figures = [f'<figure><div class="scroll"><a href="{name}.svg"><img src="{name}.png" alt="实验图 {i}" loading="eager"></a></div><figcaption>图 {i}．{caption} <a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></figcaption></figure>' for i, (name, caption) in enumerate(zip(FIGURES, captions, strict=True), 1)]
    status = "冻结协议后的正式评价" if metadata["profile"] == "full" else "开发检查；轮数不足以支持统计结论"
    intro = "；".join(f"G{g}：{_pct(index[(2,g,3,'uncertainty','signal_reject')])} → {_pct(index[(2,g,3,'wilks_joint','signal_reject')])}" for g in (2,3,5))
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>共同时间参数的信息利用 · strategy-inference</title>
<style>*{{box-sizing:border-box}}:root{{color-scheme:light}}body{{margin:0;background:#e8e3da;color:#302b25;font:17px/1.85 Georgia,"Noto Serif SC","Songti SC",serif}}main{{max-width:1140px;margin:34px auto;padding:52px 70px 68px;background:#faf7ee;border-inline:1px solid #d7cdbd;box-shadow:0 4px 22px #594d3615}}header{{border-bottom:1px solid #c8bdac;padding-bottom:23px}}.running{{font-size:12px;letter-spacing:1.3px;color:#756b5c}}h1{{font-size:34px;line-height:1.4;font-weight:500;margin:17px 0}}h2{{font-size:23px;font-weight:500;margin:36px 0 12px}}p{{margin:16px 0}}a{{color:#345d6c;text-underline-offset:4px;text-decoration-color:#a9bbc0}}.note,figcaption{{font-size:14px;color:#726654}}.scroll{{overflow-x:auto}}figure{{margin:27px 0}}img{{display:block;width:100%;min-width:970px;height:auto;background:white}}figcaption{{margin-top:12px;line-height:1.75}}table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}td,th{{text-align:left;padding:9px 11px;border-bottom:1px solid #ddd3c5;white-space:nowrap}}th{{font-weight:500;border-block:1px solid #928570}}pre{{overflow:auto;font:13px/1.7 Menlo,Consolas,monospace;background:#f0ebe0;padding:17px;border-left:2px solid #b5a58f}}.downloads{{overflow-wrap:anywhere}}footer{{border-top:1px solid #c8bdac;padding-top:17px;margin-top:39px;font-size:13px;color:#746855}}@media(max-width:700px){{body{{font-size:16px}}main{{margin:0;padding:29px 20px;box-shadow:none}}h1{{font-size:28px}}h2{{font-size:21px}}}}</style>
<main><header><div class="running">STRATEGY INFERENCE · RESEARCH NOTE IV</div><h1>共同时间参数的信息利用</h1><p>在保留覆盖预算的前提下，时间对比和多个策略方向能减少多少参数不确定性？</p><div class="note">{status}。{sum(cell['n'] for cell in metadata['cells']):,} 份独立阶段噪声；{metadata['records']:,} 条方法记录。α=5%，β=0.5%。</div></header>
<p><strong>δ=3 的信号检出：</strong>{intro}。这些是各格的点估计和点态 95% 区间；新旧方法之差使用同一份噪声配对计算。结果并不表示对所有协方差、时间参数或样本量都有功效改善。</p>
<h2>一、方法改变了什么</h2><p>旧方法在事先指定的单列上使用少量低频对比。新方法固定使用前 min(K,8) 列，在长度 4、16、64 的块上比较块内创新与块均值创新。真实共同 φ 下，这两组散布矩阵是独立 Wishart；其行列式比消去未知列间协方差。多尺度共享一次 β 预算，随后在参数集合的连续域上认证原来的 GLS 均值检验。</p><p>区间临界值来自精确有理矩和 Markov 界，幂阶选择只依赖自由度和预算。这里的 Wishart、Wilks 和投影推断原理均有经典文献依据；本项目提供的是它们在共同时间参数问题中的保守实现、连续参数证书与配对功效评价。</p><p class="note">保证仍要求固定候选集、共同平稳 Gaussian AR(1) 和 0≤φ&lt;1。同期协方差允许任意半正定结构，但选定子集奇异时保守返回无信息集合，不把重复列算成额外方向。数值证书针对输入的 binary64 值；理想 Gaussian 法则的保证不另包含测量或生成舍入误差。</p>
<h2>二、正式设计与错误率</h2><div class="scroll"><table><thead><tr><th>组</th><th>T</th><th>K</th><th>因子 φ / 特有 φ</th><th>ρ</th><th>结构</th></tr></thead><tbody>{groups}</tbody></table></div>{figures[0]}
<p>直接代入 φ 的方法保留为失效参照。已知真实 φ 的两项参照分别使用 5% 临界值和与认证方法相同的保守临界值；它们用于量化参数不确定性的代价。G8 超出共同参数模型，不能把该格的检出率当成可靠功效优势。</p>
<h2>三、参数集合的信息与边界</h2>{figures[1]}<p>单列多尺度与八维联合多尺度的比较衡量跨策略信息的作用；旧单列对比与新单列对比则衡量时间构造的变化。两项比较同时保留 Chernoff 界的保守代价。G3 的单列多尺度在 δ=3 时只有 {_pct(index[(2,3,3,'wilks_scalar','signal_reject')])}，低于旧方法；不能把时间改造单独称作成功。包含 φ=1 或无法完成符号认证时不拒绝，空集合也不用于强行拒绝。</p><p>G9 的高相关仍是正定协方差，理论上具有多个创新方向；G10 完全重复、只有一个方向，新联合方法在此回退。两种情形的差异不能被描述成对任意严重共线数据均有效的数值结论。</p>
<h2>四、功效增益与剩余代价</h2>{figures[2]}<p>下表列出 δ=3 的真实信号检出率，以及联合方法相对旧方法的配对增益。区间单位为百分点，均为点态描述，不作同时显著性主张。近单位根单列仍然困难；新的参数集合没有消除这个限制。</p><div class="scroll"><table><thead><tr><th>组</th><th>已知 φ，同临界值</th><th>旧单列</th><th>新单列</th><th>新联合</th><th>配对增益 / 百分点</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p>G8 的异质时间参数使保证失效：零均值阶段的联合方法误报率为 {_pct(index[(1,8,0,'wilks_joint','reject')])}。参数集合很窄也可能是模型失配的结果，不能把这一格的检出增益视作可靠性改善。</p>
<h2>五、证据与复现</h2><p><a href="../../../../docs/joint-uncertainty.md">完整构造与证明</a> · <a href="../../../../docs/joint-results.md">正式结果解读</a> · <a href="../../../../experiments/joint-uncertainty-protocol.json">冻结实验协议</a> · <a href="../../../../docs/references.bib">文献</a></p><pre>python scripts/joint_report.py --output results/research/joint/full
# 完整重算使用新的空目录：
python scripts/joint_uncertainty.py --profile full --workers 4 --output results/research/joint/reproduced
python scripts/verify_joint_uncertainty.py --output results/research/joint/reproduced
python scripts/verify_joint_bounds.py --output results/research/joint/reproduced
python scripts/joint_report.py --output results/research/joint/reproduced</pre><p class="downloads">{downloads}</p><p class="downloads">逐轮记录：{raw}</p><p class="note">全部记录核对哈希、设计、拒绝掩码、区间、覆盖指标与汇总算术。每格预设的三个噪声样本、全部位移和三套完整行列式/GLS 证书重新生成；其余数值证书不重复运行。另有独立的有理数概率界核验，检查所存尺度的矩存在性、外向根和覆盖预算；它不重新证明行列式多项式。计算源版本：{html.escape(metadata['git_revision'])}。</p><footer>strategy-inference · 固定候选集下的均值推断 · 保留失败情形与模型边界</footer></main></html>'''
    (output / "report.html").write_text(text)


def report(output: Path):
    renderer_digest = _sha(Path(__file__))
    metadata = json.loads((output / "metadata.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    bounds = json.loads((output / "bounds-audit.json").read_text())
    required = {"summary.csv", "paired.csv", "geometry.csv", "certificates.json"} | {f"{cell['key']}.csv.gz" for cell in metadata["cells"]}
    if metadata["status"] != "complete" or set(metadata["output_hashes"]) != required:
        raise ValueError("A complete frozen evidence manifest is required.")
    if audit["status"] != "passed" or audit["metadata_sha256"] != _sha(output / "metadata.json"):
        raise ValueError("A passed audit bound to this evidence is required.")
    if bounds["status"] != "passed" or bounds["metadata_sha256"] != _sha(output / "metadata.json") or bounds["certificates_sha256"] != _sha(output / "certificates.json"):
        raise ValueError("A passed bounds audit bound to this evidence is required.")
    if bounds["verifier_sha256"] != _sha(ROOT / "scripts/verify_joint_bounds.py"):
        raise ValueError("Bounds verifier source differs from its evidence binding.")
    for name, digest in metadata["output_hashes"].items():
        if Path(name).name != name or _sha(output / name) != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")
    index, differences = _index(_read(output / "summary.csv")), _index(_read(output / "paired.csv"))
    geometry = {(int(row["phase"]), int(row["group"]), float(row["delta"]), row["method"]): row for row in _read(output / "geometry.csv")}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7, "grid.color": "#e4e1d9", "grid.linewidth": .55,
        "svg.hashsalt": "joint-uncertainty-report", "savefig.facecolor": "white"})
    _size(output, index, metadata["protocol"])
    _information(output, index, geometry, metadata["protocol"])
    _power(output, index)
    _html(output, index, differences, metadata)
    if _sha(Path(__file__)) != renderer_digest:
        raise ValueError("Renderer source changed during presentation; rebuild from stable source.")
    names = ["report.html", *(f"{stem}.{extension}" for stem in FIGURES for extension in ("png", "svg", "pdf"))]
    provenance = dict(evidence_metadata_sha256=_sha(output / "metadata.json"), audit_sha256=_sha(output / "audit.json"),
        bounds_audit_sha256=_sha(output / "bounds-audit.json"),
        renderer_sha256=renderer_digest, output_hashes={name: _sha(output / name) for name in names})
    (output / "presentation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Three figures rebuilt from {metadata['records']:,} saved records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output)
