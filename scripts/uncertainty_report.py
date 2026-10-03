"""Rebuild three scientific figures and a book-style report from saved evidence."""

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
METHODS = ("gls_known", "gls_known_budget", "gls_fitted", "uncertainty")
LABELS = {"gls_known": r"Known $\phi$, 5%", "gls_known_budget": r"Known $\phi$, same cutoff",
          "gls_fitted": r"Fitted $\phi$, 5%", "uncertainty": r"Unknown $\phi$, certified"}
CHINESE = {"gls_known": "已知 φ，5%", "gls_known_budget": "已知 φ，相同保守临界值",
           "gls_fitted": "代入估计 φ，5%", "uncertainty": "未知 φ，区间证书"}
COLORS = {"gls_known": "#262c30", "gls_known_budget": "#8a8174", "gls_fitted": "#a15c3a", "uncertainty": "#306379"}
FIGURES = ("figure-1-uncertainty-size", "figure-2-uncertainty-boundary", "figure-3-uncertainty-power")


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _index(rows):
    return {(int(row["phase"]), int(row["group"]), float(row["delta"]), row["method"], row["mode"]): row for row in rows}


def _verify(output, metadata):
    if metadata["status"] != "complete":
        raise ValueError("A completed computation is required.")
    required = {"summary.csv", "paired.csv", "certificates.json"} | {f"{cell['key']}.csv.gz" for cell in metadata["cells"]}
    if set(metadata["output_hashes"]) != required:
        raise ValueError("Evidence manifest differs from the protocol.")
    for name, digest in metadata["output_hashes"].items():
        if Path(name).name != name or _sha(output / name) != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")
    audit = json.loads((output / "audit.json").read_text())
    if audit["status"] != "passed" or audit.get("metadata_sha256") != _sha(output / "metadata.json"):
        raise ValueError("The independent arithmetic/certificate audit must pass first.")


def _style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
                         "grid.color": "#e6e3dc", "grid.linewidth": .55,
                         "svg.hashsalt": "parameter-uncertainty-report", "savefig.facecolor": "white"})


def _save(figure, output, name):
    for extension in ("png", "svg", "pdf"):
        meta = {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight", metadata=meta)
    plt.close(figure)


def _error(rows):
    return np.maximum(0, np.array([[float(row["rate"]) - float(row["low"]) for row in rows],
                                   [float(row["high"]) - float(row["rate"]) for row in rows]]))


def _size(output, index, metadata):
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    panels = ((1, [1, 2, 3, 4, 5, 6, 7], 0, "Global null: common AR", "reject"),
              (3, [3, 4, 7], 3, "Partial null: strong FWER", "false_reject"),
              (1, [8], 0, "Heterogeneous AR: outside model", "reject"))
    for ax, (phase, groups, delta, title, mode) in zip(axes, panels, strict=True):
        for position, method in enumerate(METHODS):
            rows = [index[(phase, group, delta, method, mode)] for group in groups
                    if (phase, group, delta, method, mode) in index]
            if not rows:
                continue
            y = np.arange(len(rows)) + (position - 1.5) * .14
            ax.errorbar([float(row["rate"]) for row in rows], y, xerr=_error(rows), fmt="o", markersize=4,
                        color=COLORS[method], capsize=2, linewidth=.9, label=LABELS[method])
        ax.axvline(.05, color="#66615b", linestyle="--", linewidth=.8)
        ax.set_yticks(range(len(groups)), [f"G{group}" for group in groups])
        ax.invert_yaxis()
        ax.set_title(title, pad=13)
        ax.set_xlabel("Rejection probability")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x")
        maximum = max(float(index[(phase, group, delta, method, mode)]["high"]) for group in groups for method in METHODS
                      if (phase, group, delta, method, mode) in index)
        ax.set_xlim(0, max(.10, maximum * 1.07))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.01))
    figure.subplots_adjust(left=.06, right=.985, bottom=.19, top=.88, wspace=.32)
    _save(figure, output, FIGURES[0])


def _boundary(output, index):
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    groups = list(range(1, 9))
    for ax, mode, title in zip(axes, ("ci_contains_truth", "phi1_retained"),
                               (r"Coverage of true $\phi$ (G8: factor $\phi$ only)", r"Outer confidence set retains $\phi=1$"), strict=True):
        rows = [index[(1, group, 0, "confidence_set", mode)] for group in groups]
        x = np.arange(len(groups))
        colors = ["#306379"] * 7 + ["#a15c3a"]
        ax.bar(x, [float(row["rate"]) for row in rows], width=.65, color=colors, alpha=.84)
        ax.errorbar(x, [float(row["rate"]) for row in rows], yerr=_error(rows), fmt="none", color="#282522", capsize=3, linewidth=.8)
        if mode == "ci_contains_truth":
            ax.axhline(.995, color="#6f675e", linestyle="--", linewidth=.85)
        ax.set_ylim(0, 1.045)
        ax.set_xticks(x, [f"G{group}" for group in groups])
        ax.set_title(title, pad=12)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    figure.subplots_adjust(left=.07, right=.985, bottom=.13, top=.86, wspace=.24)
    _save(figure, output, FIGURES[1])


def _power(output, index):
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True, sharey=True)
    for ax, group, title in zip(axes.ravel(), (2, 3, 5, 6),
                                (r"G2: $T=256,\ \phi=.5,\ K=20$", r"G3: $T=512,\ \phi=.9,\ K=20$",
                                 r"G5: $T=1024,\ \phi=.96875,\ K=20$", r"G6: $T=200,\ \phi=.99,\ K=1$"), strict=True):
        for method in METHODS:
            deltas = (0, 1, 2, 3, 6)
            rows = [index[(1 if delta == 0 else 2, group, delta, method, "reject")] for delta in deltas]
            ax.errorbar(deltas, [float(row["rate"]) for row in rows], yerr=_error(rows),
                        color=COLORS[method], linestyle="--" if method == "gls_known_budget" else "-",
                        marker="s" if method == "uncertainty" else "o", markersize=4, capsize=2,
                        linewidth=1.2, label=LABELS[method])
        row = index[(1, group, 0, "confidence_set", "phi1_retained")]
        ceiling = 1 - float(row["rate"])
        ax.axhline(ceiling, color="#306379", linestyle=":", linewidth=1)
        ax.axhspan(1 - float(row["high"]), 1 - float(row["low"]), color="#306379", alpha=.07)
        ax.set_title(title, pad=12)
        ax.set_xlim(-.15, 6.2)
        ax.set_ylim(0, 1.03)
        ax.set_xticks((0, 1, 2, 3, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y")
    for ax in axes[-1]:
        ax.set_xlabel(r"Mean shift / true SD of sample mean, $\delta$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Global rejection probability")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.002))
    figure.subplots_adjust(left=.08, right=.985, bottom=.115, top=.94, hspace=.35, wspace=.16)
    _save(figure, output, FIGURES[2])


def _pct(row):
    return f"{100 * float(row['rate']):.1f}% [{100 * float(row['low']):.1f}, {100 * float(row['high']):.1f}]"


def _html(output, index, differences, metadata):
    protocol = metadata["protocol"]
    groups = protocol["groups"]
    independent = sum(cell["n"] for cell in metadata["cells"])
    title = "未知时间参数与检验功效"
    group_table = "".join(f"<tr><td>G{g['id']}</td><td>{g['n_obs']}</td><td>{g['k']}</td><td>{g['phi_factor']:g} / {g['phi_idio']:g}</td><td>{g['rho']:g}</td><td>{'正负相关、不同尺度' if g['signed_scaled'] else '共同 AR' if g['in_scope'] else '异质 AR，保证外'}</td></tr>" for g in groups)
    costs = []
    for group in (2, 3, 5, 6):
        oracle = index[(2, group, 3, "gls_known", "reject")]
        budget = index[(2, group, 3, "gls_known_budget", "reject")]
        unknown = index[(2, group, 3, "uncertainty", "reject")]
        paired = differences[(2, group, 3, "gls_known", "reject")]
        diff = f"{100 * float(paired['risk_difference']):.1f} [{100 * float(paired['low']):.1f}, {100 * float(paired['high']):.1f}]"
        costs.append(f"<tr><td>G{group}</td><td>{_pct(oracle)}</td><td>{_pct(budget)}</td><td>{_pct(unknown)}</td><td>{diff}</td></tr>")
    summary_table = "".join(f"<tr><td>G{g['id']}</td><td>{_pct(index[(1,g['id'],0,'uncertainty','reject')])}</td><td>{_pct(index[(1,g['id'],0,'confidence_set','ci_contains_truth')])}</td><td>{_pct(index[(1,g['id'],0,'confidence_set','phi1_retained')])}</td></tr>" for g in groups)
    download = " · ".join(f'<a href="{html.escape(name)}">{html.escape(name)}</a>' for name in
                          ("summary.csv", "paired.csv", "metadata.json", "certificates.json", "audit.json"))
    raw = " · ".join(f'<a href="{cell["key"]}.csv.gz">{cell["key"]}</a>' for cell in metadata["cells"])
    figure_html = []
    captions = (
        "全零均值、混合真假原假设与模型失配分别列出。第二栏只统计零均值策略被拒绝，不把检出真实信号算成误报。误差线为点态 95% Wilson 区间。",
        "置信集合的覆盖与边界保留是不同指标。G8 没有共同 φ，第一栏仅核对因子参数，不能解释为合法 nuisance 覆盖。右栏接近 100% 时，均值检验受到严重阻塞。",
        "δ=0 使用独立零均值阶段，正 δ 共用同一份噪声。虚线横线及浅色带表示由零均值阶段估计的本方法边界功效上限与点态 95% 区间；不是所有有效检验的上限。",
    )
    for number, (name, caption) in enumerate(zip(FIGURES, captions, strict=True), 1):
        figure_html.append(f'<figure><div class="figure-scroll"><a href="{name}.svg"><img src="{name}.png" alt="实验图 {number}" loading="eager"></a></div><figcaption>图 {number}．{caption} <a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></figcaption></figure>')
    quick = "开发检查，轮数不足以作统计结论。" if metadata["profile"] == "quick" else "预设正式实验。"
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · strategy-inference</title>
<style>:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#e9e4db;color:#302c26;font:17px/1.85 Georgia,"Noto Serif SC","Songti SC",serif}}main{{max-width:1080px;margin:35px auto;padding:55px 72px 70px;background:#faf7ef;box-shadow:0 4px 24px #4d413215;border-left:1px solid #d9d1c3;border-right:1px solid #d9d1c3}}header{{border-bottom:1px solid #cfc5b5;padding-bottom:24px}}.running{{font-size:12px;letter-spacing:1.4px;color:#776b59}}h1{{font-size:35px;font-weight:500;line-height:1.45;margin:16px 0}}h2{{font-size:23px;font-weight:500;margin:38px 0 14px}}p{{margin:16px 0}}a{{color:#365b69;text-decoration-color:#acbdc0;text-underline-offset:4px}}.note,figcaption{{font-size:14px;color:#746854}}figure{{margin:27px 0}}.figure-scroll{{overflow-x:auto}}img{{display:block;width:100%;min-width:900px;height:auto;background:white}}figcaption{{margin-top:11px;line-height:1.7}}.table-scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{padding:9px 11px;border-bottom:1px solid #ddd4c7;text-align:left;white-space:nowrap}}th{{font-weight:500;border-top:1px solid #8f8370;border-bottom:1px solid #8f8370}}pre{{font:13px/1.7 Menlo,Consolas,monospace;background:#f0ebe0;padding:16px;overflow-x:auto;border-left:2px solid #b8a991}}code{{font-size:.86em}}footer{{border-top:1px solid #cfc5b5;margin-top:40px;padding-top:16px;font-size:13px;color:#766953}}.downloads{{overflow-wrap:anywhere}}@media(max-width:700px){{body{{font-size:16px}}main{{margin:0;padding:30px 20px;box-shadow:none}}h1{{font-size:28px}}h2{{font-size:21px}}}}</style>
<main><header><div class="running">STRATEGY INFERENCE · RESEARCH NOTE III</div><h1>{title}</h1><p>把参数不确定性纳入检验，会换来多少可靠性，又付出多少检出能力？</p><div class="note">{quick} {independent:,} 份独立阶段噪声；{metadata['records']:,} 条方法记录。α=5%，参数置信预算 β=0.5%。</div></header>
<h2>一、检验究竟保证什么</h2><p>假定每列是平稳 Gaussian AR(1)，共享未知时间参数 0≤φ&lt;1，列间同期协方差任意，均值和边际尺度未知。先在事先固定的第 0 列构造精确 F 置信集合，再对集合中的所有参数认证 GLS 均值检验超过保守临界值。它对输入候选集提供强族错误率控制：即使部分策略确有正均值，拒绝任何零均值或负均值策略的概率仍不超过 α。</p><p>置信集合和均值检验可以使用同一份数据，不要求二者独立。概率预算为一次 β 加 K 次边际尾概率；横截面协方差无需估计。数值证书用精确整数和有理数，未决区间保留、未决检验不拒绝。返回固定水平下的决定，不编造连续 p 值。</p><p class="note">保证针对上述理想实数 Gaussian 模型；精确计算证书针对传入的 binary64 观测。数据测量或浮点生成的舍入分布误差并未被另一条有限样本定理覆盖。异质时间参数、非 Gaussian 创新和自适应候选生成均不在本轮保证内。</p>
<h2>二、独立评价与错误率</h2><div class="table-scroll"><table><thead><tr><th>组</th><th>T</th><th>K</th><th>因子 φ / 特有 φ</th><th>ρ</th><th>结构</th></tr></thead><tbody>{group_table}</tbody></table></div>{figure_html[0]}
<p>比较已知真实 φ 的 GLS、采用相同保守临界值的已知 φ GLS、直接代入估计 φ 的 GLS，以及置信集合认证方法。前两项是知道未知参数的参照，不是可直接用于未知参数数据的竞争方法。所有方法使用相同观测；没有在正式数据上调参数或校准实际尺寸。</p>
<h2>三、参数集合与接近单位根的阻塞</h2>{figure_html[1]}<p>若置信集合外包仍包含 φ=1，非退化数据的 GLS t 统计量在 φ→1 时趋于 0。程序因此无法在整个集合上认证拒绝。该集合对理想数据的常数均值平移不变；增加均值信号并不会修复这个阻塞。因此，本程序的功效对任意信号强度均不超过“集合排除 1”的概率。</p><div class="table-scroll"><table><thead><tr><th>组</th><th>认证检验误报率</th><th>参数覆盖率</th><th>保留 φ=1</th></tr></thead><tbody>{summary_table}</tbody></table></div><p class="note">表内括号均为点态 95% Wilson 区间。G8 的覆盖列只是失配诊断；它没有共同真实 φ。置信集合覆盖和程序的理论错误率保证由证明建立，不能由这些点估计替代。</p>
<h2>四、功效的代价</h2>{figure_html[2]}<p>先从 5% 已知参数参照移到同一保守临界值，衡量置信预算与最多一个自由度取整的代价；再从相同临界值参照移到未知参数方法，衡量参数集合最不利检验与保守数值包围的代价。下表以 δ=3 为例，最后一列是同一噪声上的配对差及点态 95% 区间，单位为百分点。</p><div class="table-scroll"><table><thead><tr><th>组</th><th>已知 φ，5%</th><th>已知 φ，同临界值</th><th>未知 φ，认证</th><th>总功效损失 / 百分点</th></tr></thead><tbody>{''.join(costs)}</tbody></table></div><p>这是一项有明确保守代价的可靠性方案。若近单位根时功效大幅下降，应报告下降及其原因，不能把“有效”写成“普遍优于其他方法”。G8 的失配结果保留在原始记录中，超出共同 AR 假设时不承诺错误率。</p>
<h2>五、文献与本轮工作的范围</h2><p>Dufour（1990）与 Dufour–Neifar（2002）已提出参数置信集合和投影检验；Glazer–Stark（2026）讨论保守置信集合的可靠计算。这些是方法依据。本轮工作的具体内容是固定有理投影的 F 集合、对连续参数域的符号证书、任意横截面协方差下同时均值检验的实现，以及冻结协议下的功效损失分解。它不宣称新的通用推断原理或最优功效。</p><p><a href="../../../../docs/parameter-uncertainty.md">完整构造、命题与证明</a> · <a href="../../../../docs/uncertainty-results.md">结果解读</a> · <a href="../../../../experiments/parameter-uncertainty-protocol.json">冻结实验协议</a> · <a href="../../../../docs/references.bib">文献</a></p>
<h2>六、复现与证据</h2><pre>python scripts/uncertainty_report.py --output results/research/uncertainty/full
# 重新计算需使用新的空目录：
python scripts/parameter_uncertainty.py --profile full --output results/research/uncertainty/reproduced
python scripts/verify_parameter_uncertainty.py --output results/research/uncertainty/reproduced
python scripts/uncertainty_report.py --output results/research/uncertainty/reproduced</pre><p class="downloads">{download}</p><p class="downloads">逐轮记录：{raw}</p><p class="note">审计核对全部哈希、拒绝掩码、区间包含关系、比例区间及配对计算；每格三个数据集的所有信号位移与完整证书重新生成。未对其余全部证书再次运行计算。计算源版本：{html.escape(metadata['git_revision'])}。</p><footer>strategy-inference · 固定候选集下的均值推断 · 所有原始失败结果保留</footer></main></html>'''
    (output / "report.html").write_text(text)


def report(output: Path):
    metadata = json.loads((output / "metadata.json").read_text())
    _verify(output, metadata)
    index = _index(_read(output / "summary.csv"))
    differences = _index(_read(output / "paired.csv"))
    _style()
    _size(output, index, metadata)
    _boundary(output, index)
    _power(output, index)
    _html(output, index, differences, metadata)
    names = ["report.html", *(f"{stem}.{ext}" for stem in FIGURES for ext in ("png", "svg", "pdf"))]
    presentation = {"evidence_metadata_sha256": _sha(output / "metadata.json"),
                    "audit_sha256": _sha(output / "audit.json"),
                    "renderer_sha256": _sha(Path(__file__)), "output_hashes": {name: _sha(output / name) for name in names}}
    (output / "presentation.json").write_text(json.dumps(presentation, indent=2) + "\n")
    print(f"Three figures and report rebuilt from {metadata['records']:,} saved records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output)
