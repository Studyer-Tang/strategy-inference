"""Build figures and a research report from verified Gaussian replay evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "oracle_gaussian", "gaussian_known", "gaussian_fitted", "replay_known",
    "replay_phi_fitted", "replay_rho_fitted", "replay_fitted", "replay_frozen", "gls_known",
)
LABELS = {
    "oracle_gaussian": "True-variance Gaussian",
    "gaussian_known": "Gaussian: true mean correlation",
    "gaussian_fitted": "Gaussian: fitted correlation",
    "replay_known": "Replay: true DGP",
    "replay_phi_fitted": "Replay: fitted temporal parameter",
    "replay_rho_fitted": "Replay: fitted correlation",
    "replay_fitted": "Replay: fitted DGP",
    "replay_frozen": "Replay: frozen multiplier",
    "gls_known": r"Known-$\phi$ GLS + Bonferroni",
}
CHINESE = {
    "oracle_gaussian": "真实方差 Gaussian",
    "gaussian_known": "真实均值相关临界值",
    "gaussian_fitted": "估计相关 Gaussian",
    "replay_known": "真实 DGP 重放",
    "replay_phi_fitted": "仅拟合时间参数",
    "replay_rho_fitted": "仅拟合相关参数",
    "replay_fitted": "完整拟合重放",
    "replay_frozen": "冻结校正因子",
    "gls_known": "已知 φ 的 GLS",
}
COLORS = dict(zip(METHODS, (
    "#262626", "#8c8071", "#a35c38", "#303f4b", "#aa8670",
    "#8b7695", "#356d88", "#667e50", "#704d7b",
), strict=True))
FIGURES = ("figure-1-replay-size", "figure-2-replay-fit", "figure-3-replay-power")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _verify(output: Path, metadata: dict) -> None:
    if metadata.get("status") != "complete":
        raise ValueError("Only completed replay evidence can be rendered.")
    manifest = metadata.get("output_hashes", {})
    required = {"summary.csv", "paired.csv", "calibration.json", "inner-snapshots.json"}
    required.update(f"{cell['key']}.csv.gz" for cell in metadata["cells"])
    if not required.issubset(manifest):
        raise ValueError("The evidence hash manifest is incomplete.")
    presentation = {"report.html", "presentation.json"}
    presentation.update(f"{name}.{ext}" for name in FIGURES for ext in ("png", "svg", "pdf"))
    for name, digest in manifest.items():
        if Path(name).name != name or name in presentation or name == "metadata.json":
            raise ValueError(f"Invalid evidence filename: {name}")
        if _sha(output / name) != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")


def _index(rows: list[dict]) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = (int(row["phase"]), int(row["group"]), float(row["delta"]),
               row["method"], row["mode"])
        if key in result:
            raise ValueError(f"Duplicate summary row: {key}")
        result[key] = row
    return result


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "grid.color": "#e8e6e0", "grid.linewidth": .55,
        "svg.hashsalt": "parametric-replay-report", "savefig.facecolor": "white",
    })


def _save(figure, output: Path, name: str) -> None:
    for extension in ("png", "svg", "pdf"):
        metadata = {"Date": None} if extension == "svg" else (
            {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        )
        figure.savefig(output / f"{name}.{extension}", dpi=200,
                       bbox_inches="tight", metadata=metadata)
    plt.close(figure)


def _run_label(metadata: dict) -> str:
    settings = metadata["settings"]
    label = "Development check" if metadata["profile"] == "quick" else "Prespecified experiment"
    return f"{label}: R={settings['null_replicates']}, B={settings['inner_draws']}"


def _errors(rows: list[dict]) -> np.ndarray:
    errors = np.array([[float(row["rate"]) - float(row["low"]) for row in rows],
                       [float(row["high"]) - float(row["rate"]) for row in rows]])
    if np.any(errors < -1e-12):
        raise ValueError("A saved interval does not contain its estimated rejection rate.")
    # Wilson endpoints can differ from 0 or 1 by machine rounding.
    return np.maximum(errors, 0)


def _size_figure(output: Path, metadata: dict, index: dict, groups: dict) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharex=True, sharey=True)
    selected = [index[1, group, 0.0, method, "reject"]
                for group in (2, 5, 7) for method in METHODS
                if not (group == 7 and method == "gls_known")]
    upper = min(1.0, max(.15, math.ceil(10 * max(float(row["high"]) for row in selected)) / 10))
    for axis, identifier, panel in zip(axes, (2, 5, 7), "abc", strict=True):
        group = groups[identifier]
        for position, method in enumerate(METHODS):
            if identifier == 7 and method == "gls_known":
                continue
            row = index[1, identifier, 0.0, method, "reject"]
            rate = float(row["rate"])
            axis.errorbar(rate, len(METHODS) - 1 - position, xerr=_errors([row]),
                          fmt="o", markersize=4.5, capsize=2.5, linewidth=1,
                          color=COLORS[method])
        axis.axvline(metadata["alpha"], color="#49443e", linestyle=":", linewidth=.9)
        parameters = (rf"$\phi={group['phi_factor']:g},\ \rho={group['rho']:g}$"
                      if identifier != 7 else r"$\phi_{\rm factor}=.98,\ \phi_{\rm idio}=.30$")
        axis.set(xlim=(0, upper), xlabel="Null rejection probability",
                 title=f"({panel}) G{identifier}: T={group['n_obs']}, K={group['k']}\n{parameters}")
        axis.set_ylim(-.6, len(METHODS) - .4)
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(axis="x")
    axes[0].set_yticks(np.arange(len(METHODS)), [LABELS[name] for name in reversed(METHODS)])
    axes[0].tick_params(axis="y", length=0, pad=9)
    figure.suptitle(_run_label(metadata), fontsize=11, y=.99)
    figure.tight_layout(rect=(0, 0, 1, .94), w_pad=1.7)
    _save(figure, output, FIGURES[0])


def _fits(output: Path, metadata: dict, identifiers: tuple[int, ...]) -> dict[int, dict]:
    result = {}
    cells = {int(cell["group"]): cell for cell in metadata["cells"] if cell["phase"] == 1}
    for identifier in identifiers:
        rows = [row for row in _read(output / f"p1-g{identifier:02d}.csv.gz")
                if row["method"] == "oracle_gaussian" and float(row["delta"]) == 0]
        if len(rows) != cells[identifier]["n"] or len({row["replicate"] for row in rows}) != len(rows):
            raise ValueError(f"Invalid replicate count in G{identifier} fit evidence.")
        result[identifier] = {
            field: np.array([float(row[field]) for row in rows])
            for field in ("phi_fit", "rho_fit")
        }
        if not all(np.isfinite(values).all() for values in result[identifier].values()):
            raise ValueError(f"Nonfinite fitted parameters in G{identifier} evidence.")
    return result


def _ecdf(axis, values: np.ndarray, right: float, *, left: float = 0.0, **kwargs) -> None:
    ordered = np.sort(values)
    axis.step(np.r_[left, ordered, right], np.r_[0.0, np.arange(1, len(values) + 1) / len(values), 1.0],
              where="post", **kwargs)


def _fit_figure(output: Path, metadata: dict, groups: dict) -> None:
    fits = _fits(output, metadata, (2, 5, 6, 7))
    palette = {2: "#a35c38", 5: "#356d88", 6: "#667e50", 7: "#704d7b"}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.25), sharey=True)
    ratios = {identifier: (1 - fits[identifier]["phi_fit"]) / (1 - groups[identifier]["phi_factor"])
              for identifier in (2, 5, 6)}
    if any(np.any(values <= 0) for values in ratios.values()):
        raise ValueError("Temporal gap ratios must be positive for the logarithmic axis.")
    left = min(.8, .95 * min(float(values.min()) for values in ratios.values()))
    right = max(1.15, max(float(values.max()) for values in ratios.values()) * 1.05)
    for identifier, values in ratios.items():
        _ecdf(axes[0], values, right, left=left, color=palette[identifier], linewidth=1.5,
              label=rf"G{identifier}: $\phi={groups[identifier]['phi_factor']:g}$, K={groups[identifier]['k']}")
    axes[0].axvline(1, color="#49443e", linestyle=":", linewidth=.9)
    axes[0].set(xscale="log", xlabel=r"$(1-\widehat\phi_{\mathrm{common}})/(1-\phi)$ (log scale)",
                ylabel="Empirical cumulative probability", title="(a) Temporal gap estimation")
    axes[0].set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    axes[0].set_xlim(left, right)
    axes[0].minorticks_off()
    for identifier in (5, 6, 7):
        _ecdf(axes[1], fits[identifier]["rho_fit"], 1.0, color=palette[identifier],
              linewidth=1.5, label=f"G{identifier}")
    cells = {int(cell["group"]): cell for cell in metadata["cells"] if cell["phase"] == 1}
    mean_rho = cells[7]["correlation_of_means"]
    axes[1].axvline(groups[7]["rho"], color="#49443e", linestyle=":", linewidth=.9,
                    label=r"Instantaneous $\rho=0.35$")
    axes[1].axvline(mean_rho, color="#8c8071", linestyle="--", linewidth=.9,
                    label=rf"G7: corr(means)={mean_rho:.3f}")
    axes[1].set(xlim=(0, 1), xlabel=r"Projected Pearson estimate $\widehat\rho$",
                title="(b) Cross-candidate correlation")
    for axis in axes:
        axis.set_ylim(0, 1.025)
        axis.grid(axis="y")
        axis.legend(frameon=False, fontsize=9, loc="lower right")
    figure.suptitle(_run_label(metadata), fontsize=11, y=.99)
    figure.tight_layout(rect=(0, 0, 1, .94), w_pad=2.5)
    _save(figure, output, FIGURES[1])


def _power_figure(output: Path, metadata: dict, protocol: dict, index: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    methods = ("gaussian_fitted", "replay_known", "replay_fitted", "replay_frozen")
    shifts = [0.0, *protocol["power"]["standardized_mean_shifts"]]
    for column, identifier in enumerate((4, 7)):
        selected = (*methods, "gls_known") if identifier == 4 else methods
        for row_number, mode in enumerate(("reject", "matched_reject")):
            axis = axes[row_number, column]
            for position, method in enumerate(selected):
                rows = [index[1 if delta == 0 else 2, identifier, float(delta), method, mode]
                        for delta in shifts]
                rate = np.array([float(row["rate"]) for row in rows])
                offset = (position - (len(selected) - 1) / 2) * .022
                axis.errorbar(np.array(shifts) + offset, rate, yerr=_errors(rows), fmt="o",
                              linewidth=1.2, markersize=4, capsize=2, color=COLORS[method],
                              label=LABELS[method], linestyle="--" if method == "gls_known" else "-")
            axis.axhline(metadata["alpha"], color="#49443e", linestyle=":", linewidth=.8)
            panel = "abcd"[row_number * 2 + column]
            model = "Common temporal parameter" if identifier == 4 else "Temporal misspecification"
            axis.set(title=f"({panel}) G{identifier}: {model}", ylim=(0, 1),
                     xlim=(-.15, max(shifts) + .15), xticks=shifts)
            axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            axis.grid(axis="y")
            if column == 0:
                axis.set_ylabel("Nominal rejection probability" if row_number == 0
                                else "Size-matched rejection probability")
            if row_number == 1:
                axis.set_xlabel(r"Mean shift $\delta$ in true standard-error units")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.suptitle(_run_label(metadata), fontsize=11, y=.995)
    figure.tight_layout(rect=(0, .11, 1, .95), w_pad=2.5, h_pad=2)
    _save(figure, output, FIGURES[2])


def _relative(output: Path, path: Path) -> str:
    return html.escape(Path(os.path.relpath(path, output)).as_posix(), quote=True)


def _percent(value: str | float) -> str:
    return f"{100 * float(value):.2f}%"


def _interval(row: dict) -> str:
    return f"{_percent(row['rate'])}<small>{_percent(row['low'])}–{_percent(row['high'])}</small>"


def _resolution_note(output: Path, metadata: dict, calibration: dict) -> str:
    draws = metadata["settings"]["inner_draws"]
    minimum, last_bin = 1 / (draws + 1), draws / (draws + 1)
    methods = ("replay_fitted", "replay_frozen")
    counts = {}
    for phase in (0, 1):
        with gzip.open(output / f"p{phase}-g07.csv.gz", "rt", encoding="utf-8", newline="") as stream:
            counts[phase] = {method: [0, 0] for method in methods}
            for row in csv.DictReader(stream):
                if row["method"] in methods and float(row["delta"]) == 0:
                    values = counts[phase][row["method"]]
                    values[0] += int(int(row["exceedances"]) == 0)
                    values[1] += 1
    rows = []
    for method in methods:
        cutoff = calibration["7"][method]["cutoff"]
        if cutoff <= last_bin:
            continue
        ceiling = min(1.0, max(0.0, (draws + 1) * (1 - cutoff)))
        calibration_count, calibration_n = counts[0][method]
        null_count, null_n = counts[1][method]
        rows.append(
            f"<tr><th scope=\"row\">{CHINESE[method]}</th>"
            f"<td>{calibration_count} / {calibration_n}<small>{_percent(calibration_count / calibration_n)}</small></td>"
            f"<td>{null_count} / {null_n}<small>{_percent(null_count / null_n)}</small></td>"
            f"<td>{cutoff:.9f}</td><td>{_percent(ceiling)}</td></tr>"
        )
    note = (f"<p>内部模拟只有 B={draws} 轮，因此 rank p-value 的最小值为 1/(B+1)={minimum:g}。"
            "更小的尾概率无法通过当前秩分辨率区分；独立 jitter 只在相同 p-value 的分数区间内排序。</p>")
    if rows:
        note += (
            "<div class=\"table\"><table><caption>G7：最小秩 p-value 与最后分数区间</caption>"
            "<thead><tr><th scope=\"col\">方法</th><th scope=\"col\">校准阶段达到最小值</th>"
            "<th scope=\"col\">独立零均值评价达到最小值</th><th scope=\"col\">匹配阈值 c</th>"
            "<th scope=\"col\">给定 c 的总体拒绝概率上限</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<p>这些阈值落在最后一个 jitter 分数区间，即 c&gt;B/(B+1)。只有 p-value 已达到最小值的重复才可能拒绝，"
            "且还需要独立均匀扰动越过阈值。给定本次固定 c，总体拒绝概率至多为 min{1,(B+1)(1−c)}；"
            "图中的有限外层模拟频率仍有抽样波动。G7 的尺寸匹配功效因此不能全部归因于统计量或错误生成模型，"
            "还包含既定有限 B 与 jitter 比较规则的分辨率瓶颈。它不说明任意模拟轮数、连续 p-value 或实际检验的一般功效上限。</p>"
        )
    return note


def _document(output: Path, metadata: dict, protocol: dict, index: dict, calibration: dict) -> str:
    settings, alpha = metadata["settings"], metadata["alpha"]
    quick = metadata["profile"] == "quick"
    run_kind = "开发流程检查" if quick else "预先固定协议的实验"
    warning = ("本页是 quick 开发运行，仅检查计算与展示流程，不作为正式研究结果。"
               if quick else "本页使用 full 配置；方法、格点、随机种子与评价规则在计算前固定。")
    groups = {group["id"]: group for group in protocol["groups"]}
    cells = {cell["group"]: cell for cell in metadata["cells"] if cell["phase"] == 1}
    g7_mean_rho = cells[7]["correlation_of_means"]
    null_rows = []
    for method in METHODS:
        values = []
        for identifier in groups:
            row = index.get((1, identifier, 0.0, method, "reject"))
            values.append(f"<td>{_interval(row) if row else '不适用'}</td>")
        null_rows.append(f"<tr><th scope=\"row\">{CHINESE[method]}</th>{''.join(values)}</tr>")
    design_rows = []
    for identifier, group in groups.items():
        cell = cells[identifier]
        row = index[1, identifier, 0.0, "replay_fitted", "reject"]
        screen = (f"{_percent(row['family_upper'])}<small>"
                  f"{'≤7%' if int(row['passes_tolerance']) else '>7%'}</small>"
                  if row["family_upper"] else "错配压力格点")
        design_rows.append(
            f"<tr><th scope=\"row\">G{identifier}</th><td>{group['n_obs']} / {group['k']}</td>"
            f"<td>{group['phi_factor']:g} / {group['phi_idio']:g}</td>"
            f"<td>{group['rho']:g} / {cell['correlation_of_means']:.4f}</td>"
            f"<td>{_percent(cell['projected_fraction'])}</td><td>{screen}</td></tr>"
        )
    calibration_rows = []
    for identifier, methods in calibration.items():
        for method in METHODS:
            if method not in methods:
                continue
            entry = methods[method]
            low, high = entry["conditional_size_interval"]
            calibration_rows.append(
                f"<tr><th scope=\"row\">G{identifier} · {CHINESE[method]}</th>"
                f"<td>{entry['rank']} / {entry['n']}</td><td>{entry['cutoff']:.6f}</td>"
                f"<td>{_percent(entry['unconditional_size'])}</td>"
                f"<td>{_percent(low)}–{_percent(high)}</td></tr>"
            )
    definitions = (
        ("oracle_gaussian", "使用真实有限 T 均值方差标准化各列均值，再用真实均值相关下的 Gaussian 最大值临界值。它同时知道真实尺度和相关，是诊断参照。"),
        ("gaussian_known", "使用原来的逐列拟合补尾与 HAC 最大值统计量；只有 Gaussian 临界值中的均值相关取真实值。“已知”不表示统计量内 φ 已知。"),
        ("gaussian_fitted", "保持同一最大值统计量，将 Gaussian 临界值的相关参数换成中心化 Pearson 平均相关的投影估计。"),
        ("replay_known", "从真实零均值 DGP 生成每段数据，完整重算逐列 φ、HAC、有限目标补尾因子和最大值；使用包含同值的 Monte Carlo 秩 p-value。G7 保留真实因子与个体成分的不同时间参数。"),
        ("replay_phi_fitted", "生成时将时间参数换成共同拟合 φ，横截面混合权重使用真实瞬时 ρ；统计量完整重算。"),
        ("replay_rho_fitted", "生成时保留真实时间参数，只把横截面混合权重换成拟合 ρ；统计量完整重算。"),
        ("replay_fitted", "生成时共同 φ 和 ρ 都使用原样本估计，每段模拟数据完整重算统计量。固定 K、mild persistence 下有一阶证明，未知参数下没有有限样本交换性。"),
        ("replay_frozen", "使用拟合生成模型，仍重算每段模拟数据的 HAC，只冻结观察样本的逐列补尾乘数。它用于比较随机乘数重拟合的作用。"),
        ("gls_known", "使用真实共同 φ 对每列 Gaussian 数据做 GLS Student t 检验，再用 Bonferroni 阈值。它改变均值估计器和分母；G7 没有共同时间参数，因此省略。"),
    )
    definitions_html = "".join(
        f"<dt>{CHINESE[method]}<code>{method}</code></dt><dd>{definition}</dd>"
        for method, definition in definitions
    )
    captions = (
        ("1. 零均值下的临界值校准", "G2：单列强持久性；G5：100 个相关候选；G7：时间结构错配。"
         "点为独立外层模拟的拒绝频率，横线为点态 95% Wilson 区间，竖虚线为 5%。所有区间完整显示，G7 省略 GLS。"),
        ("2. 时间参数与相关参数的拟合", "仅使用 phase 1 的独立零均值评价样本，每轮只取一次参数估计。左图使用对数横轴保留全部正的持久性缺口比值，真实目标为 1；"
         "右图显示投影后的 Pearson 平均相关。G5、G6 的瞬时相关与均值相关都为 0.35；"
         f"G7 的瞬时相关为 0.35，均值相关为 {g7_mean_rho:.4f}，两条参照线有不同含义。"),
        ("3. 名义功效与独立尺寸匹配", "只有第 0 列具有正均值，δ 表示以其真实有限 T 均值标准误计量的信号大小。"
         "δ=0 来自 phase 1 独立零均值评价，δ=1、2、3 来自 phase 2。上排使用各方法原始 5% 检验规则；"
         "下排使用 phase 0 的独立零均值校准阈值。误差条为点态 95% Wilson 区间；G4 另列已知 φ 的 GLS 参照。"),
    )
    sections = []
    for name, (title, caption) in zip(FIGURES, captions, strict=True):
        downloads = " · ".join(f'<a href="{name}.{ext}">{ext.upper()}</a>'
                               for ext in ("png", "svg", "pdf"))
        sections.append(f'<section><h2>{title}</h2><figure><div class="figure-scroll">'
                        f'<a href="{name}.svg"><img src="{name}.svg" alt="{title}" loading="lazy"></a></div>'
                        f'<figcaption>{caption}</figcaption></figure><p class="downloads">{downloads}</p></section>')
    raw_links = []
    for phase, label in ((0, "独立尺寸校准"), (1, "零均值评价"), (2, "信号评价")):
        links = [f'<a href="{cell["key"]}.csv.gz">G{cell["group"]} CSV.gz</a>'
                 for cell in metadata["cells"] if cell["phase"] == phase]
        raw_links.append(f"<p>{label}：{' · '.join(links)}</p>")
    protocol_link = _relative(output, ROOT / "experiments/parametric-replay-protocol.json")
    replay_doc = _relative(output, ROOT / "docs/parametric-replay.md")
    mechanism_doc = _relative(output, ROOT / "docs/tail-mechanism.md")
    script_link = _relative(output, Path(__file__).resolve())
    additional_links = []
    if not quick and (output / "audit.json").is_file():
        additional_links.append('<a href="audit.json">独立审计记录 JSON</a>')
    results_path = ROOT / "docs/replay-results.md"
    if results_path.is_file():
        additional_links.append(f'<a href="{_relative(output, results_path)}">正式结果解读</a>')
    additional_links_html = f"<p>{' · '.join(additional_links)}</p>" if additional_links else ""
    versions = html.escape(" · ".join(f"{name} {version}" for name, version in metadata["environment"].items()))
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,"><title>参数不确定性与最大值检验</title>
<style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#eeebe3;color:#28251f;font:17px/1.85 Georgia,"Songti SC","Noto Serif CJK SC",serif}}
main{{max-width:1190px;margin:34px auto;padding:48px 58px 40px;background:#fffdf7;border:1px solid #dfd8c8}}
header{{border-bottom:1px solid #cfc5b3;padding-bottom:24px;margin-bottom:30px}}.eyebrow{{font-size:13px;letter-spacing:.08em;color:#726753;margin:0 0 10px}}
h1{{font-size:33px;line-height:1.4;font-weight:500;margin:0 0 16px}}h2{{font-size:23px;line-height:1.5;font-weight:500;margin:40px 0 15px}}p{{margin:13px 0}}
.notice{{border-left:3px solid #8c775a;padding-left:17px}}.note,figcaption,.downloads,footer{{font-size:14px;color:#655d50}}.mobile-note{{display:none}}a{{color:#355a6d;text-underline-offset:3px}}
figure{{margin:21px 0 8px}}figure img{{width:100%;height:auto;display:block;background:white;border:1px solid #e8e3d8}}figcaption{{margin-top:12px;line-height:1.8}}
.figure-scroll{{overflow-x:auto}}.figure-scroll a{{display:block}}
.downloads{{margin:8px 0 18px}}.table{{overflow-x:auto;margin:20px 0}}table{{border-collapse:collapse;font-size:13px;line-height:1.6;width:100%}}
caption{{text-align:left;font-size:15px;margin-bottom:12px;color:#4c4438}}th,td{{padding:10px 9px;border-bottom:1px solid #dfd8ca;text-align:right;vertical-align:top}}
thead th{{border-top:1px solid #9f927c;border-bottom:1px solid #9f927c;font-weight:500;color:#463e31}}th:first-child{{text-align:left;font-weight:400}}
.results th:first-child{{min-width:164px}}.results td{{white-space:nowrap}}small{{display:block;font-size:11px;color:#746b5d;white-space:nowrap;margin-top:2px}}
.methods{{display:grid;grid-template-columns:210px 1fr;column-gap:22px;row-gap:17px;font-size:15px}}dt{{font-weight:500}}dd{{margin:0}}code{{font:12px/1.5 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}}
dt code{{display:block;margin-top:3px;color:#736955}}details{{margin:24px 0}}summary{{cursor:pointer;color:#4e4435}}footer{{border-top:1px solid #cfc5b3;margin-top:42px;padding-top:20px}}
@media(max-width:760px){{main{{margin:0;padding:28px 19px;border:0}}h1{{font-size:27px}}h2{{font-size:22px}}body{{font-size:16px}}.methods{{display:block}}dt{{margin-top:20px}}dd{{margin-top:6px}}.results{{min-width:950px}}figure img{{min-width:900px}}.mobile-note{{display:block}}}}
@media print{{body{{background:white}}main{{margin:0;max-width:none;border:0;padding:0}}details{{display:block}}figure,table{{break-inside:avoid}}a{{color:inherit}}}}
</style></head><body><main>
<header><p class="eyebrow">GAUSSIAN REPLAY · RESEARCH NOTE</p><h1>参数不确定性与最大值检验</h1>
<p>完整重放随机分母，是否足以校准策略筛选后的尾部？</p>
<p class="notice">{warning}</p><p class="note">{run_kind} · 每格零均值 {settings['null_replicates']} 轮 · 每次内部重放 {settings['inner_draws']} 轮 · 名义水平 {_percent(alpha)}</p>
<p class="note mobile-note">窄屏上可横向滚动图像查看各面板，也可点击图像打开原 SVG；图注保持正常文字宽度。</p></header>
<p>本实验在 Gaussian 时间模型内比较三件事：真实尺度和相关的参照、拟合相关下的 Gaussian 临界值，以及将逐列参数估计、中心化 HAC、补尾乘数和最大值筛选一并重算的参数重放。已知生成参数的重放与未知参数的拟合重放具有不同的保证。</p>
<p>G1–G6 使用共同 AR(1) 时间参数；G2 是强持久性压力格点，G5 用 100 个候选检查较大搜索规模。G7 刻意令共同因子 φ=0.98、个体成分 φ=0.30，违反拟合模型的共同时间参数假设。它的瞬时相关为 0.35，均值相关为 {g7_mean_rho:.4f}。</p>
<p class="note">方法证明限定固定 K、正确 Gaussian 模型与 T(1−φ)→∞、带宽 ell=o(T)。本页的有限格点不构成这一渐近条件的有限样本保证；也不建立增长 K、local-to-unity、厚尾或波动聚集下的普遍有效性。<a href="{replay_doc}">具体推导与边界</a> · <a href="{protocol_link}">固定实验协议</a>。</p>
{''.join(sections)}
<p>方法共用每轮噪声，因此图中的区间重叠不能检验成对差异。<a href="paired.csv">配对比较</a>另存相对真实 DGP 重放的拒绝不一致计数、风险差、近似区间及未经多重比较调整的 McNemar p-value。</p>
<p>最大值检验报告全局拒绝。<a href="summary.csv">汇总文件</a>还记录“拒绝且最大值候选恰为信号列”的联合频率；选中信号列不等于完成每列的个别推断，也不构成同时识别所有盈利策略的保证。</p>
<details><summary>九种方法的定义与所需信息</summary><dl class="methods">{definitions_html}</dl></details>
<section><h2>全部零均值格点</h2><p class="note">每个数值单元的第一行为拒绝频率，第二行为点态 95% Wilson Monte Carlo 区间；所有值来自 phase 1，每格 {settings['null_replicates']} 个独立外层重复。G7 的 GLS 不适用。</p>
<div class="table"><table class="results"><thead><tr><th scope="col">方法</th>{''.join(f'<th scope="col">G{identifier}<small>T={group["n_obs"]} · K={group["k"]}</small></th>' for identifier, group in groups.items())}</tr></thead><tbody>{''.join(null_rows)}</tbody></table></div>
<div class="table"><table><caption>模型参数与预定尺寸检查</caption><thead><tr><th scope="col">格点</th><th scope="col">T / K</th><th scope="col">φ 因子 / 个体</th><th scope="col">ρ 瞬时 / 均值</th><th scope="col">ρ 投影频率</th><th scope="col">完整拟合重放：尺寸上界</th></tr></thead><tbody>{''.join(design_rows)}</tbody></table></div>
<p class="note">最后一列是 G1–G6 完整拟合重放的单侧 Clopper–Pearson 上界，每格误差概率 0.05/6。协议要求该上界不超过 7%；G7 只作错配压力检查。这个有限实验筛查不证明其他模型或参数点的有效性。{'开发运行的样本量只用于检查此字段，不据此作正式判定。' if quick else ''}</p></section>
<section><h2>独立尺寸校准及其不确定性</h2>
<p>下排功效使用额外 {settings['calibration_replicates']} 轮零均值校准，随后固定阈值用于独立评价。连续校准分数的次序为 ceil((1−α)(M+1))；秩方法以独立均匀扰动消除分数同值，原 p-value 和名义检验不受扰动影响。</p>
<p>表中的 Beta 区间描述重复校准时，所选固定阈值真实尾概率的 95% 分布范围，不是额外的逐点功效区间。其跨校准平均尺寸由次序统计量精确给出；一次校准的真实尺寸仍有随机性。这里知道真实 DGP 才能额外生成校准数据，属于模拟比较参照，不能直接用于未知真实数据的检验。</p>
<div class="table"><table><thead><tr><th scope="col">格点与方法</th><th scope="col">次序 / 校准数</th><th scope="col">固定分数阈值</th><th scope="col">跨校准平均尺寸</th><th scope="col">条件尺寸 Beta 95% 区间</th></tr></thead><tbody>{''.join(calibration_rows)}</tbody></table></div>
{_resolution_note(output, metadata, calibration)}</section>
<footer><p><a href="summary.csv">拒绝与选择汇总 CSV</a> · <a href="paired.csv">配对差异 CSV</a> · <a href="calibration.json">校准 JSON</a> · <a href="inner-snapshots.json">内部重放快照 JSON</a> · <a href="metadata.json">原始运行与哈希 JSON</a> · <a href="presentation.json">展示文件哈希 JSON</a></p>
{''.join(raw_links)}
{additional_links_html}
<p><a href="{protocol_link}">实验协议</a> · <a href="{replay_doc}">参数重放推导</a> · <a href="{mechanism_doc}">补尾机制推导</a> · <a href="{script_link}">报告生成脚本</a>。补尾公式来源：<a href="https://doi.org/10.1080/01621459.2026.2676715">Liu–Chan，JASA 2026</a>；完整重放和秩检验使用既有 Monte Carlo 原理。</p>
<p>计算源码版本：<code>{html.escape(metadata['git_revision'])}</code>。协议 SHA-256：<code>{html.escape(metadata['protocol_sha256'])}</code>。每份计算源文件的冻结哈希见原始 metadata；展示脚本与输出哈希单独记录，不改写计算证据。</p>
<p>{versions}。三张图及本页仅读取已保存、哈希核验通过的证据，不重新模拟或调整检验规则。</p></footer>
</main></body></html>'''


def build(output: Path) -> None:
    """Verify immutable evidence, then write independently hashed presentation files."""
    output = output.resolve()
    metadata_path = output / "metadata.json"
    metadata_digest = _sha(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _verify(output, metadata)
    protocol_path = ROOT / "experiments/parametric-replay-protocol.json"
    if _sha(protocol_path) != metadata["protocol_sha256"]:
        raise ValueError("The protocol no longer matches this run's frozen protocol hash.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if tuple(protocol["methods"]) != METHODS or [group["id"] for group in protocol["groups"]] != list(range(1, 8)):
        raise ValueError("Unsupported replay report protocol schema.")
    index = _index(_read(output / "summary.csv"))
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    groups = {group["id"]: group for group in protocol["groups"]}
    _style()
    _size_figure(output, metadata, index, groups)
    _fit_figure(output, metadata, groups)
    _power_figure(output, metadata, protocol, index)
    (output / "report.html").write_text(
        _document(output, metadata, protocol, index, calibration), encoding="utf-8"
    )
    _verify(output, metadata)
    if _sha(metadata_path) != metadata_digest or _sha(protocol_path) != metadata["protocol_sha256"]:
        raise ValueError("Evidence metadata or protocol changed during rendering.")
    rendered = ["report.html", *(f"{name}.{ext}" for name in FIGURES for ext in ("png", "svg", "pdf"))]
    presentation = {
        "status": "complete", "profile": metadata["profile"],
        "input_hashes": {"metadata.json": metadata_digest, **metadata["output_hashes"],
                         "protocol": metadata["protocol_sha256"], "render_script": _sha(Path(__file__))},
        "output_hashes": {name: _sha(output / name) for name in rendered},
    }
    (output / "presentation.json").write_text(
        json.dumps(presentation, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
