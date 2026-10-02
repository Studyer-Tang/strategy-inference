"""Plain, self-contained HTML reports; numbers are never interpreted by an LLM."""

import csv
import html
import json
from pathlib import Path
from typing import Any

from .audit import AuditResult

STYLE = """
:root{color-scheme:light;--ink:#222b32;--muted:#626b71;--rule:#d7d9d9;--blue:#274a60}
*{box-sizing:border-box}body{margin:0;background:#f2f1ed;color:var(--ink);font:16px/1.8 Georgia,'Noto Serif SC','Songti SC',serif}
main{max-width:1040px;margin:42px auto;padding:48px 60px 56px;background:#fff;box-shadow:0 2px 14px #00000008}
header{border-top:3px solid var(--ink);padding-top:15px;margin-bottom:30px}
.eyebrow{font:12px/1.5 system-ui,sans-serif;letter-spacing:.12em;color:var(--muted)}
h1{font-size:34px;font-weight:600;line-height:1.35;margin:16px 0 8px}h2{font-size:23px;margin-top:38px;font-weight:600}
p{margin:14px 0}a{color:var(--blue);text-underline-offset:3px}.note,figcaption,footer{color:var(--muted);font-size:14px}
table{border-collapse:collapse;width:100%;font:13px/1.6 system-ui,sans-serif;margin:20px 0}
th,td{padding:10px 9px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}thead{border-top:1px solid var(--ink);border-bottom:1px solid var(--ink)}
tr.selected{background:#f1f5f6}.scroll{overflow-x:auto}figure{margin:30px 0}figure img{display:block;width:100%;height:auto}
figcaption{margin-top:12px}code{font:13px ui-monospace,SFMono-Regular,monospace;background:#f3f4f3;padding:2px 4px;overflow-wrap:anywhere}
li{margin:7px 0}footer{border-top:1px solid var(--rule);margin-top:40px;padding-top:18px}
@media(max-width:700px){main{margin:0;padding:28px 20px;box-shadow:none}h1{font-size:28px}}
@media print{body{background:white}main{margin:0;padding:0;box-shadow:none}figure{break-inside:avoid}a{color:inherit}}
"""


def document(title: str, content: str) -> str:
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body><main>{content}</main></body></html>"
    )


def write_audit_report(
    result: AuditResult,
    output: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = result.to_dict()
    if provenance:
        record["provenance"] = provenance
    (output / "audit.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = []
    for index, name in enumerate(result.names):
        label = html.escape(name)
        selected = ' class="selected"' if index == result.selected_index else ""
        interval = (
            f"[{result.simultaneous_ci_low[index]:.6g}, {result.simultaneous_ci_high[index]:.6g}]"
        )
        rows.append(
            f"<tr{selected}><td>{label}</td><td>{result.mean[index]:.6g}</td>"
            f"<td>{result.standard_error[index]:.6g}</td><td>{result.iid_pvalue[index]:.4f}</td>"
            f"<td>{result.hac_pvalue[index]:.4f}</td><td>{result.adjusted_pvalue[index]:.4f}</td>"
            f"<td>{interval}</td></tr>"
        )
    conclusion = (
        "在当前方法假设与提交的候选集下，拒绝全族原假设。"
        if result.global_pvalue <= result.alpha
        else "在当前方法假设与提交的候选集下，尚不能拒绝全族原假设。"
    )
    family = (
        "研究者声明候选集完整"
        if result.search_complete is True
        else "研究者声明候选集不完整"
        if result.search_complete is False
        else "完整搜索记录未确认"
    )
    tail_low, tail_high = result.bootstrap_tail_interval
    scale_description = (
        "每次重抽样重新计算同一滞后阶的 HAC 标准误（Bootstrap t）"
        if result.studentization == "resampled"
        else "每次重抽样使用原样本的固定 HAC 尺度"
    )
    content = f"""
    <header><div class="eyebrow">STRATEGY INFERENCE · RESEARCH RECORD</div>
    <h1>候选策略的平均收益推断</h1><p class="note">时间依赖与策略筛选 · 第一阶段</p></header>
    <p>原假设：所有提交候选的平均基准调整收益均不大于零。列级筛选规则为最大 HAC 统计量。</p>
    <p><strong>全族 p 值：{result.global_pvalue:.4f}</strong>，检验水平 {result.alpha:.3f}。{conclusion}</p>
    <p>筛选候选为 <code>{html.escape(result.selected_name)}</code>。{family}；对隐藏试验的调整不在结果中。</p>
    <div class="scroll"><table><thead><tr><th>候选</th><th>均值</th><th>HAC 标准误</th>
    <th>IID p</th><th>HAC p</th><th>max 调整 p</th><th>同时置信区间</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>
    <p class="note">IID 与 HAC p 均为未经筛选调整的列级对照。区间为 {1 - result.alpha:.1%} 双侧同时区间，
    使用 max|t*|；检验使用单侧 max t*。收益单位与输入一致，未做年化。</p>
    <h2>计算设置</h2><p>T = {result.sample_size}，K = {result.n_strategies}；
    重抽样 {result.n_resamples} 次，期望块长 {result.block_length:g}，HAC 滞后阶 {result.lags}。
    p 值分辨率为 1/{result.n_resamples + 1}。</p>
    <p class="note">固定当前数据时，Bootstrap 尾概率的 95% Wilson 区间为
    [{tail_low:.4f}, {tail_high:.4f}]。它只反映内层模拟误差，不衡量方法假设是否成立。
    若该区间跨过检验水平，应增加抽样次数再作边界判断。</p>
    <h2>解释边界</h2><ul>
    <li>Bootstrap 使用逐列去均值数据与共享时间索引；{scale_description}。</li>
    <li>结论依赖平稳性、弱依赖和适当矩条件，有限样本精度不能由一次输出保证。</li>
    <li>数据时间、完整搜索记录与交易成本需由研究者核实。该报告不检测全部数据泄漏。</li>
    <li>全族拒绝不代表未来盈利，也不意味着已确定某条策略适合交易。</li>
    </ul><footer>详细数值与输入记录：<a href="audit.json">audit.json</a>。
    数值解释由固定报告模板生成。</footer>
    """
    path = output / "audit.html"
    path.write_text(document("候选策略的平均收益推断", content), encoding="utf-8")
    return path


def write_experiment_report(output: str | Path) -> Path:
    """Render the three saved figures and selected data rows without recomputation."""
    output = Path(output)
    metadata = json.loads((output / "run-metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("The experiment run must be complete before rendering its report.")
    config = metadata["resolved_protocol"]

    def records(name: str) -> list[dict[str, str]]:
        with (output / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    dependence = records("figure-1-dependence.csv")
    selection = records("figure-2-selection.csv")
    sizes = records("figure-3-size.csv")
    method_groups = (
        (
            "单策略，φ = 0.8",
            [row for row in dependence if float(row["phi"]) == 0.8],
            ("iid_t", "hac_z", "stationary_bootstrap"),
        ),
        (
            "筛选 50 条，φ = 0.5",
            [row for row in selection if int(row["n_strategies"]) == 50],
            ("iid_winner", "hac_winner", "bootstrap_max"),
        ),
        (
            "GARCH，20 条",
            [row for row in sizes if row["process"] == "garch"],
            ("iid_winner", "hac_winner", "bootstrap_max"),
        ),
    )
    rows = []
    for label, candidates, methods in method_groups:
        values = []
        for method in methods:
            row = next(row for row in candidates if row["method"] == method)
            values.append(
                f"<td>{float(row['rate']):.2%}<br><span class='note'>"
                f"[{float(row['ci_low']):.2%}, {float(row['ci_high']):.2%}]</span></td>"
            )
        rows.append(f"<tr><td>{label}</td>{''.join(values)}</tr>")
    smoke_note = (
        "<p><strong>这是 quick 流程检查，重复次数不足以支持方法校准结论。</strong></p>"
        if metadata["profile"] == "quick"
        else ""
    )
    figure_specs = (
        (
            "figure-1-autocorrelation",
            "一 · 时间依赖",
            "零均值高斯 AR(1)，每次只检验一个策略。虚线参考线使用大样本近似；所有误差棒是点态 Monte Carlo 区间。"
            "较强依赖下，默认带宽与块长仍可能出现有限样本偏差。",
            ("figure-1-dependence.csv",),
        ),
        (
            "figure-2-selection",
            "二 · 策略筛选",
            "全部候选的真实均值都是零。每个嵌套候选集都按最大 HAC 统计量选择胜出者，"
            "IID 与 HAC 读取同一个胜出者的未调整 p 值；联合 Bootstrap 检验整族。",
            ("figure-2-selection.csv",),
        ),
        (
            "figure-3-robustness",
            "三 · 过程与检出能力",
            "左侧是三个过程下的误报率；右侧只在第一列植入正均值，比较全族检验的拒绝概率。"
            "这不是正确选中信号策略的概率。重尾创新由标准化 t₅ 成分构造；有限四阶矩不代替完整有效性证明。",
            ("figure-3-size.csv", "figure-3-power.csv"),
        ),
    )
    figures = []
    for stem, title, caption, data_files in figure_specs:
        links = [f'<a href="{stem}.pdf">PDF</a>', f'<a href="{stem}.svg">SVG</a>']
        links.extend(f'<a href="{name}">{name}</a>' for name in data_files)
        figures.append(
            f'<h2>{title}</h2><figure><img src="{stem}.png" alt="{title}的模拟实验图">'
            f"<figcaption>{caption}<br>{' · '.join(links)}</figcaption></figure>"
        )
    revision = html.escape(str(metadata.get("git_revision") or "未记录"))
    content = f"""
    <header><div class="eyebrow">STRATEGY INFERENCE · SIMULATION STUDY</div>
    <h1>时间依赖与策略筛选<br>如何影响显著性</h1>
    <p class="note">第一阶段 · 零均值检验、联合重抽样与有限样本表现</p></header>
    <p>本轮研究检验平均基准调整收益是否大于零。模拟中的均值已知，
    因而可以直接比较误报率与检出能力。实现采用固定 HAC 尺度的 stationary-bootstrap max 检验。</p>
    {smoke_note}
    <p class="note">每个主实验情形 T = {config["n_obs"]}，外层重复 {config["n_mc"]:,} 次，
    每次 Bootstrap {config["n_bootstrap"]:,} 次；名义水平 {config["alpha"]:.0%}。
    HAC 滞后 {config["hac_lags"]}，期望块长 {config["block_length"]}，种子 {metadata["seed"]}。</p>
    <h2>三个零均值情形</h2>
    <div class="scroll"><table><thead><tr><th>情形</th><th>IID 未调整</th><th>HAC 未调整</th>
    <th>联合 Bootstrap</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
    <p class="note">第一行没有筛选，Bootstrap 为单列检验。其余两行有筛选，联合方法检验提交的候选族。
    方括号内为外层拒绝比例的 95% Wilson 区间，不能解释为跨表格的同时保证。</p>
    {"".join(figures)}
    <h2>如何理解这些结果</h2>
    <p><strong>当前默认设置在本轮基准中存在有限样本误报偏差。
    第一版用于研究与核对，不能称为已校准的实务检验。</strong></p>
    <p>时间依赖调整与候选筛选调整处理不同问题。名义水平是比较基线，
    实际误报率由保存的计数估计。联合重抽样的有限样本偏差也需要保留；
    三个模拟过程的结果不能推出对所有金融数据都有效。</p>
    <p>第一版是现有统计方法的实现与模拟研究，不声明原创定理，
    不包含历史交易实证。对真实策略还需核实数据时间、成本、完整搜索记录和样本外评价。</p>
    <h2>复现与记录</h2>
    <p><code>strategy-inference reproduce --profile {html.escape(metadata["profile"])} --output results/{html.escape(metadata["profile"])} --seed {metadata["seed"]}</code></p>
    <p class="note">本轮运行用时 {metadata["elapsed_seconds"]:.1f} 秒。
    计算版本：<code>{revision}</code>。环境版本、协议、计数与文件哈希见
    <a href="run-metadata.json">运行记录</a>；预先指定的块长敏感性见
    <a href="block-length-sensitivity.csv">附表</a>，未用于挑选主图参数。</p>
    <footer>方法推导：项目内 docs/methods.md。报告直接读取已保存的 CSV 与图文件，
    所有展示结果均可回到拒绝次数核对。</footer>
    """
    path = output / "report.html"
    path.write_text(document("时间依赖与策略筛选", content), encoding="utf-8")
    return path
