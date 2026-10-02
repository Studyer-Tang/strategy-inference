"""Plain, self-contained HTML reports; numbers are never interpreted by an LLM."""

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
        interval = f"[{result.simultaneous_ci_low[index]:.6g}, {result.simultaneous_ci_high[index]:.6g}]"
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
        "研究者声明候选集完整" if result.search_complete is True else
        "研究者声明候选集不完整" if result.search_complete is False else "完整搜索记录未确认"
    )
    tail_low, tail_high = result.bootstrap_tail_interval
    content = f"""
    <header><div class="eyebrow">STRATEGY INFERENCE · RESEARCH RECORD</div>
    <h1>候选策略的平均收益推断</h1><p class="note">时间依赖与策略筛选 · 第一阶段</p></header>
    <p>原假设：所有提交候选的平均基准调整收益均不大于零。列级筛选规则为最大 HAC 统计量。</p>
    <p><strong>全族 p 值：{result.global_pvalue:.4f}</strong>，检验水平 {result.alpha:.3f}。{conclusion}</p>
    <p>筛选候选为 <code>{html.escape(result.selected_name)}</code>。{family}；对隐藏试验的调整不在结果中。</p>
    <div class="scroll"><table><thead><tr><th>候选</th><th>均值</th><th>HAC 标准误</th>
    <th>IID p</th><th>HAC p</th><th>max 调整 p</th><th>同时置信区间</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    <p class="note">IID 与 HAC p 均为未经筛选调整的列级对照。区间为 {1-result.alpha:.1%} 双侧同时区间，
    使用 max|t*|；检验使用单侧 max t*。收益单位与输入一致，未做年化。</p>
    <h2>计算设置</h2><p>T = {result.sample_size}，K = {result.n_strategies}；
    重抽样 {result.n_resamples} 次，期望块长 {result.block_length:g}，HAC 滞后阶 {result.lags}。
    p 值分辨率为 1/{result.n_resamples+1}。</p>
    <p class="note">固定当前数据时，Bootstrap 尾概率的 95% Wilson 区间为
    [{tail_low:.4f}, {tail_high:.4f}]。它只反映内层模拟误差，不衡量方法假设是否成立。
    若该区间跨过检验水平，应增加抽样次数再作边界判断。</p>
    <h2>解释边界</h2><ul>
    <li>Bootstrap 使用逐列去均值数据与共享时间索引；每次重抽样使用原样本的固定 HAC 尺度。</li>
    <li>结论依赖平稳性、弱依赖和适当矩条件，有限样本精度不能由一次输出保证。</li>
    <li>数据时间、完整搜索记录与交易成本需由研究者核实。该报告不检测全部数据泄漏。</li>
    <li>全族拒绝不代表未来盈利，也不意味着已确定某条策略适合交易。</li>
    </ul><footer>详细数值与输入记录：<a href="audit.json">audit.json</a>。
    数值解释由固定报告模板生成。</footer>
    """
    path = output / "audit.html"
    path.write_text(document("候选策略的平均收益推断", content), encoding="utf-8")
    return path
