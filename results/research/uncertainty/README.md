# 未知时间参数：正式证据

本目录保存一个预设的有限样本研究：Gaussian AR(1) 的共同时间参数未知时，用参考列的形状置信集合和连续域证书做均值同时检验。完整推导见[方法说明](../../../docs/parameter-uncertainty.md)，正式数字与功效成本解释见[结果记录](../../../docs/uncertainty-results.md)。

`full/` 是正式配置；每个单元 1000 次独立重复，计算代码冻结于 `2b8ae791875ee3f7cf508e2246fab9411ff139c7`。它包含 16,000 份独立基础噪声、31,000 次加入位移后的程序调用和 114,000 条方法记录。功效阶段的四个位移重用同一噪声，不能将其当成 4000 次独立重复。规则和 seed 均记录在[冻结协议](../../../experiments/parameter-uncertainty-protocol.json)中。

结果的重点是代价：模型内认证规则的全零假设误报很低，但 `δ=3` 的四组模型内功效仅 `.4%` 至 `1.1%`，`δ=6` 仍只有 `.5%` 至 `12.1%`。当前构造是可复核的可靠性基线，尚未展示可普遍使用的高功效方案。G8 是异质持久性的模型外诊断，没有共同真实 `φ`，其结果不属于保证范围。

## 文件与统计口径

| 文件 | 内容 |
| --- | --- |
| [report.html](full/report.html) | 书页排版的报告与三张实验图 |
| [summary.csv](full/summary.csv) | 435 行比率汇总，包括点态 95% Wilson 区间 |
| [paired.csv](full/paired.csv) | 104 行同噪声配对差、点态正态区间与精确 McNemar p 值 |
| `p1-gNN.csv.gz` | 全零假设；`false_reject` 与族拒绝相同 |
| `p2-gNN.csv.gz` | 仅候选 0 有信号；`signal_reject` 记录候选 0 的检出，四个位移共用基础噪声 |
| `p3-gNN.csv.gz` | 偶数索引有信号，奇数索引为零；`false_reject` 只计算真实零假设，检验 strong FWER |
| [certificates.json](full/certificates.json) | 每个单元预设抽查重复的精确有理区间、临界值和逐列证书诊断 |
| [metadata.json](full/metadata.json) | 环境、revision、完整协议、源码 SHA-256 和原始输出 SHA-256 |
| [audit.json](full/audit.json) | 独立审计结果及审计器哈希 |
| [presentation.json](full/presentation.json) | 展示脚本和 HTML／图像的单独哈希；不修改原始元数据 |

每个逐行文件保存所有判定掩码、置信集合的有理端点、集合长度、是否保留 `φ=1`、空集合和认证未完成诊断。区间长度是保留区间的长度之和。`certificate_unresolved` 是该数据集中未完成认证的列数；`certificate_nodes` 是各列访问节点数之和。两者与 `ci_unresolved` 的末层参数区间保留诊断不同。

G8 的 `ci_contains_truth` 字段只检查因子参数 `.98` 是否落在集合中，不能称为共同参数覆盖率。`gls_known` 和 `gls_known_budget` 在该组省略。正式实验没有尺寸匹配、额外零假设校准或根据结果调参。

所有区间均为逐单元描述，不同时覆盖全部组或位移。配对比较使用同一重复的差异，不能由单项区间是否重叠判定。正式计算保留全部噪声和失效保护结果，不删除空集合或覆盖失败的路径。

## 三张图

| 图 | SVG | PDF |
| --- | --- | --- |
| 误报与部分零假设 | [图 1](full/figure-1-uncertainty-size.svg) | [PDF](full/figure-1-uncertainty-size.pdf) |
| 参数集合与单位根端点 | [图 2](full/figure-2-uncertainty-boundary.svg) | [PDF](full/figure-2-uncertainty-boundary.pdf) |
| 已知参数、预算与未知参数的功效 | [图 3](full/figure-3-uncertainty-power.svg) | [PDF](full/figure-3-uncertainty-power.pdf) |

## 审计与复现

本轮独立审计已通过：检查 114,000 条记录、435 行汇总和 104 行配对算术，并按协议重建 93 次带位移的数据及证书。这对应 48 份独立基础噪声，每个单元抽查重复 `0、500、999`，功效单元检查全部四个位移。其余记录做哈希、掩码、置信集合与算术核对，没有重算全部 31,000 次程序调用。

以下命令从仓库根目录执行。安装可绘图环境后，重新审计并重建已有报告，不运行新的 Monte Carlo：

```bash
python -m pip install -e '.[figures]'
python scripts/verify_parameter_uncertainty.py --output results/research/uncertainty/full
python scripts/uncertainty_report.py --output results/research/uncertainty/full
```

绘图脚本要求元数据标记完成、原始输出哈希一致且独立审计通过。展示文件可重建，正式 CSV、压缩记录和原始元数据保持原样。

要重新计算完整实验，指定一个空目录：

```bash
python scripts/parameter_uncertainty.py --profile full --output results/research/uncertainty/reproduction
python scripts/verify_parameter_uncertainty.py --output results/research/uncertainty/reproduction
python scripts/uncertainty_report.py --output results/research/uncertainty/reproduction
```

已有非空证据目录不会被运行器覆盖。`quick` 每单元仅 8 次重复，用于开发和检查文件流程，不是正式统计结果。完整运行本次耗时 148.8 秒；不同环境的运行时间和末位浮点结果可能不同，应以各自元数据和哈希追踪。
