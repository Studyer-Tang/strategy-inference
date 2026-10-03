# 研究与复现 / Research index

本页索引已完成的研究及失败边界。安装和公共 API 见[中文首页](../README.md)与[英文首页](../README.en.md)。原始报告、数据、协议和版本记录均保留；整理首页不改变历史证据。

This index links completed studies, failure settings and frozen evidence. Simulation improvements do not establish validity beyond the stated assumptions.

## 已完成阶段

| 阶段 | 问题、结果与范围 | 证据 |
| --- | --- | --- |
| Stationary bootstrap | 比较固定尺度与逐轮 HAC 学生化。后者在若干场景降低观察误报，但多数预设有限场景门槛未通过，高持久性仍有失准。公共 bootstrap 保留近似适用条件。 | [结果与失败数字](results.md)、[方法](methods.md)、[完整报告](../results/calibration/full/report.html)、[冻结协议](../experiments/calibration-protocol.json)、[运行记录](../results/calibration/full/run-metadata.json) |
| Tail-scale mechanism | 区分总体截断、配对损失、中心化和随机分母。真实参数使尺度期望无偏后，筛选尾部仍可能失准。采用已知相关结构，属于机制诊断。 | [结果](tail-results.md)、[机制证明](tail-mechanism.md)、[已知参数候选规模界](oracle-selection-bound.md)、[报告与逐轮记录](../results/research/tail/full/report.html)、[协议](../experiments/tail-diagnostic-protocol.json) |
| Parametric replay | 分开研究时间拟合、相关拟合和补尾因子重估。部分格点改善，强持久性与异质时间结构仍失败；固定内层模拟次数还有分辨率与功效成本。 | [结果](replay-results.md)、[条件证明](parametric-replay.md)、[报告](../results/research/replay/full/report.html)、[协议](../experiments/parametric-replay-protocol.json)、[独立审计](../results/research/replay/full/audit.json) |
| Reference-column uncertainty | 共同未知 Gaussian AR 参数下，参考列置信集合和连续认证给保守强 FWER 决定。研究记录同时保留明显功效损失和保证范围外的诊断。 | [结果](uncertainty-results.md)、[证明](parameter-uncertainty.md)、[报告](../results/research/uncertainty/full/report.html)、[协议](../experiments/parameter-uncertainty-protocol.json)、[审计](../results/research/uncertainty/full/audit.json) |
| Joint parameter information | 用预先指定的联合方向与时间块缩小参数集合，在部分场景提高检出率。单列近单位根、重复列和共同模型错配仍是边界；作为统一入口 `gaussian_ar` 的底层方法。 | [结果](joint-results.md)、[证明](joint-uncertainty.md)、[报告](../results/research/joint/full/report.html)、[协议](../experiments/joint-uncertainty-protocol.json)、[线上实验图](https://studyer-tang.github.io/strategy-inference/research/joint/) |

原始固定尺度[开发协议](../experiments/protocol.json)和[开发报告](../results/full/report.html)继续保留。Bootstrap 独立确认运行的冻结源码及后续 quick 对照范围见[结果说明](results.md)与[冻结源码对照](../results/verification/frozen-source-comparison.json)。历史正式研究应在运行元数据指定的冻结 commit 或对应的 `v0.4.0` 等 tag 下完整复核；本轮 `v0.5.0` 改进计算与 API，不改变原协议和证据。旧审计器锁定源码哈希，不能直接用更新后的核心实现复核旧证据。现有报告与图表仍可直接读取。

[研究方案](research-plan.md)记录近期文献、后续问题与证明义务；[项目讨论](interview-notes.md)解释方法选择；[文献表](references.bib)提供论文和软件来源。已有置信集合反演、Wilks、参数投影和 bootstrap 原理不作为新的通用推断原理宣称。

## 复现

从源码目录安装绘图依赖。`quick` 检查运行流程；完整证据使用各阶段冻结的 `full` 协议。重新模拟使用新的空目录，保留原始结果及元数据。报告脚本读取保存的证据，重建图表不等于新的独立评价。

```bash
python -m pip install '.[figures]'

strategy-inference reproduce --study calibration --profile quick
strategy-inference reproduce --study calibration --profile full \
  --output results/calibration/reproduced
strategy-inference reproduce --study baseline --profile full \
  --output results/baseline-reproduced
```

补尾与参数重放：

```bash
python scripts/tail_diagnostics.py --profile full \
  --output results/research/tail/reproduced
python scripts/verify_tail_mechanism.py --output /tmp/tail-mechanism-checks.json

python scripts/replay_report.py --output results/research/replay/full
python scripts/parametric_replay.py --profile full \
  --output results/research/replay/reproduced
python scripts/replay_report.py --output results/research/replay/reproduced
python scripts/verify_parametric_replay.py --output results/research/replay/reproduced
```

参数不确定性与联合信息：

```bash
python scripts/uncertainty_report.py --output results/research/uncertainty/full
python scripts/parameter_uncertainty.py --profile full \
  --output results/research/uncertainty/reproduced
python scripts/verify_parameter_uncertainty.py \
  --output results/research/uncertainty/reproduced
python scripts/uncertainty_report.py --output results/research/uncertainty/reproduced

python scripts/joint_report.py --output results/research/joint/full
python scripts/joint_uncertainty.py --profile full --workers 4 \
  --output results/research/joint/reproduced
python scripts/verify_joint_uncertainty.py --output results/research/joint/reproduced
python scripts/verify_joint_bounds.py --output results/research/joint/reproduced
python scripts/joint_report.py --output results/research/joint/reproduced
```

## 独立核验与解读边界

- 补尾公式与作者原始 R 函数的对照只覆盖已记录的给定参数公式，不代表整篇论文的自动拟合与带宽方法均已复现。见[运行来源](../results/research/verification/author-r-runtime.json)与[逐项对照](../results/research/verification/tail-reference-comparison.json)。
- [机制数值核验](../results/research/verification/tail-mechanism-checks.json)检查矩阵恒等式、参数恒等式和有限网格极限误差，不能作为新的拒绝率 Monte Carlo 或渐近证明。
- 外层点态比例区间、跨场景同时门槛、内层重抽样误差与候选均值区间是不同对象。未通过预设证据门槛和已证明真实误报超标也须区分。
- 已知参数参照有助于隔离机制，不能冒充真实数据的未知参数方法。Gaussian AR 参数集合的模型内保证也不转移到一般 bootstrap 接口。
- 搜索调整覆盖事先确定的输入候选集。隐藏搜索、自适应生成、反复监测与停止、数据泄漏及未来市场变化不由收益矩阵自动解决。

各专题保留全部格点、区间、原始记录、失败结果及理论限制；首页方法说明仍保留使用者做选择所需的条件。

## 多步成熟反馈与尺度共享（v0.7）

[方法与独立证明](multistep-methods.md)、[固定协议模拟结果](multistep-results.md)和[运行成本](multistep-performance.md)记录每步长反馈时钟、可选尺度共享及其适用范围。四个预设过程各40条独立路径，三张图可由一个命令重现；原始记录与源码、runner和协议哈希绑定。
