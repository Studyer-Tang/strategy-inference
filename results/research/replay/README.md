# 参数重放实验

本目录属于研究分支，不是 v0.2.0 的发布验证。

计算规则与评价范围在 [parametric-replay-protocol.json](../../../experiments/parametric-replay-protocol.json) 固定。数学条件见 [parametric-replay.md](../../../docs/parametric-replay.md)。旧补尾机制实验继续保存在 `../tail/full`，没有以新实验替换。

`full` 使用三个独立阶段：零均值分数校准、名义与校准后尺寸评价、植入单列正均值的功效评价。校准只用于模拟中公平比较功效，并不是数据使用者能够实施的未知 DGP 校准。参数拟合、尾部因子冻结及真正生成参数的对照均按固定协议保留。

每个 `p{phase}-g{group}.csv.gz` 保存所有外层数据对应的方法统计量、拟合参数、p 值与拒绝记录。模拟收益矩阵可通过随机地址重新生成。`inner-snapshots.json` 保存每个阶段/场景首轮、中轮、末轮的全部内层最大值；其他轮次可通过协议的随机地址、源码和环境版本重新生成。每轮使用 199 个内层抽样，观察值加入秩计算，p 值分母为 200。

`metadata.json` 记录计算源码、协议和原始输出哈希；`presentation.json` 单独记录图表与报告哈希。重建图表先核验原始证据，不重跑 Monte Carlo。

```bash
python scripts/parametric_replay.py --profile full --output results/research/replay/full
python scripts/replay_report.py --output results/research/replay/full
```

输出目录必须为空，不覆盖既有实验。开发检查使用 `development*` 临时目录与独立 quick 种子，不作为统计结论；两次开发启动因摘要接口衔接错误中止，修复后才完成运行流程。正式计算不会删除无效轮次或对失败轮次补抽样。
