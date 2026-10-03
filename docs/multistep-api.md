# 多步在线区间 API

`MultiStepConformal` 与 `multistep_intervals` 在每个目标成熟后更新对应步长的控制器。v0.8 增加自身步长与短步长共享尺度的固定比例融合。现有回测、比较和单步区间见[时序 API](time-series.md)，递推与证明见[多步方法](multistep-methods.md)，可运行示例见[examples/multistep.py](../examples/multistep.py)。这是独立工程实现，未实现完整 AcMCP 的 PID 饱和反馈、误差预测器或跨步长优化。

当前输入是单变量、等间隔序列与连续整数观测时钟。保证的对象是**每个步长已成熟预测的历史平均误覆盖**，不是每个时点的条件覆盖，也不是整条未来路径同时覆盖。下面的 `coverage_bound` 来自理想实数递推；普通 binary64 实现没有舍入证书。

## 从回测接入

```python
import numpy as np
from strategy_inference import backtest, naive_forecast, multistep_intervals

y = np.cumsum(np.random.default_rng(17).normal(size=384))
run = backtest(
    y, {"naive": naive_forecast}, initial_train_size=128, window=128, horizon=12,
)
origins = np.asarray([split.origin for split in run.splits])
leads = run.target_indices[0] - origins[0]  # 物理步长，包含 gap
result = multistep_intervals(
    y, run.forecasts[:, :, 0], origins=origins, lead_times=leads,
    scale=np.sqrt(leads), step_size=0.1 / np.sqrt(leads), decay=0.2,
    initial_quantile=0.65, scale_decay=0.97, scale_source="blended",
    scale_share_weight=0.5,
)
print(result.summary())
```

本例 `sqrt(lead)` 来自单位创新方差随机游走的已知模拟尺度；实际使用可由初始训练数据固定尺度，不能用未来评价标签拟合。`0.1/sqrt(lead)` 仅演示按步长降低学习率，不声称最优。若回测设置 `gap>0`，必须用 `target-origin`，不能用列号代替物理步长。

## 初始化与参数

```python
MultiStepConformal(
    lead_times, *, start_time=0, alpha=0.1, step_size=0.1, decay=0.6,
    scale=1.0, initial_quantile=0.5, strategy="pooled",
    scale_decay=None, scale_source="horizon", scale_floor=1e-8,
    scale_share_weight=None,
)
```

| 参数 | 含义与要求 |
| --- | --- |
| `lead_times` | 严格递增、非空的正整数向量；每次发行都预测全部这些步长 |
| `start_time` | 首次必须观测的非负整数位置；不是已经观测过的 origin。位置和目标须能装入 int64 |
| `alpha` | 共同目标误覆盖，`0 < alpha < 1` |
| `step_size` | 正的有限标量或长度为 H 的向量；各步长的初始学习率常数 Γ，随后使用 `Γ[h] * n**(-decay)` |
| `decay` | `0 <= decay < 1`；`n` 是该控制器已经处理的成熟反馈数 |
| `scale` | 正的有限标量或 H 向量；初始残差尺度，使用原始序列单位 |
| `initial_quantile` | 共同初始阈值，位于 `[0,1]`；更新后不截断 |
| `strategy` | `"pooled"` 或 `"interlaced"`，见下文 |
| `scale_decay` | `None` 为固定尺度；`0 <= scale_decay < 1` 启用成熟残差的 EWMA RMS |
| `scale_source` | `"horizon"` 使用自身步长成熟残差；`"shortest"` 共享最小配置步长的成熟残差；`"blended"` 融合两者。后两种须启用 `scale_decay` |
| `scale_floor` | 正的有限标量或 H 向量；只限制自适应尺度更新，不改写给定初始尺度或单次发行覆盖值 |
| `scale_share_weight` | 仅用于 `"blended"`，有限标量或 H 向量，位于 `[0,1]`；`None` 时默认各步长 `0.5`。数值表示共享来源的权重，在运行中固定；其他来源传入非 `None` 权重会报错 |

`pooled` 每个步长维护一个阈值，在成熟反馈到达时更新**当前**阈值；miss 仍使用旧发行区间。`interlaced` 按 `origin % lead` 分 lane，每个 lane 独立更新。Pooled 的 `n` 是该步长所有反馈数，interlaced 的 `n` 是该 lane 的反馈数，因此同一日历长度下学习率不同，不能把比较差异全部解释为信息效率。

共享尺度保持初始比率：若最短步长为 `h*`，发行前默认尺度为 `initial_scale[h]/initial_scale[h*] * current_source_scale`，再应用各步长 floor。只有已经成熟的 `h*` 残差更新共同 RMS；`h*` 不一定为 1。其他步长的阈值与 miss 反馈仍独立。这是可选研究方案，尚不声称它优于自身步长尺度或固定尺度。

`blended` 同时维护自身步长尺度 `own[h]` 和上述共享尺度 `shared[h]`，发行默认尺度为 `(1-w[h])*own[h] + w[h]*shared[h]`。初始时两个来源均保留给定尺度；自身残差成熟时更新 own，最短步长残差成熟时更新 shared。缺少某种成熟反馈时保留该来源的上次值。权重应预先指定或使用独立训练/验证数据选择，不根据最终评价分数反复调整。

权重为 0、1 时，在两个来源都可用有限正数表示的输入范围内，分别逐位复现 `horizon`、`shortest`。融合模式始终检查两个来源和初始比率；即使某来源权重为 0，它溢出也会使调用原子失败。这提供统一的双来源诊断，不承诺与旧模式相同的极端输入接受范围。融合是信息来源之间的折中，没有通用最优权重或效率保证。

在每时发行、队列填满后，各步长都会在同一日历时点 t 收到 `y[t]`。Shortest 没有让长步长标签提前到达；差别在于预测起点的新旧、累计误差的相关性及尺度信息效率。它也不消除长步长阈值从发行到自身反馈的 h 步延迟。

## 流式时钟与记录

```python
tracker = MultiStepConformal([1, 3], scale=[1.0, np.sqrt(3)])
for t, label in enumerate(y):
    updates = tracker.observe(t, float(label))
    issued = tracker.predict([label, label])  # naive：只用当前标签
```

必须先观测当前位置，再发出未来预测。`observe(time, actual)` 要求 `time == next_time`，不能跳过或重复观测；返回当期所有成熟的 `MultiStepUpdate` 元组，尚无成熟预测时为空。允许某些时点不发行预测。`predict(predicted, *, scale=None)` 每个已观测时点至多调用一次；预测为 H 向量，覆盖尺度为正标量或 H 向量，只影响这次发行。

发行尺度与区间均冻结，之后的尺度更新不会回改旧 score。结束时不需要虚构未来标签去 flush pending；继续得到真实标签后，可按 `next_time` 继续 `observe`。无效输入、协议错误和数值溢出在当前调用提交前报错，不部分更新当期状态。

| 流式访问项 | 内容 |
| --- | --- |
| `lead_times`、`step_size`、`current_scales` | 配置步长、展开后的学习率常数、当前默认尺度元组 |
| `current_scale_weights` | 融合模式的固定共享权重元组；其他模式为 `None` |
| `last_time`、`next_time` | 最后已观测位置和下一必需位置；首次观测前 `last_time=None` |
| `n_updates`、`coverage`、`coverage_bound` | 按步长的成熟数量、实际覆盖和理想平均误覆盖误差上界；无反馈时覆盖与界为 `None` |
| `pending` | 按 target 排序的未成熟 `MultiStepInterval` 元组 |
| `lane_state(lead_time, lane=0)` | 独立字典：`quantile`、`n_updates`、`misses`；未用 lane 返回默认状态，不分配状态 |
| `summary()` | 每个步长的 `n_issued`、`n_evaluated`、`n_pending`、`coverage`、`coverage_bound` |
| `to_dict()` | 独立的严格 JSON 记录，包含配置、active lane 状态、summary 与 pending；只保存当前状态，不保留已成熟轨迹 |

融合模式的 `to_dict()` 额外保存 `scale_share_weight`、`own_scales` 和 `shared_scales`，方便检查新发行尺度的两个来源；其他模式保持原有记录结构。单次 `predict(scale=...)` 覆盖值不改变这些内部来源或权重。

`MultiStepInterval` 保存 `origin`、`target`、`lead_time`、`prediction`、`lower`、`upper`、`quantile`、`scale`、`kind`。`quantile` 是发行阈值。`kind` 为 `"finite"`、`"empty"` 或 `"unbounded"`；空集用 `(+inf,-inf)`，全域用 `(-inf,+inf)`，`to_dict()` 将二者边界转换为 `None` 并保留 kind。

`MultiStepUpdate` 保存发行位置/目标/步长、`actual`、`prediction`、冻结 `scale`、`issued_quantile`、更新前的 `previous_quantile`、更新后的 `quantile`、实际 `step_size`、有界 `score`、`miss`、`lane` 和该 lane 更新后的 `n_updates`。`new_quantile` 与 `eta` 分别是 `quantile` 与 `step_size` 的只读别名。Pooled 下 previous_quantile 可能不同于 issued_quantile。

## 批量重放与结果

```python
multistep_intervals(
    actual, predicted, *, origins, lead_times, alpha=0.1, step_size=0.1,
    decay=0.6, scale=1.0, initial_quantile=0.5, strategy="pooled",
    scale_decay=None, scale_source="horizon", scale_floor=1e-8,
    scale_share_weight=None,
)
```

`actual` 是完整的一维有限序列；`origins` 是其内部严格递增的非负整数位置；`predicted` 形状 `(F,H)`。Wrapper 从首个 origin 到 `len(actual)-1` 逐时观测，仅在指定 origins 发行；目标可超过序列末尾并保留 pending。参数同 class，但起点由首个 origin 确定。Wrapper 不训练预测器，也不验证外部预测的未来信息来源。

`MultiStepResult` 的数组均只读：

| 字段 | 形状与语义 |
| --- | --- |
| `origins`、`lead_times`、`n_origins` | F 个起点、H 个物理步长、起点数量 |
| `target_indices` | `(F,H)`，等于 origin 加物理步长 |
| `predicted`、`lower`、`upper`、`quantiles`、`scales` | `(F,H)` 的发行预测、区间边界、发行阈值与冻结尺度 |
| `evaluated` | `(F,H)`，目标已成熟并已反馈的掩码 |
| `actual`、`step_sizes` / `feedback_steps` | `(F,H)` 的目标真实值与成熟时实际学习率；未成熟位置为 NaN |
| `empty`、`unbounded` | `(F,H)` 的发行状态，对未成熟预测也有定义 |
| `misses` | `(F,H)` 的成熟 miss；未成熟默认 False，统计时必须使用 evaluated 掩码 |
| `pending` | 未成熟的发行记录元组 |
| `coverage`、`coverage_bound` | 按步长的实际成熟覆盖和理想平均误覆盖误差上界 |
| `diagnostics` | 独立的终态配置、lane、尺度、计数与 pending 字典；含下一次发行所用阈值状态 |

`records()` 返回每个 origin × lead 的完整长表，未成熟的 actual、miss、step_size 用 `None`，非有限集合边界用 `None`。`to_dict()` 合并全部记录、终态和 summary，支持 `json.dumps(..., allow_nan=False)`；这会构造轨迹列表，流式内存需求下应使用 class。`to_frame()` 需可选 pandas。

`summary()` 按步长输出数量、coverage、coverage_bound、成熟预测的 empty/unbounded 比例及 `finite_width_count`、`mean_finite_width`、`finite_width_status`、`mean_interval_score`、`interval_score_status`。宽度只平均有限且已成熟区间，须同时查看 excluded 状态比例。存在成熟空集或全域时，平均 interval score 为 `None`，状态为 `"empty_or_unbounded"`；这表示没有汇总该 score，不能当作零或删掉异常状态后的完整样本均值。

## 理论与数值范围

在每个步长固定、`0<=decay<1`、初始阈值在 `[0,1]`、尺度始终正且发行时可预测并冻结的理想递推中，成熟反馈的平均 miss 趋向 alpha；界只针对已反馈前缀。它不证明局部任意窗口、每个未来时点或所有步长联合覆盖，也不保证阈值收敛、最优宽度或共享尺度有收益。初始训练尺度和跨步长比率应先确定，后续更新只用成熟信息。

负阈值返回空集，至少 1 返回全域，不 clip。有限边界包含端点。反馈以实际返回区间为准：大中心、小半径可能使 binary64 舍入后的 miss 不同于 `score > issued_quantile`。诊断 score 不替代区间分类。理想界与数值证书的区别见[方法证明](multistep-methods.md)。

本实现参考 [ICML 2024 的递减步长 tracker](https://proceedings.mlr.press/v235/angelopoulos24a.html)。[2026 年 9 月 delayed ACI](https://arxiv.org/html/2609.07251v1) 已分析相位交织与延迟代价；[AcMCP（2026 修订）](https://arxiv.org/html/2410.13115v2) 使用成熟多步误差、PID 与误差预测；[O²CP（2026 年 8 月修订）](https://arxiv.org/html/2508.13362v3) 已研究跨步长优化。这里不继承这些完整算法的额外结论，也不将交织、尺度归一化或信息共享表述为首次提出。共享成熟尺度的正式对照另行报告，有限实验不能建立一般效率保证。
