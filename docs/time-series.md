# 时间序列评估与在线区间

`v0.6.0` 增加与模型无关的时序评估流程。原有均值检验见 [API](api.md)，后续范围见[路线图](toolbox-roadmap.md)。核心依赖仍为 NumPy 与 SciPy；滚动评估、评分和在线区间本身只用 NumPy 与标准库。

## 滚动切分

```python
rolling_splits(n_obs, *, initial_train_size, horizon=1, step=1, gap=0, window=None)
```

返回不可变 `RollingSplit` 元组。位置从零开始，训练与测试范围均为左闭右开：

- `train_start:train_stop` 是本折训练数据；`origin = train_stop - 1`。
- `test_start = train_stop + gap`；`test_stop = test_start + horizon`。
- 下一折的 `train_stop` 增加 `step`；只保留完整测试窗口。
- `window=None` 使用扩展历史；整数 `window` 是训练长度上限。若小于 `initial_train_size`，第一折就使用最近的 `window` 个观测。

例：`n_obs=12, initial_train_size=4, horizon=2, step=3, gap=1, window=3` 的两折为训练 `[1:4]`、测试 `[5:7]`，以及训练 `[4:7]`、测试 `[8:10]`。预测物理步长分别是 2 和 3，并非 1 和 2。测试目标可以跨折重叠；这代表不同预测起点，不代表独立观测。

参数要求非布尔整数；除 `gap>=0` 外均须为正。没有完整测试窗口时明确报错。输入按观测位置处理，不推断日期频率、排序或自动填补缺失值。

## 预测模型接口

```python
backtest(y, forecasters, *, initial_train_size, horizon=1, step=1, gap=0, window=None)
```

`y` 是有限实数的一维序列，允许常数序列。`forecasters` 是按插入顺序保留的 `{name: callable}` 字典。每个 callable 收到：

```python
prediction = forecaster(train, lead_times)
```

`train` 只含本折历史；`lead_times` 为 `gap+1, ..., gap+horizon`。返回长度恰为 `horizon` 的有限实数一维数组。每个模型和折获得独立、只读的训练副本及步长副本，其 NumPy base 不包含未来数据。调用者仍须避免在 closure、外部文件或预先全样本拟合的状态中使用未来信息。

任何失败折都会停止并指出模型、fold 与 origin，不静默跳过。普通 callback 异常保留为 `RuntimeError` 的 cause；预测类型、长度或数值无效时报 `ValueError`。

内置基线：

| 接口 | 预测规则 |
| --- | --- |
| `naive_forecast(train, lead_times)` | 全部步长重复最后观测 |
| `drift_forecast(train, lead_times)` | 延伸首尾之间的平均变化；训练至少两个观测 |
| `SeasonalNaive(period)(train, lead_times)` | 重复最后完整季节周期；训练至少 `period` 个观测 |

季节周期和窗口应事先确定。每折重新调用模型；若 callback 每次重新拟合，计算成本包含拟合本身。使用较小的固定 `window` 或较大的 `step` 可以限制工作量，同时改变评估设计。当前没有自动增量更新、外生变量或多序列 panel 接口。

其他预测库可通过 callable 接入。模型训练、超参数选择和预处理应限制在传入的训练资料中；本接口不替使用者确认嵌套验证过程。

## 保存的预测

`BacktestResult` 保留如下字段：

| 字段 | 形状或内容 |
| --- | --- |
| `names`、`splits` | 模型顺序、每折切片与 origin |
| `forecasts` | `(F, H, M)`，起点 × 步长 × 模型 |
| `actuals`、`target_indices` | `(F, H)`，目标值及其原序列位置 |
| `sample_size` / `n_obs` | 原序列长度 |
| `n_folds`、`horizon`、`n_models` | 三个输出维度 |

数组只读。`records()` 按 fold、步长、模型输出完整长表；`to_dict()` 为严格 JSON 可序列化的全部预测记录。`to_frame()` 需可选 pandas。

## 损失与评分

```python
forecast_loss(actual, predicted, *, loss="squared", quantile=None)
evaluate_forecasts(backtest_result, *, loss="squared", quantile=None)
interval_score(actual, lower, upper, *, alpha=0.1)
```

`forecast_loss` 支持 squared、absolute 和 pinball：分别为 `(y-p)^2`、`|y-p|`、`max(q*(y-p),(q-1)*(y-p))`。Pinball 要求显式 `0<quantile<1`，预测应是该分位数；其他损失不能携带 quantile。数组须同形，预测可比实际多出最后一维模型轴。缺失、掩码、非数值和非有限输入拒绝，未作平均的损失逐观测返回；数值超出 float 范围时报错。

`ForecastEvaluation.losses` 保留 `(F,H,M)`；`mean_loss` 为 `(H,M)`，只沿起点平均。`lead_times` 是实际物理步长，包含 gap；`origins`、`target_indices` 保持位置。`records()` / `to_dict()` / `to_frame()` 输出按步长的描述评分，没有自动置信声明。

中央区间 score 为 `upper-lower + 2/alpha * ((lower-y)_+ + (y-upper)_+)`。要求边界同形、顺序正确、没有 NaN；允许向外无穷边界，对全实数区间评分为正无穷。空集合没有该区间 score，明确报错。极大惩罚也可能为正无穷。评分描述区间宽度与漏覆盖代价，不证明覆盖有效；请分别保存空集、无界比例，不把这些状态删掉后报告平均宽度。

## 比较候选模型

```python
compare_forecasts(result, *, baseline, lead_time=None, loss="squared",
                  quantile=None, alpha=0.05, lags=None, block_length=None,
                  **bootstrap_options)
```

候选差值定义为基准损失减候选损失；正值表示候选改善。基准、候选集、损失和预测步长应事先确定。多步结果必须显式选择一个 `lead_time`，不同步长不会混成更多观测。必须有至少两种模型、至少八个等间隔起点和非退化损失差；常数差时报错，不从检验族中删除。

该接口调用现有 `test_returns(..., method="bootstrap")`，默认逐轮重新估计 HAC。`bootstrap_options` 支持 `n_resamples`、`seed`、`batch_size`、`studentization` 与 `search_complete`。Gaussian AR 保证不能转移到一般预测损失差，故此接口不允许切换到 Gaussian AR 方法。它不是 Hansen SPA 或 Model Confidence Set 实现。

默认 lag 与 block 启发式同时考虑 `ceil(lead_time/origin_step)`，上限受起点数量限制；跨度超出可估计范围时返回 `warnings`，建议延长评估资料。明确传入的参数仍按原值验证。这样的带宽选择不是最优带宽估计，也不自动保证显著性准确。

`ForecastComparison` 保存 `baseline`、`lead_time`、`origin_step`、`loss`、`quantile`、基准/候选平均损失、`warnings` 与底层 `inference: TestResult`。表格包含平均改善、相对改善、调整 p 值及拒绝决定；基准损失为零时相对改善是 `None`，不做除零。`to_dict()` 保留方法诊断，另外输出 `comparison_warnings`。

保证范围要求损失差平稳、弱依赖、合适矩条件及尺度估计。模型反复调参、损失非平稳、持续监测后停止和跨多个步长同时选择，需要额外方法；普通候选表不解决这些问题。比较可以评估电力需求、销售量或其他预测序列，不局限于收益，但适用条件始终针对对应的损失差。

## 单步在线区间

```python
AdaptiveConformal(*, alpha=.1, step_size=.1, decay=.6,
                  scale=1., initial_quantile=.5)
adaptive_intervals(actual, predicted, *, alpha=.1, step_size=.1,
                   decay=.6, scale=1., initial_quantile=.5)
```

基础更新来自 [Angelopoulos–Barber–Bates，ICML 2024](https://proceedings.mlr.press/v235/angelopoulos24a.html)。令 `miss` 为先发出区间后、收到目标值时计算的误覆盖，更新 `q_next = q + eta_t*(miss-alpha)`，其中 `eta_t = step_size*t**(-decay)`。`t` 是已反馈预测数，而非日历时间。

原论文的任意序列平均覆盖界需要有界 score。实现使用事先固定的正 `scale=c`，把绝对残差 `u` 单调映射为 `u/(c+u)`，而不是假设金融残差自然有上界。这是实现选择，没有宣称新的 conformal 原理。在理想运算中，`0<=q<1` 的区间半径为 `c*q/(1-q)`；`q<0` 返回空集，`q>=1` 返回全实数，不截断 q。

`0<alpha<1`、`step_size>0`、`scale>0`、`0<=decay<1`、`initial_quantile∈[0,1]`。在理想递推中，历史平均误覆盖与 alpha 的偏差界为 `(1+step_size)/(T*eta_T)`；它可能很宽，不是每个时点的覆盖保证。分位数收敛还需原论文的 IID、固定 score、唯一分位数、连续分布等条件及 `decay∈(.5,1)`。当前采用 binary64，区间反馈以实际返回的闭区间判定；不声称精确浮点反演或舍入认证。

流式使用严格先预测、再反馈：

```python
tracker = AdaptiveConformal(alpha=.1, scale=1.)
interval = tracker.predict(point_forecast)
feedback = tracker.update(actual_when_available)
```

只允许一个 pending prediction；必须按顺序即时反馈。第二个 pending、无预测就更新、非有限值或溢出均报错，且失败不改变状态。标签应与待反馈预测匹配，由调用者负责。类状态内存为 O(1)，不保存整段历史。

`adaptive_intervals` 回放已对齐的一维、单步预测与目标；预测本身须已在不读取未来资料的情况下构造。它不会训练、挑选模型或验证该来源。结果 `ConformalResult` 包含实际值、预测、lower/upper、empty/unbounded、misses、发出时的 quantiles、后续 step_sizes、参数及 next_quantile；数组只读。`coverage` 是已实现的平均覆盖，不是未来概率。严格 JSON 通过 kind 和 `None` 边界区分空集与无界区间。

滚动结果接入此 wrapper 时，首例应使用 `gap=0,horizon=1,step=1`，或明确确认下一次发出预测前已收到上次标签。多步迟到反馈与未来路径同时覆盖尚未实现，不能靠展平多步数组套用此保证。

## 当前与后续

0.6 提供一条能调用和核验的工作流程。经典模型接口、数据频率诊断、panel、成熟标签队列、序贯模型集合、跨步长联合推断及现代条件覆盖仍按[路线图](toolbox-roadmap.md)逐项扩展。未来方法上线前须核验原论文、独立参考、假设与复杂度；本页的基础算法和合理工程组织不作为研究创新宣称。
