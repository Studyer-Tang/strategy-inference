# 时间序列评估与在线区间

真实数据、预测评估、固定样本比较、连续模型置信集与单步在线区间的公共接口。多步成熟反馈见[多步 API](multistep-api.md)，原收益推断见[均值 API](api.md)，后续范围见[路线图](toolbox-roadmap.md)。核心依赖为 NumPy 与 SciPy。

## 真实数据与缓存

```python
available_datasets()  # ("fred_md", "bitcoin", "oikolab_weather")
load_dataset(name, *, cache_dir=None, offline=False, timeout=30.0)
read_tsf(path, *, encoding="cp1252")
```

返回 `TimeSeriesDataset`：`.names` 是原始名称元组，`data[name]` 返回 `TimeSeries`，`.values` 是只读 float64 一维数组，`.attributes` 保存文件中的属性。`.start_timestamp` 来自原文件；缺失时返回 `None`，不虚构日期或时区。数据集的 `.metadata` 保存 TSF 头，`.frequency` 返回频率字符串，`.source`、`.license`、`.revision`、`.sha256` 记录来源；本地读取不推断源网站或许可，SHA-256 对应所读文件字节。

| 名称 | 原始内容 | ZIP 大小 | 原始档案 |
| --- | --- | --- | --- |
| `fred_md` | 107 条月序列，各 728 点，匿名 T1…T107 | 169,107 B | [FRED-MD v2](https://zenodo.org/records/4654833) |
| `bitcoin` | 18 条日序列，各 4,581 点，含缺失 | 220,403 B | [Bitcoin v1](https://zenodo.org/records/5121965) |
| `oikolab_weather` | 8 条小时序列，各 100,057 点 | 1,326,101 B | [Oikolab v1](https://zenodo.org/records/5184708) |

上述 CC BY 4.0 存档由 Godahewa、Bergmeir、Webb、Hyndman 与 Montero-Manso 整理，见 [Monash Time Series Forecasting Archive（NeurIPS 2021）](https://arxiv.org/abs/2105.06643)。下载来自[官方 Hugging Face 仓库](https://huggingface.co/datasets/Monash-University/monash_tsf/tree/58aafbe2712ff481c014f562e42723f2820fd5d4)，固定 revision `58aafbe2712ff481c014f562e42723f2820fd5d4` 和各文件 SHA-256。读取原始 ZIP，无 HF SDK、pandas 或远程代码依赖。缓存默认位于 `~/.cache/strategy-inference/datasets`（尊重 `XDG_CACHE_HOME`），下载通过大小、哈希与解析校验后原子写入。每次缓存读取都校验；损坏时明确报错，移除该缓存文件后可重下。`offline=True` 缓存未命中直接报错，不联网。

TSF 的 `?` 保留为原位置 NaN；不填补、不删行、不重采样。`read_tsf` 支持单文件 TSF 或仅含一个 TSF 的 ZIP，压缩输入和解压内容均限 32 MiB，不向磁盘解压 ZIP。默认编码遵循原作者解析器，本地 UTF-8 可显式指定。无 `series_name` 属性时用一基行号命名。

FRED-MD 保留档案提供的预处理，不从匿名列名猜测指标，不是发布时点 vintage 数据。Bitcoin 的 `price` 有 560 个前导缺失，之后是 4,021 个连续有限观测。Oikolab 是供应商提供的历史气候数据，没有证据将该快照称作气象站直接实测；文件未给时区，其起点和点数也不应由文字简介的结束日期替换。三个数据集是历史研究快照。

[真实数据例子](../examples/real_data.py) 固定 FRED-MD/T1、Bitcoin/price 的有限连续后缀及 Oikolab/T1 的末 8,760 小时，按整数位置划分 70% 训练、15% 验证及其余测试目标。这不是 Monash 官方基准划分；官方 HF 的 train/validation/test 是嵌套前缀，不能拼接。每条序列给出固定的 24 个配置：两种窗口、密集／稀疏滞后、原尺度／一阶差分、三个 ridge。候选共享相同验证目标，在验证段选择，测试前冻结，逐起点仅使用已观测历史重新拟合。简单基线保持原设置。输出 MAE、原单位 RMSE、仅由初始训练段季节差分定标的 MASE，以及 strong conditional-superiority 的连续集合；集合不表示平均误差排名。v0.11 在这些已公开样例上的迭代属于开发比较。

训练段定标和逐起点评价与近期 [fev-bench（2026 版本）](https://arxiv.org/abs/2509.26468) 的设计一致；这里是三个序列的可运行例子，没有复现其完整 benchmark 或 task-bootstrap 结论。公开历史序列可能进入基础模型预训练，因此本例使用从头拟合的基线，不宣称基础模型的无污染零样本比较。

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
| `Autoregression(lags=12, ridge=0)(train, lead_times)` | 带截距密集或稀疏 AR；整数 p 使用 1…p，元组指定不同的正整数滞后并排序。训练至少 `max_lag+n_lags+1` 点，密集情形为 `2*p+1`；按物理 lead 递推 |
| `Differenced(model, period=1)(train, lead_times)` | 只在本次训练窗内计算 `y[t]-y[t-period]`，调用任意预测器，再逐物理步还原原尺度；常规／季节差分，无自动平稳性判断 |

AR 在当前训练窗内中心化并以最大绝对偏差定标，拟合目标为归一化残差平方和加 `ridge * sum(lag_coefficients**2)`，截距不惩罚。`ridge=0` 是 OLS，秩不足时使用中心化系数系统的最小范数解；正 ridge 采用增广设计和 `lstsq`，不显式求逆或构造正规方程。常数训练序列返回常数预测。实例不保留折间状态，不自动选 lag 或 penalty；参数选择须先做训练内或独立验证。

稀疏滞后如 `(1, 2, 24, 48, 168)` 只拟合五个系数；默认整数 AR 的行为保持不变。`Differenced` 支持不连续、乱序或重复的请求步长，先预测 1…最大 lead 的全部差分，再按季节锚点逐步还原，包含 gap 内的未请求步。原训练数据及回调收到的差分／lead 不共享未来数据；任一中间差分、预测或还原结果超出有限浮点范围都会报错。其工作空间随最大请求步长增长，不能把长 lead 当作只计算一个标量。

季节周期和窗口应事先确定。每折重新调用模型；若 callback 每次重新拟合，计算成本包含拟合本身。使用较小的固定 `window` 或较大的 `step` 可以限制工作量，同时改变评估设计。当前没有自动增量更新、外生变量或多序列 panel 接口。

其他预测库可通过 callable 接入。模型训练、超参数选择和预处理应限制在传入的训练资料中；本接口不替使用者确认嵌套验证过程。

## 验证集选择

```python
select_forecaster(train, validation, forecasters, *, loss="absolute",
                  quantile=None, lead_time=1, step=1, window=None)
```

训练、验证分开输入，二者均须为非空、有限一维序列；日期顺序由调用者保证。第一起点是训练段末尾，一个固定的物理 lead=h 使用 `gap=h-1, horizon=1`；验证段前 `h-1` 个标签没有这个起点发行的预测，因此不评分。所有模型使用相同目标位置、损失和给定训练窗口。验证期间后续起点可用已成熟的验证标签，回调仍只能使用当时的历史。

返回 `ForecastSelection`：`.name` 是最小平均验证损失的名称，`.forecaster` 保留该 callable，`.names`、只读 `.mean_loss` 保存完整映射顺序与分数，完全相同的分数取第一个。只读 `.target_indices` 是训练＋验证前缀中的绝对位置。`.to_dict()` 保存分数、样本数、损失、lead 与半开目标范围；当 `step>1` 时，该范围是包络，内部存在未评价目标。选中的 callable 可用于随后独立的测试段；结果不声称统计优越性。回调捕获外部未来资料仍需使用者约束。

窗口和表示选择参考 [Huang、Xu、Darlow（2026-06）](https://arxiv.org/abs/2606.27282) 对线性预测器的研究，尤其保持候选验证目标一致。本库实现递推 AR 与显式差分组合；论文采用直接多步 ridge、三折验证及更广的表示搜索，本文例子采用单步滚动验证，未复现其全套算法或 benchmark。

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

## 固定样本比较

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

## 连续模型置信集

```python
SequentialModelConfidenceSet(names, *, alpha=.05, loss="absolute",
                             quantile=None, bet_fraction=.25)
sequential_compare_forecasts(result, *, lead_time=None, alpha=.05,
                             loss="absolute", quantile=None, bet_fraction=.25)
```

这是 [Arnold 等，JRSSB 2026，Sequential model confidence sets](https://doi.org/10.1093/jrsssb/qkag066) 的 **strong conditional-superiority** 构造，采用 Proposition 3.2、Eq.6–7 与补充 H 的闭合调整；[原文 HTML v4](https://arxiv.org/html/2404.18678v4)提供完整公式。设 `L_i,t` 为模型 i 的损失，目标是对**每次**发行时的信息 \(\mathcal F_{t-1}\) 及**所有**其他固定模型 j 满足

\[
\mathbb E[L_{i,t}-L_{j,t}\mid\mathcal F_{t-1}]\le 0.
\]

在这些条件下，理想运算中的置信集以至少 `1-alpha` 的概率在全部评估时刻保留所有目标模型；不要求 IID、平稳或事先固定结束时点。目标集合可能为空。当前接口不实现原论文的累计平均风险或随时间变化的平均风险构造，不能把留下的模型解释为已证实的“目前平均损失最小者”。

`names` 至少含两个不同的非空标签。模型族、`alpha`、损失、quantile 和 betting fraction 须事先固定；模型内部可用已经可用的历史重新拟合。支持 `absolute` 或 `pinball`；后者须给出 `0<quantile<1`，预测应对应该分位数。其他损失不能带 quantile。`bet_fraction` 在 `(0,.5]` 内，默认 `.25` 是保守固定选择，不声称最优。

必须先 `predict(forecasts)` 登记 names 顺序的有限实数向量，再 `update(actual)` 反馈共同标签。只允许一个 pending 标签，标签须在下一次发行前成熟；即使某模型被排除，之后仍须提供全部原模型的预测。流式例子：

```python
from strategy_inference import SequentialModelConfidenceSet
tracker = SequentialModelConfidenceSet(("a", "b"), loss="absolute")
tracker.predict([1.0, 2.0])
print(tracker.update(1.2))
tracker.predict([1.5, 2.0])
print(tracker.to_dict())  # 第二次预测仍 pending。
```

成对损失差的事前界为 `B_ij = |forecast_i-forecast_j|`，pinball 则再乘 `max(q,1-q)`。实现直接计算差值与界的比率，不截断原始损失或观测；相同预测的因子为 1。各 pair 的财富乘上 `1 + bet_fraction*(L_i-L_j)/B_ij`，再对模型平均并作闭合检验；调整后的 e-value 达到 `1/alpha` 时排除模型。`confidence_set` 是历次集合的交集，排除永久生效，不恢复模型。

`sequential_compare_forecasts` 返回同一个 tracker，处理回测所选 lead 后仍可继续流式调用。多步回测必须指定物理 `lead_time`，相邻 origin 间隔须至少等于该 lead；重叠反馈明确报错，不展平多步数组。预测来源须只使用当时可用的信息，wrapper 不替调用者验证强条件 null。

`names`、`n_updates`、`confidence_set`、`pending` 可读取；`log_evalues` 和 `log_adjusted_evalues` 为当前模型层面的只读向量。`to_dict()` 导出参数、这两个证据向量、计数、跑动集合与 pending 的严格 JSON 快照，未保存整段预测历史。e-value 不是 p 值或模型正确的概率；当前证据可能回落，跑动集合仍保留先前排除。非法输入不改变状态。内存为 `O(M²)`；更新为 `O(M²)` 加 `O(M log M)` 的闭合调整。普通 float64 未作舍入认证。

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

滚动结果接入此 wrapper 时，首例应使用 `gap=0,horizon=1,step=1`，或确认下一次发行前已收到上次标签。多步成熟反馈使用[多步接口](multistep-api.md)，不能靠展平数组套用单步保证；整条未来路径同时覆盖尚未实现。

## 当前与后续

当前已提供滚动评估、固定样本比较、strong 连续模型置信集及单步/多步区间。经典模型适配、数据频率诊断、panel、跨步长联合推断及现代条件覆盖仍见[路线图](toolbox-roadmap.md)。既有算法的实现和接口整合不作为基础算法创新宣称。
