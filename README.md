# strategy-inference

用于时序预测比较与不确定性评估的 Python 库。按时间回测，在验证段选择预测器，评估样本外损失与依赖调整的比较结果，并用已成熟标签更新在线区间。保留预测时点、候选分数和数据来源。

[English](README.en.md) · [在线文档](https://studyer-tang.github.io/strategy-inference/library/) · [时序 API](docs/time-series.md) · [多步 API](docs/multistep-api.md) · [均值 API](docs/api.md) · [路线图](docs/toolbox-roadmap.md) · [研究与复现](docs/research.md)

## 安装

需要 Python 3.10+，核心依赖为 NumPy 与 SciPy。可直接安装 GitHub release wheel，无需 clone。尚未发布到 PyPI：

```bash
python -m pip install https://github.com/Studyer-Tang/strategy-inference/releases/download/v0.11.0/strategy_inference-0.11.0-py3-none-any.whl
```

对应标签的源码安装方式：

```bash
python -m pip install 'git+https://github.com/Studyer-Tang/strategy-inference.git@v0.11.0'
```

开发时，在源码 checkout 中运行 `python -m pip install -e '.[dev]'`。只有 `to_frame()` 等 DataFrame 功能需要可选 pandas，可用 `python -m pip install 'pandas>=2'` 安装。

## 从预测到比较与区间

下面的例子生成模拟序列，预先确定模型和比较步长，不需要下载数据：

```python
import numpy as np
from strategy_inference import (
    backtest, evaluate_forecasts, compare_forecasts, adaptive_intervals,
    sequential_compare_forecasts, naive_forecast, drift_forecast, SeasonalNaive,
)

rng = np.random.default_rng(17)
y = np.empty(384)
y[0] = rng.normal()
for t in range(1, len(y)):
    y[t] = 0.6 * y[t - 1] + 0.8 * rng.normal()

models = {
    "naive": naive_forecast,
    "seasonal": SeasonalNaive(period=12),
    "drift": drift_forecast,
}
run = backtest(
    y, models, initial_train_size=120, window=120, horizon=3,
)
scores = evaluate_forecasts(run, loss="squared")
comparison = compare_forecasts(
    run, baseline="naive", lead_time=1, loss="squared",
    n_resamples=1999, seed=17, search_complete=True,
)
sequential = sequential_compare_forecasts(run, lead_time=1, loss="absolute")
# 单步、连续 origin：先发出 naive 预测，再用该步真实值更新。
intervals = adaptive_intervals(
    run.actuals[:, 0], run.forecasts[:, 0, 0], alpha=0.1, scale=1.0,
)

print(run.forecasts.shape)        # (262, 3, 3): origin × lead × model
print(scores.mean_loss)           # (3, 3): 每个 lead、model 的平均损失
print(comparison.records())       # 相对 naive 的改善、调整 p 值与拒绝决定
print(sequential.to_dict())       # 固定模型族的连续比较快照
print(intervals.coverage)         # 已评价单步预测的实际平均覆盖
```

`mean_improvement > 0` 表示候选损失低于 baseline。上例的 bootstrap 使用平方损失，连续比较使用绝对损失；两者统计目标和条件见下文。示例演示接口，不建立模型优劣或覆盖保证。

## 真实数据

```python
from strategy_inference import Autoregression, load_dataset, backtest, naive_forecast

data = load_dataset("fred_md")  # 第一次下载 169 KB；之后校验并复用缓存
y = data["T1"].values
run = backtest(
    y, {"naive": naive_forecast, "ar": Autoregression(lags=12, ridge=1.0)},
    initial_train_size=600, window=120,
)
```

内置 `fred_md`、`bitcoin`、`oikolab_weather` 三个官方历史档案，总压缩大小约 1.7 MB。固定源版本与 SHA-256，支持 `offline=True`；不需要 Hugging Face SDK、pandas 或远程加载脚本。`read_tsf(path)` 也可读取自己的 TSF/ZIP 文件。缺失位置保持为 NaN，回测前须明确处理；软件不会自动删行或填补。

仓库内运行 `python examples/real_data.py --dataset all`，得到三个固定序列的整齐评分表。例子按时间划分训练、验证、测试段，在共同验证目标上选择窗口、滞后、差分和 ridge，在测试段逐时重新拟合；MASE 的尺度只来自训练段。`--output result.json` 保存来源、候选与选择结果，缓存完成后可加 `--offline` 复现。这些已公开样例上的迭代是开发比较。

数据来自 [Monash 时间序列档案](https://huggingface.co/datasets/Monash-University/monash_tsf)，原始记录采用 CC BY 4.0。FRED-MD 保留档案提供的数值预处理和匿名列名，未提供历史发布版本；这些数据可用于预测评估，不能据此声称无修订信息的实时交易回测。来源和具体协议见[时序 API](docs/time-series.md#真实数据与缓存)。

独立核验在当前 `main` 的源码目录中运行：`python benchmarks/forecast_validation.py --mode full --cache-dir .cache --output forecast-validation.json`（需 `.[dev]`）。[事先固定的协议](benchmarks/forecast_validation_protocol.json)包含五个此前未用于开发例子的 FRED-MD 序列、同训练窗 [statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html) 对照，以及同一均值损失目标下的 AR(1) 错误率／功效实验。[完整账本](benchmarks/results/forecast-validation-0.11.json)保留所有逐次决定与 Monte Carlo 不确定性。这是同档案多序列扩展，不是独立外部数据集验证；已知参数 oracle 仅用于校准参考。

验证集选择可直接使用公共接口：

```python
from strategy_inference import Differenced, select_forecaster

choice = select_forecaster(
    y[:400], y[400:600],
    {"level": Autoregression(12), "changes": Differenced(Autoregression((1, 2, 12)))},
    window=120, loss="absolute",
)
run = backtest(y, {"chosen": choice.forecaster}, initial_train_size=600, window=120)
print(choice.name, choice.to_dict())
```

`lags=(1, 2, 24, 168)` 可用少量系数表示短期和日／周信息；`Differenced(model, period=24)` 在小时数据上做季节差分并还原预测。默认仍是未差分的密集 AR。相关设计参考 [Huang、Xu、Darlow（2026）](https://arxiv.org/abs/2606.27282) 对线性预测器表示和验证选择的研究；本库采用独立的递推实现。

## 多步区间与成熟反馈

多步预测须等到各自目标标签成熟后才更新。批量接口保留起点和物理步长，不把回测每行当作立即反馈：

```python
from strategy_inference import multistep_intervals

walk = np.cumsum(np.random.default_rng(17).normal(size=384))
multi_run = backtest(
    walk, {"naive": naive_forecast}, initial_train_size=128, window=128, horizon=12,
)
origins = np.asarray([split.origin for split in multi_run.splits])
leads = multi_run.target_indices[0] - origins[0]  # 包含 gap 的物理步长
multi = multistep_intervals(
    walk, multi_run.forecasts[:, :, 0], origins=origins, lead_times=leads,
    scale=np.sqrt(leads), step_size=0.1 / np.sqrt(leads), decay=0.2,
    initial_quantile=0.65, strategy="pooled", scale_decay=0.97,
    scale_source="blended", scale_share_weight=0.5,
)
print(multi.summary())
```

这里 `sqrt(leads)` 来自单位创新随机游走的已知模拟尺度，按步长降低学习率只是示例。`blended` 按预先固定的权重融合自身步长 RMS 与短步长共享 RMS；权重可用标量或逐步长向量，发行尺度冻结。也可用固定尺度、`horizon` 或 `shortest`。权重应由训练/独立验证数据确定，`0.5` 是折中默认值，不是最优参数。

融合的收益与代价见[独立路径研究](docs/blended-scales-results.md)；它没有一般效率优势。

每时发行且队列填满后，各步长都在同一日历时点收到当前标签；共享尺度改变的是预测起点新旧与残差信息，不会提前取得长步长标签或消除其阈值反馈延迟。

流式 `MultiStepConformal` 按 `observe(t, y[t])` 再 `predict(path)` 运行，连续整数观测时钟不能跳步；结束时未成熟预测保留 pending。`pooled` 每个步长一个状态，`interlaced` 按起点相位拆分；两者成熟计数与学习率时钟不同。见[多步 API](docs/multistep-api.md)、[方法与证明](docs/multistep-methods.md)及[可运行例子](examples/multistep.py)。

## 已实现的接口

| 接口 | 输入与输出 |
| --- | --- |
| `rolling_splits(...)`、`backtest(y, forecasters, ...)` | 有限、等间隔的一维序列；保存训练/测试位置、每个 origin 的预测、真实值和目标索引。`window=None` 为扩展窗，整数为滚动窗上限 |
| `naive_forecast`、`SeasonalNaive(period)`、`drift_forecast`、`Autoregression(lags, ridge)` | 透明 baseline；AR/ridge 只在当前训练窗拟合。也可提供 `callback(train, lead_times)`，返回指定 lead 的一维预测 |
| `load_dataset(name)`、`read_tsf(path)` | 固定版本真实数据、校验缓存及本地 TSF；保留缺失位置和来源，不执行远程代码 |
| `Differenced(model, period)`、`select_forecaster(train, validation, models)` | 可组合的差分与预测还原；在共同验证目标上选择模型，返回可直接用于后续回测的 callable |
| `forecast_loss(...)`、`evaluate_forecasts(run, ...)` | 平方、绝对、pinball 损失；保留 origin × lead × model 损失，按 lead 分别汇总。Pinball 预测须对应指定分位数 |
| `interval_score(actual, lower, upper, alpha=...)` | 中心区间评分，包含宽度与漏覆盖距离惩罚；全域区间评分为无穷，空集不支持 |
| `compare_forecasts(run, baseline=..., lead_time=...)` | 一个预先指定 baseline、一个 lead、固定候选集的共享时间索引 max bootstrap，返回全族/候选决定及诊断 |
| `SequentialModelConfidenceSet`、`sequential_compare_forecasts(run, ...)` | 固定模型族、绝对或 pinball 损失、非重叠反馈；返回持续更新且只缩小的模型置信集及 log e-value 快照 |
| `AdaptiveConformal`、`adaptive_intervals(...)` | 单步、有序完整反馈的递减步长 quantile tracker，返回区间、空集/全域状态和实际覆盖 |
| `MultiStepConformal`、`multistep_intervals(...)` | 单变量整数时钟、多步成熟反馈，pooled/interlaced 状态、冻结发行尺度和逐步长汇总 |
| `test_returns(...)`、`infer_mean(...)` | 原有同期收益矩阵的同时或逐列均值推断 |

`run.to_dict()` 保存完整回测数组与 splits；`scores.losses` 保留逐 origin 损失，`scores.to_dict()` 导出汇总；比较和在线区间也提供 `to_dict()`。导出候选表或回测长表可使用 `to_frame()`，需要 pandas。详细参数、数组形状和结果语义见[时序 API](docs/time-series.md)。

## 方法与适用条件

回测按照输入顺序使用当前训练窗。每次 callback 得到独立的只读训练数据与 lead 数组；使用者仍须避免 callback 通过闭包、外部状态或数据源读取未来信息。缺失、非有限输入和无效预测会报错；软件不自动重排日期、填补或重新采样。

固定样本比较使用 baseline 损失减去候选损失。Bootstrap 近似需要损失差平稳、弱依赖、适当矩及非退化方差，候选族固定。多步回测必须指定 `lead_time`；默认 lag 和块长考虑重叠与样本量，但仍是启发式。回测切分不建立这些条件，也不自动处理隐藏搜索或反复查看后停止。

连续比较采用 [Arnold 等，JRSSB 2026](https://doi.org/10.1093/jrsssb/qkag066) 的 strong conditional-superiority 构造：目标模型须在每次发行时的信息条件下，对所有其他固定模型都有不高于它们的期望损失。该集合可为空；被排除的模型不会恢复，之后仍须提供全部原模型的预测。每个标签须在下一次发行前成熟，多步回测要求相邻 origin 间隔至少为所选物理 lead。`bet_fraction` 事先固定在 `(0, .5]`，默认 `.25`；不声称最优或浮点舍入认证。参数、证据字段与短流式例子见[时序 API](docs/time-series.md#连续模型置信集)。

单步在线区间实现 [Angelopoulos–Barber–Bates（ICML 2024）](https://proceedings.mlr.press/v235/angelopoulos24a.html) 的递减步长更新，采用固定尺度有界残差映射。多步模块另外显式维护成熟反馈队列；每个步长的理想递推控制已成熟预测的历史平均误覆盖，不提供逐时条件覆盖或整条路径同时覆盖。它是独立工程实现，没有复刻完整 AcMCP 的 PID 与 scorecaster。空集与全域显式保留，阈值不截断；普通浮点反馈按返回闭区间计算，没有舍入证书。完整条件见[时序 API](docs/time-series.md)与[多步方法](docs/multistep-methods.md)。

## 收益均值推断

```python
import numpy as np
from strategy_inference import test_returns

rng = np.random.default_rng(17)
returns = rng.normal(0.0, 0.01, size=(512, 3))
result = test_returns(returns, n_resamples=1999, seed=17)
print(result.global_pvalue, result.adjusted_pvalue)
```

`test_returns` 支持 NumPy、可选 DataFrame 和 `read_returns_csv` 返回的表，行是时间、列是候选。要求 `T >= 8`、同期等间隔、有限值与正样本方差。输入保持单期单位；检验基准调整收益时须先扣基准与成本。

默认 `method="bootstrap"` 使用共享 stationary-bootstrap 时间索引与每轮重新估计的 HAC 尺度，适用条件同样包含平稳、弱依赖与适当矩。`method="gaussian_ar"` 提供共同平稳 Gaussian AR(1)、`0 <= phi < 1`、事先固定候选与信息配置下的保守强 FWER 决定，返回参数集合而不提供连续 p 值；信息退化或认证未完成时保守不拒绝。完整限制见[均值 API](docs/api.md)和[联合方法证明](docs/joint-uncertainty.md)。

`audit_returns`、`uncertainty_test`、`wilks_uncertainty_test`、`infer_mean` 和 `long_run_variance` 保留；兼容接口 `audit_returns` 默认 `studentization="fixed"`，`test_returns` 默认 `"resampled"`。CLI 保留收益检验与报告：

```bash
strategy-inference test examples/demo_returns.csv \
  --method bootstrap --n-resamples 1999 --seed 17 --output result.json
```

## 文档、研究与性能

使用例子见[时序工作流](examples/time_series.py)和[多步区间](examples/multistep.py)；[路线图](docs/toolbox-roadmap.md)区分已实现与计划能力。[研究索引](docs/research.md)保留历史评价、失败边界和冻结证据，正式复核须使用 metadata 指定的 commit/tag。基础算法的实现不作为算法创新主张。

性能记录按版本保存：[预测评估](docs/time-series-performance.md)、[多步区间](docs/multistep-performance.md)、[尺度融合与重构](docs/blended-scales-performance.md)。旧版测量不代表当前新增接口，本机计时也不保证通用速度或统计校准。

```bash
python -m pytest
ruff check .
```

BSD-3-Clause 许可证。
