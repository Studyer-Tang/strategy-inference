# strategy-inference

面向预测评估、在线不确定性与策略均值推断的 Python 时序工具箱。v0.8 提供可调的成熟尺度融合：既保留自身步长的误差信息，也能利用短步长的共同波动信息。支持滚动预测、损失评估、固定候选比较、单步与多步在线区间；原有收益均值接口继续可用。

[English](README.en.md) · [在线文档](https://studyer-tang.github.io/strategy-inference/library/) · [时序 API](docs/time-series.md) · [多步 API](docs/multistep-api.md) · [均值 API](docs/api.md) · [路线图](docs/toolbox-roadmap.md) · [研究与复现](docs/research.md)

## 安装

需要 Python 3.10+，核心依赖为 NumPy 与 SciPy。可直接安装 GitHub release wheel，无需 clone。尚未发布到 PyPI：

```bash
python -m pip install https://github.com/Studyer-Tang/strategy-inference/releases/download/v0.8.1/strategy_inference-0.8.1-py3-none-any.whl
```

对应标签的源码安装方式：

```bash
python -m pip install 'git+https://github.com/Studyer-Tang/strategy-inference.git@v0.8.1'
```

开发时，在源码 checkout 中运行 `python -m pip install -e '.[dev]'`。只有 `to_frame()` 等 DataFrame 功能需要可选 pandas，可用 `python -m pip install 'pandas>=2'` 安装。

## 从预测到比较与区间

下面的例子生成模拟序列，预先确定模型和比较步长，不需要下载数据：

```python
import numpy as np
from strategy_inference import (
    backtest, evaluate_forecasts, compare_forecasts, adaptive_intervals,
    naive_forecast, drift_forecast, SeasonalNaive,
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
# 单步、连续 origin：先发出 naive 预测，再用该步真实值更新。
intervals = adaptive_intervals(
    run.actuals[:, 0], run.forecasts[:, 0, 0], alpha=0.1, scale=1.0,
)

print(run.forecasts.shape)        # (262, 3, 3): origin × lead × model
print(scores.mean_loss)           # (3, 3): 每个 lead、model 的平均损失
print(comparison.records())       # 相对 naive 的改善、调整 p 值与拒绝决定
print(intervals.coverage)         # 已评价单步预测的实际平均覆盖
```

`mean_improvement > 0` 表示候选损失低于 baseline。拒绝决定检验期望改善是否大于零，不能仅按平均损失最小选择后再解释未经调整的 p 值。该示例演示接口，实际平均覆盖也不是每个时点的覆盖概率。

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

[240 条独立路径的验证](docs/blended-scales-results.md)量化了这种折中：24 步预测含步长特有偏差时，半权重融合相对 `shortest` 的平均区间评分降低 1.31%，最坏 200 点窗口覆盖误差降低 35.0%；稳定 Gaussian 情形的评分增加 0.63%。固定尺度在多种情形得分更低，融合没有一般效率优势。

每时发行且队列填满后，各步长都在同一日历时点收到当前标签；共享尺度改变的是预测起点新旧与残差信息，不会提前取得长步长标签或消除其阈值反馈延迟。

流式 `MultiStepConformal` 按 `observe(t, y[t])` 再 `predict(path)` 运行，连续整数观测时钟不能跳步；结束时未成熟预测保留 pending。`pooled` 每个步长一个状态，`interlaced` 按起点相位拆分；两者成熟计数与学习率时钟不同。见[多步 API](docs/multistep-api.md)、[方法与证明](docs/multistep-methods.md)及[可运行例子](examples/multistep.py)。

## 已实现的接口

| 接口 | 输入与输出 |
| --- | --- |
| `rolling_splits(...)`、`backtest(y, forecasters, ...)` | 有限、等间隔的一维序列；保存训练/测试位置、每个 origin 的预测、真实值和目标索引。`window=None` 为扩展窗，整数为滚动窗上限 |
| `naive_forecast`、`SeasonalNaive(period)`、`drift_forecast` | 三种透明 baseline；也可提供 `callback(train, lead_times)`，返回指定 lead 的一维预测 |
| `forecast_loss(...)`、`evaluate_forecasts(run, ...)` | 平方、绝对、pinball 损失；保留 origin × lead × model 损失，按 lead 分别汇总。Pinball 预测须对应指定分位数 |
| `interval_score(actual, lower, upper, alpha=...)` | 中心区间评分，包含宽度与漏覆盖距离惩罚；全域区间评分为无穷，空集不支持 |
| `compare_forecasts(run, baseline=..., lead_time=...)` | 一个预先指定 baseline、一个 lead、固定候选集的共享时间索引 max bootstrap，返回全族/候选决定及诊断 |
| `AdaptiveConformal`、`adaptive_intervals(...)` | 单步、有序完整反馈的递减步长 quantile tracker，返回区间、空集/全域状态和实际覆盖 |
| `MultiStepConformal`、`multistep_intervals(...)` | 单变量整数时钟、多步成熟反馈，pooled/interlaced 状态、冻结发行尺度和逐步长汇总 |
| `test_returns(...)`、`infer_mean(...)` | 原有同期收益矩阵的同时或逐列均值推断 |

`run.to_dict()` 保存完整回测数组与 splits；`scores.losses` 保留逐 origin 损失，`scores.to_dict()` 导出汇总；比较和在线区间也提供 `to_dict()`。导出候选表或回测长表可使用 `to_frame()`，需要 pandas。详细参数、数组形状和结果语义见[时序 API](docs/time-series.md)。

## 方法与适用条件

回测按照输入顺序使用当前训练窗。每次 callback 得到独立的只读训练数据与 lead 数组；使用者仍须避免 callback 通过闭包、外部状态或数据源读取未来信息。缺失、非有限输入和无效预测会报错；软件不自动重排日期、填补或重新采样。

预测比较使用 baseline 损失减去候选损失。Bootstrap 近似需要损失差序列平稳、弱依赖、适当矩及非退化方差，渐近讨论固定候选数。重叠预测保留 origin 顺序；多步回测必须指定 `lead_time`，不会把多个 lead 当作独立样本。默认 HAC lag 和块长同时考虑重叠与样本量，仍是启发式；rolling/expanding 切分本身不建立统计条件。常数损失差会报错。候选、baseline、损失和 lead 应事先确定，跨 lead 联合检验、隐藏搜索和反复查看后停止需要另外处理。

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

日常使用从 `src/strategy_inference/` 和 `examples/` 开始；`scripts/` 保留复现与发布工具，`results/` 保存研究证据。v0.8.1 将淘汰的尺度学习原型及其专用测试移出活跃目录，正式研究的生成器、路径统计与校验统一在 `scripts/_scale_study.py`。旧方案的源码和测试仍可在 [v0.8.0 标签](https://github.com/Studyer-Tang/strategy-inference/tree/v0.8.0)复核，结果及源码 ZIP 保留原样。

[路线图](docs/toolbox-roadmap.md)区分已实现功能与待核验扩展；[研究索引](docs/research.md)保留历史评价、失败边界、冻结协议和原始证据。基础预测、bootstrap 与 conformal 算法的实现不作为算法创新主张。

历史 v0.6 的[归档页面](https://studyer-tang.github.io/strategy-inference/library/v0.6.0/)、[性能说明](docs/time-series-performance.md)和[原始记录](benchmarks/results/time-series-0.6.json)保存当时运行条件、五次热调用与源码哈希。它们绑定 v0.6 发布源码；v0.7 的[多步性能说明](docs/multistep-performance.md)和[原始记录](benchmarks/results/multistep-0.7.json)另行报告。[融合性能说明](docs/blended-scales-performance.md)区分冻结的 v0.8.0 测量和 v0.8.1 重构对照；本机工程测量不提供通用速度或统计校准保证。

历史证据保留在 [v0.5 页面与完整 benchmark](https://studyer-tang.github.io/strategy-inference/library/v0.5.0/)和[v0.5 性能说明](docs/performance.md)。旧版本计时不代表当前接口。历史正式研究应在 metadata 指定的冻结 commit/tag 下复核。

```bash
python -m pytest
ruff check .
python scripts/sync_protocols.py --check
python scripts/build_library_site.py --check
python scripts/build_toolbox_site.py --check
python scripts/build_multistep_site.py --check
python scripts/build_blended_site.py --check
```

BSD-3-Clause 许可证。
