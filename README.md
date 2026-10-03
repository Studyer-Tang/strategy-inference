# strategy-inference

用于时间依赖数据和候选策略筛选的 Python 均值推断库。输入同期收益矩阵，返回全族检验、列级决定和诊断信息。

[English](README.en.md) · [在线文档](https://studyer-tang.github.io/strategy-inference/library/) · [API](docs/api.md) · [性能](docs/performance.md) · [方法说明](docs/methods.md) · [研究与复现](docs/research.md)

## 安装

需要 Python 3.10+，核心依赖为 NumPy 与 SciPy。直接安装 GitHub v0.5.0 release 的 wheel，无需 clone：

```bash
python -m pip install https://github.com/Studyer-Tang/strategy-inference/releases/download/v0.5.0/strategy_inference-0.5.0-py3-none-any.whl
```

也可从对应标签安装源码；开发或绘图时使用 checkout：

```bash
python -m pip install 'git+https://github.com/Studyer-Tang/strategy-inference.git@v0.5.0'

# 开发与绘图：
git clone --branch v0.5.0 https://github.com/Studyer-Tang/strategy-inference.git
cd strategy-inference
python -m pip install -e '.[figures,dev]'
```

尚未发布到 PyPI。示例 CSV、研究脚本与保存的实验结果在源码仓库中。

## 最小可运行例子

```python
import numpy as np
from strategy_inference import test_returns

rng = np.random.default_rng(17)
returns = rng.normal(0.0, 0.01, size=(512, 3))
returns[:, 0] += 0.0005

result = test_returns(
    returns,
    method="bootstrap",
    names=["strategy_a", "strategy_b", "strategy_c"],
    n_resamples=1999,
    seed=17,
    search_complete=True,  # 本例的候选集在生成数据前已确定。
)
print(result.global_pvalue)    # 全部候选均值均不大于零的全族检验。
print(result.adjusted_pvalue)  # 长度 K 的列级调整 p 值。
print(result.decisions)        # 指定 alpha 下的列级拒绝决定。
```

数据为模拟的单期收益，单位是小数。默认 bootstrap 每轮重新估计 HAC 尺度；结果依赖重抽样近似。这个例子演示调用，不为任意金融序列提供有限样本保证。

## 核心接口

```text
test_returns(returns, *, method="bootstrap", alpha=0.05, names=None, **options)
```

输入支持 NumPy 数组、数值 array-like、可选 pandas DataFrame，以及 `read_returns_csv` 返回的 `ReturnTable`。行是时间、列是候选，形状为 `(T, K)`；一维输入按单列处理。要求 `T >= 8`、各列方差为正、所有值有限。DataFrame 与 CSV 自动提供列名。

观测须等间隔、同期对齐。检验基准调整收益时，应先减去基准并扣除交易成本。缺失值、非有限值和恒定列会报错，程序不会自动删行。输出均值与区间沿用输入的单期单位，不自动年化。

统一返回 `TestResult`：

| 属性 | 含义 |
| --- | --- |
| `names`、`mean` | 候选名与原样本均值 |
| `decisions`、`global_reject` | 列级拒绝决定及是否至少拒绝一列 |
| `adjusted_pvalue`、`global_pvalue` | bootstrap 调整 p 值；Gaussian AR 方法为 `None` |
| `parameter_intervals` | Gaussian AR 方法的时间参数集合外包；bootstrap 为 `None` |
| `diagnostics`、`details` | 方法诊断及原始底层结果 |

`result.to_dict()` 生成 JSON 可序列化记录；`result.to_frame()` 返回候选结果表，可通过 `python -m pip install 'pandas>=2'` 安装所需的可选依赖。完整参数与诊断见 [API 说明](docs/api.md)。

## 方法选择

| 方法 | 用途 | 适用条件和边界 |
| --- | --- | --- |
| `test_returns(..., method="bootstrap")` | 事先给定的有限候选集，联合检验均值是否全部不大于零 | 平稳、弱时间依赖、适当矩条件和一致尺度估计；渐近讨论固定 `K`，短样本或高持久性可能失准 |
| `test_returns(..., method="gaussian_ar")` | 共同未知时间参数下的保守同时决定 | 共同平稳 Gaussian AR(1)、`0 <= phi < 1`、未知边际尺度与同期协方差；模型内强 FWER 控制，可能明显保守 |
| `infer_mean(..., method="hac")` 或 `"iid"` | 事先固定候选的逐列比较 | 不调整筛选；HAC 依赖渐近条件，IID t 检验仅在独立 Gaussian 观测下有限样本精确 |

Bootstrap 默认 `studentization="resampled"`、`n_resamples=999`，列间共享 stationary-bootstrap 时间索引。可指定 `lags`、`block_length`、`n_resamples`、`seed` 与 `batch_size`；批次参数控制计算批量。默认带宽和块长是样本量启发式，重新学生化不自动消除有限样本误差。该接口不等同于 Hansen SPA。

`"gaussian_ar"` 调用 `wilks_uncertainty_test`，使用 GLS 推断，并在整个参数集合上认证决定。要求 `0 < beta < alpha < 0.5`；用于参数信息的前 `min(K, max_dimension)` 列及时间块尺度须事先确定。认证预算不足、单位根端点保留或信息退化时，方法保守地不拒绝。理想 Gaussian 分布保证与给定浮点输入的代数认证范围见[联合方法证明](docs/joint-uncertainty.md)；测量舍入误差不在该分布定理内。

原接口 `audit_returns`、`uncertainty_test`、`wilks_uncertainty_test`、`infer_mean` 和 `long_run_variance` 继续可用。`audit_returns` 为兼容历史版本仍默认 `studentization="fixed"`；统一入口的 bootstrap 默认值为 `"resampled"`。`uncertainty_test` 用事先指定的参考列构造参数集合，详见[参数不确定性方法](docs/parameter-uncertainty.md)。

调整只覆盖输入候选集。软件不能从收益矩阵确认隐藏试验、策略自适应生成、反复查看后的停止规则或搜索完整性。全族拒绝也不保证所选策略未来可盈利。

## CSV 与命令行

```python
from strategy_inference import read_returns_csv, test_returns

table = read_returns_csv("examples/demo_returns.csv")
result = test_returns(table, method="bootstrap", n_resamples=1999, seed=17)
print(result.to_dict())
```

CSV 可含一个 `date` 列，其余列为候选收益。`read_returns_csv(..., benchmark="benchmark")` 会减去指定基准列。重复表头、无效日期顺序及缺失数据会报错。示例 CSV 是模拟数据。

```bash
strategy-inference test examples/demo_returns.csv \
  --method bootstrap --n-resamples 1999 --seed 17 --output result.json

strategy-inference test examples/demo_returns.csv \
  --method gaussian_ar --output gaussian-ar.json
```

旧 `audit` 命令保留 JSON 与 HTML 报告功能。

## 研究与开发

[研究索引](docs/research.md)集中保留各阶段评价、失败边界、功效代价、冻结协议、原始数据和复现命令。已知参数补尾与拟合参数重放属于研究模块，未接入统一检验入口。推导见[方法说明](docs/methods.md)，来源见[文献表](docs/references.bib)。

```bash
python -m pytest
ruff check .
python scripts/sync_protocols.py --check
```

BSD-3-Clause 许可证。
