# 公共 API

本文说明 `v0.5.0` 的统一接口。安装与最小例子见[首页](../README.md)，公式及统计条件见[方法说明](methods.md)，研究证据见[研究索引](research.md)。

## test_returns

```python
from strategy_inference import test_returns
```

```text
test_returns(
    returns,
    *,
    method="bootstrap",
    alpha=0.05,
    names=None,
    **options,
) -> TestResult
```

列级零假设是 \(H_{0,j}:\mu_j\le0\)，全族零假设是全部候选均值均不大于零。检验对象是输入的单期均值，不是 Sharpe 比率。方法须由使用者选择，不根据检验结果自动选择。

### 输入和标签

`returns` 支持：

- 形状为 `(T, K)` 的 NumPy 数组或实数 array-like；一维数据作为 `(T, 1)`。
- 数值 pandas DataFrame，包括数值 nullable dtype。缺失值进入严格校验，不自动填补。
- `read_returns_csv` 返回的 `ReturnTable`。

要求 `T >= 8`、`K >= 1`、每列样本方差为正且所有值有限。复数、布尔、字符串收益及非数值 dtype 不作为收益输入。软件不会删去缺失行，也不会根据 DataFrame 索引重排、对齐或重新采样；行顺序就是时间顺序。观测频率和同期对齐须由使用者保证。

输入已是待检验的收益或收益差分。`test_returns` 本身不自动扣基准或交易成本。数值转换采用 `float64`，结果保持列顺序及单期单位，不自动年化。

`names` 为可选的、长度 `K` 的唯一非空字符串序列。默认标签规则为：

| 输入 | 默认标签 |
| --- | --- |
| NumPy / 普通 array-like / 一维序列 | `strategy_1`、`strategy_2`、… |
| DataFrame | 每个列标签转换成字符串 |
| `ReturnTable` | CSV 中的候选列名 |

显式 `names` 覆盖自动标签。字符串化后重复的 DataFrame 标签会报错，例如同时包含整数 `1` 与字符串 `"1"`。一维序列的标签也以 `strategy_1` 为默认值；需要其他名称时传 `names=["my_strategy"]`。

### 共同参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `method` | `"bootstrap"` | 可选 `"bootstrap"` 或 `"gaussian_ar"` |
| `alpha` | `0.05` | 列级调整决定和全族拒绝的指定错误率水平；bootstrap 要求 `0 < alpha < 1`，Gaussian AR 另有下述预算约束 |
| `names` | `None` | 输入列标签，规则见上表 |
| `**options` | 方法默认值 | 仅接受所选方法的选项，不能混用 |

未知选项和不适用于该方法的选项会抛出 `TypeError`，不会默默忽略。例如 `method="gaussian_ar", seed=17` 或 `method="bootstrap", beta=0.005` 均不接受。错误方法名抛出 `ValueError`。

## method="bootstrap"

使用共享行索引的 stationary bootstrap，按原样本 HAC 均值统计量做单步 max 调整。所有候选先去均值，以全零均值构造边界零假设。这个近似需要平稳、弱时间依赖、适当矩条件及一致尺度估计；渐近讨论固定候选数。短样本、高持久性、非平稳或尾部条件不满足时可能失准。完整候选集和调参规则应事先确定；该方法不能补回遗漏试验或反复监测后的停止规则。

| 选项 | 默认值 | 含义与范围 |
| --- | --- | --- |
| `n_resamples` | `999` | 正整数重抽样次数 \(B\) |
| `studentization` | `"resampled"` | `"resampled"` 每轮围绕该轮均值重估 HAC；`"fixed"` 使用原样本分母 |
| `lags` | `None` | Bartlett 滞后数；默认 `default_lags(T)`，显式值须为整数 `0 <= lags <= T-2` |
| `block_length` | `None` | 几何块长的期望，默认 `default_block_length(T)`；须在 `[1, T]` 内 |
| `seed` | `0` | 整数种子、`numpy.random.Generator` 或 `SeedSequence` |
| `batch_size` | `128` | 正整数计算批次上限；内部可按学生化和内存预算降低批次 |
| `search_complete` | `None` | `True`、`False` 或 `None`，记录是否确认输入覆盖完整搜索族 |

默认规则是

\[
\mathrm{lags}
=\min\left\{\left\lfloor4(T/100)^{2/9}\right\rfloor,T-2\right\},
\qquad
\mathrm{block\_length}
=\min\{\lceil2T^{1/3}\rceil,T\}.
\]

这些是样本量启发式，未估计最优带宽或最优块长。Bartlett 带宽 \(\ell=\mathrm{lags}+1\)，样本自协方差分母为 \(T\)。改变方法参数应事先规定，不能根据哪个参数给出最小 p 值再报一次未调整结果。

Bootstrap 调整 p 值采用

\[
p_j=\frac{1+\#\{b:\max_k T^*_{b,k}\ge T_j\}}{B+1}.
\]

`global_pvalue` 对应原样本最大 HAC 统计量；`decisions[j]` 等于 `adjusted_pvalue[j] <= alpha`。数值分辨率为 `1 / (B + 1)`。增加 \(B\) 可降低条件模拟误差，不自动消除统计近似误差；加一公式也不赋予一般 bootstrap 有限样本精确保证。

`search_complete` 只记录使用者提供的信息，不改变计算。`None` 表示未确认，而非自动判定完整。`seed` 为 Generator 时消耗其当前随机状态；整数种子用于可复现的独立调用。诊断中的 `seed` 只保存整数种子，Generator 或 SeedSequence 对应 `None`；`random_state_type` 保存传入类型名。导出不包含生成器或 SeedSequence 的状态，使用这类输入时须自行保存状态和调用顺序以复现。

内部保留 \(B\times K\) 统计量。`batch_size` 控制中间工作批次，不限制完整输入或返回矩阵大小。低层重抽样遇到退化、非正或非有限 HAC 尺度时会报错，不通过删去坏抽样来继续计算。

## method="gaussian_ar"

此方法调用 `wilks_uncertainty_test`。模型为共同时间系数的平稳联合 Gaussian AR(1)：

\[
X_t=\mu+\phi(X_{t-1}-\mu)+\varepsilon_t,\qquad
0\le\phi<1,
\]

\[
\varepsilon_t\overset{\rm iid}{\sim}
N\{0,(1-\phi^2)\Sigma\},\qquad
\operatorname{Cov}(X_s,X_t)=\phi^{|s-t|}\Sigma.
\]

要求平稳初始化、固定候选集和正边际方差 \(\Sigma_{jj}>0\)。同期协方差可未知；用于参数信息的联合方向发生奇异时保守回退。异质时间系数、一般 GARCH 或非 Gaussian 创新不属于此模型内保证。

方法用预先指定的前若干列与时间块构造共同 \(\phi\) 的置信集合，再认证集合上所有参数的 GLS 单侧决定。共享覆盖预算与 Bonferroni 给出模型内强 FWER 控制，即有些候选确实有信号时也控制误拒真零假设。它不采用拟合单点参数的标准误，也不提供连续 p 值。

| 选项 | 默认值 | 含义与范围 |
| --- | --- | --- |
| `beta` | `0.005` | 全族共用的参数覆盖预算；要求 `0 < beta < alpha < 0.5` |
| `max_dimension` | `8` | 参数信息用前 `min(K, max_dimension)` 列；整数 `1..8` |
| `block_lengths` | `(4, 16, 64)` | 事先给定的时间块长度；非空、互异的整数序列，每项至少 `2` |
| `ci_depth` | `16` | 参数集合外包的最大细分深度；整数 `0..64` |
| `certificate_depth` | `20` | 每个合并参数区间的检验认证细分深度；整数 `0..64` |
| `max_nodes` | `4096` | 每列认证节点预算；正整数 |
| `critical_bits` | `40` | 保守临界值的二进制界精度；整数 `8..128` |

列顺序、所用维度及时间块尺度须在看数据前确定。计算深度和节点预算控制证书搜索；未解析的区域保守地不拒绝，不当作证书已经证明零假设。参数集合为空、保留单位根端点或参数信息不可用时也可能全部不拒绝。更大的预算不保证某列一定获得拒绝。

统计保证针对理想 Gaussian 模型。整数/有理数证书针对实际提供的二进制浮点数据；输入测量及舍入误差不在该分布定理内。详细区别见[共同参数证明](joint-uncertainty.md)。

## TestResult

返回的主数组均按输入列顺序排列。`mean`、`decisions` 和存在时的 `adjusted_pvalue` 是只读副本；`diagnostics` 是只读映射。`details` 保留底层对象，不应把主结果的只读约定延伸为底层每个数组都不可修改。

| 字段或属性 | 类型与语义 |
| --- | --- |
| `method` | `"bootstrap"` 或 `"gaussian_ar"` |
| `sample_size`、`n_obs` | 观测数 \(T\)，`n_obs` 为别名 |
| `names` | 长度 \(K\) 的候选名称 tuple |
| `n_strategies` | 候选数 \(K\) |
| `alpha` | 本次决定的指定水平 |
| `mean` | 长度 \(K\) 的普通样本均值；Gaussian AR 的决定采用 GLS，此字段不替代其估计器 |
| `decisions` | 长度 \(K\) 的布尔数组；`True` 表示该候选拒绝均值不大于零 |
| `global_reject` | `decisions.any()`；至少一列拒绝 |
| `adjusted_pvalue` | Bootstrap 长度 \(K\) 的 max 调整 p 值；Gaussian AR 为 `None` |
| `global_pvalue` | Bootstrap 全族 p 值；Gaussian AR 为 `None` |
| `parameter_intervals` | Gaussian AR 参数集合外包的浮点端点 tuple；bootstrap 为 `None` |
| `diagnostics` | 下述方法诊断映射 |
| `details` | `AuditResult` 或 `WilksResult`，保留完整底层结果 |

`parameter_intervals` 是展示与导出用的浮点端点，不是参数点估计。需要精确有理数端点时读取 `details.intervals`，不要把展示时的浮点转换重新用于连续域证书。`decisions=False` 只表示本次未拒绝，不证明均值为零或没有信号。

Bootstrap 诊断包括 `studentization`、`n_resamples`、`block_length`、`lags`、`search_complete`、`pvalue_resolution`、`seed`、`random_state_type`、`warnings` 与 `conditional_bootstrap_tail_interval_95`。后两项保留短样本、有限重抽样尾部及搜索完整性提示；在只读诊断映射中为 tuple，`to_dict()` 与 JSON 中为列表。条件尾部区间是 95% Wilson 区间，描述给定观测数据后的模拟误差，不是总体均值的置信区间，也不能校验统计近似有效性。底层对应字段为 `details.warnings` 和 `details.bootstrap_tail_interval`。

Gaussian AR 诊断包括：

| 键 | 含义 |
| --- | --- |
| `shape_dimension` | 实际采用的预先指定列数 |
| `beta` | 共享覆盖预算 |
| `block_lengths` | 本次使用的时间块长度；映射中为 tuple，JSON 中为列表 |
| `phi1_retained` | 参数集合保留单位根端点 |
| `empty_parameter_set` | 参数集合外包为空，方法保守不拒绝 |
| `singular_fallback` | 至少一个时间尺度发生奇异回退 |
| `ci_unresolved_cells` | 外包搜索未解析的单元数 |
| `certificate_unresolved` | 预算/深度下未解决认证的候选数量 |

`certificate_unresolved=0` 不表示全部拒绝；证据不足也可产生已经完成计算的非拒绝。查看逐列认证标志、节点数、各尺度可用性及有理数集合时使用 `details`。

### 导出

```python
import json

record = result.to_dict()
json_text = json.dumps(record, ensure_ascii=False, allow_nan=False)
rows = result.records()       # 无需 pandas。
frame = result.to_frame()     # 需要可选 pandas。
```

`to_dict()` 返回紧凑 JSON 记录，含 `schema_version`、方法、零假设、样本和候选数、水平、全族结果、参数集合、诊断及 `candidates`。它不含完整 bootstrap 抽样或证书多项式。改变返回字典不会改变主结果。

`records()` 返回候选行：`name`、`mean`、`reject`，bootstrap 另含 `adjusted_pvalue`。Gaussian AR 行不伪造 p 值，也不放入全族参数集合。

`to_frame()` 使用上述候选行，索引名为 `strategy`。它同样不包含方法、`alpha`、全族结果或参数集合；需要完整上下文时同时保存 `to_dict()`。安装可选依赖：

```bash
python -m pip install 'pandas>=2'
```

Bootstrap 的 `details` 还包括 IID/HAC 比较、最大统计量胜出者、近似同时区间及 \(B\times K\) 抽样矩阵；Gaussian AR 的 `details` 包括精确有理数区间、每列认证状态和 Wilks 尺度记录。原始底层对象可用于进一步诊断，不是另一个已经调整过的方法选择规则。

## CSV 输入与 CLI

```python
from strategy_inference import read_returns_csv, test_returns

table = read_returns_csv("returns.csv", benchmark="benchmark")
result = test_returns(table, method="bootstrap", n_resamples=1999, seed=17)
```

`read_returns_csv(path, *, benchmark=None)` 返回 `ReturnTable(values, names, dates)`。CSV 表头须非空且互异，可含一个名为 `date` 的严格递增 ISO 日期/时间列；其他列必须是数值。`benchmark` 指定的列会从全部候选收益中减去，并从候选集移除。日期检查不推断交易频率，也不保证等间隔；`test_returns` 不根据 `dates` 改动观测。

```bash
strategy-inference test returns.csv \
  --method bootstrap --n-resamples 1999 --seed 17 --output result.json

strategy-inference test returns.csv \
  --method gaussian_ar --beta 0.005 --output result.json

strategy-inference test returns.csv --format csv --output candidates.csv
```

`--output -` 或省略输出路径时写到标准输出。`--format json` 保存方法、水平、全族结果、参数集合和诊断，并附输入文件名、哈希、基准列、版本及有效整数种子；bootstrap 未指定种子时记录 `0`，Gaussian AR 记录 `None`。`--format csv` 仅保存候选行，无需 pandas；需要可追溯的完整结果时另存 JSON。没有指定格式时，输出文件名的 `.csv` 后缀选择 CSV，其余为 JSON；`--format` 可显式覆盖。

CLI 只接受对应方法的选项。Bootstrap 提供 `--n-resamples`、`--seed`、`--batch-size`、`--block-length`、`--lags`、`--studentization`；Gaussian AR 提供 `--beta`、`--max-dimension`、`--block-lengths`。Python 的认证深度/节点选项和 `search_complete` 尚不通过 `test` CLI 暴露，需要时使用 Python API。输入校验或选项错误会返回非零退出状态。输出指向输入本身或其软/硬链接时会在计算前报错；其他既有输出文件会被替换。`--help` 与 `--version` 无需加载 NumPy/SciPy。

旧 `audit` 命令继续产生 JSON 与 HTML 报告，并有搜索完整性标志。其默认 `studentization="fixed"` 与新统一入口、新 `test` 命令的 `"resampled"` 不同；重跑旧调用时应显式指定原方法。

## 现有进阶 API

| 接口 | 用途与文档 |
| --- | --- |
| `audit_returns`、`AuditResult` | 共享时间索引的 max bootstrap 与同时区间；[方法说明](methods.md) |
| `infer_mean`、`MeanInference` | 逐列 IID 或 HAC 推断，不调整筛选；[方法说明](methods.md) |
| `long_run_variance`、`default_lags` | Bartlett HAC 尺度与默认带宽；[方法说明](methods.md) |
| `stationary_indices`、`stationary_bootstrap_means`、`stationary_bootstrap_statistics` | 低层重抽样；[bootstrap 实现](../src/strategy_inference/bootstrap.py) |
| `stationary_mean_variance` | 指定重抽样索引律下均值的精确条件方差，不是未知总体方差；[方法说明](methods.md) |
| `uncertainty_test`、`UncertaintyResult` | 事先指定参考列的参数集合方法；[证明与条件](parameter-uncertainty.md) |
| `wilks_uncertainty_test`、`WilksResult`、`WilksScale` | 联合多尺度方法；[证明与条件](joint-uncertainty.md) |

已知参数参照和拟合参数重放继续作为[研究模块](research.md)，不在 `test_returns` 中自动选择。输入集合以外的策略生成、隐藏搜索和反复监测需要另行处理。
