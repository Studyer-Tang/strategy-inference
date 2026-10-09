# 预测比较与不确定性：范围与路线图

`strategy-inference` 专注时序预测比较与不确定性评估。真实数据读取、线性基线和验证集选择服务于这一工作流。下面区分已经实现的公共接口与后续方向；列出论文不意味着其全部算法或理论已进入本库。

安装与例子见[首页](../README.md)，详细接口见[时序 API](time-series.md)、[多步 API](multistep-api.md)，原收益推断见[均值 API](api.md)。本项目的增量主要是接口整合、实现、验证和计算效率；已知预测、MCS、bootstrap 与 conformal 基础算法不作为算法创新主张。

## 已实现的评估工作流

| 能力 | 公共接口与当前边界 |
| --- | --- |
| 真实数据（v0.10） | `load_dataset`、`read_tsf`；固定版本 Monash 宏观、Bitcoin 与小时气候档案，SHA-256/离线缓存，原位保留缺失；不是实时金融 vintage 数据。真实例子按时间切分，验证调参、测试冻结 |
| 滚动与扩展窗回测 | `rolling_splits`、`backtest`；单变量数值序列、完整 horizon、可设 step/gap/window。保留 origin、目标位置和重叠预测；回调只得到当前训练数据与 lead 的独立只读副本 |
| 透明预测 baseline | `naive_forecast`、`SeasonalNaive`、`drift_forecast`、`Autoregression(lags, ridge)`；密集／稀疏滞后，只拟合当前历史。`Differenced` 可组合常规／季节差分与任意 callback，并还原物理 lead |
| 验证集选择（v0.11） | `select_forecaster(train, validation, models, ...)`；候选共享验证目标、窗口和损失，返回选中的 callable 与完整分数，确定性处理 ties；测试段另行评价 |
| 逐预测损失 | `forecast_loss`、`evaluate_forecasts`；平方、绝对、pinball，保留 origin × lead × model 数组并按 lead 汇总。`interval_score` 单独评价中心区间，不能凭评分推断覆盖保证 |
| 固定候选预测比较 | `compare_forecasts`；预先指定 baseline 与单个 lead，共享时间索引的 max bootstrap。正改善为 baseline 损失减去候选损失；依赖固定族、平稳弱依赖、适当矩和非退化方差等条件 |
| 连续模型置信集（v0.9） | `SequentialModelConfidenceSet`、`sequential_compare_forecasts`；固定模型族、绝对/pinball 损失、事前注册预测与非重叠反馈。目标为每时每 pair 的强条件优劣；集合可空，排除不恢复。见[API](time-series.md#连续模型置信集) |
| 单步在线区间 | `AdaptiveConformal`、`adaptive_intervals`；递减步长 quantile tracker，固定尺度有界残差映射，单个 pending prediction 与有序完整反馈。保留空集/全域，不截断阈值；理想递推为长期平均覆盖，普通浮点实现不提供舍入证书 |
| 收益均值推断 | `test_returns` 与既有高级接口保留；bootstrap 近似和共同平稳 Gaussian AR(1) 模型内的保守决定各自沿用原条件 |
| 结果保存 | 回测数组、逐预测损失、候选比较与在线区间结果可读取；JSON/记录表接口，部分 `to_frame()` 需可选 pandas。类型和导出范围见 API |

回测不自动保证损失平稳，strong 连续比较也不等同于累计平均风险排名。比较不会把 lead 展平成独立样本；原单步 tracker 仍要求即时有序反馈。

## v0.7 增加的能力

| 能力 | 公共接口与边界 |
| --- | --- |
| 多步成熟反馈 | `MultiStepConformal`、`multistep_intervals`；单变量连续整数观测时钟，显式 origin/target，observe 后 predict；只用已经成熟的标签，未来尾部保留 pending |
| Pooled 与 interlaced | 每步长当前状态更新，或按 `origin % lead` 分 lane；逐步长计数和平均覆盖界。两者学习时钟不同，不把差异直接当作效率结论 |
| 按步长学习率 | `step_size` 支持正标量或 H 向量；`0.1/sqrt(leads)` 是延迟 damping 例子，不声称最优 |
| 可选成熟尺度 | 固定尺度、自身步长 EWMA RMS，或最短配置步长成熟残差加固定初始尺度比率；发行尺度冻结，阈值不跨步长混用 |
| 对齐与导出 | 只读发行数组、evaluated 掩码、成熟学习率、pending 与 lane 状态；严格 JSON 和可选 pandas。summary 分开报告空集、全域及有限宽度范围 |

v0.8 的 `scale_source="blended"` 用预先指定的标量或逐步长共享权重融合两个成熟 RMS 来源；默认各占一半。它保留原阈值控制器和冻结发行尺度，只增加 O(H) 当前来源状态。接口允许在共享共同波动与保留步长特有误差之间选择折中，不把固定融合表述为新算法或普遍最优方案。

理想递推针对每个步长已成熟预测的历史平均误覆盖，不承诺逐时条件覆盖或整条路径同时覆盖；普通 binary64 结果不提供舍入证书。实现是独立工程 prototype，未包含完整 AcMCP 的 PID 饱和函数、误差预测器或 O²CP 的联合优化。证明见[多步方法](multistep-methods.md)。

研究问题是：不同预测起点新旧与累计误差相关性，能否让短步长成熟残差成为更有效的长步长尺度信息，并改善宽度、区间评分或局部恢复？每时发行、队列填满后，各步长都在同一日历时点收到当前标签；shortest 不提前取得长步长标签，也不消除其阈值反馈延迟。共享尺度与跨步长信息利用已有先例，不声称首次提出。正式对照固定预测器、训练尺度、学习时钟与目标，比较 fixed、horizon 和 shortest，并包含共同波动、突变、重尾及步长特有失配的反例；有限比较不建立一般优势。

## 近期工作与验收目标

先加强现有工作流的可核验性，再考虑新接口。以下顺序表示优先级，不承诺发布日期。

| 方向 | 计划范围 | 进入公共接口前的验证 |
| --- | --- | --- |
| 现有实现的独立核验 | 同训练窗对照 statsmodels 的密集／稀疏 OLS；在预先固定的五个未用于开发例子的 FRED-MD 序列上验证选模与回测 | 协议先提交；验证段选参数，测试段冻结；逐点预测核对后记录同机耗时。同档案扩展不视为外部数据集验证 |
| 同一目标的检验校准 | Gaussian AR(1) 的固定样本均值损失比较，对照已知参数的精确 oracle；报告错误率、功效及配对差异 | 两种方法检验同一原假设；保存逐次决定、Wilson 区间与 MC 标准误；oracle 不作为可部署方法 |
| 比较结果的稳定性 | 先研究块长与长期方差选择的敏感性，再评估多 lead 联合比较与预测器适配 | 预先指定风险目标；保留完整时间／跨 lead 协方差；用独立实现、相关 DGP 和新数据核验，不因结果挑选配置 |

当前验证协议和运行程序见 [`forecast_validation_protocol.json`](../benchmarks/forecast_validation_protocol.json) 与 [`forecast_validation.py`](../benchmarks/forecast_validation.py)。条件 conformal、完整 AcMCP/O²CP、经典 Fast MCS、漂移检测与通用 panel 模块暂不列入近期公共接口承诺。

## 近期主源与可继承的范围

- **2024，Angelopoulos–Barber–Bates：[Online conformal prediction with decaying step sizes](https://proceedings.mlr.press/v235/angelopoulos24a.html)。** 当前 tracker 的来源。Theorem 1 对有界 score、正递减步长和初始阈值给出任意序列的回顾平均覆盖界。IID 量化收敛还需要固定/稳定 score、分位数与连续性等额外条件，不能解释成任意依赖序列的逐时条件覆盖。当前固定尺度有界映射是实现选择。
- **2026，Arnold 等：[Sequential model confidence sets，JRSSB](https://doi.org/10.1093/jrsssb/qkag066)，[HTML v4](https://arxiv.org/html/2404.18678v4)。** v0.9 实现 Proposition 3.2、Eq.6–7 与补充 H 的 strong 构造，使用预测事前确定的绝对/pinball 损失差界和闭合 e-testing。原论文的 uniformly weak/weak 平均风险构造未实现；其时间变化归一化可能改变风险目标。普通固定样本 MCS 反复运行不能继承时间一致保证。
- **2025，Areces–Mohri–Hashimoto–Duchi：[Online Conformal Prediction via Online Optimization](https://proceedings.mlr.press/v267/areces25a.html)。** 分别提供 adversarial 平均保证和 stochastic 条件结论；后者使用条件误差分位数由过去数据线性表达等结构假设。它不支持任意非平稳序列逐时条件覆盖，也不是当前 `AdaptiveConformal` 的已实现能力。
- **2025 首稿，Bauer–Kazak：[Conditional Method Confidence Set](https://arxiv.org/abs/2505.21278)。** 按预测时点已知的离散状态比较方法。§2 使用有限训练窗、混合/矩条件、状态内稳定排名和一致方差估计；不能把事后挑选 regime 或 expanding-window 搜索直接当作已获保证。
- **2026（online 2025），Barde：[Large-scale model comparison with fast model confidence sets](https://doi.org/10.1016/j.jeconom.2025.106123)。** R-rule 两遍更新算法减少论文所比较实现的模型维度时间与工作内存阶数。**Proposition 1 的 ranking/output 等价结论是样本量增大时的渐近结果**；有限样本实验一致不等于任意输入严格相等。未来实现须保留经典消除法参考、共享抽样索引及差异记录/回退；一遍版本不能冒充两遍算法。添加模型的计算能力也不自动处理自适应搜索。[作者说明](https://sylvain-barde.github.io/projects/fast_mcs/)明确此渐近限制。
- **2026，Grant–Mrazik–Satchell：[Evaluating Forecasts at Multiple Horizons](https://doi.org/10.1002/for.70150)。** §3 对多步损失差向量及其长期协方差做联合比较；需要联合 CLT 与一致协方差估计。向量差异检验与预声明加权平均改善是不同统计目标，均不是当前单 lead 比较自动提供的结果。
- **2026，Pohle–Zahn–Lerch：[Uncertainty Quantification in Forecast Comparisons](https://arxiv.org/abs/2605.03997)。** 为 expected scores/skill scores 构造联合带；§3、Proposition 1 使用多元 CLT、一致长期协方差与有效 bootstrap。比值型 skill score 还要求基准期望 score 为正。其有限样本研究也报告强依赖和高维时的欠覆盖，不能由此宣称普遍精确或任意增长维数有效；已有[作者实现](https://github.com/TanjaZahn/UQforecasts)可作对照。
- **2026 修订（首稿 2024），Wang–Hyndman：[Online conformal inference for multi-step time series forecasting](https://arxiv.org/html/2410.13115v2)。** AcMCP/多步 PID 使用已经成熟的多步误差。完整长期保证涉及指定饱和函数、可容许次线性函数及 bounded scorecaster 等条件；各 horizon 的长期覆盖不能改称整条预测轨迹同时覆盖。一般非线性 AR 的误差结构推导使用 Taylor 近似，不能泛称精确 MA(h−1)。v0.7 的简单成熟反馈 tracker 未实现完整 AcMCP。
- **2026 年 9 月，El Halabi–Brandt：[Adaptive Conformal Inference Under Delayed Feedback](https://arxiv.org/html/2609.07251v1)。** 延迟 ACI 可分解成相位交织的控制器，给出延迟相关平均覆盖界；delay-to-memory 是机制相关诊断，尺度归一化也已有研究。其额外 marginal 结论使用结构化依赖假设，不能由本模块直接继承，更不能以 interlaced 的定理替代 pooled 推导。
- **2026 年 8 月修订（首稿 2025）：[Optimization-Based Online Conformal Prediction for Multi-Step Forecasting](https://arxiv.org/html/2508.13362v3)。** O²CP 在可容许控制集合内利用跨步长误差分布进行优化，保留基础算法的长期 marginal 目标。跨步长利用不是新的概念；本模块的固定比率成熟尺度共享没有实现该联合优化，也不借用其定理保证任意阈值改动。

## 与成熟库衔接

[sktime](https://www.sktime.org/en/latest/examples/01_forecasting.html) 提供统一预测、时间交叉验证和概率预测接口；[StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html) 提供高效模型、多序列与滚动评估。它们适合作为预测器来源，本库着重保留评估位置、损失和推断结果。

[arch MCS](https://arch.readthedocs.io/en/latest/multiple-comparison/generated/arch.bootstrap.MCS.html) 已支持 R/max 消除法与多种 block bootstrap，并包含 SPA/StepM 等方法；[dieboldmariano](https://github.com/edoannunziata/dieboldmariano) 提供轻量配对 DM。本库新增接口须解释自己的统计目标和对照差异，不把这些已有功能表述为首次提出。

## 版本证据

[v0.5 归档页面及 benchmark](https://studyer-tang.github.io/strategy-inference/library/v0.5.0/)绑定固定发布 commit，原数据保留；[性能说明](performance.md)描述的是该版六格实验。新模块的速度须用新版本 benchmark 验证，不沿用旧版倍数。

[v0.6 归档页面](https://studyer-tang.github.io/strategy-inference/library/v0.6.0/)与[性能说明](time-series-performance.md)同样绑定当时发布源码。它们不是 v0.7 多步接口的性能报告；新测量另行保存。归档冻结的是统计/性能证据与来源，页面相对链接可随归档位置调整。

历史正式研究须在 metadata 指定的冻结 commit/对应 tag 下完整复核；新 API 和计算优化不改写旧协议与证据。报告和图可直接读取，带 source-hash 锁的旧 auditor 不应直接配合更新后的 core 重跑。完整索引见[研究与复现](research.md)。
