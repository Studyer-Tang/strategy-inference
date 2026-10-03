# 时间依赖与策略筛选的显著性检验

本文说明 `strategy-inference` 的统计对象、算法与实验设计。问题是：在同一段历史上比较多个候选策略之后，观察到的正收益是否仍有足够的统计证据？第 1–10 节保留 v0.1 固定尺度基线，第 11 节说明 v0.2 增加的重新学生化方法、方差诊断和独立评价协议。

第一版采用均值检验、Bartlett HAC 标准误和 stationary bootstrap。实现选择固定原样本尺度的最大统计量检验，便于核对每一步计算。它没有实现 Hansen 的 SPA，也没有提出新的统计定理。本文的公式描述实际算法；渐近论证说明算法适用的条件，有限样本表现由实验报告。

Hansen 的 [SPA 原文](https://doi.org/10.1198/073500105000000063)同时涉及 studentized 统计量与样本依赖的零假设分布修正。仅给最大均值加一个 HAC 分母，不能把实现称作该方法。

## 1. 输入、目标和候选集

输入为有限实数矩阵

$$
D=(d_{tj})_{T\times K},\qquad t=1,\ldots,T,\quad j=1,\ldots,K.
$$

每行是一段等间隔评价期，每列是一个候选策略。程序接受 $T\geq8$、$K\geq1$，一维数组按单列处理；八个观测只是计算边界，不是统计精度的保证。等间隔、同期和成本定义需要由研究者确认，数组校验无法推断这些事实。主要使用方式为

$$
d_{tj}=r^{\mathrm{net}}_{tj}-r^{\mathrm{net}}_{t0},
$$

其中 $r^{\mathrm{net}}_{tj}$ 是策略在第 $t$ 期扣除约定成本后的收益，$r^{\mathrm{net}}_{t0}$ 是同一期基准收益。基准已扣成本的定义也应保存在试验记录中。收益频率、计价单位、成本规则和评价日期需要统一；不能把缺失收益静默填成零。

目标参数是每期平均差分

$$
\mu_j=\mathbb E[d_{tj}].
$$

采用单侧族假设

$$
H_0:\mu_j\leq0\ \text{对所有 }j,
\qquad
H_1:\text{至少一个 }\mu_j>0.
$$

该参数与 Sharpe ratio、因子回归 alpha、最大回撤和经济效用不同。均值显著为正也不意味着交易规模、成本建模或样本外表现足够好。一般表现差分也能使用相同计算，但需要另行说明符号方向和参数含义。

**候选集是推断的一部分。** 第一版以固定有限 $K$ 为理论范围。矩阵应包含这一次固定搜索族中所有被比较的候选项，包括效果差的项。只提交胜出策略，计算得到的是单列检验，无法还原已经发生的完整搜索。不断读取结果、修改策略并增加新列的自适应开发过程，需要对开发和评价另作设计；把最终所有曲线放进矩阵本身不能保证解决该问题。

## 2. 时间依赖为什么改变标准误

记 $x_{tj}=d_{tj}-\mu_j$，第 $j$ 列的自协方差为

$$
\gamma_j(h)=\operatorname{Cov}(x_{tj},x_{t-h,j}).
$$

在协方差平稳且自协方差绝对可和时，长程方差为

$$
\Omega_{jj}=\gamma_j(0)+2\sum_{h=1}^{\infty}\gamma_j(h).
$$

样本均值的有限样本方差有直接的分解：

$$
\operatorname{Var}(\bar d_j)
=\frac1T\left[\gamma_j(0)+2\sum_{h=1}^{T-1}
\left(1-\frac hT\right)\gamma_j(h)\right].
$$

因此，$T$ 大时应使用 $\Omega_{jj}/T$，而不只是 $\gamma_j(0)/T$。正自相关常使独立样本标准误偏小；负自相关可以产生相反方向的影响。条件异方差与均值自相关也需要区分：GARCH 收益可能没有线性自相关，但平方收益仍有依赖。

### 2.1 AR(1) 的可核对例子

令

$$
x_t=\phi x_{t-1}+\sqrt{1-\phi^2}\,\varepsilon_t,
\qquad |\phi|<1,
\qquad \mathbb E\varepsilon_t=0,
\quad\operatorname{Var}(\varepsilon_t)=\sigma^2.
$$

平稳时 $\operatorname{Var}(x_t)=\sigma^2$，且 $\gamma(h)=\sigma^2\phi^{|h|}$。用几何级数即可得到

$$
\Omega=\sigma^2\left(1+2\sum_{h=1}^{\infty}\phi^h\right)
=\sigma^2\frac{1+\phi}{1-\phi}.
$$

独立标准误相对正确标准误的平方比为

$$
R(\phi)=\frac{\Omega}{\gamma(0)}=\frac{1+\phi}{1-\phi}.
$$

在零均值、大样本正态近似下，忽略依赖并使用 $z_{1-\alpha}$ 作单侧临界值，拒绝概率近似为

$$
1-\Phi\!\left(\frac{z_{1-\alpha}}{\sqrt{R(\phi)}}\right).
$$

当 $\alpha=0.05$、$\phi=0.5$ 时，$R=3$，该近似约为 $0.171$。它解释了图中的方向和量级，不能当作有限 $T$ 的精确拒绝率；有限样本权重、方差估计和 Student 临界值都会产生差异。

## 3. 两个单策略参考检验

### 3.1 独立样本 Student 检验

令

$$
\bar d_j=\frac1T\sum_t d_{tj},\qquad
s_j^2=\frac1{T-1}\sum_t(d_{tj}-\bar d_j)^2.
$$

独立样本统计量和单侧 $p$ 值为

$$
t^{\mathrm{iid}}_j=\frac{\bar d_j}{s_j/\sqrt T},\qquad
p^{\mathrm{iid}}_j=1-F_{t_{T-1}}(t^{\mathrm{iid}}_j).
$$

仅在该列独立同分布且正态时，零均值边界下才有有限样本精确的 $t_{T-1}$ 分布。独立但非正态时通常依赖大样本近似；时间依赖下也不能仅因为使用 Student 分布，就把它视作稳健检验。这里保留它作为常见实践的参考。[SciPy 的单样本检验定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)可用于核对符号、标准误和自由度。

### 3.2 Bartlett HAC 检验

实际计算使用样本中心化残差 $\hat x_{tj}=d_{tj}-\bar d_j$。所有滞后统一采用分母 $T$：

$$
\hat\gamma_j(h)=\frac1T\sum_{t=h+1}^{T}\hat x_{tj}\hat x_{t-h,j},
\qquad h=0,\ldots,L.
$$

Bartlett 长程方差估计为

$$
\hat\Omega_{jj}
=\hat\gamma_j(0)+2\sum_{h=1}^{L}
\left(1-\frac h{L+1}\right)\hat\gamma_j(h).
$$

标准误、统计量和单侧渐近 $p$ 值为

$$
\widehat{\operatorname{se}}_j=\sqrt{\hat\Omega_{jj}/T},\qquad
z^{\mathrm{HAC}}_j=\frac{\bar d_j}{\widehat{\operatorname{se}}_j},\qquad
p^{\mathrm{HAC}}_j=1-\Phi(z^{\mathrm{HAC}}_j).
$$

分母 $T$ 与 Bartlett 权重配合，使这一估计在精确算术中非负。第一版不做 $T/(T-1)$ 小样本修正，也不做预白化。常数列或不能得到有效正标准误的列不能正常使用上述比值，应明确报错；加入一个任意小常数继续输出显著性会改变统计对象。

默认滞后为

$$
L=\min\left\{T-2,
\left\lfloor4(T/100)^{2/9}\right\rfloor\right\}.
$$

这是常用的样本量规则，与 [statsmodels 的 HAC 默认滞后](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)相同。截断到 $T-2$ 是本项目的输入边界约定。它不是对某个样本求得的最优带宽，较强持久性下仍可能低估长程方差。$L=0$ 退化为使用分母 $T$ 的零阶方差估计。

[Newey–West (1987)](https://doi.org/10.2307/1913610)给出了 HAC 估计的经典构造。这里的均值检验还需要相应的中心极限定理：对固定 $K$，$\sqrt T(\bar d-\mu)$ 收敛到具有有限长程协方差的正态向量，并且 HAC 尺度一致。文献结论带有依赖和矩条件，不能仅凭调用公式便认为所有金融序列都满足这些条件。

## 4. 先筛选再作单列检验

第一版统一按 HAC 统计量选择候选：

$$
\hat j=\operatorname*{arg\,max}_{j}z^{\mathrm{HAC}}_j,
\qquad M=\max_j z^{\mathrm{HAC}}_j.
$$

并列时使用第一列的索引，以保持确定性。这个选择规则偏向“均值相对估计不确定性较大”的策略，不等于按最高收益或最高 Sharpe ratio 排序。

实验中的两种未调整方法分别查看 $p^{\mathrm{iid}}_{\hat j}$ 和 $p^{\mathrm{HAC}}_{\hat j}$，二者使用同一个胜出者。HAC 只处理时间依赖；若在筛选后仍把 $p^{\mathrm{HAC}}_{\hat j}$ 当作事前指定单列的 $p$ 值，搜索效应仍在。

一个特殊例子能说明这一点：若 $K$ 个单列零假设检验相互独立，每个检验的拒绝概率恰为 $\alpha$，并且选择规则等价于选择最小 $p$ 值，那么至少一次拒绝的概率为

$$
1-(1-\alpha)^K.
$$

该式不是本项目一般设置的拒绝率公式。现实策略通常相关，实验中的 HAC 胜出者也不一定等于 iid 最小 $p$ 值的候选。横截面相关性、各列尺度和选择规则都会影响最大统计量的分布。

## 5. 共同索引 stationary bootstrap

### 5.1 索引生成

给定期望块长 $\ell\geq1$，设置重启概率 $q=1/\ell$。每次重抽样生成长度 $T$ 的索引序列：

1. $I_1$ 在 $\{1,\ldots,T\}$ 上均匀抽取。
2. 对 $t=2,\ldots,T$，以概率 $q$ 重新均匀抽取 $I_t$；否则令 $I_t=1+(I_{t-1}\bmod T)$。
3. 使用完整行向量 $(d_{I_t,1},\ldots,d_{I_t,K})$，所有列共享这组索引。

由此形成的连续块长服从正整数几何分布，其期望为 $1/q=\ell$；末块按所需总长度截断。数组尾部可以循环到开头，这是重抽样边界约定，不是假设真实历史周期循环。

共享索引同时保留样本中的同日横截面关系和块内时序结构。若逐列独立抽样，就会破坏候选策略间的相关性，继而改变最大统计量的分布。重抽样不会精确保留所有长滞后结构，块长过小也会削弱依赖。[Politis–Romano (1994)](https://doi.org/10.1080/01621459.1994.10476870)是 stationary bootstrap 的原始文献；[arch 的接口说明](https://arch.readthedocs.io/en/stable/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html)将 `block_size` 定义为平均块长。

第一版默认规则为

$$
\ell=\min\left\{T,\left\lceil2T^{1/3}\right\rceil\right\}.
$$

这是一项预先固定的实现选择。它满足 $\ell\to\infty$ 和 $\ell/T\to0$ 的基本尺度关系，不代表这些关系足以单独保证 bootstrap 一致性，也不代表有限样本的最佳块长。真实应用应报告约定范围内的块长和 HAC 滞后敏感性，不能只保留最容易显著的一组。

### 5.2 零假设中心化

重抽样对象为

$$
\tilde d_{tj}=d_{tj}-\bar d_j.
$$

每列样本均值都移到零，模拟族零假设的最不利边界。这样抽样时近似的是均值估计误差，而不是把观察到的正均值当作零假设信号。

“最不利”指以下坐标单调性：在固定误差分布和正尺度下，将任何 $\mu_j\leq0$ 提高到零，只会提高最大统计量。对所有候选都移到零因此可能保守，尤其当候选集中有很多明显较差的策略时。bootstrap 对实际误差分布的近似仍依赖其一致性；有限样本中不能据此直接宣称误报率必然不超过 $\alpha$。[White (2000)](https://doi.org/10.1111/1468-0262.00152)系统讨论了完整搜索族与最不利边界的重抽样检验。

### 5.3 固定原样本尺度的族检验

对第 $b$ 轮重抽样计算

$$
\bar{\tilde d}^{*(b)}_j=\frac1T\sum_{t=1}^{T}\tilde d_{I_t^{(b)},j},
\qquad
Z^{*(b)}_j=
\frac{\bar{\tilde d}^{*(b)}_j}{\widehat{\operatorname{se}}_j},
\qquad
M^{*(b)}=\max_j Z^{*(b)}_j.
$$

分母是在原数据上计算一次的 HAC 标准误。第一版不会在每次重抽样中重新估计标准误；因此准确名称是**固定尺度的最大统计量 bootstrap 检验**。在原样本尺度一致和联合 bootstrap 有效时，可以用 Slutsky 定理解释这种一阶近似；不能由此宣称重新 studentize 的高阶精度。

当 $K=1$ 时，比较 $M^{*(b)}\geq M$ 两侧的同一个正分母相消。这时得到的 $p$ 值等价于用中心化均值 bootstrap 分布检验原均值；它并非重新 studentize 的单样本 bootstrap-$t$。

取 $B$ 次重抽样，报告

$$
\hat p_{\mathrm{family}}=
\frac{1+\sum_{b=1}^{B}\mathbf1\{M^{*(b)}\geq M\}}{B+1}.
$$

`>=` 包含并列值。加一约定使结果不会为零，最小值为 $1/(B+1)$。它不是将本检验变为有限样本精确随机化检验的证明。给定数据，尾部计数本身还有 bootstrap Monte Carlo 误差：若条件尾概率为 $p_*\!$，计数比例的标准差为 $\sqrt{p_*(1-p_*)/B}$。例如 $B=999$、$p_*=0.05$ 时约为 $0.0069$，不能把接近阈值的微小差异当作确定结论。

这是对“至少存在一个正均值候选”的全族检验。一个很小的全族 $p$ 值不等于每个策略都有效，也不证明胜出者优于其他候选。

程序还报告每列的边际 bootstrap $p$ 值和单步 max 调整值：

$$
\hat p_j^{\mathrm{marginal}}
=\frac{1+\sum_b\mathbf1\{Z_j^{*(b)}\geq z_j^{\mathrm{HAC}}\}}{B+1},
\qquad
\hat p_j^{\mathrm{adjusted}}
=\frac{1+\sum_b\mathbf1\{M^{*(b)}\geq z_j^{\mathrm{HAC}}\}}{B+1}.
$$

调整值在每次抽样上使用整个候选族的最大值，因此逐列不小于边际值。对于按最大 HAC 统计量选出的 $\hat j$，$\hat p_{\hat j}^{\mathrm{adjusted}}$ 恰等于上述全族 $p$ 值。边际 bootstrap 处理该列的均值误差分布，不处理事后筛选。列级调整值仍依赖联合 bootstrap 近似；它不是 step-down 检验，也不是获胜概率。

### 5.4 同时置信区间

同一组重抽样可用于双侧同时区间，但使用的是另一种最大值：

$$
A^{*(b)}=\max_j|Z^{*(b)}_j|.
$$

令 $c_{1-\alpha}$ 为 $A^{*(b)}$ 的经验 $1-\alpha$ 分位数，给出

$$
C_j=\left[
\bar d_j-c_{1-\alpha}\widehat{\operatorname{se}}_j,
\bar d_j+c_{1-\alpha}\widehat{\operatorname{se}}_j
\right],\qquad j=1,\ldots,K.
$$

实现采用 NumPy 的 `method="higher"` 经验分位数：对升序抽样值，在零基索引 $\lceil(B-1)(1-\alpha)\rceil$ 取值，不在线性插值后的相邻值之间取点。单侧族检验使用 $\max Z_j^*$；双侧区间使用 $\max|Z_j^*|$。二者的尾部对象不同，不应拿全族 $p$ 值和某个双侧区间是否跨零作精确的数值等价解释。

程序另用 $M^{*(b)}$ 的 `higher` 分位数 $c^+_{1-\alpha}$ 计算单侧同时下界

$$
L_j=\bar d_j-c^+_{1-\alpha}\widehat{\operatorname{se}}_j.
$$

单侧下界与单侧 $p$ 值方向一致，但 `higher` 分位数和加一尾概率的离散边界可相差一个抽样值。因此本版不声称 `adjusted_pvalue <= alpha` 与 `one_sided_lower > 0` 有有限 $B$ 的严格对偶性。这些约定也没有消除 bootstrap 的统计近似误差。

在联合近似成立时，这些区间针对事件 $\{\mu_j\in C_j\ \forall j\}$ 提供渐近覆盖。因为该事件包含所选索引 $\hat j$ 的覆盖，所以筛选后查看 $C_{\hat j}$ 有联合覆盖的依据；这不是“给定某个策略恰好获胜”的条件选择推断。区间也没有直接覆盖 $\mu_j-\mu_k$，比较两个策略需要另行定义差分或联合对比。

## 6. 渐近适用范围

本版的理论说明采取固定 $K$、$T\to\infty$ 的框架。一个使用环境至少需要支持下列步骤：

- 向量差分序列具有适当的平稳性、弱依赖和矩条件，足以得到均值的联合中心极限定理。
- 各列长程方差有限且严格为正，HAC 估计一致。
- 共同索引 stationary bootstrap 一致地近似联合均值误差分布；块长随样本量适当增长。
- 最大值的极限分布在所用临界值处连续，bootstrap 次数足够大。

“弱依赖”不能只由样本自相关图定义。相关文献中的混合系数、矩阶数和带宽增长约束需要共同满足；这里不把最小假设写成一个过宽的通用定理。重尾场景使用有限四阶矩的创新，模拟结果也不延伸到无限方差分布。

还需要区分文献的具体充分条件与本项目的经验场景。例如 White (2000) 附录中引用的一个 stationary-bootstrap 一致性结果采用 $6+\epsilon$ 阶矩条件，而 $t_5$ 成分没有六阶矩。本项目没有把该定理直接套到重尾模拟上；有限四阶矩的核对也不构成完整的联合 bootstrap 一致性证明。重尾场景用于考察本版算法的有限样本行为，不是对所有重尾过程的理论有效性认证。

以下问题没有由本版处理：随 $T$ 快速增长的高维候选集、单位根或强长记忆、结构突变、在线反复检验、自适应生成候选、策略拟合参数误差的普遍修正、样本泄漏和成本模型错误。对这些情形，本版返回数值不代表理论适用性已经得到验证。

## 7. 模拟数据生成

三个场景均用于研究已知均值真值下的统计行为，不是对可交易策略的实证证明。正态 AR(1) 从精确联合平稳正态分布初始化；重尾 AR 从零初始化，GARCH 从无条件方差初始化，后二者均先丢弃 512 期再保留 $T$ 行。这两个场景只是近似平稳，丢弃前段不等于精确采到平稳分布。实验使用边际标准差 $\sigma=0.01$，不按实现出来的样本方差再标准化。完整约定保存在 [protocol.json](../experiments/protocol.json)。

### 7.1 共同与特有创新

设 $u_t$ 与 $v_{tj}$ 相互独立，且跨 $t$ 独立，均值零、方差一。构造

$$
\varepsilon_{tj}=\sqrt\rho\,u_t+\sqrt{1-\rho}\,v_{tj}.
$$

于是每列创新方差为一，不同列同一期创新协方差为 $\rho$。模拟固定 $\rho=0.35$；$\rho$ 是共同创新的方差份额，不是任意模型下收益相关性的直接参数。

正态场景的 $u_t,v_{tj}$ 均为标准正态。重尾场景先取独立的 $t_5$ 随机量，再乘以 $\sqrt{3/5}$ 使方差为一。混合后的 $\varepsilon_{tj}$ 不再严格服从 $t_5$，因此实验名称中的 heavy-tail 表示由 $t_5$ 成分构造的创新。

单位方差 $t_5$ 的四阶矩为 9。由独立性和零均值，混合创新的四阶矩为

$$
\mathbb E\varepsilon_{tj}^4
=9\big[\rho^2+(1-\rho)^2\big]+6\rho(1-\rho)
=3+6\big[\rho^2+(1-\rho)^2\big].
$$

在 $\rho=0.35$ 时为 $6.27$，并始终不大于 9。

### 7.2 正态 AR(1) 与重尾 AR(1)

两者均使用

$$
x_{tj}=\phi x_{t-1,j}+\sigma\sqrt{1-\phi^2}\,\varepsilon_{tj},
\qquad d_{tj}=\mu_j+x_{tj}.
$$

平稳时每列边际方差为 $\sigma^2$，长程方差为 $\sigma^2(1+\phi)/(1-\phi)$。所有列使用相同 $\phi$，则平稳时列间相关系数也是 $\rho$。重尾版本是由 $t_5$ 成分构造创新的 AR(1)，其收益边际分布不是 $t_5$。

### 7.3 重尾 GARCH(1,1)

设

$$
x_{tj}=\sqrt{h_{tj}}\,\varepsilon_{tj},\qquad
h_{tj}=\omega+a x_{t-1,j}^2+b h_{t-1,j},
$$

其中 $a=0.06$、$b=0.90$、$\omega=(1-a-b)\sigma^2=0.04\sigma^2$，创新采用上述重尾构造。初始条件经过 burn-in 衰减后，二阶平稳目标满足 $\mathbb E h_{tj}=\sigma^2$。这一场景不叠加 AR 均值项。

给定过去，$x_{tj}$ 的条件均值为零，所以不同时间的线性自协方差为零，均值长程方差为 $\sigma^2$；条件方差仍随时间变化。因各列条件标准差也随机，收益的无条件横截面相关性一般不等于创新参数 $\rho$。

四阶矩条件可以直接核对：

$$
\mathbb E(a\varepsilon_{tj}^2+b)^2
=a^2\mathbb E\varepsilon_{tj}^4+2ab+b^2
=0.940572<1.
$$

即使把创新四阶矩用上界 9 替代，也得到 $0.9504<1$。此设计保留重尾与波动聚集，又避免把无限方差问题混入本轮比较。GARCH 的经典出处为 [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)。

## 8. 三张图分别检验什么

所有方法在每次 Monte Carlo 重复中使用同一份数据，并按第 4 节的 HAC 规则选择同一个候选。这样方法间的差异不会混入不同的筛选规则。

| 实验 | 真值与变化项 | 需要读出的量 |
| --- | --- | --- |
| 时间依赖 | $K=1$、$\mu=0$，改变正态 AR(1) 的 $\phi$ | 各方法实际拒绝率；iid 方法与第 2.1 节近似的对应关系 |
| 候选筛选 | $\phi=0.5$、全体 $\mu_j=0$，使用嵌套候选集扩大 $K$ | 单列 HAC 不能抵消筛选效应；共同重抽样的族检验表现 |
| 校准与检出 | 正态 AR、重尾 AR、重尾 GARCH；分别使用全零均值和第一列正均值 | 误报率与族检出率，而非胜出者一定为真实信号的概率 |

嵌套候选集指先生成最大 $K$ 的矩阵，再使用其前 $k$ 列。不同 $k$ 使用同一份数据，并共享该次重复的 bootstrap 索引，以减少比较噪声。横轴增加的是已约定候选族的规模，不是模拟研究者看到显著结果后继续提出新策略。

局部备择设为

$$
\mu_1=\delta\sqrt{\Omega_{11}/T},\qquad
\mu_2=\cdots=\mu_K=0.
$$

其中 $\delta\in\{0,1,2,3,4\}$ 在实验配置中事前固定。正态和重尾 AR 使用已知的 $\Omega_{11}=\sigma^2(1+\phi)/(1-\phi)$，GARCH 使用 $\Omega_{11}=\sigma^2$。这种设置把信号大小按均值的不确定性统一，便于比较检出能力；固定 $\delta$、增大 $T$ 并不意味着经济信号保持不变。

均值平移不改变中心化残差、原样本 HAC 尺度及中心化重抽样分布。实现因此能将同一次零均值样本的 bootstrap 抽样用于对应的 mean-shift 备择，减少计算并保持配对比较。选择规则和原样本统计量仍需要随平移重新计算。

## 9. Monte Carlo 不确定性与复现

对一个配置独立生成 $R$ 份数据，令 $J_r=\mathbf1\{\hat p_r\leq\alpha\}$，报告

$$
\hat\pi=\frac1R\sum_{r=1}^{R}J_r.
$$

零假设下它估计实际误报率；备择下它估计检出率。图中的误差范围针对有限 $R$ 带来的比例估计不确定性，不是原策略均值的置信区间。使用 95% Wilson 区间，令 $z=z_{0.975}$，中心和半宽为

$$
m=\frac{\hat\pi+z^2/(2R)}{1+z^2/R},\qquad
w=\frac{z}{1+z^2/R}
\sqrt{\frac{\hat\pi(1-\hat\pi)}R+\frac{z^2}{4R^2}}.
$$

报告区间为 $[m-w,m+w]$，数值上截到 $[0,1]$。它是每个格点的点态 Monte Carlo 区间，不是跨所有格点、曲线和场景的同时区间。该区间仍是比例推断的近似，尤其不能把覆盖名义 $0.05$ 的某一根误差棒解释成方法已经得到普遍有效性证明。[Wilson (1927)](https://doi.org/10.2307/2276774)给出了这一反演思路。

外层 Monte Carlo 次数 $R$ 和内层 bootstrap 次数 $B$ 控制两种不同误差。把 $B$ 调大不会弥补 $R$ 太小，换一个随机种子也不能把方法校准得更好。对每个实验格点保留拒绝次数、重复次数、方法名称、数据参数和区间端点，图由这些表格生成。环境版本、完整参数和随机种子随结果记录，便于逐项核对。

单次审计还返回 `bootstrap_tail_interval`：在当前数据固定的条件下，用内层超越次数除以 $B$ 计算尾概率的 95% Wilson 区间。它衡量重抽样计算误差，不是总体均值区间，也不保证统计假设成立。它针对事先固定的 $B$，不是可随时停止的置信序列。若需要更精确的边界判断，应事先约定更大的 $B$ 后重新计算，不能反复追加抽样直到区间符合期望。

复现应先验证输入和公式，再运行较小配置检查流程，最后按保存的标准配置生成三图。运行成本随 $R\times B\times T\times K$ 增长。固定随机种子保证同一配置和环境下能重现抽样；没有相应数值测试时，不承诺所有 Python、NumPy 和硬件组合的逐位一致性。

### 9.1 运行命令与输出

在源码目录安装实验依赖，然后使用同一个命令运行三图：

```bash
python -m pip install '.[figures]'
strategy-inference reproduce --study baseline --profile full --output results/full --seed 20261002
```

v0.1 的 `quick` 只检查流程，使用 $T=256$、$R=40$、$B=99$；其 `full` 使用 $T=512$、$R=2000$、$B=1999$。具体版本以 [protocol.json](../experiments/protocol.json) 和该次运行保存的 `run-metadata.json` 为准。基线主实验的 $p$ 值分辨率为 $1/2000=0.0005$；在条件尾概率约 0.05 时，内层计数比例的标准差约为 0.0049。分辨率细不等于尾概率估计也同样精确。v0.2 的独立协议使用另一组抽样参数，见第 11.6 节。

| 文件 | 内容 |
| --- | --- |
| `figure-1-autocorrelation.{png,svg,pdf}` | 时间依赖图 |
| `figure-2-selection.{png,svg,pdf}` | 候选筛选图 |
| `figure-3-robustness.{png,svg,pdf}` | 三过程的校准与局部检出图 |
| `figure-1-dependence.csv` | 图 1 的计数、比例和区间 |
| `figure-2-selection.csv` | 图 2 的计数、比例和区间 |
| `figure-3-size.csv`、`figure-3-power.csv` | 图 3 的两组数据 |
| `block-length-sensitivity.csv` | 事前固定的块长敏感性附表 |
| `run-metadata.json` | 已解析配置、协议、随机种子、环境版本和文件校验信息 |

v0.1 的块长敏感性是单独的实验流，完整配置使用 $R=200$、$B=399$，块长为 $\{4,8,16,32\}$。它不用于反过来挑选主实验的默认块长。其较宽的不确定性需要在解读附表时保留。

随机流按根种子与 `(figure, scenario, replicate, purpose)` 的逻辑地址建立。模拟与重抽样使用不同 purpose，增加外层重复数不会改动已有重复的随机地址。嵌套候选集与局部均值平移共享抽样，是事前约定的配对比较；不同实验场景则用各自的流。

### 9.2 为什么不用三维重抽样数组

均值计算只需要每个原始时间行在某轮抽样中出现的次数。记该计数为 $N_{bt}$，则

$$
\bar{\tilde d}^{*(b)}_j=\frac1T\sum_{t=1}^{T}N_{bt}\tilde d_{tj}.
$$

实现先按 stationary-bootstrap 规则生成索引，再生成行计数，最后用计数矩阵乘中心化数据。这样保留了块索引所产生的抽样分布，同时不分配 $B\times T\times K$ 的收益张量。计数没有按独立 multinomial 抽取；否则会变成另一个算法。

以批大小 $C$ 计算时，索引和计数的工作内存为 $O(CT)$，另有 $O(TK)$ 输入和 $O(BK)$ 输出。矩阵乘法的总计算阶数仍为 $O(BTK)$，HAC 计算为 $O((L+1)TK)$。批处理减少峰值内存，不会从复杂度上消除策略数或重抽样次数的影响。

### 9.3 代码接口与统计对象

| 接口／字段 | 对应对象 |
| --- | --- |
| `infer_mean(..., method="iid")` | 独立 Student 单列推断；区间为双侧点态区间 |
| `infer_mean(..., method="hac")` | Bartlett HAC 单列正态近似；区间为双侧点态区间 |
| `long_run_variance(...)` | 各列 $\hat\Omega_{jj}$，不是样本均值方差本身 |
| `stationary_bootstrap_means(..., center=True)` | 中心化数据的共同索引抽样均值，形状为 $B\times K$ |
| `audit_returns(...)` | 完整候选族的固定尺度 bootstrap 推断 |
| `global_pvalue` | $\hat p_{\mathrm{family}}$ |
| `marginal_bootstrap_pvalue`／`adjusted_pvalue` | 列级边际值／单步 max 调整值 |
| `simultaneous_ci_low`／`simultaneous_ci_high` | 双侧 max-absolute 同时区间 |
| `one_sided_lower` | 单侧 max-statistic 同时下界 |
| `search_complete` | 研究者对候选族完整性的声明，不是算法识别结果 |

`search_complete=None` 是默认的未知状态；即使设为 `True`，软件也不验证隐藏搜索是否存在。原始输入为差分时直接传入数组；CSV 接口的 `--benchmark` 选项才执行同期基准列减法。两个入口不能重复减去基准。

## 10. 结果如何表述

一个合适的报告应同时交代以下信息：评价期与频率、基准与成本定义、完整候选集的来源、选择规则、HAC 滞后、平均块长、$B$、随机种子，以及 $p$ 值和区间。

例如可以写：“在已提交的固定候选族和给定评价期内，共同索引 stationary bootstrap 的全族 $p$ 值为……；同时区间为……。这一结论依赖平稳、弱依赖和候选集完整等条件。”

如果结果不显著，应写“当前数据不足以拒绝全族零假设”，而不是“已经证明所有策略无效”。如果单列 HAC 显著、全族检验不显著，应说明搜索调整改变了证据强度。拒绝全族零假设以后，仍需检验策略身份、样本外表现和经济可实施性。

本文的方法依据与软件资料在 [references.bib](references.bib) 中列出。论文承担原方法的学术来源；本项目承担有限范围内的实现、可核对说明和实验记录。

## 11. v0.2：重新学生化与独立评价

### 11.1 为什么增加另一种尺度计算

v0.1 完整模拟表明，在 $T=512$、名义水平 5% 下，固定尺度方法仍有明显误报膨胀：正态 AR(1) 的 $\phi=0.8$、$K=1$ 格点为 9.70%；$\phi=0.5$、$K=50$ 格点为 10.85%。这些结果保留为开发证据，不能在同一批样本上反复改动后，再把更好看的结果称为独立验证。

v0.2 增加 `studentization="resampled"`，每轮重抽样重新估计 HAC 尺度。`studentization="fixed"` 仍为默认值，保持既有调用的含义。两个选项不是对原始数据作不同筛选，而是构造不同的重抽样统计量；它们使用相同的原样本 HAC 统计量、候选胜出规则、中心化方式和共同时间索引。

只有一列时，固定尺度比较可以写成

$$
\frac{\bar{\tilde d}^{*(b)}}{\widehat{\operatorname{se}}}
\geq\frac{\bar d}{\widehat{\operatorname{se}}}
\quad\Longleftrightarrow\quad
\bar{\tilde d}^{*(b)}\geq\bar d.
$$

因此，对固定尺度单列检验，仅改变原样本 HAC 带宽不会改变 bootstrap $p$ 值。增加 $B$ 也只会减小当前条件重抽样尾概率的计算误差。要改变这一路线的有限样本行为，需要改变重抽样分布或统计量本身。

### 11.2 逐轮 HAC 学生化

沿用第 5 节的中心化数据 $\tilde d_{tj}=d_{tj}-\bar d_j$。第 $b$ 轮共享时间索引形成

$$
y^{*(b)}_{tj}=\tilde d_{I_t^{(b)},j},\qquad
\bar y^{*(b)}_j=\frac1T\sum_t y^{*(b)}_{tj}.
$$

计算该轮尺度时，再围绕该轮均值中心化：

$$
e^{*(b)}_{tj}=y^{*(b)}_{tj}-\bar y^{*(b)}_j,\qquad
\hat\gamma^{*(b)}_j(h)=\frac1T\sum_{t=h+1}^T
e^{*(b)}_{tj}e^{*(b)}_{t-h,j}.
$$

使用与原样本相同的滞后数 $L$ 和 Bartlett 权重，得到

$$
\hat\Omega^{*(b)}_{jj}
=\hat\gamma^{*(b)}_j(0)
+2\sum_{h=1}^L\left(1-\frac h{L+1}\right)
\hat\gamma^{*(b)}_j(h),\qquad
Z^{*(b),\mathrm{resampled}}_j
=\frac{\sqrt T\,\bar y^{*(b)}_j}{\sqrt{\hat\Omega^{*(b)}_{jj}}}.
$$

原样本仍采用 $z_j^{\mathrm{HAC}}=\sqrt T\,\bar d_j/\sqrt{\hat\Omega_{jj}}$。在每轮重抽样中取 $\max_j Z^{*(b),\mathrm{resampled}}_j$，再代入第 5.3 节的加一尾概率。列级单步调整也使用该轮的全族最大值。不能先选原胜出列，再只重抽那一列；筛选调整需要保留完整候选族。

该步骤将原样本尺度估计误差的类似变化纳入重抽样，比固定分母多模拟了一层随机性。它是否在给定样本量和数据过程中改善误报率，需要由独立评价回答。学生化并没有移除块拼接对依赖的近似、短带宽偏差或重尾影响。

### 11.3 实现、区间与计算成本

`stationary_bootstrap_statistics` 返回 $B\times K$ 的逐轮 HAC 统计量。实现先生成一批共同索引，再分列批次处理 $C\times T\times Q$ 的样本数组，其中 $C$ 为重抽样批大小，$Q$ 为列批大小。每轮均值在第二次中心化前保存，HAC 自协方差在第二次中心化后计算。列批次不产生新的索引，各列仍共享同一份抽样。

行计数矩阵足以计算均值，却没有记录观测在伪时间序列中的邻接关系，所以不能直接用第 9.2 节的均值加速步骤计算逐轮 HAC。新实现的工作内存为 $O(CTQ+CT)$，另有输入与输出；总计算阶数为 $O(BT(L+1)K)$。以有界的批大小换取较小峰值内存，而不是分配完整收益张量。

若某轮 HAC 方差为零、负数或非有限值，函数报错。它不静默删去该轮、不重新抽取直至成功，也不加任意正数继续计算；这些处理会改动目标重抽样分布。模拟中出现此类失败时也应保留诊断，不能从误报率分母中悄悄排除。

实现使用双精度浮点数。有限输入并不意味着其平方、FFT 功率或方差除以 $T$ 后仍处于可表示范围；极端计价尺度可能溢出或下溢。无效标准误、统计量及条件方差应明确报错，不能继续输出貌似有效的 $p$ 值。数值范围校验不改变正常尺度下的统计构造，也不提供任意精度计算。

双侧同时区间使用 $\max_j|Z^{*(b),\mathrm{resampled}}_j|$ 的经验分位数，再乘回**原样本**标准误：

$$
\bar d_j\ \pm\ c^{\mathrm{resampled}}_{1-\alpha}
\widehat{\operatorname{se}}_j.
$$

这仍是 max-absolute 的对称同时带，不是单列等尾 bootstrap-$t$ 区间。单侧下界使用单侧最大值分位数。加一 $p$ 值、`higher` 分位数和双侧／单侧区别仍适用第 5.4 节的说明，不新增有限 $B$ 的严格对偶性保证。

实际调用为：

```python
from strategy_inference import audit_returns, stationary_mean_variance

# excess_returns 为已经定义好的 T x K 差分矩阵。
result = audit_returns(
    excess_returns,
    studentization="resampled",
    n_resamples=1999,
    seed=17,
    search_complete=None,
)
conditional_variance = stationary_mean_variance(
    excess_returns, block_length=result.block_length
)
```

CSV 审计入口支持 `--studentization resampled`。结果对象保存 `studentization`，JSON 使用 `schema_version=2`，记录方法选项和内层条件尾概率区间。`stationary_mean_variance` 是单独的诊断接口，不会自动把其返回值替换为审计标准误。

### 11.4 stationary-bootstrap 均值方差的精确诊断

令 $a=1-1/\ell$，定义中心化原数据的循环自协方差

$$
\hat\gamma^{\mathrm{circ}}_j(h)
=\frac1T\sum_{t=1}^T
\tilde d_{tj}\tilde d_{1+((t+h-1)\bmod T),j}.
$$

在当前数据固定时，如果相隔 $h$ 步的索引之间没有重启，其概率为 $a^h$，条件协方差为循环自协方差；一旦重启，中心化行的条件期望为零。再对样本均值中的观测对计数，得到

$$
V^{\mathrm{SB}}_j
=\operatorname{Var}^*(\bar y^*_j\mid D)
=\frac1T\left[\hat\gamma^{\mathrm{circ}}_j(0)
+2\sum_{h=1}^{T-1}\left(1-\frac hT\right)
a^h\hat\gamma^{\mathrm{circ}}_j(h)\right].
$$

`stationary_mean_variance` 用 FFT 计算循环自协方差，返回各列 $V^{\mathrm{SB}}_j$，而非 $T V^{\mathrm{SB}}_j$。精确一词针对当前数据和已约定 stationary-bootstrap 索引法的条件方差；浮点计算仍有数值误差。该量不是未知总体均值方差的精确估计，也不是 Monte Carlo 置信区间。

使用第 3.2 节的非循环样本自协方差，等价表达为

$$
T V^{\mathrm{SB}}_j
=\hat\gamma_j(0)+2\sum_{h=1}^{T-1}
\left[\left(1-\frac hT\right)a^h
+\frac hT a^{T-h}\right]\hat\gamma_j(h).
$$

这是 [Nordman (2009)](https://arxiv.org/pdf/0903.0474) 式 (3) 的形式。$\ell=1$ 时返回 $\hat\gamma_j(0)/T$，与独立经验重抽样的均值方差一致。块长变大并不保证方差单调增大：有限循环样本与长块还会产生边界效应。

适当平稳、协方差和累积量可和及块长条件下，文献给出 leading bias

$$
\mathbb E[T V^{\mathrm{SB}}_j]-T\operatorname{Var}(\bar d_j)
=-\frac{G_j}{\ell}+o(1/\ell),\qquad
G_j=\sum_{h\in\mathbb Z}|h|\gamma_j(h).
$$

对正自相关 AR(1)，$G_j>0$，这能解释块重启使均值方差估计偏低的一个机制。它是渐近展开，不能直接当作有限 $T$ 的精确校正系数。

独立评价保存两个主要诊断比值：原样本 $\hat\Omega_{jj}/[T\operatorname{Var}(\bar d_j)]$ 与 $V^{\mathrm{SB}}_j/\operatorname{Var}(\bar d_j)$。总体方差在模拟中来自已知 DGP；真实数据中没有该真值。前者检查 HAC 尺度，后者检查均值重抽样分布。低方差比值可以解释误报方向，却不直接决定该样本的正确 $p$ 值，更不能据此临时缩放临界值。

### 11.5 已知协方差的模型参考

正态 AR(1) 场景在精确平稳初始化、相同 $\phi$、相同 $\sigma$ 和共同创新参数 $\rho$ 下，样本均值有已知联合正态分布。其单列有限样本方差为

$$
v_T=\frac{\sigma^2}T
\left[1+2\sum_{h=1}^{T-1}\left(1-\frac hT\right)\phi^h\right],
$$

列间协方差为 $\rho v_T$。因此

$$
Z^{\mathrm{oracle}}_j=\frac{\bar d_j}{\sqrt{v_T}},\qquad
(Z^{\mathrm{oracle}}_1,\ldots,Z^{\mathrm{oracle}}_K)
\sim N\big(0,(1-\rho)I+\rho\mathbf1\mathbf1^\top\big)
$$

在零均值边界下成立。对于 $0<\rho<1$，令 $U,\epsilon_1,\ldots,\epsilon_K$ 独立标准正态，并写 $Z_j=\sqrt\rho U+\sqrt{1-\rho}\epsilon_j$，则最大值分布为

$$
F_{K,\rho}(c)
=\int_{-\infty}^{\infty}
\Phi\!\left(\frac{c-\sqrt\rho\,u}{\sqrt{1-\rho}}\right)^K
\varphi(u)\,du.
$$

当 $\rho=0$ 时为 $\Phi(c)^K$；$\rho=1$ 或 $K=1$ 时为 $\Phi(c)$。代码通过一维数值积分求值，用确定性求根得到临界值。参考检验使用 $\max_j Z_j^{\mathrm{oracle}}$；其尾部和积分有数值容差，参考正态模型本身的分布无需渐近近似。

数值计算的支持域与参考分布的数学定义需要区分。对一般 $K>1$、$0<\rho<1$，`equicorrelated_max_quantile(p, ...)` 只接受 $p\in[10^{-12},1-10^{-12}]$，超出时明确报错。$K=1$ 或 $\rho\in\{0,1\}$ 使用解析分支，没有这一额外区间限制，但仍要求 $0<p<1$。极端概率下，未缩放的数值积分可能漏掉很远的密度峰，不能以一个看似有限的求根结果宣称支持任意尾部。

CDF 积分设置绝对误差容限 $10^{-12}$；容限不是严格的数学误差界，也不保证极小概率的相对精度。上尾函数直接积分条件生存概率，避免仅由 $1-F(c)$ 相减造成的损失，仍不承诺任意小尾部的相对精度。本文名义 5% 的模型参考使用 $p=0.95$，属于正常支持域；新增范围校验没有改动该计算路径。

`gaussian_ar_mean_variance`、`equicorrelated_max_cdf`、`equicorrelated_max_tail` 和 `equicorrelated_max_quantile` 位于 `strategy_inference.reference`。它们接收模拟生成参数，没有从策略收益估计这些参数。**不能把已知协方差最大值的临界值套到随机 HAC 分母的最大统计量上。** 真实数据没有已知 $\phi,\sigma,\rho$，重尾和 GARCH 场景也不具有这里的有限样本正态参考分布。

该参考帮助核对 Monte Carlo 流程和已有模型下的基准拒绝率。它不是审计 API 的隐藏输入，更不用于自动选取表现最好的方法。

### 11.6 独立评价协议和工程标准

新增实验以 [calibration-protocol.json](../experiments/calibration-protocol.json) 为准，保留旧基线。pilot 根种子为 `20261003`，正式评价使用未用于 pilot 的 `20261004`。主实验固定 $T=512$、$R=2000$、$B=999$，同一份数据上比较 `fixed` 与 `resampled`；两者共享索引和原样本统计量。HAC 仍采用 $L(T)$ 规则，共同期望块长仍为 $\lceil2T^{1/3}\rceil$。本轮不同时改动带宽或加入自动块长。

主图的零均值格点包括时间依赖 5 个、候选规模 5 个和过程比较 3 个，共 13 个。协议还固定 6 个 holdout 设置，每个使用 $R=1000$：

| 过程 | $T$ | $K$ | $\phi$ | $\rho$ |
| --- | ---: | ---: | ---: | ---: |
| 正态 AR | 256 | 10 | 0.7 | 0 |
| 正态 AR | 1024 | 10 | 0.7 | 0.8 |
| 正态 AR | 2048 | 10 | 0.9 | 0.35 |
| 重尾成分 AR | 1024 | 10 | 0.7 | 0 |
| 重尾成分 GARCH | 1024 | 10 | 0 | 0.8 |
| 正态 AR | 512 | 10 | -0.4 | 0.35 |

holdout 在正式评价前规定，覆盖不同长度、创新相关性、较强持久性和负自相关。它不是运行后选出的好看格点，也不意味着覆盖所有金融过程。不同格点用明确的随机地址；一个格点内的两个 bootstrap 方法保持配对。

另有两个预设敏感性场景：$(\phi,K)=(0.8,1)$ 与 $(0.5,50)$，共同默认块长乘数为 $\{0.5,1,2,4\}$，各使用 $R=1000$。敏感性是诊断附表，不用于挑出一个乘数替换主实验默认值。

工程验收只评价 `resampled` 方法在 19 个主实验与 holdout 零均值格点的误报率，不包括正均值功效格点和块长敏感性。每个格点的拒绝次数为 $s_i$、重复次数为 $R_i$，设 $\eta=0.05/19$，计算单侧 Clopper–Pearson 上界

$$
U_i=
\begin{cases}
\operatorname{Beta}^{-1}(1-\eta;\ s_i+1,R_i-s_i), & s_i<R_i,\\
1, & s_i=R_i.
\end{cases}
$$

该表达来自二项尾概率反演，[Clopper–Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404)是其原始来源。Bonferroni 分配使这 19 个上界的联合 Monte Carlo 置信水平至少为 95%，不要求格点之间相互独立；每个格点内部仍需独立重复、固定 $R_i$。

预先约定的通过条件是**所有 $U_i\leq0.07$**。7% 是本项目为名义 5% 检验规定的有限基准容忍上限，不是把名义水平改为 7%，也不是文献给出的普遍校准保证。`quick` 只验证流程，永不据此宣布通过。普通图中的 Wilson 区间仍是点态展示，不能代替这组验收上界。

如果某格点失败，按协议保存失败和全部参数，不根据正式评价结果再次改规则后继续称同一轮独立验证。如果以后修改方法，应保留此轮记录并另立协议和独立数据流。通过也只提供对这一组有限模型和样本长度的证据；不通过仍可公开为有清楚失败边界的研究实现。

功效与误报率并列报告，不能靠牺牲所有检出能力换取看似合格的误报率。局部均值平移对原样本和重抽样 HAC 的中心化残差都不产生影响，所以同一份噪声和重抽样可复用到不同 $\delta$；原样本的第一列统计量和全族最大值随信号更新。该配对下，拒绝事件对 $\delta$ 单调，可作为实现核对。

独立确认的复现入口与旧基线分开：

```bash
strategy-inference reproduce --study calibration --profile full --output results/calibration/full
```

[完整运行记录](../results/calibration/full/run-metadata.json)已保存软件执行成功的 `status=complete`，以及统计验收未过的 `assessment.status=failed`：19 个格点中 1 个上界不高于 7%，18 个未过。原运行使用冻结提交 `579697d`，没有因后续增加输入校验、安装资源查找或数值域限制而改写其源码哈希；版本差异和 quick 对照范围见[结果解读](results.md)。这种未通过记录是有限模型证据，不是“真实误报率已被证明超过 7%”的一般断言。

### 11.7 为什么暂不启用自动块长

[Politis–White (2004)](https://doi.org/10.1081/ETC-120028836)及 [Patton–Politis–White (2009)](https://doi.org/10.1080/07474930802459016)提出、修正了数据驱动块长选择。其 stationary-bootstrap 形式可写成

$$
\hat\ell_j
=\left(\frac{T\hat G_j^2}{\hat\Omega_j^2}\right)^{1/3},
$$

其中 $\hat G_j$ 与 $\hat\Omega_j$ 由 flat-top 自协方差加权估计；具体截断规则、常数和边界需参照完整算法。[arch 的官方说明](https://arch.readthedocs.io/en/stable/bootstrap/generated/arch.bootstrap.optimal_block_length.html)给出了这些约定，并明确二维输入逐列计算。

该方法针对方差估计 MSE 的块长选择，不是最大统计量检验误报率的直接优化。单位边际方差 AR(1) 的对应理论量为

$$
\ell_{\mathrm{SB}}^{\mathrm{MSE}}
=\left[T\left(\frac{2\phi}{1-\phi^2}\right)^2\right]^{1/3}.
$$

在 $T=512$ 时，$\phi=0.5$ 约为 9.69，$\phi=0.8$ 约为 21.63；它不总比当前默认 16 更大，不能预先认定能解决误报膨胀。各列建议块长还必须聚合为一个共同块长，才可保留全族共享索引；这种聚合需要独立说明与评价。GARCH 收益没有线性自相关，也不意味着涉及平方收益的尺度统计量没有依赖。

因此，v0.2 暂时只改变学生化方式，使有限样本差异能够解释。自动块长可以作为后续独立扩展，而不能根据本轮误报结果临时启用。类似地，[Newey–West (1994)](https://doi.org/10.2307/2297912)的自动 HAC 滞后选择也以协方差估计 MSE 为目标，其原文同时报告有限样本 size distortion；自动选择不是校准证书。

### 11.8 关于理论精度的表述

原尺度和重抽样尺度均一致、联合均值 bootstrap 有效、最大值分布在临界值处连续时，重新学生化可由联合收敛与 Slutsky 定理得到一阶渐近依据。v0.2 仍沿用固定有限候选族的范围，没有推导 $K$ 快速增长时的有效性。

[Götze–Künsch (1996)](https://doi.org/10.1214/aos/1069362303)的高阶结果要求合适的方差估计和相应正则条件；[Politis (2003)](https://math.ucsd.edu/~politis/StatSci03.pdf)也强调尺度估计方式的重要性。[arch 的学生化说明](https://arch.readthedocs.io/en/stable/bootstrap/confidence-intervals.html)使用“在一些条件下可能改善精度”的措辞，并不把它作为所有样本上的保证。

本项目仅重算 Bartlett HAC，没有证明其有限样本精确性或二阶正确性，也没有验证重尾模拟满足这些高阶结果的全部矩和依赖条件。论文中的学生化研究为构造提供依据；正式评价提供有限模型下的实验记录。两者应分别陈述，不能把采用一个统计术语写成已经获得相应定理。

## 12. v0.3：未知时间参数的固定水平检验

`uncertainty_test` 检验事先给定的候选族中的 `μ_j≤0`，使用 GLS 均值估计。模型是共同未知 `0≤φ<1` 的平稳 Gaussian AR(1)，每列尺度未知，同期协方差任意、对角严格为正；不能将异质时间参数或非 Gaussian 收益直接视作已满足模型。

事先固定参考列，第 0 列是默认选择。它的创新向量经固定有理正交投影分成两组，在真实参数下的方差比精确服从 F 分布。将接受条件反演成两条二次不等式，使用 Bernstein 符号证书得到连续参数集合的外包。已知候选参数的 GLS t 按保守 Student 临界值校验；只有外包中所有参数都通过认证才拒绝相应列。完整代数与证明见[参数不确定性说明](parameter-uncertainty.md)。

覆盖失败预算为全族一次 `β`，均值检验每列预算为 `(α−β)/K`。即使其他候选有信号，真实零假设列的任何拒绝概率仍不超过 `α`。这个保证不要求参考列与均值检验独立，也不要求候选之间独立。

接口返回 `UncertaintyResult`：固定水平的列级 `decisions`、精确有理 `intervals`、供展示的 `interval_bounds`、`phi1_retained`，以及外包和拒绝认证诊断。它没有连续 p 值；不能根据同一份数据反复选择水平、参考列或投影后保留最显著结果。空集合与认证预算不足均不拒绝；保留 `φ=1` 时 GLS 极限趋于零，本程序全部不拒绝。

数学分布结论针对理想 Gaussian 模型；整数和有理数证书针对传入的 binary64 输入，没有为测量舍入建立独立分布保证。新实验直接量化这套构造的保守功效损失，[结果记录](uncertainty-results.md)表明它尚不具有实用竞争力。原 bootstrap 审计接口仍保留原来的含义与限制。
