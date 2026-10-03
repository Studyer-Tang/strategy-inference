# 参数拟合与完整统计量重放

推导记录，2026-10-03。本文把项目的有限目标补尾统计量放进一个可模拟的参数模型，明确三个层次：已知生成参数的有限样本 Monte Carlo 参照、未知参数下固定候选数的渐近保证，以及仍需研究的持久性边界。

这里的参数 bootstrap 是已有方法。相关估计的一致性、Gaussian 二次型集中、连续映射和 Monte Carlo 秩检验也不是本项目的原创工具。本文的作用是把具体估计器、重放对象和保证范围写到可以核查的程度；不把正确模型内的校准升级为一般金融策略检验。

## 1. 模型与单侧全局假设

有 `T` 期、预先给定的 `K` 列候选，`K` 在本节渐近结论中保持固定。假定

\[
X_{tj}=\mu_j+\sigma_j Z_{tj},\qquad \sigma_j>0,
\]

\[
\operatorname{Cov}(Z_{sj},Z_{tk})
=\phi^{|s-t|}(R_\rho)_{jk},\qquad
R_\rho=(1-\rho)I_K+\rho\mathbf1\mathbf1',
\tag{1}
\]

其中 `Z` 联合 Gaussian、平稳边际方差为 1。理论真值限制为 `0≤φ<1`、`0≤ρ≤1`，各列共有一个时间参数。`ρ=1` 允许重复的标准化列；`K=1` 时 `ρ` 不影响分布，约定为 0。各列边际尺度 `σ_j` 可以不同。

目标是单侧全局检验

\[
H_0:\ \mu_j\le0\ \text{对所有 }j,
\qquad H_1:\ \text{至少一列 }\mu_j>0.
\tag{2}
\]

这里报告的是全局拒绝，不直接报告哪个策略已被证明显著，不将零均值实验视为任意自适应搜索的 strong FWER 证明。

设 `H=I−11'/T`、确定性 Bartlett 宽度 `ell=lags+1` 满足 `1≤ell<T`，并令

\[
W_{st}=(1-|s-t|/\ell)_+,\qquad
\widehat\Omega_j=X_j'HWHX_j/T,
\]

\[
m_n(\psi)=1+2\sum_{h=1}^{n-1}(1-h/n)\psi^h,
\qquad
\eta_{T,\ell}(\psi)=m_T(\psi)/m_\ell(\psi).
\tag{3}
\]

对 `|ψ|<1`，`m_n(ψ)>0`，因为它是长度 `n` 的平稳 AR(1) 均值方差乘 `n`。有限目标比值来自 Liu–Chan 的 Remark 3.2，详见[尾部机制说明](tail-mechanism.md)。

## 2. 固定估计规则与重放对象

记 `Y_tj=X_tj−X̄_j`，每列都采用相同的中心化自协方差估计规则：

\[
\widehat\phi_j
=\frac{\sum_{t=2}^T Y_{tj}Y_{t-1,j}}
{\sum_{t=1}^T Y_{tj}^2},\qquad
\widehat\phi_{\rm common}=K^{-1}\sum_{j=1}^K\widehat\phi_j.
\tag{4}
\]

共同参数是每列比值的平均，不能在实现或说明中改称 pooled ratio。它可以为负，不投影到 `[0,1)`；数学上非恒定列给 `|φ̂_j|<1`，平均仍在 `(-1,1)`。这不是对所有浮点退化输入的承诺。

相关估计使用中心化 Pearson 相关：

\[
\widehat r_{jk}
=\frac{\sum_tY_{tj}Y_{tk}}
{\{\sum_tY_{tj}^2\sum_tY_{tk}^2\}^{1/2}},
\]

\[
\widehat\rho_{\rm raw}
=\frac{2}{K(K-1)}\sum_{j<k}\widehat r_{jk},
\qquad
\widehat\rho=\Pi_{[0,1]}(\widehat\rho_{\rm raw}).
\tag{5}
\]

`K=1` 时两个相关估计均记为 0。投影明确限制了生成模型：它是非负等相关的约束矩估计，不是任意相关矩阵估计。原始值与投影值需要分别记录；投影不能掩盖模型错配。

检验的统计量固定为

\[
\widehat v_j
=\widehat\Omega_j\eta_{T,\ell}(\widehat\phi_j),
\qquad
S(X)=\max_{j\le K}
\frac{\sqrt T\,\bar X_j}{\sqrt{\widehat v_j}}.
\tag{6}
\]

每次 bootstrap draw 从零均值、单位边际尺度、参数
`(φ̂_common,ρ̂)` 的平稳模型生成整段 `T×K` 数据，再完整计算式 (4)、(3)、(6)。这包括每列参数拟合、HAC、随机尺度与最大值筛选。式 (6) 不使用 `ρ̂`，因此 bootstrap draw 内再次估计相关参数不会改变统计量；相关只进入生成模型。

所有规则须在评价前固定：`ell`、拟合器、约束、统计量和数值失效处理不能在观察拒绝率后调整。若尺度非正或拟合数值失效，应按协议报错，不能删除该轮后只统计成功轮次。

### 2.1 均值和边际尺度为何不需要另行拟合

对正数 `a_j` 和常数 `b_j`，变换 `X_j↦a_jX_j+b_j1` 不改变 `φ̂_j`、Pearson 相关与两个生成参数；HAC 乘以 `a_j²`。在零均值下，式 (6) 的分子与分母同乘 `a_j`，统计量不变。因此零均值生成时设 `σ_j=1` 不引入未知边际尺度。

给定同一噪声路径，加入非正均值只降低每列分子，所有拟合参数与尺度保持不变。于是

\[
S(\mu+\text{noise})\le S(\text{noise})
\quad\text{当全部 }\mu_j\le0.
\tag{7}
\]

这一单调性也保留 fitted-DGP 的零均值最不利边界，但不能单独证明 fitted-DGP 在有限样本有效；生成参数与观察统计量仍然来自同一数据。

## 3. 时间依赖下相关估计的一致性

不能把相关估计中的 `T` 个时间点当作 IID 观测。以下计算显示持久性如何降低有效信息。

设

\[
C_0=(\phi^{|s-t|})_{s,t\le T},\qquad
a_T=\operatorname{tr}(HC_0)/T=1-m_T(\phi)/T,
\]

\[
Q_{jk}=\frac{X_j'HX_k}{T\sigma_j\sigma_k},
\qquad U_T=\operatorname{tr}(HC_0HC_0).
\tag{8}
\]

常数均值被 `H` 消去。由 Gaussian 四阶矩的两种交叉配对，

\[
E Q_{jk}=(R_\rho)_{jk}a_T,
\qquad
\operatorname{Var}(Q_{jk})
=\{1+(R_\rho)_{jk}^2\}U_T/T^2.
\tag{9}
\]

对角情形系数为 2。式 (9) 可直接由 `E(Z_sj Z_tk Z_uj Z_vk)` 减去均值乘积后展开：剩下一个同列时间配对与一个带 `R_jk²` 的跨列时间配对，两项共享同一个迹。无需假设候选列独立。

因为 `H` 是正交投影，Frobenius 范数收缩给

\[
U_T=\|HC_0H\|_F^2\le\|C_0\|_F^2
\le T\frac{1+\phi^2}{1-\phi^2}
\le\frac{T}{1-\phi},\qquad 0\le\phi<1.
\tag{10}
\]

同时 `m_T(φ)≤(1+φ)/(1−φ)≤2/(1−φ)`，所以 `a_T→1` 当 `T(1−φ)→∞`。从 (9)–(10)，固定 `K` 时所有 `Q_jk−R_jk a_T` 均为 `O_p([T(1−φ)]^(−1/2))`。

Pearson 相关中的共同中心化因子 `a_T` 会抵消。具体地，若所有上三角元素满足

\[
|Q_{jk}-R_{jk}a_T|\le\epsilon\le a_T/2,
\]

则 `sqrt(Q_jj Q_kk)` 位于 `[a_T−epsilon,a_T+epsilon]`，从而

\[
|\widehat r_{jk}-R_{jk}|
\le\frac{2\epsilon}{a_T-\epsilon}
\le\frac{4\epsilon}{a_T}.
\tag{11}
\]

平均与向 `[0,1]` 的投影不会放大相对于真实 `ρ∈[0,1]` 的最大误差。并集上界和 (9) 给

\[
P\left\{|\widehat\rho-\rho|>4\epsilon/a_T\right\}
\le\frac{K(K+1)U_T}{T^2\epsilon^2}.
\tag{12}
\]

因此固定 `K`、`T(1−φ)→∞` 时

\[
\widehat\rho-\rho=O_p([T(1-\phi)]^{-1/2})=o_p(1).
\tag{13}
\]

这是正确可分离等相关模型内的结果。共同时间参数不成立、异方差动态不同或相关结构不是等相关时，式 (9) 的目标需要重新分析；简单投影不修复错配。

## 4. 固定候选数下的参数与尺度极限

以下渐近条件固定为

\[
K<\infty,\qquad \ell_T/T\to0,\qquad
0\le\phi_T<1,\qquad T(1-\phi_T)\to\infty.
\tag{14}
\]

`ell` 确定且至少为 1，可以固定，也可以增长。`ρ_T∈[0,1]` 可以随 `T` 变化。没有允许 `K` 增长，也没有允许自适应选择带宽、搜索候选或停止。

### 4.1 参数拟合的相对误差

对一列，将均值与边际尺度消去，写 `d=1−φ`、`S_0=Σ(X_t−X̄)²`、`S_1=Σ_{t=2}^T(X_t−X̄)(X_{t−1}−X̄)`。AR 表达为
`X_t=φX_{t−1}+sqrt(1−φ²)ε_t`，平稳初值独立于后续标准 Gaussian 创新。精确恒等式为

\[
S_1-\phi S_0
=\sqrt{1-\phi^2}\sum_{t=2}^TX_{t-1}\epsilon_t
-\phi X_T^2-(Td+1)\bar X^2+\bar X(X_1+X_T).
\tag{15}
\]

首项的鞅差正交给方差 `(1−φ²)(T−1)≤2Td`。Gaussian 四阶矩给 `Var(T^(−1)ΣX_t²)≤2/(Td)`，而 `E X̄²≤2/(Td)`。所以 `S_0/T→p1`，式 (15) 的其余项是 `O_p(1)`，得到

\[
\frac{|\widehat\phi_j-\phi|}{d}
=O_p\{(Td)^{-1/2}+(Td)^{-1}\}=o_p(1).
\tag{16}
\]

固定 `K` 后，平均也满足

\[
\frac{|\widehat\phi_{\rm common}-\phi|}{d}=o_p(1),
\qquad
\frac{1-\widehat\phi_{\rm common}}{1-\phi}\to_p1.
\tag{17}
\]

这些概率界的常数不依赖 `ρ`；每列的边际模型相同，固定 `K` 的并集上界即可，不需跨列独立。

### 4.2 补尾因子对拟合误差的敏感度

[机制说明第 8 节](tail-mechanism.md)已给完整导数证明。对 `0≤ψ<1`，

\[
0\le\partial_\psi\log m_n(\psi)
\le\frac{2}{1-\psi^2},\qquad
|\partial_\psi\log\eta_{T,\ell}(\psi)|
\le\frac{2}{1-\psi^2}.
\tag{18}
\]

将 `m_n` 看成几何滞后分布被递减 Bartlett 权重倾斜的和，倾斜降低平均滞后，即给第一条；两个非负导数位于同一上界内，其差给第二条。

不能将该非负参数证明直接套给负的 `φ̂`。真值 `φ≥1/8` 时，(16) 使连接真值与拟合值的路径以高概率保持非负，且距 1 至少为 `d/2`，故 (18) 至多为 `4/d`。真值 `φ<1/8` 时，路径以高概率位于 `[-1/4,1/4]`；在该区间 `m_n≥1/3`、`|m_n'|≤32/9`，于是 `|∂log eta|≤64/3`，统一于 `n`。均值定理给

\[
\eta_{T,\ell}(\widehat\phi_j)/\eta_{T,\ell}(\phi)\to_p1.
\tag{19}
\]

同一证明也适用于共同参数估计。这正是相对 `1−φ` 误差必要的原因：仅有 `φ̂−φ→p0` 不够。

### 4.3 随机 HAC 仍需单独集中

单位边际尺度下记 `Ω=(1+φ)/(1−φ)`、`M_ell=m_ell(φ)`、`v_T=m_T(φ)`，以及

\[
r=\frac{\ell\Omega}{T M_\ell},\qquad b=\ell/T+2r.
\]

[机制说明第 7 节](tail-mechanism.md)的精确二次型界为

\[
0\le1-\frac{E\widehat\Omega_j}{M_\ell}\le b,
\qquad
\frac{\operatorname{Var}(\widehat\Omega_j)}{M_\ell^2}\le2r,
\qquad
r\le\frac{\ell+2/(1-\phi)}T.
\tag{20}
\]

条件 (14) 使 `r→0`，所以 `Ω̂_j/M_ell→p1`。结合 (19)，固定 `K` 时

\[
\max_{j\le K}\left|
\frac{\widehat v_j}{\sigma_j^2m_T(\phi)}-1
\right|\to_p0.
\tag{21}
\]

参数拟合一致性并不代替 (20)。在随机尺度未集中的场景里，即使校正因子使用真实参数，也不能凭 (19) 取得 Gaussian 尾部。

### 4.4 原样本统计量与 fitted Gaussian 临界值

全零均值下，有限 `T` 已经准确有

\[
A_j=\frac{\sqrt T\bar X_j}{\sigma_j\sqrt{m_T(\phi)}},
\qquad A\sim N(0,R_\rho).
\tag{22}
\]

这是 Gaussian 均值的恒等分布，不是新 CLT。固定 `K` 的 `max|A_j|=O_p(1)` 与 (21) 给

\[
S(X)-\max_jA_j\to_p0.
\tag{23}
\]

令 `F_{K,ρ}` 为标准等相关 Gaussian 最大值的 CDF。写

\[
\max_jA_j\overset d=
\sqrt\rho\,G+\sqrt{1-\rho}\max_{j\le K}G_j
\tag{24}
\]

可见 `F_{K,ρ}(x)` 在 `(x,ρ)∈R×[0,1]` 连续。每个 `ρ` 下分布连续、严格递增，`ρ=1` 退化成一个标准正态，仍然如此。因此固定 `0<α<1/2` 的分位数 `q_{1−α}(K,ρ)` 在紧区间 `[0,1]` 连续。

式 (13) 给 `q(K,ρ̂)−q(K,ρ)→p0`。用 (23) 和连续 CDF，可得全零均值下

\[
P\{S(X)>q_{1-\alpha}(K,\widehat\rho)\}\to\alpha.
\tag{25}
\]

若 `ρ_T` 不收敛，可对任意子序列抽取在 `[0,1]` 收敛的进一步子序列；上述结论沿每条进一步子序列成立，故原序列也成立。由 (7)，非正均值全局零假设下极限上界为 `α`。

## 5. fitted-DGP 的条件 pivotal 证明

本节说明完整统计量重放为何在条件 (14) 下可行。结论针对准确模拟整个平稳 Gaussian 模型的参数 bootstrap。

给定原样本，记生成参数

\[
\psi_T=\widehat\phi_{\rm common},\qquad
\varrho_T=\widehat\rho,
\]

每次生成零均值单位边际尺度的 `X*`，按式 (6) 重新计算 `S*`。`P*` 表示条件于原样本的模拟概率。“条件概率趋零”指该条件概率本身按原样本概率趋于零。

### 5.1 负的生成参数需要单独处理

真值 `φ≥1/8` 时，(17) 使 `ψ≥0` 以高概率成立，且 `1−ψ` 与 `1−φ` 相对接近，故
`T(1−ψ)→p∞`。上面的正参数界可以条件于原样本应用。

真值 `φ<1/8` 时，`ψ` 以高概率位于 `[-1/4,1/4]`。任意固定 `c<1` 的紧区间 `|ψ|≤c` 上，AR 协方差的特征值满足

\[
\frac{1-c}{1+c}\le\lambda_{\min}(C_0)
\le\lambda_{\max}(C_0)\le\frac{1+c}{1-c}.
\tag{26}
\]

可由 AR 谱密度 `(1−ψ²)/(1−2ψ cosλ+ψ²)` 的上下界与有限 Toeplitz 二次型积分得出。因此所有 `m_n(ψ)` 有统一正下界，导数的绝对值有统一有限上界；绝对可和的自协方差给中心化与边缘偏差 `O(ell/T)`，Gaussian 二次型方差也为 `O(ell/T)`。参数拟合误差为 `O_P*(T^(−1/2))`。于是这一紧区间同样有条件拟合尺度一致性，无需错误地把 (20) 的非负损失符号扩展到负参数。

这些条件界可以写成显式形式。令 `M=(1+c)/(1−c)`，则统一于 `|ψ|≤c`，

\[
|E\widehat\Omega-m_\ell(\psi)|
\le \frac{2c}{T(1-c)^2}+\frac{3M\ell}{T},
\qquad
\operatorname{Var}(\widehat\Omega)\le\frac{2M^2\ell}{T},
\qquad m_\ell(\psi)\ge M^{-1}.
\]

第一式的首项是有限样本边缘损失的绝对值上界；展开 `HWH` 的两个中心化项，分别用 `||W||≤ell`、`||C_0||≤M`，得到剩下的 `3Mell/T`。第二式由 Gaussian 二次型方差、Frobenius 收缩与 `tr(W²)≤Tell` 得出。导数级数也给 `|m_n'(ψ)|≤2/(1−c)²`，故 `|∂log eta|≤4M/(1−c)²`。这些常数不依赖随机拟合参数或样本量，足以在上述紧区间条件化。

### 5.2 条件尺度集中与 CDF 逼近

上一小节保证，无论生成参数来自非负持久区间还是零附近的负参数紧区间，固定 `K` 都有

\[
\max_{j\le K}\left|
\frac{\widehat v_j^*}{m_T(\psi_T)}-1
\right|\xrightarrow{P^*}0\quad\text{按原样本概率成立}.
\tag{27}
\]

这里使用 (15)–(20) 的显式矩与导数界：在高概率事件上
`ell/T+[T(1−ψ)]^(−1)→0`；这些界只通过该量控制条件概率，而不是对一个任意随机参数直接援引逐点极限。

条件于原样本的准确 Gaussian 均值向量为

\[
A_j^*=\frac{\sqrt T\bar X_j^*}{\sqrt{m_T(\psi_T)}},
\qquad A^*\mid X\sim N(0,R_{\varrho_T}).
\tag{28}
\]

固定 `K` 的 Gaussian 尾界统一控制 `max|A_j*|`，所以 (27) 给 `S*−max A_j*→P*0`。对任意实数 `x` 与 `h>0`，

\[
|P^*(S^*\le x)-F_{K,\varrho_T}(x)|
\le P^*\{|S^*-\max_jA_j^*|>h\}
+\frac{2Kh}{\sqrt{2\pi}}.
\tag{29}
\]

这里直接使用条件分布的统一界：`max_j A_j*` 落入 `[x−h,x+h]` 时，至少一个坐标落入该区间；每个条件边际都是标准正态，密度至多 `1/sqrt(2π)`，故并集上界为 `2Kh/sqrt(2π)`。它不需要坐标独立，统一于所有 `varrho_T∈[0,1]`。先控制第一项，再令 `h→0`，得

\[
\sup_x|P^*(S^*\le x)-F_{K,\widehat\rho}(x)|\to_p0.
\tag{30}
\]

由 (13)、(24) 与固定维 CDF 的连续性，

\[
\boxed{\ \sup_x|P^*(S^*\le x)-F_{K,\rho}(x)|\to_p0.\ }
\tag{31}
\]

这是一阶条件 pivotal 结论。不要求拟合生成模型的整条数据分布在 total variation 下接近真实分布；那是更强、且这里没有证明的断言。也不证明有限样本临界值误差足够小，具体误差仍须由独立正式实验评价。

### 5.3 固定 Monte Carlo 轮数的尺寸

设 `B≥1`，给定原样本独立生成 `B` 个 bootstrap 数据，统计量为 `S_b*`。采用

\[
\widehat p_B
=\frac{1+\sum_{b=1}^B1[S_b^*\ge S(X)]}{B+1},
\qquad\text{拒绝当 }\widehat p_B\le\alpha.
\tag{32}
\]

固定 `B` 时，(31) 与条件独立意味着沿任意 `ρ_T→ρ_0` 的子序列，观察统计量和模拟统计量的联合极限是 `B+1` 个独立、同分布的 `F_{K,ρ_0}` 变量。可先对矩形事件条件化，利用条件 CDF 的一致收敛使模拟部分的条件概率趋近固定乘积，再与 (23) 合并。极限分布连续，出现同值的概率为零，所以观察值在 `B+1` 个值中的尾部秩均匀。

因此全零均值下

\[
P(\widehat p_B\le\alpha)
\longrightarrow\alpha_B
:=\frac{\lfloor\alpha(B+1)\rfloor}{B+1}\le\alpha.
\tag{33}
\]

不收敛的 `ρ_T` 用之前的紧子序列论证处理。非正均值全局零假设通过 (7) 给极限上界 `α_B`。例如 `α=.05,B=999` 时 `α_B=.05`；`B=100` 时则为 `5/101`。这也是为什么不能随手把有限模拟经验分位数与式 (32) 的检验当作完全相同的规则。

本文证明固定 `B` 的结论；它不自动给任意增长 `B_T` 的误差率。若用无限精度 bootstrap 分位数，(31) 与分位数连续性给全零均值渐近尺寸 `α`。有限模拟的不确定性和有限 `T` 的拟合误差是两个不同来源。

### 5.4 冻结校正因子的对照同样可以一阶有效

为隔离重拟合的有限样本作用，定义一个有意不完整的重放对照：bootstrap draw 重算 HAC，但把校正因子固定在观察样本的逐列估计上，

\[
\widehat v^*_{{\rm freeze},j}
=\widehat\Omega_j^*\eta_{T,\ell}(\widehat\phi_j).
\tag{34}
\]

原样本观察统计量仍为 (6)，生成模型仍使用共同 `ψ=φ̂_common`。由 (16)–(19)，

\[
\frac{\eta_{T,\ell}(\widehat\phi_j)}
{\eta_{T,\ell}(\psi)}\to_p1.
\]

条件 HAC 集中给 `Ω̂_j*/m_ell(ψ)→P*1`，于是 (34) 相对于 `m_T(ψ)` 的比值也趋于 1，故其条件最大值 CDF 同样满足 (31)。固定 `B` 的一阶尺寸仍为 (33)。

因此完整重拟合不能在这个固定 `K` 结论下被描述成“渐近有效的必要步骤”。它准确重放观察统计量，有机会改善有限样本的随机尺度与筛选关系；改善幅度、计算成本和失败场景需要实验回答。冻结因子的版本使用观察样本参数，不再是对同一个 `S` 函数的完全重放，不能套用下一节 known-DGP 的交换性精确保证。

## 6. 已知生成参数的有限样本 Monte Carlo 参照

现在固定有限 `T,K,ell`，真实 `φ,ρ` 已知。模拟零均值、单位边际尺度的 `B` 个独立数据，并在每个数据上完整计算同一个 `S`，包括式 (4) 的每列拟合。已知的生成参数与统计量内拟合的参数是两个不同对象。

全零均值下，逐列尺度不变性使观察统计量与模拟统计量同分布。`B+1` 个值交换，故 (32) 给

\[
P(\widehat p_B\le\alpha)\le\alpha_B.
\tag{35}
\]

没有同值时等号成立；采用 `≥` 对同值保守处理，不需要随机打破同值。这是经典 Monte Carlo 秩检验，见 [Dufour，2006，Propositions 2.2–2.4](https://jeanmariedufour.github.io/Dufour_1995_MCT_W.pdf)。它与 bootstrap 的区别是生成参数为真实值，不能把 (35) 的有限样本保证转移给数据拟合的生成参数。

对于 (2) 的复合全局零假设，耦合同一零均值噪声与同一模拟样本。(7) 使观察 `S` 下降，而 p-value 对观察 `S` 单调不增，故 p-value 只能上升。于是 (35) 对全部 `μ_j≤0` 仍成立。该保证不需要已知边际尺度 `σ_j`，但需要 Gaussian AR(1) 时间协方差、等相关横截面结构和正确平稳初值。

### 6.1 “精确”所依赖的计算规则

有限样本秩保证要求：观察值与全部模拟值调用同一统计函数，`T,K,ell` 相同，随机生成独立于观察数据并使用真实参数，数值失败不能只在模拟样本中被选择性删除。一个简单可审查的方式是按协议在数值失败时中止整次计算。

代码正确性测试和数值稳定性检查支持这些条件，但不能通过模拟拒绝率证明它们。尤其应避免把使用观察 `φ̂_j` 的冻结校正版本贴上 known-DGP exact 标签：那一版本的 `B+1` 个统计量并未按同一独立函数生成。

## 7. 已知时间参数的 GLS Student t 参照

另一个参照只需已知共同时间参数 `φ`，横截面相关可以任意，不需要等相关或估计 `ρ`。每列仍须是 Gaussian，具有时间协方差 `σ_j² C_0(φ)`，`σ_j` 未知。

令

\[
D=\mathbf1'C_0^{-1}\mathbf1
=\frac{2+(T-2)(1-\phi)}{1+\phi},
\]

\[
\widehat\mu_{{\rm GLS},j}
=\frac{\mathbf1'C_0^{-1}X_j}{D}
=\frac{X_{1j}+X_{Tj}+(1-\phi)\sum_{t=2}^{T-1}X_{tj}}
{2+(T-2)(1-\phi)}.
\tag{36}
\]

这里 `T≥2`，内部和在 `T=2` 时为空。AR 精度矩阵为

\[
C_0^{-1}=\frac1{1-\phi^2}
\begin{pmatrix}
1&-\phi&&\\
-\phi&1+\phi^2&-\phi&\\
&\ddots&\ddots&\ddots\\
&&-\phi&1
\end{pmatrix}.
\tag{37}
\]

仅内部对角元素为 `1+φ²`，端点对角为 1。计算残差尺度和统计量

\[
s_j^2
=\frac{(X_j-\widehat\mu_{{\rm GLS},j}\mathbf1)'
C_0^{-1}(X_j-\widehat\mu_{{\rm GLS},j}\mathbf1)}{T-1},
\qquad
t_j=\widehat\mu_{{\rm GLS},j}\sqrt D/s_j.
\tag{38}
\]

**有限样本证明。** 写 `C_0^(−1/2)X_j=μ_j u+σ_j g_j`，`u=C_0^(−1/2)1`、`g_j∼N(0,I_T)`。`u/||u||` 方向上的投影是一个标准正态；与其正交的 `T−1` 维残差独立，残差平方和除以 `σ_j²` 是 `χ²_{T−1}`。因此 `μ_j=0` 时

\[
t_j\sim t_{T-1}
\tag{39}
\]

准确成立。`μ_j≤0` 时分子沿同一噪声路径下降，残差尺度对均值平移不变，所以该单侧尾部概率不大于中心 t 尾部。

不需要候选列独立，Bonferroni 即给

\[
P_{H_0}\left\{\max_{j\le K}t_j>
t_{T-1,1-\alpha/K}\right\}\le\alpha.
\tag{40}
\]

这是一项有限样本、已知时间参数的参照。它不校准原来的 HAC 统计量，换了均值估计器和分母；比较应同时记录误报率与功效。它也不提供未知 `φ` 的精确保证。若把 `φ` 换成 `φ̂`，whitening 的投影方向和残差变成随机对象，(39) 的独立性论证就不能直接使用。AR 回归的普通固定设计 t 证明同样不能直接套给依赖未来创新的随机滞后回归量。

## 8. Local-to-unity：估计器的限制与待证方向

本节只证明式 (4) 的单列参数估计器在一个 local-to-unity 序列上的非退化极限。它不证明完整统计量重放的拒绝率极限，也不把“本节一阶条件不成立”写成“所有参数 bootstrap 都失败”。

### 8.1 一个精确恒等式

取 `K=1`，记 `V_T=T^(−1)Σ(X_t−X̄)²`。直接展开相邻差平方，得到

\[
T(1-\widehat\phi)
=\frac{(X_1-\bar X)^2+(X_T-\bar X)^2
+\sum_{t=2}^T(X_t-X_{t-1})^2}{2V_T}.
\tag{41}
\]

因此估计 `T(1−φ)` 不仅取决于增量能量，也取决于整段中心化路径的随机能量与端点。只用 `φ̂→p1` 无法证明相对持久性参数已经拟合准确。

### 8.2 固定 local 参数的非退化极限

假设 `φ_T=1−a/T`、固定 `a>0`，只取使 `0<φ_T<1` 的 `T`，并利用平移和边际尺度不变性令均值为 0、边际方差为 1。令平稳 OU 过程

\[
dZ_a(r)=-aZ_a(r)\,dr+\sqrt{2a}\,dW(r),
\qquad Z_a(0)=G\sim N(0,1),\quad G\perp W.
\tag{42}
\]

定义

\[
\bar Z_a=\int_0^1Z_a(r)\,dr,
\qquad V_a=\int_0^1\{Z_a(r)-\bar Z_a\}^2\,dr.
\]

则该具体估计器满足

\[
T(1-\widehat\phi)\Rightarrow
A_a:=\frac{\{Z_a(0)-\bar Z_a\}^2
+\{Z_a(1)-\bar Z_a\}^2+2a}{2V_a}.
\tag{43}
\]

**证明。** 设 `b_T=−T log(1−a/T)→a`。在时间点 `r_t=(t−1)/T` 取平稳 `Z_bT`，其协方差为 `exp(−b_T|s−t|/T)=φ_T^(|s−t|)`，所以网格值与原 AR 样本同分布。可以把所有 `Z_b` 放在同一个 `G,W` 上：

\[
Z_b(r)=e^{-br}G+\sqrt{2b}\left[
W(r)-b\int_0^re^{-b(r-u)}W(u)\,du\right].
\tag{44}
\]

这是 OU 解的确定性积分分部形式。连续 Brownian 路径上，右边对 `b` 的连续性统一于 `r∈[0,1]`，故 `Z_bT→Z_a` 一致几乎必然。Riemann 和与端点于是给 `X̄→bar Z_a`、`V_T→V_a`、`X_1→Z_a(0)`、`X_T→Z_a(1)`。

增量满足 `ΔZ_b=−b∫Z_b dr+sqrt(2b)ΔW`。网格总长度趋于 1，Brownian 增量平方和趋概率于 1；漂移平方和由路径有界性为 `O(1/T)`，交叉和由 Cauchy–Schwarz 趋零。因此 `Σ(ΔX_t)²→p2a`。`V_a>0` 几乎必然：连续路径若 `V_a=0` 则为常数，而 OU 的二次变差为 `2a>0`。代入 (41) 得 (43)。

`A_a` 确实非退化，而不只是一个未辨认的随机表达式。Brownian 路径在 `C_0[0,1]` 具有完整支撑，初值 `G` 在实线上具有完整支撑；OU 方程的连续变换及逆变换
`W(r)=[Z(r)−Z(0)+a∫_0^r Z(u)du]/sqrt(2a)` 使 `Z_a` 在 `C[0,1]` 具有完整支撑。Brownian 支撑事实也可由分段线性逼近、Gaussian 增量的正密度和 Brownian bridge 小球概率得到。

式 (43) 右边是 `V>0` 路径上的连续泛函。对确定性路径 `z_c(r)=c sin(2πr)`，`c>0`，其均值、两个端点为 0，`V=c²/2`，该泛函取值 `2a/c²`。选两个不同的 `c`，它们的足够小路径邻域有正概率，泛函值落入两个不相交区间，故 `A_a` 非退化。这样，

\[
\frac{1-\widehat\phi}{1-\phi_T}\Rightarrow A_a/a
\]

不能趋概率于 1。这里证明的是这个估计器的相对一致性失败，尚不是 bootstrap 尺寸失败。

### 8.3 不能跳过的后续证明

当 `ell=o(T)`，一个需要完整核查的统计量极限候选是

\[
S_T\ \Rightarrow\
\frac{\bar Z_a}{\sqrt{f(A_a)V_a}},
\qquad f(u)=\frac{2(u-1+e^{-u})}{u^2}.
\tag{45}
\]

式 (45) **尚未作为本文证明的定理**。它涉及带宽增长时 HAC 路径极限、随机拟合参数代入 `m_T` 的联合收敛与正尺度控制。对于 fitted-DGP，还需要证明条件于原样本时模拟的路径极限，确认极限分布对 local 参数 `a` 的敏感度，再检验随机 `A_a` 代入是否产生不消失的临界值误差。未知相关参数会添加另一层随机性。

当前可成立的陈述只有：条件 (14) 排除了 `T(1−φ_T)=a`；式 (43) 说明本项目的具体拟合器不能沿这个序列提供前面使用的相对一致性。因此 (31) 的证明不能直接延伸，任何该区域的实验结果应作为预设压力检验报告。不能据此声称所有 local-to-unity 序列的 fitted-DGP 检验必然失败，也不能把一次有限样本改善称为 uniform 有效。

## 9. 后续的参数不确定性路线

如果希望获得未知参数的有限样本保证，单纯代入估计点不够。已有的一条路线是 nuisance confidence set 加 supremum：对 `θ=(φ,ρ)` 构造覆盖率至少 `1−β` 的置信集 `C`，并令

\[
p_{\rm BB}=\min\{1,\beta+\sup_{\theta\in C}p_\theta\},
\qquad 0<\beta<\alpha,
\tag{46}
\]

其中每个 `p_θ` 是真实参数为 `θ` 时有效的 rank-MC p-value。全局零假设下

\[
P(p_{\rm BB}\le\alpha)
\le P(p_{\theta_0}\le\alpha-\beta)
+P(\theta_0\notin C)\le\alpha.
\tag{47}
\]

这里不需要置信集与检验 p-value 独立，也不需要各参数值使用独立模拟样本。可用中心化、尺度不变的 nuisance-statistic 做逐点 rank-MC 检验，并以不拒绝参数点组成置信集。覆盖率只依赖真实参数点对应的检验有效，因此共同随机数可以跨参数复用。

若将这一路线实现为正式方法，还须给出可测的随机置信集、共同模拟过程，以及可测的 supremum 或保守上界，并证明真实参数点的覆盖率。本文没有完成这项构造；式 (47) 仅是在这些条件成立时的已有原理。

这属于 [Berger–Boos，1994](https://doi.org/10.1080/01621459.1994.10476836) 与 [Dufour，2006](https://doi.org/10.1016/j.jeconom.2005.06.007) 的既有原理。近期 [Glazer–Stark，JCGS 2026](https://doi.org/10.1080/10618600.2025.2526416) 讨论保守 Monte Carlo 置信集和重复使用同一模拟样本的计算。

困难在连续参数上的可证 supremum：有限网格的最大值是完整 supremum 的下界，直接拿它代入 (46) 不能保证保守。快速单参数算法通常需要单调性或准凹性，这些条件尚未在本项目的 `φ,ρ` 统计量上建立。未处理网格间隙和优化误差前，应将网格结果称为参数敏感性分析，而非已经有效的 nuisance 校准。

## 10. 文献、原创性与结果范围

- [Liu–Chan，Tail Postcoloring in Long-Run Variance Estimation of Time Series，JASA 2026](https://doi.org/10.1080/01621459.2026.2676715)，阅读版本为 [arXiv v1](https://arxiv.org/html/2605.15596v1)：提供 AR 补尾公式和有限目标；本项目另行明确该公式放进最大值统计量后的重放流程，不声称补尾本身原创。
- [Dufour，Monte Carlo tests with nuisance parameters，Journal of Econometrics 133(2)，2006，443–477](https://doi.org/10.1016/j.jeconom.2005.06.007)，[作者 2005 年稿](https://jeanmariedufour.github.io/Dufour_1995_MCT_W.pdf)：提供有限样本秩检验、nuisance maximization 与一致点估计下 Monte Carlo bootstrap 的理论背景。本文第 5 节核查具体估计器的条件，不把一般理论称作新贡献。
- [Glazer–Stark，Fast Conservative Monte Carlo Confidence Sets，JCGS 35(1)，2026，273–282](https://doi.org/10.1080/10618600.2025.2526416)，online 2025-08-06，[公开作者稿](https://par.nsf.gov/servlets/purl/10646803)：说明保守 Monte Carlo 置信集及共用模拟样本；其计算条件不自动适用于本项目的连续二维 nuisance 优化。
- [Berger–Boos，P Values Maximized Over a Confidence Set for the Nuisance Parameter，JASA 89(427)，1994，1012–1016](https://doi.org/10.1080/01621459.1994.10476836)：式 (46) 的来源。nuisance confidence set 与 supremum 不是本项目发明。

已知参数的有限样本参照、固定 `K` 的渐近结论和待证方向应分别阅读。上述结果没有覆盖非 Gaussian 分布、波动聚集、异质 AR 参数、一般动态因子、增长 `K`、自适应候选搜索或停止。机制实验的既有结果保持冻结；新一轮联合校准实验须另用协议和种子，不能用于修改此前结果。

当前值得继续研究的增量是：在明确持久性范围内，参数不确定性、随机 HAC 与最大值筛选如何共同改变临界值；完整重放改善多少，代价多少，何时仍不足。原创性须围绕可证明的误差界或可核查的新机制逐项查重，不能只靠实现已有 bootstrap 或发现某个基线在压力场景失效来建立。
