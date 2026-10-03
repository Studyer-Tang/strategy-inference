# 已知参数下，候选规模如何进入校准上界

推导记录，2026-10-03。本文把最新尾部修正基线与已有 Gaussian 集中、反集中工具连接起来，给出可以逐项核对的**充分条件**。不是精确边界，不声称原创；拟合参数、未知相关矩阵和随机 bootstrap 临界值不在结论范围内。

## 模型与目标

沿用[机制说明](tail-mechanism.md)的平稳 Gaussian AR(1) 模型：`T` 期、`K` 个预先给定的零均值候选，边际方差 `σ²>0`，共同自回归系数 `0≤φ<1`，等相关参数 `0≤ρ≤1`。时间协方差与候选协方差可分离。

令

\[
H=I-\mathbf1\mathbf1'/T,\quad
W_{st}=(1-|s-t|/\ell)_+,\quad 1\le\ell<T,
\]

\[
C_{st}=\sigma^2\phi^{|s-t|},\qquad
\widehat\Omega_j=X_j'HWHX_j/T.
\]

总体 Bartlett 核尺度、长程方差和有限样本均值尺度为

\[
M_\ell=\sigma^2\left[1+2\sum_{h=1}^{\ell-1}(1-h/\ell)\phi^h\right],
\quad \Omega=\sigma^2\frac{1+\phi}{1-\phi},
\quad v_T=T\operatorname{Var}(\bar X_j).
\]

真实参数有限目标补尾是 `v̂_j=v_T Ω̂_j/M_ℓ`，来自 Liu–Chan Remark 3.2。令

\[
r=\frac{\ell\Omega}{T M_\ell},\quad
Z_j=\sqrt T\bar X_j/\sqrt{v_T},\quad M=\max_{j\le K}Z_j.
\]

`Z` 是标准等相关 Gaussian 向量。取固定 `0<α<1/2`，`q=q_{1−α}(K,ρ)>0` 为其真实最大值分位数。本文研究使用该临界值的随机尺度统计量

\[
\widehat M=\max_{j\le K}\frac{\sqrt T\bar X_j}{\sqrt{\widehat v_j}}.
\]

## 1. 尺度误差的全列概率界

对单列，写 `X_j=C^(1/2)g`、`g∼N(0,I)`，则

\[
\frac{\widehat\Omega_j}{M_\ell}=g'Bg,
\qquad B=\frac{C^{1/2}HWHC^{1/2}}{T M_\ell}\succeq0.
\]

机制说明中的均值和谱界给出

\[
0\le1-\operatorname{tr}(B)\le\ell/T+2r,\qquad
\|B\|_{\rm op}\le r,\qquad
\operatorname{tr}(B^2)\le r.
\tag{1}
\]

最后一个不等式也可由 `tr(B²)≤||B|| tr(B)` 得到，因为 `tr(B)≤1`。这一步使用非负 `φ` 的 AR 协方差与 Bartlett 权重；不直接扩展到负 `φ` 或任意核。

令 `λ_i≥0` 是 `B` 的特征值，`V=Σ_i λ_i(g_i²−1)`，`s²=Σ_i λ_i²`，`b=max_i λ_i`。Gaussian 平方的矩母函数给出

\[
\log E e^{tV}\le\frac{s^2t^2}{1-2bt},\quad0<t<(2b)^{-1},
\qquad \log E e^{-tV}\le s^2t^2,\quad t>0.
\]

Chernoff 优化得到

\[
P\{V>2s\sqrt u+2bu\}\le e^{-u},\qquad
P\{V<-2s\sqrt u\}\le e^{-u}.
\tag{2}
\]

这是已有的 [Laurent–Massart，2000，Lemma 1](https://doi.org/10.1214/aos/1015957395)；以上写出矩母函数步骤以免把其使用包装为新结果。

任取 `0<η<1`，设

\[
u=\log(2K/\eta),\qquad
\varepsilon=\ell/T+2r+2\sqrt{ru}+2ru.
\tag{3}
\]

由 (1)–(2) 与 union bound，

\[
P\left\{\max_{j\le K}\left|\frac{\widehat v_j}{v_T}-1\right|>\varepsilon\right\}\le\eta.
\tag{4}
\]

这里没有要求候选列独立，也没有要求样本均值与尺度独立。相关结构仅可能让 union bound 偏松。

## 2. Gaussian 最大值在阈值附近的敏感度

记 `a_K=E(M)`。根据 [Chernozhukov–Chetverikov–Kato，2015，Theorem 3(i)](https://denischetverikov.wordpress.com/wp-content/uploads/2018/07/2015cck.pdf)，任意长度为 `L` 的区间均满足

\[
P(M\in I)\le 2L(a_K+1),\qquad
a_K\le\sqrt{2\log K}.
\tag{5}
\]

因子 2 来自该定理对半宽 `L/2` 的区间给出 `4(L/2)(a_K+1)`。`K=1` 时 `a_K=0`。

等相关结构还给出一个简单的改进。若 `ρ>0`，可以写

\[
M=\sqrt\rho\,G+\sqrt{1-\rho}\,U_K,
\qquad U_K=\max_{j\le K}G_j,
\]

其中 `G,G_1,…,G_K` 独立标准 Gaussian。给定 `U_K` 后，第一项是方差 `ρ` 的 Gaussian，因此卷积密度有上界

\[
\sup_x f_M(x)\le(2\pi\rho)^{-1/2}.
\tag{6}
\]

这也覆盖 `ρ=1`。它是共同因子表达式的基础推论，不是对未知因子的估计保证。综合 (5)–(6)，定义区间概率的 Lipschitz 常数

\[
D_{K,\rho}=\begin{cases}
2(a_K+1),&\rho=0,\\
\min\{2(a_K+1),(2\pi\rho)^{-1/2}\},&\rho>0.
\end{cases}
\tag{7}
\]

则 `P(M∈I)≤D_{K,ρ}|I|`。可以用 `√(2log K)` 替换 `a_K`，得到无需额外计算期望的较松常数。

## 3. 有限样本尺寸上界

若 (3) 的 `ε<1`，在全列尺度误差不超过 `ε` 的事件上，事件夹逼给出

\[
P\{M>q\sqrt{1+\varepsilon}\}-\eta
\le P(\widehat M>q)
\le P\{M>q\sqrt{1-\varepsilon}\}+\eta.
\]

因为 `P(M>q)=α`，且阈值两侧的每个区间宽度至多 `q ε`，

\[
\boxed{\left|P(\widehat M>q)-\alpha\right|
\le\eta+D_{K,\rho}\,q\,\varepsilon.}
\tag{8}
\]

这是显式的有限样本充分界。右边可能大于 1，此时它没有实际辨别力；不能将这种松上界当作观测误报率或精确边界。若 `ε≥1`，上述正尺度阈值夹逼也不能给出 (8)。

## 4. 候选增长与相关结构的两个充分条件

先假设 `K=K_T→∞`，取 `η=1/K`，则 `u=log(2K²)=O(log K)`。由 Gaussian Chernoff 和 union bound，

\[
q\le\sqrt{2\log(K/\alpha)}.
\]

另一方面，AR 精度矩阵的 Cauchy 界给出

\[
r\le\frac{\ell+2\phi/(1-\phi)}T
\le\frac{\ell+2\tau}T,\qquad \tau=(1-\phi)^{-1}.
\tag{9}
\]

由 `M_ℓ≤Ω` 还有 `ell/T≤r`。

- **允许 `ρ_T` 任意变化的保守条件**：若 `r_T(log K_T)^3→0`，则 (3) 的 `ε_T log K_T→0`，(5) 和 (8) 保证尺寸趋于 `α`。
- **共同相关有固定正下界**：若 `ρ_T≥ρ_*>0` 且 `r_T(log K_T)^2→0`，则 (6) 和 (8) 保证尺寸趋于 `α`。

第一条主导项为 `sqrt(r log K)×log K`；第二条主导项为 `sqrt(r log K)×sqrt(log K)`。两条都是由已有工具得到的充分条件。较宽的第二条说明，相关结构不仅改变多重比较临界值，还会改变阈值对尺度误差的敏感度。它不证明第二条最优，也不意味着增加相关性在每个有限样本方法中都一定更好。

例如，温和持久性 `φ_T=1−a T^(−κ)`、`0<κ<1`，以及 `ell_T=o(T)`，使 (9) 显式连接带宽、时间记忆与候选规模。将其代入上述充分条件是本模型的推论，不是对所有金融收益过程的结论。

固定 `K` 应另取 `η_T→0`，选择足够慢的速度使 `r_T log(1/η_T)→0`；`r_T→0` 时可做到。不能机械沿用 `η=1/K` 来证明固定 `K` 的极限。

## 仍需解决的部分

本轮实验使用真实相关矩阵，所以与这里的临界值假设一致；但拟合尾因子的增长 `K` 误差没有包含在 (3)。固定 `K` 的拟合一致性见机制说明，不能仅凭该结果把所有 `φ` 换成 `φ̂`。

未知相关结构需要临界值误差，模型错配需要额外控制，随机重抽样临界值需要条件分布近似。接下来的研究应在这些环节寻找可验证的增量，并检查上述对数幂能否被具体的随机尺度结构改进；当前充分条件不作为原创定理或实践验收规则发布。
