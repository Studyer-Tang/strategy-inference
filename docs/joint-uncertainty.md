# 联合时间形状集合与连续均值认证

推导记录，2026-10-03。上一轮只用一列的时间形状构造共同 AR 参数集合。本轮研究如何使用事先指定的多列信息，同时消去未知的列间协方差。方法仍然针对共同时间参数的平稳 Gaussian AR(1)；它不要求等相关，通过枢轴消去未知同期协方差。

核心统计量是经典的 Wilks determinant ratio。它的有限样本分布、矩与 Markov/Chernoff 上界都不是本项目的新理论。本轮的具体工作是将固定的多尺度时间对比、共享参数覆盖预算和连续 GLS 认证连起来，用有理运算保存可核查的证据，并评价信息增加之后仍需承担的功效与计算成本。本文不给未经正式冻结实验支持的效果数字。

实现位于 [`wilks.py`](../src/strategy_inference/wilks.py)，接口为 `wilks_uncertainty_test`。已知时间参数的 GLS、Student 临界值和拒绝认证复用 [`uncertainty.py`](../src/strategy_inference/uncertainty.py)。较早的单列 F 集合推导见[未知时间参数说明](parameter-uncertainty.md)。

## 1. 模型、候选族与共同预算

有 `T` 期、事先给定的 `K` 列数据。假设

$$
X_t=\mu+U_t,\qquad U_t=\phi U_{t-1}+\varepsilon_t,
\qquad 0\le\phi<1,
\tag{1}
$$

$$
U_1\sim N_K(0,\Sigma),\qquad
\varepsilon_2,\ldots,\varepsilon_T
\overset{\mathrm{iid}}\sim N_K(0,(1-\phi^2)\Sigma),
\tag{2}
$$

其中初值与创新独立，`Σ` 为半正定矩阵且每个 `Σ_jj>0`。因此

$$
\operatorname{Cov}(X_t,X_v)=\phi^{|t-v|}\Sigma.
\tag{3}
$$

允许不同尺度、负的列间相关及奇异的整体协方差。这里的“任意相关”仅指式 (3) 内的任意 `Σ`；时间参数仍然共同，时间结构仍然为 AR(1)，创新仍然 Gaussian。

每列检验 `H₀j: μ_j≤0` 对 `H₁j: μ_j>0`。令 `I₀={j:μ_j≤0}`，目标为 strong family-wise error rate：

$$
P\{\text{至少拒绝一个 }j\in I_0\}\le\alpha.
\tag{4}
$$

其余列可以有任意均值。参数集合所用的列也可以有信号。

固定 `0<β<α<1/2`。均值检验每列的尾概率预算为

$$
\eta=\frac{\alpha-\beta}{K}.
\tag{5}
$$

`β` 是全族共同的参数覆盖预算，只使用一次。当前默认 `α=.05`、`β=.005`；内部用所给 binary float 的精确有理值。

参数集合使用前

$$
r=\min(K,r_{\max}),\qquad 1\le r_{\max}\le8
\tag{6}
$$

列，默认 `r_max=8`。列顺序与这个上限必须事先指定。不能根据观察到的收益、胜出列、样本相关或本次结果重新选列。其他列仍然参加式 (4) 的均值同时检验，但不向这个形状集合提供额外信息。这里的 `r` 是固定的选列数，不是估计的有效秩。

## 2. 固定的多尺度时间分解

预先指定 `S` 个互不重复的整数块长 `L₁,…,L_S≥2`，默认 `(4,16,64)`。每个尺度分配

$$
\beta_a=\beta/S.
\tag{7}
$$

不可使用的尺度也保留这份分配，不把它的预算依据本次数据重新分给其他尺度。以下先写一个块长 `L` 的分解。

令 `b` 为不超过 `⌊(T−1)/L⌋` 的最大非负奇数；当没有正奇数时取零。使用最前面的 `n=bL` 个创新，并设

$$
q=b-1,\qquad s=n-b.
\tag{8}
$$

只有 `b≥3` 且 `s≥r+2` 时，这个尺度才进入后续两侧矩计算。否则它返回整个 `[0,1]`，没有排除参数的作用。代码中不可用尺度的显示自由度将 `q` 截到非负数。

使用奇数块数使 `q` 为正偶数，方便第 5 节的有理矩化简。Wilks 分布本身不要求这个奇偶性。若原最大块数为偶数，会再省去一整块；这是计算选型带来的信息成本，应与仅省去不足一个块的方案区分。

定义块内取平均的矩阵

$$
B_L=\operatorname{diag}\left(
\frac{\mathbf1_L\mathbf1_L'}L,\ldots,
\frac{\mathbf1_L\mathbf1_L'}L\right),
\qquad J_n=\frac{\mathbf1_n\mathbf1_n'}n,
$$

$$
P_L=B_L-J_n,\qquad Q_L=I_n-B_L,
\qquad M_n=I_n-J_n.
\tag{9}
$$

`B_L` 是秩为 `b` 的正交投影，其像空间包含常数方向。因此

$$
P_L^2=P_L=P_L',\quad Q_L^2=Q_L=Q_L',\quad
P_LQ_L=0,
$$

$$
P_L\mathbf1_n=Q_L\mathbf1_n=0,\qquad
P_L+Q_L=M_n,\qquad
\operatorname{rank}(P_L)=q,\quad
\operatorname{rank}(Q_L)=s.
\tag{10}
$$

`P_L` 描述去掉整体均值后的块平均变化，`Q_L` 描述块内变化。投影矩阵完全由 `T` 和预设块长决定，没有从数据中拟合时间子空间。

## 3. 真实参数下的两个独立 Wishart 矩阵

记选定前 `r` 列为 `X_J`，其协方差子矩阵为 `Σ_J`。对任意候选 `ψ∈[0,1]`，构造 `n×r` 创新矩阵

$$
Z(\psi)_{i,\cdot}=X_{i+1,J}'-\psi X_{i,J}',
\qquad i=1,\ldots,n.
\tag{11}
$$

两组 scatter 为

$$
W_P(\psi)=Z(\psi)'P_LZ(\psi),\qquad
W_Q(\psi)=Z(\psi)'Q_LZ(\psi),
$$

$$
W_T(\psi)=W_P(\psi)+W_Q(\psi)=Z(\psi)'M_nZ(\psi).
\tag{12}
$$

在真实 `ψ=φ`，

$$
Z(\phi)=\mathbf1_n(1-\phi)\mu_J'+E,
\qquad E_{i,\cdot}\overset{\mathrm{iid}}\sim
N_r(0,\Omega),
\quad \Omega=(1-\phi^2)\Sigma_J.
\tag{13}
$$

式 (10) 消去均值。选一个确定性的正交行变换，使 `P_L`、`Q_L` 成为互不重叠的坐标投影。Gaussian 行向量经过正交变换后，协方差仍为 `I_n⊗Ω`，故两组坐标彼此独立。这给

$$
W_P(\phi)\sim\mathcal W_r(q,\Omega),\qquad
W_Q(\phi)\sim\mathcal W_r(s,\Omega),
\qquad W_P(\phi)\perp W_Q(\phi).
\tag{14}
$$

先假设 `Σ_J` 正定。因为 `s≥r`，`W_Q(φ)` 与 `W_T(φ)` 几乎处处正定。定义 Wilks 比值

$$
\Lambda_L(\psi)=\frac{\det W_Q(\psi)}{\det W_T(\psi)}
\tag{15}
$$

于分母正的候选处。`0≼W_Q≼W_T`，所以该比值在 `[0,1]` 内。对真实参数做共同的列 whitening，两个矩阵的行列式同时乘 `det(Ω⁻¹)`，比值消去全部未知 `Σ_J` 和创新尺度。

即使 `φ` 很接近 1，这个消去仍是有限样本恒等式；没有用“根与 1 保持固定距离”的渐近条件。但这不保证参数集合狭窄，也不保证浮点行列式稳定。实际计算用第 7 节的有理多项式。

`q<r` 时，`W_P` 必然奇异。式 (15) 仍然有定义；只需要分母与 `W_Q` 的正定性，不需要 `W_P` 有密度或可逆。下面证明这种情况下的分布。

## 4. Product-Beta 分布：不要求 `q≥r`

式 (14) whitening 后，可写为

$$
W_Q=G_Q'G_Q,\qquad W_P=G_P'G_P,
\quad G_Q\in\mathbb R^{s\times r},\quad
G_P\in\mathbb R^{q\times r},
\tag{16}
$$

其中两个矩阵的全部元素为独立标准 Gaussian。令 `G=(G_Q',G_P')'`。按列做 Gram 行列式的 Schur 分解。

给定前 `i−1` 列，令 `A_i` 是 `ℝ^(s+q)` 中只在前 `s` 个坐标上非零，且与 `G_Q` 的前 `i−1` 列正交的子空间；令 `D_i` 是与 `G` 的前 `i−1` 列正交的子空间。前面的 Gaussian 列几乎处处满列秩，故

$$
A_i\subseteq D_i,\qquad
\dim A_i=s-i+1,\qquad
\dim D_i=s+q-i+1.
\tag{17}
$$

包含关系来自：只在 Q 坐标上有值的向量，与总列的内积就是与其 Q 部分的内积。记总矩阵第 `i` 列为 `g_i`，它独立于给定的前面列。嵌套正交投影给

$$
U_i=\|\Pi_{A_i}g_i\|^2\sim\chi^2_{s-i+1},\qquad
V_i=\|\Pi_{D_i\cap A_i^\perp}g_i\|^2\sim\chi_q^2,
\tag{18}
$$

条件下二者独立。因此

$$
B_i=\frac{U_i}{U_i+V_i}
\sim\operatorname{Beta}\left(\frac{s-i+1}{2},\frac q2\right).
\tag{19}
$$

这个条件分布只依赖维数，与给定的前面列无关，故 `B_i` 独立于前面列的 sigma-field；而 `B₁,…,B_{i−1}` 都可由前面列确定。归纳得到 `B₁,…,B_r` 互相独立。

Gram determinant 的第 `i` 个 Schur 因子分别是 Q 部分的剩余平方长度 `U_i`，以及总矩阵的剩余平方长度 `U_i+V_i`。故

$$
\boxed{\quad
\Lambda_L(\phi)\ \overset d=\ \prod_{i=1}^r B_i,
\qquad B_i\ \text{独立},\quad
B_i\sim\operatorname{Beta}\left(\frac{s-i+1}{2},\frac q2\right).
\quad}
\tag{20}
$$

整个证明只要求 `s≥r` 和 `q>0`，没有要求 `q≥r`。未知列间相关由 whitening 消去，但没有因此假设原列彼此独立。

这是经典有限样本 Wilks 分布。[Dufour–Khalaf (2002), Appendix A.1](https://www2.cirano.qc.ca/~dufourj/Web_Site/Dufour_Khalaf_1996_MLR_W.pdf) 在 Gaussian 多元回归中给出同一独立 Beta 乘积；以其残差自由度对应 `s`、约束秩对应 `q`、响应维数对应 `r` 即得式 (20)。这里使用的是精确 determinant ratio 分布，不是把 Wilks 渐近 likelihood-ratio chi-square 定理用到近单位根样本上。

## 5. 正负整数矩与存在条件

对 `B∼Beta(a,b)`，由 Beta 积分可得

$$
E(B^h)=\frac{B(a+h,b)}{B(a,b)}
=\frac{(a)_h}{(a+b)_h},\qquad h=1,2,\ldots,
$$

$$
E(B^{-h})=\frac{B(a-h,b)}{B(a,b)},\qquad h<a.
\tag{21}
$$

负矩条件必须严格：`h=a` 时零点附近的积分已经发散。由式 (20)，

$$
M_h:=E(\Lambda^h)
=\prod_{i=1}^r\prod_{u=0}^{h-1}
\frac{s-i+1+2u}{q+s-i+1+2u},
\qquad h\ge1,
\tag{22}
$$

$$
N_h:=E(\Lambda^{-h})
=\prod_{i=1}^r\prod_{u=1}^{h}
\frac{q+s-i+1-2u}{s-i+1-2u},
\qquad 1\le h<\frac{s-r+1}{2}.
\tag{23}
$$

所以负整数幂的最大许可值是 `⌊(s−r)/2⌋`。不可把 `≤(s−r+1)/2` 当作条件，也不可跨过边界后继续计算一个符号不明的分母。

式 (22)–(23) 都是正有理数，可精确计算。当前 `q/2=v` 是整数，当幂数 `h>v` 时还可使用短乘积

$$
E(\Lambda^h)=
\prod_{i=1}^r\prod_{u=0}^{v-1}
\frac{s-i+1+2u}{s-i+1+2u+2h},
\tag{24}
$$

其中 `h` 可为许可的负整数。它由 Gamma 比值交换两组上升阶乘得到，与前两式完全相同；只是减少整数乘法，不是分布近似。

## 6. 数据无关的矩界与外向有理根

一个尺度的尾预算为 `β_L=β/S`。对每个许可整数幂，Markov 不等式给

$$
P\{\Lambda<\ell_h\}\le\beta_L/2,
\qquad
\ell_h=\left(\frac{\beta_L}{2N_h}\right)^{1/h},
\tag{25}
$$

$$
P\{\Lambda>u_h\}\le\beta_L/2,
\qquad
u_h=\left(\frac{2M_h}{\beta_L}\right)^{1/h}.
\tag{26}
$$

例如式 (25) 来自 `P(Λ^(−h)>ℓ_h^(−h))≤N_h ℓ_h^h`。这是对 `−logΛ` 的指数矩界，属于通常的 Chernoff 方法；不是精确分位数计算。若 `u_h≥1`，可直接以 1 为上界，因为 `Λ≤1`。

当前实现采用预定的有限幂集合。下侧搜索

$$
\mathcal H_-=\{1,\ldots,
\min[64,\lfloor(s-r)/2\rfloor]\}.
\tag{27}
$$

上侧令 `M=2^⌈log₂(4n)⌉`，搜索 `{1,2,3,4}`，以及不超过 `M` 的 `2^a` 与 `3·2^(a−1)`，其中 `a≥2`。这些选择只依赖样本数、投影自由度和预设参数。它们未穷尽所有存在的矩，不声称在整个 Chernoff 家族中全局最优。

每个根用 `critical_bits=40` 的 dyadic 区间夹住。若目标为 `t=A/B∈(0,1)`，在整数点 `m/2^w` 上比较

$$
\left(\frac m{2^w}\right)^h\le\frac AB
\quad\iff\quad m^hB\le A\,2^{wh}.
\tag{28}
$$

全部判断使用整数。浮点对数和指数仅提供搜索的起始猜值；必须再用式 (28) 确认 bracket 两端并完成二分。低临界值取根的下侧端点，高临界值取上侧端点，再在预定幂集合中选择

$$
\ell_L=\max_{h\in\mathcal H_-}\underline\ell_h,
\qquad
u_L=\min\left(1,\min_{h\in\mathcal H_+}\overline u_h\right).
\tag{29}
$$

这分别只向外扩大接受区域。因此

$$
P\{\Lambda_L(\phi)<\ell_L\}\le\beta_L/2,
\qquad
P\{\Lambda_L(\phi)>u_L\}\le\beta_L/2.
\tag{30}
$$

不需要为不同幂数额外支付多重比较预算：临界值与达到最大/最小值的幂都由分布自由度确定，完全没有读取观察到的 `Λ`。如果根据数据选择块长、列数、幂集合或尾概率，这个论证就不再直接适用。

这种计算的优点是无需 Monte Carlo 临界值、特殊函数分位数或参数拟合。代价是矩界可能明显宽于精确分位数，幂集合和 dyadic 精度又会继续向外移动临界值。保证覆盖并不意味着这些代价在功效上很小。

## 7. 行列式的整数多项式

对输入的每列 binary float，以共同正分母把该列全部观测变成整数 `y_{t,j}`。设相应列尺度矩阵为正的对角矩阵 `D`，则整数数据为 `X_JD`。在候选参数下，行列式比值不变，因为两个 scatter 都经同一个 congruence 变换。

令整数创新行为 `z_i(ψ)=y_{i+1,J}−ψy_{i,J}`。记

$$
G(\psi)=\sum_{i=1}^n z_i(\psi)z_i(\psi)',\qquad
b_v(\psi)=\sum_{i\text{ 属于块 }v}z_i(\psi),\qquad
b_*(\psi)=\sum_{i=1}^n z_i(\psi).
$$

由式 (9)，整数矩阵多项式为

$$
A_L(\psi)=n\left\{LG(\psi)-\sum_{v=1}^{b}
b_v(\psi)b_v(\psi)'\right\}=nL W_Q^{(y)}(\psi),
$$

$$
B_L^*(\psi)=L\{nG(\psi)-b_*(\psi)b_*(\psi)'\}
=nL W_T^{(y)}(\psi).
\tag{31}
$$

本节 `B_L^*` 与第 2 节投影 `B_L` 是不同对象。每个矩阵条目是二次以内的整数多项式。所以

$$
D_{Q,L}(\psi)=\det A_L(\psi),\qquad
D_{T,L}(\psi)=\det B_L^*(\psi)
\tag{32}
$$

次数至多 `2r≤16`，并具有整数系数。两者共同的正尺度完全抵消。

实现用 `ψ=0,1,…,2r` 处的整数 Bareiss 行列式值做 Newton 插值。次数上界使 `2r+1` 个精确值唯一确定整个多项式；超出参数域的插值点只用于代数求系数，不作统计判断。Gram 条目为整数多项式，所以最终系数也是整数；实现检查 Bareiss 除法与插值的精确性。对两条行列式系数一起除共同正 gcd，不改变比值或接受不等式。

对每个实数 `ψ`，式 (31) 仍然是正半定 Gram 矩阵，且 `0≼A_L≼B_L^*`。若 `D_T,L(ψ)=0`，则存在非零向量 `v` 使 `v'B_L^*v=0`；从半正定顺序得到 `v'A_Lv=0`，所以 `A_L` 也奇异，`D_Q,L(ψ)=0`。后续交叉相乘因此在这样的候选处保留两个零，没有将未定义的 ratio 误当作失败证据。

### 7.1 奇异的选列协方差

如果 `Σ_J` 奇异，则所有去均值时间行都位于其固定的协方差支持空间，维数小于 `r`。这是因为整个样本的噪声均在该支持内，而 `M_n` 消去候选创新的常数均值。因此任意 `ψ` 下的 `W_T(ψ)`、`W_Q(ψ)` 均奇异，两条 determinant 多项式恒为零。

当前规则检测 `D_T,L` 是否恒零，并核验这时 `D_Q,L` 也恒零；该尺度返回整个 `[0,1]`。若所有尺度都如此，整体集合为 `[0,1]`，均值认证不拒绝。没有挑选估计的独立列、没有用样本数值秩更换式 (20) 的自由度，也没有把 pseudodeterminant 代入满维 Beta law。

对正定 `Σ_J`，在真实参数处 `W_T(φ)` 几乎处处正定，故恒零 fallback 是零概率的退化例外。把该例外也处理为无信息只会保守。整体 `Σ` 可以奇异而选定 `Σ_J` 正定，此时式 (20) 仍然正常使用。

这些结论针对精确数据与数学模型。舍入可能破坏一个实数线性依赖，因而不能凭 float 上的非零行列式宣布原始实数协方差正定；第 12 节说明这个范围限制。

## 8. 多尺度交集、覆盖与连续外包

对可用且不触发恒零 fallback 的尺度定义

$$
g_{L,-}(\psi)=D_{Q,L}(\psi)-\ell_L D_{T,L}(\psi),
\qquad
g_{L,+}(\psi)=u_L D_{T,L}(\psi)-D_{Q,L}(\psi),
\tag{33}
$$

$$
C_L=\{\psi\in[0,1]:g_{L,-}(\psi)\ge0,
\ g_{L,+}(\psi)\ge0\}.
\tag{34}
$$

不可用或恒零的尺度定义 `C_L=[0,1]`。在真实参数处，式 (30) 或 fallback 给

$$
P\{\phi\notin C_L\}\le\beta/S.
$$

令

$$
C_J=\bigcap_{a=1}^S C_{L_a}.
$$

并集上界给

$$
P\{\phi\notin C_J\}
\le\sum_{a=1}^S P\{\phi\notin C_{L_a}\}\le\beta.
\tag{35}
$$

不同尺度来自同一样本，通常高度相关。式 (35) 不要求它们独立。若有不可用尺度，实际求和更小，但仍沿用原预算。

集合不必是一个区间，也不能假定 determinant curve 单调。计算从 `[0,1]` 开始，对全部接受多项式作 Bernstein 认证。

对区间 `I=[l,h]`，令 `w=(ψ−l)/(h−l)`。次数不超过 `d` 的多项式可精确写为

$$
p(\psi)=\sum_{k=0}^d b_k\binom dk
w^k(1-w)^{d-k}.
\tag{36}
$$

若 `p(l+(h−l)w)=Σ_i a_i w^i`，则

$$
b_k=\sum_{i=0}^k a_i
\frac{\binom ki}{\binom di}.
\tag{37}
$$

式 (37) 由把每个 `w^i` 展开到次数 `d` 的 Bernstein 基得到。基函数在 `[0,1]` 上非负且和为 1，所以

$$
\min_k b_k\le p(\psi)\le\max_k b_k,
\qquad \psi\in I.
\tag{38}
$$

所有系数和 dyadic 端点为有理数，清除正分母后用整数符号比较。若至少一条接受多项式的全部 Bernstein 系数严格为负，排除整个区间；若全部接受多项式的全部系数非负，保留整个区间；其余二分。到 `ci_depth=16` 仍未定的区间全部保留。相邻保留区间可以合并，不会填补被排除的正长度空隙。

记最终区间并为 `C̄_J`。每次排除都由式 (38) 证明该区间没有任何 `C_J` 中的点，因此

$$
C_J\subseteq\overline C_J,
\qquad P\{\phi\in\overline C_J\}\ge1-\beta.
\tag{39}
$$

未定区间、退化候选和有限深度都只扩大集合。空集合时全部不拒绝，避免使用“空集上的所有条件恒真”来自动宣布显著。不得只保留点估计所在的一段，或用通过的网格点替代连续外包。

一个直接推论是 `P(C̄_J=∅)≤β`：空集合必然没有覆盖真实参数。因此，在本节给定的理想模型与预设规则下，空集合也可作为水平至多 `β` 的模型冲突信号。它仍使均值检验停止拒绝；非空集合并不能证明模型正确，更不能把模型外非空路径上的均值检验解释为已经校准。

## 9. 已知时间参数的 GLS 与有理临界值

本节使用每列全部 `T` 个观测，包括没有用于某个形状尺度的末尾数据。对一列非恒定 `x=(x₁,…,x_T)'`，令 `ν=T−1`，并定义

$$
a_x(\psi)=x_1+x_T+(1-\psi)\sum_{t=2}^{T-1}x_t,
\qquad d(\psi)=T-(T-2)\psi,
$$

$$
Q_x(\psi)=x_1^2+x_T^2+(1+\psi^2)
\sum_{t=2}^{T-1}x_t^2
-2\psi\sum_{t=2}^T x_tx_{t-1},
$$

$$
H_x(\psi)=d(\psi)Q_x(\psi)-(1-\psi)a_x(\psi)^2.
\tag{40}
$$

对 `0≤ψ<1`，AR 相关矩阵 `C_ψ` 的精度矩阵 `W_ψ=(1−ψ²)C_ψ⁻¹` 为三对角矩阵：两端对角为 1，内部对角为 `1+ψ²`，相邻元素为 `−ψ`。直接求和给

$$
\mathbf1'W_\psi\mathbf1=(1-\psi)d(\psi),\quad
\mathbf1'W_\psi x=(1-\psi)a_x(\psi),\quad
x'W_\psi x=Q_x(\psi).
\tag{41}
$$

GLS 均值为 `a_x/d`，其 t 为

$$
t_x(\psi)=\frac{\sqrt\nu\,a_x(\psi)\sqrt{1-\psi}}
{\sqrt{H_x(\psi)}}.
\tag{42}
$$

非恒定 `x` 下 `H_x(ψ)>0`：正定矩阵的 Cauchy–Schwarz 给
`(1'W1)(x'Wx)−(1'Wx)²>0`，其值为 `(1−ψ)H_x(ψ)`。

在真实 `ψ=φ`，确定性的 Gaussian whitening 后，常数均值方向的一维分量与其 `T−1` 维正交残差独立。因此零均值下 `t_x(φ)∼t_ν`。对路径加常数 `b`，

$$
a_{x+b\mathbf1}=a_x+bd,\qquad H_{x+b\mathbf1}=H_x.
\tag{43}
$$

`b≤0` 只降低正尾。于是 `μ_j≤0` 的最不利边界为零，不受其他列的均值或相关影响。

Student 临界值沿用上一轮的精确外向构造。令 `ν₀=2m` 为不超过 `ν` 的最大偶数。对 `v=c/√(ν₀+c²)`，

$$
P(t_{\nu_0}>c)=\frac12-
\frac{m\binom{2m}{m}}{4^m}
\sum_{k=0}^{m-1}
\frac{(-1)^k\binom{m-1}{k}v^{2k+1}}{2k+1}.
\tag{44}
$$

这是把 Student 密度用 `x=√ν₀ v/√(1−v²)` 代换后，对 `(1−v²)^(m−1)` 展开积分的有限表达。用精确有理运算选择一个 dyadic 上侧 `v̄`，使式 (44) 不超过 `η`，并存

$$
c^2=\frac{\nu_0\bar v^2}{1-\bar v^2}.
\tag{45}
$$

减少至多一个自由度在正尾上保守。简要证明为：`f_ν(0)` 随 `ν` 增加；当 `ν₂>ν₁` 时，正轴上的密度比的对数导数为

$$
\frac{d}{dx}\log\frac{f_{\nu_2}(x)}{f_{\nu_1}(x)}
=\frac{x(\nu_2-\nu_1)(1-x^2)}
{(\nu_1+x^2)(\nu_2+x^2)}.
\tag{46}
$$

该比从大于 1 开始，先升后降，最终趋于零，仅穿过 1 一次。两种密度在正半轴都积分为 `.5`，所以 `P(t_ν₂>c)≤P(t_ν₁>c)` 对全部 `c>0` 成立。零点密度的严格单调证明与式 (44) 常数核验见[单列推导 §6](parameter-uncertainty.md#6-偶自由度-student-临界值的外向构造)。由 `η<.5`，本轮临界值为正，故

$$
P_{\mu_j\le0}\{t_{X_j}(\phi)>c\}
\le P(t_\nu>c)\le P(t_{\nu_0}>c)\le\eta.
\tag{47}
$$

## 10. 连续拒绝证书与 strong FWER

定义三次以内的多项式

$$
R_{x,c}(\psi)=\nu(1-\psi)a_x(\psi)^2-c^2H_x(\psi).
\tag{48}
$$

在 `0≤ψ<1`，

$$
t_x(\psi)>c\quad\iff\quad
a_x(\psi)>0\ \text{且}\ R_{x,c}(\psi)>0.
\tag{49}
$$

先检查正的分子方向，才可以平方；遗漏它会把大的负均值误当作单侧显著。

对 `C̄_J` 的每个闭区间，使用式 (36)–(38) 认证这两条多项式严格为正。只有全部区间都完成认证，才拒绝该列。`certificate_depth=20` 是每个合并后区间的额外二分深度；`max_nodes=4096` 是每列预算。任何深度或节点耗尽、空参数集合、未完成正性认证均返回不拒绝，并记录原因。

因此每次实际拒绝第 `j` 列都蕴含

$$
t_{X_j}(\psi)>c\quad\text{对全部 }
\psi\in\overline C_J\cap[0,1).
\tag{50}
$$

设 `A={φ∈C̄_J}`。在 `A` 上，拒绝一个真实零假设列就蕴含其真实参数 t 超过 `c`。所以

$$
\{\exists j\in I_0:\text{拒绝 }j\}
\subseteq A^c\ \cup\
\bigcup_{j\in I_0}\{t_{X_j}(\phi)>c\}.
$$

由式 (39)、(47) 和并集上界，

$$
\boxed{\quad
P\{\exists j\in I_0:\text{拒绝 }j\}
\le\beta+|I_0|\eta\le\beta+K\eta=\alpha.
\quad}
\tag{51}
$$

均值 t 与参数集合用了同一样本，没有假定它们独立；不同尺度和不同候选也没有独立性要求。`β` 只有一个共同覆盖失败事件，因此只支付一次。这是共享 nuisance confidence set 的 Berger–Boos/投影预算在本模型中的具体使用。有限计算预算只能撤回拒绝，不能破坏包含关系。

式 (51) 为预先给定的每列提供同时结论，而不只是检验“所有均值都为零”后报告胜出列。但它不授权依据本次数据扩充候选库或修改已冻结的方法。

## 11. 不变性、持久性端点与功效限制

对选定数据作逐列平移和非零尺度变换 `X_J↦X_JD+1b'`，候选创新为 `Z(ψ)D+(1−ψ)1b'`。投影消去最后一项，两组 scatter 都变为 `D'WD`，行列式同乘 `det(D)²`。所以整条 Wilks 接受集合不依赖选列的均值与边际尺度。保持列集合的任意可逆线性重参数化也保持数学上的 determinant ratio；实现仍按固定原列计算。

因此选列中含有均值信号不会破坏参数覆盖，也不会仅因为信号加大而给时间形状增加信息。整个均值拒绝规则则只对正尺度不变；均值平移本来会改变式 (4) 的推断对象。

虽然模型只允许 `φ<1`，计算域保留闭包端点 `ψ=1`。没有在该点定义平稳 Gaussian 初值分布。对固定非恒定数据，

$$
H_x(1)=2\sum_{t=2}^T(x_t-x_{t-1})^2>0,
\qquad t_x(\psi)\longrightarrow0\quad(\psi\uparrow1).
\tag{52}
$$

对应的端点多项式为 `R_x,c(1)=−c²H_x(1)<0`。所以 `1∈C̄_J` 时，没有任何列可以通过连续正性证书。即使只看 `[0,1)`，一串仍被保留、趋近 1 的候选也会阻止严格单侧拒绝。

Wilks 参数集合对所有均值平移不变，故对任意固定均值向量，

$$
P_\mu\{\text{至少一列被拒绝}\}
\le P_\mu\{1\notin\overline C_J\}
=P_0\{1\notin\overline C_J\}.
\tag{53}
$$

这是本轮构造自身的功效上限，适用于任意信号强度；不是所有未知依赖均值检验的不可能性定理，也未证明上限一定可达。多个尺度、多列和更紧的临界值可能改变右侧概率，但是否改善应由冻结实验回答。

新增尺度一方面提供不同时间对比，另一方面将每个尺度的预算从 `β` 缩为 `β/S`。因此不能仅凭“交集使用了更多信息”保证集合更窄或功效更高。增加选列维数也改变精确零假设分布与临界值；选列数越大并不自动意味着本规则越有效。

## 12. 数学分布、机器证书与研究评价

本方法有两个层次。式 (20)、(35)、(47)、(51) 针对理想 Gaussian 模型与相应数学规则；行列式、dyadic 根和 Bernstein 的有理计算认证的是“对收到的这份 binary float 输入，整个区间上成立的代数符号”。将 float 解释成精确有理数，不会自动控制观测舍入或随机数生成器相对理想实数分布的误差。

尤其在奇异协方差下，非 dyadic 的真实线性依赖可能被舍入打破。这一限制不能通过把更精确的行列式叫作精确分布来消除。若要求连观测舍入都得到有限样本保证，需要对输入误差做区间外包或明确定义离散观测模型；本轮没有完成这一层。非有限输入或恒定列在公共校验时报告错误，不删去失败列后重定义候选族。

本接口返回固定水平的逐列决策、参数区间与计算证据；没有返回已校准的任意水平 p 值，也没有可选停止保证。理论不覆盖不同列拥有不同 `φ`、因子与个体不同持久性、条件异方差、重尾分布、候选选择后扩库或看结果后调整规则。

评价应在同一批观测上分开记录：参数覆盖、所有真实零假设的误拒、预定信号列的检出、端点保留、集合外包宽度、空集合、奇异 fallback 与未完成证书数。信号列功效不能用“至少有一列被拒绝”替代，因为后者也包括误拒零假设。

与已知参数 GLS 的比较应同时保留标准 `α/K` 临界值和本轮共同预算下的临界值，以分开观察预算/Student 外向临界值的成本与参数集合/认证的成本。覆盖失败路径也必须保留；两个方法只在参数被覆盖时有逐路径包含关系。配对差异应用同路径拒绝差记录，不能根据两条独立区间是否重叠判断显著差异。

Chernoff 外向界、预定幂集合、块数与列数上限、多尺度 Bonferroni、Student 自由度、参数区间外包、计算预算都可能保守。不能因为误报很小便宣称普遍有效，更不能把开发阶段试过的选择当作事前冻结的确认性研究。

## 13. 经典前例、近期研究与贡献范围

[Dufour–Khalaf (2002), *Simulation Based Finite and Large Sample Tests in Multivariate Regressions*](https://www2.cirano.qc.ca/~dufourj/Web_Site/Dufour_Khalaf_1996_MLR_W.pdf)，*Journal of Econometrics* 111(2), 303–322，DOI `10.1016/S0304-4076(02)00108-2`。共同固定设计、满秩横截面尺度的多元回归中，Theorem 3.1 和 Corollaries 3.2–3.3 给 determinant criteria 的 nuisance invariance，Appendix A.1 给精确 Beta 乘积。它是本轮联合枢轴的直接经典前例；AR 候选创新使真参数下的固定时间对比进入这个 Gaussian 结构。

[Genest–Ouimet–Richards (2024), *On Wilks’ joint moment formulas for embedded principal minors of Wishart random matrices*](https://onlinelibrary.wiley.com/doi/10.1002/sta4.706)，*Stat* 13(2), e706，online 2024-06-12。Lemma 1 和 Theorem 1 用 Schur complements 重新证明旧的 Wishart principal-minor 矩公式；条件为正定尺度、自由度大于维数减一，定理中的幂非负。它为经典矩结构提供近期数学对照，不能被引用为本轮奇异 population covariance 或任意负矩的现成定理。这里的负矩存在条件由式 (21) 独立核验。

[Wasserman–Ramdas–Balakrishnan (2020), *Universal Inference*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/)，*PNAS* 117(29), 16880–16890。Theorem 1 给 proper-density likelihood-ratio confidence set 的有限样本保证，§6 讨论 nuisance profiling。它支持研究阶段的 proper Gaussian likelihood mixture 路线；但有效性不意味着信息效率，随机滞后设计下也必须证明 numerator 是完整路径上的正规条件密度。当前发布接口没有使用该 mixture。

[Vovk–Wang (2021), *E-values: calibration, combination and applications*](https://arxiv.org/abs/1912.06116)，*Annals of Statistics* 49(3), 1736–1754，及 [Wang (2025), *The only admissible way of merging arbitrary e-values*](https://academic.oup.com/biomet/article/112/2/asaf020/8086785)，*Biometrika* 112(2), asaf020，online 2025-03-17。后者 Theorem 1 表明，在必须适用于任意 e-value 联合分布的类中，admissible merger 是输入与常数 1 的固定加权平均。这不禁止 Gaussian 结构中的联合 Wishart 枢轴：本方法直接使用原始矩阵的模型结构，并非只合并任意边际 e-values。它也不证明某个列平均在本 AR 模型中最优。

[Takatsu (2025), *On the Precise Asymptotics of Universal Inference*](https://arxiv.org/abs/2503.14717)，arXiv 预印本，2025-03-18。其 IID regular working model 的分析使用 quadratic mean differentiability、局部 likelihood 展开、初始估计一致性与 uniform CLT 条件，说明有限样本有效集合仍可能在渐近上极其保守；studentization/bias correction 的修正追求渐近精确覆盖。它为关注集合宽度与功效提供近期依据，不能直接授予共同 AR、近单位根样本或本轮机器认证新的分布保证。

连续 nuisance 集合的投影预算已有 [Dufour (1990)](https://jeanmariedufour.research.mcgill.ca/Dufour_1990_Econometrica_ExactAR1.pdf) 与 [Berger–Boos (1994)](https://doi.org/10.1080/01621459.1994.10476836) 的前例。近期 [Glazer–Stark (2026), *Fast Conservative Monte Carlo Confidence Sets*](https://www.tandfonline.com/doi/full/10.1080/10618600.2025.2526416) 讨论保守反演与快速计算；其适用搜索条件不能替代本轮非单调 determinant 曲线上的连续证书。本轮临界值本身没有 Monte Carlo 误差。

本项目可以主张的是一个范围清楚、可审计的组合：固定多尺度块对比与固定多列的联合形状集合；精确矩配合数据无关幂搜索及外向根；degree-16 以内的 determinant 多项式与未定即保留的连续外包；共享覆盖预算下的逐列 GLS strong-FWER 证明；以冻结配对实验揭示可靠性、参数信息与检出能力之间的具体成本。Wilks law、矩界、置信集合反演、Bernstein 包络及 Berger–Boos 原理都应明确归于已有方法。它没有证明普遍最优性，也不能声称首次解决未知依赖下的金融显著性。

完整书目信息见 [`references.bib`](references.bib)。正式结果和结论应另文报告，并以冻结协议、保存原始记录和独立审计为依据。
