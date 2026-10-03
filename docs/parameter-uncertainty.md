# 未知时间参数下的有限样本检验

推导记录，2026-10-03。本轮研究一个比参数重放更窄、保证更明确的问题：候选序列确实服从共同时间参数的平稳 Gaussian AR(1)，但时间参数、每列均值、每列尺度和候选之间的相关均未知。先构造时间参数的有限样本置信集合，再在它的整个连续外包上认证均值检验。计算不能完成认证时，不拒绝。

这条推断路线已有明确前例。Dufour 的 AR 误差回归置信集合、Berger–Boos 的 nuisance confidence set 和 Bonferroni 同时检验是方法基础。本项目的工作是选择可复核的投影和临界值，给连续参数计算提供单向保守的有理证书，并量化它为可靠性付出的功效代价。这里没有提出新的置信集合反演原理，也没有建立一般金融收益分布下的有效性。

实现位于 [`uncertainty.py`](../src/strategy_inference/uncertainty.py)。本说明给数学保证与计算证书；实验结果应以另行冻结的协议和保存数据为依据。

## 1. 模型、假设与错误概率预算

有 `T` 期、事先给定的 `K` 列候选。假设

\[
X_t=\mu+U_t,\qquad
U_t=\phi U_{t-1}+\varepsilon_t,\qquad 0\le\phi<1,
\tag{1}
\]

\[
U_1\sim N_K(0,\Sigma),\qquad
\varepsilon_2,\ldots,\varepsilon_T
\overset{\mathrm{iid}}\sim N_K(0,(1-\phi^2)\Sigma),
\tag{2}
\]

其中初值与后续创新独立，`Σ` 是任意半正定矩阵，且 `Σ_jj>0`。因此

\[
\operatorname{Cov}(X_s,X_t)=\phi^{|s-t|}\Sigma.
\tag{3}
\]

允许负的列间相关、不同的边际尺度和奇异的列间协方差；不估计 `Σ`，不要求等相关。这里“任意相关”指式 (3) 内任意 `Σ`，没有允许任意跨时间协方差或不同列拥有不同的时间参数。

每列检验

\[
H_{0j}:\mu_j\le0\quad\text{对}\quad H_{1j}:\mu_j>0.
\tag{4}
\]

令 `I₀={j:μ_j≤0}`，目标是 strong family-wise error rate：

\[
P\{\text{至少拒绝一个 }j\in I_0\}\le\alpha,
\tag{5}
\]

无论其余列是否有信号。有限 `T`、任意有限 `K` 都是目标范围，不需要 `K` 固定的渐近近似。

固定 `0<β<α<1/2`，取

\[
r=\frac{\alpha-\beta}{K}.
\tag{6}
\]

本轮默认 `α=.05`、`β=.005`；`β` 是全族共同的参数覆盖预算，只收取一次。每列均值尾概率用 `r` 控制。参考列固定为第 0 列，不能根据本次数据中的胜出候选更换。

## 2. 参考列的均值与尺度不变形状统计量

只用第 0 列构造 `φ` 的置信集合。它可以有正均值；覆盖证明不假定参考列的均值为零。

### 2.1 固定的整数投影

令 `n` 为不超过 `T−1` 的最大奇数，只取前 `n` 个创新。数学上需要 `n≥5`，以便两组残差自由度都严格为正；当前接口沿用公共输入校验，要求 `T≥8`。取

\[
q=2\left\lfloor n^{1/3}/2+1/2\right\rfloor,
\qquad q\text{ 截到 }[2,n-3],\qquad s=n-1-q.
\tag{7}
\]

`q` 与 `s` 均为正偶数。省去至多一个末尾创新、采用偶数投影秩，是为使用下一节的有限多项式分布计算；这会损失信息，不是无代价的优化。

对 `k=1,…,q`、`i=0,…,n−1`，定义整数矩阵

\[
B^{\rm raw}_{ki}
=\operatorname{round}\{2^{16}\cos[\pi(i+1/2)k/n]\},
\]

\[
B_{ki}=nB^{\rm raw}_{ki}-\sum_{v=0}^{n-1}B^{\rm raw}_{kv}.
\tag{8}
\]

每一行的和精确为零。核验 `rank(B)=q` 后，令

\[
M_n=I_n-\mathbf1\mathbf1'/n,\qquad
P=B'(BB')^{-1}B,\qquad Q=M_n-P.
\tag{9}
\]

这些矩阵为有理数矩阵。直接代入给 `P'=P`、`P²=P`、`P1=0`，所以 `MP=PM=P`。由此 `Q'=Q`、`Q²=Q`、`PQ=0`，两者的秩分别为 `q`、`s`。

式 (8) 只是低频 cosine 子空间的一个固定近似。没有把舍入后的 cosine 向量当作精确正交 DCT；式 (9) 的 Gram 投影才使它成为精确的正交子空间。若 Gram 矩阵不满秩，程序必须报告失效，不能继续套用错误的 F 自由度。数据分布的结论只依赖上述投影性质，不依赖近似 DCT 的误差很小。

### 2.2 F 分布的有限样本证明

对任意候选 `ψ∈[0,1]`，参考列构造

\[
z_i(\psi)=X_{i+2,0}-\psi X_{i+1,0},\qquad i=0,\ldots,n-1,
\]

\[
L(\psi)=z(\psi)'Pz(\psi),\qquad
H_{\rm ci}(\psi)=z(\psi)'Qz(\psi),
\]

\[
F(\psi)=\frac{L(\psi)/q}{H_{\rm ci}(\psi)/s}.
\tag{10}
\]

在真实值 `ψ=φ`，

\[
z(\phi)=(1-\phi)\mu_0\mathbf1+\tau\xi,
\quad \tau^2=(1-\phi^2)\Sigma_{00}>0,
\quad \xi\sim N_n(0,I_n).
\tag{11}
\]

`P1=Q1=0` 消去未知均值。取正交坐标分别张成 `P`、`Q` 的像空间与常数方向，则 `Pξ` 和 `Qξ` 的坐标是互相独立的标准 Gaussian。因此

\[
L(\phi)/\tau^2\sim\chi_q^2,\qquad
H_{\rm ci}(\phi)/\tau^2\sim\chi_s^2,
\qquad F(\phi)\sim F_{q,s}.
\tag{12}
\]

分母在真实值处几乎处处为正。均值、尺度、时间参数本身及列间相关都不进入这个零假设分布。这个 F 结论来自候选参数下的确定性投影，不是对估计 AR 回归中的随机滞后解释变量套用普通回归 t 分布。

此外，对任何常数 `b`、正数 `a`，`X_0↦aX_0+b1` 给 `z(ψ)↦az(ψ)+(1−ψ)b1`，所以 `L`、`H_ci` 同乘 `a²`，整条 `F(ψ)` 曲线保持不变。参考列的信号不会破坏覆盖，也不会凭均值变大而使参数集合缩小。

## 3. 不使用 Monte Carlo 的外向临界值

令 `A=q/2`、`D=s/2`，均为正整数。独立 chi-square 的比值给

\[
V=\frac{L(\phi)}{L(\phi)+H_{\rm ci}(\phi)}
\sim\operatorname{Beta}(A,D),\qquad
F=\frac{sV}{q(1-V)}.
\tag{13}
\]

整数形状的 Beta CDF 有有限多项式表达：

\[
I_v(A,D)=\sum_{k=A}^{A+D-1}
\binom{A+D-1}{k}v^k(1-v)^{A+D-1-k}.
\tag{14}
\]

**证明。** 右侧是 `Binomial(A+D−1,v)` 的右尾。逐项求导后相邻项抵消，剩下 `v^(A−1)(1−v)^(D−1)/B(A,D)`；两边在 `v=0` 都为零，所以相等。

在 dyadic 点 `v=m/2^b`，式 (14) 可以用整数运算比较 CDF 与给定有理概率，不需要浮点特殊函数。40 位二分始终维护包含真实分位数的区间：低尾取满足 `I_ℓ(A,D)≤β/2` 的下侧端点，高尾取满足 `I_u(A,D)≥1−β/2` 的上侧端点。于是

\[
P(V<\ell)\le\beta/2,\qquad
P(V>u)\le\beta/2.
\tag{15}
\]

令

\[
f_- =\frac{s\ell}{q(1-\ell)},\qquad
f_+ =\frac{su}{q(1-u)}.
\tag{16}
\]

这两个有理临界值向外移动：低临界值不高于真实低分位数，高临界值不低于真实高分位数。有限位数只扩大接受集合，不是把分位数近似误差当成零。若给定概率低于可表示分辨率，端点可能使集合非常宽；实现必须保持外向含义或报告无法构造，不能偷偷改成内向截断。

## 4. 连续参数置信集合与有理外包

定义两条二次多项式

\[
g_-(\psi)=sL(\psi)-qf_-H_{\rm ci}(\psi),
\qquad
g_+(\psi)=qf_+H_{\rm ci}(\psi)-sL(\psi).
\tag{17}
\]

`z(ψ)` 对 `ψ` 为线性，两个能量都是二次多项式。定义闭集合

\[
C(X_0)=\{\psi\in[0,1]:g_-(\psi)\ge0,\ g_+(\psi)\ge0\}.
\tag{18}
\]

在 `H_ci>0` 处，这正是 `f_-≤F(ψ)≤f_+`。如果两种能量同时为零，两条不等式也为零；保留这样的退化候选。这不改变真实参数处的覆盖，因为该退化事件在式 (12) 下概率为零。

式 (15) 给

\[
P\{\phi\in C(X_0)\}\ge1-\beta.
\tag{19}
\]

虽然 `φ=1` 不属于平稳模型，计算域保留其闭包端点，以便严格处理靠近 1 的候选；这里没有在 1 处赋予一个平稳 Gaussian 初值分布。

### 4.1 Bernstein 排除证书

对区间 `I=[l,h]` 和次数不超过 `d` 的多项式 `p`，设 `w=(ψ−l)/(h−l)`，精确写成

\[
p(\psi)=\sum_{k=0}^{d}b_k
\binom dk w^k(1-w)^{d-k}.
\tag{20}
\]

`0≤w≤1` 时，这些基函数非负且和为 1，因此

\[
\min_k b_k\le p(\psi)\le\max_k b_k
\quad\text{对全部 }\psi\in I.
\tag{21}
\]

例如二次多项式的系数是 `b₀=p(l)`、`b₁=p(l)+(h−l)p'(l)/2`、`b₂=p(h)`。本轮系数与区间端点均为有理数，可以清除正分母后用整数比较其符号。

从 `[0,1]` 开始：若 `g_-` 或 `g_+` 的所有 Bernstein 系数严格为负，整个区间排除；若两条的所有系数都非负，整个区间保留；其余二分。到 `ci_depth=16` 仍未确定的区间全部保留。记保留下来的有限闭区间之并为 `C̄(X₀)`。

**外包证明。** 每一次排除都有式 (21) 证实至少一条接受不等式在整个区间严格失败。它不能删去任何 `C` 中的点。保留、二分、到限保留均不改变这个事实。因此

\[
C(X_0)\subseteq\overline C(X_0),\qquad
P\{\phi\in\overline C(X_0)\}\ge1-\beta.
\tag{22}
\]

集合可以不连通；不能只保留包含点估计的那一段，也不能用网格上的通过点代替外包。若外包为空，本轮约定全部不拒绝，防止程序用“空集上的条件恒真”自动宣布显著。这个约定进一步保守，并不需要假定空集合不可能发生。

## 5. 已知候选时间参数的 GLS t

以下逐列使用全部 `T` 个观测。对一列 `x=(x₁,…,x_T)'`，记 `ν=T−1`。候选平稳相关矩阵为 `C_ψ=(ψ^|v−w|)`。对 `0≤ψ<1`，其正定精度矩阵满足

\[
W_\psi=(1-\psi^2)C_\psi^{-1}
=\begin{pmatrix}
1&-\psi&&\\
-\psi&1+\psi^2&-\psi&\\
&\ddots&\ddots&\ddots\\
&&-\psi&1
\end{pmatrix}.
\tag{23}
\]

记

\[
a_x(\psi)=x_1+x_T+(1-\psi)\sum_{t=2}^{T-1}x_t,
\qquad d(\psi)=T-(T-2)\psi,
\]

\[
Q_x(\psi)=x_1^2+x_T^2+(1+\psi^2)
\sum_{t=2}^{T-1}x_t^2
-2\psi\sum_{t=2}^Tx_tx_{t-1},
\]

\[
H_x(\psi)=d(\psi)Q_x(\psi)-(1-\psi)a_x(\psi)^2.
\tag{24}
\]

从 tridiagonal 矩阵逐行求和得到

\[
\mathbf1'W_\psi\mathbf1=(1-\psi)d(\psi),
\quad \mathbf1'W_\psi x=(1-\psi)a_x(\psi),
\quad x'W_\psi x=Q_x(\psi).
\tag{25}
\]

因此 GLS 均值估计为 `a_x/d`，检验均值零的 t 统计量化简为

\[
t_x(\psi)=\frac{\sqrt\nu\,a_x(\psi)\sqrt{1-\psi}}
{\sqrt{H_x(\psi)}}.
\tag{26}
\]

非恒定 `x` 下 `H_x(ψ)>0`：由 `W_ψ` 正定及 Cauchy–Schwarz，`(1'W1)(x'Wx)−(1'Wx)²>0`，等号只会在 `x` 与 `1` 共线时发生；式 (25) 显示这个量等于 `(1−ψ)H_x(ψ)`。

### 5.1 精确 t 分布与复合单侧零假设

在真实 `ψ=φ` 下，确定性的创新 whitening 把 `x` 变成均值为 `μ_j u_φ`、协方差为某个正尺度乘 `I_T` 的 Gaussian 向量，其中 `u_φ` 是被 whitening 的常数方向。投影到 `u_φ` 的一维分量与其正交补的 `T−1` 个 Gaussian 分量独立。因此在 `μ_j=0` 时

\[
t_{X_j}(\phi)\sim t_{T-1}.
\tag{27}
\]

对同一噪声路径加入常数 `b`，式 (24) 直接给

\[
a_{x+b\mathbf1}=a_x+bd,
\qquad H_{x+b\mathbf1}=H_x.
\tag{28}
\]

因为 `d(φ)>0`，`b≤0` 只降低式 (26) 的分子。所以真实零假设 `μ_j≤0` 下的正尾不大于中心 t 正尾；零均值是该单列检验的最不利边界。这一论证允许其余列有任意均值和任意式 (3) 内的相关。

## 6. 偶自由度 Student 临界值的外向构造

令 `ν₀` 为不超过 `T−1` 的最大偶数，`ν₀=2m`。只少至多一个自由度。对正临界值 `c`，令

\[
v=\frac{c}{\sqrt{\nu_0+c^2}}\in(0,1),
\qquad C_m=\frac{m\binom{2m}{m}}{4^m}.
\tag{29}
\]

Student 正尾可写成有限多项式：

\[
\Pr(t_{\nu_0}>c)
=\frac12-C_m\sum_{k=0}^{m-1}
\frac{(-1)^k\binom{m-1}{k}v^{2k+1}}{2k+1}.
\tag{30}
\]

**常数与积分证明。** 代换 `x=√ν₀ v/√(1−v²)`，Student 密度乘 Jacobian 化为
`Γ(m+1/2)/(√πΓ(m))·(1−v²)^(m−1)`。半整数 Gamma 公式给该常数正是 `C_m`。在 `[0,v]` 展开二项式并积分，从 `.5` 减去这段面积，得到式 (30)。其导数为 `−C_m(1−v²)^(m−1)<0`，所以 dyadic 二分可以始终从上侧夹住目标 `v`。

选择 dyadic 上侧端点 `v̄`，通过精确有理计算证实式 (30) 不超过 `r`，再令

\[
c^2=\frac{\nu_0\bar v^2}{1-\bar v^2}.
\tag{31}
\]

只存储有理的 `c²`，后面不需要求其平方根。交替多项式在普通浮点数中可能严重抵消；证书依赖精确有理比较，不能换成浮点逐项求和后继续声称外向保证。

### 6.1 为什么减少自由度保守

对 `ν₂>ν₁>0`，两种 Student 密度的比值在零处大于 1。这由
`f_ν(0)=1/[√ν B(ν/2,1/2)]` 严格递增可知：令 `a=ν/2`，则

\[
\sqrt a\,B(a,1/2)
=\int_0^\infty e^{-u}
\{a(1-e^{-u/a})\}^{-1/2}\,du.
\tag{32}
\]

对固定 `u>0`，分母中的 `a(1−e^(−u/a))` 随 `a` 严格增加，因为导数为 `1−e^(−u/a)(1+u/a)>0`；积分严格下降，故零处密度严格增加。

对 `x>0`，密度比值的对数导数为

\[
\frac{d}{dx}\log\frac{f_{\nu_2}(x)}{f_{\nu_1}(x)}
=\frac{x(\nu_2-\nu_1)(1-x^2)}
{(\nu_1+x^2)(\nu_2+x^2)}.
\tag{33}
\]

比值在 `(0,1)` 增加、在 `(1,∞)` 下降，最后趋于零；所以正轴上只穿过 1 一次。两种密度在正半轴各积分为 `.5`，因此 `F_ν₂(x)−F_ν₁(x)` 先增加后减至零，始终为正。于是对 `c>0`，

\[
\Pr(t_{T-1}>c)\le\Pr(t_{\nu_0}>c)\le r.
\tag{34}
\]

这个尾排序只在正临界值方向成立；负临界值的方向相反。式 (6) 的 `r<.5` 保证本轮使用正临界值。

## 7. 均值拒绝的连续域证书

对每一列定义三次以内的多项式

\[
R_{x,c}(\psi)
=\nu(1-\psi)a_x(\psi)^2-c^2H_x(\psi).
\tag{35}
\]

因为 `a_x` 为线性、`H_x` 为三次以内，所有系数都可从输入和式 (31) 用有理运算得到。对非恒定数据、`0≤ψ<1`，

\[
t_x(\psi)>c
\iff a_x(\psi)>0\ \text{且}\ R_{x,c}(\psi)>0.
\tag{36}
\]

首先要求 `a_x>0` 保留单侧方向，然后才能平方不等式；没有这个符号检查，大的负均值也可能被错误拒绝。

在 `C̄` 的每个闭区间上，对 `a_x` 和 `R_x,c` 应用式 (20)–(21)。只有全部 Bernstein 系数严格为正时，才认证整个区间上的严格正性；否则进一步二分。拒绝第 `j` 列要求它在外包的**全部**区间上均完成认证。`certificate_depth=20` 计每个合并后外包区间的额外二分层数；到此限或每列 `max_nodes=4096` 仍未完成，返回不拒绝。恒定列、非有限输入或不能通过样本方差校验的数据会在计算前报错，没有删掉失效列后重定义候选族。

因此每一次实际拒绝都蕴含

\[
t_{X_j}(\psi)>c\quad\text{对全部 }
\psi\in\overline C\cap[0,1).
\tag{37}
\]

认证失败不等于数学上的最大 p 值已经大于阈值。它可能只是预算不足、区间外包较宽或 Bernstein 界较松。应分别记录不拒绝的原因，不把数值保守性当成数据证据。

### 7.1 strong FWER 证明

设 `A={φ∈C̄}`。在 `A` 上，拒绝任何真实零假设列都蕴含 `t_j(φ)>c`。所以

\[
\{\exists j\in I_0:\text{拒绝 }j\}
\subseteq A^c\ \cup\
\bigcup_{j\in I_0}\{t_{X_j}(\phi)>c\}.
\]

用式 (22)、(28)、(34) 和并集上界，

\[
\begin{aligned}
P\{\exists j\in I_0:\text{拒绝 }j\}
&\le\beta+|I_0|r\\
&\le\beta+Kr=\alpha.
\end{aligned}
\tag{38}
\]

这里没有假定参考列的形状统计量与均值 t 独立，也没有假定候选之间独立。参数覆盖失败只有一个共同事件，故 `β` 只出现一次。这与把每列都加上 `β` 再直接按 `α/K` 比较，是不同的错误预算安排。

从 nuisance p 值的角度，这是一项共享置信集合的 Berger–Boos/投影构造：在可信参数集合上考虑最不利均值检验，再加上集合覆盖失败的预算。实现认证“所有候选参数都拒绝”，没有把一组网格 p 值的最大值当作连续 supremum。式 (38) 是该已有原理在本模型中的具体证明。

## 8. 持久性端点与功效上限

### 8.1 GLS t 在 `ψ↑1` 时的极限

对固定非恒定数据，由式 (24)，

\[
Q_x(1)=\sum_{t=2}^T(x_t-x_{t-1})^2>0,
\qquad d(1)=2,
\]

\[
H_x(1)=2\sum_{t=2}^T(x_t-x_{t-1})^2>0.
\tag{39}
\]

`a_x(ψ)` 的极限有限，式 (26) 的分子有 `√(1−ψ)`，所以

\[
t_x(\psi)\longrightarrow0\quad(\psi\uparrow1).
\tag{40}
\]

相应的已知候选参数单侧 p 值趋于 `.5`。式 (35) 在端点满足 `R_x,c(1)=−c²H_x(1)<0`。因此只要计算外包包含 1，本轮端点正性证书就不可能通过，任何列都不拒绝。若集合内有一串参数趋于 1，即使只考虑开域 `[0,1)`，式 (40) 也已经阻止严格单侧显著性。

### 8.2 均值信号不会自动排除持久性端点

由第 2.2 节，参考列的所有常数平移都不改变两条置信集合多项式。固定的区间细分规则因此也不改变 `C̄`。对任何均值向量 `μ`，

\[
P_\mu\{\text{至少一列被拒绝}\}
\le P_\mu\{1\notin\overline C(X_0)\}
=P_0\{1\notin\overline C(X_0)\}.
\tag{41}
\]

这是本轮检验的一个可量化功效上限，适用于任意信号强度；并未证明上限总能达到。它不依赖参照列的均值必须为零：信号再强，也不会给这个均值不变的形状统计量额外时间结构信息。

式 (41) 不是所有未知参数均值检验的一般不可能性定理。它依赖本轮参考列置信集合、闭端点外包和 GLS envelope 的具体选择。尤其不能据此宣称高持久性的数据永远不能做有意义推断。实验应报告端点保留频率、外包宽度和认证失败原因，区分参数不可排除、Bonferroni 代价与计算保守性。

## 9. 数学对象、机器证书与评价范围

`as_integer_ratio` 把已给定的 binary float 输入解释为精确有理数；随后 Gram 投影、Beta/Student 比较和 Bernstein 符号证书都在这些有理数上进行。`α`、`β` 的内部有理数也是所给 binary float 的精确值，默认十进制书写只是通常的显示形式。cosine 的浮点计算只用于定义一份数据无关的整数基，之后核查其精确投影性质。这样可以认证“针对这份已给定输入，整个参数区间上有相应符号”。

这不自动认证测量、读取或 Gaussian 生成器将理想实数数据舍入为 float 的误差。式 (12)、(27)、(38) 的分布证明针对理想 Gaussian 模型和对应的数学规则；有理证书针对收到的数值输入。若要求连原始测量舍入也得到严格控制，需要对输入区间进一步外包或直接建模离散观测，本轮没有完成这一层。

需要区分四种保守性：参数置信集合的 `β` 预算；多候选的 Bonferroni；省略少量创新/一个 Student 自由度与临界值外向舍入；区间外包和有限认证预算。这些都有明确方向，却可能叠加成较大功效损失。不能只报误报小而省略检出能力，也不能只用不受约束的名义功效说明改进。

本轮[冻结协议](../experiments/parameter-uncertainty-protocol.json)固定参考列、投影维数、`β`、临界值精度和认证预算，在同一批观测上比较四项 GLS 均值规则：已知 `φ`、按 `α/K` 取尾概率的 `gls_known`；已知 `φ`、使用本轮同一有理临界值的 `gls_known_budget`；点估计代入的 `gls_fitted`；连续集合认证的 `uncertainty`。前两个基线的差异用于观察 `β` 预算、至多一个 Student 自由度及临界值外向舍入的成本；`gls_known_budget` 与 `uncertainty` 的差异用于观察参数外包和认证的影响。后一对只在真实参数被覆盖时有逐路径的保守包含关系，覆盖失败的路径也须保留在实验中。

四项规则保持相同的 GLS 均值目标，报告预设水平下的实际尺寸、功效和配对差异。旧参数重放涉及不同的 HAC 统计量，已在[另一份协议](../experiments/parametric-replay-protocol.json)独立评价，不混入这次配对比较。本轮没有做额外零假设校准或尺寸匹配；未来若比较尺寸匹配功效，应另设独立校准样本与冻结规则。这里还保存集合外包、端点状态和未完成认证数，以便解释不拒绝的来源。

式 (38) 不覆盖异质时间参数、因子/个体不同持久性、条件异方差、重尾分布、按同一数据扩充候选库、根据已见结果修改规则或可选停止。模型错配实验可以说明失效，不能转化成已有保证。对于预先给定的列，第 (38) 式确实允许逐列结论；这比只拒绝全局假设后报告胜出列，提供了不同且更明确的推断对象。

## 10. 文献前例与本轮工作的定位

- [Dufour (1990), *Exact Tests and Confidence Sets in Linear Regressions with Autocorrelated Errors*](https://jeanmariedufour.research.mcgill.ca/Dufour_1990_Econometrica_ExactAR1.pdf)，*Econometrica* 58(2), 475–494，DOI `10.2307/2938212`。§3 的 Proposition 2 在不要求独立性的情况下，用 nuisance confidence set 与已知参数 F 检验构造保守投影；§4 对候选 AR 参数 whitening 后的残差形状检验进行反演。这已经覆盖本轮的基本统计架构。
- [Dufour–King (1991), *Optimal Invariant Tests for the Autocorrelation Coefficient in Linear Regressions with Stationary or Nonstationary AR(1) Errors*](https://jeanmariedufour.github.io/Dufour_King_1991_JE_OptInvarTestsAutorcorLinRegAR1.pdf)，*Journal of Econometrics* 47, 115–143。Theorems 1–2 给 stationary Gaussian 固定设计模型的局部最优与点最优 invariant 形状检验。本轮分组能量 F 是一个容易计算的选型，未证明它在这些意义下最优。
- [Dufour–Neifar (2002), *Méthodes d’inférence exactes pour des processus autorégressifs : une approche fondée sur des tests induits*](https://jeanmariedufour.research.mcgill.ca/Dufour_Neifar_1994_ARpInduit_W.pdf)，*L'Actualité économique* 78(1), 19–40，DOI `10.7202/007243ar`。§3–4.1 用均值未知的 Gaussian innovations quadratic ratios 反演 AR(1)，明确将接受集合写成两条二次不等式。作者 PDF 文件名早于发表年份，首页确认 2002 年发表。二次代数反演也不能作为本项目的新原理。
- [Dou (2024), *Optimal HAR Inference*](https://onlinelibrary.wiley.com/doi/full/10.3982/QE1762)，*Quantitative Economics* 15(4), 1107–1149。Theorems 3.3、4.3 在 Whittle 型 diagonal cosine covariance 与预先指定最大 persistence 的条件下讨论 scale-invariant 最优性和功效代价。§3 的 Comment 2 明确讨论原始 AR 近 ±1 时近似或 persistence 限制的边界。这不是本轮未知 `φ`、原始 AR 样本的精确有限样本定理，但为“可靠性要付多少功效成本”提供直接对照。
- [Glazer–Stark (2026), *Fast Conservative Monte Carlo Confidence Sets*](https://www.tandfonline.com/doi/full/10.1080/10618600.2025.2526416)，*JCGS* 35(1), 273–282，online 2025-08-06。§4 与 §8.2 讨论保守 Monte Carlo 反演、变化点搜索、连续界、MMC 和 Berger–Boos。它的快速 scalar bisection 要求固定数据和模拟后 p 曲线 monotone/quasiconcave；不能直接假定 AR 形状曲线满足这个条件。本轮没有使用 Monte Carlo 近似临界值。
- [Stanley et al. (2025；v2 修订于 2026-07-02), *Confidence Intervals for Functionals in Constrained Inverse Problems via Data-Adaptive Sampling-Based Calibration*](https://arxiv.org/abs/2502.02674v2)，arXiv 预印本。v2 的 Lemma 1 使用 Berger–Boos 预算；Theorem 4 在 quantile continuity、compact set 与一致估计等条件下证明计算区间的概率收敛。其已知 Gaussian 噪声 inverse problem 与未知 AR covariance shape 不同，采样近似也不能当作有限计算量下的连续域证书。

本轮可以主张的是：把已有方法落实为一个范围清晰的均值同时检验；采用经秩核验的整数 cosine surrogate 和有限多项式临界值；只以保守方向处理连续域、空集合和计算预算；给出具体端点功效上限并在新协议中评价。它是可复核的方法实现与专题研究，不应写成“首次解决未知依赖下的金融显著性”或未经证明的普遍最优检验。

完整书目信息见 [`references.bib`](references.bib)。已有参数重放和 local-unit 机制分别见[参数重放说明](parametric-replay.md)与[补尾机制说明](tail-mechanism.md)。
