# 尾部修正的有限样本机制与适用边界

本文记录**本项目推导的机制命题，原创性未建立**。矩阵迹、Gaussian 二次型、条件正态分布、Cauchy–Schwarz 与 Jensen 不等式均为基础工具；将它们组合或代入 AR(1) 不自动构成原创方法。

除最后的参数估计附录外，校正因子使用真实的 \(\phi\)，联合检验使用真实的 \(\rho\)，即 **known phi / known rho**。附录也严格保留已知 \(\rho\) 的条件。本文不证明现有 bootstrap、估计相关矩阵、增长候选数、自适应搜索或停止规则有效。这里的符号与正式实验的 Bartlett 约定一致：\(T=200\)、代码参数 \(\mathrm{lags}=4\) 对应 \(\ell=5\)；数学结论允许文中明确规定的其他 \(T,\ell\)。

本文不改变冻结计算源码、实验结果或协议。下文的数值是解析恒等式校验，不能与正式 Monte Carlo 结果混写。

## 1. 模型、带宽和三个尺度

先考虑一列 Gaussian 数据

\[
X=(X_1,\ldots,X_T)'\sim N(\mu\mathbf1,C),\qquad
C_{st}=\sigma^2\phi^{|s-t|},\quad \sigma>0,\quad |\phi|<1.
\]

这里 \(\sigma^2\) 是**平稳边际方差**。相应创新方差为 \(\sigma^2(1-\phi^2)\)，与直接用 \(\sigma^2\) 表示创新方差的文献记号不同。假定 \(T\ge2\)。后面的偏差上界与持久性结论限制为 \(0\le\phi<1\)；精确迹和条件分布恒等式允许整个 \((-1,1)\)。

令

\[
H=I-\frac{\mathbf1\mathbf1'}T,\qquad
\ell=L+1,\qquad 1\le\ell<T,
\]

\[
w_h=(1-|h|/\ell)_+,\qquad W_{st}=w_{s-t},\qquad
\widehat\Omega=\frac1T X'HWHX.
\tag{1}
\]

\(L\) 是代码的 Bartlett 滞后数。因为 \(w_\ell=0\)，非零滞后的正权重恰好覆盖 \(h=1,\ldots,L\)，而 \(\ell=L+1\)。估计的自协方差分母均为 \(T\)。

\(W\) 是正定矩阵。为检查这一点，任取非零向量 \(z\)，将三角权重写成长度 \(\ell\) 移动窗口的重叠数，得到

\[
z'Wz=\frac1\ell\sum_{r\in\mathbb Z}
\left\{\sum_{s=1}^T z_s\,1[r\le s\le r+\ell-1]\right\}^2>0.
\]

若 \(j\) 是 \(z\) 的首个非零坐标，取窗口右端为 \(j\)，其窗口和为 \(z_j\)，故至少一个平方严格为正。此外，

\[
\|W\|_{\rm op}\le\max_s\sum_t|W_{st}|\le\ell.
\tag{2}
\]

定义以下三个不同对象：

\[
\Omega=\sum_{h\in\mathbb Z}\gamma_h
=\sigma^2\frac{1+\phi}{1-\phi},
\]

\[
M_\ell=\sum_{h\in\mathbb Z}w_h\gamma_h
=\sigma^2\left\{1+2\sum_{h=1}^{\ell-1}(1-h/\ell)\phi^h\right\},
\]

\[
v_T=T\operatorname{Var}(\bar X)
=\frac{\mathbf1'C\mathbf1}{T}
=\sigma^2\left\{1+2\sum_{h=1}^{T-1}(1-h/T)\phi^h\right\}.
\tag{3}
\]

\(\Omega\) 是无限长程方差，\(M_\ell\) 是总体核加权尺度，\(v_T\) 是有限样本均值方差乘 \(T\)。样本均值本身的方差是 \(v_T/T\)。

为简化参数比值，记

\[
m_j(\phi)=1+2\sum_{h=1}^{j-1}(1-h/j)\phi^h.
\]

于是 \(M_\ell=\sigma^2m_\ell\)，\(v_T=\sigma^2m_T\)。\(M_\ell\) 恰好等于一段长度 \(\ell\) 的平稳 AR(1) 均值方差乘 \(\ell\)；这不意味着它等于长度 \(T\) 的中心化 HAC 估计量的期望。

[Liu–Chan 原文 v1](https://arxiv.org/html/2605.15596v1) Definition 3.1、Example 3.1 的总体补尾因子为

\[
\eta_{\infty,\ell}=\frac{\Omega}{M_\ell}
=\frac{\ell(1-\phi^2)}
{\ell(1-\phi^2)-2\phi+2\phi^{\ell+1}}.
\tag{4}
\]

Remark 3.2 另外提出有限样本目标因子

\[
\eta_{{\rm fs},T,\ell}=\frac{v_T}{M_\ell}.
\tag{5}
\]

式 (4) 与式 (5) 目标不同。下文称为 finite-target oracle 的尺度为

\[
\widehat v_{\rm fs}=\frac{v_T}{M_\ell}\widehat\Omega.
\tag{6}
\]

它使用真实参数，但仍是随机尺度。另一个仅用于机制诊断的 mean-unbiased oracle 是

\[
\widehat v_{\rm mu}
=\frac{v_T}{B_{T,\ell}}\widehat\Omega,\qquad
B_{T,\ell}=E\widehat\Omega.
\tag{7}
\]

式 (7) 不是本文宣称有效的新检验，也不是原论文式 (5)。

## 2. 精确期望：截断之外还有边缘与中心化

**机制命题 1：精确迹分解。**

由 \(H\mathbf1=0\) 和二次型期望公式，

\[
B_{T,\ell}=\frac1T\operatorname{tr}(HWHC).
\tag{8}
\]

证明可以逐项检查。令 \(P=\mathbf1\mathbf1'/T\)，则

\[
HWH=W-PW-WP+PWP.
\]

利用 \(C\)、\(W\) 对称和迹的循环性质，

\[
B_{T,\ell}
=\frac{\operatorname{tr}(WC)}T
-\frac{2\mathbf1'WC\mathbf1}{T^2}
+\frac{(\mathbf1'W\mathbf1)(\mathbf1'C\mathbf1)}{T^3}.
\tag{9}
\]

第一项按每个滞后的配对数展开为

\[
\frac{\operatorname{tr}(WC)}T
=\gamma_0+2\sum_{h=1}^{\ell-1}
(1-h/T)(1-h/\ell)\gamma_h.
\tag{10}
\]

即使真实均值已知、完全不中心化，有限 \(T\) 的配对数也使式 (10) 不等于 \(M_\ell\)。

令 \(s=W\mathbf1\)、\(g=C\mathbf1\)。可以写成

\[
M_\ell-B_{T,\ell}=D_{\rm pair}+D_{\rm center},
\]

\[
D_{\rm pair}
=\frac2T\sum_{h=1}^{\ell-1}h(1-h/\ell)\gamma_h,
\]

\[
D_{\rm center}
=\frac{2s'g}{T^2}
-\frac{(\mathbf1's)(\mathbf1'g)}{T^3}.
\tag{11}
\]

这些分量分别是边缘配对损失与中心化损失。总体核截断差 \(\Omega-M_\ell\) 是另一个量，不能把三者合并称为同一截断误差。

对于 \(0\le\phi<1\)，两项损失均非负。\(D_{\rm pair}\) 的符号直接来自所有求和项非负。\(D_{\rm center}\) 的符号需要证明，不能仅由 \(W,C\) 正定推出：

1. \(s_i\)、\(g_i\) 均关于样本中点对称。
2. 在中点左侧，
   \[
   s_{i+1}-s_i=w_i-w_{T-i}\ge0,\qquad
   g_{i+1}-g_i=\gamma_i-\gamma_{T-i}\ge0.
   \]
3. 因而它们按距样本中点的距离排序后共序。由
   \[
   s'g-\frac{(\mathbf1's)(\mathbf1'g)}T
   =\frac1{2T}\sum_{i,j}(s_i-s_j)(g_i-g_j)\ge0,
   \]
   得到
   \[
   D_{\rm center}\ge
   \frac{(\mathbf1's)(\mathbf1'g)}{T^3}>0.
   \tag{12}
   \]

负 \(\phi\) 时共序与上述符号论证不适用；式 (8)–(11) 仍然正确。

由于 finite-target oracle 的相对期望为

\[
\frac{E\widehat v_{\rm fs}}{v_T}
=\frac{B_{T,\ell}}{M_\ell},
\tag{13}
\]

已知参数补尾仍留下式 (11) 的损失。原论文 Theorem 3.1 的偏差展开保留有限样本余项，本来也不是精确无偏恒等式。

### 2.1 两个可以直接重算的例子

IID 时 \(C=\sigma^2I\)，并且

\[
\mathbf1'W\mathbf1
=T\ell-\frac{\ell^2-1}{3}.
\]

这是将每个滞后的配对数 \(2(T-h)\) 与 Bartlett 权重相乘求和得到的。因此

\[
B_{T,\ell}
=\sigma^2\left\{1-\frac\ell T+
\frac{\ell^2-1}{3T^2}\right\}.
\tag{14}
\]

\(\ell=1\) 时为常见的 \((1-1/T)\sigma^2\)。\(\ell>1\) 时，仅乘 \(T/(T-1)\) 不会一般性地消除中心化偏差。

下表是正式研究约定 \(T=200\)、\(\mathrm{lags}=4\)、\(\ell=5\) 下的**解析检查**；设 \(\phi=0.99\)、边际方差 \(\sigma^2=1\)。它不是新增 Monte Carlo 拒绝率。

| 对象 | 数值 |
| --- | ---: |
| 总体核尺度 \(M_\ell\) | 4.920597604 |
| 真实均值已知时的配对期望，式 (10) | 4.881588048 |
| 中心化 HAC 精确期望 \(B_{T,\ell}\) | 2.062231418 |
| 有限样本目标 \(v_T\) | 113.263987811 |
| 长程目标 \(\Omega\) | 199 |
| finite-target oracle 的期望 | 47.469143582 |
| finite-target oracle 的期望/目标 | 0.419101821 |
| long-run oracle 的期望 | 83.401262444 |

最后一行使用 \(\Omega/M_\ell\)，两种 oracle 因子的分子不同，不能将它们称为同一个校正。

## 3. 均值和随机分母的耦合

**机制命题 2：Gaussian 条件期望与独立性。**

记 \(Y=\bar X-\mu\)、\(R=HX\)，并令

\[
\nu=\operatorname{Var}(Y)=v_T/T,\qquad
k=\operatorname{Cov}(R,Y)=\frac{HC\mathbf1}{T},\qquad
\Sigma_R=HCH.
\tag{15}
\]

联合正态的条件分布为

\[
R\mid Y=z\sim
N\left(\frac{kz}{\nu},\,
\Sigma_R-\frac{kk'}{\nu}\right).
\]

将其代入 \(T^{-1}R'WR\)，逐项取条件期望得到

\[
E(\widehat\Omega\mid Y=z)
=B_{T,\ell}
+\frac{k'Wk}{T\nu^2}(z^2-\nu).
\tag{16}
\]

利用 \(EY^4=3\nu^2\)，进一步得到

\[
\operatorname{Cov}(Y^2,\widehat\Omega)
=\frac{2k'Wk}{T}
=\frac{2(HC\mathbf1)'W(HC\mathbf1)}{T^3}.
\tag{17}
\]

因为 Gaussian 的三阶中心矩为零，\(\operatorname{Cov}(Y,\widehat\Omega)=0\)。这不等于独立：只要 \(k\) 非零，式 (16) 随 \(z^2\) 改变。

\(W\) 正定，因此 \(k\) 非零时 \(k'Wk>0\)，从条件期望已经可以否定独立性。\(k=0\) 时，\(Y\) 与整个 \(R\) 联合 Gaussian 且不相关，故 \(Y\) 与 \(R\) 独立，也与其二次型独立。于是

\[
Y\perp\widehat\Omega
\quad\Longleftrightarrow\quad HC\mathbf1=0.
\tag{18}
\]

AR(1) 的第 i 行和为

\[
g_i=\frac{\sigma^2}{1-\phi}
\{1+\phi-\phi^i-\phi^{T-i+1}\}.
\tag{19}
\]

\(T\ge3\) 时，

\[
g_2-g_1=\sigma^2\phi(1-\phi^{T-2}),
\]

在 \(0<|\phi|<1\) 下不为零。因此有限样本非循环 AR(1) 的均值和分母一般不独立。IID、\(T=2\) 的这个 AR(1) 矩阵，以及其他行和恒定的协方差矩阵是例外。确定性地乘一个正补尾因子不会消除这种耦合。

同一 Gaussian 二次型计算还给出

\[
\operatorname{Var}(\widehat\Omega)
=\frac2{T^2}\operatorname{tr}
\left\{(C^{1/2}HWHC^{1/2})^2\right\}.
\tag{20}
\]

例如将 \(X\) 写成 \(\mu\mathbf1+C^{1/2}Z\)，对称矩阵谱分解后，二次型为独立标准正态平方的加权和；每个平方的方差为 2，即得式 (20)。这里不应误写为两个完整 \(C\) 夹住 \(HWH\)。

## 4. 固定 \(T\)、\(\phi\) 趋近 \(1\)：finite-target 的拒绝反例

**机制命题 3：已知 \(\phi,\rho\) 时，固定 \(T\) 的 finite-target Gaussian 临界值失败。**

在 \(K\) 列的共同 AR(1) 模型中，假设各列边际方差为 \(\sigma^2\)，时间相关均为 \(\phi\)，横截面相关矩阵为

\[
R_\rho=(1-\rho)I_K+\rho\mathbf1\mathbf1',
\qquad 0\le\rho\le1.
\]

即 \(\operatorname{Cov}(X_{s,\(j\)},X_{t,\(k\)})
=\sigma^2\phi^{|s-t|}(R_\rho)_{jk}\)。候选数 \(K\) 固定、全零均值。令 \(F_{\max}(x,K,\rho)\) 为 \(N(0,R_\rho)\) 向量最大值的 CDF；令

\[
q_\alpha=F_{\max}^{-1}(1-\alpha,K,\rho),\qquad
0<\alpha<1/2.
\]

这个 \(q_\alpha\) 为正、有限。考虑已知 \(\phi\) 的 finite-target 测试

\[
1\left\{\max_{j\le K}
\frac{\sqrt T\,\bar X_j}{\sqrt{\widehat v_{{\rm fs},j}}}
>q_\alpha\right\}.
\tag{21}
\]

固定 \(T\) 与 \(\ell<T\)，让 \(\phi\) 从下方趋近 \(1\)。协方差矩阵趋向时间上完全共同的水平，因此

\[
(\bar X_j)_{j\le K}\Rightarrow\sigma Z,\qquad
Z\sim N(0,R_\rho),\qquad
HX_{\cdot,j}\to_p0.
\]

而 \(M_\ell\to\sigma^2\ell\)、\(v_T\to\sigma^2T\)，所以 finite-target 因子趋向 \(T/\ell\)，是有限常数；所有 \(\widehat v_{{\rm fs},j}\to_p0\)。

不必在零分母处定义比值极限。对 \(\phi<1\)，尺度几乎处处为正；式 (21) 等价于存在 \(j\) 使

\[
\sqrt T\,\bar X_j>q_\alpha\sqrt{\widehat v_{{\rm fs},j}}.
\]

右边趋于零，左边趋向 \(\sqrt T\sigma Z_j\)。Gaussian 每个坐标为零的概率均为零，有限个坐标的边界仍为零概率，故拒绝概率极限是

\[
\boxed{\quad
1-F_{\max}(0,K,\rho).
\quad}
\tag{22}
\]

\(K=1\) 或 \(\rho=1\) 时为 \(1/2\)；\(\rho=0\) 时为 \(1-2^{-K}\)。这是全零均值下的 FWER，至少为 \(1/2\)。

**式 (22) 仅适用于上述已知 \(\phi\) 的 finite-target Gaussian 临界测试。** 长程因子 \(\Omega/M_\ell\) 在此极限中发散，不能代入这段“有限因子乘趋零分母”的证明。该命题固定 \(T\)，并非 local-to-unity 的同一个极限，也不否定固定 \(\phi\) 的大样本结果。

## 5. 即使尺度期望无偏，随机分母仍可能使拒绝率偏高

**机制命题 4：已知 \(\phi\) 时，mean-unbiased oracle 的单列固定 \(T\) 极限。**

仍固定 \(T,\ell\)。为便于表示，先除去 \(\sigma\)，令边际方差为 \(1\)，并置 \(\delta=1-\phi\)。取独立标准正态 \(G,\epsilon_2,\ldots,\epsilon_T\)，使用平稳初始化

\[
X_1=G,\qquad
X_t=\phi^{t-1}G+
\sqrt{1-\phi^2}\sum_{s=2}^t\phi^{t-s}\epsilon_s.
\]

固定 \(T\) 的展开为

\[
X_t=G+\sqrt{2\delta}\,S_t+O_p(\delta),
\qquad S_1=0,\quad S_t=\sum_{s=2}^t\epsilon_s.
\tag{23}
\]

\(G\) 与 \(S\) 独立。由 \(H\) 消去常数项，

\[
\widehat\Omega=\delta Q+o_p(\delta),\qquad
Q=\frac2T S'HWH S.
\]

令 \(a=EQ\)。\(Q>0\) 且非退化：\(S\) 的 \(T-1\) 个创新经 \(H\) 映射后在残差空间仍满秩，\(W\) 正定，所以这是至少一个正特征值的 Gaussian 平方和。

期望展开也需单独验证，不能仅从随机小量推断。设 \(D_{st}=|s-t|\)、\(\Lambda_{st}=\min(s-1,t-1)\)，则

\[
C=\mathbf1\mathbf1'-\delta D+O(\delta^2),\qquad
HDH=-2H\Lambda H.
\]

代入式 (8)，得到 \(B_{T,\ell}=\delta a+O(\delta^2)\)。又 \(v_T\to T\)，故式 (7) 的均值无偏尺度满足

\[
\widehat v_{\rm mu}\Rightarrow TQ/a,\qquad
\frac{\sqrt T\,\bar X}{\sqrt{\widehat v_{\rm mu}}}
\Rightarrow G\sqrt{a/Q}.
\tag{24}
\]

对任意 \(c>0\)，令 \(f(x)=\bar\Phi(c\sqrt x)\)。直接求导：

\[
f''(x)=\frac{c\varphi(c\sqrt x)}{4x^{3/2}}
(1+c^2x)>0,\qquad x>0.
\]

由于 \(Q/a\) 非退化且期望为 1，严格 Jensen 给出

\[
P\{G\sqrt{a/Q}>c\}
=E\bar\Phi(c\sqrt{Q/a})
>\bar\Phi(c).
\tag{25}
\]

取 \(c=\Phi^{-1}(1-\alpha)>0\)，极限拒绝率严格大于 \(\alpha\)。尺度期望已被精确改为 \(v_T\)，仍未得到高斯尾部。这是独立随机尺度混合的基础事实；本文的使用场景是式 (24) 的极限。有限 \(\phi\) 下均值与 HAC 通常相关，不能用同一 Jensen 推断所有有限 \(\phi\) 都有严格偏高的拒绝率。

## 6. Local-to-unity：已知参数补尾的期望仍失败

**机制命题 5：已知 \(\phi\) 时，local-to-unity 的相对期望极限。**

现在令 \(T\) 增长，并取

\[
\phi_T=1-a/T,\qquad a>0,\qquad
1\le\ell_T<T,\qquad \ell_T/T\to0,
\]

保持平稳边际方差 \(\sigma^2\)。只使用使 \(\phi_T\in(0,1)\) 的 \(T\)。定义

\[
f(a)=\int_0^1\int_0^1e^{-a|u-v|}\,du\,dv
=\frac{2(a-1+e^{-a})}{a^2},\quad 0<f(a)<1.
\tag{26}
\]

首先，\(\ell_T(1-\phi_T)\to0\)，且对 \(h<\ell\)，
\(0\le1-\phi_T^h\le h(1-\phi_T)\)。因此

\[
\frac{M_{\ell_T}}{\sigma^2\ell_T}\to1.
\tag{27}
\]

其次，协方差双重和的 Riemann 极限给出

\[
\frac{v_T}{T\sigma^2}
=\frac1{T^2}\sum_{s,t=1}^T\phi_T^{|s-t|}
\to f(a).
\tag{28}
\]

还需要中心化项的量级。由 Bartlett 行和，

\[
0\le\ell-s_i,\qquad
\sum_i(\ell-s_i)=\frac{\ell^2-1}{3},\qquad
0\le g_i\le T\sigma^2.
\]

所以

\[
s'g=\ell\mathbf1'g+O(\sigma^2T\ell^2)
=\ell T v_T+O(\sigma^2T\ell^2).
\]

结合 \(\mathbf1's=T\ell-(\ell^2-1)/3\)，代入式 (11)，得到

\[
D_{\rm center}
=\frac{\ell v_T}{T}+O(\sigma^2\ell^2/T).
\tag{29}
\]

另一方面 \(0\le D_{\rm pair}/M_\ell\le\ell/T\to0\)。将式 (27)–(29) 代入迹分解：

\[
\boxed{\quad
\frac{B_{T,\ell_T}}{M_{\ell_T}}\to1-f(a)<1.
\quad}
\tag{30}
\]

因此已知 \(\phi\) 的 finite-target 补尾的期望/目标也趋向 \(1-f(a)\)，中心化损失不会消失。记相对尺度 \(Q_T=\widehat v_{\rm fs}/v_T\)，它是非负随机量；若它趋概率于 \(1\)，则 \(EQ_T\ge(1-\epsilon)P(Q_T\ge1-\epsilon)\) 会使期望下极限至少为 1，与式 (30) 矛盾。所以这里相对一致性确实失败。

长程补尾则是另一个对象。由于 \(\Omega/(T\sigma^2)\to2/a\)，它相对于**有限目标**的期望极限为

\[
\frac{E\{(\Omega/M_{\ell_T})\widehat\Omega\}}{v_T}
\to\frac{2\{1-f(a)\}}{a f(a)}.
\tag{31}
\]

再令 \(a\) 趋近零，这个值趋向 2/3。式 (30)–(31) 是尺度或期望的失败，不自动提供某个检验的 拒绝率极限；拒绝率需要处理随机分母和临界值的联合分布。

## 7. 固定 \(K\) 的 oracle 温和持久性充分条件

**机制命题 6：已知 \(\phi,\rho\)、固定 \(K\) 时的二次型上界与 oracle 校准。**

仍令 \(0\le\phi<1\)、\(\ell\)<\(T\)，记

\[
r_{T,\ell}=\frac{\ell\Omega}{TM_\ell},
\qquad b_{T,\ell}=\ell/T+2r_{T,\ell}.
\tag{32}
\]

### 7.1 偏差与方差上界

式 (11) 与共序证明给 \(0\le B\le M_\ell\)。因为 \(h<\ell\)，

\[
D_{\rm pair}
\le\frac\ell T(M_\ell-\gamma_0)\le\ell M_\ell/T.
\]

又 \(s_i\le\ell\)、\(g_i\le\Omega\)，且式 (11) 中减去的乘积非负，故

\[
0\le D_{\rm center}\le2\ell\Omega/T.
\]

合并即得

\[
0\le1-\frac{B_{T,\ell}}{M_\ell}\le b_{T,\ell}.
\tag{33}
\]

方差上界要在正确的对称矩阵上计算。令 \(A=HWH\)、\(G=C^{1/2}AC^{1/2}\)，则
\(\|A\|_{\rm op}\le\ell\)、\(\|C\|_{\rm op}\le\Omega\)、\(G\) 非负定。因此

\[
\operatorname{tr}(G^2)
\le\|G\|_{\rm op}\operatorname{tr}(G)
\le\ell\Omega\,(TB_{T,\ell}).
\]

结合式 (20) 和 \(B_{T,\ell}\le M_\ell\)，

\[
\frac{\operatorname{Var}(\widehat\Omega)}{M_\ell^2}
\le2r_{T,\ell}.
\tag{34}
\]

这里的矩阵是 \(C\) 的平方根夹住 A，而非 \(CAC\)。若 \(b_{T,\ell}>1\)，该界可以很松，并不承诺有限样本校准。

### 7.2 严格证明 \(r_{T,\ell}\) 的持久性比较界

令 \(d=1-\phi\)、\(\tau=1/d\)，并令 \(R_\ell=(\phi^{|s-t|})_{s,t\le\ell}\)。由式 (3)，

\[
M_\ell=\frac{\sigma^2}{\ell}\mathbf1'R_\ell\mathbf1.
\]

Cauchy–Schwarz 应用于 \(R_\ell^{1/2}\mathbf1\) 与
\(R_\ell^{-1/2}\mathbf1\)：

\[
(\mathbf1'R_\ell\mathbf1)
(\mathbf1'R_\ell^{-1}\mathbf1)\ge\ell^2.
\tag{35}
\]

\(\ell\ge2\) 时，AR(1) 精度矩阵为

\[
R_\ell^{-1}=\frac1{1-\phi^2}
\begin{pmatrix}
1&-\phi&&\\
-\phi&1+\phi^2&-\phi&\\
&\ddots&\ddots&\ddots\\
&&-\phi&1
\end{pmatrix},
\]

其中仅内部对角元素为 \(1+\phi^2\)。将所有元素求和：

\[
\mathbf1'R_\ell^{-1}\mathbf1
=\frac{2+(\ell-2)(1+\phi^2)-2(\ell-1)\phi}{1-\phi^2}
=\frac{\ell d+2\phi}{1+\phi}.
\tag{36}
\]

\(\ell=1\) 时左边为 \(1\)，最后一式仍为 \(1\)，单独验证即可。于是

\[
M_\ell\ge
\frac{\sigma^2\ell(1+\phi)}{\ell d+2\phi},
\]

\[
\boxed{\quad
r_{T,\ell}
\le\frac{\ell+2\phi/d}{T}
\le\frac{\ell+2\tau}{T}.
\quad}
\tag{37}
\]

因此这里的显式常数 2 已经过证明，不是由近似符号猜出的常数。

### 7.3 从尺度界到固定 \(K\) 的 Gaussian 最大值检验

设 \(\ell_T=o(T)\)、\(T(1-\phi_T)\to\infty\)。式 (37) 给 \(r_{T,\ell_T}\to0\)，式 (33)–(34) 给

\[
\frac{\widehat v_{{\rm fs},j}}{v_T}
=\frac{\widehat\Omega_j}{M_{\ell_T}}\to_p1.
\]

具体地，简记 \(r=r_{T,\ell}\)、\(b=b_{T,\ell}\)，任取 \(\epsilon>b\)，由 Chebyshev，

\[
P\left(\left|\frac{\widehat\Omega_j}{M_\ell}-1\right|>\epsilon\right)
\le\frac{2r}{(\epsilon-b)^2}.
\tag{38}
\]

固定 \(K\) 用并集概率上界 即可控制全部尺度，不需要列独立。选 \(\epsilon_T\to0\) 但
\(r_T/(\epsilon_T-b_T)^2\to0\)，例如当 \(r_T>0\) 时取
\(\epsilon_T=b_T+r_T^{1/4}\)，坏事件概率至多 \(2K r_T^{1/2}\)。

全零均值下

\[
Z_j=\frac{\sqrt T\,\bar X_j}{\sqrt{v_T}},
\qquad Z\sim N(0,R_\rho)
\]

对每个 \(T\) 都准确成立，没有 CLT 近似。记 \(Z_{\max}=\max_{j\le K}Z_j\)、\(q=q_\alpha>0\)。在所有尺度位于 \([1-\epsilon,1+\epsilon]\) 的事件上，

\[
\{Z_{\max}>q\sqrt{1+\epsilon}\}
\subseteq
\{\widehat M>q\}
\subseteq
\{Z_{\max}>q\sqrt{1-\epsilon}\},
\]

其中 \(\widehat M=\max_j\sqrt T\,\bar X_j/\sqrt{\widehat v_{{\rm fs},j}}\)。
加入坏事件概率后得到两侧概率夹逼。\(Z_{\max}\) 的 CDF 连续，故式 (21) 的拒绝概率趋向 \(\alpha\)。\(\rho=1\) 的退化重复列同样有连续的一维最大值分布。

这也涵盖
\(\phi_T=1-aT^{-\kappa}\)、固定 \(a>0\)、\(0<\kappa<1\)、\(\ell=o(T)\)，并与第 6 节真正 local-to-unity 的期望失败保持区分。若 \(a\) 位于一个固定正的紧区间，式 (37) 的 \(r_{T,\ell}\) 上界在 \(a\) 上一致。本文的校准结论保持固定 \(K\)；已知参数下增长 \(K\) 的额外充分条件另见 [oracle selection bound](oracle-selection-bound.md)。

对于原始候选均值的单侧全族零假设 \(\mu_j\le0\)，中心化尺度对常数平移不变。同一噪声路径上，加入非正均值只能降低各列分子，因此全零均值是这个已知 \(\rho\) 的测试的最不利情形，渐近错误率至多 \(\alpha\)。

## 8. 附录：固定 \(K\) 的参数估计候选推论

**候选推论：已知 \(\rho\)、固定 \(K\)、正确 Gaussian AR(1) 下的 finite-target Gaussian 临界测试。**

本节另行允许估计 \(\phi\)，具体估计器固定为

\[
S_0=\sum_{t=1}^T(X_t-\bar X)^2,\qquad
S_1=\sum_{t=2}^T(X_t-\bar X)(X_{t-1}-\bar X),
\qquad\widehat\phi=S_1/S_0.
\tag{39}
\]

假定每列是平稳 Gaussian AR(1)、\(0\le\phi_T<1\)、
\(T(1-\phi_T)\to\infty\)，带宽确定性地满足 \(\ell=o(T)\)。\(\sigma\) 与常数均值在式 (39) 中相消，证明可对零均值、单位边际方差数据进行。

### 8.1 精确恒等式与相对参数误差

令 \(d=1-\phi\)，并写 \(X_t=\phi X_{t-1}+\sqrt{1-\phi^2}\epsilon_t\)。
分别展开 \(S_0,S_1\)：

\[
S_0=\sum X_t^2-T\bar X^2,
\]

\[
S_1=\sum_{t=2}^TX_tX_{t-1}
-(T+1)\bar X^2+\bar X(X_1+X_T).
\]

由 AR 方程替换交叉和，得到精确恒等式

\[
S_1-\phi S_0
=\sqrt{1-\phi^2}\sum_{t=2}^TX_{t-1}\epsilon_t
-\phi X_T^2-(Td+1)\bar X^2+\bar X(X_1+X_T).
\tag{40}
\]

首项是鞅差之和。不同时间项正交，每个 \(EX_{t-1}^2=1\)，因此其方差准确为 \((1-\phi^2)(T-1)\)，它是 \(O_p(\sqrt{Td})\)。

Gaussian 四阶矩给

\[
\operatorname{Var}\left(\frac1T\sum X_t^2\right)
\le\frac2T\frac{1+\phi^2}{1-\phi^2}
\le\frac2{Td}.
\]

又 \(E\bar X^2=v_T/T\le\Omega/T\le2/(Td)\)。所以
\(S_0/T\to_p1\)。

式 (40) 中 \(\phi X_T^2=O_p(1)\)，
\((Td+1)\bar X^2=O_p(1)\)，
\(\bar X(X_1+X_T)=O_p((Td)^{-1/2})\)。
除以 \(dS_0\)，得到

\[
\frac{|\widehat\phi-\phi|}{d}
=O_p\{(Td)^{-1/2}+(Td)^{-1}\}\to_p0.
\tag{41}
\]

而且估计器在精确数学中自动满足 \(|\widehat\phi|<1\)。对 \(y_t=X_t-\bar X\)，

\[
S_0\pm S_1
=\frac12\left\{\sum_{t=2}^T(y_t\pm y_{t-1})^2+y_1^2+y_T^2\right\}>0
\]

只要 \(S_0>0\)；若右边为零，递推与端点迫使全部 \(y_t=0\)。Gaussian 非退化数据在 \(T\ge2\) 时 \(S_0>0\) 几乎必然成立。这不是浮点实现永不退化的承诺。

### 8.2 补尾因子的导数界

对 \(\phi>0\)，定义非负整数滞后 \(J\) 的分布

\[
P_\phi(J=h)=\frac{b_h\phi^h}{(1+\phi)/(1-\phi)},
\quad b_0=1,\quad b_h=2\ (h\ge1).
\]

则 \(\log\Omega\) 的导数为 \(E_\phi J/\phi=2/(1-\phi^2)\)。
\(\log m_j\) 的导数为用 \(w_j(h)=(1-h/j)_+\) 倾斜后分布的 \(EJ/\phi\)。
权重单调递减，取独立同分布的 \(J,J'\)，有

\[
2\operatorname{Cov}(J,w_j(J))
=E\{(J-J')(w_j(J)-w_j(J'))\}\le0.
\]

所以倾斜后的 \(EJ\) 不增加，得到

\[
0\le\partial_\phi\log m_j(\phi)
\le\frac2{1-\phi^2}.
\tag{42}
\]

\(\phi=0\) 由连续性成立。因此

\[
\left|\partial_\phi\log\frac{m_T(\phi)}{m_\ell(\phi)}\right|
\le\frac2{1-\phi^2},\qquad 0\le\phi<1.
\tag{43}
\]

需要处理可能为负的 \(\widehat\phi\)，不能直接把正 \(\phi\) 的倾斜证明套过去。

- 当真值 \(\phi\ge1/8\) 时，式 (41) 使连接 \(\phi\) 与 \(\widehat\phi\) 的路径以趋于 1 的概率保持非负，且 \(1-\phi_{\rm path}\ge d/2\)。式 (43) 沿路径至多为 \(4/d\)。
- 当真值 \(\phi<1/8\) 时，\(\widehat\phi-\phi\to_p0\)，路径以高概率位于 \([-1/4,1/4]\)。在这个紧区间，对所有 \(j\)，
  \[
  m_j(\theta)\ge1-2\sum_{h\ge1}(1/4)^h=1/3,
  \qquad |m_j'(\theta)|\le2\sum_{h\ge1}h(1/4)^{h-1}=32/9.
  \]
  所以 \(\log(m_T/m_\ell)\) 导数的绝对值统一不超过 \(64/3\)。

用均值定理与式 (41)，两种情形都给出

\[
\frac{\eta_{{\rm fs},T,\ell}(\widehat\phi)}
{\eta_{{\rm fs},T,\ell}(\phi)}\to_p1.
\tag{44}
\]

### 8.3 候选推论的准确范围

每列应用式 (44)，固定 \(K\) 用并集概率上界。令

\[
\widehat v_{{\rm fs,plug},j}
=\frac{m_T(\widehat\phi_j)}{m_\ell(\widehat\phi_j)}
\widehat\Omega_j.
\]

它与同带宽 finite-target oracle 的比值趋概率于 1。结合第 7 节的 oracle 相对一致性，全部列尺度相对于 \(v_T\) 趋概率于 1。仍用真实 \(\rho\) 的 Gaussian max 临界值，原来的事件夹逼给固定 \(K\)、全零均值下渐近 \(\alpha\)；常数平移不变时，\(\mu_j\le0\) 的零假设下至多 \(\alpha\)。

本推论不涉及估计 \(\rho\)、不涉及 Monte Carlo bootstrap 分布、不涉及增长 \(K\)。它只适用于式 (39) 的具体 \(\phi\) 估计器；其他自动带宽、自动阶数、拟合规则与数值失效处理需要另证。不能从式 (44) 单独推断随机 HAC 分母已经集中；第 7 节的条件不可省略。

## 9. 复现计算与文献边界

[确定性核验脚本](../scripts/verify_tail_mechanism.py)已运行，[原始记录](../results/research/verification/tail-mechanism-checks.json)另存。可重新运行到一个独立文件：

```bash
python scripts/verify_tail_mechanism.py --output /tmp/tail-mechanism-checks.json
```

实际核验范围为：

1. 72 个稠密矩阵案例，覆盖 IID、正负 \(\phi\)、\(\ell=1\) 与 \(\ell=T-1\)，核对迹分解、配对与中心化损失、精确方差及有限均值目标；其中 54 组核对非负 \(\phi\) 的偏差、方差、谱范数和持久性上界。
2. 独立重算 \(T=200,\ell=5\) 的解析表。生产实现的期望与方差和独立稠密计算的最大相对差异不超过 \(9.70\times10^{-15}\)。
3. 用稳定的残差协方差迹计算记录固定 \(T\) 下 \(B/(1-\phi)\) 的趋近误差，并与随机游走二次型的解析期望对照。
4. 12 个预设输入核对参数估计恒等式 (40)。这些输入只作代数检查，没有用于调参或评价误报率。
5. 9 个 local-to-unity 网格记录有限期望与式 (30) 极限的差异；有限网格的方向或接近程度不代替极限证明。

条件均值的 Monte Carlo 分箱、Jensen 混合分布的模拟和参数误差率模拟未运行。正文推导与上述数值检查分开成立；正式零均值评价见[结果解读](tail-results.md)。候选数增长的已知参数充分条件另见[候选规模上界](oracle-selection-bound.md)，不改变本节固定 \(K\) 的拟合参数范围。

文献使用范围：

- [Liu–Chan：Tail Postcoloring in Long-Run Variance Estimation of Time Series，阅读版本 arXiv v1](https://arxiv.org/html/2605.15596v1)：Definition 3.1、Example 3.1、Remark 3.2 给方法与目标基线；Theorem 3.1 为渐近估计结果。本文不把补尾或有限样本目标称为原创。
- [Zhang–Shao：Another look at bandwidth-free inference，JRSSB 2024](https://doi.org/10.1093/jrsssb/qkad108)：提供 sample splitting/self-normalisation 对照，其均值等式检验与本文单侧全族零假设需要对齐。不能据此宣称近单位根有效。
- [Müller：HAC Corrections for Strongly Autocorrelated Time Series，2014，作者全文](https://www.princeton.edu/~umueller/HACtest.pdf)：第 4 节提供特定 AR(1) 近单位根推断的背景；不是本文 fixed-\(K\) 随机尺度的现成保证。

上述机制命题与候选推论仍需逐项查重。若贡献只是已知恒等式、AR 模型代入和文献复现，应以机制分析项目表述；不将本文升级为已证明实际 bootstrap 或普遍金融策略检验有效的方法论文。
