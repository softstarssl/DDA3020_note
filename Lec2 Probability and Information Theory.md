# DDA3020 Lecture 02: 概率论与信息论基础

## 1. 概率、事件与随机变量 (Probability, Event, Random Variables)

### 1.1 基本定义
*   **随机试验 (Random Experiment)**: 一个过程，其结果是不确定的。例如：抛硬币两次。
*   **样本空间 (Sample Space, $S$)**: 随机试验所有可能结果的集合。
    *   例如抛硬币两次：$S = \{(H, H), (H, T), (T, H), (T, T)\}$。
*   **事件 (Event, $A$)**: 样本空间 $S$ 的子集 ($A \subseteq S$)。
    *   例如“至少有一次正面朝上”：$A = \{(H, H), (H, T), (T, H)\}$。

### 1.2 概率公理与性质
对于事件 $A \subseteq S$ 和 $B \subseteq S$：
1.  **非负性**: $P(A) \ge 0$
2.  **归一性**: $P(S) = 1$
3.  **加法法则**:
    *   如果 $A \cap B = \emptyset$ (互斥)，则 $P(A \cup B) = P(A) + P(B)$
    *   一般情况：$P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### 1.3 随机变量 (Random Variables)
*   **定义**: 一个将样本空间 $S$ 映射到实数空间 $\mathbb{R}$ 的函数 $X: S \to \mathbb{R}$。
    *   例如：定义 $X$ 为抛两次硬币中“反面”出现的次数。
    *   $X((H,H))=0, X((H,T))=1, X((T,T))=2$。
*   **状态空间 (State Space, $\mathcal{X}$)**: $X$ 的输出空间，例如 $\{0, 1, 2\}$。
*   **类型**:
    *   **离散型 (Discrete)**: 状态空间是有限或可数的。
    *   **连续型 (Continuous)**: 状态空间是不可数的（如实数区间）。

---

## 2. 离散随机变量 (Discrete Random Variables)

### 2.1 概率质量函数 (PMF)
描述随机变量 $X$ 取某个特定值 $x$ 的概率：
$$ P(X=x), \quad x \in \mathcal{X} $$
**性质**:
1.  $P(X=x) \ge 0$
2.  $\sum_{x \in \mathcal{X}} P(X=x) = 1$

### 2.2 联合概率、边缘概率与条件概率
假设有两个随机变量 $X$ 和 $Y$。

*   **联合概率 (Joint Probability)**: $P(X=x, Y=y)$
*   **边缘概率 (Marginal Probability)**:
    $$ P(X=x) = \sum_{y \in \mathcal{Y}} P(X=x, Y=y) $$
    $$ P(Y=y) = \sum_{x \in \mathcal{X}} P(X=x, Y=y) $$
*   **条件概率 (Conditional Probability)**:
    $$ P(X=x | Y=y) = \frac{P(X=x, Y=y)}{P(Y=y)} $$
*   **乘法法则**:
    $$ P(X=x, Y=y) = P(X=x | Y=y)P(Y=y) = P(Y=y | X=x)P(X=x) $$

### 2.3 贝叶斯定理 (Bayes' Rule)
结合条件概率定义与乘法法则推导：
$$ P(Y=y | X=x) = \frac{P(X=x | Y=y)P(Y=y)}{P(X=x)} $$
展开分母（全概率公式）：
$$ P(Y=y | X=x) = \frac{P(X=x | Y=y)P(Y=y)}{\sum_{y' \in \mathcal{Y}} P(X=x | Y=y')P(Y=y')} $$

*   **应用案例：医疗诊断**
    *   **设定**:
        *   $y=1$: 患癌, $y=0$: 未患癌。
        *   $x=1$: 检测阳性, $x=0$: 检测阴性。
    *   **已知数据**:
        *   先验概率 (Prior): $P(y=1) = 0.0013$ (原文例子数据修正为0.13，此处按Slide 11计算)。
            *   *注意*: Slide 11 中 $P(y=1)=0.13$ (13%)。
        *   似然概率 (Likelihood, 敏感度): $P(x=1 | y=1) = 0.8$。
        *   误报率 (False Positive): $P(x=1 | y=0) = 0.1$。
    *   **问题**: 检测为阳性时，真正患癌的概率 $P(y=1 | x=1)$ 是多少？
    *   **计算**:
        $$
        \begin{aligned}
        P(y=1 | x=1) &= \frac{P(x=1 | y=1)P(y=1)}{P(x=1 | y=1)P(y=1) + P(x=1 | y=0)P(y=0)} \\
        &= \frac{0.8 \times 0.13}{0.8 \times 0.13 + 0.1 \times (1 - 0.13)} \\
        &= \frac{0.104}{0.104 + 0.087} \\
        &= \frac{0.104}{0.191} \approx 0.5445
        \end{aligned}
        $$
    *   **结论**: 即使检测阳性，患癌概率也只有约 54%，因为先验概率较低且存在误报。

### 2.4 独立性 (Independence)
如果 $X$ 和 $Y$ 独立，记作 $X \perp Y$，则：
$$ P(X, Y) = P(X)P(Y) $$

**参数数量分析**:
假设 $X$ 有 3 个状态，$Y$ 有 4 个状态。
*   **不独立时**: 定义联合分布 $P(X, Y)$ 需要 $(3 \times 4) - 1 = 11$ 个自由参数（减1是因为总和为1）。
*   **独立时**: 需要 $(3-1) + (4-1) = 2 + 3 = 5$ 个自由参数。独立性大大减少了模型参数。

### 2.5 期望与方差 (Expectation and Variance)
*   **期望 (Expectation/Mean)**:
    $$ E[X] = \sum_{x \in \mathcal{X}} x P(X=x) $$
*   **函数的期望**:
    $$ E[f(X)] = \sum_{x \in \mathcal{X}} f(x) P(X=x) $$
*   **方差 (Variance)**: 衡量数据偏离均值的程度。
    $$ Var(X) = E[(X - E[X])^2] $$
    **推导**:
    令 $\mu = E[X]$。
    $$
    \begin{aligned}
    Var(X) &= E[(X - \mu)^2] \\
    &= E[X^2 - 2X\mu + \mu^2] \\
    &= E[X^2] - 2\mu E[X] + \mu^2 \quad (\text{因为 } \mu \text{ 是常数}) \\
    &= E[X^2] - 2\mu(\mu) + \mu^2 \\
    &= E[X^2] - \mu^2 \\
    &= E[X^2] - (E[X])^2
    \end{aligned}
    $$
*   **标准差 (Standard Deviation)**:
    $$ Std = \sqrt{Var(X)} $$

---

## 3. 连续随机变量 (Continuous Random Variables)

### 3.1 概率密度函数 (PDF)
对于连续变量，单点概率 $P(X=x)=0$。我们使用 PDF $p(x)$。
*   **区间概率**:
    $$ P(a < X < b) = \int_{a}^{b} p(x) dx $$
*   **微元解释**:
    $$ P(x < X < x + dx) \approx p(x)dx $$
*   **累积分布函数 (CDF)**:
    $$ F_X(x) = P(X < x) = \int_{-\infty}^{x} p(s) ds $$
    $$ p(x) = \frac{d}{dx}F_X(x) $$

### 3.2 连续变量的期望与方差
将离散情况下的求和 $\sum$ 替换为积分 $\int$。
*   **期望**: $\mu = E[X] = \int x p(x) dx$
*   **矩 (Moments)**: $M_k = E[X^k] = \int x^k p(x) dx$
*   **方差**: $Var(X) = E[X^2] - (E[X])^2$ (公式形式与离散相同)

---

## 4. 常见分布 (Popular Distributions)

### 4.1 伯努利分布 (Bernoulli Distribution)
适用于二值变量 $x \in \{0, 1\}$ (如抛硬币)。
*   **参数**: $\mu$ (表示 $x=1$ 的概率)。
*   **公式**:
    $$ Bern(x|\mu) = \mu^x (1-\mu)^{1-x} $$
*   **性质**:
    *   $E[x] = \mu$
    *   $Var[x] = \mu(1-\mu)$

### 4.2 二项分布 (Binomial Distribution)
进行 $N$ 次独立的伯努利试验，出现 $x=1$ (正面) 的次数 $m$。
*   **参数**: $N$ (试验次数), $\mu$ (单次成功的概率)。
*   **公式**:
    $$ Bin(m|N, \mu) = \binom{N}{m} \mu^m (1-\mu)^{N-m} $$
    其中组合数 $\binom{N}{m} = \frac{N!}{(N-m)!m!}$。
*   **性质**:
    *   $E[m] = N\mu$
    *   $Var[m] = N\mu(1-\mu)$

### 4.3 高斯分布 (Gaussian / Normal Distribution)
最常见的连续分布。
*   **一元高斯分布**:
    $$ \mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right) $$
    *   $\mu$: 均值 (Mean)
    *   $\sigma^2$: 方差 (Variance)
*   **多元高斯分布 (D维向量 x)**:
    $$ \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) $$
    *   $\boldsymbol{\mu}$: $D$ 维均值向量。
    *   $\Sigma$: $D \times D$ 协方差矩阵 (Covariance Matrix)。
    *   $|\Sigma|$: 协方差矩阵的行列式。

---

## 5. 信息论 (Information Theory)

### 5.1 信息量 (Information)
香农定义，量化事件的不确定性。
*   对于离散随机变量 $X$，取值 $x_k$ 的概率为 $p_k$。
*   **公式**:
    $$ I(x_k) = \log \frac{1}{p_k} = -\log(p_k) $$
*   **单位**: 若底数为2，单位为比特 (bit)。
*   **直觉**: 概率越小（越惊讶），信息量越大。

### 5.2 熵 (Entropy)
熵是信息量的期望值，表示整个信源的平均不确定性。
*   **公式**:
    $$ H(X) = E[I(x)] = -\sum_{x \in \mathcal{X}} p(x) \log p(x) $$
*   **二值信源熵**:
    若 $X \in \{0, 1\}$，$P(X=1)=p$，则：
    $$ H(X) = -p \log p - (1-p) \log (1-p) $$

### 5.3 交叉熵 (Cross Entropy)
衡量使用分布 $Q$ 来编码来自真实分布 $P$ 的事件所需的平均比特数。
*   **公式**:
    $$ H_{P,Q}(X) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x) $$
*   **性质**:
    1.  非负性: $H_{P,Q}(X) \ge 0$
    2.  $H_{P,Q}(X) \ge H(P)$，当且仅当 $P=Q$ 时取等号。

### 5.4 相对熵 / KL 散度 (Kullback-Leibler Divergence)
衡量两个分布 $P$ 和 $Q$ 之间的距离（或差异）。
*   **离散形式**:
    $$ D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} $$
*   **连续形式**:
    $$ D_{KL}(P||Q) = \int p(x) \log \frac{p(x)}{q(x)} dx $$
*   **性质**:
    1.  **非负性**: $D_{KL}(P||Q) \ge 0$ (由 Jensen 不等式可证)。
    2.  **非对称性**: $D_{KL}(P||Q) \neq D_{KL}(Q||P)$。

### 5.5 熵、交叉熵与 KL 散度的关系
**公式**:
$$ H_{P,Q}(X) = H(P) + D_{KL}(P||Q) $$

**详细推导**:
$$
\begin{aligned}
D_{KL}(P||Q) &= \sum_{x} P(x) \log \frac{P(x)}{Q(x)} \\
&= \sum_{x} P(x) (\log P(x) - \log Q(x)) \quad (\text{对数除法变减法}) \\
&= \sum_{x} P(x) \log P(x) - \sum_{x} P(x) \log Q(x) \\
&= - \left( -\sum_{x} P(x) \log P(x) \right) + \left( -\sum_{x} P(x) \log Q(x) \right) \\
&= -H(P) + H_{P,Q}(X)
\end{aligned}
$$
移项即得：
$$ H_{P,Q}(X) = H(P) + D_{KL}(P||Q) $$

这意味着：交叉熵 = 真实分布的熵 + 两个分布的差异(KL散度)。在机器学习中，因为训练数据的真实分布 $P$ 是固定的（即 $H(P)$ 是常数），所以**最小化交叉熵等价于最小化 KL 散度**。