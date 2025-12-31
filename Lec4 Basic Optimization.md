# DDA3020 Lecture 04: 基础优化 (Basic Optimization)

## 1. 凸集 (Convex Set)

### 1.1 仿射集 (Affine Set)
*   **仿射直线定义**: 给定两点 $x_1, x_2$，穿过这两点的直线上的所有点 $x$ 可以表示为：
    $$ x = \theta x_1 + (1 - \theta)x_2, \quad \theta \in \mathbb{R} $$
*   **仿射集定义**: 如果一个集合包含其中任意两点的直线，则该集合为仿射集。
*   **例子**: 线性方程组的解集 $\{x | Ax = b\}$ 是一个仿射集。

### 1.2 凸集 (Convex Set)
*   **线段定义**: 给定两点 $x_1, x_2$，这两点之间的线段上的所有点 $x$ 可以表示为：
    $$ x = \theta x_1 + (1 - \theta)x_2, \quad 0 \le \theta \le 1 $$
*   **凸集定义**: 如果一个集合包含其中任意两点的线段，则该集合为凸集 $C$。
    $$ x_1, x_2 \in C, 0 \le \theta \le 1 \implies \theta x_1 + (1 - \theta)x_2 \in C $$
*   **直观理解**: 集合内任意两点连线都在集合内部（没有凹陷）。

---

## 2. 凸函数 (Convex Function)

### 2.1 定义
函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是凸函数，当且仅当其定义域 $\text{dom} f$ 是凸集，且满足：
$$ f(\theta x + (1 - \theta)y) \le \theta f(x) + (1 - \theta)f(y) $$
*   **适用范围**: $\forall x, y \in \text{dom} f, \quad 0 \le \theta \le 1$。
*   **凹函数 (Concave)**: 如果 $-f$ 是凸函数，则 $f$ 是凹函数。
*   **严格凸函数 (Strictly Convex)**: 上述不等式取严格小于号 ($<$)，且 $x \neq y, 0 < \theta < 1$。

### 2.2 常见例子
*   **$\mathbb{R}$ 上的凸函数**:
    *   仿射函数: $ax + b$
    *   指数函数: $e^{ax}$
    *   幂函数: $x^\alpha$ ($\alpha \ge 1$ 或 $\alpha \le 0$)
    *   负熵: $x \log x$
*   **$\mathbb{R}^n$ 上的凸函数**:
    *   仿射函数: $f(x) = a^\top x + b$
    *   范数: $\ell_p$ 范数 $\lVert x \rVert_p$ ($p \ge 1$)
*   **矩阵空间 $\mathbb{R}^{m \times n}$ 上的凸函数**:
    *   仿射函数: $f(X) = \text{tr}(A^\top X) + b = \sum \sum a_{ij}x_{ij} + b$
    *   谱范数 (最大奇异值): $f(X) = \lVert X \rVert_2 = \sigma_{\max}(X)$

### 2.3 判定条件 (核心)

#### 一阶条件 (First-order condition)
若 $f$ 可微（梯度 $\nabla f(x)$ 存在），则 $f$ 是凸函数当且仅当：
$$ f(y) \ge f(x) + \nabla f(x)^\top (y - x), \quad \forall x, y \in \text{dom} f $$
*   **几何意义**: 凸函数的一阶泰勒近似（切线/切平面）永远位于函数的下方（全局下估计）。

#### 二阶条件 (Second-order condition)
若 $f$ 二阶可微（Hessian 矩阵 $\nabla^2 f(x)$ 存在），则 $f$ 是凸函数当且仅当：
$$ \nabla^2 f(x) \succeq 0, \quad \forall x \in \text{dom} f $$
*   **解释**: $\succeq 0$ 表示 Hessian 矩阵是**半正定矩阵 (Positive Semi-Definite, PSD)**。
*   **严格凸**: 若 $\nabla^2 f(x) \succ 0$ (正定)，则 $f$ 严格凸。

#### 例子: 二次函数与最小二乘
1.  **二次函数**: $f(x) = \frac{1}{2}x^\top P x + q^\top x + r$
    *   梯度: $\nabla f(x) = Px + q$
    *   Hessian: $\nabla^2 f(x) = P$
    *   结论: 当且仅当 $P \succeq 0$ 时，该函数为凸。
2.  **最小二乘**: $f(x) = \lVert Ax - b \rVert_2^2$
    *   梯度: $\nabla f(x) = 2A^\top(Ax - b)$
    *   Hessian: $\nabla^2 f(x) = 2A^\top A$
    *   结论: 由于对于任意 $A$，其格拉姆矩阵 $A^\top A$ 总是半正定的，因此最小二乘目标函数总是凸的。

### 2.4 Jensen 不等式 (Jensen's Inequality)
若 $f$ 是凸函数，则：
$$ f(\mathbb{E}[z]) \le \mathbb{E}[f(z)] $$
*   $z$: 随机变量。
*   基础形式: $f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y)$ 是其特例。

---

## 3. 凸优化问题 (Convex Optimization Problem)

### 3.1 标准形式
$$
\begin{aligned}
\text{minimize} \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \le 0, \quad i = 1, \dots, m \\
& h_i(x) = 0, \quad i = 1, \dots, p
\end{aligned}
$$
*   $x \in \mathbb{R}^n$: 优化变量。
*   $f_0$: 目标函数 (Cost function)。
*   $f_i$: 不等式约束。
*   $h_i$: 等式约束。

### 3.2 凸优化问题的特定要求
一个优化问题是凸优化问题，必须满足：
1.  目标函数 $f_0$ 是凸函数。
2.  不等式约束函数 $f_1, \dots, f_m$ 是凸函数。
3.  **等式约束函数必须是仿射的**: $h_i(x) = a_i^\top x - b_i = 0$ (即 $Ax = b$)。

### 3.3 局部最优与全局最优 (Local vs Global Optima)
**定理**: 凸优化问题的任意局部最优解都是全局最优解。

**详细证明 (反证法)**:
1.  假设 $x$ 是局部最优解，但不是全局最优。这意味着存在一个可行解 $y$，使得 $f_0(y) < f_0(x)$。
2.  由于 $x$ 是局部最优，存在半径 $r > 0$，使得在邻域 $\lVert z - x \rVert_2 \le r$ 内，$f_0(z) \ge f_0(x)$。
3.  构造点 $z = \theta y + (1 - \theta)x$，取 $\theta = \frac{r}{2\lVert y - x \rVert_2}$。
    *   这保证了 $z$ 位于 $x$ 和 $y$ 的连线上，且 $z$ 非常接近 $x$ ($\lVert z - x \rVert_2 = 0.5r < r$)，即 $z$ 在 $x$ 的局部邻域内。
4.  根据凸函数的性质：
    $$ f_0(z) \le \theta f_0(y) + (1 - \theta)f_0(x) $$
5.  因为假设 $f_0(y) < f_0(x)$，代入上式：
    $$ f_0(z) < \theta f_0(x) + (1 - \theta)f_0(x) = f_0(x) $$
6.  **矛盾**: 我们得出了 $f_0(z) < f_0(x)$，但这与 $x$ 是局部最优解（即在邻域内 $f_0(z) \ge f_0(x)$）矛盾。
7.  因此，假设不成立，$x$ 必须是全局最优解。

---

## 4. 无约束最小化: 梯度下降法 (Unconstrained Minimization)

### 4.1 通用下降方法 (General Descent Method)
迭代更新公式：
$$ x^{(k+1)} = x^{(k)} + t^{(k)}\Delta x^{(k)} $$
*   $\Delta x^{(k)}$: 搜索方向 (Search direction)。
*   $t^{(k)} > 0$: 步长 (Step size)。
*   **下降条件**: 需满足 $f(x^{(k+1)}) < f(x^{(k)})$。
*   基于一阶泰勒展开: $f(x^+) \approx f(x) + t\nabla f(x)^\top \Delta x$。
    *   若要下降，必须满足 **$\nabla f(x)^\top \Delta x < 0$**。

### 4.2 梯度下降法 (Gradient Descent)
*   **搜索方向**: 选择负梯度方向 $\Delta x = -\nabla f(x)$。
    *   原因: $\nabla f(x)^\top (-\nabla f(x)) = -\lVert \nabla f(x) \rVert_2^2 < 0$，保证了下降。
*   **停止准则**: 通常当梯度足够小 $\lVert \nabla f(x) \rVert_2 \le \epsilon$ 时停止。

### 4.3 步长选择 (Line Search)
1.  **精确直线搜索 (Exact Line Search)**:
    $$ t = \arg\min_{t>0} f(x + t\Delta x) $$
2.  **回溯直线搜索 (Backtracking Line Search)** (非精确，常用):
    *   参数: $\alpha \in (0, 0.5), \beta \in (0, 1)$。
    *   初始化 $t = 1$。
    *   重复 $t := \beta t$，直到满足 Armijo 条件：
        $$ f(x + t\Delta x) < f(x) + \alpha t \nabla f(x)^\top \Delta x $$

---

## 5. 有约束最小化: 拉格朗日对偶与 KKT (Constrained Minimization)

考虑一般优化问题（注意此处符号变化）：
$$
\begin{aligned}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & h_i(x) \le 0, \quad i = 1, \dots, m \\
& \ell_j(x) = 0, \quad j = 1, \dots, r
\end{aligned}
$$

### 5.1 拉格朗日函数 (Lagrangian Function)
$$ L(x, u, v) = f(x) + \sum_{i=1}^m u_i h_i(x) + \sum_{j=1}^r v_j \ell_j(x) $$
*   $u_i$: 对应不等式约束的拉格朗日乘子 (Lagrange multiplier)，要求 $u_i \ge 0$。
*   $v_j$: 对应等式约束的拉格朗日乘子。

### 5.2 拉格朗日对偶函数 (Dual Function)
$$ g(u, v) = \min_{x \in \mathbb{R}^n} L(x, u, v) $$
*   $g(u, v)$ 是关于 $(u, v)$ 的凹函数（即使原问题不是凸的）。
*   它是原问题最优值 $p^*$ 的下界。

### 5.3 对偶问题 (Dual Problem)
$$
\begin{aligned}
\max_{u, v} \quad & g(u, v) \\
\text{s.t.} \quad & u \ge 0
\end{aligned}
$$

### 5.4 KKT 条件 (Karush-Kuhn-Tucker Conditions)
对于凸优化问题（且满足 Slater 条件），$x^*$ 和 $(u^*, v^*)$ 分别是原问题和对偶问题的最优解的**充要条件**是满足 KKT 条件：

1.  **平稳性 (Stationarity)**: 拉格朗日函数对 $x$ 的梯度为 0。
    $$ 0 \in \partial f(x) + \sum_{i=1}^m u_i \partial h_i(x) + \sum_{j=1}^r v_j \partial \ell_j(x) $$
    (若可微，则 $\nabla f(x) + \sum u_i \nabla h_i(x) + \sum v_j \nabla \ell_j(x) = 0$)
2.  **互补松弛性 (Complementary Slackness)**:
    $$ u_i \cdot h_i(x) = 0, \quad \forall i $$
    *   这意味着若约束无效 ($h_i(x) < 0$)，则乘子 $u_i$ 必须为 0；若乘子 $u_i > 0$，则约束必须取等号 ($h_i(x) = 0$)。
3.  **原问题可行性 (Primal Feasibility)**:
    $$ h_i(x) \le 0, \quad \ell_j(x) = 0, \quad \forall i, j $$
4.  **对偶可行性 (Dual Feasibility)**:
    $$ u_i \ge 0, \quad \forall i $$

---

## 6. 优化与机器学习 (Optimization and Machine Learning)

*   **凸最小化应用**: 线性回归 (Linear Regression), 逻辑回归 (Logistic Regression), 支持向量机 (SVM)。
*   **梯度下降应用**: 线性回归, 逻辑回归, 神经网络 (Neural Networks)。
*   **拉格朗日与 KKT 应用**: SVM, K-Means, 高斯混合模型 (GMM), 主成分分析 (PCA)。
*   **学习目标**: 给定 ML 模型，需能判断：
    1.  是凸优化还是非凸优化？
    2.  是否存在局部/全局最优？
    3.  应采用哪种优化方法？