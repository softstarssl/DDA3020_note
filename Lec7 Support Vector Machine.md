# DDA3020 Lecture 07: 支持向量机 (Support Vector Machine)

## 1. 动机 (Motivation)

### 1.1 分类问题回顾
给定训练数据集 $D = \{(x_i, y_i)\}_{i=1}^m$，其中 $x_i \in \mathbb{R}^n$，$y_i \in \{-1, +1\}$。
假设函数为符号函数：
$$ y = \text{Sgn}(f_w(x)) = \text{Sgn}(w^\top x + b) $$
要求：
*   若 $y_i = +1$，则 $w^\top x_i + b > 0$
*   若 $y_i = -1$，则 $w^\top x_i + b < 0$

### 1.2 为什么需要 SVM？
对于线性可分的数据，存在无数个超平面可以将正负样本完美分开。
*   **逻辑回归 (Logistic Regression)**: 损失函数是凸的但不严格凸（除非有正则化），可能收敛到任意一个能分开数据的解，取决于初始值和迭代停止时间。
*   **直觉**: 我们希望决策边界尽可能远离两类数据点（即位于“中间”），这样泛化能力更强。

**核心思想**: **最大化间隔 (Large Margin)**。即寻找一个超平面，使得离超平面最近的正样本和负样本的距离最大化。

---

## 2. 推导 I: 最大间隔 (Large Margin) - 几何视角

### 2.1 几何基础
定义超平面为 $f_{w,b}(x) = w^\top x + b = 0$。

**引理 1**: 向量 $w$ 正交于超平面。
*证明*:
1. 取超平面上任意两点 $x_1, x_2$。
2. 满足 $w^\top x_1 + b = 0$ 和 $w^\top x_2 + b = 0$。
3. 相减得 $w^\top (x_1 - x_2) = 0$。
4. $x_1 - x_2$ 是超平面上的向量，故 $w$ 与超平面正交。

**推导 1**: 任意点 $x$ 到超平面 $w^\top x + b = 0$ 的距离 $r$。
1. 设 $x_p$ 为 $x$ 在超平面上的投影点。
2. $x$ 可以表示为 $x_p$ 加上沿法向量 $w$ 方向的一段距离：
   $$ x = x_p + r \frac{w}{\|w\|} $$
   其中 $|r|$ 即为距离，$\frac{w}{\|w\|}$ 是单位法向量。
3. 两边同时左乘 $w^\top$ 并加上 $b$：
   $$ w^\top x + b = w^\top (x_p + r \frac{w}{\|w\|}) + b $$
   $$ w^\top x + b = (w^\top x_p + b) + r \frac{w^\top w}{\|w\|} $$
4. 因为 $x_p$ 在超平面上，故 $w^\top x_p + b = 0$。且 $w^\top w = \|w\|^2$。
   $$ w^\top x + b = r \frac{\|w\|^2}{\|w\|} = r \|w\| $$
5. 解得距离（取绝对值）：
   $$ \text{distance} = |r| = \frac{|w^\top x + b|}{\|w\|} $$

### 2.2 间隔 (Margin) 的定义
对于所有训练样本，我们只关心分类正确的超平面，即 $y_i(w^\top x_i + b) > 0$。
样本集到超平面的**几何间隔** $\gamma$ 定义为所有样本点距离的最小值：
$$ \gamma = \min_i \frac{y_i(w^\top x_i + b)}{\|w\|} $$

### 2.3 优化目标构建
我们的目标是最大化这个最小间隔：
$$ \max_{w,b} \left( \min_i \frac{y_i(w^\top x_i + b)}{\|w\|} \right) $$

**缩放不变性 (Scaling Constraint)**:
超平面 $w^\top x + b = 0$ 与 $c w^\top x + c b = 0$ 是同一个平面。我们可以通过缩放 $(w, b)$ 来固定分子。
令离超平面最近的点的**函数间隔**为 1，即：
$$ \min_i y_i(w^\top x_i + b) = 1 $$
这意味着对于所有 $i$，都有 $y_i(w^\top x_i + b) \geq 1$，且至少有一个等号成立。

优化问题转化为：
$$ \max_{w,b} \frac{1}{\|w\|} \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1, \forall i $$

等价于最小化 $\|w\|^2$ (为了方便求导，通常写成 $\frac{1}{2}\|w\|^2$)：

$$
\begin{aligned}
\min_{w,b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y_i(w^\top x_i + b) \geq 1, \quad i = 1, \dots, m
\end{aligned}
$$
这就是 **硬间隔 SVM (Hard Margin SVM)** 的原始优化问题。

---

## 3. 推导 II: Hinge Loss 视角

### 3.1 逻辑回归 vs SVM 损失函数
*   **逻辑回归**: 使用对数损失 (Log Loss)。
    $$ J(w) = \sum \log(1 + e^{-y_i(w^\top x_i + b)}) $$
*   **SVM**: 使用 **Hinge Loss**。
    $$ \text{Loss}(z) = \max(0, 1 - z), \quad \text{where } z = y_i(w^\top x_i + b) $$
    *   如果 $y_i(w^\top x_i + b) \geq 1$（分类正确且有足够间隔），损失为 0。
    *   如果 $y_i(w^\top x_i + b) < 1$，损失随距离线性增加。

### 3.2 正则化目标函数
SVM 的目标函数可以看作是 Hinge Loss 加上 $L_2$ 正则化项：
$$ J(w, b) = C \sum_{i=1}^m \max(0, 1 - y_i(w^\top x_i + b)) + \frac{1}{2} \|w\|^2 $$
*   $C$ 是正则化系数的倒数。
*   如果要求所有样本分类正确且满足间隔（即 Loss 部分为 0），则等价于硬间隔 SVM：
    $$ \min \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad 1 - y_i(w^\top x_i + b) \leq 0 $$

---

## 4. 拉格朗日对偶性与 KKT 条件 (Lagrange Duality & KKT)

为了求解上述约束优化问题，我们使用拉格朗日乘子法。

### 4.1 一般形式
原问题 (Primal):
$$ \min_x f(x) \quad \text{s.t.} \quad h_i(x) \leq 0, \ l_j(x) = 0 $$
拉格朗日函数:
$$ L(x, u, v) = f(x) + \sum u_i h_i(x) + \sum v_j l_j(x) $$
对偶函数:
$$ g(u, v) = \min_x L(x, u, v) $$
对偶问题 (Dual):
$$ \max_{u, v} g(u, v) \quad \text{s.t.} \quad u \geq 0 $$

### 4.2 KKT 条件 (Karush-Kuhn-Tucker Conditions)
对于凸优化问题，最优解必须满足 KKT 条件：
1.  **平稳性 (Stationarity)**: $\nabla_x L = 0$
2.  **互补松弛性 (Complementary Slackness)**: $u_i h_i(x) = 0$
3.  **原始可行性 (Primal Feasibility)**: $h_i(x) \leq 0, l_j(x) = 0$
4.  **对偶可行性 (Dual Feasibility)**: $u_i \geq 0$

---

## 5. 利用对偶性优化 SVM

### 5.1 构造拉格朗日函数
原问题：
$$ \min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad 1 - y_i(w^\top x_i + b) \leq 0 $$
引入拉格朗日乘子 $\alpha_i \geq 0$：
$$ L(w, b, \alpha) = \frac{1}{2}\|w\|^2 + \sum_{i=1}^m \alpha_i \left( 1 - y_i(w^\top x_i + b) \right) $$

### 5.2 求解对偶问题
**第一步：最小化 $L(w, b, \alpha)$ 关于 $w, b$ (求偏导并令为 0)**
1.  针对 $w$:
    $$ \frac{\partial L}{\partial w} = w - \sum_{i=1}^m \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^m \alpha_i y_i x_i $$
2.  针对 $b$:
    $$ \frac{\partial L}{\partial b} = -\sum_{i=1}^m \alpha_i y_i = 0 \implies \sum_{i=1}^m \alpha_i y_i = 0 $$

**第二步：将 $w$ 和约束代回 $L$ 得到 $g(\alpha)$**
$$
\begin{aligned}
L(w, b, \alpha) &= \frac{1}{2} (\sum_i \alpha_i y_i x_i)^\top (\sum_j \alpha_j y_j x_j) + \sum_i \alpha_i - \sum_i \alpha_i y_i (w^\top x_i + b) \\
&= \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j + \sum_i \alpha_i - \sum_i \alpha_i y_i w^\top x_i - b \underbrace{\sum_i \alpha_i y_i}_{0} \\
&= \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j + \sum_i \alpha_i - \sum_i \alpha_i y_i (\sum_j \alpha_j y_j x_j)^\top x_i \\
&= \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j + \sum_i \alpha_i - \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j \\
&= \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^\top x_j
\end{aligned}
$$

**第三步：最大化对偶函数 (SVM 对偶问题)**
$$
\begin{aligned}
\max_\alpha \quad & \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j (x_i^\top x_j) \\
\text{s.t.} \quad & \alpha_i \geq 0, \quad \forall i \\
& \sum_{i=1}^m \alpha_i y_i = 0
\end{aligned}
$$

### 5.3 解的解释与支持向量
*   **支持向量 (Support Vectors)**: 根据互补松弛条件 $\alpha_i (1 - y_i(w^\top x_i + b)) = 0$。
    *   若 $\alpha_i > 0$，则必须有 $1 - y_i(w^\top x_i + b) = 0$，即 $y_i(w^\top x_i + b) = 1$。这些点位于间隔边界上，称为支持向量。
    *   若 $1 - y_i(w^\top x_i + b) < 0$（点在间隔外且分类正确），则必须有 $\alpha_i = 0$。这些点不影响模型构建。
*   **计算 $w$**: $w^* = \sum_{i \in S} \alpha_i y_i x_i$ ($S$ 是支持向量集合)。
*   **计算 $b$**: 对于任意支持向量 $x_j$ ($j \in S$)，有 $y_j(w^\top x_j + b) = 1$。利用 $y_j^2=1$ 两边乘以 $y_j$:
    $$ w^\top x_j + b = y_j \implies b = y_j - w^\top x_j $$
    为了数值稳定性，通常取所有支持向量计算出的 $b$ 的平均值：
    $$ b^* = \frac{1}{|S|} \sum_{j \in S} (y_j - \sum_{i \in S} \alpha_i y_i x_i^\top x_j) $$

---

## 6. 软间隔 SVM (SVM with Slack Variables)

当数据不可线性分时，引入松弛变量 $\xi_i$ 允许少量错误。

### 6.1 原始问题
$$
\begin{aligned}
\min_{w,b,\xi} \quad & \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \xi_i \\
\text{s.t.} \quad & y_i(w^\top x_i + b) \geq 1 - \xi_i, \quad \forall i \\
& \xi_i \geq 0, \quad \forall i
\end{aligned}
$$
*   $C$: 惩罚系数。$C$ 越大，对错误惩罚越重（越接近硬间隔）；$C$ 越小，容忍度越高。

### 6.2 拉格朗日函数
引入乘子 $\alpha_i \geq 0$ (对应分类约束) 和 $\mu_i \geq 0$ (对应 $\xi_i \geq 0$)：
$$ L(w, b, \xi, \alpha, \mu) = \frac{1}{2}\|w\|^2 + C\sum \xi_i + \sum \alpha_i(1 - \xi_i - y_i(w^\top x_i + b)) - \sum \mu_i \xi_i $$

### 6.3 KKT 推导与对偶问题
求导：
1.  $\frac{\partial L}{\partial w} = 0 \implies w = \sum \alpha_i y_i x_i$
2.  $\frac{\partial L}{\partial b} = 0 \implies \sum \alpha_i y_i = 0$
3.  $\frac{\partial L}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \implies \alpha_i = C - \mu_i$

由于 $\mu_i \geq 0$ 且 $\alpha_i \geq 0$，结合 $\alpha_i = C - \mu_i$，我们得到约束：
$$ 0 \leq \alpha_i \leq C $$

**软间隔对偶问题**:
形式与硬间隔完全相同，仅约束条件改变。
$$
\begin{aligned}
\max_\alpha \quad & \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j (x_i^\top x_j) \\
\text{s.t.} \quad & 0 \leq \alpha_i \leq C, \quad \forall i \\
& \sum_{i=1}^m \alpha_i y_i = 0
\end{aligned}
$$

### 6.4 $\alpha_i$ 的物理意义
*   $\alpha_i = 0$: 样本正确分类且在间隔外 ($\xi_i=0$)。
*   $0 < \alpha_i < C$: 样本在间隔边界上 ($\xi_i=0$)，是支持向量。
*   $\alpha_i = C$: 样本在间隔内 ($\xi_i > 0$)，可能是误分类或在间隔内但分类正确。

---

## 7. 核函数 SVM (SVM with Kernels)

### 7.1 思想
对于非线性可分数据，将其映射到高维特征空间 $\phi(x)$，使其在高维空间线性可分。
决策函数变为：$f(x) = w^\top \phi(x) + b$。

### 7.2 核技巧 (Kernel Trick)
观察对偶问题，数据仅以内积形式 $x_i^\top x_j$ 出现。
定义核函数 $k(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$。
我们不需要显式计算 $\phi(x)$，只需计算核函数。

**核化后的对偶问题**:
$$ \max_\alpha \sum \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j k(x_i, x_j) $$

**核化后的预测**:
$$ w^\top \phi(x) + b = \sum_{i=1}^m \alpha_i y_i k(x_i, x) + b $$

### 7.3 常用核函数
1.  **多项式核 (Polynomial Kernel)**:
    $$ k(x_i, x_j) = (x_i^\top x_j + 1)^d $$
2.  **高斯核 / 径向基核 (RBF Kernel)**:
    $$ k(x_i, x_j) = \exp\left( -\frac{\|x_i - x_j\|^2}{2\sigma^2} \right) = \exp(-\gamma \|x_i - x_j\|^2) $$
    *   这是最常用的核，对应无穷维特征空间。
3.  **Sigmoid 核**:
    $$ k(x_i, x_j) = \tanh(\kappa x_i^\top x_j + \theta) $$

---

## 8. 其他要点

### 8.1 多分类 SVM
*   **One-vs-Rest (One-vs-All)**: 训练 $K$ 个二分类器。第 $k$ 个分类器将第 $k$ 类设为正，其余为负。预测时取 $w^{(k)\top}x + b^{(k)}$ 最大的类别。
*   **One-vs-One**: 训练 $K(K-1)/2$ 个分类器，两两投票。

### 8.2 SVM vs 逻辑回归
| 特性 | SVM | 逻辑回归 (Logistic Regression) |
| :--- | :--- | :--- |
| **损失函数** | Hinge Loss (忽略远离边界的正确样本) | Log Loss (所有样本都贡献损失) |
| **概率输出** | 无直接概率 (需 Platt Scaling) | 有直接概率输出 |
| **核函数** | 易于结合核函数 (对偶形式高效) | 较难结合核函数 (计算量大) |
| **适用场景** | $n$ (特征) 大 $m$ (样本) 小; 或非线性问题 | $n$ 小 $m$ 大 (需手动特征工程) |
| **稀疏性** | 解是稀疏的 (仅依赖支持向量) | 解是非稀疏的 (依赖所有数据) |

### 8.3 实践建议
*   使用现成库 (如 libsvm, sklearn)。
*   **特征缩放 (Feature Scaling)**: 使用高斯核前必须进行归一化/标准化，否则距离计算会被大尺度特征主导。
*   **参数选择**: 需调节 $C$ (正则化) 和核参数 (如 RBF 的 $\sigma$ 或 $\gamma$)。