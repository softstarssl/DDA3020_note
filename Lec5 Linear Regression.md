# DDA3020 Lecture 05: 线性回归 (Linear Regression)

## 1. 符号与数学基础 (Notations & Math Basics)

### 1.1 线性函数与仿射函数
*   **线性函数 (Linear Function)**: 满足叠加性（齐次性 + 可加性）。
    $$ f(\alpha x + \beta y) = \alpha f(x) + \beta f(y) $$
    形式通常为内积：$f(x) = a^\top x$。
*   **仿射函数 (Affine Function)**: 线性函数加上一个偏置（offset）。
    $$ f(x) = a^\top x + b $$
    *注意：在机器学习中，我们通常通过扩展特征向量（增加一维 1）将仿射函数转化为线性函数的形式处理。*

### 1.2 矩阵微积分 (Matrix Calculus)
采用分母布局 (Denominator layout) 的常用求导公式：
1.  **向量对向量**: 若 $f(w) = X^\top w$（$X$ 与 $w$ 无关），则：
    $$ \frac{d(X^\top w)}{dw} = X $$
2.  **标量对向量**: 若 $f(w) = y^\top X w$，则：
    $$ \frac{d(y^\top X w)}{dw} = X^\top y $$
3.  **二次型对向量**: 若 $f(w) = w^\top A w$（$A$ 对称），则：
    $$ \frac{d(w^\top A w)}{dw} = 2Aw $$
    *   推导特例：对于线性回归中的 $w^\top X^\top X w$，令 $A = X^\top X$（对称阵），则导数为 $2X^\top X w$。

---

## 2. 线性回归建模 (Modeling)

### 2.1 确定性视角 (Deterministic Perspective)
*   **假设函数 (Hypothesis)**:
    $$ f_w(x) = w_0 + w_1 x_1 + \dots + w_d x_d = w^\top x $$
    其中 $x = [1, x_1, \dots, x_d]^\top$ 是增广特征向量。
*   **损失函数 (Loss Function)**: 衡量单个样本预测误差。
    $$ L(f_w(x_i), y_i) = (f_w(x_i) - y_i)^2 \quad (\text{Squared Error Loss}) $$
*   **目标函数 (Cost Function)**: 所有样本的平均损失（经验风险）。
    $$ J(w) = \frac{1}{m} \sum_{i=1}^m (w^\top x_i - y_i)^2 $$

### 2.2 概率视角 (Probabilistic Perspective)
假设输入 $x$ 与输出 $y$ 的关系包含观测噪声 $\epsilon$：
$$ y = w^\top x + \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \sigma^2) $$
这意味着给定 $x$ 和 $w$， $y$ 服从正态分布：
$$ p(y|x, w) = \mathcal{N}(w^\top x, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y - w^\top x)^2}{2\sigma^2} \right) $$

#### 极大似然估计 (MLE) 推导
我们需要找到 $w$ 使得观测数据出现的概率（似然）最大。
1.  **似然函数**:
    $$ L(w; D) = \prod_{i=1}^m p(y_i | x_i, w) $$
2.  **对数似然 (Log-Likelihood)**:
    $$ \log L(w; D) = \sum_{i=1}^m \log \left( \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y_i - w^\top x_i)^2}{2\sigma^2} \right) \right) $$
3.  **化简**:
    $$ \log L(w; D) = \sum_{i=1}^m \left[ \log(\frac{1}{\sqrt{2\pi}\sigma}) - \frac{1}{2\sigma^2}(y_i - w^\top x_i)^2 \right] $$
    $$ \log L(w; D) = m \log(\dots) - \frac{1}{2\sigma^2} \sum_{i=1}^m (y_i - w^\top x_i)^2 $$
4.  **最大化**:
    最大化 $\log L$ 等价于最小化 $\sum (y_i - w^\top x_i)^2$。
    **结论**: 高斯噪声假设下的极大似然估计 (MLE) 等价于最小二乘法 (Least Squares)。

---

## 3. 线性回归求解 (Learning)

定义数据矩阵 $X \in \mathbb{R}^{m \times (d+1)}$ 和 目标向量 $y \in \mathbb{R}^m$。
$$ J(w) = \sum_{i=1}^m (x_i^\top w - y_i)^2 = \| Xw - y \|_2^2 = (Xw - y)^\top (Xw - y) $$

### 3.1 解析解 (Analytical Solution / Closed-form)
**详细推导**:
1.  展开目标函数:
    $$
    \begin{aligned}
    J(w) &= (w^\top X^\top - y^\top)(Xw - y) \\
    &= w^\top X^\top X w - w^\top X^\top y - y^\top X w + y^\top y
    \end{aligned}
    $$
    注意 $w^\top X^\top y$ 是标量，其转置等于自身，即 $w^\top X^\top y = (w^\top X^\top y)^\top = y^\top X w$。
    $$ J(w) = w^\top X^\top X w - 2y^\top X w + y^\top y $$
2.  对 $w$ 求导并令其为 0:
    $$ \nabla_w J(w) = \frac{\partial (w^\top X^\top X w)}{\partial w} - \frac{\partial (2y^\top X w)}{\partial w} $$
    利用矩阵微积分公式：
    $$ \nabla_w J(w) = 2X^\top X w - 2X^\top y = 0 $$
3.  求解方程:
    $$ X^\top X w = X^\top y $$
    若 $X^\top X$ 可逆，则：
    $$ w^* = (X^\top X)^{-1} X^\top y $$

### 3.2 梯度下降法 (Gradient Descent)
当特征维度 $d$ 很大时，计算 $(X^\top X)^{-1}$ 的复杂度 $O(d^3)$ 太高，改用迭代法。
*   **更新规则**:
    $$ w \leftarrow w - \eta \nabla J(w) $$
*   **梯度**:
    $$ \nabla J(w) = X^\top (Xw - y) $$
*   **复杂度对比**:
    *   解析解: $O(d^3 + md^2)$，适合小 $d$。
    *   梯度下降: $O(T \cdot md)$，适合大 $d$ ($T$ 为迭代次数)。

---

## 4. 扩展应用 (Extensions)

### 4.1 多输出线性回归 (Multiple Outputs)
*   **场景**: 目标 $Y \in \mathbb{R}^{m \times h}$ (每个样本有 $h$ 个输出)。
*   **参数**: $W \in \mathbb{R}^{(d+1) \times h}$。
*   **损失函数**: 迹范数 (Trace of error matrix)。
    $$ J(W) = \text{trace}((XW - Y)^\top (XW - Y)) $$
*   **解析解**:
    $$ W^* = (X^\top X)^{-1} X^\top Y $$

### 4.2 线性回归用于分类 (Classification)
*   **二分类**: $y_i \in \{-1, +1\}$。
    *   训练: 视为回归问题求解 $w^*$。
    *   预测: $y_{pred} = \text{sgn}(x_{new}^\top w^*)$。
*   **多分类**: $Y$ 使用 One-hot 编码。
    *   训练: 求解 $W^*$。
    *   预测: $y_{pred} = \arg\max_{k} (x_{new}^\top W^*)_k$。

---

## 5. 线性回归变体 (Variants)

### 5.1 岭回归 (Ridge Regression) - $L_2$ 正则化
**动机**:
1.  解决 $X^\top X$ 不可逆（奇异矩阵）的问题（如特征多重共线性）。
2.  防止过拟合（参数值过大导致模型对输入微小变化敏感）。

**目标函数**:
$$ J(w) = (Xw - y)^\top (Xw - y) + \lambda \|w\|^2 $$
*(注：通常不对偏置 $w_0$ 进行正则化，令 $I_d$ 为首个对角元素为0的单位阵)*

**解析解推导**:
1.  求导:
    $$ \nabla_w J(w) = 2X^\top X w - 2X^\top y + 2\lambda I_d w = 0 $$
2.  整理:
    $$ (X^\top X + \lambda I_d) w = X^\top y $$
3.  结果:
    $$ w^* = (X^\top X + \lambda I_d)^{-1} X^\top y $$
    *性质: 只要 $\lambda > 0$，$(X^\top X + \lambda I)$ 总是可逆的。*

**概率视角 (MAP)**:
假设参数 $w$ 服从均值为 0 的高斯先验：$p(w) \sim \mathcal{N}(0, \tau^2 I)$。
$$
\begin{aligned}
w_{MAP} &= \arg\max_w [\log p(y|X, w) + \log p(w)] \\
&= \arg\max_w [-\frac{1}{2\sigma^2}\sum(y_i - w^\top x_i)^2 - \frac{1}{2\tau^2}w^\top w] \\
&\equiv \arg\min_w [\sum(y_i - w^\top x_i)^2 + \lambda \|w\|_2^2] \quad (\text{其中 } \lambda = \frac{\sigma^2}{\tau^2})
\end{aligned}
$$

### 5.2 Lasso 回归 - $L_1$ 正则化
**特点**: 产生稀疏解（某些 $w_j$ 变为 0），用于特征选择。
**概率视角**:
假设参数 $w$ 服从拉普拉斯先验 (Laplacian Prior)：
$$ p(w) = \frac{1}{2b} \exp\left(-\frac{\|w\|_1}{b}\right) $$
**目标函数**:
$$ w_{MAP} = \arg\min_w \sum (y_i - w^\top x_i)^2 + \lambda \|w\|_1 $$
*注: 由于 $|w|$ 在 0 处不可导，无法得到闭式解，通常使用坐标下降法或转化为线性规划求解。*

### 5.3 多项式回归 (Polynomial Regression)
**核心思想**: 线性模型无法处理非线性数据（如 XOR 问题）。通过基函数扩展 (Basis Expansion) 将低维非线性问题映射到高维线性空间。
*   **映射**: $\phi(x) = [1, x_1, x_2, x_1 x_2, x_1^2, x_2^2, \dots]^\top$。
*   **模型**: $f(x) = w^\top \phi(x)$。
*   **本质**: 对参数 $w$ 仍是线性的，因此仍属于线性回归家族。

### 5.4 鲁棒回归 (Robust Linear Regression)
**动机**: 最小二乘法 ($L_2$ Loss) 对异常值 (Outliers) 非常敏感（误差平方放大影响）。
**方法**:
1.  **$L_1$ Loss (Least Absolute Deviations)**:
    $$ J(w) = \sum |x_i^\top w - y_i| $$
    对应噪声服从拉普拉斯分布。
2.  **Huber Loss**: 在误差较小时用 $L_2$，误差较大时用 $L_1$。

**求解 $L_1$ 回归**:
由于不可导，可转化为**迭代重加权最小二乘法 (Iteratively Reweighted Least Squares, IRLS)**:
$$ w^{(k+1)} = \arg\min_w \sum \frac{1}{2} \frac{(x_i^\top w - y_i)^2}{\mu_i}, \quad \text{where } \mu_i = |x_i^\top w^{(k)} - y_i| $$

---

## 6. 总结：先验与正则化的对应关系

| 似然分布 $p(y|x,w)$ | 先验分布 $p(w)$ | 回归方法 | 损失函数 + 正则项 |
| :--- | :--- | :--- | :--- |
| Gaussian | Uniform | 最小二乘 (Least Squares) | $L_2$ Loss |
| Gaussian | Gaussian | 岭回归 (Ridge) | $L_2$ Loss + $L_2$ Reg |
| Gaussian | Laplace | Lasso 回归 | $L_2$ Loss + $L_1$ Reg |
| Laplace | Uniform | 鲁棒回归 (Robust) | $L_1$ Loss |