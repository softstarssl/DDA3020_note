# DDA3020 Lecture 06: 逻辑回归 (Logistic Regression)

## 1. 背景回顾 (Review)

在进入逻辑回归之前，简要回顾线性回归的关键点，以便对比。

*   **线性假设**: $f_w(x) = w^\top x$ (包含偏置项 $w_0$ 在增广向量中)。
*   **优化目标**: 最小化残差平方和 (RSS) 或均方误差 (MSE)。
    $$ J(w) = \frac{1}{2m} \sum_{i=1}^m (w^\top x_i - y_i)^2 $$
*   **求解方法**:
    *   闭式解 (Closed-form): $w^* = (X^\top X)^{-1}X^\top y$
    *   梯度下降 (Gradient Descent): $w \leftarrow w - \eta X^\top(Xw - y)$
*   **概率视角**: 假设噪声服从高斯分布 $y = w^\top x + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$，则 MLE 等价于最小二乘法。

---

## 2. 分类问题与表示 (Classification and Representation)

### 2.1 为什么不用线性回归做分类？
对于二分类问题 $y \in \{0, 1\}$：
1.  **输出范围**: 线性回归 $f_w(x) = w^\top x$ 的输出范围是 $(-\infty, +\infty)$，而我们需要的是概率值 $[0, 1]$。
2.  **对异常值敏感**: 引入一个非常大的正样本（outlier）会显著改变回归直线，导致决策边界偏移，从而错误分类正常的样本。
3.  **性质不符**: 我们希望得到的是属于正类的“概率”。

### 2.2 假设函数 (Hypothesis Representation)
为了将输出限制在 $[0, 1]$ 之间，引入 **Sigmoid 函数** (或 Logistic 函数)：
$$ g(z) = \frac{1}{1 + e^{-z}} $$

**逻辑回归模型**:
$$ f_w(x) = g(w^\top x) = \frac{1}{1 + e^{-w^\top x}} $$

**参数解释**:
*   $w$: 权重向量 (包含偏置 $w_0$)。
*   $x$: 特征向量 (增广了 $x_0=1$)。

**概率解释**:
模型输出被解释为给定输入 $x$ 下，$y=1$ 的条件概率：
$$ f_w(x) = P(y=1 | x; w) $$
由于只有两类，则 $P(y=0 | x; w) = 1 - f_w(x)$。

### 2.3 决策边界 (Decision Boundary)
*   当 $f_w(x) \geq 0.5$ 时，预测 $y=1$。即 $w^\top x \geq 0$。
*   当 $f_w(x) < 0.5$ 时，预测 $y=0$。即 $w^\top x < 0$。
*   **决策边界**: 由方程 $w^\top x = 0$ 定义。
    *   它可以是线性的（线性决策边界）。
    *   如果引入多项式特征（如 $x_1^2, x_1 x_2$），决策边界可以是非线性的（圆形、椭圆等）。

---

## 3. 代价函数 (Cost Function)

### 3.1 为什么不能用均方误差 (MSE)?
如果直接将 Sigmoid 函数代入 MSE 公式：
$$ J(w) = \frac{1}{2m} \sum_{i=1}^m (\frac{1}{1 + e^{-w^\top x_i}} - y_i)^2 $$
由于 Sigmoid 函数的非线性，会导致 $J(w)$ 关于 $w$ 是 **非凸 (Non-convex)** 的，存在许多局部极小值，不利于梯度下降求解。

### 3.2 交叉熵损失 (Cross-Entropy Loss)
我们采用基于极大似然估计推导出的交叉熵损失函数。对于单个样本 $(x, y)$：
$$ \text{cost}(f_w(x), y) = \begin{cases} -\log(f_w(x)) & \text{if } y=1 \\ -\log(1 - f_w(x)) & \text{if } y=0 \end{cases} $$

*   **直观理解**:
    *   若 $y=1$，预测 $f_w(x) \to 1$，损失 $\to 0$；若 $f_w(x) \to 0$，损失 $\to \infty$（惩罚极大）。
    *   若 $y=0$，预测 $f_w(x) \to 0$，损失 $\to 0$；若 $f_w(x) \to 1$，损失 $\to \infty$。

**统一形式**:
利用 $y \in \{0, 1\}$ 的特性，可以将上述分段函数合并：
$$ \text{cost}(f_w(x), y) = -y \log(f_w(x)) - (1-y) \log(1 - f_w(x)) $$

**整体代价函数**:
$$ J(w) = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \log(f_w(x_i)) + (1 - y_i) \log(1 - f_w(x_i)) \right] $$
*此函数关于 $w$ 是凸函数 (Convex)，保证梯度下降能收敛到全局最优解。*

---

## 4. 梯度下降求解 (Gradient Descent)

### 4.1 Sigmoid 函数求导预备
令 $g(z) = \frac{1}{1+e^{-z}} = (1+e^{-z})^{-1}$。
$$
\begin{aligned}
g'(z) &= -(1+e^{-z})^{-2} \cdot (-e^{-z}) \\
&= \frac{e^{-z}}{(1+e^{-z})^2} \\
&= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} \\
&= \frac{1}{1+e^{-z}} \cdot (1 - \frac{1}{1+e^{-z}}) \\
&= g(z)(1 - g(z))
\end{aligned}
$$

### 4.2 代价函数梯度推导 (详细步骤)
我们需要计算 $\frac{\partial J(w)}{\partial w_j}$。
针对单个样本的损失 $L = -y \log(a) - (1-y) \log(1-a)$，其中 $a = f_w(x) = g(z)$，$z = w^\top x$。

利用链式法则：$\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_j}$

1.  **第一部分** $\frac{\partial L}{\partial a}$:
    $$ \frac{\partial L}{\partial a} = -\frac{y}{a} - \frac{1-y}{1-a} \cdot (-1) = \frac{a(1-y) - y(1-a)}{a(1-a)} = \frac{a-ay-y+ay}{a(1-a)} = \frac{a-y}{a(1-a)} $$
2.  **第二部分** $\frac{\partial a}{\partial z}$ (Sigmoid 导数):
    $$ \frac{\partial a}{\partial z} = a(1-a) $$
3.  **第三部分** $\frac{\partial z}{\partial w_j}$:
    $$ z = w_0 x_0 + \dots + w_j x_j + \dots \implies \frac{\partial z}{\partial w_j} = x_j $$

**合并**:
$$
\frac{\partial L}{\partial w_j} = \left( \frac{a-y}{a(1-a)} \right) \cdot (a(1-a)) \cdot x_j = (a - y) x_j = (f_w(x) - y)x_j
$$

### 4.3 梯度下降更新规则
对于所有 $m$ 个样本的平均梯度：
$$ \frac{\partial J(w)}{\partial w} = \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_i $$

**更新公式**:
$$ w := w - \eta \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_i $$
其中 $\eta$ 是学习率。
*注意：这与线性回归的更新规则形式上完全一致，区别仅在于 $f_w(x)$ 的定义不同（线性回归是 $w^\top x$，逻辑回归是 Sigmoid）。*

---

## 5. 多分类问题 (Multi-class Classification)

当 $y \in \{1, \dots, C\}$ 时。

### 5.1 一对多 (One-vs-All / One-vs-Rest)
*   **方法**: 训练 $C$ 个二分类逻辑回归模型。对于第 $j$ 类，将该类样本设为正例 ($y=1$)，其余所有类设为负例 ($y=0$)。
*   **模型**: 得到 $f_{w^{(1)}}, \dots, f_{w^{(C)}}$。
*   **预测**: 选择概率最大的类别。
    $$ \hat{y} = \arg\max_{j} f_{w^{(j)}}(x) $$

### 5.2 Softmax 回归 (Softmax Regression)
直接处理多分类问题。
*   **假设函数**:
    $$ P(y=j | x; W) = \frac{e^{w_j^\top x}}{\sum_{c=1}^C e^{w_c^\top x}} $$
    其中 $W = [w_1, \dots, w_C]$ 是参数矩阵。
*   **代价函数**:
    $$ J(W) = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^C I(y_i = j) \log \left( \frac{e^{w_j^\top x_i}}{\sum_{c=1}^C e^{w_c^\top x_i}} \right) $$
    其中 $I(\cdot)$ 是指示函数，当条件为真时为 1，否则为 0。
*   **梯度**:
    $$ \nabla_{w_j} J(W) = \frac{1}{m} \sum_{i=1}^m [ P(y=j|x_i; W) - I(y_i=j) ] x_i $$

---

## 6. 正则化逻辑回归 (Regularized Logistic Regression)

为了防止过拟合（Overfitting），在代价函数中加入正则项。

### 6.1 代价函数 ($L_2$ 正则化)
$$ J_{reg}(w) = J(w) + \frac{\lambda}{2m} \sum_{j=1}^d w_j^2 $$
*   **注意**: 通常不对偏置项 $w_0$ 进行正则化，求和从 $j=1$ 开始。
*   $\lambda$: 正则化参数。$\lambda$ 越大，惩罚越重，模型越简单（欠拟合风险）；$\lambda$ 越小，模型越复杂（过拟合风险）。

### 6.2 梯度下降更新
*   对于 $w_0$ (不正则化):
    $$ w_0 := w_0 - \eta \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_{i,0} $$
*   对于 $w_j$ ($j = 1, \dots, d$):
    $$
    \begin{aligned}
    w_j &:= w_j - \eta \left[ \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_{i,j} + \frac{\lambda}{m} w_j \right] \\
    &:= w_j (1 - \eta \frac{\lambda}{m}) - \eta \frac{1}{m} \sum_{i=1}^m (f_w(x_i) - y_i) x_{i,j}
    \end{aligned}
    $$
    *(项 $1 - \eta \frac{\lambda}{m}$ 通常小于 1，体现了权重衰减 Weight Decay 的效果)*

---

## 7. 概率视角 (Probabilistic Perspective)

逻辑回归可以从概率建模的角度推导出来。

### 7.1 假设
假设 $y$ 服从 **伯努利分布 (Bernoulli Distribution)**:
$$ y | x; w \sim \text{Bernoulli}(\mu) $$
其中 $\mu = f_w(x) = \text{Sigmoid}(w^\top x)$。
概率质量函数为:
$$ P(y | x; w) = \mu^y (1-\mu)^{1-y} $$

### 7.2 极大似然估计 (MLE)
似然函数 $L(w)$ 为所有样本概率的乘积：
$$ L(w) = \prod_{i=1}^m P(y_i | x_i; w) = \prod_{i=1}^m (f_w(x_i))^{y_i} (1 - f_w(x_i))^{1-y_i} $$

对数似然 (Log-Likelihood):
$$ \log L(w) = \sum_{i=1}^m [ y_i \log(f_w(x_i)) + (1-y_i) \log(1 - f_w(x_i)) ] $$

最大化对数似然 $\max_w \log L(w)$ 等价于最小化负对数似然，即最小化交叉熵代价函数 $J(w)$。
$$ \max_w L(w) \iff \min_w J(w) $$

### 7.3 正则化与先验分布 (Priors)
*   **$L_2$ 正则化**: 对应于假设参数 $w$ 服从 **高斯先验 (Gaussian Prior)** $w \sim \mathcal{N}(0, \tau^2 I)$ 的最大后验估计 (MAP)。
*   **$L_1$ 正则化**: 对应于假设参数 $w$ 服从 **拉普拉斯先验 (Laplace Prior)** $w \sim \text{Laplace}(0, b)$ 的 MAP。

---

## 8. 总结：线性回归 vs 逻辑回归

| 特性       | 线性回归 (Linear Regression)    | 逻辑回归 (Logistic Regression)           |     |     |
| :------- | :-------------------------- | :----------------------------------- | --- | --- |
| **任务**   | 回归 (预测连续值)                  | 分类 (预测离散值/概率)                        |     |     |
| **假设函数** | $f_w(x) = w^\top x$         | $f_w(x) = \frac{1}{1+e^{-w^\top x}}$ |     |     |
| **输出范围** | $(-\infty, +\infty)$        | $[0, 1]$                             |     |     |
| **代价函数** | 均方误差 (MSE)                  | 交叉熵 (Cross-Entropy)                  |     |     |
| **凸性**   | 总是凸函数                       | 交叉熵下是凸函数 (MSE下非凸)                    |     |     |
| **求解方法** | 解析解 或 梯度下降                  | 梯度下降 (无闭式解)                          |     |     |
| **概率模型** | 高斯分布 $y\|x\sim \mathcal{N}$ | 伯努利分布 $y\|x \sim \text{Bernoulli}$   |     |     |
|          |                             |                                      |     |     |
