# DDA3020 Machine Learning - Lecture 12: Over/Under-Fitting and Bias-Variance Trade-off

**讲师**: Juexiao Zhou (School of Data Science, CUHK-SZ)
**日期**: Nov 4/6, 2025

---

## 1. 过拟合、欠拟合与模型复杂度 (Overfitting, Underfitting and Model Complexity)

### 1.1 学习的目标
机器学习的核心目标是 **预测 (Prediction)**。
*   **学习过程**: 通过训练数据求解参数。
    *   线性回归: $w_b = (X^T X)^{-1} X^T y$
    *   多项式回归: $w_b = (P^T P)^{-1} P^T y$
*   **预测过程**: 将新数据代入模型。
    *   $f_{w,b}(X_{new}) = X_{new} w_b$

### 1.2 欠拟合 (Underfitting)
*   **定义**: 模型无法很好地预测其训练数据的标签。即训练误差很高。
*   **主要原因**:
    1.  **模型太简单**: 例如用线性模型拟合非线性数据。
    2.  **特征信息量不足**: 特征工程做得不够好。
*   **解决方案**:
    *   尝试更复杂的模型（如增加多项式阶数）。
    *   构建具有更高预测能力的特征。
*   **表现**: 高偏差 (High Bias)，低方差 (Low Variance)。

### 1.3 过拟合 (Overfitting)
*   **定义**: 模型在训练数据上表现极好，但在测试数据上表现很差。泛化能力弱。
*   **主要原因**:
    1.  **模型太复杂**: 例如决策树太深、神经网络太深或太宽、多项式阶数过高（如9阶多项式拟合少量点）。
    2.  **特征过多但训练样本过少**。
*   **解决方案**:
    *   增加训练数据量。
    *   降低模型复杂度（正则化、剪枝等）。
*   **表现**: 低偏差 (Low Bias)，高方差 (High Variance)。

### 1.4 模型复杂度与拟合的关系
*   **低复杂度 (如线性/1阶)**: 欠拟合，训练误差高，测试误差高。
*   **中等复杂度 (如2-4阶)**: 拟合良好 (Good fit)，训练误差适中，测试误差低。
*   **高复杂度 (如9阶)**: 过拟合，训练误差极低（甚至为0），测试误差极高。

---

## 2. 偏差-方差权衡 (Bias-Variance Trade-off)

### 2.1 实验观察 (Experimental Observations)
随着模型复杂度（Model Complexity）的增加：
1.  **训练误差 (Training Error)**: 持续下降，趋近于0。
2.  **测试误差 (Testing Error)**: 先下降后上升（呈U型曲线）。
    *   **低复杂度区域**: 测试误差高 $\rightarrow$ 高偏差，低方差。
    *   **高复杂度区域**: 测试误差高 $\rightarrow$ 低偏差，高方差。
    *   **最佳点**: 偏差与方差的平衡点，测试误差最低。

### 2.2 统计学分析 (Statistical Analysis) - **核心推导**

#### 2.2.1 问题设定
*   **训练集**: $D = \{(x_i, y_i)\}_{i=1}^n$，独立同分布 (i.i.d.) 采样自分布 $P(X, Y)$。
*   **真实关系**:
    $$ y = t(x) + \epsilon $$
    *   $t(x)$: 未知的真实目标函数 (Target Function)，即 $E[y|x]$。
    *   $\epsilon$: 噪声，服从正态分布 $\epsilon \sim \mathcal{N}(0, \sigma^2)$。
*   **模型学习**: 算法 $A$ 基于数据集 $D$ 学习到的假设函数为 $h_D(x) = A(D)$。

#### 2.2.2 期望假设与期望测试误差
*   **期望假设函数 (Expected Hypothesis)**:
    $$ \bar{h}(x) = E_{D \sim P^n}[h_D(x)] $$
    *解释*: 这是在无数个不同训练集上训练出的模型的平均预测结果。

*   **期望测试误差 (Expected Test Error)**:
    我们需要评估算法 $A$ 在特定测试样本 $(x, y)$ 上的表现，并对所有可能的训练集 $D$ 求期望：
    $$ E_{(x,y)\sim P, D\sim P^n} \left[ (h_D(x) - y)^2 \right] $$

#### 2.2.3 误差分解推导 (Decomposition Derivation)
我们将误差项 $(h_D(x) - y)^2$ 进行分解。为了简化符号，省略下标，默认期望是对 $D$ 和 $(x,y)$ 进行的。

**第一步：引入 $\bar{h}(x)$ 进行分解**
$$
\begin{aligned}
E[(h_D(x) - y)^2] &= E[(h_D(x) - \bar{h}(x) + \bar{h}(x) - y)^2] \\
&= E[\underbrace{(h_D(x) - \bar{h}(x))^2}_{A^2} + \underbrace{(\bar{h}(x) - y)^2}_{B^2} + \underbrace{2(h_D(x) - \bar{h}(x))(\bar{h}(x) - y)}_{2AB}]
\end{aligned}
$$

**第二步：分析交叉项 (Cross-term)**
$$
\begin{aligned}
E_{D, (x,y)} [(h_D(x) - \bar{h}(x))(\bar{h}(x) - y)] &= E_{(x,y)} [ E_D [h_D(x) - \bar{h}(x)] \cdot (\bar{h}(x) - y) ] \\
&= E_{(x,y)} [ (\underbrace{E_D[h_D(x)]}_{\bar{h}(x)} - \bar{h}(x)) \cdot (\bar{h}(x) - y) ] \\
&= E_{(x,y)} [ 0 \cdot (\bar{h}(x) - y) ] = 0
\end{aligned}
$$
*注*: 因为 $\bar{h}(x)$ 是常数（相对于 $D$ 的期望而言），且 $y$ 与 $D$ 无关。

因此，误差简化为：
$$ E[(h_D(x) - y)^2] = E[(h_D(x) - \bar{h}(x))^2] + E[(\bar{h}(x) - y)^2] $$

**第三步：进一步分解第二项 $E[(\bar{h}(x) - y)^2]$**
引入真实目标函数 $t(x)$：
$$
\begin{aligned}
E[(\bar{h}(x) - y)^2] &= E[(\bar{h}(x) - t(x) + t(x) - y)^2] \\
&= E[(\bar{h}(x) - t(x))^2] + E[(t(x) - y)^2] + 2E[(\bar{h}(x) - t(x))(t(x) - y)]
\end{aligned}
$$

**第四步：分析新的交叉项**
$$
\begin{aligned}
E[(\bar{h}(x) - t(x))(t(x) - y)] &= E_x [ E_{y|x} [ (\bar{h}(x) - t(x))(t(x) - y) ] ] \\
&= E_x [ (\bar{h}(x) - t(x)) (t(x) - \underbrace{E_{y|x}[y]}_{t(x)}) ] \\
&= E_x [ (\bar{h}(x) - t(x)) \cdot 0 ] = 0
\end{aligned}
$$
*注*: $t(x)$ 是 $y$ 的真实均值，即 $t(x) = E[y|x]$。

#### 2.2.4 最终公式与解释
综合上述步骤，期望测试误差分解为三项：

$$
\underbrace{E[(h_D(x) - y)^2]}_{\text{Total Error}} = \underbrace{E_D[(h_D(x) - \bar{h}(x))^2]}_{\text{Variance}} + \underbrace{E_x[(\bar{h}(x) - t(x))^2]}_{\text{Bias}^2} + \underbrace{E_{x,y}[(t(x) - y)^2]}_{\text{Noise}}
$$

1.  **方差 (Variance)**: $E_D[(h_D(x) - \bar{h}(x))^2]$
    *   **含义**: 描述了模型对训练数据集 $D$ 的敏感程度。如果你换一个训练集，预测结果变化有多大？
    *   **关联**: 高方差 $\leftrightarrow$ 过拟合 (Over-specialized)。
2.  **偏差平方 (Bias$^2$)**: $E_x[(\bar{h}(x) - t(x))^2]$
    *   **含义**: 即使有无限的数据，模型预测的平均值 $\bar{h}(x)$ 与真实值 $t(x)$ 之间的固有差距。这是由模型本身的假设（如线性假设）决定的。
    *   **关联**: 高偏差 $\leftrightarrow$ 欠拟合 (Underfitting)。
3.  **噪声 (Noise)**: $E_{x,y}[(t(x) - y)^2] = \sigma^2$
    *   **含义**: 数据本身的固有噪声。
    *   **性质**: 不可约减误差 (Irreducible Error)，这是性能的上限，无法通过优化模型来消除。

---

### 2.3 实际应用与分析

#### 2.3.1 权衡 (The Trade-off)
*   **模型复杂度 $\uparrow$**:
    *   方差 (Variance) $\uparrow$ (不同数据集训练出的模型差异变大)。
    *   偏差 (Bias) $\downarrow$ (平均预测更接近真实值)。
*   **总误差**: 先减后增，存在一个最优复杂度。

#### 2.3.2 典型模型分析
*   **决策树 (Decision Trees)**:
    *   **单棵剪枝树 (Pruned Tree)**: 高偏差，低方差 (欠拟合)。
    *   **单棵深树 (Deep Tree)**: 低偏差，高方差 (过拟合)。
*   **随机森林 (Random Forests)**:
    *   引入数据随机性（Bagging）和特征随机性。
    *   **效果**: 显著降低 **方差**，但不能保证降低偏差（偏差通常与单棵树相似）。
*   **Boosting**: 可以降低偏差。

#### 2.3.3 两种状态 (Regimes) 与对策
1.  **Regime 1: 高方差 (High Variance / Overfitting)**
    *   **症状**: 训练误差 $\ll$ 测试误差；训练误差很低，测试误差高。
    *   **对策**:
        *   增加训练样本 (Add more training instances)。
        *   降低模型复杂度 (Reduce model complexity)。
2.  **Regime 2: 高偏差 (High Bias / Underfitting)**
    *   **症状**: 训练误差本身就很高。
    *   **对策**:
        *   增加特征 (Add more features)。
        *   使用更复杂的模型 (非线性模型、核方法等)。

---

### 2.4 计算练习 (Exercise Calculation)

**题目设定**:
*   真实模型: $y = t(x) + \epsilon, \quad t(x=5)=9.5, \quad \epsilon \sim \mathcal{N}(0, 0.5)$ (即噪声方差 $\sigma^2=0.5$, 但本题中具体样本的 $\epsilon$ 给定为 0.5)。
*   测试样本: $(x, y) = (5, 10)$ (因为 $9.5 + 0.5 = 10$)。
*   10个模型对 $x=5$ 的预测值: $\{9, 11, 23, 6, 8, 12, 10, 4, 13, 7\}$。

**计算目标**: Empirical MSE, Bias$^2$, Variance。

**计算过程**:
1.  **平均预测 ($\bar{h}(x)$)**:
    $$ \bar{h} = \frac{9+11+23+6+8+12+10+4+13+7}{10} = 10.3 $$
2.  **偏差平方 (Bias$^2$)**:
    $$ (\bar{h}(x) - t(x))^2 = (10.3 - 9.5)^2 = 0.8^2 = 0.64 $$
3.  **方差 (Variance)**:
    $$ \text{Var} = \frac{1}{10} \sum_{i=1}^{10} (h_i(x) - \bar{h}(x))^2 $$
    $$ = \frac{1}{10} [(9-10.3)^2 + \dots + (7-10.3)^2] = 24.81 $$
4.  **经验均方误差 (Empirical MSE)**:
    $$ \text{MSE} = \frac{1}{10} \sum_{i=1}^{10} (h_i(x) - y)^2 $$
    $$ = \frac{1}{10} [(9-10)^2 + (11-10)^2 + \dots] = 24.9 $$

**验证**:
理论上 $E[\text{MSE}] \approx \text{Bias}^2 + \text{Variance} + \text{Noise}^2$ (针对单点)。
在此样本中，噪声贡献为 $(t(x)-y)^2 = (9.5-10)^2 = 0.25$。
$0.64 (\text{Bias}^2) + 24.81 (\text{Var}) + 0.25 (\text{Noise}) = 25.7$。
由于是经验估算（样本量仅为10），数值上 $24.9 \approx 25.7$，存在微小差异属正常统计波动。