# DDA3020 Machine Learning - Lecture 17: Principal Component Analysis (PCA)

**讲师**: Juexiao Zhou (School of Data Science, CUHK-SZ)
**日期**: Dec 2/4, 2025

---

## 1. 预备知识：子空间投影 (Preliminary: Projection onto a subspace)

在介绍 PCA 之前，首先需要定义数据如何从高维空间投影到低维子空间。

### 1.1 符号定义
*   **数据集**: $D = \{x^{(1)}, ..., x^{(N)}\}$，其中 $x^{(n)} \in \mathbb{R}^D$，$D$ 为原始维度。
*   **均值向量 (Mean)**: $\mu = \frac{1}{N} \sum_{n=1}^N x^{(n)} \in \mathbb{R}^D$。
*   **子空间 (Subspace) $S$**: 由 $K$ 个正交基向量 (Orthonormal basis) $\{u_k\}_{k=1}^K$ 张成，其中 $u_k \in \mathbb{R}^D$。
    *   **正交归一性**: $\|u_k\| = 1$，且当 $i \neq j$ 时 $u_i^T u_j = 0$。即 $U^T U = I_K$。
    *   **基矩阵**: $U = [u_1, ..., u_K] \in \mathbb{R}^{D \times K}$。

### 1.2 投影与重构
对于任意数据点 $x \in \mathbb{R}^D$，我们可以将其近似（重构）为：

$$ \tilde{x} = \mu + \text{Proj}_S (x - \mu) = \mu + \sum_{k=1}^K z_k u_k $$

或者写作矩阵形式：
$$ \tilde{x} = \mu + Uz $$

其中：
1.  **$z$ (Representation/Code)**: $x$ 在低维子空间中的坐标表示。
    $$ z_k = u_k^T (x - \mu) \implies z = U^T(x - \mu) \in \mathbb{R}^K $$
2.  **$\tilde{x}$ (Reconstruction)**: $x$ 在原始空间中的重构（投影点）。
    $$ \tilde{x} = \mu + U U^T (x - \mu) \in \mathbb{R}^D $$

### 1.3 正交定理 (Orthogonal Theorem)
**定理**: 误差向量 $x - \tilde{x}$ 正交于子空间 $S$。即：
$$ U^T (x - \tilde{x}) = 0 $$

**推导**:
1.  根据定义：$x - \tilde{x} = x - (\mu + Uz) = (x - \mu) - Uz$
2.  代入 $z = U^T(x - \mu)$：
    $$ x - \tilde{x} = (x - \mu) - U U^T (x - \mu) $$
3.  左乘 $U^T$：
    $$ \begin{aligned} U^T(x - \tilde{x}) &= U^T(x - \mu) - U^T U U^T (x - \mu) \\ &= U^T(x - \mu) - I \cdot U^T (x - \mu) \quad (\because U^T U = I) \\ &= z - z = 0 \end{aligned} $$

---

## 2. 降维 (Dimensionality Reduction)

*   **目标**: 寻找一个低维数据向量来表示原始的高维数据向量。
*   **PCA**: 一种典型的**无监督 (Unsupervised)** 线性降维方法。
*   **输入**: 数据集 $D \in \mathbb{R}^D$。
*   **输出**: 基向量 $\{u_k\}_{k=1}^K$ 和新的表示 $z^{(n)} \in \mathbb{R}^K$ ($K < D$)。
*   **用途**:
    *   可视化 (Visualization)
    *   缓解过拟合 (Alleviate overfitting)
    *   减少计算成本 (Reduce computational cost)

---

## 3. PCA 的推导 (Derivations)

PCA 的目标是找到最佳的子空间 $U$。什么是“最佳”？有两种等价的解释。

### 3.1 推导 1：最大化方差 (Maximal Variance)
**直觉**: 好的投影应该尽可能保留数据的信息，即投影后的数据分布越分散（方差越大）越好。

**目标函数**:
寻找 $U$ 使得重构数据 $\tilde{x}$ 的方差最大化：
$$ \max_{U, U^T U = I} \frac{1}{N} \sum_{n=1}^N \|\tilde{x}^{(n)} - \tilde{\mu}\|^2 $$

**推导步骤**:
1.  **计算重构均值 $\tilde{\mu}$**:
    $$ \tilde{\mu} = \frac{1}{N} \sum \tilde{x}^{(n)} = \mu + U(\frac{1}{N} \sum z^{(n)}) = \mu + U \cdot U^T (\underbrace{\frac{1}{N}\sum x^{(n)}}_{\mu} - \mu) = \mu $$
    即重构数据的中心依然是 $\mu$。

2.  **简化目标函数**:
    $$ \begin{aligned} \text{Var} &= \frac{1}{N} \sum_{n=1}^N \|\tilde{x}^{(n)} - \mu\|^2 \\ &= \frac{1}{N} \sum_{n=1}^N \|Uz^{(n)}\|^2 \quad (\because \tilde{x} - \mu = Uz) \\ &= \frac{1}{N} \sum_{n=1}^N (Uz^{(n)})^T (Uz^{(n)}) \\ &= \frac{1}{N} \sum_{n=1}^N z^{(n)T} \underbrace{U^T U}_{I} z^{(n)} \\ &= \frac{1}{N} \sum_{n=1}^N \|z^{(n)}\|^2 \end{aligned} $$
    这表明：**最大化重构数据的方差 $\iff$ 最大化低维表示 $z$ 的方差**。

3.  **引入协方差矩阵**:
    代入 $z^{(n)} = U^T(x^{(n)} - \mu)$：
    $$ \begin{aligned} J(U) &= \frac{1}{N} \sum_{n=1}^N \|U^T(x^{(n)} - \mu)\|^2 \\ &= \frac{1}{N} \sum_{n=1}^N \text{Trace}\left( (U^T(x^{(n)} - \mu)) (U^T(x^{(n)} - \mu))^T \right) \quad (\text{利用性质 } \|v\|^2 = \text{Tr}(vv^T)) \\ &= \frac{1}{N} \sum_{n=1}^N \text{Trace}\left( U^T (x^{(n)} - \mu)(x^{(n)} - \mu)^T U \right) \\ &= \text{Trace}\left( U^T \left[ \frac{1}{N} \sum_{n=1}^N (x^{(n)} - \mu)(x^{(n)} - \mu)^T \right] U \right) \end{aligned} $$

    定义**经验协方差矩阵 (Empirical Covariance Matrix)** $\Sigma$:
    $$ \Sigma = \frac{1}{N} \sum_{n=1}^N (x^{(n)} - \mu)(x^{(n)} - \mu)^T $$

    **最终优化问题**:
    $$ \max_{U} \text{Trace}(U^T \Sigma U) \quad \text{s.t.} \quad U^T U = I $$

### 3.2 推导 2：最小化重构误差 (Minimal Reconstruction Error)
**直觉**: 好的投影应该使得重构后的点 $\tilde{x}$ 与原始点 $x$ 之间的距离（误差）最小。

**目标函数**:
$$ \min_{U, U^T U = I} \frac{1}{N} \sum_{n=1}^N \|x^{(n)} - \tilde{x}^{(n)}\|^2 $$

### 3.3 等价性证明
利用勾股定理 (Pythagorean Theorem)，我们可以证明上述两个目标是等价的。

$$ \|x^{(n)} - \mu\|^2 = \| (x^{(n)} - \tilde{x}^{(n)}) + (\tilde{x}^{(n)} - \mu) \|^2 $$

由于 $x - \tilde{x}$ 正交于子空间 $S$，而 $\tilde{x} - \mu$ 位于子空间 $S$ 内（因为 $\tilde{x} = \mu + Uz$），两者正交。交叉项为 0。
$$ \|x^{(n)} - \mu\|^2 = \underbrace{\|x^{(n)} - \tilde{x}^{(n)}\|^2}_{\text{Reconstruction Error}} + \underbrace{\|\tilde{x}^{(n)} - \mu\|^2}_{\text{Projected Variance}} $$

对所有样本求和并除以 $N$：
$$ \text{Total Variance (Constant)} = \text{Reconstruction Error} + \text{Projected Variance} $$

*   因为原始数据的总方差是常数，所以 **最小化重构误差 $\iff$ 最大化投影方差**。

---

## 4. PCA 算法求解 (Algorithm & Solution)

我们需要求解优化问题：
$$ \max_{U} \text{Trace}(U^T \Sigma U) \quad \text{s.t.} \quad U^T U = I $$

### 4.1 拉格朗日乘子法 (Lagrangian Multiplier)
构造拉格朗日函数 $L(U, \Lambda)$，其中 $\Lambda$ 是由拉格朗日乘子组成的对角矩阵（对应 $K$ 个约束）：

$$ L(U, \Lambda) = \text{Trace}(U^T \Sigma U) + \text{Trace}(\Lambda^T(I - U^T U)) $$
*注：这里 $\Lambda = \text{diag}([\lambda_1, ..., \lambda_K])$。*

### 4.2 求解最优解
对 $U$ 求导并令其为 0：
$$ \frac{\partial L}{\partial U} = 2\Sigma U - 2U\Lambda = 0 $$

整理得：
$$ \Sigma U = U \Lambda $$

对于单个列向量 $u_k$ (即 $U$ 的第 $k$ 列) 和对应的 $\lambda_k$：
$$ \Sigma u_k = \lambda_k u_k $$

**结论**:
1.  最优解 $u_k$ 是协方差矩阵 $\Sigma$ 的**特征向量 (Eigenvector)**。
2.  对应的 $\lambda_k$ 是 $\Sigma$ 的**特征值 (Eigenvalue)**。

### 4.3 选择哪些特征向量？
将最优解代入目标函数：
$$ \text{Trace}(U^T \Sigma U) = \sum_{k=1}^K u_k^T \Sigma u_k = \sum_{k=1}^K u_k^T (\lambda_k u_k) = \sum_{k=1}^K \lambda_k \underbrace{u_k^T u_k}_{1} = \sum_{k=1}^K \lambda_k $$

**策略**: 为了最大化目标函数，我们需要选择**最大的 $K$ 个特征值**对应的特征向量。

### 4.4 算法总结
1.  **中心化**: 计算均值 $\mu$，并将数据中心化 $x^{(n)} \leftarrow x^{(n)} - \mu$。
2.  **计算协方差**: $\Sigma = \frac{1}{N} \sum_{n=1}^N x^{(n)} {x^{(n)}}^T$ (或矩阵形式 $\frac{1}{N} X X^T$)。
3.  **特征分解 (SVD)**: 对 $\Sigma$ 进行特征值分解，得到特征值 $\{\lambda_i\}$ 和特征向量 $\{q_i\}$。
4.  **排序与选择**: 将特征值从大到小排序，选择前 $K$ 个特征向量构成矩阵 $U = [q_1, ..., q_K]$。
5.  **转换**: 计算新的表示 $z^{(n)} = U^T (x^{(n)} - \mu)$。

---

## 5. 性质与应用 (Properties & Applications)

### 5.1 去相关性 (Decorrelation)
PCA 得到的新特征 $z$ 是互不相关的（Decorrelated）。
计算 $z$ 的协方差矩阵：
$$ \begin{aligned} \text{Cov}(z) &= \text{Cov}(U^T(x-\mu)) \\ &= U^T \text{Cov}(x) U \\ &= U^T \Sigma U \\ &= U^T Q \Lambda_{all} Q^T U \end{aligned} $$
由于 $U$ 是由 $Q$ 的前 $K$ 列组成的，且特征向量矩阵正交，最终结果为：
$$ \text{Cov}(z) = \text{diag}(\lambda_1, ..., \lambda_K) $$
这是一个**对角矩阵**，意味着 $z$ 的各个维度之间没有相关性。

### 5.2 应用：人脸识别 (Eigenfaces)
*   数据：$N$ 张人脸图像，每张图拉直为一个高维向量。
*   PCA：提取“特征脸 (Eigenfaces)”，即协方差矩阵的特征向量。
*   效果：仅使用前几个特征脸（如 Top-3）即可实现较高精度的分类，且特征脸可视化后具有人脸的轮廓特征。

### 5.3 变体 (Variants)
PCA 只能处理线性数据结构，对于非线性数据有以下变体：
*   **Kernel PCA**: 利用核技巧处理非线性（Bishop Chapter 12.3）。
*   **Probabilistic PCA**: 概率生成模型视角（Bishop Chapter 12.2）。
*   **Nonlinear PCA**: 非线性 PCA。
*   **Robust PCA**: 对异常值鲁棒的 PCA。