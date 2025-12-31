# DDA3020 Machine Learning - Lecture 15: K-means Clustering

**讲师**: Juexiao Zhou (School of Data Science, CUHK-SZ)
**日期**: Nov 18/20, 2025

---

## 1. K-means 聚类 (K-means Clustering)

### 1.1 定义
*   **基本概念**: K-means 聚类是一种**矢量量化 (Vector Quantization)** 方法，起源于信号处理。
*   **目标**: 将 $n$ 个观测值（样本）划分为 $K$ 个簇 (Clusters)。
*   **规则**: 每个观测值属于距离其最近的**均值**（簇中心或簇质心，Cluster Centroid）所在的簇。簇中心作为该簇的原型。
*   **数学目标**: 最小化**簇内方差 (Within-cluster variances)**，通常度量为平方欧几里得距离。

### 1.2 算法流程 (Basic Algorithm)
K-means 是一种迭代算法，主要包含以下步骤：

1.  **初始化 (Initialization)**:
    *   选择簇的数量 $K$。
    *   随机在特征空间中放置 $K$ 个特征向量作为初始质心 (Centroids) $\{c_1, ..., c_K\}$。

2.  **分配 (Assignment)**:
    *   计算每个样本 $x$ 到每个质心 $c$ 的距离（通常使用欧几里得距离）。
    *   将每个样本分配给距离最近的质心（相当于给样本打上质心 ID 的标签）。

3.  **更新/重拟合 (Update/Refitting)**:
    *   对于每个簇，计算所有被分配到该簇的样本的**平均特征向量**。
    *   将这些平均向量作为新的质心位置。

4.  **迭代**:
    *   重新计算距离并修改分配。
    *   重复上述过程，直到**收敛**（即质心位置重新计算后，样本的分配不再发生变化）。

---

### 1.3 优化视角 (Optimization Perspective)

这是本讲的核心理论部分，解释了 K-means 到底在优化什么，以及为什么这样做是有效的。

#### 1.3.1 目标函数 (Objective Function)
给定数据集 $\{x_i\}_{i=1}^n$，K-means 旨在通过最小化数据点到其分配的簇中心的平方距离之和，来找到簇中心 $c = \{c_j\}_{j=1}^K$ 和分配矩阵 $r$。

**目标函数 $J(c, r)$**:
$$ J(c, r) = \sum_{i=1}^{n} \sum_{k=1}^{K} r_{ik} ||x_i - c_k||^2 $$

**约束条件**:
1.  $r_{ik} \in \{0, 1\}$: $r_{ik}=1$ 表示样本 $x_i$ 被分配给簇 $k$，否则为 0。
2.  $\sum_{k=1}^{K} r_{ik} = 1$: 每个样本必须且只能属于一个簇。

#### 1.3.2 求解方法：坐标下降法 (Coordinate Descent)
该问题可以通过坐标下降算法求解，即交替更新 $c$ 和 $r$。

**步骤 1: 分配 (Assignment) - 固定 $c$，更新 $r$**
我们要解决以下子问题：
$$ \min_{r} \sum_{i=1}^{n} \sum_{k=1}^{K} r_{ik} ||x_i - c_k||^2 $$
$$ \text{s.t. } r_{ik} \in \{0, 1\}, \sum_{k} r_{ik} = 1 $$

*   **推导**:
    由于每个数据点 $x_i$ 的分配是独立的，我们可以对每个 $i$ 分别最小化：
    $$ \min_{r_i} \sum_{k=1}^{K} r_{ik} ||x_i - c_k||^2 $$
    为了使上式最小，且 $r_{ik}$ 只能取 0 或 1，我们显然应该将 $r_{ik}=1$ 赋给使 $||x_i - c_k||^2$ 最小的那个 $k$。
*   **最优解**:
    $$ k^* = \arg\min_{1 \le k \le K} ||x_i - c_k||^2 $$
    $$ r_{ik^*} = 1, \quad \text{其他为 } 0 $$
    *解释*: 这正是算法中的“将 $x_i$ 分配给最近的簇”。

**步骤 2: 重拟合 (Refitting) - 固定 $r$，更新 $c$**
我们要解决以下子问题：
$$ \min_{c} \sum_{i=1}^{n} \sum_{k=1}^{K} r_{ik} ||x_i - c_k||^2 $$

*   **推导**:
    $c_1, ..., c_K$ 可以独立优化。对于特定的簇中心 $c_k$，目标函数为：
    $$ J(c_k) = \sum_{i=1}^{n} r_{ik} ||x_i - c_k||^2 $$
    这是一个关于 $c_k$ 的凸函数。为了求极小值，我们对 $c_k$ 求偏导并令其为 0。
    $$ \frac{\partial J}{\partial c_k} = \sum_{i=1}^{n} r_{ik} \cdot \frac{\partial}{\partial c_k} (x_i - c_k)^T (x_i - c_k) $$
    $$ \frac{\partial J}{\partial c_k} = \sum_{i=1}^{n} r_{ik} \cdot 2(c_k - x_i) = 0 $$
    消去常数 2：
    $$ \sum_{i=1}^{n} r_{ik} c_k - \sum_{i=1}^{n} r_{ik} x_i = 0 $$
    $$ c_k \sum_{i=1}^{n} r_{ik} = \sum_{i=1}^{n} r_{ik} x_i $$
*   **最优解**:
    $$ c_k = \frac{\sum_{i=1}^{n} r_{ik} x_i}{\sum_{i=1}^{n} r_{ik}} $$
    *解释*:
    *   分母 $\sum r_{ik}$ 是分配给簇 $k$ 的样本数量。
    *   分子 $\sum r_{ik} x_i$ 是分配给簇 $k$ 的所有样本向量之和。
    *   结论：$c_k$ 就是该簇所有样本的**均值 (Mean)**。

#### 1.3.3 收敛性 (Convergence)
*   **保证**:
    *   当分配 $r$ 改变时，数据点到新中心的距离更近，目标函数 $J$ 下降。
    *   当中心 $c$ 移动到均值时，根据最小二乘性质，目标函数 $J$ 下降。
    *   由于 $J$ 有下界（$\ge 0$）且每步单调递减，算法必然收敛。
*   **局部最优 (Local Minimum)**:
    *   目标函数 $J$ 是**非凸 (Non-convex)** 的。
    *   K-means **不能保证**收敛到全局最优，可能会陷入局部最优。
    *   **解决方案**: 多次运行 K-means（使用不同的随机初始化），选择目标函数值最小的那次结果。

### 1.4 应用示例
*   **矢量量化 (Vector Quantization)**: 图像压缩。将大量颜色点归类为 $K$ 种代表色，用质心颜色代替区域颜色。

### 1.5 K-means 变体 (Variants) - 可选
*   **Fuzzy C-means**: 软聚类，一个点可以以不同概率属于多个簇。
*   **Constrained K-means**: 带有约束条件的聚类。
*   **Accelerated K-means**: 加速算法（如利用三角不等式减少距离计算）。

---

## 2. 聚类性能评估 (Performance Evaluation)

由于无监督学习没有标签，评估比较困难。主要分为两类指标。

### 2.1 内部评估指标 (Internal Evaluation Metrics)
不需要真实标签，仅基于数据本身的分布。

#### 轮廓系数 (Silhouette Coefficient)
对于单个样本 $i$：
1.  **$a$**: 样本 $i$ 与**同簇**中所有其他点的平均距离（簇内紧密度）。
2.  **$b$**: 样本 $i$ 与**最近的邻居簇**（即不包含 $i$ 的簇中，距离 $i$ 最近的那个簇）中所有点的平均距离（簇间分离度）。

**公式**:
$$ s = \frac{b - a}{\max(a, b)} $$

**分段形式**:
$$ s = \begin{cases} 1 - a/b & \text{if } a < b \\ 0 & \text{if } a = b \\ b/a - 1 & \text{if } a > b \end{cases} $$

**解释**:
*   $s \in (-1, 1)$。
*   $s$ 越接近 1，表示 $a \ll b$，聚类效果越好（簇内紧密，簇间分离）。
*   整个数据集的轮廓系数是所有样本 $s$ 值的均值。

### 2.2 外部评估指标 (External Evaluation Metrics)
需要真实标签（Ground Truth）作为参考。

#### Rand Index (RI)
给定 $n$ 个样本 $S$，比较两个聚类结果 $X$（算法结果）和 $Y$（真实标签）。
考虑所有可能的样本对（共 $\binom{n}{2}$ 对），定义以下计数：
*   **$a$**: 在 $X$ 中同簇，在 $Y$ 中也同簇的对数（TP）。
*   **$b$**: 在 $X$ 中不同簇，在 $Y$ 中也不同簇的对数（TN）。
*   **$c$**: 在 $X$ 中同簇，但在 $Y$ 中不同簇的对数（FP）。
*   **$d$**: 在 $X$ 中不同簇，但在 $Y$ 中同簇的对数（FN）。

**公式**:
$$ RI = \frac{a + b}{a + b + c + d} = \frac{a + b}{\binom{n}{2}} = \frac{a + b}{n(n-1)/2} $$

**解释**:
*   $RI \in [0, 1]$。
*   分数越高表示相似度越高（聚类结果越接近真实标签）。

#### Adjusted Rand Index (ARI)
RI 的问题在于，对于随机聚类结果，RI 不为 0。ARI 对随机性进行了调整。
使用列联表 (Contingency Table)，$n_{ij}$ 表示同时属于 $X_i$ 和 $Y_j$ 的样本数。

**公式**:
$$ ARI = \frac{\sum_{ij} \binom{n_{ij}}{2} - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2}}{ \frac{1}{2} [\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}] - [\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}] / \binom{n}{2} } $$

*   其中 $a_i$ 是 $X_i$ 的行和，$b_j$ 是 $Y_j$ 的列和。
*   公式结构理解：$ARI = \frac{\text{Index} - \text{Expected Index}}{\text{Max Index} - \text{Expected Index}}$。

**解释**:
*   ARI 可以是负数。
*   ARI = 1 表示完美聚类。
*   ARI $\approx$ 0 表示随机聚类。

---

## 3. 其他聚类算法 (Other Clusterings)
除了 K-means，还有许多其他方法：
*   **层次聚类 (Hierarchical clustering)**
*   **基于图的聚类 (Graph based clustering)**
*   **基于密度的聚类 (Density based clustering)**
*   **概率聚类 (Probabilistic clustering)**