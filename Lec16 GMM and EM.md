# DDA3020 Lecture 16: Gaussian Mixture Models & EM Algorithm

## 1. 混合模型 (Mixture Models) 与 GMM 基础

### 1.1 混合模型概念
在无监督聚类中，我们没有类标签 $z$。混合模型通过引入**隐变量 (Latent Variable)** $z$ 来对观测数据 $x$ 的边缘分布进行建模：
$$
p(x) = \sum_{z} p(x, z) = \sum_{z} p(x|z)p(z)
$$

### 1.2 高斯混合模型 (GMM)
GMM 是最常见的混合模型，它假设数据是由 $K$ 个高斯分布线性组合而成的。

**概率密度函数 (PDF):**
$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

**参数定义:**
*   $x$: $d$ 维特征向量。
*   $\pi_k$: 混合系数 (Mixing coefficients)，即先验概率 $p(z=k)$。
    *   约束条件: $\sum_{k=1}^{K} \pi_k = 1$ 且 $0 \le \pi_k \le 1$。
*   $\mu_k$: 第 $k$ 个高斯分量的均值。
*   $\Sigma_k$: 第 $k$ 个高斯分量的协方差矩阵。
*   $\mathcal{N}(x | \mu_k, \Sigma_k)$: 多元高斯分布，公式为：
    $$
    \mathcal{N}(x | \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)
    $$

---

## 2. GMM 的最大似然估计 (直接推导)

### 2.1 对数似然函数 (Log-Likelihood)
给定数据集 $X = \{x^{(1)}, \dots, x^{(N)}\}$，GMM 的对数似然函数为：
$$
\ln p(X | \pi, \mu, \Sigma) = \sum_{n=1}^{N} \ln \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x^{(n)} | \mu_k, \Sigma_k) \right)
$$

**难点:** 由于对数函数内部存在求和 ($\ln \sum$)，无法直接得到闭式解（Closed-form solution），令导数为0会得到相互依赖的方程组。

### 2.2 引入 "责任" (Responsibility)
定义 $\gamma(z_{nk})$ 为第 $k$ 个高斯分量对第 $n$ 个样本的**后验概率**（即责任）：
$$
\gamma(z_{nk}) = p(z^{(n)}=k | x^{(n)}) = \frac{\pi_k \mathcal{N}(x^{(n)} | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(n)} | \mu_j, \Sigma_j)}
$$
令 $N_k = \sum_{n=1}^{N} \gamma(z_{nk})$ 为分配给第 $k$ 个簇的有效样本数。

### 2.3 参数更新公式推导 (详细步骤)

#### (1) 关于均值 $\mu_k$ 的推导
目标：对 $\ln p(X)$ 关于 $\mu_k$ 求导并令其为 0。
利用链式法则，注意 $\mu_k$ 只出现在第 $k$ 个分量中：
$$
\frac{\partial \ln p(X)}{\partial \mu_k} = \sum_{n=1}^{N} \frac{\partial}{\partial \mu_k} \ln \left( \sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(n)} | \mu_j, \Sigma_j) \right)
$$
$$
= \sum_{n=1}^{N} \frac{1}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(n)}|\dots)} \cdot \frac{\partial}{\partial \mu_k} (\pi_k \mathcal{N}(x^{(n)} | \mu_k, \Sigma_k))
$$
已知高斯分布对均值的导数为 $\mathcal{N}(x|\mu, \Sigma) \Sigma^{-1}(x-\mu)$，故：
$$
= \sum_{n=1}^{N} \underbrace{\frac{\pi_k \mathcal{N}(x^{(n)} | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(n)} | \mu_j, \Sigma_j)}}_{\gamma(z_{nk})} \cdot \Sigma_k^{-1}(x^{(n)} - \mu_k) = 0
$$
$$
\sum_{n=1}^{N} \gamma(z_{nk}) \Sigma_k^{-1}(x^{(n)} - \mu_k) = 0
$$
两边同乘 $\Sigma_k$ 去掉协方差项：
$$
\sum_{n=1}^{N} \gamma(z_{nk}) x^{(n)} - \sum_{n=1}^{N} \gamma(z_{nk}) \mu_k = 0
$$
$$
\mu_k \sum_{n=1}^{N} \gamma(z_{nk}) = \sum_{n=1}^{N} \gamma(z_{nk}) x^{(n)}
$$
**结果:**
$$
\mu_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) x^{(n)}
$$

#### (2) 关于协方差 $\Sigma_k$ 的推导
类似地，对 $\Sigma_k$ 求导并令为 0，可得：
**结果:**
$$
\Sigma_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) (x^{(n)} - \mu_k)(x^{(n)} - \mu_k)^T
$$

#### (3) 关于混合系数 $\pi_k$ 的推导 (拉格朗日乘子法)
目标：最大化 $\ln p(X)$，约束条件为 $\sum_{k=1}^{K} \pi_k = 1$。
构造拉格朗日函数：
$$
\mathcal{L} = \ln p(X) + \lambda \left( \sum_{k=1}^{K} \pi_k - 1 \right)
$$
对 $\pi_k$ 求导：
$$
\frac{\partial \mathcal{L}}{\partial \pi_k} = \sum_{n=1}^{N} \frac{\mathcal{N}(x^{(n)} | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(n)} | \mu_j, \Sigma_j)} + \lambda = 0
$$
两边同乘 $\pi_k$：
$$
\sum_{n=1}^{N} \underbrace{\frac{\pi_k \mathcal{N}(x^{(n)} | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x^{(n)} | \dots)}}_{\gamma(z_{nk})} + \lambda \pi_k = 0
$$
$$
N_k + \lambda \pi_k = 0
$$
对所有 $k$ 求和以解出 $\lambda$：
$$
\sum_{k=1}^{K} N_k + \lambda \sum_{k=1}^{K} \pi_k = 0 \implies \sum_{n=1}^{N} \underbrace{\sum_{k=1}^{K} \gamma(z_{nk})}_{1} + \lambda(1) = 0 \implies N + \lambda = 0 \implies \lambda = -N
$$
代回原式：
$$
N_k - N \pi_k = 0
$$
**结果:**
$$
\pi_k = \frac{N_k}{N}
$$

---

## 3. GMM 与 K-Means 的比较

| 特性       | K-Means                               | GMM (EM算法)                    |
| :------- | :------------------------------------ | :---------------------------- |
| **分配方式** | 硬分配 (Hard Assignment)                 | 软分配 (Soft Assignment, 概率性)    |
| **模型假设** | 假设簇是球状的 (协方差为单位阵)                     | 可以学习任意椭球状的协方差                 |
| **更新中心** | 仅基于归属该簇的数据点的均值                        | 基于所有数据点的加权均值 (权重为责任 $\gamma$) |
| **本质**   | GMM 的特例 ($\Sigma \to 0$ 或 $\Sigma=I$) | 概率密度估计                        |
|          |                                       |                               |

---

## 4. 期望最大化 (EM) 算法理论

### 4.1 隐变量模型与对数似然分解
对于包含隐变量 $z$ 的模型，对数似然为：
$$
\ln p(X|\theta) = \ln \sum_z p(X, z | \theta)
$$
引入任意关于 $z$ 的分布 $q(z)$，利用 **Jensen不等式** 或直接分解：
$$
\ln p(X|\theta) = \mathcal{L}(q, \theta) + KL(q || p)
$$
其中：
1.  **ELBO (Evidence Lower Bound, 下界):**
    $$
    \mathcal{L}(q, \theta) = \sum_z q(z) \ln \frac{p(X, z | \theta)}{q(z)}
    $$
2.  **KL 散度 (KL Divergence):**
    $$
    KL(q || p) = - \sum_z q(z) \ln \frac{p(z | X, \theta)}{q(z)} \ge 0
    $$

### 4.2 Jensen 不等式
*   若 $f$ 是凸函数 (Convex): $f(E[X]) \le E[f(X)]$
*   若 $f$ 是凹函数 (Concave, 如 $\ln$): $f(E[X]) \ge E[f(X)]$
*   利用 Jensen 不等式证明下界：
    $$
    \ln p(X|\theta) = \ln \sum_z q(z) \frac{p(X, z|\theta)}{q(z)} \ge \sum_z q(z) \ln \frac{p(X, z|\theta)}{q(z)} = \mathcal{L}(q, \theta)
    $$

### 4.3 EM 算法流程
EM 通过坐标下降法 (Coordinate Descent) 迭代最大化下界 $\mathcal{L}$。

#### **E-Step (Expectation): 固定 $\theta$，更新 $q$**
目标：使下界 $\mathcal{L}$ 贴近真实的对数似然 $\ln p(X|\theta)$。
这等价于最小化 $KL(q || p)$。当 $KL=0$ 时，下界最紧。
**最优解:**
$$
q^{(new)}(z) = p(z | X, \theta^{(old)})
$$
即：$q$ 分布应当取当前参数下的**后验分布**。

#### **M-Step (Maximization): 固定 $q$，更新 $\theta$**
目标：最大化下界 $\mathcal{L}$ 以更新参数。
$$
\theta^{(new)} = \arg \max_{\theta} \mathcal{L}(q, \theta) = \arg \max_{\theta} \sum_z q(z) \ln p(X, z | \theta)
$$
注意：$\ln q(z)$ 项对 $\theta$ 是常数，因此这等价于最大化**期望完整数据对数似然 (Expected Complete Data Log-Likelihood)**：
$$
Q(\theta, \theta^{(old)}) = E_{z \sim p(z|X, \theta^{(old)})} [\ln p(X, z | \theta)]
$$

---

## 5. EM 算法应用于 GMM

将通用的 EM 框架套用到 GMM 模型中。
完整数据对数似然 $\ln p(X, Z | \theta)$ 为：
$$
\ln p(X, Z) = \sum_{n=1}^{N} \sum_{k=1}^{K} \mathbb{I}(z^{(n)}=k) \left( \ln \pi_k + \ln \mathcal{N}(x^{(n)} | \mu_k, \Sigma_k) \right)
$$

### 5.1 E-Step: 计算责任 (Responsibilities)
计算隐变量的后验概率（即期望）：
$$
\gamma(z_{nk}) = E[\mathbb{I}(z^{(n)}=k)] = p(z^{(n)}=k | x^{(n)}, \theta^{(old)})
$$
$$
\gamma(z_{nk}) = \frac{\pi_k^{(old)} \mathcal{N}(x^{(n)} | \mu_k^{(old)}, \Sigma_k^{(old)})}{\sum_{j=1}^{K} \pi_j^{(old)} \mathcal{N}(x^{(n)} | \mu_j^{(old)}, \Sigma_j^{(old)})}
$$

### 5.2 M-Step: 最大化期望函数
我们需要最大化的函数 $Q$ 是：
$$
Q(\theta) = \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma(z_{nk}) \left( \ln \pi_k + \ln \mathcal{N}(x^{(n)} | \mu_k, \Sigma_k) \right)
$$
这与第 2 节中直接求导的公式形式完全一致，只是其中的指示变量被替换为了责任 $\gamma(z_{nk})$。

**更新公式 (同 2.3 节):**
1.  $\mu_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) x^{(n)}$
2.  $\Sigma_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) (x^{(n)} - \mu_k^{new})(x^{(n)} - \mu_k^{new})^T$
3.  $\pi_k^{new} = \frac{N_k}{N}$

其中 $N_k = \sum_{n=1}^{N} \gamma(z_{nk})$。

---

## 6. GMM 总结

### 优点
*   **灵活性:** 能拟合复杂的数据分布（通用近似器）。
*   **软聚类:** 提供概率性的归属，适合重叠簇。
*   **密度估计:** 不仅是聚类，还能给出概率密度函数。
*   **处理缺失值:** 能够处理不完整的数据集。

### 缺点
*   **局部最优:** EM 算法只能保证收敛到局部最优，结果依赖于初始化（通常用 K-Means 初始化）。
*   **K的选择:** 需要预先指定高斯分量的个数 $K$。
*   **奇异性问题:** 如果某个簇只有一个点，协方差矩阵可能不可逆（奇异）。
*   **维度灾难:** 在高维数据上表现不佳。


# HW3 T2 

# 离散多项分布混合模型的 EM 算法推导 (基于标量隐变量)

## 1. 建模及引入隐变量

**观测数据**: 数据集包含 $N$ 个样本 $\{x^{(1)}, \dots, x^{(N)}\}$。
每个样本 $x^{(n)}$ 是一个 $D \times M$ 的二值矩阵（或向量），元素记为 $x_{ij}^{(n)}$。
*   $n = 1, \dots, N$
*   $i = 1, \dots, D$ (维度/特征)
*   $j = 1, \dots, M$ (状态/取值)
*   约束：$\sum_{j=1}^M x_{ij}^{(n)} = 1$ (每个特征必取且仅取一个状态)。

**隐变量**: 对于每个样本 $x^{(n)}$，引入一个离散隐变量 $z^{(n)}$，表示该样本所属的混合成分（簇）。
$$ z^{(n)} \in \{1, 2, \dots, K\} $$

**联合概率分布**:
根据图片中的定义，样本 $x^{(n)}$ 和隐变量 $z^{(n)}$ 取值为 $k$ 的联合概率为：
$$ p(x^{(n)}, z^{(n)} = k) = \pi_k p(x^{(n)} | \mu_k) = \pi_k \prod_{i=1}^D \prod_{j=1}^M (\mu_{kij})^{x_{ij}^{(n)}} $$

其中：
*   $\pi_k = p(z^{(n)} = k)$ 是混合系数，满足 $\sum_{k=1}^K \pi_k = 1$。
*   $\mu_{kij} = p(x_{ij}^{(n)}=1 | z^{(n)}=k)$ 是参数，满足 $\sum_{j=1}^M \mu_{kij} = 1$。

---

## 2. 完整数据的对数似然函数

为了使用 EM 算法，我们首先写出包含隐变量的**完整数据对数似然函数**。
对于单个样本 $(x^{(n)}, z^{(n)})$，其对数似然可以借助指示函数 $\mathbb{I}(z^{(n)} = k)$ 来表示（当 $z^{(n)}=k$ 时为 1，否则为 0）：

$$ \ln p(x^{(n)}, z^{(n)}) = \sum_{k=1}^K \mathbb{I}(z^{(n)} = k) \ln \left( \pi_k \prod_{i=1}^D \prod_{j=1}^M (\mu_{kij})^{x_{ij}^{(n)}} \right) $$

将所有 $N$ 个样本累加，得到总体的完整数据对数似然：

$$ \mathcal{L}_{complete} = \sum_{n=1}^N \sum_{k=1}^K \mathbb{I}(z^{(n)} = k) \left( \ln \pi_k + \sum_{i=1}^D \sum_{j=1}^M x_{ij}^{(n)} \ln \mu_{kij} \right) $$

---

## 3. E-Step (期望步)

E-step 的目的是计算在给定观测数据 $x^{(n)}$ 和当前参数 $\theta^{old} = \{\pi^{old}, \mu^{old}\}$ 下，隐变量 $z^{(n)}$ 的后验概率。我们将这个后验概率记为 $\gamma_{nk}$ (Responsibility)。

$$ \gamma_{nk} = p(z^{(n)} = k | x^{(n)}, \theta^{old}) $$

根据贝叶斯公式：
$$ p(z^{(n)} = k | x^{(n)}) = \frac{p(x^{(n)} | z^{(n)} = k) p(z^{(n)} = k)}{\sum_{l=1}^K p(x^{(n)} | z^{(n)} = l) p(z^{(n)} = l)} $$

代入具体的分布公式：
$$ \gamma_{nk} = \frac{\pi_k \prod_{i=1}^D \prod_{j=1}^M (\mu_{kij})^{x_{ij}^{(n)}}}{\sum_{l=1}^K \pi_l \prod_{i=1}^D \prod_{j=1}^M (\mu_{lij})^{x_{ij}^{(n)}}} $$

这一步计算出的 $\gamma_{nk}$ 将用于 M-step 中计算指示函数 $\mathbb{I}(z^{(n)} = k)$ 的期望值：
$$ \mathbb{E}[\mathbb{I}(z^{(n)} = k)] = 1 \cdot p(z^{(n)}=k | x^{(n)}) + 0 \cdot p(z^{(n)} \neq k | x^{(n)}) = \gamma_{nk} $$

---

## 4. M-Step (最大化步)

M-step 的目标是最大化 Q 函数。Q 函数是完整数据对数似然函数关于隐变量后验分布的期望。

$$ Q(\theta, \theta^{old}) = \mathbb{E}_{Z|X, \theta^{old}} [\mathcal{L}_{complete}] $$

将 $\mathbb{E}[\mathbb{I}(z^{(n)} = k)]$ 替换为 $\gamma_{nk}$：

$$ Q(\theta, \theta^{old}) = \sum_{n=1}^N \sum_{k=1}^K \gamma_{nk} \left( \ln \pi_k + \sum_{i=1}^D \sum_{j=1}^M x_{ij}^{(n)} \ln \mu_{kij} \right) $$

我们需要分别针对 $\pi_k$ 和 $\mu_{kij}$ 最大化这个 Q 函数，同时满足各自的约束条件。

### 4.1 求解混合系数 $\pi_k$

提取 Q 函数中与 $\pi_k$ 相关的部分：
$$ Q_{\pi} = \sum_{n=1}^N \sum_{k=1}^K \gamma_{nk} \ln \pi_k $$
约束条件：$\sum_{k=1}^K \pi_k = 1$。

构造拉格朗日函数（乘子为 $\lambda$）：
$$ \Lambda_{\pi} = \sum_{n=1}^N \sum_{k=1}^K \gamma_{nk} \ln \pi_k + \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) $$

对 $\pi_k$ 求偏导并令为 0：
$$ \frac{\partial \Lambda_{\pi}}{\partial \pi_k} = \sum_{n=1}^N \frac{\gamma_{nk}}{\pi_k} + \lambda = 0 $$
$$ \sum_{n=1}^N \gamma_{nk} = -\lambda \pi_k $$

对两边关于 $k$ 求和（利用 $\sum_{k=1}^K \pi_k = 1$ 和 $\sum_{k=1}^K \gamma_{nk} = 1$）：
$$ \sum_{k=1}^K \sum_{n=1}^N \gamma_{nk} = \sum_{k=1}^K (-\lambda \pi_k) $$
$$ \sum_{n=1}^N \left( \sum_{k=1}^K \gamma_{nk} \right) = -\lambda \cdot 1 $$
$$ \sum_{n=1}^N 1 = N = -\lambda \implies \lambda = -N $$

代回原式解得：
$$ \pi_k = \frac{\sum_{n=1}^N \gamma_{nk}}{N} $$
令 $N_k = \sum_{n=1}^N \gamma_{nk}$，则 $\pi_k = \frac{N_k}{N}$。

### 4.2 求解分量参数 $\mu_{kij}$

提取 Q 函数中与 $\mu_{kij}$ 相关的部分（针对特定的 $k$ 和 $i$）：
$$ Q_{\mu} = \sum_{n=1}^N \gamma_{nk} \sum_{j=1}^M x_{ij}^{(n)} \ln \mu_{kij} $$
约束条件：对于任意 $k, i$，满足 $\sum_{j=1}^M \mu_{kij} = 1$。

构造拉格朗日函数（乘子为 $\beta_{ki}$）：
$$ \Lambda_{\mu} = \sum_{n=1}^N \gamma_{nk} \sum_{j=1}^M x_{ij}^{(n)} \ln \mu_{kij} + \beta_{ki} \left( \sum_{j=1}^M \mu_{kij} - 1 \right) $$

对 $\mu_{kij}$ 求偏导并令为 0：
$$ \frac{\partial \Lambda_{\mu}}{\partial \mu_{kij}} = \sum_{n=1}^N \gamma_{nk} \frac{x_{ij}^{(n)}}{\mu_{kij}} + \beta_{ki} = 0 $$
$$ \sum_{n=1}^N \gamma_{nk} x_{ij}^{(n)} = -\beta_{ki} \mu_{kij} $$

对两边关于 $j$ 求和（利用 $\sum_{j=1}^M \mu_{kij} = 1$ 和 $\sum_{j=1}^M x_{ij}^{(n)} = 1$）：
$$ \sum_{j=1}^M \sum_{n=1}^N \gamma_{nk} x_{ij}^{(n)} = \sum_{j=1}^M (-\beta_{ki} \mu_{kij}) $$
$$ \sum_{n=1}^N \gamma_{nk} \left( \sum_{j=1}^M x_{ij}^{(n)} \right) = -\beta_{ki} \cdot 1 $$
$$ \sum_{n=1}^N \gamma_{nk} \cdot 1 = -\beta_{ki} $$
$$ N_k = -\beta_{ki} $$

代回原式解得：
$$ \mu_{kij} = \frac{\sum_{n=1}^N \gamma_{nk} x_{ij}^{(n)}}{N_k} $$

---

## 5. 结论：EM 算法迭代方程

**E-Step (评估责任):**
$$ \gamma_{nk} = \frac{\pi_k \prod_{i=1}^D \prod_{j=1}^M (\mu_{kij})^{x_{ij}^{(n)}}}{\sum_{l=1}^K \pi_l \prod_{i=1}^D \prod_{j=1}^M (\mu_{lij})^{x_{ij}^{(n)}}} $$

**M-Step (参数更新):**
计算有效样本数： $N_k = \sum_{n=1}^N \gamma_{nk}$

更新混合系数：
$$ \pi_k^{new} = \frac{N_k}{N} $$

更新分量参数：
$$ \mu_{kij}^{new} = \frac{\sum_{n=1}^N \gamma_{nk} x_{ij}^{(n)}}{N_k} $$