# Lecture 16：Gaussian Mixture Models & EM 算法（超详细，傻子也能看懂版）

> 课程：DDA3020 Machine Learning  
> 主题：Gaussian Mixture Model (GMM，高斯混合模型) +  
> Expectation–Maximization Algorithm (EM，期望最大化算法)  
> 核心：  
> - 用若干个高斯分布“拼起来”拟合复杂数据分布  
> - 有**隐藏变量 (latent variables)**，用 EM 算法做极大似然估计

---

## 0. 全局符号与参数说明（一定要看）

- $x$：观测到的特征 (observed feature, data point)，向量，$x \in \mathbb{R}^d$  
- $z$：类别/成分标签 (class label / component index)，离散变量  
  - $z \in \{1,\dots,K\}$，表示“由第 $k$ 个高斯生成”
- $K$：混合成分个数 (number of components / clusters)  
- $N$：样本个数 (number of samples)

数据集：

$$
\mathcal{D} = \{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}, 
\quad x^{(n)} \in \mathbb{R}^d
$$

GMM 参数（整套参数通常记为 $\theta$ 或 $\varepsilon$）：

- **混合系数 (mixing coefficients)**：  
  $$
  \omega = \{\omega_1,\dots,\omega_K\},\quad
  \omega_k \ge 0,\quad
  \sum_{k=1}^K \omega_k = 1
  $$
  含义：随机选一个样本，来自第 $k$ 个高斯的先验概率。

- **均值 (means)**：  
  $$
  \mu = \{\mu_1,\dots,\mu_K\},\quad
  \mu_k \in \mathbb{R}^d
  $$
  第 $k$ 个高斯的中心位置。

- **协方差矩阵 (covariance matrices)**：  
  $$
  \Sigma = \{\Sigma_1,\dots,\Sigma_K\},\quad
  \Sigma_k \in \mathbb{R}^{d\times d}
  $$
  - 对称正定矩阵，描述第 $k$ 个高斯的“形状”和“尺度”；
  - $|\Sigma_k|$：行列式 (determinant)，出现在密度公式中；
  - $\Sigma_k^{-1}$：逆矩阵 (inverse)，用于 Mahalanobis 距离。

常用符号：

- $N(x\mid \mu,\Sigma)$：多元高斯分布的密度 (multivariate Gaussian density)
- $\gamma_k^{(n)}$：第 $n$ 个样本属于第 $k$ 个成分的“责任度” (responsibility)
- $N_k$：第 $k$ 个成分的“有效样本数” (effective number of samples)

---

## 1. 混合模型 (Mixture Models) 的基本思想

### 1.1 从有标签的联合分布到无标签的边缘分布

在最简单的监督学习中，我们可以写出 $(x,z)$ 的**联合分布 (joint distribution)**：

$$
p(x,z) = p(x\mid z)\,p(z)
$$

- $p(z)$：标签的先验分布 (prior over classes)，比如类别 $1$、$2$ 的概率；
- $p(x\mid z)$：在类别为 $z$ 的条件下，特征 $x$ 的分布 (class-conditional distribution)。

如果 $z$ 是已知标签，没什么问题。

**无监督聚类**：我们没有 $z$（不知道每个点属于哪个簇），只能看到 $x$。  
那我们只能建模**边缘分布 (marginal distribution)**：

$$
p(x)
= \sum_z p(x,z)
= \sum_z p(x\mid z) p(z)
$$

> 这就是**混合模型 (mixture model)** 的一般形式：  
> **“由不同成分（簇）按一定权重混合而成的分布”**。

---

## 2. 高斯混合模型 GMM（Gaussian Mixture Model）

### 2.1 定义

最常用的混合模型：**高斯混合模型 (Gaussian Mixture Model, GMM)**

GMM 假设：  
$$
p(x) = \sum_{k=1}^K \omega_k \, N(x\mid \mu_k, \Sigma_k)
$$

其中：

- $\omega_k$：混合系数 (mixing coefficient)，满足
  $$
  \omega_k \ge 0,\quad \sum_{k=1}^K \omega_k = 1
  $$
- $N(x\mid \mu_k, \Sigma_k)$：第 $k$ 个高斯成分的密度。

### 2.2 多元高斯分布公式（一定要会）

维度为 $d$ 时，多元高斯密度：

$$
N(x\mid \mu_k, \Sigma_k)
=
\frac{1}{\sqrt{(2\pi)^d\,|\Sigma_k|}}
\exp\left(
-\frac{1}{2}
(x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k)
\right)
$$

解释：

- $(x-\mu_k)$：偏离均值的向量；
- $(x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k)$：**Mahalanobis 距离 (Mahalanobis distance)** 的平方  
  （考虑到了不同方向上的方差不同、特征相关性）；
- $|\Sigma_k|$：决定“体积”，方差越大，密度峰值越低。

### 2.3 GMM 的能力：通用密度近似器

只要给足够多的高斯成分 $K$，GMM 可以逼近非常复杂的分布形状（多峰、弯曲等），所以说：

> **GMM 是一个非常灵活的密度估计器 (density estimator)**，  
> 理论上可以作为“通用近似器 (universal approximator of densities)”。

---

## 3. 用极大似然 (Maximum Likelihood) 拟合 GMM

### 3.1 数据集与参数

数据集：

$$
X = \{x^{(1)},\dots,x^{(N)}\}
$$

参数集合（全部记成 $\theta$ 或 $\varepsilon$）：

$$
\theta = \{\omega_1,\dots,\omega_K;\ \mu_1,\dots,\mu_K;\ \Sigma_1,\dots,\Sigma_K\}
$$

### 3.2 似然函数和对数似然

对每个样本 $x^{(n)}$，在 GMM 下的密度是：

$$
p(x^{(n)} \mid \theta)
= \sum_{k=1}^K \omega_k N(x^{(n)}\mid \mu_k,\Sigma_k)
$$

假设样本独立同分布 (i.i.d.)，则**似然函数 (likelihood)**：

$$
L(\theta)
= p(X\mid\theta)
= \prod_{n=1}^N p(x^{(n)}\mid\theta)
= \prod_{n=1}^N 
\left(
\sum_{k=1}^K \omega_k N(x^{(n)}\mid\mu_k,\Sigma_k)
\right)
$$

通常优化**对数似然 (log-likelihood)**：

$$
\ell(\theta)
= \log L(\theta)
= \sum_{n=1}^N \log
\left(
\sum_{k=1}^K \omega_k N(x^{(n)}\mid\mu_k,\Sigma_k)
\right)
$$

> 难点：里面有 $\log(\sum_k \cdot)$，  
> 即所谓 **log-sum-exp** 结构。  
> 直接对 $\mu_k,\Sigma_k,\omega_k$ 求导并令 0，**得不到简单闭式解**。

这就是我们需要 EM 的根源。不过在讲 EM 之前，先展示一下“直接从 MLE 形式推导更新公式”的思路。

---

## 4. 从 MLE 直接推导 GMM 的交替更新（带责任度 γ）

核心技巧：引入一个“软标签” $\gamma_k^{(n)}$，表示第 $n$ 个数据点由第 $k$ 个高斯生成的**后验概率 (posterior probability)**，也叫**责任度 (responsibility)**。

### 4.1 定义责任度 γ

根据贝叶斯公式：

$$
\gamma_k^{(n)}
= p(z^{(n)}=k \mid x^{(n)},\theta)
= \frac{
\omega_k\,N(x^{(n)}\mid\mu_k,\Sigma_k)
}{
\sum_{j=1}^K \omega_j\,N(x^{(n)}\mid\mu_j,\Sigma_j)
}
$$

解释：

- 分子：  
  $p(z^{(n)}=k)\,p(x^{(n)}\mid z^{(n)}=k)$  
  = “属于第 $k$ 类的先验概率” × “在该类下生成此点的概率”

- 分母：  
  $p(x^{(n)})$ = 所有类别之和

$\gamma_k^{(n)}$ 满足：

- $\gamma_k^{(n)} \ge 0$  
- $\sum_{k=1}^K \gamma_k^{(n)} = 1$（对固定 $n$，是一个**类别分布**）

再定义：

$$
N_k = \sum_{n=1}^N \gamma_k^{(n)}
$$

- $N_k$：第 $k$ 个高斯“负责”的有效数据点数量（软计数，effective count）。

有了这些量，更新公式会非常漂亮。

---

### 4.2 推导均值更新公式 $\mu_k$

我们从对数似然出发：

$$
\ell(\theta)
= \sum_{n=1}^N \log
\left(
\sum_{j=1}^K \omega_j N(x^{(n)}\mid\mu_j,\Sigma_j)
\right)
$$

对 $\mu_k$ 求偏导：

#### 第一步：对 log-sum 结构求导（链式法则）

对任意 $n$，

令：
$$
f_n(\theta)
= \sum_{j=1}^K \omega_j N(x^{(n)}\mid\mu_j,\Sigma_j)
$$

那么：
$$
\ell(\theta)
= \sum_{n=1}^N \log f_n(\theta)
$$

对 $\mu_k$：

$$
\frac{\partial \ell}{\partial \mu_k}
= \sum_{n=1}^N
\frac{1}{f_n(\theta)}
\frac{\partial f_n(\theta)}{\partial \mu_k}
$$

注意 $f_n(\theta)$ 对 $\mu_k$ 的依赖只在第 $j=k$ 项上：

$$
f_n(\theta)
= \sum_{j=1}^K \omega_j N(x^{(n)}\mid\mu_j,\Sigma_j)
\Rightarrow
\frac{\partial f_n}{\partial \mu_k}
= \omega_k \frac{\partial}{\partial \mu_k} N(x^{(n)}\mid\mu_k,\Sigma_k)
$$

因此：

$$
\frac{\partial \ell}{\partial \mu_k}
= \sum_{n=1}^N
\frac{1}{\sum_{j} \omega_j N_j^{(n)}}
\omega_k \frac{\partial N_k^{(n)}}{\partial \mu_k}
$$

其中简记：
$$
N_k^{(n)} = N(x^{(n)}\mid\mu_k,\Sigma_k)
$$

#### 第二步：多元高斯对均值的导数

对于高斯密度：

$$
N(x\mid\mu_k,\Sigma_k)
=
\frac{1}{\sqrt{(2\pi)^d |\Sigma_k|}}
\exp\left(
-\frac{1}{2}
(x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k)
\right)
$$

先对 $\mu_k$ 求对数的导数：

1. 取对数：
   $$
   \log N(x\mid\mu_k,\Sigma_k)
   = \text{const} 
   - \frac{1}{2}
   (x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k)
   $$

2. 对 $\mu_k$ 求导：
   - 注意 $(x-\mu_k)$ 是 $\mu_k$ 的函数；
   - 展开二次型：
     $$
     (x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k)
     = (x^\top - \mu_k^\top)\Sigma_k^{-1}(x-\mu_k)
     $$
     可以证明（这是标准结果）：
     $$
     \frac{\partial}{\partial \mu_k}
     \left[
     (x-\mu_k)^\top \Sigma_k^{-1}(x-\mu_k)
     \right]
     = -2 \Sigma_k^{-1}(x-\mu_k)
     $$

   于是：
   $$
   \frac{\partial}{\partial \mu_k}
   \log N(x\mid\mu_k,\Sigma_k)
   = -\frac{1}{2} \cdot (-2)\Sigma_k^{-1}(x-\mu_k)
   = \Sigma_k^{-1}(x-\mu_k)
   $$

3. 由链式法则：
   $$
   \frac{\partial N(x\mid\mu_k,\Sigma_k)}{\partial \mu_k}
   = N(x\mid\mu_k,\Sigma_k)
     \cdot \frac{\partial}{\partial \mu_k}\log N(x\mid\mu_k,\Sigma_k)
   $$
   即：
   $$
   \frac{\partial N(x\mid\mu_k,\Sigma_k)}{\partial \mu_k}
   = N(x\mid\mu_k,\Sigma_k)\, \Sigma_k^{-1}(x-\mu_k)
   $$

代回 $N_k^{(n)}$ 的导数：

$$
\frac{\partial N_k^{(n)}}{\partial \mu_k}
= N_k^{(n)} \Sigma_k^{-1}(x^{(n)}-\mu_k)
$$

#### 第三步：整理成 γ 的形式

回到：

$$
\frac{\partial \ell}{\partial \mu_k}
= \sum_{n=1}^N
\frac{1}{\sum_j \omega_j N_j^{(n)}}
\omega_k N_k^{(n)} \Sigma_k^{-1}(x^{(n)}-\mu_k)
$$

定义责任度：

$$
\gamma_k^{(n)}
= \frac{\omega_k N_k^{(n)}}{\sum_j \omega_j N_j^{(n)}}
$$

于是：

$$
\frac{\partial \ell}{\partial \mu_k}
= \sum_{n=1}^N
\gamma_k^{(n)} 
\Sigma_k^{-1}(x^{(n)}-\mu_k)
$$

设这个导数为 0（极大值点的一阶条件）：

$$
\sum_{n=1}^N
\gamma_k^{(n)} 
\Sigma_k^{-1}(x^{(n)}-\mu_k)
= 0
$$

左乘 $\Sigma_k$（可逆）：

$$
\sum_{n=1}^N
\gamma_k^{(n)} 
(x^{(n)}-\mu_k)
= 0
$$

展开：

$$
\sum_{n=1}^N
\gamma_k^{(n)} x^{(n)}
-
\sum_{n=1}^N
\gamma_k^{(n)} \mu_k
= 0
$$

注意 $\mu_k$ 不依赖于 $n$，可以提出和号：

$$
\sum_{n=1}^N
\gamma_k^{(n)} x^{(n)}
-
\mu_k \sum_{n=1}^N \gamma_k^{(n)}
= 0
$$

定义：

$$
N_k = \sum_{n=1}^N \gamma_k^{(n)}
$$

于是：

$$
\sum_{n=1}^N
\gamma_k^{(n)} x^{(n)}
= \mu_k N_k
$$

解出 $\mu_k$：

$$
\boxed{
\mu_k
= \frac{1}{N_k}
\sum_{n=1}^N
\gamma_k^{(n)} x^{(n)}
}
$$

解释：

- $\mu_k$ 是所有样本 $x^{(n)}$ 的**加权平均**，权重是责任度 $\gamma_k^{(n)}$；
- 这就是“**按软分配后的加权中心**”。

---

### 4.3 推导协方差更新公式 $\Sigma_k$

同样从对数似然出发，对 $\Sigma_k$ 求导较麻烦，需要一些矩阵求导公式。  
最后结果是：

$$
\boxed{
\Sigma_k
= \frac{1}{N_k}
\sum_{n=1}^N
\gamma_k^{(n)}
(x^{(n)}-\mu_k)(x^{(n)}-\mu_k)^\top
}
$$

直观理解：

- 就是“**加权协方差**”；
- 权重是 $\gamma_k^{(n)}$。

略示推导思路（不要求你会手算，但知道用的工具）：

1. 仍然用责任度把 log-likelihood 写为：

   $$
   \ell(\theta)
   = \sum_{n,k} \gamma_k^{(n)} \log N(x^{(n)}\mid\mu_k,\Sigma_k)
   + \text{其它与 }\Sigma_k\text{ 无关项}
   $$

2. 把 $\log N$ 展开，关注 $\Sigma_k$ 部分：

   $$
   \log N(x^{(n)}\mid\mu_k,\Sigma_k)
   = -\frac{1}{2}\log|\Sigma_k|
     -\frac{1}{2}(x^{(n)}-\mu_k)^\top \Sigma_k^{-1}(x^{(n)}-\mu_k)
     + \text{常数}
   $$

3. 使用矩阵微积分中的两个公式（可以背）：

   - $\frac{\partial}{\partial \Sigma} \log|\Sigma| = (\Sigma^{-1})^\top = \Sigma^{-1}$（对称矩阵）  
   - $\frac{\partial}{\partial \Sigma^{-1}} (x-\mu)^\top \Sigma^{-1}(x-\mu)
      = (x-\mu)(x-\mu)^\top$

4. 令导数为 0，整理得出上面的协方差更新形式。

---

### 4.4 推导混合系数更新公式 $\omega_k$

记住约束：$\sum_{k=1}^K \omega_k = 1$。

我们使用**拉格朗日乘子法 (Lagrange multipliers)**。

#### 第一步：构造拉格朗日函数

$$
\tilde{\ell}(\theta,\lambda)
= \ell(\theta)
+ \lambda \left( \sum_{k=1}^K \omega_k - 1 \right)
$$

对 $\omega_k$ 求偏导并令 0：

$$
\frac{\partial \tilde{\ell}}{\partial \omega_k}
= \frac{\partial \ell}{\partial \omega_k} + \lambda = 0
$$

#### 第二步：求 $\frac{\partial \ell}{\partial \omega_k}$

仍从：

$$
\ell(\theta)
= \sum_{n=1}^N \log
\left(
\sum_{j=1}^K \omega_j N_j^{(n)}
\right)
$$

对固定 $n$：

$$
\frac{\partial}{\partial \omega_k}
\log
\left(
\sum_{j=1}^K \omega_j N_j^{(n)}
\right)
=
\frac{1}{\sum_j \omega_j N_j^{(n)}}
\frac{\partial}{\partial \omega_k}
\left(
\sum_{j=1}^K \omega_j N_j^{(n)}
\right)
=
\frac{N_k^{(n)}}{\sum_j \omega_j N_j^{(n)}}
$$

注意：

$$
\gamma_k^{(n)}
= \frac{\omega_k N_k^{(n)}}{\sum_j \omega_j N_j^{(n)}}
\Rightarrow
\frac{N_k^{(n)}}{\sum_j \omega_j N_j^{(n)}}
= \frac{\gamma_k^{(n)}}{\omega_k}
$$

于是：

$$
\frac{\partial \ell}{\partial \omega_k}
= \sum_{n=1}^N \frac{\gamma_k^{(n)}}{\omega_k}
$$

#### 第三步：联立约束求解

代回：

$$
\frac{\partial \tilde{\ell}}{\partial \omega_k}
= \sum_{n=1}^N \frac{\gamma_k^{(n)}}{\omega_k}
+ \lambda = 0
$$

乘以 $\omega_k$：

$$
\sum_{n=1}^N \gamma_k^{(n)}
+ \lambda \omega_k = 0
$$

记 $N_k = \sum_{n=1}^N \gamma_k^{(n)}$，得到：

$$
N_k + \lambda \omega_k = 0
\Rightarrow
\omega_k = -\frac{N_k}{\lambda}
$$

对所有 $k$ 求和，利用约束 $\sum_k \omega_k = 1$：

$$
\sum_{k=1}^K \omega_k
= -\frac{1}{\lambda} \sum_{k=1}^K N_k
= 1
$$

而：

$$
\sum_{k=1}^K N_k
= \sum_{k=1}^K \sum_{n=1}^N \gamma_k^{(n)}
= \sum_{n=1}^N \sum_{k=1}^K \gamma_k^{(n)}
= \sum_{n=1}^N 1
= N
$$

所以：

$$
-\frac{1}{\lambda} N = 1
\Rightarrow
\lambda = -N
$$

代回：

$$
\omega_k
= -\frac{N_k}{\lambda}
= \frac{N_k}{N}
$$

**最终结论：**

$$
\boxed{
\omega_k
= \frac{N_k}{N}
= \frac{1}{N}
\sum_{n=1}^N \gamma_k^{(n)}
}
$$

含义：

- 第 $k$ 个混合系数 = “平均来说，有多少比例的数据属于第 $k$ 个成分”。

---

### 4.5 GMM 交替更新算法（就是 EM，但从 MLE 直接看）

算法步骤（每次迭代）：

1. **E-step（责任度计算）**  

   给定当前参数 $(\omega_k,\mu_k,\Sigma_k)$，对每一个样本 $x^{(n)}$ 和每一个成分 $k$，计算：

   $$
   \gamma_k^{(n)}
   = \frac{
     \omega_k N(x^{(n)}\mid\mu_k,\Sigma_k)
   }{
     \sum_{j=1}^K \omega_j N(x^{(n)}\mid\mu_j,\Sigma_j)
   }
   $$

2. **M-step（参数更新）**

   先算有效样本数：

   $$
   N_k
   = \sum_{n=1}^N \gamma_k^{(n)}
   $$

   然后更新：

   $$
   \mu_k
   = \frac{1}{N_k} \sum_{n=1}^N \gamma_k^{(n)} x^{(n)}
   $$
   $$
   \Sigma_k
   = \frac{1}{N_k} \sum_{n=1}^N \gamma_k^{(n)}
     (x^{(n)} - \mu_k)(x^{(n)} - \mu_k)^\top
   $$
   $$
   \omega_k
   = \frac{N_k}{N}
   $$

3. 计算新的对数似然 $\ell(\theta)$，检查是否收敛（变化很小则停止）。

> 这其实就是后面要讲的 EM 算法在 GMM 上的一个特例形式。

---

## 5. GMM vs K-Means 的对比

### 5.1 K-Means 聚类回顾

- 要求聚成 $K$ 个簇；
- **Assignment step（分配步骤）**：  
  每个点被硬分配 (hard assignment) 到最近的簇中心：
  $$
  c^{(n)} = \arg\min_k \|x^{(n)} - \mu_k\|^2
  $$
- **Refitting step（重拟合步骤）**：  
  每个簇中心取所有被分配到它的点的**平均**：
  $$
  \mu_k
  = \frac{1}{|C_k|}
    \sum_{n \in C_k} x^{(n)}
  $$
  其中 $C_k$ 是被分到第 $k$ 簇的样本集合。

特点：

- 簇形状是“圆球”+“硬分配”；
- 只是基于欧式距离，**没有概率意义**。

### 5.2 GMM + EM（软 K-Means）

- GMM 中：
  - **E-step**：计算 $\gamma_k^{(n)}$，是“软分配”：每个点对每个簇都有一个概率；
  - **M-step**：用这些概率作为权重，做加权平均和加权协方差。

- 如果我们强行把所有 $\Sigma_k$ 固定为单位矩阵 $I$，
  并（或）强行令所有 $\omega_k = 1/K$，  
  那么：

  - 责任度的形式接近于对距离的 softmax：
    $$
    \gamma_k^{(n)}
    \propto
    \exp\left(-\frac{1}{2}\|x^{(n)}-\mu_k\|^2\right)
    $$
  - M-step 就是不同比例的“加权均值”。

- **类比：**
  - K-Means：硬分配（权重只有 0 或 1）；  
  - GMM-EM：软分配（权重是 $(0,1)$ 之间的概率）。

> 因此，GMM 可以看作是 **K-Means 的概率版 / 软版**，  
> 同时还能通过协方差矩阵刻画“椭圆形簇”。

---

## 6. 隐变量视角 (Latent Variable View) & LVM

### 6.1 隐变量模型 (Latent Variable Model, LVM)

**定义**：  
有些变量在训练或测试时是看不到的（从不观测），我们叫它们**隐变量 (latent variable)** 或**隐藏变量 (hidden variable)**。

- 观测变量 (observed variables)：$x$  
- 隐变量 (latent variables)：$z$

**隐变量模型 (Latent Variable Model)**：  
用一些不可见的 $z$ 来解释观测到的 $x$，比如：

- GMM：$z$ 表示“是哪个高斯组件”；
- 主题模型：$z$ 表示“是哪个主题”；
- 因子分析：$z$ 是连续潜在因素。

按隐变量类型分：

- 连续隐变量模型：如因子分析 (factor analysis)
- 离散隐变量模型：如混合模型 (mixture models, 包括 GMM)

### 6.2 GMM 的隐变量表达

在 GMM 中，我们显式引入 $z$：

1. **先验分布 (prior over z)**：

   $$
   z \sim \text{Categorical}(\omega)
   $$
   即：
   $$
   p(z=k) = \omega_k,\quad
   \omega_k\ge 0,\ \sum_k \omega_k=1
   $$

2. **条件分布 (likelihood)**：

   $$
   p(x\mid z=k) = N(x\mid \mu_k,\Sigma_k)
   $$

则边缘分布：

$$
p(x)
= \sum_{k=1}^K p(z=k)p(x\mid z=k)
= \sum_{k=1}^K \omega_k N(x\mid\mu_k,\Sigma_k)
$$

这和之前 GMM 的定义完全一致，只是现在把“背后哪个高斯在生成 $x$”的选择用隐变量 $z$ 表示出来。

---

## 7. Jensen 不等式 (Jensen’s Inequality) 预备知识

### 7.1 凸函数/凹函数 (convex / concave functions)

- 函数 $f$ 是**凸的 (convex)**，如果对任意 $x,y$ 以及 $t\in[0,1]$：
  $$
  f(tx + (1-t)y)
  \le
  t f(x) + (1-t) f(y)
  $$

- 函数 $f$ 是**凹的 (concave)**，如果对任意 $x,y$ 以及 $t\in[0,1]$：
  $$
  f(tx + (1-t)y)
  \ge
  t f(x) + (1-t) f(y)
  $$

**对数函数 $\log$ 是凹函数**。

### 7.2 Jensen 不等式（非常重要）

**定理（Jensen’s Inequality）**：

- 若 $f$ 是凸函数，$X$ 是随机变量，则：
  $$
  f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]
  $$

- 若 $f$ 是凹函数，则：
  $$
  f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]
  $$

因为 $\log$ 是凹的，因此：

$$
\log \mathbb{E}[Y]
\ge
\mathbb{E}[\log Y]
$$

我们后面会用这个来把 $\log \sum$ 变成 $\sum \log$ 的形式，从而构造**下界 (lower bound)**。

---

## 8. 一般隐变量模型的对数似然分解（ELBO）

### 8.1 记号再整理一次

- 观测数据：$D=\{x^{(1)},\dots,x^{(N)}\}$  
- 每个数据 $x^{(n)}$ 关联一个隐变量 $z^{(n)}$  
- 整个隐变量集合：$z=\{z^{(1)},\dots,z^{(N)}\}$  
- 参数：$\varepsilon$（包括所有我们要学的参数，比如 $\omega,\mu,\Sigma$）

目标：最大化对数似然：

$$
\log p(D;\varepsilon)
= \sum_{n=1}^N \log p(x^{(n)};\varepsilon)
$$

其中：

$$
p(x^{(n)};\varepsilon)
= \sum_{z^{(n)}} p(z^{(n)}, x^{(n)};\varepsilon)
$$

因为 $z^{(n)}$ 不可见，“把 $z^{(n)}$ 消掉（做和/积分）”之后，形式会变复杂。

### 8.2 引入辅助分布 $q(z)$

我们**人为引入**一组辅助分布：

- 对于每个 $n$，定义 $q_n(z^{(n)})$，是关于 $z^{(n)}$ 的一个分布；
- 假设不同样本的隐变量分布相互独立：

  $$
  q(z) = \prod_{n=1}^N q_n(z^{(n)})
  $$

注意：

- $q_n(z^{(n)})$ 一开始是**任意的**，不是模型的一部分；
- 它不是 $p(z;\omega)$，也不是 $p(z\mid x;\varepsilon)$；
- 后面我们会选一个最优的 $q$ 来帮助优化。

### 8.3 对单个样本的 log-likelihood 分解

先看单个 $(x,z)$：

$$
\log p(x;\varepsilon)
= \log \sum_z p(x,z;\varepsilon)
$$

往里乘一个 $q(z)$，再除以 $q(z)$（相当于乘以 1）：

$$
\log p(x;\varepsilon)
= \log \sum_z q(z)\,\frac{p(x,z;\varepsilon)}{q(z)}
$$

把求和看成是关于 $z$ 的期望（记 $E_q$ 为对 $q(z)$ 的期望）：

$$
\sum_z q(z)\,\frac{p(x,z;\varepsilon)}{q(z)}
= \mathbb{E}_{q(z)}
\left[
\frac{p(x,z;\varepsilon)}{q(z)}
\right]
$$

于是：

$$
\log p(x;\varepsilon)
= \log \mathbb{E}_{q(z)}
\left[
\frac{p(x,z;\varepsilon)}{q(z)}
\right]
$$

因为 $\log$ 是凹函数，应用 Jensen 不等式：

$$
\log p(x;\varepsilon)
\ge
\mathbb{E}_{q(z)}
\left[
\log \frac{p(x,z;\varepsilon)}{q(z)}
\right]
$$

定义：

$$
\mathcal{L}(q;\varepsilon)
=
\mathbb{E}_{q(z)}
\left[
\log p(x,z;\varepsilon)
- \log q(z)
\right]
$$

于是对单个样本，有：

$$
\log p(x;\varepsilon)
\ge
\mathcal{L}(q;\varepsilon)
$$

$\mathcal{L}$ 就是所谓的 **ELBO（evidence lower bound, 证据下界）**。

### 8.4 用 KL 散度的等价分解

我们可以进一步把上式写成：

$$
\log p(x;\varepsilon)
= \mathcal{L}(q;\varepsilon)
+ \mathrm{KL}\big(q(z)\,\|\,p(z\mid x;\varepsilon)\big)
$$

推导如下：

1. 从贝叶斯公式：
   $$
   p(z\mid x;\varepsilon) = \frac{p(x,z;\varepsilon)}{p(x;\varepsilon)}
   $$

2. 定义 KL 散度：
   $$
   \mathrm{KL}\big(q(z)\,\|\,p(z\mid x;\varepsilon)\big)
   =
   \mathbb{E}_{q(z)}
   \left[
     \log \frac{q(z)}{p(z\mid x;\varepsilon)}
   \right]
   $$

   展开分子分母：
   $$
   \begin{aligned}
   \mathrm{KL}
   &= \mathbb{E}_q
   \left[
     \log q(z)
     - \log p(z\mid x;\varepsilon)
   \right]
   \\[4pt]
   &= \mathbb{E}_q
   \left[
     \log q(z)
     - \log \frac{p(x,z;\varepsilon)}{p(x;\varepsilon)}
   \right]
   \\[4pt]
   &= \mathbb{E}_q
   \left[
     \log q(z)
     - \log p(x,z;\varepsilon)
     + \log p(x;\varepsilon)
   \right]
   \\[4pt]
   &= \mathbb{E}_q[\log q(z)]
     - \mathbb{E}_q[\log p(x,z;\varepsilon)]
     + \log p(x;\varepsilon)
   \end{aligned}
   $$

3. 注意 $\log p(x;\varepsilon)$ 与 $z$ 无关，可以从期望里拿出来：

   $$
   \mathrm{KL}
   = -\mathcal{L}(q;\varepsilon)
     + \log p(x;\varepsilon)
   $$

   整理得：

   $$ 
   \log p(x;\varepsilon)
   = \mathcal{L}(q;\varepsilon)
   + \mathrm{KL}\big(q(z)\,\|\,p(z\mid x;\varepsilon)\big)
   $$

由于 KL 散度总是 $\ge 0$，所以 $\mathcal{L}(q;\varepsilon)$ 确实是 $\log p(x;\varepsilon)$ 的**下界**。

### 8.5 拓展到整个数据集

对每个样本 $x^{(n)}$ 都有一个 $q_n(z^{(n)})$，  
于是整个数据集：

$$
\log p(D;\varepsilon)
=
\sum_{n=1}^N \log p(x^{(n)};\varepsilon)
$$

每一项都有上面的分解：

$$
\log p(x^{(n)};\varepsilon)
=
\mathcal{L}_n(q_n;\varepsilon)
+ \mathrm{KL}\big( q_n(z^{(n)}) \,\|\, p(z^{(n)}\mid x^{(n)};\varepsilon) \big)
$$

求和得：

$$
\log p(D;\varepsilon)
= \underbrace{
  \sum_{n=1}^N \mathcal{L}_n(q_n;\varepsilon)
}_{\mathcal{L}(q;\varepsilon)}
+ \sum_{n=1}^N
\mathrm{KL}\big( q_n(z^{(n)}) \,\|\, p(z^{(n)}\mid x^{(n)};\varepsilon) \big)
\tag{1}
$$

记总的下界：

$$
\mathcal{L}(q;\varepsilon)
= \sum_{n=1}^N 
\mathbb{E}_{q_n(z^{(n)})}
\left[
  \log \frac{p(x^{(n)},z^{(n)};\varepsilon)}{q_n(z^{(n)})}
\right]
$$

---

## 9. EM 算法的一般形式（General EM Algorithm）

目标：最大化 $\log p(D;\varepsilon)$ 很难，  
但从式 (1) 知道：

- $\log p(D;\varepsilon)$ **≥** $\mathcal{L}(q;\varepsilon)$  
- 而且二者相差一个**非负的 KL 散度和**

于是我们可以：

> 通过最大化 $\mathcal{L}(q;\varepsilon)$ 的方式，间接提高 $\log p(D;\varepsilon)$。

这就是 EM 背后的思想。

### 9.1 坐标上升 (coordinate ascent) 思路

我们把变量分成两块：

- **隐变量分布 $q(z)$**（辅助分布）
- **模型参数 $\varepsilon$**

采用交替优化：

1. 给定 $\varepsilon$，最大化 $\mathcal{L}(q;\varepsilon)$ 对 $q$；
2. 给定 $q$，最大化 $\mathcal{L}(q;\varepsilon)$ 对 $\varepsilon$。

就是所谓的**E-step** 和 **M-step**。

---

### 9.2 E-step：固定 $\varepsilon$，优化 $q(z)$

从式 (1)：

$$
\log p(D;\varepsilon)
= \mathcal{L}(q;\varepsilon)
+ \sum_{n=1}^N
\mathrm{KL}\big(q_n(z^{(n)}) \,\|\, p(z^{(n)}\mid x^{(n)};\varepsilon)\big)
$$

对于固定的 $\varepsilon$，左边 $\log p(D;\varepsilon)$ 是常数。  
由于 KL 散度 $\ge 0$，要让 $\mathcal{L}(q;\varepsilon)$ 尽可能大，  
等价于让每一项 KL 尽可能小：

- 最小值是 0；
- 当且仅当 $q_n(z^{(n)}) = p(z^{(n)}\mid x^{(n)};\varepsilon)$ 时达到。

**因此**：

$$
\boxed{
q_n(z^{(n)})
= p(z^{(n)}\mid x^{(n)};\varepsilon_{\text{old}})
}
$$

这就是 EM 的 **E-step（期望步）**：

- 计算“后验分布 (posterior)” $p(z^{(n)}\mid x^{(n)};\varepsilon_{\text{old}})$；
- 作为最优的 $q_n$。

此时，有一个重要结论：

- 每个 KL 散度都变为 0；
- 所以：
  $$
  \mathcal{L}(q;\varepsilon_{\text{old}})
  = \log p(D;\varepsilon_{\text{old}})
  $$
  下界和真对数似然**完全相等**（下界“贴紧”原函数）。

---

### 9.3 M-step：固定 $q(z)$，优化 $\varepsilon$

现在把 $q$ 固定（刚刚在 E-step 得到的最优 $q$），  
我们要最大化 $\mathcal{L}(q;\varepsilon)$ 对 $\varepsilon$：

$$
\varepsilon_{\text{new}}
= \arg\max_\varepsilon
\mathcal{L}(q;\varepsilon)
$$

回忆：

$$
\mathcal{L}(q;\varepsilon)
=
\sum_{n=1}^N
\mathbb{E}_{q_n(z^{(n)})}
\big[
\log p(x^{(n)},z^{(n)};\varepsilon)
\big]
-
\sum_{n=1}^N
\mathbb{E}_{q_n(z^{(n)})}
\big[
\log q_n(z^{(n)})
\big]
$$

第二项与 $\varepsilon$ 无关，可以当成常数。这时 M-step 等价于最大化：

$$
\sum_{n=1}^N
\mathbb{E}_{q_n(z^{(n)})}
\big[
\log p(x^{(n)},z^{(n)};\varepsilon)
\big]
$$

这叫做 **expected complete-data log-likelihood（期望完全数据对数似然）**。

> 总结：
> - E-step：求 $q_n(z^{(n)}) = p(z^{(n)}\mid x^{(n)};\varepsilon_{\text{old}})$；
> - M-step：用这个 $q$，最大化完全数据对数似然的期望。

---

### 9.4 EM 收敛性：为什么 log-likelihood 单调上升

设当前参数为 $\varepsilon_{\text{old}}$，E-step 得到 $q$，M-step 得到 $\varepsilon_{\text{new}}$。

我们有三步不等式（对应课件）：

1. 由于 $\log p(D;\varepsilon) \ge \mathcal{L}(q;\varepsilon)$ 对任意 $q,\varepsilon$ 成立：

   $$
   \log p(D;\varepsilon_{\text{new}})
   \ge
   \mathcal{L}(q;\varepsilon_{\text{new}})
   $$

2. M-step 中我们选择 $\varepsilon_{\text{new}}$ 使 $\mathcal{L}(q;\varepsilon)$ 最大，所以：

   $$
   \mathcal{L}(q;\varepsilon_{\text{new}})
   \ge
   \mathcal{L}(q;\varepsilon_{\text{old}})
   $$

3. 在 E-step 之后，我们有 $\mathcal{L}(q;\varepsilon_{\text{old}}) = \log p(D;\varepsilon_{\text{old}})$（因为 KL=0）：

   $$
   \mathcal{L}(q;\varepsilon_{\text{old}})
   = \log p(D;\varepsilon_{\text{old}})
   $$

三条合在一起：

$$
\log p(D;\varepsilon_{\text{new}})
\ge
\mathcal{L}(q;\varepsilon_{\text{new}})
\ge
\mathcal{L}(q;\varepsilon_{\text{old}})
=
\log p(D;\varepsilon_{\text{old}})
$$

于是：

> 每一次 EM 迭代都**不会降低**对数似然（单调不减），  
> 最终收敛到某个局部极大值或鞍点。

---

## 10. EM 在 GMM 上的具体形式（再回到 GMM）

现在把一般 EM 框架套回 GMM 模型。

### 10.1 GMM 的联合分布

对单个样本 $(x,z)$：

- 先验：
  $$
  p(z=k;\varepsilon) = \omega_k
  $$
- 条件：
  $$
  p(x\mid z=k;\varepsilon) = N(x\mid \mu_k,\Sigma_k)
  $$

联合分布：

$$
p(x,z=k;\varepsilon)
= p(z=k;\varepsilon)\,p(x\mid z=k;\varepsilon)
= \omega_k N(x\mid \mu_k,\Sigma_k)
$$

---

### 10.2 E-step：计算后验 $p(z^{(n)}=k\mid x^{(n)};\varepsilon_{\text{old}})$

根据贝叶斯公式：

$$
p(z^{(n)}=k\mid x^{(n)};\varepsilon_{\text{old}})
=
\frac{
p(z^{(n)}=k;\varepsilon_{\text{old}})\,
p(x^{(n)}\mid z^{(n)}=k;\varepsilon_{\text{old}})
}{
p(x^{(n)};\varepsilon_{\text{old}})
}
$$

代入 GMM 各项：

- $p(z^{(n)}=k;\varepsilon_{\text{old}}) = \omega_k^{\text{old}}$  
- $p(x^{(n)}\mid z^{(n)}=k;\varepsilon_{\text{old}}) = N(x^{(n)}\mid\mu_k^{\text{old}},\Sigma_k^{\text{old}})$  
- $p(x^{(n)};\varepsilon_{\text{old}}) = \sum_{j=1}^K \omega_j^{\text{old}} N(x^{(n)}\mid\mu_j^{\text{old}},\Sigma_j^{\text{old}})$

得到：

$$
\gamma_k^{(n)}
=
p(z^{(n)}=k\mid x^{(n)};\varepsilon_{\text{old}})
=
\frac{
\omega_k^{\text{old}} N(x^{(n)}\mid\mu_k^{\text{old}},\Sigma_k^{\text{old}})
}{
\sum_{j=1}^K
\omega_j^{\text{old}} N(x^{(n)}\mid\mu_j^{\text{old}},\Sigma_j^{\text{old}})
}
$$

这就是 GMM-EM 的 **E-step**。

---

### 10.3 M-step：最大化期望完全数据对数似然

**完全数据对数似然 (complete-data log-likelihood)**：

对所有数据点 $(x^{(n)}, z^{(n)})$：

$$
\log p(D,Z;\varepsilon)
=
\sum_{n=1}^N \log p(x^{(n)},z^{(n)};\varepsilon)
$$

但 $z^{(n)}$ 不可见，我们用 E-step 得到的 $q_n(z^{(n)})$（即后验）来求期望：

$$
\mathbb{E}_{q}
\big[
\log p(D,Z;\varepsilon)
\big]
=
\sum_{n=1}^N
\mathbb{E}_{q_n(z^{(n)})}
\big[
\log p(x^{(n)},z^{(n)};\varepsilon)
\big]
$$

在 GMM 中：

$$
\log p(x^{(n)},z^{(n)}=k;\varepsilon)
= \log \omega_k + \log N(x^{(n)}\mid\mu_k,\Sigma_k)
$$

于是：

$$
\begin{aligned}
\mathbb{E}_{q}
\big[
\log p(D,Z;\varepsilon)
\big]
&=
\sum_{n=1}^N
\sum_{k=1}^K
\gamma_k^{(n)}
\left[
\log \omega_k
+ \log N(x^{(n)}\mid\mu_k,\Sigma_k)
\right]
\end{aligned}
$$

M-step 要在 $\omega_k,\mu_k,\Sigma_k$ 上最大化上述式子。  
这和我们在第 4 节中直接对 log-likelihood 求导得到的形式**完全一致**。  
因此 M-step 的结果就是：

- 有效样本数：
  $$
  N_k = \sum_{n=1}^N \gamma_k^{(n)}
  $$
- 更新均值：
  $$
  \mu_k
  = \frac{1}{N_k}
    \sum_{n=1}^N
    \gamma_k^{(n)} x^{(n)}
  $$
- 更新协方差：
  $$
  \Sigma_k
  = \frac{1}{N_k}
    \sum_{n=1}^N
    \gamma_k^{(n)}
    (x^{(n)} - \mu_k)(x^{(n)} - \mu_k)^\top
  $$
- 更新混合系数：
  $$
  \omega_k
  = \frac{N_k}{N}
  $$

---

### 10.4 EM for GMM：完整算法小结

1. **初始化**  
   随机初始化或用 K-Means 的结果初始化：
   - $\omega_k^{(0)}, \mu_k^{(0)}, \Sigma_k^{(0)}$

2. **循环直到收敛**（迭代 index $t=0,1,2,\dots$）：

   - **E-step**：

     $$
     \gamma_k^{(n)(t+1)}
     =
     \frac{
     \omega_k^{(t)}
     N(x^{(n)}\mid\mu_k^{(t)},\Sigma_k^{(t)})
     }{
     \sum_{j=1}^K
     \omega_j^{(t)}
     N(x^{(n)}\mid\mu_j^{(t)},\Sigma_j^{(t)})
     }
     $$

   - **M-step**：

     $$
     N_k^{(t+1)} = \sum_{n=1}^N \gamma_k^{(n)(t+1)}
     $$
     $$
     \mu_k^{(t+1)}
     =
     \frac{1}{N_k^{(t+1)}}
     \sum_{n=1}^N
     \gamma_k^{(n)(t+1)} x^{(n)}
     $$
     $$
     \Sigma_k^{(t+1)}
     =
     \frac{1}{N_k^{(t+1)}}
     \sum_{n=1}^N
     \gamma_k^{(n)(t+1)}
     (x^{(n)} - \mu_k^{(t+1)})
     (x^{(n)} - \mu_k^{(t+1)})^\top
     $$
     $$
     \omega_k^{(t+1)}
     = \frac{N_k^{(t+1)}}{N}
     $$

3. 每次迭代后计算 $\log p(D;\varepsilon^{(t+1)})$，直到增幅很小或达到最大迭代次数。

---

## 11. GMM & EM 的优点与缺点（考试常考概念）

### 11.1 GMM 的优点

1. **灵活的分布表示 (flexible density representation)**  
   - 多个高斯 + 不同均值 & 协方差 ⇒ 能拟合复杂的、多峰的分布。

2. **概率视角的聚类 (probabilistic clustering)**  
   - 每个样本有一个对各簇的后验概率 $\gamma_k^{(n)}$；
   - 可以做“软聚类 (soft clustering)”而不是硬标签。

3. **适合重叠簇 (overlapping clusters)**  
   - 若两个簇中间有重叠区域，GMM 能用后验概率平滑处理；
   - K-Means 只能硬切分。

4. **密度估计能力 (density estimation)**  
   - 不只是给出簇标签，还给出任意点的密度值 $p(x)$；
   - 可用于异常检测 (outlier detection)：那些密度特别低的点可能为异常点。

5. **缺失数据处理 (handling missing data)**  
   - EM 本身就是为“有隐变量/缺失数据”设计的；
   - 当部分特征缺失时，可以把缺失值也视作隐变量，通过 EM 来估计。

### 11.2 GMM / EM 的缺点

1. **组件数 $K$ 难确定 (choosing K)**  
   - $K$ 太小：欠拟合 (underfitting)；  
   - $K$ 太大：过拟合 (overfitting)；
   - 现实中常用 BIC/AIC 或交叉验证来选。

2. **对初始化敏感 (initialization sensitivity)**  
   - EM 只能保证收敛到**局部最优**；
   - 不同初始参数可能收敛到不同解；
   - 常用：多次随机初始化，或者先用 K-Means 初始化均值。

3. **高斯假设 (Gaussian assumption)**  
   - 如果真实数据分布偏离高斯很多（比如强非对称、重尾分布等），GMM 表示能力受限；
   - 这时可能需要其它更复杂的模型。

4. **高维数据的困难 (curse of dimensionality)**  
   - 每个协方差矩阵 $\Sigma_k$ 是 $d\times d$，参数量 $\mathcal{O}(d^2)$；
   - 高维情况下，估计准确协方差需要大量数据；
   - 可能需要简化协方差结构（对角矩阵、共享协方差等）。

5. **协方差矩阵必须可逆 (non-singular requirement)**  
   - 需要 $\Sigma_k$ 是正定矩阵；
   - 如果有维度高度相关或样本数太少，很容易出现奇异矩阵；
   - 需要正则化（例如 $\Sigma_k + \epsilon I$）。

6. **需要能算出后验 $p(z\mid x;\varepsilon)$**  
   - EM 要求 E-step 能算出隐变量的后验分布；
   - 一些复杂模型的后验没有闭式解，就不能直接用标准 EM，只能用近似方法（变分 EM、MCMC 等）。

---

## 12. 总结：这一讲的知识框架

1. **混合模型 (Mixture models)**  
   - $p(x) = \sum_z p(x\mid z)p(z)$  
   - GMM 是其中一种，$p(x\mid z=k)$ 为高斯。

2. **高斯混合模型 (GMM)**  
   - $p(x) = \sum_{k=1}^K \omega_k N(x\mid\mu_k,\Sigma_k)$  
   - 参数：$\omega_k,\mu_k,\Sigma_k$；  
   - 可以近似复杂分布，是强大的密度估计器。

3. **MLE 拟合 GMM 的 log-likelihood**  
   - $\ell(\theta) = \sum_n \log \sum_k \omega_k N(x^{(n)}\mid\mu_k,\Sigma_k)$  
   - 直接优化困难（log-sum-exp）。

4. **责任度 (responsibilities) $\gamma_k^{(n)}$**  
   - $\gamma_k^{(n)}=p(z^{(n)}=k\mid x^{(n)},\theta)$  
   - 解释为“软标签”、“簇 $k$ 对样本 $n$ 的责任”。

5. **从 MLE 求导推导出更新公式**  
   - 均值：
     $$
     \mu_k
     = \frac{1}{N_k}
     \sum_n \gamma_k^{(n)} x^{(n)}
     $$
   - 协方差：
     $$
     \Sigma_k
     = \frac{1}{N_k}
     \sum_n \gamma_k^{(n)}
     (x^{(n)} - \mu_k)(x^{(n)} - \mu_k)^\top
     $$
   - 混合系数：
     $$
     \omega_k
     = \frac{N_k}{N}
     $$
   - 其中 $N_k = \sum_n \gamma_k^{(n)}$。

6. **GMM 与 K-Means 的关系**  
   - K-Means：硬分配（0/1），仅用距离；
   - GMM：软分配（概率），有协方差，结果是“软 K-Means”。

7. **隐变量模型 (Latent variable models)**  
   - GMM 可以看成 LVM：$z$ 是离散隐变量，描述“哪个高斯生成该点”。

8. **Jensen 不等式 & 对数似然的下界分解**  
   - 定义辅助分布 $q(z)$；
   - 得到：
     $$
     \log p(D;\varepsilon)
     = \mathcal{L}(q;\varepsilon)
     + \sum_n \mathrm{KL}\big(q_n(z^{(n)})\,\|\,p(z^{(n)}\mid x^{(n)};\varepsilon)\big)
     $$
   - $\mathcal{L}(q;\varepsilon)$ 为 ELBO，是 $\log p(D;\varepsilon)$ 的下界。

9. **EM 算法的一般形式**  
   - E-step：固定 $\varepsilon$，令 $q_n(z^{(n)})=p(z^{(n)}\mid x^{(n)};\varepsilon)$，使 KL=0、下界贴紧真值；
   - M-step：固定 $q$，最大化下界 $\mathcal{L}(q;\varepsilon)$，等价于最大化 expected complete-data log-likelihood。

10. **EM 在 GMM 上的具体 E/M 步**  
    - E-step：
      $$
      \gamma_k^{(n)}
      = p(z^{(n)}=k\mid x^{(n)};\varepsilon_{\text{old}})
      $$
    - M-step：对 $\omega_k,\mu_k,\Sigma_k$ 的闭式更新公式。

11. **EM 收敛性**  
    - 每次迭代后，对数似然 $\log p(D;\varepsilon)$ 单调不减；
    - 但只能保证收敛到局部最优。

12. **GMM 的优缺点**  
    - 优点：灵活、软聚类、可处理重叠簇、可做密度估计、异常检测、缺失数据；  
    - 缺点：$K$ 难选、对初始化敏感、高斯假设、维度灾难、协方差可逆性要求、对复杂模型 E-step 可能不可解。

---

如果你需要，我可以在下一步：

- 专门给你整理一页“GMM & EM 公式总表（考试速记版）”；  
- 或者出几道简单数值题（比如 $K=2,d=1$），带你一步一步算一轮 E-step 和 M-step。
  






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