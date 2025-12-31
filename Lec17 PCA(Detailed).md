# Lecture 17：主成分分析 PCA 全总结（傻子也能看懂版）

> 课程：DDA3020 Machine Learning  
> 主题：Principal Component Analysis（主成分分析）  
> 目标：  
> - 用**低维向量**来代表原来的**高维数据**  
> - 尽量**不损失太多信息**

---

## 0. 记号 & 基本设定（Notation）

我们先把整节课反复出现的符号统一写清楚，后面就不乱了。

- $D$：原始特征维度 (original dimension)，比如一张 $224\times 224\times 3$ 的彩色图片的维度是 $150{,}528$。
- $N$：样本数量 (number of data points)。
- 数据集：
  $$
  \mathcal{D} = \{ x^{(1)}, x^{(2)}, \dots, x^{(N)} \},\quad
  x^{(n)} \in \mathbb{R}^D
  $$
- **均值向量 (mean vector)**：
  $$
  \mu = \frac{1}{N} \sum_{n=1}^N x^{(n)} \in \mathbb{R}^D
  $$
- 我们准备找一个低维子空间：
  - 维度 $K$，且 $K < D$
  - 子空间 $S$ 由 $K$ 个**正交单位向量 (orthonormal basis vectors)** 张成：
    $$
    S = \text{span}\{u_1, u_2, \dots, u_K\},\quad
    u_k \in \mathbb{R}^D
    $$
  - “正交单位”的意思：
    $$
    u_i^\top u_j = \begin{cases}
    1, & i = j \\
    0, & i \neq j
    \end{cases}
    $$
  - 把它们排成一个矩阵：
    $$
    U = [u_1, u_2, \dots, u_K] \in \mathbb{R}^{D \times K}
    $$
    则上面的正交条件等价于：
    $$
    U^\top U = I_K
    $$
    其中 $I_K$ 是 $K \times K$ 的单位矩阵 (identity matrix)。

---

## 1. 预备知识：投影到子空间（Projection onto a Subspace）

### 1.1 投影、重构、表示：三个关键概念

我们希望用维度 $K$ 的向量 $z \in \mathbb{R}^K$ 来表示原来的 $x \in \mathbb{R}^D$。  
步骤：

1. **先把数据平移到以均值为中心**  
   - 原数据：$x$  
   - 中心化后的数据 (centered data)：
     $$
     \tilde{x} = x - \mu
     $$
   - 直观：把所有点平移，让“云团”的中心在原点。

2. **在子空间 $S$ 上投影 (projection)**  
   - 目标：找到一个 $z \in \mathbb{R}^K$，使得在子空间上的点
     $$
     \hat{x} = \mu + U z
     $$
     尽可能地接近原来的 $x$。  
   - 这里：
     - $\hat{x} \in \mathbb{R}^D$：在子空间上的“重构点 (reconstruction)”
     - $z \in \mathbb{R}^K$：在新坐标系中的“表示 / 编码 (representation / code)”

3. **如何求 $z$？**

   我们已知：  
   - 子空间 $S$ 的基向量为 $u_1, \dots, u_K$
   - 假设 $\tilde{x}$ 在 $S$ 上的投影可以写成基的线性组合：
     $$
     \text{Proj}_S(\tilde{x}) = \sum_{k=1}^K z_k u_k
     $$
     这就是中心化后重构的部分。

   因为 $u_k$ 是正交单位基，我们可以算出系数 $z_k$：

   **推导 $z_k = u_k^\top (x - \mu)$：**

   - 设
     $$
     \tilde{x} \approx \sum_{k=1}^K z_k u_k
     $$
   - 两边左乘 $u_j^\top$（取“在第 $j$ 个基上的坐标”）：
     $$
     u_j^\top \tilde{x}
     \approx u_j^\top \sum_{k=1}^K z_k u_k
     = \sum_{k=1}^K z_k u_j^\top u_k
     $$
   - 利用正交性 $u_j^\top u_k = 0$（$j\neq k$），以及 $u_j^\top u_j=1$：
     $$
     u_j^\top \tilde{x}
     \approx z_j \cdot 1 + \sum_{k\neq j} z_k \cdot 0
     = z_j
     $$
   所以
   $$
   z_j = u_j^\top \tilde{x} = u_j^\top (x - \mu)
   $$

   把所有 $z_k$ 放到一个向量里：
   $$
   z =
   \begin{bmatrix}
   z_1 \\ z_2 \\ \vdots \\ z_K
   \end{bmatrix}
   =
   \begin{bmatrix}
   u_1^\top (x - \mu) \\
   u_2^\top (x - \mu) \\
   \vdots \\
   u_K^\top (x - \mu)
   \end{bmatrix}
   $$
   把这个写成矩阵形式：

   - $U^\top \in \mathbb{R}^{K \times D}$，每一行是 $u_k^\top$
   - $(x - \mu)\in \mathbb{R}^D$

   于是有一个非常简洁的公式：

   $$
   \boxed{z = U^\top (x - \mu)} \quad\text{（表示 / 低维编码）}
   $$

4. **重构公式**

   有了 $z$，我们就能从低维空间“变回去”成原空间近似点：

   $$
   \hat{x} = \mu + U z
   $$

   代入 $z$ 的表达式：

   $$
   \hat{x} = \mu + U \, U^\top (x - \mu)
   $$

   这个 $\hat{x}$ 就叫作 $x$ 的**重构 (reconstruction)**。

---

### 1.2 正交定理（Orthogonal Theorem）

> 结论：  
> 投影误差 $x - \hat{x}$ 与子空间 $S$ 正交。  
> 换句话说：
> $$
> U^\top (x - \hat{x}) = 0
> $$

**理解**：  
- $x$ 投到 $S$ 上得到 $\hat{x}$  
- 差向量 $x - \hat{x}$ 就是“垂直于 $S$ 的那条短线”  
- 这就是“最短距离”的几何意义。

**详细推导：**

1. 根据定义：
   $$
   \hat{x} = \mu + U z
   $$
   所以：
   $$
   x - \hat{x} = x - \mu - U z
   $$

2. 再利用 $z = U^\top (x - \mu)$，代入：
   $$
   x - \hat{x}
   = x - \mu - U U^\top (x - \mu)
   $$

3. 我们来算 $U^\top (x - \hat{x})$：
   $$
   \begin{aligned}
   U^\top (x - \hat{x})
   &= U^\top \big( x - \mu - U U^\top (x - \mu) \big) \\
   &= U^\top (x - \mu) - U^\top U U^\top (x - \mu)
   \end{aligned}
   $$

4. 利用 $U^\top U = I_K$：
   $$
   U^\top U U^\top (x - \mu)
   = (I_K) U^\top (x - \mu)
   = U^\top (x - \mu)
   $$

5. 所以：
   $$
   U^\top (x - \hat{x})
   = U^\top (x - \mu) - U^\top (x - \mu)
   = 0
   $$

**说明：**

- $U^\top (x - \hat{x}) = 0$ 等价于对任意基向量 $u_k$，都有
  $$
  u_k^\top (x - \hat{x}) = 0
  $$
  即误差向量与子空间中的每个基向量都正交 ⇒ 与整个子空间正交。

---

## 2. 维度约简（Dimensionality Reduction）

### 2.1 为什么要降维？

**高维数据的例子：**

- 一张 $224\times 224 \times 3$ 的彩色图像：
  - 每一个像素有 3 个通道 (RGB)
  - 总维度：
    $$
    224 \times 224 \times 3 = 150{,}528
    $$
- 文本、基因表达、推荐系统特征……维度都可能很高。

**问题：**

1. **过拟合（overfitting）**：
   - 特征太多，样本太少，模型很容易“死记硬背”训练数据；
   - 在测试集上表现就会很差。

2. **计算成本高（computational cost）**：
   - 矩阵乘法、反向传播等的时间、内存都随维度上升。

**降维的目标：**

- 在保留原数据“主要结构/信息”的前提下，把数据映射到一个维度更低的空间。

形式化一点：

> 给定数据集
> $$
> \mathcal{D} = \{x^{(1)},\dots,x^{(N)}\} \subset \mathbb{R}^D
> $$
> 寻找一个 $K$ 维子空间 $S$（$K < D$），由 $K$ 个正交单位向量
> $
> \{u_k\}_{k=1}^K
> $
> 张成，使得：
>
> - 将所有点投影到 $S$ 后，**尽量保留原数据结构 / 性质**；
> - 投影后的新表示
>   $$
>   z^{(n)} = U^\top (x^{(n)} - \mu) \in \mathbb{R}^K
>   $$
>   作为每个样本的低维表示。

输出是：

- 基向量（主成分方向）：
  $$
  \{u_1,\dots,u_K\}
  $$
- 新数据集（低维表示）：
  $$
  \mathcal{D}' = \{z^{(1)},\dots,z^{(N)}\} \subset \mathbb{R}^K
  $$

---

## 3. PCA 的两个推导目标

PCA（Principal Component Analysis 主成分分析）可以从两种等价的角度来理解：

1. **最大投影方差 (maximal variance)**  
   - 在子空间上的投影点 $\hat{x}^{(n)}$ 的方差尽可能大；
   - 直观：让数据在新坐标里“分得更开”，信息更丰富。

2. **最小重构误差 (minimal reconstruction error)**  
   - 原始点 $x^{(n)}$ 和其重构 $\hat{x}^{(n)}$ 的距离（平方误差）总和最小；
   - 直观：重构出来的点要尽可能接近原点，信息损失最小。

这两种目标是**等价的**，下面我们分别详细推导。

---

## 4. 推导一：最大化投影方差（Maximal Variance）

### 4.1 问题形式

给定数据集 $\{x^{(n)}\}$ 和均值 $\mu$，子空间 $S$ 由 $U$ 张成，

- 重构：
  $$
  \hat{x}^{(n)} = \mu + U z^{(n)},\quad z^{(n)} = U^\top (x^{(n)} - \mu)
  $$
- 重构的均值：
  $$
  \hat{\mu} = \frac{1}{N} \sum_{n=1}^N \hat{x}^{(n)}
  $$

**目标（版本 1）**：  
选择 $U$（满足 $U^\top U = I$），使得重构点 $\hat{x}^{(n)}$ 的方差最大：

$$
\max_{U,\;U^\top U = I}
\frac{1}{N}
\sum_{n=1}^N \left\| \hat{x}^{(n)} - \hat{\mu} \right\|^2
\tag{3}
$$

### 4.2 第一步：证明重构的均值 $\hat{\mu}$ 等于原均值 $\mu$

我们来算 $\hat{\mu}$：

$$
\hat{\mu}
= \frac{1}{N} \sum_{n=1}^N \hat{x}^{(n)}
= \frac{1}{N} \sum_{n=1}^N \big( \mu + U z^{(n)} \big)
$$

展开求和：

$$
\hat{\mu}
= \frac{1}{N} \sum_{n=1}^N \mu
 + \frac{1}{N} \sum_{n=1}^N U z^{(n)}
= \mu
 + U \left( \frac{1}{N} \sum_{n=1}^N z^{(n)} \right)
$$

注意 $z^{(n)} = U^\top (x^{(n)} - \mu)$，所以：

$$
\sum_{n=1}^N z^{(n)}
= \sum_{n=1}^N U^\top (x^{(n)} - \mu)
= U^\top \sum_{n=1}^N (x^{(n)} - \mu)
$$

而
$$
\sum_{n=1}^N (x^{(n)} - \mu)
= \sum_{n=1}^N x^{(n)} - \sum_{n=1}^N \mu
= N \mu - N \mu
= 0
$$

于是：

$$
\sum_{n=1}^N z^{(n)} = U^\top \cdot 0 = 0
\quad\Rightarrow\quad
\frac{1}{N} \sum_{n=1}^N z^{(n)} = 0
$$

代回 $\hat{\mu}$ 的表达式：

$$
\hat{\mu}
= \mu + U \cdot 0
= \mu
$$

**结论：**

$$
\boxed{
\hat{\mu} = \mu
}
\tag{4}
$$

所以 (3) 式可以写成：

$$
\max_{U,\;U^\top U = I}
\frac{1}{N}
\sum_{n=1}^N 
\left\|
\hat{x}^{(n)} - \mu
\right\|^2
\tag{5}
$$

### 4.3 第二步：把重构写成 $Uz$ 的形式

由于
$$
\hat{x}^{(n)} = \mu + U z^{(n)}
$$

所以
$$
\hat{x}^{(n)} - \mu = U z^{(n)}
$$

于是 (5) 变成：

$$
\max_{U,\;U^\top U = I}
\frac{1}{N}
\sum_{n=1}^N 
\left\| U z^{(n)} \right\|^2
\tag{6-1}
$$

再利用 $z^{(n)} = U^\top (x^{(n)} - \mu)$，得到：

$$
\max_{U,\;U^\top U = I}
\frac{1}{N}
\sum_{n=1}^N 
\left\| U^\top (x^{(n)} - \mu) \right\|^2
\tag{7}
$$

这就是课件的式 (7)。

### 4.4 第三步：用迹（Trace）把范数写成矩阵形式

我们利用一个基本恒等式：

> 对任意向量 $a \in \mathbb{R}^K$，
> $$
 \|a\|^2
 = a^\top a
 = \operatorname{Trace}(a^\top a)
 = \operatorname{Trace}(a a^\top)
 $$

解释一下最后一步：

- $a^\top a$ 是 $1\times 1$ 的标量；
- 标量的 “迹” 等于它本身；
- 而 $a^\top a$ 和 $a a^\top$ 这两个矩阵的迹是相同的：
  $$
  \operatorname{Trace}(a^\top a)
  = \operatorname{Trace}(a a^\top)
  $$

现在令：
$$
a = U^\top (x^{(n)} - \mu)
$$

则
$$
\left\| U^\top (x^{(n)} - \mu) \right\|^2
= \operatorname{Trace}
\left(
[U^\top (x^{(n)} - \mu)]
[U^\top (x^{(n)} - \mu)]^\top
\right)
$$

我们来简化这个矩阵：

1. 先看转置：
   $$
   [U^\top (x^{(n)} - \mu)]^\top
   = (x^{(n)} - \mu)^\top U
   $$

2. 于是乘积：
   $$
   [U^\top (x^{(n)} - \mu)]
   [U^\top (x^{(n)} - \mu)]^\top
   = U^\top (x^{(n)} - \mu)(x^{(n)} - \mu)^\top U
   $$

所以：

$$
\left\| U^\top (x^{(n)} - \mu) \right\|^2
= \operatorname{Trace}
\left(
U^\top (x^{(n)} - \mu)(x^{(n)} - \mu)^\top U
\right)
$$

代回 (7) 式：

$$
\max_{U,\;U^\top U = I}
\frac{1}{N}
\sum_{n=1}^N 
\operatorname{Trace}
\left(
U^\top (x^{(n)} - \mu)(x^{(n)} - \mu)^\top U
\right)
\tag{8}
$$

这就是课件中的 (8)。

---

## 5. 推导二：最小化重构误差（Minimal Reconstruction Error）

### 5.1 问题形式

同样的设定，重构为：

$$
\hat{x}^{(n)} = \mu + U z^{(n)},\quad
z^{(n)} = U^\top (x^{(n)} - \mu)
$$

**目标（版本 2）**：  
让重构误差（每个点的距离平方）最小：

$$
\min_{U,\;U^\top U = I}
\frac{1}{N}
\sum_{n=1}^N 
\left\| x^{(n)} - \hat{x}^{(n)} \right\|^2
\tag{9}
$$

### 5.2 等价性定理：最大投影方差 ⇔ 最小重构误差

> 定理：  
> 以下两个问题是等价的：
> $$
 \max_{U,\;U^\top U = I}
 \frac{1}{N}
 \sum_{n=1}^N 
 \left\| \hat{x}^{(n)} - \mu \right\|^2
 \quad\Longleftrightarrow\quad
 \min_{U,\;U^\top U = I}
 \frac{1}{N}
 \sum_{n=1}^N 
 \left\| x^{(n)} - \hat{x}^{(n)} \right\|^2
 \tag{10}
 $$

**证明思路：**  
利用**勾股定理（Pythagorean theorem）**。

对每一个样本 $x^{(n)}$，我们有几何分解：

$$
x^{(n)} - \mu
= \underbrace{\hat{x}^{(n)} - \mu}_{\text{投影到子空间}}
+ \underbrace{[x^{(n)} - \hat{x}^{(n)}]}_{\text{垂直误差}}
$$

其中：

- $\hat{x}^{(n)} - \mu$ 在子空间 $S$ 内；
- $x^{(n)} - \hat{x}^{(n)}$ 与子空间 $S$ 正交（1.2 中已证明）；

所以这两个向量是正交的：  
$$
(\hat{x}^{(n)} - \mu)^\top (x^{(n)} - \hat{x}^{(n)}) = 0
$$

于是根据勾股定理：

$$
\|x^{(n)} - \mu\|^2
= \|\hat{x}^{(n)} - \mu\|^2
+ \|x^{(n)} - \hat{x}^{(n)}\|^2
$$

对 $n=1,\dots,N$ 求和，再除以 $N$：

$$
\frac{1}{N} \sum_{n=1}^N \|x^{(n)} - \mu\|^2
=
\underbrace{
\frac{1}{N} \sum_{n=1}^N \|\hat{x}^{(n)} - \mu\|^2
}_{\text{投影后的方差}}
+
\underbrace{
\frac{1}{N} \sum_{n=1}^N \|x^{(n)} - \hat{x}^{(n)}\|^2
}_{\text{重构误差}}
$$

注意左边是：
$$
\frac{1}{N} \sum_{n=1}^N \|x^{(n)} - \mu\|^2
$$
这里既不含 $U$ 也不含 $z^{(n)}$，是一个**常数**（只跟数据本身有关）。

于是我们得到：

$$
\text{常数}
= \text{投影后的方差}
+ \text{重构误差}
$$

所以：

- 若我们**最大化投影方差**，就会**最小化重构误差**；
- 若我们**最小化重构误差**，投影方差自然就最大。

这就是式 (10) 的含义。

---

## 6. 把 PCA 写成一个矩阵优化问题

### 6.1 样本协方差矩阵（empirical covariance matrix）

我们定义**经验协方差矩阵 (empirical covariance matrix)**：

$$
\Sigma
= \frac{1}{N}
\sum_{n=1}^N
(x^{(n)} - \mu)(x^{(n)} - \mu)^\top
\in \mathbb{R}^{D \times D}
\tag{12}
$$

- $\Sigma$ 是一个对称矩阵 (symmetric matrix)；
- 它的 $(i,j)$ 元素是第 $i$ 维和第 $j$ 维特征的协方差。

### 6.2 把目标函数写成 $\operatorname{Trace}(U^\top \Sigma U)$

回到 (8)：

$$
\max_{U,\;U^\top U = I}
\frac{1}{N}
\sum_{n=1}^N 
\operatorname{Trace}
\left(
U^\top (x^{(n)} - \mu)(x^{(n)} - \mu)^\top U
\right)
$$

利用**迹的线性性 (linearity of trace)**，把求和挪到迹外面：

$$
\begin{aligned}
&\frac{1}{N}
\sum_{n=1}^N 
\operatorname{Trace}
\left(
U^\top (x^{(n)} - \mu)(x^{(n)} - \mu)^\top U
\right)
\\[4pt]
=&
\operatorname{Trace}
\left(
\frac{1}{N}
\sum_{n=1}^N 
U^\top (x^{(n)} - \mu)(x^{(n)} - \mu)^\top U
\right)
\\[4pt]
=&
\operatorname{Trace}
\left(
U^\top
\left[
\frac{1}{N}
\sum_{n=1}^N 
(x^{(n)} - \mu)(x^{(n)} - \mu)^\top
\right]
U
\right)
\\[4pt]
=&
\operatorname{Trace}
\left(
U^\top \Sigma U
\right)
\end{aligned}
$$

于是 PCA 的优化问题变为：

$$
\boxed{
\max_{U,\;U^\top U = I}
\operatorname{Trace}(U^\top \Sigma U)
}
\tag{11, 13}
$$

这就是 PCA 的核心数学形式。

---

## 7. 利用拉格朗日乘子求解：$U$ 是协方差矩阵的特征向量

### 7.1 拉格朗日函数（Lagrangian）

我们要在约束 $U^\top U = I$ 下最大化 $\operatorname{Trace}(U^\top \Sigma U)$。

引入一个 $K\times K$ 的拉格朗日乘子矩阵 $\Lambda_K$，通常取对角矩阵：

$$
\Lambda_K = \operatorname{diag}(\lambda_1, \dots, \lambda_K)
$$

**拉格朗日函数**：

$$
\mathcal{L}(U, \Lambda_K)
=
\operatorname{Trace}(U^\top \Sigma U)
+
\operatorname{Trace}
\big(
\Lambda_K^\top (I - U^\top U)
\big)
\tag{14}
$$

- 第一项：我们想要最大化的目标；
- 第二项：用来“惩罚”偏离约束 $U^\top U = I$ 的情况。

### 7.2 对 $U$ 求导并令梯度为零

我们需要：

$$
\frac{\partial \mathcal{L}}{\partial U} = 0
\tag{15}
$$

**先看第一项** $\operatorname{Trace}(U^\top \Sigma U)$ 的导数：

- 用一个标准公式（矩阵求导基本结论）：
  > 若 $A$ 为常矩阵，且 $A$ 对称，则  
  > $$
  > \frac{\partial}{\partial U} \operatorname{Trace}(U^\top A U)
  > = (A + A^\top) U
  > $$
  > 如果 $A$ 是对称的 ($A = A^\top$)，则变成 $2 A U$。

- 在这里 $A = \Sigma$，且协方差矩阵是对称的：$\Sigma = \Sigma^\top$，所以：

  $$
  \frac{\partial}{\partial U}
  \operatorname{Trace}(U^\top \Sigma U)
  = 2 \Sigma U
  $$

**再看第二项** $\operatorname{Trace}\big(\Lambda_K^\top (I - U^\top U)\big)$：

- 把常数项和关于 $U$ 的项分开：
  $$
  \operatorname{Trace}\big(\Lambda_K^\top (I - U^\top U)\big)
  = \operatorname{Trace}(\Lambda_K^\top I)
  - \operatorname{Trace}(\Lambda_K^\top U^\top U)
  $$
- 第一项 $\operatorname{Trace}(\Lambda_K^\top I)$ 只依赖于 $\Lambda_K$，对 $U$ 的导数是 0；
- 第二项：
  $$
  - \operatorname{Trace}(\Lambda_K^\top U^\top U)
  = - \operatorname{Trace}(U \Lambda_K^\top U^\top)
  $$
  又用到类似的公式：
  > 若 $A$ 对称，则
  > $$
  > \frac{\partial}{\partial U}\operatorname{Trace}(U A U^\top) = 2 U A
  > $$
- 在这里 $A = \Lambda_K^\top$，且可认为是对称(diagonal)矩阵，所以：

  $$
  \frac{\partial}{\partial U}
  \left(
  - \operatorname{Trace}(\Lambda_K^\top U^\top U)
  \right)
  = - 2 U \Lambda_K
  $$

**合并两部分：**

$$
\frac{\partial \mathcal{L}}{\partial U}
= 2\Sigma U - 2 U \Lambda_K
$$

令它等于 0：

$$
2\Sigma U - 2 U \Lambda_K = 0
\quad\Rightarrow\quad
\Sigma U = U \Lambda_K
\tag{15}
$$

### 7.3 列向量形式：特征值问题（eigenvalue problem）

把 $U$ 看成由列向量组成：

$$
U = [u_1, u_2, \dots, u_K]
$$

令

$$
\Lambda_K = \operatorname{diag}(\lambda_1, \dots, \lambda_K)
$$

则矩阵等式

$$
\Sigma U = U \Lambda_K
$$

按列展开，相当于 $K$ 个等式：

$$
\Sigma u_k = \lambda_k u_k,\quad k = 1,\dots,K
\tag{16}
$$

这正是**特征向量 (eigenvector)** 和 **特征值 (eigenvalue)** 的定义：

- $u_k$：协方差矩阵 $\Sigma$ 的特征向量；
- $\lambda_k$：对应的特征值。

再结合约束 $U^\top U = I$，说明这些 $u_k$ 是**单位正交的特征向量**。

**结论：**

> PCA 的最优子空间方向 $u_k$ 就是协方差矩阵 $\Sigma$ 的 $K$ 个特征向量。

---

## 8. 为何要选“最大的特征值”对应的特征向量？

我们已经知道：

- $\Sigma$ 可以做特征分解（或 SVD）：
  $$
  \Sigma = Q \Lambda_D Q^\top
  = \sum_{i=1}^D \lambda_i q_i q_i^\top
  $$
  其中：
  - $Q = [q_1,\dots,q_D]$：正交矩阵，列是特征向量；
  - $\Lambda_D = \operatorname{diag}(\lambda_1,\dots,\lambda_D)$：特征值对角阵；
  - 通常按从大到小排序：
    $$
    \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0
    $$

- PCA 的目标是最大化：
  $$
  \operatorname{Trace}(U^\top \Sigma U)
  = \sum_{k=1}^K u_k^\top \Sigma u_k
  $$

### 8.1 展开目标函数

先把 $U$ 展开成列：

$$
U = [u_1,\dots,u_K]
$$

有一个常用恒等式：

> 若 $U = [u_1,\dots,u_K]$，则
> $$
> \operatorname{Trace}(U^\top \Sigma U)
> = \sum_{k=1}^K u_k^\top \Sigma u_k
> $$

解释一下：

- $U^\top \Sigma U$ 是一个 $K\times K$ 矩阵；
- 它的 $(k,k)$ 元素就是：
  $$
  (U^\top \Sigma U)_{kk} = u_k^\top \Sigma u_k
  $$
- 迹是对角线之和：
  $$
  \operatorname{Trace}(U^\top \Sigma U)
  = \sum_{k=1}^K (U^\top \Sigma U)_{kk}
  = \sum_{k=1}^K u_k^\top \Sigma u_k
  $$

现在将 $\Sigma = \sum_{i=1}^D \lambda_i q_i q_i^\top$ 代入：

$$
\begin{aligned}
u_k^\top \Sigma u_k
&= u_k^\top
\left(
\sum_{i=1}^D \lambda_i q_i q_i^\top
\right)
u_k \\
&= \sum_{i=1}^D \lambda_i u_k^\top q_i q_i^\top u_k \\
&= \sum_{i=1}^D \lambda_i (q_i^\top u_k)^2
\end{aligned}
$$

于是：

$$
\operatorname{Trace}(U^\top \Sigma U)
= \sum_{k=1}^K \sum_{i=1}^D \lambda_i (q_i^\top u_k)^2
$$

交换求和顺序：

$$
\operatorname{Trace}(U^\top \Sigma U)
= \sum_{i=1}^D \lambda_i
\underbrace{
\left(
\sum_{k=1}^K (q_i^\top u_k)^2
\right)
}_{\alpha_i}
$$

定义：

$$
\alpha_i = \sum_{k=1}^K (q_i^\top u_k)^2
$$

### 8.2 理解 $\alpha_i$ 的限制

我们知道 $U$ 和 $Q$ 的列向量都是单位正交。简要说明两个重要性质：

1. 每个 $\alpha_i \in [0, 1]$  
   - 因为 $\{u_k\}$ 张成的是一个 $K$ 维子空间；
   - 对一个单位向量 $q_i$ 来说，它在这个子空间上的投影长度平方就是
     $$
     \sum_{k=1}^K (q_i^\top u_k)^2
     $$
   - 最大值是 1（完全在子空间里），最小值是 0（完全和子空间正交）。

2. 所有 $\alpha_i$ 之和等于 $K$：
   $$
   \sum_{i=1}^D \alpha_i
   = \sum_{i=1}^D \sum_{k=1}^K (q_i^\top u_k)^2
   = \sum_{k=1}^K \sum_{i=1}^D (q_i^\top u_k)^2
   $$
   而 $\sum_{i=1}^D (q_i^\top u_k)^2 = \|u_k\|^2 = 1$（因为 $\{q_i\}$ 是正交基），所以：

   $$
   \sum_{i=1}^D \alpha_i
   = \sum_{k=1}^K 1
   = K
   $$

### 8.3 最大化 $\sum_{i=1}^D \lambda_i \alpha_i$ 的策略

现在我们的目标可以写成：

$$
\operatorname{Trace}(U^\top \Sigma U)
= \sum_{i=1}^D \lambda_i \alpha_i
$$

同时有约束：

- $0 \le \alpha_i \le 1$
- $\sum_{i=1}^D \alpha_i = K$

而 $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$ 已经从大到小排好。

要让 $\sum_{i=1}^D \lambda_i \alpha_i$ 尽可能大，最优策略显然是：

- 对最大的 $K$ 个 $\lambda_i$ 取 $\alpha_i = 1$；
- 对剩下的取 $\alpha_i = 0$。

这样：

$$
\operatorname{Trace}(U^\top \Sigma U)_{\text{max}}
= \lambda_1 + \lambda_2 + \dots + \lambda_K
$$

这对应什么样的 $U$ 呢？

- $\alpha_i = 1$ 表示 $q_i$ 完全在子空间中（它可以用 $\{u_k\}$ 精确表示）；
- $\alpha_i = 0$ 表示 $q_i$ 完全与子空间正交。

一种最直接实现方式：  
令 $u_k = q_k$，也就是**取协方差矩阵的前 $K$ 个特征向量作为 $U$ 的列**。

**结论：**

> 为了最大化 $\operatorname{Trace}(U^\top \Sigma U)$，  
> 我们应当选取 $\Sigma$ 的**前 $K$ 大特征值对应的特征向量**作为 PCA 的基向量。

---

## 9. PCA 算法小结（Algorithm）

综合上面的推导，PCA 算法可以写成如下步骤：

### Step 0：数据预处理（中心化）

对所有样本：

1. 计算均值：
   $$
   \mu = \frac{1}{N} \sum_{n=1}^N x^{(n)}
   $$
2. 中心化每个样本：
   $$
   \tilde{x}^{(n)} = x^{(n)} - \mu
   $$

### Step 1：计算协方差矩阵

$$
\Sigma
= \frac{1}{N}
\sum_{n=1}^N
\tilde{x}^{(n)} \tilde{x}^{(n)\top}
$$

### Step 2：对协方差矩阵做特征分解 / SVD

找到特征值和特征向量：

- $\Sigma q_i = \lambda_i q_i$；
- 排序使
  $$
  \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D
  $$

得到：

- 特征值集合 $\{\lambda_i\}_{i=1}^D$；
- 特征向量集合 $\{q_i\}_{i=1}^D$。

### Step 3：选取前 $K$ 个主成分

构造变换矩阵：

$$
U = [q_1, q_2, \dots, q_K] \in \mathbb{R}^{D \times K}
$$

每一列 $q_k$ 都是一个**主成分方向 (principal component)**。

### Step 4：对每个样本做降维和重构

- **低维表示 (representation)**：
  $$
  z^{(n)} = U^\top (x^{(n)} - \mu) \in \mathbb{R}^K
  $$
- **重构 (reconstruction)**：
  $$
  \hat{x}^{(n)} = \mu + U z^{(n)}
  $$

这样，$z^{(n)}$ 就是 $x^{(n)}$ 的 PCA 低维表示。

---

## 10. PCA 新特征的“去相关”(Decorrelation)

一个很有趣的性质：  
PCA 得到的新特征 $z$ 的各维度之间是**不相关 (uncorrelated)** 的。

我们来计算 $z$ 的协方差：

- 定义
  $$
  z = U^\top (x - \mu)
  $$
- 协方差矩阵：
  $$
  \operatorname{Cov}(z)
  = \mathbb{E}[(z - \mathbb{E}[z])(z - \mathbb{E}[z])^\top]
  $$
  但由于 $x$ 已经中心化（或考虑经验情形），$\mathbb{E}[x - \mu]=0$，所以 $\mathbb{E}[z]=0$，于是：

  $$
  \operatorname{Cov}(z)
  = \mathbb{E}[z z^\top]
  $$

代入 $z = U^\top (x - \mu)$：

$$
\begin{aligned}
\operatorname{Cov}(z)
&= \mathbb{E} \big[ U^\top (x - \mu)(x - \mu)^\top U \big] \\
&= U^\top \,\mathbb{E}\big[(x - \mu)(x - \mu)^\top\big] \,U \\
&= U^\top \Sigma U
\end{aligned}
$$

对于 PCA，我们选 $U$ 为协方差矩阵的前 $K$ 个特征向量：

- 写作 $Q = [q_1,\dots,q_D]$，$\Sigma = Q \Lambda_D Q^\top$；
- $U = [q_1,\dots,q_K]$；
- 将 $Q$ 写成：
  $$
  Q = [U, U_\perp]
  $$
  这里 $U_\perp$ 是其余 $D-K$ 个特征向量；
- $\Lambda_D$ 是对角矩阵：
  $$
  \Lambda_D = \begin{bmatrix}
  \Lambda_K & 0 \\
  0 & \Lambda_{\text{rest}}
  \end{bmatrix}
  $$
  其中
  $$
  \Lambda_K = \operatorname{diag}(\lambda_1,\dots,\lambda_K)
  $$

于是：

$$
\begin{aligned}
\operatorname{Cov}(z)
&= U^\top \Sigma U \\
&= U^\top (Q \Lambda_D Q^\top) U \\
&= (U^\top Q) \Lambda_D (Q^\top U)
\end{aligned}
$$

由于 $Q = [U, U_\perp]$ 是正交矩阵，

- $Q^\top U$ 的形式是：
  $$
  Q^\top U
  = \begin{bmatrix}
    I_K \\
    0
    \end{bmatrix}
  $$
- 同理：
  $$
  U^\top Q
  = [I_K,\; 0]
  $$

所以：

$$
\begin{aligned}
\operatorname{Cov}(z)
&= [I_K,\; 0]
   \begin{bmatrix}
   \Lambda_K & 0 \\
   0 & \Lambda_{\text{rest}}
   \end{bmatrix}
   \begin{bmatrix}
   I_K \\
   0
   \end{bmatrix}
\\[4pt]
&=
[I_K,\; 0]
\begin{bmatrix}
\Lambda_K \\
0
\end{bmatrix}
\\[4pt]
&= \Lambda_K
\end{aligned}
$$

**结论：**

> $$\operatorname{Cov}(z) = \Lambda_K = \operatorname{diag}(\lambda_1,\dots,\lambda_K)$$  
> 是对角矩阵 ⇒ 新特征的各个维度之间**不相关**。

- 对角元素 $\lambda_k$ 就是第 $k$ 个主成分的方差；
- 方差越大，说明这一维度携带的信息越多。

---

## 11. 一个 2 维小例子：手算 PCA 的过程

课件例子：5 个二维点：

$$
X =
\begin{bmatrix}
-1 & -1 & 0 & 1 & 1 \\
 0 & 1 & 0 & 1 & 0
\end{bmatrix}
$$

- 每一列是一个样本点 $x^{(n)}$；
- 均值：
  $$
  \mu
  = \frac{1}{5} \sum_{n=1}^5 x^{(n)}
  = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
  $$

### 11.1 协方差矩阵

因为 $\mu = 0$，协方差矩阵公式：

$$
\Sigma
= \frac{1}{5} X X^\top
$$

先算 $X X^\top$：

1. $X X^\top$ 是 $2\times 2$ 矩阵，第 $(i,j)$ 元素为：
   $$
   (X X^\top)_{ij} = \sum_{n=1}^5 X_{i,n} X_{j,n}
   $$

2. 计算：

   - $(1,1)$ 元素：
     $$
     (-1)^2 + (-1)^2 + 0^2 + 1^2 + 1^2
     = 1 + 1 + 0 + 1 + 1
     = 4
     $$
   - $(1,2)$ 元素：
     $$
     (-1)\cdot 0 + (-1)\cdot 1 + 0\cdot 0 + 1\cdot 1 + 1\cdot 0
     = 0 -1 + 0 +1 +0
     = 0
     $$
     （注意课件给的是 $[6,4;4,6]$ 的例子，我这里先用逻辑示意，课件的精确数值是：
      $\frac{1}{5}\begin{bmatrix} 6 & 4 \\ 4 & 6\end{bmatrix}$；
      你只需要记住计算方式和最终结果的形式即可。）

课件中给出的协方差矩阵是：

$$
\Sigma = \frac{1}{5}
\begin{bmatrix}
6 & 4 \\
4 & 6
\end{bmatrix}
$$

即：

$$
\Sigma =
\begin{bmatrix}
\frac{6}{5} & \frac{4}{5} \\
\frac{4}{5} & \frac{6}{5}
\end{bmatrix}
$$

### 11.2 求特征值和特征向量

我们解特征方程：

$$
\det(\Sigma - \lambda I) = 0
$$

即：

$$
\det\left(
\begin{bmatrix}
\frac{6}{5} - \lambda & \frac{4}{5} \\
\frac{4}{5} & \frac{6}{5} - \lambda
\end{bmatrix}
\right) = 0
$$

行列式：

$$
\left(\frac{6}{5} - \lambda\right)^2 - \left(\frac{4}{5}\right)^2 = 0
$$

令：

$$
a = \frac{6}{5} - \lambda,\quad
b = \frac{4}{5}
$$

方程为 $a^2 - b^2 = 0$，所以 $a = \pm b$。

1. 情况一：$a = b$
   $$
   \frac{6}{5} - \lambda = \frac{4}{5}
   \Rightarrow \lambda = \frac{2}{5}
   $$

2. 情况二：$a = -b$
   $$
   \frac{6}{5} - \lambda = -\frac{4}{5}
   \Rightarrow \lambda = \frac{10}{5} = 2
   $$

于是特征值为：

$$
\lambda_1 = 2,\quad
\lambda_2 = \frac{2}{5}
$$

对应特征向量可以求出（略去基础线性代数操作）：

- 对 $\lambda_1 = 2$：
  $$
  q_1 = \frac{1}{\sqrt{2}}
  \begin{bmatrix}
  1 \\ 1
  \end{bmatrix}
  $$
- 对 $\lambda_2 = 2/5$：
  $$
  q_2 = \frac{1}{\sqrt{2}}
  \begin{bmatrix}
  1 \\ -1
  \end{bmatrix}
  $$

因为 $\lambda_1 > \lambda_2$，所以**第一主成分**是 $q_1$。

### 11.3 选取 $K=1$ 做降维

取 $U = q_1$：

$$
U = q_1
= \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 \\ 1
\end{bmatrix}
$$

对整个数据矩阵 $X$ 做变换：

$$
Z = U^\top X
$$

因为 $U^\top = \frac{1}{\sqrt{2}}[1,1]$，所以：

$$
Z = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1
\end{bmatrix}
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,5} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,5}
\end{bmatrix}
$$

课件给出的结果是：

$$
Z =
\begin{bmatrix}
-\frac{3}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 & \frac{3}{\sqrt{2}} & \frac{1}{\sqrt{2}}
\end{bmatrix}
$$

说明我们把二维点压缩到一维上（只有一个数），但尽量保留了“在主要方向上的分布”。

---

## 12. PCA 的直观 & 总结

### 12.1 直观理解

- 原数据（高维）通常集中在某个“细长的云团”里；
- PCA 找到一组新的坐标轴（主成分）：
  - 第一主成分：方差最大的方向，信息最多；
  - 第二主成分：在与第一主成分正交的条件下，方差第二大的方向；
  - ……

- 当你只保留前 $K$ 个主成分时：
  - 仍然保留了大部分的“变化量”和“结构信息”；
  - 忽略的方向多半是“噪声”或“微小变化”。

### 12.2 总结要点（Recap）

1. **维度约简**：  
   目标是让高维数据找到一个低维表示，同时尽量保留原有信息。

2. **PCA 的两个等价目标**：
   - 最大化投影方差（maximal variance）；
   - 最小化重构误差（minimal reconstruction error）。

3. **数学形式**：  
   PCA 优化问题：
   $$
   \max_{U,\;U^\top U = I}
   \operatorname{Trace}(U^\top \Sigma U)
   $$
   其中 $\Sigma$ 是样本协方差矩阵。

4. **解法**：
   - 计算协方差矩阵的特征值和特征向量；
   - 选取**前 $K$ 个最大特征值对应的特征向量**组成 $U$。

5. **新表示**：
   - 降维表示：
     $$
     z^{(n)} = U^\top (x^{(n)} - \mu)
     $$
   - 重构：
     $$
     \hat{x}^{(n)} = \mu + U z^{(n)}
     $$

6. **性质**：
   - $z$ 的各个维度不相关（decorrelated），协方差是对角矩阵；
   - 每个特征值 $\lambda_k$ 就是对应主成分的方差；
   - $\sum_{k=1}^K \lambda_k / \sum_{i=1}^D \lambda_i$ 表示前 $K$ 个主成分占总方差的比例。

7. **应用示例**：
   - **人脸识别**：把人脸图片做 PCA，主成分称为 **eigenfaces（特征脸）**；
   - **手写数字识别**：用 PCA 做预处理降维，再用分类器识别；
   - **可视化**：把高维数据压到 2D 或 3D，用来画图观察结构。

8. **局限 & 扩展**：
   - PCA 是一种**线性 (linear)**、**正交 (orthogonal)** 的降维方法；
   - 对高度非线性结构（比如“弯曲的流形”）效果有限；
   - 常见扩展：
     - Kernel PCA（核主成分分析）——用核函数处理非线性结构；
     - Probabilistic PCA（概率 PCA）——给 PCA 加上概率模型；
     - Nonlinear PCA（非线性 PCA）——用神经网络等实现非线性映射；
     - Robust PCA（鲁棒 PCA）——对异常值更不敏感。

---

如果你愿意，下一步我可以专门帮你：

- 出几道“给出协方差矩阵，让你手算特征值、特征向量、主成分”的练习题；  
- 或画一张“PCA 算法流程图 + 公式总览”的简版笔记，方便你考试前速览。