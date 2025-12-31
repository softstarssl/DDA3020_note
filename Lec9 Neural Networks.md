# DDA3020 Machine Learning - Lecture 09: Neural Networks

**讲师**: Juexiao Zhou (CUHK-SZ)
**日期**: Oct 21, 2025

---

## 1. 课程回顾：之前的分类模型 (Recall of Previous Models)

在进入神经网络之前，回顾两个核心的线性分类模型。这些模型通常假设特征向量 $\mathbf{x}$ 是给定的。

### 1.1 逻辑回归 (Logistic Regression)
*   **假设函数 (Hypothesis Function)**:
    $$h_w(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}}$$
    *   $\sigma(\cdot)$: Sigmoid 激活函数，将输出压缩到 $(0,1)$ 区间，表示概率。
*   **代价函数 (Cost Function)**:
    $$Cost = -\log \sigma(y \cdot \mathbf{w}^\top \mathbf{x})$$
    *   这是负对数似然（Negative Log-Likelihood）。
*   **学习算法 (Learning Algorithm)**: 梯度下降
    $$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_\mathbf{w} J(\mathbf{w})$$

### 1.2 支持向量机 (SVM)
*   **假设函数**:
    $$h_w(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$$
*   **代价函数 (Hinge Loss)**:
    $$Cost = \max(0, 1 - y \cdot (\mathbf{w}^\top \mathbf{x} + b))$$
*   **学习算法**: 拉格朗日对偶性 (Lagrange duality) 和 KKT 条件。

### 1.3 为什么要引入神经网络？
*   **特征提取难题**: 传统的机器学习（如 LR, SVM）假设输入 $\mathbf{x}$ 是处理好的向量。但在图像（Image）或文本（Text）任务中，原始数据（如像素）很难直接线性分类。
*   **手工特征 vs 学习特征**: 过去依赖手工特征（SIFT, HOG 等）。神经网络将**特征学习 (Feature Learning)** 和**分类器学习 (Classifier Learning)** 结合在一起，实现端到端的学习。

---

## 2. 感知机模型 (Perceptron Model)

### 2.1 神经元 (Neuron) 与 M-P 模型
*   **生物灵感**: 大脑有约 $10^{11}$ 个神经元，每个连接约 $10^4$ 个其他神经元。
*   **M-P 神经元模型 (1943)**: 模拟生物神经元的电位累积和阈值激发机制。

### 2.2 感知机定义
感知机由输入层和输出层（阈值逻辑单元）组成。
*   **公式**:
    $$y = f(\mathbf{w}^\top \mathbf{x} + b) = \text{Sgn}(\mathbf{w}^\top \mathbf{x} + b)$$
    *   $\text{Sgn}(\cdot)$: 符号函数，大于0输出+1，否则输出-1。
*   **目标函数 (Objective Function)** (基于均方误差):
    $$J(\mathbf{w}) = \frac{1}{2}(y - t)^2 = \frac{1}{2}(\text{Sgn}(\mathbf{w}^\top \mathbf{x} + b) - t)^2$$
    *   $t$: 真实标签 (Ground-truth label)。
*   **学习规则 (Gradient Descent)**:
    $$\mathbf{w} \leftarrow \mathbf{w} - \eta(y - t)\mathbf{x}$$
    *   *注意*: 这里假设 $\text{Sgn}$ 的梯度近似为 1（实际上 $\text{Sgn}$ 不可导，这是感知机的启发式更新规则）。

### 2.3 激活函数 (Activation Functions)
神经元的核心是非线性激活函数。常见类型：
1.  **Linear**: $y=z$
2.  **ReLU (Rectified Linear Unit)**: $y = \max(0, z)$ (最常用)
3.  **Soft ReLU**: $y = \log(1+e^z)$
4.  **Hard Threshold**: $z>0 \to 1, z \le 0 \to 0$
5.  **Logistic (Sigmoid)**: $y = \frac{1}{1+e^{-z}}$
6.  **Tanh**: $y = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

### 2.4 逻辑门与 XOR 问题
*   **能力**: 感知机可以模拟简单的布尔逻辑门：AND, OR, NOT。
*   **局限性**: 感知机**无法解决 XOR (异或) 问题**。
    *   原因: XOR 问题在二维空间中是**非线性可分 (Non-linearly separable)** 的。单层感知机只能画一条直线（线性边界）。

---

## 3. 多层前馈神经网络 (Multi-layer Feedforward NN)

### 3.1 定义与结构
*   **结构**: 输入层 $\to$ 隐藏层(s) $\to$ 输出层。
*   **连接方式**:
    *   单向传播 (Feedforward)。
    *   层与层之间全连接 (Fully connected)。
    *   同层内无连接，跨层无跳跃连接。

### 3.2 数学表达 (复合函数)
网络计算本质上是函数的复合：
$$
\begin{aligned}
\mathbf{h}^{(1)} &= f^{(1)}(\mathbf{x}) \\
\mathbf{h}^{(2)} &= f^{(2)}(\mathbf{h}^{(1)}) \\
\dots \\
y &= f^{(L)}(\mathbf{h}^{(L-1)})
\end{aligned}
$$
简写为: $y = f^{(L)} \circ \dots \circ f^{(1)}(\mathbf{x})$

### 3.3 解决 XOR 问题 (特征空间变换)
多层网络通过隐藏层将原始空间中的**非线性可分**数据，变换到隐藏层空间，使其变得**线性可分**。
*   **示例**:
    *   输入 $\mathbf{x} \in \mathbb{R}^2$。
    *   隐藏层 $\mathbf{h} = g_2(\mathbf{W}\mathbf{x} + \mathbf{c})$。
    *   输出 $y = g_1(\mathbf{w}^\top \mathbf{h} + b)$。
    *   通过适当的权重 $\mathbf{W}$ 和 $\mathbf{c}$，原始的 XOR 分布被扭曲/折叠，使得在 $\mathbf{h}$ 空间可以用一条直线分开。

### 3.4 可视化理解
*   第一层隐藏单元 $\sigma(\mathbf{w}_j^\top \mathbf{x})$ 充当**特征检测器 (Feature Detector)**。
*   例如在手写数字识别 (MNIST) 中，隐藏层神经元可能在检测特定的笔画（横、竖、圆弧）。

---

## 4. 反向传播 (Backpropagation) - 核心推导

这是训练神经网络的核心算法，利用**链式法则 (Chain Rule)** 计算梯度。

### 4.1 链式法则基础
如果 $L$ 是 $y$ 的函数，$y$ 是 $x$ 的函数，即 $L(y(x))$，则：
$$\frac{d L}{d x} = \frac{d L}{d y} \cdot \frac{d y}{d x}$$

### 4.2 详细推导示例 (单神经元)
假设损失函数为平方误差，且包含一个 Sigmoid 激活函数：
*   **模型**: $z = wx + b$, $y = \sigma(z)$
*   **损失**: $L = \frac{1}{2}(y - t)^2$
*   **目标**: 求 $\frac{\partial L}{\partial w}$ 和 $\frac{\partial L}{\partial b}$。

**推导步骤**:
1.  **分解计算图**:
    $$w, x, b \xrightarrow{z=wx+b} z \xrightarrow{y=\sigma(z)} y \xrightarrow{L=\frac{1}{2}(y-t)^2} L$$

2.  **从后向前计算偏导数**:
    *   对输出 $y$ 的导数:
        $$\frac{\partial L}{\partial y} = \frac{\partial}{\partial y}(\frac{1}{2}(y-t)^2) = (y - t)$$
    *   对线性输出 $z$ 的导数 (利用链式法则):
        $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z}$$
        已知 Sigmoid 导数 $\sigma'(z) = \sigma(z)(1-\sigma(z)) = y(1-y)$ (此处讲义简化记为 $\sigma'(z)$)。
        $$\frac{\partial L}{\partial z} = (y - t) \cdot \sigma'(z)$$
    *   对权重 $w$ 的导数:
        $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}$$
        因为 $z = wx + b$，所以 $\frac{\partial z}{\partial w} = x$。
        $$\frac{\partial L}{\partial w} = (y - t)\sigma'(z) \cdot x$$
    *   对偏置 $b$ 的导数:
        $$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b}$$
        因为 $z = wx + b$，所以 $\frac{\partial z}{\partial b} = 1$。
        $$\frac{\partial L}{\partial b} = (y - t)\sigma'(z) \cdot 1$$

### 4.3 多层网络的反向传播
对于多层网络，我们将误差项 $\delta$ (即 $\frac{\partial L}{\partial z}$) 从输出层向输入层传播。

**符号定义**:
*   $L$: 损失函数
*   $y_i$: 节点 $i$ 的输出
*   $z_j$: 节点 $j$ 的加权输入
*   $w_{ij}$: 从 $i$ 到 $j$ 的权重

**传播公式**:
1.  **计算当前层的梯度**:
    $$\frac{\partial L}{\partial z_j} = \frac{\partial L}{\partial y_j} \cdot \frac{d y_j}{d z_j} = \frac{\partial L}{\partial y_j} \cdot \sigma'(z_j)$$
    *(对于 Sigmoid，$\sigma'(z_j) = y_j(1-y_j)$)*

2.  **将梯度传回上一层 (关键)**:
    $$\frac{\partial L}{\partial y_i} = \sum_{j \in \text{Children}(i)} \frac{\partial L}{\partial z_j} \cdot \frac{\partial z_j}{\partial y_i}$$
    因为 $z_j = \sum_k w_{kj} y_k + b$，所以 $\frac{\partial z_j}{\partial y_i} = w_{ij}$。
    $$\frac{\partial L}{\partial y_i} = \sum_{j} w_{ij} \frac{\partial L}{\partial z_j}$$

3.  **计算权重的梯度**:
    $$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}} = \frac{\partial L}{\partial z_j} \cdot y_i$$

### 4.4 计算图 (Computational Graph)
*   **定义**: 节点表示变量，边表示函数操作。
*   **前向传播 (Forward Pass)**: 按照拓扑顺序计算每个节点的值 $v_i = f(\text{Parents}(v_i))$。
*   **反向传播 (Backward Pass)**: 按照反向拓扑顺序计算梯度。
    $$v_N = 1 \quad (\text{Loss 对自身的导数})$$
    $$\overline{v_i} = \sum_{j \in \text{Children}(v_i)} \overline{v_j} \cdot \frac{\partial v_j}{\partial v_i}$$
    *(其中 $\overline{v}$ 表示 Loss 对 $v$ 的导数)*

---

## 5. 计算复杂度 (Computational Cost)

假设网络层维度为 $m \times d$ (输入 $d$, 输出 $m$)。

### 5.1 前向传播
*   计算: $\mathbf{y} = g(\mathbf{W}\mathbf{x} + \mathbf{b})$
*   复杂度: $O(md)$ (矩阵向量乘法)。

### 5.2 反向传播
*   计算: 需要计算 $\frac{\partial L}{\partial \mathbf{W}}$ 和传递给下一层的误差。
*   复杂度: $O(md)$。
*   **结论**: 反向传播的计算成本大约是前向传播的 **2倍**。
    *   总复杂度与连接数（权重数）成正比。

---

## 6. 深度神经网络 (Deep Neural Networks)

### 6.1 深度的意义
*   **深度线性网络无效**: 多层线性网络等价于单层线性网络 ($W^{(3)}W^{(2)}W^{(1)}x = W'x$)。必须引入**非线性激活函数**。
*   **通用近似定理 (Universal Approximation Theorem)**: 包含一个非线性隐藏层的前馈神经网络可以以任意精度逼近任何连续函数。
    *   *既然一层就够了，为什么要做深 (Deep)？*

### 6.2 为什么要 Deep?
1.  **参数效率**: 浅层网络要达到同样的表现，可能需要指数级增加的宽度（神经元数量）。
2.  **泛化能力**: 深层网络通常比极宽的浅层网络泛化能力更好。
3.  **特征层级**: 深度结构允许学习从低级特征（边缘）到高级特征（形状、物体）的层级表示。

---

## 7. 卷积神经网络 (CNN) 的引入动机

虽然全连接 DNN 很强，但在处理高维图像时面临问题：
*   **参数爆炸**: 对于大图像，全连接层参数过多（例如 $1000 \times 1000$ 图像，隐藏层 100 个神经元，参数量为 $10^8$）。

### 7.1 解决方案 (CNN 的两个核心 Trick)
1.  **稀疏连接 (Sparse Connection)**:
    *   每个输出神经元只连接输入的一小部分区域。
    *   **感受野 (Receptive Field)**: 输出神经元能“看到”的输入区域。随着层数加深，感受野会变大。
2.  **参数共享 (Shared Parameters)**:
    *   在图像的不同位置使用相同的权重（卷积核/滤波器）。
    *   假设特征（如垂直边缘）在图像的任何位置都可能出现且特征相同。

*   **对比**: 卷积核参数可能只有 9 个 ($3\times3$)，而全连接层可能有 25 个 ($5\times5$ 输入全连接)。

---

## 8. 总结 (Summary)

1.  **感知机**: 只能解决线性可分问题，无法解决 XOR。
2.  **多层前馈网络 (MLP)**: 通过非线性激活函数和层级结构，具备通用函数拟合能力。
3.  **反向传播 (BP)**: 基于链式法则的高效梯度计算方法，利用计算图实现自动化求导。
4.  **深度学习**: 比浅层网络更高效地表示复杂函数，具有更好的泛化性。
5.  **CNN 预告**: 通过稀疏连接和参数共享解决图像处理中的参数冗余问题。