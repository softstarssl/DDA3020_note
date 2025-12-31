# DDA3020 Machine Learning - Lecture 10: Convolutional Neural Networks (CNN)

**讲师**: Juexiao Zhou (CUHK-SZ)
**日期**: Oct 23, 2025

---

## 1. CNN 简史 (History of CNN)

卷积神经网络的发展经历了几个关键节点，从早期的文档识别到现代的深度视觉任务。

*   **1998年 (LeNet)**: LeCun, Bottou, Bengio, Haffner 提出了基于梯度的学习方法应用于文档识别。这是 CNN 的早期雏形，成功用于手写数字识别。
*   **2006年**: Hinton 和 Salakhutdinov 发表论文，重新激活了深度学习 (Deep Learning) 的研究。
*   **2012年 (AlexNet)**: Krizhevsky, Sutskever, Hinton 在 ImageNet 分类比赛中取得突破性成果。这标志着深度卷积神经网络（Deep CNN）时代的正式爆发。
*   **现状**: CNN 现已无处不在，广泛应用于：
    *   图像分类 (Classification)
    *   图像检索 (Retrieval)
    *   目标检测 (Detection)
    *   图像分割 (Segmentation)
    *   人脸识别、视频分析等。

---

## 2. 卷积神经网络架构概览 (CNN Architecture)

### 2.1 对比传统神经网络
*   **线性模型**: $f(x) = Wx$
*   **两层神经网络**: $f(x) = W_2 \max(0, W_1 x)$
*   **CNN**: 是一系列层的堆叠，主要包括卷积层 (Convolutional Layer)、激活函数 (Activation, 如 ReLU)、池化层 (Pooling Layer) 和全连接层 (Fully Connected Layer)。

### 2.2 典型结构
一个典型的 ConvNet 结构如下：
`Input -> [[CONV -> ReLU] * N -> POOL?] * M -> [FC -> ReLU] * K -> FC`
*   **CONV**: 卷积层，提取特征。
*   **ReLU**: 激活层，引入非线性。
*   **POOL**: 池化层，降维。
*   **FC**: 全连接层，用于最终的分类输出。

---

## 3. 卷积层 (Convolutional Layer) - 核心组件

卷积层是 CNN 的核心，负责在保留空间结构的同时提取特征。

### 3.1 核心概念
*   **输入 (Input Volume)**: 假设输入图像尺寸为 $32 \times 32 \times 3$ (宽 $\times$ 高 $\times$ RGB通道)。
*   **滤波器 (Filter/Kernel)**: 卷积核。
    *   **深度一致性**: 滤波器的深度（Depth/Channels）必须与输入数据的深度一致。例如输入是 $3$ 通道，滤波器也必须是 $5 \times 5 \times \mathbf{3}$。
*   **卷积操作 (Convolution Operation)**:
    *   滤波器在输入图像的空间维度（宽和高）上滑动 (Slide)。
    *   在每个位置，计算滤波器与输入图像对应局部区域的**点积 (Dot Product)**，并加上偏置 $b$。
    *   **数学表达**: $w^T x + b$。
        *   其中 $w$ 是滤波器参数，$x$ 是局部感受野内的像素值。
        *   结果是一个标量 (Scalar)。

### 3.2 特征图 (Activation Map)
*   一个滤波器在整个图像上滑动并计算后，生成一个二维的**特征图 (Activation Map)**。
*   **多滤波器**: 如果我们有 $K$ 个不同的滤波器（例如 6 个），我们会得到 $K$ 个特征图。
*   **堆叠**: 将这 $K$ 个特征图在深度方向堆叠，形成该层的输出体积。
    *   例如：输入 $32 \times 32 \times 3$，使用 6 个 $5 \times 5 \times 3$ 的滤波器，输出将是 $28 \times 28 \times 6$ (假设无填充，步长为1)。

### 3.3 空间尺寸计算 (Spatial Dimensions) - **重点推导**

输出特征图的尺寸取决于四个超参数：
1.  **输入尺寸 (Input Size)**: $N$ (假设宽和高相等，即 $N \times N$)
2.  **滤波器尺寸 (Filter Size)**: $F$ ($F \times F$)
3.  **步长 (Stride)**: $S$ (滤波器每次滑动的像素数)
4.  **零填充 (Zero Padding)**: $P$ (在输入边缘填充 0 的圈数)

#### 公式推导
假设我们要计算输出的宽度/高度：
1.  **总长度**: 原始输入长度 $N$ 加上两边的填充 $2P$，总长度为 $N + 2P$。
2.  **滤波器覆盖**: 滤波器占据了 $F$ 的长度。
3.  **可滑动距离**: 滤波器中心可以移动的有效区域长度为 $(N + 2P) - F$。
4.  **步数计算**: 每次移动 $S$，所以滑动的步数为 $\frac{N + 2P - F}{S}$。
5.  **输出点数**: 步数代表移动了多少次，加上初始位置（第1个点），总输出点数为步数 + 1。

#### 核心公式
$$ \text{Output Size} = \frac{N - F + 2P}{S} + 1 $$

*注意*: 计算结果必须是整数。如果不能整除，说明该滤波器配置无法适配输入尺寸（通常会导致报错或需要调整填充）。

#### 示例计算
*   **场景 1**: 输入 $N=7$, 滤波器 $F=3$, 步长 $S=1$, 填充 $P=0$。
    $$ \text{Out} = \frac{7 - 3 + 0}{1} + 1 = 5 \quad (\text{输出 } 5 \times 5) $$
*   **场景 2**: 输入 $N=7$, 滤波器 $F=3$, 步长 $S=2$, 填充 $P=0$。
    $$ \text{Out} = \frac{7 - 3 + 0}{2} + 1 = 3 \quad (\text{输出 } 3 \times 3) $$
*   **场景 3**: 输入 $N=7$, 滤波器 $F=3$, 步长 $S=3$, 填充 $P=0$。
    $$ \text{Out} = \frac{7 - 3 + 0}{3} + 1 = 2.33 \quad (\text{无效配置，无法整除}) $$

### 3.4 零填充的意义 (Padding)
*   **保持尺寸**: 如果不加填充，每次卷积后图像都会变小。使用填充可以保持输出尺寸与输入一致。
*   **常用设置**: 步长 $S=1$ 时，为了保持尺寸 ($N_{out} = N_{in}$)，需要设置 $P = \frac{F-1}{2}$。
    *   $F=3 \implies P=1$
    *   $F=5 \implies P=2$
    *   $F=7 \implies P=3$

### 3.5 参数数量计算 (Parameter Sharing) - **重点**

卷积层利用**参数共享**机制，大大减少了参数量。

假设：
*   输入体积: $W_1 \times H_1 \times C$
*   滤波器数量: $K$
*   滤波器尺寸: $F \times F$
*   (注意：每个滤波器的深度自动为 $C$)

#### 参数计算公式
1.  **单个滤波器的权重数**: $F \times F \times C$
2.  **单个滤波器的偏置数**: $1$
3.  **单个滤波器的总参数**: $(F \times F \times C) + 1$
4.  **该层总参数 (Total Parameters)**:
    $$ \text{Total Params} = K \times ((F \times F \times C) + 1) $$

#### 示例
*   **输入**: $32 \times 32 \times 3$
*   **配置**: 10 个 $5 \times 5$ 的滤波器 (Stride=1, Pad=2)。
*   **输出尺寸**:
    $$ \frac{32 + 2(2) - 5}{1} + 1 = 32 \implies \text{Output Volume: } 32 \times 32 \times 10 $$
*   **参数数量**:
    *   每个滤波器权重: $5 \times 5 \times 3 = 75$
    *   加偏置: $75 + 1 = 76$
    *   总参数 ($K=10$): $76 \times 10 = 760$

### 3.6 卷积层总结 (Summary)
*   **输入**: $W_1 \times H_1 \times C$
*   **超参数**: $K$ (Filter数量), $F$ (Filter尺寸), $S$ (Stride), $P$ (Padding)
*   **输出**: $W_2 \times H_2 \times K$
    *   $W_2 = (W_1 - F + 2P)/S + 1$
    *   $H_2 = (H_1 - F + 2P)/S + 1$
*   **参数量**: $(F \cdot F \cdot C + 1) \cdot K$

---

## 4. 池化层 (Pooling Layer)

### 4.1 作用
*   **降维**: 减小特征图的空间尺寸 (Spatial dimensions)。
*   **减少计算量**: 减少后续层的参数和计算量。
*   **控制过拟合**: 提高模型的鲁棒性。

### 4.2 机制
*   独立地对每一个深度切片（特征图）进行操作。
*   **无参数**: 池化层通常没有需要学习的参数（Weights），只有超参数（尺寸和步长）。
*   **最大池化 (Max Pooling)**: 最常用的池化方法。在滤波器窗口内取最大值。

### 4.3 尺寸变化
假设输入 $W_1 \times H_1 \times C$，池化核大小 $F$，步长 $S$。
*   **输出**: $W_2 \times H_2 \times C$
    *   $W_2 = (W_1 - F)/S + 1$
    *   $H_2 = (H_1 - F)/S + 1$
*   *注意*: 池化层通常不使用 Zero Padding。

### 4.4 常用设置
*   $F=2, S=2$: 将宽高减半 (最常见)。
*   $F=3, S=2$: 重叠池化 (Overlapping Pooling)。

---

## 5. 全连接层 (Fully Connected Layer)

*   **位置**: 通常位于 CNN 的末端。
*   **连接方式**: 当前层的所有神经元与上一层的所有输出连接（与传统神经网络相同）。
*   **作用**: 将卷积层提取的分布式特征映射到样本标记空间（如分类得分）。
*   **输入**: 卷积/池化层的输出通常是 3D 体积，进入 FC 层前需要被**展平 (Flatten)** 成 1D 向量。

---

## 6. 推荐资源 (Further Reading)
*   **课程**: CS231n: Deep Learning for Computer Vision (Stanford)
*   **框架**: PyTorch, TensorFlow