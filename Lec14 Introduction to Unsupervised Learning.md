# DDA3020 Machine Learning - Lecture 14: Introduction to Unsupervised Learning

**讲师**: Juexiao Zhou 
**日期**: Nov 18/20, 2025

---

## 1. 动机 (Motivation)

为什么要进行无监督学习？主要基于以下两点动机：

### 1.1 人类的能力
人类具有将未标记数据划分为若干组（即聚类）的天生能力，从而发现数据中有用的结构。
*   **应用实例**:
    *   **说话人分离 (Speaker diarization)**: 在一段录音中区分谁在什么时候说话。
    *   **图像分割 (Image segmentation)**: 将图像分割成多个有意义的区域。
    *   **人脸聚类 (Face clustering)**: 根据身份将人脸分组。

### 1.2 现实的困难 (标签稀缺)
在实际应用中，获取足够的标签非常困难：
*   **昂贵 (Expensive)**: 某些任务（如医学影像分析）需要专家进行标注，成本极高。
*   **耗时 (Time-consuming)**: 深度神经网络的监督学习需要大规模标记数据库（例如 ImageNet 包含 1000 个类别的 100 万张标记图像）。

**结论**: 利用未标记数据进行机器学习（即无监督学习）是非常有用且必要的。

---

## 2. 定义 (Definition)

### 2.1 无监督学习定义
*   **数据集**: 一个未标记样本的集合 $D = \{x_i\}_{i=1}^M$。
    *   $x$: 特征向量 (Feature vector)。
*   **目标**: 创建一个模型，将特征向量 $x$ 作为输入，将其转换为另一个向量或一个值，用于解决实际问题。
*   **典型应用**:
    *   异常检测 (Anomaly detection)
    *   数据压缩 (Data compression)
    *   新物种/类别发现 (Discovery of new species)

### 2.2 监督学习 vs 无监督学习

| 特性 | 监督学习 (Supervised Learning) | 无监督学习 (Unsupervised Learning) |
| :--- | :--- | :--- |
| **训练集** | $D = \{(x_i, y_i)\}_{i=1}^N$ (包含标签 $y$) | $D = \{(x_i)\}_{i=1}^N$ (无标签) |
| **目标** | 拟合输入 $x$ 到标签 $y$ 的关系，标签代表期望行为。 | 由于缺乏标签，没有稳固的参考点来判断模型质量。 |
| **评估指标** | 基于标签定义，如准确率 (Accuracy)。 | 基于具体任务定义（如聚类效果、降维后的结构保留度）。 |

---

## 3. 主要方法 (Main Approaches)

本讲介绍了五种主要的无监督学习方法。

### 3.1 聚类 (Clustering)
*   **定义**: 将一组对象分组，使得同一组（称为簇 Cluster）中的对象在某种意义上比其他组的对象更相似。
*   **典型算法**: K-means 聚类（将在后续课程详细介绍）。

### 3.2 降维 (Dimensionality Reduction)
*   **主成分分析 (PCA)**:
    *   **定义**: 一种将可能相关的变量观测值转换为线性不相关变量值（称为主成分）的技术。
    *   **目标**: 寻找一个新的低维空间来表示原始高维空间中的数据点，同时尽可能保留数据的结构（方差）。
    *   **示例**: 找到两个正交坐标（PC1 和 PC2）来表示原始 3D 空间中的数据。

### 3.3 密度估计 (Density Estimation)
*   **背景**: 在机器学习中，我们假设训练集 $D = \{(x_i)\}_{i=1}^N$ 采样自某个分布 $P(X)$。但在实践中，我们无法显式写出该潜在分布。
*   **任务**: 基于观察到的数据估计该分布的概率密度函数 (PDF)。

#### 核心模型：核密度估计 (Kernel Density Estimation, KDE)

**1. 问题设定**:
设 $D = \{(x_i)\}_{i=1}^N$ 为一维数据集，样本来自具有未知概率密度函数 $f$ 的分布。任务是基于 $D$ 建模 $f$ 的形状。

**2. KDE 模型公式**:
$$ \hat{f}_b(x) = \frac{1}{Nb} \sum_{i=1}^{N} k\left( \frac{x - x_i}{b} \right) $$

**3. 公式详细解释与参数说明**:
*   **$\hat{f}_b(x)$**: 在点 $x$ 处的估计概率密度。
*   **$N$**: 样本总数。
*   **$x_i$**: 第 $i$ 个观测数据点。
*   **$k(\cdot)$**: **核函数 (Kernel function)**。它定义了每个数据点对周围密度的贡献形状。
    *   核函数通常是对称的，且积分为 1（即 $\int k(z)dz = 1$）。
*   **$b$**: **核大小 (Kernel size)** 或 **带宽 (Bandwidth)**。
    *   这是一个**超参数 (Hyper-parameter)**。
    *   $b > 0$。
    *   **作用**: 控制估计曲线的平滑程度。$b$ 太小会导致曲线过度拟合（出现很多尖峰），$b$ 太大会导致曲线过度平滑（丢失细节）。通常使用 **K-fold 交叉验证**来调整。
*   **$\frac{1}{Nb}$**: 归一化系数，确保最终的 $\hat{f}_b(x)$ 积分为 1，从而构成一个合法的概率密度函数。

**4. 高斯核 (Gaussian Kernel) 示例**:
$$ k(z) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{z^2}{2} \right) $$
这是标准正态分布的密度函数。

**5. 结合高斯核的 KDE 推导**:
将高斯核代入 KDE 通用公式：
$$ \hat{f}_b(x) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{b} \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1}{2} \left( \frac{x - x_i}{b} \right)^2 \right) $$
$$ \hat{f}_b(x) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\sqrt{2\pi b^2}} \exp\left( -\frac{(x - x_i)^2}{2b^2} \right) $$
*   **直观理解**: 这相当于在每一个数据点 $x_i$ 上放置了一个均值为 $x_i$、标准差为 $b$ 的高斯分布（小土包）。最终的密度估计是所有这些小高斯分布的**平均值**。

### 3.4 自编码器 (Autoencoder)
*   **定义**: 一种人工神经网络，用于学习未标记数据的有效编码（Efficient coding）。
*   **结构**:
    *   **输入**: 图像/数据 $x$。
    *   **中间层**: 激活向量可视为输入的**低维表示 (Low-dimensional representation)**。
    *   **输出**: 重构的图像/数据 $\hat{x}$。
*   **应用**:
    *   图像修复 (Recover old/blurring images)。
    *   风格迁移 (Style transfer)。
    *   **DeepFakes**: 生成虚假视频（如换脸）。

### 3.5 自监督学习 (Self-supervised Learning)
*   **定义**: 一类很有前景的方法，通过学习编码“什么使两个事物相似或不同”来构建表示。
*   **核心机制**: 对比学习 (Contrastive learning)。
*   **优势**:
    1.  **预训练大规模深度神经网络 (Foundation Models)**:
        *   利用海量数据进行预训练，提供良好的特征表示。
        *   模型可泛化到不同的下游任务。
    2.  **鲁棒性**:
        *   对训练集中的噪声标签或恶意标签不敏感。
        *   **安全应用**: 成功应用于防御后门攻击 (Backdoor attacks)，例如 DBD (De-Backdoor) 方法。

---

## 4. 总结与延伸阅读
*   后续课程将重点讲解 **聚类 (Clustering)** 和 **降维 (Dimensionality reduction)**。
*   **参考书籍**: Book1, Chapter 9.
*   **KDE 参考**: [Scikit-learn Density Estimation](https://scikit-learn.org/stable/modules/density.html)
*   **自监督学习参考**: [SimCLR](https://github.com/sthalles/SimCLR), [DBD Paper](https://openreview.net/pdf?id=TySnJ-0RdKI)