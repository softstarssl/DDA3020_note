# DDA3020 Machine Learning - Lecture 13: Performance Evaluation

**讲师**: Juexiao Zhou (School of Data Science, CUHK-SZ)
**日期**: Nov 11/13, 2025

---

## 1. 动机：机器学习算法的性能评估 (Motivation)

### 1.1 机器学习的定义回顾
根据 Tom Mitchell 的定义，机器学习包含三个要素：
*   **经验 (Experience, E)**: 对应训练数据。
*   **任务 (Task, T)**: 对应监督学习中的分类或回归任务。
*   **性能度量 (Performance Measure, P)**: 本讲的核心主题。即如何衡量模型在任务 T 上的表现。

### 1.2 机器学习的工作流 (Workflow)
一个典型的学习算法通常包含三个部分：
1.  **损失函数 (Loss Function)**: 例如均方误差 ($\text{MSE}$) $\frac{1}{n}\sum_{i}^n(f(x_i; w, b) - y_i)^2$。
2.  **目标函数 (Objective Function)**: 基于损失函数的优化准则（例如最小化 MSE）。
3.  **优化过程 (Optimization Routine)**: 利用训练数据寻找最优解的算法（例如梯度下降）。

**评估的重要性**:
*   上述步骤仅是训练过程。
*   我们需要知道算法在**未知数据 (Novel Data)** 上的准确性，而不仅仅是训练数据。
*   **目标**: 利用有限的数据评估算法的泛化能力。

---

## 2. 交叉验证 (Cross-validation)

### 2.1 超参数调优 (Hyper-parameter Tuning)
*   **参数 (Parameters)**: 由学习算法基于训练集自动学习得到的变量（例如线性模型中的 $w, b$）。
*   **超参数 (Hyper-parameters)**: 在学习算法运行之前确定的变量，通常手动选择。
    *   *例子*: 多项式回归的阶数、决策树的最大深度、随机森林的树数量、SVM 中的正则化参数 $C$、梯度下降的学习率。

**如何选择超参数？ (模型选择问题)**
1.  **Idea 1**: 选择在**所有数据**上表现最好的超参数。
    *   *问题*: 导致过拟合，无法评估泛化能力。
2.  **Idea 2**: 将数据分为**训练集 (Train)** 和 **测试集 (Test)**。
    *   *问题*: 测试集用于选择超参数后，就不能再作为“未见数据”来评估最终性能。
3.  **Idea 3**: 将数据分为 **训练集 (Train)**、**验证集 (Validation)** 和 **测试集 (Test)**。
    *   *流程*: 在 Train 上训练，在 Validation 上选择超参数，在 Test 上评估最终性能。
    *   *问题*: 性能受数据随机划分的影响较大（某次划分可能导致验证集不具代表性）。

### 2.2 K折交叉验证 (K-fold Cross-validation)
为了解决上述问题，引入 K-fold 交叉验证。

**步骤**:
1.  将训练数据分割为 $K$ 个互不重叠的折 (Folds)。
2.  进行 $K$ 次试验：
    *   每次选择其中 **1个折** 作为验证集。
    *   其余 **K-1个折** 作为训练集。
    *   训练模型并计算验证集上的误差。
3.  计算 $K$ 次试验的**平均结果**。
4.  选择平均性能最好的超参数。

**优缺点**:
*   **优点**: 评估结果比单次划分更稳定、更彻底。
*   **缺点**:
    *   计算成本高（需要训练 $K$ 次）。
    *   引入了新的超参数 $K$（通常设为 5 到 10）。
    *   $K$ 太大 $\rightarrow$ 验证集太小，且各次训练集重叠度高 $\rightarrow$ 过拟合风险。
    *   $K$ 太小 $\rightarrow$ 训练数据不足 $\rightarrow$ 欠拟合风险。

---

## 3. 回归任务的评估指标 (Evaluation Metrics for Regression)

对于回归问题，主要关注预测值 $\hat{y}_i$ 与真实值 $y_i$ 的差异。

### 3.1 均方误差 (Mean Square Error, MSE)
$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
*   对大误差惩罚更重（平方项）。

### 3.2 平均绝对误差 (Mean Absolute Error, MAE)
$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$
*   反映预测误差的平均绝对大小。

---

## 4. 分类任务的评估指标 (Evaluation Metrics for Classification)

### 4.1 混淆矩阵 (Confusion Matrix)
以二分类为例，定义 Class 1 为正类 (Positive)，Class 2 为负类 (Negative)。

| | 预测为 Positive | 预测为 Negative |
| :--- | :---: | :---: |
| **真实为 Positive** | **TP** (True Positive) | **FN** (False Negative) <br> (Type II Error) |
| **真实为 Negative** | **FP** (False Positive) <br> (Type I Error) | **TN** (True Negative) |

### 4.2 基础指标
1.  **准确率 (Accuracy)**:
    $$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
    *   *局限性*: 在类别不平衡（Imbalanced）的数据集中失效。

2.  **精确率 (Precision)**: 预测为正的样本中有多少是真的正样本。
    $$ \text{Precision} = \frac{TP}{TP + FP} $$

3.  **召回率 (Recall)**: 真实的正样本中有多少被预测出来了。
    $$ \text{Recall} = \frac{TP}{TP + FN} $$

### 4.3 代价敏感准确率 (Cost-sensitive Accuracy)
在某些场景下（如医疗诊断），不同错误的代价不同（漏诊癌症 vs 误诊健康人）。
设定代价矩阵或权重：$C_{p,p}$ (正类预测正确的收益/代价), $C_{n,p}$ (负类预测为正类的代价) 等。

**公式**:
$$ \text{Cost-sensitive Acc} = \frac{C_{p,p} \cdot TP + C_{n,n} \cdot TN}{C_{p,p} \cdot TP + C_{n,n} \cdot TN + C_{p,n} \cdot FN + C_{n,p} \cdot FP} $$

### 4.4 归一化比率 (Normalized Rates)
将数值归一化到 [0, 1] 区间：

*   **真正例率 (TPR)** / Recall: $TPR = \frac{TP}{TP + FN}$
*   **假负例率 (FNR)**: $FNR = \frac{FN}{TP + FN} = 1 - TPR$
*   **真负例率 (TNR)**: $TNR = \frac{TN}{FP + TN}$
*   **假正例率 (FPR)**: $FPR = \frac{FP}{FP + TN} = 1 - TNR$

**平衡准确率 (Balanced Accuracy)**:
$$ \text{Accuracy}_{bal} = \frac{TPR + TNR}{2} = 1 - \frac{FPR + FNR}{2} $$

### 4.5 阈值与曲线 (Thresholds and Curves)
分类器通常输出一个分数或概率，通过设定**阈值 (Threshold, $\tau$)** 来决定类别。
*   改变 $\tau$ 会改变 TP, FP, TN, FN 的值，从而形成一系列**操作点 (Operating Points)**。

#### 4.5.1 等误差率 (Equal Error Rate, EER)
*   随着阈值变化，FPR 下降，FNR 上升。
*   **EER**: 当 $FPR = FNR$ 时的误差率。

#### 4.5.2 DET 曲线 (Detection Error Trade-off)
*   **X轴**: FPR
*   **Y轴**: FNR
*   **特点**: 曲线越靠近左下角越好。

#### 4.5.3 ROC 曲线 (Receiver Operating Characteristic)
*   **X轴**: FPR (False Positive Rate)
*   **Y轴**: TPR (True Positive Rate)
*   **特点**: 曲线越靠近**左上角**越好。
*   **对角线 ($y=x$)**: 代表随机猜测 (Random Guess)。

### 4.6 ROC曲线下的面积 (AUC - Area Under Curve)

#### 4.6.1 定义与性质
*   **定义**: AUC 衡量了分类器将随机抽取的正样本排在随机抽取的负样本之前的概率。
*   **范围**: 0 到 1。
    *   AUC = 1: 完美分类。
    *   AUC = 0.5: 随机猜测。
    *   AUC = 0: 完全错误的预测（全部反着来）。
*   **性质**:
    *   **尺度不变性 (Scale-invariant)**: 输出数值的范围不影响 AUC，只看相对排序。
    *   **阈值不变性**: 不需要设定特定阈值，衡量的是整体排序质量。

#### 4.6.2 AUC 的详细计算公式推导
假设我们有 $m^+$ 个正样本和 $m^-$ 个负样本。
设 $g(x)$ 为预测器输出的分数。

定义 $e_{ij}$ 为第 $i$ 个正样本与第 $j$ 个负样本的分数差：
$$ e_{ij} = g(x_i^+) - g(x_j^-) $$

引入 **Heaviside 阶跃函数** $u(e)$:
$$
u(e) = \begin{cases}
1, & \text{if } e > 0 \\
0.5, & \text{if } e = 0 \\
0, & \text{if } e < 0
\end{cases}
$$

**AUC 计算公式**:
$$ AUC = \frac{1}{m^+ m^-} \sum_{i=1}^{m^+} \sum_{j=1}^{m^-} u(e_{ij}) $$

**解释**:
1.  遍历所有正负样本对 $(x_i^+, x_j^-)$。
2.  如果正样本得分高于负样本 ($e_{ij} > 0$)，得 1 分。
3.  如果得分相等，得 0.5 分。
4.  如果正样本得分低，得 0 分。
5.  计算平均得分（除以总对数 $m^+ m^-$）。

#### 4.6.3 AUC 计算示例 (Example)
**数据**:
*   2个正样本 ($m^+=2$): $x_1^+, x_2^+$
*   2个负样本 ($m^-=2$): $x_1^-, x_2^-$
*   预测分数 $g(x)$:
    *   $g(x_1^-) = 0.1$
    *   $g(x_2^-) = 0.4$
    *   $g(x_1^+) = 0.35$
    *   $g(x_2^+) = 0.8$

**计算步骤**:
我们需要计算 $2 \times 2 = 4$ 个 $e_{ij}$：

1.  $e_{11} = g(x_1^+) - g(x_1^-) = 0.35 - 0.1 = 0.25 > 0 \Rightarrow u=1$
2.  $e_{12} = g(x_1^+) - g(x_2^-) = 0.35 - 0.4 = -0.05 < 0 \Rightarrow u=0$
3.  $e_{21} = g(x_2^+) - g(x_1^-) = 0.8 - 0.1 = 0.7 > 0 \Rightarrow u=1$
4.  $e_{22} = g(x_2^+) - g(x_2^-) = 0.8 - 0.4 = 0.4 > 0 \Rightarrow u=1$

**最终 AUC**:
$$ AUC = \frac{1}{2 \times 2} (1 + 0 + 1 + 1) = \frac{3}{4} = 0.75 $$

### 4.7 多分类评估 (Multicategory)
*   混淆矩阵扩展为 $N \times N$。
*   ROC 在多分类中定义复杂，无统一标准。

---

## 5. 计算与可维护性 (Computational and Maintainability)
*   **计算效率**: 速度快、并行化。
*   **软件质量**: 灵活性、可扩展性、模块化、可维护性。
*   **权衡 (Trade-off)**: 过度追求底层优化的计算效率（如汇编级并行）可能会牺牲代码的可读性和可维护性（如面向对象特性的丢失）。需要在两者间仔细权衡。