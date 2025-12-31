# DDA3020 Machine Learning - Lecture 11: Recurrent Neural Networks and Transformer

**讲师**: Juexiao Zhou (SDS, CUHK-SZ)
**日期**: Oct 28/30, 2025

---

## 1. 序列数据分析 (Sequential Data Analysis)

### 1.1 动机 (Motivation)
*   **回顾**: MLP (多层感知机) 和 CNN (卷积神经网络) 主要用于处理表格数据和图像数据。
*   **序列数据**: 数据以序列形式排列，**顺序 (Order)** 至关重要。
*   **特点**:
    *   输入长度可变 (Variable length input)。
    *   顺序可变 (Variable order)。
    *   例如："I visited Paris in 2014" vs "In 2014, I visited Paris"。

### 1.2 典型任务
1.  **时间序列预测 (Time series prediction)**: 股票、天气等。
2.  **语音识别 (Speech recognition)**: 音频 $\to$ 文本 (Audio to text)。
3.  **机器翻译 (Machine translation)**: 文本 $\to$ 文本 (Text to text)。
4.  **图像描述 (Image captioning)**: 图像 $\to$ 文本 (Image to text)。
5.  **其他**: 文本生成 (ChatGPT), 视频生成 (SORA), 生物序列分析 (DNA)。

---

## 2. 循环神经网络 (Recurrent Neural Network, RNN)

### 2.1 基础架构 (Basic Architecture)

RNN 的核心思想是在时间步 (Time Steps) 之间**共享参数**。

#### 核心公式
在时刻 $t$，RNN 处理输入 $x_t$ 和上一时刻的隐藏状态 $h_{t-1}$：

$$
\begin{aligned}
h_t &= f_W(h_{t-1}, x_t) \\
\hat{y}_t &= g_{W'}(h_t)
\end{aligned}
$$

*   **$h_t$ (Hidden State)**: 当前时刻的隐藏状态，包含了直到时刻 $t$ 的序列信息。
*   **$x_t$ (Input)**: 当前时刻的输入向量。
*   **$\hat{y}_t$ (Output)**: 当前时刻的预测输出。
*   **$W, W'$ (Parameters)**: 权重矩阵，**在所有时间步共享**。

#### 具体形式 (Vanilla RNN)
通常使用 Tanh 作为激活函数：

$$
\begin{aligned}
h_t &= \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h) \\
\hat{y}_t &= W_{hy} h_t + b_y
\end{aligned}
$$

**参数解释**:
*   $W_{hh}$: 隐藏层到隐藏层的权重矩阵。
*   $W_{xh}$: 输入层到隐藏层的权重矩阵。
*   $W_{hy}$: 隐藏层到输出层的权重矩阵 (即上文的 $W'$ )。
*   $b_h, b_y$: 偏置项。

### 2.2 损失函数与训练 (Loss & Training)

#### 损失函数
*   **单步损失**: $L_t(y_t, \hat{y}_t)$，例如交叉熵损失。
*   **总损失**: 所有时间步损失之和。
    $$ E(\theta) = \sum_{t=1}^{T} L(y_t, \hat{y}_t) $$
    其中 $\theta = \{W_{hh}, W_{xh}, W_{hy}, b\}$。

#### 随时间反向传播 (Backpropagation Through Time, BPTT)
RNN 的训练类似于前馈神经网络，但需要将网络按时间展开。

**梯度推导 (以 $W_{hh}$ 为例)**:
由于参数共享，计算总损失对 $W_{hh}$ 的梯度时，需要对每个时间步的梯度求和。根据链式法则：

$$ \frac{\partial E}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}} $$

对于单个时间步 $t$，$\frac{\partial L_t}{\partial W_{hh}}$ 依赖于 $h_t$，而 $h_t$ 依赖于 $h_{t-1}$，以此类推。
$$ \frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial \hat{y}_t} \frac{\partial \hat{y}_t}{\partial h_t} \sum_{k=1}^{t} \left( \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}} \right) $$
其中 $\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$ 是连乘项。

### 2.3 梯度问题 (Gradient Exploding & Vanishing)

由于 BPTT 中存在连乘项 $\prod \frac{\partial h_j}{\partial h_{j-1}}$ (通常涉及 $W_{hh}$ 的幂)：
1.  **梯度爆炸 (Gradient Exploding)**: 梯度变得极大，导致数值不稳定。
    *   *解决*: **梯度裁剪 (Gradient Clipping)** (设置阈值截断梯度)。
2.  **梯度消失 (Gradient Vanishing)**: 梯度趋近于 0，导致网络无法学习长距离依赖 (Long-term dependencies)。
    *   *解决*: 使用 **LSTM** 或 **GRU**。

---

## 3. 长短期记忆网络 (LSTM)

### 3.1 概述
*   **提出者**: Hochreiter & Schmidhuber (1997)。
*   **目的**: 缓解梯度消失问题，捕捉长距离依赖。
*   **核心**: 引入了 **细胞状态 (Cell State, $c_t$)** 和 **门控机制 (Gating Mechanisms)**。

### 3.2 核心公式与结构
LSTM 在每个时间步维护两个状态：$h_t$ (隐藏状态) 和 $c_t$ (细胞状态)。

#### 门控计算
所有门的值域为 $(0, 1)$，由 Sigmoid ($\sigma$) 激活。
1.  **遗忘门 (Forget Gate)**: 决定丢弃多少旧的细胞状态信息。
    $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
2.  **输入门 (Input Gate)**: 决定更新多少新信息到细胞状态。
    $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
3.  **候选细胞状态**: 创建新的候选值向量。
    $$ \tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) $$
4.  **输出门 (Output Gate)**: 决定基于当前细胞状态输出什么值。
    $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$

#### 状态更新
1.  **更新细胞状态**: 旧状态 $\times$ 遗忘比例 + 新候选值 $\times$ 输入比例。
    $$ c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t $$
    *(注: $\odot$ 表示逐元素乘法/Hadamard Product)*
    *解释*: 加法操作 ($+$) 使得梯度可以更顺畅地反向传播，缓解梯度消失。
2.  **更新隐藏状态**:
    $$ h_t = o_t \odot \tanh(c_t) $$

### 3.3 RNN 的其他变体 (Extensions)
*   **GRU (Gated Recurrent Unit)** (Cho et al., 2014):
    *   将遗忘门和输入门合并为“更新门”。
    *   没有单独的细胞状态 $c_t$，只有 $h_t$。
    *   **优点**: 参数比 LSTM 少，计算更高效。
*   **多层 RNN (Multi-layer RNNs)**: 垂直堆叠多个 RNN 层，增加模型深度。

### 3.4 RNN 及其变体的局限性
1.  **长距离依赖困难**: 尽管 LSTM 改善了这一点，但对于极长序列仍有挑战。
2.  **无法并行化 (Limited Parallelization)**: 必须先计算 $h_{t-1}$ 才能计算 $h_t$，导致训练速度慢，无法充分利用 GPU。

---

## 4. Transformer

### 4.1 概述
*   **来源**: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)。
*   **核心**: 完全抛弃循环和卷积，完全基于 **注意力机制 (Attention Mechanism)**。
*   **优势**:
    *   易于并行化 (Easier parallelization)。
    *   更有效地处理长距离依赖。
    *   更高的模型容量。

### 4.2 关键组件
1.  **输入嵌入 (Input Embeddings)**: 将词转换为向量。
2.  **位置编码 (Positional Encodings)**: 由于没有循环结构，模型不知道词的顺序，需显式加入位置信息向量。

### 4.3 注意力机制 (Attention Mechanism) - **核心推导**

#### 定义
假设输入矩阵为 $X \in \mathbb{R}^{n \times d}$ ($n$ 为样本/词数，$d$ 为维度)。
引入三个参数矩阵：
*   $W_Q \in \mathbb{R}^{d \times d'}$
*   $W_K \in \mathbb{R}^{d \times d'}$
*   $W_V \in \mathbb{R}^{d \times d'}$

#### 计算 Query, Key, Value
$$ Q = X W_Q, \quad K = X W_K, \quad V = X W_V $$
*   **Q (Query)**: 查询向量。
*   **K (Key)**: 键向量，用于被查询匹配。
*   **V (Value)**: 值向量，包含实际内容信息。

#### 缩放点积注意力 (Scaled Dot-Product Attention) 公式
$$ Z = \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q K^\top}{\sqrt{d'}} \right) V $$

**详细步骤解释**:
1.  **相似度计算 ($QK^\top$)**: 计算 Query 和 Key 的点积。如果 $Q_i$ 和 $K_j$ 相似（点积大），说明第 $i$ 个词应该关注第 $j$ 个词。得到矩阵维度 $n \times n$。
2.  **缩放 ($\frac{1}{\sqrt{d'}}$)**: 除以维度的平方根，防止点积数值过大导致 Softmax 进入梯度极小区域。
3.  **归一化 (Softmax)**: 对每一行进行 Softmax，得到概率分布（权重和为1）。
    *   **Attention Map**: $A = \text{softmax}(\frac{Q K^\top}{\sqrt{d'}}) \in \mathbb{R}^{n \times n}$。
4.  **加权求和 ($AV$)**: 用计算出的权重对 Value ($V$) 进行加权求和，得到最终输出 $Z$。

#### 自注意力 vs 交叉注意力
*   **自注意力 (Self-Attention)**: $Q, K, V$ 都来自同一个输入源 $X$。用于理解序列内部的关联（如代词指代）。
*   **交叉注意力 (Cross-Attention)**: $Q$ 来自一个源（如解码器），$K, V$ 来自另一个源（如编码器）。用于序列到序列任务（如翻译）。

---

## 5. 总结 (Summary)

| 特性        | RNN / LSTM       | Transformer             |
| :-------- | :--------------- | :---------------------- |
| **处理方式**  | 序列化 (Sequential) | 并行化 (Parallel)          |
| **长距离依赖** | 较弱 (受梯度消失影响)     | 强 (直接通过 Attention 连接)   |
| **核心操作**  | 递归更新 $h_t$       | 矩阵乘法 + Softmax          |
| **位置信息**  | 隐含在处理顺序中         | 需显式添加位置编码               |
| **主要应用**  | 早期 NLP, 时间序列     | 现代 NLP (LLMs), 视觉 (ViT) |
