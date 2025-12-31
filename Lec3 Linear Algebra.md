# DDA3020 Lecture 03: 线性代数 (Linear Algebra)

## 1. 向量、矩阵及其范数 (Vector, Matrix, and their Norms)

### 1.1 标量 (Scalar)
*   **定义**: 一个简单的数值（实数），如 $15$ 或 $-3.2$。
*   **符号**: 斜体字母，如 $x$ 或 $a$。
*   **运算符号**:
    *   求和 (Summation): $\sum_{i=1}^m x_i = x_1 + x_2 + \dots + x_m$
    *   连乘 (Product): $\prod_{i=1}^m x_i = x_1 \cdot x_2 \cdot \dots \cdot x_m$

### 1.2 向量 (Vector)
*   **定义**: 有序的标量列表，称为属性 (attributes)。
*   **符号**: 粗体小写字母，如 $\mathbf{x}$ 或 $\mathbf{w}$。通常以列向量形式表示：
    $$ \mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} $$
*   **索引**: $x^{(j)}$ 或 $x_j$ 表示向量 $\mathbf{x}$ 的第 $j$ 个维度的值。
    *   *注意*: 不要与幂运算混淆。$(x^{(j)})^2$ 表示第 $j$ 个元素的平方。

### 1.3 矩阵 (Matrix)
*   **定义**: 按行和列排列的矩形数字阵列。
*   **符号**: 粗体大写字母，如 $\mathbf{X}$ 或 $\mathbf{W}$。
    $$ \mathbf{X} = \begin{bmatrix} x_{1,1} & x_{1,2} \\ x_{2,1} & x_{2,2} \end{bmatrix} $$
*   **索引**: $x_{i,j}$ 表示第 $i$ 行第 $j$ 列的元素。

### 1.4 向量与矩阵运算
假设 $\mathbf{x}, \mathbf{y}$ 为向量，$\mathbf{X}, \mathbf{W}$ 为矩阵，$a$ 为标量。

1.  **加减法**: 对应元素相加减。
2.  **标量乘法**: 每个元素乘以标量 $a\mathbf{x}$。
3.  **转置 (Transpose)**: 行列互换。
    *   向量转置: $\mathbf{x}^\top = [x_1, x_2]$
    *   矩阵转置: $(\mathbf{X}^\top)_{i,j} = \mathbf{X}_{j,i}$
4.  **点积/内积 (Dot Product)**:
    $$ \mathbf{x} \cdot \mathbf{y} = \mathbf{x}^\top \mathbf{y} = \sum_{i} x_i y_i $$
5.  **迹 (Trace)**: 方阵对角线元素之和。
    $$ \text{tr}(\mathbf{X}) = \sum_{i=1}^n x_{i,i} $$
6.  **矩阵-向量乘法**: $\mathbf{X}\mathbf{w}$，结果为向量。
7.  **矩阵-矩阵乘法**: $\mathbf{X}\mathbf{W}$。
    *   $(\mathbf{X}\mathbf{W})_{i,j} = \sum_{k} x_{i,k} w_{k,j}$ (第 $i$ 行与第 $j$ 列的点积)。

### 1.5 向量范数 (Vector Norms)
范数 $\lVert \cdot \rVert$ 用于衡量向量的大小（长度）。需满足非负性、齐次性和三角不等式。

*   **$\ell_2$-范数 (欧几里得范数)**:
    $$ \lVert \mathbf{x} \rVert_2 = \sqrt{\sum_{i=1}^d x_i^2} $$
*   **$\ell_1$-范数 (曼哈顿距离)**:
    $$ \lVert \mathbf{x} \rVert_1 = \sum_{i=1}^d |x_i| $$
*   **$\ell_p$-范数 ($p \ge 1$)**:
    $$ \lVert \mathbf{x} \rVert_p = \left( \sum_{i=1}^d |x_i|^p \right)^{1/p} $$
*   **$\ell_0$-范数**:
    $$ \lVert \mathbf{x} \rVert_0 = \text{number of nonzero elements in } \mathbf{x} $$
    *   *解释*: 向量中非零元素的个数。

### 1.6 矩阵范数 (Matrix Norms)
*   **Frobenius 范数**:
    $$ \lVert \mathbf{X} \rVert_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{i,j}^2} $$
    *   *解释*: 矩阵所有元素的平方和的平方根。
*   **谱范数 (Spectral Norm)**:
    $$ \lVert \mathbf{X} \rVert_2 = \sigma_{\max}(\mathbf{X}) $$
    *   *解释*: 矩阵最大的奇异值 (singular value)。

---

## 2. 矩阵逆、行列式与独立性 (Matrix Inverse, Determinant, Independence)

### 2.1 矩阵逆 (Matrix Inverse)
*   **定义**: 对于 $d \times d$ 方阵 $\mathbf{A}$，若存在矩阵 $\mathbf{B}$ 使得 $\mathbf{AB} = \mathbf{BA} = \mathbf{I}$，则 $\mathbf{A}$ 是可逆的 (invertible/nonsingular)，$\mathbf{B} = \mathbf{A}^{-1}$。
*   **计算公式**:
    $$ \mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \text{adj}(\mathbf{A}) $$
    *   **$\det(\mathbf{A})$**: 行列式。
    *   **$\text{adj}(\mathbf{A})$**: 伴随矩阵 (Adjugate matrix)，即代数余子式矩阵 $\mathbf{C}$ 的转置 ($\mathbf{C}^\top$)。
    *   **代数余子式 (Cofactor)**: $C_{i,j} = M_{i,j} \times (-1)^{i+j}$，其中 $M_{i,j}$ 是去除第 $i$ 行第 $j$ 列后的子矩阵的行列式。

### 2.2 基于 SVD 的计算
对于矩阵 $\mathbf{A}$ 进行奇异值分解 (SVD): $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$，其中 $\mathbf{\Sigma} = \text{diag}(\sigma_1, \sigma_2, \dots)$。
*   **逆矩阵**: $\mathbf{A}^{-1} = \mathbf{V}\mathbf{\Sigma}^{-1}\mathbf{U}^\top$，其中 $\mathbf{\Sigma}^{-1} = \text{diag}(\sigma_1^{-1}, \sigma_2^{-1}, \dots)$。
*   **行列式**: $\det(\mathbf{A}) = \prod_i \sigma_i$ (所有奇异值的乘积)。

### 2.3 线性相关与无关
*   **线性相关 (Linearly Dependent)**: 存在不全为零的系数 $\beta_1, \dots, \beta_m$ 使得：
    $$ \beta_1 \mathbf{x}_1 + \dots + \beta_m \mathbf{x}_m = 0 $$
*   **线性无关 (Linearly Independent)**: 上述等式仅在 $\beta_1 = \dots = \beta_m = 0$ 时成立。

---

## 3. 线性方程组 (Systems of Linear Equations)

考虑方程组 $\mathbf{X}\mathbf{w} = \mathbf{y}$，其中 $\mathbf{X} \in \mathbb{R}^{m \times d}$。
*   $m$: 方程数量 (样本数)。
*   $d$: 未知数数量 (特征维度)。

### 3.1 适定系统 (Square / Even-determined System)
*   **条件**: $m = d$ (方程数等于未知数数)，且 $\mathbf{X}$ 可逆（行/列线性无关）。
*   **解法**:
    $$ \mathbf{w} = \mathbf{X}^{-1}\mathbf{y} $$
*   **推导**:
    $$ \begin{aligned} \mathbf{X}\mathbf{w} &= \mathbf{y} \\ \mathbf{X}^{-1}\mathbf{X}\mathbf{w} &= \mathbf{X}^{-1}\mathbf{y} \quad (\text{两边左乘 } \mathbf{X}^{-1}) \\ \mathbf{I}\mathbf{w} &= \mathbf{X}^{-1}\mathbf{y} \\ \mathbf{w} &= \mathbf{X}^{-1}\mathbf{y} \end{aligned} $$

### 3.2 超定系统 (Over-determined System)
*   **条件**: $m > d$ (方程数多于未知数数)。通常没有精确解。
*   **目标**: 寻找近似解。
*   **解法 (左逆 Left-Inverse)**:
    $$ \mathbf{w} = \mathbf{X}^\dagger \mathbf{y} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} $$
*   **推导**:
    1.  定义左逆 $\mathbf{B}$ 满足 $\mathbf{B}\mathbf{X} = \mathbf{I}$。
    2.  对于超定矩阵 $\mathbf{X}$，其左逆通常计算为 $\mathbf{X}^\dagger = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top$ (前提是 $\mathbf{X}^\top \mathbf{X}$ 可逆)。
    3.  在方程两边左乘 $\mathbf{X}^\dagger$:
        $$ \begin{aligned} \mathbf{X}\mathbf{w} &\approx \mathbf{y} \\ \mathbf{X}^\dagger \mathbf{X} \mathbf{w} &= \mathbf{X}^\dagger \mathbf{y} \\ \mathbf{I} \mathbf{w} &= (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} \\ \mathbf{w} &= (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} \end{aligned} $$
    *   *注*: 这通常对应于最小二乘解 (Least Squares Solution)。

### 3.3 欠定系统 (Under-determined System)
*   **条件**: $m < d$ (未知数数多于方程数)。通常有无穷多解。
*   **目标**: 寻找满足约束的特定解（如 $\mathbf{w} = \mathbf{X}^\top \mathbf{a}$ 形式的解）。
*   **解法 (右逆 Right-Inverse)**:
    $$ \mathbf{w} = \mathbf{X}^\dagger \mathbf{y} = \mathbf{X}^\top (\mathbf{X}\mathbf{X}^\top)^{-1} \mathbf{y} $$
*   **推导**:
    1.  定义右逆 $\mathbf{B}$ 满足 $\mathbf{X}\mathbf{B} = \mathbf{I}$。
    2.  对于欠定矩阵 $\mathbf{X}$，其右逆通常计算为 $\mathbf{X}^\dagger = \mathbf{X}^\top (\mathbf{X}\mathbf{X}^\top)^{-1}$ (前提是 $\mathbf{X}\mathbf{X}^\top$ 可逆)。
    3.  为了从无穷多解中选一个，我们限制搜索空间，假设 $\mathbf{w}$ 可以表示为 $\mathbf{X}$ 行向量的线性组合，即令 $\mathbf{w} = \mathbf{X}^\top \mathbf{a}$ ($\mathbf{a}$ 是辅助向量)。
    4.  代入原方程 $\mathbf{X}\mathbf{w} = \mathbf{y}$:
        $$ \begin{aligned} \mathbf{X}(\mathbf{X}^\top \mathbf{a}) &= \mathbf{y} \\ (\mathbf{X}\mathbf{X}^\top) \mathbf{a} &= \mathbf{y} \end{aligned} $$
    5.  求解 $\mathbf{a}$:
        $$ \mathbf{a} = (\mathbf{X}\mathbf{X}^\top)^{-1} \mathbf{y} $$
    6.  回代求 $\mathbf{w}$:
        $$ \mathbf{w} = \mathbf{X}^\top \mathbf{a} = \mathbf{X}^\top (\mathbf{X}\mathbf{X}^\top)^{-1} \mathbf{y} $$