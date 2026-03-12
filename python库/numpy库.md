NumPy 是 Python 中最重要的科学计算库之一，是许多高级库（如 Pandas、SciPy、Matplotlib、PyTorch、TensorFlow）的基石。它主要提供**高效的多维数组对象（ndarray）**和大量的数学函数，能够对整个数组进行快速向量化运算，避免慢速的 Python 循环。

### 为什么用 NumPy？
- **速度快**：底层用 C 实现，数组存储连续，运算向量化。
- **内存高效**：同类型数据紧密存储。
- **广播机制**：自动对不同形状数组进行运算。
- **功能丰富**：线性代数、傅里叶变换、随机数、统计等一应俱全。

### 基本导入
```python
import numpy as np
```

### 1. 创建数组（ndarray）
```python
# 从列表/元组创建
arr1 = np.array([1, 2, 3, 4])                  # 一维数组
arr2 = np.array([[1, 2], [3, 4]])              # 二维数组

# 常用创建函数
np.zeros((3, 4))          # 3x4 全零数组，dtype 默认 float64
np.ones((2, 3, 4))        # 三维全1数组
np.full((2, 3), 7)        # 指定填充值

np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]，类似 range
np.linspace(0, 1, 5)      # [0., 0.25, 0.5, 0.75, 1.] 等间隔

np.random.random((3, 3))  # 随机浮点数 [0, 1)
np.random.randint(0, 10, (2, 3))  # 随机整数
np.eye(3)                 # 3x3 单位矩阵
```

### 2. 数组基本属性
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape     # (2, 3) 形状
arr.ndim      # 2     维度数
arr.dtype     # int64 数据类型
arr.size      # 6     元素总数
arr.itemsize  # 8     每个元素字节数
```


常用 dtype：int32/int64, float32/float64, bool, complex 等。可以手动指定：
```python
np.array([1, 2, 3], dtype=np.float64)
```

### 3. 索引与切片
和 Python 列表类似，但支持多维和广播。
```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr[0, 1]         # 2（第0行第1列）
arr[1:]           # 后两行：[[4,5,6], [7,8,9]]
arr[:, 1]         # 第1列：[2, 5, 8]
arr[0:2, 1:3]     # 子矩阵 [[2,3], [5,6]]

# 布尔索引
arr[arr > 5]      # [6, 7, 8, 9]

# 花式索引（fancy indexing）
arr[[0, 2], [1, 2]]  # 取 (0,1) 和 (2,2) 位置的元素 → [2, 9]
```


### 4. 数学运算与广播（Broadcasting）
NumPy 的核心优势：元素级运算自动向量化。
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b          # [5, 7, 9]
a * b          # [4, 10, 18]
a ** 2         # [1, 4, 9]
np.sqrt(a)     # 开方

# 广播：形状不同的数组自动扩展
a = np.array([[1, 2, 3], [4, 5, 6]])   # (2, 3)
b = np.array([10, 20, 30])              # (3,)
a + b                                  # 每行都加 [10,20,30]
# 结果：[[11,22,33], [14,25,36]]
```

通用函数（ufunc）：sin, cos, exp, log, sqrt, mean, std 等，都支持向量化。


### 5. 聚合/统计函数
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.sum()          # 总和 21
arr.sum(axis=0)    # 按列求和 [5, 7, 9]
arr.sum(axis=1)    # 按行求和 [6, 15]

arr.mean(), 
arr.std(),
 arr.var()
arr.min(),
 arr.max()
arr.argmax()       # 最大值索引
np.median(arr)
np.percentile(arr, 50)  # 中位数
```

```
arr.sum() = 21          # 总和
arr.sum(axis=0) = [5 7 9]    # 按列求和
arr.sum(axis=1) = [ 6 15]    # 按行求和
arr.mean() = 3.5          # 均值
arr.std() = 1.7078          # 标准差
arr.var() = 2.9167          # 方差
arr.min() = 1          # 最小值
arr.max() = 6          # 最大值
arr.argmax() = 5       # 最大值索引（扁平化后为第5个元素，即6）
np.median(arr) = 3.5        # 中位数
np.percentile(arr, 50) = 3.5  # 50%分位数（中位数）
```


### 6. 形状操作（Reshaping）
不复制数据，仅改变视图。
```python
arr = np.arange(12)   # [0,1,2,3,~11]

arr.reshape(3, 4)      # 变成 3x4    [0,1,2,3][4,5,6,7][8,9,10,11]
arr.resize(3, 4)       # 原地修改（慎用）  [0,1,2,3][4,5,6,7][8,9,10,11]

arr.ravel()            # 展平为一维   [0,1,2,3,~11]
arr.flatten()          # 展平并复制   [0,1,2,3,~11]

# 拼接与拆分
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

np.concatenate((a, b), axis=0)   # 垂直拼接
np.vstack((a, b))                # 同上
np.hstack((a, a))                # 水平拼接

np.split(arr, 3)                 # 等分拆分
```


### 7. 线性代数（numpy.linalg）
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.dot(a, b)                # 矩阵乘法
a @ b                       # Python 3.5+ 同样是矩阵乘法

np.linalg.inv(a)            # 逆矩阵
np.linalg.det(a)            # 行列式
np.linalg.eig(a)            # 特征值与特征向量
np.linalg.svd(a)            # 奇异值分解
np.linalg.solve(a, b)       # 解线性方程 ax = b
```


### 8. 随机数模块（numpy.random）
```python
np.random.seed(42)          # 设置种子，可复现

np.random.rand(3, 3)        # [0,1) 均匀分布
np.random.randn(3, 3)       # 标准正态分布
np.random.randint(0, 10, 5) # 随机整数

np.random.shuffle(arr)      # 原地打乱
np.random.choice(arr, 5)    # 有放回抽样
```

### 9. 保存与加载
```python
np.save('my_arr.npy', arr)        # 保存单个数组
np.load('my_arr.npy')

np.savez('arrays.npz', a=a, b=b)  # 保存多个
data = np.load('arrays.npz')
data['a']

np.savetxt('data.csv', arr, delimiter=',')  # 保存为文本
np.loadtxt('data.csv', delimiter=',')
```

### 小贴士
- **视图 vs 复制**：切片通常返回视图（修改会影响原数组），用 copy() 显式复制。
- **性能**：尽量使用向量化操作，避免 for 循环。
- **与 PyTorch/TensorFlow 互操作**：np.array ↔ torch.tensor 很容易转换（torch.from_numpy、tensor.numpy()）。

NumPy 是科学计算的入门必备，几乎所有数据处理任务都会用到它。掌握数组创建、索引、广播、聚合这几块，就能覆盖 80% 的常见场景。如果有具体任务或想看某个函数的例子，再告诉我！