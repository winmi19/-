

我帮你整理一个**竞赛/刷题常用分类体系（非常实用）**👇

---
0-1背包  ：最大限度提高水果口味，蓝桥云课
恰好0-1背包
完全01背包
# 一、线性 DP（最基础 ⭐⭐⭐）

👉 状态沿着数组/序列推进

### 典型特征

- 从左到右
    
- 每个状态只和前面几个有关
    

### 常见题型

- 最长上升子序列（LIS）
    
- 打家劫舍
    
- 子序列问题
    

### 模板

```python
dp[i] = ...
dp[i] = f(dp[i-1], dp[i-2], ...)
```

---

# 二、背包 DP（超级重点 ⭐⭐⭐⭐⭐）

👉 本质：**选 or 不选**

### 分类

#### 1. 0-1 背包

每个物品只能选一次

```python
for i in range(n):
    for j in range(W, w[i]-1, -1):
        dp[j] = max(dp[j], dp[j-w[i]] + v[i])
```

---

#### 2. 完全背包

可以无限选

```python
for i in range(n):
    for j in range(w[i], W+1):
        dp[j] = max(dp[j], dp[j-w[i]] + v[i])
```

---

#### 3. 多重背包

有数量限制

---

#### 4. 分组背包

每组只能选一个

---

# 三、区间 DP（很经典 ⭐⭐⭐⭐）

👉 在一个区间内做决策

### 特征

- 枚举区间长度
    
- 枚举分割点
    

### 典型题

- 石子合并
    
- 括号匹配
    
- 区间最大值
    

### 模板

```python
for len in range(2, n+1):
    for l in range(...):
        r = l + len - 1
        for k in range(l, r):
            dp[l][r] = min/max(...)
```

---

# 四、网格 DP（二维 DP ⭐⭐⭐⭐）

👉 在矩阵上移动

### 特征

- 从左上走到右下
    
- 只能向右/向下
    

### 典型题

- 最小路径和
    
- 不同路径数
    

### 模板

```python
dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
```

---

# 五、树形 DP（进阶 ⭐⭐⭐⭐）

👉 在树结构上做 DP

### 特征

- DFS + DP
    

### 典型题

- 树的最大独立集
    
- 树上选点问题
    

---

# 六、状态压缩 DP（位运算 ⭐⭐⭐⭐）

👉 状态用二进制表示

### 典型题

- TSP（旅行商问题）
    
- 集合选择问题
    

```python
dp[mask][i]
```

---

# 七、数位 DP（偏难 ⭐⭐⭐⭐⭐）

👉 处理“数字限制”的问题

### 典型题

- 数字中不能出现某些数
    
- 统计区间内满足条件的数
    

---

# 八、计数 DP（组合类 ⭐⭐⭐）

👉 统计方案数

### 特征

- 用加法
    
- 常配合取模
    

---

# 九、概率 DP（较少 ⭐⭐⭐）

👉 求期望、概率

---

# 🔥总结（最重要）

你可以记这 **6大核心套路**：

|类型|关键词|
|---|---|
|线性DP|子序列|
|背包DP|选/不选|
|区间DP|切一刀|
|网格DP|上/左|
|树形DP|DFS|
|状压DP|二进制|

---

# 💡你刚刚那道“树等间隔”属于哪类？

👉 属于：

**线性 DP + 枚举参数（d）**

本质还是：

```python
dp[i][d] = dp[i-d][d] + 1
```

---

# 🧠学习建议（很关键）

按这个顺序学最舒服：

1. 线性 DP（必须熟）
    
2. 背包（重点）
    
3. 网格 DP
    
4. 区间 DP
    
5. 树形 DP
    
6. 状压 DP
    

---

如果你愿意，我可以给你一份：

👉「每种 DP 必刷 3 题清单（快速进阶版）」




# 动态规划分类

背包问题（Knapsack Problem）是动态规划（DP）中最经典、最必须掌握的领域。在算法竞赛（如蓝桥杯）中，它通常不会直接考原题，但会作为核心逻辑嵌套在复杂题意中。

根据物品的**数量限制**，我们主要将其分为三类。

---

## 1. 0/1 背包（最基础）

**规则**：每个物品只有**一件**，要么装（1），要么不装（0）。

- **状态转移方程**：
    
    $$dp[j] = \max(dp[j], dp[j - weight[i]] + value[i])$$
    

### Python 实现区别：

为了节省空间，我们通常使用**一维数组**。关键点在于**重量循环必须“逆序”**。

- **原因**：逆序是为了保证在计算 $dp[j]$ 时，$dp[j - weight[i]]$ 保存的是“上一层”的状态（即还没拿这个物品时的状态），防止同一个物品被拿多次。
    

Python

```
# 核心代码片段
for i in range(n):  # 遍历物品
    for j in range(capacity, weights[i] - 1, -1):  # 逆序！
        dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
```

---

## 2. 完全背包（Unlimited）

**规则**：每个物品有**无限件**，只要背包装得下，你可以一直拿同一种。

### Python 实现区别：

代码和 0/1 背包几乎一模一样，但**重量循环必须“正序”**。

- **原因**：正序意味着当你在计算 $dp[j]$ 时，用到的 $dp[j - weight[i]]$ 可能是已经拿过该物品后的状态。这种“覆盖”正好模拟了无限次选取的逻辑。
    

Python

```
# 核心代码片段
for i in range(n):
    for j in range(weights[i], capacity + 1):  # 正序！
        dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
```

---

## 3. 多重背包（Bounded）

**规则**：每个物品有**限定的数量** $s_i$。

### Python 实现区别：

多重背包有三种处理方式，性能差异很大：

1. **暴力拆分**：把 $s_i$ 个物品看成 $s_i$ 个独立的 0/1 背包物品。代码最简单，但容易超时。
    
2. **二进制优化（常用）**：将 $s_i$ 拆分成 $1, 2, 4, \dots$ 等 2 的幂次之和。这样能把 $O(N \cdot S \cdot V)$ 降到 $O(N \cdot \log S \cdot V)$。
    
3. **单调队列优化**：最难，能达到 $O(NV)$，蓝桥杯国赛级别。
    

**二进制优化代码逻辑：**

Python

```
# 将 s 个物品拆分存入新的列表
new_weights, new_values = [], []
for i in range(n):
    num = s[i]
    k = 1
    while num >= k:
        new_weights.append(k * weights[i])
        new_values.append(k * values[i])
        num -= k
        k *= 2
    if num > 0:
        new_weights.append(num * weights[i])
        new_values.append(num * values[i])
# 然后对 new_weights 跑一次 0/1 背包
```

---

## 总结与对比

|**类型**|**物品数量**|**核心逻辑（一维 DP）**|**时间复杂度**|
|---|---|---|---|
|**0/1 背包**|1个|`j` 从 `V` 到 `w[i]` (**逆序**)|$O(NV)$|
|**完全背包**|无限|`j` 从 `w[i]` 到 `V` (**正序**)|$O(NV)$|
|**多重背包**|$s_i$ 个|二进制拆分后转 0/1 背包|$O(NV \log S)$|

---

### 给你的刷题建议：

1. **先练 0/1 背包的二维和一维实现**，理解为什么一维要逆序（这是面试和笔试的高频考点）。
    
2. **注意初始化**：
    
    - 如果要求“**恰好装满**”：`dp[0] = 0`, 其余设为 `-inf`。
        
    - 如果要求“**不超过容量**”：全部设为 `0`。
        

**你想让我带你用一个具体的蓝桥杯题目（比如“采药”或“小明的背包”）来实战演练一下 0/1 背包的代码吗？**



# dp问题
数字三角形20
```
import os

import sys

  

n = int(input())

a=[]

for _ in range(n):

  a.append(list(map(int,input().split())))

  
  

#动态规划

dp=[[0]*n for _ in range(n)]

dp[0][0]=a[0][0]

  

for i in range(1,n):

  for j in range(i+1):   #运行出一个三角形就可以 ！！！！

    #如果在左侧，只能从右上下来

    if j==0:  

      dp[i][j]=dp[i-1][j]+a[i][j]

  

    #如果在右侧，只能从左上下来

    elif j==i:

      dp[i][j]=dp[i-1][j-1]+a[i][j]

  

    else:

      dp[i][j]=max(dp[i-1][j-1],dp[i-1][j])+a[i][j]  #报错索引错误

  
  

#判断终点位置   ！！！！！！！！！！！！

#如果n是奇数，一定在正中间，

if n%2==1:

  print(dp[n-1][(n-1)//2])

else:

  print(max(dp[n-1][n//2-1],dp[n-1][n//2]))
```

## dp 倒2 平面切割
```
import os
import sys

  

# 请在此输入您的代码

n = int(input())

a = [list(map(int, input().split())) for _ in range(n)]

  

dp = [[0] * n for _ in range(n)]

dp[0][0] = a[0][0]

  

for i in range(1, n):

    for j in range(i + 1):

        if j == 0:

            dp[i][j] = dp[i - 1][j] + a[i][j]

        elif j == i:

            dp[i][j] = dp[i - 1][j - 1] + a[i][j]

        else:

            dp[i][j] = max(dp[i - 1][j - 1], dp[i - 1][j]) + a[i][j]

  

# 根据左右步数差不超过1来确定终点

if n % 2 == 1:

    # 终点唯一

    print(dp[n - 1][(n - 1) // 2])

else:

    # 终点有两个

    print(max(dp[n - 1][n // 2 - 1], dp[n - 1][n // 2]))

```


## 25园艺题

```
import os

import sys

  

n = int(input())

# 为了方便索引对齐，前面加个0，nums[1]就是第一个数

nums = [0] + list(map(int, input().split()))

# 重要修改 1：使用字典数组。f[i] 是个字典，记录以第 i 个数结尾的所有可能公差。

# 这样不仅省内存，还能直接通过公差找到前一项。

f = [{} for _ in range(n + 1)]

  

ans = 1

# 从第1个数开始遍历

for i in range(1, n + 1):

    # 遍历 i 之前的所有数字 j

    for j in range(1, i):

        # 重要修改 2：计算数值公差（注意：你原来的 d 是下标差，这里要数值差）

        d = nums[i] - nums[j]

        # 如果 nums[j] 那里已经存在公差为 d 的序列，就在它基础上加1

        # 如果没有，说明 i 和 j 第一次组成公差为 d 的序列，长度为 2

        f[i][d] = f[j].get(d, 1) + 1

        # 更新最大值

        if f[i][d] > ans:

            ans = f[i][d]

  

print(ans)
```

分解因子：
