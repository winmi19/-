
Python 的 **random** 模块是标准库中用于生成**伪随机数**的模块。它基于 Mersenne Twister 算法，提供各种随机数生成函数，包括均匀分布、各种统计分布、序列随机操作等。

以下是 **random** 模块中所有公共函数和类的用法介绍（基于 Python 3.12+ 版本）。我将它们分类，便于理解。每项包括函数签名、功能说明和简单示例。

### 1. 基本随机数生成
这些函数生成基本的随机浮点数或整数。

- **random.random()**  
  返回 `[0.0, 1.0)` 范围内的浮点数（均匀分布）。  
  示例：  
  ```python
  import random
  print(random.random())  # 如 0.764761
  ```

- **random.uniform(a, b)**  
  返回 `[a, b]` 范围内的浮点数（均匀分布）。  
  示例：  
  ```python
  print(random.uniform(1, 10))  # 如 7.342
  ```

- **random.randint(a, b)**  
  返回` [a, b]` 范围内的随机整数（包含 a 和 b）。  
  示例：  
  ```python
  print(random.randint(1, 10))  # 如 7
  ```

- **random.randrange(start, stop=None, step=1)**  
  从 range(start, stop, step) 中返回随机整数（不包含 stop）。 ` [a,b)`
  示例：  
  ```python
  print(random.randrange(0, 101, 5))  # 0~100 的 5 的倍数
  ```

- **random.getrandbits(k)**  
  返回具有 k 位随机位的整数（用于生成大随机数）。  
  示例：  
  ```python
  print(random.getrandbits(32))  # 如 2147483647
  ```

- **random.randbytes(n)** (Python 3.9+)  
  返回 n 个随机字节（bytes 类型，常用于加密）。  
  示例：  
  ```python
  print(random.randbytes(8))  # 如 b'\x1f\x8b...'
  ```

### 2. 序列随机操作
这些函数用于从列表、序列中随机选取或打乱。

- **random.choice(seq)**  
  从**非空序列 seq** 中随机返回一个元素。  
  示例：  
  ```python
  fruits = ['apple', 'banana', 'orange']
  
  print(random.choice(fruits))  # 如 'banana'
  ```

- **random.choices(population, weights=None, k=1)**  
  从 population 中有放回地抽取 k 个元素（可加权重）。  (可重复)
  示例：  
  ```python
  print(random.choices(['red', 'blue', 'green'], k=5))
  # 如 ['blue', 'red', 'green', 'red', 'blue']
  ```

- **random.sample(population, k)**  
  从 population 中无放回地抽取 k 个唯一元素。  
  示例：  
  ```python
  print(random.sample(range(100), k=5))  # 如 [45, 23, 67, 89, 12]
  ```

- **random.shuffle(x)**  
  原地打乱序列 x（修改原列表）。  
  示例：  
  ```python
  lst = [1, 2, 3, 4, 5]
  random.shuffle(lst)
  print(lst)  # 如 [3, 1, 5, 2, 4]
  ```

### 3. 统计分布函数
这些生成符合特定概率分布的随机数。

- **random.triangular(low, high, mode)**  
  三角分布（默认 mode 为中间值）。  
  示例：  
  ```python
  print(random.triangular(0, 10, 5))  # 倾向于 5 附近
  ```

- **random.normalvariate(mu, sigma)**  
  正态（高斯）分布，均值 mu，标准差 sigma。  
  示例：  
  ```python
  print(random.normalvariate(0, 1))  # 标准正态
  ```

- **random.gauss(mu, sigma)**  
  与 normalvariate 类似，但更快（线程不安全）。  

- **random.lognormvariate(mu, sigma)**  
  对数正态分布。  

- **random.expovariate(lambd)**  
  指数分布（lambd 为速率）。  

- **random.gammavariate(alpha, beta)**  
  Gamma 分布。  

- **random.betavariate(alpha, beta)**  
  Beta 分布。  

- **random.paretovariate(alpha)**  
  Pareto 分布。  

- **random.weibullvariate(alpha, beta)**  
  Weibull 分布。  

- **random.vonmisesvariate(mu, kappa)**  
  von Mises 分布（圆形分布）。  

- **random.binomialvariate(n, p)** (Python 3.12+)  
  二项分布（n 次试验，每次成功概率 p）。  

### 4. 随机数生成器状态管理
用于控制随机种子和状态（可复现随机序列）。

- **random.seed(a=None, version=2)**  
  初始化随机种子（a 可为 int、str、bytes 等）。  
  示例：  
  ```python
  random.seed(123)  # 固定种子，后续随机数可复现
  ```

- **random.getstate()**  
  返回当前随机生成器状态（可保存）。  

- **random.setstate(state)**  
  恢复之前保存的状态。  

### 5. 类
- **random.Random([seed])**  
  创建独立的随机生成器实例（不影响全局 random）。  
  用法示例：  
  ```python
  rng = random.Random(123)  # 独立实例
  print(rng.random())
  ```

- **random.SystemRandom**  
  使用操作系统随机源（更安全，适用于加密），用法类似 Random。  
  示例：  
  ```python
  sr = random.SystemRandom()
  print(sr.randint(1, 100))
  ```



### 注意事项
- 大多数模块级函数（如 random.random()）共享同一个全局生成器实例。
- 如需多个独立随机序列，使用 random.Random() 创建实例。
- 随机数是**伪随机**的（可复现），不适合高安全加密场景（用 SystemRandom 或 secrets 模块）。
- 导入方式：`import random`

如果需要某个函数的更详细示例或源码解释，随时告诉我！