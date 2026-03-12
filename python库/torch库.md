### PyTorch（torch）库详尽介绍：深度学习常用用法

**PyTorch**（通常导入为 `torch`）是目前最流行的开源深度学习框架之一，由 Meta（Facebook）维护。它以**动态计算图**（eager execution）为核心，代码直观、调试友好，非常适合研究和快速原型开发。相比 TensorFlow，PyTorch 在学术界和工业界（尤其是研究型项目）占有率更高。

PyTorch 的核心是 **Tensor**（张量），它是 NumPy ndarray 的超集，支持 GPU 加速和自动求导（autograd）。深度学习中，几乎所有操作都围绕 Tensor 展开。

#### 基本导入
```python
import torch
import torch.nn as nn          # 神经网络模块
import torch.optim as optim    # 优化器
import torch.nn.functional as F  # 函数式接口（常用激活、损失）
from torch.utils.data import DataLoader, TensorDataset  # 数据加载
```

#### 1. Tensor：核心数据结构
Tensor 是 PyTorch 的基本单元，类似 NumPy 数组，但支持：
- GPU 加速
- 自动求导（记录运算历史）

```python
# 创建 Tensor
x = torch.tensor([1, 2, 3])                    # 从列表
x = torch.zeros(2, 3)                         # 全零，形状 (2,3)
x = torch.ones(3, 3)
x = torch.full((2, 4), 7.5)                    # 填充值
x = torch.randn(3, 3)                         # 标准正态随机
x = torch.arange(0, 10, 2)                     # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5)                   # 等间隔

# 从 NumPy 创建（共享内存，注意修改会相互影响）
import numpy as np
np_arr = np.array([[1, 2], [3, 4]])
x = torch.from_numpy(np_arr)                  # 或 torch.tensor(np_arr)（复制）

# 指定设备和类型
x = torch.randn(3, 3, device='cuda')          # GPU（需有 CUDA）
x = torch.randn(3, 3, dtype=torch.float16)    # 半精度
```

**常用属性**
```python
x.shape          # 或 x.size() → torch.Size([3, 3])
x.dtype          # 数据类型
x.device         # 设备 'cpu' 或 'cuda:0'
x.ndim           # 维度数
x.numel()        # 元素总数
```

#### 2. Tensor 操作（索引、切片、数学运算）
与 NumPy 高度类似，支持广播。

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8, 9], [10, 11, 12]])

a + b            # 元素加
a * b            # 元素乘
a @ b.t()        # 矩阵乘法（或 torch.matmul(a, b.T)）
torch.mm(a, b.t())  # 同上

# 广播
a = torch.randn(2, 3)
b = torch.randn(3)       # 自动扩展到每行加 b
a + b

# 索引与切片
a[0, 1]                  # 第0行第1列
a[:, 1]                  # 第1列
a[0:2, 1:3]              # 子张量
a[a > 0]                 # 布尔索引

# 形状操作（不复制数据）
a.view(3, 2)             # reshape，需元素数匹配
a.reshape(3, 2)          # 同上，更灵活
a.permute(1, 0)          # 交换维度（多维常用）
a.transpose(0, 1)        # 交换两个维度
a.flatten()              # 展平
torch.cat([a, b], dim=0) # 拼接
torch.stack([a, b], dim=0) # 新维度堆叠
```

**常用数学函数**（向量化）
```python
torch.sqrt(a),    # 开平方
torch.exp(a),     # e指数
 torch.log(a)     # 对数
torch.sin(a),     # sin函数
 torch.cos(a)     # cos函数
torch.mean(a),    # 平方
torch.std(a),     #标准差
torch.var(a)      #方差
a.sum(dim=1),     # 相加
a.max(dim=0)   # 返回值和索引：values, indices = a.max(dim=0)
```

#### 3. 与 NumPy 互操作
```python
# Tensor → NumPy（在 CPU 上）
np_arr = x.cpu().numpy()

# NumPy → Tensor
x = torch.from_numpy(np_arr)
```

#### 4. 自动求导（autograd）——深度学习核心
PyTorch 的杀手级特性：动态计算图，自动计算梯度。

```python
x = torch.tensor(2.0, requires_grad=True)   # 标记需要梯度
y = x ** 2 + 3 * x
y.backward()                                # 反向传播
x.grad                                      # → tensor(7.)  (dy/dx = 2x + 3)
```

在神经网络中，所有参数（nn.Parameter）默认 `requires_grad=True`，训练时自动积累梯度。

```python
# 清零梯度（每步训练必须）
optimizer.zero_grad()
loss.backward()
optimizer.step()        # 更新参数
```

#### 5. torch.nn：构建神经网络（深度学习最常用）
之前介绍过，这里扩展常见层和用法。

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # 卷积
        self.bn1 = nn.BatchNorm2d(64)                            # 批归一化
        self.fc1 = nn.Linear(64 * 28 * 28, 512)                  # 全连接
        self.dropout = nn.Dropout(0.5)                           # Dropout
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # 常用 F.relu（函数式）
        x = x.view(x.size(0), -1)             # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.out(x)
        return x                                  # 或 return F.log_softmax(x, dim=1)

model = Net()
model.to('cuda')                                  # 移到 GPU
```

**常见层**
- 卷积：`nn.Conv1d/2d/3d`
- 池化：`nn.MaxPool2d`, `nn.AvgPool2d`
- 归一化：`nn.BatchNorm2d`, `nn.LayerNorm`
- 循环网络：`nn.LSTM`, `nn.GRU`
- Transformer：`nn.Transformer`, `nn.MultiheadAttention`
- 激活：`nn.ReLU`, `nn.LeakyReLU`, `nn.GELU`（Transformer 常用）
- 损失：`nn.CrossEntropyLoss`（分类，包含 softmax），`nn.MSELoss`, `nn.L1Loss`

**容器**
- `nn.Sequential`：快速串联层
- `nn.ModuleList` / `nn.ModuleDict`：动态层列表/字典

#### 6. 数据加载（torch.utils.data）——深度学习必备
```python
from torchvision import datasets, transforms   # torchvision 常用数据集/变换

transform = transforms.Compose([
    transforms.ToTensor(),                     # 转 Tensor，归一化到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 均值/标准差
])

train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# 自定义 Dataset
class MyDataset(TensorDataset):
    # 继承 TensorDataset 或 Dataset
    pass
```

#### 7. 优化器与训练循环（深度学习标准流程）
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam 最常用
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度

for epoch in range(20):
    model.train()                                      # 训练模式（启用 Dropout/BN）
    for data, target in train_loader:
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)         # 或 nn.CrossEntropyLoss()
        loss.backward()
        optimizer.step()
    
    scheduler.step()                                   # 更新学习率
    
    # 验证阶段
    model.eval()                                       # 评估模式
    with torch.no_grad():                              # 关闭梯度
        # 计算准确率...
        pass
```

#### 8. GPU 与多卡支持
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data = data.to(device)

# 多卡 DataParallel（简单）或 DistributedDataParallel（高级）
model = nn.DataParallel(model)
```

#### 9. 其他深度学习常用子库
- **torchvision**：图像数据集（MNIST, CIFAR, ImageNet）、模型（ResNet, ViT）、变换。
- **torchaudio**：音频处理。
- **torchtext**：NLP 数据集、词嵌入。
- **torch.nn.functional**：无参数函数（如 F.softmax, F.interpolate 上采样）。
- **torch.save / torch.load**：保存/加载模型（推荐保存 state_dict）。

#### 小贴士（深度学习实战）
- **调试**：用 `print(x.shape)`、`torch.summary`（需 pip install torchsummary）查看模型结构。
- **加速**：用 `torch.compile(model)`（PyTorch 2.0+）加速训练。
- **混合精度**：`torch.cuda.amp` 自动混合精度训练，节省显存。
- **常见错误**：忘记 `model.train()/eval()` 切换、梯度未清零、设备不一致。

PyTorch 的优势在于**灵活**：你可以随时插入 Python 代码调试计算图。掌握 Tensor 操作、autograd、nn.Module、DataLoader、训练循环，就能构建 90% 的深度学习模型（CNN、Transformer、Diffusion 等）。

如果你想看某个具体模型（如 ResNet、BERT）的实现例子，或者某个功能的代码演示，再告诉我！