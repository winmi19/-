

# 架构简图
![架构图](C:\Users\HP\Desktop\vjepa_struc.png)

放大如下：

![[Pasted image 20260120172206.png]]
![[Pasted image 20260120172226.png]]



---


# 结构介绍

## 1. 核心目录

### `app/` (训练应用层)

这是所有预训练循环（Training Loops）的入口。

- **`main.py`**: 单机/本地调试用的入口脚本。用于在本地机器上启动预训练。
    
- **`main_distributed.py`**: 分布式训练入口，通常用于 Slurm 集群环境。
    
- **`vjepa/`**: 包含 V-JEPA 具体的预训练逻辑代码（例如 `train.py` 就在这里，虽然你在上传的文件列表中直接给出了 `train.py`，但在架构中它通常位于此处）。
    

### `evals/` (评估应用层)

用于评估训练好的模型，不允许在此处进行模型的主干（Backbone）训练，通常是冻结主干训练探测头（Probe）。

- **`image_classification/`**: 训练注意力探测头（Attentive Probe）进行图像分类。
    
- **`video_classification/`**: 训练注意力探测头进行视频分类。
    
- **`main.py` / `main_distributed.py`**: 评估任务的启动脚本。
    

### `src/` (核心代码包)

包含模型定义、数据处理和工具函数。

- **`datasets/`**: 数据集定义、DataLoader 实现。
    
- **`models/`**: V-JEPA 的模型架构定义（Vision Transformers 等）。
    
- **`masks/`**: 掩码生成策略（如 `TubeMaskCollator`，`Multiblock3d`），这是 JEPA 类方法的核心。
    
- **`utils/`**: 通用工具，如分布式设置、日志记录、张量操作等。
    

### `configs/` (配置文件)

- **`pretrain/`**: 预训练的 YAML 配置文件（定义学习率、Mask 策略、数据路径等）。
    
- **`evals/`**: 评估任务的 YAML 配置文件。
    

## 2. 关键文件详情

- **`README.md`**: 项目的官方说明文档，包含安装、数据准备和运行命令。
    
- **`main.py` (App)**: 负责解析参数、初始化进程组、读取配置，并将执行权交给具体的应用逻辑（`app.scaffold`）。
    
- **`train.py`**: 核心训练循环。实现了 V-JEPA 的具体算法：加载数据 -> 前向传播（Target/Context Encoder） -> 计算损失 -> 反向传播 -> 更新 EMA 模型。



---

# main.py解析

```python
#导入参数解析库，用于处理命令行参数

import argparse


#导入多进程库，用于在单机上模拟分布式训练（每个 GPU 一个进程）

import multiprocessing as mp


#导入美化打印库，用于打印配置信息

import pprint


#导入 yaml 库，用于读取 .yaml 配置文件

import yaml

  

#从 app.scaffold 模块导入 main 函数并重命名为 app_main“

#scaffold 通常是训练流程的脚手架，负责连接配置和具体的训练函数（如 train.py 中的 main）”

from app.scaffold import main as app_main

#从 src.utils.distributed 导入初始化分布式的工具”

from src.utils.distributed import init_distributed

  

"创建参数解析器“

parser = argparse.ArgumentParser()

"添加 --fname 参数：指定要加载的配置文件路径，默认为 'configs.yaml'”

parser.add_argument(

    '--fname', type=str,

    help='name of config file to load',

    default='configs.yaml')

#添加 --devices 参数：指定在本地机器上使用哪些 GPU 设备，默认为 ['cuda:0']

#nargs='+' 表示可以接受多个值，例如 --devices cuda:0 cuda:1

parser.add_argument(

    '--devices', type=str, nargs='+', default=['cuda:0'],

    help='which devices to use on local machine')

  
  

"""定义每个进程的具体执行函数

rank: 当前进程的序号（0, 1, 2...）

fname: 配置文件路径

world_size: 总进程数（总 GPU 数）

devices: 设备列表"""


def process_main(rank, fname, world_size, devices):

    import os

    # 设置当前进程可见的 CUDA 设备。

    # 例如，如果 devices=['cuda:0', 'cuda:1']，且 rank=1，则设置为 '1'。

    # 这样在 PyTorch 内部，该进程会认为自己只有这一块 GPU（索引为 0）。

    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

  

    import logging

    from src.utils.logging import get_logger

    # 获取日志记录器，force=True 确保重新初始化

    logger = get_logger(force=True)

    # 只有 rank 0 (主进程) 打印 INFO 级别的日志，其他进程只打印 ERROR

    # 这样可以避免多进程训练时控制台输出混乱

    if rank == 0:

        logger.setLevel(logging.INFO)

    else:

        logger.setLevel(logging.ERROR)

  

    # 记录当前加载的配置文件名

    logger.info(f'called-params {fname}')

  

    # 加载 YAML 配置文件

    params = None

    with open(fname, 'r') as y_file:

        params = yaml.load(y_file, Loader=yaml.FullLoader)

        logger.info('loaded params...')

  

    # 如果是主进程，打印完整的配置参数，并将参数备份到日志目录中

    if rank == 0:

        pprint.PrettyPrinter(indent=4).pprint(params)

        # 将配置参数 dump 到日志文件夹下的 params-pretrain.yaml，方便后续复现

        dump = os.path.join(params['logging']['folder'], 'params-pretrain.yaml')

        with open(dump, 'w') as f:

            yaml.dump(params, f)

  

    # 初始化分布式环境 (Distributed Data Parallel, DDP)

    # 这会设置 master_addr, master_port 等，让不同进程可以通信

    # rank_and_world_size 参数手动指定了当前的 rank 和总数

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))

    logger.info(f'Running... (rank: {rank}/{world_size})')

  

    # 启动应用主逻辑

    # app_main 通常会根据 params['app'] 里的配置去调用真正的 train.py

    app_main(params['app'], args=params)

  
  

"程序入口点"

if __name__ == '__main__':

    # 解析命令行参数

    args = parser.parse_args()

    # 获取 GPU 数量

    num_gpus = len(args.devices)

    # 设置多进程启动方式为 'spawn'

    # spawn 是 CUDA 兼容性最好的方式，可以避免 fork 带来的一些死锁或上下文问题

    mp.set_start_method('spawn')

    # 循环为每个 GPU 启动一个进程

    for rank in range(num_gpus):

        mp.Process(

            target=process_main, # 目标函数

            args=(rank, args.fname, num_gpus, args.devices) # 传递参数

        ).start() # 启动进程
```




# 训练流程
![流程图](D:\google_gemini3\deepseek_mermaid_20260120_ca24d3.png)



---


# train.py解析


本文档对 V-JEPA 的训练核心脚本 `train.py` 进行逐行拆解。为了方便理解，代码被分为多个逻辑模块。每个模块后附有详细的中文解释及**外部调用说明**（标注调用的文件及其功能）。

## 1. 导入与环境配置

```
import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # 尝试设置当前进程可见的 CUDA 设备 ID。
    # 在分布式训练（如 Slurm）中，通常每个进程只应看到一个 GPU。
    # 这里使用环境变量 SLURM_LOCALID 来指定。
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

# --- 核心模块导入 ---
from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch

from app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
)
from app.vjepa.transforms import make_transforms
```

### 🧩 外部调用档案 (External Function Calls)

|                    |                                |                                                                        |
| ------------------ | ------------------------------ | ---------------------------------------------------------------------- |
| **函数/类名**          | **来源文件 (Source)**              | **功能说明**                                                               |
| `init_data`        | `src/datasets/data_manager.py` | 初始化数据集和 DataLoader，负责视频数据的读取和预处理。                                      |
| `TubeMaskCollator` | `src/masks/random_tube.py`     | 生成“管状”掩码（在时间轴上一致的掩码），用于简单的掩码策略。                                        |
| `MB3DMaskCollator` | `src/masks/multiblock3d.py`    | **核心组件**。生成多块 3D 掩码（Multi-Block Masking），这是 V-JEPA 论文中使用的关键掩码策略。       |
| `apply_masks`      | `src/masks/utils.py`           | 工具函数，用于将生成的掩码应用到 Transformer 的输出特征上，提取需要预测的目标区域。                       |
| `init_distributed` | `src/utils/distributed.py`     | 初始化 PyTorch 的分布式后端（NCCL），设置进程组。                                        |
| `init_video_model` | `app/vjepa/utils.py`           | **核心组件**。构建并初始化 Vision Transformer (ViT) 的编码器（Encoder）和预测器（Predictor）。 |
| `make_transforms`  | `app/vjepa/transforms.py`      | 构建视频数据的数据增强流水线（裁剪、翻转、颜色抖动等）。                                           |

## 2. 全局设置与日志

```
# -- 硬编码的日志配置
log_timings = True       # 是否记录时间消耗
log_freq = 10            # 日志打印频率（每多少个 iteration）
checkpoint_freq = 1      # 检查点保存频率（每多少个 epoch）
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)          # 设置 numpy 随机种子
torch.manual_seed(_GLOBAL_SEED)       # 设置 torch 随机种子
torch.backends.cudnn.benchmark = True # 开启 cudnn benchmark 以加速固定尺寸输入的训练

logger = get_logger(__name__)         # 获取当前模块的日志记录器
```

## 3. 主函数：参数解析

```
def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  从 CONFIG 文件传入的参数 (PASSED IN PARAMS)
    # ----------------------------------------------------------------------- #
    
    # -- META (元数据配置)
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt # 是否加载检查点
    r_file = cfgs_meta.get('read_checkpoint', None)                 # 指定读取的检查点文件
    # ... (省略部分简单参数获取代码) ...
    
    # 确定混合精度训练的数据类型 (BF16 或 FP16)
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK (掩码配置)
    cfgs_mask = args.get('mask')


    # -- MODEL (模型配置)
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')           # 例如 'vit_h_16'
    pred_depth = cfgs_model.get('pred_depth')           # 预测器网络的深度
    pred_embed_dim = cfgs_model.get('pred_embed_dim')   # 预测器的嵌入维度
    # ...


    # -- DATA (数据配置)
    cfgs_data = args.get('data')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d') # 掩码类型，默认多块3D
    # ... (获取 batch_size, num_frames 等) ...



    # -- LOSS (损失函数配置)
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')   # 损失函数的指数 (例如 L1 为 1, L2 为 2)
    reg_coeff = cfgs_loss.get('reg_coeff') # 正则化系数

    # -- OPTIMIZATION (优化器配置)
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)      # Iterations Per Epoch (每个 epoch 的迭代次数)
    # ... (获取学习率 lr, weight_decay 等) ...
```

### 💡 解读

这部分代码主要是从 `args` 字典（由 YAML 配置文件解析而来）中提取各种超参数。

- **关键参数**:
    
    - `mixed_precision`: 是否使用混合精度训练（加速显存利用）。
        
    - `pred_depth`: V-JEPA 中预测器（Predictor）是一个轻量级的 Transformer，这里定义其层数。
        
    - `mask_type`: 决定了 V-JEPA 学习多大难度的任务（遮挡多少、怎么遮挡）。
        

## 4. 初始化核心组件

```
    # -- 初始化分布式后端
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- 设置当前设备
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0') # 由于最开始设置了 CUDA_VISIBLE_DEVICES，这里总是用 0
        torch.cuda.set_device(device)

    # -- 路径配置 (略) ...
    
    # -- 初始化 CSV 日志记录器
    csv_logger = CSVLogger(...) 

    # -- 初始化模型 (核心)
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        # ... 传入大量模型参数
        device=device,
        model_name=model_name,
        # ...
    )
    # 深拷贝编码器作为目标编码器 (Teacher)
    target_encoder = copy.deepcopy(encoder)

    # -- 制作数据增强 Transform
    # 根据掩码类型选择 Collator (整理器)
    if mask_type == 'multiblock3d':
        logger.info('Initializing basic multi-block mask')
        mask_collator = MB3DMaskCollator(...) # 使用多块3D掩码策略
    else:
        logger.info('Initializing random tube mask')
        mask_collator = TubeMaskCollator(...)
        
    transform = make_transforms(...) # 创建数据增强

    # -- 初始化数据加载器
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         data=dataset_type,
         root_path=dataset_paths,
         collator=mask_collator, # 将掩码生成器传入 DataLoader
         # ...
         world_size=world_size,
         rank=rank,
         # ...
    )

    # -- 初始化优化器
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        # ...
    )
    
    # -- 包装为 DDP (分布式并行模型)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    
    # 冻结目标编码器 (Teacher) 的梯度，它不通过反向传播更新
    for p in target_encoder.parameters():
        p.requires_grad = False
```

### 🧩 外部调用档案

- **`init_video_model`**: 调用 `app/vjepa/utils.py`。这是构建 ViT 架构的地方。注意它返回了两个模型：`encoder` (Context Encoder) 和 `predictor` (Predictor)。
    
- **`target_encoder = copy.deepcopy(encoder)`**: V-JEPA 是一种基于动量蒸馏（Momentum Distillation）的方法，所以需要一个 Teacher 模型，初始化时它是 Student 的完全拷贝。
    
- **`MB3DMaskCollator`**: 调用 `src/masks/multiblock3d.py`。这个类非常重要，它在数据加载阶段动态生成掩码，决定了模型在训练中要预测哪些部分。
    

## 5. 训练主循环

```
    # -- TRAINING LOOP (训练循环)
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # 更新 Sampler 的 epoch，确保每个 epoch 数据的随机顺序不同
        unsupervised_sampler.set_epoch(epoch)

        # ... (重置各种统计 Meter) ...

        for itr in range(ipe):
            # 1. 获取数据
            try:
                # udata: 视频数据, masks_enc: 编码器掩码, masks_pred: 预测器掩码
                udata, masks_enc, masks_pred = next(loader)
            except Exception:
                # 如果 DataLoader 耗尽，重新创建迭代器
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred = next(loader)

            # 2. 数据搬运到 GPU 的辅助函数
            def load_clips():
                # 将 batch 中的 clips 拼接并移至 GPU
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
                # ... (处理掩码并移至 GPU) ...
                return (clips, _masks_enc, _masks_pred)
            clips, masks_enc, masks_pred = load_clips()

            # 3. 定义单步训练逻辑 (核心算法)
            def train_step():
                # 更新学习率和权重衰减
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # --- 核心函数 A: 目标编码器前向传播 (Teacher) ---
                def forward_target(c):
                    with torch.no_grad(): # 目标网络不计算梯度
                        h = target_encoder(c)
                        h = F.layer_norm(h, (h.size(-1),))
                        # 仅保留预测器需要预测的那些区域的特征 (masks_pred)
                        # 调用了 src.masks.utils.apply_masks
                        h = apply_masks(h, masks_pred, concat=False) 
                        return h

                # --- 核心函数 B: 上下文编码器 + 预测器前向传播 (Student) ---
                def forward_context(c, h):
                    # 编码器仅处理可见区域 (masks_enc)
                    z = encoder(c, masks_enc)
                    # 预测器基于编码特征 z 和位置掩码，尝试预测被遮挡的特征
                    z = predictor(z, h, masks_enc, masks_pred)
                    return z

                # --- 核心函数 C: 损失计算 ---
                def loss_fn(z, h):
                    loss = 0.
                    for zi, hi in zip(z, h):
                        # 计算预测特征 zi 和目标特征 hi 之间的距离 (L1 或 L2)
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss /= len(masks_pred)
                    return loss

                # --- 核心函数 D: 正则化 ---
                def reg_fn(z):
                    # 计算 batch 内特征的方差，鼓励特征多样性，防止模型坍塌
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # --- Step 3.1: 前向计算 ---
                loss_jepa, loss_reg = 0., 0.
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)      # 计算目标特征
                    z = forward_context(clips, h)  # 计算预测特征
                    loss_jepa = loss_fn(z, h)      # 预测损失
                    
                    # 正则化损失：希望预测的特征方差不要太小
                    pstd_z = reg_fn(z)
                    loss_reg += torch.mean(F.relu(1.-pstd_z))
                
                # 总损失 = JEPA 损失 + 正则系数 * 正则损失
                loss = loss_jepa + reg_coeff * loss_reg

                # --- Step 3.2: 反向传播与优化 ---
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer) # 反缩放以便梯度裁剪
                else:
                    loss.backward()
                
                # 梯度裁剪 (Clip Grad)
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                
                # 优化器更新参数
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad() # 清空梯度

                # --- Step 3.3: EMA 动量更新 (关键) ---
                # 将 Encoder 的参数以动量 m 更新给 Target Encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        # param_k = m * param_k + (1-m) * param_q
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (...) # 返回统计信息

            # 4. 执行训练步并计时
            (loss, ...), gpu_etime_ms = gpu_timer(train_step)

            # ... (日志记录与打印，略) ...
        
        # 5. 保存检查点
        if epoch % checkpoint_freq == 0:
            save_checkpoint(epoch + 1, latest_path)
```

### 💡 深度逻辑解读

1. **Teacher-Student 结构**:
    
    - `target_encoder` (Teacher) 提供回归目标。
        
    - `encoder` + `predictor` (Student) 尝试预测 Teacher 在被遮挡区域的输出。
        
    - Teacher 的参数不是通过梯度下降更新的，而是通过 EMA (指数移动平均) 缓慢跟随 Student。这保证了目标的稳定性，避免“模型坍塌”（即模型输出全为0或常数来“作弊”降低 loss）。
        
2. **掩码策略**:
    
    - `masks_enc`: 传给 `encoder`。这是模型**看得见**的部分。
        
    - `masks_pred`: 传给 `apply_masks`（用于 Target）和 `predictor`。这是模型**看不见**但需要**预测**的部分。
        
3. **预测器 (Predictor)**:
    
    - 它的输入不仅仅是 `encoder` 的输出，还包括 `masks_pred` (Position Embeddings)。它需要根据 Encoder 提供的上下文信息，推断出指定位置的内容。
        

### 🧩 外部调用档案

- **`apply_masks`**: 位于 `src/masks/utils.py`。注意它在 `forward_target` 中被调用。因为 Teacher 看到的是全图，但我们只计算被遮挡部分的 Loss，所以用这个函数把对应位置的特征抠出来。
    
- **`gpu_timer`**: 位于 `src/utils/logging.py`。用于精确测量 GPU 执行 `train_step` 的时间，用于性能监控。



---



# `main -> scaffold -> train` 调用链解释


这种设计**不是多此一举**，而是为了实现**控制反转 (Inversion of Control)** 和 **关注点分离**。这在大型深度学习框架（如 FAIR/Meta 的项目）中非常常见。

- **`main.py` (基础设施层):**
    
    - **职责**: 它只关心“怎么启动”。负责多进程 (Multiprocessing)、GPU 分配、分布式环境初始化 (DDP)、读取 YAML 配置文件。
        
    - **好处**: 它不需要知道你在跑什么模型。无论是跑 V-JEPA、MAE 还是 ResNet，启动分布式环境的代码是一样的。写一次，处处通用。
        
- **`scaffold.py` (路由层/脚手架):**
    
    - **职责**: 它是一个“动态路由”。它根据配置文件里的字符串（比如 `app: vjepa`），动态地去寻找对应的文件夹并加载里面的 `train.py`。
        
    - **好处**: 如果你想加一个新算法（比如 `app/new_model/train.py`），你不需要修改 `main.py` 的一行代码，只需要在配置文件里改个名字，`scaffold` 就会自动帮你找到并加载它。这是一种**依赖注入**的思想。
        
- **`train.py` (业务逻辑层):**
    
    - **职责**: 只关心具体的训练算法（Loss 计算、前向传播）。
        
    - **好处**: 代码纯粹，便于研究人员阅读和修改算法，而不必被复杂的分布式启动代码干扰。
        

**结论**: 如果你只是写一个几百行的小脚本，直接调 `train.py` 更好。但对于这种包含多个实验、多种模型的大型库，这种分层架构极大地降低了维护成本。



---

# scaffold.py解读

`scaffold.py` 是连接底层架构 (`main.py`) 和业务逻辑 (`train.py`) 的桥梁。它的核心作用是**动态导入**。

```
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# ... (版权声明，略) ...

import importlib  # [关键] 导入 Python 的标准库 importlib。
                  # 它的作用是允许代码在"运行时"动态地导入模块，而不是在文件开头写死 import xxx。
import logging
import sys

# 设置日志的基础配置
# stream=sys.stdout: 确保日志输出到标准输出（控制台），这样 Slurm 或命令行能看到。
# level=logging.INFO: 只记录 INFO 及以上级别的日志（忽略调试信息）。
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):
    """
    Args:
        app (str): 从配置文件中读取的应用名称，例如 'vjepa'。
        args (dict): 完整的参数字典。
        resume_preempt (bool): 是否从抢占中恢复。
    """
    
    # 记录一条日志，告诉用户正在启动哪个应用的预训练。
    # 比如：Running pre-training of app: vjepa
    logger.info(f'Running pre-training of app: {app}')

    # [核心逻辑]
    # 1. f'app.{app}.train': 这是一个 f-string。如果 app='vjepa'，这里就变成了字符串 'app.vjepa.train'。
    #    这对应了文件路径 app/vjepa/train.py。
    #
    # 2. importlib.import_module(...): 这行代码等价于你在代码里写 `import app.vjepa.train`。
    #    但它的强大之处在于，它是在程序跑起来之后，根据变量 `app` 的值决定的。
    #    这意味着 main.py 不需要知道 'vjepa' 的存在，它只负责传一个字符串过来。
    #
    # 3. .main(...): 导入模块后，直接调用该模块里的 `main` 函数。
    #    这就是为什么 train.py 里必须有一个 main 函数的原因。
    return importlib.import_module(f'app.{app}.train').main(
        args=args,
        resume_preempt=resume_preempt)
```

## 总结：数据流向

1. **用户**运行 `python main.py --fname configs/vjepa.yaml`。
    
2. **`main.py`** 读取 yaml，发现 `app: vjepa`。
    
3. **`main.py`** 调用 `scaffold.main(app='vjepa', ...)`。
    
4. **`scaffold.py`** 动态执行 `import app.vjepa.train`。
    
5. **`scaffold.py`** 调用 `app.vjepa.train.main(...)`。
    
6. **`train.py`** 开始训练循环。

---


# src/masks/default.py 代码解析

这段代码定义了一个名为 `DefaultCollator` 的类。在 V-JEPA 的数据加载流水线中，它的角色是**“数据打包员”**。

### 1. 核心作用：什么是 Collator？

在 PyTorch 训练模型时，数据读取通常分为三步：

1. **Dataset（仓库）**：负责从硬盘读取单个视频或图像。
    
2. **Sampler（排队）**：决定按什么顺序读取。
    
3. **DataLoader（装箱）**：把读取到的多个单个数据“打包”成一个 **Batch（批次）**。
    

`DefaultCollator` 里的 **Collator** 就是负责“装箱”的逻辑。

### 代码逐行拆解

```
class DefaultCollator(object):
    def __call__(self, batch):
        # 1. 使用 PyTorch 默认的打包工具
        # 它把一个列表的视频张量合并成一个高维张量
        collated_batch = torch.utils.data.default_collate(batch)
        
        # 2. 返回结果
        # V-JEPA 的数据接口通常期待返回三个东西：(数据, 掩码1, 掩码2)
        # 这个“默认打包器”不处理掩码，所以后两个返回 None
        return collated_batch, None, None
```

## 2. 深度解析：`__call__` 是什么？

这是 Python 中的一个“魔法方法”。它的作用是：**让一个类的实例对象可以像函数一样被直接调用。**

### 通俗比喻

普通类就像一个“工具箱”，你得打开箱子拿工具（调用方法）。

实现了 `__call__` 的类就像一个“带按钮的机器”，你不用打开它，直接按一下机器（直接调用对象名字）它就工作。

### 如何调用？

如果没有 `__call__`，你可能需要这样写：

```
collator = DefaultCollator()
result = collator.combine(batch) # 必须指明调用 combine 方法
```

有了 `__call__`，你可以直接这样写：

```
collator = DefaultCollator()
result = collator(batch)        # 像函数一样直接调用对象！
```

### 为什么在 PyTorch 里要这么写？

PyTorch 的 `DataLoader` 有一个参数叫 `collate_fn`（打包函数）。它期待接收一个**函数**。

通过实现 `__call__`，我们可以把 `DefaultCollator` 的**实例**伪装成一个函数传进去，同时又能利用“类”的特性（比如在类里保存一些配置参数）。

## 3. 为什么返回三个值？

你会发现代码最后返回了 `collated_batch, None, None`。

这是为了**对齐接口**。V-JEPA 的核心是“掩码建模（Masking）”。其他的 Collator（比如专门处理遮挡的 `MaskCollator`）会返回：

1. `collated_batch`: 原始视频数据。
    
2. `mask_context`: 告诉模型哪些地方是可见的。
    
3. `mask_target`: 告诉模型哪些地方是要预测的。
    

而这个 `DefaultCollator` 是一个**最简单的备选方案**，它不生成任何掩码信息。为了让程序主逻辑不崩溃（主逻辑可能等着接收 3 个变量），它即便没东西给，也要塞两个 `None` 占位。

## 总结

- **它的身份**：数据打包的保底工具。
    
- **它的功能**：把散装的数据变成整齐的批次。
    
- **`__call__` 的意义**：让这个类可以被 PyTorch 的 DataLoader 直接当成函数使用，简洁又专业。

---
# src/masks/multiblock3d.py 逐行解析

这个文件实现了 V-JEPA 的核心任务：**3D 掩码生成**。它不仅在空间（长宽）上遮挡，还在时间（帧）上遮挡。

## 1. 顶部导入与全局变量

```
import math
from multiprocessing import Value  # 用于多进程间的共享变量
from logging import getLogger
import torch

_GLOBAL_SEED = 0
logger = getLogger()
```

## 2. MaskCollator 类：掩码管理器

这个类是对外暴露的接口，负责管理多个掩码生成策略。

### 初始化 `__init__`

```
class MaskCollator(object):
    def __init__(self, cfgs_mask, crop_size=(224, 224), num_frames=16, patch_size=(16, 16), tubelet_size=2):
        super(MaskCollator, self).__init__()
        self.mask_generators = []
        # 根据配置文件 (yaml)，可能同时存在多种掩码策略（比如大块和小块结合）
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                # 从配置中提取缩放、比例、块数等参数
                spatial_pred_mask_scale=m.get('spatial_scale'),
                temporal_pred_mask_scale=m.get('temporal_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                max_keep=m.get('max_keep', None),
            )
            self.mask_generators.append(mask_generator)
```

### 核心方法 `__call__`

```
    def __call__(self, batch):
        batch_size = len(batch)
        # 1. 先用默认工具把视频数据打包成 Batch [B, C, T, H, W]
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        # 2. 遍历所有的生成器，为这组 Batch 生成对应的掩码
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        # 返回：打包后的视频、给 Encoder 看的掩码、给 Predictor 预测用的掩码
        return collated_batch, collated_masks_enc, collated_masks_pred
```

## 3. _MaskGenerator 类：掩码工厂

这是真正的算法所在地，负责计算具体的坐标。

### 共享计数器与状态

```
class _MaskGenerator(object):
    def __init__(self, ...):
        # 计算网格尺寸：比如 224/16 = 14，16/2 = 8。网格就是 8x14x14
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size
        
        # 【关键】multiprocessing.Value: 
        # 因为 DataLoader 会开多个进程，为了保证随机种子同步，
        # 使用一个跨进程共享的计数器来生成 seed。
        self._itr_counter = Value('i', -1)
```

### 采样逻辑：决定“洞”的大小

`_sample_block_size` 函数负责决定遮挡块的长、宽、高（时间步）。

```
    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        # 1. 采样时间轴上的长度 (t)
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # 2. 采样空间轴上的面积 (h * w)
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # 3. 采样长宽比 (aspect_ratio) 并通过开根号计算出 h 和 w
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        return (t, h, w)
```

### 采样位置：决定“洞”在哪

`_sample_block_mask` 随机选一个起始点 `(top, left, start)`，然后挖掉一个块。

```
    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        # 随机找左上角坐标和起始帧
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start:start+t, top:top+h, left:left+w] = 0 # 0 表示被遮挡（要预测的部分）
        return mask
```

### 生成掩码索引 `__call__`

这是最核心的逻辑：

1. **确定尺寸**：先确定这一批数据要挖多大的洞。
    
2. **挖洞**：对 Batch 里的每个样本，随机挖 `npred` 个洞。
    
3. **取反**：
    
    - `mask_p` (Predictor mask): 值为 0 的地方（洞的位置）。
        
    - `mask_e` (Encoder mask): 值为 1 的地方（留下的背景）。
        
4. **对齐**：因为每个样本挖掉的比例可能略有不同，代码最后会取一个 `min_keep`，确保一个 Batch 里的掩码长度整齐划一，方便 GPU 并行计算。
    

## 4. 总结：这代码在干什么？

它的本质是**视频版“完形填空”题目生成器**。

- 它先决定题目（Mask）的大小。
    
- 再决定题目的位置。
    
- 最后把一张完整的视频切碎，告诉模型：“这些碎片（Encoder Mask）给你看，剩下的空白处（Predictor Mask）你给我猜出来。”

---
# src/masks/random_tube.py 逐行解析

如果说 `multiblock3d.py` 是在视频里“挖大洞”，那么 `random_tube.py` 就是在视频里“插吸管”。它会随机选择一些像素点，然后把这些点在**整个时间轴**上全部遮住。

## 1. 核心概念：什么是“随机管道 (Random Tube)”？

在视频处理中，如果我们在空间上随机遮住 90% 的格子，并且让这些遮挡在所有帧（时间维度）上都保持在**同一个位置**，那么从侧面看，这些遮挡就像一根根穿透视频的“管道”。

## 2. MaskCollator 类

这个类的结构与 `multiblock3d.py` 基本一致，是一个外部包装器。

- **初始化**：它读取配置文件中的 `ratio`（遮挡比例）。
    
- **作用**：遍历 `cfgs_mask`，为每一个配置创建一个 `_MaskGenerator`。
    
- **返回值**：同样返回 `(视频数据, Encoder掩码, Predictor掩码)`。
    

## 3. _MaskGenerator 类：管道工厂

这是实现“管道”逻辑的核心类。

### 初始化 `__init__`

```
def __init__(self, ..., ratio=0.9):
    # 计算空间网格：224/16 = 14 -> 14x14 = 196 个格子
    self.num_patches_spatial = self.height * self.width 
    self.ratio = ratio # 比如 0.9，表示遮住 90%

    # 计算每帧保留多少个格子：196 * (1 - 0.9) = 19.6 -> 19 个
    self.num_keep_spatial = int(self.num_patches_spatial * (1. - self.ratio))
    # 总共保留的 Token 数量 = 每帧保留数 * 时间长度
    self.num_keep = self.num_keep_spatial * self.duration
```

### 核心逻辑：`sample_mask` 函数

这是该文件最精妙的地方，通过几行代码实现了“管道”效果：

```
def sample_mask():
    # 1. 创建一个 1D 数组，0 表示遮住，1 表示保留
    mask = np.hstack([
        np.zeros(self.num_patches_spatial - self.num_keep_spatial), # 177 个 0
        np.ones(self.num_keep_spatial),                            # 19 个 1
    ])
    
    # 2. 随机打乱空间位置
    np.random.shuffle(mask)
    
    # 3. 【关键步】使用 np.tile 进行“平铺”
    # np.tile(mask, (self.duration, 1)) 的意思是：
    # 把这组“随机 0/1 序列”在时间维度（duration）上复制一份。
    # 结果：如果第 1 帧的左上角被遮了，那么第 2, 3, ..., 8 帧的左上角也全被遮了。
    mask = torch.tensor(np.tile(mask, (self.duration, 1)))
    
    # 4. 展平并提取索引
    mask = mask.flatten()
    mask_p = torch.argwhere(mask == 0).squeeze() # 预测目标（0的部分）
    mask_e = torch.nonzero(mask).squeeze()        # 训练输入（1的部分）
    return mask_e, mask_p
```

## 4. 与 Multi-block 的区别

|   |   |   |
|---|---|---|
|**特性**|**Multi-block (多块掩码)**|**Random Tube (随机管道掩码)**|
|**形状**|大的方形块（Blocks）|离散的像素点/格子（Patches）|
|**分布**|局部集中|全局随机分散|
|**时间一致性**|块在时间轴上可以有变化|强制时间轴完全一致（管道状）|
|**学习目标**|强迫模型根据周围环境联想大块内容|强迫模型根据极少数的采样点恢复全局特征|

## 5. 关键参数解析

1. **`ratio: 0.9`**: 这是一个非常极端的比例。意味着模型只能看到 10% 的像素，却要尝试还原剩下的 90%。
    
2. **`np.tile`**: 这是实现“管道”效果的灵魂函数。如果不使用 `tile` 而是在每一帧都随机重新分配 0 和 1，那就不叫“管道”，而叫“时空随机噪声”。
    
3. **`multiprocessing.Value`**: 同样出现在这里。在多进程加载数据时（`num_workers > 0`），它保证了每个进程拿到的随机序列是不同的，避免所有 Batch 的掩码都长得一模一样。
    

## 6. 总结：这代码在干什么？

这段代码是一个**“极简主义题目生成器”**。它在视频的每一帧上都戳出同样的“针孔”，让模型从这些细小的、贯穿始终的“针孔”中窥探整段视频的动作和语义。

---
# src/masks/utils.py 深度解析

`apply_masks` 函数的作用是根据生成的掩码索引，从原始的特征序列中“提取”出我们想要的那些部分。

## 1. 函数参数含义

```
def apply_masks(x, masks, concat=True):
    """
    :param x: 输入张量，形状为 [B, N, D]
              B: Batch Size (一批视频的数量)
              N: Num Patches (总共切了多少个块，如 14x14x8)
              D: Feature Dim (每个块的特征长度，如 384)
    :param masks: 掩码列表，里面的每个元素形状为 [B, K]
              K: 想要保留的块的数量 (Indices)
    """
```

## 2. 核心难点拆解

你提到的这一行是整个函数最关键的**维度匹配**操作：

`mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))`

### 为什么要这么做？

我们要使用的 `torch.gather` 函数有一个硬性规定：**索引张量的维度必须和输入张量的维度完全一致。**

- 输入 `x` 的维度是 3 维：`[B, N, D]`。
    
- 原始掩码 `m` 的维度只有 2 维：`[B, K]`。
    

如果我们直接用 `m` 去拿数据，程序会报错，因为它不知道在第 3 维（特征维度 $D$）上该怎么拿。我们需要把索引“拉伸”，让它覆盖整个特征向量。

### 拆解步骤：

假设 $B=2, K=10, D=384$：

1. **`m.unsqueeze(-1)`**:
    
    - 作用：在最后增加一个维度。
        
    - 变化：`[2, 10]` $\rightarrow$ `[2, 10, 1]`。
        
    - 此时，每个索引值被包在了一个小括号里，变成了类似 `[[[idx1], [idx2]...]]` 的样子。
        
2. **`.repeat(1, 1, x.size(-1))`**:
    
    - 作用：在最后一个维度上重复 $D$（384）次。
        
    - 变化：`[2, 10, 1]` $\rightarrow$ `[2, 10, 384]`。
        
    - **结果**：现在，原本的一个索引值（比如索引 5），变成了 384 个连续的 5。这告诉程序：“请把第 5 个块的全部 384 个特征数字都给我拿出来”。
        

## 3. 提取过程：`torch.gather`

```
all_x += [torch.gather(x, dim=1, index=mask_keep)]
```

- `dim=1`：告诉程序我们在“块（Patches）”这个维度上进行挑选。
    
- 通过上面准备好的 `mask_keep`，程序会从 `N` 个块中精准地挑出 `K` 个块。
    
- **提取后的形状**：`[B, K, D]`。
    

## 4. 拼接逻辑：`concat`

代码最后有一个判断：

```
if not concat:
    return all_x # 返回一个列表，里面装的是多个提取后的结果

return torch.cat(all_x, dim=0) # 把所有结果在 Batch 维度上接起来
```

### 为什么需要 `torch.cat`？

在 V-JEPA 训练中，由于我们可能同时有多个掩码策略（比如 4 个不同的遮挡块），`all_x` 就会包含 4 个 `[B, K, D]` 的张量。

- 执行 `torch.cat(..., dim=0)` 后，它们会合并成一个形状为 `[4*B, K, D]` 的大张量。
    
- 这样做的目的是为了**并行计算**：把 4 个任务合并成 1 个更大的 Batch 喂给显卡，效率更高。
    

## 总结：这行代码在干什么？

**那行代码实际上是在做“广播（Broadcasting）”的准备工作。**

它把**“我要拿第 5 号块”**这句话，变成了**“我要拿第 5 号块的第 1 个特征、第 2 个特征……直到第 384 个特征”**。只有这样，PyTorch 的底层计算引擎才能明白如何从内存中把那一整串特征向量完整地抽离出来。

---
---

# 问题：
### 1.`logger.info('loaded params...')`这句命令`[INFO][2026-02-04 07:39:38][process_main] loaded params...`对应这个结果吗，为什么，Info什么意思

是的，
在工业级代码（如 V-JEPA）中，开发者通常使用 Python 的 `logging` 模块而不是简单的 `print`。这条输出是一个典型的**格式化日志**：

- **`[INFO]`**: 这是**日志级别（Level）**。
    
- **`[2026-02-04 07:39:38]`**: 这是**时间戳**，记录代码运行到这一行的精确时间。
    
- **`[process_main]`**: 这是**函数名/模块名**。说明这行代码是在 `main.py` 里的 `process_main` 函数中被触发的。
    
- **`loaded params...`**: 这是**实际的消息内容**，即 `logger.info()` 括号里写的字符串。

### 2.yaml取帧解析
`num_clips: 1 # 每个视频取几个片段，预训练通常是 1。
`num_frames: 16 # 【时间长度】每个视频片段包含 16 帧画面。
`tubelet_size: 2 # 【切块厚度】模型把视频切成小方块时，时间维度上每 2 帧切一刀。
`sampling_rate: 4 # 【抽帧频率】每隔 4 帧取一帧。`

 **视频抽帧逻辑详解

你没看懂的那一段核心是：**“原本连贯的视频，是怎么变成模型手里的一叠照片的？”**

我们把原始视频想象成一卷很长的**电影胶卷**。

- **`sampling_rate: 4` (步长/间隔)**：
    
    意思是“**每隔 4 帧取 1 帧**”。
    
    - 模型不会要把每一帧都看一遍（那样数据量太大，而且相邻帧太像了，没信息量）。
        
    - 它看第 1 帧，然后跳过第 2、3、4 帧，直接看第 5 帧，以此类推。
        
    - **比喻**：就像看书不逐字读，而是“一目十行”，每 4 行只读 1 行。
        
- **`num_frames: 16` (目标帧数)**：
    
    意思是“**模型手里最终只拿 16 张照片**”。
    
    不管原始视频有多长，经过上面的“跳着选”，凑够 16 张就停。
    
- **`tubelet_size: 2` (切块厚度)**：
    
    这是进入模型**内部**后的处理。
    
    - 模型手里现在有 16 张照片。
        
    - `tubelet_size: 2` 意思是把这 16 张照片，**每 2 张叠在一起**，打包成一个“小方块”（Token）。
        
    - 所以模型实际上处理的是 $16 \div 2 = 8$ 个时间维度上的特征块。
        

**🧮 算账时间（为什么说是 64 帧跨度？）：**

为了凑齐这 16 张“跳着选”的照片，我们需要跨越多少原始视频？

$$16 \text{ (要几张)} \times 4 \text{ (每几张取一次)} = 64 \text{ 帧}$$

- 这意味着，虽然模型只看了 16 帧，但它覆盖了原始视频中 **64 帧** 的时间跨度（大约 2 秒多的内容）。
    

---

### 3. Epochs 与 Warmup 的关系

**Q: 如果设置总的 epoch 数是 300，那么从哪里设置？**

**A:** 就在配置文件的这里：

YAML

```
optimization:
  # ... 其他参数
  epochs: 300  # <--- 就是这里设置总轮次
```

**Q: Warmup 包含在里面吗？**

**A: 包含在里面。** (这是最重要的点！)

在深度学习的主流代码库（包括 Meta 的 V-JEPA、MAE 等）中，`epochs` 指的是**总的训练进度条**。`warmup` 只是这个进度条开头的**一个阶段**，而不是额外的附加时间。

**时间轴演示：**

如果你设置了 `epochs: 300` 和 `warmup: 40`，训练过程是这样的：

- **第 1 ~ 40 Epoch (Warmup 阶段)**：
    
    - 学习率从 `start_lr` (0.0002) **直线爬升** 到 `lr` (0.000625)。
        
    - 这是为了防止模型刚开始什么都不懂的时候，步子迈太大扯到蛋（梯度爆炸）。
        
- **第 41 ~ 300 Epoch (Main/Decay 阶段)**：
    
    - 学习率从 `lr` (0.000625) **按余弦曲线(Cosine)慢慢下降** 到 `final_lr` (0.000001)。
        
    - 这是正式学习阶段，越学越精细。
        
- **第 300 Epoch 结束**：
    
    - 训练彻底停止。
        

**总结公式：**

$$\text{总训练轮次} = \text{Warmup轮次} + \text{正式衰减轮次}$$

$$300 = 40 + 260$$

所以你不需要把 `epochs` 设置成 340，保持 300 即可。



### 4.Vision Transformer (ViT) 架构参数通俗解析

这段代码定义了模型的基础结构。以下是每个参数在“模型流水线”中的具体含义：

|   |   |   |   |
|---|---|---|---|
|**参数名**|**术语**|**通俗比喻**|**详细解释**|
|**`patch_size`**|切块大小|**网格的尺寸**|图像进入模型前会被切成小方格。`16` 表示每个方格是 16x16 像素。格子越小，细节越多，但计算量越大。|
|**`embed_dim`**|嵌入维度|**信息的宽度**|图像块被切开后会转化为一串数字（向量）。`192` 表示每个格子用 192 个数字来描述。数字越多，模型“记性”越好。|
|**`depth`**|深度|**流水线的工序**|图像信息要经过多少层 Transformer 模块。`12` 表示有 12 层。层数越深，模型逻辑推理能力越强。|
|**`num_heads`**|多头注意力数|**观察者的数量**|模型在看图时，会有多个“头”同时从不同角度看。`3` 表示有 3 个观察者。有的看颜色，有的看形状，有的看纹理。|
|**`mlp_ratio`**|MLP 放大比例|**思考的空间**|在每一层中间，模型会把信息放大再缩小。`4` 表示先把 192 维放大 4 倍到 768 维进行计算，再缩回来。|
|**`qkv_bias`**|QKV 偏置|**微调偏见**|是否在计算“注意力”时加入一个可学习的偏差。`True` 相当于给模型增加了一点灵活调整的余地。|
|**`norm_layer`**|归一化层|**标准化质检**|就像在每道工序后把零件“磨平、校准”。这里使用的是 `LayerNorm`，防止数字变得太大或太小导致训练失控。|

#### 🔍 为什么叫 "Tiny"？

我们可以对比一下你配置文件里用的 **`vit_large`**：

- **`vit_tiny`**: `embed_dim=192`, `depth=12`, `num_heads=3` (轻量，适合手机或快速调试)
    
- **`vit_large`**: `embed_dim=1024`, `depth=24`, `num_heads=16` (庞大，需要多张 4090 才能跑动)
    

#### 💡 关于 `**kwargs`

代码末尾的 `**kwargs` 是 Python 的一个语法，意思是：**“如果还有其他细碎的参数（比如你想临时改个名字或加个特殊标记），统统传给后面的 VisionTransformer 类。”** 它像是一个“等”字，增加了代码的通用性。


### 5.`cfgs` 是 "configurations"
cfgs_model = args.get('model') 为什么要叫cfgs这个名字