# 架构图
![mae架构图](C:\Users\HP\Desktop\mae_struc.png)

### 架构流程详解

1. **`main_pretrain.py` (总指挥)**:
    
    - **作用**: 程序的入口。负责这一团乱麻的初始化工作。
        
    - **调用**: 它调用 `models_mae` 来“出生”一个模型，调用 `util` 来设置分布式环境，调用 `torchvision` 加载图片。
        
    - **流程**: 设置参数 -> 建立数据管道 -> 初始化模型 -> 准备优化器 -> 进入死循环（Epoch）。
        
2. **`models_mae.py` (大脑 - 图一通常对应的文件)**:
    
    - **作用**: 定义了 MAE 的核心网络结构。
        
    - **核心逻辑**:
        
        - `patch_embed`: 把图片切成块。
            
        - `random_masking`: **核心创新点**，随机扔掉 75% 的块。
            
        - `encoder`: 只处理剩下的 25% 的块（速度快的原因）。
            
        - `decoder`: 把扔掉的块补回来（用 0 填充），尝试复原图片。
            
        - `forward_loss`: 计算复原的图片和原图（归一化后）的 MSE 损失。
            
3. **`engine_pretrain.py` (工头)**:
    
    - **作用**: 负责脏活累活，即“训练一个 Epoch”。
        
    - **流程**: 从 DataLoader 拿一批数据 -> 丢给模型 -> 算 Loss -> 反向传播 -> 更新参数 -> 打印日志。


---

该架构的设计目的是为了**自监督预训练 (Self-Supervised Pre-training)**。

- **输入 (Input)**: 一张完整的图片（例如 224x224）。
    
- **切块与掩码 (Patchify & Mask)**:
    
    - 代码对应: `models_mae.py` 中的 `random_masking` 函数。
        
    - 逻辑: 将图片切成 16x16 的小块。**关键操作**：随机扔掉 75% 的块，只保留 25%。这是为了强迫模型学习图片的语义，而不是死记硬背。
        
- **编码器 (Encoder)**:
    
    - 代码对应: `models_mae.py` 中的 `forward_encoder`。
        
    - 逻辑: 标准的 ViT (Vision Transformer)。**注意**：它只看那 25% 可见的块。因为输入少，所以计算量极小（仅为完整 ViT 的 1/4 左右）。
        
- **解码器 (Decoder)**:
    
    - 代码对应: `models_mae.py` 中的 `forward_decoder`。
        
    - 逻辑: 一个轻量级的 Transformer。它接收编码器的输出，加上那些“被扔掉位置”的占位符（Mask tokens），试图还原整张图。
        
- **重构目标 (Reconstruction Target)**:
    
    - 代码对应: `models_mae.py` 中的 `forward_loss`。
        
    - 逻辑: 计算预测的像素值与原始像素值的均方误差 (MSE)。通常会先对每个 Patch 的像素做归一化 (`norm_pix_loss`) 以提高训练稳定性。



# main_pretrain(入口脚本)讲解

“导入必要的库
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

”timm 是 PyTorch 图像模型库，这里用作基础组件
import timm

“强制版本检查，确保复现性
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

”导入自定义工具库
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

“核心：导入 MAE 模型定义
import models_mae

”核心：导入训练引擎（训练一个 epoch 的逻辑）
from engine_pretrain import train_one_epoch


def get_args_parser():
    """
    定义命令行参数解析器。
    这里包含了训练所需的所有超参数。
    """
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    
    # --- 基础训练参数 ---
    parser.add_argument('--batch_size', default=64, type=int,
                        help='每张 GPU 的 Batch size (总 batch_size = batch_size * accum_iter * num_gpus)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='梯度累积次数。显存不够时，可以通过累积多次梯度再一次更新参数，变相增大 batch size')

    # --- 模型参数 ---
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='要训练的模型名称，对应 models_mae.py 中的函数名')

    parser.add_argument('--input_size', default=224, type=int,
                        help='输入图片大小')

    # 关键参数：Mask Ratio
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='掩码比例。MAE 论文发现 0.75 (75%) 效果最好，比 BERT 的 15% 高得多')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='是否在计算 Loss 前对每个 Patch 的像素做归一化。这通常能提升表示学习的效果')
    parser.set_defaults(norm_pix_loss=False)

    # --- 优化器参数 ---
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减，防止过拟合 (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='学习率 (绝对值)。如果不设，会根据 batch size 自动计算')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='基础学习率 (base_lr)。计算公式: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='余弦退火调度器的最小学习率')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='预热 Epoch 数。训练初期先用小学习率慢慢升上来，防止模型跑飞')

    # --- 数据集参数 ---
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='ImageNet 数据集路径')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='模型保存路径')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='Tensorboard 日志路径')
    parser.add_argument('--device', default='cuda',
                        help='训练设备')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='断点续训的 checkpoint 路径')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='开始的 epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='是否锁页内存，通常能加速数据从 CPU 到 GPU 的传输')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # --- 分布式训练参数 ---
    parser.add_argument('--world_size', default=1, type=int,
                        help='分布式进程数')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='建立分布式通信的 URL')

    return parser


def main(args):
    # 1. 初始化分布式环境 (DDP)
    # 这包括设置 GPU device，world_size, rank 等
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 2. 固定随机种子
    # 保证实验可复现。注意种子要加上 rank，保证不同 GPU 上的随机数种子不同（如果是数据增强需要）
    # 但这里 seed = args.seed + misc.get_rank() 实际上是让不同 GPU 种子不同
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True # 加速卷积运算

    # 3. 数据增强 pipeline
    # MAE 的预训练只需要非常简单的增强：随机裁剪缩放 + 翻转。
    # 不需要像对比学习 (MoCo/SimCLR) 那样复杂的增强，因为 Masking 本身就是极强的增强。
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # 加载数据集 (ImageNet train set)
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    # 4. 设置 Sampler
    # 如果是分布式训练，使用 DistributedSampler 确保不同 GPU 拿到不重叠的数据
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # 5. 设置 Tensorboard Logger (仅在主进程)
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # 6. 数据加载器 DataLoader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True, # 丢弃最后一个不完整的 batch
    )
    
    # 7. 定义模型
    # 从 models_mae.py 中根据名字加载模型 (例如 mae_vit_large_patch16)
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # 计算有效 Batch Size = 单卡 batch * 梯度累积次数 * 卡数
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    # 8. 自动调整学习率
    # Linear Scaling Rule: 学习率应该和 Batch Size 成正比
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # 9. 包装分布式模型 (DDP)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # 10. 配置优化器
    # param_groups: 这里的技巧是，对于 Bias (偏置) 和 Norm (归一化层) 的参数，不进行权重衰减 (Weight Decay = 0)
    # 这是一个通用的 Transformer 训练技巧。
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    # 混合精度训练的 Scaler
    loss_scaler = NativeScaler()

    # 11. 加载断点 (如果有)
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    # --- 12. 训练主循环 ---
    for epoch in range(args.start_epoch, args.epochs):
        # 分布式训练需要每个 epoch 设置 sampler 的 epoch，保证数据 shuffle 的随机性不同
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # 调用 engine_pretrain.py 中的 train_one_epoch 执行真正的训练
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        # 保存模型 Checkpoint (每 20 个 epoch 或最后一个 epoch)
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # 记录日志
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)




# MAE 项目使用指南与架构说明

## 第一部分：终端运行命令 (Terminal Commands)

假设你已经配置好了环境（PyTorch, timm 等），并且 ImageNet 数据集路径为 `/path/to/imagenet`。

### 1. 最简单的运行版本 (Default)

直接使用默认参数运行。默认模型是 `mae_vit_large_patch16`，默认 mask 比例 0.75。

```
python main_pretrain.py \
    --data_path /path/to/imagenet \
    --output_dir ./output_dir
```

### 2. 修改核心变量的版本 (Customized)

这里演示如何修改：**Batch Size**（显存不够时调小）、**Epochs**（训练轮数）、**模型类型**（改用 Base 模型）、**Mask 比例**。

```
python main_pretrain.py \
    --data_path /path/to/imagenet \
    --output_dir ./output_dir_base \
    --model mae_vit_base_patch16 \
    --batch_size 128 \
    --epochs 200 \
    --mask_ratio 0.5 \
    --warmup_epochs 20
```

- `--model mae_vit_base_patch16`: 切换为更小的 Base 模型（默认是 Large）。
    
- `--mask_ratio 0.5`: 将掩码比例改为 50%（虽然论文说 75% 最好，但你可以改）。
    
- `--batch_size 128`: 增大批次大小。
    

### 3. 单机多卡分布式训练 (Distributed)

MAE 这种大模型通常需要多卡训练。假设你有 8 张 GPU：

```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
    --data_path /path/to/imagenet \
    --output_dir ./output_dir \
    --batch_size 64 \
    --accum_iter 4
```

- `--nproc_per_node=8`: 使用 8 张卡。
    
- `--accum_iter 4`: 梯度累积。实际 Batch Size = 64 * 8 (卡数) * 4 (累积) = 2048。这是在大 Batch Size 下训练的关键技巧。
    

## 第二部分：项目仓库结构与文件作用 (Project Structure)

这个仓库的代码结构非常清晰，遵循了 Facebook Research (Meta) 的一贯风格（如 DeiT）。

### 核心代码文件

1. **`main_pretrain.py` (入口脚本)**
    
    - **作用**：程序的指挥官。
        
    - **职责**：解析参数、建立分布式环境、构建数据集 (`ImageFolder`)、初始化模型 (`models_mae`)、配置优化器 (`AdamW`)，最后调用 `engine_pretrain` 开始循环训练。
        
2. **`engine_pretrain.py` (训练引擎)**
    
    - **作用**：负责“脏活累活”，即具体执行一个 Epoch 的训练。
        
    - **职责**：从 DataLoader 取数据 -> 丢进模型 -> 算 Loss -> 反向传播 -> 更新参数 -> 打印日志。它不关心模型长什么样，只管算梯度。
        
3. **`models_mae.py` (MAE 模型定义)**
    
    - **作用**：**核心文件**，定义了 Masked Autoencoder 的架构。
        
    - **职责**：包含 Encoder (ViT)、Decoder (轻量级) 以及最关键的 `random_masking` (随机挖洞) 和 `forward_loss` (计算像素重建误差)。**这是预训练阶段专用的模型。**
        
4. **`models_vit.py` (标准 ViT 模型定义)**
    
    - **作用**：标准的 Vision Transformer 架构。
        
    - **职责**：它主要用于 **Fine-tuning (微调)** 阶段。预训练完成后，我们会扔掉 MAE 的 Decoder，只保留 Encoder，并把它加载到这个标准的 ViT 中进行分类任务训练。
        

### 工具代码文件 (`util/`)

5. **`util/misc.py` (杂项工具)**
    
    - **作用**：各种辅助函数。
        
    - **职责**：分布式设置 (`init_distributed_mode`)、保存/加载模型 (`save_model`, `load_model`)、统计指标 (`MetricLogger`) 等。
        
6. **`util/pos_embed.py` (位置编码)**
    
    - **作用**：生成 Transformer 所需的位置编码。
        
    - **职责**：特别是生成 `sin-cos` 形式的 2D 位置编码，这是 MAE 初始化时必须的。
        
7. **`util/lr_sched.py` (学习率调度)**
    
    - **作用**：控制学习率如何变化。
        
    - **职责**：实现了 Warmup + Cosine Decay 的学习率调整策略。
        
8. **`util/datasets.py` (虽然你没上传，但通常存在)**
    
    - **作用**：自定义的数据集读取逻辑（如果有的话，但 MAE 预训练主要直接用 `torchvision` 的 `ImageFolder`）。




# MAE 核心代码深度解析

## 第一部分：Engine 解析 (`engine_pretrain.py`)

Q: 训练逻辑调用是不是要看 engine_pretrain.py？

A: 是的。 main_pretrain.py 只是搭建舞台，真正的“演出”（训练循环）是在这里进行的。

### `engine_pretrain.py` 逐行中文解析

```
# 导入必要的数学和系统库
import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    """
    执行一个 Epoch 的训练逻辑
    """
    # 1. 将模型设置为训练模式 (启用 Dropout, BatchNorm 等)
    model.train(True)
    
    # 2. 初始化日志记录器，用于打印平滑后的 Loss 等信息
    metric_logger = misc.MetricLogger(delimiter="  ")
    # 添加学习率记录
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20 # 每 20 个 step 打印一次日志

    accum_iter = args.accum_iter # 梯度累积步数

    optimizer.zero_grad() # 清空梯度

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 3. 遍历数据加载器 (Data Loader)
    # metric_logger.log_every 会自动处理打印逻辑
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # 4. 调整学习率
        # MAE 使用基于 step (iteration) 的更新，而不是基于 epoch，这样更平滑
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # 5. 数据搬运到 GPU
        samples = samples.to(device, non_blocking=True)

        # 6. 前向传播 (Forward) + 混合精度 (AMP)
        with torch.cuda.amp.autocast():
            # 调用 model(samples)。这里会自动触发 models_mae.py 中的 forward()
            # 注意：MAE 预训练不需要标签 (target)，只需要图片 (samples)
            # mask_ratio 控制这一个 batch 挖掉多少块
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        # 检查 Loss 是否炸了 (Infinity)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 7. 梯度累积处理
        # Loss 除以 accum_iter，因为梯度是累加的，平均值才对
        loss /= accum_iter
        
        # 8. 反向传播 (Backward) + 优化器更新 (Step)
        # loss_scaler 处理混合精度的梯度缩放
        # update_grad=True 只有在累积够了步数后才真正更新权重
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # 如果更新了权重，就清空梯度，准备下一轮累积
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize() # 等待 GPU 计算完成

        # 9. 记录日志
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # 10. Tensorboard 写入 (如果是主进程且到了更新步数)
        loss_value_reduce = misc.all_reduce_mean(loss_value) # 多卡训练时，计算所有卡的平均 Loss
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # 打印本 Epoch 的平均统计数据
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
```



## 第二部分：models_mae.py 与 models_vit.py 的区别

**Q: 为什么有两个模型文件？有什么区别？为什么预训练调用前者？**

|   |   |   |
|---|---|---|
|**特性**|**models_mae.py**|**models_vit.py**|
|**用途**|**预训练 (Pre-training)**|**微调 (Fine-tuning) / 分类**|
|**任务**|图像重建 (画出被遮挡的部分)|图像分类 (识别是什么物体)|
|**结构**|**Encoder + Decoder**|**Only Encoder + Classifier Head**|
|**输入**|部分图片 (25% 的 patches)|完整图片 (100% 的 patches)|
|**输出**|重建的像素值|类别概率 (Softmax)|
|**Loss**|MSE Loss (均方误差)|CrossEntropy Loss (交叉熵)|
|**调用原因**|`main_pretrain.py` 是为了让模型“学习理解世界”，需要 Decoder 来验证它是否理解，所以调用 MAE。|预训练结束后，Decoder 被丢弃。我们将 Encoder 的权重加载到 `models_vit.py` 中，接上分类头，去跑 ImageNet 分类刷榜。|



## 第三部分：模型代码逐行解析

### 1. `models_mae.py` 解析 (核心关注 Masking 和 Decoder)

```
class MaskedAutoencoderViT(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # --------------------------------------------------------------------------
        # Encoder 部分 (和 ViT 几乎一样)
        self.patch_embed = PatchEmbed(...) # 将图片切成 16x16 的块
        self.cls_token = ... # 分类 Token (虽然预训练不用它分类，但为了保持结构一致)
        self.blocks = ...    # Transformer Encoder Blocks (处理可见的 patches)
        self.norm = ...      # 归一化层
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder 部分 (MAE 特有)
        # 将 Encoder 的输出维度 (如 1024) 映射到 Decoder 维度 (如 512)
        # Decoder 通常比 Encoder 小很多，这让 MAE 训练非常快
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask Token: 一个可学习的向量，代表“被挖掉的地方”
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder 的 Transformer Blocks
        self.decoder_blocks = ... 
        
        # 预测头: 将 Decoder 输出映射回像素 (16*16*3 = 768 个像素值)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

    # --- 核心函数 1: 随机掩码 (Random Masking) ---
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # N: Batch大小, L: Patch数量, D: 维度
        len_keep = int(L * (1 - mask_ratio)) # 计算需要保留多少个 patch
        
        # 技巧: 生成随机噪声，然后排序
        noise = torch.rand(N, L, device=x.device)
        
        # argsort 返回的是索引。数值小的排前面 -> 保留；数值大的排后面 -> 丢弃
        ids_shuffle = torch.argsort(noise, dim=1) 
        ids_restore = torch.argsort(ids_shuffle, dim=1) # 用于后面还原顺序

        # 取前 len_keep 个索引
        ids_keep = ids_shuffle[:, :len_keep]
        
        # torch.gather: 根据索引从 x 中取出数据
        # 结果 x_masked 只包含原图 25% 的数据！大大减少了计算量
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 生成 mask: 0 代表保留，1 代表被挖掉
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore) # 还原顺序对应的 mask

        return x_masked, mask, ids_restore

    # --- 核心函数 2: Encoder 前向传播 ---
    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x) # [N, 3, 224, 224] -> [N, 196, 1024]
        x = x + self.pos_embed[:, 1:, :] # 加上位置编码
        
        # 关键步骤: 随机扔掉 75% 的 patches
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # 加上 CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 扔进 Transformer Blocks 跑
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    # --- 核心函数 3: Decoder 前向传播 ---
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x) # 映射维度 1024 -> 512

        # 构造完整的序列: Encoder输出 + Mask Tokens
        # 我们需要把“挖掉”的部分用 mask_token 填回去，这样才能恢复成完整的图片形状
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        # 拼接：[可见的patches, mask_tokens]
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) 
        
        # 还原顺序: 把乱序的 patch 变回原始图片的位置顺序
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
        
        # 加上 CLS token
        x = torch.cat([x[:, :1, :], x_], dim=1) 

        # 加上 Decoder 的位置编码 (因为 Mask token 也需要位置信息才知道自己在哪里)
        x = x + self.decoder_pos_embed

        # 扔进 Decoder Blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 预测像素值
        x = self.decoder_pred(x)
        # 移除 CLS token (因为它不包含像素信息)
        x = x[:, 1:, :]
        return x

    def forward(self, imgs, mask_ratio=0.75):
        # 1. Encoder 提取特征 (只处理可见部分)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        # 2. Decoder 尝试复原 (处理全部部分)
        pred = self.forward_decoder(latent, ids_restore)
        # 3. 计算 Loss
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
```


### 2. `models_vit.py` 解析 (标准分类模型)

```
# 继承自 timm 库的标准 VisionTransformer
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ 
    标准的 ViT，增加了一些功能支持（如全局平均池化）
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        
        # 如果启用 global_pool (GAP)，通常用于分类效果更好，替代仅使用 CLS token
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # 删除原始的 norm 层，改用 fc_norm

    def forward_features(self, x):
        B = x.shape[0]
        # 切块 [N, 3, H, W] -> [N, L, D]
        x = self.patch_embed(x)

        # 加上 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 加上位置编码 (注意：这里没有 masking！处理的是完整图片)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过 Transformer Blocks
        for blk in self.blocks:
            x = blk(x)

        # 输出处理
        if self.global_pool:
            # 如果是 GAP，去掉 CLS token，对所有 patch 取平均
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            # 否则标准做法：只取 CLS token 用于分类
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
        # 之后 outcomes 会被传给 self.head (分类全连接层) 输出类别概率
```