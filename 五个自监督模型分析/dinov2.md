
# 1. 代码仓库结构全解

你上传的图片展示了一个非常标准的深度学习项目结构。你可以把它想象成一个**“大工厂”**，每个文件夹都是一个不同的部门。

#### **核心目录 (`dinov2/` 文件夹内)**

这是工厂的“核心车间”，所有的核心技术都在这里。

- **`configs` (配置部)**: 存放 `.yaml` 文件。就像“施工图纸”或“配方单”，规定了模型有多大、学习率是多少、跑多少轮。**这是你最常修改的地方。**
    
- **`data` (原料部)**: 负责加载图片、进行预处理（裁剪、翻转）。它把原始图片变成模型能“吃”的数字矩阵。
    
- **`layers` (零件部)**: 存放自定义的神经网络层（比如特殊的 Attention 模块）。
    
- **`loss` (质检部)**: 定义了“损失函数”。它负责计算模型做得好不好，告诉模型“你刚才猜错了，离正确答案差这么远”。
    
- **`models` (组装部)**: 这里定义了 Vision Transformer (ViT) 的整体架构，把零件组装成完整的机器人。
    
- **`train` (生产线)**: 存放训练循环的代码。也就是机器日夜不停运转、学习参数的地方。
    
- **`utils` (后勤部)**: 存放各种小工具，比如记录日志、保存模型权重、计算时间等。
    
- **`eval` (测试部)**: 训练好的模型要在这里进行考试（评估），看在分类或分割任务上表现如何。
    

## **外层目录**

- **`scripts`**: 存放一键运行的脚本（比如 `lint.sh` 是用来检查代码格式是否漂亮的，和训练无关）。
    
- **`notebooks`**: 存放 `.ipynb` 文件，像是“演示草稿本”，用来快速画图或测试小段代码。
    
- **`requirements.txt` / `conda.yaml`**: 环境清单。告诉电脑运行这个工厂需要安装哪些软件（PyTorch, Numpy 等）。
    

---

# 2. 详细代码解释 (逐行/逐块解析)

你上传了两个 `train.py`。为了区分，我们把 **`train (1).py` 称为“启动器 (Launcher)”**，把 **`train.py` 称为“执行者 (Worker)”**。

#### **文件一：`train (1).py` —— 启动器 (The Launcher)**

**功能**：这个文件不干累活。它的作用是**“派单”**。它负责检查配置，然后使用 `submitit` 工具把训练任务发送到服务器集群（如果有的话）或者在本地启动。

- **导入部分 (Imports)**:
    
    Python
    
    ```
    import logging...
    from dinov2.run.submit import get_args_parser, submit_jobs
    ```
    
    这里引入了“派单员” (`submit_jobs`)。
    
- class Trainer(object):
    
    这是一个包装壳。
    
    - `__init__`: 记下你的命令（参数 `args`）。
        
    - `__call__`: **关键动作**。当任务开始执行时，它才真正去引用 `dinov2.train.main` (执行者) 并让它干活。
        
    - `checkpoint`: 这是一个“存档”功能。如果服务器因为时间到了要把任务杀掉，这个函数会说：“等等，让我存个档，然后重新排队继续算。”
        
- def main():
    
    整个脚本的入口。
    
    1. `get_args_parser`: 拿小本本记下你运行命令时输入的参数（比如 `--config-file xxx`）。
        
    2. `setup_logging`: 打开录音笔（日志），准备记录发生了什么。
        
    3. `assert os.path.exists`: 检查你的“配方单”（config文件）是不是真的存在，不存在就报错。
        
    4. **`submit_jobs(Trainer, args...)`**: **最重要的一行**。它把上面的 `Trainer` 打包，扔给计算集群去跑。
        

#### **文件二：`train.py` —— 执行者 (The Worker)**

**功能**：这是真正的**苦力**。它负责搭建模型、加载数据、计算误差、更新参数。**所有的数学运算都在这里发生。**

- **`get_args_parser`**: 定义了这个脚本能接受哪些具体指令（比如 `--output-dir` 结果存哪）。
    
- **`build_optimizer` (组装引擎)**:
    
    - 创建 `AdamW` 优化器。你可以把它理解为“老师傅”，他根据误差来微调模型的螺丝（参数），让模型越来越准。
        
- **`build_schedulers` (制定课程表)**:
    
    - 它决定了学习率（Learning Rate）怎么变。通常是“先热身（warmup），再猛学，最后慢慢学（cosine decay）”。
        
    - 这里还控制了 **Teacher-Student** 架构中的“动量（momentum）”和“温度（temperature）”。（DINO 的核心机制，稍后解释）。
        
- do_train (核心循环 —— 生产流水线):
    
    这是代码的心脏，咱们拆细一点：
    
    1. **准备阶段**:
        
        - `model.train()`: 告诉模型“现在是上课时间，不是考试，要允许参数变化”。
            
        - `inputs_dtype = torch.half`: 使用半精度（FP16），为了省显存、跑得快。
            
        - `make_data_loader`: 打开“传送带”，把图片源源不断地送进来。
            
    2. **循环阶段 (`for data in metric_logger...`)**:
        
        - `lr_schedule[...]`: 每一轮都看一眼课程表，调整当天的学习率。
            
        - `optimizer.zero_grad()`: **清空之前的计算**。就像把黑板擦干净，准备解下一道题。
            
        - `loss_dict = model.forward_backward(data, ...)`: **最关键的一步**。
            
            - **Forward (前向)**: 模型看图片，猜特征。
                
            - **Loss**: 算出猜得有多离谱。
                
            - **Backward (反向)**: 自动计算每个参数该怎么调才能减小误差。
                
        - `optimizer.step()`: **更新参数**。老师傅动手拧螺丝了。
            
        - `model.update_teacher(mom)`: **DINO 特有步骤**。更新“老师模型”。老师模型不是直接训练的，而是学生模型参数的“滑动平均”（EMA）。也就是说，老师是过去的学生的平均值，比学生更稳重。
            
    3. **收尾**:
        
        - 每隔一段时间 (`periodic_checkpointer`)，保存一下模型（`.pth` 文件），防止断电白跑。
            

---

# 3. DINOv2 项目架构全景图

![dinov2架构全景图](D:\google_gemini3\dinov2_struc.png)


```
graph TD
    User[用户 (你)] -->|运行命令| Launcher[启动器: train (1).py]
    
    subgraph "任务调度 (Submission)"
        Launcher -->|读取配置| Config[Configs (.yaml)]
        Launcher -->|提交任务| Submitit[Submitit (集群管理)]
    end
    
    Submitit -->|启动进程| Worker[执行者: train.py]
    
    subgraph "训练循环 (Training Loop)"
        Worker -->|1. 构建| Model[模型: SSLMetaArch (models/)]
        Worker -->|2. 加载| DataLoader[数据加载: data/ (含增强)]
        Worker -->|3. 创建| Optim[优化器 & 课程表]
        
        loop "每一批次 (Batch)"
            DataLoader -->|图片| Model
            Model -->|计算 Loss| Loss[DINO Loss]
            Loss -->|梯度| Optim
            Optim -->|更新参数| Model
            Model -->|EMA 更新| Teacher[老师模型]
        end
        
        Worker -->|定期保存| Checkpoint[权重文件 (.pth)]
        Worker -->|定期测试| Eval[eval/ (测试代码)]
    end
```

**调用逻辑总结：**

1. 你运行 `train (1).py`。
    
2. 它把任务扔给服务器。
    
3. 服务器运行 `train.py`。
    
4. `train.py` 召唤 `SSLMetaArch`（大脑）和 `DataLoader`（眼睛）。
    
5. 大脑不断看图、自我纠错、进化，最后把练好的“大脑”存成 `.pth` 文件。
    

---

# 4. 解释 DINOv2

DINOv2 (Self-distillation with NO labels v2) 是 Meta 推出的一种自监督学习计算机视觉模型。

- 核心逻辑（简单版）：
    
    传统的 AI 像是一个学生，需要老师在每一张图上打勾（比如“这是猫”）。
    
    DINOv2 像是一个天才自学者。你给它一堆图，不告诉它图里是什么。它通过对比同一张图的不同裁剪部分来学习。
    
    - **逻辑**：“这张图的左上角（猫耳朵）和整张图（猫）应该属于同一个东西。”
        
    - **架构**：它有两个分身，**Student（学生）** 和 **Teacher（老师）**。
        
        1. 把一张图剪成两块：块A 和 块B。
            
        2. 学生看 块A，老师看 块B。
            
        3. 强制让学生输出的特征去模仿老师的特征。
            
    - **结果**：学完后，不需要任何标签，它就能极好地理解图片内容（提取特征）。
        
- 为什么它这么强？
    
    它用了海量数据（1.42亿张图），而且不需要人工标注。这让它学到的特征非常通用，不管是做分割、分类还是深度估计，效果都非常好。
    


---
---


# 5. DINOv2 启动脚本解析 (`dinov2/dinov2/run/train.py`)

文件定位：这是任务的发射台 (Launcher)。

主要作用：它不进行具体的模型训练计算，而是负责配置环境、解析参数，并调用 submitit 工具将训练任务提交到计算集群（或者在本地启动进程）。

## 1. 导入部分 (Imports)

```
import logging
import os
import sys

from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser
from dinov2.run.submit import get_args_parser, submit_jobs

logger = logging.getLogger("dinov2")
```

- **`setup_logging`**: 引入日志工具，确保程序运行时的输出（比如“开始训练”、“报错了”）能被记录下来。
    
- **`get_train_args_parser`**: 从核心训练代码（即另一个 `train.py`）中借用参数解析器。因为它需要知道你想跑什么配置。
    
- **`submit_jobs`**: **关键函数**。这是一个封装好的工具，用来把任务分发给 GPU。
    

## 2. `Trainer` 类 (核心任务包装器)

这个类是一个“任务包”，它定义了当任务被分配到某个 GPU 上时，具体该干什么。

### `__init__(self, args)`

```
class Trainer(object):
    def __init__(self, args):
        self.args = args
```

- **作用**：初始化。当你创建一个 `Trainer` 时，把你的命令参数（比如 `batch_size`, `output_dir`）存到 `self.args` 里带在身上。
    

### `__call__(self)`

```
    def __call__(self):
        from dinov2.train import main as train_main

        self._setup_args()
        train_main(self.args)
```

- **作用**：**这是真正干活的入口**。当服务器（或本地进程）开始运行这个任务时，会自动调用这个函数。
    
- **逻辑**：
    
    1. `from dinov2.train import main as train_main`: 此时才导入真正的训练主函数（避免在提交任务前就加载巨大的模型）。
        
    2. `self._setup_args()`: 设置好当前进程的 ID 等信息。
        
    3. `train_main(self.args)`: **核心调用**。把控制权移交给 `train.py` 的 `main` 函数，开始真正的训练。
        

### `checkpoint(self)`

```
    def checkpoint(self):
        import submitit

        logger.info(f"Requeuing {self.args}")
        empty = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty)
```

- **作用**：**防中断存档机制**。
    
- **场景**：在大型集群上，任务往往有时间限制（比如只能跑 3 天）。如果时间到了还没跑完，集群会杀掉任务。
    
- **逻辑**：这个函数会被自动调用。它会生成一个新的“任务包”，重新去排队（Requeue），确保训练能接着跑，而不是直接断掉。
    

### `_setup_args(self)`

```
    def _setup_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.output_dir.replace("%j", str(job_env.job_id))
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Args: {self.args}")
```

- **作用**：环境配置。
    
- **细节**：
    
    - 它获取当前任务的 ID (`job_id`)。
        
    - 它把输出目录里的 `%j` 替换成实际的任务 ID。比如 `output_dir/run_%j` 变成 `output_dir/run_12345`。这样每次跑实验结果都在不同文件夹，不会覆盖。
        

## 3. `main()` 函数 (脚本入口)

这是你在这个文件里直接运行的代码。

```
def main():
    description = "Submitit launcher for DINOv2 training"
    # 1. 获取参数解析器
    train_args_parser = get_train_args_parser(add_help=False)
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    # 2. 设置日志
    setup_logging()

    # 3. 检查配置文件是否存在
    assert os.path.exists(args.config_file), "Configuration file does not exist!"

    # 4. 提交任务
    submit_jobs(Trainer, args, name="dinov2:train")
    return 0
```

- **逐条解释**：
    
    1. **参数组合**：它不仅接受提交任务的参数（比如“用几个节点”），还把 `train.py` 需要的参数（比如“学习率多少”）也拿了过来。这样你在这里输参数，最后能传给真正的训练代码。
        
    2. **`submit_jobs(Trainer, args, ...)`**: **这是最终的一击**。
        
        - 它拿着 `Trainer` 这个类。
            
        - 拿着你输入的 `args`。
            
        - 调用底层的 `submitit` 库。
            
        - **结果**：如果是在集群，它发送任务；如果是本地运行，它直接启动进程。
            

## 4. 文件调用关系总结

1. **用户** 在命令行运行 `python train (1).py --config-file ...`。
    
2. **`main()`** 函数运行，解析参数。
    
3. **`submit_jobs`** 被调用，它初始化 **`Trainer`** 类。
    
4. **`Trainer.__call__`** 被触发（在计算节点上）。
    
5. **`Trainer`** 内部导入并调用 **`dinov2/train.py`** 的 `main` 函数。
    
6. **正式训练开始**。


## 代码详解


’导入系统标准库

import logging  # 用于记录日志（打印运行信息）

import os       # 用于操作系统交互（检查文件是否存在等）

import sys      # 用于系统参数（退出程序等）

  

’导入DINOv2 自己的模块

from dinov2.logging import setup_logging  # 专门配置好的日志工具

‘注意：这里从 dinov2.train 导入了 get_args_parser 并重命名为 get_train_args_parser

‘因为下面 submit 也有一个 get_args_parser，为了防止名字冲突

from dinov2.train import get_args_parser as get_train_args_parser

from dinov2.run.submit import get_args_parser, submit_jobs  # 导入提交任务的工具

  

’获取一个名为 "dinov2" 的日志记录器

logger = logging.getLogger("dinov2")

  
  

‘ 定义 Trainer 类

’这个类是一个“包装壳”，它的主要作用是告诉 Submitit 工具：

‘当你在服务器节点上把代码跑起来的时候，请执行这个类里面的逻辑。”

class Trainer(object):

    def __init__(self, args):

        # 初始化：把外部传入的参数（如 batch_size, 路径等）存到自己身上

        self.args = args

  

    def __call__(self):

        # 【核心执行入口】

        # 当这个对象被当做函数调用时（trainer()），会运行这里的代码

        # 这段代码是在计算节点（GPU服务器）上实际运行的

        # 为什么要在函数里面 import？

        # 因为如果一开始就 import，可能会在提交任务的主节点加载巨大的模型，导致卡死。

        # 只有真正到了计算节点，才加载训练主函数。

        from dinov2.train import main as train_main

  

        # 设置当前任务的参数（比如修改输出目录名）

        self._setup_args()

        # 开始真正的训练！调用 dinov2/train.py 里的 main 函数

        train_main(self.args)

  

    def checkpoint(self):

        # 【断点续传机制】

        # 如果你用的是 Slurm 这种集群调度系统，任务有时间限制（比如限时 72 小时）。

        # 当时间快到了，系统会发信号。submitit 会自动调用这个 checkpoint 函数。

        import submitit

  

        logger.info(f"Requeuing {self.args}") # 记录日志：我要重新排队了

        # 创建一个新的 Trainer 实例（把自己复制一份）

        empty = type(self)(self.args)

        # 返回一个“延迟提交”对象。

        # 意思是：告诉集群，“我还没干完，请把这个新任务重新加到队列里，下次接着跑”。

        return submitit.helpers.DelayedSubmission(empty)

  

    def _setup_args(self):

        # 设置参数的辅助函数

        import submitit

  

        # 获取当前的任务环境信息（比如是第几个节点，任务ID是多少）

        job_env = submitit.JobEnvironment()

        # 【关键路径修改】

        # 把输出目录里的 "%j" 替换成真实的任务 ID。

        # 比如：output_dir="/checkpoint/run_%j" -> "/checkpoint/run_12345"

        # 这样每次运行都在不同的文件夹，不会覆盖。

        self.args.output_dir = self.args.output_dir.replace("%j", str(job_env.job_id))

        # 打印当前进程的信息，方便调试

        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

        logger.info(f"Args: {self.args}")

  
  

’‘’‘’‘’‘脚本的主入口函数

def main():

    description = "Submitit launcher for DINOv2 training"

    # 1. 获取训练脚本需要的参数（比如 --lr, --batch-size）

    # add_help=False 是因为我们要把这些参数拼接到总参数里，先不处理帮助文档

    train_args_parser = get_train_args_parser(add_help=False)

    # 2. 组合参数

    # parents=[train_args_parser] 意思是：

    # “我这个启动脚本，不仅接受提交任务的参数（比如用几个节点），

    #  也直接接受训练脚本的所有参数，并透传给它。”

    parents = [train_args_parser]

    args_parser = get_args_parser(description=description, parents=parents)

    # 3. 解析用户输入的命令行参数

    args = args_parser.parse_args()

  

    # 4. 初始化日志格式

    setup_logging()

  

    # 5. 检查配置文件

    # 必须指定配置文件，否则报错

    assert os.path.exists(args.config_file), "Configuration file does not exist!"

    # 6. 【提交任务】

    # submit_jobs 会根据 args 里的设置（是本地跑还是提交到集群），

    # 启动 Trainer 类。

    # name="dinov2:train" 是给任务起个名字，方便在集群队列里看。

    submit_jobs(Trainer, args, name="dinov2:train")

    return 0

  

’‘’标准 Python 写法：如果直接运行这个文件，就执行 main()

if __name__ == "__main__":

    sys.exit(main())


---
---


# 6. 训练脚本train.py详细解释

  

import argparse

import logging

import math

import os

from functools import partial # 用于固定函数的一部分参数

  

‘导入定期保存检查点（Checkpoint）的工具

from fvcore.common.checkpoint import PeriodicCheckpointer

import torch

  

’导入 DINOv2 的数据处理模块

from dinov2.data import SamplerType, make_data_loader, make_dataset

from dinov2.data import collate_data_and_cast, DataAugmentationDINO, CellAugmentationDINO, MaskingGenerator

import dinov2.distributed as distributed # 分布式训练工具

from dinov2.fsdp import FSDPCheckpointer # 专门用于 FSDP（全分片数据并行）的保存工具

from dinov2.logging import MetricLogger  # 记录 Loss 等指标的工具

from dinov2.utils.config import setup    # 读取 yaml 配置文件的工具

from dinov2.utils.utils import CosineScheduler # 余弦退火调度器（用于调整学习率）

  

‘导入模型架构（Student + Teacher 的结构就在这里面）

from dinov2.train.ssl_meta_arch import SSLMetaArch

  

’允许 TF32 格式运算，这是一种在 NVIDIA Ampere 显卡上加速矩阵乘法的技术

‘PyTorch 1.12 默认关了，这里手动开启以加速

torch.backends.cuda.matmul.allow_tf32 = True

  

logger = logging.getLogger("dinov2")

  
  

’定义参数解析器：决定了这个脚本能接收哪些命令行参数

def get_args_parser(add_help: bool = True):

    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)

    # 配置文件路径

    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")

    # 是否禁止从断点恢复（默认是会自动恢复的）

    parser.add_argument(

        "--no-resume",

        action="store_true",

        help="Whether to not attempt to resume from the checkpoint directory. ",

    )

    # 只进行评估，不训练

    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")

    # 指定评估的类型

    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")

    # 允许在命令行末尾直接覆盖配置文件里的参数

    # 比如：python train.py ... train.batch_size=64

    parser.add_argument(

        "opts",

        help="""

Modify config options at the end of the command. For Yacs configs, use

space-separated "PATH.KEY VALUE" pairs.

For python-based LazyConfig, use "path.key=value".

        """.strip(),

        default=None,

        nargs=argparse.REMAINDER,

    )

    # 输出目录

    parser.add_argument(

        "--output-dir",

        "--output_dir",

        default="",

        type=str,

        help="Output directory to save logs and checkpoints",

    )

  

    return parser

  
  

‘构建优化器 (Optimizer)

’也就是决定如何根据梯度来更新模型参数

def build_optimizer(cfg, params_groups):

    # 使用 AdamW 优化器，这是 Transformer 训练的标配

    # betas: 控制动量的参数

    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

  
  

‘构建调度器 (Schedulers)

’这里决定了训练过程中各个超参数（学习率、动量等）随时间如何变化

def build_schedulers(cfg):

    # DINOv2 定义的一“轮”有多长

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

    # 1. 学习率 (LR) 调度配置

    lr = dict(

        base_value=cfg.optim["lr"],           # 最高学习率

        final_value=cfg.optim["min_lr"],      # 最低学习率（训练结束时）

        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH, # 总步数

        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH, # 热身步数（刚开始慢慢学）

        start_warmup_value=0,                 # 热身从 0 开始

    )

    # 2. 权重衰减 (Weight Decay) 调度配置

    wd = dict(

        base_value=cfg.optim["weight_decay"],

        final_value=cfg.optim["weight_decay_end"],

        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,

    )

    # 3. 动量 (Momentum) 调度配置

    # 这是给 Teacher 模型更新用的。

    # 动量越大，Teacher 更新越慢，越稳定。通常是先小后大（0.996 -> 1.0）。

    momentum = dict(

        base_value=cfg.teacher["momentum_teacher"],

        final_value=cfg.teacher["final_momentum_teacher"],

        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,

    )

    # 4. 老师温度 (Teacher Temperature) 调度配置

    # 温度控制 Softmax 输出的平滑程度。

    # 温度越低，输出越尖锐（越自信）。DINO 训练中通常温度会逐渐降低，让老师越来越确信。

    teacher_temp = dict(

        base_value=cfg.teacher["teacher_temp"],

        final_value=cfg.teacher["teacher_temp"],

        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,

        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,

        start_warmup_value=cfg.teacher["warmup_teacher_temp"],

    )

  

    # 实例化 CosineScheduler（余弦退火调度器）

    lr_schedule = CosineScheduler(**lr)

    wd_schedule = CosineScheduler(**wd)

    momentum_schedule = CosineScheduler(**momentum)

    teacher_temp_schedule = CosineScheduler(**teacher_temp)

    last_layer_lr_schedule = CosineScheduler(**lr) # 最后一层单独的 LR

  

    # 这里的逻辑是：让最后一层的学习率在训练初期（freeze_last_layer_epochs 内）保持为 0

    # 目的是为了稳定训练，先冻结最后一层。

    last_layer_lr_schedule.schedule[

        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH

    ] = 0  

  

    logger.info("Schedulers ready.")

  

    return (

        lr_schedule,

        wd_schedule,

        momentum_schedule,

        teacher_temp_schedule,

        last_layer_lr_schedule,

    )

  
  

'应用调度器：把当前步骤算出来的 LR, WD 等赋值给优化器

def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):

    for param_group in optimizer.param_groups:

        is_last_layer = param_group["is_last_layer"]

        lr_multiplier = param_group["lr_multiplier"]

        wd_multiplier = param_group["wd_multiplier"]

        # 设置权重衰减

        param_group["weight_decay"] = wd * wd_multiplier

        # 设置学习率（如果是最后一层，用 last_layer_lr，否则用通用 lr）

        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

  
  

'执行测试/评估逻辑

def do_test(cfg, model, iteration):

    # 取出 Teacher 模型的权重

    # 为什么取 Teacher？因为在自监督学习中，EMA 更新的 Teacher 模型泛化能力通常比 Student 好。

    new_state_dict = model.teacher.state_dict()

  

    if distributed.is_main_process(): # 只在主进程保存，防止多个进程同时写文件冲突

        iterstring = str(iteration)

        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)

        os.makedirs(eval_dir, exist_ok=True)

        # 保存 checkpoint

        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")

        torch.save({"teacher": new_state_dict}, teacher_ckp_path)

  
  

'【核心】执行训练循环

def do_train(cfg, model, resume=False):

    model.train() # 将模型设置为训练模式（启用 Dropout, BatchNorm 更新等）

    inputs_dtype = torch.half # 使用半精度 (FP16)，节省显存

    fp16_scaler = model.fp16_scaler  # 混合精度缩放器

  

    # 1. 准备优化器

    optimizer = build_optimizer(cfg, model.get_params_groups())

    # 2. 准备各种参数调度器

    (

        lr_schedule,

        wd_schedule,

        momentum_schedule,

        teacher_temp_schedule,

        last_layer_lr_schedule,

    ) = build_schedulers(cfg)

  

    # 3. 准备检查点保存工具 (Checkpointer)

    # FSDPCheckpointer 支持大规模模型的切片保存

    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

  

    # 尝试恢复训练（如果之前跑挂了，从断点继续）

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

  

    # 计算总迭代次数

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

  

    # 定期保存器：每隔一定周期保存一次

    periodic_checkpointer = PeriodicCheckpointer(

        checkpointer,

        period=3 * OFFICIAL_EPOCH_LENGTH, # 每 3 个 epoch 存一次

        max_iter=max_iter,

        max_to_keep=3, # 最多保留最近的 3 个存档，节省硬盘

    )

  

    # 4. 数据预处理准备

    img_size = cfg.crops.global_crops_size # 大图尺寸

    patch_size = cfg.student.patch_size    # Patch 尺寸 (如 14)

    n_tokens = (img_size // patch_size) ** 2 # 一张图切成多少个 Token

    # 初始化 Mask 生成器：负责在图片上“挖洞”

    mask_generator = MaskingGenerator(

        input_size=(img_size // patch_size, img_size // patch_size),

        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size, # 最多挖掉 50%

    )

  

    # 初始化数据增强模块

    # DINO 的精髓：一张图变成 -> 2张大图 (Global) + 几张小图 (Local)

    if cfg.train.cell_augmentation:

        data_transform = CellAugmentationDINO(...) # 细胞图像专用增强

    else:

        data_transform = DataAugmentationDINO(

            cfg.crops.global_crops_scale,

            cfg.crops.local_crops_scale,

            cfg.crops.local_crops_number,

            global_crops_size=cfg.crops.global_crops_size,

            local_crops_size=cfg.crops.local_crops_size,

        )

  

    # 定义如何把一堆图片拼成一个 batch

    collate_fn = partial(

        collate_data_and_cast,

        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max, # IBOT 的 mask 比例

        mask_probability=cfg.ibot.mask_sample_probability,

        n_tokens=n_tokens,

        mask_generator=mask_generator,

        dtype=inputs_dtype,

    )

  

    # 5. 准备 DataLoader (数据传送带)

    dataset = make_dataset(

        dataset_str=cfg.train.dataset_path,

        transform=data_transform,

        target_transform=lambda _: (),

    )

    # 使用 Sharded Infinite Sampler，确保分布式训练数据不重复且源源不断

    sampler_type = SamplerType.SHARDED_INFINITE

    data_loader = make_data_loader(

        dataset=dataset,

        batch_size=cfg.train.batch_size_per_gpu,

        num_workers=cfg.train.num_workers, # 多少个 CPU 线程搬运数据

        shuffle=True,

        seed=start_iter,

        sampler_type=sampler_type,

        sampler_advance=0,

        drop_last=True,

        collate_fn=collate_fn,

    )

  

    # 6. 开始训练循环

    iteration = start_iter

  

    logger.info("Starting training from iteration {}".format(start_iter))

    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")

    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)

    header = "Training"

  

    # metric_logger.log_every 会自动打印进度条和日志

    for data in metric_logger.log_every(

        data_loader,

        10, # 每 10 步打印一次

        header,

        max_iter,

        start_iter,

    ):

        current_batch_size = data["collated_global_crops"].shape[0] / 2

        if iteration > max_iter:

            return

  

        # 6.1 应用参数调度

        lr = lr_schedule[iteration]

        wd = wd_schedule[iteration]

        mom = momentum_schedule[iteration]

        teacher_temp = teacher_temp_schedule[iteration]

        last_layer_lr = last_layer_lr_schedule[iteration]

        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

  

        # 6.2 计算 Loss

        optimizer.zero_grad(set_to_none=True) # 清空梯度

        # model.forward_backward 是 SSLMetaArch 里的方法

        # 它包含了：Student 前向，Teacher 前向，算 DINO Loss + IBOT Loss，反向传播

        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

  

        # 6.3 梯度裁剪 (防止梯度爆炸)

        if fp16_scaler is not None:

            if cfg.optim.clip_grad:

                fp16_scaler.unscale_(optimizer) # 先反缩放才能裁剪

                for v in model.student.values():

                    v.clip_grad_norm_(cfg.optim.clip_grad)

            fp16_scaler.step(optimizer) # 更新参数

            fp16_scaler.update() # 更新缩放因子

        else:

            if cfg.optim.clip_grad:

                for v in model.student.values():

                    v.clip_grad_norm_(cfg.optim.clip_grad)

            optimizer.step()

  

        # 6.4 更新 Teacher 模型 (EMA)

        # Teacher 参数 = m * Teacher参数 + (1-m) * Student参数

        model.update_teacher(mom)

  

        # 6.5 分布式日志同步

        if distributed.get_global_size() > 1:

            for v in loss_dict.values():

                torch.distributed.all_reduce(v) # 把所有 GPU 的 Loss 加起来

        # 计算平均 Loss

        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

  

        # 检查是否出现 NaN (数值爆炸)，如果有就报错停止

        if math.isnan(sum(loss_dict_reduced.values())):

            logger.info("NaN detected")

            raise AssertionError

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

  

        # 更新日志记录器

        metric_logger.update(lr=lr)

        metric_logger.update(wd=wd)

        metric_logger.update(mom=mom)

        metric_logger.update(last_layer_lr=last_layer_lr)

        metric_logger.update(current_batch_size=current_batch_size)

        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

  

        # 6.6 定期保存和评估

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:

            do_test(cfg, model, f"training_{iteration}")

            torch.cuda.synchronize()

        # 记录步数，看看是不是该保存 Checkpoint 了

        periodic_checkpointer.step(iteration)

  

        iteration = iteration + 1

    # 训练结束，同步各进程日志

    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

  
  

'' 脚本入口函数

def main(args):

    # 1. 设置配置 (读取 yaml，合并命令行参数)

    cfg = setup(args)

  

    # 2. 实例化模型 SSLMetaArch

    # 这就是 DINOv2 的本体，包含了 Student 和 Teacher 两个网络

    model = SSLMetaArch(cfg).to(torch.device("cuda"))

    model.prepare_for_distributed_training() # 包装成 DDP 或 FSDP 模型

  

    logger.info("Model:\n{}".format(model))

    # 3. 如果只是评估模式

    if args.eval_only:

        iteration = (

            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)

            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)

            .get("iteration", -1)

            + 1

        )

        return do_test(cfg, model, f"manual_{iteration}")

  

    # 4. 进入训练主循环

    do_train(cfg, model, resume=not args.no_resume)

  
  

if __name__ == "__main__":

    # 解析参数并运行 main

    args = get_args_parser(add_help=True).parse_args()

    main(args)

