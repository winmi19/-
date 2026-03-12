#MAE 大模型离线环境部署详细指南

本文档参照 DINOv2 部署流程，针对 **MAE (Masked Autoencoders)** 的特殊依赖（特别是 `timm` 版本兼容性）进行了深度定制。本方案适用于无法连接外网的服务器环境。

## 准备工作：目录规划

请在**有外网的电脑**上，严格按照以下结构整理文件。这非常重要，否则 Docker 构建会失败。

```
mae_deploy/                <-- 总文件夹
├── mae/                   <-- MAE 项目源码 (从 GitHub 克隆或解压 zip)
├── wheels/                <-- 依赖包存放目录 (由脚本生成)
│   ├── pip/               <-- 存放 torch, timm 等 .whl 文件
│   ├── src/               <-- 存放源码包 (可选)
│   └── miniconda.sh       <-- Conda 安装程序
├── Dockerfile             <-- (本文提供的 Dockerfile)
└── download.py            <-- (本文提供的下载脚本)
```

## 第一阶段：有网环境操作（下载资源）

在您的个人电脑（Windows/Mac/Linux）上操作。

### 步骤 1：获取 MAE 源码

将您手头的 `mae` 代码（包含 `main_pretrain.py` 等文件）放入 `mae_deploy/mae/` 目录中。

### 步骤 2：下载离线依赖

1. 确保您的电脑安装了 Python。
    
2. 将本文提供的 `download.py` 保存到 `mae_deploy/` 根目录。
    
3. 运行脚本：
    
    ```
    python download.py
    ```
    
    > **注意**：脚本会自动从国内镜像源（清华源）下载 PyTorch 2.0、Miniconda 以及 MAE 必须的 `timm==0.3.2`。
    

### 步骤 3：打包传输

检查 `mae_deploy/wheels/pip` 目录下是否有大量的 `.whl` 文件。确认无误后，将整个 `mae_deploy` 文件夹压缩打包：

```
# Windows 上可以使用压缩软件打包为 mae_deploy.zip 或 .tar.gz
# Mac/Linux:
tar -czvf mae_deploy.tar.gz mae_deploy
```

**使用 U 盘或内网跳转机将压缩包传输到离线服务器。**

## 第二阶段：离线服务器操作（构建镜像）

在无法联网的 GPU 服务器上操作。

### 步骤 1：解压资源

```
tar -xzvf mae_deploy.tar.gz
cd mae_deploy
```

### 步骤 2：构建 Docker 镜像

确保当前目录下有 `Dockerfile` 和 `wheels` 文件夹。执行构建命令：

```
# -t 指定镜像名称为 mae-offline:v1
# 注意命令最后有一个点 "."，代表使用当前目录上下文
docker build -t mae-offline:v1 .
```

> **重点说明**：构建过程中，Docker 会自动执行一个 `sed` 命令来修复 `timm 0.3.2` 的 Bug。这是因为 MAE 强制要求旧版 timm，而旧版 timm 在新版 PyTorch 下会报错。我们的 Dockerfile 已经自动为您处理了这个棘手的兼容性问题。

## 第三阶段：运行与验证

### 步骤 1：启动容器

我们将代码目录挂载到容器内，这样您可以随时修改代码而无需重建镜像。

```
# --gpus all: 启用所有显卡
# --ipc=host: 解决多卡训练共享内存不足的问题 (强烈建议加上)
# -v: 挂载目录
docker run --gpus all --ipc=host -it --rm \
    -v $(pwd)/mae:/workspace/mae \
    mae-offline:v1 /bin/bash
```

### 步骤 2：验证环境

进入容器后，终端前缀应显示 `(mae)`，表示环境已激活。

1. **验证 PyTorch 和 CUDA**:
    
    ```
    python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    ```
    
    _预期输出_: `Torch: 2.0.0+cu117` (或类似), `CUDA: True`
    
2. **验证 MAE 核心依赖 timm**:
    
    ```
    python -c "import timm; print(f'Timm: {timm.__version__}')"
    ```
    
    _预期输出_: `Timm: 0.3.2`
    
    > 如果这一步没有报错，说明我们的自动修复补丁生效了。如果报错 `ImportError: cannot import name 'container_abcs'`, 请检查 Dockerfile 构建日志。
    

### 步骤 3：跑通代码 (测试)

尝试运行预训练脚本的一个小测试（假设您在 `/workspace/mae` 目录下）：

```
cd /workspace/mae
# 运行一个小批量的测试 (假设您已经准备好了 ImageNet 数据集，或者先跑通代码逻辑)
# 如果没有数据集，代码会报错找不到文件，但可以验证依赖加载是否正常
python main_pretrain.py --model mae_vit_base_patch16 --batch_size 2 --epochs 1 --data_path /tmp/null
```

## 常见问题排查

1. **依赖报错 `GLIBC_XXX not found`**:
    
    - 原因：您下载 `whl` 包的电脑系统版本太旧，或者太新导致下载的包与服务器 Docker 基础镜像（Ubuntu 22.04）不匹配。
        
    - 解决：`download.py` 中使用了 `--platform manylinux1_x86_64` 参数尽量规避此问题。如果仍出现，建议在与服务器系统架构相似的电脑上运行下载脚本。
        
2. **`timm` 版本错误**:
    
    - MAE 代码中明确写了 `assert timm.__version__ == "0.3.2"`。请不要随意升级 `timm`，否则需要修改大量源码。请信任本文 Dockerfile 中的自动修复逻辑。



https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
writing image sha256:b3a3bdf5f0bf45dfa904b8d036bf9599e70f989f9cd96aaee5ce722825cd4984                         0.0s 
 => => naming to docker.io/library/maei


docker run -d --name maec --gpus all -v $(pwd)/mae:/workspace/mae -v /data/xuwenmin/data_upload:/dataset maei tail -f /dev/null  /bin/bash


docker run -d --name maec --gpus all -v $(pwd)/mae:/workspace/mae -v /data/xuwenmin/data_upload:/dataset maei tail -f /dev/null  /bin/bash
98fb33dfaf3f0825c10deb131c541a97986a0118a6ff301744530685f72f1a20(路径错误)

docker run -d --name maec --gpus all -v $(pwd)/mae:/workspace/mae -v /data/xuwenmin/data_upload:/dataset maei tail -f /dev/null  /bin/bash
5fc6b5e96ed4c51bdf64ff85dc010d2f859c492ae1136cc22cb2384b9af341b6


后续测试
问gemini3:
#!/usr/bin/env python3
"""
MAE 环境测试脚本 - 修复版
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse
def test_environment():
    """测试基础环境"""
    print("=" * 50)
    print("MAE 环境测试")
    print("=" * 50)    
    # 1. 检查Python和PyTorch
    print("1. 检查Python和PyTorch环境...")
    print(f"   Python版本: {sys.version}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")    
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name()}")
        print(f"   GPU数量: {torch.cuda.device_count()}")    
    # 2. 检查关键包
    print("\n2. 检查关键依赖包...")
    try:
        import timm
        print(f"   ✓ timm版本: {timm.__version__}")
    except ImportError:
        print("   ✗ timm未安装")
        return False    
    return True
def test_model_loading():
    """测试模型加载"""
    print("\n3. 测试模型加载...")
    try:
        # 使用绝对导入
        sys.path.insert(0, '/workspace/mae')
        from models_mae import mae_vit_base_patch16        
        model = mae_vit_base_patch16()
        print("   ✓ MAE ViT-Base 模型创建成功")        
        # 测试模型参数
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数量: {num_params:,}")        
        return model
    except Exception as e:
        print(f"   ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
def test_data_loading(dataset_path):
    """测试数据加载"""
    print(f"\n4. 测试数据加载 from {dataset_path}...")    
    # 查找测试图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_image = None    
    if os.path.isfile(dataset_path):
        test_image = dataset_path
    else:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break    
    if not test_image:
        print("   ⚠️ 未找到测试图片，使用随机数据")
        return None    
    print(f"   找到测试图片: {test_image}")    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])    
    try:
        img = Image.open(test_image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        print(f"   ✓ 图片加载成功, 形状: {img_tensor.shape}")
        return img_tensor
    except Exception as e:
        print(f"   ✗ 图片加载失败: {e}")
        return None
def test_inference(model, input_data, use_cuda=True):
    """测试推理"""
    print("\n5. 测试模型推理...")    
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    print(f"   使用设备: {device}")    
    model = model.to(device)    
    if input_data is None:
        print("   使用随机数据进行测试...")
        input_data = torch.randn(2, 3, 224, 224).to(device)
    else:
        input_data = input_data.to(device)    
    model.eval()    
    try:
        with torch.no_grad():
            print("   进行前向传播...")
            loss, pred, mask = model(input_data)            
            print(f"   ✓ 推理成功!")
            print(f"   输入形状: {input_data.shape}")
            print(f"   损失值: {loss.item():.4f}")
            print(f"   预测形状: {pred.shape}")            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                print(f"   GPU内存占用: {memory_allocated:.1f} MB")            
        return True        
    except Exception as e:
        print(f"   ✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
def main():
    parser = argparse.ArgumentParser(description='MAE环境测试')
    parser.add_argument('--data_path', type=str, default='/dataset', 
                       help='数据路径，可以是图片文件或目录')
    parser.add_argument('--cpu', action='store_true', 
                       help='强制使用CPU（即使CUDA可用）')
    args = parser.parse_args()    
    print("开始MAE环境测试...")    
    # 1. 测试环境
    if not test_environment():
        print("\n❌ 环境测试失败!")
        sys.exit(1)    
    # 2. 测试模型加载
    model = test_model_loading()
    if model is None:
        print("\n❌ 模型加载失败!")
        sys.exit(1)    
    # 3. 测试数据加载
    input_data = test_data_loading(args.data_path)    
    # 4. 测试推理
    success = test_inference(model, input_data, use_cuda=not args.cpu)    
    # 5. 总结
    print("\n" + "=" * 50)
    if success:
        print("🎉 MAE 环境测试通过！")
    else:
        print("❌ MAE 环境测试失败！")
    print("=" * 50)
if __name__ == "__main__":
    main()这是别人给我的test_run.py 没跑通 报错如下vim test_run.py
(mae) root@9296b04d137c:/workspace/mae# python test_run.py --data_path /dataset
开始MAE环境测试...
==================================================
MAE 环境测试
==================================================
1. 检查Python和PyTorch环境...
   Python版本: 3.8.20 (default, Oct  3 2024, 15:24:27) 
[GCC 11.2.0]
   PyTorch版本: 1.12.1+cu113
   CUDA可用: True
   GPU设备: NVIDIA GeForce RTX 4090
   GPU数量: 2
2. 检查关键依赖包...
   ✗ timm未安装
❌ 环境测试失败! ①mae需要装这个依赖吗 ②可以直接用官方的demo吗 上面有demo ③你推荐哪个



docker run -d --name maec2 --gpus all -v $(pwd)/mae:/workspace/mae -v $(pwd)/wheels:/workspace/wheels -v /data/xuwenmin/imagenet:/dataset dinoi tail -f /dev/null
ac02e4296e512f84f849de16fc61ab09f8a0f4173294ac67d8228226712868dd
(用dino的cuda11.8来建mae镜像)