
本文档提供在无外网环境的工作站上，基于 `docker.1ms.run/nvidia/cuda:11.8.0-devel-ubuntu20.04` 镜像部署 DINOv2 模型的完整操作流程。

## 准备工作：目录规划

无论是在有网机器还是离线工作站，请始终保持以下目录结构，以确保脚本和 Dockerfile 能正确找到文件：

```
dinov2_deploy/
├── dinov2/             # (项目源码，需从 GitHub 克隆或下载 zip)
├── wheels/             # (存放所有依赖包的目录)
│   ├── pip/            # (存放 .whl 文件)
│   └── src/            # (存放 .zip 源码包)
├── Dockerfile          # (构建镜像的配置文件)
└── download_deps.sh    # (下载资源的辅助脚本)
```

## 第一阶段：有网环境操作 (资源获取)

在可以连接外网的机器上执行此阶段，目标是下载所有构建镜像所需的依赖。

步骤 1：准备下载脚本

确保 download_deps.sh 位于 dinov2_deploy/ 根目录下，并具有执行权限。

步骤 2：执行下载

运行以下命令自动下载 Miniconda 安装包、PyTorch 离线包及通用 Python 库：

```
chmod +x download_deps.sh
./download_deps.sh
```

_脚本运行结束后，请检查 `wheels/` 目录下是否有 `miniconda.sh`，以及 `wheels/pip/` 下是否有大量 `.whl` 文件。_

**步骤 3：补充特殊依赖 (如果脚本未覆盖)**

- **DINOv2 源码**：如果你还没有代码，请在 `dinov2_deploy/` 目录下克隆项目：
    
    ```
    git clone [https://github.com/facebookresearch/dinov2.git](https://github.com/facebookresearch/dinov2.git)
    ```
    
- **submitit**：脚本会自动下载此源码包到 `wheels/src/`。
    
- **NVIDIA cuml (可选)**：如果脚本中注释掉了 `cuml`，而你需要它，请手动取消注释并运行，或手动下载对应的 whl 文件放入 `wheels/pip/`。
    

步骤 4：打包传输

将整理好的 dinov2_deploy 整个文件夹压缩（例如 tar -czvf dinov2_deploy.tar.gz dinov2_deploy），通过U盘或内网传输到离线工作站。

## 第二阶段：离线工作站操作 (镜像构建)

在无法连接外网的工作站上执行此阶段。

步骤 1：解压资源

将压缩包解压到任意位置：

```
tar -xzvf dinov2_deploy.tar.gz
cd dinov2_deploy
```

步骤 2：确认文件就位

执行 ls -R 确认：

1. `Dockerfile` 在当前目录。
    
2. `wheels/miniconda.sh` 存在。
    
3. `wheels/pip/` 下有 `torch-2.0.0+cu118...whl` 等文件。
    

步骤 3：构建 Docker 镜像

使用提供的 Dockerfile 构建镜像。该过程会自动将 wheels 目录传入容器并在内部进行离线安装。

```
# -t 指定镜像名称和标签 (dinov2-offline:v1)
# . 代表使用当前目录下的 Dockerfile
docker build -t dinov2-offline:v1 .
```

_构建成功后，你会看到 `Successfully built ...` 和 `Successfully tagged ...` 的提示。_

## 第三阶段：运行与验证

步骤 1：启动容器

启动容器并挂载代码目录。建议挂载 dinov2 源码目录，这样你可以在宿主机修改代码，容器内直接运行。

```
# --gpus all: 启用 GPU
# -v $(pwd)/dinov2:/workspace/dinov2: 将当前目录下的 dinov2 源码挂载到容器内的 /workspace/dinov2
docker run --gpus all -it --rm -v $(pwd)/dinov2:/workspace/dinov2 dinov2-offline:v1 /bin/bash
```

步骤 2：验证环境 (在容器内)

进入容器后，默认应该已经激活了 dinov2 环境（由 Dockerfile 中的 .bashrc 配置）。如果没有，手动执行 conda activate dinov2。

验证 PyTorch 和 CUDA：

```
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

_预期输出：`Torch: 2.0.0+cu118, CUDA: True`_

验证 xFormers：

```
python -c "import xformers; print(f'xFormers: {xformers.__version__}')"
```

_预期输出：`xFormers: 0.0.18`_

## 第四阶段：镜像分发 (可选)

如果你需要将配置好的环境给其他人使用。

步骤 1：导出镜像

在工作站宿主机上执行：

```
docker save -o dinov2_offline_image_v1.tar dinov2-offline:v1
```

步骤 2：他人导入

其他人拿到 dinov2_offline_image_v1.tar 后执行：

```
docker load -i dinov2_offline_image_v1.tar
```


docker run -d --name dinov2c2 --gpus all -v $(pwd)/dinov2:/workspace/dinov2 -v /data/xuwenmin/data_upload:/dataset dinov2i tail -f 
/dev/n
03da55fb1b0d57d28617139b4f5193e625951929bba75e959a10688f6ecc1b72
一个镜像建了两个容器

docker run -d --name dinov2c2 --gpus all -v $(pwd)/dinov2:/workspace/dinov2 -v /data/xuwenmin/data_upload:/dataset dinov2i tail -f /dev/null
2c16331fd07e3aaf1019cc5387878e5d4d0c6ac27d8074671f679bf50ef16a75
第二个废了 用第三个

docker run -d --name dinov2c3 --gpus all -v $(pwd)/dinov2:/workspace/dinov2 -v /data/xuwenmin/data_upload:/dataset dinov2i tail -f /dev/null
ea251ed2a39222f963e9bf39fd78d325be6015ed5685e4d55db11a9496a53c85

	docker run -d --memory 64g --memory-swap 64g --shm-size 8g --name dinov2c4 --gpus all -v /home/xuwenmin/dinov2_deploy:/workspace/dinov2 -v /data/xuwenmin/data_upload:/dataset  dinov2i tail -f /dev/null
5de38448cb4e6016ea5fa95420aa5898bd932b00ac837272e6329bee4ab4d9a3


 docker run -d --memory 64g --memory-swap 64g --shm-size 8g --name dinov2c4 --gpus all -v /home/xuwenmin/dinov2_deploy:/workspace/dinov2 -v /data/xuwenmin/data_upload:/dataset  dinov2i tail -f /dev/null
125d6dc4888d89b523b57e88738cf234859bab2f8e51e53f959f806fd5da53ca
最后一个，共享内存变大


![[Pasted image 20251231165836.png]]