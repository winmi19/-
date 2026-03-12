# DINO 离线环境使用指南

本环境包含 Facebook DINO (v1) 模型所需的完整运行环境。 已适配工作站显卡驱动，基于 CUDA 11.8 + PyTorch 2.1.2。

## 0. 准备工作 (在有网电脑上)

1. 确保目录结构如下：
    
    - `dino_deploy/`
        
        - `download.py`
            
        - `Dockerfile`
            
        - `dino_src/` (这里放 clone 下来的 dino 源代码)
            
2. 运行下载脚本：
    
    ```
    python download.py
    ```
    
    等待 `wheels/pip` 目录下生成一系列 `.whl` 文件。
    
3. 使用 Xftp 将整个 `dino_deploy` 文件夹上传到工作站。
    

## 1. 构建镜像 (在离线工作站上)

进入目录并构建 Docker 镜像：

```
cd dino_deploy
docker build -t dino-offline:v1 .
```

_注意：构建过程会使用目录下的 wheels 自动安装 PyTorch，无需联网。_

## 2. 启动容器

使用以下命令启动容器，并挂载源代码：

```
docker run --gpus all -it --rm \
    --shm-size=8g \
    -v $(pwd)/dino_src:/workspace/dino \
    dino-offline:v1
```

- `--gpus all`: 启用显卡。
    
- `--shm-size=8g`: DINO 训练数据加载需要较大的共享内存，建议加上。
    
- `-v ...`: 将当前目录下的源代码挂载到容器内的 `/workspace/dino`。
    

## 3. 验证环境

进入容器后，运行以下 Python 代码验证 CUDA 是否可用：

```
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"
```

## 4. 开始训练 (示例)

单机多卡训练 (以 4 卡为例)：

```
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch vit_small --data_path /path/to/imagenet/train --output_dir ./output
```


命令指标：
我要跑通 dino的代码，参照V-JEPA的离线部署 要求①有脚本download.py②有dockerfile 拉取的镜像要从docker.1ms.run里面拉③dino_deploy目录下挂载着四样东西 dockerfile ,download.py,生成的驱动wheel，还有源代码 ④最后用xftp来传输 不用压缩镜像 ⑤要生成镜像的 要工作站的人都能用的 ⑥考虑用什么cuda版本 下载的依赖一定一定看看清楚跟你选择的版本兼不兼容 一定一定 这上面是dino的README还有一个代码 修改一下


报错如下：
docker build -t dinoi .
[+] Building 111.1s (6/12)                                                                                                                          docker:default
 => [internal] load build definition from Dockerfile                                                                                                          0.0s
 => => transferring dockerfile: 1.96kB                                                                                                                        0.0s
 => [internal] load metadata for docker.1ms.run/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04                                                                   3.3s
 => [internal] load .dockerignore                                                                                                                             0.0s
 => => transferring context: 2B                                                                                                                               0.0s
 => [1/8] FROM docker.1ms.run/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95            90.2s
 => => resolve docker.1ms.run/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95             0.0s
 => => sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95 743B / 743B                                                                    0.0s
 => => sha256:68075f2beca1cfd3f243ec110000716dff39d895f4d5e0d3faba7ace430f9633 1.43GB / 1.43GB                                                               58.0s
 => => sha256:bd746eb3b9953805ebe644847a227e218b5da775f47007c69930569a75c9ad7d 2.84kB / 2.84kB                                                                0.0s
 => => sha256:d0117ee15b5fd0bbcb42c8fd3e35f9bc0f06fe3a947a4ec240f9b73738c7cf54 17.79kB / 17.79kB                                                              0.0s
 => => extracting sha256:68075f2beca1cfd3f243ec110000716dff39d895f4d5e0d3faba7ace430f9633                                                                    32.1s
 => [internal] load build context                                                                                                                            26.3s
 => => transferring context: 2.41GB                                                                                                                          26.2s
 => ERROR [2/8] RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list &&     sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /et  17.6s
------                                                                                                                                                             
 > [2/8] RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list &&     sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list &&     apt-get update && apt-get install -y --no-install-recommends     python3.9     python3.9-dev     python3-pip     git     wget     libgl1-mesa-glx     libglib2.0-0     && rm -rf /var/lib/apt/lists/*:                                                                                                                     
5.554 Get:1 http://mirrors.aliyun.com/ubuntu jammy InRelease [270 kB]                                                                                              
5.946 Get:3 http://mirrors.aliyun.com/ubuntu jammy-updates InRelease [128 kB]                                                                                      
6.226 Get:4 http://mirrors.aliyun.com/ubuntu jammy-backports InRelease [127 kB]
6.360 Get:5 http://mirrors.aliyun.com/ubuntu jammy-security InRelease [129 kB]
6.700 Get:6 http://mirrors.aliyun.com/ubuntu jammy/multiverse amd64 Packages [266 kB]
6.970 Get:7 http://mirrors.aliyun.com/ubuntu jammy/main amd64 Packages [1792 kB]
7.352 Get:8 http://mirrors.aliyun.com/ubuntu jammy/universe amd64 Packages [17.5 MB]
9.796 Get:9 http://mirrors.aliyun.com/ubuntu jammy/restricted amd64 Packages [164 kB]
9.984 Get:10 http://mirrors.aliyun.com/ubuntu jammy-updates/multiverse amd64 Packages [69.2 kB]
10.05 Get:11 http://mirrors.aliyun.com/ubuntu jammy-updates/restricted amd64 Packages [6222 kB]
10.97 Get:12 http://mirrors.aliyun.com/ubuntu jammy-updates/main amd64 Packages [3876 kB]
11.39 Get:2 https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64  InRelease [1581 B]
11.57 Get:13 http://mirrors.aliyun.com/ubuntu jammy-updates/universe amd64 Packages [1596 kB]
11.61 Get:14 https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64  Packages [2153 kB]
11.82 Get:15 http://mirrors.aliyun.com/ubuntu jammy-backports/main amd64 Packages [83.9 kB]
12.00 Get:16 http://mirrors.aliyun.com/ubuntu jammy-backports/universe amd64 Packages [35.2 kB]
12.06 Get:17 http://mirrors.aliyun.com/ubuntu jammy-security/main amd64 Packages [3539 kB]
12.59 Get:18 http://mirrors.aliyun.com/ubuntu jammy-security/multiverse amd64 Packages [60.9 kB]
12.68 Get:19 http://mirrors.aliyun.com/ubuntu jammy-security/universe amd64 Packages [1290 kB]
12.90 Get:20 http://mirrors.aliyun.com/ubuntu jammy-security/restricted amd64 Packages [6008 kB]
14.09 Fetched 45.3 MB in 14s (3293 kB/s)
14.09 Reading package lists...
15.42 Reading package lists...
16.64 Building dependency tree...
16.89 Reading state information...
17.01 E: Unable to locate package python3.9-dev
17.01 E: Couldn't find any package by glob 'python3.9-dev'
17.01 E: Couldn't find any package by regex 'python3.9-dev'
------
Dockerfile:16
--------------------
  15 |     # 这里的 apt 源可以根据工作站情况替换，默认使用官方或阿里源
  16 | >>> RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
  17 | >>>     sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
  18 | >>>     apt-get update && apt-get install -y --no-install-recommends \
  19 | >>>     python3.9 \
  20 | >>>     python3.9-dev \
  21 | >>>     python3-pip \
  22 | >>>     git \
  23 | >>>     wget \
  24 | >>>     libgl1-mesa-glx \
  25 | >>>     libglib2.0-0 \
  26 | >>>     && rm -rf /var/lib/apt/lists/*
  27 |     
--------------------
ERROR: failed to solve: process "/bin/sh -c sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list &&     sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list &&     apt-get update && apt-get install -y --no-install-recommends     python3.9     python3.9-dev     python3-pip     git     wget     libgl1-mesa-glx     libglib2.0-0     && rm -rf /var/lib/apt/lists/*" did not complete successfully: exit code: 100


![[Pasted image 20251129112306.png]]



docker run -d --name dinoc --gpus all -v $(pwd)/dino:/workspace/dino -v /data/xuwenmin/imagenet:/dataset dinoi tail -f /dev/null
f8ac8c7eda2958c0586ad656d90bc1470742212aaebb936705de91e2aa1828a1


main_dino.py

