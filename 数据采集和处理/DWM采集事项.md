# readme
![[Pasted image 20260122122316.png]]

# DWM疑惑解答

### 1. 关于相机序列号 (Camera Serial Number)

- **只有一个相机还需要改吗？**
    
    - **建议修改。** 即使你只插了一个传感器，你的电脑可能还有自带的摄像头（如笔记本内置摄像头）。如果不指定序列号，程序可能会错误地调用你的电脑摄像头，导致程序报错或画面黑屏。
        
    - **修改方法：** 在 `main.py` 文件中，你可以填入序列号（字符串），也可以填数字索引（如 `0` 或 `1`）。如果你确定电脑没其他摄像头，填 `0` 通常也能跑，但填序列号最保险。
        
- **多个相机能采集吗？**
    
    - **理论上可以，但有限制。** 说明书 **2.2.1** 提到，单个传感器需要约 **4.5MB/s** 的带宽。由于 USB 2.0 带宽限制，**不能**通过 USB 集线器（Hub）同时连多个。如果你把它们分别插在电脑主板不同的 USB 接口上（确保带宽足够），通过运行多个程序实例并指定不同序列号，是可以采集的。
        
- **我是用别产品的相机：**
    
    - **特别提醒：** 这个软件（SDK）是专门配合 **DM-Tac 传感器**的内部光学结构和硅胶设计的。如果你使用的是其他品牌的普通相机或网络摄像头，虽然代码可能能打开画面，但**无法计算出触觉数据**（如深度图 Depth、切向力 Shear），因为算法不匹配。
        
- **为什么要修改：** 主要是为了**“身份识别”**。就像给特定的人打电话要拨特定的号码一样，告诉程序去连接哪一个具体的设备。
    

### 2. API 是什么意思？

- **简单理解：** API (Application Programming Interface) 就像是厂家为你准备好的**“控制按钮”**或**“工具箱”**。
    
- **在说明书中：** 参考说明书 **2.4 章节**。厂家已经把复杂的底层代码（比如怎么驱动相机、怎么用显卡算力计算形变）都写好了，封装成了简单的函数。
    
    - 你不需要知道怎么算深度，你只需要写一行代码调用 `sensor.getDepth()`（这就是一个 API），传感器就会直接把计算好的深度数据给你。
        

### 3. 传感器周身的序列与型号 (WS判断)

- **序列号含义：** 传感器周身的字符串通常是**产品序列号 (SN)**，用于唯一标识这台设备。说明书提到，USB 线缆上的黄色标签上也有这个号（如 `M2505150032`）。
    
- **判断是否为 WS 型号：**
    
    - 请看说明书 **1.4.1 外形尺寸** 和 **2.2.3 电气连接**。
        
    - **WS 的最大特征**：它没有像普通鼠标那样的一根圆圆的 USB 线直接连出来，而是通过一根**扁平的柔性排线 (FPC)** 连接到一个外部的小电路板上。
        
    - **WM / WL 型号**：通常机身比较大，线缆是一体的。
        
    - 如果你的图一中看到传感器屁股后面拖着一根**扁扁的软排线**，那它就是 **DM-Tac WS (Small)** 型号。
        
![[480be5b885c2cf5380d586357ccacb37.jpg]]
WM 中等尺寸

### 4. 传感器背部标签解读

这张图片展示的是传感器本体的详细身份信息。

- **DM-Tac WM**:
    
    - 这是**产品型号**。
        
    - **WM** 代表 **Medium (中号)** 传感器 。
        
    - **重要判断**：您之前问是否为 WS (Small) 型号，根据此标签和说明书的尺寸表，这是**中号**，不是小号。且该传感器线缆为圆形黑色线，并非 WS 型号特有的扁平排线 (FPC) ，因此它更坚固，使用时不像 WS 那样需要特别小心排线折断。
        
- **SN: DMVT01WM0693**:
    
    - 这是**设备序列号 (Device Serial Number)**。用于厂家售后追溯生产批次。
        
- **黄色标签 (M2505150198)**:
    
    - **关键信息**：根据说明书，这通常是**相机序列号** 。在运行代码 `main.py` 时，你需要填入代码中的序列号通常是指这个黄色标签上的字符串，或者通过代码自动读取到的 ID。


---

### 5. “软件复位时，传感器表面必须悬空、无任何接触”是什么意思？

- **原理解释：** 视触觉传感器的原理是“通过对比现在的图像和原本的图像来计算形变”。
    
- **归零操作（Tare）：** 当你运行程序或点击复位（Reset）时，机器会把**当前这一瞬间**的状态当作“0”。
    
    - **正确做法：** 传感器硅胶面朝下或朝侧面，**不要**触碰任何物体。此时复位，系统记住了“平整的硅胶”是“0”。
        
    - **错误做法：** 如果你把传感器压在桌子上，然后点击复位。系统会误以为“压扁的样子”才是“0”。当你把传感器拿起来（恢复平整）时，系统反而会觉得传感器“凸出来”了，数据就全乱了。
        


---

### 6. 目录与安装问题 (pip install .)

你提到的图二和图三对应了说明书中的资源结构：

- **U盘根目录(DM-Tac W)解释：**
    
    - 根据说明书 **2.1 包装清单**，U盘里通常有三个东西：
        
        1. **使用说明书 PDF**（文档）。
            
        2. **SDK 文件夹/压缩包**（名字通常叫 `Daimon-Tactile-Publish...`）：这里面是代码。
            
        3. **外壳数模**（名字含 `外壳数模`）：这是 3D 图纸，给你设计夹具用的。
            
- **SDK(Daimon-Tactile-Publish 20250909)解释：**
    
        
    - **操作步骤：**
        
        1. **不要直接在 U 盘里运行。** 先把图三里的 **SDK 文件夹** 整个**复制**到你的电脑硬盘上（比如 D盘或桌面的某个文件夹），这个电脑上的文件夹就是你的**“本地目录”**。
            
        2. 打开命令行终端（CMD 或 PowerShell 或 Linux Terminal）。
            
        3. 使用 `cd` 命令进入到这个**本地目录**（必须进到能看到 `setup.py` 或 `main.py` 的这一层）。
            
        4. **执行命令：** 就在这个路径下输入 `pip install .` （注意最后有个点 `.`，意思是“安装当前目录下的包”）。
            

总结你的操作路径：

U盘 -> 复制 SDK 文件夹到电脑 -> 进文件夹 -> 右键“在终端打开” -> 输入 pip install . -> 安装完后输入 python main.py 启动。


---


### 7. `pip install .` 到底是安装什么？

这个命令是 Python 开发中非常经典的安装方式。

- **`.` (点) 的意思**：代表**“当前目录”**（Current Directory）。也就是你终端光标所在的那个文件夹（图二那个位置）。
    
- **安装了什么？**：
    
    - 当你输入这个命令，`pip`（Python的包管理工具）会立刻在当前目录下寻找一个名为 **`setup.py`**（或者 `pyproject.toml`）的文件。
        
    - **`setup.py` 就是“配方单”**：这个文件里写明了软件的名字、版本，最重要的是写了 **`install_requires`（依赖列表）**。
        
    - **自动下载依赖**：`pip` 读取到 `setup.py` 里写着“我需要 numpy, opencv, cupy”等库时，它就会自动去互联网（PyPI）把这些库下载并安装到你的电脑里。
        
- **SDK文件夹里的 py 文件**：
    
    - SDK 文件夹里有很多 `.py` 文件（源代码），`pip install .` 也会把这些源代码打包，安装到你 Python 的系统库目录（site-packages）里。这样，以后你在任何地方写 `import daimon_sensor`（假设包名是这个），电脑都能找到它，而不需要你每次都跑到这个文件夹里来运行。
        

**总结**：`pip install .` = “请读取当前目录下的 `setup.py` 配方，把这个软件本身装好，顺便把它需要的葱姜蒜（依赖库）也全部自动买回来装好。”


---


### 8. README 解析

你提供的这段 README 是开发者的“快速上手指南”，我们一句句拆解：

- **`# Work with Python 3.8/3.9/3.10/3.11`**
    
    - **含义**：代码支持这些 Python 版本。
        
    - **注意冲突**：这里提到支持 **3.11**，但之前的 PDF 说明书里说“不包括 3.11”。通常 **README 更新**（它是代码里的文件，通常比 PDF 更新），所以大概率是支持 3.11 的。但为了稳妥，如果你遇到问题，优先用 3.8-3.10。
        
- **`Make sure you have cuda toolkit 12.x installed`**
    
    - **含义**：这个传感器需要用显卡加速计算（Cupy库），它默认你是 **CUDA 12** 的显卡驱动环境。
        
    - **`otherwise you might need to modify setup.py`**：这句话非常关键！
        
        - **如果你是 CUDA 11**：你需要打开 `setup.py` 文件（用记事本或代码编辑器），找到里面写着 `cupy-cuda12x` 的地方，把它改成 `cupy-cuda11x`。否则安装会报错或者装错版本。
            
- **`## Install the package` -> `pip install .`**
    
    - **含义**：这就是执行我们在第2点解释的安装步骤。
        
- **`## Plug in the sensor`**
    
    - **含义**：插上传感器的 USB 线。
        
- **`## Run` -> `python main.py`**
    
    - **含义**：安装完成、设备插好后，运行主程序启动画面。




---

### 9. U盘根目录解读

这是 U 盘插入电脑后看到的顶层文件夹，对应说明书中的“包装清单” 。

- **`Daimon-Tactile-Publish 20250909`**:
    
    - 这是**SDK 软件开发包**。就是你需要复制到电脑本地、然后进去执行 `pip install .` 的那个文件夹。
        
- **`DM-Tac W-外壳数模-20250417`**:
    
    - 这是**3D 图纸文件**。里面通常包含 `.stp` 或 `.stl` 格式的 3D 模型。
        
    - **用途**：如果你需要把这个传感器安装到你的机器人末端，你需要根据这个 3D 模型来设计和打印连接支架（法兰） 。
        
- **`DM-Tac W产品说明书 V3.0...pdf`**:
    
    - 这是**用户手册**。即本对话最开始解析的那份 PDF 文档。
        

---

### 10. SDK 文件夹文件功能详解

这个文件夹是核心的开发环境，每一个文件的作用如下：

- **核心运行文件：**
    
    - **`setup.py`**: **安装脚本**。当你输入 `pip install .` 时，电脑实际上就是在运行这个文件。它告诉电脑这个软件叫什么、版本是多少、依赖哪些库（如 cupy, opencv）。
        
    - **`requirements.txt`**: **依赖列表**。纯文本列出了所有需要的第三方库（如 `numpy==1.24.4`）。`setup.py` 会读取它。
        
    - **`main.py`**: **主程序入口**。这是你平时主要运行的文件。你需要打开它修改相机序列号，然后运行它来查看传感器的实时画面（深度图、切向力图等） 。
        
- **代码库文件：**
    
    - **`dmrobotics` (文件夹)**: **源代码核心包**。这里面装着所有具体的算法逻辑（如何从图像解算出力的数学公式）。安装后，你在其他 Python 脚本里 `import dmrobotics` 调用的就是这里面的内容。
        
    - **`dmrobotics.egg-info` (文件夹)**: **安装信息**。这是运行安装命令后自动生成的，记录了安装的元数据，不用管它。
        
- **说明与配置：**
    
    - **`README.md`**: **使用说明文档**。就是你刚才发的那个“How to use”文本文件，用 Markdown 格式写的简易教程。
        
    - **`MANIFEST.in`**: **清单文件**。告诉打包工具在安装时，除了 `.py` 代码外，还需要包含哪些非代码文件（比如许可证、说明书等）。
        
    - **`sdk_log.log`**: **运行日志**。如果程序报错或崩溃了，可以打开这个文件查看具体的错误记录。
        
- **其他：**
    
    - **`.git` / `.gitignore`**: **版本控制文件**。这是开发者用来管理代码版本的，对你作为使用者来说没有用，可以忽略。
        
    - **`build` (文件夹)**: **构建临时目录**。安装过程中生成的临时文件，安装完后可以忽略。


# 注意事项

可以打开设备管理器，找连接的外部设备



# 部署全过程

## 阶段一：宿主机准备：

宿主机准备：

- 插入传感器

- 确认设备节点  ls /dev/video

	这里这里发现D405相机有6个数据流接口，如图所示：
	![[d743248abac000b4f47c8471b3873518.jpg]]



## 阶段二：拉取镜像

- **确认SDK文件夹路径**：/home/shuangmulin/DWM_datacollect/DM-Tac W/Daimon-Tactile-Publish 20250909
- **启动容器命令**：

```
docker run -it --name dm_sensor_env --gpus all --privileged --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "/home/shuangmulin/DWM_datacollect/DM-Tac W/Daimon-Tactile-Publish 20250909:/workspace/sdk" --device=/dev/video2:/dev/video2 --device=/dev/video3:/dev/video3 --device=/dev/video4:/dev/video4 --device=/dev/video5:/dev/video5 --device=/dev/video6:/dev/video6 --device=/dev/video7:/dev/video7 docker.1ms.run/nvidia/cuda:12.6.0-devel-ubuntu22.04 bash
```




## 阶段三：配置环境

- apt-get update  更换软件源并更新系统

- 安转python 3.10
```
apt-get install -y python3.10 python3-pip git 
```

```
# 安装 OpenCV 运行所需的系统库 (核心步骤)
apt-get install -y libgl1-mesa-glx libglib2.0-0
```

```
ln -s /usr/bin/python3.10 /usr/bin/python
```
-  创建软链接 (可选)


## 阶段四：安装SDK

1. 进入SDK路径

2. 执行安装命令 pip install .


## 阶段五：运行程序

python main.py 





---
这里遇到了曲折：
### 问题一：电脑没安装docker

- sudo apt-get update # 更新软件包索引

- sudo apt-get install apt-transport-https ca-certificates curl software-properties-common  # 安装依赖

- curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - # 添加 Docker 官方 GPG 密钥

- sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"  # 添加 Docker 仓库

- sudo apt-get update
- sudo apt-get install docker-ce
- 安装 Docker CE


### 问题二：Docker无法找到NVIDIA GPU驱动程序

提示如下：
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].

	 解决流程：
	 1. 首先检查NVIDAI驱动是否已经安装
	 nvidai-smi  检查nvidia驱动
	 nvcc --version  检查cuda版本
	 
	 2. 安装NVIDIA Container Toolkit
	   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

      curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
      # 添加NVIDIA容器工具包存储库
    
      
	  sudo apt-get update
	  sudo apt-get install -y nvidia-container-toolkit
	  # 更新并安装
	  
	  sudo systemctl restart docker
	  # 重启Docker服务
	  

安装 **NVIDIA Container Toolkit** 是为了让 Docker 容器能够访问和使用宿主机的 NVIDIA GPU。这是一个必要的中间层，原因如下：

#### 为什么需要 NVIDIA Container Toolkit？

##### 1. Docker 默认不支持 GPU

- Docker 最初是为 CPU 应用设计的
    
- 默认 Docker 容器无法直接访问 GPU 硬件
    
- 需要特殊的运行时来桥接 GPU 驱动
    

##### 2. GPU 虚拟化技术

- GPU 不是普通的 PCIe 设备，不能简单映射到容器
    
- NVIDIA 开发了特殊的驱动架构：
    
    - **CUDA** **驱动**：在宿主机上
        
    - **容器运行时**：在 Docker 和驱动之间桥接



### 问题三：cuda版本低

拉取了 docker.1ms.run/nvidia/cuda:12.2.0-devel-ubuntu22.04
cuda12.2跟RXT 5060不适配。显卡太新，cuda版本太低，导致编译/加载的kernel不适配

解决方案：升级cuda版本到12.6
**无法解决** ：硬件版本过高，选择降低换台电脑

**报错信息**：
```
python main.py
Traceback (most recent call last):
  File "/workspace/sdk/main.py", line 9, in <module>
    sensor = Sensor(dev_serial_id)  # serial IDS
  File "/workspace/sdk/dmrobotics/__init__.py", line 11, in __init__
    self.hardware = dmSDK.DMV1(dev_id,KEEP_FPS_Print = KEEP_FPS_Print)
  File "<frozen dmSDK>", line 344, in __init__
  File "cupy/_core/raw.pyx", line 487, in cupy._core.raw.RawModule.get_function
  File "cupy/_core/raw.pyx", line 100, in cupy._core.raw.RawKernel.kernel.__get__
  File "cupy/_core/raw.pyx", line 117, in cupy._core.raw.RawKernel._kernel
  File "cupy/_util.pyx", line 67, in cupy._util.memoize.decorator.ret
  File "cupy/_core/raw.pyx", line 538, in cupy._core.raw._get_raw_module
  File "cupy/_core/core.pyx", line 2359, in cupy._core.core.compile_with_cache
  File "cupy/_core/core.pyx", line 2377, in cupy._core.core.compile_with_cache
  File "/usr/local/lib/python3.10/dist-packages/cupy/cuda/compiler.py", line 536, in _compile_module_with_cache
    return _compile_with_cache_cuda(
  File "/usr/local/lib/python3.10/dist-packages/cupy/cuda/compiler.py", line 607, in _compile_with_cache_cuda
    mod.load(cubin)
  File "cupy/cuda/function.pyx", line 263, in cupy.cuda.function.Module.load
  File "cupy/cuda/function.pyx", line 265, in cupy.cuda.function.Module.load
  File "cupy_backends/cuda/api/driver.pyx", line 226, in cupy_backends.cuda.api.driver.moduleLoadData
  File "cupy_backends/cuda/api/driver.pyx", line 63, in cupy_backends.cuda.api.driver.check_status
cupy_backends.cuda.api.driver.CUDADriverError: CUDA_ERROR_NO_BINARY_FOR_GPU: no kernel image is available for execution on the device什么问题

```




# 最终解决方案：

激活虚拟环境 .venv

在虚拟环境中 pip install .

