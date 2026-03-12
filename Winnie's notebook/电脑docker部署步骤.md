更新软件包列表
sudo apt update

安装 Docker
sudo apt install docker.io

启动 Docker 服务
sudo systemctl start docker

设置 Docker 开机自启
sudo systemctl enable docker

添加当前用户到 docker 组（避免每次都要加 sudo）
sudo usermod -aG docker $USER

重新登录或重启使组权限生效



拉取1ms镜像源
docker pull docker.1ms.run/nvidia/cuda:12.2.0-devel-ubuntu22.04


运行挂载命令
docker run -it --name dm_sensor_env --gpus all --privileged --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v "/home/shuangmulin/DWM_datacollect/DM-Tac W/Daimon-Tactile-Publish 20250909:/workspace/sdk" --device=/dev/video2:/dev/video2 --device=/dev/video3:/dev/video3 --device=/dev/video4:/dev/video4 --device=/dev/video5:/dev/video5 --device=/dev/video6:/dev/video6 --device=/dev/video7:/dev/video7 docker.1ms.run/nvidia/cuda:12.2.0-devel-ubuntu22.04 bash


