docker exec -it [容器名（dinov2)]  /bin/bash进入容器

docker save

docker load 

docker run -d --name dinov3 （容器名字）--gpus all 
-v /data/linmiaoju/miniconda3:/data/miniconda3 
-v /data/linmiaoju/dinov2:/workspace/dinov2 
dinov2-test（镜像）
tail -f /dev/null (运行容器永久）
/bin/bash

docker run -d --name vjepac --gpus all -v $(pwd)/jepa:/workspace/jepa -v /data/xuwenmin/imagenet:/dataset vjepai tail -f /dev/null

docker exec -it dinov2（容器名字） /bin/bash

docker build -t vjepai .
docker build -t dinoi .
docker build -t maei2 .

docker stop dinov2c
docker rm dinov2c

docker cp /home/xuwenmin/dinov2_deploy/wheels/pip/antlr4-python3-runtime-4.9.3.tar.gz dinov2c2:/workspace/wheels/pip/
Successfully copied 119kB to dinov2c2:/workspace/wheels/pip/


free -h 看共享内存

ndivia-smi 看显卡情况

```
docker image

docker ps -as #看存储

docker ps -a #状态

docker rm 。。。 #容器名

docker stop 。。。 #要先停容器，再删

docker rmi 。。。 #镜像

docker system df #看占据

du -sh ~/* |sort -hr | head -n 10
```

```
ssh xuwenmin@172.31.179.162 #连接工作站 powershell
```

```
du -h --max-depth=1 / | sort -hr
```

![[Pasted image 20260303152723.png]]
