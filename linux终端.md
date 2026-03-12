
```
clear  #清空终端
chomd +x  main.py  #添加可执行权限
./main.py
```

```
#!/usr/bin/env python3   #系统环境变量中寻找python
```

终端里面默认把命令当做可执行文件执行
```
echo   #输出打印指令  

echo $ros_version 打印出ros的版本

ros_disro ros的版本发行名字

printenv 打印出环境变量列表

export 修改环境变量 （只在当前终端）
```

安装ros2
```
sudo apt update
wget http://fishros.com/install -O
fishros && bash fishros
ros2 #测试 新终端
```


```mermaid.js
gantt
    title VR遥操作开发记录
    dateFormat  MM-DD
    section 初始阶段
    VR串流与vive_ros2编译 :a1, 11-23, 5d
    编译franka_ros2适配层 :after a1, 5d
```