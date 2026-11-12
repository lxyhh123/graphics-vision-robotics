# TurtleBot3 Gazebo 仿真配置与实验

本指南采用 Ubuntu 22.04、ROS 2 Humble 和 ROBOTIS 的 Humble 仿真包。按照 [ROS 2 Humble 安装指南](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) 安装 ROS 2，按照 [ROBOTIS PC Setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/) 的 Humble 说明配置 TurtleBot3 依赖和 colcon。仿真包以 [ROBOTIS 仿真手册](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/) 为准。

## 编译

在本仓库根目录打开 Bash。若已安装 Humble 版 turtlebot3_gazebo，则跳过克隆仿真包。

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
git clone -b humble https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git src/turtlebot3_simulations
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 观察激光扫描与里程计

每个终端均从仓库根目录进入并 source 环境。三个终端使用相同的 ROS_DOMAIN_ID，建议在独立仿真环境中运行。

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=23
```

终端 A 启动仿真。

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

终端 B 启动本仓库的观测节点。

```bash
ros2 run course_scan_tools scan_summary --ros-args -p use_sim_time:=true
```

终端 C 查看话题与输出。

```bash
ros2 topic list
ros2 topic info /scan -v
ros2 topic echo /course/front_distance --once
ros2 topic echo /odom --once
```

节点订阅 LaserScan，发布正前方 ±30° 内最小有效距离（米）。无有效采样时输出正无穷。话题名不同时可用 `-r scan:=实际话题名` 重映射。节点仅观测传感器。使用仿真中的键盘遥控改变朝向，比较正前方读数，记录话题名、ROS 版本、两个朝向的读数及 RViz 或 Gazebo 截图。

需要开展二维激光建图时，接续 [ROBOTIS SLAM 仿真教程](https://emanual.robotis.com/docs/en/platform/turtlebot3/slam_simulation/)。视觉里程计和 RGB-D 数据格式练习见[数据集目录](../datasets/README.md)。视觉 SLAM 接入 Unity3D、抓取仿真及实体机器人联调需另行实验验证。
