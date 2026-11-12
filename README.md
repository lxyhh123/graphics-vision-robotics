# 图形视觉机器人课程群实践

按课程群实验指导书修订稿组织。四项实验各 4 学时，共 16 学时，输入数据由程序生成。

| 实验 | 任务与提交物 |
| --- | --- |
| [E1 三维场景搭建与变换](labs/E1/experiment1.py) | 比较层次变换与投影方式，提交对照图、坐标 JSON、手算和 AI 建议核验 |
| [E2 软件光栅化与光照材质设计](labs/E2/experiment2.py) | 比较覆盖、遮挡、光照分量与材质，提交图像、参数和 AI 建议核验 |
| [E3 带孔点云处理与曲面重建](labs/E3/experiment3.py) | 完成去噪、采样、法线估计与带孔三角化，提交阶段图、网格与误差对照 |
| [E4 点云刚体配准与三维模型融合](labs/E4/experiment4.py) | 完成 ICP 和体素融合，提交共同评价集指标、失败案例和融合图 |

## 在 Windows PowerShell 中运行

从仓库根目录执行。每次调整参数时指定独立的输出目录。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe labs/E1/experiment1.py --output outputs/e1
.\.venv\Scripts\python.exe labs/E2/experiment2.py --stage all --materials labs/E2/materials_example.json --output outputs/e2
.\.venv\Scripts\python.exe labs/E3/experiment3.py --output outputs/e3
.\.venv\Scripts\python.exe labs/E4/experiment4.py --mode all --output outputs/e4
```

E4 提供学生代码骨架。按指导书完成 `icp` 与 `voxel_fuse` 后，再执行 E4 的批量对照命令。

## ROS 与课程资源

- [TurtleBot3 Gazebo 配置及观测实验](docs/gazebo.md)
- [ROS 2 激光扫描观测功能包](ros2_ws/src/course_scan_tools)
- [标准数据集下载与教学样例](datasets/README.md)
- [学生作品分享与提交](projects/README.md)
