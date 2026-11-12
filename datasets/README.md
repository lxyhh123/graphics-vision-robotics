# 数据集

| 数据 | 获取方式 | 教学任务 |
| --- | --- | --- |
| TUM RGB-D SLAM Benchmark | [TUM 官方下载页](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)，选 `freiburg1_xyz` | 检查 RGB、深度与轨迹时间戳，理解关联与轨迹评估 |
| 前向激光扫描教学样例 | [scan_sample.csv](scan_sample.csv) | 用有效量程 0.12–3.5 米和前向 ±30° 计算最近距离，期望值 0.8 米 |
| E3、E4 合成点云 | 按[实验命令](../README.md)生成 | 对照点云处理与配准误差，保留输入参数及随机种子 |

TUM 文件从原作者站点下载。解压后先检查 `rgb.txt`、`depth.txt`、`groundtruth.txt`。前两份索引记录时间戳和相对图像路径，轨迹文件提供带时间戳的参考位姿。任选一帧 RGB，按最接近的时间戳找到对应深度帧，记录两者差值并说明关联阈值。使用或转发数据时按原站点要求引用并标注许可。

CSV 为课程自制数据，角度单位为度，距离单位为米。它用于核验计算，Gazebo 现场采样请随实验报告另行记录。
