

\# Autonomous UAV Target Search



\## 室内无人机自主探索与目标搜索系统



An autonomous indoor UAV exploration and

target search system based on ROS2, PX4,

Gazebo, SLAM and YOLOv11.



!\[ROS2](https://img.shields.io/badge/ROS2-Robotics-blue)

!\[PX4](https://img.shields.io/badge/PX4-Autopilot-blue)

!\[Gazebo](https://img.shields.io/badge/Gazebo-Simulation-orange)

!\[Python](https://img.shields.io/badge/Python-Development-yellow)

!\[YOLO](https://img.shields.io/badge/YOLOv11-Detection-green)



\## 1. 项目简介 | Overview



本项目面向室内未知环境下的无人机自主探索

与目标搜索任务，融合自主定位与建图、路径规划、

目标检测及飞行控制等技术，构建无人机自主搜索系统。



项目基于 ROS2 与 PX4 搭建控制框架，

依托 Gazebo 进行仿真验证，实现室内环境探索、

地图构建、路径规划及目标识别等功能。



\## 2. 核心功能 | Features



\- 基于 SLAM 的室内自主定位与地图构建

\- 基于 A\* 算法的全局路径规划

\- 基于 APF 的局部避障

\- 基于 MPC 的局部运动规划

\- 基于 YOLOv11 的目标检测

\- 基于状态机的自主搜索任务管理

\- ROS2 与 PX4 飞行控制系统通信

\- 多房间环境自主探索与仿真验证



\## 3. 系统架构 | Architecture



!\[System Architecture](docs/architecture.png)



\## 4. 技术栈 | Technology Stack



| Module | Technology |

| --- | --- |

| Robotics Framework | ROS2 |

| Flight Control | PX4 |

| Simulation | Gazebo |

| Mapping | SLAM Toolbox |

| Object Detection | YOLOv11, OpenCV |

| Path Planning | A\*, APF, MPC |

| Programming | Python, C++ |



\## 5. 仿真结果 | Simulation Results



\### 室内环境仿真



!\[Gazebo](assets/gazebo-demo.png)



\### 自主定位与地图构建



!\[SLAM](assets/slam-mapping.png)



\### 目标检测与识别



!\[Detection](assets/target-detection.png)



\## 6. 科研成果 | Research Achievements



\- 省级大学生创新创业训练计划项目

\- 计算机软件著作权

\- 软件著作权登记号：2026SR0599868



\## 7. 环境配置与运行 | Getting Started



具体软件版本、依赖安装及运行指令

以项目实际开发环境为准。



\## 8. 作者与贡献 | Contributors



\- 吉林大学

\- 电子科学与工程学院



项目成员及个人贡献按实际情况填写。



\## 9. 版权说明 | Copyright



项目相关知识产权归相应权利人所有。

第三方组件遵循其各自的许可协议。



