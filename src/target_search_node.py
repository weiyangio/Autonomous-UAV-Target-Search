#!/usr/bin/env python3
"""
简化重写版 target_search 节点
目标：
1) 稳定 PX4 Offboard 速度控制
2) 稳健高度估计（优先 dist_bottom，抗突变）
3) 深度学习目标检测（YOLO）
4) 室内自主搜索（local/apf 优先 + lidar fallback）
"""

import atexit
import concurrent.futures
import csv
import json
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw, ImageFont
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleAttitude, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
import tf2_ros
from ultralytics import YOLO

from indoor_search.msg import ApfCommand, LocalCommand, SearchPath as _SearchPath
from safety_monitor import SafetyMonitor
from target_confirmation import TargetConfirmation


# ============================================================================
# TargetKalmanFilter
# ============================================================================
# 轻量级目标位置 Kalman 滤波器（基于 cv2.KalmanFilter，4状态·2测量）
# 状态向量: [cx_norm, cy_norm, vx, vy]  (归一化坐标 -0.5~0.5，速度単位/帧)
# 测量向量: [cx_norm, cy_norm]
# 用途：在目标短暂遮挡/帧率不足时，通过运动模型预测目标位置，
#        维持视觉伺服的连续性，防止短时丢失即触发 lost_abort。
# ============================================================================
class TargetKalmanFilter:
    """
    目标位置 Kalman 滤波器（4状态/2测量）。
    状态 = [cx_norm, cy_norm, vx, vy]，测量 = [cx_norm, cy_norm]。
    cx/cy_norm 为归一化坐标（图像中心为 0，范围约 -0.5~0.5）。
    """

    def __init__(self,
                 proc_noise: float = 0.02,
                 meas_noise: float = 0.05,
                 init_err_cov: float = 1.0) -> None:
        """
        参数:
          proc_noise  — 过程噪声方差（越大=速度越快变化，state 越不信任预测）
          meas_noise  — 测量噪声方差（越大=越信任历史预测而非当前测量）
          init_err_cov— 初始误差协方差（影响滤波器收敛速度）
        """
        # 4 状态（cx,cy,vx,vy）/ 2 测量（cx,cy）
        self._kf = cv2.KalmanFilter(4, 2)

        # 状态转移矩阵（匀速运动：cx+=vx*dt，等 dt=1 步长时：cx[t+1]=cx[t]+vx[t]）
        self._kf.transitionMatrix = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # 测量矩阵（只观测 cx, cy）
        self._kf.measurementMatrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # 过程噪声
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * proc_noise

        # 测量噪声
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise

        # 初始后验误差协方差
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * init_err_cov

        # 是否已初始化（接收到首次测量）
        self.initialized: bool = False

        # 跟踪置信度 [0.0~1.0]：有测量=升，无测量=衰减
        self.conf: float = 0.0

        # 上一次有测量更新的 monotonic 时间
        self.last_meas_t: float = 0.0

        # 上一次 predict/update 的 monotonic 时间（控制 dt）
        self._last_tick_t: float = 0.0

    # ------------------------------------------------------------------
    def init(self, cx_norm: float, cy_norm: float) -> None:
        """用首次测量初始化滤波器状态。"""
        import time as _kf_t
        self._kf.statePost = np.array(
            [[cx_norm], [cy_norm], [0.0], [0.0]], dtype=np.float32
        )
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        self.initialized = True
        self.conf = 1.0
        self.last_meas_t = _kf_t.monotonic()
        self._last_tick_t = self.last_meas_t

    # ------------------------------------------------------------------
    def predict(self, dt: float = 1.0) -> tuple:
        """
        向前预测 dt 个步长，返回 (cx_norm, cy_norm, vx, vy)。
        若尚未初始化，返回 (0,0,0,0)。
        """
        if not self.initialized:
            return 0.0, 0.0, 0.0, 0.0
        # 根据 dt 动态调整 transitionMatrix 的速度分量
        self._kf.transitionMatrix[0, 2] = float(dt)
        self._kf.transitionMatrix[1, 3] = float(dt)
        pred = self._kf.predict()
        return (float(pred[0, 0]), float(pred[1, 0]),
                float(pred[2, 0]), float(pred[3, 0]))

    # ------------------------------------------------------------------
    def update(self, cx_norm: float, cy_norm: float) -> tuple:
        """
        用测量值更新滤波器，返回滤波后的 (cx_norm, cy_norm)。
        若尚未初始化则先执行 init。
        """
        import time as _kf_t
        if not self.initialized:
            self.init(cx_norm, cy_norm)
            return cx_norm, cy_norm
        meas = np.array([[cx_norm], [cy_norm]], dtype=np.float32)
        corrected = self._kf.correct(meas)
        self.conf = min(1.0, self.conf + 0.3)
        self.last_meas_t = _kf_t.monotonic()
        return float(corrected[0, 0]), float(corrected[1, 0])

    # ------------------------------------------------------------------
    def decay(self, dt_sec: float) -> None:
        """
        无测量时衰减置信度（dt_sec 为距上次 tick 的秒数）。
        置信度线性衰减：0.8/s，低于 0.05 视为完全失去。
        """
        self.conf = max(0.0, self.conf - 0.8 * dt_sec)

    # ------------------------------------------------------------------
    def get_conf(self) -> float:
        """返回当前跟踪置信度 [0, 1]。"""
        return float(self.conf)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """重置滤波器，清除历史状态。"""
        self.initialized = False
        self.conf = 0.0
        self.last_meas_t = 0.0
        self._last_tick_t = 0.0


class TargetSearch(Node):
    def __init__(self) -> None:
        super().__init__("target_search")
        self.bridge = CvBridge()

        self.fmu_prefix = self.declare_parameter("fmu_prefix", "/fmu").value

        # 地图边界：通过ROS参数可覆盖
        # maze.world 墙体范围 ±6m，初始设为 -8~+8 覆盖全迷宫
        # 这样墙外未被激光扫到的区域会显示为灰色（unknown），与实际情况一致
        _bx0 = float(self.declare_parameter("map_bound_xmin", -8.0).value)
        _bx1 = float(self.declare_parameter("map_bound_xmax",  8.0).value)
        _by0 = float(self.declare_parameter("map_bound_ymin", -8.0).value)
        _by1 = float(self.declare_parameter("map_bound_ymax",  8.0).value)
        self.bounds = np.array([[_bx0, _bx1], [_by0, _by1], [-2.5, -0.05]], dtype=float)
        self._bounds_dirty = False  # 是否需要重建地图grid
        self.max_speed = float(self.declare_parameter("max_speed", 0.8).value)
        self.max_acc = float(self.declare_parameter("max_acc", 0.6).value)

        self.alt_target_z = float(self.declare_parameter("alt_target_z", -1.5).value)
        self.alt_takeoff_band = float(self.declare_parameter("alt_takeoff_band", 0.12).value)
        self.alt_takeoff_hold_time = float(self.declare_parameter("alt_takeoff_hold_time", 1.0).value)
        self.alt_kp = float(self.declare_parameter("alt_kp", 0.7).value)   # 降低避免z超调
        self.alt_kd = float(self.declare_parameter("alt_kd", 0.35).value)  # 增加阻尼抑制超调
        self.alt_max_climb_speed = float(self.declare_parameter("alt_max_climb_speed", 0.30).value)
        self.alt_max_descend_speed = float(self.declare_parameter("alt_max_descend_speed", 0.25).value)  # 降低下降速度减少z超调

        self.z_jump_reject_m = float(self.declare_parameter("z_jump_reject_m", 1.8).value)
        self.z_jump_reject_dt = float(self.declare_parameter("z_jump_reject_dt", 0.4).value)
        self.z_jump_reinit_hits = int(self.declare_parameter("z_jump_reinit_hits", 3).value)
        self.z_recovery_rate_mps = float(self.declare_parameter("z_recovery_rate_mps", 1.6).value)
        self.z_positive_clamp = float(self.declare_parameter("z_positive_clamp", 0.30).value)
        # XY跳变过滤：仅拒绝真实的大跳变（>2m），EKF高频抖动(2ms/1.6cm=9.3m/s)是假阳性
        # dt窗口设为0.1s，大于该窗口的连续运动不算"跳变"
        self.xy_jump_reject_dist = float(self.declare_parameter("xy_jump_reject_dist", 2.0).value)
        self.xy_jump_reject_dt = float(self.declare_parameter("xy_jump_reject_dt", 0.1).value)
        self.xy_jump_reject_speed = float(self.declare_parameter("xy_jump_reject_speed", 15.0).value)
        self.pose_bounds_guard_margin = float(self.declare_parameter("pose_bounds_guard_margin", 1.0).value)

        self.safety_max_pose_age_sec = float(self.declare_parameter("safety_max_pose_age_sec", 1.5).value)
        self.safety_pose_age_ceiling_sec = float(self.declare_parameter("safety_pose_age_ceiling_sec", 3.5).value)
        self.safety_pose_age_adaptive_mult = float(self.declare_parameter("safety_pose_age_adaptive_mult", 4.0).value)
        self.pose_dt_ema_alpha = float(self.declare_parameter("pose_dt_ema_alpha", 0.15).value)

        self.prefer_dist_bottom_altitude = bool(self.declare_parameter("prefer_dist_bottom_altitude", True).value)
        self.dist_bottom_alpha = float(self.declare_parameter("dist_bottom_alpha", 0.25).value)
        self.dist_bottom_min = float(self.declare_parameter("dist_bottom_min", 0.08).value)
        self.dist_bottom_max = float(self.declare_parameter("dist_bottom_max", 4.0).value)

        self.lidar_topic = self.declare_parameter("lidar_topic", "/px4/lidar_controller/out").value
        self.lidar_fallback_topic = self.declare_parameter("lidar_fallback_topic", "/px4/lidar_controller/out").value
        self.lidar_timeout = float(self.declare_parameter("lidar_timeout", 1.0).value)
        # LiDAR安装旋转补偿：iris.sdf中rplidar_link pose=(0,0,0.1, 0,0,0)，yaw=0
        # 即传感器x轴与base_link.x轴平行，angle=0指向机头前方(+X_body)
        # 无需旋转补偿，lidar_mount_yaw_deg=0.0
        self.lidar_mount_yaw_deg = float(self.declare_parameter("lidar_mount_yaw_deg", 0.0).value)
        self.lidar_sector_percentile = float(self.declare_parameter("lidar_sector_percentile", 20.0).value)
        self.lidar_sector_ema_alpha = float(self.declare_parameter("lidar_sector_ema_alpha", 0.35).value)
        self.lidar_sector_min_points = int(self.declare_parameter("lidar_sector_min_points", 4).value)
        self.obstacle_slowdown_start = float(self.declare_parameter("obstacle_slowdown_start", 1.2).value)
        self.obstacle_cmd_stop_dist = float(self.declare_parameter("obstacle_cmd_stop_dist", 0.35).value)  # 提高：距墙0.35m强制停止，避免贴墙滑行
        # ★[J3] escape_speed 0.10→0.20：原0.10 m/s逃逸速度过慢（仅略高于零速），
        #   CORNER-TRAP期间需要足够推力脱离墙角，0.20 m/s 兼顾安全与有效性
        self.obstacle_escape_speed = float(self.declare_parameter("obstacle_escape_speed", 0.20).value)
        self.lidar_min_valid_dist = float(self.declare_parameter("lidar_min_valid_dist", 0.15).value)  # 过滤机体反射噪声
        self.map_max_tilt_deg = float(self.declare_parameter("map_max_tilt_deg", 20.0).value)
        self.map_max_beam_dir_z = float(self.declare_parameter("map_max_beam_dir_z", 0.25).value)
        self.map_max_beam_z_dev = float(self.declare_parameter("map_max_beam_z_dev", 0.6).value)

        self.rgb_topic = self.declare_parameter("rgb_topic", "/px4/rgb_camera/image_raw").value
        self.rgb_fallback_topic = self.declare_parameter("rgb_fallback_topic", "/px4/camera/color/image_raw").value

        self.model_path = self.declare_parameter("model_path", "/home/ubuntu/PX4-Autopilot/yolo11n.pt").value
        self.detect_conf = float(self.declare_parameter("detect_conf", 0.5).value)
        self.detect_every_n = int(self.declare_parameter("detect_every_n", 1).value)
        self.detect_interval_sec = float(self.declare_parameter("detect_interval_sec", 0.25).value)  # ★ 0.7→0.25：提高检测频率，保证 track_timeout 内有足够刷新
        self.target_confirm_hits = int(self.declare_parameter("target_confirm_hits", 5).value)  # ★ 10→5：YOLO确认更快
        self.enable_red_detect = bool(self.declare_parameter("enable_red_detect", True).value)
        self.red_min_area_ratio = float(self.declare_parameter("red_min_area_ratio", 0.004).value)
        self.red_confirm_hits = int(self.declare_parameter("red_confirm_hits", 3).value)
        self.target_track_timeout = float(self.declare_parameter("target_track_timeout", 1.5).value)  # ★ 0.8→1.5：与detect_interval=0.25匹配，允许6帧内丢失不退出
        self.target_align_kp = float(self.declare_parameter("target_align_kp", 0.7).value)
        self.target_align_max = float(self.declare_parameter("target_align_max", 0.25).value)
        self.target_forward_speed = float(self.declare_parameter("target_forward_speed", 0.18).value)
        self.target_approach_area_ratio = float(self.declare_parameter("target_approach_area_ratio", 0.08).value)
        self.target_land_z = float(self.declare_parameter("target_land_z", -0.20).value)
        self.target_descent_speed = float(self.declare_parameter("target_descent_speed", 0.12).value)
        # ====== 任务8：目标接近参数重构 ======================================
        # 分级目标丢失超时
        self.target_track_timeout_short  = float(self.declare_parameter("target_track_timeout_short",  1.5).value)  # FIX5: 1.0->1.5s, more resilience
        self.target_track_timeout_sweep  = float(self.declare_parameter("target_track_timeout_sweep",  3.0).value)  # 中等丢失→扇扫
        # ★[A5][规则2] target_track_timeout_abort: 非终端时放弃门槛(6s)
        # 终端锁定后此参数控制的是"强制降落触发时机"，不再导致回搜索
        self.target_track_timeout_abort  = float(self.declare_parameter("target_track_timeout_abort",  6.0).value)  # 长时丢失→放弃（终端锁定时→强制降落）
        # 图像中心容忍度（归一化坐标，-0.5~0.5）
        self.target_center_tol_x         = float(self.declare_parameter("target_center_tol_x",         0.12).value)  # 水平容忍
        self.target_center_tol_y         = float(self.declare_parameter("target_center_tol_y",         0.15).value)  # 垂直容忍
        self.target_center_dwell_sec     = float(self.declare_parameter("target_center_dwell_sec",     0.3).value)  # FIX6: 0.5->0.3s   # 居中稳定时间(s)
        # 悬停确认时长
        self.target_hover_confirm_sec    = float(self.declare_parameter("target_hover_confirm_sec",    1.0).value)
        # 面积阈值（归一化，图像面积占比）
        # target_approach_area_ratio 已存在（0.08→进入悬停确认）
        self.target_land_area_ratio      = float(self.declare_parameter("target_land_area_ratio",      0.20).value)  # 面积大→可降落
        # 视觉伺服 Yaw P 增益（rad/s per normalized pixel）
        self.target_yaw_kp               = float(self.declare_parameter("target_yaw_kp",               1.2).value)
        self.target_yaw_max              = float(self.declare_parameter("target_yaw_max",               0.6).value)  # rad/s
        # 前进速度自适应范围
        self.target_forward_speed_max    = float(self.declare_parameter("target_forward_speed_max",    0.25).value)
        self.target_forward_speed_min    = float(self.declare_parameter("target_forward_speed_min",    0.05).value)
        # 降落速度
        self.target_descend_speed        = float(self.declare_parameter("target_descend_speed",        0.10).value)
        # 搜索阶段偏航扫描速率 (rad/s)
        self.target_search_yaw_scan_rate = float(self.declare_parameter("target_search_yaw_scan_rate", 0.3).value)

        # ============================================================
        # ★ 模块G：目标检测与接近参数集中定义
        # ============================================================
        # --- G1: OpenCV 红色目标检测参数 ---
        # HSV 双段红色范围（H: 0-10 和 170-180 都是红色，S/V 最低阈值过滤弱色）
        self.red_hsv_lower_1 = np.array(self.declare_parameter(
            "red_hsv_lower_1", [0, 120, 60]).value, dtype=np.uint8)
        self.red_hsv_upper_1 = np.array(self.declare_parameter(
            "red_hsv_upper_1", [10, 255, 255]).value, dtype=np.uint8)
        self.red_hsv_lower_2 = np.array(self.declare_parameter(
            "red_hsv_lower_2", [170, 120, 60]).value, dtype=np.uint8)
        self.red_hsv_upper_2 = np.array(self.declare_parameter(
            "red_hsv_upper_2", [180, 255, 255]).value, dtype=np.uint8)
        # 形态学开运算核尺寸（去除小噪点）
        _morph_open  = int(self.declare_parameter("red_morph_open_kernel",  3).value)
        _morph_close = int(self.declare_parameter("red_morph_close_kernel", 5).value)
        self.red_morph_open_kernel  = np.ones((_morph_open,  _morph_open),  np.uint8)
        self.red_morph_close_kernel = np.ones((_morph_close, _morph_close), np.uint8)
        # 检测最小面积（帧面积占比）：>= 此值才认为是有效轮廓
        # 0.0015 ≈ 320×240 图像中 115 像素，足以区分远处小目标与噪点
        self.red_min_area_ratio_detect = float(self.declare_parameter(
            "red_min_area_ratio_detect", 0.0015).value)

        # --- G2: 三级确认与进入接近阈值 ---
        # Level-1 候选：面积 >= red_min_area_ratio_detect，连续 >=3 帧
        # Level-2 确认：面积 >= red_min_area_ratio，streak >= red_confirm_hits，中心稳定
        # Level-3 进入接近：确认后还需要面积 >= target_approach_enter_area_ratio
        # 理由：避免目标太远太小就开始接近，导致很快丢失
        # 默认 0.02 ≈ 目标占图像 2%（约 40×40px / 320×240），此时目标已清晰可见
        self.target_approach_enter_area_ratio = float(self.declare_parameter(
            "target_approach_enter_area_ratio", 0.003).value)  # FIX1: 0.02->0.003
        # 进入接近时还要求目标连续居中的帧数（防止目标在边缘时就开始接近）
        self.target_approach_enter_center_frames = int(self.declare_parameter(
            "target_approach_enter_center_frames", 2).value)

        # --- G3: 目标跟踪（Kalman滤波器 + 可选OpenCV Tracker）---
        # OpenCV tracker 类型：prefer CSRT（精度高），KCF 作为后备，"none" 禁用
        self.target_tracker_type = str(self.declare_parameter(
            "target_tracker_type", "CSRT").value)
        # tracker 失败后用 IoU 门限来判断重检测结果是否适合重初始化
        # 0.2 = bbox 重叠 20% 以上才重初始化（防止新目标被误跟）
        self.target_tracker_reinit_iou_thresh = float(self.declare_parameter(
            "target_tracker_reinit_iou_thresh", 0.2).value)
        # 无测量情况下 Kalman 预测最多维持多少秒（超时后停止使用预测）
        self.target_prediction_max_hold_sec = float(self.declare_parameter(
            "target_prediction_max_hold_sec", 1.5).value)
        # Kalman 过程噪声（越大=状态更新越快，适合快速移动目标）
        self.target_kf_proc_noise  = float(self.declare_parameter(
            "target_kf_proc_noise", 0.02).value)
        # Kalman 测量噪声（越大=越信任预测而非当前测量，可抑制检测抖动）
        self.target_kf_meas_noise  = float(self.declare_parameter(
            "target_kf_meas_noise", 0.03).value)

        # --- G4: 视觉伺服（冗余声明，支持运行时覆盖）---
        # 以下参数在前面 "任务8" 块中已声明；此处是为了文档完整性的注释
        # target_yaw_kp=1.2, target_yaw_max=0.6 rad/s
        # target_center_tol_x=0.12, target_center_tol_y=0.15
        # target_center_dwell_sec=0.5 s（居中稳定才前进）
        # target_forward_speed_max=0.25, target_forward_speed_min=0.05 m/s
        # target_descend_speed=0.10 m/s

        # --- G5: 悬停确认与降落终端 ---
        # 进入悬停确认的面积阈值（已在 target_approach_area_ratio=0.08 中定义）
        # target_hover_confirm_area_ratio 与 target_approach_area_ratio 共用同一参数
        # 下降中目标丢失时暂停下降的帧数
        self.target_descend_pause_on_miss = int(self.declare_parameter(
            "target_descend_pause_on_miss", 3).value)
        # 触发 LAND 的高度阈值（米，NED坐标系，z为负=高，-0.20 ≈ 地面以上 20cm）
        self.target_land_trigger_height = float(self.declare_parameter(
            "target_land_trigger_height", -0.25).value)
        # 触发 LAND 的面积阈值（>=20%=目标已经非常近，可降落）
        self.target_land_trigger_area_ratio = float(self.declare_parameter(
            "target_land_trigger_area_ratio", 0.20).value)

        # ──────────────────────────────────────────────────────────────────
        # ★[G3] 视觉目标世界坐标估计参数
        # 功能：目标可见时估算其在地图平面(z≈0)的世界坐标，短时丢失时作为 SSG 导航目标
        # 原理：针孔相机模型，假设目标位于地面平面 z=0 或已知平面
        # ──────────────────────────────────────────────────────────────────
        # 相机水平视场角(°)，典型值：广角62°, 普通60°, 窄角45°
        self.target_camera_hfov_deg = float(self.declare_parameter(
            "target_camera_hfov_deg", 62.0).value)
        # 最小置信度：低于此值的世界坐标估计不使用（防止不稳定估计污染导航目标）
        self.target_world_est_min_conf = float(self.declare_parameter(
            "target_world_est_min_conf", 0.50).value)
        # EMA 平滑系数（0~1，越大=越跟随测量，越小=越平滑）
        # 0.25 在 10Hz 检测率下约 4帧收敛。增大可提高响应性，减小可抑制抖动
        self.target_world_est_alpha = float(self.declare_parameter(
            "target_world_est_alpha", 0.25).value)
        # 世界坐标目标保持有效时间(s)（超时后清除，回到普通探索）
        self.target_world_goal_hold_sec = float(self.declare_parameter(
            "target_world_goal_hold_sec", 8.0).value)
        # 世界坐标目标到达判定距离(m)
        self.target_world_goal_reach_dist = float(self.declare_parameter(
            "target_world_goal_reach_dist", 0.60).value)

        # ── ★[G5] 终端目标链参数（单一提交点方案）─────────────────────────────────
        # 世界坐标估计历史长度（用于方差估计，越大越稳定）
        self.target_world_history_len = int(self.declare_parameter(
            "target_world_history_len", 15).value)
        # 兼容旧参数（保留以防外部config引用）
        self.target_stage1_offset_m = float(self.declare_parameter(
            "target_stage1_offset_m", 1.0).value)
        self.target_stage2_offset_m = float(self.declare_parameter(
            "target_stage2_offset_m", 0.40).value)
        self.target_stage1_reach_dist = float(self.declare_parameter(
            "target_stage1_reach_dist", 0.50).value)
        self.target_stage2_reach_dist = float(self.declare_parameter(
            "target_stage2_reach_dist", 0.30).value)
        self.target_stage_hover_sec = float(self.declare_parameter(
            "target_stage_hover_sec", 2.0).value)
        # 保守降落提交条件参数
        self.target_descend_commit_height = float(self.declare_parameter(
            "target_descend_commit_height", -0.6).value)  # NED z 阈值（负=高度）
        self.target_commit_conf_min = float(self.declare_parameter(
            "target_commit_conf_min", 0.55).value)   # 最低世界坐标置信度
        self.target_commit_var_max = float(self.declare_parameter(
            "target_commit_var_max", 0.30).value)    # 最大方差 (m²)
        self.target_commit_speed_max = float(self.declare_parameter(
            "target_commit_speed_max", 0.25).value)  # 允许的最大水平速度 (m/s), ★[C3] was 0.15
        # 阶段导航最大速度 (m/s)
        self.target_stage_nav_speed = float(self.declare_parameter(
            "target_stage_nav_speed", 0.20).value)

        # ── ★[G6] 单一提交点方案专属参数 ────────────────────────────────────────────
        # 弱确认帧数阈值（少于 red_confirm_hits，更早冻结搜索 FSM）
        self.target_weak_confirm_hits = int(self.declare_parameter(
            "target_weak_confirm_hits", 2).value)  # 连续2帧即视为弱确认
        # 弱确认：目标中心允许的最大帧间跳变（归一化，0~1）
        self.target_weak_confirm_max_center_jump = float(self.declare_parameter(
            "target_weak_confirm_max_center_jump", 0.20).value)
        # 弱确认：目标不允许太靠近图像边缘（归一化距离，0=边缘，0.5=中心）
        self.target_weak_confirm_max_edge_margin = float(self.declare_parameter(
            "target_weak_confirm_max_edge_margin", 0.06).value)
        # 世界坐标建立阶段：最少悬停帧数时间 (s)（达到后才尝试生成提交点）
        self.target_world_establish_sec = float(self.declare_parameter(
            "target_world_establish_sec", 1.0).value)  # ★[C4] 1.0s 稳定估计(was 1.5s)
        # 单一提交点：距目标世界坐标的安全偏置 (m)（可接受停留在目标周围1m内）
        self.target_commit_offset_m = float(self.declare_parameter(
            "target_commit_offset_m", 0.80).value)  # 0.8m离目标即可
        # 提交点到达判定距离 (m)
        self.target_commit_reach_dist = float(self.declare_parameter(
            "target_commit_reach_dist", 0.40).value)  # 到达半径
        # 提交悬停时长 (s)
        self.target_commit_hover_sec = float(self.declare_parameter(
            "target_commit_hover_sec", 2.0).value)  # 悬停2s积累估计

        # ════════════════════════════════════════════════════════════════
        # ★[B] ROOM_ENTRY_SCAN 房间进入视觉扇扫参数
        # 无人机进入新房间后主动将前视相机覆盖左前/正前/右前扇区
        # ════════════════════════════════════════════════════════════════
        self.room_entry_scan_yaw_span_deg = float(self.declare_parameter(
            "room_entry_scan_yaw_span_deg", 120.0).value)  # 总扇扫角度 ±60°
        self.room_entry_scan_step_deg = float(self.declare_parameter(
            "room_entry_scan_step_deg", 40.0).value)   # 每步转角，覆盖3扇区
        self.room_entry_scan_hold_sec = float(self.declare_parameter(
            "room_entry_scan_hold_sec", 1.0).value)    # 每扇区悬停时长
        self.room_entry_scan_timeout_sec = float(self.declare_parameter(
            "room_entry_scan_timeout_sec", 20.0).value)  # 总扇扫超时

        # ════════════════════════════════════════════════════════════════
        # ★ 保守降落补充参数集中定义（统一默认值）
        # ════════════════════════════════════════════════════════════════
        self.target_world_var_max = float(self.declare_parameter(
            "target_world_var_max", 0.30).value)       # 世界坐标历史方差上限
        self.target_world_recent_seen_sec = float(self.declare_parameter(
            "target_world_recent_seen_sec", 5.0).value)  # 最近N秒内必须见到目标

        self.pose: Optional[VehicleLocalPosition] = None
        self.pose_initialized = False
        self.pose_init_start_time = self.get_clock().now()
        self.pose_init_timeout = float(self.declare_parameter("pose_init_timeout", 8.0).value)
        self.pose_init_stable_count = 0
        self.pose_init_required_stable = int(self.declare_parameter("pose_init_required_stable", 8).value)

        self.last_pose_msg_time = None
        self.last_pose_arrival_time = None
        self.pose_dt_ema_sec = None
        self.last_valid_z: Optional[float] = None
        self.last_valid_x: Optional[float] = None
        self.last_valid_y: Optional[float] = None
        self.altitude_est_z: Optional[float] = None
        self.z_jump_reject_count = 0
        self.xy_jump_reject_count = 0
        self.last_coord_warn_time = self.get_clock().now()

        self.att_rot: Optional[R] = None
        self.att_yaw: Optional[float] = None      # ENU yaw（π/2 - NED_yaw），与 bridge 约定一致
        self.att_yaw_ned: Optional[float] = None  # NED yaw（原始 PX4 四元数解算）
        self.att_pitch: Optional[float] = None
        self.att_roll: Optional[float] = None
        self.nav_state = 0
        self.arming_state = 0
        self.armed = False
        self.armed_since = None
        self.offboard_mode = False

        self.last_scan: Optional[LaserScan] = None
        self.last_scan_time = None
        self.last_scan_source = ""
        self._scan_seq: int = 0             # scan_cb 每接到新帧就+1
        self._last_integrated_seq: int = -1  # 上次已积分的帧号，相同帧号不再积分
        # 锚点位移门控阈值：HOVER_SCAN期间如果实际位置跟锁定点偏差>0.08m则拒绝建图
        self._hover_scan_pos_gate_m: float = 0.08   # 比速度门控更可靠
        import time as _init_mt
        self._last_reset_mono: float = _init_mt.monotonic()  # 上次EKF reset单调时间
        # ── EKF reset 检测与地图清空 ─────────────────────────────────────────
        # 修复A：EKF reset时清空地图（防止新旧坐标系地图叠加）
        self._last_xy_reset_counter: int = -1   # 上次位置reset counter
        self._last_hdg_reset_counter: int = -1  # 上次heading reset counter
        self._last_z_reset_counter: int  = -1   # 上次z reset counter（新增）
        self._map_pause_until: float = 0.0      # monotonic：在此之前暂停建图
        self._map_pause_sec: float = 3.0        # reset后暂停3.0s再重新建图（延长以等待EKF稳定）
        # EKF reset 状态机：STABLE → PAUSE → STABLE（用于日志 PAUSE/RESUME 打印）
        self._ekf_reset_state: str = "STABLE"   # "STABLE" | "PAUSE" | "REINIT"
        self._ekf_reset_count: int = 0          # 累计 EKF reset 次数（用于统计）
        # HOVER_SCAN 有效扫描帧计数（每次扫描周期清零，完成时打印）
        self._hover_valid_frames: int = 0
        # ── 位姿单调时间戳门控 ───────────────────────────────────────────────
        self._last_px4_ts: int = 0              # 上次接受的PX4 timestamp(us)
        self.lidar_ranges = {"front": None, "left": None, "right": None, "back": None}

        self.last_local_cmd: Optional[LocalCommand] = None
        self.last_local_stamp = None
        self.last_apf_cmd: Optional[ApfCommand] = None
        self.last_apf_stamp = None

        self.prev_vel = np.zeros(3, dtype=float)
        self.last_loop_time = self.get_clock().now()
        self.offboard_setpoint_counter = 0
        # ── startup 检测 ────────────────────────────────────────────────────────
        self._cmd_loop_ready: bool = False          # 首帧 command_loop 发出 [CMD-LOOP-READY]
        # ── offboard 恢复 ───────────────────────────────────────────────────────
        self._offboard_recovery_count: int = 0      # 已尝试切回 offboard 次数
        self._last_offboard_recovery_mono: float = 0.0  # 上次尝试的单调时间

        self.takeoff_done = False
        self.alt_takeoff_hold_start = None
        self.state = "TAKEOFF"
        # ★ 起飞后稳定化冻结：takeoff_done=True 之后等待 takeoff_stabilize_sec 秒
        # 在此期间保持位置不动，让 EKF/SLAM 在稳定悬停状态下建立初始地图
        # 避免起飞阶段电机振动+高度变化导致 SLAM 初始地图噪声过大
        self.takeoff_stabilize_sec = float(
            self.declare_parameter("takeoff_stabilize_sec", 3.0).value
        )
        self._takeoff_stab_done: bool = False   # 稳定化是否已完成
        self._takeoff_stab_start: Optional[float] = None  # monotonic 开始时间

        self.target_found = False
        # ── 两级目标确认机制 ──────────────────────────────────────────────────────
        # target_candidate: 宽松检测 - 任意检测到即为候选（frame ≥ 3）
        # target_confirmed: 严格确认 - 面积/边缘/中心漂移均通过后才允许进入 TARGET_APPROACH
        self.target_candidate: bool = False
        self.target_confirmed: bool = False
        self._target_cand_frames: int = 0         # 候选连续帧计数
        self._target_conf_frames: int = 0         # 确认连续帧计数
        self._target_cand_center_hist: list = []  # 中心点历史 [(cx_norm, cy_norm), ...]
        self._target_confirm_start_t: Optional[float] = None  # 首次候选的时间
        # TARGET_APPROACH 两阶段：Phase1=水平贴近，Phase2=允许下降
        self._target_approach_phase: int = 1       # 1=水平接近中, 2=允许下降
        self._target_xy_stable_since: Optional[float] = None  # Phase2 开始时间
        self._target_phase2_xy_thresh: float = 0.25  # 进入 Phase2 所需的像素偏差阈值（归一化）
        self._target_phase2_stable_sec: float = 1.5  # 需要稳定多久才切到 Phase2
        # ====== 任务1/2/3/4：完整目标接近状态机 ================================
        # 独占控制标志：True=探索逻辑冻结，目标接近完全接管
        self._approach_exclusive: bool = False
        # 内部子状态：'locked_far' | 'align' | 'forward' | 'hover_confirm' | 'descend' | 'land' | 'lost_short' | 'lost_sweep'
        self._approach_inner_state: str = 'align'
        # 目标丢失计时（monotonic）
        self._tgt_lost_since: Optional[float] = None
        # 扇扫状态：累计旋转量、方向
        self._sweep_yaw_start: Optional[float] = None
        self._sweep_dir: float = 1.0
        self._sweep_angle_acc: float = 0.0
        # 悬停确认进入时间
        self._hover_confirm_enter_t: Optional[float] = None
        # 目标居中稳定计时
        self._tgt_centered_since: Optional[float] = None
        # 当前 yaw 伺服输出（用于搜索/接近 分离）
        self._servo_yawspeed: float = 0.0
        # 接近阶段进入时间（用于日志）
        self._approach_enter_t: Optional[float] = None
        self.last_rgb_image = None
        self.rgb_counter = 0
        self.last_detect_time = self.get_clock().now()
        self.target_confirmation = TargetConfirmation(
            required_hits=self.target_confirm_hits,
            conf_threshold=self.detect_conf,
            timeout_sec=1.2,
        )
        self.red_confirmation = TargetConfirmation(
            required_hits=self.red_confirm_hits,
            conf_threshold=self.red_min_area_ratio,
            timeout_sec=0.8,
        )
        self.target_last_seen_time = None
        self.target_offset_x = 0.0
        self.target_offset_y = 0.0
        self.target_area_ratio = 0.0
        self.target_source = "none"       # "opencv" | "tracker" | "kalman" | "none"
        self.target_bbox = None           # (x, y, w, h) 或 None，像素坐标
        self.target_center_px = None      # (cx_px, cy_px) 或 None
        self.target_track_conf = 0.0      # [0,1] 综合跟踪置信度
        self.target_lost_stage = "none"   # "none"|"short"|"scan"|"abort"

        # ── ★[G3] 视觉目标世界坐标估计状态 ──────────────────────────────────
        # 当目标在图像中可见时，基于当前姿态/高度估算其地面投影坐标
        self._tgt_world_est_xy: Optional[np.ndarray] = None  # EMA 滤波后的 [x,y] ENU
        self._tgt_world_est_conf: float = 0.0                 # 估计置信度 [0,1]
        self._tgt_world_est_count: int = 0                    # 累积更新次数
        self._tgt_world_est_last_t: float = 0.0               # 最后更新 monotonic
        # ── ★[G4] 视觉目标 SSG 辅助导航目标 ──────────────────────────────────
        # 目标短时丢失但世界坐标稳定时，临时作为 SSG viewpoint_goal 让无人机靠近
        self._visual_target_goal: Optional[np.ndarray] = None  # [x,y] ENU
        self._visual_target_goal_set_t: float = 0.0            # 设置时间 monotonic

        # ── ★[G6] 单一提交点终端链状态变量 ──────────────────────────────────────
        # 单一提交点世界坐标 [x,y] ENU
        self._tgt_commit_goal: Optional[np.ndarray] = None
        # 提交点悬停进入时间 (monotonic)
        self._tgt_commit_hover_enter_t: Optional[float] = None
        # 世界坐标建立阶段进入时间 (monotonic)
        self._tgt_world_establish_enter_t: Optional[float] = None
        # 弱确认连续帧计数
        self._tgt_weak_confirm_cnt: int = 0
        # 弱确认状态标志（防止重复进入日志）
        self._tgt_weak_confirmed: bool = False
        # ★[A1][A6] 终端任务锁定 & 任务完成锁存
        # target_terminal_locked: 弱确认后立即设置，此后禁止回搜索
        # _mission_done: LAND发送后设置，禁止任何探索/再起飞
        self.target_terminal_locked: bool = False
        self._mission_done: bool = False
        # 兼容旧P1/P2变量（防止引用报错）
        self._tgt_stage1_goal: Optional[np.ndarray] = None
        self._tgt_stage2_goal: Optional[np.ndarray] = None
        self._tgt_stage_hover_enter_t: Optional[float] = None
        # 世界坐标历史（用于方差估计）
        self._tgt_world_xy_history: list = []   # 最近N次估计 [(x, y), ...]
        self._tgt_world_xy_var: float = 999.0   # 当前位置估计方差 (m²)
        # target_search.txt 日志文件句柄
        self._target_log_file = None

        # ── ★[B] ROOM_ENTRY_SCAN 房间进入扇扫状态变量 ──────────────────────
        self._room_entry_scan_active: bool = False     # 是否正在扇扫
        self._room_entry_scan_start_t: float = 0.0    # 开始时间 (monotonic)
        self._room_entry_scan_yaw_base: float = 0.0   # 进入时无人机 yaw 角
        self._room_entry_scan_step: int = 0            # 当前扇区索引(0=左,1=中,2=右)
        self._room_entry_scan_step_enter_t: float = 0.0  # 当前扇区进入时间
        self._room_entry_scan_steps = []               # 各扇区目标 yaw 列表
        self._room_entry_target_found: bool = False    # 扇扫中发现目标
        self._last_room_entry_trigger_t: float = 0.0  # 上次触发扇扫时间(防重复)
        self._last_hover_scan_room_id: int = -1        # ★[B] 上次HOVER_SCAN所在房间，用于新房间检测

        # ── 目标 Kalman 滤波器（归一化坐标：0~1，中心=0.5）─────────────────────
        # 用于在检测间隙预测目标位置，抑制检测抖动
        self._kf_target = TargetKalmanFilter(
            proc_noise=self.target_kf_proc_noise,
            meas_noise=self.target_kf_meas_noise,
        )
        # ── OpenCV Tracker（CSRT/KCF/MOSSE，用于短时遮挡跟踪）────────────────
        self._cv_tracker = None           # OpenCV tracker 实例（动态创建）
        self._cv_tracker_ok = False       # 上一次 update 是否成功
        self._cv_tracker_frame_w = 1      # tracker 初始化时的帧宽（归一化用）
        self._cv_tracker_frame_h = 1      # tracker 初始化时的帧高
        self._cv_tracker_last_t = 0.0     # 上次 tracker 更新的 monotonic 时间
        # ── 目标接近进入条件额外计数器──────────────────────────────────────────
        self._approach_center_ok_cnt = 0  # 连续符合居中条件的帧数（用于进入接近门限）
        # ── 下降暂停帧计数器 ────────────────────────────────────────────────────
        self._descend_pause_cnt = 0       # 下降中目标丢失暂停计数

        # NED高度边界说明：
        #   bounds[2] = [-2.5, -0.05]（即飞行高度范围 0.05m~2.5m）
        #   z_lower_bound = bounds[2,0] - 裕量：高度上限保护（z太负=飞太高）
        #   z_upper_bound = bounds[2,1] + 0.35：起飞前z≈0为地面，+0.35防误触发
        #   裕量0.5m，允许无人机飞到-3.0m才触发SAFE_HOLD（防止PID超调频繁触发）
        self.safety_monitor = SafetyMonitor(
            max_pose_age_sec=self.safety_max_pose_age_sec,
            max_scan_age_sec=max(1.0, self.lidar_timeout * 1.2),
            z_lower_bound=float(self.bounds[2, 0] - 1.0),   # 高度上限：bounds=-2.5 → 触发=-3.5m（留超调裕量）
            z_upper_bound=float(self.bounds[2, 1] + 0.35),  # 高度下限：bounds=-0.05 → 触发=+0.30m
            min_offboard_hold_sec=8.0,  # 起飞切换 offboard 模式需要更长的容忍窗口
            obstacle_stop_dist=0.18,    # 降低阈值：避免起落架/支架反射噪声(~0.22m)误触发
            obstacle_release_dist=0.22, # 解除阈值匹配降低后的stop值
            obstacle_hold_hits=3,       # 需要连续3次才触发，减少短暂反射误判
            obstacle_hold_max_sec=3.0,  # 死锁最长3s，防止EKF漂移
        )
        self.last_safety_warn = self.get_clock().now()

        self.search_yaw_dir = 1
        self.last_yaw_flip = self.get_clock().now()
        self.locked_yaw: Optional[float] = None  # 起飞完成后锁定偏航，保持地图坐标系稳定

        # 地图模块（在线更新 + 持久输出）
        self.map_output_dir = Path("/home/ubuntu/PX4-Autopilot/map_output")
        self.map_output_dir.mkdir(parents=True, exist_ok=True)
        self.map_res = float(self.declare_parameter("map_res", 0.1).value)
        xs = np.arange(self.bounds[0, 0], self.bounds[0, 1] + self.map_res, self.map_res)
        ys = np.arange(self.bounds[1, 0], self.bounds[1, 1] + self.map_res, self.map_res)
        self.grid_shape = (len(xs), len(ys))
        self.grid = np.full(self.grid_shape, 128, dtype=np.uint8)  # unknown
        self.plan_grid = np.full(self.grid_shape, 128, dtype=np.uint8)  # ★ 规划图层（Layer2稳定，B3施加前）——A*/frontier/房间检测用
        self.vis_grid  = np.full(self.grid_shape, 128, dtype=np.uint8)  # ★ 显示图层（B3滤波后）——仅用于可视化

        # ─────────────────────────────────────────────────────────────────────
        # ★ 固定规划地图元数据（_plan_*）：启动时一次性确定，之后永不被 SLAM 覆盖
        # SLAM origin/size 在回环、扩展、EKF reset 后会变化（正常 SLAM 行为），
        # 但业务规划/显示坐标系必须保持静止——这是地图稳定的基础。
        # map_cb 只用 _plan_* 来做 "固定世界坐标 → SLAM当前格" 的投影，不修改这些字段。
        # ─────────────────────────────────────────────────────────────────────
        self._plan_bounds_xy = np.array([
            [float(self.bounds[0, 0]), float(self.bounds[0, 1])],
            [float(self.bounds[1, 0]), float(self.bounds[1, 1])],
        ], dtype=float)   # [[xmin,xmax],[ymin,ymax]]  单位：m（固定世界坐标）
        self._plan_res = float(self.map_res)           # 固定规划分辨率（永不被 SLAM 覆盖）
        self._plan_ni  = int(self.grid_shape[0])       # 固定 X 轴格数
        self._plan_nj  = int(self.grid_shape[1])       # 固定 Y 轴格数
        # SLAM 原生地图（仅供调试保存使用，不参与规划/显示）
        self._slam_native_grid: Optional[np.ndarray] = None
        self._slam_native_meta: dict = {}

        self.prob_occ_th = int(self.declare_parameter("prob_occ_th", 180).value)
        self.prob_free_th = int(self.declare_parameter("prob_free_th", 70).value)
        # ── SLAM 原始值占据/空闲阈值 ─────────────────────────────────────────────
        # slam_toolbox: 0=free, 100=fully occupied
        # 50 仅是中等置信度，不应直接标为墙；≥65 才是可信占据，0-30 才是可信空闲
        # 中间值(31~64)→ unknown，大幅减少室内假墙
        self._slam_raw_occ_th  = int(self.declare_parameter("slam_raw_occ_threshold",  65).value)
        self._slam_raw_free_th = int(self.declare_parameter("slam_raw_free_threshold", 30).value)
        # ★ 地图质量门控阈值：plan 网格中 occupied 格数 < 此值 → MAP_THIN（地图墙体不足，禁止激进导航）
        self._min_occ_for_planning = int(self.declare_parameter("min_occ_cells_for_planning", 10).value)
        self._slam_occ_cell_count: int = 0    # plan 网格中当前 occ 格数（每帧更新）
        self._map_quality_ok: bool = False    # occ 格数 >= _min_occ_for_planning
        # ── HOVER_SCAN 稳定门控阈值（放宽：PX4悬停位置噪声约±10cm）──────────────
        # 原阈值 drift>0.05m/vz>0.06 在正常悬停下拒绝大量有效帧，导致建图帧太少
        self._hover_gate_drift_m   = float(self.declare_parameter("hover_gate_drift_m",  0.15).value)
        self._hover_gate_vz_mps    = float(self.declare_parameter("hover_gate_vz_mps",   0.12).value)
        self._hover_gate_tilt_rad  = float(self.declare_parameter("hover_gate_tilt_rad", 0.20).value)
        # 拒绝原因统计（每100帧汇报一次）
        self._hg_total:     int = 0
        self._hg_rejected:  int = 0
        self._hg_rej_drift: int = 0
        self._hg_rej_vz:    int = 0
        self._hg_rej_tilt:  int = 0
        # ── 可视化：fit_room_rectangle 独立颜色+置信度门槛 ────────────────────────
        # vis_draw_room_box=True 才画拟合边框（橙色）；至少要有足够的已知格子
        self._vis_draw_room_box     = bool(self.declare_parameter("vis_draw_room_box",           True).value)
        self._vis_room_box_min_free = int(self.declare_parameter("vis_room_box_min_free_cells", 1500).value)
        self._vis_room_box_min_occ  = int(self.declare_parameter("vis_room_box_min_occ_cells",   300).value)
        self.map_update_interval = float(self.declare_parameter("map_update_interval", 0.15).value)
        self.seg_update_interval = float(self.declare_parameter("seg_update_interval", 1.0).value)
        self.map_save_interval = float(self.declare_parameter("map_save_interval", 2.0).value)
        # ===== 地图稳定性参数 =====
        # 当 SLAM origin 在一次 /map 回调中跳变 > map_stab_origin_jump_m，认为地图不稳定
        # 当地图尺寸变化 > map_stab_size_jump_cells，同样视为不稳定
        self.map_stab_origin_jump_m = float(self.declare_parameter("map_stab_origin_jump_m", 2.0).value)
        self.map_stab_size_jump_cells = int(self.declare_parameter("map_stab_size_jump_cells", 500).value)
        # ── 时域投票滤波参数（B2）──────────────────────────────────────────
        self._map_temporal_enabled = bool(self.declare_parameter("map_temporal_filter_enabled", True).value)
        self._map_temporal_window  = int(self.declare_parameter("map_temporal_window", 5).value)
        self._map_occ_vote_th      = int(self.declare_parameter("map_occ_vote_th",  3).value)
        self._map_free_vote_th     = int(self.declare_parameter("map_free_vote_th", 3).value)
        # ── 空间去噪参数（B3）────────────────────────────────────────────
        self._map_spatial_enabled    = bool(self.declare_parameter("map_spatial_filter_enabled", True).value)
        self._map_occ_close_m        = float(self.declare_parameter("map_occ_close_m",    0.15).value)
        self._map_occ_open_m         = float(self.declare_parameter("map_occ_open_m",     0.10).value)
        self._map_occ_min_comp_cells = int(self.declare_parameter("map_occ_min_component_cells", 3).value)
        # ── 三层地图参数（5.1/5.2）──────────────────────────────────────────
        # Layer1: _raw_slam_grid   每帧 SLAM 重投影原始结果（uint8，逐帧覆盖，仅供诊断）
        # Layer2: _stable_occ_grid log-odds EMA 融合稳定层（uint8，跨帧持久化）
        # Layer3: self.grid        空间去噪后规划/显示层（= Layer2 经形态学滤波）
        # stable_lo_enabled=True 时 Layer2 替代 B2 时域投票滤波器（内存更低、收敛更平滑）
        self._stable_lo_enabled  = bool(self.declare_parameter("stable_lo_enabled",  True).value)
        self._stable_lo_alpha    = float(self.declare_parameter("stable_lo_alpha",    0.70).value)   # 历史权重
        self._stable_lo_obs      = float(self.declare_parameter("stable_lo_obs",      1.5).value)    # 每帧 +/-logit
        self._stable_lo_occ_th   = float(self.declare_parameter("stable_lo_occ_th",   0.8).value)    # occ 决策阈值
        self._stable_lo_free_th  = float(self.declare_parameter("stable_lo_free_th", -0.8).value)    # free 决策阈值
        self._stable_lo_max      = float(self.declare_parameter("stable_lo_max",      4.0).value)    # logit 截断防饱和
        _s_ni, _s_nj = int(self.grid_shape[0]), int(self.grid_shape[1])
        self._raw_slam_grid:   np.ndarray = np.full((_s_ni, _s_nj), 128, dtype=np.uint8)   # Layer1
        self._stable_log_odds: np.ndarray = np.zeros((_s_ni, _s_nj), dtype=np.float32)    # Layer2 log-odds
        self._stable_occ_grid: np.ndarray = np.full((_s_ni, _s_nj), 128, dtype=np.uint8)  # Layer2 uint8
        self.map_stab_pause_sec = float(self.declare_parameter("map_stab_pause_sec", 2.0).value)
        self._map_stability_ok: bool = True   # 地图稳定状态（False时暂停路径规划）
        self._map_stab_pause_until: float = 0.0  # monotonic：在此之前暂停规划
        # C3: MAP-STAB 频繁跳变警告追踪（不自动重启 SLAM，只记录警告）
        self._map_stab_jump_times: list = []   # [monotonic_ts, ...] 最近跳变时局
        self._map_stab_freq_warned: bool = False    # 是否已打印频繁警告
        # ===== 可视化发布参数 =====
        # 发布 /indoor_search/map_vis （sensor_msgs/Image），支持全局和局部跟随视图
        self.vis_enabled = bool(self.declare_parameter("vis_enabled", True).value)
        self.vis_mode = str(self.declare_parameter("vis_mode", "global").value)  # "global" | "local"（默认全局，可用 ros2 param set 改为 local）
        self.vis_local_range_m = float(self.declare_parameter("vis_local_range_m", 10.0).value)  # 局部视图半径(m)
        self.vis_max_dim = int(self.declare_parameter("vis_max_dim", 512).value)  # 发布图像最大边(px)
        self.vis_interval = float(self.declare_parameter("vis_interval", 0.5).value)   # 发布周期(s)
        self.last_vis_update = self.get_clock().now()
        # ★ 显示边界（与计算边界解耦）：固定可视化画布范围，不随SLAM动态地图扩展
        # display_fixed_bounds=True（默认开启）：可视化始终固定在世界坐标范围，不随SLAM扩展漂移
        # display_fixed_bounds=False：显示完整SLAM原生地图（调试用）
        self._display_fixed_bounds = bool(self.declare_parameter("display_fixed_bounds", True).value)
        self._display_xmin = float(self.declare_parameter("display_xmin", -8.0).value)
        self._display_xmax = float(self.declare_parameter("display_xmax",  8.0).value)
        self._display_ymin = float(self.declare_parameter("display_ymin", -8.0).value)
        self._display_ymax = float(self.declare_parameter("display_ymax",  8.0).value)
        # seg 形态学预处理参数
        self.seg_morph_close_m = float(self.declare_parameter("seg_morph_close_m", 0.15).value)  # 闭运算宽度(m)，填补墙面缝隙
        self.seg_min_free_cells = int(self.declare_parameter("seg_min_free_cells", 40).value)   # 自由区小连通域最小格数
        self.last_map_update = self.get_clock().now()
        self.last_seg_update = self.get_clock().now()
        self.last_map_save = self.get_clock().now()
        self.room_labels = np.zeros(self.grid_shape, dtype=np.int32)
        self.rooms_info = []
        self.doors_info = []
        self.confirmed_doors: list = []  # 只含 confirmed_door（两侧均为已知房间）
        self.corridor_connectors: list = []  # ★[E] 走廊连接器（corridor_connector 类型）
        self.frontier_gaps: list = []    # frontier_gap（一侧未知，不进入导航）

        # ===== Stop-Scan-Go：地图冻结开关 =====
        # 只有 HOVER_SCAN 阶段 _map_frozen=False，允许 update_map_from_scan
        # 移动/导航阶段 _map_frozen=True，A* 用冻结的 grid，激光只做避障
        self._map_frozen: bool = True  # 起飞前冻结，HOVER_SCAN时解冻

        # ===== Belief 目标概率地图 =====
        # belief_grid[i,j] ∈ [0,1]，初始均匀（用先验分布）
        # 正证据：YOLO检测到目标→周边高斯增强
        # 负证据：扫描通过的可视区域→乘以衰减系数，逐渐降低目标概率
        self.belief_grid = np.full(self.grid_shape, 0.3, dtype=np.float32)  # 初始先验 0.3
        self.belief_decay_coeff = float(self.declare_parameter("belief_decay_coeff", 0.85).value)   # 负证据衰减
        self.belief_detect_boost = float(self.declare_parameter("belief_detect_boost", 0.9).value)  # 检测到目标时峰值belief
        self.belief_detect_sigma = float(self.declare_parameter("belief_detect_sigma", 1.5).value)  # 检测高斯增强半径(m)

        # ===== Stop-Scan-Go：扫描点选择参数 =====
        self.ssg_nav_speed = float(self.declare_parameter("ssg_nav_speed", 0.60).value)     # ★ 0.40→0.60：加快搜索导航速度
        self.ssg_reach_dist = float(self.declare_parameter("ssg_reach_dist", 0.60).value)   # 到达下一扫描点的判定距离
        self.ssg_stuck_timeout = float(self.declare_parameter("ssg_stuck_timeout", 18.0).value)  # 卡住超时→重选点
        self.ssg_r_safe = float(self.declare_parameter("ssg_r_safe", 0.35).value)           # 候选点安全裕度(m)
        self.ssg_lidar_range = float(self.declare_parameter("ssg_lidar_range", 6.0).value)  # InfoGain 激光射程(m)
        self.ssg_ig_sigma = float(self.declare_parameter("ssg_ig_sigma", 3.0).value)        # InfoGain 距离衰减σ(m)
        self.ssg_w_ig = float(self.declare_parameter("ssg_w_ig", 1.0).value)               # 信息增益权重
        self.ssg_w_tg = float(self.declare_parameter("ssg_w_tg", 2.0).value)               # 目标增益权重
        self.ssg_w_cost = float(self.declare_parameter("ssg_w_cost", 0.4).value)            # 路径代价权重
        self.ssg_w_risk = float(self.declare_parameter("ssg_w_risk", 0.8).value)            # 风险权重
        self.ssg_w_revisit = float(self.declare_parameter("ssg_w_revisit", 0.6).value)      # 重访惩罚权重（含热格惩罚）
        self.ssg_belief_confirm_th = float(self.declare_parameter("ssg_belief_confirm_th", 0.60).value)  # 目标确认阈值
        self.ssg_n_candidates = int(self.declare_parameter("ssg_n_candidates", 50).value)  # 候选点数量
        self.ssg_visited_scan_pts: list = []   # 历史扫描点 [(x,y),...]，用于重访惩罚
        self.ssg_viewpoint_goal: Optional[np.ndarray] = None   # 当前导航目标扫描点
        self.ssg_goal_type: str = "none"    # 当前目标类型：none/frontier/coverage/confirmed_door/frontier_gap_probe/frontier_gap_center/fallback
        self._frontier_gaps_cached: list = []   # frontier_gap TTL缓存
        self._frontier_gaps_cache_t: float = 0.0
        self._door_fail_counts: dict = {}   # {goal_sig: int} 每个门口目标的失败计数
        self._target_blacklist: dict = {}   # {goal_sig: expiry_mono} 短期黑名单（90s TTL）
        self._ssg_stuck_pos: Optional[np.ndarray] = None
        self._ssg_stuck_since: Optional[float] = None
        self._ssg_last_select_time: float = 0.0
        # ★[D1] 空间冷却 TTL 网格：禁止无人机短期内重访同一格
        # 格子大小 1.5m，TTL 90s，惩罚系数0.5（叠加到 revisit penalty)
        self._spatial_cooldown: dict = {}          # {(ix,iy): expiry_mono}
        self.spatial_cooldown_cell_m: float = float(
            self.declare_parameter("spatial_cooldown_cell_m", 1.5).value)
        self.spatial_cooldown_ttl_sec: float = float(
            self.declare_parameter("spatial_cooldown_ttl_sec", 90.0).value)
        self.spatial_cooldown_penalty: float = float(
            self.declare_parameter("spatial_cooldown_penalty", 0.50).value)

        # ===== 门候选稳定性 TTL 缓存 =====
        # 防止每帧分割结果因噪声波动导致 doors_info 一帧有一帧没有，影响规划连续性
        self._door_cand_cache: dict = {}       # sig -> {"data": door_dict, "hits": int, "misses": int}
        self._door_confirmed_stable: list = [] # 已通过 N 帧确认的稳定门列表（用于导航）
        self._door_confirm_frames: int = 2     # ★[E2] 2帧即可标记稳定（原3帧），加快门导航确认
        # ★[E1] 5帧稳定即可硬切（原8帧检测过慢，导致门口导航前中期 doors=0）
        # ★[E2] 进一步降至4帧≈2s@2Hz；配合_door_miss_tolerance=4同步调整
        # 2Hz分割速率下：4帧≈2s更快; 5帧≈2.5s偏慢
        self._door_hard_cut_frames: int = 4    # ★[E2] was 5→4：进一步加快硬切确认
        self._door_miss_tolerance: int = 4     # ★[E2] was 5→4：配套缩短缺失容忍
        # ★[C] 硬切迟滞：已经达到 hard_cut 的门，需要 N 次连续缺失才能被移除（防止环境遮挡导致分割抖动）
        self._door_hard_cut_release_frames: int = 12  # was 14→12：适度降低迟滞，配合更早检测
        self._door_hard_confirmed_sigs: set = set()   # 已达到 hard_cut 阈值的门签名集合
        # ★[C5] 房间计数稳定化：防止 rooms=2→1 振荡
        # 仅在连续 _rooms_confirm_frames 帧都报告相同房间数时才切换到更高值
        # 从高房间数降低需要 _rooms_drop_frames 连续帧（防止单帧遮挡导致 seg 退回 coarse）
        self._seg_stable_n_rooms: int = 0        # 当前稳定化后的房间数（经迟滞）
        self._seg_raw_n_rooms_buf: list = []     # 最近 N 帧原始房间数（滑动窗口）
        self._rooms_confirm_frames: int = 2      # ★[E2] 新房间数连续2帧即接受（原3帧），加快多房间识别
        self._rooms_drop_frames: int = 6         # 房间数降低前需连续此帧数确认（降）
        self._seg_rooms_log_t: float = 0.0       # 上次 [ROOMS] 日志时间
        # ★[A6] 分级门模型三层状态列表
        # soft_gap:       hits 1-2，仅可视化提示，不参与分割、不作导航目标
        # stable_soft_gap:hits 3-4，可作导航目标（door transit），不参与硬切分割
        # confirmed_door: hits≥5 AND room_a>0, room_b>0，参与正式房间切断
        self.soft_gaps: list = []              # 候选门（低置信，仅可视化）
        self.stable_soft_gaps: list = []       # 稳定软门（可导航，不硬切）
        self._soft_gaps_prev_cnt: int = 0      # 上帧 soft_gap 数量（变化时触发日志）
        self._stable_soft_prev_cnt: int = 0    # 上帧 stable_soft 数量（变化时触发日志）
        # ★[A5] 目标锁定：进入门通行任务后，防止被重新选点打断
        self._goal_switch_hold_sec: float = 12.0   # 选定门目标后的锁定时间(s)
        self._goal_hold_until: float = 0.0         # 此时刻前不允许选新目标
        # ★[B] 目标切换防抖：防止 SSG 在短时间内频繁替换规划目标导致飞行抖动
        self.goal_switch_min_interval_sec: float = float(
            self.declare_parameter("goal_switch_min_interval_sec", 2.0).value)
        self.room_relabel_hysteresis_sec: float = float(
            self.declare_parameter("room_relabel_hysteresis_sec", 3.0).value)
        self.door_reassoc_hysteresis_sec: float = float(
            self.declare_parameter("door_reassoc_hysteresis_sec", 4.0).value)
        self._goal_switch_last_t: float = 0.0          # 上次切换时间 (monotonic)
        self._goal_switch_last_type: str = "none"      # 上次目标类型
        self._door_assoc_prev: dict = {}               # {sig: (room_a, room_b)} 门最近关联
        self._door_reassoc_last_t: dict = {}           # {sig: monotonic} 上次 DOOR-REASSOC 时间
        # ★[D2] 进入新房间稳定区跟踪
        self._room_entered_id: int = -1                # 上次通过门进入的房间 ID
        self._room_entered_t: float = 0.0              # 进入新房间的时间 (monotonic)
        self._room_stable_logged: bool = False         # 是否已记录 ROOM-STABLE-ZONE
        self.room_stable_zone_sec: float = float(
            self.declare_parameter("room_stable_zone_sec", 4.0).value)  # 稳定区判定时长(s)
        # ★ Phase5: 世界坐标门缓存（TTL，独立于 grid 坐标系）
        # 键 = (round(wx/0.3)*0.3, round(wy/0.3)*0.3)，值保存门数据和最后见到时间
        # 即使 grid 重框/SLAM jump，世界坐标门仍然保留 _door_world_ttl_sec 秒
        self._door_world_cache: dict = {}      # wsig -> {"door": dict, "last_seen": float, "conf": int}
        self._door_world_ttl_sec: float = float(self.declare_parameter(
            "door_world_ttl_sec", 8.0).value)  # 世界坐标门缓存保持时间
        self._door_world_match_dist_m: float = float(self.declare_parameter(
            "door_world_match_dist_m", 0.5).value)  # 世界坐标匹配半径

        # ===== STUCK 进展检测与恢复 =====
        # 若探索区域长时间无增长且无人机处于集聚状态，触发 STUCK 恢复流程
        self._stuck_count: int = 0             # STUCK 累计触发次数
        self._stuck_last_trigger_t: float = 0.0  # 上次 STUCK 触发时间（防抖：120s 冷却）
        self._explore_free_window: list = []   # [(now_sec, free_count)] 滑动窗口（60s）
        self._virtual_door_cands: list = []    # STUCK 后生成的虚拟出口候选 [(wx, wy)]

        self.room_min_area = float(self.declare_parameter("room_min_area", 2.0).value)
        self.door_min_width = float(self.declare_parameter("door_min_width", 0.8).value)
        self.door_max_width = float(self.declare_parameter("door_max_width", 2.5).value)
        # ★ 修复：默认值从 3 降至 1，避免对稀疏细片墙体过度滑平（远处墙体标记消失的根因之一）
        self.seg_median_ksize = int(self.declare_parameter("seg_median_ksize", 1).value)

        # ===== ★[H] 分割参数组织 =====
        # --- 房间掩模正则化（Task A）---
        self.room_mask_open_kernel_m   = float(self.declare_parameter("room_mask_open_kernel_m",  0.0).value)   # m, 0=禁用
        self.room_mask_close_kernel_m  = float(self.declare_parameter("room_mask_close_kernel_m", 0.0).value)  # m, 0=禁用
        self.room_min_component_area_m2 = float(self.declare_parameter("room_min_component_area_m2", 0.5).value)  # m²
        self.room_polygon_simplify_eps  = float(self.declare_parameter("room_polygon_simplify_eps", 0.15).value)   # m
        # --- 门几何约束（Task C）---
        self.door_width_min_m          = float(self.declare_parameter("door_width_min_m", 0.55).value)   # m
        self.door_width_max_m          = float(self.declare_parameter("door_width_max_m", 1.80).value)   # m
        self.door_jamb_support_min_len  = float(self.declare_parameter("door_jamb_support_min_len", 0.20).value)  # m 门框支撑最小长度
        self.door_depth_max_m          = float(self.declare_parameter("door_depth_max_m", 2.50).value)   # m 门洞深度上限（大→不限制）
        self.door_room_min_area_m2     = float(self.declare_parameter("door_room_min_area_m2", 1.0).value)  # m²
        # --- 走廊检测（Task B）---
        self.corridor_aspect_ratio_min  = float(self.declare_parameter("corridor_aspect_ratio_min", 3.5).value)   # 走廊宽高比
        self.corridor_width_max_m       = float(self.declare_parameter("corridor_width_max_m", 1.80).value)   # m 走廊最大宽度
        self.corridor_parallel_wall_len = float(self.declare_parameter("corridor_parallel_wall_len", 1.5).value)  # m 走廊平行墙最小长度
        # --- 时序保护（Task F）---
        self.door_upgrade_min_map_occ   = int(self.declare_parameter("door_upgrade_min_map_occ", 80).value)   # 最少占据格数
        self.door_upgrade_min_room_area_m2 = float(self.declare_parameter("door_upgrade_min_room_area_m2", 1.0).value)  # m²
        self.door_side_occ_th = float(self.declare_parameter("door_side_occ_th", 0.20).value)

        # ===== Stop-Scan-Go 状态机 =====
        # 核心：HOVER_SCAN 完整360°扫描建图 → 冻结地图 → A*飞向最佳下一扫描点 → 悬停扫描 → 循环
        # explore_state: "HOVER_SCAN" | "NAV_TO_VIEWPOINT"
        self.explore_state = "HOVER_SCAN"        # 初始先扫描
        self.mapping_min_free_cells = int(self.declare_parameter("mapping_min_free_cells", 200).value)
        self._last_explore_state_log: float = 0.0
        # 兼容旧变量（部分可视化/分割逻辑仍用）
        self.frontier_goal: Optional[np.ndarray] = None
        self.current_room_id: int = -1
        self.visited_rooms: set = set()
        self.door_nav_goal: Optional[np.ndarray] = None
        self.door_nav_goal_id: int = -1
        # ── slam_gate_mode 追踪（向 bridge 节点发布强制打开/关闭门控）──────────
        self._slam_gate_mode_current: str = ""   # 已发送给 bridge 的当前模式
        self._prev_explore_state: str = ""       # 空串 → 首帧会立即检测并发布
        # ── Phase7: 进度看门狗（goal-stuck 检测与恢复）──────────────────────
        # 检测无人机长时间无法离开某位置（被障碍物/路径规划失败卡住）
        self._progress_wd_pos: Optional[tuple] = None   # 参考位置 (x,y)
        self._progress_wd_since: float = 0.0            # 参考位置记录时刻（monotonic）
        self._progress_wd_trig_cnt: int = 0             # 触发次数（诊断用）
        self._progress_wd_backstep_frames: int = 0      # 剩余后退帧数（触发后非零）
        self._progress_wd_backstep_dir: Optional[np.ndarray] = None  # 后退方向
        self._progress_wd_timeout_sec: float = float(
            self.declare_parameter("progress_wd_timeout_sec", 8.0).value
        )   # 判定卡住的时间阈值（秒）
        self._progress_wd_min_move_m: float = 0.20     # 判定有进度的最小位移（米）
        # ── MAP_RECOVER 完整恢复状态机 (Phase9) ────────────────────────
        # 触发条件：EKF reset和/或地图大幅跳变 → 进入恢复模式
        # 退出条件：连续渐进帧数足够 AND EKF 15s下无新 reset
        self._map_recover_active: bool = False       # 当前是否处于地图恢复模式
        self._map_recover_since_mono: float = 0.0    # 恢复模式开始时刻（monotonic）
        self._hover_scan_retry_cnt: int = 0          # 连续重试 HOVER_SCAN 次数
        self._hover_scan_max_retry: int = int(
            self.declare_parameter("hover_scan_max_retry", 5).value
        )                                            # 超过后强制 frontier 逃逸
        self._occ_stable_frames: int = 0             # 地图占据变化稳定帧计数（退出恢复用）
        self._occ_stable_needed: int = 5             # 退出恢复需要的连续渐进帧数──────────────────────────────────────────
        # 4.3 CORNER-TRAP 角落贴墙状态
        self._corner_trap_until: float = 0.0          # monotonic: 在此时刻前强制逃出模式
        self._corner_trap_door_block_until: float = 0.0  # 临时禁止 door 作目标的时居
        self._corner_trap_escape_dir: Optional[np.ndarray] = None  # 逃出方向 [vx,vy]
        self._corner_trap_trigger_count: int = 0   # ★ 触发次数（穿闹模式下不兼第1次就清除全局目标）
        # ★[J3] corner_front_th_m 0.80→0.40m：原0.80m在正常走廊导航中极易误触发
        #   （前方0.80m内有墙=正常转角，不是真正角落），收紧到0.40m只处理真实角落
        self._corner_front_th_m: float = float(self.declare_parameter("corner_front_th_m", 0.40).value)
        self._corner_side_th_m:  float = float(self.declare_parameter("corner_side_th_m",  0.28).value)
        self._corner_block_sec:  float = float(self.declare_parameter("corner_block_sec",  4.0).value)
        # CMD-FRAME 切换计数器（每100次打印一次诊断）
        self._publish_setpoint_cnt: int = 0
        # HOVER_SCAN_ESCAPE 子状态变量
        self._hover_scan_escape_active: bool = False
        self._hover_scan_escape_dir: Optional[np.ndarray] = None
        self._hover_scan_escape_start: float = 0.0
        self._hover_scan_escape_safe_dist: float = float(
            self.declare_parameter("hover_scan_escape_safe_dist_m", 0.35).value
        )
        self._hover_scan_escape_speed: float = float(
            self.declare_parameter("hover_scan_escape_speed_mps", 0.07).value
        )  # ★[FIX1] 0.12→0.07 m/s: HOVER_SCAN escape 仅作小幅偏置，不独占 XY
        self._hover_scan_escape_log_t: float = 0.0
        # 严格 ESCAPE 恢复条件：clearance 必须连续超过此阈值 >= resume_hold_sec 才退出 ESCAPE
        self._hover_scan_resume_clearance_m: float = float(
            self.declare_parameter("hover_scan_resume_clearance_m", 0.32).value
        )  # ★[H1] 降至0.32m（原0.45m在窄廊clearance=0.41m时永远无法成功逃脱→fail_count无限递增→SAFE_HOLD 30s死循环）
        self._hover_scan_escape_min_dur_s: float = float(
            self.declare_parameter("hover_scan_escape_min_dur_s", 0.40).value
        )
        self._hover_scan_resume_hold_s: float = float(
            self.declare_parameter("hover_scan_resume_hold_s", 0.50).value
        )
        self._hover_scan_escape_clear_since: float = 0.0   # clearance > resume 阈值的起始时刻
        self._hover_scan_last_escape_t: float = 0.0        # 上次 ESCAPE 结束的 monotonic 时刻
        # ── ESCAPE 升级策略：连续失败计数 + 升级等级 ─────────────────────────────
        self._hover_scan_escape_fail_count: int = 0    # 连续逃离失败次数（成功后清零）
        self._hover_scan_escape_level: int = 0          # 0=normal, 1=extended_hold, 2=map_recover
        # scan_done 额外条件：有效帧数 >= min_valid_frames
        self._hover_scan_min_valid_frames: int = int(
            self.declare_parameter("hover_scan_min_valid_frames", 40).value
        )
        # MAP-RECOVER-HARD：大跳变计数 + SAFE_HOLD 保护
        self._large_jump_count: int = 0          # 近期大跳变次数
        self._large_jump_window_start: float = 0.0  # 计数窗口开始时刻
        self._large_jump_window_sec: float = float(
            self.declare_parameter("large_jump_window_sec", 30.0).value
        )
        self._large_jump_abort_count: int = int(
            self.declare_parameter("large_jump_abort_count", 3).value
        )
        self._large_jump_hard_hold_sec: float = float(
            self.declare_parameter("large_jump_hard_hold_sec", 5.0).value
        )
        self._map_recover_hard_until: float = 0.0  # monotonic: 强制HOLD到此刻
        self._recover_stable_frames_needed: int = int(
            self.declare_parameter("recover_stable_frames", 20).value
        )
        # att_cb 1Hz诊断日志计时
        self._att_enu_log_t: float = 0.0
        self._wall_stick_since: float = 0.0      # 贴墙状态开始时刻（monotonic）
        self._wall_stick_pos: Optional[tuple] = None   # 贴墙时的位置 (x,y)
        self._wall_stick_active: bool = False    # 当前是否处于贴墙回退
        self._wall_stick_recover_dir: Optional[np.ndarray] = None  # 回退方向
        self._wall_stick_detect_dist_m = float(self.declare_parameter(
            "wall_stick_detect_dist_m", 0.35).value)   # 触发贴墙检测的激光最近距离
        self._wall_stick_hold_sec = float(self.declare_parameter(
            "wall_stick_hold_sec", 1.5).value)          # 持续N秒无位移 + 有指令 → 贴墙
        self._wall_stick_min_move_m = float(self.declare_parameter(
            "wall_stick_min_move_m", 0.08).value)       # 期望位移阈值

        # ===== HOVER_SCAN 悬停旋转扫描参数 =====
        # 360°旋转扫描，建立当前扫描点周边完整地图，旋转完成后冻结地图
        self.hover_scan_duration = float(self.declare_parameter("hover_scan_duration", 22.0).value)  # ★[B1] 40→22s：截短最大扫描时长（0.40rad/s旋转22s≈约504°已足够）
        self.hover_scan_yaw_speed = float(self.declare_parameter("hover_scan_yaw_speed", 0.40).value) # ★ 0.2→0.40：加快 HOVER_SCAN 旋转（0.40≈23°/s）
        self.hover_scan_start_time: Optional[float] = None
        self.hover_scan_yaw_start: Optional[float] = None
        self.hover_scan_accum_angle: float = 0.0
        self._hover_scan_hold_pos: Optional[tuple] = None   # 悬停扫描锚定位置 ENU (East=pose.x, North=pose.y) + NED_z，用于 position setpoint
        self._hover_scan_anchor_map: Optional[tuple] = None  # 悬停扫描锚点 SLAM map frame (x,y)，用于 drift gate（与 _map_px/_map_py 同坐标系）
        self._hover_scan_last_yaw: Optional[float] = None  # 上一帧yaw，用于增量累积
        self.hover_scan_min_angle_deg = float(self.declare_parameter("hover_scan_min_angle_deg", 270.0).value)  # ★[B1] 340→270°：降低完成门槛，加快扫描循环
        # HOVER_SCAN 硬避障触发阈值：近障 < 此值 → 暂停旋转 + 暂停扫描积分 + 进入 ESCAPE
        self._hover_scan_emerg_stop_dist = float(self.declare_parameter(
            "hover_scan_emerg_stop_dist_m", 0.25).value)  # ★[FIX1] 0.45→0.25m: 仅真正贴墙时才触发 escape

        # ===== A* 路径规划参数 =====
        self.astar_inflation_radius = float(self.declare_parameter("astar_inflation_radius", 0.3).value)
        # ★ astar_treat_unknown_as_occ：正式 ROS2 参数，不再 getattr 硬编码
        # True（默认）= 未知格视为障碍，保守但可能堵死门口
        # False = 未知格视为可通行，用于门口穿越等宽松场景
        self.astar_treat_unknown_as_occ: bool = bool(
            self.declare_parameter("astar_treat_unknown_as_occ", True).value
        )
        self.astar_path: Optional[list] = None
        self.astar_path_idx: int = 0
        self.astar_waypoint_reach_dist = float(self.declare_parameter("astar_waypoint_reach_dist", 0.5).value)
        self.astar_replan_interval = float(self.declare_parameter("astar_replan_interval", 3.0).value)
        self.astar_last_plan_time: float = 0.0

        # ★[A] Carrot-lookahead 路径跟随参数
        self.carrot_lookahead_m: float = float(self.declare_parameter("carrot_lookahead_m", 0.9).value)
        self.carrot_scan_max_pts: int   = int(self.declare_parameter("carrot_scan_max_pts", 40).value)
        self._carrot_idx: int = 0          # 上一次 carrot 在路径上的索引

        # ★[B] nav_to_goal_cmd 膨胀碰撞检查距离
        self.nav_check_max_dist_m: float = float(self.declare_parameter("nav_check_max_dist_m", 1.5).value)

        # ★[C] wall_follow fallback 门控
        self.enable_wall_follow_fallback: bool = bool(
            self.declare_parameter("enable_wall_follow_fallback", False).value)
        self.wall_follow_max_sec: float = float(self.declare_parameter("wall_follow_max_sec", 0.8).value)

        # ★[D] SAFE_HOLD 脱离参数
        self.safe_hold_escape_speed: float = float(self.declare_parameter("safe_hold_escape_speed", 0.28).value)
        self.safe_hold_clear_ok_m: float   = float(self.declare_parameter("safe_hold_clear_ok", 0.6).value)
        self.safe_hold_clear_dwell_s: float = float(self.declare_parameter("safe_hold_clear_dwell", 0.5).value)
        self.safe_hold_max_lock_s: float   = float(self.declare_parameter("safe_hold_max_lock_s", 1.5).value)  # ★[B-ARCH] 3.0→1.5s：减少SAFE_HOLD霸占时间，加快恢复
        self.planar_invalid_trigger_s: float = float(self.declare_parameter("planar_invalid_trigger_s", 0.4).value)
        self.planar_invalid_recover_s: float = float(self.declare_parameter("planar_invalid_recover_s", 1.0).value)
        self._planar_invalid_since: float = 0.0    # monotonic 时间：tilt 首次超阈值
        self._planar_recover_since: float = 0.0    # monotonic 时间：tilt 首次低于阈值
        self._safe_hold_since: float = 0.0         # monotonic 时间：进入 SAFE_HOLD
        self._safe_hold_escape_clr_since: float = 0.0  # clearance 持续 OK 的开始时间

        # ★[E] 目标点安全净空与 pullback 参数
        self.goal_min_clearance_normal: float = float(self.declare_parameter("goal_min_clearance_normal", 0.5).value)
        self.goal_min_clearance_door: float   = float(self.declare_parameter("goal_min_clearance_door", 0.35).value)
        self.goal_pullback_m: float           = float(self.declare_parameter("goal_pullback_m", 0.45).value)

        # ════════════════════════════════════════════════════════════════════
        # [H-ARCH] 控制架构参数集中区（软件著作权整理用）
        # ════════════════════════════════════════════════════════════════════
        # ── 控制仲裁参数 ─────────────────────────────────────────────────
        # local_cmd 时间戳有效窗口 (s)：超过此窗口认为 local_mpc 已 stale
        self.control_source_switch_hold_sec: float = float(
            self.declare_parameter("control_source_switch_hold_sec", 0.35).value)
        # SAFE_HOLD 触发条件净空 (m)：障碍物距离 < 此值才允许进入 SAFE_HOLD
        self.safe_hold_trigger_clearance: float = float(
            self.declare_parameter("safe_hold_trigger_clearance", 0.35).value)
        # SAFE_HOLD 解除条件净空 (m)：障碍物距离 > 此值后才允许退出 SAFE_HOLD
        self.safe_hold_release_clearance: float = float(
            self.declare_parameter("safe_hold_release_clearance", 0.60).value)
        # ── 走廊/门目标模式参数 ─────────────────────────────────────────
        # 走廊对中增益（m/s per m of asymmetry）
        # 注意：当 local_mpc 主导时此值被旁路，仅 FALLBACK 时注入
        self.corridor_bias_gain: float = float(
            self.declare_parameter("corridor_bias_gain", 0.12).value)
        # 走廊对中最大注入速度 (m/s)
        self.corridor_bias_max: float = float(
            self.declare_parameter("corridor_bias_max", 0.10).value)
        # 门目标激活距离 (m)：到 gap 中心距离 < 此值时才启用 door_mode
        self.door_mode_trigger_dist: float = float(
            self.declare_parameter("door_mode_trigger_dist", 1.5).value)
        # 门目标释放距离 (m)：通过门后 > 此值才关闭 door_mode
        self.door_mode_release_dist: float = float(
            self.declare_parameter("door_mode_release_dist", 0.5).value)
        # ── 空间冷却参数（已在上方__init__中declare，此处作集中文档注释）──
        # spatial_cooldown_cell_m=1.5m  粗网格格子大小
        # spatial_cooldown_ttl_sec=90s  冷却有效时长
        # spatial_cooldown_penalty=0.50 挡加惩罚系数（叠加到revisit penalty）
        # 冷却触发点：HOVER_SCAN完成、door_give_up、hscan_esc_fail
        # ── NEAR-WALL-HARDSTOP 参数 ──────────────────────────────────────
        # 门口目标：lock=0.5s(PASSAGE_COMMIT)/1.0s(普通door)，阈值 front<0.18|near<0.13
        # 非门口导航(NAV_TO_VIEWPOINT)侧向近墙：lock=2.0s，dot>0.3才触发（方向感知）
        # 普通近墙(非door/非NAV_侧向)：thr=0.25m lock=2.0s ★[B1]was 5s→2s
        # ── 目标视觉伺服参数（集中定义，避免分散） ──────────────────────
        # 以下已在"任务8"中声明，此处仅作架构文档注释
        # target_track_timeout_short=1.5s (lost_short 触发)
        # target_track_timeout_sweep=3.0s (lost_scan 触发)
        # target_track_timeout_abort=6.0s (lost_abort 触发)
        # target_yaw_kp=1.2, target_yaw_max=0.6 rad/s
        # target_forward_speed_min=0.05, target_forward_speed_max=0.25 m/s
        # target_descend_speed=0.10 m/s
        # ── 门/房间分割参数（★[E1] 加速早期检测）──────────────────────
        # _door_hard_cut_frames=5 (★[E1] was 8)：5帧≈2.5s于2Hz更新速率下可确认硬切
        # _door_hard_cut_release_frames=12 (★[E1] was 14)：防抖动但更快响应缺失
        # _door_miss_tolerance=5 (★[E1] was 8)：门短暂消失容忍帧数
        # ── HOVER_SCAN 门口保护参数（★[B2/B3] 强化过门不中断）──────────
        # HOVER-SKIP 净空阈值=0.28m (was 0.38m)：窄门(0.28-0.38m净空)时直接跳过全扫描
        # HOVER_SCAN_ESCAPE 门口保护：door_transit_mode/door_goal时跳过ESCAPE直接恢复NAV

        self.enable_logging = bool(self.declare_parameter("enable_logging", True).value)
        self.log_writer = None
        self.log_file = None
        self._event_log_file = None   # 结构化事件日志（_init_csv_log中创建）
        self._diag_summary_path = None
        self._diag_summary_last_sec: float = 0.0
        self.last_log_flush = self.get_clock().now()
        self.log_flush_interval = 0.5
        if self.enable_logging:
            self._init_csv_log()

        self._init_detector()
        self._init_pub_sub()

        self.create_timer(0.05, self.command_loop)
        if self.vis_enabled:
            self.create_timer(max(0.2, self.vis_interval), self._publish_map_vis)
        self.get_logger().info("target_search rewritten node ready")

    def _init_detector(self) -> None:
        self.yolo_model = None
        try:
            self.yolo_model = YOLO(self.model_path)
            self.get_logger().info(f"YOLO loaded: {self.model_path}")
        except Exception as e:
            self.get_logger().warn(f"YOLO disabled: {e}")
        # ★ 异步推理线程池——max_workers=1确保推理不重叠、不积压
        self._infer_executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="yolo_infer"
            )
        )
        self._infer_future: Optional[concurrent.futures.Future] = None  # 当前待执行的推理任务
        self._infer_drop_count: int = 0   # 被丢帧数（推理占用中

    def _yolo_infer_worker(self, frame: np.ndarray) -> Optional[dict]:
        """后台线程：执行 YOLO 推理，不阻塞 ROS spin 线程。
        只开放读 self.yolo_model（load后不操作其他节点状态）。
        """
        if self.yolo_model is None:
            return None
        try:
            results = self.yolo_model(frame, verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return None
            boxes = results[0].boxes
            conf = (boxes.conf.detach().cpu().numpy()
                    if hasattr(boxes.conf, "detach") else boxes.conf.cpu().numpy())
            xyxy = (boxes.xyxy.detach().cpu().numpy()
                    if hasattr(boxes.xyxy, "detach") else boxes.xyxy.cpu().numpy())
            i = int(np.argmax(conf))
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            return {"conf": float(conf[i]),
                    "cx": 0.5 * (x1 + x2), "cy": 0.5 * (y1 + y2),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        except Exception as exc:
            return {"_error": str(exc)}

    def _init_csv_log(self) -> None:
        log_dir = Path("/home/ubuntu/PX4-Autopilot/log")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts_name = int(time.time())
        # ── ① 飞行轨迹 CSV（原有）──────────────────────────────────────────
        log_path = log_dir / f"target_search_{ts_name}.csv"
        self.log_file = log_path.open("w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "t_sec", "state", "x", "y", "z", "yaw", "vx", "vy", "vz", "speed",
            "lidar_front", "lidar_left", "lidar_right", "lidar_back", "scan_age_sec",
            "armed", "offboard", "nav_state", "target_found", "target_streak", "safety_ok", "safety_reason"
        ])
        # ── ② 结构化事件日志（新增）────────────────────────────────────────
        # 记录关键事件：EKF reset / 起飞 / HOVER_SCAN / 目标发现 / 地图跳变等
        _evt_path = log_dir / f"events_{ts_name}.txt"
        try:
            self._event_log_file = _evt_path.open("w", encoding="utf-8")
            self._event_log_file.write(
                f"# indoor_search events log  started={ts_name}\n"
                f"# format: [wall_time_sec] [tag] message\n"
            )
            self._event_log_file.flush()
        except Exception as _e:
            self._event_log_file = None
            self.get_logger().warn(f"[DIAG] 事件日志创建失败: {_e}")
        # ── ⑤ 目标终端专用日志 target_search.txt （软件著作权整理用）───────────
        _tgt_log_path = log_dir / f"target_search.txt"
        try:
            self._target_log_file = _tgt_log_path.open("w", encoding="utf-8")
            self._target_log_file.write(
                f"# target_search terminal log  started={ts_name}\n"
                f"# CODE_VERSION: 2026-03-10_project_final\n"
                f"# format: [wall_time] [tag] state px_xy world_xy conf trigger\n"
                f"# 记录目标发现→确认→WorldEst→CommitPoint→CommitHover→Conservative→降落全链路事件\n"
                f"# tags: TARGET-DETECT TARGET-WEAK-CONFIRMED TARGET-WORLD-EST "
                f"TARGET-COMMIT-POINT TARGET-HOVER-COMMIT TARGET-CONSERVATIVE-DESCEND TARGET-LAND\n"
            )
            self._target_log_file.flush()
            self.get_logger().info(f"[DIAG] 目标终端日志: {_tgt_log_path}")
        except Exception as _e:
            self._target_log_file = None
            self.get_logger().warn(f"[DIAG] target_search.txt 创建失败: {_e}")
        # ── ③ 启动参数快照（新增）──────────────────────────────────────────
        # 记录所有关键参数值，方便复现问题时确认参数配置
        _params_path = log_dir / f"params_{ts_name}.txt"
        try:
            _param_lines = [
                f"# indoor_search 启动参数快照  ts={ts_name}\n",
                f"# 生成时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                "[地图边界]\n",
                f"  map_bound_xmin={self.bounds[0,0]:.2f}  map_bound_xmax={self.bounds[0,1]:.2f}\n",
                f"  map_bound_ymin={self.bounds[1,0]:.2f}  map_bound_ymax={self.bounds[1,1]:.2f}\n",
                f"  map_bound_zmin={self.bounds[2,0]:.2f}  map_bound_zmax={self.bounds[2,1]:.2f}\n",
                f"  map_res={self.map_res}  grid_shape={self.grid_shape}\n\n",
                "[显示边界]\n",
                f"  display_fixed_bounds={self._display_fixed_bounds}\n",
                f"  display_xmin={self._display_xmin}  display_xmax={self._display_xmax}\n",
                f"  display_ymin={self._display_ymin}  display_ymax={self._display_ymax}\n\n",
                "[飞行参数]\n",
                f"  alt_target_z={self.alt_target_z}  max_speed={self.max_speed}  max_acc={self.max_acc}\n",
                f"  takeoff_stabilize_sec={self.takeoff_stabilize_sec}\n",
                f"  ssg_nav_speed={self.ssg_nav_speed}  ssg_reach_dist={self.ssg_reach_dist}\n",
                f"  hover_scan_duration={self.hover_scan_duration}  hover_scan_yaw_speed={self.hover_scan_yaw_speed}\n\n",
                "[SLAM参数]\n",
                f"  slam_map_topic={getattr(self, '_slam_map_topic', '/map')}\n",
                f"  slam_reset_on_ekf_reset={getattr(self, '_slam_reset_on_ekf', True)}\n",
                f"  map_pause_sec={self._map_pause_sec}\n",
                f"  map_stab_origin_jump_m={self.map_stab_origin_jump_m}\n",
                f"  map_stab_size_jump_cells={self.map_stab_size_jump_cells}\n\n",
                "[LiDAR参数]\n",
                f"  lidar_topic={self.lidar_topic}\n",
                f"  lidar_mount_yaw_deg={self.lidar_mount_yaw_deg}\n",
                f"  lidar_min_valid_dist={self.lidar_min_valid_dist}\n\n",
                "[安全参数]\n",
                f"  safety_max_pose_age_sec={self.safety_max_pose_age_sec}\n",
                f"  obstacle_cmd_stop_dist={self.obstacle_cmd_stop_dist}\n\n",
                "[目标检测参数]\n",
                f"  model_path={self.model_path}\n",
                f"  detect_conf={self.detect_conf}\n",
                f"  target_confirm_hits={self.target_confirm_hits}\n",
                f"  enable_red_detect={self.enable_red_detect}\n\n",
                "[日志文件路径]\n",
                f"  CSV轨迹:     {log_path}\n",
                f"  事件日志:    {_evt_path}\n",
                f"  地图JSON:    /home/ubuntu/PX4-Autopilot/map_output/map_state_latest.json\n",
                f"  地图PNG:     /home/ubuntu/PX4-Autopilot/map_output/map_latest.png\n",
                f"  实时摘要:    /home/ubuntu/PX4-Autopilot/log/diag_summary.txt\n",
                f"  ROS日志:     ~/.ros/log/  (ROS2自动管理)\n",
            ]
            with _params_path.open("w", encoding="utf-8") as _pf:
                _pf.writelines(_param_lines)
        except Exception as _e:
            self.get_logger().warn(f"[DIAG] 参数快照写入失败: {_e}")
        # ── ④ 实时诊断摘要路径（固定文件名，方便 tail -f）──────────────────
        # 每60s覆盖写入一份最新系统状态（命令：tail -f /home/ubuntu/PX4-Autopilot/log/diag_summary.txt）
        self._diag_summary_path = log_dir / "diag_summary.txt"
        self._diag_summary_last_sec: float = 0.0
        # 诊断摘要Timer（60s）
        self.create_timer(60.0, self._diag_summary_cb)
        self.get_logger().info(
            f"[DIAG] 日志文件已创建:\n"
            f"  CSV轨迹:   {log_path}\n"
            f"  事件日志:  {_evt_path}\n"
            f"  参数快照:  {_params_path}\n"
            f"  实时摘要:  {self._diag_summary_path}  (每60s更新, tail -f 监看)"
        )
        # ── ★ 启动版本标记 + 固定规划坐标系诊断 ─────────────────────────────
        # CODE_VERSION: 2026-03-10_project_final
        # 核心改动：map_cb 不再 use_direct=True 覆盖 bounds/res/shape，
        #           改为固定规划坐标系重采样（_plan_* 参数）
        import hashlib as _fid_hl, time as _fid_t
        _fid_data = open(__file__, 'rb').read()
        self._fid_sha = _fid_hl.sha256(_fid_data).hexdigest()[:16]
        self._fid_starttime = _fid_t.strftime('%Y-%m-%dT%H:%M:%S')
        import os as _fid_os
        _fid_mtime = _fid_t.strftime(
            '%Y-%m-%dT%H:%M:%S',
            _fid_t.localtime(_fid_os.path.getmtime(__file__))
        )
        self.get_logger().info(
            f"[FILE-ID] ★ node=target_search  file={__file__}  "
            f"sha256={self._fid_sha}  started={self._fid_starttime}  "
            f"mtime={_fid_mtime}  version=2026-03-09_through_fix"
        )
        # ★ 核心参数摘要（一眼确认运行配置）
        _fp_r_safe   = getattr(self, 'ssg_r_safe', '?')
        _fp_bl_ttl   = 90.0    # target blacklist TTL
        _fp_clr_lam  = 0.4     # A* clearance lambda
        _fp_d_app    = 1.2     # door approach dist
        _fp_d_exit   = 1.0     # door exit dist
        _fp_pullback  = 0.55   # frontier inward pull-back
        _fp_dwell_t  = 0.5    # arrival dwell required
        self.get_logger().info(
            f"[PARAMS-SUMMARY] ★★★ through_fix key params:"
            f" r_safe={_fp_r_safe}  blacklist_ttl={_fp_bl_ttl}s"
            f" astar_clr_lambda={_fp_clr_lam}  door_approach={_fp_d_app}m"
            f" door_exit={_fp_d_exit}m  frontier_pullback={_fp_pullback}m"
            f" arrive_dwell={_fp_dwell_t}s"
        )
        # 同步写入事件日志（方便离线分析运行代码归属）
        if self._event_log_file is not None:
            try:
                self._event_log_file.write(
                    f"[{_fid_t.time():.3f}] [FILE-ID] node=target_search "
                    f"sha256={self._fid_sha} started={self._fid_starttime} "
                    f"file={__file__}\n"
                )
                self._event_log_file.flush()
            except Exception:
                pass
        # ★ install/source 同步校验：SHA256 比对（colcon build --merge-install 是复制不是链接）
        import os as _sync_os
        _install_node_path = _sync_os.path.join(
            _sync_os.path.dirname(__file__).replace(
                '/ros2_ws/src/indoor_search/src',
                '/ros2_ws/install/lib/indoor_search'
            ),
            'target_search_node.py'
        )
        _src_real = _sync_os.path.realpath(__file__)
        if _sync_os.path.exists(_install_node_path):
            _inst_sha = _fid_hl.sha256(open(_install_node_path,'rb').read()).hexdigest()[:16]
            _code_sync_ok = (_inst_sha == self._fid_sha)
        else:
            _inst_sha = 'NOT_FOUND'
            _code_sync_ok = False
        _is_symlink = _sync_os.path.islink(_install_node_path)
        self.get_logger().info(
            f"[CODE-SYNC] running={__file__} sha={self._fid_sha} | "
            f"install={_install_node_path} sha={_inst_sha} | "
            f"is_symlink={_is_symlink} sha_match={_code_sync_ok}"
        )
        if not _code_sync_ok:
            self.get_logger().error(
                f"[CODE-SYNC-MISMATCH] ★★★ install/lib SHA={_inst_sha} ≠ src SHA={self._fid_sha}！"
                f" 运行时代码与当前源码不一致！"
                f" 请立即：colcon build --packages-select indoor_search --merge-install"
            )
        # ★ 跨节点 lidar_mount_yaw_deg 一致性检查
        # 规范：target_search / px4_slam_bridge 均应设为 0.0
        # 任何偏差将导致 SLAM 地图方向与扇区判断不一致，产生 90° 导航偏差
        _expected_yaw = 0.0
        _actual_yaw   = float(self.lidar_mount_yaw_deg)
        if abs(_actual_yaw - _expected_yaw) > 5.0:
            self.get_logger().error(
                f"[LIDAR-YAW-MISMATCH] ★ lidar_mount_yaw_deg={_actual_yaw:.1f}° "
                f"(期望 {_expected_yaw:.1f}°) — 与 px4_slam_bridge_node 不一致！"
                f"SLAM 地图将发生 {_actual_yaw - _expected_yaw:.0f}° 旋转偏差。"
                f"请在 launch 文件中统一设置 lidar_mount_yaw_deg:=0.0"
            )
        else:
            self.get_logger().info(
                f"[LIDAR-YAW-OK] lidar_mount_yaw_deg={_actual_yaw:.1f}° "
                f"== 期望 {_expected_yaw:.1f}° ✓ (与 bridge_node 一致)"
            )

        self.get_logger().info(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════╗\n"
            f"║  target_search_node  CODE_VERSION: 2026-03-10_project_final  ║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  ★ 固定规划坐标系（_plan_*）：                                ║\n"
            f"║    xmin={self._plan_bounds_xy[0,0]:.1f}  xmax={self._plan_bounds_xy[0,1]:.1f}  "
            f"ymin={self._plan_bounds_xy[1,0]:.1f}  ymax={self._plan_bounds_xy[1,1]:.1f}  "
            f"res={self._plan_res:.3f}m  grid={self._plan_ni}×{self._plan_nj}\n"
            f"║  ★ display_fixed_bounds={self._display_fixed_bounds}  "
            f"display=({self._display_xmin},{self._display_ymin})→({self._display_xmax},{self._display_ymax})\n"
            f"║  ★ SLAM 地图仅作为数据源，self.bounds/map_res/grid_shape 永不被覆盖 ║\n"
            f"╚══════════════════════════════════════════════════════════════╝"
        )

    def _log_event(self, tag: str, msg: str) -> None:
        """将关键事件同时写入事件日志文件和 ROS logger。
        tag 示例: EKF-RESET / TAKEOFF-DONE / HOVER-SCAN-START / SEG-OK / TARGET-FOUND / MAP-STAB
        """
        import time as _et
        _wall = _et.time()
        _line = f"[{_wall:.3f}] [{tag}] {msg}\n"
        if self._event_log_file is not None:
            try:
                self._event_log_file.write(_line)
                self._event_log_file.flush()
            except Exception:
                pass
        self.get_logger().info(f"[{tag}] {msg}")

    def _log_target_event(self, tag: str, msg: str,
                          px: tuple = None,
                          world_xy: 'Optional[np.ndarray]' = None,
                          conf: float = -1.0,
                          trigger: str = "") -> None:
        """写入 target_search.txt 目标终端专用日志。
        同时转发到事件日志（events*.txt）。
        """
        import time as _tle_t
        _wall = _tle_t.time()
        _state_str = getattr(self, '_approach_inner_state', 'none')
        _px_str = f"px=({px[0]},{px[1]})" if px else "px=none"
        _wxy_str = (f"world=({world_xy[0]:.2f},{world_xy[1]:.2f})"
                    if world_xy is not None else "world=none")
        _line = (f"[{_wall:.3f}] [{tag}] state={_state_str} "
                 f"{_px_str} {_wxy_str} conf={conf:.2f} trigger={trigger} {msg}\n")
        if self._target_log_file is not None:
            try:
                self._target_log_file.write(_line)
                self._target_log_file.flush()
            except Exception:
                pass
        # 同时写入事件日志文件
        self._log_event(tag, msg)

    def _diag_summary_cb(self) -> None:
        """每60s覆盖写入 diag_summary.txt，供实时监看。
        命令: tail -f /home/ubuntu/PX4-Autopilot/log/diag_summary.txt
        """
        import time as _dt
        if self._diag_summary_path is None:
            return
        try:
            _now_w = _dt.time()
            _free  = int(np.sum(self.grid <= self.prob_free_th))
            _occ   = int(np.sum(self.grid >= self.prob_occ_th))
            _unk   = self.grid_shape[0] * self.grid_shape[1] - _free - _occ
            _state = getattr(self, 'state', 'N/A')
            _ekf   = getattr(self, '_ekf_reset_state', 'N/A')
            _ekf_n = getattr(self, '_ekf_reset_count', 0)
            _stab  = getattr(self, '_map_stability_ok', True)
            _slam_r = getattr(self, '_slam_map_received', False)
            _td    = getattr(self, 'takeoff_done', False)
            _ts_done = getattr(self, '_takeoff_stab_done', False)
            _tf_ok  = getattr(self, '_tf_chain_ok', False)
            _rooms  = len(getattr(self, 'rooms_info', []))
            _doors  = len(getattr(self, 'doors_info', []))
            _tgt    = getattr(self, 'target_found', False)
            _max_b  = float(np.max(self.belief_grid)) if hasattr(self, 'belief_grid') else 0.0
            _px     = getattr(self, '_map_px', 0.0)
            _py     = getattr(self, '_map_py', 0.0)
            _explore = getattr(self, 'explore_state', 'N/A')
            _map_age = 0.0
            if getattr(self, '_last_map_stamp', 0.0) > 0.0:
                _map_age = _dt.monotonic() - self._last_map_stamp

            _lines = [
                f"# indoor_search 实时诊断摘要  更新时间: {_dt.strftime('%Y-%m-%d %H:%M:%S')}\n",
                f"# tail -f {self._diag_summary_path}  |  每60s自动更新\n\n",
                f"[飞行状态]\n",
                f"  state={_state}  explore={_explore}\n",
                f"  takeoff_done={_td}  takeoff_stab_done={_ts_done}\n",
                f"  target_found={_tgt}  max_belief={_max_b:.3f}\n",
                f"  drone_map_pos=({_px:.2f},{_py:.2f})\n\n",
                f"[EKF状态]\n",
                f"  ekf_reset_state={_ekf}  ekf_reset_count={_ekf_n}\n\n",
                f"[SLAM/地图状态]\n",
                f"  slam_map_received={_slam_r}  map_age={_map_age:.1f}s\n",
                f"  map_stability_ok={_stab}\n",
                f"  grid_shape={self.grid_shape}  free={_free} occ={_occ} unk={_unk}\n",
                f"  rooms={_rooms}  doors={_doors}\n\n",
                f"[TF链状态]\n",
                f"  tf_chain_ok={_tf_ok}\n\n",
                f"[日志文件路径]\n",
                f"  CSV轨迹:  /home/ubuntu/PX4-Autopilot/log/target_search_*.csv\n",
                f"  事件日志: /home/ubuntu/PX4-Autopilot/log/events_*.log\n",
                f"  参数快照: /home/ubuntu/PX4-Autopilot/log/params_*.txt\n",
                f"  地图JSON: /home/ubuntu/PX4-Autopilot/map_output/map_state_latest.json\n",
                f"  地图PNG:  /home/ubuntu/PX4-Autopilot/map_output/map_latest.png\n",
            ]
            with self._diag_summary_path.open("w", encoding="utf-8") as _sf:
                _sf.writelines(_lines)
        except Exception as _e:
            self.get_logger().debug(f"[DIAG] diag_summary 写入失败: {_e}")

    def _room_quality_cb(self) -> None:  # 5.3 ROOM-QUALITY 30s 定期日志
        """每30s将房间/门质量指标写入 events_*.log（[ROOM-QUALITY] 标签）。"""
        import time as _rq_t
        _now = _rq_t.monotonic()
        if _now - getattr(self, '_room_quality_last_t', 0.0) < 25.0:
            return  # 节流（计时器抖动保护）
        self._room_quality_last_t = _now
        _rooms  = getattr(self, 'rooms_info', [])
        _doors  = getattr(self, 'doors_info', [])
        _n_rooms = len(_rooms)
        _n_doors = len(_doors)
        _free   = int(np.sum(self.grid <= self.prob_free_th))
        _occ    = int(np.sum(self.grid >= self.prob_occ_th))
        _total  = self.grid_shape[0] * self.grid_shape[1]
        _cov_pct = (_free + _occ) / max(1, _total) * 100.0
        _stable_lo_active = getattr(self, '_stable_lo_enabled', False)
        _lo_occ  = int(np.sum(self._stable_log_odds >= self._stable_lo_occ_th))  if _stable_lo_active else -1
        _lo_free = int(np.sum(self._stable_log_odds <= self._stable_lo_free_th)) if _stable_lo_active else -1
        _recover = getattr(self, '_map_recover_active', False)
        _stab_f  = getattr(self, '_occ_stable_frames', 0)
        _ekf_cnt = getattr(self, '_ekf_reset_count', 0)
        self._log_event(
            "ROOM-QUALITY",
            f"rooms={_n_rooms} doors={_n_doors} "
            f"free={_free} occ={_occ} cov={_cov_pct:.1f}% "
            f"stable_lo={'on' if _stable_lo_active else 'off'} "
            f"lo_occ={_lo_occ} lo_free={_lo_free} "
            f"map_recover={_recover} stable_frames={_stab_f} "
            f"ekf_reset_count={_ekf_cnt}"
        )

    def _map_quality_cb(self) -> None:  # 6.3 MAP-QUALITY 30s 定期日志
        """Layer1(raw)/Layer2(stable)/Layer3(display) 质量分析，写入 events_*.log。"""
        import time as _mq_t
        _now = _mq_t.monotonic()
        if _now - getattr(self, '_map_quality_last_t', 0.0) < 25.0:
            return
        self._map_quality_last_t = _now
        # Layer1 raw
        _raw = getattr(self, '_raw_slam_grid', None)
        if _raw is not None:
            _r_occ  = int(np.sum(_raw >= self.prob_occ_th))
            _r_free = int(np.sum(_raw <= self.prob_free_th))
            _r_unk  = _raw.size - _r_occ - _r_free
        else:
            _r_occ = _r_free = _r_unk = -1
        # Layer2 stable
        _s2 = getattr(self, '_stable_occ_grid', None)
        if _s2 is not None:
            _s_occ  = int(np.sum(_s2 >= self.prob_occ_th))
            _s_free = int(np.sum(_s2 <= self.prob_free_th))
            _s_unk  = _s2.size - _s_occ - _s_free
        else:
            _s_occ = _s_free = _s_unk = -1
        # Layer3 display = self.grid
        _d_occ  = int(np.sum(self.grid >= self.prob_occ_th))
        _d_free = int(np.sum(self.grid <= self.prob_free_th))
        _d_unk  = self.grid.size - _d_occ - _d_free
        # 连通域数量（Layer3）
        try:
            from scipy.ndimage import label as _lbl
            _occ_mask_q = (self.grid >= self.prob_occ_th).astype(np.uint8)
            _, _conn_comp = _lbl(_occ_mask_q)
        except Exception:
            _conn_comp = -1
        # 简易质量分（0~100）： free_cov * 稳定性
        _total = self.grid.size
        _cov   = (_d_free + _d_occ) / max(1, _total)
        _match = (_d_occ / max(1, _s_occ) if _s_occ > 0 else 0.0) if _s2 is not None else 1.0
        _score = int(min(100.0, _cov * 100.0 * min(1.0, _match)))
        self._log_event(
            "MAP-QUALITY",
            f"raw_occ={_r_occ} raw_free={_r_free} raw_unk={_r_unk} "
            f"stable_occ={_s_occ} stable_free={_s_free} stable_unk={_s_unk} "
            f"added={max(0, _d_occ - _s_occ)} removed={max(0, _s_occ - _d_occ)} "
            f"display_occ={_d_occ} display_free={_d_free} "
            f"conn_comp={_conn_comp} score={_score}"
        )

    def _init_pub_sub(self) -> None:
        qos_scan = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.ts_pub = self.create_publisher(TrajectorySetpoint, f"{self.fmu_prefix}/in/trajectory_setpoint", 10)
        self.offboard_pub = self.create_publisher(OffboardControlMode, f"{self.fmu_prefix}/in/offboard_control_mode", 10)
        self.vehicle_cmd_pub = self.create_publisher(VehicleCommand, f"{self.fmu_prefix}/in/vehicle_command", 10)

        # 修复A：只保留一个位姿来源，优先订阅 _v1（仿真中通常存在）
        # 不再同时订阅两个话题，避免乱序/不同reset源导致pos_cb看到"瞬间大跳"污染地图
        self.create_subscription(VehicleLocalPosition, f"{self.fmu_prefix}/out/vehicle_local_position_v1", self.pos_cb, qos_profile_sensor_data)
        self.create_subscription(VehicleAttitude, f"{self.fmu_prefix}/out/vehicle_attitude", self.att_cb, qos_profile_sensor_data)
        # 修复A（status同样问题）：只保留一个status来源
        self.create_subscription(VehicleStatus, f"{self.fmu_prefix}/out/vehicle_status_v1", self.status_cb, qos_profile_sensor_data)

        self.create_subscription(LaserScan, self.lidar_topic, lambda m: self.scan_cb(m, "primary"), qos_scan)
        if self.lidar_fallback_topic and self.lidar_fallback_topic != self.lidar_topic:
            self.create_subscription(LaserScan, self.lidar_fallback_topic, lambda m: self.scan_cb(m, "fallback"), qos_scan)

        self.create_subscription(LocalCommand, "local_cmd", self.local_cb, 10)
        self.create_subscription(ApfCommand, "apf_cmd", self.apf_cb, 10)
        # ── bridge /odom 主订阅（ENU 权威来源）────────────────────────────────
        # 优先于 VehicleLocalPosition 用于 odom 侧位姿显示与 _get_map_pose fallback
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.rgb_topic, self.rgb_cb, 10)
        if self.rgb_fallback_topic and self.rgb_fallback_topic != self.rgb_topic:
            self.create_subscription(Image, self.rgb_fallback_topic, self.rgb_cb, 10)

        # ── 方案C：订阅 slam_toolbox 发布的 /map（OccupancyGrid）────────────
        # ★ FIX(2026-02): 只保留单个 RELIABLE+TRANSIENT_LOCAL 订阅。
        # 之前的"双 QoS 订阅"会导致 map_cb 每次 /map 更新被调用两次：
        #   - 第一次调用写入 self.grid
        #   - 第二次调用在第一次写入未完成时再次覆写，造成竞争与重影
        # slam_toolbox async_slam_toolbox_node 的 /map publisher 使用
        # RELIABLE + TRANSIENT_LOCAL，此处订阅者必须匹配才能接收。
        from rclpy.qos import QoSDurabilityPolicy as DUR
        qos_map_tl = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=DUR.TRANSIENT_LOCAL,
        )
        self._slam_map_topic = self.declare_parameter("slam_map_topic", "/map").value
        # 单一订阅：RELIABLE + TRANSIENT_LOCAL（与 slam_toolbox publisher QoS 一致）
        self.create_subscription(OccupancyGrid, self._slam_map_topic, self.map_cb, qos_map_tl)
        # ── 地图可视化 Image 发布（全局/局部跟随，用于rviz/rqt_image_view调试）───
        if self.vis_enabled:
            self._map_vis_pub = self.create_publisher(Image, "/indoor_search/map_vis", 1)
            self.get_logger().info(f"[VIS] 地图可视化已开启: /indoor_search/map_vis  mode={self.vis_mode}")
        else:
            self._map_vis_pub = None
        # ── slam_gate_mode 发布器（通知 bridge 节点是否强制打开扫描门控）──────
        self._slam_gate_mode_pub = self.create_publisher(String, "/slam_gate_mode", 10)
        # ── FSM 状态广播（供 local_mpc_node 抢占判断）────────────────────────────
        # local_mpc_node 订阅此 topic；当状态为 HOVER_SCAN/SAFE_HOLD/MAP_RECOVER
        # 等时停止发布 local_cmd，避免刷新 last_local_stamp 干扰仲裁。
        self._fsm_state_pub = self.create_publisher(String, '/indoor_search/fsm_state', 1)
        # ★ goal_type 广播：供 APF 判断是否进入门口降斥模式
        self._goal_type_pub = self.create_publisher(String, '/indoor_search/goal_type', 1)
        # ★[A1] SSG A* 路径发布：供 local_mpc 在 NAV_TO_VIEWPOINT 期间跟踪
        #   解决 global_planner 在 SEARCH/NAV_TO_VIEWPOINT 阶段停止发布
        #   → local_fresh=False → CTRL-ARB 降级 → 规划-控制闭环断开的根本问题
        self._ssg_path_pub = self.create_publisher(_SearchPath, 'ssg_path', 10)
        self._ssg_path_pub_t: float = 0.0   # 节流：上次发布 monotonic 时间
        # ── RGB 目标检测调试图像发布器（供 rqt_image_view / rviz 调试用）──────
        # 叠加：OpenCV 检测框（绿）、Kalman 预测中心（蓝）、tracker 框（橙）、状态文字
        self._rgb_debug_pub = self.create_publisher(
            Image, "/indoor_search/target_debug", 1
        )
        self._rgb_debug_log_t: float = 0.0   # 节流：避免频繁打印调试日志
        # slam 地图元数据缓存
        self._slam_map_origin_x: float = 0.0
        self._slam_map_origin_y: float = 0.0
        self._slam_map_res: float = 0.05
        self._slam_map_width: int  = 0
        self._slam_map_height: int = 0
        self._slam_map_received: bool = False
        self._map_cb_cnt: int = 0
        self._grid_hist: list = []   # 时域投票缓存：存储最近 N 帧的离散化类别数组
        self._last_map_stamp: float = 0.0      # monotonic time of last /map callback
        self._map_last_cb_mono: float = 0.0    # Phase-6：上帧回调 monotonic，用于 Hz 计算
        self._map_cb_intervals: list = []      # Phase-6：最近 20 帧间隔历史（滑窗平均 Hz）
        self._map_watchdog_warned: bool = False
        # Phase-6：房间轮廓多边形（lid→list of (ix,iy) 顶点），用于可视化
        self._room_polygons: dict = {}
        # Phase-6：避障硬停止诊断节流时间戳
        self._prox_stop_diag_t: float = 0.0
        # Watchdog：5s 未收到 /map 时报警
        import time as _wdt
        self._map_watchdog_start: float = _wdt.monotonic()
        self.create_timer(5.0, self._map_watchdog_cb)
        self.get_logger().info(
            f"[SLAM] 订阅 OccupancyGrid 话题: {self._slam_map_topic} "
            f"(单一QoS: RELIABLE+TRANSIENT_LOCAL，与 slam_toolbox 匹配)"
        )

        # TF2：查询无人机在 slam map frame 中的位姿
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        # map frame 位姿缓存（command_loop 每帧更新，避免多处重复 TF 查询）
        self._map_px: float = 0.0
        self._map_py: float = 0.0
        self._map_yaw: float = 0.0
        self._last_map_pose: tuple = (0.0, 0.0, 0.0)
        # ★ plan frame 位姿缓存（plan frame ≡ ENU 固定规划网格坐标系）
        # 经 _map_pose_to_plan_pose() 从 map frame 转换而来，补偿 OccupancyGrid
        # origin.orientation 旋转。典型情况（slam 零旋转）：_plan_px ≈ _map_px。
        # 语义约定：凡访问 self.grid（plan 网格）的操作，必须用 _plan_px/_plan_py；
        #          map frame 原始值仅用于 TF/SLAM 日志。
        self._plan_px:  float = 0.0   # 无人机在 plan frame 的 East  坐标（m）
        self._plan_py:  float = 0.0   # 无人机在 plan frame 的 North 坐标（m）
        self._plan_yaw: float = 0.0   # 无人机在 plan frame 的 ENU   yaw（rad）
        # SLAM OccupancyGrid origin.orientation 的 yaw 缓存（map_cb 每帧更新）
        # 典型值 ≈ 0；非零时 map_cb 和 _map_pose_to_plan_pose 据此旋转补偿
        self._slam_origin_yaw: float = 0.0
        # ★ map→odom TF 旋转缓存（用于 _map_vel_to_odom_vel）
        # 初始化为 None（表示未取得），首次 TF 查询成功后填充
        self._map_to_odom_rot_cache = None

        # ── bridge /odom 主坐标源缓存（ENU）──────────────────────────────────
        # 权威来源：由 px4_slam_bridge_node 发布的 /odom（x=East, y=North, z=Up）
        # 用于 odom 侧显示、备用位姿、_get_map_pose fallback、底栏 ENU 显示
        self._odom_x_enu:      float = 0.0
        self._odom_y_enu:      float = 0.0
        self._odom_z_up:       float = 0.0
        self._odom_yaw_enu:    float = 0.0
        self._odom_stamp_mono: float = 0.0
        self._odom_received:   bool  = False
        self._odom_pose_stale: bool  = True   # True=超过 0.5s 未更新
        # map pose 缓存质量字段（_get_map_pose 写入，command_loop 读取）
        self._map_pose_valid:  bool  = False
        self._map_pose_stale:  bool  = True
        self._map_pose_source: str   = 'invalid'
        self._map_pose_age_sec: float = 999.0
        self._last_map_pose_mono: float = 0.0  # monotonic 时间，_get_map_pose 每次成功写入
        # TF 质量标志（_tf_chain_watchdog_cb 写入，供 command_loop 读取）
        self._tf_quality_ok:     bool = False
        self._tf_quality_reason: str  = 'not_checked'
        # ── 平面位姿有效性（基于机体倾角，不依赖 Euler 轴约定）────────────────
        self._planar_pose_valid: bool = True             # tilt 未超阈值 → 2D 位姿有效
        self._planar_pose_invalid_reason: str = 'not_checked'
        self._att_tilt_local: float = 0.0               # 本地计算的机体倾角（°）
        self._planar_tilt_threshold_deg: float = float(
            self.declare_parameter('planar_tilt_threshold_deg', 20.0).value
        )

        # TF chain watchdog：周期检查 map→odom / odom→base_link / base_link→laser_link
        self._tf_chain_ok:        bool  = False   # 上一次检查是否全部通畅
        self._tf_chain_last_ok:   float = 0.0     # 最后一次全通畅的 monotonic 时间
        self._tf_chain_warn_since: float = 0.0    # 开始异常的 monotonic 时间
        self._tf_chain_paused:    bool  = False   # TF 断链时暂停建图/规划
        import time as _tfw
        self._tf_chain_node_start: float = _tfw.monotonic()
        # TF watchdog 每 5s 运行一次（与 map watchdog 错开避免日志爆刷）
        self.create_timer(5.0, self._tf_chain_watchdog_cb)
        # ROOM-QUALITY 定期质量日志（5.3）：每30s写入 events_*.log
        self.create_timer(30.0, self._room_quality_cb)
        # MAP-QUALITY 定期地图质量日志（6.3）：每30s写入 events_*.log
        self.create_timer(30.0, self._map_quality_cb)

        # slam_toolbox reset 服务钩子（EKF reset 时可选调用）
        # 参数 slam_reset_on_ekf_reset: true 时在重置时调用 /slam_toolbox/clear_changes
        self._slam_reset_on_ekf: bool = bool(
            self.declare_parameter("slam_reset_on_ekf_reset", True).value
        )
        # ── slam_toolbox /clear_changes 服务客户端 ──────────────────────────
        # 仅在 slam_reset_on_ekf_reset=true 时创建，避免无用连接
        self._slam_clear_cli = None
        if self._slam_reset_on_ekf:
            try:
                from slam_toolbox.srv import Clear as _SlamClear
                self._slam_clear_cli = self.create_client(
                    _SlamClear, '/slam_toolbox/clear_changes'
                )
                self.get_logger().info(
                    "[SLAM] slam_clear_changes 客户端已创建 "
                    "(slam_reset_on_ekf_reset=true)"
                )
            except ImportError:
                self.get_logger().warn(
                    "[SLAM] slam_toolbox.srv.Clear 导入失败，"
                    "slam_reset_on_ekf_reset 功能自动禁用"
                )
                self._slam_reset_on_ekf = False

        # ── slam_toolbox /pause_new_measurements Pause 服务客户端 ───────────
        # 大幅 EKF 跳变时调用 Pause(True) 停止 slam 接收扫描帧，
        # 避免错位扫描被积分入地图（clear_changes 只清优化史，不停扫描积分）。
        # 恢复时调用 Pause(False)。
        self._slam_pause_cli = None
        self._slam_pause_active: bool = False   # 记录 slam 当前是否处于暂停状态
        # 大幅跳变暂停时长（EKF jump > large_jump_th_m 时用此值替代默认 map_pause_sec）
        self._map_pause_sec_large: float = float(
            self.declare_parameter("map_pause_sec_large", 8.0).value
        )
        self._ekf_large_jump_th_m: float = float(
            self.declare_parameter("ekf_large_jump_th_m", 5.0).value
        )
        try:
            from slam_toolbox.srv import Pause as _SlamPause  # noqa
            self._slam_pause_cli = self.create_client(
                _SlamPause, '/slam_toolbox/pause_new_measurements'
            )
            self.get_logger().info(
                "[SLAM] slam_pause 客户端已创建 (Pause service available)"
            )
        except ImportError:
            self.get_logger().warn(
                "[SLAM] slam_toolbox.srv.Pause 导入失败，大幅EKF reset 时无法暂停 slam 扫描积分"
            )

        # ── 启动身份/配置日志（同时写入 log/events_*.log，满足「日志输出到log目录」要求）──
        import os as _osf
        _sf_msg = (
            f"__file__={_osf.path.abspath(__file__)} "
            f"CODE_VERSION=2026-03-10_project_final sha={self._fid_sha}"
        )
        _sc_msg = (
            f"slam_reset_on_ekf={self._slam_reset_on_ekf} "
            f"pause_cli={'OK' if self._slam_pause_cli is not None else 'NONE'} "
            f"clear_cli={'OK' if self._slam_clear_cli is not None else 'NONE'} "
            f"map_pause_sec={self._map_pause_sec} "
            f"map_pause_sec_large={self._map_pause_sec_large} "
            f"ekf_large_jump_th={self._ekf_large_jump_th_m}m"
        )
        self._log_event("SEARCH-FILE", _sf_msg)      # → log/events_*.log
        self._log_event("SLAM-RESET-CONFIG", _sc_msg)  # → log/events_*.log

    # ================================================================
    # ================================================================
    # slam_gate_mode：通知 bridge 节点门控模式
    # ================================================================
    def _set_slam_gate_mode(self, mode: str) -> None:
        """向 /slam_gate_mode 话题发布门控模式指令。
        mode='hover_strict' → HOVER_SCAN 期间宽松但不完全绕过的门控（首选）
        mode='normal'       → 恢复正常姿态/速度门控
        mode='force_open'   → 仅调试用，正常流程禁止，会自动降级为 hover_strict
        重复值不再重复发布，避免话题刷屏。
        """
        if mode == 'force_open':
            _dbg = bool(getattr(self, '_slam_gate_force_open_debug', False))
            if not _dbg:
                self.get_logger().warn(
                    "[SLAM-GATE-MODE-WARN] 试图使用 force_open，"
                    "正常流程应使用 hover_strict，已自动降级"
                )
                mode = 'hover_strict'
        if mode == self._slam_gate_mode_current:
            return
        self._slam_gate_mode_current = mode
        msg = String()
        msg.data = mode
        self._slam_gate_mode_pub.publish(msg)
        self.get_logger().info(f"[SLAM-GATE-MODE] → {mode!r}")

    # ================================================================
    # slam_pause_scan：通过 slam_toolbox Pause 服务暂停/恢复扫描摄入
    # ================================================================
    def _slam_pause_scan(self, pause: bool) -> None:
        """调用 /slam_toolbox/pause_new_measurements 暂停或恢复 SLAM 扫描积分。

        pause=True  → slam_toolbox 停止接收新扫描帧（大幅EKF跳变防护）
        pause=False → slam_toolbox 恢复接收扫描（恢复建图）

        注意：即使 Pause 服务不可用（导入失败/服务未就绪），也只打印 WARN，不抛异常。
        Bridge 节点的 TF-FREEZE 机制已从发布端阻断大多数错位扫描，
        slam Pause 是第二道防线（阻断 slam_toolbox 本身的接收端）。
        """
        cli = getattr(self, '_slam_pause_cli', None)
        if cli is None:
            return  # Pause 服务不可用（导入失败），静默跳过
        if not cli.service_is_ready():
            self.get_logger().warn(
                f"[SLAM-PAUSE] /slam_toolbox/pause_new_measurements 服务未就绪 "
                f"({'pause' if pause else 'resume'} 请求被跳过)"
            )
            return
        try:
            # slam_toolbox Pause.Request 是空请求（无任何字段），调用一次就切换 pause/resume
            # 只有当期望状态与当前状态不同时才调用，避免误toggle
            if pause == getattr(self, '_slam_pause_active', False):
                return  # 已处于目标状态，不重复调用
            from slam_toolbox.srv import Pause as _SlamPause
            req = _SlamPause.Request()  # 空请求，无需设置任何字段
            fut = cli.call_async(req)
            self._slam_pause_active = pause
            action_str = "暂停" if pause else "恢复"

            def _done_cb(f: "rclpy.Future") -> None:
                try:
                    f.result()
                    self.get_logger().warn(
                        f"[SLAM-PAUSE] {action_str} 扫描摄入成功 "
                        f"(pause_active={self._slam_pause_active})"
                    )
                except Exception as _e:
                    self.get_logger().warn(
                        f"[SLAM-PAUSE] {action_str} 调用失败: {_e}"
                    )
                    self._slam_pause_active = not pause  # 回滚状态

            fut.add_done_callback(_done_cb)
        except Exception as _ex:
            self.get_logger().warn(f"[SLAM-PAUSE] call_async 异常: {_ex}")

    # SLAM watchdog：5s 未收到 /map 时打印诊断信息
    # ================================================================
    def _map_watchdog_cb(self) -> None:
        import time as _wt
        if self._slam_map_received:
            self._map_watchdog_warned = False
            return
        elapsed = _wt.monotonic() - self._map_watchdog_start
        if elapsed < 5.0:
            return
        self._map_watchdog_warned = True
        self.get_logger().warn(
            f"[SLAM-WD] {elapsed:.0f}s 未收到 /map！排查清单：\n"
            f"  1) ros2 topic hz /scan        → 应 >10Hz（bridge在发布？）\n"
            f"  2) ros2 topic info /map -v    → Publisher QoS 应 TRANSIENT_LOCAL\n"
            f"  3) ros2 run tf2_ros tf2_echo odom base_link  → 应有数据\n"
            f"  4) ros2 run tf2_ros tf2_echo base_link laser_link → 应有静态TF\n"
            f"  5) 检查 slam_toolbox 日志: tail -20 /tmp/slam_toolbox.log\n"
            f"  6) use_sim_time:=true 时需要 /clock 话题正常发布"
        )

    # ================================================================
    # TF chain watchdog：每 5s 检查 map→odom / odom→base_link / base_link→laser_link
    # ================================================================
    def _tf_chain_watchdog_cb(self) -> None:
        """
        周期性检查 TF 链完整性。
        TF 断链时：打印明确错误并设置 _tf_chain_paused=True（暂停建图/规划）。
        TF 恢复时：打印 RESUME 并清除暂停标志。

        检查链路：
          map → odom          (slam_toolbox 发布)
          odom → base_link    (px4_slam_bridge 发布，频率 ~50Hz)
          base_link → laser_link (静态 TF，bridge 启动时发布)
        """
        import time as _wt
        now_mono = _wt.monotonic()
        warmup_sec = 20.0   # 启动后 20s 内不输出错误（等待节点全部就绪）
        uptime = now_mono - self._tf_chain_node_start

        # 检查三段 TF
        _chains = [
            ("map",        "odom"),
            ("odom",       "base_link"),
            ("base_link",  "laser_link"),
        ]
        results: dict = {}
        for parent, child in _chains:
            try:
                self._tf_buffer.lookup_transform(
                    parent, child,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                results[f"{parent}→{child}"] = True
            except Exception as err:
                results[f"{parent}→{child}"] = False

        all_ok = all(results.values())

        if all_ok:
            if not self._tf_chain_ok:
                # 从断链恢复
                broken_duration = now_mono - self._tf_chain_warn_since if self._tf_chain_warn_since > 0 else 0.0
                self.get_logger().info(
                    f"[TF-WD] ✓ TF链全部恢复(断链持续{broken_duration:.1f}s)：{list(results.keys())} "
                    f"→ 恢复建图/规划"
                )
                self._tf_chain_paused = False
                self._tf_chain_warn_since = 0.0
            else:
                self.get_logger().debug(
                    f"[TF-WD] ✓ {' | '.join(f'{k}=OK' for k in results)}"
                )
            self._tf_chain_ok = True
            self._tf_chain_last_ok = now_mono
        else:
            failed = [k for k, v in results.items() if not v]
            ok     = [k for k, v in results.items() if v]
            if uptime < warmup_sec:
                self.get_logger().debug(
                    f"[TF-WD] 启动预热中({uptime:.0f}s<{warmup_sec}s) 等待TF: {failed}"
                )
            else:
                if not self._tf_chain_paused:
                    self._tf_chain_warn_since = now_mono
                self.get_logger().error(
                    f"[TF-WD] ❌ TF链断链！缺失:{failed} 正常:{ok}\n"
                    f"  → 建图/规划已暂停，请检查：\n"
                    f"  ros2 run tf2_ros tf2_echo odom base_link\n"
                    f"  ros2 run tf2_ros tf2_echo base_link laser_link\n"
                    f"  ros2 run tf2_ros tf2_echo map odom\n"
                    f"  (若 map→odom 缺失：slam_toolbox 未运行 或 /scan 无数据)"
                )
                self._tf_chain_paused = True   # 暂停标志（command_loop 可检查此标志）
            self._tf_chain_ok = False

        # ── 质量检查（链路存在但数据可能过期/不可信）─────────────────────────
        # 只在链路存在时做质量评估
        _quality_issues = []
        if all_ok:
            import time as _tq_t
            _now_q = _tq_t.monotonic()
            # 检查 /odom 新鲜度
            _odom_age_q = _now_q - getattr(self, '_odom_stamp_mono', 0.0)
            if getattr(self, '_odom_received', False):
                if _odom_age_q > 0.5:
                    _quality_issues.append(f"odom_stale({_odom_age_q:.1f}s>0.5s)")
            else:
                if uptime > 5.0:
                    _quality_issues.append("odom_not_received")
            # 检查 map pose 年龄
            _map_pose_age_q = _now_q - getattr(self, '_last_map_pose_mono', 0.0)
            if _map_pose_age_q > 2.0 and getattr(self, '_slam_map_received', False):
                _quality_issues.append(f"map_pose_stale({_map_pose_age_q:.1f}s>2s)")
            # 计算 odom 经 map←odom 变换后与 map→base_link 的偏差（可选，昂贵）
            # 只在两者都有时才检查
            _mp_src = getattr(self, '_map_pose_source', '')
            if _mp_src == 'tf_map_base' and getattr(self, '_odom_received', False):
                try:
                    t_mo = self._tf_buffer.lookup_transform(
                        'map', 'odom', rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.05))
                    _tx = float(t_mo.transform.translation.x)
                    _ty = float(t_mo.transform.translation.y)
                    _q_mo = [t_mo.transform.rotation.x, t_mo.transform.rotation.y,
                             t_mo.transform.rotation.z, t_mo.transform.rotation.w]
                    _rot_mo = R.from_quat(_q_mo)
                    _op = _rot_mo.apply([self._odom_x_enu, self._odom_y_enu, 0.0])
                    _mx_est = _tx + _op[0]
                    _my_est = _ty + _op[1]
                    _diff = float(np.hypot(_mx_est - self._map_px, _my_est - self._map_py))
                    if _diff > 1.0:
                        _quality_issues.append(
                            f"map_odom_diverged({_diff:.2f}m>1.0m)"
                        )
                except Exception:
                    pass

            if _quality_issues:
                self._tf_quality_ok = False
                self._tf_quality_reason = ' | '.join(_quality_issues)
                self.get_logger().warn(
                    f"[TF-WD-QUALITY] TF链存在但质量不佳: {self._tf_quality_reason}"
                )
            else:
                self._tf_quality_ok = True
                self._tf_quality_reason = 'ok'
        else:
            self._tf_quality_ok = False
            self._tf_quality_reason = f"chain_broken:{failed}"

    # ================================================================
    # slam_toolbox /map 回调：将 slam 地图写入内部 self.grid
    # ★ 唯一写入源：self.grid 只在此函数内被赋值（__init__初始化除外）
    # ================================================================
    def map_cb(self, msg: OccupancyGrid) -> None:
        """
        接收 slam_toolbox 发布的 OccupancyGrid，转换为内部 uint8 栅格。

        slam_toolbox 值约定：
          -1      → 未知 (unknown)
          0       → 空闲 (free)
          1~100   → 占据 (occupied, 通常100)

        内部 grid 约定（uint8）：
          128 → 未知
          50  → 空闲
          220 → 占据

        策略：
          ★ 唯一占据来源：self.grid 只在此函数中写入（__init__初始化除外）。
            update_map_from_scan 只更新 belief_grid（目标概率），不写 occupancy。
          use_direct=True：直接采用 slam map 原生坐标系（origin/res/W/H），
            彻底消除"固定bounds重采样"导致的偏移/扭曲/重影问题。
        """
        import time as _mt
        if msg is None:
            return

        # 首次收到地图时打印唯一来源声明
        if not getattr(self, '_grid_source_logged', False):
            self._grid_source_logged = True
            self.get_logger().info(
                "[SLAM-MAP] ★ occupancy 唯一来源 = SLAM /map (map_cb)；"
                "update_map_from_scan 仅更新 belief，不写 occupancy/self.grid"
            )
        meta = msg.info
        sres = float(meta.resolution)
        W    = int(meta.width)
        H    = int(meta.height)
        ox   = float(meta.origin.position.x)
        oy   = float(meta.origin.position.y)

        if sres <= 0 or W == 0 or H == 0:
            self.get_logger().warn("[SLAM-MAP] 收到空 OccupancyGrid（W/H/res 为0），忽略")
            return

        # ── 地图稳定性跳变检测（Task3）──────────────────────────────────────
        # 首次收到地图不检测跳变；后续每帧与上一帧比较
        import time as _stab_mt
        if self._slam_map_received:
            _origin_jump = float(np.hypot(ox - self._slam_map_origin_x,
                                          oy - self._slam_map_origin_y))
            _old_size = self._slam_map_width * self._slam_map_height
            _new_size = W * H
            _size_delta = _new_size - _old_size
            # ★ Phase5: 区分"地图扩展"（正常）和"真实跳变"
            # 真实跳变 = 原点偏移过大 OR 地图异常缩小（SLAM重框）
            # 地图扩展 = 新区域被探索，地图增大 —— 不触发 MAP-STAB，不清空门缓存
            _size_shrank = _size_delta < -self.map_stab_size_jump_cells
            _jumped = (_origin_jump > self.map_stab_origin_jump_m or _size_shrank)
            _expansion_only = (not _jumped) and (_size_delta > 5000)

            if _expansion_only:
                self.get_logger().debug(
                    f"[MAP-EXPAND] 地图扩展 {_old_size}→{_new_size}格"
                    f"(+{_size_delta}) origin_ok={_origin_jump:.3f}m"
                )
            elif _jumped:
                self._map_stability_ok = False
                self._map_stab_pause_until = _stab_mt.monotonic() + self.map_stab_pause_sec
                # ★ 跳变立即清空时域投票缓存，防止旧坐标系帧参与新一轮投票
                _hist_len_before = len(self._grid_hist)
                self._grid_hist = []
                # ★ 跳变使门候选坐标失效，一并清空
                self._door_cand_cache = {}
                self._door_confirmed_stable = []
                # ★ Phase5: 清空世界坐标门缓存
                if hasattr(self, '_door_world_cache'):
                    self._door_world_cache.clear()
                # C3: 记录本次跳变时间，过滤 60s 内的记录
                _now_stab = _stab_mt.monotonic()
                self._map_stab_jump_times.append(_now_stab)
                self._map_stab_jump_times = [
                    t for t in self._map_stab_jump_times if _now_stab - t < 60.0
                ]
                _jump_cnt_60s = len(self._map_stab_jump_times)
                self._log_event(
                    "MAP-STAB",
                    f"SLAM地图跳变 origin_jump={_origin_jump:.2f}m "
                    f"size_delta={_size_delta:+d}格 shrank={_size_shrank} → 规划暂停 {self.map_stab_pause_sec:.1f}s "
                    f"origin=({ox:.2f},{oy:.2f}) W={W} H={H} "
                    f"近60s内跳变次数={_jump_cnt_60s} "
                    f"清空时域缓存(before={_hist_len_before}帧)"
                )
                # C3: 60s 内 >=3 次跳变，打印频繁警告
                if _jump_cnt_60s >= 3:
                    self.get_logger().warn(
                        f"[MAP-STAB] ⚠ SLAM 地图跳变频繁！近60s内 {_jump_cnt_60s} 次 → "
                        f"建议检查EKF/LiDAR质量。不自动重启SLAM"
                    )
                    self._map_stab_freq_warned = True
                else:
                    self._map_stab_freq_warned = False
                # ★ 大幅跳变（>3× 阈值）: 强制重新悬停扫描以修复建图
                _large_jump_th = self.map_stab_origin_jump_m * 3.0
                if _origin_jump > _large_jump_th:
                    self.get_logger().warn(
                        f"[MAP-STAB] 大幅跳变 {_origin_jump:.2f}m > {_large_jump_th:.2f}m → "
                        f"强制切换 HOVER_SCAN 重建地图 + slam Pause"
                    )
                    # 延长规划暂停（避免在不稳定地图上发出 A* 指令）
                    _large_stab_pause = max(self.map_stab_pause_sec, 5.0)
                    self._map_stab_pause_until = _stab_mt.monotonic() + _large_stab_pause
                    self._map_frozen = True
                    self.explore_state = "HOVER_SCAN"
                    self.hover_scan_start_time = None
                    self._hover_scan_hold_pos = None
                    self._hover_scan_anchor_map = None
                    self._hover_scan_last_yaw = None
                    self.hover_scan_accum_angle = 0.0
                    self._hover_valid_frames = 0
                    self.ssg_viewpoint_goal = None
                    # 房间/门语义信息基于旧坐标系，一并清空
                    self.rooms_info = []
                    self.doors_info = []
                    # slam Pause: 大幅 MAP-STAB 跳变时暂停扫描积分，防止错位地图持续增长
                    # 恢复由 MAP-STAB-RECOVER 逻辑触发（_map_stability_ok 恢复后）
                    self._slam_pause_scan(True)
            elif not self._map_stability_ok and _stab_mt.monotonic() > self._map_stab_pause_until:
                self._map_stability_ok = True
                self._log_event("MAP-STAB-RECOVER", "地图稳定恢复 → 路径规划恢复")
                # 若 MAP-STAB 大幅跳变时暂停了 slam 扫描，现在随地图稳定一起恢复
                if getattr(self, '_slam_pause_active', False):
                    self._slam_pause_scan(False)
                    self.get_logger().info("[MAP-STAB-RECOVER] slam 扫描摄入已恢复（MAP-STAB Pause → Resume）")

        # 缓存 slam meta（无论是否稳定都更新，确保下帧比较基准正确）
        self._slam_map_res     = sres
        self._slam_map_width   = W
        self._slam_map_height  = H
        self._slam_map_origin_x = ox
        self._slam_map_origin_y = oy
        self._last_map_stamp   = _mt.monotonic()

        # ★ MAP-STAB early return：跳变/不稳定期间不写 self.grid
        # Meta 已更新（确保下帧比较基准正确），但本帧 grid 跳过，防止脏帧叠加
        if not self._map_stability_ok:
            _remaining = max(0.0, self._map_stab_pause_until - _mt.monotonic())
            self.get_logger().debug(
                f"[MAP-STAB] 跳变期间跳过 grid 写入（pause剩余 {_remaining:.1f}s）"
            )
            return

        # slam_data shape: (H, W) row-major (row=y_idx, col=x_idx)
        slam_data = np.array(msg.data, dtype=np.int8).reshape((H, W))

        # ── 统计 slam 地图内的值分布（每10次打印一次）──────────────────────
        slam_free_n = int(np.sum((slam_data >= 0) & (slam_data <= self._slam_raw_free_th)))
        slam_occ_n  = int(np.sum(slam_data >= self._slam_raw_occ_th))
        slam_unk_n  = int(np.sum(slam_data < 0))

        # ── 保存 SLAM 原生地图（调试诊断用，不影响规划坐标系）──────────────────────
        self._slam_native_grid = slam_data  # (H, W) int8
        self._slam_native_meta = {"ox": ox, "oy": oy, "sres": sres, "W": W, "H": H}
        slam_known = slam_free_n + slam_occ_n

        # ── ★ 核心修复：固定规划坐标系重采样（原 use_direct=True 是根因，已废弃）──────
        # 之前 use_direct=True 每次回调都覆盖 self.bounds/map_res/grid_shape，
        # 使规划坐标系跟随 SLAM 动态 origin 漂移——这是地图重叠/漂移/显示比例不固定的根因。
        #
        # 现在改为：
        #   ① self.bounds / self.map_res / self.grid_shape  永不被修改（固定世界坐标）
        #   ② 每帧将 SLAM 当前 origin/res 用于将固定世界坐标映射到 SLAM 当前格索引：
        #        slam_col[i] = floor((world_x[i] - slam_ox) / slam_res)
        #        slam_row[j] = floor((world_y[j] - slam_oy) / slam_res)
        #   ③ SLAM 回环/扩展后 slam_ox/slam_oy 变化，映射自动更新到正确位置（这是正确做法）
        sx_p  = float(self._plan_bounds_xy[0, 0])
        sy_p  = float(self._plan_bounds_xy[1, 0])
        res_p = self._plan_res
        ni_p  = self._plan_ni
        nj_p  = self._plan_nj
        wx = sx_p + np.arange(ni_p, dtype=float) * res_p   # 固定世界 x 坐标序列（不变）
        wy = sy_p + np.arange(nj_p, dtype=float) * res_p   # 固定世界 y 坐标序列（不变）
        # ── ★ SE(2) 完整变换：提取 origin.orientation 旋转，避免扭曲 ─────────────
        # slam_toolbox 正常情况下 orientation=identity（yaw=0），走快速路径。
        # loop closure 后若 SLAM 输出非零旋转栅格，使用全 2D 旋转映射。
        _oq_map = msg.info.origin.orientation
        import math as _mc_math
        try:
            _slam_yaw_raw = float(
                R.from_quat([_oq_map.x, _oq_map.y, _oq_map.z, _oq_map.w]).as_euler('zyx')[0]
            )
        except Exception:
            _slam_yaw_raw = 0.0
        self._slam_origin_yaw = _slam_yaw_raw  # 缓存供 _map_pose_to_plan_pose 使用
        if abs(_slam_yaw_raw) < 1e-4:
            # 快速路径：无旋转（典型情况），plan 坐标 = SLAM 世界坐标
            slam_col = np.floor((wx - ox) / sres).astype(int)  # 固定世界 x → SLAM 当前列索引
            slam_row = np.floor((wy - oy) / sres).astype(int)  # 固定世界 y → SLAM 当前行索引
            c2d  = slam_col[:, np.newaxis]   # (ni_p, 1) 广播
            r2d  = slam_row[np.newaxis, :]   # (1,  nj_p) 广播
        else:
            # 完整 SE(2) 路径：对每个 plan 格计算逆旋转后的 SLAM 格索引
            # 变换：plan 世界点 (wx[i], wy[j]) → 相对 origin 平移 → 逆旋转 → 除 sres 得格号
            _c_r = _mc_math.cos(-_slam_yaw_raw)   # 逆旋转余弦
            _s_r = _mc_math.sin(-_slam_yaw_raw)   # 逆旋转正弦
            wx_rel = wx - ox   # (ni_p,) 相对 SLAM origin 的 x
            wy_rel = wy - oy   # (nj_p,) 相对 SLAM origin 的 y
            # 旋转后局部坐标（2D outer product 展开为 ni_p×nj_p 矩阵）
            wx_loc = (_c_r * wx_rel[:, np.newaxis]
                      - _s_r * wy_rel[np.newaxis, :])   # (ni_p, nj_p)
            wy_loc = (_s_r * wx_rel[:, np.newaxis]
                      + _c_r * wy_rel[np.newaxis, :])   # (ni_p, nj_p)
            c2d = np.floor(wx_loc / sres).astype(int)   # (ni_p, nj_p)
            r2d = np.floor(wy_loc / sres).astype(int)   # (ni_p, nj_p)
        valid = (c2d >= 0) & (c2d < W) & (r2d >= 0) & (r2d < H)
        valid_count = int(np.sum(valid))
        total_cells = ni_p * nj_p
        c_clip = np.clip(c2d, 0, W - 1)
        r_clip = np.clip(r2d, 0, H - 1)
        raw    = slam_data[r_clip, c_clip].astype(np.int32)   # (ni_p, nj_p)
        # ★ 全量重建（不继承旧图）→ 彻底消灭历史残影
        # 每帧从 unknown 重建，只写当前 SLAM /map 覆盖的区域。
        # 未覆盖区域保持 unknown（128），不保留任何历史 occupied/free。
        # 时域投票缓存(B2) 在 window 帧内保持障碍连续性，超期自动消退，避免闪烁。
        # ★ 这是消灭「SLAM origin/size 跳变后旧坐标系墙体残留」的关键修复。
        new_grid = np.full((ni_p, nj_p), 128, dtype=np.uint8)
        _r_occ  = self._slam_raw_occ_th   # 默认65：≥65才是可信占据，抑制中间值假墙
        _r_free = self._slam_raw_free_th  # 默认30：0-30才是可信空闲
        new_grid[valid & (raw >= _r_occ)]               = 220  # 占据
        new_grid[valid & (raw >= 0) & (raw <= _r_free)] = 50   # 空闲（可信）
        # 中间值(31~_r_occ-1)及 raw<0 均保持 unknown(128)，不误标墙体

        # ── Layer1: 原始SLAM重投影快照（逐帧覆盖，供诊断/比较）──────────────
        self._raw_slam_grid[:] = new_grid
        # [MAP-RAW] 1Hz: Layer1 raw projection stats（在时域/空间滤波之前）
        import time as _mr_t
        _mr_now = _mr_t.monotonic()
        if _mr_now - getattr(self, '_map_raw_log_t', 0.0) > 1.0:
            self._map_raw_log_t = _mr_now
            _raw_free_n = int(np.sum(new_grid <= self.prob_free_th))
            _raw_occ_n  = int(np.sum(new_grid >= self.prob_occ_th))
            _raw_unk_n  = int(np.sum(new_grid == 128))
            self.get_logger().debug(
                f"[MAP-RAW] free={_raw_free_n} occ={_raw_occ_n} unk={_raw_unk_n} "
                f"slam_occ_input={slam_occ_n}"
            )

        # ── Layer2/B2: 时域融合（优先 log-odds EMA，fallback 投票滤波）────────
        _occ_before = int(np.sum(new_grid >= self.prob_occ_th))
        if self._stable_lo_enabled:
            # 5.2 log-odds EMA 稳定融合（替代投票滤波，内存更低收敛更平滑）
            _lo_obs   = self._stable_lo_obs
            _lo_alpha = self._stable_lo_alpha
            _lo_max   = self._stable_lo_max
            _occ_raw  = valid & (new_grid >= self.prob_occ_th)
            _free_raw = valid & (new_grid <= self.prob_free_th)
            # EMA 更新：仅对 SLAM 已覆盖的格子（有效区域）更新 log-odds
            self._stable_log_odds[_occ_raw]  = (
                _lo_alpha * self._stable_log_odds[_occ_raw]
                + (1.0 - _lo_alpha) * _lo_obs
            )
            self._stable_log_odds[_free_raw] = (
                _lo_alpha * self._stable_log_odds[_free_raw]
                + (1.0 - _lo_alpha) * (-_lo_obs)
            )
            np.clip(self._stable_log_odds, -_lo_max, _lo_max, out=self._stable_log_odds)
            # 从 log-odds 导出 Layer2 稳定占据图
            self._stable_occ_grid[:] = 128
            self._stable_occ_grid[self._stable_log_odds >= self._stable_lo_occ_th]  = 220
            self._stable_occ_grid[self._stable_log_odds <= self._stable_lo_free_th] = 50
            # Layer3 = 空间去噪(B3) 对 Layer2 的输出 → 用 copy 避免 B3 污染 Layer2
            new_grid = self._stable_occ_grid.copy()
        elif self._map_temporal_enabled and self._map_temporal_window >= 2:
            # 降级路径：B2 投票滤波（stable_lo_enabled=False 时保留）
            _cat = np.zeros(new_grid.shape, dtype=np.uint8)
            _cat[valid & (new_grid <= self.prob_free_th)]  = 1
            _cat[valid & (new_grid >= self.prob_occ_th)]   = 2
            self._grid_hist.append(_cat)
            if len(self._grid_hist) > self._map_temporal_window:
                self._grid_hist.pop(0)
            if len(self._grid_hist) >= 2:
                _stack = np.stack(self._grid_hist, axis=0)  # (N, ni, nj)
                _vote_occ  = np.sum(_stack == 2, axis=0)
                _vote_free = np.sum(_stack == 1, axis=0)
                _known_any = np.any(_stack > 0, axis=0)
                _to_occ  = _known_any & (_vote_occ  >= self._map_occ_vote_th)
                _to_free = _known_any & (_vote_free >= self._map_free_vote_th) & ~_to_occ
                new_grid[_to_occ]  = 220
                new_grid[_to_free] = 50

        # ★ plan_grid保存点：Layer2输出，B3施加之前 → A*/frontier/房间检测用
        _pre_b3_grid = new_grid.copy()

        # ── B3: 空间去噪（形态学 + 小连通域去除）──────────────────────
        if self._map_spatial_enabled:
            from scipy.ndimage import label as _sc_label
            import cv2 as _cv2_sp
            _occ_mask_sp = (new_grid >= self.prob_occ_th).astype(np.uint8)
            # 1. 闭运算：填补墙缝
            _close_c = max(1, int(self._map_occ_close_m / self.map_res))
            _kel_cl  = _cv2_sp.getStructuringElement(
                _cv2_sp.MORPH_RECT, (2 * _close_c + 1, 2 * _close_c + 1))
            _occ_mask_sp = _cv2_sp.morphologyEx(_occ_mask_sp, _cv2_sp.MORPH_CLOSE, _kel_cl)
            # 2. 开运算：去毛刺噪点
            _open_c  = max(1, int(self._map_occ_open_m  / self.map_res))
            _kel_op  = _cv2_sp.getStructuringElement(
                _cv2_sp.MORPH_RECT, (2 * _open_c  + 1, 2 * _open_c  + 1))
            _occ_mask_sp = _cv2_sp.morphologyEx(_occ_mask_sp, _cv2_sp.MORPH_OPEN, _kel_op)
            # 3. 删除小连通域孤立噪点
            if self._map_occ_min_comp_cells > 1:
                _lbl, _cnt = _sc_label(_occ_mask_sp)
                for _li in range(1, _cnt + 1):
                    if int(np.sum(_lbl == _li)) < self._map_occ_min_comp_cells:
                        _occ_mask_sp[_lbl == _li] = 0
            # 写回：原占据格被开运算去除的→恢复 unknown，闭运算补的→ occupied
            # 守护：闭运算不能将 unknown 区域（SLAM未覆盖）填为 occupied
            # 只允许在 SLAM 已知区域（free 或 occ）内填补墙缝
            _known_before_sp = (new_grid != 128).astype(np.uint8)
            _occ_mask_sp = (_occ_mask_sp & _known_before_sp)
            _was_occ = (new_grid >= self.prob_occ_th)
            new_grid[_was_occ & (_occ_mask_sp == 0)] = 128  # 伪障碍 → unknown
            new_grid[_occ_mask_sp == 1]              = 220  # 墙缝补全 → occupied（仅限已知区域）
            # ★[B1] 过切保护：B3 把 occupied 削减超过 65% → 撤销写回，还原 pre-B3 网格
            _occ_b3_before = int(np.sum(_was_occ))
            _occ_b3_after  = int(np.sum(new_grid >= self.prob_occ_th))
            if _occ_b3_before > 10 and _occ_b3_after < _occ_b3_before * 0.35:
                new_grid = _pre_b3_grid.copy()   # 撤销 B3 写回
                import time as _b3sp_t
                if _b3sp_t.monotonic() - getattr(self, '_b3_skip_log_t', 0.0) > 2.0:
                    self._b3_skip_log_t = _b3sp_t.monotonic()
                    self.get_logger().warn(
                        f"[MAP-SPATIAL-SKIP] B3 over_prune: "
                        f"occ_before={_occ_b3_before} occ_after={_occ_b3_after} "
                        f"({_occ_b3_after/max(1,_occ_b3_before)*100:.0f}%) < 35%% "
                        f"→ 撤销B3写回，保留pre-B3网格"
                    )

        # ── MAP-STABLE diagnostic + MAP-ROLLBACK protection ────────────────────
        # [MAP-STABLE] 1Hz: Layer2+B3 输出统计（时域+空间滤波后）
        import time as _ms_t
        _ms_now = _ms_t.monotonic()
        if _ms_now - getattr(self, '_map_stable_log_t', 0.0) > 1.0:
            self._map_stable_log_t = _ms_now
            _free_stable = int(np.sum(new_grid <= self.prob_free_th))
            _occ_stable  = int(np.sum(new_grid >= self.prob_occ_th))
            _unk_stable  = int(np.sum(new_grid == 128))
            self.get_logger().debug(
                f"[MAP-STABLE] free={_free_stable} occ={_occ_stable} unk={_unk_stable} "
                f"raw_occ_before_filter={_occ_before}"
            )
        # [MAP-ROLLBACK]: 后处理把 occupied 清空为0，但 SLAM 原始输入有 occ → 回退 Layer1
        # 防止时域/空间滤波过激进导致地图"零墙体"覆盖正常 self.grid
        _occ_after_stable = int(np.sum(new_grid >= self.prob_occ_th))
        # ★[B2] MAP-ROLLBACK 需连续 3 帧 stable_occ==0 才触发，防止单帧误回滚
        if _occ_after_stable == 0 and _occ_before > 0 and self._map_cb_cnt > 5:
            self._rollback_consec = getattr(self, '_rollback_consec', 0) + 1
            if self._rollback_consec >= 3:
                new_grid = self._raw_slam_grid.copy()
                _pre_b3_grid = new_grid.copy()  # ★ MAP-ROLLBACK同时必须同步 plan_grid
                _occ_rb = int(np.sum(new_grid >= self.prob_occ_th))
                self._rollback_consec = 0
                self.get_logger().warn(
                    f"[MAP-ROLLBACK] reason=occ_cleared_by_filter "
                    f"raw_occ={_occ_before} stable_occ=0 consec=3 "
                    f"rollback_to=Layer1 occ={_occ_rb}"
                )
            else:
                import time as _rb_t
                if _rb_t.monotonic() - getattr(self, '_rollback_warn_t', 0.0) > 0.5:
                    self._rollback_warn_t = _rb_t.monotonic()
                    self.get_logger().warn(
                        f"[MAP-ROLLBACK-PENDING] consec={self._rollback_consec}/3 "
                        f"raw_occ={_occ_before} stable_occ=0 (等待连续3帧确认)"
                    )
        else:
            self._rollback_consec = 0  # 恢复正常帧则重置计数器

        # ── Phase7: occ-delta 地图质量诊断（障碍物增删变化量）────────────────
        # 量化每帧障碍物变化，用于检测 SLAM 漂移/翻转问题
        _prev_occ_mask = (self.grid >= self.prob_occ_th)
        _new_occ_mask  = (new_grid  >= self.prob_occ_th)
        if _prev_occ_mask.shape == _new_occ_mask.shape:
            _occ_added     = int(np.sum(_new_occ_mask & ~_prev_occ_mask))
            _occ_removed   = int(np.sum(_prev_occ_mask & ~_new_occ_mask))
            _occ_delta_pct = ((_occ_added + _occ_removed) /
                              max(1, int(np.sum(_prev_occ_mask | _new_occ_mask)))) * 100.0
            # 高变化率预警（≥25% 表示地图严重不稳定，可能是 SLAM 重定位或坐标漂移）
            # 阈值 Phase9 从 15% 上调为 25%，减少飞行初期的误报噪声
            # 抑制条件：
            #   ① 早期建图（_map_cb_cnt ≤ 30）: 从0开始建图，变化率天然极高（100%），是正常现象
            #   ② 障碍格过少（prev_occ ≤ 100）: 样本量不足，变化率无参考意义
            # 额外诊断：EKF reset 暂停期间不打印（已有 EKF-RESET 日志覆盖）
            _in_ekf_pause = (_mt.monotonic() < self._map_pause_until)
            if (_occ_delta_pct >= 25.0
                    and int(np.sum(_prev_occ_mask)) > 100  # 需要足够障碍格
                    and self._map_cb_cnt > 30              # 跳过初期建图阶段
                    and not _in_ekf_pause):                # EKF暂停期不重复告警
                # 地图不稳定计数器── MAP_RECOVER 退出用
                self._occ_stable_frames = 0
                self.get_logger().warn(
                    f"[MAP-UNSTABLE] 地图障碍变化率过高: "
                    f"新增={_occ_added} 消除={_occ_removed} "
                    f"变化率={_occ_delta_pct:.1f}% "
                    f"可能为 SLAM 重定位或角度漂移"
                )
            else:
                # 占据稳定：为 MAP_RECOVER 退出申请计分
                if getattr(self, '_map_recover_active', False):
                    self._occ_stable_frames = getattr(self, '_occ_stable_frames', 0) + 1
        else:
            _occ_added = _occ_removed = 0
            _occ_delta_pct = -1.0   # 形状变化帧（地图尺寸扩展），跳过计算

        # ★ 双图层赋值：
        #   plan_grid = Layer2稳定层（B3前）——A*/frontier/房间门检测用，保留完整墙体
        #   vis_grid  = B3滤波后（或MAP-ROLLBACK）——斠用于可视化
        self.plan_grid = _pre_b3_grid           # 规划图层：将B3影响隔离
        self.vis_grid  = new_grid.copy()         # 显示图层
        self.grid      = self.plan_grid          # ★ self.grid 指向规划层，防止 B3 删墙影响 A*
        # ★ [MAP-LAYER] 评估B3对墙体的删减量（每2s）
        import time as _ml_t
        if _ml_t.monotonic() - getattr(self, '_map_layer_log_t', 0.0) > 2.0:
            self._map_layer_log_t = _ml_t.monotonic()
            _plan_occ2  = int(np.sum(self.plan_grid >= self.prob_occ_th))
            _plan_free2 = int(np.sum(self.plan_grid <= self.prob_free_th))
            _vis_occ2   = int(np.sum(self.vis_grid  >= self.prob_occ_th))
            _vis_free2  = int(np.sum(self.vis_grid  <= self.prob_free_th))
            if abs(_plan_occ2 - _vis_occ2) > 50 or _plan_occ2 > _vis_occ2 + 50:
                # 墙体有显著差异时打warn，正常时打debug
                self.get_logger().warn(
                    f"[MAP-LAYER] plan_occ={_plan_occ2} plan_free={_plan_free2} "
                    f"vis_occ={_vis_occ2} vis_free={_vis_free2} "
                    f"B3删墙={_plan_occ2-_vis_occ2}格"
                )
            else:
                self.get_logger().debug(
                    f"[MAP-LAYER] plan_occ={_plan_occ2} vis_occ={_vis_occ2} "
                    f"plan_free={_plan_free2} vis_free={_vis_free2}"
                )
        # ★ 更新地图质量统计（每帧，供规划器/APF/可视化查询）
        self._slam_occ_cell_count = int(np.sum(self.grid >= self.prob_occ_th))
        self._map_quality_ok = (self._slam_occ_cell_count >= self._min_occ_for_planning)
        if not self._map_quality_ok and getattr(self, '_map_cb_cnt', 0) % 20 == 1:
            self.get_logger().warn(
                f"[MAP-QUALITY] MAP_THIN: plan_grid occ={self._slam_occ_cell_count} "
                f"< min={self._min_occ_for_planning} "
                f"→ 地图墙体不足，建议悬停重扫/等待 SLAM 建立"
            )
        mode_str = (f"固定规划坐标系重采样 "
                    f"bounds=({sx_p:.1f},{sy_p:.1f})"
                    f"→({sx_p+ni_p*res_p:.1f},{sy_p+nj_p*res_p:.1f})"
                    f" res={res_p:.3f}m grid={ni_p}×{nj_p}")

        self._slam_map_received = True
        self._map_watchdog_warned = False

        self._map_cb_cnt += 1

        # ── Phase-6：/map 发布频率追踪 ──────────────────────────────────────
        import time as _hz_mt
        _now_mono = _hz_mt.monotonic()
        if self._map_last_cb_mono > 0.0:
            _interval = _now_mono - self._map_last_cb_mono
            self._map_cb_intervals.append(_interval)
            if len(self._map_cb_intervals) > 20:
                self._map_cb_intervals.pop(0)
        self._map_last_cb_mono = _now_mono
        _map_avg_hz = (
            1.0 / max(1e-9, (sum(self._map_cb_intervals) / len(self._map_cb_intervals)))
            if self._map_cb_intervals else 0.0
        )

        # 每10次打印一次诊断（首次必打）
        if self._map_cb_cnt <= 3 or self._map_cb_cnt % 10 == 0:
            free_c = int(np.sum(self.grid < self.prob_free_th))
            occ_c  = int(np.sum(self.grid > self.prob_occ_th))
            unk_c  = self.grid_shape[0] * self.grid_shape[1] - free_c - occ_c
            _occ_after = int(np.sum(self.grid >= self.prob_occ_th))
            _tf_str = (
                f"layer2={'lo-ema' if self._stable_lo_enabled else 'vote'}"
                + (f"(α={self._stable_lo_alpha:.2f} obs={self._stable_lo_obs:.1f})" if self._stable_lo_enabled else
                   f"(win={len(self._grid_hist)}/{self._map_temporal_window})")
                + f" spatial={'on' if self._map_spatial_enabled else 'off'} "
                f"occ={_occ_before}->{_occ_after}"
            )
            self.get_logger().info(
                f"[SLAM-MAP] #{self._map_cb_cnt} 模式={mode_str} "
                f"slam={W}×{H} res={sres:.3f}m origin=({ox:.2f},{oy:.2f}) "
                f"map_hz={_map_avg_hz:.1f}Hz "
                f"slam值分布: free={slam_free_n} occ={slam_occ_n} unk={slam_unk_n} "
                f"valid_count={valid_count}/{total_cells} "
                f"内部grid: free={free_c} occ={occ_c} unk={unk_c} | {_tf_str} | "
                f"Δocc=+{_occ_added}/-{_occ_removed}({_occ_delta_pct:.1f}%)"
            )
            if _map_avg_hz > 0 and _map_avg_hz < 0.8:
                self.get_logger().warn(
                    f"[SLAM-MAP] ⚠ /map 更新频率过低 ({_map_avg_hz:.2f}Hz < 0.8Hz)！"
                    f"可能导致墙体建立缓慢/断续。"
                    f"建议：slam_toolbox map_update_interval ≤ 0.5s"
                )
            if slam_known == 0:
                self.get_logger().warn(
                    "[SLAM-MAP] slam地图中无任何已知格（全-1）！"
                    "可能原因：\n"
                    "  - /scan 无数据（bridge未启动或话题名错误）\n"
                    "  - odom→base_link TF 缺失导致 slam_toolbox 拒绝处理 scan\n"
                    "  - use_sim_time 但 /clock 延迟导致时间戳过滤\n"
                    "验证: ros2 topic hz /scan  (应>10Hz)"
                )

    # ===================================================================
    # 地图可视化发布（全局 + 局部跟随视图）
    # ===================================================================
    def _publish_map_vis(self) -> None:
        """
        发布地图可视化图像到 /indoor_search/map_vis。

        图层顺序（从底到顶，高层不能覆盖低层原始信息）：
          1. SLAM 底图层：raw occupancy cells（白=free，黑=occ，灰=unknown）
          2. 激光调试层：当前 scan 端点投影（黄色小点）
          3. 分割结果层：房间区域半透明着色（35%不透明度，保留底图可见）
          4. 规划层：A*路径（蓝线）+ 候选视点目标（绿圆）
          5. 无人机层：红点+朝向箭头
          6. 文字信息层：地图元数据 + 状态

        关键设计：
          - 房间着色 = alpha 混合（35%），SLAM 底图边界始终可见
          - 激光端点显示在 SLAM 底图上方（调试 TF/时间戳问题利器）
          - 全局/局部 可切换，局部以无人机为中心裁剪
        """
        if self._map_vis_pub is None:
            return
        if not getattr(self, '_slam_map_received', False):
            return
        try:
            import time as _mt
            W, H = self.grid_shape  # (x格数, y格数)
            if W == 0 or H == 0:
                return

            grid = self.grid  # (W, H) uint8: 50=free, 220=occ, 128=unknown

            # ── 层1：SLAM 底图（原始 occupancy，不做任何后处理覆盖）─────────
            # ★ Y轴修复：grid[ix, iy]中iy=0=South、iy=H-1=North；
            # 图像row=0=顶部，row=H-1=底部；正北朝上规范 → North应在顶部
            # 因此必须flipud：image[row,col]=grid[col, H-1-row]（North在顶部）
            # 修复前：gt=grid.T（South在顶部），无人机用H-1-iy（North在顶部）→ 镜像错位！
            # 修复后：两者都是North在顶部 → 无人机与房间正确对齐
            gt = np.flipud(grid.T).astype(np.int32)   # (H, W), North at top ★
            slam_layer = np.zeros((H, W, 3), dtype=np.uint8)
            slam_layer[(gt > self.prob_free_th) & (gt < self.prob_occ_th)] = (150, 150, 150)  # unknown=中灰
            slam_layer[gt <= self.prob_free_th]  = (245, 245, 245)   # free=近白
            slam_layer[gt >= self.prob_occ_th]   = (20,  20,  20)    # occ=近黑
            disp = slam_layer.copy()

            # ── 层2：激光端点调试层（黄色点，投影到 plan frame）──────────────
            # 利用当前 last_scan + 无人机 plan 位姿手动投影（_plan_px/_plan_py/_plan_yaw）
            # fix: 使用 plan frame 坐标与 self.grid 对齐（_map_px/_map_py 仅在 log 中保留）
            scan_l = self.last_scan
            drone_px_val = getattr(self, '_plan_px', getattr(self, '_map_px', None))
            drone_py_val = getattr(self, '_plan_py', getattr(self, '_map_py', None))
            drone_yaw_val = getattr(self, '_plan_yaw', getattr(self, '_map_yaw', 0.0))
            if (scan_l is not None and drone_px_val is not None
                    and drone_py_val is not None):
                ranges_s = np.array(scan_l.ranges, dtype=float)
                angles_s = (scan_l.angle_min
                             + np.arange(len(ranges_s), dtype=float)
                             * scan_l.angle_increment)
                # 注意：scan 在 laser_link 系，laser_link = base_link + Rz(mount_yaw)
                # _lidar_raw_to_body_angles 已加 mount_yaw → body 系角度
                angles_body = self._lidar_raw_to_body_angles(angles_s)
                valid_s = (np.isfinite(ranges_s)
                           & (ranges_s > 0.05)
                           & (ranges_s < (scan_l.range_max if scan_l.range_max > 0 else 1e6)))
                r_v = ranges_s[valid_s]
                a_v = angles_body[valid_s]
                yaw_map = float(drone_yaw_val)
                # body 角（_lidar_raw_to_body_angles 已加 mount_yaw）采用 **顺时针正** 约定；
                # ENU yaw 为**逆时针正**，故世界方向 = yaw_enu - body_angle_CW
                ex_m = float(drone_px_val) + r_v * np.cos(yaw_map - a_v)
                ey_m = float(drone_py_val) + r_v * np.sin(yaw_map - a_v)
                for ex, ey in zip(ex_m, ey_m):
                    p_col, p_row = self.world_to_grid(float(ex), float(ey))
                    p_col = int(np.clip(p_col, 0, W - 1))
                    p_row = int(np.clip(H - 1 - p_row, 0, H - 1))
                    # 只在 unknown/free 区域画端点（在 occ 上画没意义）
                    if gt[p_row, p_col] < self.prob_occ_th:
                        disp[p_row, p_col] = (0, 210, 255)   # 黄色（BGR: 0,210,255）

            # ── 层3：房间着色（35% 不透明度半透明混合）──────────────────────
            # 不替换原始 disp，而是 alpha 混合：保留 SLAM 底图和激光端点可见
            _room_palette = [
                (80, 160, 255), (80, 255, 160), (200, 80, 255),
                (255, 220, 60), (60, 200, 255), (255, 80, 180),
                (100, 255, 100), (255, 100, 100), (100, 100, 255),
            ]
            rl = getattr(self, 'room_labels', None)
            if rl is not None and rl.shape == (W, H):
                room_overlay = disp.copy()
                for r_info in getattr(self, 'rooms_info', []):
                    rid = r_info.get("label", 0)
                    col = np.array(_room_palette[(rid - 1) % len(_room_palette)], dtype=np.uint8)
                    # ★ 房间标签也需flipud，与背景图保持North在顶部的一致性
                    mask_r = (np.flipud(rl.T) == rid)   # (H, W) bool, North at top ★
                    room_overlay[mask_r] = col
                # 35% 着色，65% 保留原始 SLAM 底图
                disp = cv2.addWeighted(disp, 0.65, room_overlay, 0.35, 0)

            # ── 层4：门（青色实心圆 + 黑色轮廓）───────────────────────────
            for d in getattr(self, 'doors_info', []):
                dix = int(np.clip(d.get("ix", 0), 0, W - 1))
                diy = int(np.clip(d.get("iy", 0), 0, H - 1))
                px_c = dix
                px_r = int(np.clip(H - 1 - diy, 0, H - 1))
                cv2.circle(disp, (px_c, px_r), 5, (0, 0, 0), -1)
                cv2.circle(disp, (px_c, px_r), 4, (0, 230, 230), -1)

            # ── 层4b：A*路径（橙线）────────────────────────────────────────
            if self.astar_path is not None and len(self.astar_path) > 1:
                for pi in range(len(self.astar_path) - 1):
                    ax, ay = self.world_to_grid(
                        float(self.astar_path[pi][0]), float(self.astar_path[pi][1]))
                    bx, by = self.world_to_grid(
                        float(self.astar_path[pi + 1][0]), float(self.astar_path[pi + 1][1]))
                    ax = int(np.clip(ax, 0, W - 1))
                    ay = int(np.clip(H - 1 - ay, 0, H - 1))
                    bx = int(np.clip(bx, 0, W - 1))
                    by = int(np.clip(H - 1 - by, 0, H - 1))
                    cv2.line(disp, (ax, ay), (bx, by), (0, 100, 240), 2)

            # ── 层4c：SSG候选视点目标（绿色实心圆）────────────────────────────
            vp_goal = getattr(self, 'ssg_viewpoint_goal', None)
            if vp_goal is not None:
                gx, gy = self.world_to_grid(float(vp_goal[0]), float(vp_goal[1]))
                gx = int(np.clip(gx, 0, W - 1))
                gy = int(np.clip(H - 1 - gy, 0, H - 1))
                cv2.circle(disp, (gx, gy), 6, (0, 0, 0), -1)
                cv2.circle(disp, (gx, gy), 5, (50, 220, 50), -1)

            # ── 层4d：★[G5] 目标终端语义点可视化 ─────────────────────────────
            # 图例：天蓝圈=visual world_est  青绿实心圆=P1接近点  紫色实心圆=P2提交点
            _vis_w_est = getattr(self, '_tgt_world_est_xy', None)
            _vis_w_conf = float(getattr(self, '_tgt_world_est_conf', 0.0))
            if _vis_w_est is not None and _vis_w_conf > 0.15:
                _wx, _wy = self.world_to_grid(float(_vis_w_est[0]), float(_vis_w_est[1]))
                _wx = int(np.clip(_wx, 0, W - 1))
                _wy = int(np.clip(H - 1 - _wy, 0, H - 1))
                _r_est = max(6, int(14 * _vis_w_conf))
                cv2.circle(disp, (_wx, _wy), _r_est + 3, (0, 0, 0), 1)
                cv2.circle(disp, (_wx, _wy), _r_est, (0, 210, 255), 1)
                cv2.putText(disp, f"W{_vis_w_conf:.2f}",
                            (_wx + _r_est + 2, _wy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 150, 200), 1)
            _vis_cp = getattr(self, '_tgt_commit_goal', None)
            if _vis_cp is not None:
                _cppx, _cppy = self.world_to_grid(float(_vis_cp[0]), float(_vis_cp[1]))
                _cppx = int(np.clip(_cppx, 0, W - 1))
                _cppy = int(np.clip(H - 1 - _cppy, 0, H - 1))
                cv2.circle(disp, (_cppx, _cppy), 8, (0, 0, 0), -1)
                cv2.circle(disp, (_cppx, _cppy), 7, (0, 165, 255), -1)  # 橙色=CP
                cv2.putText(disp, "CP", (_cppx + 9, _cppy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 140, 220), 1)

            # ── 层5：无人机位置（四层色彩区分）───────────────────────────────
            #   tf_map_base(直接TF可信) = 红圆+箭头
            #   tf_map_odom_plus_odom(fallback推算) = 琥珀圆+FBK文字
            #   stale(缓存) = 灰圆+STALE文字
            #   planar_invalid(姿态失效,tilt>20°) = 红叉+TILT文字
            drone_col: Optional[int] = None
            drone_row: Optional[int] = None
            _mp_valid_pv  = getattr(self, '_map_pose_valid', False)
            _mp_stale_pv  = getattr(self, '_map_pose_stale', True)
            _mp_src_pv2   = getattr(self, '_map_pose_source', '?')
            _planar_ok_pv = getattr(self, '_planar_pose_valid', True)
            if drone_px_val is not None and drone_py_val is not None and _mp_valid_pv:
                dix_g, diy_g = self.world_to_grid(float(drone_px_val), float(drone_py_val))
                drone_col = int(np.clip(dix_g, 0, W - 1))
                drone_row = int(np.clip(H - 1 - diy_g, 0, H - 1))
                if not _planar_ok_pv:
                    # tilt > 20°：红叉标记
                    cv2.line(disp, (drone_col - 8, drone_row - 8),
                             (drone_col + 8, drone_row + 8), (0, 0, 210), 3)
                    cv2.line(disp, (drone_col + 8, drone_row - 8),
                             (drone_col - 8, drone_row + 8), (0, 0, 210), 3)
                    cv2.putText(disp, "TILT", (drone_col + 9, drone_row - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
                elif _mp_stale_pv:
                    # stale：灰色圆+STALE文字
                    cv2.circle(disp, (drone_col, drone_row), 8, (180, 180, 180), -1)
                    cv2.putText(disp, "STALE", (drone_col + 9, drone_row - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
                elif _mp_src_pv2 != 'tf_map_base':
                    # fallback推算：琥珀圆+FBK文字
                    cv2.circle(disp, (drone_col, drone_row), 8, (0, 0, 0), -1)
                    cv2.circle(disp, (drone_col, drone_row), 7, (0, 140, 210), -1)
                    cv2.putText(disp, "FBK", (drone_col + 9, drone_row - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 180), 1)
                else:
                    # fresh tf_map_base：红圆+箭头
                    cv2.circle(disp, (drone_col, drone_row), 8, (0, 0, 0), -1)
                    cv2.circle(disp, (drone_col, drone_row), 7, (0, 0, 255), -1)
                    yaw_v = float(drone_yaw_val)
                    arr_len = max(8, int(1.0 / self.map_res))
                    ex_a = int(np.clip(drone_col + arr_len * np.cos(yaw_v), 0, W - 1))
                    ey_a = int(np.clip(drone_row - arr_len * np.sin(yaw_v), 0, H - 1))
                    cv2.arrowedLine(disp, (drone_col, drone_row), (ex_a, ey_a),
                                    (0, 0, 255), 2, tipLength=0.4)

                # ── 层5b：map-tf 原始坐标对比点（空心小圆，白色）─────────────
                # 与层5主标记（plan frame 实心圆）对比，验证 frame 对齐。
                # 正常情况（slam origin 旋转≈0）：两圆重叠，说明 frame 一致。
                # 若两圆分离：说明 OccupancyGrid origin.orientation != identity，
                #   map frame 和 plan frame 之间有旋转差（SE(2) 旋转补偿已修正）。
                _raw_map_px = float(getattr(self, '_map_px', drone_px_val or 0.0))
                _raw_map_py = float(getattr(self, '_map_py', drone_py_val or 0.0))
                _raw_col, _raw_row = self.world_to_grid(_raw_map_px, _raw_map_py)
                _raw_col = int(np.clip(_raw_col, 0, W - 1))
                _raw_row = int(np.clip(H - 1 - _raw_row, 0, H - 1))
                cv2.circle(disp, (_raw_col, _raw_row), 12, (255, 255, 255), 1)  # 白色空心圆=map-tf
                _slam_yaw_deg = abs(float(getattr(self, '_slam_origin_yaw', 0.0)) * 57.3)
                if _slam_yaw_deg > 3.0:
                    cv2.putText(disp, f"MAP!={_slam_yaw_deg:.0f}deg",
                                (_raw_col + 13, _raw_row - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)

            # ── 层5.5：display_fixed_bounds 裁剪（固定世界坐标窗口）────────────
            # ★ 修复：使用固定规划参数（_plan_*）而非 SLAM 动态 origin
            # 因为 disp 已按固定规划栅格构建（格i=world_x=sx_p+i*res_p），
            # 裁剪时须用相同的坐标系，否则裁剪窗口会随 SLAM origin 漂移。
            if getattr(self, '_display_fixed_bounds', True):
                _ox  = float(self._plan_bounds_xy[0, 0])   # 固定规划 xmin
                _oy  = float(self._plan_bounds_xy[1, 0])   # 固定规划 ymin
                _sr  = self._plan_res                       # 固定规划分辨率
                def _w2c(wx): return int(np.clip((wx - _ox) / _sr, 0, W - 1))
                def _w2r(wy): return int(np.clip(H - 1 - (wy - _oy) / _sr, 0, H - 1))
                c0_d = _w2c(self._display_xmin)
                c1_d = _w2c(self._display_xmax) + 1
                r1_d = _w2r(self._display_ymin) + 1
                r0_d = _w2r(self._display_ymax)
                c0_d, c1_d = sorted([max(0, c0_d), min(W, c1_d)])
                r0_d, r1_d = sorted([max(0, r0_d), min(H, r1_d)])
                if c1_d > c0_d and r1_d > r0_d:
                    disp = disp[r0_d:r1_d, c0_d:c1_d].copy()
                    W_d = c1_d - c0_d
                    H_d = r1_d - r0_d
                    # 更新 drone_col/drone_row（相对裁剪后的子图）
                    if drone_col is not None:
                        drone_col = int(np.clip(drone_col - c0_d, 0, W_d - 1))
                        drone_row = int(np.clip(drone_row - r0_d, 0, H_d - 1))

            # ── 层6：裁剪（局部跟随 / 全局）────────────────────────────────
            vis_img = disp
            if (self.vis_mode == "local"
                    and drone_col is not None and drone_row is not None):
                half_c = max(10, int(self.vis_local_range_m / (2.0 * self.map_res)))
                r0 = max(0, drone_row - half_c)
                r1 = min(H, drone_row + half_c)
                c0 = max(0, drone_col - half_c)
                c1 = min(W, drone_col + half_c)
                crop = disp[r0:r1, c0:c1]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    vis_img = crop.copy()

            # ── 缩放到 vis_max_dim ──────────────────────────────────────────
            vh, vw = vis_img.shape[:2]
            if max(vh, vw) > self.vis_max_dim:
                sc = self.vis_max_dim / max(vh, vw)
                vis_img = cv2.resize(
                    vis_img, (int(vw * sc), int(vh * sc)),
                    interpolation=cv2.INTER_NEAREST)

            # ── 层7a：任务7 目标接近可视化层 ──────────────────────────────
            _approach_ex = getattr(self, '_approach_exclusive', False)
            _appr_inner  = getattr(self, '_approach_inner_state', 'approach')
            _tgt_conf    = getattr(self, 'target_confirmed', False)
            _tgt_found   = getattr(self, 'target_found', False)
            _tgt_area    = float(getattr(self, 'target_area_ratio', 0.0))
            _tgt_dx      = float(getattr(self, 'target_offset_x', 0.0))
            _tgt_dy      = float(getattr(self, 'target_offset_y', 0.0))
            _tgt_seen_ago = 99.0
            if getattr(self, 'target_last_seen_time', None) is not None:
                _tgt_seen_ago = (self.get_clock().now() - self.target_last_seen_time).nanoseconds * 1e-9
            # 在无人机位置周围画目标接近状态指示圈
            if drone_col is not None and drone_row is not None:
                # 状态颜色编码
                if _approach_ex:
                    if _appr_inner == 'descend':
                        _ring_col = (0, 0, 255)    # 红圆 = 下降
                    elif _appr_inner == 'hover_confirm':
                        _ring_col = (0, 180, 255)  # 橙圆 = 悬停确认
                    elif _appr_inner == 'locked_far':
                        _ring_col = (0, 200, 255)  # 黄色 = 锁定但距离远
                    elif _appr_inner in ('lost_short', 'lost_sweep'):
                        _ring_col = (255, 140, 0)  # 蓝圆 = 丢失中
                    elif _appr_inner == 'forward':
                        _ring_col = (0, 200, 50)   # 翠绿 = 前进逼近
                    else:
                        _ring_col = (80, 255, 80)  # 浅绿 = 对准 yaw
                    cv2.circle(disp, (drone_col, drone_row), 16, _ring_col, 2)
                    # 目标伺服方向箭头
                    _vis_vw = vis_img.shape[1]
                    _servo_yaw = getattr(self, '_servo_yawspeed', 0.0)
                    _arr_dx = int(np.clip(-_tgt_dx * 20, -15, 15))
                    _arr_dy = int(np.clip(-_tgt_dy * 20, -15, 15))
                    cv2.arrowedLine(disp,
                        (drone_col, drone_row),
                        (int(np.clip(drone_col + _arr_dx, 0, W-1)),
                         int(np.clip(drone_row + _arr_dy, 0, H-1))),
                        (255, 255, 0), 2, tipLength=0.4)
                elif _tgt_found:
                    # 发现但未独占：黄色虚圆
                    cv2.circle(disp, (drone_col, drone_row), 16, (0, 200, 255), 1)
            # 目标状态文字（叠加在图像左上角状态栏之后）
            if _approach_ex or _tgt_found:
                _state_labels = {
                    'align':         'ALIGN(yaw)',
                    'approach':      'ALIGN(yaw)',    # 向后兼容
                    'forward':       'FORWARD',
                    'hover_confirm': 'HOVER-CONFIRM',
                    'descend':       'DESCEND',
                    'land':          'LAND',
                    'lost_short':    'LOST(hold)',
                    'lost_sweep':    'LOST(scan)',
                    'locked_far':    'LOCKED(far)',
                    'lost_nav':      'LOST(nav)',
                    # ★[G6] 单提交点新状态
                    'world_establish':      'WORLD-EST',
                    'commit_nav':           'COMMIT-NAV',
                    'commit_hover':         'COMMIT-HOVER',
                    'conservative_descend': 'CONSERV-DESC',
                }
                _st_txt = _state_labels.get(_appr_inner, _appr_inner.upper()) if _approach_ex else (
                    'CONFIRMED' if _tgt_conf else 'FOUND'
                )
                _st_col = {
                    'align':         (80, 255, 80),
                    'approach':      (80, 255, 80),
                    'forward':       (0, 200, 50),
                    'hover_confirm': (0, 180, 255),
                    'descend':       (0, 0, 255),
                    'land':          (0, 0, 200),
                    'lost_short':    (0, 140, 255),
                    'lost_sweep':    (0, 100, 255),
                    'locked_far':    (0, 200, 255),
                    'lost_nav':      (0, 80, 200),
                    # ★[G6] 单提交点新状态颜色
                    'world_establish':      (0, 200, 255),  # 天蓝 = 世界坐标建立
                    'commit_nav':           (0, 255, 180),  # 青绿 = 提交点导航
                    'commit_hover':         (50, 200, 140), # 深青 = 提交点悬停
                    'conservative_descend': (0, 40, 255),   # 深蓝 = 保守下降
                }.get(_appr_inner, (200, 200, 0))
                _tgt_overlay_lines = [
                    f"TARGET:{_st_txt}",
                    f"area={_tgt_area:.3f} dx={_tgt_dx:.2f} dy={_tgt_dy:.2f}",
                    f"seen={_tgt_seen_ago:.1f}s",
                ]
                for _tl_i, _tl in enumerate(_tgt_overlay_lines):
                    _tl_y = vis_img.shape[0] - 30 + _tl_i * 11
                    cv2.putText(vis_img, _tl, (3, _tl_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2)
                    cv2.putText(vis_img, _tl, (3, _tl_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, _st_col, 1)

            # ── 层7：文字叠加（状态信息）────────────────────────────────────
            # ★ 地图过期告警：若 /map 超过 5s 未更新则显示红色警告
            map_age_sec = float('inf')
            if getattr(self, '_last_map_stamp', None) is not None:
                import time as _mt2
                map_age_sec = _mt2.monotonic() - float(self._last_map_stamp)

            free_c = int(np.sum(self.grid <= self.prob_free_th))
            occ_c  = int(np.sum(self.grid >= self.prob_occ_th))
            total_c = self.grid_shape[0] * self.grid_shape[1]
            expl_pct = 100.0 * free_c / max(1, total_c)

            map_stale_warn = ""
            if map_age_sec > 5.0:
                map_stale_warn = f"  !MAP_STALE {map_age_sec:.0f}s!"

            _mp_src_pv = getattr(self, '_map_pose_source', '?')
            _tilt_vis_pv = getattr(self, '_att_tilt_local', 0.0)
            _planar_ok_line = (
                f"tilt={_tilt_vis_pv:.0f}deg OK" if getattr(self, '_planar_pose_valid', True)
                else f"PLANAR_INVALID tilt={_tilt_vis_pv:.0f}deg"
            )
            _odom_enu_str = (
                f"bridge-odom/ENU east={self._odom_x_enu:.2f} north={self._odom_y_enu:.2f}"
                if self._odom_received else "odom=N/A"
            )
            # ★ plan vs map delta（用于验证frame统一是否真的生效）
            _plan_px_vis = float(getattr(self, '_plan_px', 0.0))
            _plan_py_vis = float(getattr(self, '_plan_py', 0.0))
            _map_px_vis  = float(getattr(self, '_map_px', 0.0))
            _map_py_vis  = float(getattr(self, '_map_py', 0.0))
            _plan_map_delta = float(np.hypot(_plan_px_vis - _map_px_vis,
                                             _plan_py_vis - _map_py_vis))
            _slam_rot = float(getattr(self, '_slam_origin_yaw', 0.0)) * 57.3
            _coord_str = (
                f"plan=({_plan_px_vis:.2f},{_plan_py_vis:.2f}) "
                f"map=({_map_px_vis:.2f},{_map_py_vis:.2f}) "
                f"Δ={_plan_map_delta:.3f}m slam_rot={_slam_rot:.1f}deg"
            )
            # ★ 地图质量指示：occ格数太少 → MAP_THIN告警
            _occ_cells = int(np.sum(self.grid >= self.prob_occ_th))
            _map_thin = _occ_cells < getattr(self, '_min_occ_for_planning', 5)
            _map_quality_str = (f"MAP_THIN(occ={_occ_cells})" if _map_thin
                                else f"occ={_occ_cells}")
            info_lines = [
                (f"[vis:{self.vis_mode}] res={self.map_res:.3f}m "
                 f"map={W}x{H} src=slam_toolbox{map_stale_warn}",
                 (0, 50, 220) if not map_stale_warn else (0, 0, 255)),
                (f"free={expl_pct:.1f}% {_map_quality_str} unk={total_c-free_c-occ_c} "
                 f"orig=({self.bounds[0,0]:.2f},{self.bounds[1,0]:.2f})",
                 (200, 60, 0) if _map_thin else (20, 130, 20)),
                (f"rooms={len(getattr(self,'rooms_info',[]))} "
                 f"doors={len(getattr(self,'doors_info',[]))} "
                 f"state={getattr(self,'explore_state','?')} "
                 f"map={'stale' if _mp_stale_pv else 'fresh'}/{_mp_src_pv} "
                 f"{_planar_ok_line}",
                 (20, 100, 20) if getattr(self, '_planar_pose_valid', True)
                 else (0, 0, 200)),
                (_odom_enu_str, (0, 80, 160)),
                (_coord_str,
                 (0, 0, 200) if _plan_map_delta > 0.1 else (0, 130, 0)),
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            for li, (txt, col_fg) in enumerate(info_lines):
                y_pos = 13 + li * 13
                cv2.putText(vis_img, txt, (3, y_pos), font, 0.30, (0, 0, 0), 2)
                cv2.putText(vis_img, txt, (3, y_pos), font, 0.30, col_fg, 1)

            # ── 发布 ROS Image ─────────────────────────────────────────────
            now_t = self.get_clock().now()
            img_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
            img_msg.header.stamp = now_t.to_msg()
            img_msg.header.frame_id = "map"
            self._map_vis_pub.publish(img_msg)

        except Exception as _ve:
            self.get_logger().warn(f"[VIS] 可视化错误: {_ve}")

    def pos_cb(self, msg: VehicleLocalPosition) -> None:
        now = self.get_clock().now()

        # 修复2：单调时间戳门控 —— 丢弃旧消息/乱序消息，防止两流交错导致pos_cb看到回跳
        px4_ts = int(getattr(msg, 'timestamp', 0))
        if px4_ts > 0:
            if px4_ts <= self._last_px4_ts:
                return  # 时间戳倒退/重复，直接丢弃
            self._last_px4_ts = px4_ts

        if self.last_pose_arrival_time is not None:
            dt_arrival = (now - self.last_pose_arrival_time).nanoseconds * 1e-9
            if 0.0 < dt_arrival < 5.0:
                if self.pose_dt_ema_sec is None:
                    self.pose_dt_ema_sec = float(dt_arrival)
                else:
                    a = float(np.clip(self.pose_dt_ema_alpha, 0.01, 1.0))
                    self.pose_dt_ema_sec = a * float(dt_arrival) + (1.0 - a) * float(self.pose_dt_ema_sec)
        self.last_pose_arrival_time = now

        # ENU: x=东(East)=NED_y, y=北(North)=NED_x
        raw_x = float(msg.y)   # ENU_x = East = NED_y
        raw_y = float(msg.x)   # ENU_y = North = NED_x
        raw_z = float(msg.z)
        x_meas = raw_x
        y_meas = raw_y
        z_meas = raw_z

        # ★ bounds guard：raw_x/raw_y 已是 ENU (East, North)，self.bounds 亦为 ENU 范围。
        # 注意：SLAM map 接管后 self.bounds 变为固定规划 ENU frame（origin=slam_origin），
        # 与 raw_x/raw_y（ENU odom 坐标）之间可能存在 loop-closure 引起的系统性偏差（数米）。
        # 因此：SLAM map 已收到时跳过此 bounds 门控，改由后续的 XY jump 检测保护。
        # SLAM map 未收到时（启动初期）使用初始 ±8m 固定边界检查。
        _slam_active = getattr(self, '_slam_map_received', False)
        if not _slam_active:
            x_min = float(self.bounds[0, 0] - self.pose_bounds_guard_margin)
            x_max = float(self.bounds[0, 1] + self.pose_bounds_guard_margin)
            y_min = float(self.bounds[1, 0] - self.pose_bounds_guard_margin)
            y_max = float(self.bounds[1, 1] + self.pose_bounds_guard_margin)
            if not (x_min <= x_meas <= x_max and y_min <= y_meas <= y_max):
                if self.last_valid_x is not None and self.last_valid_y is not None:
                    if (now - self.last_coord_warn_time).nanoseconds * 1e-9 > 0.8:
                        self.get_logger().error(
                            f"[COORD ERROR] XY out of bounds (pre-SLAM): x={raw_x:.3f}, y={raw_y:.3f}, "
                            f"reject and keep last=({self.last_valid_x:.3f},{self.last_valid_y:.3f})"
                        )
                        self.last_coord_warn_time = now
                    x_meas = float(self.last_valid_x)
                    y_meas = float(self.last_valid_y)

        if self.last_valid_x is not None and self.last_valid_y is not None and self.last_pose_msg_time is not None:
            dt = (now - self.last_pose_msg_time).nanoseconds * 1e-9
            dxy = float(np.hypot(x_meas - self.last_valid_x, y_meas - self.last_valid_y))
            speed_xy = dxy / max(dt, 1e-3)
            # dt<5ms 表示同一仿真 tick 内收到两帧，速度值无意义（会被 clamp 至1ms导致虚高）
            # 此时仅做距离检查，忽略速度阈值，避免假 COORD ERROR
            speed_check = dt >= 0.005 and speed_xy > self.xy_jump_reject_speed
            if dt <= self.xy_jump_reject_dt and (
                dxy > self.xy_jump_reject_dist or speed_check
            ):
                self.xy_jump_reject_count += 1
                if (now - self.last_coord_warn_time).nanoseconds * 1e-9 > 0.8:
                    self.get_logger().error(
                        f"[COORD ERROR] Sudden XY jump: x={x_meas:.3f}, y={y_meas:.3f}, "
                        f"last=({self.last_valid_x:.3f},{self.last_valid_y:.3f}), "
                        f"dxy={dxy:.3f}m dt={dt:.3f}s v={speed_xy:.2f}m/s, rejecting"
                    )
                    self.last_coord_warn_time = now
                # 修复B+A：持续跳变 → 识别为EKF reset，接受新位姿并清空地图防止叠图
                recover_hits = max(3, self.z_jump_reinit_hits)
                if self.xy_jump_reject_count >= recover_hits:
                    # 接受新位姿；触发地图清空和暂停重建
                    import time as _monotime
                    self._trigger_map_reset("XY jump EKF reset", x_meas, y_meas,
                                            delta_xy_m=float(dxy))
                    self.xy_jump_reject_count = 0
                else:
                    x_meas = float(self.last_valid_x)
                    y_meas = float(self.last_valid_y)
            else:
                self.xy_jump_reject_count = 0

        if self.prefer_dist_bottom_altitude:
            dist_bottom_valid = bool(getattr(msg, "dist_bottom_valid", False))
            dist_bottom = float(getattr(msg, "dist_bottom", np.nan))
            if dist_bottom_valid and np.isfinite(dist_bottom) and self.dist_bottom_min <= dist_bottom <= self.dist_bottom_max:
                z_from_bottom = -dist_bottom
                if self.altitude_est_z is None:
                    self.altitude_est_z = z_from_bottom
                else:
                    a = float(np.clip(self.dist_bottom_alpha, 0.01, 1.0))
                    self.altitude_est_z = a * z_from_bottom + (1.0 - a) * self.altitude_est_z
                z_meas = float(self.altitude_est_z)

        # ★[F2] 起飞完成后首次进入飞行高度时重置 last_valid_z，仅触发一次
        # 原条件 last_valid_z > -0.10 会让哨兵值 -0.05 每帧都触发（死循环），
        # 新条件：仅在 last_valid_z 处于地面/哨兵范围内（< 0.8m）且尚未重置时执行一次。
        if (self.takeoff_done
                and self.last_valid_z is not None
                and self.last_valid_z < 0.8
                and not getattr(self, '_z_takeoff_reset_done', False)):
            self.get_logger().info(
                f"[COORD] takeoff_done: resetting last_valid_z from {self.last_valid_z:.3f} to raw {z_meas:.3f}"
            )
            self.last_valid_z = None
            self.z_jump_reject_count = 0
            self._z_takeoff_reset_done = True  # 只重置一次

        if self.last_valid_z is not None and self.last_pose_msg_time is not None:
            dt = (now - self.last_pose_msg_time).nanoseconds * 1e-9
            dz = abs(z_meas - self.last_valid_z)
            # dt<5ms 同 XY：同一仿真 tick，Z 轴速度同样无意义，仅做距离检查
            if dt <= self.z_jump_reject_dt and dz > self.z_jump_reject_m and (dt >= 0.005 or dz > 0.5):
                self.z_jump_reject_count += 1
                if self.z_jump_reject_count < max(1, self.z_jump_reinit_hits):
                    self.get_logger().error(
                        f"[COORD ERROR] Sudden Z jump detected: z={z_meas:.3f} (raw={raw_z:.3f}), "
                        f"last z={self.last_valid_z:.3f}, dz={dz:.3f}m in dt={dt:.3f}s, rejecting sample "
                        f"({self.z_jump_reject_count}/{self.z_jump_reinit_hits})"
                    )
                    z_meas = float(self.last_valid_z)
                else:
                    if self.altitude_est_z is not None and np.isfinite(self.altitude_est_z):
                        z_recovered = float(self.altitude_est_z)
                    else:
                        dt_eff = max(dt, 0.05)
                        max_step = max(0.1, self.z_recovery_rate_mps * dt_eff)
                        z_recovered = float(self.last_valid_z + np.clip(z_meas - self.last_valid_z, -max_step, max_step))
                    self.get_logger().warn(
                        f"[COORD RECOVER] persistent z jump, switch to recovered z={z_recovered:.3f} "
                        f"from raw={raw_z:.3f}, last={self.last_valid_z:.3f}"
                    )
                    z_meas = z_recovered
                    self.z_jump_reject_count = 0
            else:
                self.z_jump_reject_count = 0

        dynamic_z_upper = float(self.bounds[2, 1] + 0.02)
        positive_z_limit = min(self.z_positive_clamp, dynamic_z_upper)
        if z_meas > positive_z_limit:
            if (now - self.last_coord_warn_time).nanoseconds * 1e-9 > 1.0:
                self.get_logger().warn(
                    f"[COORD WARN] positive z sample z={z_meas:.3f} raw={raw_z:.3f}, clamp to last valid"
                )
                self.last_coord_warn_time = now
            z_meas = float(self.last_valid_z if self.last_valid_z is not None else -0.05)

        if not self.pose_initialized:
            elapsed = (now - self.pose_init_start_time).nanoseconds * 1e-9
            valid = (-3.5 < z_meas < -0.03)
            if valid:
                self.pose_init_stable_count += 1
            else:
                self.pose_init_stable_count = 0

            if self.pose_init_stable_count >= self.pose_init_required_stable or elapsed > self.pose_init_timeout:
                self.pose_initialized = True
                self.get_logger().info(
                    f"[INIT] pose initialized at z={z_meas:.3f}, stable={self.pose_init_stable_count}, elapsed={elapsed:.1f}s"
                )

        self.pose = msg
        self.pose.x = float(x_meas)
        self.pose.y = float(y_meas)
        self.pose.z = float(z_meas)
        self.last_valid_x = float(x_meas)
        self.last_valid_y = float(y_meas)
        self.last_valid_z = float(z_meas)
        self.last_pose_msg_time = now

        # 修复A+：用 PX4 自带的 reset counter 检测真实EKF reset（比跳变检测更可靠）
        xy_rc = int(getattr(msg, "xy_reset_counter", -1))
        hdg_rc = int(getattr(msg, "heading_reset_counter", -1))
        z_rc   = int(getattr(msg, "z_reset_counter", -1))
        if self._last_xy_reset_counter >= 0 and xy_rc != self._last_xy_reset_counter:
            delta_xy = getattr(msg, "delta_xy", None)
            dxy_str = f"delta_xy={list(delta_xy)}" if delta_xy is not None else ""
            # 计算 EKF 跳变幅度（用于大幅跳变检测）
            _delta_xy_m = (float(np.hypot(delta_xy[0], delta_xy[1]))
                           if delta_xy is not None and len(delta_xy) >= 2 else 0.0)
            self.get_logger().warn(
                f"[EKF-RESET] xy_reset_counter {self._last_xy_reset_counter}->{xy_rc} "
                f"{dxy_str} (Δxy={_delta_xy_m:.2f}m), clearing map"
            )
            self._trigger_map_reset("xy_reset_counter", float(x_meas), float(y_meas),
                                    delta_xy_m=_delta_xy_m)
        if self._last_hdg_reset_counter >= 0 and hdg_rc != self._last_hdg_reset_counter:
            delta_hdg = float(getattr(msg, "delta_heading", 0.0))
            self.get_logger().warn(
                f"[EKF-RESET] heading_reset_counter {self._last_hdg_reset_counter}->{hdg_rc} "
                f"delta={np.rad2deg(delta_hdg):.1f}°, clearing map"
            )
            self._trigger_map_reset("heading_reset_counter", float(x_meas), float(y_meas),
                                    delta_xy_m=0.0)  # heading reset 不涉及 XY 跳变
        if self._last_z_reset_counter >= 0 and z_rc != self._last_z_reset_counter:
            delta_z = float(getattr(msg, "delta_z", 0.0))
            self.get_logger().warn(
                f"[EKF-RESET] z_reset_counter {self._last_z_reset_counter}->{z_rc} "
                f"delta_z={delta_z:.3f}m"
            )
            # z reset 仅清空 belief，不重置 HOVER_SCAN（不影响水平建图）
            if hasattr(self, "belief_grid"):
                self.belief_grid[:] = 0.3
            self.get_logger().info("[EKF-RESET] z_reset: belief cleared（不重置水平建图流程）")
        if xy_rc >= 0:
            self._last_xy_reset_counter = xy_rc
        if hdg_rc >= 0:
            self._last_hdg_reset_counter = hdg_rc
        if z_rc >= 0:
            self._last_z_reset_counter = z_rc

    def _get_map_pose(self) -> tuple:
        """查询无人机在 SLAM global map frame 的位姿。

        返回: (x, y, yaw, valid, stale, source)
          valid  : False → 位姿不可用，不能用于控制
          stale  : True  → 来自缓存而非新鲜 TF（禁止影响关键决策）
          source : 'tf_map_base' | 'tf_map_odom_plus_odom' |
                   'cache_recent' | 'cache_stale' | 'odom_fallback' | 'invalid'

        优先级:
          1. TF map→base_link（新鲜）
          2. TF map→odom + /odom 位置（odom 新鲜 ≤0.5s）
          3. 缓存 ≤0.2s → valid=True, stale=False
          4. 缓存 ≤1.0s → valid=True, stale=True（禁止用于控制）
          5. /odom 直接用（仅启动阶段 map≈odom 时有意义，stale=True）
          6. invalid — PX4 NED fallback 已移除（pose.x/y 是 NED，不是 map frame）

        ★ 重要修复：att_yaw 在 att_cb() 里已是 ENU yaw（π/2 - heading_ned 已做过），
          这里不再重复转换。
        """
        import time as _gmp_t
        _now_gmp = _gmp_t.monotonic()
        _att_yaw_enu = float(self.att_yaw or 0.0)  # 已是 ENU yaw，直接使用，不再转换

        # ── 1. 新鲜 TF map→base_link ──────────────────────────────────────────
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            x = float(t.transform.translation.x)
            y = float(t.transform.translation.y)
            q = [t.transform.rotation.x, t.transform.rotation.y,
                 t.transform.rotation.z, t.transform.rotation.w]
            yaw = float(R.from_quat(q).as_euler('zyx')[0])
            self._last_map_pose = (x, y, yaw)
            self._last_map_pose_mono = _now_gmp
            return x, y, yaw, True, False, 'tf_map_base'
        except Exception:
            pass

        # ── 2. TF map→odom + /odom 位置变换到 map 坐标 ───────────────────────
        _odom_age = _now_gmp - self._odom_stamp_mono
        if self._odom_received and _odom_age < 0.5:
            try:
                t2 = self._tf_buffer.lookup_transform(
                    'map', 'odom', rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05))
                _tx = float(t2.transform.translation.x)
                _ty = float(t2.transform.translation.y)
                _q2 = [t2.transform.rotation.x, t2.transform.rotation.y,
                       t2.transform.rotation.z, t2.transform.rotation.w]
                _rot2 = R.from_quat(_q2)
                # 将 /odom ENU 位置旋转到 map frame
                _odom_pt = _rot2.apply([self._odom_x_enu, self._odom_y_enu, 0.0])
                _mx = _tx + _odom_pt[0]
                _my = _ty + _odom_pt[1]
                # yaw: 使用 att_yaw（已是 ENU），不做二次转换
                self._last_map_pose = (_mx, _my, _att_yaw_enu)
                self._last_map_pose_mono = _now_gmp
                return _mx, _my, _att_yaw_enu, True, False, 'tf_map_odom_plus_odom'
            except Exception:
                pass

        # ── 3 & 4. 缓存 fallback ──────────────────────────────────────────────
        # cache_recent ≤0.2s（原0.3s收紧，防止旧坐标混入控制）
        # cache_stale  ≤1.0s（原2.0s收紧，减少过时数据影响决策）
        _cache_age = _now_gmp - getattr(self, '_last_map_pose_mono', 0.0)
        if hasattr(self, '_last_map_pose') and self._last_map_pose != (0.0, 0.0, 0.0):
            if _cache_age <= 0.2:
                return (*self._last_map_pose, True, False, 'cache_recent')
            elif _cache_age <= 1.0:
                return (*self._last_map_pose, True, True, 'cache_stale')

        # ── 5. /odom 直接当 map 估计（仅启动阶段 map≈odom 时有意义）──────────
        if self._odom_received and _odom_age < 1.0:
            return (self._odom_x_enu, self._odom_y_enu, _att_yaw_enu,
                    True, True, 'odom_fallback')

        # ── 6. px4_fallback 已移除 ──────────────────────────────────────────
        # pose.x/pose.y 是 PX4 NED odom，不是 SLAM map frame，混用会造成
        # 坐标系污染（grid 查询偏差、规划目标错位）。返回 invalid，
        # 让调用层进入 SAFE_HOLD，不发布错误位置。

        return 0.0, 0.0, 0.0, False, True, 'invalid'

    def _trigger_map_reset(self, reason: str, x: float, y: float,
                           delta_xy_m: float = 0.0) -> None:
        """EKF reset时：暂停建图 + 清空 belief/语义层，不清除 slam 托管的 grid。
        状态机转换: STABLE → PAUSE（在 command_loop 中 pause 过期后 → STABLE）

        delta_xy_m: EKF 跳变的 XY 幅度（米）。
          >= _ekf_large_jump_th_m (default 5m) → 使用 _map_pause_sec_large (default 8s)
                                                   + 调用 slam_toolbox Pause 暂停扫描摄入
          < threshold                            → 使用 _map_pause_sec (default 3s)，正常处理
        """
        import time as _monotime
        # ── 大幅跳变检测 ──────────────────────────────────────────────────────
        _is_large = delta_xy_m >= float(getattr(self, '_ekf_large_jump_th_m', 5.0))
        _effective_pause_sec = (
            float(getattr(self, '_map_pause_sec_large', 8.0))
            if _is_large else self._map_pause_sec
        )
        # 注意：self.grid 由 slam_toolbox /map 回调管理，不在此处清除，
        # 否则会使地图可视化一直为空（每次 EKF reset 后要等 1 Hz /map 回调才恢复）。
        if hasattr(self, "belief_grid"):
            self.belief_grid[:] = 0.3  # 重置到初始先验
        if hasattr(self, "room_labels"):
            self.room_labels[:] = 0
        # ★ EKF reset：清空占据图（短暂空图比保留错误地图安全得多）
        # 旧 grid 基于错误坐标系，会造成持续叠图；清空后等 SLAM /map 重建
        self.grid[:] = 128
        # ★ EKF reset：清空时域滤波缓存，避免旧图残影叠入新建图
        self._grid_hist = []
        # ★ EKF reset：同步清空三层地图稳定层（log-odds + Layer2 stable + Layer1 raw）
        if hasattr(self, '_stable_log_odds'):
            self._stable_log_odds[:] = 0.0
        if hasattr(self, '_stable_occ_grid'):
            self._stable_occ_grid[:] = 128
        if hasattr(self, '_raw_slam_grid'):
            self._raw_slam_grid[:] = 128
        # ★ EKF reset：清空门候选缓存，避免旧 EKF 坐标系下的门信息残存
        self._door_cand_cache = {}
        self._door_confirmed_stable = []
        self.rooms_info = []
        self.doors_info = []
        self._map_pause_until = _monotime.monotonic() + _effective_pause_sec
        # SSG：地图重置后强制重新扫描
        self._map_frozen = True   # 先冻结，等 _map_pause_until 过后由 HOVER_SCAN 解冻
        self.explore_state = "HOVER_SCAN"
        self.hover_scan_start_time = None   # 重置：确保下次进入HOVER_SCAN重新初始化
        self._hover_scan_hold_pos = None    # 重置锚定位置（PX4 NED setpoint用），下次扫描开始时重新记录
        self._hover_scan_anchor_map = None  # 重置 SLAM map frame 锚点（drift gate用）
        self._hover_scan_last_yaw = None
        self.hover_scan_accum_angle = 0.0
        self._hover_valid_frames = 0        # 重置有效帧计数
        self.ssg_viewpoint_goal = None
        self.astar_path = None
        # 重置探索扩散标志
        self._ssg_clustered = False
        # 记录 reset 单调时间，用于过滤 reset 前的旧 scan（防坐标系污染）
        import time as _mt
        self._last_reset_mono = _mt.monotonic()
        # 同时重置 scan 去重计数器，确保 reset 后首帧 scan 也能正确积分
        self._last_integrated_seq = -1

        # ── Phase9: 进入 MAP_RECOVER 状态（地图恢复模式）────────────────────
        self._map_recover_active = True
        self._map_recover_since_mono = _mt.monotonic()
        self._occ_stable_frames = 0   # 重置稳定帧计数，等待地图稳定退出恢复
        # 大跳变时使用更多稳定帧才允许恢复
        if _is_large:
            self._occ_stable_needed = getattr(self, '_recover_stable_frames_needed', 20)
        else:
            self._occ_stable_needed = 5

        # ── EKF reset 状态机 STABLE → PAUSE ─────────────────────────────────
        prev_state = getattr(self, '_ekf_reset_state', 'STABLE')
        self._ekf_reset_state = "PAUSE"
        self._ekf_reset_count = getattr(self, '_ekf_reset_count', 0) + 1
        self._log_event(
            "EKF-RESET",
            f"{prev_state}→PAUSE (#{self._ekf_reset_count}) reason={reason} "
            f"at ({x:.2f},{y:.2f}) Δxy={delta_xy_m:.2f}m "
            f"pause={_effective_pause_sec:.1f}s ({'大幅跳变' if _is_large else '普通reset'})"
        )
        self.get_logger().warn(
            f"[MAP-RECOVER] enter #{self._ekf_reset_count}: reason={reason} "
            f"Δxy={delta_xy_m:.2f}m pause={_effective_pause_sec:.1f}s large={_is_large}"
        )

        # ── 大幅跳变：调用 slam_toolbox Pause 停止扫描摄入 ───────────────────
        # Pause 才能真正阻断错位扫描积分入地图
        # Bridge 的 TF-FREEZE 是第一道防线，Pause 是第二道防线。
        if _is_large:
            self._slam_pause_scan(True)
            # 统计近期大跳变次数（滑动窗口 large_jump_window_sec）
            _now_lj = _mt.monotonic()
            _win_sec = getattr(self, '_large_jump_window_sec', 30.0)
            if _now_lj - getattr(self, '_large_jump_window_start', 0.0) > _win_sec:
                self._large_jump_count = 0
                self._large_jump_window_start = _now_lj
            self._large_jump_count = getattr(self, '_large_jump_count', 0) + 1
            _abort_cnt = getattr(self, '_large_jump_abort_count', 3)
            _hard_hold = getattr(self, '_large_jump_hard_hold_sec', 5.0)
            self._map_recover_hard_until = _now_lj + _hard_hold
            # ★[G2] 大幅跳变紧急 XY 锁定：command_loop 所有分支禁止发出 XY 逃离速度
            # 原因：EKF 跳 44m+ 后 att_yaw 极不可靠，任何 XY 速度都可能把无人机打向墙壁
            self._ekf_emergency_xy_hold_until = _now_lj + _effective_pause_sec
            self.get_logger().warn(
                f"[EKF-EMERG-XY-HOLD] 大幅跳变 Δxy={delta_xy_m:.2f}m"
                f" → 紧急锁定 XY 速度 {_effective_pause_sec:.1f}s"
            )
            self._log_event(
                "MAP-RECOVER-HARD",
                f"enter reason={reason} Δxy={delta_xy_m:.2f}m "
                f"large_jump_count={self._large_jump_count}/{_abort_cnt} "
                f"hard_hold={_hard_hold:.1f}s slam_pause=True ekf_xy_hold={_effective_pause_sec:.1f}s"
            )
            self.get_logger().warn(
                f"[MAP-RECOVER-HARD] enter reason={reason} Δxy={delta_xy_m:.2f}m "
                f"count={self._large_jump_count}/{_abort_cnt} hard_hold={_hard_hold:.0f}s"
            )
            # 连续大跳变达到上限 → 进入 SAFE_HOLD，停止探索
            if self._large_jump_count >= _abort_cnt:
                self._large_jump_count = 0
                self._large_jump_window_start = _now_lj
                self._log_event(
                    "MAP-RECOVER-ABORT",
                    f"too_many_large_jumps ({_abort_cnt}次/{_win_sec:.0f}s) → SAFE_HOLD"
                )
                self.get_logger().error(
                    f"[MAP-RECOVER-ABORT] {_abort_cnt}次大跳变/{_win_sec:.0f}s "
                    f"→ SAFE_HOLD，停止探索，等待 EKF 稳定"
                )
                self.explore_state = "HOVER_SCAN"
                self.state = "SAFE_HOLD"
            else:
                self.get_logger().warn(
                    f"[EKF-RESET-LARGE] Δxy={delta_xy_m:.2f}m ≥ "
                    f"{getattr(self,'_ekf_large_jump_th_m',5.0):.1f}m "
                    f"→ 扩展暂停 {_effective_pause_sec:.1f}s + slam Pause 已请求"
                )

        # ── 可选：调用 slam_toolbox clear_changes 服务（param_gated）────────
        if getattr(self, '_slam_reset_on_ekf', False):
            cli = getattr(self, '_slam_clear_cli', None)
            if cli is not None and cli.service_is_ready():
                try:
                    from slam_toolbox.srv import Clear as _SlamClear
                    _req = _SlamClear.Request()
                    _fut = cli.call_async(_req)
                    _fut.add_done_callback(self._slam_clear_done_cb)
                    self.get_logger().warn(
                        "[EKF-STATE] → 已异步调用 /slam_toolbox/clear_changes "
                        f"(reset #{getattr(self, '_ekf_reset_count', 0)})"
                    )
                except Exception as _sce:
                    self.get_logger().warn(
                        f"[EKF-STATE] clear_changes call_async 失败: {_sce}"
                    )
            elif cli is not None:
                self.get_logger().warn(
                    "[EKF-STATE] /slam_toolbox/clear_changes 服务未就绪，跳过 SLAM clear "
                    "(slam_toolbox 是否已启动？)"
                )
            else:
                self.get_logger().warn(
                    "[EKF-STATE] slam_reset_on_ekf_reset=true 但客户端未初始化 "
                    "(slam_toolbox.srv.Clear 导入失败？)"
                )

    def _slam_clear_done_cb(self, future) -> None:
        """slam_toolbox /clear_changes 异步调用完成回调。
        注意：clear_changes 在 interactive mode 禄用时将被忽略，不能作为地图已清除的依据。
        地图恢复主流程是：pause_new_measurements + 停规划 + 等稳定。
        """
        try:
            future.result()
            self.get_logger().info(
                "[EKF-STATE] /slam_toolbox/clear_changes 请求已返回，"
                "但不作为重置成功判据——若 interactive mode 禄用则已被忽略"
            )
        except Exception as exc:
            self.get_logger().warn(f"[EKF-STATE] /slam_toolbox/clear_changes 请求异常: {exc}")

    def att_cb(self, msg: VehicleAttitude) -> None:
        now = self.get_clock().now()
        self.att_rot = R.from_quat([msg.q[1], msg.q[2], msg.q[3], msg.q[0]])
        try:
            e = self.att_rot.as_euler("zyx")
            _yaw_ned = float(e[0])    # PX4/NED yaw: 0=北, 顺时针为正
            self.att_yaw_ned = _yaw_ned
            # ── ENU yaw = π/2 - NED_yaw（与 bridge 约定完全一致）─────────────────
            # heading=0(北) → yaw_enu=π/2(面北ENU) ✓  heading=π/2(东) → yaw_enu=0 ✓
            self.att_yaw = float((math.pi / 2.0 - _yaw_ned + math.pi) % (2.0 * math.pi) - math.pi)
            self.att_pitch = float(e[1])
            self.att_roll  = float(e[2])
        except Exception:
            self.att_yaw_ned = None
            self.att_yaw     = None
            self.att_pitch   = None
            self.att_roll    = None
        # ── 计算机体倾角 tilt_deg（与坐标系约定无关，用四元数旋转 z 轴）──────
        # body-z 轴在 NED world 中的方向（FRD: [0,0,1] = 机体朝下; 悬停时 = [0,0,1]）
        # tilt = body_z 与 world_z [0,0,1] 的夹角；悬停时 ≈ 0°，侧翻 90° 时 = 90°
        try:
            _bz = self.att_rot.apply([0.0, 0.0, 1.0])
            _cz = float(np.clip(_bz[2], -1.0, 1.0))
            self._att_tilt_local = float(math.degrees(math.acos(abs(_cz))))
        except Exception:
            self._att_tilt_local = 0.0
        # ── 更新平面位姿有效性（★[D] 加入滞回去抖）────────────────────────────
        _tilt_thr = getattr(self, '_planar_tilt_threshold_deg', 20.0)
        import time as _planar_t
        _planar_now = _planar_t.monotonic()
        if self._att_tilt_local > _tilt_thr:
            # tilt 超阈值：记录首次超阈值时间
            if getattr(self, '_planar_invalid_since', 0.0) == 0.0:
                self._planar_invalid_since = _planar_now
            _trigger_s = getattr(self, 'planar_invalid_trigger_s', 0.4)
            if _planar_now - self._planar_invalid_since >= _trigger_s:
                if getattr(self, '_planar_pose_valid', True):
                    # 首次进入 invalid：记录日志
                    self.get_logger().warn(
                        f"[PLANAR] enter tilt={self._att_tilt_local:.1f}deg "
                        f"after={_planar_now - self._planar_invalid_since:.2f}s "
                        f"threshold={_tilt_thr:.0f}deg"
                    )
                self._planar_pose_valid = False
                self._planar_pose_invalid_reason = (
                    f"tilt={self._att_tilt_local:.1f}deg>{_tilt_thr:.1f}deg"
                )
            # 未到触发时间：暂不更改 valid 状态
            self._planar_recover_since = 0.0   # 恢复计时器复位
        else:
            self._planar_invalid_since = 0.0    # 超阈值计时器复位
            # tilt 低于阈值：记录首次低于阈值时间
            if getattr(self, '_planar_recover_since', 0.0) == 0.0:
                self._planar_recover_since = _planar_now
            _recover_s = getattr(self, 'planar_invalid_recover_s', 1.0)
            if _planar_now - self._planar_recover_since >= _recover_s:
                if not getattr(self, '_planar_pose_valid', True):
                    # 首次恢复：记录日志
                    self.get_logger().info(
                        f"[PLANAR] exit tilt={self._att_tilt_local:.1f}deg "
                        f"after={_planar_now - self._planar_recover_since:.2f}s"
                    )
                self._planar_pose_valid = True
                self._planar_pose_invalid_reason = 'ok'
        # ── [ATT-ENU] 1Hz 节流诊断日志（通过 _log_event 写入 log/events_*）────────
        import time as _att_t
        _tnow_att = _att_t.monotonic()
        if _tnow_att - getattr(self, '_att_enu_log_t', 0.0) >= 1.0:
            self._att_enu_log_t = _tnow_att
            if self.att_yaw is not None and self.att_yaw_ned is not None:
                _p_deg = math.degrees(float(self.att_pitch)) if self.att_pitch is not None else float('nan')
                _r_deg = math.degrees(float(self.att_roll))  if self.att_roll  is not None else float('nan')
                self._log_event(
                    "ATT-FRAME",
                    f"yaw_ned={math.degrees(self.att_yaw_ned):.1f}deg "
                    f"yaw_enu={math.degrees(self.att_yaw):.1f}deg "
                    f"pitch={_p_deg:.1f}deg roll={_r_deg:.1f}deg"
                )
                self._log_event(
                    "ATT-RAW-CHECK",
                    f"px4_raw_yaw={math.degrees(self.att_yaw_ned):.1f}deg "
                    f"enu_yaw={math.degrees(self.att_yaw):.1f}deg "
                    f"formula=pi/2-ned_yaw "
                    f"diff={abs(math.degrees(self.att_yaw_ned) - math.degrees(self.att_yaw)):.1f}deg"
                )
    def status_cb(self, msg: VehicleStatus) -> None:
        armed_now = (msg.arming_state == 2)
        if armed_now and not self.armed:
            self.armed_since = self.get_clock().now()
        if (not armed_now) and self.armed:
            self.armed_since = None
            # 无人机被disarm（降落/强制锁定）→ 重置起飞标志，等待重新arm后再起飞
            if self.takeoff_done:
                self.get_logger().warn(
                    "[STATUS] 无人机被disarm，重置takeoff_done以便重新起飞"
                )
                self.takeoff_done = False
                self.alt_takeoff_hold_start = None
                self.locked_yaw = None   # 重置locked_yaw，重新起飞时重新锁定
        self.armed = armed_now
        self.nav_state = int(msg.nav_state)
        self.arming_state = int(msg.arming_state)
        prev_offboard = getattr(self, 'offboard_mode', False)
        self.offboard_mode = (msg.nav_state == 14)
        # offboard 模式恢复后重置恢复计数，允许下次 offboard_lost 时重新尝试
        if self.offboard_mode and not prev_offboard and self._offboard_recovery_count > 0:
            self.get_logger().info(
                f"[OFFBOARD-RECOVER] offboard 模式已恢复，重置恢复计数 "
                f"(之前尝试 {self._offboard_recovery_count} 次)"
            )
            self._offboard_recovery_count = 0
            self._last_offboard_recovery_mono = 0.0

    def _lidar_raw_to_body_angles(self, raw_angles: np.ndarray) -> np.ndarray:
        """将激光雷达 raw 角度统一转换为机体(body) frame 角度。
        body  0° = 机体前方 (+X)
        body +90° = 机体右方 (+Y)
        body ±180° = 机体后方
        body -90° = 机体左方 (-Y)

        公式：body_angle = raw_angle + lidar_mount_yaw_deg
        lidar_mount_yaw_deg 默认 0°（已修正）：传感器 X 轴与机体前方(+X_body)对齐，
        与 px4_slam_bridge_node 中的 TF base_link→laser_link = Rz(0°) 保持一致。
        历史注：原值为 -90°，导致 SLAM 地图 90° CCW 旋转，已于修复版统一为 0°。
        """
        mount_offset_rad = np.deg2rad(float(self.lidar_mount_yaw_deg))
        body = raw_angles + mount_offset_rad
        # ★ 修复A：归一化到 [-π, π]，防止 body_min=-270°/body_max=90° 扇区误判
        # 激光雷达 raw_angles 通常是 0→2π，加上 -π/2 偏移后变成 -π/2→3π/2，
        # 不归一化会导致 front/left/right/back 扇区判断全部错位，障碍/地图写入方向偏错 90°
        return np.arctan2(np.sin(body), np.cos(body))

    def scan_cb(self, msg: LaserScan, source: str) -> None:
        if msg is None or len(msg.ranges) == 0:
            return
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()
        self.last_scan_source = source
        self._scan_seq += 1   # 每帧新scan序号+1，供 update_map_from_scan 去重用

        ranges = np.array(msg.ranges, dtype=float)
        raw_angles = msg.angle_min + np.arange(len(ranges), dtype=float) * msg.angle_increment

        # ── 统一角度变换：raw → body frame（与 update_map_from_scan 使用同一函数）──
        # body  0°    = 机体前方(+X)
        # body +90°   = 机体右方(+Y)
        # body ±180°  = 机体后方
        # body -90°   = 机体左方(-Y)
        body_angles = self._lidar_raw_to_body_angles(raw_angles)

        # DEBUG 日志：每 60 扫描帧输出一次传感器角度范围
        if not hasattr(self, '_scan_dbg_cnt'):
            self._scan_dbg_cnt = 0
        self._scan_dbg_cnt += 1
        if self._scan_dbg_cnt % 60 == 1:
            self.get_logger().info(
                f"[LIDAR-DBG] angle_min={np.rad2deg(msg.angle_min):.1f}° "
                f"angle_max={np.rad2deg(msg.angle_max):.1f}° "
                f"mount_yaw={self.lidar_mount_yaw_deg:.1f}° "
                f"body_min={np.rad2deg(body_angles.min()):.1f}° "
                f"body_max={np.rad2deg(body_angles.max()):.1f}° "
                f"npts={len(ranges)}"
            )

        # 扇区定义（body frame，±35° 范围，与 mount_offset 无关）：
        #   front: body(-35°, +35°)   = 机体前方
        #   right: body(+55°, +125°)  = 机体右方
        #   back:  body(±145°, ±180°) = 机体后方
        #   left:  body(-125°, -55°)  = 机体左方
        sectors = {
            "front": [(-np.deg2rad(35),  np.deg2rad(35))],
            "right": [( np.deg2rad(55),  np.deg2rad(125))],
            "back":  [( np.deg2rad(145), np.pi), (-np.pi, -np.deg2rad(145))],
            "left":  [(-np.deg2rad(125), -np.deg2rad(55))],
        }

        for key, intervals in sectors.items():
            mask = np.zeros_like(body_angles, dtype=bool)
            for a0, a1 in intervals:
                if a0 <= a1:
                    mask |= (body_angles >= a0) & (body_angles <= a1)
                else:
                    mask |= (body_angles >= a0) | (body_angles <= a1)
            vals = ranges[mask]
            vals = vals[np.isfinite(vals)]
            # 过滤机体/起落架近距反射噪声
            min_dist = float(getattr(self, "lidar_min_valid_dist", 0.15))
            vals = vals[vals >= min_dist]
            if vals.size < max(1, self.lidar_sector_min_points):
                dist = None if vals.size == 0 else float(np.min(vals))
            else:
                p = float(np.clip(self.lidar_sector_percentile, 1.0, 50.0))
                dist = float(np.percentile(vals, p))

            if dist is not None:
                last_val = self.lidar_ranges.get(key)
                if last_val is not None and np.isfinite(last_val):
                    a = float(np.clip(self.lidar_sector_ema_alpha, 0.05, 1.0))
                    dist = a * dist + (1.0 - a) * float(last_val)
            self.lidar_ranges[key] = dist

        # DEBUG：每 60 帧输出扇区距离，便于验证方向映射是否正确
        if self._scan_dbg_cnt % 60 == 1:
            def _fmt(v):
                return f"{v:.2f}" if (v is not None and np.isfinite(v)) else "---"
            self.get_logger().info(
                f"[LIDAR-SECTORS] F={_fmt(self.lidar_ranges.get('front'))} "
                f"L={_fmt(self.lidar_ranges.get('left'))} "
                f"R={_fmt(self.lidar_ranges.get('right'))} "
                f"B={_fmt(self.lidar_ranges.get('back'))} (body frame扇区)"
            )

    # ===================================================================
    # plan frame ↔ SLAM map frame SE(2) 变换工具
    # ===================================================================
    def _tf2d_from_pose(self, x: float, y: float, yaw: float) -> np.ndarray:
        """构造 3×3 SE(2) 齐次变换矩阵 T = [R|t; 0 1]。"""
        import math as _m2d
        c, s = _m2d.cos(yaw), _m2d.sin(yaw)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]], dtype=float)

    def _invert_tf2d(self, T: np.ndarray) -> np.ndarray:
        """求 SE(2) 矩阵的逆变换。"""
        Rt = T[:2, :2].T
        t = T[:2, 2]
        Ti = np.eye(3, dtype=float)
        Ti[:2, :2] = Rt
        Ti[:2, 2] = -Rt @ t
        return Ti

    def _apply_tf2d(self, T: np.ndarray, x: float, y: float) -> tuple:
        """将 SE(2) 变换应用于点 (x, y)，返回变换后的 (x', y')。"""
        v = T @ np.array([x, y, 1.0], dtype=float)
        return float(v[0]), float(v[1])

    def _map_pose_to_plan_pose(self, x_map: float, y_map: float, yaw_map: float) -> tuple:
        """将 SLAM map frame 位姿转换到 plan frame 坐标。

        说明：
          - plan frame = 固定 ENU 规划网格坐标系（由 _plan_bounds_xy 定义，永不变）
          - SLAM map frame = slam_toolbox 输出的 /map TF frame（ENU，与 odom 偏差来自
            loop closure 或初始位置偏移）
          - 两者之间的差异来自 OccupancyGrid origin.orientation 的旋转分量
            （_slam_origin_yaw，由 map_cb 每帧更新）
          - 典型情况：slam_toolbox 输出零旋转 → 此函数为恒等变换
          - 非零旋转：以 SLAM 栅格原点为旋转中心，对 map frame 位姿做逆旋转补偿

        Args:
            x_map, y_map: SLAM map frame 坐标（来自 _get_map_pose）
            yaw_map:      SLAM map frame ENU yaw（rad）
        Returns:
            (x_plan, y_plan, yaw_plan): plan frame 坐标
        """
        import math as _mpp
        _yaw = float(getattr(self, '_slam_origin_yaw', 0.0))
        if abs(_yaw) < 1e-4:
            # 快速路径：无旋转，plan frame = map frame
            return x_map, y_map, yaw_map
        # 以 SLAM 栅格索引零点（ox, oy）为旋转中心，施加逆旋转
        _ox = float(getattr(self, '_slam_map_origin_x', 0.0))
        _oy = float(getattr(self, '_slam_map_origin_y', 0.0))
        _dx, _dy = x_map - _ox, y_map - _oy
        _c, _s = _mpp.cos(-_yaw), _mpp.sin(-_yaw)
        _xp = _ox + _c * _dx - _s * _dy
        _yp = _oy + _s * _dx + _c * _dy
        return _xp, _yp, (yaw_map - _yaw)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip((x - self.bounds[0, 0]) / self.map_res, 0, self.grid_shape[0] - 1))
        iy = int(np.clip((y - self.bounds[1, 0]) / self.map_res, 0, self.grid_shape[1] - 1))
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        x = float(self.bounds[0, 0] + ix * self.map_res)
        y = float(self.bounds[1, 0] + iy * self.map_res)
        return x, y

    def _map_vel_to_odom_vel(self, vx_map: float, vy_map: float) -> tuple:
        """将 SLAM map 帧速度向量旋转到 odom 帧（标准 ROS ENU: x=东, y=北）。

        SLAM loop closure 后 map→odom TF 包含旋转分量，不可忽略。
        odom 帧 = ENU (x=东, y=北, z=上)，符合 REP-103。
        注意：输出是 ENU 速度 (vx=East, vy=North)，调用方（publish_setpoint）
        还需做 ENU→NED swap：[vel[1], vel[0], ...]，才能送给 PX4 TrajectorySetpoint。

        Args:
            vx_map, vy_map: SLAM map frame 速度分量 (m/s)
        Returns:
            (vx_odom, vy_odom): odom/ENU frame 速度分量 (vx=East, vy=North, m/s)
        """
        try:
            t = self._tf_buffer.lookup_transform(
                'odom', 'map',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.02))
            q = [t.transform.rotation.x, t.transform.rotation.y,
                 t.transform.rotation.z, t.transform.rotation.w]
            rot = R.from_quat(q)
            v_odom = rot.apply([vx_map, vy_map, 0.0])
            self._map_to_odom_rot_cache = rot
            return float(v_odom[0]), float(v_odom[1])
        except Exception:
            # TF 不可用时使用缓存旋转（常见于启动初期或短暂 TF 断联）
            if self._map_to_odom_rot_cache is not None:
                v_odom = self._map_to_odom_rot_cache.apply([vx_map, vy_map, 0.0])
                return float(v_odom[0]), float(v_odom[1])
            # 最终降级：不旋转（map 初始启动 ≈ odom，误差微小）
            return vx_map, vy_map

    def update_map_from_scan(self, now) -> None:
        """
        纯 slam_toolbox 模式：占据栅格由 /map 回调（map_cb）维护，
        此函数仅更新 belief_grid 的负证据衰减（HOVER_SCAN 阶段地图解冻时执行）。
        """
        if self.pose is None or self.last_scan is None:
            return
        if not self.scan_is_recent(now):
            return

        # 频率控制
        if self._scan_seq == self._last_integrated_seq:
            return
        if (now - self.last_map_update).nanoseconds * 1e-9 < self.map_update_interval:
            return
        self.last_map_update = now
        self._last_integrated_seq = self._scan_seq

        # slam_toolbox 还未提供地图时不做 belief 衰减
        if not getattr(self, '_slam_map_received', False):
            return

        # ★ 修复：不再用 _map_frozen 阻断建图。
        # 导航期间（NAV_TO_VIEWPOINT）地图也应继续更新，否则无人机飞到新区域后
        # 地图永远不会扩展（"地图锁死" 根因）。
        # 仅在 HOVER_SCAN_ESCAPE 激活期间跳过，因为此时位姿可能抖动。
        if getattr(self, '_hover_scan_escape_active', False):
            return

        import time as _mitp
        _mi_now = _mitp.monotonic()

        # [MAP-INTEGRATE-SKIP] Guard 1: EKF reset 后 2s 内跳过，防止位移错位扫描写入
        _last_reset_ago = _mi_now - getattr(self, '_last_reset_mono', 0.0)
        if _last_reset_ago < 2.0:
            if _mi_now - getattr(self, '_map_integrate_skip_log_t', 0.0) > 1.0:
                self._map_integrate_skip_log_t = _mi_now
                self._log_event(
                    "MAP-INTEGRATE-SKIP",
                    f"reason=recent_ekf_reset reset_ago={_last_reset_ago:.2f}s < 2.0s"
                )
            return

        # [MAP-INTEGRATE-SKIP] Guard 2: MAP_RECOVER hard_hold 期间跳过 belief 写入
        if _mi_now < getattr(self, '_map_recover_hard_until', 0.0):
            if _mi_now - getattr(self, '_map_integrate_skip_log_t', 0.0) > 1.0:
                self._map_integrate_skip_log_t = _mi_now
                _remain = self._map_recover_hard_until - _mi_now
                self._log_event(
                    "MAP-INTEGRATE-SKIP",
                    f"reason=map_recover_hard_hold remaining={_remain:.1f}s"
                )
            return

        # [MAP-INTEGRATE-SKIP] Guard 3: HOVER_SCAN anchor drift 过大（无人机已离开
        # HOVER 起始点太远），此时扫描角度偏差过大，不应更新 belief
        _hover_anchor = getattr(self, '_hover_scan_anchor_pos', None)
        if _hover_anchor is not None and self._map_px is not None:
            _dx = float(self._map_px) - float(_hover_anchor[0])
            _dy = float(self._map_py) - float(_hover_anchor[1])
            _drift = float(np.hypot(_dx, _dy))
            _drift_gate = getattr(self, '_hover_scan_pos_gate_m', 0.30)
            if _drift > _drift_gate:
                if _mi_now - getattr(self, '_map_integrate_skip_log_t', 0.0) > 1.0:
                    self._map_integrate_skip_log_t = _mi_now
                    self._log_event(
                        "MAP-INTEGRATE-SKIP",
                        f"reason=hover_anchor_drift drift={_drift:.3f}m > gate={_drift_gate:.2f}m"
                    )
                return

        # ★ 修复D补充：额外姿态/速度门控——大倾斜角或大垂向速度时跳过 belief 更新
        # 防止无人机位置/朝向不稳时对 belief_grid 写入虚假负证据（扫描弧影）
        import time as _mitp2
        _mi_now2 = _mitp2.monotonic()
        _scan_drop_reason: Optional[str] = None
        # Guard: HOVER_SCAN_ESCAPE 激活 → 边逃边扫会引入错位数据
        if self._hover_scan_escape_active:
            _scan_drop_reason = "hover_scan_escape_active"
        # Guard: CORNER_TRAP 激活 → 卡角时扫描无效
        elif _mi_now2 < getattr(self, '_corner_trap_until', 0.0):
            _scan_drop_reason = "corner_trap_active"
        # Guard: 近障（< 0.45m）→ 不应继续扫图
        else:
            _near_mi = self._nearest_lidar_distance()
            if _near_mi is not None and _near_mi < 0.45:
                _scan_drop_reason = f"too_close nearest={_near_mi:.2f}m"
        if _scan_drop_reason is not None:
            if _mi_now2 - getattr(self, '_scan_drop_log_t', 0.0) > 1.0:
                self._scan_drop_log_t = _mi_now2
                self.get_logger().debug(f"[SCAN-DROP] reason={_scan_drop_reason}")
            return
        # Gate: 姿态/速度（阈值收紧到 8°）
        if self.att_pitch is not None and self.att_roll is not None:
            pitch_deg = abs(float(self.att_pitch)) * 180.0 / np.pi
            roll_deg  = abs(float(self.att_roll))  * 180.0 / np.pi
            if pitch_deg > 8.0 or roll_deg > 8.0:
                self.get_logger().debug(
                    f"[SCAN-DROP] reason=tilt pitch={pitch_deg:.1f}° roll={roll_deg:.1f}° > 8°"
                )
                return
        if self.pose is not None:
            vz_now = float(getattr(self.pose, 'vz', 0.0))
            if abs(vz_now) > 0.05:
                self.get_logger().debug(
                    f"[SCAN-DROP] reason=vz={vz_now:.3f}m/s > 0.05"
                )
                return

        scan = self.last_scan
        ranges = np.array(scan.ranges, dtype=float)
        raw_angles = scan.angle_min + np.arange(len(ranges), dtype=float) * scan.angle_increment
        angles = self._lidar_raw_to_body_angles(raw_angles)
        valid = np.isfinite(ranges) & (ranges > 0.05)
        if scan.range_max > 0:
            valid &= (ranges < scan.range_max)
        ranges = ranges[valid]
        angles = angles[valid]
        if ranges.size == 0:
            return

        # ★ 使用 plan frame 坐标（_plan_px/_plan_py）访问 self.grid（plan 网格）
        px = float(self._plan_px)
        py = float(self._plan_py)
        yaw = float(self._plan_yaw)
        cos_y = float(np.cos(yaw))
        sin_y = float(np.sin(yaw))

        # [MAP-INTEGRATE] 标记本次地图积分
        self._log_event("MAP-INTEGRATE", f"layer=occupancy_only pts={int(ranges.size)}")
        # 8 等角方向，沿射线直到命中障碍或边界，沿途 free 格衰减 belief
        n_dir = 8
        for i_dir in range(n_dir):
            a_dir = (2.0 * np.pi * i_dir) / n_dir
            bx_d = float(np.cos(a_dir))
            by_d = float(np.sin(a_dir))
            dx_map = cos_y * bx_d - sin_y * by_d   # ENU map frame，非 NED
            dy_map = sin_y * bx_d + cos_y * by_d   # ENU map frame，非 NED
            r_vis = 0.1
            while r_vis < self.ssg_lidar_range:
                vx_b = px + dx_map * r_vis
                vy_b = py + dy_map * r_vis
                if not (self.bounds[0, 0] <= vx_b <= self.bounds[0, 1] and
                        self.bounds[1, 0] <= vy_b <= self.bounds[1, 1]):
                    break
                vix, viy = self.world_to_grid(vx_b, vy_b)
                if self.grid[vix, viy] >= self.prob_occ_th:
                    break
                self.belief_grid[vix, viy] *= self.belief_decay_coeff
                self.belief_grid[vix, viy] = max(0.0, float(self.belief_grid[vix, viy]))
                r_vis += self.map_res * 2.0

    # ================================================================
    # A* 路径规划器：在占据栅格地图上规划无碰撞路径
    # ================================================================
    def astar_plan(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> Optional[list]:
        """
        在当前栅格地图上用A*算法规划从start到goal的路径。
        返回路径点列表 [(x0,y0), (x1,y1), ...]（世界坐标），或None（无路径）。
        障碍物先膨胀 astar_inflation_radius，防止飞机擦墙。
        ★ astar_treat_unknown_as_occ=True（默认）：将未知格视为障碍，
          防止规划路径穿过未扫描区域（未扫描区可能有墙）。
        """
        W, H = self.grid_shape
        occ_mask = self.grid >= self.prob_occ_th

        # ★ 未知格当障碍：slam_toolbox 未知区存在不可预测障碍物，
        #   由 astar_treat_unknown_as_occ 参数控制（True=保守/False=宽松门口穿越）
        _treat_unk_as_occ_now = getattr(self, 'astar_treat_unknown_as_occ', True)
        if _treat_unk_as_occ_now:
            unk_mask = (self.grid > self.prob_free_th) & (self.grid < self.prob_occ_th)
            occ_mask = occ_mask | unk_mask

        # 障碍物膨胀（安全余量）
        infl_cells = max(1, int(self.astar_inflation_radius / self.map_res))
        occ_inflated = binary_dilation(occ_mask, iterations=infl_cells)

        # ★ A* clearance 代价地图（距最近障碍栅格的距离，单位:m）
        # 用于给低 clearance 格子加入额外代价，使路径自然偏向通道中线
        try:
            from scipy.ndimage import distance_transform_edt as _astar_dt_edt
            _clr_map = _astar_dt_edt(~occ_mask).astype(float) * self.map_res   # meters
        except Exception:
            _clr_map = None
        _ASTAR_CLR_LAMBDA = 4.0   # ★ clearance 代价权重（1.5→4.0，强中线偏置，路径贴墙→走通道中线；值越高越远离墙壁）
        _ASTAR_CLR_EPS    = 0.05  # 避免除零

        # 起点/终点转格子坐标
        sx, sy = self.world_to_grid(float(start_xy[0]), float(start_xy[1]))
        gx, gy = self.world_to_grid(float(goal_xy[0]), float(goal_xy[1]))
        sx = int(np.clip(sx, 0, W - 1))
        sy = int(np.clip(sy, 0, H - 1))
        gx = int(np.clip(gx, 0, W - 1))
        gy = int(np.clip(gy, 0, H - 1))

        # ★ 起点吸附：若起点落在膨胀障碍区，找最近自由格后继续规划
        if occ_inflated[sx, sy]:
            _start_cells = np.argwhere(~occ_inflated)
            if _start_cells.shape[0] == 0:
                return None
            _sd = np.abs(_start_cells[:, 0] - sx) + np.abs(_start_cells[:, 1] - sy)
            _sb = _start_cells[np.argmin(_sd)]
            import time as _ss_t
            if _ss_t.monotonic() - getattr(self, '_astar_start_snap_log_t', 0.0) > 2.0:
                self._astar_start_snap_log_t = _ss_t.monotonic()
                self.get_logger().info(
                    f"[ASTAR-START-SNAP] 起点({sx},{sy})在膨胀障碍区 → 吸附到({int(_sb[0])},{int(_sb[1])})"
                )
            sx, sy = int(_sb[0]), int(_sb[1])

        if occ_inflated[gx, gy]:
            # 目标被障碍覆盖：寻找附近最近自由格
            free_cells = np.argwhere(~occ_inflated)
            if free_cells.shape[0] == 0:
                return None
            dists = np.abs(free_cells[:, 0] - gx) + np.abs(free_cells[:, 1] - gy)
            best = free_cells[np.argmin(dists)]
            gx, gy = int(best[0]), int(best[1])

        if sx == gx and sy == gy:
            return [(float(goal_xy[0]), float(goal_xy[1]))]

        # A* 搜索
        import heapq
        INF = float('inf')
        g_cost = np.full((W, H), INF, dtype=float)
        came_from = {}
        g_cost[sx, sy] = 0.0

        def h(x, y):
            return float(np.sqrt((x - gx)**2 + (y - gy)**2))

        open_heap = [(h(sx, sy), sx, sy)]
        dirs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        step_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

        found = False
        while open_heap:
            f, cx, cy = heapq.heappop(open_heap)
            if cx == gx and cy == gy:
                found = True
                break
            if f > g_cost[cx, cy] + h(cx, cy) + 1e-6:
                continue
            for (dx, dy), sc in zip(dirs8, step_costs):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if occ_inflated[nx, ny]:
                    continue
                # ★ 禁止斜穿墙角（对角移动时检查两个相邻正交格必须都是自由格）
                if dx != 0 and dy != 0:
                    if occ_inflated[cx + dx, cy] or occ_inflated[cx, cy + dy]:
                        continue  # 两侧任一被堵 → 禁止该对角扩展
                # ★ clearance 代价：低 clearance 格子代价更高，路径自然偏向通道中线
                if _clr_map is not None:
                    _clr_v = max(float(_clr_map[nx, ny]), _ASTAR_CLR_EPS)
                    _clr_penalty = _ASTAR_CLR_LAMBDA / _clr_v
                else:
                    _clr_penalty = 0.0
                ng = g_cost[cx, cy] + sc + _clr_penalty
                if ng < g_cost[nx, ny]:
                    g_cost[nx, ny] = ng
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_heap, (ng + h(nx, ny), nx, ny))

        if not found:
            # ★ 详细 A* 失败日志
            import time as _af_t
            if _af_t.monotonic() - getattr(self, '_astar_fail_log_t', 0.0) > 1.5:
                self._astar_fail_log_t = _af_t.monotonic()
                _unk_cells = int(np.sum((self.grid > self.prob_free_th) & (self.grid < self.prob_occ_th)))
                _total_cells = W * H
                _unk_pct = 100.0 * _unk_cells / _total_cells if _total_cells > 0 else 0.0
                _goal_in_raw_occ = bool(self.grid[gx, gy] >= self.prob_occ_th)
                _goal_in_infl = bool(occ_inflated[gx, gy])
                _start_in_infl = bool(occ_inflated[sx, sy])
                self.get_logger().warn(
                    f"[ASTAR-FAIL] start_grid=({sx},{sy}) goal_grid=({gx},{gy}) "
                    f"start_in_occ={_start_in_infl} goal_in_occ_raw={_goal_in_raw_occ} "
                    f"goal_in_occ_infl={_goal_in_infl} "
                    f"unk%={_unk_pct:.1f} infl={self.astar_inflation_radius:.3f}m "
                    f"unk_as_occ={getattr(self,'astar_treat_unknown_as_occ',True)}"
                )
            return None

        # 回溯路径
        path_cells = []
        cur = (gx, gy)
        while cur != (sx, sy):
            path_cells.append(cur)
            cur = came_from.get(cur)
            if cur is None:
                return None
        path_cells.append((sx, sy))
        path_cells.reverse()

        # 路径简化（去掉共线点，减少航点数量）
        # 使用 line-of-sight 检测：若 A->C 无障碍则跳过中间点B
        def has_los(ax, ay, bx, by):
            """检查格 (ax,ay) 到 (bx,by) 之间是否有障碍（Bresenham line）"""
            dx, dy = abs(bx - ax), abs(by - ay)
            sx2 = 1 if bx > ax else -1
            sy2 = 1 if by > ay else -1
            err = dx - dy
            cx2, cy2 = ax, ay
            while True:
                if occ_inflated[cx2, cy2]:
                    return False
                if cx2 == bx and cy2 == by:
                    return True
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    cx2 += sx2
                if e2 < dx:
                    err += dx
                    cy2 += sy2

        # ★ [A4b] 净空感知路径简化：LOS检查同时要求路径线段中间点净空 >= 阈值
        # 防止简化后路径走门框边缘（A*代价加权找到了中线，但直线连接会绕过中线）
        _MIN_SIMPL_CLR_M = 0.20   # 简化线段中间采样点最小净空(m)
        def has_los_with_clr(ax, ay, bx, by):
            """LOS + 净空双重检查：确保简化后直线不穿越低净空区"""
            if not has_los(ax, ay, bx, by):
                return False
            if _clr_map is None:
                return True
            _dx_s, _dy_s = bx - ax, by - ay
            _seg_len = max(abs(_dx_s), abs(_dy_s))
            if _seg_len < 4:   # 短线段跳过净空检查
                return True
            # 均匀采样中间3点，要求最小净空
            for _t_s in (0.25, 0.5, 0.75):
                _sx_s = int(round(ax + _t_s * _dx_s))
                _sy_s = int(round(ay + _t_s * _dy_s))
                if 0 <= _sx_s < W and 0 <= _sy_s < H:
                    if float(_clr_map[_sx_s, _sy_s]) < _MIN_SIMPL_CLR_M:
                        return False
            return True

        simplified = [path_cells[0]]
        i = 0
        while i < len(path_cells) - 1:
            j = len(path_cells) - 1
            while j > i + 1:
                if has_los_with_clr(path_cells[i][0], path_cells[i][1],
                                    path_cells[j][0], path_cells[j][1]):
                    break
                j -= 1
            simplified.append(path_cells[j])
            i = j

        # 转回世界坐标
        path_world = []
        for ix, iy in simplified:
            wx, wy = self.grid_to_world(ix, iy)
            path_world.append((wx, wy))

        return path_world

    # ================================================================
    # 矩形房间边界拟合：从占据格中找到最外层墙壁，拟合轴对齐矩形
    # 返回 (xmin_cell, xmax_cell, ymin_cell, ymax_cell) 或None
    # ================================================================
    def fit_room_rectangle(self, free_mask: np.ndarray, occ_mask: np.ndarray):
        """
        策略：
        1. 找到所有障碍物格，投影到X轴和Y轴
        2. 找到自由区域的凸包边界（X/Y的min/max）
        3. 在边界附近搜索连续障碍物段作为「墙」
        4. 返回矩形四条边的格子坐标
        """
        W, H = self.grid_shape
        if np.count_nonzero(free_mask) < 50:
            return None

        # 找到自由区域的最大范围
        free_xs, free_ys = np.where(free_mask)
        if len(free_xs) == 0:
            return None
        fx_min, fx_max = int(free_xs.min()), int(free_xs.max())
        fy_min, fy_max = int(free_ys.min()), int(free_ys.max())

        # 在自由区域边缘向外找最近的连续障碍物「墙」
        wall_search = max(3, int(0.5 / self.map_res))  # 向外搜索距离

        def find_wall_line(axis, side, search_range):
            """在自由区域edge外side方向搜索最近的障碍行/列"""
            if axis == 'x':
                # 在fy_min..fy_max列范围内搜索
                for di in search_range:
                    col = di
                    if 0 <= col < W:
                        # 计算这列在y范围内的障碍率
                        seg = occ_mask[col, max(0, fy_min):min(H, fy_max+1)]
                        # ★ 修复：原0.3平均阈值对稀疏地图过于严格，改为存在任意障碍格
                        if len(seg) > 0 and np.any(seg):
                            return col
            else:  # axis == 'y'
                for dj in search_range:
                    row = dj
                    if 0 <= row < H:
                        seg = occ_mask[max(0, fx_min):min(W, fx_max+1), row]
                        if len(seg) > 0 and np.any(seg):
                            return row
            return None

        xmin_wall = find_wall_line('x', 'min', range(fx_min - 1, max(-1, fx_min - wall_search - 1), -1))
        xmax_wall = find_wall_line('x', 'max', range(fx_max + 1, min(W, fx_max + wall_search + 1)))
        ymin_wall = find_wall_line('y', 'min', range(fy_min - 1, max(-1, fy_min - wall_search - 1), -1))
        ymax_wall = find_wall_line('y', 'max', range(fy_max + 1, min(H, fy_max + wall_search + 1)))

        # 若某侧没找到墙，用自由区边界作为估计
        xmin_c = xmin_wall if xmin_wall is not None else fx_min
        xmax_c = xmax_wall if xmax_wall is not None else fx_max
        ymin_c = ymin_wall if ymin_wall is not None else fy_min
        ymax_c = ymax_wall if ymax_wall is not None else fy_max

        return (xmin_c, xmax_c, ymin_c, ymax_c)

    # ================================================================
    # Phase-6 根因修复：Manhattan 墙段提取 + 基于真实墙段的门缺口检测
    # 替代原「外包矩形」策略，正确处理迷宫内部墙和走廊结构
    # ================================================================
    def _extract_wall_segments(
        self,
        occ_mask: np.ndarray,
        min_len_m: float = 0.4,
        morph_dilate_cells: int = 1,
    ) -> list:
        """
        从占据格栅提取轴对齐（Manhattan）墙段。
        策略：
          ① 先对 occ_mask 做 1 格膨胀，合并相邻 1 格噪声间隙
          ② 逐行扫描：找连续占据 run → 水平段（constant iy）
          ③ 逐列扫描：找连续占据 run → 垂直段（constant ix）
          ④ 长度 < min_len_cells 丢弃（去除单格孤立噪点）
        返回: list[dict]
          {'ix0','iy0','ix1','iy1','orient'('H'|'V'),'len_cells'}
        全部坐标为 planning-grid cell 索引 (ix=x_dir, iy=y_dir)。
        """
        import cv2 as _cv2_ws
        min_len_cells = max(2, int(min_len_m / self.map_res))
        W, H = self.grid_shape

        # 轻度膨胀：填 1 格间隙，使断续墙体连成段
        if morph_dilate_cells >= 1:
            _kel = _cv2_ws.getStructuringElement(
                _cv2_ws.MORPH_RECT,
                (2 * morph_dilate_cells + 1, 2 * morph_dilate_cells + 1))
            _occ_u8 = occ_mask.astype(np.uint8)
            _occ_u8 = _cv2_ws.dilate(_occ_u8, _kel)
            occ_ex = _occ_u8.astype(bool)
        else:
            occ_ex = occ_mask

        segments: list = []

        # ── 水平段（H）：constant iy，沿 ix 方向 ──
        for iy in range(H):
            run_start = None
            for ix in range(W + 1):
                occ = (ix < W) and bool(occ_ex[ix, iy])
                if occ:
                    if run_start is None:
                        run_start = ix
                else:
                    if run_start is not None:
                        length = ix - run_start
                        if length >= min_len_cells:
                            segments.append({
                                'ix0': run_start, 'iy0': iy,
                                'ix1': ix - 1,    'iy1': iy,
                                'orient': 'H',    'len_cells': length,
                            })
                        run_start = None

        # ── 垂直段（V）：constant ix，沿 iy 方向 ──
        for ix in range(W):
            run_start = None
            for iy in range(H + 1):
                occ = (iy < H) and bool(occ_ex[ix, iy])
                if occ:
                    if run_start is None:
                        run_start = iy
                else:
                    if run_start is not None:
                        length = iy - run_start
                        if length >= min_len_cells:
                            segments.append({
                                'ix0': ix, 'iy0': run_start,
                                'ix1': ix, 'iy1': iy - 1,
                                'orient': 'V', 'len_cells': length,
                            })
                        run_start = None

        return segments

    def _detect_doors_from_wall_segments(
        self,
        free_mask: np.ndarray,
        occ_mask: np.ndarray,
        wall_segments: list,
    ) -> list:
        """
        基于真实墙段（Manhattan）的门缺口检测。
        对每一段墙，沿墙延伸方向扫描墙内自由缺口：
          ① 缺口宽度在 [door_min, door_max] 之间
          ② 缺口前端和后端各有 ≥1 格占据格（墙支撑）
          ③ 缺口格本身为 free（非 unknown）
        返回候选列表: [(ix, iy, orient_str, width_cells), ...]
        """
        door_min_cells = max(2, int(self.door_min_width / self.map_res))
        door_max_cells = max(door_min_cells + 1, int(self.door_max_width / self.map_res))
        W, H = self.grid_shape
        # 噪声容忍：缺口内允许 ≤noise_tol 格 unknown（非free非occ），继续计入 gap
        noise_tol = max(1, int(0.12 / self.map_res))

        raw_candidates: list = []

        def _has_wall_support(coord_arr: np.ndarray, pos: int, mask_2d, ax: str, side: int) -> bool:
            """在 pos 的 side 方向 1~3 格内是否有占据格"""
            W2, H2 = mask_2d.shape
            for k in range(1, 4):
                p = pos + side * k
                if ax == 'ix':
                    if 0 <= p < W2 and bool(mask_2d[p, coord_arr]):
                        return True
                else:
                    if 0 <= p < H2 and bool(mask_2d[coord_arr, p]):
                        return True
            return False

        for seg in wall_segments:
            orient = seg['orient']

            if orient == 'H':
                # 水平墙：constant iy，沿 ix 扫描
                iy = seg['iy0']
                in_gap = False
                gap_start = -1
                unk_run = 0  # 连续 unknown 计数

                for ix in range(seg['ix0'], seg['ix1'] + 2):
                    in_bounds = (ix < W) and (ix <= seg['ix1'])
                    is_occ  = in_bounds and bool(occ_mask[ix, iy])
                    is_free = in_bounds and bool(free_mask[ix, iy])

                    if in_bounds and is_free and not is_occ:
                        if not in_gap:
                            in_gap = True
                            gap_start = ix
                            unk_run = 0
                    elif in_bounds and not is_occ and not is_free and in_gap:
                        # unknown 格：容忍计入
                        unk_run += 1
                        if unk_run > noise_tol:
                            in_gap = False
                            gap_start = -1
                            unk_run = 0
                    else:
                        if in_gap and gap_start >= 0:
                            gap_end = ix - 1
                            free_cnt = sum(
                                1 for gx in range(gap_start, gap_end + 1)
                                if gx < W and bool(free_mask[gx, iy])
                            )
                            if door_min_cells <= free_cnt <= door_max_cells:
                                l_ok = _has_wall_support(iy, gap_start, occ_mask, 'ix', -1)
                                r_ok = _has_wall_support(iy, gap_end,   occ_mask, 'ix', +1)
                                if l_ok and r_ok:
                                    mid_ix = (gap_start + gap_end) // 2
                                    raw_candidates.append((mid_ix, iy, 'horizontal', free_cnt))
                        in_gap = False
                        gap_start = -1
                        unk_run = 0

            else:  # orient == 'V'
                # 垂直墙：constant ix，沿 iy 扫描
                ix = seg['ix0']
                in_gap = False
                gap_start = -1
                unk_run = 0

                for iy_s in range(seg['iy0'], seg['iy1'] + 2):
                    in_bounds = (iy_s < H) and (iy_s <= seg['iy1'])
                    is_occ  = in_bounds and bool(occ_mask[ix, iy_s])
                    is_free = in_bounds and bool(free_mask[ix, iy_s])

                    if in_bounds and is_free and not is_occ:
                        if not in_gap:
                            in_gap = True
                            gap_start = iy_s
                            unk_run = 0
                    elif in_bounds and not is_occ and not is_free and in_gap:
                        unk_run += 1
                        if unk_run > noise_tol:
                            in_gap = False
                            gap_start = -1
                            unk_run = 0
                    else:
                        if in_gap and gap_start >= 0:
                            gap_end = iy_s - 1
                            free_cnt = sum(
                                1 for gy in range(gap_start, gap_end + 1)
                                if gy < H and bool(free_mask[ix, gy])
                            )
                            if door_min_cells <= free_cnt <= door_max_cells:
                                b_ok = _has_wall_support(ix, gap_start, occ_mask, 'iy', -1)
                                t_ok = _has_wall_support(ix, gap_end,   occ_mask, 'iy', +1)
                                if b_ok and t_ok:
                                    mid_iy = (gap_start + gap_end) // 2
                                    raw_candidates.append((ix, mid_iy, 'vertical', free_cnt))
                        in_gap = False
                        gap_start = -1
                        unk_run = 0

        return raw_candidates

    # ================================================================
    # 新版门检测辅助
    # ================================================================
    def _estimate_local_width_cells(self, free_mask: np.ndarray, ix: int, iy: int, axis: str, max_cells: int) -> int:
        c1 = 0
        c2 = 0
        if axis == "x":
            i = ix - 1
            while i >= 0 and c1 < max_cells and free_mask[i, iy]:
                c1 += 1
                i -= 1
            i = ix + 1
            while i < self.grid_shape[0] and c2 < max_cells and free_mask[i, iy]:
                c2 += 1
                i += 1
        else:
            j = iy - 1
            while j >= 0 and c1 < max_cells and free_mask[ix, j]:
                c1 += 1
                j -= 1
            j = iy + 1
            while j < self.grid_shape[1] and c2 < max_cells and free_mask[ix, j]:
                c2 += 1
                j += 1
        return c1 + c2 + 1

    # ================================================================
    # ★[B] 走廊区域分类
    # ================================================================
    def _classify_corridor_region(
        self,
        free_mask: np.ndarray,
        occ_mask: np.ndarray,
        ix: int, iy: int,
        orientation: str,
    ) -> tuple:
        """
        判定门候选点 (ix,iy) 处是否处于走廊（长宽比过大的通道），
        而不是真正的门洞。
        返回 (is_corridor: bool, reason: str)
        判定逻辑：
          - 沿门穿越方向（穿过门的轴）测量"深度"（自由延伸长度）
          - 沿门宽方向测量"宽度"
          - 若 depth/width > corridor_aspect_ratio_min → 走廊
          - 若 width > corridor_width_max_m → 走廊（过宽）
        """
        W, H = self.grid_shape
        res = self.map_res
        # 穿越方向 = 走廊延伸方向
        # orientation="horizontal" → 穿越 y 轴（沿 +/-y 测深度）；宽度沿 x
        # orientation="vertical"   → 穿越 x 轴（沿 +/-x 测深度）；宽度沿 y
        def _run(axis: str, sign: int) -> int:
            count = 0
            if axis == "x":
                i = ix + sign
                while 0 <= i < W and free_mask[i, iy] and count < 200:
                    count += 1; i += sign
            else:
                j = iy + sign
                while 0 <= j < H and free_mask[ix, j] and count < 200:
                    count += 1; j += sign
            return count

        if orientation == "horizontal":
            depth_cells = _run("y", -1) + _run("y", 1) + 1
            width_cells = _run("x", -1) + _run("x", 1) + 1
        else:
            depth_cells = _run("x", -1) + _run("x", 1) + 1
            width_cells = _run("y", -1) + _run("y", 1) + 1

        depth_m = depth_cells * res
        width_m = width_cells * res

        if width_m <= 0.01:
            return False, "zero_width"

        # 走廊：depth/width 过大 且 宽度未超出"走廊最大宽度"范围（否则是广场，不是走廊）
        if width_m > self.corridor_width_max_m:
            return False, f"width_too_large({width_m:.2f}m)"

        aspect = depth_m / width_m
        if aspect >= self.corridor_aspect_ratio_min:
            # 进一步验证：走廊两侧应有平行墙段（增加置信度）
            # 检查沿宽度方向±corridor_parallel_wall_len范围内是否有连续墙
            _par_len_cells = max(2, int(self.corridor_parallel_wall_len / res))
            _waxis = "x" if orientation == "horizontal" else "y"
            _wall_side1, _wall_side2 = False, False
            if _waxis == "x":
                # 检查 ix-1 和 ix+1 行（贴墙侧）
                _lo = max(0, iy - _par_len_cells // 2)
                _hi = min(H, iy + _par_len_cells // 2 + 1)
                if ix > 0:
                    _wall_side1 = bool(np.any(occ_mask[ix - 1, _lo:_hi]))
                if ix < W - 1:
                    _wall_side2 = bool(np.any(occ_mask[ix + 1, _lo:_hi]))
            else:
                _lo = max(0, ix - _par_len_cells // 2)
                _hi = min(W, ix + _par_len_cells // 2 + 1)
                if iy > 0:
                    _wall_side1 = bool(np.any(occ_mask[_lo:_hi, iy - 1]))
                if iy < H - 1:
                    _wall_side2 = bool(np.any(occ_mask[_lo:_hi, iy + 1]))
            if _wall_side1 and _wall_side2:
                _corr_msg = f"corridor(aspect={aspect:.1f},w={width_m:.2f}m,d={depth_m:.2f}m)"
                self.get_logger().debug(
                    f"[CORRIDOR-CLASSIFY] ix={ix},iy={iy} ori={orientation} "
                    f"aspect={aspect:.1f} w={width_m:.2f}m d={depth_m:.2f}m → is_corridor"
                )
                return True, _corr_msg
        return False, f"ok(aspect={aspect:.1f},w={width_m:.2f}m)"

    # ================================================================
    # ★[C] 门几何有效性检查
    # ================================================================
    def _check_door_geometry(
        self,
        d: dict,
        free_mask: np.ndarray,
        occ_mask: np.ndarray,
        rooms_dict: dict,
    ) -> tuple:
        """
        对候选 confirmed_door 进行几何有效性校验。
        返回 (geom_ok: bool, reason: str, door_subtype: str)
          door_subtype: 'confirmed_door' | 'corridor_connector'
        校验项：
          1. 宽度约束：door_width_min_m ≤ width_m ≤ door_width_max_m
          2. 走廊排除：_classify_corridor_region 判定为走廊 → corridor_connector
          3. 门框支撑：门旁两侧应有墙体（jamb support）
          4. 两侧房间面积：≥ door_room_min_area_m2
          5. 地图质量门：occ_cells ≥ door_upgrade_min_map_occ
        """
        ix = int(d.get("ix", 0))
        iy = int(d.get("iy", 0))
        ori = str(d.get("orientation", "vertical"))
        w_m = float(d.get("width_m", 0.0))
        room_a = int(d.get("room_a", -1))
        room_b = int(d.get("room_b", -1))

        # 1. 宽度约束
        if w_m < self.door_width_min_m:
            return False, f"too_narrow({w_m:.2f}<{self.door_width_min_m})", "confirmed_door"
        if w_m > self.door_width_max_m:
            return False, f"too_wide({w_m:.2f}>{self.door_width_max_m})", "confirmed_door"

        # 2. 地图质量门（occ格数）
        _occ_cnt = int(np.count_nonzero(occ_mask))
        if _occ_cnt < self.door_upgrade_min_map_occ:
            return False, f"map_not_ready(occ={_occ_cnt}<{self.door_upgrade_min_map_occ})", "confirmed_door"

        # 3. 两侧房间面积
        for _rid in (room_a, room_b):
            if _rid > 0 and _rid in rooms_dict:
                _area = float(rooms_dict[_rid].get("area_m2", 0.0))
                if _area < self.door_room_min_area_m2:
                    return False, f"room_too_small(R{_rid}={_area:.1f}<{self.door_room_min_area_m2})", "confirmed_door"

        # 4. 走廊排除（仅当房间够大时才检查走廊）
        _is_corr, _corr_reason = self._classify_corridor_region(
            free_mask, occ_mask, ix, iy, ori)
        if _is_corr:
            # 返回 geom_ok=True 但 subtype=corridor_connector，让调用方保留为走廊边
            self.get_logger().debug(
                f"[DOOR-GEOM] ix={ix},iy={iy} w={w_m:.2f}m → corridor_connector ({_corr_reason})"
            )
            return True, _corr_reason, "corridor_connector"

        # 5. 门框墙体支撑（jamb support）
        _res = self.map_res
        W, H = self.grid_shape
        _jamb_cells = max(1, int(self.door_jamb_support_min_len / _res))
        # 门的宽度方向 = 与穿越方向垂直
        # horizontal(穿越y) → 宽度沿x → jamb在±x方向检查iy处是否有墙
        # vertical(穿越x)   → 宽度沿y → jamb在±y方向检查ix处是否有墙
        _half_w_cells = max(1, int(round(w_m / (2.0 * _res))))
        _jamb_ok = False
        if ori == "horizontal":
            # jamb在左右(ix±_half_w_cells)侧 ±1 格 iy 方向
            for _side in (-1, 1):
                _jx = ix + _side * (_half_w_cells + 1)
                if 0 <= _jx < W:
                    _lo = max(0, iy - _jamb_cells)
                    _hi = min(H, iy + _jamb_cells + 1)
                    if np.any(occ_mask[_jx, _lo:_hi]):
                        _jamb_ok = True
                        break
        else:
            for _side in (-1, 1):
                _jy = iy + _side * (_half_w_cells + 1)
                if 0 <= _jy < H:
                    _lo = max(0, ix - _jamb_cells)
                    _hi = min(W, ix + _jamb_cells + 1)
                    if np.any(occ_mask[_lo:_hi, _jy]):
                        _jamb_ok = True
                        break
        if not _jamb_ok:
            return False, f"no_jamb_support(w={w_m:.2f}m,ori={ori})", "confirmed_door"

        self.get_logger().debug(
            f"[DOOR-CONFIRMED] ix={ix},iy={iy} ori={ori} w={w_m:.2f}m "
            f"Ra={room_a} Rb={room_b} → confirmed_door"
        )
        return True, "ok", "confirmed_door"

    def detect_doors(self, free_mask: np.ndarray, occ_mask: np.ndarray) -> list[dict]:
        """
        Phase-6 根因修复版门检测策略（适合迷宫/多房间）：
        1. 策略1（主）：提取真实 Manhattan 墙段，逐段扫描缺口
           → 正确处理迷宫内部墙、走廊间隔墙（原「外包矩形」策略只能处理单房间）
        2. 策略2（补）：向量化双侧夹墙检测，补充走廊/窄通道情形
        合并去重后输出 doors 列表。
        """
        door_min_cells = max(2, int(self.door_min_width / self.map_res))
        door_max_cells = max(door_min_cells + 1, int(self.door_max_width / self.map_res))
        W, H = self.grid_shape

        candidates = []  # list of (ix, iy, orientation, width_cells)

        # ---- 策略1（主）：基于真实墙段的缺口检测 ----
        # 提取 Manhattan 轴对齐墙段 → 在每段墙上找门洞缺口
        # 优势：正确处理迷宫内部墙（不依赖外包矩形边界），门支持率提升
        _wall_segs = self._extract_wall_segments(occ_mask, min_len_m=0.4)
        self._last_wall_seg_count = len(_wall_segs)   # Phase-6：供诊断日志使用
        _seg_cands = self._detect_doors_from_wall_segments(free_mask, occ_mask, _wall_segs)
        candidates += _seg_cands
        self.get_logger().debug(
            f"[DOOR-WSEG] 墙段数={len(_wall_segs)} "
            f"缺口候选={len(_seg_cands)} "
            f"door_range=[{door_min_cells},{door_max_cells}]格"
        )

        # ---- 策略2：向量化双侧夹墙检测（走廊/内门）----
        # ★ 修复：原O(W×H)嵌套Python循环在0.05m分辨率(117×237格)耗时>2s，
        #         改为scipy.ndimage向量化操作，速度提升100x+
        from scipy.ndimage import maximum_filter
        wall_look = max(2, int(0.4 / self.map_res))
        occ_u8 = occ_mask.astype(np.uint8)
        # 构造单向扫描footprint：中心格本身不算，只看一侧wall_look格内是否有障碍
        fp_neg_x = np.zeros((2 * wall_look + 1, 1), dtype=np.uint8)  # -x方向
        fp_neg_x[:wall_look, 0] = 1
        has_wall_neg_x = (maximum_filter(occ_u8, footprint=fp_neg_x, mode='constant', cval=0) > 0)
        fp_pos_x = np.zeros((2 * wall_look + 1, 1), dtype=np.uint8)  # +x方向
        fp_pos_x[wall_look + 1:, 0] = 1
        has_wall_pos_x = (maximum_filter(occ_u8, footprint=fp_pos_x, mode='constant', cval=0) > 0)
        fp_neg_y = np.zeros((1, 2 * wall_look + 1), dtype=np.uint8)  # -y方向
        fp_neg_y[0, :wall_look] = 1
        has_wall_neg_y = (maximum_filter(occ_u8, footprint=fp_neg_y, mode='constant', cval=0) > 0)
        fp_pos_y = np.zeros((1, 2 * wall_look + 1), dtype=np.uint8)  # +y方向
        fp_pos_y[0, wall_look + 1:] = 1
        has_wall_pos_y = (maximum_filter(occ_u8, footprint=fp_pos_y, mode='constant', cval=0) > 0)
        # horizontal候选：free且±x两侧均有墙（在x方向被夹住的通道）
        h_cand_mask = free_mask & has_wall_neg_x & has_wall_pos_x
        # vertical候选：free且±y两侧均有墙
        v_cand_mask = free_mask & has_wall_neg_y & has_wall_pos_y
        # 只对通过向量化预过滤的候选格做width检查（大幅减少Python调用次数）
        for ix, iy in np.argwhere(h_cand_mask):
            wy = self._estimate_local_width_cells(free_mask, int(ix), int(iy), "y", door_max_cells * 2)
            if door_min_cells <= wy <= door_max_cells:
                candidates.append((int(ix), int(iy), "horizontal", wy))
        for ix, iy in np.argwhere(v_cand_mask):
            wx = self._estimate_local_width_cells(free_mask, int(ix), int(iy), "x", door_max_cells * 2)
            if door_min_cells <= wx <= door_max_cells:
                candidates.append((int(ix), int(iy), "vertical", wx))

        # ---- 策略3（新）[E1]：dist_field 瓶颈检测（软门/窄通道补漏）----
        # 用 distance_transform_edt 找局部最窄点（dt < 0.75m 且是局部最小值）
        # 解决走廊中央无墙缺口、被策略1/2漏检的软门
        try:
            from scipy.ndimage import (distance_transform_edt as _dtedt,
                                       minimum_filter as _dtminf)
            _dt = _dtedt(free_mask).astype(float) * self.map_res  # 距墙米数
            _DT_LO, _DT_HI = 0.30, 0.75   # 窄通道宽度范围（单侧距墙）
            _in_range = (_dt >= _DT_LO) & (_dt <= _DT_HI) & free_mask
            # 沿 ix 方向的局部最小（横向通道）；沿 iy 方向的局部最小（纵向通道）
            _locmin_x = (_dt == _dtminf(_dt, size=(5, 1), mode='constant', cval=9999.0))
            _locmin_y = (_dt == _dtminf(_dt, size=(1, 5), mode='constant', cval=9999.0))
            _bn_mask = _in_range & (_locmin_x | _locmin_y)
            _bn_pts = [tuple(p) for p in np.argwhere(_bn_mask)]
            _bn_gap  = max(3, int(0.3 / self.map_res))
            _bn_done: set = set()
            _bn_cands = []
            for _bix, _biy in _bn_pts:
                if (_bix, _biy) in _bn_done:
                    continue
                # 收集邻近点 → 取 dt 最小者为代表
                _cluster = [(r, c) for (r, c) in _bn_pts
                            if abs(r - _bix) <= _bn_gap and abs(c - _biy) <= _bn_gap]
                if not _cluster:
                    _cluster = [(_bix, _biy)]
                _rep = min(_cluster, key=lambda rc: _dt[rc[0], rc[1]])
                for _rc in _cluster:
                    _bn_done.add(_rc)
                _brix, _briy = _rep
                # _locmin_y (size=(1,5)): 沿 Y 轴方向扫描得到局部最小值
                #   → 在 ±Y 方向被墙夹住 → 走廊沿 X 轴延伸 → 穿越方向 X → orientation="vertical"
                # _locmin_x (size=(5,1)): 沿 X 轴方向扫描得到局部最小值
                #   → 在 ±X 方向被墙夹住 → 走廊沿 Y 轴延伸 → 穿越方向 Y → orientation="horizontal"
                _ori = "vertical" if _locmin_y[_brix, _briy] else "horizontal"
                # waxis 测量瓶颈截面宽度（被夹住的方向 = 穿越方向的垂直方向）
                # vertical(X穿越, ±Y夹住) → 截面在 Y 方向 → waxis="y"
                # horizontal(Y穿越, ±X夹住) → 截面在 X 方向 → waxis="x"
                _waxis = "y" if _ori == "vertical" else "x"
                _wc = self._estimate_local_width_cells(
                    free_mask, _brix, _briy, _waxis, door_max_cells * 2)
                if door_min_cells <= _wc <= door_max_cells + 2:
                    _bn_cands.append((_brix, _briy, _ori, _wc))
            candidates += _bn_cands
            if _bn_cands:
                self.get_logger().debug(
                    f"[DOOR-BN] dist_field 瓶颈候选={len(_bn_cands)} "
                    f"(dt_range=[{_DT_LO},{_DT_HI}]m)"
                )
        except Exception as _bn_exc:
            self.get_logger().debug(f"[DOOR-BN] skip: {_bn_exc}")

        # ---- 去重（合并距离<min_gap的候选）----
        doors = []
        min_gap = max(3, int(0.5 / self.map_res))
        for ix, iy, ori, w in candidates:
            if not (0 <= ix < W and 0 <= iy < H):
                continue
            too_close = False
            for d in doors:
                if abs(d["ix"] - ix) <= min_gap and abs(d["iy"] - iy) <= min_gap:
                    too_close = True
                    break
            if too_close:
                continue
            wx, wy = self.grid_to_world(ix, iy)
            doors.append({
                "id": len(doors),
                "ix": ix,
                "iy": iy,
                "x": wx,
                "y": wy,
                "orientation": ori,
                "width_m": float(w * self.map_res),
                "room_a": -1,
                "room_b": -1,
            })
        return doors

    def update_room_door_segmentation(self, now) -> None:
        if (now - self.last_seg_update).nanoseconds * 1e-9 < self.seg_update_interval:
            return
        self.last_seg_update = now

        # ── 质量门控：EKF reset暂停期/TF断链/地图未就绪时跳过，避免垃圾地图产生错误标签 ──
        if getattr(self, '_ekf_reset_state', 'STABLE') != 'STABLE':
            return
        if getattr(self, '_tf_chain_paused', False):
            return
        if not getattr(self, '_slam_map_received', False):
            return

        # ★ 地图已知区域门控：known_ratio < 0.03 时跳过分割（避免全未知时运行消耗资源）
        # ★ 修复：原阈值 0.30 在 16m×16m 固定坐标系下需要探索 78m² 才触发分割，
        #         导致整个探索阶段 rooms_info=[] / doors_info=[]，状态机永久困在一个房间。
        #         新阈值 0.03 = 3%（约780格=7.8m²），扫描一小块房间后立即开始检测门/房间。
        total_cells = int(self.grid_shape[0]) * int(self.grid_shape[1])
        if total_cells > 0:
            # 已知单元格 = 非 unknown
            known_cells = int(np.sum((self.grid <= self.prob_free_th) | (self.grid >= self.prob_occ_th)))
            known_ratio = known_cells / total_cells
            _min_known = 0.03   # 3% 已知区域即可运行分割（约一间小屋的扫描量）
            if known_ratio < _min_known:
                _last_t = getattr(self, '_last_seg_skip_log_t', 0.0)
                # 每 15s 打印一次（避免日志刷屏）
                if (now.nanoseconds * 1e-9 - _last_t) > 15.0:
                    self._last_seg_skip_log_t = now.nanoseconds * 1e-9
                    self.get_logger().info(
                        f"[SEG-SKIP] known_ratio={known_ratio:.2f} < {_min_known}: "
                        f"地图已知区域不足，跳过房间/门分割 "
                        f"(已知={known_cells}格 / 总={total_cells}格)"
                    )
                return

        grid_seg = self.grid
        k = int(max(1, self.seg_median_ksize))
        if k > 1:
            if k % 2 == 0:
                k += 1
            grid_seg = cv2.medianBlur(self.grid, k)

        # ★ 分割预处理增强：形态学闭运算填补墙面缝隙 + 小连通域去噪
        # 闭运算：将距离迟 seg_morph_close_m 内的垃圾结合进墙面，补赟廣度小于 seg_morph_close_m的缝隙
        _close_cells = max(1, int(self.seg_morph_close_m / self.map_res))
        _close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * _close_cells + 1, 2 * _close_cells + 1))
        occ_raw = (grid_seg >= self.prob_occ_th).astype(np.uint8)
        occ_closed = cv2.morphologyEx(occ_raw, cv2.MORPH_CLOSE, _close_kernel)
        free_mask = (grid_seg <= self.prob_free_th) & (occ_closed == 0)
        occ_mask  = occ_closed.astype(bool)
        # 小连通域去噪：删除孤立的小自由区域（通常是短暂拆除的蓄意圆/噪声）
        if self.seg_min_free_cells > 0:
            from scipy.ndimage import label as _nlabel
            _fl, _fn = _nlabel(free_mask.astype(np.uint8))
            for _lid in range(1, _fn + 1):
                if int(np.sum(_fl == _lid)) < self.seg_min_free_cells:
                    free_mask[_fl == _lid] = False
        if np.count_nonzero(free_mask) < 20:
            # ★ 修复：分割失败时不立刻清空 rooms/doors overlay
            # 标记为降级状态，保留上一帧稳定结果，避免"远处墙体标记突然消失"
            _overlay_degraded = getattr(self, '_seg_overlay_degraded', False)
            if not _overlay_degraded:
                self._seg_overlay_degraded = True
                self.get_logger().warn(
                    f"[SEG-FAIL] free_mask={np.count_nonzero(free_mask)} < 20: "
                    f"分割失败，overlay标记为降级状态，保留上一帧结果（overlay_degraded=True）"
                )
            else:
                import time as _sf_t
                _sf_now = _sf_t.monotonic()
                if _sf_now - getattr(self, '_seg_fail_log_t', 0.0) > 5.0:
                    self._seg_fail_log_t = _sf_now
                    self.get_logger().debug(
                        f"[SEG-FAIL] overlay_degraded=True, kept last rooms={len(self.rooms_info)} "
                        f"doors={len(self.doors_info)} frontier_gaps={len(getattr(self,'frontier_gaps',[]))}"
                    )
            return  # 不清空，维持上一帧稳定 overlay

        doors = self.detect_doors(free_mask=free_mask, occ_mask=occ_mask)

        # ★ 墙体硬门控：detect_doors 之后立即检查 wallsegs（在里面被设置）
        # wallsegs == 0 → 没有可信墙段，生成的房间轮廓仅是自由空间外缘，会误导导航
        _wsegs_now = getattr(self, '_last_wall_seg_count', 0)
        if _wsegs_now == 0:
            import time as _mss_t
            _mss_now = _mss_t.monotonic()
            if _mss_now - getattr(self, '_map_struct_skip_log_t', 0.0) >= 5.0:
                self._map_struct_skip_log_t = _mss_now
                self.get_logger().info(
                    "[MAP-STRUCT-SKIP] wallsegs=0 → 跳过房间多边形/标签提取，"
                    "等待地图墙体成形（至少需要1段墙）"
                )
            return

        free_blocked = free_mask.copy()
        # ★[A6/A7] 分级硬切策略：
        #   - 只有达到 _door_hard_cut_frames（≥5帧）的稳定门才参与 free_blocked 硬切
        #   - 低稳定度门候选（soft_gap/stable_soft_gap）只做导航目标，不做硬切
        #   - 使用上一帧的 _door_confirmed_stable 作为硬切依据（本帧稳定缓存尚未更新）
        # 粗分模式（早期地图）：_door_confirmed_stable 为空 → 不切 → 单大连通域 → 正确
        # 细分模式（地图充分）：stable_confirmed 门参与切割 → 正式房间分割
        _stable_for_cut = [
            c["data"] for c in self._door_cand_cache.values()
            if c["hits"] >= getattr(self, '_door_hard_cut_frames', 5)
        ]
        # ★[A7] 渐进式模式判断：若没有高稳定度门，使用"粗分"模式（不切割）
        _seg_mode = "fine" if _stable_for_cut else "coarse"
        import time as _seg_t
        if _seg_t.monotonic() - getattr(self, '_seg_mode_log_t', 0.0) > 5.0:
            self._seg_mode_log_t = _seg_t.monotonic()
            _n_soft   = sum(1 for c in self._door_cand_cache.values() if c["hits"] < 3)
            _n_stable = sum(1 for c in self._door_cand_cache.values() if 3 <= c["hits"] < getattr(self, '_door_hard_cut_frames', 5))
            _n_hard   = len(_stable_for_cut)
            self.get_logger().info(
                f"[SEG-MODE] mode={_seg_mode} "
                f"soft_gap={_n_soft}(<3帧,仅可视化) "
                f"stable_soft={_n_stable}(3-4帧,可导航) "
                f"hard_cut={_n_hard}(≥5帧,参与分割) "
                f"→ {'正式房间切断激活' if _stable_for_cut else '粗分模式(单大连通域)'}"
            )
        for d in _stable_for_cut:
            _dix, _diy = d["ix"], d["iy"]
            _ori_d = str(d.get("orientation", ""))
            # 按门宽阻断整段线段，使连通域真正被切断
            _half_w = max(1, int(round(d.get("width_m", self.map_res) / (2.0 * self.map_res))))
            if _ori_d == "horizontal":  # 水平墙(y=const)，缺口沿ix方向展开
                for _off in range(-_half_w, _half_w + 1):
                    _bix = int(np.clip(_dix + _off, 0, self.grid_shape[0] - 1))
                    free_blocked[_bix, _diy] = False
            elif _ori_d == "vertical":  # 垂直墙(x=const)，缺口沿iy方向展开
                for _off in range(-_half_w, _half_w + 1):
                    _biy = int(np.clip(_diy + _off, 0, self.grid_shape[1] - 1))
                    free_blocked[_dix, _biy] = False
            else:
                for _di in range(-1, 2):
                    for _dj in range(-1, 2):
                        _bix2 = int(np.clip(_dix + _di, 0, self.grid_shape[0] - 1))
                        _biy2 = int(np.clip(_diy + _dj, 0, self.grid_shape[1] - 1))
                        free_blocked[_bix2, _biy2] = False

        labels, n = label(free_blocked.astype(np.uint8))
        # ★[A] 房间掩模正则化：形态学开/闭运算去噪 + 小连通域清除
        # 目的：平滑扫描噪声导致的锯齿边界，去除细长噪声岛，提高房间多边形质量
        _fb_reg = free_blocked.copy()
        _open_k = int(round(self.room_mask_open_kernel_m / self.map_res)) if self.room_mask_open_kernel_m > 0 else 0
        _close_k = int(round(self.room_mask_close_kernel_m / self.map_res)) if self.room_mask_close_kernel_m > 0 else 0
        if _open_k >= 1 or _close_k >= 1:
            try:
                from scipy.ndimage import binary_opening, binary_closing
                if _open_k >= 1:
                    _fb_reg = binary_opening(_fb_reg, iterations=_open_k)
                if _close_k >= 1:
                    _fb_reg = binary_closing(_fb_reg, iterations=_close_k)
            except Exception as _morph_e:
                self.get_logger().debug(f"[SEG-MORPH] 形态学正则化失败: {_morph_e}")
        # 小连通域清除（在正则化之后再清一次，去除孤立噪声块）
        _min_comp_cells = max(4, int(self.room_min_component_area_m2 / (self.map_res ** 2)))
        _fb_before = int(np.count_nonzero(_fb_reg))
        _fb_lbl, _fb_n = label(_fb_reg.astype(np.uint8))
        for _fbl in range(1, _fb_n + 1):
            if int(np.sum(_fb_lbl == _fbl)) < _min_comp_cells:
                _fb_reg[_fb_lbl == _fbl] = False
        _fb_after = int(np.count_nonzero(_fb_reg))
        # ★[B] [ROOM-MASK-REG] 日志：记录正则化前后自由格数变化
        import time as _rmr_t
        if _rmr_t.monotonic() - getattr(self, '_room_mask_reg_log_t', 0.0) > 5.0:
            self._room_mask_reg_log_t = _rmr_t.monotonic()
            self.get_logger().debug(
                f"[ROOM-MASK-REG] open_k={_open_k} close_k={_close_k} "
                f"free_before={_fb_before} free_after={_fb_after} "
                f"removed={_fb_before - _fb_after} comp_min={_min_comp_cells}"
            )
        # 正则化掩模覆盖但不影响 free_blocked（保留原始门切割语义）
        # 用正则化掩模重新标注：使用 _fb_reg 而不影响 free_blocked（二者均用于后续不同目的）
        labels, n = label(_fb_reg.astype(np.uint8))
        self.room_labels = labels.astype(np.int32)

        rooms = []
        min_cells = max(10, int(self.room_min_area / (self.map_res * self.map_res)))
        rid = 1
        for lid in range(1, n + 1):
            idx = np.argwhere(labels == lid)
            if idx.shape[0] < min_cells:
                continue
            cx = int(np.mean(idx[:, 0]))
            cy = int(np.mean(idx[:, 1]))
            wx, wy = self.grid_to_world(cx, cy)
            rooms.append({
                "id": rid,
                "label": int(lid),
                "cell_count": int(idx.shape[0]),
                "area_m2": float(idx.shape[0] * self.map_res * self.map_res),
                "center": [float(wx), float(wy)],
            })
            rid += 1

        # ★ [A2/ROOM-PERSIST] 持久化房间ID：跨帧匹配，尽量保留稳定ID
        # 匹配策略：中心点欧氏距离 + 面积相似度 greedy最近邻匹配
        _old_rooms_persist = getattr(self, '_persistent_rooms', {})  # {stable_id: {center, area}}
        _new_to_stable   = {}   # 新rooms索引 → 稳定ID
        _used_stable_ids = set()
        if _old_rooms_persist:
            for _ni, _nr in enumerate(rooms):
                _nr_cx, _nr_cy = float(_nr['center'][0]), float(_nr['center'][1])
                _nr_area = float(_nr['area_m2'])
                _best_sid = None
                _best_sc  = float('inf')
                for _sid, _sor in _old_rooms_persist.items():
                    if _sid in _used_stable_ids:
                        continue
                    _sor_cx, _sor_cy = float(_sor['center'][0]), float(_sor['center'][1])
                    _sor_area = float(_sor['area_m2'])
                    _dist_r = float(np.hypot(_nr_cx - _sor_cx, _nr_cy - _sor_cy))
                    _ar = max(_nr_area, _sor_area) / (min(_nr_area, _sor_area) + 0.01)
                    # 中心距 < 2.5m 且面积倍数 < 5 才考虑
                    if _dist_r < 2.5 and _ar < 5.0:
                        _sc = _dist_r + _ar * 0.4
                        if _sc < _best_sc:
                            _best_sc = _sc
                            _best_sid = _sid
                if _best_sid is not None:
                    _new_to_stable[_ni] = _best_sid
                    _used_stable_ids.add(_best_sid)
        _max_sid = max(_old_rooms_persist.keys(), default=0)
        _stable_rooms     = []
        _new_persist_map  = {}
        _new_cnt = 0; _reused_cnt = 0
        for _ni, _nr in enumerate(rooms):
            if _ni in _new_to_stable:
                _sid2 = _new_to_stable[_ni]
                _reused_cnt += 1
            else:
                _max_sid += 1
                _sid2 = _max_sid
                _new_cnt  += 1
            _nr2 = dict(_nr)
            _nr2['id']    = _sid2
            _nr2['label'] = _nr['label']   # 原始scipy label，供后续 room_labels 查找
            _stable_rooms.append(_nr2)
            _new_persist_map[_sid2] = {'center': _nr['center'], 'area_m2': _nr['area_m2']}
        self._persistent_rooms = _new_persist_map
        rooms = _stable_rooms
        import time as _rp_lt
        if _rp_lt.monotonic() - getattr(self, '_room_persist_log_t', 0.0) > 5.0:
            self._room_persist_log_t = _rp_lt.monotonic()
            self.get_logger().info(
                f"[ROOM-PERSIST] 持久ID分配: total={len(rooms)} "
                f"reused(稳定)={_reused_cnt} new_id={_new_cnt} "
                f"ids={[r['id'] for r in rooms]}"
            )

        label_to_room = {r["label"]: r["id"] for r in rooms}
        # rooms_dict：{room_id: room_dict}，供几何检查时查询房间面积
        _rooms_dict_lookup = {r["id"]: r for r in rooms}
        valid_doors = []
        for d in doors:
            ix, iy = d["ix"], d["iy"]
            neigh = set()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < self.grid_shape[0] and 0 <= ny < self.grid_shape[1]:
                        lid = int(labels[nx, ny])
                        if lid > 0 and lid in label_to_room:
                            neigh.add(label_to_room[lid])
            neigh_list = sorted(list(neigh))
            d["room_a"] = int(neigh_list[0]) if len(neigh_list) > 0 else -1
            d["room_b"] = int(neigh_list[1]) if len(neigh_list) > 1 else -1
            # ★ 修复：保留两类门：
            #   ① 已确认跨房间门（两侧均识别到不同房间，room_a ≠ room_b > 0）
            #   ② 出口探针门（一侧已知房间，另一侧未知 room_b=-1）—— 最重要的探索线索！
            #      原始过滤器要求 room_a>0 AND room_b>0 导致早期仅一个房间时 doors_info 永远为空，
            #      无人机永远无法获得穿越门洞的导航目标，困在初始房间。
            _is_cross = d["room_a"] > 0 and d["room_b"] > 0 and d["room_a"] != d["room_b"]
            _is_exit  = (d["room_a"] > 0 and d["room_b"] < 0) or (d["room_b"] > 0 and d["room_a"] < 0)
            if _is_cross or _is_exit:
                if _is_cross:
                    # ★[C/D] 几何有效性校验：confirmed_door 需通过宽度/走廊/门框检查
                    _geom_ok, _geom_reason, _geom_subtype = self._check_door_geometry(
                        d, free_mask, occ_mask, _rooms_dict_lookup)
                    if _geom_ok:
                        d["door_type"] = _geom_subtype   # 'confirmed_door' 或 'corridor_connector'
                        d["geom_reason"] = _geom_reason
                        valid_doors.append(d)
                    else:
                        self.get_logger().debug(
                            f"[DOOR-GEOM-REJECT] ix={ix},iy={iy} reason={_geom_reason}"
                        )
                else:
                    # frontier_gap：一侧未知，不做几何校验（探索性门候选），仅做宽度初筛
                    _w_m = float(d.get("width_m", 0.0))
                    if _w_m >= self.door_width_min_m:  # 最低宽度过滤
                        d["door_type"] = "frontier_gap"
                        d["geom_reason"] = "frontier_exit"
                        valid_doors.append(d)

        # ── [DOOR-CLASSIFY] 周期性门分类日志 ─────────────────────────────────────
        import time as _dc_mt
        _now_dc = _dc_mt.monotonic()
        if _now_dc - getattr(self, '_door_classify_log_t', 0.0) > 5.0:
            self._door_classify_log_t = _now_dc
            _n_confirmed = sum(1 for _dv in valid_doors if _dv.get("door_type") == "confirmed_door")
            _n_corridor  = sum(1 for _dv in valid_doors if _dv.get("door_type") == "corridor_connector")
            _n_frontier  = sum(1 for _dv in valid_doors if _dv.get("door_type") == "frontier_gap")
            self.get_logger().info(
                f"[DOOR-CLASSIFY] confirmed={_n_confirmed} corridor={_n_corridor} "
                f"frontier_gap={_n_frontier} "
                f"candidates={len(doors)} accepted={len(valid_doors)}"
            )

        self.rooms_info = rooms

        # ★ [A3/REGION-GRAPH] 轻量房间图增量更新
        # 节点: 房间ID  边: gap/door（含置信度）
        import time as _rg_t
        _rg_now = _rg_t.monotonic()
        _rg = getattr(self, '_region_graph', {'rooms': {}, 'edges': {}})
        _cur_room_ids = {r['id'] for r in rooms}
        # 更新/添加房间节点
        for _r in rooms:
            _rid3 = _r['id']
            if _rid3 not in _rg['rooms']:
                _rg['rooms'][_rid3] = {'center': _r['center'], 'area': _r['area_m2'],
                                       'neighbors': set(), 'visit_count': 0}
            else:
                _rg['rooms'][_rid3]['center'] = _r['center']
                _rg['rooms'][_rid3]['area']   = _r['area_m2']
        # 移除消失的房间节点
        for _rid3 in list(_rg['rooms'].keys()):
            if _rid3 not in _cur_room_ids:
                del _rg['rooms'][_rid3]
        # 更新/添加边（门/gap）
        _last_crossed = getattr(self, '_last_crossed_gap_pos', None)
        _cross_t_mono = getattr(self, '_last_crossed_gap_at', 0.0)
        for _d3 in valid_doors:
            _eid3 = f"{int(_d3['ix']//2)}_{int(_d3['iy']//2)}"   # 粗粒度key，允许小幅抖动
            _traversed3 = False
            if _last_crossed is not None:
                import math as _rg_m
                _traversed3 = (
                    _rg_m.hypot(_d3['x'] - _last_crossed[0], _d3['y'] - _last_crossed[1]) < 1.5
                    and _rg_now - _cross_t_mono < 120.0
                )
            if _eid3 not in _rg['edges']:
                _rg['edges'][_eid3] = {
                    'pos': (_d3['x'], _d3['y']),
                    'room_a': _d3['room_a'], 'room_b': _d3['room_b'],
                    'conf': 1, 'type': _d3.get('door_type', 'unknown'),
                    'width': _d3.get('width_m', 0.8), 'traversed': _traversed3,
                }
            else:
                _rg['edges'][_eid3]['conf']   = min(_rg['edges'][_eid3]['conf'] + 1, 30)
                _rg['edges'][_eid3]['room_a'] = _d3['room_a']
                _rg['edges'][_eid3]['room_b'] = _d3['room_b']
                if _traversed3:
                    _rg['edges'][_eid3]['traversed'] = True
            # 登记邻接
            for _ra3 in (_d3['room_a'], _d3['room_b']):
                if _ra3 > 0 and _ra3 in _rg['rooms']:
                    _rg['rooms'][_ra3]['neighbors'].add(_eid3)
        # 更新当前房间的 visit_count
        _cur_room_id_now = getattr(self, 'current_room_id', -1)
        if _cur_room_id_now > 0 and _cur_room_id_now in _rg['rooms']:
            _rg['rooms'][_cur_room_id_now]['visit_count'] = (
                _rg['rooms'][_cur_room_id_now].get('visit_count', 0) + 1)
        self._region_graph = _rg
        if _rg_now - getattr(self, '_rg_log_t', 0.0) > 8.0:
            self._rg_log_t = _rg_now
            _n_edges3   = len(_rg['edges'])
            _conf_edges = sum(1 for _e in _rg['edges'].values() if _e['conf'] >= 3)
            _trav_edges = sum(1 for _e in _rg['edges'].values() if _e.get('traversed', False))
            _room_info3 = [f"R{_ri}(nbr={len(_rv['neighbors'])})" for _ri, _rv in _rg['rooms'].items()]
            self.get_logger().info(
                f"[REGION-GRAPH] 节点={len(_rg['rooms'])} 边={_n_edges3} "
                f"confirmed(≥3)={_conf_edges} traversed={_trav_edges} "
                f"rooms={_room_info3}"
            )

        # ★ 分割成功时复位降级标记
        if getattr(self, '_seg_overlay_degraded', False):
            self._seg_overlay_degraded = False
            self.get_logger().info(
                f"[SEG-RECOVER] overlay_degraded复位: rooms={len(rooms)} doors_raw={len(doors)}"
            )

        # ── Phase9 房间/门质量冻结：地图恢复履止阈内不更新 rooms/doors ────────
        import time as _rq_mt
        _now_rq = _rq_mt.monotonic()
        _room_frozen = (
            getattr(self, '_map_recover_active', False)
            or _now_rq < self._map_pause_until
        )
        if _room_frozen:
            # 地图恢复期间保持上一版房间轮廓不更新
            if _now_rq - getattr(self, '_room_freeze_log_mono', 0.0) > 5.0:
                self._room_freeze_log_mono = _now_rq
                self.get_logger().debug(
                    f"[ROOM-FREEZE] 地图恢复期间房间/门冻结 "
                    f"recover={getattr(self,'_map_recover_active',False)} "
                    f"pause={max(0.0, self._map_pause_until - _now_rq):.1f}s"
                )
            # 退出，不执行后续的门候选更新
            return

        # ── Phase7: 门检测地图质量门控（降级模式）────────────────────────────────
        # 仅使用世界坐标 TTL 缓存中的历史门作为输出，避免噪声门污染导航。
        _cur_wsegs  = getattr(self, '_last_wall_seg_count', 0)
        _cur_occ    = int(np.count_nonzero(occ_mask))
        _door_map_quality_ok = (_cur_wsegs >= 4 and _cur_occ >= 50)
        if not _door_map_quality_ok:
            _deg_reason = (
                f"wallsegs={_cur_wsegs}<4" if _cur_wsegs < 4 else
                f"occ_cells={_cur_occ}<50"
            )
            import time as _dg_mt
            _now_dg = _dg_mt.monotonic()
            if _now_dg - getattr(self, '_door_degrade_log_t', 0.0) >= 5.0:
                self._door_degrade_log_t = _now_dg
                self.get_logger().debug(
                    f"[DOOR-DEGRADE] 门检测降级: {_deg_reason} "
                    f"— 跳过缓存更新，使用历史世界缓存门"
                )

        # ===== 门候选 TTL 稳定性缓存 =====
        # 目的：防止每帧分割噪声导致 doors_info 一帧有一帧没，
        #       需要连续 _door_confirm_frames 帧检测到的门才作为稳定门使用。
        _current_sigs: set = set()
        for _d in valid_doors:
            # 用网格坐标（粗粒度0.2m）作为门的签名——允许小幅位置抖动
            _sig = f"{int(_d['ix'] // 2)}_{int(_d['iy'] // 2)}"
            _current_sigs.add(_sig)
            # ★ Phase7: 降级模式下跳过新增/更新，避免噪声门污染缓存
            if not _door_map_quality_ok:
                continue
            if _sig not in self._door_cand_cache:
                self._door_cand_cache[_sig] = {"data": _d, "hits": 1, "misses": 0}
            else:
                _old_hits = self._door_cand_cache[_sig]["hits"]
                self._door_cand_cache[_sig]["hits"] = min(_old_hits + 1, 20)  # 上限20防溢出
                _new_hits = self._door_cand_cache[_sig]["hits"]
                self._door_cand_cache[_sig]["misses"] = 0
                self._door_cand_cache[_sig]["data"] = _d  # 更新最新数据
                # ★[B] DOOR-REASSOC：门的房间关联发生变化时记录（防抖）
                _new_assoc = (int(_d.get('room_a', -1)), int(_d.get('room_b', -1)))
                _prev_assoc = self._door_assoc_prev.get(_sig)
                if _prev_assoc is not None and _prev_assoc != _new_assoc:
                    import time as _da_tim
                    _da_now = _da_tim.monotonic()
                    _da_last = self._door_reassoc_last_t.get(_sig, 0.0)
                    if _da_now - _da_last > self.door_reassoc_hysteresis_sec:
                        self._door_reassoc_last_t[_sig] = _da_now
                        self.get_logger().info(
                            f"[DOOR-REASSOC] sig={_sig} "
                            f"room_ab {_prev_assoc}→{_new_assoc}"
                        )
                        self._log_event("DOOR-REASSOC",
                            f"sig={_sig} old_ab={_prev_assoc} new_ab={_new_assoc}")
                self._door_assoc_prev[_sig] = _new_assoc
                # ★[C] 记录首次达到 hard_cut 阈值（HARD-CUT-ENTER）
                # ★[F] 时序保护：只有地图质量足够且门类型为 confirmed_door 时才允许进入硬切
                if _old_hits < self._door_hard_cut_frames <= _new_hits:
                    _hc_dtype = _d.get("door_type", "unknown")
                    _hc_occ   = int(np.count_nonzero(occ_mask))
                    _hc_map_ok = (_hc_occ >= self.door_upgrade_min_map_occ
                                  and _hc_dtype == "confirmed_door")
                    if _hc_map_ok and _sig not in self._door_hard_confirmed_sigs:
                        self._door_hard_confirmed_sigs.add(_sig)
                        self.get_logger().warn(
                            f"[HARD-CUT-ENTER] sig={_sig} hits={_new_hits} "
                            f"dtype={_hc_dtype} occ={_hc_occ} "
                            f"(门已稳定达到硬切阈值, 将使用迟滞保护)"
                        )
                        self._log_event("HARD-CUT-ENTER", f"sig={_sig} hits={_new_hits} dtype={_hc_dtype}")
                    elif not _hc_map_ok:
                        self.get_logger().debug(
                            f"[HARD-CUT-BLOCK] sig={_sig} hits={_new_hits} "
                            f"dtype={_hc_dtype} occ={_hc_occ} "
                            f"min_occ={self.door_upgrade_min_map_occ} — 时序保护阻止硬切升级"
                        )
        # 未命中帧：增加 miss 计数，超容忍度则移除
        # ★[C] 硬切迟滞：已达到 hard_cut 阈值的门使用更高的移除阈值，防止短暂遮挡导致分割回退
        for _sig in list(self._door_cand_cache.keys()):
            if _sig not in _current_sigs:
                self._door_cand_cache[_sig]["misses"] += 1
                # 已硬切确认的门需要 _door_hard_cut_release_frames 次缺失才移除（迟滞保护）
                _release_thr = (
                    self._door_hard_cut_release_frames
                    if _sig in self._door_hard_confirmed_sigs
                    else self._door_miss_tolerance
                )
                if self._door_cand_cache[_sig]["misses"] > _release_thr:
                    _expired = self._door_cand_cache.pop(_sig)
                    _was_hard = _sig in self._door_hard_confirmed_sigs
                    self._door_hard_confirmed_sigs.discard(_sig)
                    if _was_hard:
                        self.get_logger().warn(
                            f"[HARD-CUT-EXIT] sig={_sig} hits={_expired['hits']} "
                            f"misses={_expired['misses']} reason=miss_exceeded"
                        )
                        self._log_event("HARD-CUT-EXIT", f"sig={_sig} hits={_expired['hits']} misses={_expired['misses']}")
                    else:
                        self.get_logger().debug(
                            f"[DOOR-STABLE] 门候选过期移除 sig={_sig} "
                            f"(hits={_expired['hits']} misses={_expired['misses']})"
                        )
        # 确认稳定门：hits >= 阈值
        self._door_confirmed_stable = [
            c["data"] for c in self._door_cand_cache.values()
            if c["hits"] >= self._door_confirm_frames
        ]

        # ★ Phase5: 世界坐标 TTL 门缓存更新
        import time as _wc_mt
        _now_wc = _wc_mt.monotonic()
        _wc_quant = 0.3  # 量化精度 0.3m，对应 3 个规划格
        # 用稳定门（或至少当前候选）更新世界坐标缓存
        _doors_for_wc = self._door_confirmed_stable if self._door_confirmed_stable else valid_doors
        _matched_keys = set()
        for _wd in _doors_for_wc:
            _wx = float(_wd.get('x', 0.0))
            _wy = float(_wd.get('y', 0.0))
            # 在缓存中搜索最近邻
            _best_key = None
            _best_dist = float('inf')
            for _key in self._door_world_cache:
                _kx = _key[0] * _wc_quant
                _ky = _key[1] * _wc_quant
                _dd = abs(_kx - _wx) + abs(_ky - _wy)
                if _dd < self._door_world_match_dist_m and _dd < _best_dist:
                    _best_dist = _dd
                    _best_key = _key
            if _best_key is not None:
                _matched_keys.add(_best_key)
                self._door_world_cache[_best_key]['door'] = _wd
                self._door_world_cache[_best_key]['last_seen'] = _now_wc
                self._door_world_cache[_best_key]['conf'] = min(
                    self._door_world_cache[_best_key]['conf'] + 1, 20)
            else:
                _wsig = (round(_wx / _wc_quant), round(_wy / _wc_quant))
                self._door_world_cache[_wsig] = {
                    'door': _wd, 'last_seen': _now_wc, 'conf': 1}
        # TTL 过期清理
        for _key in list(self._door_world_cache.keys()):
            if _now_wc - self._door_world_cache[_key]['last_seen'] > self._door_world_ttl_sec:
                self._door_world_cache.pop(_key)
        # 从世界缓存恢复的门列表（按最近见到时间排序）
        _wc_doors = [
            v['door'] for v in sorted(
                self._door_world_cache.values(),
                key=lambda z: z['last_seen'], reverse=True
            )
        ]

        # ── doors_info 策略：只允许 confirmed_door（两侧均为已知房间）入局 ───────────
        # ★ frontier_gap（room_b=-1）不进入 doors_info，避免把裂缝当成真实门
        # ★ corridor_connector 进入区域图边（E），但不作为导航目标（不是真实门洞）
        _confirmed_only_cur = [d for d in valid_doors if d.get("door_type") == "confirmed_door"]
        _corridor_only_cur  = [d for d in valid_doors if d.get("door_type") == "corridor_connector"]
        _frontier_gaps_cur   = [d for d in valid_doors if d.get("door_type") == "frontier_gap"]
        self.confirmed_doors = _confirmed_only_cur
        self.corridor_connectors = _corridor_only_cur  # ★[E] 走廊连接器列表
        # ★ frontier_gaps TTL缓存：质量差时保留上一帧有效结果（避免分割抖动导致导航候选消失）
        import time as _fg_t
        _now_fg = _fg_t.monotonic()
        _fg_ttl = 15.0   # 15s TTL
        if _frontier_gaps_cur:
            self.frontier_gaps = _frontier_gaps_cur
            self._frontier_gaps_cached = _frontier_gaps_cur
            self._frontier_gaps_cache_t = _now_fg
        else:
            _fg_age = _now_fg - getattr(self, '_frontier_gaps_cache_t', 0.0)
            if _fg_age < _fg_ttl and getattr(self, '_frontier_gaps_cached', []):
                self.frontier_gaps = self._frontier_gaps_cached
                _lt_fg = getattr(self, '_fg_ttl_log_t', 0.0)
                if _now_fg - _lt_fg > 5.0:
                    self._fg_ttl_log_t = _now_fg
                    self.get_logger().debug(
                        f"[FG-TTL] frontier_gaps空帧，使用缓存 {len(self._frontier_gaps_cached)}个 "
                        f"age={_fg_age:.1f}s < {_fg_ttl:.0f}s"
                    )
            else:
                self.frontier_gaps = []

        # ★[A6] 三层门分级：soft_gap / stable_soft_gap / confirmed_door
        # 使用当前 _door_cand_cache hits 值进行分级
        _hard_cut_th = getattr(self, '_door_hard_cut_frames', 5)
        _nav_th = self._door_confirm_frames  # 3
        _soft_g_list    = []  # hits 1-2：仅可视化
        _stable_sg_list = []  # hits 3-(hard_cut-1)：可导航，不硬切
        _hard_cut_list  = []  # hits ≥ hard_cut_th：参与确认分割（已在前面用于 free_blocked）
        for _cval in self._door_cand_cache.values():
            _cd = _cval["data"]
            _ch = _cval["hits"]
            if _ch < _nav_th:
                _soft_g_list.append(_cd)
            elif _ch < _hard_cut_th:
                _stable_sg_list.append(_cd)
            else:
                _hard_cut_list.append(_cd)
        self.soft_gaps        = _soft_g_list
        self.stable_soft_gaps = _stable_sg_list
        # ★[C] SOFT-GAP / STABLE-SOFT-GAP 过渡日志（计数变化时触发，不超过2s节流）
        import time as _sg_log_t
        _sg_now = _sg_log_t.monotonic()
        _sg_cnt     = len(_soft_g_list)
        _ssg_cnt    = len(_stable_sg_list)
        _hc_cnt     = len(_hard_cut_list)
        _sg_prev    = getattr(self, '_soft_gaps_prev_cnt', 0)
        _ssg_prev   = getattr(self, '_stable_soft_prev_cnt', 0)
        if (_sg_cnt != _sg_prev
                and _sg_now - getattr(self, '_soft_gap_log_t', 0.0) > 2.0):
            self._soft_gap_log_t = _sg_now
            _sg_sigs = [f"{c['ix']},{c['iy']}" for c in _soft_g_list[:4]]
            self.get_logger().info(
                f"[SOFT-GAP] 候选门数量变化 {_sg_prev}→{_sg_cnt} (hits<{_nav_th}仅可视化) "
                f"sigs={_sg_sigs}"
            )
            self._log_event("SOFT-GAP",
                f"cnt={_sg_prev}→{_sg_cnt}")
        if (_ssg_cnt != _ssg_prev
                and _sg_now - getattr(self, '_stable_sg_log_t', 0.0) > 1.5):
            self._stable_sg_log_t = _sg_now
            _ssg_sigs = [f"{c['ix']},{c['iy']}" for c in _stable_sg_list[:4]]
            self.get_logger().info(
                f"[STABLE-SOFT-GAP] 稳定软门数量变化 {_ssg_prev}→{_ssg_cnt} "
                f"(hits≥{_nav_th}<{_hard_cut_th}可导航) "
                f"hard_cut={_hc_cnt} sigs={_ssg_sigs}"
            )
            self._log_event("STABLE-SOFT-GAP",
                f"cnt={_ssg_prev}→{_ssg_cnt} hard_cut={_hc_cnt}")
        self._soft_gaps_prev_cnt  = _sg_cnt
        self._stable_soft_prev_cnt = _ssg_cnt
        # frontier_gaps 供导航使用：包含 stable_soft_gap（hits≥3）+ 原始 frontier_gap_cur 中尚未升级的
        # 目的：确保导航候选总保持最新（stable_sg 比 confirmed_stable 响应更快）
        _nav_door_list = _stable_sg_list + _hard_cut_list + _frontier_gaps_cur
        # 去重（按坐标粗粒度）
        _nav_seen = set()
        _nav_dedup = []
        for _nd in _nav_door_list:
            _nd_sig = f"{int(_nd.get('ix',0)//2)}_{int(_nd.get('iy',0)//2)}"
            if _nd_sig not in _nav_seen:
                _nav_seen.add(_nd_sig)
                _nav_dedup.append(_nd)
        if _nav_dedup:  # 只要有新候选就更新，否则保留 TTL 缓存
            self.frontier_gaps = _nav_dedup
            self._frontier_gaps_cached = _nav_dedup
            self._frontier_gaps_cache_t = _now_fg
        _confirmed_stable_only = [
            d for d in self._door_confirmed_stable
            if d.get("door_type") == "confirmed_door"
        ]
        if _confirmed_stable_only:
            self.doors_info = _confirmed_stable_only
        elif _confirmed_only_cur:
            self.doors_info = _confirmed_only_cur
        elif _wc_doors:
            # 世界缓存降级：只取 confirmed_door 类型
            _wc_confirmed = [d for d in _wc_doors if d.get("door_type") == "confirmed_door"]
            self.doors_info = _wc_confirmed
        # else: doors_info 保持上一帧结果不清空
        # [DOOR-CLASS] 日志每5s一次
        import time as _dc2_mt
        _now_dc2 = _dc2_mt.monotonic()
        if _now_dc2 - getattr(self, '_door_class2_log_t', 0.0) > 5.0:
            self._door_class2_log_t = _now_dc2
            self.get_logger().info(
                f"[DOOR-CLASS] doors_info={len(self.doors_info)} "
                f"confirmed_stable={len(_confirmed_stable_only)} "
                f"confirmed_cur={len(_confirmed_only_cur)} "
                f"frontier_gap={len(_frontier_gaps_cur)} "
                f"soft_gap={len(self.soft_gaps)}(仅可视) "
                f"stable_soft={len(self.stable_soft_gaps)}(可导航) "
                f"hard_cut={len(_hard_cut_list)}(参与分割) "
                f"seg_mode={_seg_mode} "
                f"raw={len(valid_doors)}"
            )

        # ★ Phase-6：计算房间轮廓多边形（用于可视化；每次分割完成后更新）
        try:
            self._room_polygons = self._extract_room_polygons(
                free_mask, self.room_labels, rooms
            )
        except Exception as _pe:
            self._room_polygons = {}
            self.get_logger().debug(f"[SEG] 轮廓多边形提取失败: {_pe}")

        # ★ 分割结果增强诊断日志（Phase-6：增加墙段/polygons/occ-free 统计）
        _n_rooms = len(rooms)
        # ★[C5] 房间计数迟滞稳定化：防止单帧遮挡导致 rooms 2→1 闪烁振荡
        # 维护最近 max(_rooms_confirm_frames, _rooms_drop_frames) 帧滑动窗口
        _rcbuf = self._seg_raw_n_rooms_buf
        _rcbuf.append(_n_rooms)
        _rc_win = max(self._rooms_confirm_frames, self._rooms_drop_frames)
        if len(_rcbuf) > _rc_win:
            _rcbuf.pop(0)
        _prev_stable = self._seg_stable_n_rooms
        # 升：新房间数连续 _rooms_confirm_frames 帧 → 接受升级
        if _n_rooms > _prev_stable:
            _consec_new = sum(1 for r in _rcbuf[-self._rooms_confirm_frames:] if r >= _n_rooms)
            if _consec_new >= self._rooms_confirm_frames:
                self._seg_stable_n_rooms = _n_rooms
        # 降：降低至少需要 _rooms_drop_frames 帧连续低于当前稳定值
        elif _n_rooms < _prev_stable:
            _consec_low = sum(1 for r in _rcbuf[-self._rooms_drop_frames:] if r <= _n_rooms)
            if _consec_low >= self._rooms_drop_frames:
                self._seg_stable_n_rooms = _n_rooms
        _stable_n_rooms = self._seg_stable_n_rooms
        import time as _rooms_t
        _seg_now = _rooms_t.monotonic()
        if _stable_n_rooms != _prev_stable:
            # 稳定房间数变化时立即记录（不节流）
            _direction = "升" if _stable_n_rooms > _prev_stable else "降"
            self.get_logger().warn(
                f"[ROOMS] ★ 房间数稳定值变化 {_prev_stable}→{_stable_n_rooms} ({_direction}) "
                f"raw={_n_rooms} buf={_rcbuf} "
                f"hard_cut={sum(1 for c in self._door_cand_cache.values() if c['hits'] >= self._door_hard_cut_frames)}"
            )
            self._log_event("ROOMS",
                f"stable={_prev_stable}→{_stable_n_rooms} raw={_n_rooms} direction={_direction}")
            self._seg_rooms_log_t = _seg_now
        elif _seg_now - self._seg_rooms_log_t > 10.0:
            self._seg_rooms_log_t = _seg_now
            self.get_logger().info(
                f"[ROOMS] raw={_n_rooms} stable={_stable_n_rooms} buf={_rcbuf[-3:]}"
            )
        _n_doors = len(self.doors_info)
        _fr_cells  = int(np.count_nonzero(free_mask))
        _occ_cells = int(np.count_nonzero(occ_mask))
        _unk_cells = total_cells - _fr_cells - _occ_cells
        _wsegs = getattr(self, '_last_wall_seg_count', 0)   # 上次 detect_doors 写入
        _poly_verts = {lid: len(pts) for lid, pts in self._room_polygons.items()}
        if _n_rooms == 0:
            self._log_event(
                "SEG-FAIL",
                f"房间=0 门=0 free={_fr_cells}格 occ={_occ_cells}格 unk={_unk_cells}格 "
                f"min_cells={max(10, int(self.room_min_area / (self.map_res**2)))} "
                f"wallsegs={_wsegs}"
            )
        else:
            _room_detail_parts = []
            _res2 = self.map_res * self.map_res
            _ox_w = self.bounds[0, 0]
            _oy_w = self.bounds[1, 0]
            for _ri, _r in enumerate(rooms):
                _lid = _r.get("label", _ri + 1)
                _rmask = (self.room_labels == _lid)
                _area_m2 = float(np.count_nonzero(_rmask)) * _res2
                _rixs = np.where(_rmask)[0]
                _riys = np.where(_rmask)[1]
                if len(_rixs) > 0:
                    _wx_min = _ox_w + float(_rixs.min()) * self.map_res
                    _wx_max = _ox_w + float(_rixs.max()) * self.map_res
                    _wy_min = _oy_w + float(_riys.min()) * self.map_res
                    _wy_max = _oy_w + float(_riys.max()) * self.map_res
                    _bbox_str = (f"x=[{_wx_min:.1f},{_wx_max:.1f}] "
                                 f"y=[{_wy_min:.1f},{_wy_max:.1f}]")
                else:
                    _bbox_str = "bbox=N/A"
                _nv = _poly_verts.get(_lid, 0)
                _room_detail_parts.append(
                    f"R{_lid}:{_area_m2:.1f}m²({_bbox_str})poly={_nv}顶"
                )
            self._log_event(
                "SEG-OK",
                f"房间={_n_rooms} 门={_n_doors}("
                f"稳定={len(self._door_confirmed_stable)} "
                f"候选={len(valid_doors)} 缓存={len(self._door_cand_cache)}) "
                f"frontier_gaps={len(getattr(self,'frontier_gaps',[]))} "
                f"raw_grid_occ={int(np.sum(self.grid >= self.prob_occ_th))} "
                f"seg_occ={_occ_cells} free={_fr_cells} unk={_unk_cells} "
                f"wallsegs={_wsegs} overlay_degraded={getattr(self,'_seg_overlay_degraded',False)} "
                f"| " + " ".join(_room_detail_parts)
            )

    # ================================================================
    # PIL字体辅助（全局缓存，支持CJK中文）
    # ================================================================
    _pil_font_cache: dict = {}

    @staticmethod
    def _get_pil_font(size: int):
        cache = TargetSearch._pil_font_cache
        if size in cache:
            return cache[size]
        cjk_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        font = None
        for p in cjk_paths:
            if Path(p).exists():
                try:
                    font = ImageFont.truetype(p, size)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
        cache[size] = font
        return font

    @staticmethod
    def _pil_text(draw: ImageDraw.ImageDraw, xy, text: str, font,
                  fill=(20, 20, 20), anchor="la"):
        """PIL文字绘制（RGB颜色，左上角anchor）"""
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)

    # ================================================================
    # Phase-6：房间轮廓多边形提取（基于 room_labels + free_mask）
    # ================================================================
    def _extract_room_polygons(
        self, free_mask: np.ndarray, room_labels: np.ndarray, rooms_info: list
    ) -> dict:
        """
        对每个房间连通域提取轮廓多边形。
        返回: {label_id: [(ix,iy), ...]}
          - 坐标为 planning-grid cell 索引
          - 已用 cv2.approxPolyDP 简化（ε = arc_length × 0.025）
        用于可视化：以折线绘制房间边界，比外包矩形更真实。
        ★ 新增：接触规划边框或地图质量不足时跳过该多边形（[ROOM-POLY-INVALID]）
        """
        import cv2 as _cv2_rp
        polygons: dict = {}
        if room_labels is None or len(rooms_info) == 0:
            return polygons
        _gi, _gj = self.grid_shape   # (ni, nj)
        _wsegs = getattr(self, '_last_wall_seg_count', 0)
        _occ   = int(np.count_nonzero(self.grid >= self.prob_occ_th))
        _map_recover_now = getattr(self, '_map_recover_active', False)
        _escape_now      = getattr(self, '_hover_scan_escape_active', False)
        for r in rooms_info:
            lid = int(r["label"])
            # ── 地图质量门控 ─────────────────────────────────────────────────
            if _wsegs < 20 or _occ < 80 or _map_recover_now or _escape_now:
                _reason = ("low_wallsegs" if _wsegs < 20 else
                           "low_occ" if _occ < 80 else
                           "map_recover" if _map_recover_now else "escape_active")
                self.get_logger().debug(
                    f"[ROOM-POLY-SKIP] room={lid} reason={_reason} "
                    f"wallsegs={_wsegs} occ={_occ}"
                )
                self.get_logger().debug(
                    f"[ROOM-POLY-INVALID] room={lid} reason={_reason} "
                    f"wallsegs={_wsegs} occ={_occ}"
                )
                continue
            mask = ((room_labels == lid) & free_mask).astype(np.uint8)
            contours, _ = _cv2_rp.findContours(
                mask.T,   # (W,H) → 转置为 (H,W)=图像坐标
                _cv2_rp.RETR_EXTERNAL,
                _cv2_rp.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            cnt = max(contours, key=_cv2_rp.contourArea)
            if len(cnt) < 3:
                continue
            # approxPolyDP 简化多边形（ε 约 1 格分辨率）
            _verts_raw = len(cnt)
            epsilon = max(1.0, 0.025 * float(_cv2_rp.arcLength(cnt, True)))
            approx = _cv2_rp.approxPolyDP(cnt, epsilon, True)
            _verts_simplified = len(approx)
            self.get_logger().debug(
                f"[ROOM-POLY-SIMPLIFY] room={lid} "
                f"verts_raw={_verts_raw} verts_simplified={_verts_simplified} "
                f"epsilon={epsilon:.1f}"
            )
            pts = []
            _touches_border = False
            for pt in approx.reshape(-1, 2):
                ix, iy = int(pt[0]), int(pt[1])
                # 检查是否接触规划边框（1格容差）
                if ix <= 1 or iy <= 1 or ix >= _gi - 2 or iy >= _gj - 2:
                    _touches_border = True
                pts.append((ix, iy))
            if _touches_border:
                self.get_logger().debug(
                    f"[ROOM-POLY-SKIP] room={lid} reason=touch_plan_border verts={len(pts)}"
                )
                self.get_logger().debug(
                    f"[ROOM-POLY-INVALID] room={lid} reason=touch_plan_border verts={len(pts)}"
                )
                continue
            if len(pts) >= 3:
                polygons[lid] = pts
                self.get_logger().debug(
                    f"[ROOM-POLY-OK] room={lid} verts={len(pts)} wall_support={_wsegs}"
                )
        return polygons

    def save_map_outputs(self, now) -> None:
        if (now - self.last_map_save).nanoseconds * 1e-9 < self.map_save_interval:
            return
        self.last_map_save = now

        occ_mask = self.grid >= self.prob_occ_th
        free_mask = self.grid <= self.prob_free_th
        unk_mask = (~occ_mask) & (~free_mask)

        np.save(self.map_output_dir / "occupancy_latest.npy", self.grid)

        _occ_count_npy = int(np.count_nonzero(occ_mask))
        _free_count_npy = int(np.count_nonzero(free_mask))
        _unk_count_npy = int(np.count_nonzero(unk_mask))

        state = {
            "timestamp_sec": float(now.nanoseconds * 1e-9),
            "resolution": float(self.map_res),
            "bounds": self.bounds.tolist(),
            "grid_shape": list(self.grid_shape),
            "plan_bounds_xy": self._plan_bounds_xy.tolist(),
            "plan_res": self._plan_res,
            "slam_native_meta": self._slam_native_meta,
            "counts": {
                "obstacle": _occ_count_npy,
                "free_explored": _free_count_npy,
                "unknown": _unk_count_npy,
            },
            "rooms": self.rooms_info,
            "doors": self.doors_info,
        }
        # ★[BUG-FIX] 自定义 JSON encoder：把 numpy 整数/浮点/bool 转为 Python 原生类型
        # 根因：rooms_info/doors_info 中含 numpy.int64 等，json.dump 无法序列化 → 节点崩溃
        # 修复：使用自定义 encoder + try/except，确保序列化失败时不崩溃节点
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        try:
            with open(self.map_output_dir / "map_state_latest.json", "w", encoding="utf-8") as f:
                json.dump(state, f, cls=_NumpyEncoder, ensure_ascii=False, indent=2)
        except Exception as _jse:
            self.get_logger().warn(
                f"[MAP-JSON-FAIL] map_state_latest.json 序列化失败 (节点继续运行): {_jse}"
            )

        # ── [MAP-STATUS] 周期性状态日志：map_pose可信度 / 地图冻结 / SLAM暂停 / 探索状态 ──
        import time as _mst_t
        if _mst_t.monotonic() - getattr(self, '_map_status_log_t', 0.0) > 5.0:
            self._map_status_log_t = _mst_t.monotonic()
            _mps_valid  = getattr(self, '_map_pose_valid', '?')
            _mps_stale  = getattr(self, '_map_pose_stale', '?')
            _mps_src    = getattr(self, '_map_pose_source', '?')
            _frozen     = getattr(self, '_map_frozen', '?')
            _slam_pause = getattr(self, '_slam_pause_active', False)
            _expl       = getattr(self, 'explore_state', '?')
            _esc_fail   = getattr(self, '_hover_scan_escape_fail_count', 0)
            _map_rec    = getattr(self, '_map_recover_active', False)
            _sh_until   = getattr(self, '_hscan_safe_hold_until', 0.0)
            _sh_remain  = max(0.0, _sh_until - _mst_t.monotonic())
            # ★ 增强：实时显示建图门控结果（是否允许积分、原因）
            _allow_int, _int_reason = self._allow_slam_integrate_now()
            _door_cnt = len(getattr(self, 'doors_info', []))
            _gap_cnt  = len(getattr(self, '_frontier_gaps_cached', []))
            _goal_type = getattr(self, 'ssg_goal_type', 'none')
            self.get_logger().info(
                f"[MAP-STATUS] pose_valid={_mps_valid} stale={_mps_stale} src={_mps_src} | "
                f"map_frozen={_frozen} integrate={_allow_int}({_int_reason}) | "
                f"explore={_expl} goal_type={_goal_type} | "
                f"doors={_door_cnt} gaps={_gap_cnt} esc_fail={_esc_fail} "
                f"map_recover={_map_rec} sh_remain={_sh_remain:.0f}s | "
                f"occ={_occ_count_npy} free={_free_count_npy}"
            )

        # ── 地图导出一致性检查 ────────────────────────────────────────────────
        import time as _mec_t
        if _mec_t.monotonic() - getattr(self, '_map_export_check_t', 0.0) > 10.0:
            self._map_export_check_t = _mec_t.monotonic()
            _json_occ = state["counts"]["obstacle"]
            _json_free = state["counts"]["free_explored"]
            _json_unk = state["counts"]["unknown"]
            _total = _json_occ + _json_free + _json_unk
            _grid_total = int(self.grid_shape[0]) * int(self.grid_shape[1])
            _consistent = (_total == _grid_total)
            if not _consistent:
                self.get_logger().warn(
                    f"[MAP-EXPORT-CHECK] inconsistent: occ={_json_occ} free={_json_free} "
                    f"unk={_json_unk} sum={_total} != grid_total={_grid_total}"
                )
            else:
                self.get_logger().debug(
                    f"[MAP-EXPORT-CHECK] occ={_json_occ} free={_json_free} "
                    f"unk={_json_unk} total={_total} consistent=1"
                )

        # ── SLAM 原生地图调试 PNG（map_native_debug.png）──────────────────────
        # 直接从 SLAM 原始数据渲染，方便与 map_fixed（业务地图）对比，
        # 排查问题到底在 SLAM 层还是在投影层。
        try:
            _nm = self._slam_native_grid
            _meta = self._slam_native_meta
            if _nm is not None and _meta:
                _H, _W = _nm.shape
                _native_img = np.full((_H, _W, 3), (180, 180, 180), dtype=np.uint8)
                _native_img[_nm < 0] = (180, 180, 180)    # unknown = 灰
                _native_img[(_nm >= 0) & (_nm < 50)] = (144, 220, 144)  # free = 绿
                _native_img[_nm >= 50] = (200, 40, 40)    # occ = 红
                with open(str(self.map_output_dir / "map_native_debug.png"), "wb") as _pf:
                    import io as _io
                    pil_n = PILImage.fromarray(_native_img)
                    _bio = _io.BytesIO()
                    pil_n.save(_bio, format="PNG")
                    _pf.write(_bio.getvalue())
        except Exception as _de:
            pass  # 调试图失败不影响主流程

        # ==============================================================
        # ---- 占据栅格地图可视化（PIL实现，支持中文标签）----
        # 颜色约定：
        #   未探索（unknown）= 浅灰  #B4B4B4
        #   已探索自由空间   = 绿色  #6EC96E
        #   障碍/墙体        = 红色  #D03030  (更直观：红=墙)
        #   门               = 蓝色  #1E6EF0
        #   无人机当前位置   = 橙色  #FF8C00（实心圆点）
        #   Frontier探索目标 = 青色  #00CED1
        # ==============================================================
        COLOR_UNK   = (180, 180, 180)   # 未知灰
        COLOR_FREE  = (144, 220, 144)   # 已探索绿
        COLOR_WALL  = (200,  40,  40)   # 墙体红
        COLOR_DOOR  = ( 30, 100, 220)   # 确认门蓝
        COLOR_CORR  = ( 30, 180, 180)   # 走廊连接器青色
        COLOR_DRONE = (255, 140,   0)   # 无人机橙
        COLOR_FRONT = (  0, 200, 200)   # Frontier青
        COLOR_VP    = (160,  32, 240)   # ★[H3] SSG viewpoint 普通扫描点 紫色
        COLOR_VP_DOOR = (255, 160,   0)  # ★[H3] SSG viewpoint 门/gap相关目标 琥珀色
        COLOR_VTG   = (220,  20,  60)   # ★[H3] visual_target_goal 世界坐标目标 深红色

        # 1. 构建基础颜色数组 (W, H, 3) → 转置为图像坐标 (H, W, 3)
        vis = np.full((self.grid_shape[0], self.grid_shape[1], 3), COLOR_UNK, dtype=np.uint8)
        vis[free_mask] = COLOR_FREE
        vis[occ_mask]  = COLOR_WALL

        # 1.5 Belief 热力叠加（高belief区域叠加蓝色tint）
        belief_norm = np.clip(self.belief_grid, 0.0, 1.0)
        # 仅在 free 或 unknown 区域叠加（不覆盖墙体）
        non_occ = ~occ_mask
        alpha_b = belief_norm[non_occ] * 0.5  # 最大50%透明度
        blue = np.array([30, 100, 240], dtype=np.float32)
        vis_f = vis[non_occ].astype(np.float32)
        vis[non_occ] = np.clip(
            vis_f * (1.0 - alpha_b[:, None]) + blue[None, :] * alpha_b[:, None],
            0, 255
        ).astype(np.uint8)

        # 2. 房间分区（半透明叠加色）
        room_palette = [
            (180, 230, 180), (180, 200, 240), (240, 210, 180),
            (220, 180, 230), (200, 240, 240), (240, 240, 180),
        ]
        for r in self.rooms_info:
            lid = int(r["label"])
            mask = self.room_labels == lid
            base_color = np.array(room_palette[r["id"] % len(room_palette)], dtype=np.float32)
            vis[mask & free_mask] = (
                0.6 * vis[mask & free_mask].astype(np.float32) +
                0.4 * base_color
            ).astype(np.uint8)

# 3. Phase-6：房间真实轮廓边框（在 numpy vis 层描点，覆盖在房间半透明色之上）
        # 用 _room_polygons（来自 _extract_room_polygons，连通域边界多边形）
        # 替代原「外包矩形」策略——更真实、适合非矩形/迷宫结构
        COLOR_ROOM_BOX = (200, 120, 0)  # 保留定义（图例/legend 仍使用该名称）
        _free_cnt = int(np.sum(free_mask))
        _occ_cnt  = int(np.sum(occ_mask))
        _POLY_COLORS = [
            (50, 170, 50),   # 绿色
            (50, 50, 200),   # 蓝色
            (180, 90, 0),    # 橙色
            (150, 0, 150),   # 紫色
            (0,  150, 150),  # 青色
            (150, 150, 0),   # 黄绿
        ]
        # 在 vis 数组中描画多边形轮廓线段（1格宽橙/彩边框）
        for _poly_i, (_lid_p, _pts) in enumerate(getattr(self, '_room_polygons', {}).items()):
            _pc = _POLY_COLORS[_poly_i % len(_POLY_COLORS)]
            # 连线各顶点形成闭合轮廓
            for _k in range(len(_pts)):
                _p0 = _pts[_k]
                _p1 = _pts[(_k + 1) % len(_pts)]
                # Bresenham 线段（仅 Manhattan 方向，足够 polygon 边的近似）
                _x0, _y0 = int(_p0[0]), int(_p0[1])
                _x1, _y1 = int(_p1[0]), int(_p1[1])
                _dx, _dy = abs(_x1 - _x0), abs(_y1 - _y0)
                _sx, _sy = (1 if _x1 >= _x0 else -1), (1 if _y1 >= _y0 else -1)
                _err = _dx - _dy
                _xi, _yi = _x0, _y0
                for _ in range(max(_dx, _dy) + 1):
                    if 0 <= _xi < self.grid_shape[0] and 0 <= _yi < self.grid_shape[1]:
                        vis[_xi, _yi] = _pc
                    if _xi == _x1 and _yi == _y1:
                        break
                    _e2 = 2 * _err
                    if _e2 > -_dy:
                        _err -= _dy
                        _xi  += _sx
                    if _e2 < _dx:
                        _err += _dx
                        _yi  += _sy
        # 始终计算 rect（兼容旧接口，不再用于绘制）
        rect = self.fit_room_rectangle(free_mask, occ_mask)

        # 5. ★[H3] 下一扫描目标点（SSG viewpoint，按目标类型区分颜色）
        # - 普通coverage/frontier → 紫色 COLOR_VP
        # - 门/gap相关 → 琥珀色 COLOR_VP_DOOR
        # - visual_target_world → 深红色 COLOR_VTG
        if self.ssg_viewpoint_goal is not None:
            _vp_gtype = getattr(self, 'ssg_goal_type', 'none')
            _DOOR_TYPES = ('frontier_gap_probe','frontier_gap_pre','frontier_gap_center',
                           'frontier_gap_post','confirmed_door')
            if _vp_gtype == 'visual_target_world':
                _vp_color = COLOR_VTG
            elif _vp_gtype in _DOOR_TYPES:
                _vp_color = COLOR_VP_DOOR
            else:
                _vp_color = COLOR_VP
            vix, viy = self.world_to_grid(float(self.ssg_viewpoint_goal[0]), float(self.ssg_viewpoint_goal[1]))
            for dxi in range(-3, 4):
                for dyi in range(-3, 4):
                    if abs(dxi) == abs(dyi) or dxi == 0 or dyi == 0:  # 星形
                        nx, ny = vix + dxi, viy + dyi
                        if 0 <= nx < self.grid_shape[0] and 0 <= ny < self.grid_shape[1]:
                            vis[nx, ny] = _vp_color

        # 5c. ★[H3] 视觉目标世界坐标估计点（_tgt_world_est_xy，深红色小菱形4×4）
        # 语义完全不同于 SSG viewpoint：这是 OpenCV 检测到的目标在地面的投影估计
        _twe_vis = getattr(self, '_tgt_world_est_xy', None)
        _twe_vis_conf = float(getattr(self, '_tgt_world_est_conf', 0.0))
        if _twe_vis is not None and _twe_vis_conf >= 0.25:
            _twx, _twy = self.world_to_grid(float(_twe_vis[0]), float(_twe_vis[1]))
            # 菱形（|dxi|+|dyi|<=2）
            for dxi in range(-3, 4):
                for dyi in range(-3, 4):
                    if abs(dxi) + abs(dyi) <= 2:
                        nx, ny = _twx + dxi, _twy + dyi
                        if 0 <= nx < self.grid_shape[0] and 0 <= ny < self.grid_shape[1]:
                            vis[nx, ny] = COLOR_VTG

        # 5b. Frontier目标（青色，兼容旧接口）
        if self.frontier_goal is not None:
            gix, giy = self.world_to_grid(float(self.frontier_goal[0]), float(self.frontier_goal[1]))
            for dxi in range(-2, 3):
                for dyi in range(-2, 3):
                    nx, ny = gix + dxi, giy + dyi
                    if 0 <= nx < self.grid_shape[0] and 0 <= ny < self.grid_shape[1]:
                        vis[nx, ny] = COLOR_FRONT

        # 6. 无人机位置（三色区分）：
        #   tf_map_base(直接TF) = 实心橙  COLOR_DRONE
        #   tf_map_odom_plus_odom(推算) = 琥珀色 (220,150,0)
        #   stale/cache = 灰色  (160,160,160)
        #   planar_invalid = 红色叉  (200,0,0)
        drone_ix, drone_iy = -1, -1
        _mp_valid_vis  = getattr(self, '_map_pose_valid', False)
        _mp_stale_vis  = getattr(self, '_map_pose_stale', True)
        _mp_src_vis    = getattr(self, '_map_pose_source', 'invalid')
        _planar_ok_vis = getattr(self, '_planar_pose_valid', True)
        if (self.pose is not None or self._odom_received) and _mp_valid_vis:
            # ★ 用 plan frame 坐标在 plan 网格上定位无人机（_plan_px/_plan_py）
            drone_ix, drone_iy = self.world_to_grid(self._plan_px, self._plan_py)
            if not _planar_ok_vis:
                _draw_color = (200, 0, 0)           # 红色 = 姿态失效
            elif _mp_stale_vis:
                _draw_color = (160, 160, 160)       # 灰色 = stale 缓存
            elif _mp_src_vis != 'tf_map_base':
                _draw_color = (220, 150, 0)         # 琥珀色 = fallback 推算
            else:
                _draw_color = COLOR_DRONE           # 橙色 = tf_map_base 可信
            for dxi in range(-2, 3):
                for dyi in range(-2, 3):
                    if abs(dxi) + abs(dyi) <= 3:
                        nx, ny = drone_ix + dxi, drone_iy + dyi
                        if 0 <= nx < self.grid_shape[0] and 0 <= ny < self.grid_shape[1]:
                            vis[nx, ny] = _draw_color

        # 7. 变换为图像坐标（x→右列, y→上行; flipud使y=0在底部）
        vis_img = np.flipud(np.transpose(vis, (1, 0, 2)))  # shape: (H_cells, W_cells, 3)
        map_rows, map_cols = vis_img.shape[:2]  # H格数, W格数

        # 8. 自适应缩放至固定画布
        #    画布布局：左侧120px图例，右侧地图区，底部30px坐标标注，顶部30px信息栏
        CANVAS_W, CANVAS_H = 900, 600
        LEGEND_W = 130
        TOP_BAR  = 35
        BOT_BAR  = 52
        avail_w = CANVAS_W - LEGEND_W - 10
        avail_h = CANVAS_H - TOP_BAR - BOT_BAR
        scale_w = max(1, avail_w // max(map_cols, 1))
        scale_h = max(1, avail_h // max(map_rows, 1))
        scale = max(3, min(scale_w, scale_h, 14))  # 3~14px/格

        map_px_w = map_cols * scale
        map_px_h = map_rows * scale
        # 居中放置地图
        map_off_x = LEGEND_W + max(0, (avail_w - map_px_w) // 2)
        map_off_y = TOP_BAR  + max(0, (avail_h - map_px_h) // 2)

        # 9. 用PIL创建画布（白底）
        canvas_pil = PILImage.new("RGB", (CANVAS_W, CANVAS_H), (248, 248, 248))
        # 放大地图並贴入
        vis_pil = PILImage.fromarray(vis_img, mode="RGB")
        vis_pil = vis_pil.resize((map_px_w, map_px_h), PILImage.NEAREST)
        canvas_pil.paste(vis_pil, (map_off_x, map_off_y))
        draw = ImageDraw.Draw(canvas_pil)

        # 10. 绘制地图边框
        draw.rectangle(
            [map_off_x, map_off_y, map_off_x + map_px_w, map_off_y + map_px_h],
            outline=(80, 80, 80), width=2
        )

        # 11. Phase-6：绘制房间真实轮廓多边形（PIL层，加粗有色边框）
        # 坐标转换：grid cell (ix,iy) → PIL canvas pixel
        def gx_to_px(gx): return map_off_x + gx * scale
        def gy_to_py(gy): return map_off_y + (map_rows - 1 - gy) * scale
        _POLY_COLORS_PIL = [
            (30, 160, 30),   # 绿
            (30, 30, 200),   # 蓝
            (180, 80, 0),    # 橙
            (150, 0, 150),   # 紫
            (0, 150, 150),   # 青
            (140, 130, 0),   # 黄绿
        ]
        for _poly_i2, (_lid_p2, _pts2) in enumerate(getattr(self, '_room_polygons', {}).items()):
            if len(_pts2) < 2:
                continue
            _pc2 = _POLY_COLORS_PIL[_poly_i2 % len(_POLY_COLORS_PIL)]
            # 将多边形顶点转为 PIL 像素坐标
            _pil_pts = [
                (gx_to_px(int(p[0])), gy_to_py(int(p[1])))
                for p in _pts2
            ]
            # 闭合多边形轮廓线（不填充，只画边）
            if len(_pil_pts) >= 3:
                draw.polygon(_pil_pts, outline=_pc2, fill=None)
                # 加粗：再偏移1px重复画
                _pil_pts_w = [(x+1, y) for x, y in _pil_pts]
                draw.polygon(_pil_pts_w, outline=_pc2, fill=None)
            # 在第一个顶点附近标注房间标签
            _lx, _ly = _pil_pts[0]
            font_poly = self._get_pil_font(11)
            draw.text((_lx + 3, _ly - 12), f"R{_lid_p2}", font=font_poly,
                      fill=_pc2)

        # 12. 坐标刻度线和标签（每1m一条）
        font_tick = self._get_pil_font(11)
        for xi in range(self.grid_shape[0]):
            wx_m = float(self.bounds[0, 0] + xi * self.map_res)
            if abs(wx_m - round(wx_m)) < self.map_res * 0.6:
                col_px = map_off_x + xi * scale
                if map_off_x <= col_px <= map_off_x + map_px_w:
                    draw.line([(col_px, map_off_y), (col_px, map_off_y + map_px_h)],
                              fill=(160, 160, 160), width=1)
                    draw.text((col_px + 2, map_off_y + map_px_h + 3),
                              f"{wx_m:.0f}m", font=font_tick, fill=(60, 60, 60))
        for yi in range(self.grid_shape[1]):
            wy_m = float(self.bounds[1, 0] + yi * self.map_res)
            if abs(wy_m - round(wy_m)) < self.map_res * 0.6:
                row_px = map_off_y + (map_rows - 1 - yi) * scale
                if map_off_y <= row_px <= map_off_y + map_px_h:
                    draw.line([(map_off_x, row_px), (map_off_x + map_px_w, row_px)],
                              fill=(160, 160, 160), width=1)
                    draw.text((map_off_x - 28, row_px - 7),
                              f"{wy_m:.0f}m", font=font_tick, fill=(60, 60, 60))

        # 13. 门标签（在门位置标注"门"字，按类型着色）
        font_door = self._get_pil_font(13)
        for d in self.doors_info:
            ix_d, iy_d = int(d["ix"]), int(d["iy"])
            col_px = map_off_x + ix_d * scale
            row_px = map_off_y + (map_rows - 1 - iy_d) * scale
            _dtype = d.get("door_type", "confirmed_door")
            _dc = COLOR_CORR if _dtype == "corridor_connector" else COLOR_DOOR
            draw.ellipse([(col_px - 5, row_px - 5), (col_px + 5, row_px + 5)],
                         fill=_dc, outline=(0, 0, 100), width=1)
            _dlabel = "廊" if _dtype == "corridor_connector" else f"门{d['id']+1}"
            draw.text((col_px + 6, row_px - 8), _dlabel, font=font_door,
                      fill=(0, 60, 180))
        # 13a. 走廊连接器（corridor_connector）单独显示（青色圆）
        for d in getattr(self, 'corridor_connectors', []):
            ix_d, iy_d = int(d["ix"]), int(d["iy"])
            col_px = map_off_x + ix_d * scale
            row_px = map_off_y + (map_rows - 1 - iy_d) * scale
            draw.ellipse([(col_px - 4, row_px - 4), (col_px + 4, row_px + 4)],
                         fill=COLOR_CORR, outline=(0, 100, 100), width=1)
            draw.text((col_px + 5, row_px - 7), "廊", font=font_door,
                      fill=(0, 120, 130))

        # 13b. 无人机 map pose 标注（PIL层）—— 四层状态颜色区分
        #   tf_map_base = 实心橙圈（最可信）
        #   tf_map_odom_plus_odom = 空心琥珀圈 + FBK 标记（推算位置）
        #   stale/cache = 灰圈 + STALE 文字
        #   planar_invalid = 红叉（机体倾翻）
        _mp_valid_v  = getattr(self, '_map_pose_valid', False)
        _mp_stale_v  = getattr(self, '_map_pose_stale', True)
        _mp_src_v    = getattr(self, '_map_pose_source', '?')
        _planar_ok_v = getattr(self, '_planar_pose_valid', True)
        _planar_rsn  = getattr(self, '_planar_pose_invalid_reason', '?')
        if drone_ix >= 0 and drone_iy >= 0 and _mp_valid_v:
            _d_col = map_off_x + drone_ix * scale
            _d_row = map_off_y + (map_rows - 1 - drone_iy) * scale
            if not _planar_ok_v:
                # 姿态失效：红色叉 + TILT 文字（机体倾翻，2D位置不可信）
                draw.line([(_d_col - 9, _d_row - 9), (_d_col + 9, _d_row + 9)],
                          fill=(220, 0, 0), width=3)
                draw.line([(_d_col + 9, _d_row - 9), (_d_col - 9, _d_row + 9)],
                          fill=(220, 0, 0), width=3)
                draw.text((_d_col + 11, _d_row - 10), f"TILT/{_planar_rsn[:10]}",
                          font=font_door, fill=(200, 0, 0))
            elif _mp_stale_v:
                # stale 缓存：灰色圆圈 + STALE 文字
                draw.ellipse([(_d_col - 8, _d_row - 8), (_d_col + 8, _d_row + 8)],
                             outline=(120, 120, 120), width=2)
                draw.text((_d_col + 10, _d_row - 10), f"STALE/{_mp_src_v}",
                          font=font_door, fill=(120, 120, 120))
            elif _mp_src_v not in ('tf_map_base',):
                # fallback推算（如 tf_map_odom_plus_odom）：空心琥珀圈 + FBK 标记
                draw.ellipse([(_d_col - 8, _d_row - 8), (_d_col + 8, _d_row + 8)],
                             outline=(210, 140, 0), width=2)
                draw.text((_d_col + 10, _d_row - 10),
                          f"FBK/{_mp_src_v[:12]}",
                          font=font_door, fill=(160, 100, 0))
            else:
                # tf_map_base：实心橙圈（最可信的直接TF位置）
                draw.ellipse([(_d_col - 8, _d_row - 8), (_d_col + 8, _d_row + 8)],
                             outline=(255, 140, 0), width=2)
                draw.text((_d_col + 10, _d_row - 10), f"/tf_map_base",
                          font=font_door, fill=(180, 80, 0))

        # 14. 顶部信息栏
        font_info = self._get_pil_font(13)
        n_free = int(np.count_nonzero(free_mask))
        n_occ  = int(np.count_nonzero(occ_mask))
        n_unk  = int(np.count_nonzero(unk_mask))
        total  = max(1, self.grid_shape[0] * self.grid_shape[1])
        pct_free = 100.0 * n_free / total
        t_sec = now.nanoseconds * 1e-9
        info1 = f"栅格地图  分辨率={self.map_res}m  {self.grid_shape[0]}×{self.grid_shape[1]}格  缩放={scale}×"
        info2 = (f"探索{pct_free:.1f}%  自由={n_free}格  障碍={n_occ}格  未知={n_unk}格  "
                 f"房间={len(self.rooms_info)}个  门={len(self.doors_info)}个  t={t_sec:.0f}s")
        draw.rectangle([(0, 0), (CANVAS_W, TOP_BAR - 2)], fill=(230, 235, 245))
        draw.text((6, 4),  info1, font=font_info, fill=(30, 30, 30))
        draw.text((6, 18), info2, font=font_info, fill=(60, 60, 60))

        # 15. 图例区（左侧 0~LEGEND_W px）
        font_leg  = self._get_pil_font(13)
        font_leg_title = self._get_pil_font(14)
        draw.rectangle([(0, TOP_BAR), (LEGEND_W - 5, CANVAS_H)], fill=(240, 240, 248))
        draw.text((6, TOP_BAR + 6), "图例", font=font_leg_title, fill=(20, 20, 20))
        # ★[H3] 图例语义统一：严格区分 SSG扫描点 / 门目标点 / 视觉目标世界坐标
        _vp_goal_type_now = getattr(self, 'ssg_goal_type', 'none')
        _vp_label = (
            f"SSG扫描目标({_vp_goal_type_now[:10]})"
            if self.ssg_viewpoint_goal is not None else "SSG扫描目标(无)"
        )
        leg_entries = [
            ("未探索区域", COLOR_UNK),
            ("已探索空间", COLOR_FREE),
            ("SLAM障碍(真实)", COLOR_WALL),
            ("房间真实轮廓", (30, 160, 30)),
            ("门/通道(确认)", COLOR_DOOR),
            ("走廊连接器", COLOR_CORR),
            ("稳定软门(可导航)", (255, 140, 0)),
            ("早期软门(仅视觉)", (255, 220, 0)),
            ("无人机", COLOR_DRONE),
            ("探索目标(Frontier)", COLOR_FRONT),
            (_vp_label + "(普通)", COLOR_VP),
            ("SSG目标★门/gap", COLOR_VP_DOOR),   # ★[H3] 门/gap类型目标点 琥珀色
            ("视觉目标世界坐标◆", COLOR_VTG),      # ★[H3] visual_target_goal 深红色
            ("Carrot跟踪点", (0, 230, 80)),
        ]
        ly = TOP_BAR + 28
        for label_cn, lc in leg_entries:
            draw.rectangle([(6, ly), (20, ly + 12)], fill=lc, outline=(80, 80, 80), width=1)
            draw.text((24, ly), label_cn, font=font_leg, fill=(20, 20, 20))
            ly += 20

        # 16. 比例尺（地图区右下角）
        bar_len_px = int(1.0 / self.map_res) * scale  # 1m对应像素数
        bar_x0 = map_off_x + map_px_w - bar_len_px - 10
        bar_y0 = map_off_y + map_px_h + 8
        draw.line([(bar_x0, bar_y0), (bar_x0 + bar_len_px, bar_y0)],
                  fill=(40, 40, 40), width=3)
        draw.text((bar_x0 + bar_len_px // 2 - 8, bar_y0 + 3), "1 m",
                  font=font_tick, fill=(40, 40, 40))

        # 17. 无人机信息标注（底部栏）—— 双行：map坐标 / px4 odom坐标分离显示
        if self.pose is not None:
            _mx_str = f"{self._map_px:.2f}" if self._map_px is not None else "N/A"
            _my_str = f"{self._map_py:.2f}" if self._map_py is not None else "N/A"
            # ★ 底栏 odom 来源：优先使用 bridge /odom（ENU），备用 pose.x/y（已 ENU 转换）
            if self._odom_received:
                _x_odom = float(self._odom_x_enu)
                _y_odom = float(self._odom_y_enu)
                _z_odom = float(self._odom_z_up)
                _odom_src = 'bridge'
            elif self.pose is not None:
                _x_odom = float(self.pose.x)
                _y_odom = float(self.pose.y)
                _z_odom = float(self.pose.z)
                _odom_src = 'px4fb'
            else:
                _x_odom = _y_odom = _z_odom = 0.0
                _odom_src = 'none'
            _esc_str = "ESC" if getattr(self, '_hover_scan_escape_active', False) else "-"
            # map pose 质量标注
            _mp_src  = getattr(self, '_map_pose_source', '?')
            _mp_stale = getattr(self, '_map_pose_stale', True)
            _mp_valid = getattr(self, '_map_pose_valid', False)
            _mp_qual = ("stale" if _mp_stale else "fresh") if _mp_valid else "invalid"
            # odom 新鲜度
            import time as _vf_t
            _now_vf = _vf_t.monotonic()
            _odom_age_ms = int((_now_vf - self._odom_stamp_mono) * 1000) if self._odom_received else -1
            _planar_str = getattr(self, '_planar_pose_invalid_reason', 'ok')
            _tilt_vis   = getattr(self, '_att_tilt_local', 0.0)
            _planar_ok_b = getattr(self, '_planar_pose_valid', True)
            _planar_tag = (
                f"planar=OK(tilt={_tilt_vis:.0f}deg)" if _planar_ok_b
                else f"PLANAR_INVALID({_planar_str})"
            )
            # ★ 底栏显示优先级：map-main（真实 SLAM map frame）为主；plan-grid 和 debug 为辅
            # map-main: 直接来自 _map_px/_map_py（TF/cache），独立于网格坐标变换
            # plan-grid: SE(2)变换后的网格坐标，用于 grid 索引（仅作为调试参考）
            _plan_x_str = f"{self._plan_px:.2f}"
            _plan_y_str = f"{self._plan_py:.2f}"
            _plan_yaw_deg_str = f"{float(getattr(self,'_plan_yaw',0.0))*57.3:.0f}"
            _slam_rot_str = (f"(slam_rot={float(getattr(self,'_slam_origin_yaw',0.0))*57.3:.1f}deg)"
                             if abs(float(getattr(self,'_slam_origin_yaw',0.0))) > 0.035 else "")
            drone_info_l1 = (
                f"[map-main] map=({_mx_str},{_my_str})m  src={_mp_qual}/{_mp_src}  "
                f"yaw={_plan_yaw_deg_str}deg {_slam_rot_str}  {_planar_tag}  "
                f"[dbg-plan] plan=({_plan_x_str},{_plan_y_str})"
            )
            drone_info_l2 = (
                f"状态={self.state}  esc={_esc_str}  "
                f"[bridge-odom/ENU] east={_x_odom:.2f} north={_y_odom:.2f} "
                f"up={_z_odom:.2f}m  src={_odom_src}  odom_age={_odom_age_ms}ms"
            )
            draw.rectangle([(LEGEND_W, CANVAS_H - BOT_BAR + 2), (CANVAS_W, CANVAS_H)],
                           fill=(245, 240, 230))
            draw.text((LEGEND_W + 6, CANVAS_H - BOT_BAR + 5),
                      drone_info_l1, font=font_info, fill=(80, 40, 0))
            draw.text((LEGEND_W + 6, CANVAS_H - BOT_BAR + 22),
                      drone_info_l2, font=font_info, fill=(0, 60, 100))
            # 5s 节流 [VIS-FRAME] 日志
            if _now_vf - getattr(self, '_vis_frame_log_t', 0.0) > 5.0:
                self._vis_frame_log_t = _now_vf
                self.get_logger().debug(
                    f"[VIS-FRAME] map-main=({_mx_str},{_my_str}) map_qual={_mp_qual}/{_mp_src} "
                    f"slam_origin_yaw={float(getattr(self,'_slam_origin_yaw',0.0))*57.3:.1f}deg "
                    f"dbg-plan=({_plan_x_str},{_plan_y_str}) "
                    f"bridge_odom_enu=({_x_odom:.2f},{_y_odom:.2f},{_z_odom:.2f}) "
                    f"odom_age={_odom_age_ms}ms"
                )

        # 17.5 ★ A* 规划路径叠加（1~2Hz节流，橙色折线+黄色目标点）
        # 目的：在地图上直观显示当前规划路径，帮助诊断"路径本身是否回穿"
        # 算力低：路径点数量少，只做简单折线，不重新计算任何东西
        import time as _vis_t
        _path_now_vis = _vis_t.monotonic()   # 独立获取时间戳（_now_vf 在条件块内可能未赋值）
        if _path_now_vis - getattr(self, '_path_vis_last_t', 0.0) > 0.8:   # ~1.2Hz
            self._path_vis_last_t = _path_now_vis
            _astar_vis = getattr(self, 'astar_path', None)
            _goal_vis  = getattr(self, 'ssg_viewpoint_goal', None)
            _gtype_vis = getattr(self, 'ssg_goal_type', 'none')
            _path_pts_drawn = 0
            try:
                # ★[K4] 单点路径可视化修复：原来要求 >1 个点，单点路径（A*起终点相同时）不显示
                if _astar_vis is not None and len(_astar_vis) >= 1:
                    _pip_pts = []
                    for _wp in _astar_vis:
                        _wix, _wiy = self.world_to_grid(float(_wp[0]), float(_wp[1]))
                        _px_c = gx_to_px(_wix)
                        _py_c = gy_to_py(_wiy)
                        _pip_pts.append((_px_c, _py_c))
                    _path_color = (255, 140, 0) if _gtype_vis not in (
                        'frontier_gap_pre','frontier_gap_center','frontier_gap_post',
                        'frontier_gap_probe','confirmed_door') else (255, 220, 0)
                    for _pp_i in range(len(_pip_pts) - 1):
                        draw.line([_pip_pts[_pp_i], _pip_pts[_pp_i + 1]], fill=_path_color, width=2)
                    _path_pts_drawn = len(_pip_pts)
                    # 当前跟踪起始点（橙色小圆）
                    _pidx = getattr(self, 'astar_path_idx', 0)
                    if 0 <= _pidx < len(_pip_pts):
                        _tx, _ty = _pip_pts[_pidx]
                        draw.ellipse([(_tx - 4, _ty - 4), (_tx + 4, _ty + 4)],
                                     fill=(255, 100, 0), outline=(180, 60, 0), width=1)
                    # ★[A] Carrot 点可视化（亮绿色圆 + idx 标注）
                    _cidx = getattr(self, '_carrot_idx', _pidx)
                    if 0 <= _cidx < len(_pip_pts) and _cidx != _pidx:
                        _ctx, _cty = _pip_pts[_cidx]
                        draw.ellipse([(_ctx - 6, _cty - 6), (_ctx + 6, _cty + 6)],
                                     fill=(0, 230, 80), outline=(0, 140, 40), width=2)
                        draw.text((_ctx + 7, _cty - 6), f"C{_cidx}", font=font_info,
                                  fill=(0, 160, 40))
                    # 当前跟踪段高亮（蓝色）
                    if 0 <= _pidx < _cidx < len(_pip_pts):
                        for _si in range(_pidx, _cidx):
                            draw.line([_pip_pts[_si], _pip_pts[_si + 1]],
                                      fill=(0, 180, 255), width=3)
                # 目标终点（黄色圆圈）
                if _goal_vis is not None:
                    _gix, _giy = self.world_to_grid(float(_goal_vis[0]), float(_goal_vis[1]))
                    _gpx_c, _gpy_c = gx_to_px(_gix), gy_to_py(_giy)
                    _goal_color = (255, 240, 0) if _gtype_vis not in (
                        'frontier_gap_pre','frontier_gap_center','frontier_gap_post',
                        'frontier_gap_probe','confirmed_door') else (0, 255, 100)
                    draw.ellipse([(_gpx_c - 6, _gpy_c - 6), (_gpx_c + 6, _gpy_c + 6)],
                                 fill=None, outline=_goal_color, width=2)
                # ★ [A11] 增强可视化：门穿越阶段中间点 + gap位置 + region graph置信度
                # ── 门阶段提交中间点（品红色小圆）────────────────────────────
                _dpc_vis = getattr(self, '_door_phase_commit', None)
                _dpc_drawn = 0
                if _dpc_vis is not None:
                    for _dpc_key, _dpc_clr in [
                            ('center', (255, 0, 220)),    # 品红: 门中心
                            ('post',   (180, 0, 255))]:   # 紫: 门后点
                        _dpc_pt = _dpc_vis.get(_dpc_key)
                        if _dpc_pt is not None:
                            _dpc_ix, _dpc_iy = self.world_to_grid(float(_dpc_pt[0]), float(_dpc_pt[1]))
                            _dpc_px, _dpc_py = gx_to_px(_dpc_ix), gy_to_py(_dpc_iy)
                            draw.ellipse([(_dpc_px-5, _dpc_py-5), (_dpc_px+5, _dpc_py+5)],
                                         fill=_dpc_clr, outline=(80, 0, 80), width=1)
                            _dpc_drawn += 1
                # ── [A10] 三级门标记：soft_gap(黄菱)/stable_soft_gap(橙方)/frontier_gap(蓝方) ──
                # 层1: soft_gap（hits 1-2）→ 小黄菱形，仅可视化
                _soft_gaps_vis = getattr(self, 'soft_gaps', [])
                for _sg_v in _soft_gaps_vis[:10]:
                    _sg_ix, _sg_iy = self.world_to_grid(float(_sg_v.get('x', _sg_v.get('cx', 0))),
                                                         float(_sg_v.get('y', _sg_v.get('cy', 0))))
                    _sg_px, _sg_py = gx_to_px(_sg_ix), gy_to_py(_sg_iy)
                    # 菱形 = 旋转45°方块
                    draw.polygon([(_sg_px, _sg_py-4), (_sg_px+4, _sg_py),
                                  (_sg_px, _sg_py+4), (_sg_px-4, _sg_py)],
                                 fill=(255, 220, 0), outline=(200, 160, 0))   # 黄色
                # 层2: stable_soft_gap（hits 3-4）→ 橙色方块
                _stable_sg_vis = getattr(self, 'stable_soft_gaps', [])
                for _ssg_v in _stable_sg_vis[:10]:
                    _ssg_ix, _ssg_iy = self.world_to_grid(float(_ssg_v.get('x', _ssg_v.get('cx', 0))),
                                                           float(_ssg_v.get('y', _ssg_v.get('cy', 0))))
                    _ssg_px, _ssg_py = gx_to_px(_ssg_ix), gy_to_py(_ssg_iy)
                    draw.rectangle([(_ssg_px-4, _ssg_py-4), (_ssg_px+4, _ssg_py+4)],
                                   fill=(255, 140, 0), outline=(200, 80, 0))   # 橙色
                # 层3: frontier_gap（已穿越/导航候选）→ 深蓝方块（无置信度数字，减少杂乱）
                _fgaps_vis = getattr(self, 'frontier_gaps', [])
                for _fg_v in _fgaps_vis[:8]:   # 最多显示8个gap
                    _fg_ix, _fg_iy = self.world_to_grid(float(_fg_v['x']), float(_fg_v['y']))
                    _fg_px, _fg_py = gx_to_px(_fg_ix), gy_to_py(_fg_iy)
                    draw.rectangle([(_fg_px-3, _fg_py-3), (_fg_px+3, _fg_py+3)],
                                   fill=COLOR_DOOR, outline=(0, 40, 160))     # 深蓝，不标数字
                # ── region graph 边（高置信度用粗线，低置信度用细线）──────────
                _rg_vis = getattr(self, '_region_graph', {})
                for _eid_v, _ev in list(_rg_vis.get('edges', {}).items())[:12]:
                    _ex, _ey = float(_ev['pos'][0]), float(_ev['pos'][1])
                    _econf = int(_ev.get('conf', 0))
                    _e_ix, _e_iy = self.world_to_grid(_ex, _ey)
                    _e_px, _e_py = gx_to_px(_e_ix), gy_to_py(_e_iy)
                    _e_clr = (0, 255, 80) if _ev.get('traversed', False) else (
                             (255, 200, 0) if _econf >= 3 else (200, 200, 0))
                    _e_w = 3 if _ev.get('traversed', False) else (2 if _econf >= 5 else 1)
                    draw.ellipse([(_e_px-4, _e_py-4), (_e_px+4, _e_py+4)],
                                 fill=None, outline=_e_clr, width=_e_w)
                # ★ 路径绘制日志（5s节流）
                if _path_now_vis - getattr(self, '_path_vis_log_t', 0.0) > 5.0:
                    self._path_vis_log_t = _path_now_vis
                    _gv_str2 = f"({_goal_vis[0]:.2f},{_goal_vis[1]:.2f})" if _goal_vis is not None else "None"
                    self.get_logger().debug(
                        f"[PATH-VIS] astar_pts={_path_pts_drawn} goal_type={_gtype_vis} "
                        f"goal={_gv_str2} gaps={len(_fgaps_vis)} "
                        f"dpc={'yes' if _dpc_drawn>0 else 'no'} "
                        f"rg_edges={len(_rg_vis.get('edges',{}))} "
                        f"draw_ok={'yes' if _path_pts_drawn>0 else 'no(no_path)'}"
                    )
            except Exception as _pv_e:
                self.get_logger().debug(f"[PATH-VIS] 绘制失败: {_pv_e}")

        # 18. 保存PNG
        canvas_arr = np.array(canvas_pil)
        cv2_img = cv2.cvtColor(canvas_arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.map_output_dir / "map_latest.png"), cv2_img)

        # 历史快照（每10s一次）
        if not hasattr(self, '_last_map_snapshot_sec'):
            self._last_map_snapshot_sec = 0.0
        if t_sec - self._last_map_snapshot_sec > 10.0:
            self._last_map_snapshot_sec = t_sec
            snap_name = f"map_{int(t_sec)}.png"
            cv2.imwrite(str(self.map_output_dir / snap_name), cv2_img)

    def local_cb(self, msg: LocalCommand) -> None:
        # 直接接受 ENU 坐标（local_mpc_node 已输出 ENU: x=East, y=North）
        # 禁止在此处做任何坐标交换，否则会再次错乱 90°
        self.last_local_cmd = msg
        self.last_local_stamp = self.get_clock().now()

    def _odom_cb(self, msg: Odometry) -> None:
        """bridge /odom 回调：更新 ENU 主坐标缓存。
        这是 target_search 内 odom 视角位置的权威来源（x=East, y=North, z=Up）。
        VehicleLocalPosition 仅用于 reset counter / delta_xy / delta_z 等 PX4 诊断。
        """
        import time as _oc_t
        self._odom_x_enu   = float(msg.pose.pose.position.x)  # ENU East
        self._odom_y_enu   = float(msg.pose.pose.position.y)  # ENU North
        self._odom_z_up    = float(msg.pose.pose.position.z)  # ENU Up
        self._odom_stamp_mono = _oc_t.monotonic()
        self._odom_pose_stale = False
        # 从四元数提取 ENU yaw（可选，用于显示/备用）
        try:
            from scipy.spatial.transform import Rotation as _RR
            _q = msg.pose.pose.orientation
            _qv = [_q.x, _q.y, _q.z, _q.w]
            self._odom_yaw_enu = float(_RR.from_quat(_qv).as_euler('zyx')[0])
        except Exception:
            pass
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info(
                f"[ODOM-CB] bridge /odom 首帧到达: ENU east={self._odom_x_enu:.2f}m "
                f"north={self._odom_y_enu:.2f}m up={self._odom_z_up:.2f}m → "
                f"odom 已成为 target_search 横向主位姿来源"
            )

    def apf_cb(self, msg: ApfCommand) -> None:
        self.last_apf_cmd = msg
        self.last_apf_stamp = self.get_clock().now()

    # ==================================================================
    # 模块A：OpenCV 红色目标检测
    # ==================================================================
    def _detect_red_target_opencv(self, frame: np.ndarray) -> Optional[dict]:
        """
        基于 HSV 双阈值的红色目标检测。
        返回 dict{cx, cy, cx_norm, cy_norm, area_ratio, bbox(x,y,w,h)}
        若未检测到返回 None。
        主链路：HSV → 双段mask → 形态学 → 最大轮廓 → 面积门控。
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 红色双段阈值（H=0-10 / H=170-180）
        mask1 = cv2.inRange(hsv, self.red_hsv_lower_1, self.red_hsv_upper_1)
        mask2 = cv2.inRange(hsv, self.red_hsv_lower_2, self.red_hsv_upper_2)
        mask = cv2.bitwise_or(mask1, mask2)
        # 形态学去噪：先 open（去小噪点），再 close（填充内部小洞）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.red_morph_open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.red_morph_close_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area_px = float(cv2.contourArea(c))
        area_ratio = area_px / max(1.0, float(w * h))
        # 最低面积门控（过滤噪点）
        if area_ratio < self.red_min_area_ratio_detect:
            return None
        m = cv2.moments(c)
        if m["m00"] < 1e-6:
            return None
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        bx, by, bw, bh = cv2.boundingRect(c)
        return {
            "cx": cx, "cy": cy,
            "cx_norm": cx / float(w), "cy_norm": cy / float(h),
            "area_ratio": area_ratio,
            "bbox": (bx, by, bw, bh),
            "mask": mask,   # 供调试图像使用
        }

    # ==================================================================
    # 模块A.2：OpenCV Tracker 初始化与更新（CSRT / KCF / MOSSE 三档降级）
    # ==================================================================
    def _init_target_tracker(self, frame: np.ndarray, bbox: tuple) -> None:
        """
        用检测到的 bbox 初始化 OpenCV tracker。
        bbox = (x, y, w, h)（像素坐标）。
        按优先级尝试 CSRT → KCF → MOSSE，均失败则禁用 tracker。
        """
        import time as _tr_t
        h, w = frame.shape[:2]
        self._cv_tracker = None
        self._cv_tracker_ok = False
        self._cv_tracker_frame_w = w
        self._cv_tracker_frame_h = h
        # 按配置类型选择 tracker
        _pref = self.target_tracker_type.upper()
        _try_order = []
        if _pref == "CSRT":
            _try_order = ["CSRT", "KCF", "MOSSE"]
        elif _pref == "KCF":
            _try_order = ["KCF", "MOSSE"]
        elif _pref == "MOSSE":
            _try_order = ["MOSSE"]
        else:
            _try_order = []  # "none" → 禁用
        for _ttype in _try_order:
            try:
                if _ttype == "CSRT":
                    _t = cv2.TrackerCSRT_create()
                elif _ttype == "KCF":
                    _t = cv2.TrackerKCF_create()
                else:
                    _t = cv2.legacy.TrackerMOSSE_create()
                _t.init(frame, bbox)
                self._cv_tracker = _t
                self._cv_tracker_ok = True
                self._cv_tracker_last_t = _tr_t.monotonic()
                self.get_logger().info(
                    f"[TARGET-TRACK-INIT] tracker={_ttype} "
                    f"bbox={bbox} frame={w}x{h}"
                )
                return
            except Exception as _te:
                self.get_logger().debug(
                    f"[TARGET-TRACK-INIT] {_ttype} 初始化失败: {_te}, 尝试下一级"
                )
        self.get_logger().warn(
            "[TARGET-TRACK-INIT] 所有 tracker 类型初始化失败，退化为纯 Kalman 预测"
        )

    def _update_target_tracker(self, frame: np.ndarray) -> Optional[dict]:
        """
        更新 OpenCV tracker。
        返回 dict{cx, cy, cx_norm, cy_norm, bbox} 或 None（tracker 失败）。
        """
        import time as _tr2_t
        if self._cv_tracker is None or not self._cv_tracker_ok:
            return None
        try:
            ok, bbox = self._cv_tracker.update(frame)
            self._cv_tracker_last_t = _tr2_t.monotonic()
            if not ok:
                self._cv_tracker_ok = False
                self.get_logger().debug("[TARGET-TRACK] tracker update failed → 标记失败")
                return None
            x, y, bw, bh = [float(v) for v in bbox]
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            fw = float(self._cv_tracker_frame_w)
            fh = float(self._cv_tracker_frame_h)
            return {
                "cx": cx, "cy": cy,
                "cx_norm": cx / max(1.0, fw),
                "cy_norm": cy / max(1.0, fh),
                "bbox": (int(x), int(y), int(bw), int(bh)),
            }
        except Exception as _e:
            self._cv_tracker_ok = False
            self.get_logger().debug(f"[TARGET-TRACK] update 异常: {_e}")
            return None

    # ==================================================================
    # 模块A.3：Kalman 状态更新与预测（融合检测/跟踪结果）
    # ==================================================================
    def _update_target_filter(self, cx_norm: float, cy_norm: float) -> tuple:
        """
        用测量值更新 Kalman 滤波器，返回滤波后的 (cx_norm, cy_norm)。
        同时更新 target_track_conf。
        """
        kx, ky = self._kf_target.update(cx_norm, cy_norm)
        self.target_track_conf = float(self._kf_target.get_conf())
        return kx, ky

    def _predict_target_state(self, dt: float = 1.0) -> tuple:
        """
        Kalman 向前预测 dt 步，返回 (cx_norm, cy_norm, vx, vy)。
        衰减置信度（模拟无测量状态）。
        """
        self._kf_target.decay(dt * self.detect_interval_sec)
        self.target_track_conf = float(self._kf_target.get_conf())
        return self._kf_target.predict(dt)

    def _update_target_world_xy_est(self) -> None:
        """★[G3] 当目标在图像中可见时，估算目标在地图平面(z≈地面)的世界坐标。

        原理：针孔相机模型，假设目标位于地面平面（相机光轴垂直向下时正好对准地面）。
            world_lat_offset = tan(hfov/2) * cx_norm_offset * flight_height
            用机体 yaw 旋转 body_frame 偏移 → ENU 世界坐标偏移。

        条件：需要有效姿态(att_yaw) + 位置(pose) + 目标候选(target_candidate)。
        输出：更新 self._tgt_world_est_xy（EMA滤波）+ self._tgt_world_est_conf。
        """
        import math as _wxy_m, time as _wxy_t
        _wxy_now = _wxy_t.monotonic()

        # 前提条件检查
        if (self.pose is None
                or self.att_yaw is None
                or not self.target_candidate
                or self.target_area_ratio < 0.001):
            # 无目标时置信度缓慢衰减（每次调用衰减0.02）
            if self._tgt_world_est_conf > 0.0:
                self._tgt_world_est_conf = max(0.0, self._tgt_world_est_conf - 0.02)
            return

        # 飞行高度（正值, m）：使用NED z（负=高），取绝对值作为离地高度估计
        _height = max(0.20, abs(float(self.pose.z)))

        # 图像归一化偏差：target_offset_x = (cx_n - 0.5) * 2 ∈ [-1, 1]
        # 转回 cx_n_offset ∈ [-0.5, 0.5]（中心=0）
        _cx_n_offset = self.target_offset_x * 0.5   # 正=目标在右
        _cy_n_offset = self.target_offset_y * 0.5   # 正=目标在下（图像y轴向下）

        # 相机水平/垂直视场角（水平 hfov，垂直 = hfov*(H/W)，简化取相同）
        _hfov_rad = float(self.target_camera_hfov_deg) * _wxy_m.pi / 180.0
        _tan_half_h = _wxy_m.tan(_hfov_rad / 2.0)

        # body frame 偏移（假设相机朝前，前方=+x_body，右方=+y_body，向下=+z_body）
        # cy_n_offset < 0 → 目标偏上 → 目标在无人机前方更远处 → +前向偏移
        _x_off_body = _height * _tan_half_h * (-_cy_n_offset)   # 前向偏移
        _y_off_body = _height * _tan_half_h * _cx_n_offset       # 右向偏移

        # body → ENU：用 yaw 角旋转（ENU yaw=0 = East，yaw=π/2 = North）
        _yaw = float(self.att_yaw)
        _cos_y = _wxy_m.cos(_yaw)
        _sin_y = _wxy_m.sin(_yaw)
        _dx_enu = _cos_y * _x_off_body - _sin_y * _y_off_body
        _dy_enu = _sin_y * _x_off_body + _cos_y * _y_off_body

        # 世界坐标估计 = 无人机位置 + 地面投影偏移
        _est_x = self._plan_px + _dx_enu
        _est_y = self._plan_py + _dy_enu

        # Bounds guard：估计超出规划地图范围时忽略
        _bxy = float(getattr(self, '_plan_bounds_xy',
                             np.array([[-8, 8], [-8, 8], [-3, 3]]))[0, 1])
        if abs(_est_x) > _bxy * 1.2 or abs(_est_y) > _bxy * 1.2:
            return

        # EMA 更新
        _alpha = float(self.target_world_est_alpha)
        if self._tgt_world_est_xy is None or self._tgt_world_est_conf < 0.05:
            # 首次或置信度极低时直接初始化
            self._tgt_world_est_xy = np.array([_est_x, _est_y], dtype=float)
            self._tgt_world_est_conf = min(0.40, float(self.target_track_conf) * 0.7)
        else:
            self._tgt_world_est_xy = ((1.0 - _alpha) * self._tgt_world_est_xy
                                      + _alpha * np.array([_est_x, _est_y]))
            # 置信度缓慢累积（每次更新+0.015，受 track_conf 调制）
            _conf_inc = 0.015 * max(0.3, float(self.target_track_conf))
            self._tgt_world_est_conf = min(1.0, self._tgt_world_est_conf + _conf_inc)

        self._tgt_world_est_count += 1
        self._tgt_world_est_last_t = _wxy_now

        # ★[G5] 历史记录 + 方差估计（用于保守降落条件判断）
        self._tgt_world_xy_history.append((_est_x, _est_y))
        _hist_len = int(getattr(self, 'target_world_history_len', 15))
        if len(self._tgt_world_xy_history) > _hist_len:
            self._tgt_world_xy_history.pop(0)
        if len(self._tgt_world_xy_history) >= 4:
            _hx = np.array([p[0] for p in self._tgt_world_xy_history])
            _hy = np.array([p[1] for p in self._tgt_world_xy_history])
            self._tgt_world_xy_var = float(np.var(_hx) + np.var(_hy))
        else:
            self._tgt_world_xy_var = 999.0

        # 每20次更新打印一次（节流）
        if self._tgt_world_est_count % 20 == 1:
            self.get_logger().info(
                f"[TARGET-WORLD-EST] ★[G3] "
                f"xy=({self._tgt_world_est_xy[0]:.2f},{self._tgt_world_est_xy[1]:.2f}) "
                f"conf={self._tgt_world_est_conf:.2f} count={self._tgt_world_est_count} "
                f"height={_height:.2f}m cx_off={_cx_n_offset:.3f} cy_off={_cy_n_offset:.3f} "
                f"drone=({self._plan_px:.2f},{self._plan_py:.2f}) yaw={_yaw:.2f}"
            )

    def _compute_iou(self, bbox_a: tuple, bbox_b: tuple) -> float:
        """
        计算两个 bbox 的 IoU（Intersection over Union）。
        bbox = (x, y, w, h)。
        """
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return float(inter / max(1.0, union))

    # ==================================================================
    # 模块A.4：调试图像发布（叠加检测框、tracker框、预测中心、状态文字）
    # ==================================================================
    def _publish_target_debug_image(
        self,
        frame: np.ndarray,
        det_result: Optional[dict],
        trk_result: Optional[dict],
        kf_cx_norm: float,
        kf_cy_norm: float,
        kf_conf: float,
    ) -> None:
        """
        在 frame 上叠加调试信息并发布到 /indoor_search/target_debug。
        颜色约定：
          绿色实框 = OpenCV 检测框
          橙色虚框 = OpenCV Tracker 框
          蓝色圆点 = Kalman 预测中心
          白色文字 = 状态机子状态 + 置信度
        """
        import time as _vis_t
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. OpenCV 检测框（绿色）
        if det_result is not None:
            bx, by, bw, bh = det_result["bbox"]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 220, 0), 2)
            cv2.putText(vis, f"DET {det_result['area_ratio']:.3f}",
                        (bx, max(12, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        # 2. Tracker 框（橙色）
        if trk_result is not None and self._cv_tracker_ok:
            bx, by, bw, bh = trk_result["bbox"]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 140, 255), 2)
            cv2.putText(vis, "TRK",
                        (bx, max(12, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

        # 3. Kalman 预测中心（蓝色实心圆）
        if self._kf_target.initialized and kf_conf > 0.1:
            kfx = int(kf_cx_norm * w)
            kfy = int(kf_cy_norm * h)
            cv2.circle(vis, (kfx, kfy), 6, (255, 80, 0), -1)
            cv2.putText(vis, f"KF {kf_conf:.2f}",
                        (kfx + 8, kfy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 0), 1)

        # 4. 图像中心十字线（灰色）
        cv2.line(vis, (w // 2 - 12, h // 2), (w // 2 + 12, h // 2), (180, 180, 180), 1)
        cv2.line(vis, (w // 2, h // 2 - 12), (w // 2, h // 2 + 12), (180, 180, 180), 1)

        # 5. 状态文字（左上角）
        _st = getattr(self, '_approach_inner_state', 'search')
        _confirmed = "CONF" if self.target_confirmed else ("CAND" if self.target_candidate else "NONE")
        _src = self.target_source
        _area = self.target_area_ratio
        _lost = self.target_lost_stage
        _lines = [
            f"STATE: {_st}  TARG: {_confirmed}",
            f"SRC: {_src}  AREA: {_area:.3f}  CONF: {kf_conf:.2f}",
            f"LOST: {_lost}  TRK: {'OK' if self._cv_tracker_ok else 'FAIL'}",
        ]
        for _li, _line in enumerate(_lines):
            cv2.putText(vis, _line, (6, 16 + _li * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)

        try:
            _dbg_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            _dbg_msg.header.stamp = self.get_clock().now().to_msg()
            self._rgb_debug_pub.publish(_dbg_msg)
        except Exception as _pe:
            pass  # 调试图像非关键路径，失败静默

    # ==================================================================
    # 主回调：rgb_cb — 负责帧率控制、检测/跟踪/Kalman融合、确认状态机
    # ==================================================================
    def rgb_cb(self, msg: Image) -> None:
        """
        RGB 图像回调（主链路：OpenCV 红色检测 + Kalman 滤波 + tracker 辅助）。

        流程：
          1. 帧率门控（detect_interval_sec）
          2. 调用 _detect_red_target_opencv 进行主检测
          3. 若在 target_approach 阶段，同时更新 OpenCV Tracker
          4. 融合 detection + tracker → Kalman 滤波 → 输出稳定位置估计
          5. 三级目标确认状态机（detect → candidate → confirmed）
          6. 发布调试图像
        """
        import time as _rcb_t
        self.last_rgb_image = msg
        self.rgb_counter += 1

        now = self.get_clock().now()
        # 帧率门控：detect_interval_sec（默认 0.25s = 4Hz）
        if self.rgb_counter % max(1, self.detect_every_n) != 0:
            return
        _elapsed = (now - self.last_detect_time).nanoseconds * 1e-9
        if _elapsed < self.detect_interval_sec:
            return
        self.last_detect_time = now
        _now_m = _rcb_t.monotonic()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h, w = frame.shape[:2]

            # ─────────────────────────────────────────────────────────────
            # 步骤 1：YOLO 异步推理（辅助检测链，不影响主 OpenCV 链）
            # ─────────────────────────────────────────────────────────────
            yolo_result: Optional[dict] = None
            if self._infer_future is not None and self._infer_future.done():
                try:
                    yolo_result = self._infer_future.result(timeout=0)
                except Exception as _fe:
                    self.get_logger().warn(f"[YOLO-ASYNC] future error: {_fe}")
                self._infer_future = None
            if self._infer_future is None and self.yolo_model is not None:
                self._infer_future = self._infer_executor.submit(
                    self._yolo_infer_worker, frame.copy()
                )
            elif self._infer_future is not None:
                self._infer_drop_count += 1

            # ─────────────────────────────────────────────────────────────
            # 步骤 2：OpenCV 红色主检测（主链路）
            # ─────────────────────────────────────────────────────────────
            det_result: Optional[dict] = None
            if self.enable_red_detect:
                det_result = self._detect_red_target_opencv(frame)

            # YOLO 辅助检测（补充红色主链）
            yolo_det: Optional[dict] = None
            if yolo_result is not None and "_error" not in yolo_result:
                _yc = yolo_result["conf"]
                _yx, _yy = yolo_result["cx"], yolo_result["cy"]
                _x1, _y1 = yolo_result["x1"], yolo_result["y1"]
                _x2, _y2 = yolo_result["x2"], yolo_result["y2"]
                _yar = max(0.0, min(1.0, ((_x2 - _x1) * (_y2 - _y1)) / max(1.0, float(w * h))))
                self.target_confirmation.update(max_conf=_yc, now_sec=now.nanoseconds * 1e-9)
                if _yc >= self.detect_conf and _yar >= self.red_min_area_ratio_detect:
                    _ybbox = (int(_x1), int(_y1), int(_x2 - _x1), int(_y2 - _y1))
                    yolo_det = {
                        "cx": _yx, "cy": _yy,
                        "cx_norm": _yx / float(w), "cy_norm": _yy / float(h),
                        "area_ratio": _yar, "bbox": _ybbox,
                    }

            # ─────────────────────────────────────────────────────────────
            # 步骤 3：选择主检测结果（OpenCV 优先，YOLO 补充）
            # ─────────────────────────────────────────────────────────────
            primary_det: Optional[dict] = None
            _det_src = "none"
            if det_result is not None:
                primary_det = det_result
                _det_src = "opencv"
                # 更新 red_confirmation 计数
                self.red_confirmation.update(
                    max_conf=det_result["area_ratio"],
                    now_sec=now.nanoseconds * 1e-9
                )
            elif yolo_det is not None:
                # OpenCV 未检测，用 YOLO 补充
                primary_det = yolo_det
                _det_src = "yolo"

            # ─────────────────────────────────────────────────────────────
            # 步骤 4：OpenCV Tracker 更新（仅在接近阶段启用，减少 CPU 开销）
            # ─────────────────────────────────────────────────────────────
            trk_result: Optional[dict] = None
            _in_approach = getattr(self, '_approach_exclusive', False)
            if _in_approach:
                trk_result = self._update_target_tracker(frame)
                # tracker 结果验证：若与当前 detection 偏差过大，重新初始化
                if (primary_det is not None
                        and trk_result is not None
                        and "bbox" in primary_det):
                    _iou = self._compute_iou(primary_det["bbox"], trk_result["bbox"])
                    if _iou < self.target_tracker_reinit_iou_thresh:
                        # 检测框与 tracker 框分歧过大，用新检测重新初始化 tracker
                        self.get_logger().debug(
                            f"[TARGET-TRACK] IoU={_iou:.2f} < {self.target_tracker_reinit_iou_thresh:.2f} "
                            f"→ 用新检测重新初始化 tracker"
                        )
                        self._init_target_tracker(frame, primary_det["bbox"])
                        trk_result = None  # 本帧 tracker 结果作废，避免 bbox 跳跃

            # ─────────────────────────────────────────────────────────────
            # 步骤 5：融合 detection + tracker → Kalman 更新 → 稳定位置估计
            # ─────────────────────────────────────────────────────────────
            _fused_cx_norm: Optional[float] = None
            _fused_cy_norm: Optional[float] = None
            _fused_bbox: Optional[tuple] = None
            _fused_area_ratio: float = 0.0
            _fused_src = "none"

            if primary_det is not None:
                # 检测结果存在 → 更新 Kalman（最高优先级）
                _raw_cx = primary_det["cx_norm"]
                _raw_cy = primary_det["cy_norm"]
                _kfx, _kfy = self._update_target_filter(_raw_cx, _raw_cy)
                _fused_cx_norm = _kfx
                _fused_cy_norm = _kfy
                _fused_bbox = primary_det.get("bbox")
                _fused_area_ratio = primary_det["area_ratio"]
                _fused_src = _det_src

                # 若 tracker 未初始化 或 tracker 已失败，用新检测重新初始化
                if (_in_approach
                        and _fused_bbox is not None
                        and (self._cv_tracker is None or not self._cv_tracker_ok)):
                    self._init_target_tracker(frame, _fused_bbox)

                # 打印 [TARGET-DETECT] 日志（1s 节流）
                if _now_m - getattr(self, '_det_log_t', 0.0) > 1.0:
                    self._det_log_t = _now_m
                    self.get_logger().info(
                        f"[TARGET-DETECT] source={_fused_src} "
                        f"bbox={_fused_bbox} "
                        f"cx={_raw_cx:.3f} cy={_raw_cy:.3f} "
                        f"area={_fused_area_ratio:.4f} "
                        f"kf_conf={self.target_track_conf:.2f}"
                    )
                    # ★[C] 写入 target_search.txt（仅 opencv/yolo 来源，非 tracker 传播）
                    if _fused_src in ("opencv", "yolo"):
                        self._log_target_event("TARGET-DETECT",
                            f"source={_fused_src} area={_fused_area_ratio:.4f} "
                            f"cx_n={_raw_cx:.3f} cy_n={_raw_cy:.3f}",
                            world_xy=getattr(self, '_tgt_world_est_xy', None),
                            conf=self._tgt_world_est_conf, trigger="detect")

            elif trk_result is not None:
                # 无主检测，但 tracker 仍有结果 → 用 tracker bbox 做 Kalman 更新
                _kfx, _kfy = self._update_target_filter(
                    trk_result["cx_norm"], trk_result["cy_norm"]
                )
                _fused_cx_norm = _kfx
                _fused_cy_norm = _kfy
                _fused_bbox = trk_result["bbox"]
                # 面积比由 bbox 推算（tracker 无面积，用相对估算）
                _fb = trk_result["bbox"]
                _fused_area_ratio = float((_fb[2] * _fb[3]) / max(1.0, float(w * h)))
                _fused_src = "tracker"
                # 打印 [TARGET-TRACK] 日志（1.5s 节流）
                if _now_m - getattr(self, '_trk_log_t', 0.0) > 1.5:
                    self._trk_log_t = _now_m
                    self.get_logger().info(
                        f"[TARGET-TRACK] source=tracker "
                        f"bbox={_fused_bbox} "
                        f"cx_n={trk_result['cx_norm']:.3f} cy_n={trk_result['cy_norm']:.3f} "
                        f"kf_conf={self.target_track_conf:.2f}"
                    )

            else:
                # 无检测，无 tracker → 仅 Kalman 预测（如置信度充足）
                _kf_pred = self._predict_target_state(dt=1.0)
                _kfcx, _kfcy = _kf_pred[0], _kf_pred[1]
                _pred_age = _now_m - self._kf_target.last_meas_t
                if (self.target_track_conf > 0.25
                        and _pred_age < self.target_prediction_max_hold_sec):
                    _fused_cx_norm = _kfcx
                    _fused_cy_norm = _kfcy
                    _fused_src = "kalman"
                    if _now_m - getattr(self, '_pred_log_t', 0.0) > 1.0:
                        self._pred_log_t = _now_m
                        self.get_logger().info(
                            f"[TARGET-TRACK] source=predict "
                            f"cx_n={_kfcx:.3f} cy_n={_kfcy:.3f} "
                            f"age={_pred_age:.2f}s "
                            f"kf_conf={self.target_track_conf:.2f}"
                        )
                else:
                    # Kalman 置信度也不足：无目标
                    _kf_pred = (0.0, 0.0, 0.0, 0.0)
                    _fused_src = "none"

            # ─────────────────────────────────────────────────────────────
            # 步骤 6：三级目标确认状态机
            # ─────────────────────────────────────────────────────────────
            _have_target = (_fused_cx_norm is not None)
            _edge_margin = 0.08  # 贴近图像边缘则降低信任度

            if _have_target:
                _cx_n = _fused_cx_norm
                _cy_n = _fused_cy_norm
                _area = _fused_area_ratio
                # 贴边检测：中心点距图像边缘 < 8% 认为不可信
                _edge_ok = (
                    _edge_margin < _cx_n < 1.0 - _edge_margin
                    and _edge_margin < _cy_n < 1.0 - _edge_margin
                )

                # ── Level-1：候选确立（面积 >= detect 门限，连续 >=3 帧）
                if _area >= self.red_min_area_ratio_detect:
                    self._target_cand_frames += 1
                    if self._target_confirm_start_t is None:
                        self._target_confirm_start_t = now.nanoseconds * 1e-9
                    self._target_cand_center_hist.append((_cx_n, _cy_n))
                    if len(self._target_cand_center_hist) > 20:
                        self._target_cand_center_hist.pop(0)
                else:
                    self._target_cand_frames = max(0, self._target_cand_frames - 1)

                if self._target_cand_frames >= 3:
                    _prev_cand = self.target_candidate
                    self.target_candidate = True
                    self.target_found = True
                    self.target_last_seen_time = now
                    self.target_source = _fused_src
                    # 归一化偏差：0=中心，±1=边缘
                    self.target_offset_x = (_cx_n - 0.5) * 2.0
                    self.target_offset_y = (_cy_n - 0.5) * 2.0
                    self.target_area_ratio = float(np.clip(_area, 0.0, 1.0))
                    self.target_bbox = _fused_bbox
                    self.target_center_px = (
                        int(_cx_n * w), int(_cy_n * h)
                    )
                    # ★[G3] 每帧目标可见时更新世界坐标估计
                    self._update_target_world_xy_est()
                    if not _prev_cand:
                        self.get_logger().info(
                            f"[TARGET-CAND] 候选确立 "
                            f"source={_fused_src} area={_area:.4f} "
                            f"cx_n={_cx_n:.3f} frames={self._target_cand_frames}"
                        )

                    # ── Level-2：严格确认（面积+中心稳定+streak）
                    _stable_center = True
                    if len(self._target_cand_center_hist) >= 5:
                        _tail = self._target_cand_center_hist[-5:]
                        _cxs = [p[0] for p in _tail]
                        _cys = [p[1] for p in _tail]
                        _stable_center = (
                            (max(_cxs) - min(_cxs)) < 0.35
                            and (max(_cys) - min(_cys)) < 0.35
                        )
                    _conf_area_ok = _area >= max(self.red_min_area_ratio, 0.003)
                    if _fused_src in ("opencv", "yolo"):
                        _streak = (self.red_confirmation.hit_streak
                                   if _fused_src == "opencv"
                                   else self.target_confirmation.hit_streak)
                        _need_streak = (self.red_confirm_hits
                                        if _fused_src == "opencv"
                                        else self.target_confirm_hits)
                    else:
                        # tracker/kalman 不参与 streak 主计数，使用上次 streak
                        _streak = max(
                            self.red_confirmation.hit_streak,
                            self.target_confirmation.hit_streak
                        )
                        _need_streak = self.red_confirm_hits

                    # ★[G6] 弱确认：比 full confirm 更早标记（2帧即触发，仅用于日志）
                    if _edge_ok and _conf_area_ok:
                        self._tgt_weak_confirm_cnt += 1
                    else:
                        # ★[E] 粘性衰减：miss 帧不立刻归零，每帧 -1（防止单帧遮挡重置计数）
                        self._tgt_weak_confirm_cnt = max(0, self._tgt_weak_confirm_cnt - 1)
                    if (not self._tgt_weak_confirmed
                            and self._tgt_weak_confirm_cnt >= int(getattr(
                                self, 'target_weak_confirm_hits', 2))):
                        self._tgt_weak_confirmed = True
                        # ★[A1] 弱确认触发终端任务锁定 — 此后搜索逻辑全部冻结
                        self.target_terminal_locked = True
                        self._mission_done = False  # 确保任务完成标志初始为 False
                        self.get_logger().warn(
                            f"[TARGET-WEAK-CONFIRMED] ★[A1] 弱确认 → 终端任务锁定! "
                            f"cnt={self._tgt_weak_confirm_cnt} area={_area:.3f} cx_n={_cx_n:.3f}"
                        )
                        self._log_target_event("TARGET-WEAK-CONFIRMED",
                            f"cnt={self._tgt_weak_confirm_cnt} area={_area:.3f} terminal_locked=True",
                            world_xy=getattr(self, '_tgt_world_est_xy', None),
                            conf=self._tgt_world_est_conf, trigger="weak_confirm")

                    if _conf_area_ok and _stable_center and _edge_ok and _streak >= _need_streak:
                        _prev_conf = self.target_confirmed
                        self.target_confirmed = True
                        if not _prev_conf:
                            self._log_event(
                                "TARGET-CONFIRMED",
                                f"source={_fused_src} area={_area:.3f} "
                                f"streak={_streak}/{_need_streak} "
                                f"cx_n={_cx_n:.3f}",
                            )
                            self.get_logger().warn(
                                f"[TARGET-CONFIRMED] ★ 目标确认 "
                                f"source={_fused_src} area={_area:.3f} "
                                f"streak={_streak}/{_need_streak}"
                            )
                    else:
                        if self._target_cand_frames % 8 == 1:
                            self.get_logger().info(
                                f"[TARGET-CAND-WAIT] src={_fused_src} area={_area:.4f} "
                                f"streak={_streak}/{_need_streak} "
                                f"stable={_stable_center} edge={_edge_ok}"
                            )

                # Belief 地图正证据
                if self.pose is not None and self.target_candidate:
                    self._belief_boost_at_pose(self._plan_px, self._plan_py)

                # 首次 found 事件
                if self.target_found and not getattr(self, '_target_found_logged', False):
                    self._target_found_logged = True
                    self._log_event(
                        "TARGET-FOUND",
                        f"source={self.target_source} "
                        f"area={self.target_area_ratio:.3f}",
                    )

            else:
                # 本帧无目标（检测+tracker+kalman 全失败）
                self._target_cand_frames = max(0, self._target_cand_frames - 1)
                self._tgt_weak_confirm_cnt = 0  # ★[G6] 弱确认计数重置
                if self._target_cand_frames == 0 and self.target_candidate:
                    self.target_candidate = False
                    # ★[A2] 终端锁定后不重置弱确认 — 避免目标遮挡时放弃终端任务
                    if not getattr(self, 'target_terminal_locked', False):
                        self._tgt_weak_confirmed = False  # ★[G6] 弱确认状态重置
                    else:
                        # 终端锁定：仅记录候选消失但不清除终端链状态
                        self.get_logger().debug(
                            "[TARGET-CAND-LOST-TERMINAL] 候选消失但终端锁定，保持 weak_confirmed"
                        )
                    self._ssg_freeze_logged = False   # ★[C] 解冻标志重置
                    self._target_confirm_start_t = None
                    if not getattr(self, 'target_terminal_locked', False):
                        self.get_logger().debug("[TARGET-CAND-LOST] 候选消失")

            # ─────────────────────────────────────────────────────────────
            # 步骤 7：调试图像发布（2Hz 节流，不影响控制性能）
            # ─────────────────────────────────────────────────────────────
            _kf_cx_n = float(self._kf_target._kf.statePost[0, 0]) if self._kf_target.initialized else 0.5
            _kf_cy_n = float(self._kf_target._kf.statePost[1, 0]) if self._kf_target.initialized else 0.5
            if _now_m - getattr(self, '_dbg_img_t', 0.0) > 0.5:  # 2Hz
                self._dbg_img_t = _now_m
                self._publish_target_debug_image(
                    frame, det_result, trk_result,
                    float(_kf_cx_n), float(_kf_cy_n),
                    self.target_track_conf,
                )

        except Exception as e:
            self.get_logger().warn(f"[RGB-CB] 图像处理异常: {e}", throttle_duration_sec=2.0)

    def _can_commit_conservative_descent(self) -> tuple:
        """★[G5] 保守降落条件判断。
        返回 (can_commit: bool, reason: str)。
        ★[H-FIX] 终端任务锁定时放宽条件：跳过 var/seen_ago/obstacle 检查，
        只保留 conf/height/vxy/pose 基础约束，确保已经到达提交点的任务不被阻断。
        """
        import time as _cd_t, math as _cd_m
        _terminal_locked = getattr(self, 'target_terminal_locked', False)
        # 7. 位姿质量（前置，terminal 也需要）
        if self.pose is None:
            return False, "no_pose"
        if getattr(self, 'att_yaw', None) is None:
            return False, "no_attitude"
        # 1. 世界坐标置信度
        _conf = float(getattr(self, '_tgt_world_est_conf', 0.0))
        _conf_min = float(getattr(self, 'target_commit_conf_min', 0.55))
        # ★[H-FIX] 终端锁定：降低置信度门槛（世界估计短暂丢失也继续）
        _actual_conf_min = 0.15 if _terminal_locked else _conf_min
        if _conf < _actual_conf_min:
            return False, f"conf={_conf:.2f}<{_actual_conf_min:.2f}"
        # 2. 世界坐标方差（★[H-FIX] terminal 时跳过，var=999 是历史帧不足，不代表不安全）
        if not _terminal_locked:
            _var = float(getattr(self, '_tgt_world_xy_var', 999.0))
            _var_max = float(getattr(self, 'target_commit_var_max', 0.30))
            if _var > _var_max:
                return False, f"var={_var:.3f}>{_var_max:.3f}"
        # 3. 高度低于提交阈值
        _z = float(self.pose.z)
        _commit_h = float(getattr(self, 'target_descend_commit_height', -0.6))
        if _z > _commit_h:
            return False, f"height_z={_z:.2f}>{_commit_h:.2f}"
        # 4. 当前水平速度足够小
        _vxy = float(_cd_m.hypot(
            float(getattr(self.pose, 'vx', 0.0)),
            float(getattr(self.pose, 'vy', 0.0))
        ))
        _spd_max = float(getattr(self, 'target_commit_speed_max', 0.25))  # ★[H-FIX] 0.15→0.25
        if _vxy > _spd_max:
            return False, f"vxy={_vxy:.2f}>{_spd_max:.2f}"
        # 5. 最近确实看到过目标（★[H-FIX] terminal 时跳过，丢失也要降落）
        if not _terminal_locked:
            _seen_ago = 99.0
            if getattr(self, 'target_last_seen_time', None) is not None:
                _seen_ago = (self.get_clock().now() - self.target_last_seen_time).nanoseconds * 1e-9
            _max_seen_ago = float(getattr(self, 'target_track_timeout_abort', 6.0)) * 0.8
            if _seen_ago > _max_seen_ago:
                return False, f"seen_ago={_seen_ago:.1f}s>{_max_seen_ago:.1f}s"
        # 6. 局部障碍安全（★[H-FIX] terminal 时跳过，不因近距干扰阻断降落）
        if not _terminal_locked:
            if self.last_scan is not None:
                _ranges = np.array(self.last_scan.ranges)
                _valid = _ranges[(_ranges > 0.05) & (_ranges < 3.0)]
                if len(_valid) > 0 and float(np.min(_valid)) < 0.35:
                    return False, f"obstacle_near={float(np.min(_valid)):.2f}m<0.35m"
        return True, "all_ok"

    def _build_target_commit_point(self) -> Optional[np.ndarray]:
        """★[G6] 生成单一提交点。
        基于 target_world_xy_est，在目标方向上退让 target_commit_offset_m，
        得到一个可接受的接近点（允许停留在目标周围 1m 左右）。
        返回 [x, y] ENU 坐标，或 None（world_est 不可用时）。
        """
        w_xy = getattr(self, '_tgt_world_est_xy', None)
        if w_xy is None:
            return None
        px = float(getattr(self, '_plan_px', 0.0))
        py = float(getattr(self, '_plan_py', 0.0))
        dx = float(w_xy[0]) - px
        dy = float(w_xy[1]) - py
        dist = float(np.hypot(dx, dy))
        offset = float(getattr(self, 'target_commit_offset_m', 0.80))
        if dist > 0.1:
            # 退让距离：从无人机方向向目标靠近，停在 offset 处
            ratio = max(0.0, (dist - offset) / dist)
            cx = px + dx * ratio
            cy = py + dy * ratio
        else:
            # 已经很接近目标，直接用目标世界坐标
            cx, cy = float(w_xy[0]), float(w_xy[1])
        return np.array([cx, cy], dtype=float)

    def _check_conservative_landing_condition(self) -> tuple:
        """★[G6] 公开别名：委托给 _can_commit_conservative_descent()。
        返回 (ok: bool, reason: str)
        供外部调用或调试日志使用的标准化接口。
        """
        return self._can_commit_conservative_descent()

    def target_approach_velocity(self, now, z: float) -> np.ndarray:
        """[兼容保留] 转发到新方法 _target_approach_servo_cmd"""
        vel, _ = self._target_approach_servo_cmd(now, z)
        return vel

    def _target_approach_servo_cmd(self, now, z: float):
        """任务2/3/4：完整视觉伺服闭环 + 六级目标接近状态机。
        返回 (vel_enu_3d, yawspeed_rad_s)。
        子状态：
          'align'          — 仅偏航伺服，目标居中后→'forward'
          'forward'        — 缓慢前进 + 软偏航纠偏，面积≥approach_area→'hover_confirm'
          'hover_confirm'  — 悬停确认：目标居中 N 秒后→'descend'
          'descend'        — 视觉伺服+下降，高度/面积满足→'land'
          'land'           — 发送降落命令
          'lost_short'     — 短时丢失：悬停保持（Kalman 维持）
          'lost_sweep'     — 中等丢失：小角度扇扫恢复
        向后兼容：'approach' 等价于 'align' 处理。
        """
        import time as _ap_t
        _now_m = _ap_t.monotonic()

        # ────────────────────── 目标视觉数据 ──────────────────────────────────
        _seen_ago = 99.0
        if self.target_last_seen_time is not None:
            _seen_ago = (now - self.target_last_seen_time).nanoseconds * 1e-9

        # ★[E-ARCH] target_visible 阈值与 lost_short 对齐，消除1.0~1.5s的间隙
        # 此间隙会导致目标"不可见"但非"lost_short"，tgt_centered_since被重置，
        # 导致 locked_far→align 反复被打断而无法稳定升级
        _target_visible = (_seen_ago < self.target_track_timeout_short)  # 0.5->1.5s

        _dx = float(getattr(self, 'target_offset_x', 0.0))   # 归一化水平偏差 (-1~1)
        _dy = float(getattr(self, 'target_offset_y', 0.0))   # 归一化垂直偏差
        _area = float(getattr(self, 'target_area_ratio', 0.0))

        # ★[I7] TARGET-SERVO 周期日志（1s 节流）
        if _now_m - getattr(self, '_servo_tick_log_t', 0.0) > 1.0:
            self._servo_tick_log_t = _now_m
            _yaw_err_deg = _dx * 90.0  # 近似：dx*90≈pixel偏差折算角度（归一化后）
            self.get_logger().info(
                f"[TARGET-SERVO] inner={self._approach_inner_state} "
                f"dx={_dx:.3f} dy={_dy:.3f} area={_area:.4f} "
                f"yaw_err≈{_yaw_err_deg:.1f}° "
                f"visible={_target_visible}({_seen_ago:.1f}s ago) "
                f"confirmed={self.target_confirmed}"
            )

        # ────────────────────── 目标丢失分级 ──────────────────────────────────
        _lost_short = _seen_ago > self.target_track_timeout_short
        _lost_sweep = _seen_ago > self.target_track_timeout_sweep
        _lost_abort = _seen_ago > self.target_track_timeout_abort

        # ────────────────────── 向后兼容：'approach' → 'align' ───────────────
        if self._approach_inner_state == 'approach':
            self._approach_inner_state = 'align'
        _st = self._approach_inner_state

        # ────────────────────── 子状态迁移 ───────────────────────────────────
        if _target_visible:
            # 目标重新出现：从 lost_* 恢复
            if _st in ('lost_short', 'lost_sweep'):
                self.get_logger().info(
                    f"[TARGET-REACQUIRE] 目标重捕获 seen_ago={_seen_ago:.2f}s → 恢复 align"
                )
                self._log_event("TARGET-REACQUIRE", f"inner_st={_st} seen_ago={_seen_ago:.2f}s")
                self._tgt_lost_since = None
                self._sweep_yaw_start = None
                self._approach_inner_state = 'align'
                # ★[I6] align 重入日志（从lost恢复）
                self.get_logger().warn(
                    f"[TARGET-APPROACH-ALIGN-ENTER] ★ from={_st} "
                    f"area={_area:.4f} dx={_dx:.3f} dy={_dy:.3f}")
                _st = 'align'


            # locked_far → align：面积达到门槛且已居中 0.5s 以上
            if _st == 'locked_far':
                _lf_centered = (abs(_dx) < self.target_center_tol_x
                                and abs(_dy) < self.target_center_tol_y)
                if _lf_centered:
                    if not hasattr(self, '_lf_centered_since') or self._lf_centered_since is None:
                        self._lf_centered_since = _now_m
                else:
                    self._lf_centered_since = None
                _lf_dwell = _now_m - self._lf_centered_since if self._lf_centered_since else 0.0
                # FIX2: area gate removed -- centering dwell only
                if _lf_dwell >= 0.3:
                    self._approach_inner_state = 'align'
                    self._lf_centered_since = None
                    self._tgt_centered_since = None
                    self.get_logger().warn(
                        f"[TARGET-LOCKED-FAR-UPGRADE] ★ area={_area:.4f}>={self.target_approach_enter_area_ratio:.4f} "
                        f"centered={_lf_dwell:.2f}s → 升级到 align"
                    )
                    self._log_event("TARGET-LOCKED-FAR-UPGRADE",
                        f"area={_area:.4f} dx={_dx:.3f}")
                    # ★[I6] align 进入日志
                    self.get_logger().warn(
                        f"[TARGET-APPROACH-ALIGN-ENTER] ★ from=locked_far "
                        f"area={_area:.4f} dx={_dx:.3f} dy={_dy:.3f}")
                    _st = 'align'

            # 居中状态维护
            _centered = (abs(_dx) < self.target_center_tol_x
                         and abs(_dy) < self.target_center_tol_y)
            if _centered:
                if self._tgt_centered_since is None:
                    self._tgt_centered_since = _now_m
            else:
                self._tgt_centered_since = None
            _centered_dwell = (_now_m - self._tgt_centered_since
                               if self._tgt_centered_since else 0.0)

            # align → forward：居中停留 0.3s 以上
            if _st == 'align' and _centered_dwell >= 0.3:
                self._approach_inner_state = 'forward'
                self.get_logger().info(
                    f"[TARGET-ALIGN-DONE] 目标居中 {_centered_dwell:.2f}s → 进入 forward"
                )
                self._log_event("TARGET-ALIGN-DONE",
                    f"centered_dwell={_centered_dwell:.2f}s dx={_dx:.3f}")
                # ★[I6] forward 进入日志
                self.get_logger().warn(
                    f"[TARGET-APPROACH-FORWARD-ENTER] ★ from=align "
                    f"area={_area:.4f} dx={_dx:.3f} dy={_dy:.3f}")
                _st = 'forward'

            # ★[G6] forward → world_establish（无论world_est是否可用，先积累估计）
            if _st == 'forward':
                if (_area >= self.target_approach_area_ratio
                        and _centered_dwell >= self.target_center_dwell_sec):
                    self._approach_inner_state = 'world_establish'
                    self._tgt_world_establish_enter_t = _now_m
                    self._tgt_commit_goal = None
                    self.get_logger().warn(
                        f"[TARGET-WORLD-ESTABLISH-ENTER] ★[G6] area={_area:.3f}>="
                        f"{self.target_approach_area_ratio:.3f} → 进入世界坐标建立"
                    )
                    self._log_target_event("TARGET-WORLD-ESTABLISH",
                        f"area={_area:.3f} dx={_dx:.3f}",
                        world_xy=getattr(self, '_tgt_world_est_xy', None),
                        conf=self._tgt_world_est_conf, trigger="forward_area_ok")
                    _st = 'world_establish'

            # hover_confirm → descend 或 退回 forward
            if _st == 'hover_confirm':
                _hc_dur = _now_m - (self._hover_confirm_enter_t or _now_m)
                if _hc_dur >= self.target_hover_confirm_sec:
                    self._approach_inner_state = 'descend'
                    self.get_logger().warn(
                        f"[TARGET-DESCEND-ENTER] ★ 悬停确认{_hc_dur:.1f}s → 进入下降"
                    )
                    self._log_event("TARGET-DESCEND-ENTER",
                        f"area={_area:.3f} z={z:.3f}")
                    _st = 'descend'
                elif not _centered:
                    self._approach_inner_state = 'forward'
                    self._hover_confirm_enter_t = None
                    # ★[I6] forward 重入日志（从hover_confirm退回）
                    self.get_logger().info(
                        f"[TARGET-APPROACH-FORWARD-ENTER] from=hover_confirm_decentered "
                        f"area={_area:.4f} dx={_dx:.3f}")
                    _st = 'forward'

            # ★[G6] world_establish → commit_nav：估计建立够稳定后生成单一提交点
            if _st == 'world_establish':
                _est_dur = _now_m - (self._tgt_world_establish_enter_t or _now_m)
                _est_conf = float(getattr(self, '_tgt_world_est_conf', 0.0))
                _est_var = float(getattr(self, '_tgt_world_xy_var', 999.0))
                _establish_sec = float(getattr(self, 'target_world_establish_sec', 1.5))
                _min_conf_est = float(getattr(self, 'target_commit_conf_min', 0.55)) * 0.6
                if _est_dur >= _establish_sec and _est_conf >= _min_conf_est:
                    _cp = self._build_target_commit_point()
                    if _cp is not None:
                        self._tgt_commit_goal = _cp
                        self._approach_inner_state = 'commit_nav'
                        self.get_logger().warn(
                            f"[TARGET-COMMIT-POINT] ★[G6] est={_est_dur:.1f}s "
                            f"conf={_est_conf:.2f} var={_est_var:.3f} "
                            f"→ CP=({_cp[0]:.2f},{_cp[1]:.2f})"
                        )
                        self._log_target_event("TARGET-COMMIT-POINT",
                            f"cp=({_cp[0]:.2f},{_cp[1]:.2f}) est={_est_dur:.1f}s",
                            world_xy=_cp, conf=_est_conf, trigger="world_established")
                        _st = 'commit_nav'

            # ★[H-FIX-D] commit_nav 到达检查：在可见分支也运行，避免漏检
            if _st == 'commit_nav':
                _cp_vis_ck = getattr(self, '_tgt_commit_goal', None)
                if _cp_vis_ck is not None:
                    _d_cp_vis = float(np.hypot(
                        float(getattr(self, '_plan_px', 0.0)) - _cp_vis_ck[0],
                        float(getattr(self, '_plan_py', 0.0)) - _cp_vis_ck[1]
                    ))
                    if _d_cp_vis < float(getattr(self, 'target_commit_reach_dist', 0.40)):
                        self._approach_inner_state = 'commit_hover'
                        self._tgt_commit_hover_enter_t = _now_m
                        self._conservative_descend_enter_t = None  # 重置计时器
                        self.get_logger().warn(
                            f"[TARGET-HOVER-COMMIT] ★[H-FIX-D] 可见分支到达CP "
                            f"dist={_d_cp_vis:.2f}m → 悬停确认"
                        )
                        self._log_target_event("TARGET-HOVER-COMMIT",
                            f"dist={_d_cp_vis:.2f}m seen_visible=True",
                            world_xy=_cp_vis_ck, conf=self._tgt_world_est_conf,
                            trigger="commit_reached_visible")
                        _st = 'commit_hover'

            # ★[G6] commit_hover → conservative_descend：悬停足够后直接进入
            if _st == 'commit_hover':
                _hov_dur = _now_m - (self._tgt_commit_hover_enter_t or _now_m)
                if _hov_dur >= float(getattr(self, 'target_commit_hover_sec', 2.0)):
                    _can_c, _why_c = self._can_commit_conservative_descent()
                    # ★[H-FIX-C] terminal_locked 时跳过保守条件门控，直接进入降落
                    _force_enter = getattr(self, 'target_terminal_locked', False)
                    if _can_c or _target_visible or _force_enter:
                        self._approach_inner_state = 'conservative_descend'
                        self._conservative_descend_enter_t = None  # 重置计时器
                        self.get_logger().warn(
                            f"[TARGET-CONSERVATIVE-DESCEND-ENTER] ★[G6] hover={_hov_dur:.1f}s "
                            f"can_commit={_can_c} visible={_target_visible} force={_force_enter} → 保守下降"
                        )
                        self._log_target_event("TARGET-CONSERVATIVE-DESCEND",
                            f"hover={_hov_dur:.1f}s can_commit={_can_c} force={_force_enter}",
                            world_xy=getattr(self, '_tgt_world_est_xy', None),
                            conf=self._tgt_world_est_conf, trigger="commit_hover_done")
                        _st = 'conservative_descend'

        else:
            # 目标不可见
            # ★[G6] world_establish/commit_nav/commit_hover/conservative_descend
            # 不依赖实时视觉，不应被目标丢失中断
            if _st not in ('lost_short', 'lost_sweep', 'land',
                           'world_establish', 'commit_nav', 'commit_hover', 'conservative_descend'):
                if self._tgt_lost_since is None:
                    self._tgt_lost_since = _now_m
                if _lost_abort:
                    if getattr(self, 'target_terminal_locked', False):
                        # ★[A5] 终端锁定：即使abort超时也不中断，强制切入 conservative_descend
                        import time as _sab_t
                        self.get_logger().warn(
                            f"[TARGET-SERVO-ABORT-TERMINAL] ★[A5] 终端锁定 seen_ago={_seen_ago:.1f}s "
                            f"inner={_st} → 强制 conservative_descend (不回搜索)"
                        )
                        if getattr(self, '_tgt_commit_goal', None) is None:
                            _wxy_sa = getattr(self, '_tgt_world_est_xy', None)
                            if _wxy_sa is not None:
                                self._tgt_commit_goal = _wxy_sa.copy()
                        self._approach_inner_state = 'conservative_descend'
                        self._tgt_commit_hover_enter_t = _sab_t.monotonic()
                        _st = 'conservative_descend'
                    else:
                        return np.zeros(3, dtype=float), 0.0
                elif _lost_sweep and _st != 'lost_sweep':
                    self._approach_inner_state = 'lost_sweep'
                    self._sweep_yaw_start = float(self.att_yaw) if self.att_yaw else 0.0
                    self._sweep_dir = 1.0
                    self._sweep_angle_acc = 0.0
                    self.get_logger().warn(
                        f"[TARGET-LOST-SWEEP] 中等丢失{_seen_ago:.1f}s → 开始扇扫恢复"
                    )
                    # ★[I8] TARGET-LOST-SCAN 标准化别名
                    self.get_logger().warn(
                        f"[TARGET-LOST-SCAN] seen_ago={_seen_ago:.1f}s → 扇扫模式激活"
                    )
                    self._log_event("TARGET-LOST-SWEEP",
                        f"sweep=True seen_ago={_seen_ago:.1f}s")
                    _st = 'lost_sweep'
                elif _lost_short and _st not in ('lost_short', 'lost_sweep'):
                    self._approach_inner_state = 'lost_short'
                    self.get_logger().info(
                        f"[TARGET-LOST-SHORT] 短时丢失{_seen_ago:.1f}s → 悬停保持"
                    )
                    self._log_event("TARGET-LOST-SHORT",
                        f"hover=True seen_ago={_seen_ago:.1f}s")
                    _st = 'lost_short'
            if _st == 'descend':
                # 下降中丢失：暂停 descend_pause_on_miss 帧，再退回 hover_confirm
                self._descend_pause_cnt = getattr(self, '_descend_pause_cnt', 0) + 1
                if self._descend_pause_cnt >= self.target_descend_pause_on_miss:
                    self._descend_pause_cnt = 0
                    self._approach_inner_state = 'hover_confirm'
                    self._hover_confirm_enter_t = _now_m
                    self.get_logger().warn(
                        "[TARGET-DESCEND-ABORT] 下降中目标丢失 → 退回悬停确认"
                    )
                    self._log_event("TARGET-DESCEND-ABORT",
                        f"seen_ago={_seen_ago:.2f}s z={z:.3f}")
                    _st = 'hover_confirm'

            # ★[G6][A4] 不可见时 commit_hover 保守降落判断
            if _st == 'commit_hover':
                _hov_dur_nv = _now_m - (self._tgt_commit_hover_enter_t or _now_m)
                _hover_sec = float(getattr(self, 'target_commit_hover_sec', 2.0))
                if _hov_dur_nv >= _hover_sec:
                    _can_c_nv, _why_c_nv = self._can_commit_conservative_descent()
                    # ★[规则3] 终端锁定：不管保守条件，目标丢失也直接降落
                    _force_nv = getattr(self, 'target_terminal_locked', False)
                    if _can_c_nv or _force_nv:
                        self._approach_inner_state = 'conservative_descend'
                        _why_str = _why_c_nv if _can_c_nv else "terminal_locked_force"
                        self.get_logger().warn(
                            f"[TARGET-CONSERVATIVE-DESCEND-ENTER] ★[G6][A4] "
                            f"目标不可见 hover={_hov_dur_nv:.1f}s force={_force_nv} → 保守降落 reason={_why_str}"
                        )
                        self._log_target_event("TARGET-CONSERVATIVE-DESCEND",
                            f"conservative=True hover={_hov_dur_nv:.1f}s force={_force_nv}",
                            world_xy=getattr(self, '_tgt_world_est_xy', None),
                            conf=self._tgt_world_est_conf,
                            trigger="conservative_commit_hover")
                        _st = 'conservative_descend'
                    else:
                        # 保守条件不满足：回 commit_nav 重新导航
                        _cp_bk = getattr(self, '_tgt_commit_goal', None)
                        if _cp_bk is not None:
                            self._approach_inner_state = 'commit_nav'
                            self.get_logger().warn(
                                f"[TARGET-COMMIT-NAV-BACK] ★[G6] 保守条件不满足: {_why_c_nv} → 回 commit_nav")
                            _st = 'commit_nav'
                        else:
                            self._approach_inner_state = 'world_establish'
                            self._tgt_world_establish_enter_t = _now_m
                            _st = 'world_establish'

            # ★[G6] commit_nav 到达检查（基于位置，与视觉无关）
            if _st == 'commit_nav':
                _cp_ck = getattr(self, '_tgt_commit_goal', None)
                if _cp_ck is not None:
                    _d_cp = float(np.hypot(
                        float(getattr(self, '_plan_px', 0.0)) - _cp_ck[0],
                        float(getattr(self, '_plan_py', 0.0)) - _cp_ck[1]
                    ))
                    if _d_cp < float(getattr(self, 'target_commit_reach_dist', 0.40)):
                        self._approach_inner_state = 'commit_hover'
                        self._tgt_commit_hover_enter_t = _now_m
                        self._conservative_descend_enter_t = None  # ★[H-FIX] 重置计时器
                        self.get_logger().warn(
                            f"[TARGET-HOVER-COMMIT] ★[G6] 到达提交点 dist={_d_cp:.2f}m → 悬停确认"
                        )
                        self._log_target_event("TARGET-HOVER-COMMIT",
                            f"dist={_d_cp:.2f}m seen={_seen_ago:.1f}s",
                            world_xy=_cp_ck, conf=self._tgt_world_est_conf,
                            trigger="commit_reached")
                        _st = 'commit_hover'

        # ────────────────────── 控制输出 ──────────────────────────────────────
        _st = self._approach_inner_state
        _vx_body = 0.0
        _vy_body = 0.0
        _vz = 0.0
        _yawspeed = 0.0

        if _st == 'lost_short':
            # ★[F] 短时丢失：零速悬停 + Kalman预测软偏航（使相机持续对准预测位置加速重捕获）
            _kf_conf_ls = float(self.target_track_conf)
            if (self._kf_target.initialized
                    and _kf_conf_ls > 0.20
                    and _seen_ago <= self.target_prediction_max_hold_sec):
                # 用Kalman预测的归一化水平坐标（cx_norm:0~1，中心=0.5）折算偏航误差
                _kf_ls_pred = self._kf_target.predict(dt=1.0)
                _kfcx_ls = float(_kf_ls_pred[0])  # 归一化cx（-0.5~0.5，中心=0）
                # cx_norm已是以图像中心为0；直接作为偏航误差（正=目标在右→顺时针）
                _yaw_err_ls = _kfcx_ls
                # 软偏航：幅度=正常伺服×0.4（不过激，以免遮挡时乱转）
                _yawspeed = float(np.clip(
                    -self.target_yaw_kp * _yaw_err_ls * 0.4,
                    -self.target_yaw_max * 0.5, self.target_yaw_max * 0.5
                ))
                if _now_m - getattr(self, '_lost_track_log_t', 0.0) > 0.5:
                    self._lost_track_log_t = _now_m
                    self.get_logger().info(
                        f"[TARGET-LOST-TRACK] ★[F] Kalman预测引导偏航 "
                        f"kf_cx={_kfcx_ls:.3f} yaw_err={_yaw_err_ls:.3f} "
                        f"yawspeed={_yawspeed:.3f} kf_conf={_kf_conf_ls:.2f} "
                        f"seen_ago={_seen_ago:.2f}s"
                    )

            # ★[G4] 视觉世界坐标辅助：若有稳定世界坐标估计 → 切换到 lost_nav 子状态
            # 让 local_mpc 接管导航，向 visual_target_goal 靠近，再次被视觉捕获
            import time as _vtg_t
            _vtg_now = _vtg_t.monotonic()
            _twe_conf  = float(getattr(self, '_tgt_world_est_conf', 0.0))
            _twe_xy    = getattr(self, '_tgt_world_est_xy', None)
            _twe_last  = float(getattr(self, '_tgt_world_est_last_t', 0.0))
            _twe_fresh = (_twe_last > 0.0 and _vtg_now - _twe_last < 6.0)
            _min_conf  = float(getattr(self, 'target_world_est_min_conf', 0.50))
            if (_twe_fresh and _twe_conf >= _min_conf
                    and _twe_xy is not None
                    and getattr(self, 'pose', None) is not None):
                # 设置世界坐标导航目标
                _vtg_goal = getattr(self, '_visual_target_goal', None)
                if _vtg_goal is None:
                    self._visual_target_goal = _twe_xy.copy()
                    self._visual_target_goal_set_t = _vtg_now
                    self.get_logger().warn(
                        f"[TARGET-WORLD-GOAL-SET] ★[G4] 短时丢失+世界坐标稳定 "
                        f"goal=({_twe_xy[0]:.2f},{_twe_xy[1]:.2f}) "
                        f"conf={_twe_conf:.2f} → 切换 lost_nav 主动靠近"
                    )
                    self._log_event("TARGET-WORLD-GOAL-SET",
                        f"goal=({_twe_xy[0]:.2f},{_twe_xy[1]:.2f}) conf={_twe_conf:.2f}")
                # 将世界坐标设为 SSG 视点目标，local_mpc 接管导航
                self.ssg_viewpoint_goal = self._visual_target_goal.copy()
                self.ssg_goal_type = 'visual_target_world'
                self.astar_path = None
                self.astar_path_idx = 0
                self.astar_last_plan_time = 0.0
                # 切到 lost_nav 子状态（local_mpc 导航，不归零速度）
                self._approach_inner_state = 'lost_nav'
                _st = 'lost_nav'
                if _now_m - getattr(self, '_lost_nav_log_t', 0.0) > 2.0:
                    self._lost_nav_log_t = _now_m
                    self.get_logger().info(
                        f"[TARGET-LOST-NAV] ★[G4] → local_mpc 向 world_goal 导航 "
                        f"goal=({self._visual_target_goal[0]:.2f},{self._visual_target_goal[1]:.2f}) "
                        f"drone=({self._plan_px:.2f},{self._plan_py:.2f})"
                    )

        elif _st == 'lost_nav':
            # ★[G4] 世界坐标导航子状态：目标可见时立即恢复接近链，保持 yaw 伺服
            _vtg_now2 = getattr(self, '_vtg_set_check_t', 0.0)
            import time as _lnav_t2
            _lnav_now2 = _lnav_t2.monotonic()
            _vtg_active = (getattr(self, '_visual_target_goal', None) is not None)
            _vtg_elapsed = _lnav_now2 - float(getattr(self, '_visual_target_goal_set_t', 0.0))
            # 超时清除世界坐标目标
            if _vtg_elapsed > float(getattr(self, 'target_world_goal_hold_sec', 8.0)):
                _old = self._visual_target_goal
                self._visual_target_goal = None
                if _old is not None:
                    self.get_logger().info(
                        f"[TARGET-WORLD-GOAL-CLEAR] reason=timeout "
                        f"elapsed={_vtg_elapsed:.1f}s "
                        f"old=({_old[0]:.2f},{_old[1]:.2f}) → 切回 lost_sweep"
                    )
                    self._log_event("TARGET-WORLD-GOAL-CLEAR", f"reason=timeout")
                self._approach_inner_state = 'lost_sweep'
            elif _target_visible:
                # 目标重新可见 → 立即切回 align 逼近
                self._visual_target_goal = None
                self.ssg_viewpoint_goal = None
                self.astar_path = None
                self._approach_inner_state = 'align'
                self._tgt_centered_since = None
                self.get_logger().warn(
                    f"[TARGET-REACQUIRED-G4] ★[G4] world_est导航后目标重新可见 "
                    f"dx={_dx:.3f} area={_area:.4f} → ALIGN"
                )
                self._log_event("TARGET-REACQUIRED-G4",
                    f"dx={_dx:.3f} area={_area:.4f}")
            # 在 lost_nav 期间持续轻微 yaw 伺服（保持相机大概朝向目标方向）
            if _st == 'lost_nav' and self._kf_target.initialized:
                _kf_pred_lnav = self._kf_target.predict(dt=1.0)
                _kfcx_lnav = float(_kf_pred_lnav[0])
                _yawspeed = float(np.clip(
                    -self.target_yaw_kp * _kfcx_lnav * 0.3,
                    -self.target_yaw_max * 0.4, self.target_yaw_max * 0.4
                ))

        elif _st == 'lost_sweep':
            # 扇扫：小幅 yaw 摆动寻找目标（±30度）
            _sweep_spd = float(self.target_yaw_max * 0.5)
            _yawspeed = self._sweep_dir * _sweep_spd
            _da = abs(_sweep_spd * 0.05)  # 假设50ms周期
            self._sweep_angle_acc = getattr(self, '_sweep_angle_acc', 0.0) + _da
            if self._sweep_angle_acc > 0.52:  # ~30度
                self._sweep_dir *= -1.0
                self._sweep_angle_acc = 0.0

        elif _st == 'locked_far':
            # 已确认目标但面积不足：仅偏航伺服使目标居中，不前进
            _yaw_err = _dx
            _yawspeed = float(np.clip(
                -self.target_yaw_kp * _yaw_err,
                -self.target_yaw_max, self.target_yaw_max
            ))
            # FIX3: slow forward when yaw-aligned breaks area-deadlock
            if abs(_dx) < self.target_center_tol_x * 2.0 and _target_visible:
                _vx_body = self.target_forward_speed_min  # ~0.05 m/s
            if _now_m - getattr(self, '_locked_far_log_t', 0.0) > 1.0:
                self._locked_far_log_t = _now_m
                self.get_logger().info(
                    f'[TARGET-LOCKED-FAR] yaw_err={_yaw_err:.3f} '
                    f'yawspeed={_yawspeed:.3f} '
                    f'area={_area:.4f} need={self.target_approach_enter_area_ratio:.4f}'
                )

        elif _st in ('align', 'approach'):
            # 阶段A：仅 yaw P 控制让目标居中，不前进
            _yaw_err = _dx  # 正=目标在右→需顺时针yaw
            _yawspeed = float(np.clip(
                -self.target_yaw_kp * _yaw_err,
                -self.target_yaw_max, self.target_yaw_max
            ))
            if _now_m - getattr(self, '_align_log_t', 0.0) > 1.0:
                self._align_log_t = _now_m
                self.get_logger().info(
                    f"[TARGET-ALIGN] dx={_dx:.3f} yaw_err={_yaw_err:.3f} "
                    f"yawspeed={_yawspeed:.3f} area={_area:.4f}"
                )

        elif _st == 'forward':
            # 阶段B：居中已确认，缓慢前进 + 软 yaw 纠偏
            _yaw_err = _dx
            _yawspeed = float(np.clip(
                -self.target_yaw_kp * _yaw_err * 0.6,
                -self.target_yaw_max * 0.7, self.target_yaw_max * 0.7
            ))
            # 前进速度：面积越近越慢
            _area_ratio_clamp = min(1.0, _area / max(self.target_approach_area_ratio, 0.001))
            _fwd_spd = float(np.interp(
                _area_ratio_clamp,
                [0.0, 0.8, 1.0],
                [self.target_forward_speed_max, self.target_forward_speed_min, 0.0]
            ))
            _vx_body = float(np.clip(_fwd_spd, 0.0, self.target_forward_speed_max))
            # 侧向纠偏（小股 vy 补偿横向偏差）
            if abs(_dx) > self.target_center_tol_x * 1.5:
                _vy_body = float(np.clip(
                    -self.target_align_kp * _dx * 0.4,
                    -self.target_align_max * 0.5, self.target_align_max * 0.5
                ))
            if _now_m - getattr(self, '_servo_log_t', 0.0) > 1.0:
                self._servo_log_t = _now_m
                self.get_logger().info(
                    f"[TARGET-SERVO-FWD] dx={_dx:.3f} area={_area:.3f} "
                    f"yawspeed={_yawspeed:.3f} vx={_vx_body:.2f} vy={_vy_body:.2f}"
                )

        elif _st == 'hover_confirm':
            # 悬停确认：零速，保持 yaw 伺服
            _yaw_err = _dx
            _yawspeed = float(np.clip(
                -self.target_yaw_kp * _yaw_err * 0.5,
                -self.target_yaw_max * 0.3, self.target_yaw_max * 0.3
            ))
            _hc_rem = self.target_hover_confirm_sec - (
                _now_m - (self._hover_confirm_enter_t or _now_m)
            )
            if _now_m - getattr(self, '_hover_conf_log_t', 0.0) > 0.5:
                self._hover_conf_log_t = _now_m
                self.get_logger().info(
                    f"[TARGET-HOVER-CONFIRM] 确认中 剩余{_hc_rem:.1f}s "
                    f"area={_area:.3f} dx={_dx:.3f}"
                )

        elif _st == 'descend':
            # 下降阶段：视觉伺服 + vz 下降
            _yaw_err = _dx
            _yawspeed = float(np.clip(
                -self.target_yaw_kp * _yaw_err * 0.6,
                -self.target_yaw_max * 0.5, self.target_yaw_max * 0.5
            ))
            _vy_body = float(np.clip(
                -self.target_align_kp * _dx * 0.5,
                -self.target_align_max, self.target_align_max
            ))
            _vz = float(np.clip(
                self.target_descend_speed, 0.0, self.alt_max_descend_speed
            ))
            if _now_m - getattr(self, '_descend_log_t', 0.0) > 1.0:
                self._descend_log_t = _now_m
                self.get_logger().info(
                    f"[TARGET-DESCEND] z={z:.3f} area={_area:.3f} "
                    f"vz={_vz:.3f} dx={_dx:.3f}"
                )
            # 触发 land：面积 ≥ land_area_ratio 或 高度 ≤ land_z
            _land_by_area = _area >= getattr(self, 'target_land_trigger_area_ratio',
                                              self.target_land_area_ratio)
            _land_by_height = z <= getattr(self, 'target_land_trigger_height',
                                            self.target_land_z)
            if _land_by_area or _land_by_height:
                self._approach_inner_state = 'land'
                _st = 'land'
                self.get_logger().warn(
                    f"[TARGET-LAND-TRIGGER] ★ area={_area:.3f} z={z:.3f} "
                    f"by_area={_land_by_area} by_height={_land_by_height}"
                )
                self._log_event("TARGET-LAND-TRIGGER",
                    f"area={_area:.3f} z={z:.3f}")
                return np.zeros(3, dtype=float), 0.0

        elif _st == 'land':
            return np.zeros(3, dtype=float), 0.0

        elif _st in ('commit_nav', 'world_establish'):
            # ★[G6] commit_nav P控制飞向CP；world_establish零速悬停积累估计
            import math as _snav_m
            _px_nav = float(getattr(self, '_plan_px', 0.0))
            _py_nav = float(getattr(self, '_plan_py', 0.0))
            _vx_enu_nav, _vy_enu_nav = 0.0, 0.0
            if _st == 'commit_nav':
                _snav_goal = getattr(self, '_tgt_commit_goal', None)
                if _snav_goal is not None:
                    _dpx_nav = _snav_goal[0] - _px_nav
                    _dpy_nav = _snav_goal[1] - _py_nav
                    _dist_nav = float(_snav_m.hypot(_dpx_nav, _dpy_nav))
                    _spd_nav = min(float(getattr(self, 'target_stage_nav_speed', 0.20)),
                                   _dist_nav * 0.8)
                    if _dist_nav > 0.05:
                        _vx_enu_nav = (_dpx_nav / _dist_nav) * _spd_nav
                        _vy_enu_nav = (_dpy_nav / _dist_nav) * _spd_nav
                    if _now_m - getattr(self, '_snav_log_t', 0.0) > 1.0:
                        self._snav_log_t = _now_m
                        self.get_logger().info(
                            f"[TARGET-COMMIT-NAV] ★[G6] CP=({_snav_goal[0]:.2f},{_snav_goal[1]:.2f}) "
                            f"drone=({_px_nav:.2f},{_py_nav:.2f}) "
                            f"venu=({_vx_enu_nav:.2f},{_vy_enu_nav:.2f})"
                        )
            else:
                # world_establish: 悬停积累估计，每0.5s打印状态
                if _now_m - getattr(self, '_west_log_t', 0.0) > 0.5:
                    self._west_log_t = _now_m
                    _est_d = _now_m - (self._tgt_world_establish_enter_t or _now_m)
                    self.get_logger().info(
                        f"[TARGET-WORLD-ESTABLISH] ★[G6] dur={_est_d:.1f}s "
                        f"conf={self._tgt_world_est_conf:.2f} "
                        f"var={getattr(self, '_tgt_world_xy_var', 999.0):.3f}"
                    )
            # 软 yaw 伺服朝向世界目标估计方向
            _wxy_nav = getattr(self, '_tgt_world_est_xy', None)
            if _wxy_nav is not None:
                _wdx = _wxy_nav[0] - _px_nav
                _wdy = _wxy_nav[1] - _py_nav
                _w_yaw_nav = float(_snav_m.atan2(_wdy, _wdx))
                _cur_yaw_nav = float(getattr(self, 'att_yaw', _w_yaw_nav) or _w_yaw_nav)
                _yaw_err_nav = _snav_m.atan2(
                    _snav_m.sin(_w_yaw_nav - _cur_yaw_nav),
                    _snav_m.cos(_w_yaw_nav - _cur_yaw_nav)
                )
                _yawspeed = float(np.clip(
                    self.target_yaw_kp * _yaw_err_nav * 0.5,
                    -self.target_yaw_max * 0.6, self.target_yaw_max * 0.6
                ))
            elif _target_visible:
                _yawspeed = float(np.clip(
                    -self.target_yaw_kp * _dx * 0.4,
                    -self.target_yaw_max * 0.4, self.target_yaw_max * 0.4
                ))
            self._servo_yawspeed = _yawspeed
            return np.array([_vx_enu_nav, _vy_enu_nav, 0.0], dtype=float), _yawspeed

        elif _st == 'commit_hover':
            # ★[G6] 提交点悬停：零速 + 软 yaw 伺服保持朝向
            if _target_visible:
                _yawspeed = float(np.clip(
                    -self.target_yaw_kp * _dx * 0.4,
                    -self.target_yaw_max * 0.3, self.target_yaw_max * 0.3
                ))
            _hov_dur_log = _now_m - (self._tgt_commit_hover_enter_t or _now_m)
            if _now_m - getattr(self, '_shov_log_t', 0.0) > 0.5:
                self._shov_log_t = _now_m
                self.get_logger().info(
                    f"[TARGET-COMMIT-HOVER] ★[G6] dur={_hov_dur_log:.1f}s "
                    f"area={_area:.3f} w_conf={self._tgt_world_est_conf:.2f} "
                    f"seen={_seen_ago:.1f}s"
                )

        elif _st == 'conservative_descend':
            # ★[G6][A4][H-FIX-B] 保守下降：终端任务锁定后立即执行降落
            # 记录进入时间（首次计时）
            import time as _cde_t
            if getattr(self, '_conservative_descend_enter_t', None) is None:
                self._conservative_descend_enter_t = _cde_t.monotonic()
            _cd_elapsed = _cde_t.monotonic() - self._conservative_descend_enter_t

            _terminal_locked = getattr(self, 'target_terminal_locked', False)
            _can_dc, _why_dc = self._can_commit_conservative_descent()
            # ★[H-FIX-B] 终端锁定时：0.5s 计时后无条件触发 LAND
            # （_can_dc 在 terminal 模式已放宽，此时几乎必然 True；保留 0.5s 是防零帧竞态）
            _terminal_force_land = (
                _terminal_locked
                and not _target_visible
                and _seen_ago > float(getattr(self, 'target_commit_hover_sec', 2.0))
            )
            _terminal_commit_land = _terminal_locked and _cd_elapsed >= 0.5

            if _can_dc or _terminal_force_land or _terminal_commit_land:
                if _target_visible:
                    _yawspeed = float(np.clip(
                        -self.target_yaw_kp * _dx * 0.5,
                        -self.target_yaw_max * 0.4, self.target_yaw_max * 0.4
                    ))
                _vz = float(np.clip(
                    self.target_descend_speed, 0.0, self.alt_max_descend_speed
                ))
                if (_terminal_force_land or _terminal_commit_land) and not _can_dc:
                    if _now_m - getattr(self, '_tfl_log_t', 0.0) > 2.0:
                        self._tfl_log_t = _now_m
                        self.get_logger().warn(
                            f"[TARGET-DESCEND-EXEC] ★[H-FIX-B] 终端锁定 "
                            f"elapsed={_cd_elapsed:.1f}s seen_ago={_seen_ago:.1f}s "
                            f"→ 强制降落 z={z:.3f} (why_dc={_why_dc})"
                        )
                        self._log_event("TARGET-DESCEND-EXEC",
                            f"force=True z={z:.3f} elapsed={_cd_elapsed:.1f}s why_dc={_why_dc}")
                # ★[H-FIX-B] LAND 触发：面积 OR 高度 OR 终端计时（任一满足即降落）
                _land_by_area_dc = _area >= getattr(self, 'target_land_trigger_area_ratio', 0.20)
                _land_by_height_dc = z <= getattr(self, 'target_land_trigger_height', -0.25)
                # ★[H-FIX-B] 终端锁定：0.5s 后无条件触发 LAND（不依赖面积/高度）
                _land_by_terminal_commit = _terminal_commit_land
                if _land_by_area_dc or _land_by_height_dc or _land_by_terminal_commit:
                    self._approach_inner_state = 'land'
                    _trigger_reason = ("area" if _land_by_area_dc else
                                       ("height" if _land_by_height_dc else "terminal_commit"))
                    self.get_logger().warn(
                        f"[TARGET-LAND-TRIGGER] ★[H-FIX-B] conservative_descend "
                        f"reason={_trigger_reason} area={_area:.3f} z={z:.3f} "
                        f"elapsed={_cd_elapsed:.1f}s"
                    )
                    self._log_target_event("TARGET-LAND",
                        f"area={_area:.3f} z={z:.3f} reason={_trigger_reason}",
                        world_xy=getattr(self, '_tgt_world_est_xy', None),
                        conf=self._tgt_world_est_conf, trigger="conservative_descend")
                    self._log_event("TARGET-LAND",
                        f"area={_area:.3f} z={z:.3f} reason={_trigger_reason}")
                    return np.zeros(3, dtype=float), 0.0
            else:
                # 条件恶化 → 暂停下降
                _vz = 0.0
                if _now_m - getattr(self, '_dc_pause_log_t', 0.0) > 2.0:
                    self._dc_pause_log_t = _now_m
                    self.get_logger().warn(
                        f"[TARGET-CONSERVATIVE-DESCEND-PAUSE] ★[G6] 条件未满足: {_why_dc}"
                    )
            if _now_m - getattr(self, '_dc_log_t', 0.0) > 1.0:
                self._dc_log_t = _now_m
                self.get_logger().info(
                    f"[TARGET-CONSERVATIVE-DESCEND] can={_can_dc} z={z:.3f} "
                    f"vz={_vz:.3f} elapsed={_cd_elapsed:.1f}s terminal_commit={_terminal_commit_land} "
                    f"conf={self._tgt_world_est_conf:.2f}"
                )

        # ─── 事件日志：居中稳定、足够近 ────────────────────────────────────
        _centered2 = (abs(_dx) < self.target_center_tol_x)
        _cd2 = (_now_m - self._tgt_centered_since) if self._tgt_centered_since else 0.0
        if _st in ('align', 'approach') and _centered2:
            if _cd2 >= self.target_center_dwell_sec and not getattr(self, '_centered_logged', False):
                self._centered_logged = True
                self.get_logger().info(
                    f"[TARGET-CENTERED] 居中稳定{_cd2:.1f}s area={_area:.3f} → 准备前进"
                )
                self._log_event("TARGET-CENTERED", f"dwell={_cd2:.1f}s area={_area:.3f}")
        else:
            if _st not in ('align', 'approach'):
                self._centered_logged = False

        if _area >= self.target_approach_area_ratio and _st == 'forward':
            if not getattr(self, '_close_enough_logged', False):
                self._close_enough_logged = True
                self.get_logger().info(
                    f"[TARGET-CLOSE-ENOUGH] area={_area:.3f}>="
                    f"{self.target_approach_area_ratio:.3f}"
                )
                self._log_event("TARGET-CLOSE-ENOUGH", f"area={_area:.3f}")
        elif _st != 'forward':
            self._close_enough_logged = False

        # 速度从 body 系转到 ENU 发布系
        if self.att_rot is None:
            return np.array([0.0, 0.0, _vz], dtype=float), _yawspeed
        v_ned = self.att_rot.apply(np.array([_vx_body, _vy_body, 0.0], dtype=float))
        self._servo_yawspeed = _yawspeed
        return np.array([float(v_ned[1]), float(v_ned[0]), float(_vz)], dtype=float), _yawspeed


    def is_cmd_fresh(self, stamp, now, timeout: float = 0.35) -> bool:
        if stamp is None:
            return False
        return (now - stamp).nanoseconds * 1e-9 <= timeout

    def scan_is_recent(self, now) -> bool:
        if self.last_scan_time is None:
            return False
        return (now - self.last_scan_time).nanoseconds * 1e-9 <= self.lidar_timeout

    # ===================================================================
    # Frontier 探索：基于占据栅格地图的前沿目标选择
    # ===================================================================
    # 房间探索状态机辅助方法
    # ===================================================================

    def get_current_room_id(self) -> int:
        """根据无人机当前位置查询所在房间ID（使用 room_labels 栅格）。"""
        if self.pose is None or self.room_labels is None:
            return -1
        # ★ plan frame 坐标访问 room_labels（plan 网格）
        pix, piy = self.world_to_grid(self._plan_px, self._plan_py)
        lid = int(self.room_labels[pix, piy])
        for r in self.rooms_info:
            if r["label"] == lid:
                return int(r["id"])
        return -1

    def get_room_coverage(self, room_id: int) -> float:
        """计算指定房间已探索自由格占（自由+未知）的比例。"""
        if not self.rooms_info:
            return 0.0
        room = next((r for r in self.rooms_info if r["id"] == room_id), None)
        if room is None:
            return 0.0
        lid = room["label"]
        room_cells = np.argwhere(self.room_labels == lid)
        if room_cells.shape[0] == 0:
            return 0.0
        free_count = 0
        for ix, iy in room_cells:
            if self.grid[ix, iy] <= self.prob_free_th:
                free_count += 1
        return float(free_count) / float(room_cells.shape[0])

    def select_frontier_in_room(self, room_id: int = -1) -> Optional[np.ndarray]:
        """在指定房间内选择frontier目标；room_id=-1时全图选择。"""
        if self.pose is None:
            return None

        free_mask = self.grid <= self.prob_free_th
        unk_mask  = (self.grid > self.prob_free_th) & (self.grid < self.prob_occ_th)
        occ_mask  = self.grid >= self.prob_occ_th

        unk_dilated = binary_dilation(unk_mask, iterations=1)
        frontier_mask = free_mask & unk_dilated

        # 若指定了房间，则限制在该房间内
        if room_id > 0:
            room = next((r for r in self.rooms_info if r["id"] == room_id), None)
            if room is not None:
                room_mask = (self.room_labels == room["label"])
                # 允许稍微超出房间边界（膨胀2格），避免边缘frontier被完全过滤
                room_dilated = binary_dilation(room_mask, iterations=3)
                frontier_mask = frontier_mask & room_dilated

        frontier_cells = np.argwhere(frontier_mask)
        if frontier_cells.shape[0] == 0:
            return None

        occ_dilated = binary_dilation(occ_mask, iterations=int(0.4 / self.map_res))
        safe_frontier = frontier_cells[~occ_dilated[frontier_cells[:, 0], frontier_cells[:, 1]]]
        if safe_frontier.shape[0] == 0:
            safe_frontier = frontier_cells

        px, py = self._plan_px, self._plan_py
        pix = int(np.clip((px - self.bounds[0, 0]) / self.map_res, 0, self.grid_shape[0] - 1))
        piy = int(np.clip((py - self.bounds[1, 0]) / self.map_res, 0, self.grid_shape[1] - 1))

        dists = np.sqrt((safe_frontier[:, 0] - pix) ** 2 + (safe_frontier[:, 1] - piy) ** 2)
        rng_noise = np.random.uniform(0, 1.5, size=dists.shape)
        scores = dists * self.map_res + rng_noise

        dist_m = dists * self.map_res
        mid_mask = (dist_m >= 0.5) & (dist_m <= 6.0)
        if np.any(mid_mask):
            candidates = safe_frontier[mid_mask]
            scores_c = scores[mid_mask]
        else:
            candidates = safe_frontier
            scores_c = scores

        idx = int(np.argmin(scores_c))
        gix, giy = candidates[idx]
        gx = float(self.bounds[0, 0] + gix * self.map_res)
        gy = float(self.bounds[1, 0] + giy * self.map_res)
        return np.array([gx, gy], dtype=float)

    def nav_to_goal_cmd(self, goal_xy: np.ndarray, speed: float, now) -> Optional[np.ndarray]:
        """生成朝 goal_xy 移动的《odom 帧》水平速度，同时通过地图检测路径障碍。返回 None 若路径完全阻塞。
        ★[B] 升级：使用膨胀障碍（与 A* 一致）、检查整段到目标的路径（而非固定 0.6m）。
        ★ goal_xy / _plan_px / _plan_py 均在 plan frame 下（ENU，访问 self.grid）。
        ★ 返回速度已经过 map→odom TF 旋转，可直接作为 PX4 TrajectorySetpoint.velocity。"""
        if self.pose is None or self.att_rot is None:
            return None
        px, py = self._plan_px, self._plan_py
        dx = float(goal_xy[0]) - px
        dy = float(goal_xy[1]) - py
        dist = float(np.linalg.norm([dx, dy]))
        if dist < 0.05:
            return None

        # ★[B] 构建与 A* 一致的膨胀障碍地图
        # ★[K1] 门目标使用与 A* 相同的 35% 膨胀（full 100% 膨胀会封堵门洞入口方向,
        #        导致 nav_to_goal 始终返回 None → [REPLAN nav_blocked door_goal→hold] → 零速撞墙）
        from scipy.ndimage import binary_dilation as _bd
        _occ_raw = self.grid >= self.prob_occ_th
        _nav_is_door_goal = getattr(self, 'ssg_goal_type', 'none') in (
            'frontier_gap_pre', 'frontier_gap_center', 'frontier_gap_post',
            'frontier_gap_probe', 'confirmed_door')
        _nav_infl_pct = 0.35 if _nav_is_door_goal else 1.0
        _infl_cells = max(1, int(self.astar_inflation_radius * _nav_infl_pct / self.map_res))
        _occ_infl = _bd(_occ_raw, iterations=_infl_cells)

        # ★[B] 检查距离：min(dist_to_goal, nav_check_max_dist_m)，使用膨胀障碍
        map_dir = np.array([dx / dist, dy / dist], dtype=float)
        check_dist_m = min(dist, float(getattr(self, 'nav_check_max_dist_m', 1.5)))
        steps = max(3, int(check_dist_m / self.map_res))
        W, H  = self.grid_shape
        _first_hit = None
        _blocked = False
        for s in range(1, steps + 1):
            rx = px + map_dir[0] * s * self.map_res
            ry = py + map_dir[1] * s * self.map_res
            gx, gy = self.world_to_grid(rx, ry)
            gx = int(np.clip(gx, 0, W - 1))
            gy = int(np.clip(gy, 0, H - 1))
            if _occ_infl[gx, gy]:
                _blocked = True
                _first_hit = (gx, gy)
                break

        import time as _nvc_t
        if _nvc_t.monotonic() - getattr(self, '_nav_check_log_t', 0.0) > 1.5:
            self._nav_check_log_t = _nvc_t.monotonic()
            self.get_logger().debug(
                f"[NAV-CHECK] dist={dist:.2f}m max_check={check_dist_m:.2f}m "
                f"blocked={_blocked} first_hit={_first_hit}"
            )
        if _blocked:
            self.get_logger().debug(
                f"[NAV-BLOCKED] switch_to=replan first_hit={_first_hit}"
            )
            return None  # 路径被阻，返回 None 让调用方触发重规划

        sp = float(np.clip(speed * min(1.0, dist / 1.0), 0.1, speed))
        # ★ 关键修复：map frame 速度向量必须经 map→odom TF 旋转转到 odom frame
        vx_odom, vy_odom = self._map_vel_to_odom_vel(map_dir[0] * sp, map_dir[1] * sp)
        return np.array([vx_odom, vy_odom], dtype=float)

    # ★[A1] SSG 路径发布：将当前 A* 规划路径广播给 local_mpc 跟踪
    # 解决 global_planner 在 SEARCH/NAV_TO_VIEWPOINT 期间停止广播导致 local_fresh=False 问题
    def _publish_ssg_path(self, force: bool = False) -> None:
        """★[A1] 以 2Hz 节流发布 target_search 内部 A* 路径，供 local_mpc 跟踪。
        路径有效时 local_mpc 会实时发布 local_cmd → local_fresh=True → CTRL-ARB LOCAL_DOMINANT 模式。
        路径为空/耗尽时发布 empty 消息，local_mpc 回退到 global_path。
        """
        import time as _spub_t
        _now_sp = _spub_t.monotonic()
        if not force and _now_sp - self._ssg_path_pub_t < 0.4:
            return  # 2.5Hz 节流
        self._ssg_path_pub_t = _now_sp
        if (self.astar_path is None
                or not hasattr(self, 'astar_path_idx')
                or self.astar_path_idx >= len(self.astar_path)):
            # 路径耗尽或不存在 → 发布空消息，通知 local_mpc 放弃 SSG 优先级
            _empty = _SearchPath()
            self._ssg_path_pub.publish(_empty)
            return
        from geometry_msgs.msg import Pose as _GPose
        msg = _SearchPath()
        _z = float(getattr(self, 'alt_target_z', -1.5))
        # ★[K3] ssg_path 坐标系修复：将路径点从 plan(map) 帧转换到 odom 帧后发布
        # 根本原因：ssg_path 原来用 plan 帧坐标，local_mpc 用 odom 帧位置，
        #   当 map→odom TF 漂移时（map_odom_diverged>1m）local_mpc 计算方向错误，
        #   导致 MPC-BLOCKED 循环、ssg_path=empty 循环、最终撞墙。
        _k3_has_tf = False
        _k3_rot = None
        _k3_tx  = None
        try:
            import rclpy.duration as _k3_dur
            _k3_t = self._tf_buffer.lookup_transform(
                'odom', 'map', rclpy.time.Time(),
                timeout=_k3_dur.Duration(seconds=0.01))
            from scipy.spatial.transform import Rotation as _K3R
            _k3_q = [_k3_t.transform.rotation.x, _k3_t.transform.rotation.y,
                     _k3_t.transform.rotation.z, _k3_t.transform.rotation.w]
            _k3_rot = _K3R.from_quat(_k3_q)
            _k3_tx  = np.array([_k3_t.transform.translation.x,
                                 _k3_t.transform.translation.y,
                                 _k3_t.transform.translation.z])
            _k3_has_tf = True
        except Exception:
            pass  # TF 不可用时降级到 plan 帧（map≈odom 时误差可忽略）
        for wp in self.astar_path[self.astar_path_idx:]:
            p = _GPose()
            if _k3_has_tf:
                _p_map = np.array([float(wp[0]), float(wp[1]), _z])
                _p_odom = _k3_tx + _k3_rot.apply(_p_map)
                p.position.x = float(_p_odom[0])   # ENU East (odom frame)
                p.position.y = float(_p_odom[1])   # ENU North (odom frame)
            else:
                p.position.x = float(wp[0])   # ENU East (plan frame fallback)
                p.position.y = float(wp[1])   # ENU North (plan frame fallback)
            p.position.z = _z
            msg.poses.append(p)
        self._ssg_path_pub.publish(msg)

    # ===================================================================
    # Stop-Scan-Go 核心探索状态机
    # ===================================================================

    def explore_cmd(self, now) -> np.ndarray:
        """
        Stop-Scan-Go 状态机（两态循环）：
          HOVER_SCAN       → 悬停原地旋转360°扫描，解冻地图建图，完成后冻结地图 → 选下一扫描点
          NAV_TO_VIEWPOINT → A*飞向下一扫描点（地图冻结，激光仅做避障），到达后→HOVER_SCAN
        循环保证：每次建图都在静止状态，消除运动引起的地图漂移和角度偏移。
        """
        # 任务6：目标独占期间探索状态机被冻结，直接返回零速
        if getattr(self, '_approach_exclusive', False):
            return np.zeros(3, dtype=float)
        # ★[A6][C1] MISSION_DONE / 终端任务锁定：彻底冻结所有探索
        if getattr(self, '_mission_done', False):
            # 任务完成，禁止任何探索指令
            return np.zeros(3, dtype=float)
        if getattr(self, 'target_terminal_locked', False):
            # 弱确认后即冻结 frontier/SSG 探索，等待 _approach_exclusive 接管
            # （_approach_exclusive 会在 C1 快捷路径或 target_confirmed 时设置）
            if self.explore_state not in ('POST_DOOR_COMMIT', 'ROOM_ENTRY_SCAN'):
                # 允许 POST_DOOR_COMMIT/ROOM_ENTRY_SCAN 完成穿门动作，其余阶段冻结
                return np.zeros(3, dtype=float)
        now_sec = now.nanoseconds * 1e-9

        # ===== STUCK 进展检测（每帧运行，滑动窗口判断探索是否停滞）=====
        _free_now = int(np.sum(self.grid <= self.prob_free_th))
        self._explore_free_window.append((now_sec, _free_now))
        # 只保留最近 60s 的记录
        self._explore_free_window = [
            (t, f) for (t, f) in self._explore_free_window
            if now_sec - t < 60.0
        ]
        # 判断条件：①至少有30s采样 ②近30s内新探索<50格 ③处于集聚状态 ④距上次触发>120s
        if (len(self._explore_free_window) >= 2 and
                getattr(self, '_ssg_clustered', False) and
                now_sec - self._stuck_last_trigger_t > 120.0):
            _oldest_t, _oldest_f = self._explore_free_window[0]
            _dt_span = now_sec - _oldest_t
            if _dt_span >= 30.0:
                _delta_free = _free_now - _oldest_f
                if _delta_free < 50:  # 30s 内新探索格 < 50 ≈ 0.5m² → STUCK
                    self._stuck_count += 1
                    self._stuck_last_trigger_t = now_sec
                    self._explore_free_window.clear()  # 重置窗口
                    self._log_event(
                        "EXPLORE-WATCHDOG",
                        f"第{self._stuck_count}次：近{_dt_span:.0f}s 新探索={_delta_free}格 "
                        f"<50格阈值，集聚={getattr(self,'_ssg_clustered',False)}，触发STUCK恢复"
                    )
                    self.get_logger().warn(
                        f"[EXPLORE-WATCHDOG] #{self._stuck_count} 探索停滞 "
                        f"Δfree={_delta_free}格/{_dt_span:.0f}s → 重置目标，优先尝试穿越门洞"
                    )
                    # STUCK 恢复：清除当前路径/目标，强制切回 HOVER_SCAN 重新建图
                    self.ssg_viewpoint_goal = None
                    self.astar_path = None
                    self.explore_state = "HOVER_SCAN"
                    self.hover_scan_start_time = None
                    # 若有门候选，选最远门作为 STUCK 恢复的虚拟出口目标
                    self._virtual_door_cands = []
                    if self.doors_info and self.pose is not None:
                        _door_dists = []
                        for _dd in self.doors_info:
                            try:
                                _dwx, _dwy = self.grid_to_world(
                                    int(_dd["ix"]), int(_dd["iy"]))
                                _dist = float(np.hypot(_dwx - self._plan_px,
                                                       _dwy - self._plan_py))
                                _door_dists.append((_dist, _dwx, _dwy))
                            except Exception:
                                continue
                        if _door_dists:
                            _, _dvx, _dvy = sorted(_door_dists, reverse=True)[0]
                            self._virtual_door_cands = [(_dvx, _dvy)]
                            self._log_event(
                                "STUCK-DOOR",
                                f"STUCK后虚拟出口目标：({_dvx:.2f},{_dvy:.2f}) "
                                f"距无人机={_door_dists[0][0]:.1f}m"
                            )


        import time as _ssgt
        _ssgt_now = _ssgt.monotonic()
        # ── Phase9 MAP_RECOVER 退出检查 ──────────
        if getattr(self, '_map_recover_active', False):
            _recover_sec  = _ssgt_now - getattr(self, '_map_recover_since_mono', _ssgt_now)
            _ekf_quiet    = (_ssgt_now - getattr(self, '_last_reset_mono', 0.0)) > 15.0
            _needed       = getattr(self, '_occ_stable_needed', 5)
            _stable_cnt   = getattr(self, '_occ_stable_frames', 0)
            _stable       = _stable_cnt >= _needed
            # 大幅跳变时还必须等 hard_hold 结束
            _hard_until   = getattr(self, '_map_recover_hard_until', 0.0)
            _hard_ok      = _ssgt_now >= _hard_until
            _is_hard_mode = _hard_until > 0.0 and _needed > 5  # 大跳变模式
            if _recover_sec >= 3.0 and _ekf_quiet and _stable and _hard_ok:
                self._map_recover_active = False
                self._map_recover_hard_until = 0.0
                self._hover_scan_retry_cnt = 0
                # 大跳变模式恢复：需要恢复 slam scan
                if _is_hard_mode:
                    self._slam_pause_scan(False)
                    self._log_event(
                        "MAP-RECOVER-HARD",
                        f"stable_frames={_stable_cnt}/{_needed} "
                        f"ekf_quiet={_ekf_quiet} recover_sec={_recover_sec:.1f}s "
                        f"resume_slam=True"
                    )
                    self.get_logger().info(
                        f"[MAP-RECOVER-HARD] stable_frames={_stable_cnt}/{_needed} "
                        f"recover_sec={_recover_sec:.1f}s → resume_slam=True 恢复探索"
                    )
                self._log_event(
                    "RECOVER-DONE",
                    f"after {_recover_sec:.1f}s ekf_quiet={_ekf_quiet} "
                    f"stable_frames={_stable_cnt}/{_needed} hard_mode={_is_hard_mode}"
                )
                self.get_logger().info(
                    f"[MAP-RECOVER] exit: after {_recover_sec:.1f}s "
                    f"ekf_quiet={_ekf_quiet} stable_frames={_stable_cnt}/{_needed} "
                    f"hard_mode={_is_hard_mode} → 恢复正常探索"
                )
            elif _is_hard_mode and not _hard_ok:
                # 还在 hard_hold 期间，打印等待剩余时间（5s 一次）
                _remaining = _hard_until - _ssgt_now
                if _ssgt_now - getattr(self, '_hard_hold_log_t', 0.0) > 5.0:
                    self._hard_hold_log_t = _ssgt_now
                    self.get_logger().warn(
                        f"[MAP-RECOVER-HARD] hard_hold剩余{_remaining:.1f}s "
                        f"stable_frames={_stable_cnt}/{_needed} ekf_quiet={_ekf_quiet}"
                    )
        if _ssgt_now - getattr(self, '_last_explore_state_log_mono', 0.0) > 5.0:
            self._last_explore_state_log_mono = _ssgt_now
            free_count = int(np.sum(self.grid <= self.prob_free_th))
            max_belief = float(np.max(self.belief_grid))
            vp = self.ssg_viewpoint_goal
            self.get_logger().info(
                f"[SSG] state={self.explore_state} frozen={self._map_frozen} "
                f"free={free_count} max_belief={max_belief:.2f} "
                f"vp={None if vp is None else f'({vp[0]:.1f},{vp[1]:.1f})'} "
                f"scan_pts={len(self.ssg_visited_scan_pts)}"
            )

        if self.explore_state == "HOVER_SCAN":
            return self._hover_scan_cmd(now, now_sec)

        if self.explore_state == "NAV_TO_VIEWPOINT":
            return self._nav_to_viewpoint_cmd(now, now_sec)

        # ★[A] POST_DOOR_COMMIT：穿门后的短暂惯性推进保护窗口
        # 目的：穿越门口后立刻给无人机 post_door_commit_timeout 秒（或 dist 达标）的自由行进时间，
        # 彻底跳开门框几何区域再开始 HOVER_SCAN，从根本上切断
        # "DOOR-CROSSED → HOVER_SCAN → 近门框 ESCAPE → SAFE_HOLD" 的失效链。
        # local_mpc 在此状态下持续发布（不抢占），继续推进 ssg_path。
        if self.explore_state == "POST_DOOR_COMMIT":
            return self._post_door_commit_cmd(now, now_sec)

        # ★[B] ROOM_ENTRY_SCAN：进入新房间后的前视视觉扇扫覆盖
        if self.explore_state == "ROOM_ENTRY_SCAN":
            return self._room_entry_scan_cmd(now, now_sec)

        # fallback
        self.explore_state = "HOVER_SCAN"
        self.hover_scan_start_time = None
        return np.zeros(3, dtype=float)

    def _post_door_commit_cmd(self, now, now_sec: float) -> np.ndarray:
        """
        ★[A] POST_DOOR_COMMIT 保护状态：穿门成功后短暂推进阶段。
        - 无人机在穿越门口后立即进入此状态，持续 post_door_commit_timeout 秒
          或直到离门口起始位置超过 post_door_commit_dist 米。
        - 若 ssg_viewpoint_goal 有目标，继续 NAV 推进；否则使用保存的穿门方向
          (_post_door_commit_push_dir) 以固定速度主动推进，确保飞出门框区域。
        - 超时或距离满足后转入 NAV_TO_VIEWPOINT（若有目标）或普通 HOVER_SCAN。
        - ★ 修复：避免因 ssg_goal 被 reach_reached 清空而在门口附近零速悬停，
          导致 HOVER-NEAR-WALL 重选同一 gap 产生无限穿门循环。
        """
        import time as _pdc_t
        _now_m = _pdc_t.monotonic()
        _commit_start  = getattr(self, '_post_door_commit_start_t', 0.0)
        _commit_pos    = getattr(self, '_post_door_commit_start_pos', None)
        _commit_timeout = float(getattr(self, 'post_door_commit_timeout', 6.0))
        _commit_dist    = float(getattr(self, 'post_door_commit_dist', 1.5))

        _elapsed = _now_m - _commit_start
        _dist_moved = 0.0
        if _commit_pos is not None and self.pose is not None:
            import math as _pdc_m
            _dist_moved = _pdc_m.hypot(
                float(self.pose.x) - _commit_pos[0],
                float(self.pose.y) - _commit_pos[1]
            )

        # 1Hz 节流诊断日志
        if _now_m - getattr(self, '_pdc_log_t', 0.0) > 1.0:
            self._pdc_log_t = _now_m
            self.get_logger().info(
                f"[POST-DOOR-COMMIT] elapsed={_elapsed:.1f}s/{_commit_timeout:.1f}s "
                f"dist_moved={_dist_moved:.2f}m/{_commit_dist:.1f}m "
                f"ssg_goal={getattr(self,'ssg_viewpoint_goal',None) is not None}"
            )

        # 退出条件：超时 OR 移动距离已达标
        _exit_pdc = (_elapsed >= _commit_timeout or _dist_moved >= _commit_dist)
        if _exit_pdc:
            _reason = "timeout" if _elapsed >= _commit_timeout else "dist_ok"
            # ★[B] POST_DOOR_COMMIT 退出后触发 ROOM_ENTRY_SCAN（若冷却期通过）
            # 优先进行一次房间内视觉扇扫，降低目标漏检率
            import time as _pdc_exit_t
            _re_cooldown = float(getattr(self, 'room_entry_scan_timeout_sec', 20.0)) * 2.0
            _last_re_t   = float(getattr(self, '_last_room_entry_trigger_t', 0.0))
            _can_room_entry = (
                _pdc_exit_t.monotonic() - _last_re_t > max(30.0, _re_cooldown * 0.5)
                and not getattr(self, '_tgt_weak_confirmed', False)  # 已弱确认则不触发
            )
            _next_state = "HOVER_SCAN"
            if _can_room_entry:
                # 触发 ROOM_ENTRY_SCAN
                _yaw_now = float(self.att_yaw) if self.att_yaw is not None else 0.0
                _span_deg = float(getattr(self, 'room_entry_scan_yaw_span_deg', 120.0))
                _step_deg = float(getattr(self, 'room_entry_scan_step_deg', 40.0))
                import math as _pdc_re_m
                _span_rad = _span_deg * _pdc_re_m.pi / 180.0
                _step_rad = _step_deg * _pdc_re_m.pi / 180.0
                # 生成各步目标 yaw：左前→正前→右前
                _steps = []
                _n = max(3, int(round(_span_deg / _step_deg)) + 1)
                for _si in range(_n):
                    _ang = _yaw_now - _span_rad / 2.0 + _si * _span_rad / (_n - 1.0)
                    _steps.append(_ang)
                self._room_entry_scan_active = True
                self._room_entry_scan_start_t = _pdc_exit_t.monotonic()
                self._room_entry_scan_yaw_base = _yaw_now
                self._room_entry_scan_step = 0
                self._room_entry_scan_step_enter_t = _pdc_exit_t.monotonic()
                self._room_entry_scan_steps = _steps
                self._room_entry_target_found = False
                self._last_room_entry_trigger_t = _pdc_exit_t.monotonic()
                _next_state = "ROOM_ENTRY_SCAN"
                self.get_logger().warn(
                    f"[ROOM-ENTRY-SCAN-ENTER] POST_DOOR_COMMIT退出 → 房间视觉扇扫 "
                    f"yaw_base={_yaw_now:.2f}rad steps={len(_steps)} span={_span_deg:.0f}°"
                )
                self._log_event("ROOM-ENTRY-SCAN-ENTER",
                    f"reason=post_door yaw_base={_yaw_now:.2f} steps={len(_steps)}")
            self.get_logger().warn(
                f"[POST-DOOR-COMMIT-EXIT] reason={_reason} elapsed={_elapsed:.1f}s "
                f"dist={_dist_moved:.2f}m → {_next_state}"
            )
            self._log_event("POST-DOOR-COMMIT-EXIT",
                f"reason={_reason} elapsed={_elapsed:.1f}s dist={_dist_moved:.2f}m")
            if self.ssg_viewpoint_goal is not None and _next_state == "HOVER_SCAN":
                self.explore_state = "NAV_TO_VIEWPOINT"
            else:
                self.explore_state = _next_state
                if _next_state == "HOVER_SCAN":
                    self.hover_scan_start_time = None
            return np.zeros(3, dtype=float)

        # 在 POST_DOOR_COMMIT 期间：若 ssg 有目标继续 NAV 推进，否则沿穿门方向主动推进
        if self.ssg_viewpoint_goal is not None:
            return self._nav_to_viewpoint_cmd(now, now_sec)
        # ★[PDC-PUSH] 无 ssg 目标时：用穿门方向推进，确保无人机离开门框进入新区域
        # 根本fix：避免 POST_DOOR_COMMIT 因 vel=0 导致 dist=0.09m（barely moved）后
        # HOVER-NEAR-WALL 重选同一个gap入口，陷入无限穿门循环
        _pdc_push_n = getattr(self, '_post_door_commit_push_dir', None)
        if _pdc_push_n is not None:
            _push_spd = 0.30   # m/s 穿门后推进速度
            # ★[H-FIX-PDC-PROX] 检查推进方向的 LIDAR 净空，防止撞入紧凑空间
            # 根本原因：PROX-BRAKE 在门口模式放宽到 0.35m/s，PDC push 0.30m/s 不被拦截
            # 修复：在推进方向净空不足时减速/停止推进，其余控制权不变
            import math as _pdc_prox_m
            _pdc_ref_range = None
            _yaw_pdc = float(self.att_yaw) if self.att_yaw is not None else 0.0
            # 把 ENU 推进向量投影到体坐标系，选取最对齐的 LIDAR 扇区
            _px_enu = float(_pdc_push_n[0])
            _py_enu = float(_pdc_push_n[1])
            _pm = _pdc_prox_m.hypot(_px_enu, _py_enu)
            if _pm > 0.01:
                _px_enu /= _pm
                _py_enu /= _pm
                # body forward = ENU (cos(yaw), sin(yaw))
                _sectors_body = {
                    'front': ( _pdc_prox_m.cos(_yaw_pdc),  _pdc_prox_m.sin(_yaw_pdc)),
                    'back':  (-_pdc_prox_m.cos(_yaw_pdc), -_pdc_prox_m.sin(_yaw_pdc)),
                    'left':  (-_pdc_prox_m.sin(_yaw_pdc),  _pdc_prox_m.cos(_yaw_pdc)),
                    'right': ( _pdc_prox_m.sin(_yaw_pdc), -_pdc_prox_m.cos(_yaw_pdc)),
                }
                _best_dot = -1.0
                for _sec, (_ex, _ey) in _sectors_body.items():
                    _dot = _px_enu * _ex + _py_enu * _ey
                    if _dot > _best_dot:
                        _best_dot = _dot
                        _r = self.lidar_ranges.get(_sec)
                        _pdc_ref_range = float(_r) if _r is not None else None
            # 净空阈值：停止=0.35m，全速=0.55m
            _pdc_stop_dist = 0.35
            _pdc_full_dist = 0.55
            if _pdc_ref_range is not None and _pdc_ref_range <= _pdc_stop_dist:
                if _now_m - getattr(self, '_pdc_prox_log_t', 0.0) > 1.0:
                    self._pdc_prox_log_t = _now_m
                    self.get_logger().warn(
                        f"[PDC-PUSH-BLOCKED] 推进方向净空不足 "
                        f"clr={_pdc_ref_range:.2f}m <= {_pdc_stop_dist:.2f}m → 停止推进"
                    )
                return np.zeros(3, dtype=float)
            elif _pdc_ref_range is not None and _pdc_ref_range < _pdc_full_dist:
                _scale = (_pdc_ref_range - _pdc_stop_dist) / (_pdc_full_dist - _pdc_stop_dist)
                _push_spd *= float(np.clip(_scale, 0.0, 1.0))
            return np.array([_pdc_push_n[0] * _push_spd, _pdc_push_n[1] * _push_spd, 0.0],
                            dtype=float)
        return np.zeros(3, dtype=float)

    def _room_entry_scan_cmd(self, now, now_sec: float) -> np.ndarray:
        """
        ★[B] ROOM_ENTRY_SCAN：进入新房间后的系统性前视视觉扇扫。
        目的：主动将前视相机覆盖左前、正前、右前扇区，减少目标因视角盲区导致漏检。
        流程：
          1. 进入时保存当前 yaw 为基准，生成 N 个扇区目标 yaw（左→中→右）
          2. 每个扇区转到目标 yaw 后悬停 room_entry_scan_hold_sec 秒
          3. 悬停期间运行红色检测，若弱确认则提前退出进入目标链
          4. 完成所有扇区或超时后退出到 HOVER_SCAN
        输出：纯 yaw 旋转速度（XY 位置保持，靠 local_mpc 维持）
        """
        import time as _res_t, math as _res_m
        _now_m = _res_t.monotonic()

        _steps = list(getattr(self, '_room_entry_scan_steps', []))
        _total_timeout = float(getattr(self, 'room_entry_scan_timeout_sec', 20.0))
        _hold_sec      = float(getattr(self, 'room_entry_scan_hold_sec', 1.0))
        _step_idx      = int(getattr(self, '_room_entry_scan_step', 0))
        _step_enter_t  = float(getattr(self, '_room_entry_scan_step_enter_t', _now_m))
        _scan_start_t  = float(getattr(self, '_room_entry_scan_start_t', _now_m))
        _yaw_now       = float(self.att_yaw) if self.att_yaw is not None else 0.0

        # ── ★[H-FIX-RES-PROX] 旋转扫描净空安全检查 ─────────────────────────────
        # 根本原因：POST_DOOR_COMMIT push 可能将无人机推至距墙 <0.40m 处，
        # 120° 偏航旋转时机身/螺旋桨扫过近墙方向导致物理撞击。
        # 修复：若任意方向净空 < 0.40m，中止扫描退回 HOVER_SCAN。
        # 此检查不影响控制权分层，仅在 ROOM_ENTRY_SCAN 内部安全退出。
        _res_min_near_vals = [v for v in self.lidar_ranges.values() if v is not None]
        _res_min_near = min(_res_min_near_vals) if _res_min_near_vals else 9.9
        _res_prox_abort_threshold = 0.40  # m
        if _res_min_near < _res_prox_abort_threshold:
            if _now_m - getattr(self, '_res_prox_log_t', 0.0) > 1.0:
                self._res_prox_log_t = _now_m
                self.get_logger().warn(
                    f"[ROOM-ENTRY-SCAN-PROX-ABORT] 净空不足 near={_res_min_near:.2f}m "
                    f"< {_res_prox_abort_threshold:.2f}m → 中止旋转扫描，退回 HOVER_SCAN"
                )
                self._log_event("ROOM-ENTRY-SCAN-EXIT",
                    f"reason=prox_abort near={_res_min_near:.2f}m")
            self._room_entry_scan_active = False
            self.explore_state = "HOVER_SCAN"
            self.hover_scan_start_time = None
            return np.zeros(3, dtype=float)

        # ── 超时退出 ──
        if _now_m - _scan_start_t > _total_timeout:
            self.get_logger().warn(
                f"[ROOM-ENTRY-SCAN-EXIT] reason=timeout elapsed={_now_m-_scan_start_t:.1f}s "
                f"→ HOVER_SCAN"
            )
            self._log_event("ROOM-ENTRY-SCAN-EXIT", "reason=timeout")
            self._room_entry_scan_active = False
            self.explore_state = "HOVER_SCAN"
            self.hover_scan_start_time = None
            return np.zeros(3, dtype=float)

        # ── 目标发现提前退出 ──
        # 若弱确认触发（检测线程已更新状态），立即退出扇扫、冻结搜索
        if getattr(self, '_tgt_weak_confirmed', False):
            self.get_logger().warn(
                f"[ROOM-ENTRY-SCAN-EXIT] reason=target_found "
                f"step={_step_idx}/{len(_steps)} → TARGET 链"
            )
            self._log_event("ROOM-ENTRY-SCAN-EXIT", "reason=target_found")
            self._room_entry_scan_active = False
            # 弱确认已触发，由目标确认逻辑负责状态转换；保持 HOVER_SCAN 待目标链接管
            self.explore_state = "HOVER_SCAN"
            self.hover_scan_start_time = None
            return np.zeros(3, dtype=float)

        # ── 无步骤配置时退出 ──
        if not _steps:
            self.explore_state = "HOVER_SCAN"
            self.hover_scan_start_time = None
            return np.zeros(3, dtype=float)

        # ── 完成所有扇区退出 ──
        if _step_idx >= len(_steps):
            self.get_logger().warn(
                f"[ROOM-ENTRY-SCAN-EXIT] reason=all_sectors_done steps={len(_steps)} "
                f"→ HOVER_SCAN"
            )
            self._log_event("ROOM-ENTRY-SCAN-EXIT",
                f"reason=all_sectors_done steps={len(_steps)}")
            self._room_entry_scan_active = False
            self.explore_state = "HOVER_SCAN"
            self.hover_scan_start_time = None
            return np.zeros(3, dtype=float)

        # ── 当前扇区控制 ──
        _target_yaw = float(_steps[_step_idx])
        _yaw_err    = _target_yaw - _yaw_now
        # 角度归一化到 [-π, π]
        while _yaw_err >  _res_m.pi: _yaw_err -= 2.0 * _res_m.pi
        while _yaw_err < -_res_m.pi: _yaw_err += 2.0 * _res_m.pi

        _yaw_reached = abs(_yaw_err) < 0.10   # ~5.7°
        _step_elapsed = _now_m - _step_enter_t

        _sector_name = "left" if _step_idx == 0 else ("right" if _step_idx == len(_steps)-1 else "front")

        # 诊断日志（2Hz）
        if _now_m - getattr(self, '_res_log_t', 0.0) > 0.5:
            self._res_log_t = _now_m
            self.get_logger().info(
                f"[ROOM-ENTRY-SCAN-STEP] sector={_sector_name}({_step_idx+1}/{len(_steps)}) "
                f"yaw_target={_res_m.degrees(_target_yaw):.1f}° "
                f"yaw_err={_res_m.degrees(_yaw_err):.1f}° "
                f"held={_step_elapsed:.1f}s/{_hold_sec:.1f}s "
                f"elapsed_total={_now_m-_scan_start_t:.1f}s"
            )
            self._log_event("ROOM-ENTRY-SCAN-STEP",
                f"sector={_sector_name} step={_step_idx} yaw_target={_res_m.degrees(_target_yaw):.1f}")

        # 若 yaw 已到达目标且悬停时间满足 → 推进到下一扇区
        if _yaw_reached and _step_elapsed >= _hold_sec:
            self._room_entry_scan_step  = _step_idx + 1
            self._room_entry_scan_step_enter_t = _now_m
            self.get_logger().info(
                f"[ROOM-ENTRY-SCAN-STEP] sector={_sector_name} done → next"
            )
            return np.zeros(3, dtype=float)

        # 计算偏航速率命令（向目标 yaw 旋转）
        _yaw_kp   = 0.8   # P-gain 较快转向
        _yaw_max  = float(getattr(self, 'hover_scan_yaw_speed', 0.4))
        # ★ Fix-YAW-SIGN: att_yaw 为 ENU（正=CCW），ts.yawspeed 为 NED（正=CW）
        # 要减小 ENU yaw 误差 → 需 NED 正向旋转 → 对 ENU yaw_err 取反
        _yawspeed = float(np.clip(-_yaw_kp * _yaw_err, -_yaw_max, _yaw_max))

        # 返回零 XY（位置保持），旋转控制靠 yawspeed 通道
        # local_mpc 会保持当前 XY 位置
        self._servo_yawspeed = _yawspeed
        return np.zeros(3, dtype=float)

    def _hover_scan_cmd(self, now, now_sec: float) -> np.ndarray:
        """HOVER_SCAN 状态机：原地旋转360°扫描建图，完成后→NAV_TO_VIEWPOINT。"""
        import time as _hmt
        _now_mono = _hmt.monotonic()
        # ── EKF reset 保护暂停期，等待 pause 过期再开始建图 ──────────────────
        if _now_mono < self._map_pause_until:
            remaining = self._map_pause_until - _now_mono
            _t = getattr(self, '_hover_pause_log_mono', 0.0)
            if _now_mono - _t > 2.0:
                self._hover_pause_log_mono = _now_mono
                self.get_logger().info(
                    f"[EKF-STATE] PAUSE：等待EKF稳定 {remaining:.1f}s 后开始建图..."
                )
            return np.zeros(3, dtype=float)

        # pause 过期后，若状态仍是 PAUSE → 转为 STABLE 并打印 RESUME
        if getattr(self, '_ekf_reset_state', 'STABLE') == "PAUSE":
            self._ekf_reset_state = "STABLE"
            self.get_logger().info(
                f"[EKF-STATE] PAUSE→STABLE: EKF stabilization complete "
                f"(EKF reset #{getattr(self,'_ekf_reset_count',0)}) → 恢复建图/规划"
            )
            if getattr(self, '_slam_pause_active', False):
                self._slam_pause_scan(False)
                self.get_logger().info("[EKF-STATE] slam 扫描摄入已恢复（Pause → Resume）")

        # ★[F3] 近墙保护：若任意方向间隙 < 0.28m 或前向 < 0.30m，拒绝启动 HOVER_SCAN
        # ★[L1] 提升触发阈值（0.35→0.28m）：走廊/窄道内正常探索时 lidar 经常读到 0.30-0.35m，
        #        原 0.35m 阈值导致几乎每次 HOVER_SCAN 都被打断→迫使 frontier_escape 替代门目标选择。
        #        0.28m = 真正紧贴墙壁的危险距离，正常走廊不会触发。
        _hs_front = self.lidar_ranges.get('front') or 9.9
        _hs_lidar_vals = [v for v in self.lidar_ranges.values() if v is not None]
        _hs_min_all = min(_hs_lidar_vals) if _hs_lidar_vals else 9.9
        if _hs_front < 0.30 or _hs_min_all < 0.28:
            # ★[L1] 近墙时优先选最近且净空足够的 frontier_gap_pre
            # 保证即使被近墙打断也本能指向真实门洞，而非地图边界处的假门
            import math as _fnw_m
            _fnw_gaps = list(getattr(self, 'frontier_gaps', []))
            _fnw_px   = self._plan_px if self.pose else 0.0
            _fnw_py   = self._plan_py if self.pose else 0.0
            _fnw_chosen = None
            if _fnw_gaps and self.pose is not None:
                try:
                    _fnw_dt = self._ssg_compute_dist_transform()
                except Exception:
                    _fnw_dt = None
                # ★ 遍历所有 gap（按距离排序），跳过"假门"（gap_center净空不足）
                _GAP_CTR_MIN_CLR_L1 = 0.33   # gap中心需有此净空才是真实可穿越门洞
                _sorted_gaps_l1 = sorted(_fnw_gaps,
                    key=lambda g: _fnw_m.hypot(g['x'] - _fnw_px, g['y'] - _fnw_py))
                for _fnw_cand_gap in _sorted_gaps_l1:
                    _gwx = float(_fnw_cand_gap['x'])
                    _gwy = float(_fnw_cand_gap['y'])
                    _ori_fnw = str(_fnw_cand_gap.get('orientation', ''))
                    # ★[PDC-GUARD] 跳过60s内已穿越的gap，防止穿门后掉头重返已穿越门口入口侧
                    # 根本fix：DOOR-CROSSED后无人机在门口附近，L1会选同一gap的pre点（从背面），
                    # 导致DOOR-PHASE重新路由穿回 Room1 → 无限循环。60s冷却期确保飞出新区域。
                    _l1_lcgp = getattr(self, '_last_crossed_gap_pos', None)
                    if (_l1_lcgp is not None
                            and _now_mono - float(getattr(self, '_last_crossed_gap_at', 0.0)) < 60.0
                            and _fnw_m.hypot(_gwx - _l1_lcgp[0], _gwy - _l1_lcgp[1]) < 1.5):
                        continue  # 近期已穿越此gap，跳过L1重选
                    if _ori_fnw == 'horizontal':  # 水平墙(y=const) → 穿越方向y轴
                        _nx_fnw, _ny_fnw = 0.0, (1.0 if _fnw_py >= _gwy else -1.0)
                    elif _ori_fnw == 'vertical':  # 垂直墙(x=const) → 穿越方向x轴
                        _nx_fnw, _ny_fnw = (1.0 if _fnw_px >= _gwx else -1.0), 0.0
                    else:
                        _dd_fnw = max(0.001, _fnw_m.hypot(_fnw_px - _gwx, _fnw_py - _gwy))
                        _nx_fnw, _ny_fnw = (_fnw_px - _gwx) / _dd_fnw, (_fnw_py - _gwy) / _dd_fnw
                    # 检查 gap_center 净空：太低说明这不是真实门洞
                    _gcx_l1, _gcy_l1 = self.world_to_grid(_gwx, _gwy)
                    _gcx_l1 = int(np.clip(_gcx_l1, 0, self.grid_shape[0]-1))
                    _gcy_l1 = int(np.clip(_gcy_l1, 0, self.grid_shape[1]-1))
                    _gap_ctr_clr_l1 = (float(_fnw_dt[_gcx_l1, _gcy_l1])
                                       if _fnw_dt is not None else 1.0)
                    if _gap_ctr_clr_l1 < _GAP_CTR_MIN_CLR_L1:
                        continue   # 假门（地图边界伪门），跳过
                    _pre_x = _gwx + _nx_fnw * 1.2
                    _pre_y = _gwy + _ny_fnw * 1.2
                    # 检查 gap_pre 点本身可达且净空足够
                    _pre_gx, _pre_gy = self.world_to_grid(_pre_x, _pre_y)
                    _pre_gx = int(np.clip(_pre_gx, 0, self.grid_shape[0]-1))
                    _pre_gy = int(np.clip(_pre_gy, 0, self.grid_shape[1]-1))
                    _pre_clr = float(_fnw_dt[_pre_gx, _pre_gy]) if _fnw_dt is not None else 1.0
                    if not (self.grid[_pre_gx, _pre_gy] <= self.prob_free_th and
                            _pre_clr >= 0.40):   # ★[WALL-FIX] 0.30→0.40: gap_pre必须距墙≥0.40m
                        continue   # gap_pre 不可达或太近墙
                    # ★[J4] 跳过最近多次失败的门目标：防止与 DOOR-GIVE-UP 形成级联循环。
                    # DOOR-GIVE-UP 清除目标后重置了 _ssg_stuck_since（★[J3]），
                    # 但 L1 若选择同一目标，首帧 stuck timer 被重置后新窗口开始，仍会在18s内再次超时。
                    # 而空间冷却只影响打分权重，不是硬排除────所以必须在此做硬过滤。
                    _l1_fail_key = f"{_pre_x:.1f}_{_pre_y:.1f}"
                    _l1_fail_cnt = getattr(self, '_door_fail_counts', {}).get(_l1_fail_key, 0)
                    if _l1_fail_cnt > 3:
                        continue   # 最近已多次失败，跳过此目标，避免 DOOR-GIVE-UP 级联
                    # ✓ 找到有效 gap
                    _fnw_chosen = np.array([_pre_x, _pre_y], dtype=float)
                    self.ssg_viewpoint_goal = _fnw_chosen
                    self.ssg_goal_type = 'frontier_gap_pre'
                    self.explore_state = 'NAV_TO_VIEWPOINT'
                    self.hover_scan_start_time = None
                    self.get_logger().warn(
                        f'[HOVER-NEAR-WALL] front={_hs_front:.2f}m min={_hs_min_all:.2f}m '
                        f'→ [L1] gap_pre=({_pre_x:.2f},{_pre_y:.2f}) clr={_pre_clr:.2f}m '
                        f'gap=({_gwx:.2f},{_gwy:.2f}) ori={_ori_fnw} gap_ctr_clr={_gap_ctr_clr_l1:.2f}m'
                    )
                    self._log_event('HOVER-NEAR-WALL',
                        f'front={_hs_front:.2f}m min={_hs_min_all:.2f}m → gap_pre=({_pre_x:.2f},{_pre_y:.2f})')
                    return np.zeros(3, dtype=float)
            if _fnw_chosen is None:
                _fe = self._find_frontier_escape_point()
            else:
                _fe = None
            if _fe is not None:
                self.ssg_viewpoint_goal = _fe
                self.ssg_goal_type = "frontier_escape"
                self.explore_state = "NAV_TO_VIEWPOINT"
                self.hover_scan_start_time = None
                self.get_logger().warn(
                    f"[HOVER-NEAR-WALL] front={_hs_front:.2f}m min_all={_hs_min_all:.2f}m "
                    f"→ 跳过 HOVER_SCAN 直接 frontier 逃逸 ({_fe[0]:.2f},{_fe[1]:.2f})"
                )
                self._log_event(
                    "HOVER-NEAR-WALL",
                    f"front={_hs_front:.2f}m min={_hs_min_all:.2f}m → ({_fe[0]:.2f},{_fe[1]:.2f})"
                )
                return np.zeros(3, dtype=float)
            # 无 frontier 时继续执行 HOVER_SCAN（SAFE_HOLD_ESCAPE 会接管）

        # ★ Phase9: 初始化移至障碍物检查之前，防止障碍物过近时永久无法初始化（卡死根因修复）
        if self.hover_scan_start_time is None:
            # ─────────────────────────────────────────────────────────────
            # ★[B-ARCH] HOVER_SCAN 快速跳过：若有未完成的门/gap 类目标且净空足够，
            # 直接恢复 NAV_TO_VIEWPOINT，不做全扫描（防止门口附近反复扫描浪费时间）
            import time as _hskip_t
            _hskip_goal_type = getattr(self, 'ssg_goal_type', 'none')
            _hskip_door_types = ('frontier_gap_pre', 'frontier_gap_center',
                                 'frontier_gap_post', 'frontier_gap_probe', 'confirmed_door')
            _hskip_has_door = (
                _hskip_goal_type in _hskip_door_types
                and self.ssg_viewpoint_goal is not None
                and self.pose is not None
            )
            if _hskip_has_door:
                _hskip_near = self._nearest_lidar_distance()
                # ★[B2] HOVER-SKIP 净空阈值 0.38→0.28m：窄门门框净空(0.28-0.38m)时
                # 原0.38m会拒绝SKIP → 触发full HOVER_SCAN → ESCAPE → FAIL → SAFE_HOLD循环
                # 改为0.28m：只有真正极近(<0.28m可能卡住)时才进入full scan
                # 配合 _door_transit_mode：过门模式时即使近墙也允许恢复导航
                _hskip_ok_clr = (
                    _hskip_near is None
                    or _hskip_near > 0.28
                    or getattr(self, '_door_transit_mode', False)  # 过门模式强制允许
                )
                if _hskip_ok_clr:
                    # 净空足够，不做全扫描，直接恢复导航
                    self.explore_state = "NAV_TO_VIEWPOINT"
                    self.get_logger().info(
                        f"[HOVER-SKIP] door_type={_hskip_goal_type} → 跳过全扫描直接恢复导航 "
                        f"goal=({self.ssg_viewpoint_goal[0]:.2f},{self.ssg_viewpoint_goal[1]:.2f}) "
                        f"near={_hskip_near:.2f}m"
                    )
                    return np.zeros(3, dtype=float)
            # ── Phase9 HOVER_SCAN 重试次数上限：超过后强制 frontier 逃逸 ──────────
            _retry_cnt = getattr(self, '_hover_scan_retry_cnt', 0)
            _max_retry  = getattr(self, '_hover_scan_max_retry', 5)
            if _retry_cnt >= _max_retry:
                _frontier_esc2 = self._find_frontier_escape_point()
                if _frontier_esc2 is not None:
                    self.ssg_viewpoint_goal = _frontier_esc2
                    self.ssg_goal_type = "frontier_escape"
                    self.explore_state = "NAV_TO_VIEWPOINT"
                    self._hover_scan_retry_cnt = 0
                    self._map_recover_active = False
                    self._log_event(
                        "HOVER-MAX-RETRY",
                        f"retry={_retry_cnt}>={_max_retry} → 强制 frontier 逃逸 "
                        f"target=({_frontier_esc2[0]:.2f},{_frontier_esc2[1]:.2f})"
                    )
                    return np.zeros(3, dtype=float)
                else:
                    # frontier 也没有，重设计数继续扫描
                    self._hover_scan_retry_cnt = max(0, _retry_cnt - 2)
                    self.get_logger().warn(
                        f"[HOVER-MAX-RETRY] 无 frontier，重设计数为 {self._hover_scan_retry_cnt} 继续扫描"
                    )
            # ★ 锁点黑名单检查：若当前位置在某次失败锁点的 0.8m 范围内，
            # 且这个锁点失败时间距现在 < 90s，则不在此处重新开启 HOVER_SCAN
            # 而是先把无人机导航到附近的 frontier 点再扫描
            import math as _bm
            _cur_px = float(self.pose.x) if self.pose else 0.0
            _cur_py = float(self.pose.y) if self.pose else 0.0
            _fail_anch = getattr(self, '_hs_fail_anchors', [])
            _in_blacklist = any(
                _bm.hypot(_cur_px - _fp[0], _cur_py - _fp[1]) < 0.8
                and _now_mono - _ft < 90.0
                for _ft, _fp in _fail_anch
            )
            if _in_blacklist:
                _frontier_bl = self._find_frontier_escape_point()
                if _frontier_bl is not None:
                    self.ssg_viewpoint_goal = _frontier_bl
                    self.ssg_goal_type = "frontier_escape"
                    self.astar_path = None
                    self.astar_path_idx = 0
                    self.explore_state = "NAV_TO_VIEWPOINT"
                    self.get_logger().warn(
                        f"[HOVER-BLACKLIST] 当前位置在失败锇点 0.8m 范围内，"
                        f"导航到 frontier ({_frontier_bl[0]:.2f},{_frontier_bl[1]:.2f}) 再扫描"
                    )
                    self._log_event("HOVER-BLACKLIST", f"nav_to ({_frontier_bl[0]:.2f},{_frontier_bl[1]:.2f})")
                    return np.zeros(3, dtype=float)
            # ★[B] 新房间检测：若当前房间与上次 HOVER_SCAN 房间不同 → 触发 ROOM_ENTRY_SCAN
            # 覆盖场景：非穿门路径进入新房间后第一次执行 HOVER_SCAN
            import time as _hs_re_t
            _cur_rid  = int(getattr(self, 'current_room_id', -1))
            _last_rid = int(getattr(self, '_last_hover_scan_room_id', -1))
            _re_cd    = float(getattr(self, 'room_entry_scan_timeout_sec', 20.0)) * 2.0
            _last_re  = float(getattr(self, '_last_room_entry_trigger_t', 0.0))
            _hs_re_now = _hs_re_t.monotonic()
            _new_room_entry = (
                _cur_rid >= 0
                and _last_rid >= 0
                and _cur_rid != _last_rid
                and _hs_re_now - _last_re > max(30.0, _re_cd * 0.5)
                and not getattr(self, '_tgt_weak_confirmed', False)
            )
            if _new_room_entry:
                _yaw_now_re = float(self.att_yaw) if self.att_yaw is not None else 0.0
                _span_deg_re = float(getattr(self, 'room_entry_scan_yaw_span_deg', 120.0))
                import math as _hs_re_m
                _span_rad_re = _span_deg_re * _hs_re_m.pi / 180.0
                _n_re = max(3, int(round(_span_deg_re / float(getattr(self, 'room_entry_scan_step_deg', 40.0)))) + 1)
                _steps_re = [_yaw_now_re - _span_rad_re / 2.0 + _si * _span_rad_re / (_n_re - 1.0)
                             for _si in range(_n_re)]
                self._room_entry_scan_active = True
                self._room_entry_scan_start_t = _hs_re_now
                self._room_entry_scan_yaw_base = _yaw_now_re
                self._room_entry_scan_step = 0
                self._room_entry_scan_step_enter_t = _hs_re_now
                self._room_entry_scan_steps = _steps_re
                self._room_entry_target_found = False
                self._last_room_entry_trigger_t = _hs_re_now
                self._last_hover_scan_room_id = _cur_rid  # 更新房间记录
                # ★ Fix-RCA1: 在转入 ROOM_ENTRY_SCAN 之前把 hold_pos 更新为当前位置，
                # 防止使用上一轮 HOVER_SCAN 留下的过时 hold_pos（可能相差 1-3m）→ 猛飞
                if self.pose is not None:
                    self._hover_scan_hold_pos = (
                        float(self.pose.x),
                        float(self.pose.y),
                        float(self.pose.z),
                    )
                else:
                    self._hover_scan_hold_pos = None
                self.explore_state = "ROOM_ENTRY_SCAN"
                self.get_logger().warn(
                    f"[ROOM-ENTRY-SCAN-ENTER] HOVER_SCAN检测到新房间 room={_cur_rid}(旧={_last_rid}) "
                    f"→ 房间视觉扇扫 yaw_base={_yaw_now_re:.2f}rad steps={len(_steps_re)} span={_span_deg_re:.0f}°"
                )
                self._log_event("ROOM-ENTRY-SCAN-ENTER",
                    f"reason=new_room room={_cur_rid} last={_last_rid} yaw_base={_yaw_now_re:.2f} steps={len(_steps_re)}")
                return np.zeros(3, dtype=float)
            # 更新房间记录（无论是否触发扇扫，每次启动 HOVER_SCAN 时更新）
            if _cur_rid >= 0:
                self._last_hover_scan_room_id = _cur_rid
            self.hover_scan_start_time = _now_mono   # ← 使用 monotonic()，不受 sim_time 影响
            self.hover_scan_yaw_start = float(self.att_yaw) if self.att_yaw is not None else 0.0
            self.hover_scan_accum_angle = 0.0
            self._hover_scan_last_yaw = float(self.att_yaw) if self.att_yaw is not None else None
            self._hover_valid_frames = 0
            self._hg_total = 0; self._hg_rejected = 0
            self._hg_rej_drift = 0; self._hg_rej_vz = 0; self._hg_rej_tilt = 0
            if self.pose is not None:
                self._hover_scan_hold_pos = (
                    float(self.pose.x),
                    float(self.pose.y),
                    float(self.pose.z),
                )
            else:
                self._hover_scan_hold_pos = None
            self._hover_scan_anchor_map = (float(self._plan_px), float(self._plan_py))
            self._map_frozen = False
            self.locked_yaw = None
            self._log_event(
                "HOVER-SCAN-START",
                f"yaw_start={np.rad2deg(self.hover_scan_yaw_start):.1f}° "
                f"anchor_plan=({self._plan_px:.2f},{self._plan_py:.2f}) "
                f"map=({self._map_px:.2f},{self._map_py:.2f}) → 地图解冻"
            )

        # ★ Phase5/Phase9: 障碍物过近检查（立即停止旋转统计，触发 command_loop escape）
        _near_dist = self._nearest_lidar_distance()
        _emerg_active = (_near_dist is not None and _near_dist < self._hover_scan_emerg_stop_dist)
        if _emerg_active:
            _t_emerg = getattr(self, '_hover_emerg_log_mono', 0.0)
            if _now_mono - _t_emerg > 1.0:
                self._hover_emerg_log_mono = _now_mono
                self.get_logger().warn(
                    f"[HOVER-EMERG] 障碍过近 {_near_dist:.2f}m < {self._hover_scan_emerg_stop_dist:.2f}m"
                    f" → 停止旋转统计，委托 command_loop 执行 ESCAPE"
                )
            # 确保 command_loop 的 escape 状态已激活（同时设 escape_dir，避免 command_loop 崩溃）
            if not getattr(self, '_hover_scan_escape_active', False):
                _esc_v_hem = self.apply_proximity_constraints(np.zeros(3, dtype=float))
                _en_hem = float(np.linalg.norm(_esc_v_hem[:2]))
                if _en_hem > 0.01:
                    self._hover_scan_escape_dir = _esc_v_hem[:2] / _en_hem
                else:
                    _yw_hem = float(self.att_yaw) if self.att_yaw is not None else 0.0
                    self._hover_scan_escape_dir = np.array([-np.cos(_yw_hem), -np.sin(_yw_hem)])
                self._hover_scan_escape_active = True
                import time as _hem_t
                self._hover_scan_escape_start = _hem_t.monotonic()
                self._hover_scan_escape_clear_since = 0.0
            elif self._hover_scan_escape_dir is None:
                # 防御：escape_active=True 但 dir 为 None（二次触发时补设方向）
                _yw_hem2 = float(self.att_yaw) if self.att_yaw is not None else 0.0
                self._hover_scan_escape_dir = np.array([-np.cos(_yw_hem2), -np.sin(_yw_hem2)])
            # 立即返回零速（不继续统计旋转角度）
            return np.zeros(3, dtype=float)
        _esc_vel = np.zeros(3, dtype=float)

        # ★ HOVER_SCAN 稳定门控（放宽阈值：PX4悬停噪声约±10cm，原5cm过严）
        # 使用可配置参数：_hover_gate_drift_m / _hover_gate_vz_mps / _hover_gate_tilt_rad
        # 每100帧输出门控统计（[HOVER_SCAN-GATE]），方便诊断拒绝原因
        _can_scan = True
        if self._hover_scan_anchor_map is not None:
            _anchor_map = self._hover_scan_anchor_map
            _drift  = float(np.hypot(self._plan_px - _anchor_map[0], self._plan_py - _anchor_map[1]))
            _vz     = float(getattr(self.pose, 'vz', 0.0)) if self.pose is not None else 0.0
            _pitch  = float(self.att_pitch) if self.att_pitch is not None else 0.0
            _roll   = float(self.att_roll)  if self.att_roll  is not None else 0.0
            _rej_drift = _drift > self._hover_gate_drift_m
            _rej_vz    = abs(_vz) > self._hover_gate_vz_mps
            _rej_tilt  = (abs(_pitch) > self._hover_gate_tilt_rad
                          or abs(_roll) > self._hover_gate_tilt_rad)
            self._hg_total += 1
            if _rej_drift: self._hg_rej_drift += 1
            if _rej_vz:    self._hg_rej_vz    += 1
            if _rej_tilt:  self._hg_rej_tilt  += 1
            if _rej_drift or _rej_vz or _rej_tilt:
                _can_scan = False
                self._hg_rejected += 1
                _t = getattr(self, '_scan_gate_warned_mono', 0.0)
                if _now_mono - _t > 1.5:
                    self._scan_gate_warned_mono = _now_mono
                    self.get_logger().debug(
                        f"[HOVER_SCAN] 门控拒建图: "
                        f"drift={_drift:.3f}m({'✗' if _rej_drift else '✓'} th={self._hover_gate_drift_m}) "
                        f"vz={_vz:.3f}({'✗' if _rej_vz else '✓'} th={self._hover_gate_vz_mps}) "
                        f"tilt={np.rad2deg(max(abs(_pitch), abs(_roll))):.1f}°"
                        f"({'✗' if _rej_tilt else '✓'} th={np.rad2deg(self._hover_gate_tilt_rad):.1f}°)"
                    )
            # 每100帧汇报门控有效率
            if self._hg_total > 0 and self._hg_total % 100 == 0:
                _acc = 100.0 * (self._hg_total - self._hg_rejected) / self._hg_total
                self.get_logger().info(
                    f"[HOVER_SCAN-GATE] 累计={self._hg_total} "
                    f"有效={self._hg_total-self._hg_rejected}({_acc:.0f}%) "
                    f"拒绝: 漂移={self._hg_rej_drift} vz={self._hg_rej_vz} 倾斜={self._hg_rej_tilt}"
                )
        # 统计有效扫描帧并记录（每20帧打印一次）
        if _can_scan:
            self._hover_valid_frames += 1
            if self._hover_valid_frames % 20 == 0:
                self.get_logger().info(
                    f"[HOVER_SCAN] 有效帧={self._hover_valid_frames} "
                    f"angle={np.rad2deg(self.hover_scan_accum_angle):.0f}°"
                )
        # 根据稳定性决定当前帧是否允许 belief 写入（_map_frozen 控制 update_map_from_scan）
        self._map_frozen = not _can_scan

        # ★ Phase9: elapsed 使用 monotonic()——不受 sim_time 缩放影响
        elapsed = _now_mono - self.hover_scan_start_time
        free_count = int(np.sum(self.grid <= self.prob_free_th))

        # 累计旋转角度：用增量累加（正确处理超过180°的旋转）
        # 每帧与上一帧yaw的差值累加，normalize到[-π,π]后取绝对值叠加
        if self.att_yaw is not None:
            if not hasattr(self, '_hover_scan_last_yaw') or self._hover_scan_last_yaw is None:
                self._hover_scan_last_yaw = float(self.att_yaw)
            else:
                d_yaw = float(self.att_yaw) - self._hover_scan_last_yaw
                while d_yaw > np.pi: d_yaw -= 2 * np.pi
                while d_yaw < -np.pi: d_yaw += 2 * np.pi
                self.hover_scan_accum_angle += abs(d_yaw)
                self._hover_scan_last_yaw = float(self.att_yaw)

        min_angle = np.deg2rad(self.hover_scan_min_angle_deg)
        # scan_done 条件加严：角度 + 有效帧 + 2s内无ESCAPE + 安全clearance + 姿态 + 地图质量
        import time as _hst
        _now_hst = _hst.monotonic()
        _no_recent_escape = (_now_hst - self._hover_scan_last_escape_t) > 2.0
        _scan_near = self._nearest_lidar_distance()
        # ★ 修复：clearance 阈值从 0.55m 降至 0.30m——走廊内壁距通常 0.3~0.5m，
        # 0.55m 要求导致 HOVER_SCAN 在任何真实走廊中都永远完成不了。
        _scan_clearance_ok = (_scan_near is None or _scan_near >= 0.30)
        _scan_att_ok = (
            (self.att_roll is None or abs(self.att_roll) <= np.deg2rad(8.0))
            and (self.att_pitch is None or abs(self.att_pitch) <= np.deg2rad(8.0))
        )
        _scan_wsegs = getattr(self, '_last_wall_seg_count', 0)
        _scan_occ = int(np.count_nonzero(self.grid >= self.prob_occ_th)) if self.grid is not None else 0
        # ★ 修复：地图质量阈值从 (wsegs≥20, occ≥80) 降至 (wsegs≥3, occ≥10)——
        # 原阈值在初始建图阶段永远无法满足，导致完成条件永久封锁（地图锁死根因之一）。
        _scan_map_quality = (_scan_wsegs >= 3 or _scan_occ >= 10)
        _angle_done = (
            self.hover_scan_accum_angle >= min_angle
            and free_count >= self.mapping_min_free_cells
            and self._hover_valid_frames >= self._hover_scan_min_valid_frames
            and _no_recent_escape
            and not self._hover_scan_escape_active
            and _scan_clearance_ok
            and _scan_att_ok
            and _scan_map_quality
        )
        # [HSCAN-DONE-CHECK] 每2秒打印一次当前旋转进度
        if _now_hst - getattr(self, '_hscan_angle_log_t', 0.0) > 2.0:
            self._hscan_angle_log_t = _now_hst
            _near_str_hsc = f"{_scan_near:.2f}" if _scan_near is not None else "None"
            self.get_logger().debug(
                f"[HSCAN-DONE-CHECK] elapsed={elapsed:.1f}s "
                f"accum={np.rad2deg(self.hover_scan_accum_angle):.1f}°"
                f"/{np.rad2deg(min_angle):.0f}° "
                f"frames={self._hover_valid_frames}/{self._hover_scan_min_valid_frames} "
                f"near={_near_str_hsc}(≥0.55={_scan_clearance_ok}) "
                f"att_ok={_scan_att_ok} wsegs={_scan_wsegs}(≥20) occ={_scan_occ}(≥80) "
                f"no_esc={_no_recent_escape} pass={int(_angle_done)}"
            )
        # [HOVER-SCAN-FAIL] 超时但未完成完整条件 → 重置并重试，不标记为成功
        if elapsed >= self.hover_scan_duration and not _angle_done:
            self.get_logger().warn(
                f"[HSCAN-TIMEOUT] reason=timeout_insufficient_quality "
                f"elapsed={elapsed:.1f}s "
                f"accum={np.rad2deg(self.hover_scan_accum_angle):.1f}° < {np.rad2deg(min_angle):.1f}° "
                f"valid_frames={self._hover_valid_frames}/{self._hover_scan_min_valid_frames} "
                f"clearance_ok={_scan_clearance_ok} att_ok={_scan_att_ok} "
                f"map_quality_ok={_scan_map_quality}(wsegs={_scan_wsegs},occ={_scan_occ}) "
                f"→ reset and retry"
            )
            self.hover_scan_start_time = None
            self.hover_scan_accum_angle = 0.0
            self._hover_scan_last_yaw = None
            return np.zeros(3, dtype=float)
        # ★ 降级完成条件：escape 失败 >=1 次 或 已扫描 >10s，放宽到 120°+10帧（无需clearance/att/quality）
        # 降低门槛：10s（原12s），120°（原180°），10帧（原15帧）
        _esc_fail_cnt = getattr(self, '_hover_scan_escape_fail_count', 0)
        _degraded_mode = (_esc_fail_cnt >= 1 or elapsed > 10.0)
        _angle_done_degraded = (
            _degraded_mode
            and self.hover_scan_accum_angle >= np.deg2rad(120.0)
            and self._hover_valid_frames >= 10
            and not self._hover_scan_escape_active
        )
        if _angle_done_degraded and not _angle_done:
            self.get_logger().warn(
                f"[HSCAN-DEGRADED-DONE] 降级完成 esc_fail={_esc_fail_cnt} "
                f"elapsed={elapsed:.1f}s "
                f"accum={np.rad2deg(self.hover_scan_accum_angle):.0f}° "
                f"frames={self._hover_valid_frames} → 强制结束扫描，去下一位置"
            )
        scan_done = _angle_done or _angle_done_degraded

        if scan_done:
            # ★ 修复：扫描完成后不再冻结地图。
            # 导航期间（NAV_TO_VIEWPOINT）地图应继续更新，否则无人机无法感知新区域。
            # update_map_from_scan() 内部已有质量门控（ESCAPE激活/近墙/倾斜等），不依赖 _map_frozen。
            # self._map_frozen = True  # ← 已移除，地图持续更新
            if self.att_yaw is not None:
                self.locked_yaw = float(self.att_yaw)

            # 记录本次扫描点（用于重访惩罚）
            if self.pose is not None:
                sp = (self._plan_px, self._plan_py)
                self.ssg_visited_scan_pts.append(sp)
                if len(self.ssg_visited_scan_pts) > 30:
                    self.ssg_visited_scan_pts.pop(0)
                # ★[D1] 空间冷却TTL网格：标记本次扫描位置（90s内不重访）
                self._spatial_cooldown_stamp(self._plan_px, self._plan_py)

            # ★ 探索扩散检测：如果无人机近期扫描点都集中在小区域，激活"远距探索模式"
            # 防止无人机在一个房间内反复小步移动而不穿越到新房间。
            # 策略：检查最近 N 个扫描点的分布半径；若 < spread_radius_m，标记聚集状态。
            _spread_check_n = 5       # 至少需要 N 个扫描点才判断
            _spread_radius_m = 2.5    # 若最近 N 点都在半径 2.5m 内 = 聚集
            if len(self.ssg_visited_scan_pts) >= _spread_check_n:
                _recent = np.array(self.ssg_visited_scan_pts[-_spread_check_n:])
                _center = np.mean(_recent, axis=0)
                _max_r = float(np.max(np.linalg.norm(_recent - _center, axis=1)))
                _was_clustered = getattr(self, '_ssg_clustered', False)
                self._ssg_clustered = (_max_r < _spread_radius_m)
                if self._ssg_clustered and not _was_clustered:
                    self.get_logger().warn(
                        f"[SSG-SPREAD] 检测到无人机集聚（近{_spread_check_n}次扫描半径={_max_r:.1f}m"
                        f" < {_spread_radius_m}m）→ 激活远距探索模式（降低距离惩罚权重）"
                    )
                elif not self._ssg_clustered and _was_clustered:
                    self.get_logger().info(
                        f"[SSG-SPREAD] 无人机已扩散（近{_spread_check_n}次扫描半径={_max_r:.1f}m"
                        f" >= {_spread_radius_m}m）→ 恢复正常探索模式"
                    )
            else:
                self._ssg_clustered = False

            # 选下一扫描点
            next_vp = self._ssg_select_viewpoint()
            if next_vp is not None:
                self.ssg_viewpoint_goal = next_vp
                self.astar_path = None
                self.astar_path_idx = 0
                self.astar_last_plan_time = 0.0
                self._ssg_stuck_pos = None
                self._ssg_stuck_since = None
                self.explore_state = "NAV_TO_VIEWPOINT"
                self._log_event(
                    "HOVER-SCAN-DONE",
                    f"elapsed={elapsed:.1f}s free={free_count} "
                    f"angle={np.rad2deg(self.hover_scan_accum_angle):.0f}° "
                    f"有效帧={self._hover_valid_frames} "
                    f"→ NAV_TO_VIEWPOINT ({next_vp[0]:.2f},{next_vp[1]:.2f})"
                )
            else:
                # ★ Phase9: 无候选扫描点时 → 尝试 frontier 逃逸点，而非只是继续扫描
                # 驼留在同一位置反复扫描是探索卡死的根因
                _frontier_esc = self._find_frontier_escape_point()
                if _frontier_esc is not None:
                    self.ssg_viewpoint_goal = _frontier_esc
                    self.ssg_goal_type = "frontier_escape"
                    self.astar_path = None
                    self.astar_path_idx = 0
                    self.astar_last_plan_time = 0.0
                    self._ssg_stuck_pos = None
                    self._ssg_stuck_since = None
                    self.explore_state = "NAV_TO_VIEWPOINT"
                    self._log_event(
                        "HOVER-SCAN-DONE-FRONTIER",
                        f"elapsed={elapsed:.1f}s 无常规视点 → frontier逃逸点 ({_frontier_esc[0]:.2f},{_frontier_esc[1]:.2f})"
                    )
                else:
                    # 最后手段：重新解冻地图，等候选点出现
                    self.get_logger().warn("[HOVER_SCAN] 无候选扫描点且无frontier，重新解冻地图继续扫描")
                    self._map_frozen = False
                    self.locked_yaw = None

            self.hover_scan_start_time = None
            self.hover_scan_accum_angle = 0.0
            self._hover_scan_last_yaw = None
            return np.zeros(3, dtype=float)

        # ★ Phase9: HOVER_SCAN 卡死看门狗：超过 MAX_HOVER_MONO_SEC 没有完成，强制不完成切 NAV
        _max_hover_mono = float(getattr(self, '_hover_scan_max_mono_sec', 90.0))
        if elapsed > _max_hover_mono:
            self.get_logger().warn(
                f"[HOVER-WD] HOVER_SCAN 超时 {elapsed:.0f}s > {_max_hover_mono:.0f}s → 强制切 NAV_TO_VIEWPOINT/SAFE_HOLD"
            )
            # ★ 修复：看门狗超时也不冻结地图
            # self._map_frozen = True  # 已移除
            if self.att_yaw is not None:
                self.locked_yaw = float(self.att_yaw)
            next_vp = self._ssg_select_viewpoint()
            if next_vp is None:
                next_vp = self._find_frontier_escape_point()
            if next_vp is not None:
                self.ssg_viewpoint_goal = next_vp
                self.astar_path = None
                self.astar_path_idx = 0
                self.explore_state = "NAV_TO_VIEWPOINT"
                self._log_event("HOVER-WD", f"forced NAV_TO_VIEWPOINT at ({next_vp[0]:.2f},{next_vp[1]:.2f})")
            else:
                self.get_logger().warn("[HOVER-WD] 无贞点且无frontier，重置 HOVER_SCAN")
            self.hover_scan_start_time = None
            self.hover_scan_accum_angle = 0.0
            self._hover_scan_last_yaw = None
            return np.zeros(3, dtype=float)

        # 未完成：XY悬停，偏航持续旋转（由 command_loop 的 yawspeed 控制）
        return _esc_vel

    def _nav_to_viewpoint_cmd(self, now, now_sec: float) -> np.ndarray:
        """
        A*导航到下一扫描点（地图冻结期间）。
        到达后切 HOVER_SCAN。卡住超时→重选目标。
        """
        # ── 地图稳定性门控（Task3）────────────────────────────────────────────
        # 若检测到 SLAM origin/size 跳变，暂停导航，切回 HOVER_SCAN 等待地图稳定
        # ★ 门口目标例外：地图跳变时不切 HOVER_SCAN（避免中断门口穿越），原地等待
        import time as _nvmt
        if not getattr(self, '_map_stability_ok', True):
            _rem = getattr(self, '_map_stab_pause_until', 0.0) - _nvmt.monotonic()
            if _rem > 0.0:
                _lt = getattr(self, '_map_stab_nav_log_t', 0.0)
                _gtype_early2 = getattr(self, 'ssg_goal_type', 'none')
                _is_door_early = _gtype_early2 in (
                    'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                    'frontier_gap_post', 'confirmed_door')
                if now_sec - _lt > 3.0:
                    self._map_stab_nav_log_t = now_sec
                    _action = "hover_in_place(door)" if _is_door_early else "HOVER_SCAN"
                    self.get_logger().warn(
                        f"[GOAL-REPLACE] reason=map_unstable pause_rem={_rem:.1f}s type={_gtype_early2} → {_action}"
                    )
                if not _is_door_early:
                    self.explore_state = "HOVER_SCAN"
                    self.hover_scan_start_time = None
                    self.astar_path = None
                return np.zeros(3, dtype=float)

        # ─── 周期诊断日志（2s一次）：显示当前导航目标类型、距离、控制来源 ──────────
        _gtype_early = getattr(self, 'ssg_goal_type', 'none')
        _nav_log_t = getattr(self, '_nav_vp_log_t', 0.0)
        if now_sec - _nav_log_t > 2.0:
            self._nav_vp_log_t = now_sec
            _vp = self.ssg_viewpoint_goal
            if _vp is not None and self.pose is not None:
                _d2g = float(np.linalg.norm([self._plan_px - _vp[0], self._plan_py - _vp[1]]))
                self.get_logger().info(
                    f"[NAV-STATUS] type={_gtype_early} goal=({_vp[0]:.2f},{_vp[1]:.2f}) "
                    f"dist={_d2g:.2f}m ctrl_src=explore(SSG_primary)"
                )

        if self.ssg_viewpoint_goal is None:
            self.explore_state = "HOVER_SCAN"
            self.hover_scan_start_time = None
            return np.zeros(3, dtype=float)

        if self.pose is None:
            return np.zeros(3, dtype=float)

        px, py = self._plan_px, self._plan_py
        pos_now = np.array([px, py])   # 提前定义，供 DOOR-PHASE-COMMIT 等提前返回路径使用
        dist = float(np.linalg.norm([px - self.ssg_viewpoint_goal[0], py - self.ssg_viewpoint_goal[1]]))

        # ── 按目标类型选择到达判定距离 ────────────────────────────────────────────
        # frontier_gap_probe（穿门外侧点）：最小判定距离，确保真正穿过门洞
        # confirmed_door / frontier_gap_center：中等判定距离
        # 普通 frontier / coverage（室内扫描点）：原始较大判定距离
        _gtype = getattr(self, 'ssg_goal_type', 'none')
        if _gtype == 'frontier_gap_probe':
            _reach_dist = 0.25          # 穿门外侧点：0.25m，确保真正穿过
        elif _gtype in ('confirmed_door', 'frontier_gap_center'):
            _reach_dist = 0.35          # 门中心/内侧点：0.35m
        elif _gtype in ('frontier_gap_pre', 'frontier_gap_post'):
            _reach_dist = 0.30          # 门入口/出口点：0.30m
        else:
            _reach_dist = self.ssg_reach_dist   # 普通点：0.60m

        # 到达扫描点：触发 HOVER_SCAN
        # ★ Phase12: 到点需连续满足 _ARRIVE_DWELL_T 秒，且不能在近墙状态下误判
        _ARRIVE_DWELL_T    = 0.5    # 到点连续满足时间(s)
        _ARRIVE_CLR_GUARD  = 0.22   # 低于此clearance时不得判到达(m)
        _near_now = self._nearest_lidar_distance()
        _clr_ok   = (_near_now is None or _near_now >= _ARRIVE_CLR_GUARD)
        if dist < _reach_dist and _clr_ok:
            _dwell_t0 = getattr(self, '_arrive_dwell_t', None)
            if _dwell_t0 is None:
                self._arrive_dwell_t = now_sec
            elif now_sec - _dwell_t0 >= _ARRIVE_DWELL_T:
                self._arrive_dwell_t = None  # 重置

                # ★ 三阶段门穿越提交（DOOR-PHASE-COMMIT）：
                # pre→center→post 顺序执行，中途跳过 HOVER_SCAN，避免重选普通目标打断穿门序列
                # ─────────────────────────────────────────────────────────────
                if _gtype == 'frontier_gap_pre':
                    # 找到当前目标附近的 frontier_gap 作为门中心
                    import math as _dpc_m
                    _best_fgap = None
                    _best_fgap_d = 99.0
                    for _fg in getattr(self, 'frontier_gaps', []):
                        _fd = _dpc_m.hypot(_fg['x'] - self.ssg_viewpoint_goal[0],
                                           _fg['y'] - self.ssg_viewpoint_goal[1])
                        if _fd < _best_fgap_d:
                            _best_fgap_d = _fd
                            _best_fgap = _fg
                    if _best_fgap is not None and _best_fgap_d < 1.6:
                        _gap_wx, _gap_wy = float(_best_fgap['x']), float(_best_fgap['y'])
                        _ori_g = str(_best_fgap.get("orientation", ""))
                        if _ori_g == "horizontal":  # 水平墙(y=const) → 穿越方向y轴
                            _ndx, _ndy = 0.0, (1.0 if py >= _gap_wy else -1.0)
                        elif _ori_g == "vertical":  # 垂直墙(x=const) → 穿越方向x轴
                            _ndx, _ndy = (1.0 if px >= _gap_wx else -1.0), 0.0
                        else:
                            _ddx_ = px - _gap_wx; _ddy_ = py - _gap_wy
                            _dd_ = max(0.001, _dpc_m.hypot(_ddx_, _ddy_))
                            _ndx, _ndy = _ddx_ / _dd_, _ddy_ / _dd_
                        _post_x = _gap_wx - _ndx * 1.0
                        _post_y = _gap_wy - _ndy * 1.0
                        # ★ 安全校验1：检查 gap_center 净空（核心防撞逻辑）
                        # _plan_bounds_xy 是配置的全图范围(±8m)，无法代表物理房间边界，
                        # 必须改用距离变换(dt_map)来判断 gap 是否真实可穿越。
                        # 典型假门：gap center距墙0.3m（地图边界伪门），dt<0.33 → ABORT
                        # 典型真门：开口≥0.8m，center距门框≥0.4m，dt≥0.4 → PASS
                        _GAP_CTR_MIN_CLR = 0.33   # 真实可穿越门洞的中心净空下限
                        try:
                            _dpc_dt = self._ssg_compute_dist_transform()
                        except Exception:
                            _dpc_dt = None
                        _gap_cgx, _gap_cgy = self.world_to_grid(_gap_wx, _gap_wy)
                        _gap_cgx = int(np.clip(_gap_cgx, 0, self.grid_shape[0]-1))
                        _gap_cgy = int(np.clip(_gap_cgy, 0, self.grid_shape[1]-1))
                        _gap_ctr_clr = float(_dpc_dt[_gap_cgx, _gap_cgy]) if _dpc_dt is not None else 1.0
                        if _gap_ctr_clr < _GAP_CTR_MIN_CLR:
                            # 假门洞（地图边界伪门）：中心净空不足，增加失败计数
                            _dfc_key = f"{_gap_wx:.1f}_{_gap_wy:.1f}"
                            if not hasattr(self, '_door_fail_counts'):
                                self._door_fail_counts = {}
                            self._door_fail_counts[_dfc_key] = (
                                self._door_fail_counts.get(_dfc_key, 0) + 1
                            )
                            _dfc_cnt = self._door_fail_counts[_dfc_key]
                            self._log_event("DOOR-PHASE-ABORT",
                                f"gap_center净空不足 clr={_gap_ctr_clr:.2f}<{_GAP_CTR_MIN_CLR} "
                                f"gap=({_gap_wx:.2f},{_gap_wy:.2f}) fail={_dfc_cnt} → 假边界门洞")
                            self.get_logger().warn(
                                f"[DOOR-PHASE-ABORT] gap_center净空 {_gap_ctr_clr:.2f}m < "
                                f"{_GAP_CTR_MIN_CLR}m，判定为地图边界假门 "
                                f"gap=({_gap_wx:.2f},{_gap_wy:.2f}) fail={_dfc_cnt} → HOVER_SCAN"
                            )
                            # ★[D2] 清除目标锁定和强制过门模式，防止 goal_hold_until 继续锁定
                            # 导致下次 _ssg_select_viewpoint 仍然强制选同一假门的 gap_pre → 无限循环
                            self._goal_hold_until = 0.0
                            self._passage_commit_active = False
                            # 不提交，fall-through → 正常到达处理 → HOVER_SCAN
                        else:
                            # ★ 安全校验2：post点校验（使用实际栅格，不再用 _plan_bounds_xy）
                            def _dpc_post_valid(px_, py_):
                                _gx_, _gy_ = self.world_to_grid(px_, py_)
                                if not (0 <= _gx_ < self.grid_shape[0] and
                                        0 <= _gy_ < self.grid_shape[1]):
                                    return False
                                # 明确占据格拒绝
                                if self.grid[_gx_, _gy_] >= self.prob_occ_th:
                                    return False
                                # 距墙极近（<0.15m）时也拒绝（进入已知占据边缘）
                                if _dpc_dt is not None and _dpc_dt[_gx_, _gy_] < 0.15:
                                    return False
                                return True
                            _post_ok = _dpc_post_valid(_post_x, _post_y)
                            if not _post_ok:
                                for _td in (0.7, 0.5, 0.3, 0.0):
                                    _ptx = _gap_wx - _ndx * _td
                                    _pty = _gap_wy - _ndy * _td
                                    if _dpc_post_valid(_ptx, _pty):
                                        _post_x, _post_y = _ptx, _pty
                                        _post_ok = True
                                        break
                            if not _post_ok:
                                # 兜底：post退化为 gap_center（已验证净空足够）
                                _post_x, _post_y = _gap_wx, _gap_wy
                            # 存储三阶段信息供后续阶段使用
                            self._door_phase_commit = {
                                'center': (_gap_wx, _gap_wy),
                                'post':   (_post_x, _post_y),
                                'gap_pos': (_gap_wx, _gap_wy),
                            }
                            self.get_logger().warn(
                                f"[DOOR-PHASE] ★ gap_pre到达→直接提交gap_center=({_gap_wx:.2f},{_gap_wy:.2f}) "
                                f"跳过HOVER_SCAN 门法向=({_ndx:.1f},{_ndy:.1f}) "
                                f"gap_dist={_best_fgap_d:.2f}m post=({_post_x:.2f},{_post_y:.2f}) "
                                f"gap_ctr_clr={_gap_ctr_clr:.2f}m"
                            )
                            self._log_event("DOOR-PHASE",
                                f"pre→center gap=({_gap_wx:.2f},{_gap_wy:.2f}) post=({_post_x:.2f},{_post_y:.2f})")
                            self.ssg_viewpoint_goal = np.array([_gap_wx, _gap_wy], dtype=float)
                            self.ssg_goal_type = "frontier_gap_center"
                            self.astar_path = None
                            self.astar_path_idx = 0
                            self.astar_last_plan_time = 0.0
                            self._ssg_stuck_since = now_sec
                            self._ssg_stuck_pos = pos_now.copy()
                            return np.zeros(3, dtype=float)

                elif _gtype == 'frontier_gap_center':
                    _dpc = getattr(self, '_door_phase_commit', None)
                    if _dpc is not None:
                        _post_x2, _post_y2 = _dpc['post']
                        # ★ 二次安全校验：center到达时再次确认post点合法性（使用实际栅格占据，非plan_bounds）
                        _pgx2, _pgy2 = self.world_to_grid(_post_x2, _post_y2)
                        _post2_valid = (0 <= _pgx2 < self.grid_shape[0] and
                                        0 <= _pgy2 < self.grid_shape[1] and
                                        self.grid[_pgx2, _pgy2] < self.prob_occ_th)
                        if not _post2_valid:
                            self._door_phase_commit = None
                            self._log_event("DOOR-PHASE-ABORT",
                                f"center到达后post二次校验失败 post=({_post_x2:.2f},{_post_y2:.2f}) "
                                f"→ 放弃出口段 → HOVER_SCAN")
                            self.get_logger().warn(
                                f"[DOOR-PHASE-ABORT] gap_center到达但post占据/越界=({_post_x2:.2f},{_post_y2:.2f}) "
                                f"→ 放弃门穿越出口段 → HOVER_SCAN"
                            )
                            # 不返回，落入后续到达清除逻辑 → HOVER_SCAN
                        else:
                            self.get_logger().warn(
                                f"[DOOR-PHASE] ★ gap_center到达→直接提交gap_post=({_post_x2:.2f},{_post_y2:.2f}) "
                                f"跳过HOVER_SCAN 正在穿越中"
                            )
                            self._log_event("DOOR-PHASE", f"center→post post=({_post_x2:.2f},{_post_y2:.2f})")
                            self.ssg_viewpoint_goal = np.array([_post_x2, _post_y2], dtype=float)
                            self.ssg_goal_type = "frontier_gap_post"
                            self.astar_path = None
                            self.astar_path_idx = 0
                            self.astar_last_plan_time = 0.0
                            self._ssg_stuck_since = now_sec
                            self._ssg_stuck_pos = pos_now.copy()
                            return np.zeros(3, dtype=float)

                elif _gtype == 'frontier_gap_post':
                    import time as _cross_t
                    _dpc2 = getattr(self, '_door_phase_commit', None)
                    if _dpc2 is not None and 'gap_pos' in _dpc2:
                        # ★ 通过 DOOR-PHASE 正常穿门 → 记录回穿禁止信息
                        _crossed_gap = _dpc2['gap_pos']
                        self._last_crossed_gap_pos = _crossed_gap
                        self._last_crossed_gap_at = _cross_t.monotonic()
                        self._door_phase_commit = None
                        self._passage_commit_active = False   # 退出强制过门模式
                        self._near_gap_fail_count = 0
                        self.get_logger().warn(
                            f"[DOOR-CROSSED] ★★ 穿门成功! gap_pos=({_crossed_gap[0]:.2f},{_crossed_gap[1]:.2f}) "
                            f"→ 激活回穿禁止10s，进入POST_DOOR_COMMIT保护"
                        )
                        self._log_event("DOOR-CROSSED",
                            f"gap=({_crossed_gap[0]:.2f},{_crossed_gap[1]:.2f}) reentry_block=10s → POST_DOOR_COMMIT")
                        # ★[A] 穿门成功→POST_DOOR_COMMIT保护：惯性推进，跳过HOVER_SCAN
                        import time as _pdc_enter_t
                        self._post_door_commit_start_t = _pdc_enter_t.monotonic()
                        self._post_door_commit_start_pos = (
                            float(self.pose.x) if self.pose is not None else 0.0,
                            float(self.pose.y) if self.pose is not None else 0.0,
                        )
                        self._hover_scan_escape_fail_count = 0  # 重置失败计数
                        # ★[F] 穿门成功：标记门口区域空间冷却防止重选
                        self._spatial_cooldown_stamp(self._plan_px, self._plan_py)
                        self.explore_state = "POST_DOOR_COMMIT"
                    else:
                        # ★[D3] 未经 DOOR-PHASE 直接抵达 frontier_gap_post
                        # 判断无人机是否确实已穿越门洞：
                        #   - 在当前位置附近查找最近的 frontier_gap
                        #   - 若无人机在 gap 的"出口侧"（法向量方向 dot < 0），认为已成功穿越
                        #   - 成功穿越 → DOOR-CROSSED 处理 + 短暂（15s）黑名单
                        #   - 否则（假门/同侧滑动）→ 90s 黑名单
                        import math as _fgp_m
                        _fgp_gaps = list(getattr(self, 'frontier_gaps', []))
                        _fgp_post = self.ssg_viewpoint_goal
                        _nearest_gap_d = float('inf')
                        _nearest_gap   = None
                        for _fg in _fgp_gaps:
                            _fgd = _fgp_m.hypot(_fg['x'] - _fgp_post[0], _fg['y'] - _fgp_post[1])
                            if _fgd < _nearest_gap_d:
                                _nearest_gap_d = _fgd
                                _nearest_gap = _fg
                        _actually_crossed = False
                        _gap_ref_pos = None
                        if _nearest_gap is not None and _nearest_gap_d < 1.8:
                            # 计算 gap 法向量（从 gap 指向当初接近侧，即 drone 之前在的侧）
                            # "接近侧" = gap 法向量 = 目标选定时 gap_pre 所在侧方向
                            # gap_post = gap_center - n*1.0，所以从 gap_center 到 drone 的向量
                            # == -n*1.0（drone 在出口侧），dot(drone-gap, n) < 0 → 已穿越
                            _gx_ref = float(_nearest_gap['x'])
                            _gy_ref = float(_nearest_gap['y'])
                            _ori_ref = str(_nearest_gap.get('orientation', ''))
                            _dx_gap = px - _gx_ref
                            _dy_gap = py - _gy_ref
                            # gap_pre 侧方向 n（单位向量）：用当前 drone 位置估算
                            # gap_post 在 "gap - n*1.0" 处，n = (gap → pre) 方向
                            # drone 到达 gap_post 意味着 drone 在 gap 的 post 侧（-n 侧）
                            # 简单判断：drone 到 gap_post 的距离 < gap 到 gap_post 的距离+容差
                            # 即 gap_post 确实在门另一侧（非同侧滑动）
                            _post2gap = _fgp_m.hypot(_gx_ref - _fgp_post[0], _gy_ref - _fgp_post[1])
                            _drone2gap = _fgp_m.hypot(px - _gx_ref, py - _gy_ref)
                            # 若 drone 在 post 侧且 gap 在 drone 和 pre 上方，认为已穿越
                            # 条件：gap_post 距 gap_center ~ 1.0m；drone 也在 gap_center 约 1.0m 处
                            # 且 drone 已经飞入 post 区域（不是停在 pre 侧）
                            if _drone2gap > 0.4 and _post2gap > 0.3:
                                # 进一步：gap_post 与 drone 是否在 gap_center 同侧（即都在出口侧）
                                # 方法：比较 (drone-gap) 和 (post-gap) 点积
                                _dot = _dx_gap * (_fgp_post[0] - _gx_ref) + _dy_gap * (_fgp_post[1] - _gy_ref)
                                if _dot > 0.0:
                                    # 同向 → drone 和 post 在同一侧（出口侧）→ 已穿越
                                    _actually_crossed = True
                                    _gap_ref_pos = (_gx_ref, _gy_ref)

                        _bl_key_fgp = f"{_fgp_post[0]:.1f}_{_fgp_post[1]:.1f}"
                        if not hasattr(self, '_target_blacklist'):
                            self._target_blacklist = {}

                        if _actually_crossed:
                            # 真实穿门（直接导航方式）→ DOOR-CROSSED 处理
                            self._last_crossed_gap_pos = _gap_ref_pos
                            self._last_crossed_gap_at  = _cross_t.monotonic()
                            self._door_phase_commit = None
                            self._passage_commit_active = False
                            self._near_gap_fail_count = 0
                            # 短暂黑名单（15s）防止立即重选同一 post 点
                            self._target_blacklist[_bl_key_fgp] = _cross_t.monotonic() + 15.0
                            self.get_logger().warn(
                                f"[DOOR-CROSSED-DIRECT] ★ 直接导航穿门成功 "
                                f"gap=({_gap_ref_pos[0]:.2f},{_gap_ref_pos[1]:.2f}) "
                                f"post=({_fgp_post[0]:.2f},{_fgp_post[1]:.2f}) "
                                f"→ DOOR-CROSSED + 15s短暂黑名单"
                            )
                            self._log_event("DOOR-CROSSED-DIRECT",
                                f"gap=({_gap_ref_pos[0]:.2f},{_gap_ref_pos[1]:.2f}) "
                                f"post=({_fgp_post[0]:.2f},{_fgp_post[1]:.2f}) blacklist=15s → POST_DOOR_COMMIT")
                            # ★[A] DOOR-CROSSED-DIRECT→POST_DOOR_COMMIT保护
                            import time as _pdc2_t
                            self._post_door_commit_start_t = _pdc2_t.monotonic()
                            self._post_door_commit_start_pos = (
                                float(self.pose.x) if self.pose is not None else 0.0,
                                float(self.pose.y) if self.pose is not None else 0.0,
                            )
                            self._hover_scan_escape_fail_count = 0
                            # ★[F] 直接穿门：标记门口区域空间冷却防止重选
                            self._spatial_cooldown_stamp(self._plan_px, self._plan_py)
                            self.explore_state = "POST_DOOR_COMMIT"
                        else:
                            # 未真实穿越（可能是假门/同侧滑动）→ 90s 黑名单
                            self._target_blacklist[_bl_key_fgp] = _cross_t.monotonic() + 90.0
                            self._door_phase_commit = None
                            self.get_logger().warn(
                                f"[GAP-POST-NOGATE] frontier_gap_post直达且未确认穿越 "
                                f"→ 90s黑名单 key={_bl_key_fgp} "
                                f"nearest_gap={_nearest_gap_d:.2f}m"
                            )
                            self._log_event("GAP-POST-NOGATE",
                                f"pos=({_fgp_post[0]:.2f},{_fgp_post[1]:.2f}) "
                                f"blacklisted=90s nearest_gap_d={_nearest_gap_d:.2f}m")
                # ─────────────────────────────────────────────────────────────

                self.get_logger().info(
                    f"[GOAL-REPLACE] old=({self.ssg_viewpoint_goal[0]:.2f},{self.ssg_viewpoint_goal[1]:.2f}) "
                    f"new=None reason=reach_reached type={_gtype} dist={dist:.3f}m < {_reach_dist:.2f}m "
                    f"dwell={_ARRIVE_DWELL_T}s  near={_near_now if _near_now else 'n/a'} "
                    f"→ HOVER_SCAN"
                )
                self._log_event(
                    "GOAL-REPLACE",
                    f"reason=reach_reached type={_gtype} dist={dist:.3f}m < {_reach_dist:.2f}m "
                    f"dwell={_ARRIVE_DWELL_T}s "
                    f"old=({self.ssg_viewpoint_goal[0]:.2f},{self.ssg_viewpoint_goal[1]:.2f}) → HOVER_SCAN"
                )
                # ★[PDC-PUSH] 穿门类型目标到达：在清除 ssg_goal 前保存推进方向
                # 方向 = gap_center → gap_post（继续穿越方向），POST_DOOR_COMMIT 用此方向主动推进
                # 若无 _last_crossed_gap_pos（非 DOOR-PHASE 路径），退后用 drone→goal 方向
                _pdc_push_save_types = ('frontier_gap_post', 'frontier_gap_center',
                                        'frontier_gap_probe', 'confirmed_door')
                if _gtype in _pdc_push_save_types and self.ssg_viewpoint_goal is not None:
                    import math as _pdc_sp_m
                    _pdc_sp_sav  = self.ssg_viewpoint_goal
                    _pdc_sp_lcgp = getattr(self, '_last_crossed_gap_pos', None)
                    _pdc_sp_dir  = None
                    if _pdc_sp_lcgp is not None:
                        _pdc_sdx = float(_pdc_sp_sav[0]) - float(_pdc_sp_lcgp[0])
                        _pdc_sdy = float(_pdc_sp_sav[1]) - float(_pdc_sp_lcgp[1])
                        _pdc_sd  = _pdc_sp_m.hypot(_pdc_sdx, _pdc_sdy)
                        if _pdc_sd > 0.05:
                            _pdc_sp_dir = (_pdc_sdx / _pdc_sd, _pdc_sdy / _pdc_sd)
                    if _pdc_sp_dir is None and self.pose is not None:
                        _pdc_sdx2 = float(_pdc_sp_sav[0]) - float(self.pose.x)
                        _pdc_sdy2 = float(_pdc_sp_sav[1]) - float(self.pose.y)
                        _pdc_sd2  = _pdc_sp_m.hypot(_pdc_sdx2, _pdc_sdy2)
                        if _pdc_sd2 > 0.05:
                            _pdc_sp_dir = (_pdc_sdx2 / _pdc_sd2, _pdc_sdy2 / _pdc_sd2)
                    self._post_door_commit_push_dir = _pdc_sp_dir
                    if _pdc_sp_dir is not None:
                        self._log_event('PDC-PUSH-SAVED',
                            f'type={_gtype} push_dir=({_pdc_sp_dir[0]:.2f},{_pdc_sp_dir[1]:.2f})')
                self.ssg_viewpoint_goal = None
                self.ssg_goal_type = "none"
                self.astar_path = None
                # ★[A] 门类型目标到达（gap_post/gap_center/confirmed_door）→ POST_DOOR_COMMIT
                # 非门类型目标到达 → 普通 HOVER_SCAN
                _door_arrive_types = ('frontier_gap_post', 'frontier_gap_center',
                                      'frontier_gap_probe', 'confirmed_door')
                if _gtype in _door_arrive_types:
                    import time as _pdc3_t
                    self._post_door_commit_start_t = _pdc3_t.monotonic()
                    self._post_door_commit_start_pos = (
                        float(self.pose.x) if self.pose is not None else 0.0,
                        float(self.pose.y) if self.pose is not None else 0.0,
                    )
                    self._hover_scan_escape_fail_count = 0
                    self.explore_state = "POST_DOOR_COMMIT"
                    # ★[F] POST_DOOR_COMMIT 进入时标记当前门口位置空间冷却
                    # 防止系统在 commit 窗口期内重新选取同一区域的frontier目标
                    self._spatial_cooldown_stamp(self._plan_px, self._plan_py)
                    self.get_logger().warn(
                        f"[POST-DOOR-COMMIT-ENTER] type={_gtype} → 进入穿门保护"
                    )
                    self._log_event("POST-DOOR-COMMIT-ENTER",
                        f"type={_gtype} pos=({self._post_door_commit_start_pos[0]:.2f},"
                        f"{self._post_door_commit_start_pos[1]:.2f})")
                else:
                    # ★[FIX2] POST_DOOR_COMMIT 保护：若当前在 PDC 窗口内，禁止 reach_reached 将
                    # explore_state 切换为 HOVER_SCAN——保持 POST_DOOR_COMMIT 继续惯性推进，
                    # 由 _post_door_commit_cmd 自行决定何时退出。
                    if getattr(self, 'explore_state', '') == 'POST_DOOR_COMMIT':
                        pass  # 保持 POST_DOOR_COMMIT，目标已清空，由 PDC 推进逻辑继续
                    else:
                        self.explore_state = "HOVER_SCAN"
                # ★[D1] 到达viewpoint：在空间冷却网格中标记当前位置（90s内不重选）
                self._spatial_cooldown_stamp(self._plan_px, self._plan_py)
                self.hover_scan_start_time = None
                return np.zeros(3, dtype=float)
        else:
            # 距离不满足或近墙：重置驻留计时器
            if getattr(self, '_arrive_dwell_t', None) is not None:
                self._arrive_dwell_t = None
            if dist < _reach_dist and not _clr_ok:
                import time as _adw_t
                if _adw_t.monotonic() - getattr(self, '_arrive_clr_block_log_t', 0.0) > 2.0:
                    self._arrive_clr_block_log_t = _adw_t.monotonic()
                    self.get_logger().warn(
                        f"[ARRIVE-CLR-BLOCK] dist={dist:.2f}m<{_reach_dist:.2f}m satisfied but "
                        f"near={_near_now:.2f}m<{_ARRIVE_CLR_GUARD:.2f}m guard: 禁止近墙误判到达 "
                        f"type={_gtype}"
                    )

        # 卡住检测
        if self._ssg_stuck_pos is not None:
            moved = float(np.linalg.norm(pos_now - self._ssg_stuck_pos))
            if moved > 0.20:
                self._ssg_stuck_pos = pos_now.copy()
                self._ssg_stuck_since = now_sec
            elif self._ssg_stuck_since is not None and (now_sec - self._ssg_stuck_since) > self.ssg_stuck_timeout:
                _old_goal = self.ssg_viewpoint_goal
                _is_door_stuck = _gtype in (
                    'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                    'frontier_gap_post', 'confirmed_door')
                if _is_door_stuck:
                    # ★ 门口目标：卡死后不立刻放弃，先重试最多3次（重规划A*），再记失败
                    # 每次重试：清除旧路径、重置卡死计时器、继续导航
                    import time as _dst
                    _door_fails = getattr(self, '_door_fail_counts', {})
                    if not hasattr(self, '_door_fail_counts'):
                        self._door_fail_counts = {}
                        _door_fails = self._door_fail_counts
                    _gsig = f"{_old_goal[0]:.1f}_{_old_goal[1]:.1f}"
                    _fail_cnt = _door_fails.get(_gsig, 0) + 1
                    _door_fails[_gsig] = _fail_cnt
                    self._door_fail_counts = _door_fails
                    if _fail_cnt <= 3:
                        self.get_logger().warn(
                            f"[DOOR-RETRY] goal=({_old_goal[0]:.2f},{_old_goal[1]:.2f}) "
                            f"type={_gtype} attempt={_fail_cnt}/3 stuck={now_sec - self._ssg_stuck_since:.0f}s "
                            f"→ 重规划A*重试"
                        )
                        self._log_event(
                            "DOOR-RETRY",
                            f"goal=({_old_goal[0]:.2f},{_old_goal[1]:.2f}) type={_gtype} "
                            f"attempt={_fail_cnt}/3 stuck={now_sec - self._ssg_stuck_since:.0f}s"
                        )
                        # 重置卡死计时器 + 清除旧路径（强制重规划）
                        self._ssg_stuck_since = now_sec
                        self._ssg_stuck_pos = pos_now.copy()
                        self.astar_path = None
                        self.astar_path_idx = 0
                        self.astar_last_plan_time = 0.0
                        return np.zeros(3, dtype=float)
                    else:
                        # 超过3次重试失败：放弃该门口目标
                        self.get_logger().warn(
                            f"[DOOR-GIVE-UP] goal=({_old_goal[0]:.2f},{_old_goal[1]:.2f}) "
                            f"type={_gtype} failed {_fail_cnt} times → HOVER_SCAN"
                        )
                        self._log_event(
                            "DOOR-GIVE-UP",
                            f"goal=({_old_goal[0]:.2f},{_old_goal[1]:.2f}) type={_gtype} "
                            f"attempts={_fail_cnt} → HOVER_SCAN"
                        )
                        # ★[D-ARCH] 门口多次失败时标记空间冷却，防止立即重返该区域
                        self._spatial_cooldown_stamp(float(_old_goal[0]), float(_old_goal[1]))
                        self.get_logger().warn(
                            f"[SPATIAL-COOLDOWN] door_give_up stamp "
                            f"goal=({_old_goal[0]:.2f},{_old_goal[1]:.2f}) type={_gtype}"
                        )
                        self.ssg_viewpoint_goal = None
                        self.ssg_goal_type = "none"
                        self.astar_path = None
                        self.explore_state = "HOVER_SCAN"
                        self.hover_scan_start_time = None
                        # ★[J3] 重置卡死追踪变量（与非门口处理★[J2]一致）：
                        # DOOR-GIVE-UP 后若不重置，L1 近墙防护立刻重选门目标 →
                        # 下一帧卡死计时器仍满足 → 再次触发 → 0.8s 内 cascade 4→13 次飞速循环
                        self._ssg_stuck_since = now_sec
                        self._ssg_stuck_pos = pos_now.copy()
                        return np.zeros(3, dtype=float)
                else:
                    # ★ 非门口目标卡死：加入 90s 黑名单 + 区域60s降权
                    import time as _bl_t
                    _bl_key = f"{_old_goal[0]:.1f}_{_old_goal[1]:.1f}"
                    _bl_expiry = _bl_t.monotonic() + 90.0
                    if not hasattr(self, '_target_blacklist'):
                        self._target_blacklist = {}
                    self._target_blacklist[_bl_key] = _bl_expiry
                    # ★ 区域降权：1.5m 半径局域内 coverage 候选同时冷却 60s
                    if not hasattr(self, '_region_penalty'):
                        self._region_penalty = {}
                    _rp_ri = int(round(_old_goal[0] / 1.5))
                    _rp_rj = int(round(_old_goal[1] / 1.5))
                    _rp_exp = _bl_t.monotonic() + 60.0
                    for _rdi in range(-1, 2):
                        for _rdj in range(-1, 2):
                            self._region_penalty[(_rp_ri + _rdi, _rp_rj + _rdj)] = _rp_exp
                    # 清理过期黑名单条目（避免无限增长）
                    _bl_now = _bl_t.monotonic()
                    self._target_blacklist = {k: v for k, v in self._target_blacklist.items() if v > _bl_now}
                    # ★ PASSAGE_COMMIT：如果平常目标卡死时附近存在 frontier_gap，连续发生则强制提升过门优先级
                    import math as _pc_math
                    _gaps_here = [
                        d for d in getattr(self, 'frontier_gaps', [])
                        if _pc_math.hypot(d['x'] - _old_goal[0], d['y'] - _old_goal[1]) < 2.5
                    ]
                    if _gaps_here:
                        # 在门附近失败 → PASSAGE_COMMIT 计数
                        if not hasattr(self, '_near_gap_fail_count'):
                            self._near_gap_fail_count = 0
                            self._near_gap_fail_pos  = None
                        _ngfp = getattr(self, '_near_gap_fail_pos', None)
                        _same_area = (_ngfp is not None and
                                      _pc_math.hypot(_ngfp[0]-_old_goal[0], _ngfp[1]-_old_goal[1]) < 2.0)
                        if _same_area:
                            self._near_gap_fail_count += 1
                        else:
                            self._near_gap_fail_count = 1
                            self._near_gap_fail_pos = (_old_goal[0], _old_goal[1])
                        _pc_count = self._near_gap_fail_count
                        self.get_logger().warn(
                            f"[NEAR-GAP-FAIL] 门附近平常目标失败 count={_pc_count} "
                            f"gap={[(d['x'],d['y']) for d in _gaps_here[:2]]} "
                            f"pos=({_old_goal[0]:.2f},{_old_goal[1]:.2f})"
                        )
                        # 失败次数达戰30s内🕒：激活 PASSAGE_COMMIT 60秒，强制过门
                        if _pc_count >= 2:
                            self._passage_commit_active = True
                            self._passage_commit_until  = _bl_now + 60.0
                            self._near_gap_fail_count   = 0    # 重置计数
                            self.get_logger().warn(
                                f"[PASSAGE-COMMIT] ★★★ 强制过门模式已激活! "
                                f"count=2 gap=({_gaps_here[0]['x']:.2f},{_gaps_here[0]['y']:.2f}) "
                                f"TTL=60s → 评分将为 gap目标给予+6.0压倒性加成"
                            )
                            self._log_event(
                                "PASSAGE-COMMIT",
                                f"activated count=2 gap=({_gaps_here[0]['x']:.2f},{_gaps_here[0]['y']:.2f}) TTL=60s"
                            )
                    self.get_logger().warn(
                        f"[GOAL-REPLACE] old=({_old_goal[0]:.2f},{_old_goal[1]:.2f}) "
                        f"new=None reason=stuck_timeout type={_gtype} "
                        f"stuck={now_sec - self._ssg_stuck_since:.0f}s → HOVER_SCAN "
                        f"blacklist_90s=True key={_bl_key}"
                    )
                    self._log_event(
                        "GOAL-REPLACE",
                        f"reason=stuck_timeout type={_gtype} "
                        f"stuck={now_sec - self._ssg_stuck_since:.0f}s "
                        f"blacklisted=90s old=({_old_goal[0]:.2f},{_old_goal[1]:.2f}) → HOVER_SCAN"
                    )
                    self.ssg_viewpoint_goal = None
                    self.ssg_goal_type = "none"
                    self.astar_path = None
                    self.explore_state = "HOVER_SCAN"
                    self.hover_scan_start_time = None
                    # ★[J2] 重置卡死追踪变量：防止下一帧选新目标后立刻再次触发stuck_timeout风暴
                    # 不重置时：explore_state→HOVER_SCAN→立刻选新目标→NAV_TO_VIEWPOINT
                    # →_ssg_stuck_since仍是18s前→moved<0.20m→再次触发→30+次/秒循环
                    self._ssg_stuck_since = now_sec
                    self._ssg_stuck_pos = pos_now.copy()
                    return np.zeros(3, dtype=float)
        else:
            self._ssg_stuck_pos = pos_now.copy()
            self._ssg_stuck_since = now_sec

        # A* 路径规划（周期性重规划）
        # ★ 门口目标使用专用膨胀半径（第一次规划就用小膨胀，不等失败后才放宽）
        _is_door_goal = _gtype in ('frontier_gap_pre', 'frontier_gap_center', 'frontier_gap_post',
                                    'frontier_gap_probe', 'confirmed_door')
        _astar_infl_eff = max(self.map_res, self.astar_inflation_radius * 0.35) if _is_door_goal else self.astar_inflation_radius
        need_replan = (
            self.astar_path is None
            or self.astar_path_idx >= len(self.astar_path)
            or (now_sec - self.astar_last_plan_time) > self.astar_replan_interval
        )
        if need_replan:
            _orig_infl = self.astar_inflation_radius
            _orig_unk = getattr(self, 'astar_treat_unknown_as_occ', True)
            self.astar_inflation_radius = _astar_infl_eff
            # ★ 门口目标：关闭 unknown_as_occ（允许穿越未扫描区，门洞常常被标为 unknown）
            if _is_door_goal:
                self.astar_treat_unknown_as_occ = False
            new_path = self.astar_plan(np.array([px, py]), self.ssg_viewpoint_goal)
            self.astar_inflation_radius = _orig_infl
            self.astar_treat_unknown_as_occ = _orig_unk
            if new_path is not None and len(new_path) > 0:
                self.astar_path = new_path
                self.astar_path_idx = 0
                self.astar_last_plan_time = now_sec
                # ★[I3] A*终点同步：若路径终点因 snap 偏离原始目标 > 0.20m，
                # 用实际可达的路径终点更新 ssg_viewpoint_goal，
                # 避免到达检测和 PROG-WD 用墙体坐标导致无人机永远无法到达
                _i3_eff = np.array(new_path[-1], dtype=float)
                _i3_snap = float(np.linalg.norm(_i3_eff - self.ssg_viewpoint_goal))
                if _i3_snap > 0.20:
                    import time as _i3_t
                    if _i3_t.monotonic() - getattr(self, '_i3_log_t', 0.0) > 2.0:
                        self._i3_log_t = _i3_t.monotonic()
                        self.get_logger().info(
                            f"[I3-GOAL-SNAP] A*终点偏移 snap={_i3_snap:.2f}m "
                            f"orig=({self.ssg_viewpoint_goal[0]:.2f},{self.ssg_viewpoint_goal[1]:.2f}) "
                            f"→ eff=({_i3_eff[0]:.2f},{_i3_eff[1]:.2f}) 已同步ssg_viewpoint_goal"
                        )
                    self.ssg_viewpoint_goal = _i3_eff
            else:
                # ★ A*失败降级策略：先放宽膨胀半径重试，再失败才 fallback 直线/wall_follow
                # 原因：默认astar_inflation_radius=0.3m对狭窄走廊过于保守，
                #         过道富裕时路径被除掉 → 第二次用 40%膨胀 再试
                _orig_infl2 = self.astar_inflation_radius
                _orig_unk2 = getattr(self, 'astar_treat_unknown_as_occ', True)
                self.astar_inflation_radius = max(self.map_res, _orig_infl2 * 0.55)   # ★[WALL-FIX] 0.4→0.55，放宽但不过于激进
                # 宽松重试时也允许 unknown 通行
                if _is_door_goal:
                    self.astar_treat_unknown_as_occ = False
                new_path = self.astar_plan(np.array([px, py]), self.ssg_viewpoint_goal)
                self.astar_inflation_radius = _orig_infl2
                self.astar_treat_unknown_as_occ = _orig_unk2
                if new_path is not None and len(new_path) > 0:
                    self.get_logger().info(
                        f"[NAV_VP] A*放宽膨胀重试成功（infl={self.astar_inflation_radius*0.55:.3f}m）")
                    self.astar_path = new_path
                    self.astar_path_idx = 0
                    self.astar_last_plan_time = now_sec
                    # ★[I3] 宽松重试同样同步路径终点
                    _i3_eff2 = np.array(new_path[-1], dtype=float)
                    _i3_snap2 = float(np.linalg.norm(_i3_eff2 - self.ssg_viewpoint_goal))
                    if _i3_snap2 > 0.20:
                        self.ssg_viewpoint_goal = _i3_eff2
                else:
                    # 放宽后仍失败：门目标 → 原地等待重新规划（禁止 wall_follow 沿墙蹭）
                    # ★ Phase12: wall_follow = 沿墙切向前进，门口反而会更卡
                    if _is_door_goal:
                        import time as _astar_door_t
                        if _astar_door_t.monotonic() - getattr(self, '_astar_door_fail_log_t', 0.0) > 2.0:
                            self._astar_door_fail_log_t = _astar_door_t.monotonic()
                            self.get_logger().warn(
                                f"[ASTAR-DOOR-WAIT] A*失败但门目标禁止wall_follow: type={_gtype} "
                                f"goal=({self.ssg_viewpoint_goal[0]:.2f},{self.ssg_viewpoint_goal[1]:.2f}) "
                                f"→ 原地等待重新规划"
                            )
                        return np.zeros(3, dtype=float)
                    # 普通目标：直线尝试，若又失败则 wall_follow
                    self.get_logger().warn(
                        f"[GOAL-REPLACE] old=({self.ssg_viewpoint_goal[0]:.2f},{self.ssg_viewpoint_goal[1]:.2f}) "
                        f"new=direct_line reason=astar_failed type={_gtype} infl={_astar_infl_eff:.3f}m")
                    vel = self.nav_to_goal_cmd(self.ssg_viewpoint_goal, self.ssg_nav_speed, now)
                    if vel is None:
                        # ★ A8: A*失败+直线失败 → 原地停止等待重规划（禁用wall_follow，防止贴墙滑行）
                        import time as _a8t
                        if _a8t.monotonic() - getattr(self, '_a8_stop_log_t', 0.0) > 2.0:
                            self._a8_stop_log_t = _a8t.monotonic()
                            self.get_logger().warn(
                                f"[ASTAR-STOP-WAIT] A*+直线均失败 type={_gtype} "
                                f"→ 原地停止等待重规划 (wall_follow已禁用,防止贴墙滑行)"
                            )
                        return np.zeros(3, dtype=float)
                    return np.array([vel[0], vel[1], 0.0], dtype=float)

        # ★ door_transit_mode 几何检测：不只靠 goal_type，路径上事实在穿门时也启用
        # 条件：与任意 frontier_gap 距离 < 1.0m，或当前 goal_type 是门类型，或有门阶段锁定
        import math as _dtm_m
        _dtm_near_gap_d = min(
            (_dtm_m.hypot(float(g['x']) - px, float(g['y']) - py)
             for g in getattr(self, 'frontier_gaps', [])),
            default=99.0
        )
        _door_transit_mode = (
            _is_door_goal
            or _dtm_near_gap_d < 1.0
            or getattr(self, '_door_phase_commit', None) is not None
        )
        # 持久存储供 NEAR-WALL-HARDSTOP 使用（即使 goal_type 是 coverage 也能触发门穿越保护）
        self._door_transit_mode = _door_transit_mode
        if _door_transit_mode and not _is_door_goal:
            import time as _dtm_t
            if _dtm_t.monotonic() - getattr(self, '_dtm_log_t', 0.0) > 3.0:
                self._dtm_log_t = _dtm_t.monotonic()
                self.get_logger().info(
                    f"[DOOR-TRANSIT] 几何触发 dtm=True 近gap={_dtm_near_gap_d:.2f}m "
                    f"goal_type={_gtype} phase_locked={'yes' if getattr(self,'_door_phase_commit',None) else 'no'}"
                )

        # ★[A] Carrot-lookahead 路径跟随（替换原"0.5m 跳点直线追踪"）
        # 每帧找路径上"LOS 可见 + 距离 ≤ lookahead_m"的最远点作为 carrot
        if self.astar_path and self.astar_path_idx < len(self.astar_path):
            # 构建膨胀障碍（供 carrot LOS 检测 + nav_to_goal_cmd 共用）
            # ★[J1] carrot LOS 用 50% 膨胀（A*规划用100%膨胀保守安全，
            #   LOS 检测若也用100%，走廊内膨胀区将覆盖可行路径→CARROT-FAIL→原地不动）
            # ★[K2] 门目标 carrot LOS 进一步降至 30%（与 A* 门规划 35% 匹配），
            #   防止 carrot 看不穿自己规划出来的门洞路径（50% 仍会封堵窄门 LOS）
            from scipy.ndimage import binary_dilation as _cbd
            _cocc = self.grid >= self.prob_occ_th
            _is_door_goal_los = _gtype in (
                'frontier_gap_pre', 'frontier_gap_center', 'frontier_gap_post',
                'frontier_gap_probe', 'confirmed_door')
            _los_infl_pct = 0.30 if _is_door_goal_los else 0.50
            _cinfl_los = max(1, int(self.astar_inflation_radius * _los_infl_pct / self.map_res))  # ★[K2]
            _cinfl     = max(1, int(self.astar_inflation_radius / self.map_res))         # 100%（仍供其他用途）
            _cocc_infl = _cbd(_cocc, iterations=_cinfl_los)  # ★[J1/K2] LOS使用小膨胀

            # 先推进：跳过已经"到达"的路径点（< 0.3m）
            while self.astar_path_idx < len(self.astar_path) - 1:
                _wp0 = self.astar_path[self.astar_path_idx]
                _d0  = float(np.linalg.norm([px - float(_wp0[0]), py - float(_wp0[1])]))
                if _d0 < 0.30:             # ★ 到达阈值 0.3m（原 0.5m，收紧）
                    self.astar_path_idx += 1
                else:
                    break

            # 选 carrot 点
            _carrot_xy, _carrot_idx = self._pick_carrot_point(
                self.astar_path,
                self.astar_path_idx,
                self.carrot_lookahead_m,
                _cocc_infl,
            )
            import time as _car_t
            _car_now = _car_t.monotonic()
            if _carrot_xy is None:
                # ★[H4] LOS 失败时路径切线降级导航（根本修复：走廊内膨胀 LOS 永远失败根因）
                # 原因：无人机处于走廊（距墙 0.3m），膨胀占用栅格 LOS 失败 → CARROT-FAIL 每帧触发
                #       → micro_escape/hold_prev_vel ≈ 0 速度 → 原地不动
                # 方案：有有效路径点时，直接以路径方向半速行进（A*路径本身无障碍）
                _cf_near = self._nearest_lidar_distance()
                _astar_len_h4 = len(self.astar_path) if self.astar_path is not None else 0
                _cf_action = "replan+stop"
                _cf_ret = np.zeros(3, dtype=float)
                _h4_used = False
                if (self.astar_path is not None
                        and self.astar_path_idx < len(self.astar_path)
                        and _cf_near is not None and _cf_near > 0.20):
                    _wp_h4   = self.astar_path[self.astar_path_idx]
                    _dx_h4   = float(_wp_h4[0]) - px
                    _dy_h4   = float(_wp_h4[1]) - py
                    _d_h4    = float(np.sqrt(_dx_h4 ** 2 + _dy_h4 ** 2))
                    if _d_h4 > 0.10:
                        # 半速沿路径方向行进；apply_proximity_constraints 仍会做近墙保护
                        _spd_h4  = min(0.20, self.ssg_nav_speed * 0.5)
                        _cf_ret  = np.array([
                            _dx_h4 / _d_h4 * _spd_h4,
                            _dy_h4 / _d_h4 * _spd_h4,
                            0.0,
                        ], dtype=float)
                        _cf_action = f"path_tangent d={_d_h4:.2f}m near={_cf_near:.2f}m"
                        _h4_used = True
                    elif self.astar_path_idx < len(self.astar_path) - 1:
                        # ★[I5] 已到达当前路径点 → 循环推进索引直到找到足够远的点
                        # 原来只推进1步，如果连续多个点都在 0.10m 内（snap 贴墙时常见）
                        # 会卡在原地反复执行此分支而无法产生有效速度
                        _i5_advanced = 0
                        while self.astar_path_idx < len(self.astar_path) - 1:
                            self.astar_path_idx += 1
                            _i5_advanced += 1
                            _wp_next = self.astar_path[self.astar_path_idx]
                            _d_next = float(np.sqrt(
                                (float(_wp_next[0]) - px) ** 2 +
                                (float(_wp_next[1]) - py) ** 2
                            ))
                            if _d_next > 0.10:
                                break  # 找到足够远的点
                        _cf_action = f"advance_wp×{_i5_advanced} near={_cf_near:.2f}m"
                        _h4_used = True
                if not _h4_used:
                    # 原有 micro_escape / hold_prev_vel 保底逻辑（路径为空时仍需）
                    if _cf_near is not None and _cf_near < 0.6:
                        _esc_until = getattr(self, '_carrot_micro_esc_until', 0.0)
                        if _car_now < _esc_until and getattr(self, '_carrot_micro_esc_dir', None) is not None:
                            _ed = self._carrot_micro_esc_dir
                            _cf_ret = np.array([_ed[0] * 0.22, _ed[1] * 0.22, 0.0])
                            _cf_action = f"micro_escape near={_cf_near:.2f}m"
                        else:
                            _esc_dir = self._compute_escape_dir_from_lidar()
                            if _esc_dir is not None:
                                self._carrot_micro_esc_dir   = _esc_dir
                                self._carrot_micro_esc_until = _car_now + 2.0  # ★[G5]
                                _cf_ret = np.array([_esc_dir[0] * 0.22, _esc_dir[1] * 0.22, 0.0])
                                _cf_action = f"micro_escape_start near={_cf_near:.2f}m"
                            else:
                                self.astar_path = None
                                _cf_action = "replan(no_esc_dir)"
                    elif _cf_near is not None:
                        _hold_until = getattr(self, '_carrot_hold_until', 0.0)
                        if _car_now < _hold_until:
                            _cf_ret = np.array([self.prev_vel[0] * 0.4,
                                                self.prev_vel[1] * 0.4, 0.0])
                            _cf_action = "hold_prev_vel"
                        else:
                            self._carrot_hold_until = _car_now + 0.4
                            _cf_ret = np.array([self.prev_vel[0] * 0.4,
                                                self.prev_vel[1] * 0.4, 0.0])
                            self.astar_path = None
                            _cf_action = "hold+replan"
                    else:
                        self.astar_path = None
                if _car_now - getattr(self, '_carrot_fail_log_t', 0.0) > 1.5:
                    self._carrot_fail_log_t = _car_now
                    _astar_len = len(self.astar_path) if self.astar_path is not None else 0
                    self.get_logger().warn(
                        f"[CARROT-FAIL] no_visible_point idx={self.astar_path_idx}/"
                        f"{_astar_len} near={_cf_near} action={_cf_action}"
                    )
                    if "replan" in _cf_action:
                        self._log_event("REPLAN", f"reason=carrot_fail action={_cf_action}")
                return _cf_ret

            # 更新全局 carrot 索引（供可视化使用）
            self._carrot_idx = _carrot_idx
            if _car_now - getattr(self, '_carrot_log_t', 0.0) > 1.0:
                self._carrot_log_t = _car_now
                self.get_logger().debug(
                    f"[CARROT] idx_start={self.astar_path_idx} idx_carrot={_carrot_idx} "
                    f"lookahead={self.carrot_lookahead_m:.1f}m "
                    f"los_ok=True carrot=({_carrot_xy[0]:.2f},{_carrot_xy[1]:.2f})"
                )

            vel = self.nav_to_goal_cmd(_carrot_xy, self.ssg_nav_speed, now)
            if vel is None:
                # ★[C] carrot 可见但 nav_check 仍被阻（罕见）→ 不 wall_follow，重规划
                if _is_door_goal:
                    if _car_now - getattr(self, '_car_door_blk_log_t', 0.0) > 2.0:
                        self._car_door_blk_log_t = _car_now
                        self.get_logger().warn(
                            f"[REPLAN] reason=nav_blocked door_goal={_gtype} → hold"
                        )
                    return np.zeros(3, dtype=float)
                self.astar_path = None
                if _car_now - getattr(self, '_a8b_stop_log_t', 0.0) > 2.0:
                    self._a8b_stop_log_t = _car_now
                    self.get_logger().warn(
                        f"[REPLAN] reason=nav_blocked type={_gtype} "
                        f"→ 清除路径+停止等待重规划 (wall_follow已禁用)"
                    )
                return np.zeros(3, dtype=float)

            # ★ 路径主导速度投影（PATH-DOMINANT）：
            # 当近墙时，削除速度向量中朝向障碍物的法向分量，仅保留沿路径切线的安全分量
            _near_for_proj = self._nearest_lidar_distance()
            _PATH_PROJ_TH = 0.55
            if (_near_for_proj is not None and _near_for_proj < _PATH_PROJ_TH
                    and self.att_yaw is not None):
                _yaw_p = float(self.att_yaw)
                _sec_dirs = {
                    'front': np.array([ np.cos(_yaw_p),  np.sin(_yaw_p)]),
                    'back':  np.array([-np.cos(_yaw_p), -np.sin(_yaw_p)]),
                    'left':  np.array([-np.sin(_yaw_p),  np.cos(_yaw_p)]),
                    'right': np.array([ np.sin(_yaw_p), -np.cos(_yaw_p)]),
                }
                _nearest_obs_dir = None
                _nearest_obs_d   = 9.0
                for _sk_p, _sv_p in _sec_dirs.items():
                    _sd_p = self.lidar_ranges.get(_sk_p)
                    if _sd_p is not None and np.isfinite(_sd_p) and _sd_p < _nearest_obs_d:
                        _nearest_obs_d = _sd_p
                        _nearest_obs_dir = _sv_p
                if _nearest_obs_dir is not None:
                    _v2d = vel[:2]
                    _v_toward_obs = float(np.dot(_v2d, _nearest_obs_dir))
                    if _v_toward_obs > 0.0:
                        _v2d_proj = _v2d - _v_toward_obs * _nearest_obs_dir
                        _mag_orig = float(np.linalg.norm(_v2d))
                        _mag_proj = float(np.linalg.norm(_v2d_proj))
                        if _mag_proj > 0.01:
                            _v2d_proj = _v2d_proj * (_mag_orig / _mag_proj)
                        vel = _v2d_proj
                        if _near_for_proj < 0.30:
                            if _car_t.monotonic() - getattr(self, '_path_proj_log_t', 0.0) > 2.0:
                                self._path_proj_log_t = _car_t.monotonic()
                                self.get_logger().debug(
                                    f"[PATH-DOMINANT] near={_near_for_proj:.2f}m "
                                    f"削除朝障碍物分量 v_obs={_v_toward_obs:.2f} "
                                    f"|v|: {_mag_orig:.2f}→{_mag_proj:.2f}"
                                )
            return np.array([vel[0], vel[1], 0.0], dtype=float)

        self.astar_path = None
        return np.zeros(3, dtype=float)

    # ===================================================================
    # Stop-Scan-Go：候选扫描点生成 + 评分选择
    # ===================================================================

    def _belief_boost_at_pose(self, px: float, py: float) -> None:
        """在位置 (px,py) 处以高斯增强 belief_grid（正证据：检测到目标）。"""
        sigma_cells = max(1.0, self.belief_detect_sigma / self.map_res)
        bix, biy = self.world_to_grid(px, py)
        r_cells = int(sigma_cells * 3)
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                ni, nj = bix + di, biy + dj
                if 0 <= ni < self.grid_shape[0] and 0 <= nj < self.grid_shape[1]:
                    d2 = (di * di + dj * dj)
                    w = float(np.exp(-d2 / (2.0 * sigma_cells * sigma_cells)))
                    self.belief_grid[ni, nj] = float(np.clip(
                        self.belief_grid[ni, nj] + w * (self.belief_detect_boost - self.belief_grid[ni, nj]),
                        0.0, 1.0
                    ))

    def _ssg_compute_dist_transform(self):
        """计算障碍物距离变换（clearance map），用于候选点安全过滤和风险评分。"""
        occ_mask = self.grid >= self.prob_occ_th
        # distance_transform_edt 返回每格距离最近障碍物的距离（格数）
        dt = distance_transform_edt(~occ_mask) * self.map_res  # 转为米
        return dt

    def _ssg_generate_candidates(self, dt_map) -> list:
        """
        生成候选扫描点（混合来源：Frontier + 覆盖采样 + 门候选 + frontier_gap穿门探针）。
        返回 [(x, y, goal_type), ...] 世界坐标列表（3元组）。
        goal_type: 'frontier' | 'coverage' | 'confirmed_door' | 'frontier_gap_center' | 'frontier_gap_probe'
        """
        candidates = []
        if self.pose is None:
            return candidates
        px, py = self._plan_px, self._plan_py

        free_mask = self.grid <= self.prob_free_th
        unk_mask = (self.grid > self.prob_free_th) & (self.grid < self.prob_occ_th)
        occ_mask = self.grid >= self.prob_occ_th

        # ── A. Frontier 候选（free邻接unknown的格子）────────────────────
        # ★ Phase12: frontier 候选必须内缩 0.55m，不能直接用边界格
        # 逻辑：利用 dt_map 梯度方向（指向远离障碍物方向），从边界格向内陆偏移
        _FRONT_PULLBACK_M = 0.55         # 内缩距离(m)
        _FRONT_MIN_CLR    = 0.32         # 内缩后最小clearance(m)
        unk_dil = binary_dilation(unk_mask, iterations=1)
        frontier_mask = free_mask & unk_dil
        occ_dil = binary_dilation(occ_mask, iterations=max(1, int(self.ssg_r_safe / self.map_res)))
        safe_front = frontier_mask & (~occ_dil)
        front_cells = np.argwhere(safe_front)
        _n_front_raw = 0; _n_front_ok = 0; _n_front_rej = 0
        if front_cells.shape[0] > 0:
            n_front = min(self.ssg_n_candidates // 2, front_cells.shape[0])
            spacing_cells = max(1, int(0.7 / self.map_res))
            sampled = front_cells[::spacing_cells][:n_front]
            _n_front_raw = len(sampled)
            _pb_cells = int(round(_FRONT_PULLBACK_M / self.map_res))
            _sg0, _sg1 = self.grid_shape
            for cell in sampled:
                ix, iy = int(cell[0]), int(cell[1])
                # 计算 dt_map 局部梯度（有限差分）→ 指向内陆（clearance 增大方向）
                _ix0 = max(0, ix - 1); _ix1 = min(_sg0 - 1, ix + 1)
                _iy0 = max(0, iy - 1); _iy1 = min(_sg1 - 1, iy + 1)
                _gx = float(dt_map[_ix1, iy] - dt_map[_ix0, iy]) / ((_ix1 - _ix0) * self.map_res + 1e-9)
                _gy = float(dt_map[ix, _iy1] - dt_map[ix, _iy0]) / ((_iy1 - _iy0) * self.map_res + 1e-9)
                _gn = max(1e-6, float(np.hypot(_gx, _gy)))
                _gxn, _gyn = _gx / _gn, _gy / _gn
                # 尝试几个递减步长找到有效内缩点
                _placed = False
                for _frac in (1.0, 0.7, 0.4):
                    _nix = int(round(ix + _pb_cells * _frac * _gxn))
                    _niy = int(round(iy + _pb_cells * _frac * _gyn))
                    if not (0 <= _nix < _sg0 and 0 <= _niy < _sg1):
                        continue
                    if not free_mask[_nix, _niy]:
                        continue
                    if dt_map[_nix, _niy] < _FRONT_MIN_CLR:
                        continue
                    wx, wy = self.grid_to_world(_nix, _niy)
                    candidates.append((wx, wy, "frontier"))
                    _n_front_ok += 1
                    _placed = True
                    break
                if not _placed:
                    _n_front_rej += 1
        # 内缩统计日志（每5s节流）
        import time as _fc_t
        if _fc_t.monotonic() - getattr(self, '_front_pb_log_t', 0.0) > 5.0:
            self._front_pb_log_t = _fc_t.monotonic()
            if _n_front_raw > 0:
                self.get_logger().debug(
                    f'[FRONT-PULLBACK] raw={_n_front_raw} ok={_n_front_ok} '
                    f'rej={_n_front_rej} pullback={_FRONT_PULLBACK_M}m'
                )

        # ── B. 覆盖采样候选（free区域均匀网格）──────────────────────────
        spacing_m = 1.0  # 采样间隔
        spacing_c = max(1, int(spacing_m / self.map_res))
        ix_arr = np.arange(0, self.grid_shape[0], spacing_c)
        iy_arr = np.arange(0, self.grid_shape[1], spacing_c)
        # ★[K2] coverage 候选安全距离提升：ssg_r_safe(0.35m)→0.50m，
        # 与 _ssg_select_viewpoint 的 _ABS_MIN_CLR(0.45m)保持一致避免大量候选被后级过滤
        _COV_MIN_CLR = max(0.50, self.ssg_r_safe)
        for ix in ix_arr:
            for iy in iy_arr:
                if free_mask[ix, iy] and dt_map[ix, iy] >= _COV_MIN_CLR:
                    wx, wy = self.grid_to_world(ix, iy)
                    candidates.append((wx, wy, "coverage"))

        # ── C. 已确认门候选（三段式：接近 → 门中心 → 穿越出口）────────────────────
        # ★ 三段式目标：预对准点（drone侧1.2m） + 门中心（transit） + 出口（对侧1.0m）
        import math as _door_math_c
        for d in self.doors_info:
            wx, wy = float(d["x"]), float(d["y"])
            _ori_c = str(d.get("orientation", ""))
            if _ori_c == "horizontal":  # 水平墙(y=const) → 穿越方向y轴
                _nx, _ny = 0.0, (1.0 if py >= wy else -1.0)
            elif _ori_c == "vertical":  # 垂直墙(x=const) → 穿越方向x轴
                _nx, _ny = (1.0 if px >= wx else -1.0), 0.0
            else:
                # 未知方向：退回用无人机→门连线估计
                _ddx = px - wx; _ddy = py - wy
                _ddist = max(0.001, _door_math_c.hypot(_ddx, _ddy))
                _nx, _ny = _ddx / _ddist, _ddy / _ddist
            _cp_x = wx + _nx * 1.2; _cp_y = wy + _ny * 1.2
            _ce_x = wx - _nx * 1.0; _ce_y = wy - _ny * 1.0
            candidates.append((_cp_x, _cp_y, "confirmed_door"))   # 接近点
            candidates.append((wx, wy, "confirmed_door"))          # 门中心
            candidates.append((_ce_x, _ce_y, "confirmed_door"))    # 出口点
            self.get_logger().debug(
                f"[DOOR-GEOM] confirmed ori={_ori_c} ctr=({wx:.2f},{wy:.2f}) "
                f"pre=({_cp_x:.2f},{_cp_y:.2f}) post=({_ce_x:.2f},{_ce_y:.2f})"
            )

        # ── D. Frontier_gap 三段式穿门候选 ──────────────────────────────────────────
        # ★ 基于 orientation 决定穿越方向，不再靠"机体当前朝向/机体到门中心连线"
        import math as _gap_math_d
        _APPROACH_DIST = 1.2   # 接近点从门中心向drone侧退回距离(m)
        _EXIT_DIST     = 1.0   # 出口点从门中心向未知侧前进距离(m)
        for d in getattr(self, 'frontier_gaps', []):
            wx, wy = float(d["x"]), float(d["y"])
            _ori_g = str(d.get("orientation", ""))
            if _ori_g == "horizontal":  # 水平墙(y=const) → 穿越方向y轴
                _nx, _ny = 0.0, (1.0 if py >= wy else -1.0)
            elif _ori_g == "vertical":  # 垂直墙(x=const) → 穿越方向x轴
                _nx, _ny = (1.0 if px >= wx else -1.0), 0.0
            else:
                _ddx = px - wx; _ddy = py - wy
                _ddist = max(0.001, _gap_math_d.hypot(_ddx, _ddy))
                _nx, _ny = _ddx / _ddist, _ddy / _ddist
            _gp_x = wx + _nx * _APPROACH_DIST; _gp_y = wy + _ny * _APPROACH_DIST
            _ge_x = wx - _nx * _EXIT_DIST;     _ge_y = wy - _ny * _EXIT_DIST
            candidates.append((_gp_x, _gp_y,  "frontier_gap_pre"))    # 门前对中点
            candidates.append((wx,    wy,      "frontier_gap_center"))  # ★ Phase12: inner→center，语义更准确
            candidates.append((_ge_x, _ge_y,   "frontier_gap_post"))   # 穿门后恢复点
            self.get_logger().debug(
                f"[DOOR-GEOM] gap ori={_ori_g} ctr=({wx:.2f},{wy:.2f}) "
                f"pre=({_gp_x:.2f},{_gp_y:.2f}) post=({_ge_x:.2f},{_ge_y:.2f})"
            )

        return candidates

    def _ssg_info_gain(self, cx: int, cy: int) -> float:
        """
        计算从格子 (cx,cy) 扫描一圈的信息增益（附近unknown数×距离权重）。
        近似：在半径 R_lidar 内统计 unknown 格，按 exp(-d/σ) 衰减。
        """
        r_cells = int(self.ssg_lidar_range / self.map_res)
        sg = self.grid_shape
        x0 = max(0, cx - r_cells); x1 = min(sg[0], cx + r_cells + 1)
        y0 = max(0, cy - r_cells); y1 = min(sg[1], cy + r_cells + 1)
        sub = self.grid[x0:x1, y0:y1]
        unk = (sub > self.prob_free_th) & (sub < self.prob_occ_th)
        if not np.any(unk):
            return 0.0
        xi, yi = np.where(unk)
        dx = (xi + x0 - cx) * self.map_res
        dy = (yi + y0 - cy) * self.map_res
        dists = np.sqrt(dx * dx + dy * dy)
        weights = np.exp(-dists / max(0.1, self.ssg_ig_sigma))
        return float(np.sum(weights))

    def _ssg_target_gain(self, cx: int, cy: int) -> float:
        """从格子 (cx,cy) 可视半径内 belief_grid 的期望目标概率总和。"""
        r_cells = int(self.ssg_lidar_range / self.map_res)
        sg = self.grid_shape
        x0 = max(0, cx - r_cells); x1 = min(sg[0], cx + r_cells + 1)
        y0 = max(0, cy - r_cells); y1 = min(sg[1], cy + r_cells + 1)
        sub_belief = self.belief_grid[x0:x1, y0:y1]
        sub_grid = self.grid[x0:x1, y0:y1]
        free = sub_grid <= self.prob_free_th
        return float(np.sum(sub_belief[free]))

    def _ssg_revisit_penalty(self, wx: float, wy: float) -> float:
        """距历史扫描点越近惩罚越大（避免反复扫同一区域）。
        ★[D1] 同时叠加空间冷却TTL网格惩罚：90s内飞过的格受额外0.5惩罚。
        """
        import time as _rp_t
        _now_rp = _rp_t.monotonic()
        # ── 老版：基于历史扫描点 exp(-d) ──────────────────
        base_pen = 0.0
        if self.ssg_visited_scan_pts:
            pts = np.array(self.ssg_visited_scan_pts)
            dists = np.sqrt((pts[:, 0] - wx) ** 2 + (pts[:, 1] - wy) ** 2)
            k = min(5, len(dists))
            nearest = np.sort(dists)[:k]
            base_pen = float(np.sum(np.exp(-nearest / max(0.5, self.ssg_lidar_range * 0.3))))
        # ── ★[D1] TTL 网格惩罚 ─────────────────────────────
        _cell = float(getattr(self, 'spatial_cooldown_cell_m', 1.5))
        _pen  = float(getattr(self, 'spatial_cooldown_penalty', 0.50))
        _cooldown = getattr(self, '_spatial_cooldown', {})
        ix = int(wx / _cell)
        iy = int(wy / _cell)
        # 检查当前格及相邻8格（1格邻域）
        _grid_pen = 0.0
        for _di in (-1, 0, 1):
            for _dj in (-1, 0, 1):
                _key = (ix + _di, iy + _dj)
                _exp = _cooldown.get(_key, 0.0)
                if _exp > _now_rp:
                    _remain = _exp - _now_rp
                    _ttl = float(getattr(self, 'spatial_cooldown_ttl_sec', 90.0))
                    # 惩罚随剩余TTL线性衰减：刚标记=1.0, 即将过期趋0
                    _decay = _remain / max(_ttl, 1.0)
                    _grid_pen += _decay * _pen * (1.0 if (_di == 0 and _dj == 0) else 0.4)
        return base_pen + _grid_pen

    def _spatial_cooldown_stamp(self, wx: float, wy: float, ttl_override: float = None) -> None:
        """★[D1] 在空间冷却网格中标记位置 (wx, wy)，TTL = spatial_cooldown_ttl_sec。
        ttl_override: 若非None，使用此值替代默认 TTL（用于 Tier-3 双倍惩罚等场景）。
        同时清理已过期的条目，防止字典无限增长。
        """
        import time as _sc_t
        _now_sc = _sc_t.monotonic()
        _cell = float(getattr(self, 'spatial_cooldown_cell_m', 1.5))
        _ttl  = float(ttl_override if ttl_override is not None
                      else getattr(self, 'spatial_cooldown_ttl_sec', 90.0))
        if not hasattr(self, '_spatial_cooldown'):
            self._spatial_cooldown = {}
        ix = int(wx / _cell)
        iy = int(wy / _cell)
        self._spatial_cooldown[(ix, iy)] = _now_sc + _ttl
        # 清理过期条目（每次stamp顺带清理，O(n)但字典小）
        _expired = [k for k, v in self._spatial_cooldown.items() if v <= _now_sc]
        for k in _expired:
            del self._spatial_cooldown[k]
        # ★[I3] 空间冷却日志
        self.get_logger().info(
            f"[SPATIAL-COOLDOWN] stamp cell=({ix},{iy}) world=({wx:.2f},{wy:.2f}) "
            f"ttl={_ttl:.0f}s active_cells={len(self._spatial_cooldown)}"
        )

    def _find_frontier_escape_point(self) -> Optional[np.ndarray]:
        """Phase9: 基于 frontier 的逃逸目标点（无语义视点时的兜底），
        frontier = free 格旁边有 unknown 邻格；从中选 unknown 密度最高且
        距当前位置 > 0.5m 的格，返回世界坐标 (wx, wy)。
        ★[L3] 优先级提升：若存在已知 frontier_gaps，优先返回最近的 gap_pre 点，
        确保即使在兜底逃逸路径中也尽量推动无人机穿越门洞而非飞向地图边缘。
        """
        # ★[L3] 有 frontier_gap 时最优先返回 gap_pre 点（确保门穿越路径优先于随机逃逸）
        try:
            import math as _l3m
            _l3_gaps = list(getattr(self, 'frontier_gaps', []))
            if _l3_gaps and self.pose is not None:
                _l3_px = self._plan_px
                _l3_py = self._plan_py
                _l3_dt = None
                try:
                    _l3_dt = self._ssg_compute_dist_transform()
                except Exception:
                    pass
                # 对所有 gap 尝试找合法 gap_pre 点
                _sorted_gaps = sorted(_l3_gaps,
                    key=lambda g: _l3m.hypot(g['x'] - _l3_px, g['y'] - _l3_py))
                for _g in _sorted_gaps:
                    _gwx, _gwy = float(_g['x']), float(_g['y'])
                    _ori = str(_g.get('orientation', ''))
                    # ★[PDC-GUARD] 跳过60s内已穿越的gap（L3逃逸同样需防重选）
                    import time as _l3_guard_t
                    _l3_cross_gp = getattr(self, '_last_crossed_gap_pos', None)
                    if (_l3_cross_gp is not None
                            and _l3_guard_t.monotonic() - float(getattr(self, '_last_crossed_gap_at', 0.0)) < 60.0
                            and _l3m.hypot(_gwx - _l3_cross_gp[0], _gwy - _l3_cross_gp[1]) < 1.5):
                        continue  # 近期已穿越此gap，跳过L3重选，避免穿门后掉头回Room1
                    if _ori == 'horizontal':  # 水平墙(y=const) → 穿越方向y轴
                        _nx, _ny = 0.0, (1.0 if _l3_py >= _gwy else -1.0)
                    elif _ori == 'vertical':  # 垂直墙(x=const) → 穿越方向x轴
                        _nx, _ny = (1.0 if _l3_px >= _gwx else -1.0), 0.0
                    else:
                        _dd = max(0.001, _l3m.hypot(_l3_px - _gwx, _l3_py - _gwy))
                        _nx, _ny = (_l3_px - _gwx) / _dd, (_l3_py - _gwy) / _dd
                    _pre_x = _gwx + _nx * 1.2
                    _pre_y = _gwy + _ny * 1.2
                    # ★ 检查 gap_center 净空（防假门洞，用 dt 代替 plan_bounds）
                    _gcx_l3, _gcy_l3 = self.world_to_grid(_gwx, _gwy)
                    _gcx_l3 = int(np.clip(_gcx_l3, 0, self.grid_shape[0]-1))
                    _gcy_l3 = int(np.clip(_gcy_l3, 0, self.grid_shape[1]-1))
                    _gap_ctr_clr_l3 = (float(_l3_dt[_gcx_l3, _gcy_l3])
                                       if _l3_dt is not None else 1.0)
                    if _gap_ctr_clr_l3 < 0.33:   # 地图边界假门，跳过
                        continue
                    _pgx, _pgy = self.world_to_grid(_pre_x, _pre_y)
                    _pgx = int(np.clip(_pgx, 0, self.grid_shape[0]-1))
                    _pgy = int(np.clip(_pgy, 0, self.grid_shape[1]-1))
                    if self.grid[_pgx, _pgy] > self.prob_free_th:
                        continue
                    _pclr = float(_l3_dt[_pgx, _pgy]) if _l3_dt is not None else 1.0
                    if _pclr < 0.40:   # 逃逸模式门接近点净空要求（提升至0.40m防近墙）
                        continue
                    _dist_to_pre = _l3m.hypot(_pre_x - _l3_px, _pre_y - _l3_py)
                    if _dist_to_pre < 0.5:
                        continue
                    self.get_logger().info(
                        f'[L3-ESCAPE-GAP] \u9003\u9038\u70b9\u4f18\u5148\u9009 gap_pre=({_pre_x:.2f},{_pre_y:.2f}) '
                        f'gap=({_gwx:.2f},{_gwy:.2f}) ori={_ori} clr={_pclr:.2f}m dist={_dist_to_pre:.2f}m'
                    )
                    return np.array([_pre_x, _pre_y], dtype=float)
        except Exception as _l3e:
            self.get_logger().debug(f'[L3-ESCAPE-GAP] exception: {_l3e}')
        # 无可用 gap_pre → 回退到原有 frontier 逃逸逻辑
        try:
            from scipy.ndimage import uniform_filter as _uf
            if self.grid is None:
                return None
            ni, nj = self.grid.shape
            free_mask = (self.grid <= self.prob_free_th)
            unk_mask  = (self.grid > self.prob_free_th) & (self.grid < self.prob_occ_th)
            # frontier: free 格有 unknown 邻格（4-邻域）—— 仅用于判断是否有待探索区域
            frontier_mask = np.zeros((ni, nj), dtype=bool)
            for _di, _dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                _shifted = np.roll(unk_mask, (_di, _dj), axis=(0, 1))
                frontier_mask |= (free_mask & _shifted)
            if not np.any(frontier_mask):
                return None
            # ★[BORDER-FIX] 用 15×15 窗口计算 unknown 密度，让内陆格也能感知到附近未探索区
            _unk_density = _uf(unk_mask.astype(np.float32), size=15, mode='constant')
            # 无人机在栅格中的位置
            sx = float(self.bounds[0][0])
            sy = float(self.bounds[1][0])
            res = float(self.map_res)
            if self.pose is not None:
                # 使用 plan frame 坐标（_plan_px/_plan_py），与 self.grid 同坐标系
                _dx = int((float(getattr(self, '_plan_px', self.pose.x)) - sx) / res)
                _dy = int((float(getattr(self, '_plan_py', self.pose.y)) - sy) / res)
            else:
                _dx, _dy = ni // 2, nj // 2
            # ★[BORDER-FIX] 候选集改为全体满足净空的 free 格（非仅 frontier 边界格）
            # 防止逃逸点落在地图边界（frontier 格紧邻 unknown，获得虚假高净空）
            _all_fi, _all_fj = np.where(free_mask)
            if len(_all_fi) == 0:
                return None
            # ★ 障碍物距离变换（只含 occ，用于避墙）
            try:
                _esc_dt = self._ssg_compute_dist_transform()
            except Exception:
                _esc_dt = None
            # ★[BORDER-FIX] 边界距离变换：free 以外的格（occ + unknown）均视为障碍
            # 用于过滤紧邻地图边界的假净空点
            from scipy.ndimage import distance_transform_edt as _bdt_fn
            _border_dt = _bdt_fn(free_mask) * res   # 每格到最近非 free 格的距离
            # ★[K1] 主净空门槛（距已知障碍物）0.60m
            _ESC_MIN_CLR    = max(0.60, float(getattr(self, 'goal_min_clearance_normal', 0.5)))
            # ★[BORDER-FIX] 边界净空门槛（距任意非 free 格，包括 unknown）0.40m
            _ESC_BORDER_CLR = 0.40
            # 采样最多 400 点（全 free 格集合更大，步长相应调整）
            _step = max(1, len(_all_fi) // 400)
            _all_fi = _all_fi[::_step]; _all_fj = _all_fj[::_step]
            _best_score = -1e9
            _best_wp = None
            _esc_rej_clr = 0; _esc_rej_border = 0
            for ix, iy in zip(_all_fi.tolist(), _all_fj.tolist()):
                _d = float(np.hypot(ix - _dx, iy - _dy)) * res
                if _d < 0.5:
                    continue
                # ★ 硬过滤1：已知障碍物净空（防落在墙边）
                if _esc_dt is not None:
                    if 0 <= ix < _esc_dt.shape[0] and 0 <= iy < _esc_dt.shape[1]:
                        if _esc_dt[ix, iy] < _ESC_MIN_CLR:
                            _esc_rej_clr += 1
                            continue
                # ★[BORDER-FIX] 硬过滤2：边界净空（防落在地图边界/unknown 旁）
                if 0 <= ix < _border_dt.shape[0] and 0 <= iy < _border_dt.shape[1]:
                    if _border_dt[ix, iy] < _ESC_BORDER_CLR:
                        _esc_rej_border += 1
                        continue
                # 评分：unknown 密度 + 障碍净空加成（双重限幅防虚假高分）
                _raw_clr    = float(_esc_dt[ix, iy]) if _esc_dt is not None else 0.0
                _raw_border = float(_border_dt[ix, iy]) if _border_dt is not None else 0.0
                # 两个净空都限幅：取较小值参与加分，避免未知区域给出虚假超高净空
                _clr_bonus = min(_raw_clr, _ESC_MIN_CLR * 2.0, _raw_border * 2.0)
                _score = float(_unk_density[ix, iy]) - 0.08 * _d + 2.0 * _clr_bonus
                if _score > _best_score:
                    _best_score = _score
                    _wx = sx + (ix + 0.5) * res
                    _wy = sy + (iy + 0.5) * res
                    _best_wp = np.array([_wx, _wy], dtype=float)
            if _esc_rej_clr > 0 or _esc_rej_border > 0:
                self.get_logger().debug(
                    f'[GOAL-REJECT] reason=escape_filter rej_occ={_esc_rej_clr} '
                    f'rej_border={_esc_rej_border} '
                    f'min_clr={_ESC_MIN_CLR:.2f} border_clr={_ESC_BORDER_CLR:.2f}'
                )
            return _best_wp
        except Exception as _fe:
            self.get_logger().debug(f"[FRONTIER-ESC] exception: {_fe}")
            return None

    def _ssg_select_viewpoint(self) -> Optional[np.ndarray]:
        """
        生成候选扫描点 → 过滤 → 评分 → 返回最优目标点。
        评分：α·InfoGain + β·TargetGain - γ·Cost - δ·Risk - η·Revisit
        """
        if self.pose is None:
            return None
        px, py = self._plan_px, self._plan_py

        # ★[C] 弱确认冻结：目标弱确认后停止替换frontier目标，避免打断目标跟踪链
        if getattr(self, '_tgt_weak_confirmed', False):
            if not getattr(self, '_ssg_freeze_logged', False):
                self._ssg_freeze_logged = True
                self.get_logger().warn(
                    "[SSG-FREEZE] ★[C] 弱确认激活→冻结 ssg_select_viewpoint，"
                    "保持当前目标直至目标链接管"
                )
                self._log_event("SSG-FREEZE", "reason=target_weak_confirmed")
            return None   # 不替换目标，保持当前 ssg_viewpoint_goal 不变


        if getattr(self, '_virtual_door_cands', []):
            _vx, _vy = self._virtual_door_cands.pop(0)
            self.get_logger().info(
                f"[SSG-SELECT] STUCK恢复虚拟出口目标 ({_vx:.2f},{_vy:.2f}) "
                f"剩余={len(self._virtual_door_cands)}"
            )
            return np.array([_vx, _vy], dtype=float)

        # ★[A5] 目标锁定：进入门通行任务后，锁定时間内强制重选 frontier_gap（不选普通 coverage）
        import time as _gh_t, math as _gh_m
        _gh_now = _gh_t.monotonic()
        _goal_hold_active = _gh_now < getattr(self, '_goal_hold_until', 0.0)
        if _goal_hold_active:
            _hold_rem = getattr(self, '_goal_hold_until', 0.0) - _gh_now
            _force_gaps = list(getattr(self, 'frontier_gaps', []))
            if _force_gaps:
                _best_fg_lock = min(_force_gaps,
                                    key=lambda g: _gh_m.hypot(g['x']-px, g['y']-py))
                _ori_gl = str(_best_fg_lock.get("orientation", ""))
                if _ori_gl == "horizontal":  # 水平墙(y=const) → 穿越方向y轴
                    _fnx_l, _fny_l = 0.0, (1.0 if py >= _best_fg_lock['y'] else -1.0)
                elif _ori_gl == "vertical":  # 垂直墙(x=const) → 穿越方向x轴
                    _fnx_l, _fny_l = (1.0 if px >= _best_fg_lock['x'] else -1.0), 0.0
                else:
                    _dd_l = max(0.001, _gh_m.hypot(px-_best_fg_lock['x'], py-_best_fg_lock['y']))
                    _fnx_l, _fny_l = (px-_best_fg_lock['x'])/_dd_l, (py-_best_fg_lock['y'])/_dd_l
                _pre_xl = _best_fg_lock['x'] + _fnx_l * 1.2
                _pre_yl = _best_fg_lock['y'] + _fny_l * 1.2
                # ★ 检查 gap_center 净空（用 dt_map 代替 plan_bounds 验证是否真实门洞）
                try:
                    _gl_dt = self._ssg_compute_dist_transform()
                except Exception:
                    _gl_dt = None
                _gcx_gl, _gcy_gl = self.world_to_grid(
                    _best_fg_lock['x'], _best_fg_lock['y'])
                _gcx_gl = int(np.clip(_gcx_gl, 0, self.grid_shape[0]-1))
                _gcy_gl = int(np.clip(_gcy_gl, 0, self.grid_shape[1]-1))
                _gap_ctr_clr_gl = float(_gl_dt[_gcx_gl, _gcy_gl]) if _gl_dt is not None else 1.0
                if _gap_ctr_clr_gl < 0.33:
                    # 假门，解除锁定，不强制派送
                    self._goal_hold_until = 0.0
                    self.get_logger().info(
                        f"[GOAL-LOCK] gap_center净空不足({_gap_ctr_clr_gl:.2f}m<0.33m) "
                        f"gap=({_best_fg_lock['x']:.2f},{_best_fg_lock['y']:.2f}) → 解除锁定避免假门"
                    )
                else:
                    self.ssg_goal_type = "frontier_gap_pre"
                    self.get_logger().warn(
                        f"[GOAL-LOCK] 门锁定激活 remain={_hold_rem:.1f}s → 强制frontier_gap_pre "
                        f"=({_pre_xl:.2f},{_pre_yl:.2f}) gap=({_best_fg_lock['x']:.2f},{_best_fg_lock['y']:.2f})"
                    )
                    return np.array([_pre_xl, _pre_yl], dtype=float)
            else:
                # 无可用 gap → 解锁
                self._goal_hold_until = 0.0
                self.get_logger().info("[GOAL-LOCK] 锁定期内无可用gap，解除锁定")

        # 计算距离变换（clearance map）
        try:
            dt_map = self._ssg_compute_dist_transform()
        except Exception:
            dt_map = np.ones(self.grid_shape, dtype=float)

        # 生成候选点
        raw_candidates = self._ssg_generate_candidates(dt_map)
        if not raw_candidates:
            self.get_logger().warn("[SSG-SELECT] 无候选点，回退到附近free点")
            return self._ssg_fallback_viewpoint()

        # 两段式决策：最大belief是否超过确认阈值
        max_belief = float(np.max(self.belief_grid))
        target_mode = max_belief >= self.ssg_belief_confirm_th
        if target_mode:
            self.get_logger().info(f"[SSG-SELECT] TARGET_MODE: max_belief={max_belief:.2f} → 优先TargetGain")
            w_ig, w_tg = self.ssg_w_ig * 0.3, self.ssg_w_tg * 3.0
        else:
            w_ig, w_tg = self.ssg_w_ig, self.ssg_w_tg

        # ★ 当前房间优先：每次选点前更新 current_room_id，记入 visited_rooms
        import time as _rp_t
        _rp_now = _rp_t.monotonic()
        if _rp_now - getattr(self, '_room_id_update_t', 0.0) > 2.0:
            self._room_id_update_t = _rp_now
            _new_rid = self.get_current_room_id()
            if _new_rid >= 0 and _new_rid != self.current_room_id:
                if self.current_room_id >= 0:
                    self.visited_rooms.add(self.current_room_id)
                    self.get_logger().warn(
                        f"[ROOM-STATE] 进入新房间 room={_new_rid} 旧={self.current_room_id} "
                        f"→ 添加旧房间到 visited_rooms={sorted(self.visited_rooms)}"
                    )
                # ★[B] ROOM-SWITCH 日志
                _since_last_rsw = _rp_now - getattr(self, '_room_switch_last_t', 0.0)
                if _since_last_rsw >= self.room_relabel_hysteresis_sec:
                    self._room_switch_last_t = _rp_now
                    self.get_logger().warn(
                        f"[ROOM-SWITCH] old={self.current_room_id} new={_new_rid} "
                        f"visited={sorted(self.visited_rooms)}"
                    )
                    self._log_event("ROOM-SWITCH",
                        f"old={self.current_room_id} new={_new_rid} "
                        f"visited={sorted(self.visited_rooms)}")
                # ★[D2] ROOM-ENTERED：过门后进入新房间时记录
                _cross_at = float(getattr(self, '_last_crossed_gap_at', 0.0))
                _door_just_crossed = (_rp_now - _cross_at < 15.0)   # 15s 内有穿门记录
                if _door_just_crossed:
                    _gap_ref = getattr(self, '_last_crossed_gap_pos', None)
                    self.get_logger().warn(
                        f"[ROOM-ENTERED] room={_new_rid} via_door={_gap_ref} "
                        f"elapsed_since_cross={_rp_now - _cross_at:.1f}s"
                    )
                    self._log_event("ROOM-ENTERED",
                        f"room={_new_rid} via_door={_gap_ref} "
                        f"cross_elapsed={_rp_now - _cross_at:.1f}s")
                    # 开始稳定区计时
                    self._room_entered_id = int(_new_rid)
                    self._room_entered_t = _rp_now
                    self._room_stable_logged = False
                self.current_room_id = _new_rid
        # ★[D2] ROOM-STABLE-ZONE：进入房间后保持稳定超过阈值时记录（1次/房间）
        if (not self._room_stable_logged
                and self._room_entered_id == self.current_room_id
                and self._room_entered_id >= 0
                and (_rp_now - self._room_entered_t) >= self.room_stable_zone_sec):
            self._room_stable_logged = True
            _stable_dur = _rp_now - self._room_entered_t
            self.get_logger().warn(
                f"[ROOM-STABLE-ZONE] room={self._room_entered_id} "
                f"stable_dur={_stable_dur:.1f}s >= {self.room_stable_zone_sec:.1f}s"
            )
            self._log_event("ROOM-STABLE-ZONE",
                f"room={self._room_entered_id} stable_dur={_stable_dur:.1f}s")
        _room_cov = self.get_room_coverage(self.current_room_id) if self.current_room_id >= 0 else 0.0
        if _rp_now - getattr(self, '_room_state_log_t', 0.0) > 5.0:
            self._room_state_log_t = _rp_now
            self.get_logger().info(
                f"[ROOM-STATE] cur_room={self.current_room_id} "
                f"visited={sorted(self.visited_rooms)} coverage={_room_cov:.1%} "
                f"rooms_known={len(self.rooms_info)}"
            )

        # ★ PASSAGE_COMMIT：若当前处于强制过门模式且 TTL 未过期，对门候选给予压倒性加成
        # 目的：确保 frontier_gap_* 评分远高于所有 coverage/frontier，使其必然被选中
        import time as _pct
        _pc_now = _pct.monotonic()
        _pc_active = (
            getattr(self, '_passage_commit_active', False)
            and _pc_now < getattr(self, '_passage_commit_until', 0.0)
        )
        if _pc_active:
            _pc_remain = getattr(self, '_passage_commit_until', 0.0) - _pc_now
            if _pc_now - getattr(self, '_pc_log_t', 0.0) > 5.0:
                self._pc_log_t = _pc_now
                self.get_logger().warn(
                    f"[PASSAGE-COMMIT] 激活中 remain={_pc_remain:.0f}s "
                    f"gaps={len(getattr(self,'frontier_gaps',[]))} pose=({px:.2f},{py:.2f}) "
                    f"→ gap目标+6.0压倒加成将导过才抓住导航权"
                )
        else:
            # TTL 到期自动退出
            if getattr(self, '_passage_commit_active', False) and _pc_now >= getattr(self, '_passage_commit_until', 0.0):
                self._passage_commit_active = False
                self.get_logger().info("[PASSAGE-COMMIT] TTL 到期，退出强制过门模式")

        # ★ 探索扩散模式：若无人机近期扫描点聚集在小区域（_ssg_clustered=True），
        #   大幅增大 IG 权重、减小距离惩罚，强迫选择远处高信息增益目标（如门口另一侧）。
        #   这是防止无人机"困在一个房间"的最后防线。
        _clustered = getattr(self, '_ssg_clustered', False)
        if _clustered and not target_mode:
            w_ig = w_ig * 2.5          # 大幅偏好未知区域
            _ssg_w_cost_eff = self.ssg_w_cost * 0.25   # 大幅降低距离惩罚（允许飞远）
            self.get_logger().debug(
                "[SSG-SPREAD] 远距探索模式激活: w_ig×2.5, cost×0.25"
            )
        else:
            _ssg_w_cost_eff = self.ssg_w_cost

        best_score = -1e9
        best_pt = None
        best_log = {}
        best_goal_type = "none"

        # ★ 两阶段评分：先预评（无A*），只对 top-K 运行精确A*路径规划
        _ASTAR_TOP_K = 12   # 只对预评分前12名运行A*
        _min_ig_gate = 0.05 if not target_mode else 0.0
        _rej_no_ig = 0     # 被 IG 门控拒绝的候选点数
        # ★ 门目标类型集合（所有安全/过滤逻辑统一引用）
        _DOOR_TYPES = frozenset(('frontier_gap_pre', 'frontier_gap_center', 'frontier_gap_post',
                                  'frontier_gap_probe', 'confirmed_door'))
        # 门口目标使用较小的安全裕度（允许靠近门框），但仍 >= ABS_MIN_CLR
        _door_r_safe = max(self.map_res, self.ssg_r_safe * 0.5)

        # ★ 候选池两段式保底：先收集 所有 门/gap候选，再用常规候选填满剩余配额
        # 这样保证即使 frontier/coverage 数量多，门候选也不会被截断
        _door_raws    = [(wx, wy, gt) for wx, wy, gt in raw_candidates if gt in _DOOR_TYPES]
        _regular_raws = [(wx, wy, gt) for wx, wy, gt in raw_candidates if gt not in _DOOR_TYPES]
        _regular_quota = max(4, self.ssg_n_candidates - len(_door_raws))  # 普通候选最多取此数量
        _pass_two_batches = _door_raws + _regular_raws[:_regular_quota]   # 合并，门候选在前

        pre_scored = []
        _n_door_added = 0; _n_reg_added = 0
        _n_rej_clr = 0; _n_rej_blk = 0

        for wx, wy, _gtype in _pass_two_batches:
            cix, ciy = self.world_to_grid(wx, wy)
            if not (0 <= cix < self.grid_shape[0] and 0 <= ciy < self.grid_shape[1]):
                continue
            if self.grid[cix, ciy] > self.prob_free_th:
                continue
            # ★ 硬过滤1: 绝对最小安全距离
            # 门类型 0.35m（允许靠近门框但不能太近）；普通目标 0.45m（远离墙角）
            _ABS_MIN_CLR = 0.35 if _gtype in _DOOR_TYPES else 0.45   # ★[WALL-FIX] 0.30→0.35
            if dt_map[cix, ciy] < _ABS_MIN_CLR:
                _n_rej_clr += 1
                self.get_logger().debug(
                    f'[GOAL-REJECT] reason=clearance_too_low dt={dt_map[cix,ciy]:.2f}<{_ABS_MIN_CLR:.2f} '
                    f'type={_gtype} pt=({wx:.2f},{wy:.2f}) [{"door-threshold" if _gtype in _DOOR_TYPES else "normal-threshold(0.45m)"}]'
                )
                continue
            # ★ 硬过滤2: 类型相关安全距离
            _r_safe_eff = _door_r_safe if _gtype in _DOOR_TYPES else self.ssg_r_safe
            if dt_map[cix, ciy] < _r_safe_eff:
                _n_rej_clr += 1
                self.get_logger().debug(
                    f'[GOAL-REJECT] reason=type_clearance_fail dt={dt_map[cix,ciy]:.2f}<{_r_safe_eff:.2f} '
                    f'type={_gtype} pt=({wx:.2f},{wy:.2f})'
                )
                continue
            dist_to_here = float(np.sqrt((wx - px) ** 2 + (wy - py) ** 2))
            # ★[D4] 最小距离门控（按类型区分）：
            # - coverage/frontier：1.0m（避免在当前位置附近反复选停留点）
            # - 门类型：0.5m（允许近距离逼近门框，防止因短距被过滤整个 gap 失效）
            _min_dist_eff = 0.5 if _gtype in _DOOR_TYPES else 1.0
            if dist_to_here < _min_dist_eff:
                continue
            # ★ 黑名单过滤
            # confirmed_door / frontier_gap_pre / frontier_gap_center 用独立失败计数控制，不走全局黑名单
            # frontier_gap_post 到达后会被加入黑名单，需要检查（防止反复重选无效frontier边缘点）
            import time as _bl_chk
            _bl_now2 = _bl_chk.monotonic()
            _bl_key2 = f"{wx:.1f}_{wy:.1f}"
            _bl_exp2 = getattr(self, '_target_blacklist', {}).get(_bl_key2, 0.0)
            _BL_BYPASS_TYPES = frozenset(('confirmed_door', 'frontier_gap_pre', 'frontier_gap_center'))
            if _bl_exp2 > _bl_now2 and _gtype not in _BL_BYPASS_TYPES:
                _n_rej_blk += 1
                continue
            # ★ 回穿禁止：刚穿过的门在10s内同一位置gap候选直接拒绝
            # 防止「刚进第二个房间就被重新规划拉回去」
            _cross_gap_pos = getattr(self, '_last_crossed_gap_pos', None)
            _cross_gap_at  = getattr(self, '_last_crossed_gap_at', 0.0)
            if (_cross_gap_pos is not None and
                    _gtype in _DOOR_TYPES and
                    _bl_now2 - _cross_gap_at < 10.0 and
                    float(np.hypot(wx - _cross_gap_pos[0], wy - _cross_gap_pos[1])) < 1.2):
                _reentry_remain = 10.0 - (_bl_now2 - _cross_gap_at)
                if _bl_now2 - getattr(self, '_reentry_block_log_t', 0.0) > 3.0:
                    self._reentry_block_log_t = _bl_now2
                    self.get_logger().warn(
                        f'[REENTRY-BLOCK] 拒绝回穿候选 type={_gtype} pt=({wx:.1f},{wy:.1f}) '
                        f'last_crossed_gap=({_cross_gap_pos[0]:.1f},{_cross_gap_pos[1]:.1f}) '
                        f'禁止剩余={_reentry_remain:.0f}s'
                    )
                _n_rej_blk += 1
                continue
            # ★ 区域惩罚过滤：普通目标如果落在近期失败区域内则跳过（门目标豁免）
            if _gtype not in _DOOR_TYPES and hasattr(self, '_region_penalty'):
                _rp_now2 = _bl_now2   # 复用同一时间戳
                _rp_ri2 = int(round(wx / 1.5))
                _rp_rj2 = int(round(wy / 1.5))
                if self._region_penalty.get((_rp_ri2, _rp_rj2), 0.0) > _rp_now2:
                    _n_rej_blk += 1
                    continue
            ig = self._ssg_info_gain(cix, ciy)
            # IG 门控：门口所有类型豁免（门口目标IG可能为0，不应被拒绝）
            if ig < _min_ig_gate and _gtype not in _DOOR_TYPES:
                _rej_no_ig += 1
                continue
            # ★[D4] coverage 候选绝对 IG 门控：已探索区域 coverage 点 IG 极低但非零，
            # 归一化后干扰选点（normalization 会把 IG=0.05 放大到 _nig=1.0 if it's the max）
            # 对 coverage 类型施加更严格的绝对原始 IG 阈值：需有有效未知格在邻域
            # 阈值 0.5 ≈ 仅有 1 个 unknown 格在 2m 内（exp(-2/2.4)≈0.43），极低探索价值
            if _gtype == 'coverage' and ig < 0.5:
                _rej_no_ig += 1
                continue
            tg = self._ssg_target_gain(cix, ciy)
            risk = 1.0 / (dt_map[cix, ciy] + 0.1)
            revisit = self._ssg_revisit_penalty(wx, wy)
            # ★[D] REVISIT-PENALTY debug log
            if revisit > 0.15:
                self.get_logger().debug(
                    f'[REVISIT-PENALTY] goal=({wx:.2f},{wy:.2f}) '
                    f'type={_gtype} revisit={revisit:.3f}'
                )
            # 门口失败过滤（≥3次直接跳过）
            _door_fail_key = f"{wx:.1f}_{wy:.1f}"
            _fail_cnt_cand = getattr(self, '_door_fail_counts', {}).get(_door_fail_key, 0)
            if _fail_cnt_cand >= 3 and _gtype in _DOOR_TYPES:
                self.get_logger().debug(
                    f'[GOAL-REJECT] reason=gate_fail_limit fail={_fail_cnt_cand} '
                    f'type={_gtype} pt=({wx:.2f},{wy:.2f})'
                )
                continue
            pre_scored.append((0.0, wx, wy, cix, ciy, ig, tg, risk, revisit, dist_to_here,
                               _gtype, _fail_cnt_cand))
            if _gtype in _DOOR_TYPES:
                _n_door_added += 1
            else:
                _n_reg_added += 1

        # ★ [CAND-POOL] 统计日志（2s节流）
        import time as _cp_t
        if _cp_t.monotonic() - getattr(self, '_cand_pool_log_t', 0.0) > 2.0:
            self._cand_pool_log_t = _cp_t.monotonic()
            _n_gaps = len(getattr(self, 'frontier_gaps', []))
            _n_doors = len(self.doors_info)
            self.get_logger().info(
                f"[CAND-POOL] raw={len(raw_candidates)} door_raw={len(_door_raws)} "
                f"reg_quota={_regular_quota} → pool={len(pre_scored)} "
                f"(door={_n_door_added} reg={_n_reg_added} rej_clr={_n_rej_clr} "
                f"rej_blk={_n_rej_blk} rej_ig={_rej_no_ig}) "
                f"gaps={_n_gaps} doors_info={_n_doors}"
            )

        # ★ 评分归一化：将 IG/TG/cost/risk/revisit 全部缩放到 [0,1]，再加权
        _norm_ps = []
        if pre_scored:
            _all_ig  = [p[5] for p in pre_scored]
            _all_tg  = [p[6] for p in pre_scored]
            _all_rsk = [p[7] for p in pre_scored]
            _all_rev = [p[8] for p in pre_scored]
            _all_dst = [p[9] for p in pre_scored]
            _ig_max  = max(max(_all_ig),  1e-3)
            _tg_max  = max(max(_all_tg),  1e-3)
            _rsk_max = max(max(_all_rsk), 1e-3)
            _rev_max = max(max(_all_rev), 1e-3)
            _dst_max = max(max(_all_dst), 1e-3)
            # ★[L2] 门目标加成大幅提升：从 0.25→2.5，远超归一化 IG/TG (max=1.0×w_ig)，
            # 确保 frontier_gap 候选在候选池中始终压倒 coverage/frontier 类目标
            _BN_POST  = 2.5    # frontier_gap_post 加成（已知出口侧，最高优先）
            _BN_PRE   = 2.0    # frontier_gap_pre / frontier_gap_probe 加成
            _BN_INNER = 1.8    # inner / confirmed_door 加成
            _VPEN     = 0.80   # 已访问旧房间惩罚（> 任何单项加成）
            _FPEN_U   = 0.15   # 门口失败 per count 扣分
            # ★[E2] 门分割提前稳定（5帧）后门候选出现更早，降低cost权重促进选门
            # 若当前候选池中存在门类候选，对所有门候选额外降低距离惩罚系数
            _has_door_in_pool = any(
                p[10] in _DOOR_TYPES for p in pre_scored
            )
            _ssg_w_cost_door = _ssg_w_cost_eff * 0.6 if _has_door_in_pool else _ssg_w_cost_eff
            for (_, wx, wy, cix, ciy, ig, tg, risk, revisit, dist_to_here, _gtype, _fc) in pre_scored:
                _nig  = ig  / _ig_max
                _ntg  = tg  / _tg_max
                _nrsk = risk / _rsk_max
                _nrev = revisit / _rev_max
                _ndst = dist_to_here / _dst_max
                # ★[E2] frontier_gap_probe加成补齐（穿门探针≥pre优先级）
                if _gtype == 'frontier_gap_post':   _db = _BN_POST
                elif _gtype in ('frontier_gap_pre', 'frontier_gap_probe'):  _db = _BN_PRE
                elif _gtype in ('frontier_gap_center', 'confirmed_door'): _db = _BN_INNER
                else: _db = 0.0
                # ★ PASSAGE_COMMIT：选中门候选给予压倒性加成（+6.0，远超任何单项 IG/TG/距离项）
                if _pc_active and _gtype in _DOOR_TYPES:
                    _db = 6.0   # 压倒性加成：确保门候选必然赢得评分
                _fp = _fc * _FPEN_U
                _vp = 0.0
                if self.visited_rooms and self.room_labels is not None:
                    _clx = int(np.clip(cix, 0, self.grid_shape[0]-1))
                    _cly = int(np.clip(ciy, 0, self.grid_shape[1]-1))
                    _cl_lid = int(self.room_labels[_clx, _cly])
                    for _ri in self.rooms_info:
                        if _ri["label"] == _cl_lid:
                            if int(_ri["id"]) in self.visited_rooms and int(_ri["id"]) != self.current_room_id:
                                _vp = _VPEN
                            break
                # ★[E2] 门候选使用降低cost权重（_ssg_w_cost_door），其他类型不变
                _cost_w_eff2 = _ssg_w_cost_door if _gtype in _DOOR_TYPES else _ssg_w_cost_eff
                _ns = (w_ig * _nig + w_tg * _ntg
                       - _cost_w_eff2 * _ndst
                       - self.ssg_w_risk * _nrsk
                       - self.ssg_w_revisit * _nrev
                       + _db - _fp - _vp)
                _norm_ps.append((_ns, wx, wy, cix, ciy, ig, tg, risk, revisit, dist_to_here,
                                  _gtype, _dst_max))
        pre_scored = _norm_ps

        # 按归一化预评分降序，只对 top-K 运行精确 A*
        pre_scored.sort(key=lambda t: t[0], reverse=True)
        checked = 0
        _door_astar_infl = max(self.map_res, self.astar_inflation_radius * 0.35)
        for pre_s, wx, wy, cix, ciy, ig, tg, risk, revisit, dist_to_here, _gtype, _dst_max_r in pre_scored[:_ASTAR_TOP_K]:
            _orig_infl = self.astar_inflation_radius
            _orig_unk_sel = getattr(self, 'astar_treat_unknown_as_occ', True)  # ★[D1] 保存
            if _gtype in _DOOR_TYPES:
                self.astar_inflation_radius = _door_astar_infl
                # ★[D1] 门口候选允许穿越未知区：frontier_gap_post 在 room2 未知区域内
                # 若 unknown=occ，A* 无法规划到 room2，导致 goal snap 到门口 → 黑名单 90s
                # 解决方案：候选评分阶段也允许 unknown 作为可通行，与导航阶段保持一致
                self.astar_treat_unknown_as_occ = False
            path = self.astar_plan(np.array([px, py]), np.array([wx, wy]))
            self.astar_inflation_radius = _orig_infl
            self.astar_treat_unknown_as_occ = _orig_unk_sel  # ★[D1] 恢复
            # ★[I4] effective_goal 检查：A* 终点吸附偏移处理
            # - snap > 1.5m → 原始目标在深墙内，黑名单丢弃
            # - snap 0.20-1.5m → 目标贴墙被 snap，用 snap 后坐标替换 wx/wy 继续评分
            #   （原来只处理 >1.5m，0.2-1.5m 的贴墙目标直接用墙体坐标→无人机冲墙）
            _eff_end = np.array(path[-1], dtype=float) if (path and len(path) > 0) else np.array([wx, wy])
            _snap_d = float(np.linalg.norm(_eff_end - np.array([wx, wy])))
            self.astar_inflation_radius = _orig_infl
            if _snap_d > 1.5 and path is not None:
                import time as _eg_t2
                _bl_key_eg = f"{wx:.1f}_{wy:.1f}"
                if not hasattr(self, '_target_blacklist'):
                    self._target_blacklist = {}
                self._target_blacklist[_bl_key_eg] = _eg_t2.monotonic() + 90.0
                self.get_logger().warn(
                    f'[GOAL-REJECT] reason=astar_snap_too_far '
                    f'orig=({wx:.2f},{wy:.2f}) eff=({_eff_end[0]:.2f},{_eff_end[1]:.2f}) '
                    f'snap={_snap_d:.2f}m → blacklisted 90s'
                )
                continue
            if path is None:
                continue
            # ★[I4] snap 0.20-1.5m：用实际可达路径终点替换原始坐标，避免后续导航冲墙
            if _snap_d > 0.20:
                import time as _i4_t
                if _i4_t.monotonic() - getattr(self, '_i4_log_t', 0.0) > 3.0:
                    self._i4_log_t = _i4_t.monotonic()
                    self.get_logger().info(
                        f'[I4-GOAL-SNAP] 目标贴墙 snap={_snap_d:.2f}m '
                        f'orig=({wx:.2f},{wy:.2f}) → eff=({_eff_end[0]:.2f},{_eff_end[1]:.2f}) '
                        f'用路径终点替代原始目标'
                    )
                wx, wy = float(_eff_end[0]), float(_eff_end[1])
                # ★[K4] snap后重新校验clearance（snap点可能紧贴墙壁）
                _snap_gc_x, _snap_gc_y = self.world_to_grid(wx, wy)
                _snap_gc_x = int(np.clip(_snap_gc_x, 0, self.grid_shape[0]-1))
                _snap_gc_y = int(np.clip(_snap_gc_y, 0, self.grid_shape[1]-1))
                _snap_clr = float(dt_map[_snap_gc_x, _snap_gc_y])
                _snap_abs_min = _door_r_safe if _gtype in _DOOR_TYPES else 0.40
                if _snap_clr < _snap_abs_min:
                    import time as _k4_t
                    if _k4_t.monotonic() - getattr(self, '_k4_log_t', 0.0) > 3.0:
                        self._k4_log_t = _k4_t.monotonic()
                        self.get_logger().warn(
                            f'[K4-SNAP-REJECT] snap点净空不足 clr={_snap_clr:.2f}m<{_snap_abs_min:.2f}m '
                            f'type={_gtype} snap=({wx:.2f},{wy:.2f}) 丢弃候选'
                        )
                    self.astar_inflation_radius = _orig_infl
                    continue
            path_len = float(sum(
                np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                for i in range(len(path) - 1)
            )) if len(path) > 1 else dist_to_here
            # 用与预评分阶段相同的 _dst_max 归一化 A* 代价
            _ncost = path_len / max(_dst_max_r, 1e-3)
            _nig2  = ig   / max(_ig_max,  1e-3) if pre_scored else ig
            _ntg2  = tg   / max(_tg_max,  1e-3) if pre_scored else tg
            _nrsk2 = risk / max(_rsk_max, 1e-3) if pre_scored else risk
            _nrev2 = revisit / max(_rev_max, 1e-3) if pre_scored else revisit
            _fc2   = getattr(self, '_door_fail_counts', {}).get(f"{wx:.1f}_{wy:.1f}", 0)
            _fp2   = _fc2 * 0.15
            _vis_pen2 = 0.0
            if self.visited_rooms and self.room_labels is not None:
                _clx2 = int(np.clip(cix, 0, self.grid_shape[0]-1))
                _cly2 = int(np.clip(ciy, 0, self.grid_shape[1]-1))
                _cl_lid2 = int(self.room_labels[_clx2, _cly2])
                for _ri2 in self.rooms_info:
                    if _ri2["label"] == _cl_lid2:
                        if int(_ri2["id"]) in self.visited_rooms and int(_ri2["id"]) != self.current_room_id:
                            _vis_pen2 = 0.80
                        break
            # ★[L2][E2] A* 精确评分阶段同步更新门候选加成（与预评分阶段一致）
            # frontier_gap_probe 补齐加成，与 frontier_gap_pre 相同
            if _gtype == 'frontier_gap_post':   _db2 = 2.5
            elif _gtype in ('frontier_gap_pre', 'frontier_gap_probe'):  _db2 = 2.0
            elif _gtype in ('frontier_gap_center', 'confirmed_door'): _db2 = 1.8
            else: _db2 = 0.0
            # ★ PASSAGE_COMMIT 在 A* 精确评分阶段同样给压倒性加成
            if _pc_active and _gtype in _DOOR_TYPES:
                _db2 = 6.0
            # ★[E2] 门候选A*阶段也使用降低cost权重
            _cost_w_astar = _ssg_w_cost_door if _gtype in _DOOR_TYPES else _ssg_w_cost_eff
            score = (w_ig * _nig2 + w_tg * _ntg2
                     - _cost_w_astar * _ncost
                     - self.ssg_w_risk * _nrsk2
                     - self.ssg_w_revisit * _nrev2
                     + _db2 - _fp2 - _vis_pen2)
            if score > best_score:
                best_score = score
                best_pt = np.array([wx, wy], dtype=float)
                best_goal_type = _gtype
                best_log = {"ig": ig, "tg": tg, "cost": path_len, "risk": risk, "revisit": revisit}
            checked += 1

        if best_pt is not None:
            # ★[E] 目标点净空检查 + pullback
            _is_door_goal_e = best_goal_type in (
                'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                'frontier_gap_post', 'confirmed_door'
            )
            _min_clr_e = (float(getattr(self, 'goal_min_clearance_door', 0.35))
                          if _is_door_goal_e
                          else float(getattr(self, 'goal_min_clearance_normal', 0.5)))
            _gx_e, _gy_e = self.world_to_grid(float(best_pt[0]), float(best_pt[1]))
            _gx_e = int(np.clip(_gx_e, 0, self.grid_shape[0] - 1))
            _gy_e = int(np.clip(_gy_e, 0, self.grid_shape[1] - 1))
            _clr_e = float(dt_map[_gx_e, _gy_e])
            if _clr_e < _min_clr_e:
                _pb_m = float(getattr(self, 'goal_pullback_m', 0.45))
                _new_pt = self._pullback_point(best_pt, dt_map, _min_clr_e, _pb_m)
                if _new_pt is not None:
                    _new_gx, _new_gy = self.world_to_grid(float(_new_pt[0]), float(_new_pt[1]))
                    _new_gx = int(np.clip(_new_gx, 0, self.grid_shape[0] - 1))
                    _new_gy = int(np.clip(_new_gy, 0, self.grid_shape[1] - 1))
                    _new_clr = float(dt_map[_new_gx, _new_gy])
                    self.get_logger().info(
                        f"[PULLBACK] old=({best_pt[0]:.2f},{best_pt[1]:.2f}) "
                        f"new=({_new_pt[0]:.2f},{_new_pt[1]:.2f}) "
                        f"dist={float(np.hypot(_new_pt[0]-best_pt[0],_new_pt[1]-best_pt[1])):.2f}m "
                        f"ok={_new_clr >= _min_clr_e}"
                    )
                    self.get_logger().debug(
                        f"[GOAL-CLEAR] type={best_goal_type} clr={_clr_e:.2f}m < {_min_clr_e:.2f}m "
                        f"→ pullback new_clr={_new_clr:.2f}m ok={_new_clr >= _min_clr_e}"
                    )
                    best_pt = _new_pt
                else:
                    # ★[K3] pullback失败时拒绝该候选，回退到fallback（原来保留靠墙原始点，已修复）
                    self.get_logger().warn(
                        f"[GOAL-CLEAR] type={best_goal_type} clr={_clr_e:.2f}m < {_min_clr_e:.2f}m "
                        f"pullback failed → REJECT+FALLBACK (was keeping wall-adjacent original, now fixed)"
                    )
                    self.ssg_goal_type = "fallback"
                    return self._ssg_fallback_viewpoint()
            else:
                self.get_logger().debug(
                    f"[GOAL-CLEAR] type={best_goal_type} clr={_clr_e:.2f}m ok={_clr_e >= _min_clr_e}"
                )
            # ★[C1] 统一安全可达性最终门控（黑名单 + 边界 + 净空二次确认）
            _c1_ok, _c1_reason = self._is_goal_reachable_and_safe(best_pt, best_goal_type, dt_map)
            if not _c1_ok:
                self.get_logger().warn(
                    f"[GOAL-REJECT] reason=safety_gate_fail detail={_c1_reason} "
                    f"type={best_goal_type} goal=({best_pt[0]:.2f},{best_pt[1]:.2f})"
                )
                self._log_event("GOAL-REJECT",
                    f"reason=safety_gate_fail detail={_c1_reason} type={best_goal_type} "
                    f"goal=({best_pt[0]:.2f},{best_pt[1]:.2f})")
                self.ssg_goal_type = "fallback"
                return self._ssg_fallback_viewpoint()
            self.ssg_goal_type = best_goal_type
            # ★[A5] 门目标锁定：选中门类目标时激活 _goal_hold_until
            _DOOR_TYPES_SET = frozenset((
                'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                'frontier_gap_post', 'confirmed_door'
            ))
            if best_goal_type in _DOOR_TYPES_SET:
                _hold_deadline = time.monotonic() + self._goal_switch_hold_sec
                self._goal_hold_until = _hold_deadline
                self.get_logger().warn(
                    f"[GOAL-LOCK] 门目标锁定激活 type={best_goal_type} "
                    f"hold={self._goal_switch_hold_sec:.0f}s "
                    f"goal=({best_pt[0]:.2f},{best_pt[1]:.2f})"
                )
            self.get_logger().info(
                f"[GOAL-SET] type={best_goal_type} goal=({best_pt[0]:.2f},{best_pt[1]:.2f}) "
                f"score={best_score:.2f} ig={best_log['ig']:.1f} tg={best_log['tg']:.1f} "
                f"cost={best_log['cost']:.2f} risk={best_log['risk']:.2f} revisit={best_log['revisit']:.2f} "
                f"cur_room={self.current_room_id} visited={sorted(self.visited_rooms)} "
                f"checked={checked} rej_ig={_rej_no_ig} "
                f"clustered={_clustered} cost_w={_ssg_w_cost_eff:.2f}"
            )
            # ★[I2] SSG-GOAL：标准化目标选择日志，含空间冷却信息
            import time as _ssg_gl_t
            _ssg_gl_now = _ssg_gl_t.monotonic()
            _cell_m = float(getattr(self, 'spatial_cooldown_cell_m', 1.5))
            _ix_g = int(best_pt[0] / _cell_m)
            _iy_g = int(best_pt[1] / _cell_m)
            _cd = getattr(self, '_spatial_cooldown', {})
            _cd_expiry = _cd.get((_ix_g, _iy_g), 0.0)
            _cd_remain = max(0.0, _cd_expiry - _ssg_gl_now)
            self.get_logger().info(
                f"[SSG-GOAL] type={best_goal_type} "
                f"goal=({best_pt[0]:.2f},{best_pt[1]:.2f}) "
                f"score={best_score:.3f} room={self.current_room_id} "
                f"cooldown={_cd_remain:.0f}s revisit={best_log['revisit']:.2f}"
            )
            self._log_event(
                "GOAL-SET",
                f"type={best_goal_type} goal=({best_pt[0]:.2f},{best_pt[1]:.2f}) "
                f"score={best_score:.2f} ig={best_log['ig']:.1f} checked={checked} "
                f"cur_room={self.current_room_id} visited={sorted(self.visited_rooms)}"
            )
            # ★[B] GOAL-SWITCH 防抖与日志
            import time as _gsw_t
            _gsw_now = _gsw_t.monotonic()
            _prev_vp = getattr(self, 'ssg_viewpoint_goal', None)
            _DOOR_SWITCH_FREEZE = frozenset((
                'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                'frontier_gap_post', 'confirmed_door'
            ))
            if _prev_vp is not None and best_goal_type not in _DOOR_SWITCH_FREEZE:
                _dist_new = float(np.linalg.norm(best_pt - _prev_vp))
                if _dist_new > 0.5:
                    _since_sw = _gsw_now - self._goal_switch_last_t
                    if _since_sw < self.goal_switch_min_interval_sec:
                        # 防抖：切换间隔过短，抑制本次切换
                        self.get_logger().info(
                            f"[GOAL-SWITCH] suppressed type={best_goal_type} "
                            f"dist={_dist_new:.2f}m interval={_since_sw:.1f}s"
                            f"<{self.goal_switch_min_interval_sec:.1f}s"
                        )
                        self._log_event("GOAL-SWITCH",
                            f"suppressed type={best_goal_type} dist={_dist_new:.2f}m "
                            f"interval={_since_sw:.1f}s")
                        self.ssg_goal_type = self._goal_switch_last_type or best_goal_type
                        return _prev_vp
                    # 切换间隔充足，记录 GOAL-SWITCH
                    self.get_logger().info(
                        f"[GOAL-SWITCH] old=({_prev_vp[0]:.2f},{_prev_vp[1]:.2f}) "
                        f"new=({best_pt[0]:.2f},{best_pt[1]:.2f}) "
                        f"type={best_goal_type} dist={_dist_new:.2f}m "
                        f"interval={_since_sw:.1f}s"
                    )
                    self._log_event("GOAL-SWITCH",
                        f"old=({_prev_vp[0]:.2f},{_prev_vp[1]:.2f}) "
                        f"new=({best_pt[0]:.2f},{best_pt[1]:.2f}) "
                        f"type={best_goal_type} dist={_dist_new:.2f}m")
            self._goal_switch_last_t = _gsw_now
            self._goal_switch_last_type = best_goal_type
            return best_pt

        # Task7: 若 IG 门控导致所有候选点被拒绝，降级忽略 IG 门控再选一次
        if _rej_no_ig > 0 and not target_mode:
            self.get_logger().warn(
                f"[SSG-SELECT] 所有候选被IG门控拒绝(rej_ig={_rej_no_ig})，"
                "地图可能已充分探索，忽略IG门控重试"
            )
            for wx, wy, _gtype in raw_candidates:
                cix, ciy = self.world_to_grid(wx, wy)
                if not (0 <= cix < self.grid_shape[0] and 0 <= ciy < self.grid_shape[1]):
                    continue
                if self.grid[cix, ciy] > self.prob_free_th:
                    continue
                if dt_map[cix, ciy] < self.ssg_r_safe:
                    continue
                path = self.astar_plan(np.array([px, py]), np.array([wx, wy]))
                if path is not None:
                    self.ssg_goal_type = _gtype
                    self.get_logger().info(
                        f"[GOAL-SET] type={_gtype} goal=({wx:.2f},{wy:.2f}) reason=ig_gate_degraded"
                    )
                    return np.array([wx, wy], dtype=float)

        self.ssg_goal_type = "fallback"
        self.get_logger().warn("[SSG-SELECT] 所有候选点被过滤，使用fallback")
        return self._ssg_fallback_viewpoint()

    def _ssg_fallback_viewpoint(self) -> Optional[np.ndarray]:
        """兜底：选择最近的 frontier 格作为扫描目标。★ 加入 clearance 过滤，不再吐墙边点。"""
        if self.pose is None:
            return None
        px, py = self._plan_px, self._plan_py
        free_mask = self.grid <= self.prob_free_th
        unk_mask = (self.grid > self.prob_free_th) & (self.grid < self.prob_occ_th)
        unk_dil = binary_dilation(unk_mask, iterations=1)
        front_mask = free_mask & unk_dil
        cells = np.argwhere(front_mask)
        if cells.shape[0] == 0:
            return None
        # ★ clearance 过滤：不允许 fallback 点贴墙
        _fb_dt = None
        try:
            _fb_dt = self._ssg_compute_dist_transform()
        except Exception:
            pass
        _FB_MIN_CLR = 0.30
        if _fb_dt is not None:
            _cx = np.clip(cells[:, 0], 0, _fb_dt.shape[0]-1)
            _cy = np.clip(cells[:, 1], 0, _fb_dt.shape[1]-1)
            _clr_vals = _fb_dt[_cx, _cy]
            _valid = _clr_vals >= _FB_MIN_CLR
            if np.any(_valid):
                cells = cells[_valid]
            else:
                self.get_logger().warn(
                    f'[GOAL-REJECT] reason=fallback_all_clearance_fail min_clr={_FB_MIN_CLR:.2f}'
                )
        pix = int(np.clip((px - self.bounds[0, 0]) / self.map_res, 0, self.grid_shape[0] - 1))
        piy = int(np.clip((py - self.bounds[1, 0]) / self.map_res, 0, self.grid_shape[1] - 1))
        dists = np.sqrt((cells[:, 0] - pix) ** 2 + (cells[:, 1] - piy) ** 2)
        mid = (dists * self.map_res >= 0.5) & (dists * self.map_res <= 8.0)
        if np.any(mid):
            cells = cells[mid]; dists = dists[mid]
        idx = int(np.argmin(dists))
        gx, gy = self.grid_to_world(int(cells[idx, 0]), int(cells[idx, 1]))
        return np.array([gx, gy], dtype=float)

    def _obstacle_aware_forward(self) -> np.ndarray:
        """兜底前进：仅在 enable_wall_follow_fallback=True 时使用 wall_follow，否则原地停住。"""
        if getattr(self, 'enable_wall_follow_fallback', False):
            import time as _owf_t
            _owf_now = _owf_t.monotonic()
            _owf_start = getattr(self, '_wall_follow_start_t', 0.0)
            if _owf_start == 0.0:
                self._wall_follow_start_t = _owf_now
                _owf_start = _owf_now
            _wf_dur = _owf_now - _owf_start
            _max_wf  = float(getattr(self, 'wall_follow_max_sec', 0.8))
            _near_wf = self._nearest_lidar_distance()
            _clr_ok_wf = (_near_wf is None or _near_wf >= float(getattr(self, 'safe_hold_clear_ok_m', 0.6)))
            if _wf_dur > _max_wf or _clr_ok_wf:
                _exit = "clearance_ok" if _clr_ok_wf else "timeout"
                self.get_logger().warn(
                    f"[WALL-FOLLOW] enabled=True duration={_wf_dur:.2f}s exit_reason={_exit}"
                )
                self._wall_follow_start_t = 0.0
                return np.zeros(3, dtype=float)
            return self.wall_follow_cmd()
        else:
            self._wall_follow_start_t = 0.0
    # ----------------------------------------------------------------
    # ★[C1] 统一安全可达性检查（consolidates clearance + blacklist checks）
    # ----------------------------------------------------------------
    def _is_goal_reachable_and_safe(
        self,
        pt: np.ndarray,
        goal_type: str,
        dt_map=None,
    ):
        """★[C1] 统一目标安全性判断：黑名单 + 净空 + (可选) 占用格检测。
        返回 (ok: bool, reason: str)。
        本函数被 _ssg_select_viewpoint 等调用，作为最终 accept/reject 门控。
        不执行 A* 规划（规划已在主循环中完成），避免重复开销。
        """
        import time as _irs_t
        _irs_now = _irs_t.monotonic()
        _sig = f"{float(pt[0]):.1f}_{float(pt[1]):.1f}"

        # 1. 黑名单过滤
        _bl = getattr(self, '_target_blacklist', {})
        if _sig in _bl and _bl[_sig] > _irs_now:
            return False, f"blacklisted(ttl={_bl[_sig] - _irs_now:.0f}s)"

        # 2. 占用格检测（目标不得落在占用格上）
        if self.grid is not None:
            _gx, _gy = self.world_to_grid(float(pt[0]), float(pt[1]))
            _gx = int(np.clip(_gx, 0, self.grid_shape[0] - 1))
            _gy = int(np.clip(_gy, 0, self.grid_shape[1] - 1))
            if self.grid[_gx, _gy] >= self.prob_occ_th:
                return False, "goal_on_occ_cell"

        # 3. 净空检查（需要 dt_map）
        if dt_map is not None:
            _gx2, _gy2 = self.world_to_grid(float(pt[0]), float(pt[1]))
            _gx2 = int(np.clip(_gx2, 0, dt_map.shape[0] - 1))
            _gy2 = int(np.clip(_gy2, 0, dt_map.shape[1] - 1))
            _clr = float(dt_map[_gx2, _gy2])
            _DOOR_TYPES = frozenset((
                'frontier_gap_pre', 'frontier_gap_center', 'frontier_gap_post',
                'frontier_gap_probe', 'confirmed_door'
            ))
            _min_clr = (self.goal_min_clearance_door
                        if goal_type in _DOOR_TYPES
                        else self.goal_min_clearance_normal)
            if _clr < _min_clr * 0.8:   # 留有 20% 容差防止地图噪声
                return False, f"clearance_too_low({_clr:.2f}m<{_min_clr * 0.8:.2f}m)"

        # 4. 地图边界检测
        if self.bounds is not None:
            bx0, bx1 = float(self.bounds[0, 0]), float(self.bounds[0, 1])
            by0, by1 = float(self.bounds[1, 0]), float(self.bounds[1, 1])
            if not (bx0 <= float(pt[0]) <= bx1 and by0 <= float(pt[1]) <= by1):
                return False, "out_of_bounds"

        return True, "ok"

    # ----------------------------------------------------------------
    # ★[A] Carrot-lookahead：在 A* 路径上找"前方可见最远点"作为跟踪点
    # ----------------------------------------------------------------
    def _pick_carrot_point(
        self,
        path: list,
        start_idx: int,
        lookahead_m: float,
        occ_inflated,         # np.ndarray bool, shape=(W,H)
    ):
        """在 path[start_idx:] 上找最远的"LOS 可见且距离 ≤ lookahead_m"点。
        返回 (carrot_xy np.ndarray, carrot_idx int) 或 (None, start_idx)。
        LOS 采用均匀采样栅格检测，检查 occ_inflated（膨胀障碍）。
        ★[FIX-CARROT] 若所有 LOS 均失败但首个路径点很近（<0.5m），直接返回该点：
          A* 起点 snap 场景：drone 在膨胀区，carrot LOS 从膨胀区出发必然失败，
          但 drone 其实就在 A* 起点附近，直接执行第一个 waypoint 是安全的。
        """
        if not path or start_idx >= len(path):
            return None, start_idx

        px, py = self._plan_px, self._plan_py
        best_xy  = None
        best_idx = start_idx
        scan_end = min(len(path), start_idx + self.carrot_scan_max_pts)

        for i in range(start_idx, scan_end):
            wp = path[i]
            dx, dy = float(wp[0]) - px, float(wp[1]) - py
            d = float(np.sqrt(dx * dx + dy * dy))
            if d > lookahead_m:
                break              # 超出 lookahead，停止
            # LOS 检测：从 pos→wp 均匀采样，检查膨胀占据
            los_ok = True
            steps = max(2, int(d / (self.map_res * 1.5)))
            for s in range(1, steps + 1):
                t   = s / steps
                rx  = px + dx * t
                ry  = py + dy * t
                gx, gy = self.world_to_grid(rx, ry)
                W, H = self.grid_shape
                gx = int(np.clip(gx, 0, W - 1))
                gy = int(np.clip(gy, 0, H - 1))
                if occ_inflated[gx, gy]:
                    los_ok = False
                    break
            if los_ok:
                best_xy  = np.array([float(wp[0]), float(wp[1])], dtype=float)
                best_idx = i
            else:
                break              # LOS 被阻，不往更远走

        # ★[FIX-CARROT] A* snap 补偿：若未找到 carrot 点但首个 waypoint 很近(<0.5m)，
        # 直接返回首个 waypoint（drone 在膨胀区时 LOS 必然失败，但实际可以安全移动）
        if best_xy is None and start_idx < len(path):
            _wp0 = path[start_idx]
            _d0 = float(np.sqrt((float(_wp0[0]) - px) ** 2 + (float(_wp0[1]) - py) ** 2))
            if _d0 < 0.5:   # 0.5m 内：drone 很近，直接执行
                best_xy  = np.array([float(_wp0[0]), float(_wp0[1])], dtype=float)
                best_idx = start_idx
                import time as _cp_snap_t
                if _cp_snap_t.monotonic() - getattr(self, '_carrot_snap_log_t', 0.0) > 2.0:
                    self._carrot_snap_log_t = _cp_snap_t.monotonic()
                    self.get_logger().debug(
                        f'[CARROT-SNAP] LOS失败但wp近 d={_d0:.2f}m < 0.5m → 直接执行wp'
                    )

        return best_xy, best_idx

    # ----------------------------------------------------------------
    # ★[D] 计算逃离方向（从 LiDAR 最大空弧）
    # ----------------------------------------------------------------
    def _compute_escape_dir_from_lidar(self) -> Optional[np.ndarray]:
        """从 4 扇区 LiDAR 距离中找最大空弧方向（map/ENU frame）。
        返回单位向量 [ex, ey] 或 None（无 LiDAR 数据）。"""
        if self.att_yaw is None:
            return None
        yaw = float(self.att_yaw)
        sector_dirs = {
            'front': np.array([ np.cos(yaw),  np.sin(yaw)]),
            'back':  np.array([-np.cos(yaw), -np.sin(yaw)]),
            'left':  np.array([-np.sin(yaw),  np.cos(yaw)]),
            'right': np.array([ np.sin(yaw), -np.cos(yaw)]),
        }
        best_d   = -1.0
        best_dir = None
        for sk, sv in sector_dirs.items():
            v = self.lidar_ranges.get(sk)
            if v is not None and np.isfinite(v):
                if float(v) > best_d:
                    best_d   = float(v)
                    best_dir = sv
        return best_dir.copy() if best_dir is not None else None

    # ----------------------------------------------------------------
    # ★[E] 目标点 pullback：沿 dt_map 梯度拉回到 clearance≥min_clr
    # ----------------------------------------------------------------
    def _pullback_point(
        self,
        goal_xy: np.ndarray,
        dt_map,          # np.ndarray float, clearance in metres
        min_clr: float,
        pullback_m: float,
    ) -> Optional[np.ndarray]:
        """若 goal_xy 净空 < min_clr，沿 dt_map 梯度方向拉回最多 pullback_m。
        成功返回新坐标，失败返回 None。"""
        W, H = self.grid_shape
        gx0, gy0 = self.world_to_grid(float(goal_xy[0]), float(goal_xy[1]))
        gx0 = int(np.clip(gx0, 1, W - 2))
        gy0 = int(np.clip(gy0, 1, H - 2))
        clr0 = float(dt_map[gx0, gy0])
        if clr0 >= min_clr:
            return goal_xy.copy()   # 已满足
        # 梯度方向（dt_map 梯度指向离墙最快增大方向）
        gx_grad = float(dt_map[gx0 + 1, gy0] - dt_map[gx0 - 1, gy0]) * 0.5
        gy_grad = float(dt_map[gx0, gy0 + 1] - dt_map[gx0, gy0 - 1]) * 0.5
        mag = float(np.sqrt(gx_grad ** 2 + gy_grad ** 2))
        if mag < 1e-4:
            return None
        # 梯度是格子空间，转换到世界方向（栅格各向同性）
        wx_dir = gx_grad / mag
        wy_dir = gy_grad / mag
        steps  = max(4, int(pullback_m / self.map_res))
        best_pt  = None
        best_clr = clr0
        for s in range(1, steps + 1):
            rx = float(goal_xy[0]) + wx_dir * s * self.map_res
            ry = float(goal_xy[1]) + wy_dir * s * self.map_res
            pgx, pgy = self.world_to_grid(rx, ry)
            pgx = int(np.clip(pgx, 0, W - 1))
            pgy = int(np.clip(pgy, 0, H - 1))
            c = float(dt_map[pgx, pgy])
            if c > best_clr:
                best_clr = c
                best_pt  = np.array([rx, ry], dtype=float)
                if c >= min_clr:
                    break
        return best_pt

    def wall_follow_cmd(self) -> np.ndarray:
        """沿墙巡航指令（body frame 扇区，与 scan_cb 使用相同的 body_angles）：
          front: body(-35°,+35°)   = 机体前方 (+X)
          right: body(+55°,+125°)  = 机体右方 (+Y)
          back:  body(±145°,±180°) = 机体后方 (-X)
          left:  body(-125°,-55°)  = 机体左方 (-Y)
        vy>0 = body+Y = 向右；vy<0 = body-Y = 向左。
        """
        vx = 0.35  # 默认向前（body+X）
        vy = 0.0
        front = self.lidar_ranges.get("front")
        left  = self.lidar_ranges.get("left")
        right = self.lidar_ranges.get("right")
        back  = self.lidar_ranges.get("back")

        # 前方障碍处理
        if front is not None:
            if front < 0.45:
                vx = 0.0
                # 前方堵死：选左右空旷侧转向
                if left is not None and right is not None:
                    # left<right → 左侧更近 → 向右（body+Y）逃
                    vy = 0.35 if left < right else -0.35
                elif right is not None:
                    vy = 0.35   # 只有右侧数据：向右
                elif left is not None:
                    vy = -0.35  # 只有左侧数据：向左
            elif front < 0.9:
                # 前方减速
                vx *= max(0.1, (front - 0.45) / 0.45)

        # 后方保护：后方太近时向前推力
        if back is not None and back < 0.35:
            vx = max(vx, 0.15)

        # 横向平衡：靠近哪侧就远离哪侧
        # left<right(左近) → 向右(vy正) → (left-right) < 0 → *(negative coeff) > 0
        # 使用 (right-left)*0.5 等价于 (left-right)*-0.5，但更清晰
        if left is not None and right is not None:
            balance = float(np.clip((right - left) * 0.5, -0.3, 0.3))
            # 仅当非已触发硬转向时叠加横向修正
            if abs(vy) < 0.1:
                vy += balance
            else:
                vy = float(np.clip(vy + balance * 0.3, -0.5, 0.5))

        if self.att_rot is None:
            return np.array([0.0, 0.0, 0.0], dtype=float)

        # att_rot: body(FRD) → NED；v_ned[0]=NED_North, v_ned[1]=NED_East
        # publish_setpoint 期望 ENU 输入(vel[0]=East, vel[1]=North)，这里做 NED→ENU
        v_ned = self.att_rot.apply(np.array([vx, vy, 0.0], dtype=float))
        return np.array([float(v_ned[1]), float(v_ned[0]), 0.0], dtype=float)  # ENU: x=East=NED_y, y=North=NED_x

    def _nearest_lidar_distance(self) -> Optional[float]:
        vals = []
        for k in ("front", "left", "right", "back"):
            v = self.lidar_ranges.get(k)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        if not vals:
            return None
        return float(min(vals))

    def apply_proximity_constraints(self, vel: np.ndarray) -> np.ndarray:
        out = vel.copy()
        near = self._nearest_lidar_distance()
        if near is None:
            return out

        # ★ [DOOR-FIX] 门口目标/门穿越模式：大幅收紧 PROX-STOP 阈值
        #   正常模式: stop_dist=0.35m（全向近墙硬停）
        #   门口模式: stop_dist→0.13m（仅在真实碰墙时才停）
        #   原因：穿门时侧方门框距离约 0.25~0.35m，若用普通阈值会触发"逃逸速度"
        #         导致无人机被推离门口、在门边反复悬停/震荡
        _prox_in_door_mode = (
            getattr(self, '_door_transit_mode', False)
            or getattr(self, 'ssg_goal_type', 'none') in (
                'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                'frontier_gap_post', 'confirmed_door')
        )
        _prox_stop_th = 0.13 if _prox_in_door_mode else self.obstacle_cmd_stop_dist
        # ★[I4] DOOR-MODE on/off 日志（状态切换时打印）
        import time as _dm_ts_t
        _dm_ts_now = _dm_ts_t.monotonic()
        _prev_door_mode = getattr(self, '_prox_door_mode_prev', False)
        if _prox_in_door_mode != _prev_door_mode:
            self._prox_door_mode_prev = _prox_in_door_mode
            _dm_reason = (
                "door_transit" if getattr(self, '_door_transit_mode', False)
                else f"goal_type={getattr(self,'ssg_goal_type','none')}"
            )
            self.get_logger().info(
                f"[DOOR-MODE] {'on' if _prox_in_door_mode else 'off'} "
                f"reason={_dm_reason} "
                f"stop_th={_prox_stop_th:.2f}m "
                f"(was {self.obstacle_cmd_stop_dist:.2f}m)"
            )

        if near <= _prox_stop_th:
            if self.att_rot is None:
                out[0] = 0.0
                out[1] = 0.0
                return out

            front = self.lidar_ranges.get("front")
            left = self.lidar_ranges.get("left")
            right = self.lidar_ranges.get("right")
            back = self.lidar_ranges.get("back")
            sectors = {
                "front": front if front is not None else 1e9,
                "left": left if left is not None else 1e9,
                "right": right if right is not None else 1e9,
                "back": back if back is not None else 1e9,
            }
            blocked = min(sectors, key=sectors.get)

            vx_body = 0.0
            vy_body = 0.0
            esc = float(np.clip(self.obstacle_escape_speed, 0.0, self.max_speed))
            # 逃逸方向：远离阻挡扇区对应的 body 方向
            # front=body+X → 逃body-X → vx=-esc
            # back =body-X → 逃body+X → vx=+esc
            # left =body-Y → 逃body+Y → vy=+esc
            # right=body+Y → 逃body-Y → vy=-esc
            if blocked == "front":
                vx_body = -esc
            elif blocked == "back":
                vx_body = esc
            elif blocked == "left":
                vy_body = esc   # 左侧(body-Y)障碍 → 向右(body+Y)逃
            else:               # right
                vy_body = -esc  # 右侧(body+Y)障碍 → 向左(body-Y)逃

            # ★ 修复: att_rot 变换结果 v_ned 是 NED 帧 [North, East, Down]
            # out 是 ENU 帧 [East, North, Up]，需要 swap: ENU_x=NED_y, ENU_y=NED_x
            v_ned = self.att_rot.apply(np.array([vx_body, vy_body, 0.0], dtype=float))
            out[0] = float(v_ned[1])   # ENU East = NED East (v_ned[1])
            out[1] = float(v_ned[0])   # ENU North = NED North (v_ned[0])
            # Phase-6：硬停贵诊断日志（节流：每 2s 最多印一次）
            import time as _px_t
            _now_pxt = _px_t.monotonic()
            if _now_pxt - self._prox_stop_diag_t >= 2.0:
                self._prox_stop_diag_t = _now_pxt
                _fmt = lambda v: f"{v:.2f}" if v and np.isfinite(v) else "--"
                self.get_logger().warn(
                    f"[PROX-STOP] 硬停触发! near={near:.2f}m <= {_prox_stop_th:.2f}m "
                    f"door_mode={_prox_in_door_mode} "
                    f"blocked={blocked} 主副机测距: "
                    f"F={_fmt(front)} L={_fmt(left)} R={_fmt(right)} B={_fmt(back)} "
                    f"逃逸速={esc:.2f}m/s"
                )
            return out

        # ── PROX-BRAKE: stop_dist ~ 0.60m 区间强制帽速 0.08m/s ──────────────────
        #    硬停 (near < stop_dist) 之上、常规减速 (near < slowdown_start) 之内的
        #    紧急制动带，防止激光噪声导致短暂超过 stop_dist 后瞬间全速撞墙。
        # ★ 门口目标例外：降低制动阈值至0.28m，放宽帽速至0.35m/s，允许通过窄门
        _is_door_brake = (
            getattr(self, 'ssg_goal_type', 'none') in (
                'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                'frontier_gap_post', 'confirmed_door')
            # ★ door_transit_mode 几何触发：路径上事实在穿门时也放宽制动带
            or getattr(self, '_door_transit_mode', False)
        )
        if _is_door_brake:
            _prox_brake_threshold = 0.28   # m：门口模式减小制动带，允许贴近门框
            _prox_brake_max_xy    = 0.35   # m/s：门口允许较高穿越速度
        else:
            _prox_brake_threshold = 0.60   # m：制动带（resume_clearance=0.45m，brake保持0.60m提前减速）
            _prox_brake_max_xy    = 0.08   # m/s
        if near is not None and near <= _prox_brake_threshold:
            xy_spd = float(np.linalg.norm(out[:2]))
            if xy_spd > _prox_brake_max_xy:
                out[0] *= _prox_brake_max_xy / xy_spd
                out[1] *= _prox_brake_max_xy / xy_spd
            import time as _pb_t
            _now_pb = _pb_t.monotonic()
            if _now_pb - getattr(self, '_prox_brake_log_t', 0.0) > 0.5:
                self._prox_brake_log_t = _now_pb
                self.get_logger().warn(
                    f"[PROX-BRAKE] near={near:.2f}m <= {_prox_brake_threshold:.2f}m "
                    f"door_mode={_is_door_brake} "
                    f"xy_limited={min(xy_spd, _prox_brake_max_xy):.3f}m/s "
                    f"(was {xy_spd:.3f}m/s)"
                )

        start = max(self.obstacle_slowdown_start, self.obstacle_cmd_stop_dist + 0.05)
        if near < start:
            # ★[J5] 门模式减速区：使用门模式有效停止距离(_prox_stop_th=0.13m)而非通用距离(0.35m)。
            # 根因：门框侧方净空约 0.27-0.35m，若用 obstacle_cmd_stop_dist=0.35m 作为下限，
            # scale=(0.27-0.35)/0.85 = 负值 → clip→0 → XY速度归零，无人机在门口完全停住。
            # 修复：门模式下用 _prox_stop_th=0.13m，允许在门框侧方时仍保留前向速度分量。
            _scale_stop_th = _prox_stop_th if _prox_in_door_mode else self.obstacle_cmd_stop_dist
            scale = float(np.clip((near - _scale_stop_th) / (start - _scale_stop_th), 0.0, 1.0))
            # ── 投影式避障（Phase-6）：
            #    消去指向最近障碍物方向的速度分量，允许无人机沿墙滑行
            #    而不是直接将整个水平速度缩放为零——减少贴墙推墙现象
            # ★ Phase12门目标例外：门目标禁止投影滑行，改为走廊对中纠偏
            #    原因：投影后剩余分量 = 沿墙切向，门口会变成沿门框蹭过
            _is_door_proj = getattr(self, 'ssg_goal_type', 'none') in (
                'frontier_gap_pre', 'frontier_gap_center', 'frontier_gap_post',
                'frontier_gap_probe', 'confirmed_door')
            if _is_door_proj:
                # ★ 走廊对中模式：利用左右激光差计算横向纠偏速度
                # 而不是投影消去分量（投影 = 沿墙滑行，不适合门口）
                _left_d  = self.lidar_ranges.get("left")   if self.lidar_ranges.get("left")  is not None else 2.0
                _right_d = self.lidar_ranges.get("right")  if self.lidar_ranges.get("right") is not None else 2.0
                _left_d  = max(0.1, float(_left_d))
                _right_d = max(0.1, float(_right_d))
                # 走廊对中纠偏：左靠近 → 向右加速 | 右靠近 → 向左加速
                # ★[C1] 0.25→0.12, ±0.20→±0.10：降低增益+限幅防左右振荡
                _CENTER_GAIN = 0.12   # 纠偏增益 (m/s per m of imbalance)
                _lateral_err = _left_d - _right_d   # 正: 左宽右窄 → 向右(body+Y)修正
                _lateral_corr_body = float(np.clip(_lateral_err * _CENTER_GAIN, -0.10, 0.10))
                if self.att_rot is not None:
                    _corr_ned = self.att_rot.apply(np.array([0.0, _lateral_corr_body, 0.0], dtype=float))
                    _corr_enu = np.array([float(_corr_ned[1]), float(_corr_ned[0]), 0.0])
                    # ★[C1-ARCH] 走廊偏置职责隔离：
                    # 当 local_mpc 正在提供路径跟踪命令时（local_fresh=True），
                    # 跳过直接速度注入——local_mpc 的路径控制已包含精确横向约束；
                    # 双重叠加会造成走廊左右振荡（corridor centering 反向修正 local_mpc 的修正）。
                    # 仅当 local_fresh=False（FALLBACK模式）时才注入，用于兜底对中。
                    _corridor_local_fresh = self.is_cmd_fresh(
                        self.last_local_stamp, self.get_clock().now())
                    # ★[FIX-C ARCH] 门模式走廊居中纠偏：无论 LOCAL_DOMINANT 还是 FALLBACK 均全量注入
                    #   根本原因：local_mpc 只跟踪路径点（不感知门框几何），无横向居中约束；
                    #   在 LOCAL_DOMINANT 模式下仅注入 8%（≈0.008 m/s）等于无效，
                    #   无人机以 >0.35 m/s 穿越 0.27m 宽度的门框 → 侧方碰撞。
                    #   走廊居中纠偏是门穿越的安全约束（planning/safety 层职责），
                    #   应始终叠加到最终输出速度上（加法叠加，不与 local_mpc 产生冲突）。
                    if not _corridor_local_fresh or _prox_in_door_mode:
                        # FALLBACK 模式 OR 门穿越模式：完整注入走廊纠偏
                        out[:2] += _corr_enu[:2]
                    elif abs(_lateral_err) > 0.40:
                        # ★[D] LOCAL_DOMINANT 普通模式偏差极大(>0.4m)：进一步降低至4%避免振荡
                        # 原8%在走廊内与local_mpc路径修正方向相反时会造成左右振荡
                        out[:2] += _corr_enu[:2] * 0.04
                import time as _cp_t
                _cp_now = _cp_t.monotonic()
                if _cp_now - getattr(self, '_center_proj_log_t', 0.0) > 1.0:
                    self._center_proj_log_t = _cp_now
                    _corridor_local_fresh_log = self.is_cmd_fresh(
                        self.last_local_stamp, self.get_clock().now())
                    self.get_logger().info(
                        f'[CORRIDOR-CENTER] door_mode=True L={_left_d:.2f}m R={_right_d:.2f}m '
                        f'err={_lateral_err:.2f}m corr={_lateral_corr_body:.3f}m/s(body) '
                        f'near={near:.2f}m local_fresh={_corridor_local_fresh_log}'
                    )
                    # [I5/C1] CORRIDOR-BIAS 标准化日志（含职责隔离状态）
                    _corridor_local_fresh_log2 = self.is_cmd_fresh(
                        self.last_local_stamp, self.get_clock().now())
                    # 门模式下始终注入全量纠偏，日志反映实际行为
                    _cb_active = abs(_lateral_corr_body) > 1e-3 and (
                        not _corridor_local_fresh_log2 or _prox_in_door_mode)
                    self.get_logger().info(
                        f'[CORRIDOR-BIAS] {"on" if _cb_active else "off(local_mpc_suppressed)"} '
                        f'value={_lateral_corr_body:.3f}m/s '
                        f'L={_left_d:.2f}m R={_right_d:.2f}m err={_lateral_err:.2f}m '
                        f'local_dominant={_corridor_local_fresh_log2}'
                    )
            elif self.att_rot is not None:
                _body_normals = {
                    "front": np.array([1.,  0., 0.]),
                    "back":  np.array([-1., 0., 0.]),
                    "left":  np.array([0., -1., 0.]),
                    "right": np.array([0.,  1., 0.]),
                }
                # proj_buf：比 stop_dist 多 0.35m 开始做投影限制
                _proj_buf = self.obstacle_cmd_stop_dist + 0.35
                for _sec, _bn in _body_normals.items():
                    _d = self.lidar_ranges.get(_sec)
                    if _d is not None and np.isfinite(_d) and _d < _proj_buf:
                        _wall_n_ned = self.att_rot.apply(_bn)   # 体 FRD → NED [N, E, D]
                        # ★ 修复: out 是 ENU，_wall_n_ned 是 NED，帧不一致无法直接 dot
                        # NED→ENU: ENU_x=NED_y, ENU_y=NED_x, ENU_z=-NED_z
                        _wall_n_enu = np.array([_wall_n_ned[1], _wall_n_ned[0], -_wall_n_ned[2]])
                        _approach = float(np.dot(out[:3], _wall_n_enu))  # ENU·ENU ✓
                        if _approach > 0:            # 正在朝此障碍物运动
                            out[:3] -= _approach * _wall_n_enu  # 投影消去趋壁分量（ENU）
            # 统一减速（对剩余可行速度按距离比例降速）
            out[0] *= scale
            out[1] *= scale
        return out

    def _apply_safety_clamp(self, vel: np.ndarray, now, now_sec: float) -> np.ndarray:
        """
        统一安全限幅（Phase5）：在最终发布前执行，不可绕过。
        功能：
          1) 最近激光测距 < obstacle_cmd_stop_dist → 强制 apply_proximity_constraints
          2) 贴墙检测：连续 wall_stick_hold_sec 内位置无变化且有水平速度指令 → [WALL-STICK] 告警 + 后退
          3) 贴墙回退完成后解锁正常指令
        """
        out = vel.copy()
        import time as _sc_mt
        _now_m = _sc_mt.monotonic()

        # ── 1) 近障强制避障 ────────────────────────────────────────────────
        _near = self._nearest_lidar_distance()
        if _near is not None and _near <= self.obstacle_cmd_stop_dist:
            out = self.apply_proximity_constraints(out)

        # 仅对速度控制模式运行 wall-stick 检测（HOVER_SCAN 用 position setpoint 无需）
        # ★ 目标接近模式：跳过 wall-stick（视觉伺服期间 XY 位移小是正常的）
        if self.explore_state == "HOVER_SCAN" or self.target_confirmed:
            self._wall_stick_active = False
            self._wall_stick_since = 0.0
            return out

        # ── 2) 贴墙检测 ────────────────────────────────────────────────────
        _has_xy_cmd = float(np.linalg.norm(out[:2])) > 0.05
        _pos_x = float(self.pose.x) if self.pose is not None else None
        _pos_y = float(self.pose.y) if self.pose is not None else None

        if _has_xy_cmd and _pos_x is not None and _near is not None and _near < self._wall_stick_detect_dist_m:
            if self._wall_stick_pos is None:
                self._wall_stick_pos = (_pos_x, _pos_y)
                self._wall_stick_since = _now_m
            else:
                _moved = float(np.hypot(_pos_x - self._wall_stick_pos[0],
                                        _pos_y - self._wall_stick_pos[1]))
                _stuck_sec = _now_m - self._wall_stick_since
                if _moved < self._wall_stick_min_move_m and _stuck_sec >= self._wall_stick_hold_sec:
                    if not self._wall_stick_active:
                        self._wall_stick_active = True
                        self._wall_stick_recover_dir = -out[:2] / (np.linalg.norm(out[:2]) + 1e-9)
                        _fmt_lr = lambda v: self.lidar_ranges.get(v)
                        self.get_logger().warn(
                            f"[WALL-STICK] 检测到贴墙！near={_near:.2f}m "
                            f"pos=({_pos_x:.2f},{_pos_y:.2f}) 无位移{_stuck_sec:.1f}s "
                            f"vel=({out[0]:.2f},{out[1]:.2f}) "
                            f"F={self.lidar_ranges.get('front')} L={self.lidar_ranges.get('left')} "
                            f"R={self.lidar_ranges.get('right')} B={self.lidar_ranges.get('back')} "
                            f"→ 反向逃脱 dir=({self._wall_stick_recover_dir[0]:.2f},{self._wall_stick_recover_dir[1]:.2f})"
                        )
        else:
            if self._wall_stick_active:
                self.get_logger().info("[WALL-STICK-RECOVER] 贴墙状态解除，恢复正常控制")
            self._wall_stick_active = False
            self._wall_stick_pos = None
            self._wall_stick_since = 0.0
            self._wall_stick_recover_dir = None

        # ── 3) 贴墙回退 ────────────────────────────────────────────────────
        if self._wall_stick_active and self._wall_stick_recover_dir is not None:
            _esc_speed = min(float(self.obstacle_escape_speed), 0.4)
            out[0] = float(self._wall_stick_recover_dir[0]) * _esc_speed
            out[1] = float(self._wall_stick_recover_dir[1]) * _esc_speed

        # ── 4) 4.3 CORNER-TRAP：前方贴墙 + 侧向也封闭，角落拘禁检测 ─────────
        import time as _ct_t
        _ct_now = _ct_t.monotonic()
        _front  = self.lidar_ranges.get('front')
        _left   = self.lidar_ranges.get('left')
        _right  = self.lidar_ranges.get('right')
        _in_corner = (
            _front is not None and _front < self._corner_front_th_m
            and (
                (_left  is not None and _left  < self._corner_side_th_m)
                or (_right is not None and _right < self._corner_side_th_m)
            )
            and _has_xy_cmd
        )
        if _in_corner or _ct_now < self._corner_trap_until:
            if _ct_now >= self._corner_trap_until:  # 新触发
                _fwd   = out[:2].copy()
                _fwd_n = np.linalg.norm(_fwd)
                _esc   = -_fwd / (_fwd_n + 1e-9)
                if _left is not None and _left < self._corner_side_th_m:
                    _side = np.array([_esc[1], -_esc[0]])   # 旋90°右开阔
                else:
                    _side = np.array([-_esc[1], _esc[0]])   # 旋90°左开阔
                _blend = _esc + 0.5 * _side
                self._corner_trap_escape_dir = _blend / (np.linalg.norm(_blend) + 1e-9)
                self._corner_trap_until           = _ct_now + self._corner_block_sec
                self._corner_trap_door_block_until = _ct_now + self._corner_block_sec * 2.0
                self._corner_trap_trigger_count = getattr(self, '_corner_trap_trigger_count', 0) + 1
                # ★ 穿门模式保护：目标类型为门口时，第1次触发仅本地逃逸，不清除全局路径
                # 理由：窄门穿越一侧近墙是正常现象，不应立刻重规划
                _is_door_mode = getattr(self, 'ssg_goal_type', 'none') in (
                    'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                    'frontier_gap_post', 'confirmed_door')
                _door_first_trigger = _is_door_mode and self._corner_trap_trigger_count <= 2
                if not _door_first_trigger:
                    # 普通模式或多次触发：清除全局目标，触发重规划
                    self.ssg_viewpoint_goal = None
                    self.astar_path = None
                if _door_first_trigger:
                    self.get_logger().warn(
                        f"[DOOR-THREADING] 穿门近墙触发（第{self._corner_trap_trigger_count}次）"
                        f" front={_front:.2f}m L={_left} R={_right} "
                        f"→ 局部逃逸但保留全局目标 goal_type={getattr(self,'ssg_goal_type','?')}"
                    )
                    self._log_event("DOOR-THREADING",
                        f"trigger={self._corner_trap_trigger_count} front={_front:.2f}m "
                        f"goal_preserved=True esc=({self._corner_trap_escape_dir[0]:.2f},"
                        f"{self._corner_trap_escape_dir[1]:.2f})")
                else:
                    self._log_event(
                        "CORNER-TRAP",
                        f"front={_front:.2f}m left={_left}m right={_right}m "
                        f"vel=({out[0]:.2f},{out[1]:.2f}) "
                        f"esc_dir=({self._corner_trap_escape_dir[0]:.2f},"
                        f"{self._corner_trap_escape_dir[1]:.2f}) "
                        f"block={self._corner_block_sec:.1f}s door_block={self._corner_block_sec*2:.0f}s"
                    )
                    self.get_logger().warn(
                        f"[CORNER-TRAP] 角落拘禁! front={_front:.2f}m L={_left} R={_right} "
                        f"esc=({self._corner_trap_escape_dir[0]:.2f},"
                        f"{self._corner_trap_escape_dir[1]:.2f}) "
                        f"封锁门导航{self._corner_block_sec*2:.0f}s"
                    )
            # 逃出期间强制退出方向
            if self._corner_trap_escape_dir is not None:
                _esc_spd = min(float(self.obstacle_escape_speed), 0.45)
                out[0] = float(self._corner_trap_escape_dir[0]) * _esc_spd
                out[1] = float(self._corner_trap_escape_dir[1]) * _esc_spd
        else:
            if self._corner_trap_until > 0 and _ct_now >= self._corner_trap_until:
                self._corner_trap_escape_dir = None  # 解除逃出状态
                self._corner_trap_trigger_count = 0   # ★ 解除时重置计数器

        return out

    def clip_bounds(self, vel: np.ndarray) -> np.ndarray:
        if self.pose is None:
            return vel
        out = vel.copy()
        # ★ 使用 SLAM map frame 位置（_map_px/_map_py）对比 SLAM 地图边界（self.bounds）
        # 原来用 pose.x/pose.y（PX4 NED odom）vs self.bounds（SLAM map frame）会在 loop closure
        # 后产生坐标系偏差，导致边界剥裁失效。
        x = float(getattr(self, '_map_px', self.pose.x))
        y = float(getattr(self, '_map_py', self.pose.y))
        margin = 0.35
        if x > self.bounds[0, 1] - margin and out[0] > 0:
            out[0] = 0.0
        if x < self.bounds[0, 0] + margin and out[0] < 0:
            out[0] = 0.0
        if y > self.bounds[1, 1] - margin and out[1] > 0:
            out[1] = 0.0
        if y < self.bounds[1, 0] + margin and out[1] < 0:
            out[1] = 0.0
        return out

    def apply_smoothing(self, vel: np.ndarray, now) -> np.ndarray:
        dt = max((now - self.last_loop_time).nanoseconds * 1e-9, 1e-3)
        dv = vel - self.prev_vel
        max_dv = self.max_acc * dt
        dv = np.clip(dv, -max_dv, max_dv)
        out = self.prev_vel + dv
        spd = float(np.linalg.norm(out[:2]))
        if spd > self.max_speed:
            out[:2] = out[:2] / spd * self.max_speed
        out[2] = float(np.clip(out[2], -self.alt_max_climb_speed, self.alt_max_descend_speed))
        return out

    def publish_setpoint(self, vel: np.ndarray, now_us: int, yaw: float = 0.0, yawspeed: float = 0.0) -> None:
        # ★ NaN/inf 有效性保护：防止向 PX4 发送无效 setpoint 触发 "Invalid offboard setpoints"
        _vel_clean = np.array([float(vel[0]), float(vel[1]), float(vel[2])], dtype=float)
        if not np.all(np.isfinite(_vel_clean)):
            self.get_logger().error(
                f"[SETPOINT-INVALID] reason=nan_or_inf vel={_vel_clean} "
                f"→ 替换为零速度悬停，切 SAFE_HOLD"
            )
            _vel_clean = np.zeros(3, dtype=float)
            self.state = "SAFE_HOLD"
        _yaw_clean = float(yaw)  # ★[B] 保留 NaN：NaN 告知 PX4 使用 yawspeed 控制（不锁定航向角）
        # 注意：仅在 yaw 为有限值时才做范围截断；NaN 让 PX4 进入纯偏航速率控制模式
        _yawspeed_clean = float(yawspeed) if np.isfinite(float(yawspeed)) else 0.0
        ts = TrajectorySetpoint()
        ts.timestamp = now_us
        ts.position = [float("nan"), float("nan"), float("nan")]
        ts.acceleration = [float("nan"), float("nan"), float("nan")]
        # vel in ENU [East, North, NED_z]; PX4 TrajectorySetpoint 需要 NED [North, East, NED_z]
        _px4_vel = [float(_vel_clean[1]), float(_vel_clean[0]), float(_vel_clean[2])]  # ENU→NED
        ts.velocity = _px4_vel
        ts.yaw = _yaw_clean
        ts.yawspeed = _yawspeed_clean
        self.ts_pub.publish(ts)
        # ── 2 [CMD-FRAME]+[CMD-FINAL] 速度帧诊断日志（每100次记录一次）────────────
        self._publish_setpoint_cnt += 1
        if self._publish_setpoint_cnt % 100 == 1:
            _state_str = getattr(self, 'state', 'N/A')
            _exp_str   = getattr(self, 'explore_state', 'N/A')
            _map_pos   = (f"({self._map_px:.2f},{self._map_py:.2f})" if hasattr(self,'_map_px') else '?')
            self.get_logger().debug(
                f"[CMD-FRAME] #{self._publish_setpoint_cnt} "
                f"enu_vel=({_vel_clean[0]:.2f},{_vel_clean[1]:.2f},{_vel_clean[2]:.3f}) "
                f"px4_ned=({_px4_vel[0]:.2f},{_px4_vel[1]:.2f},{_px4_vel[2]:.3f}) "
                f"state={_state_str}/{_exp_str} map_pos={_map_pos}"
            )
            self._log_event(
                "CMD-FINAL",
                f"vel_enu=({_vel_clean[0]:.2f},{_vel_clean[1]:.2f},{_vel_clean[2]:.3f}) "
                f"vel_ned=({_px4_vel[0]:.2f},{_px4_vel[1]:.2f},{_px4_vel[2]:.3f}) "
                f"state={_state_str}/{_exp_str}"
            )

        mode = OffboardControlMode()
        mode.timestamp = now_us
        mode.position = False
        mode.velocity = True
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        self.offboard_pub.publish(mode)

    def publish_position_yaw_setpoint(self, hold_x: float, hold_y: float, hold_z: float,
                                       now_us: int, yawspeed: float = 0.0,
                                       enable_spin: bool = True) -> None:
        """位置保持 + 偏航旋转：XYZ 全用 position setpoint，叠加 yawspeed 旋转。
        用于 HOVER_SCAN 阶段：无人机位置锚定在扫描起始点，仅偏航旋转扫描。
        enable_spin=False 时停止旋转（ESCAPE/SAFE_HOLD/近墙阻断期间）。
        """
        ts = TrajectorySetpoint()
        ts.timestamp = now_us
        # hold_x=ENU East, hold_y=ENU North; PX4 需要 NED position [North, East, z]
        ts.position = [float(hold_y), float(hold_x), float(hold_z)]  # ENU→NED
        ts.velocity = [float("nan"), float("nan"), float("nan")]
        ts.acceleration = [float("nan"), float("nan"), float("nan")]
        if enable_spin:
            ts.yaw = float("nan")       # yaw 由 yawspeed 控制
            ts.yawspeed = float(yawspeed)
        else:
            # 止转：锁定到 locked_yaw 或当前 yaw
            _lk = float(getattr(self, 'locked_yaw', None) or
                        (self.att_yaw if self.att_yaw is not None else 0.0))
            ts.yaw = float(_lk)
            ts.yawspeed = 0.0
        # [HSCAN-SPIN] 节流日志 1Hz
        import time as _hsp_t
        _hsp_now = _hsp_t.monotonic()
        if _hsp_now - getattr(self, '_hscan_spin_log_t', 0.0) > 1.0:
            self._hscan_spin_log_t = _hsp_now
            self.get_logger().debug(
                f"[HSCAN-SPIN] enable={enable_spin} "
                f"yaw={ts.yaw:.3f} yawspeed={ts.yawspeed:.3f}"
            )
        self.ts_pub.publish(ts)

        mode = OffboardControlMode()
        mode.timestamp = now_us
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        self.offboard_pub.publish(mode)

    def publish_takeoff_setpoint(self, target_z: float, now_us: int, yaw: float = 0.0) -> None:
        """起飞专用：发送完整 position setpoint（X/Y/Z均有效）。
        PX4 mc_pos_control 要求 position setpoint 中所有分量均不能为 NaN，
        否则报 invalid setpoints 并触发 failsafe。
        X/Y 使用当前位置固定悬停，Z 飞向目标高度。
        """
        # 使用当前位置作为XY保持点（无pose时用0，PX4会在原地起飞）
        hold_x = float(self.pose.x) if self.pose is not None else 0.0
        hold_y = float(self.pose.y) if self.pose is not None else 0.0

        ts = TrajectorySetpoint()
        ts.timestamp = now_us
        # hold_x=ENU East, hold_y=ENU North; PX4 需要 NED [North, East, z]
        ts.position = [hold_y, hold_x, float(target_z)]  # ENU→NED：全部有效，无NaN
        ts.velocity = [float("nan"), float("nan"), float("nan")]
        ts.acceleration = [float("nan"), float("nan"), float("nan")]
        ts.yaw = float(yaw)
        ts.yawspeed = float("nan")
        self.ts_pub.publish(ts)

        mode = OffboardControlMode()
        mode.timestamp = now_us
        mode.position = True   # ← position mode
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        self.offboard_pub.publish(mode)

    def log_status(self, now, vel: np.ndarray, safety_ok: bool = True, safety_reason: str = "ok") -> None:
        if not self.enable_logging or self.log_writer is None or self.pose is None:
            return
        scan_age = np.nan
        if self.last_scan_time is not None:
            scan_age = (now - self.last_scan_time).nanoseconds * 1e-9
        self.log_writer.writerow([
            now.nanoseconds * 1e-9,
            self.state,
            float(self.pose.x), float(self.pose.y), float(self.pose.z),
            float(self.att_yaw) if self.att_yaw is not None else np.nan,
            float(vel[0]), float(vel[1]), float(vel[2]),
            float(np.linalg.norm(vel[:2])),
            self.lidar_ranges["front"] if self.lidar_ranges["front"] is not None else np.nan,
            self.lidar_ranges["left"] if self.lidar_ranges["left"] is not None else np.nan,
            self.lidar_ranges["right"] if self.lidar_ranges["right"] is not None else np.nan,
            self.lidar_ranges["back"] if self.lidar_ranges["back"] is not None else np.nan,
            scan_age,
            int(self.armed),
            int(self.offboard_mode),
            int(self.nav_state),
            int(self.target_found),
            int(self.target_confirmation.hit_streak),
            int(safety_ok),
            safety_reason,
        ])
        if (now - self.last_log_flush).nanoseconds * 1e-9 > self.log_flush_interval:
            if self.log_file is not None:
                self.log_file.flush()
            if self._event_log_file is not None:
                try:
                    self._event_log_file.flush()
                except Exception:
                    pass
            self.last_log_flush = now

    def _allow_slam_integrate_now(self) -> tuple:
        """判断当前帧是否允许 SLAM 建图积分。
        Returns (allow: bool, reason: str).
        False 条件（任意一个触发即返回 False）：
          - HOVER_SCAN_ESCAPE 激活中（运动中，位姿抖动）
          - MAP_RECOVER hard-hold 期间
          - EKF reset / corner_trap 暂停期
          - 近障 < 0.20m（极近墙，数据不可信）—— 仅限非 HOVER_SCAN 状态
          - 姿态倾斜 > 8°
          - 垂直速度 vz > 0.05m/s（垂直运动中）
        ★ 修复：去掉 SAFE_HOLD 封锁——无人机静止时 SLAM 应继续吸收扫描；
           去掉 0.45m 近墙封锁——HOVER_SCAN 本来就是要扫附近的墙，封锁会导致地图永远不更新。
           近墙封锁只在非 HOVER_SCAN 导航运动中生效，且阈值降至 0.20m（机体半径）。
        """
        import time as _asi_t
        _now = _asi_t.monotonic()
        if getattr(self, '_hover_scan_escape_active', False):
            return False, "hover_escape"
        # MAP_RECOVER hard-hold: 大跳变稳定期内禁止积分
        if (_now < getattr(self, '_map_recover_hard_until', 0.0)):
            return False, "map_recover_hard"
        if _now < getattr(self, '_map_pause_until', 0.0):
            return False, "map_pause"
        if _now < getattr(self, '_corner_trap_until', 0.0):
            return False, "corner_trap"
        # 近墙封锁：仅限导航运动中（非 HOVER_SCAN），且阈值降至 0.20m
        _in_hover_scan = (getattr(self, 'explore_state', '') == 'HOVER_SCAN')
        if not _in_hover_scan:
            _near = self._nearest_lidar_distance()
            if _near is not None and _near < 0.20:
                return False, f"near_wall_nav({_near:.2f}m<0.20)"
        if ((self.att_roll is not None and abs(self.att_roll) > np.deg2rad(8.0))
                or (self.att_pitch is not None and abs(self.att_pitch) > np.deg2rad(8.0))):
            _r = float(np.rad2deg(self.att_roll or 0))
            _p = float(np.rad2deg(self.att_pitch or 0))
            return False, f"att_tilt(r={_r:.1f}p={_p:.1f})"
        # ── 平面位姿无效（tilt > 20°）：禹如机体已倚坡/碰墙，不得积分 ───────────
        if not getattr(self, '_planar_pose_valid', True):
            return False, f"planar_invalid({getattr(self,'_planar_pose_invalid_reason','?')})"
        # ── map pose 完全无效（收退到 px4_fallback 或 invalid ）：禁止积分 ───
        _mp_src = getattr(self, '_map_pose_source', '')
        if _mp_src in ('px4_fallback', 'invalid'):
            return False, f"map_src_unreliable({_mp_src})"
        if self.pose is not None and abs(float(self.pose.vz)) > 0.05:
            return False, f"vz_fast({float(self.pose.vz):.3f}m/s)"
        return True, "ok"

    def command_loop(self) -> None:
        # ★[BUG-FIX] 顶层异常保护: 任何未捕获异常都应记录并继续运行，
        # 而不是让节点崩溃导致 PX4 失去 offboard setpoint 触发意外降落
        try:
            self._command_loop_impl()
        except Exception as _cl_exc:
            import traceback
            _tb = traceback.format_exc()
            _exc_key = type(_cl_exc).__name__
            # 节流：同类异常每5s只打一次
            _now_cl = __import__('time').monotonic()
            _last_t = getattr(self, f'_cl_exc_log_{_exc_key}', 0.0)
            if _now_cl - _last_t > 5.0:
                setattr(self, f'_cl_exc_log_{_exc_key}', _now_cl)
                self.get_logger().error(
                    f"[CMD-LOOP-EXC] 未捕获异常 ({_exc_key})，节点继续运行: {_cl_exc}\n{_tb}"
                )
                self._log_event("CMD-LOOP-EXC", f"exc={_exc_key} msg={_cl_exc}")

    def _command_loop_impl(self) -> None:
        now = self.get_clock().now()
        now_us = int(now.nanoseconds * 1e-3)

        self.offboard_setpoint_counter += 1

        # ── 广播当前 FSM 状态（local_mpc_node / apf / global_planner 据此决定是否停发指令）──
        # 同时追加 PLANAR_INVALID 标签（tilt > 20°），让下游节点立即抑制输出
        _fsm_pub_msg = String()
        _fsm_tag = str(self.state)
        if not getattr(self, '_planar_pose_valid', True):
            _fsm_tag = f"{_fsm_tag}/PLANAR_INVALID"
        _fsm_pub_msg.data = _fsm_tag
        self._fsm_state_pub.publish(_fsm_pub_msg)
        # \u2605 \u5e7f\u64ad\u5f53\u524d\u5bfc\u822a\u76ee\u6807\u7c7b\u578b\uff08APF \u7528\u4e8e\u5224\u65ad\u662f\u5426\u8fdb\u5165\u95e8\u53e3\u9001\u5f3a\u6a21\u5f0f\uff09
        _gt_msg = String(); _gt_msg.data = str(getattr(self, 'ssg_goal_type', 'none'))
        self._goal_type_pub.publish(_gt_msg)

        # ── startup 检测：首帧发出就绪信号（让启动脚本的 wait_for_string_in_log 通过）──
        if not self._cmd_loop_ready:
            self._cmd_loop_ready = True
            self.get_logger().info(
                "[CMD-LOOP-READY] command_loop 已启动，offboard 心跳开始发布 (20 Hz)"
            )

        # 每帧更新 map frame 位姿缓存（供所有栅格操作使用，避免多处重复 TF 查询）
        # 同时更新 map pose 质量字段（valid/stale/source/age）
        import time as _mp_ck_t
        if self.pose is not None or self._odom_received:
            _mp_result = self._get_map_pose()
            self._map_px          = _mp_result[0]
            self._map_py          = _mp_result[1]
            self._map_yaw         = _mp_result[2]
            self._map_pose_valid  = bool(_mp_result[3])
            self._map_pose_stale  = bool(_mp_result[4])
            self._map_pose_source = str(_mp_result[5])
            self._map_pose_age_sec = float(
                _mp_ck_t.monotonic() - getattr(self, '_last_map_pose_mono', 0.0)
            )
            # ★ plan frame 位姿变换（考虑 OccupancyGrid origin.orientation 旋转补偿）
            # 典型情况：slam_toolbox 输出零旋转 → _plan_px = _map_px（恒等变换）
            # 语义约定：访问 self.grid 的所有操作必须使用 _plan_px/_plan_py/_plan_yaw
            _plan_r = self._map_pose_to_plan_pose(
                self._map_px, self._map_py, self._map_yaw)
            self._plan_px  = _plan_r[0]
            self._plan_py  = _plan_r[1]
            self._plan_yaw = _plan_r[2]

        if not self.pose_initialized or self.pose is None:
            # 位姿未初始化时也发 position setpoint（而非velocity）
            # 这样PX4切入offboard模式后可以立即接受位置指令而无需重新切换控制模式
            self.publish_takeoff_setpoint(self.alt_target_z, now_us)
            self.save_map_outputs(now)
            self.last_loop_time = now
            return

        # ── 安全检查（先于建图）────────────────────────────────────────────────
        # 必须先做安全判定，才能决定是否允许积分扫描数据
        pose_age_limit = self.safety_max_pose_age_sec
        if self.pose_dt_ema_sec is not None and np.isfinite(self.pose_dt_ema_sec):
            adaptive_limit = float(self.pose_dt_ema_sec) * max(1.5, self.safety_pose_age_adaptive_mult)
            pose_age_limit = max(self.safety_max_pose_age_sec, adaptive_limit)
            pose_age_limit = min(pose_age_limit, self.safety_pose_age_ceiling_sec)

        safety_state = self.safety_monitor.evaluate(
            self,
            now,
            enforce_altitude_bounds=self.takeoff_done,
            pose_age_limit_sec=pose_age_limit,
        )

        # ── 在线建图 + 房间/门分割（_allow_slam_integrate_now 门控）─────────────
        # 调用正式门控方法（含 SAFE_HOLD / escape / recover / near / att / vz 检查）
        import time as _si_t
        _si_now = _si_t.monotonic()
        _allow_integrate, _si_reason = self._allow_slam_integrate_now()
        if _allow_integrate:
            self.update_map_from_scan(now)
            self.update_room_door_segmentation(now)
        else:
            if _si_now - getattr(self, '_scan_integrate_skip_log_t', 0.0) > 2.0:
                self._scan_integrate_skip_log_t = _si_now
                self.get_logger().info(f"[SCAN-INTEGRATE-SKIP] reason={_si_reason}")
        self.save_map_outputs(now)
        if not safety_state.safe:
            # HOVER_SCAN 悬停扫描阶段：obstacle_too_close 不中断（无人机静止，障碍物仅反射噪声）
            # 仅允许 z_too_high/z_too_low/pose_stale/offboard_lost 中断
            if (self.takeoff_done
                    and getattr(self, "explore_state", None) == "HOVER_SCAN"
                    and safety_state.reason.startswith("obstacle_too_close")):
                # ★ 不再豁免 obstacle_too_close！必须进入 ESCAPE，不能继续旋转扫描
                import time as _hsb_t
                _near_hsb = self._nearest_lidar_distance()
                _near_str_hsb = f"{_near_hsb:.2f}" if _near_hsb is not None else "None"
                self.get_logger().warn(
                    f"[HSCAN-BLOCK] reason=obstacle_too_close near={_near_str_hsb}m "
                    f"state={self.explore_state} → 预触发 HOVER_SCAN_ESCAPE"
                )
                self._log_event("HSCAN-BLOCK", f"reason=obstacle_too_close near={_near_str_hsb}")
                # 预触发 ESCAPE（若不在 escape 中），确保主逻辑执行 escape 速度而非 yawspeed
                if not self._hover_scan_escape_active:
                    _esc_v = self.apply_proximity_constraints(np.zeros(3, dtype=float))
                    _en_hsb = float(np.linalg.norm(_esc_v[:2]))
                    if _en_hsb > 0.01:
                        self._hover_scan_escape_dir = _esc_v[:2] / _en_hsb
                    else:
                        _yw_hsb = float(self.att_yaw) if self.att_yaw is not None else 0.0
                        self._hover_scan_escape_dir = np.array([-np.cos(_yw_hsb), -np.sin(_yw_hsb)])
                    self._hover_scan_escape_active = True
                    self._hover_scan_escape_start = _hsb_t.monotonic()
                    self._hover_scan_escape_clear_since = 0.0
                    _estr_hsb = (f"({self._hover_scan_escape_dir[0]:.2f},"
                                 f"{self._hover_scan_escape_dir[1]:.2f})")
                    self.get_logger().warn(f"[HSCAN-ESC-ENTER] near={_near_str_hsb}m dir={_estr_hsb}")
                    self._log_event("HSCAN-ESC-ENTER", f"near={_near_str_hsb} dir={_estr_hsb}")
                pass  # 继续进入 HOVER_SCAN 主逻辑（escape 分支处理 escape 速度）
            else:
                # 起飞阶段：只有障碍物过近或高度越界才阻断，其他（pose_stale/scan_stale/offboard）允许
                is_takeoff_phase = not self.takeoff_done
                if is_takeoff_phase:
                    # 起飞时允许通过的情况：pose_stale / scan_stale / offboard_lost / offboard_transition
                    bypass_reasons = ("offboard_lost", "offboard_transition", "pose_stale",
                                      "pose_missing", "pose_stamp_missing", "scan_stale")
                    if any(safety_state.reason.startswith(r) for r in bypass_reasons):
                        pass  # fall through to normal takeoff logic below
                        # 但如果高度越界仍需阻断（z_too_high_alt / z_too_low_alt 由 enforce_altitude_bounds 管）
                    else:
                        # obstacle_too_close 等仍然阻断起飞
                        if (now - self.last_safety_warn).nanoseconds * 1e-9 > 1.0:
                            self.get_logger().warn(f"[SAFETY] {safety_state.reason}, action={safety_state.action}")
                            self.last_safety_warn = now
                        # 起飞阶段安全阻断：继续发 position setpoint 保持PX4 offboard控制模式活跃
                        # 不能发velocity zeros，否则PX4可能退出offboard
                        cur_yaw = float(self.att_yaw) if self.att_yaw is not None else 0.0
                        self.publish_takeoff_setpoint(self.alt_target_z, now_us, yaw=cur_yaw)
                        vel_cmd = np.zeros(3, dtype=float)
                        self.log_status(now, vel_cmd, safety_ok=False, safety_reason=safety_state.reason)
                        self.prev_vel = vel_cmd
                        self.last_loop_time = now
                        self.state = "SAFE_HOLD"
                        return
                else:
                    if (now - self.last_safety_warn).nanoseconds * 1e-9 > 1.0:
                        self.get_logger().warn(f"[SAFETY] {safety_state.reason}, action={safety_state.action}")
                        self.last_safety_warn = now
                    # ── offboard_lost: 自动发 DO_SET_MODE 切回 offboard 模式 ──────────
                    if safety_state.reason == "offboard_lost":
                        import time as _ort
                        _now_m = _ort.monotonic()
                        _max_tries = 6            # 最多尝试 6 次
                        _interval  = 5.0          # 每 5s 尝试一次
                        if (self._offboard_recovery_count < _max_tries
                                and _now_m - self._last_offboard_recovery_mono > _interval):
                            self._last_offboard_recovery_mono = _now_m
                            self._offboard_recovery_count += 1
                            _vc = VehicleCommand()
                            _vc.timestamp = now_us
                            _vc.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
                            _vc.param1 = 1.0   # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                            _vc.param2 = 6.0   # PX4_CUSTOM_MAIN_MODE_OFFBOARD
                            _vc.target_system = 1
                            _vc.target_component = 1
                            _vc.source_system = 255
                            _vc.source_component = 0
                            _vc.from_external = True
                            self.vehicle_cmd_pub.publish(_vc)
                            self.get_logger().warn(
                                f"[OFFBOARD-RECOVER] 发送切回 offboard 命令 "
                                f"(#{self._offboard_recovery_count}/{_max_tries}), "
                                f"间隔={_interval:.0f}s  — 若连续失败请手动执行: "
                                f"commander mode offboard"
                            )
                        elif (self._offboard_recovery_count >= _max_tries
                              and _now_m - self._last_offboard_recovery_mono > 60.0):
                            # 6 次均失败后静默 60s 再重置，防止刷屏
                            self._offboard_recovery_count = 0
                    sv = SafetyMonitor.safety_velocity(
                        action=safety_state.action,
                        climb_speed=min(self.alt_max_climb_speed, 0.20),
                        descend_speed=min(self.alt_max_descend_speed, 0.20),
                    )
                    if safety_state.action == "hold":
                        if safety_state.reason.startswith("obstacle_too_close"):
                            # 障碍物过近：通过 apply_proximity_constraints 产生逃脱速度，避免零速贴墙
                            escape_vel = self.apply_proximity_constraints(np.zeros(3, dtype=float))
                            vel_cmd = self.apply_smoothing(escape_vel, now)
                        else:
                            # pose_stale/scan_stale：XY停止，但保持高度（PD控制 z → alt_target_z）
                            # 避免 vz=0 导致 PX4 在 offboard 模式下无人机持续下沉
                            if self.pose is not None and self.takeoff_done:
                                z_hold = float(self.pose.z)
                                z_err_hold = self.alt_target_z - z_hold
                                vz_hold = float(np.clip(
                                    self.alt_kp * z_err_hold - self.alt_kd * self.prev_vel[2],
                                    -self.alt_max_climb_speed, self.alt_max_descend_speed
                                ))
                                vel_cmd = np.array([0.0, 0.0, vz_hold], dtype=float)
                            else:
                                vel_cmd = np.zeros(3, dtype=float)
                    else:
                        vel_cmd = self.apply_smoothing(np.array([sv[0], sv[1], sv[2]], dtype=float), now)
                    self.publish_setpoint(vel_cmd, now_us)
                    self.log_status(now, vel_cmd, safety_ok=False, safety_reason=safety_state.reason)
                    self.prev_vel = vel_cmd
                    self.last_loop_time = now
                    self.state = "SAFE_HOLD"
                    return

        z = float(self.pose.z)
        z_err = self.alt_target_z - z

        # ── 靠墙硬停保护：方向感知，门口目标仅检查前向堵死 ──────────────────────
        # 例外：HOVER_SCAN_ESCAPE（正在逃离），SAFE_HOLD（已经悬停），起飞阶段
        # ★ 门口目标（frontier_gap_probe/inner/confirmed_door）：
        #   - 侧方贴门框（left/right < 0.35m）是正常穿门现象，不触发硬停
        #   - 仅当正前方 < 0.18m 或全向最近 < 0.13m 才硬停（避免撞墙）
        #   - 持久锁缩短为 1s（避免门口短暂碰触导致 5s 停顿）
        # ★ 普通目标：保持原 0.35m 全向阈值 + 5s 持久锁
        import time as _hw_t
        _hw_near = self._nearest_lidar_distance()
        _hw_is_door_goal = (
            getattr(self, 'ssg_goal_type', 'none') in (
                'frontier_gap_probe', 'frontier_gap_pre', 'frontier_gap_center',
                'frontier_gap_post', 'confirmed_door')
            # ★ door_transit_mode 几何触发：路径上事实在穿门（即使 goal_type 不是门类型）
            or getattr(self, '_door_transit_mode', False)
        )
        # ★ PASSAGE_COMMIT：强制过门模式激活时缩短 NEAR-WALL-HARDSTOP 锁定时间
        _pc_active_hw = (
            getattr(self, '_passage_commit_active', False)
            and _hw_t.monotonic() < getattr(self, '_passage_commit_until', 0.0)
        )
        _hw_front = self.lidar_ranges.get("front")
        _hw_front_v = float(_hw_front) if (_hw_front is not None and np.isfinite(_hw_front)) else 9.0
        if _hw_is_door_goal:
            # PASSAGE_COMMIT激活：极短锁 0.5s；普通门目标：1.0s（原 2s）
            _hw_lock_secs = 0.5 if _pc_active_hw else 1.0
            # 门口模式：仅前方堵死才硬停
            _hw_threshold_ok = (
                (_hw_front_v < 0.18)                          # 前方极近
                or (_hw_near is not None and _hw_near < 0.13) # 全向极近（撞上了）
            )
        else:
            # ★ 速度方向感知版近墙保护（非门口目标）：
            # 在 NAV_TO_VIEWPOINT 时仅在当前速度朝向最近障碍物时才硬停
            # 允许无人机沿墙平行导航（侧向近墙但速度切向 → 不硬停）
            _hw_vel_mag = float(np.linalg.norm(self.prev_vel[:2]))
            _hw_nav_active = (
                getattr(self, 'explore_state', '') == 'NAV_TO_VIEWPOINT'
                and _hw_near is not None and _hw_near < 0.35
                and _hw_vel_mag > 0.06
                and self.att_yaw is not None
            )
            if _hw_nav_active:
                # 最近扇区 → ENU 方向向量（检查速度是否朝向该扇区）
                _hw_yaw = float(self.att_yaw)
                _hw_sec_vecs = {
                    'front': np.array([ np.cos(_hw_yaw),  np.sin(_hw_yaw)]),
                    'back':  np.array([-np.cos(_hw_yaw), -np.sin(_hw_yaw)]),
                    'left':  np.array([-np.sin(_hw_yaw),  np.cos(_hw_yaw)]),
                    'right': np.array([ np.sin(_hw_yaw), -np.cos(_hw_yaw)]),
                }
                _hw_near_sec, _hw_near_sec_d = None, 9.0
                for _sk, _sv in _hw_sec_vecs.items():
                    _sd = self.lidar_ranges.get(_sk)
                    if _sd is not None and np.isfinite(_sd) and _sd < _hw_near_sec_d:
                        _hw_near_sec_d = _sd
                        _hw_near_sec = _sk
                if _hw_near_sec is not None:
                    _hw_obs_dir = _hw_sec_vecs[_hw_near_sec]
                    _hw_vel_dir = self.prev_vel[:2] / _hw_vel_mag
                    _hw_dot = float(np.dot(_hw_vel_dir, _hw_obs_dir))
                    # dot > 0.3：速度有显著朝障碍物分量 → 触发硬停
                    # dot <= 0.3：侧向沿墙 → 跳过硬停（仅减速由 apply_proximity_constraints 处理）
                    _hw_threshold_ok = (_hw_dot > 0.3)
                    _hw_lock_secs = 2.0
                    if not _hw_threshold_ok and _hw_t.monotonic() - getattr(self, '_hw_side_log_t', 0.0) > 2.0:
                        self._hw_side_log_t = _hw_t.monotonic()
                        self.get_logger().info(
                            f"[NEAR-WALL-SIDE] near={_hw_near:.2f}m sec={_hw_near_sec} "
                            f"dot={_hw_dot:.2f}<=0.3 → 侧向近墙，速度切向，跳过硬停"
                        )
                else:
                    _hw_threshold_ok = True
                    # ★[B1] 2.0s（原5s）：NAV导航中未找到最近扇区也不应长时间锁定
                    _hw_lock_secs = 2.0
            else:
                _hw_threshold_ok = (_hw_near is not None and _hw_near < 0.25)  # ★[H3] 0.35→0.25：窄廊(0.28-0.35m)不再每帧触发硬停
                # ★[B1] 2.0s（原5s）：减少在普通近墙情况下的长期SAFE_HOLD，加快导航恢复
                # 原5s锁定导致：门口through慢→被判为"严重失败"→长期悬停→回搜索→重规划→循环
                _hw_lock_secs = 2.0  # ★[B1] was 5s→2s

        # ★[H2] 持久锁已激活时跳过 NEAR-WALL-HARDSTOP：防止锁期间 HARDSTOP 持续扩展
        # 锁期间由 plock_near_wall 逃脱逻辑接管，HARDSTOP 不应干扰
        _hw_plock_active = (_hw_t.monotonic() < getattr(self, '_safe_hold_persistent_until', 0.0))
        # ★[D] POST_DOOR_COMMIT / ROOM_ENTRY_SCAN 进入门后抑制近墙硬停（防止徘徊）
        _pdc_start_t = float(getattr(self, '_post_door_commit_start_t', 0.0))
        _post_door_suppress_sec = float(getattr(self, 'post_door_safehold_suppress_sec', 15.0))
        _in_post_door_win = (
            _pdc_start_t > 0.0
            and (_hw_t.monotonic() - _pdc_start_t) < _post_door_suppress_sec
            and getattr(self, 'explore_state', '') in ('POST_DOOR_COMMIT', 'ROOM_ENTRY_SCAN')
        )
        _hw_active = (
            _hw_threshold_ok
            and self.takeoff_done
            and not getattr(self, '_hover_scan_escape_active', False)
            and self.state not in ("SAFE_HOLD", "TAKEOFF", "TAKEOFF_STABILIZE")
            and not getattr(self, '_approach_exclusive', False)  # ★ 独占控制期间：跳过近墙硬停，让视觉伺服正常工作
            and not _hw_plock_active        # ★[H2] 持久锁已激活：让 plock_escape 运行，不重复延锁
            and not _in_post_door_win       # ★[D] 进门后 15s 内抑制近墙硬停
        )
        if _hw_active:
            _vz_hw = float(np.clip(
                self.alt_kp * z_err - self.alt_kd * self.prev_vel[2],
                -self.alt_max_climb_speed, self.alt_max_descend_speed,
            ))
            _hw_cmd = np.array([0.0, 0.0, _vz_hw], dtype=float)
            if _hw_t.monotonic() - getattr(self, '_hw_log_t', 0.0) > 1.0:
                self._hw_log_t = _hw_t.monotonic()
                _hw_thr_str = "front<0.18m|near<0.13m" if _hw_is_door_goal else "near<0.25m(H3)"
                _hw_near_s = f"{_hw_near:.2f}m" if _hw_near is not None else "None"
                self.get_logger().warn(
                    f"[NEAR-WALL-HARDSTOP] near={_hw_near_s} front={_hw_front_v:.2f}m "
                    f"door_goal={_hw_is_door_goal} thr={_hw_thr_str} "
                    f"state={self.state}/{getattr(self,'explore_state','?')} "
                    f"→ 硬停 XY lock={_hw_lock_secs:.0f}s"
                )
                self._log_event("NEAR-WALL-HARDSTOP",
                                f"near={_hw_near_s} door={_hw_is_door_goal} lock={_hw_lock_secs:.0f}s state={self.state}")
            # 设置持久锁：防止贴墙条件短暂消失后立即重飞
            _plock_hw_until = getattr(self, '_safe_hold_persistent_until', 0.0)
            _hw_now_m = _hw_t.monotonic()
            if _hw_now_m + _hw_lock_secs > _plock_hw_until:
                # ★ Fix-B3-A: 新开一轮锁时记录开始时间，供 B3 计算真实持锁时长
                if _hw_now_m >= _plock_hw_until:
                    self._safe_hold_lock_start_t = _hw_now_m
                self._safe_hold_persistent_until = _hw_now_m + _hw_lock_secs
                self._safe_hold_reason = f"near_wall:{_hw_near:.2f}m"
            # ★[G2] SAFE-HOLD-ENTER log（首次进入时打印）
            _hw_prev_state = getattr(self, '_hw_prev_safe_hold', False)
            if not _hw_prev_state:
                _hw_near_s2 = f"{_hw_near:.2f}m" if _hw_near is not None else "None"
                self.get_logger().warn(
                    f"[SAFE-HOLD-ENTER] reason=near_wall near={_hw_near_s2} "
                    f"door_goal={_hw_is_door_goal} lock={_hw_lock_secs:.1f}s "
                    f"state={self.state}/{getattr(self,'explore_state','?')}"
                )
            self._hw_prev_safe_hold = True
            self.publish_setpoint(_hw_cmd, now_us)
            self.log_status(now, _hw_cmd, safety_ok=False, safety_reason=f"near_wall:{_hw_near:.2f}m")
            self.prev_vel = _hw_cmd
            self.last_loop_time = now
            self.state = "SAFE_HOLD"
            return
        else:
            # ★[G2] SAFE-HOLD-EXIT log（离开近墙SAFE_HOLD时打印）
            if getattr(self, '_hw_prev_safe_hold', False):
                self.get_logger().info(
                    f"[SAFE-HOLD-EXIT] reason=near_wall_cleared "
                    f"state={self.state}/{getattr(self,'explore_state','?')} → resuming"
                )
            self._hw_prev_safe_hold = False

        if not self.takeoff_done:
            # 起飞阶段：使用 position setpoint 指定目标高度
            # 原因：PX4 velocity-only setpoint 在地面时不会触发电机起飞
            # 必须用 position setpoint（position=True in OffboardControlMode）
            in_band = abs(z_err) <= self.alt_takeoff_band
            if in_band:
                if self.alt_takeoff_hold_start is None:
                    self.alt_takeoff_hold_start = now
                hold = (now - self.alt_takeoff_hold_start).nanoseconds * 1e-9
                if hold >= self.alt_takeoff_hold_time:
                    self.takeoff_done = True
                    self.state = "SEARCH"
                    # 起飞完成后锁定偏航，防止搜索时随机旋转导致地图坐标系叠加偏差
                    if self.att_yaw is not None:
                        self.locked_yaw = float(self.att_yaw)
                        self._log_event(
                            "TAKEOFF-DONE",
                            f"z={z:.3f} locked_yaw={np.rad2deg(self.locked_yaw):.1f}°"
                        )
                    else:
                        self.locked_yaw = 0.0
                        self._log_event("TAKEOFF-DONE", f"z={z:.3f} locked_yaw=0°(default)")
            else:
                self.alt_takeoff_hold_start = None

            # 发送 position setpoint（目标高度 alt_target_z）
            cur_yaw = float(self.att_yaw) if self.att_yaw is not None else 0.0
            self.publish_takeoff_setpoint(self.alt_target_z, now_us, yaw=cur_yaw)
            # log记录使用等效vz供分析（NaN-safe）
            vz_equiv = float(np.clip(self.alt_kp * z_err, -self.alt_max_climb_speed, self.alt_max_descend_speed))
            vel_cmd = np.array([0.0, 0.0, vz_equiv], dtype=float)
            self.log_status(now, vel_cmd, safety_ok=True, safety_reason="ok")
            self.prev_vel = np.zeros(3, dtype=float)
            self.last_loop_time = now
            return

        # ── 起飞后稳定化冻结（Task4）────────────────────────────────────────
        # takeoff_done=True 后等待 takeoff_stabilize_sec 秒保持悬停，
        # 让 EKF/SLAM 在平稳状态建立初始地图，然后再开始搜索飞行
        import time as _stabmt
        if not self._takeoff_stab_done:
            if self._takeoff_stab_start is None:
                self._takeoff_stab_start = _stabmt.monotonic()
                self._log_event(
                    "TAKEOFF-STAB-START",
                    f"等待 {self.takeoff_stabilize_sec:.1f}s EKF/SLAM稳定化"
                )
            _stab_elapsed = _stabmt.monotonic() - self._takeoff_stab_start
            if _stab_elapsed < self.takeoff_stabilize_sec:
                self.state = "TAKEOFF_STABILIZE"
                _cur_yaw_s = float(self.att_yaw) if self.att_yaw is not None else 0.0
                if self.pose is not None:
                    self.publish_position_yaw_setpoint(
                        float(self.pose.x), float(self.pose.y),
                        float(self.alt_target_z), now_us, yawspeed=0.0,
                        enable_spin=False  # 起飞稳定化期间不旋转
                    )
                else:
                    self.publish_takeoff_setpoint(
                        self.alt_target_z, now_us, yaw=_cur_yaw_s
                    )
                vel_cmd = np.zeros(3, dtype=float)
                self.log_status(now, vel_cmd, safety_ok=True, safety_reason="ok")
                self.prev_vel = np.zeros(3, dtype=float)
                self.last_loop_time = now
                return
            else:
                self._takeoff_stab_done = True
                self._log_event(
                    "TAKEOFF-STAB-DONE",
                    f"稳定化完成 elapsed={_stab_elapsed:.1f}s → 开始搜索/建图"
                )

        self.state = "SEARCH"

        # ★ 持久锁：SAFE_HOLD / NEAR-WALL / PLANAR-INVALID 触发后保持最短封锁时间，
        # 防止条件短暂消失后下一帧立即恢复飞行导致反复贴墙。
        # 定时器由各触发点设置：self._safe_hold_persistent_until = monotonic() + T
        import time as _plock_t
        _plock_now = _plock_t.monotonic()
        _plock_until = getattr(self, '_safe_hold_persistent_until', 0.0)
        # ★ 目标已确认 或 弱确认（C2）：清除持久锁，让视觉伺服立即接管
        if (self.target_confirmed or getattr(self, '_tgt_weak_confirmed', False)) and _plock_now < _plock_until:
            self._safe_hold_persistent_until = 0.0
            _clr_reason = "target_confirmed" if self.target_confirmed else "weak_confirmed★[C2]"
            self.get_logger().warn(
                f"[SAFE-HOLD-LOCK-CLEAR] {_clr_reason} → 清除持久锁，进入 TARGET_APPROACH"
            )
            _plock_until = 0.0
        if _plock_now < _plock_until:
            # ★[B3] 条件恢复即早退出：若净空已恢复到 safe_hold_release_clearance 以上
            # 且已持锁 ≥1s，允许提前清除 persistent_lock，避免不必要的长期悬停
            _sh_rel_clr = float(getattr(self, 'safe_hold_release_clearance', 0.60))
            # ★ Fix-B3-A: 用真实持锁时长判断 held_enough
            # 原公式 (remaining < _sh_max-1) 当锁时长 < _sh_max 时会立刻为 True 导致锁瞬间释放，
            # 典型场景：planar_invalid 设 1.5s 锁，_sh_max=4.0 → (1.5<3.0)=True → 0ms 后释放
            _sh_lock_start = getattr(self, '_safe_hold_lock_start_t', _plock_until)
            _sh_held_secs = _plock_now - _sh_lock_start
            _sh_held_enough = _sh_held_secs >= 1.0
            _sh_near_now = self._nearest_lidar_distance()
            _sh_clear_ok = (_sh_near_now is None or _sh_near_now >= _sh_rel_clr)
            # ★ Fix-B3-B: planar_invalid 原因时还需确认倾角已恢复（_planar_pose_valid=True）
            # 否则倾角仍 >20° 时释放锁→导航命令继续→飞机持续倾斜乱撞
            _sh_reason = getattr(self, '_safe_hold_reason', '')
            if 'planar_invalid' in _sh_reason:
                _sh_clear_ok = _sh_clear_ok and getattr(self, '_planar_pose_valid', True)
            if _sh_held_enough and _sh_clear_ok:
                _plock_reason_early = getattr(self, '_safe_hold_reason', 'safe_hold_lock')
                _sh_near_str = f"{_sh_near_now:.2f}" if _sh_near_now is not None else "None"
                # 节流：防止每帧都打印（锁刚释放又被重设时会连续触发）
                if _plock_now - getattr(self, '_b3_log_t', 0.0) > 1.0:
                    self._b3_log_t = _plock_now
                    self.get_logger().info(
                        f"[SAFE-HOLD-EXIT] ★[B3] 条件恢复提前释放 "
                        f"near={_sh_near_str}m >= {_sh_rel_clr:.2f}m "
                        f"held={_sh_held_secs:.1f}s remain={_plock_until - _plock_now:.1f}s "
                        f"reason={_plock_reason_early} → resuming navigation"
                    )
                self._safe_hold_persistent_until = 0.0
                self._plock_was_active = False
                _plock_until = 0.0
            if _plock_now < _plock_until:  # 再次检查（可能刚被清除）
                pass  # 继续持锁逻辑
        if _plock_now < _plock_until:
            _plock_remain = _plock_until - _plock_now
            if _plock_now - getattr(self, '_plock_log_t', 0.0) > 3.0:
                self._plock_log_t = _plock_now
                _plock_reason = getattr(self, '_safe_hold_reason', 'safe_hold_lock')
                self.get_logger().warn(
                    f"[SAFE-HOLD-LOCK] 持久锁生效 remain={_plock_remain:.1f}s "
                    f"reason={_plock_reason} → 禁止恢复导航"
                )
            self._plock_was_active = True  # ★ 用于退出检测
            _vz_pl = float(np.clip(
                self.alt_kp * z_err - self.alt_kd * self.prev_vel[2],
                -self.alt_max_climb_speed, self.alt_max_descend_speed,
            ))
            # ★[D] SAFE_HOLD 持久锁期间：若近墙则主动脱离，不再站桩
            _near_pl = self._nearest_lidar_distance()
            _sh_stop_dist_pl = getattr(self, 'stop_dist', 0.35)
            # ★[G2] EKF 大幅跳变紧急期间：禁止 XY 逃离（att_yaw 可能失效，方向随机）
            _plock_ekf_emerg = (_plock_now < getattr(self, '_ekf_emergency_xy_hold_until', 0.0))
            if _near_pl is not None and _near_pl < _sh_stop_dist_pl and not _plock_ekf_emerg:
                # ★[G4] 方向锁定 3s 防震荡：逃离方向一旦确定，3s 内不重算
                _pld_lock_until = getattr(self, '_plock_esc_dir_lock_until', 0.0)
                if _plock_now < _pld_lock_until and getattr(self, '_plock_esc_dir_locked', None) is not None:
                    _esc_dir_pl = self._plock_esc_dir_locked
                else:
                    _esc_dir_pl = self._compute_escape_dir_from_lidar()
                    if _esc_dir_pl is not None:
                        self._plock_esc_dir_locked = _esc_dir_pl
                        self._plock_esc_dir_lock_until = _plock_now + 3.0
                if _esc_dir_pl is not None:
                    _esc_sp_pl = float(getattr(self, 'safe_hold_escape_speed', 0.28))
                    _vel_pl = np.array([_esc_dir_pl[0]*_esc_sp_pl, _esc_dir_pl[1]*_esc_sp_pl, _vz_pl], dtype=float)
                    if _plock_now - getattr(self, '_plock_esc_log_t', 0.0) > 1.5:
                        self._plock_esc_log_t = _plock_now
                        _locked_str = '(locked)' if _plock_now < _pld_lock_until else '(new)'
                        self.get_logger().warn(
                            f"[SAFE-HOLD] mode=escape near={_near_pl:.2f}m "
                            f"v_xy=({_vel_pl[0]:.2f},{_vel_pl[1]:.2f}) reason=plock_near_wall dir={_locked_str}"
                        )
                else:
                    _vel_pl = np.array([0.0, 0.0, _vz_pl], dtype=float)
            else:
                if _plock_ekf_emerg and _near_pl is not None and _near_pl < _sh_stop_dist_pl:
                    if _plock_now - getattr(self, '_plock_ekf_log_t', 0.0) > 2.0:
                        self._plock_ekf_log_t = _plock_now
                        self.get_logger().warn(
                            f"[SAFE-HOLD-EKF-HOLD] EKF 紧急锁定期间禁 XY 逃离 near={_near_pl:.2f}m → 原地高度保持"
                        )
                _vel_pl = np.array([0.0, 0.0, _vz_pl], dtype=float)
            self.publish_setpoint(_vel_pl, now_us)
            self.log_status(now, _vel_pl, safety_ok=False, safety_reason="safe_hold_lock")
            self.prev_vel = _vel_pl
            self.last_loop_time = now
            self.state = "SAFE_HOLD"
            return
        else:
            # ★[G2] SAFE-HOLD-EXIT：持久锁刚刚到期 → 恢复导航
            if getattr(self, '_plock_was_active', False):
                _plock_reason_out = getattr(self, '_safe_hold_reason', 'safe_hold_lock')
                self.get_logger().info(
                    f"[SAFE-HOLD-EXIT] persistent_lock expired "
                    f"reason={_plock_reason_out} "
                    f"state={self.state}/{getattr(self,'explore_state','?')} → resuming navigation"
                )
                self._plock_was_active = False

        # ★ 平面位姿失效硬保护：机体倾角 > 20° 时不得继续正常导航
        # 根据：ATT-FRAME 日志显示 roll -83°~-91°，此时 2D 扇区/map 解释全部失真
        if not getattr(self, '_planar_pose_valid', True) and self.takeoff_done:
            import time as _ppv_t
            _ppv_now = _ppv_t.monotonic()
            _ppv_thr = getattr(self, '_planar_tilt_threshold_deg', 20.0)
            _tilt_cur = getattr(self, '_att_tilt_local', 0.0)
            if _ppv_now - getattr(self, '_planar_invalid_log_t', 0.0) > 1.0:
                self._planar_invalid_log_t = _ppv_now
                self.get_logger().error(
                    f"[PLANAR-INVALID] tilt={_tilt_cur:.1f}deg > {_ppv_thr:.0f}deg "
                    f"→ 强制 SAFE_HOLD：禁止继续导航/建图/扇区解释"
                )
                self._log_event(
                    "PLANAR-INVALID",
                    f"tilt={_tilt_cur:.1f}deg > {_ppv_thr:.0f}deg → SAFE_HOLD"
                )
            # ★[D] 近墙时主动脱离，而非 XY=0
            _vz_ppv = float(np.clip(
                self.alt_kp * (self.alt_target_z - float(self.pose.z if self.pose else 0.0))
                - self.alt_kd * self.prev_vel[2],
                -self.alt_max_climb_speed, self.alt_max_descend_speed,
            ))
            _near_ppv = self._nearest_lidar_distance()
            _sh_stop_dist = getattr(self, 'stop_dist', 0.35)
            # ★[G1] HOVER_SCAN ESC-FAIL 压制期间：PLANAR_INVALID 不发 XY 逃离
            # 原因：此时 att_yaw 可能已被 EKF 跳变污染，escape 方向随机会打墙
            _ppv_in_hscan_suppress = (
                _ppv_now < getattr(self, '_hscan_safe_hold_until', 0.0)
                and getattr(self, 'explore_state', '') == 'HOVER_SCAN'
            )
            # ★[G2] EKF 大幅跳变紧急期间：禁止 XY 逃离
            _ppv_ekf_emerg = (_ppv_now < getattr(self, '_ekf_emergency_xy_hold_until', 0.0))
            if (_near_ppv is not None and _near_ppv < _sh_stop_dist
                    and not _ppv_in_hscan_suppress and not _ppv_ekf_emerg):
                # ★[G4] 方向锁定 3s 防震荡：PLANAR_INVALID 逃离方向一旦确定3s不重算
                _ppv_dir_lock_until = getattr(self, '_ppv_esc_dir_lock_until', 0.0)
                if _ppv_now < _ppv_dir_lock_until and getattr(self, '_ppv_esc_dir_locked', None) is not None:
                    _esc_dir = self._ppv_esc_dir_locked
                else:
                    _esc_dir = self._compute_escape_dir_from_lidar()
                    if _esc_dir is not None:
                        self._ppv_esc_dir_locked = _esc_dir
                        self._ppv_esc_dir_lock_until = _ppv_now + 3.0
                if _esc_dir is not None:
                    _esc_sp = float(getattr(self, 'safe_hold_escape_speed', 0.28))
                    _vel_ppv = np.array([_esc_dir[0] * _esc_sp, _esc_dir[1] * _esc_sp, _vz_ppv], dtype=float)
                    if _ppv_now - getattr(self, '_sh_esc_log_t', 0.0) > 1.5:
                        self._sh_esc_log_t = _ppv_now
                        _ppv_locked_str = '(locked)' if _ppv_now < _ppv_dir_lock_until else '(new)'
                        self.get_logger().warn(
                            f"[SAFE-HOLD] mode=escape near={_near_ppv:.2f}m "
                            f"v_xy=({_vel_ppv[0]:.2f},{_vel_ppv[1]:.2f}) reason=planar_invalid dir={_ppv_locked_str}"
                        )
                else:
                    _vel_ppv = np.array([0.0, 0.0, _vz_ppv], dtype=float)
            else:
                _vel_ppv = np.array([0.0, 0.0, _vz_ppv], dtype=float)
                if _ppv_now - getattr(self, '_sh_hover_log_t', 0.0) > 2.0:
                    self._sh_hover_log_t = _ppv_now
                    _ppv_reason = ('ekf_emerg_hold' if _ppv_ekf_emerg
                                   else ('hscan_suppress' if _ppv_in_hscan_suppress else 'clear'))
                    self.get_logger().debug(
                        f"[SAFE-HOLD] mode=hold_xy near={_near_ppv if _near_ppv else 'N/A'} "
                        f"reason=planar_invalid/{_ppv_reason}"
                    )
            self.publish_setpoint(_vel_ppv, now_us)
            self.log_status(now, _vel_ppv, safety_ok=False, safety_reason=f"planar_invalid(tilt={_tilt_cur:.0f}deg)")
            self.prev_vel = _vel_ppv
            self.last_loop_time = now
            self.state = "SAFE_HOLD"
            # ★[D] 持久锁改为 safe_hold_max_lock_s（原 8s → 3s）
            _sh_max = float(getattr(self, 'safe_hold_max_lock_s', 3.0))
            _ppv_lock_until = getattr(self, '_safe_hold_persistent_until', 0.0)
            if _ppv_now + _sh_max > _ppv_lock_until:
                # ★ Fix-B3-A: 新开一轮锁时记录开始时间，供 B3 计算真实持锁时长
                if _ppv_now >= _ppv_lock_until:
                    self._safe_hold_lock_start_t = _ppv_now
                self._safe_hold_persistent_until = _ppv_now + _sh_max
                self._safe_hold_reason = f"planar_invalid:tilt={_tilt_cur:.0f}deg"
            return


        # ════════════════════════════════════════════════════════════════════
        # ★[C1] weak_confirm 直通 world_establish：弱确认即入目标链，跳过 locked_far/align/forward
        # 根因修复：原代码 _tgt_weak_confirmed 仅用于日志，不触发任何状态转换。
        # 修复：当弱确认触发 + 有世界坐标估计 → 直接进入 world_establish，
        # 无需等待 streak 达到 target_confirmed 门槛（streak 3~5帧可能因遮挡永远达不到）。
        # ════════════════════════════════════════════════════════════════════
        import time as _wce_t
        _wce_now = _wce_t.monotonic()
        _wc_shortcut_ok = (
            getattr(self, '_tgt_weak_confirmed', False)
            and not self._approach_exclusive
            and not self.target_confirmed    # 尚未全确认（全确认走下面的原路径）
            and getattr(self, '_tgt_world_est_xy', None) is not None
            and float(getattr(self, '_tgt_world_est_conf', 0.0)) >= 0.20
        )
        if _wc_shortcut_ok:
            self._approach_exclusive = True
            self._approach_inner_state = 'world_establish'
            self._tgt_world_establish_enter_t = _wce_now
            self._tgt_commit_goal = None
            self._frozen_explore_state = self.explore_state
            self.explore_state = 'TARGET_APPROACH_LOCK'
            self.ssg_viewpoint_goal = None
            self.astar_path = None
            # 弱确认时同样清除持久锁（防止 SAFE_HOLD 阻挡目标链）
            self._safe_hold_persistent_until = 0.0
            _wc_wxy = self._tgt_world_est_xy
            _wc_conf = float(self._tgt_world_est_conf)
            self.get_logger().warn(
                f"[TARGET-WEAK-COMMIT-ENTER] ★[C1] weak_confirmed → world_establish shortcut "
                f"conf={_wc_conf:.2f} "
                f"xy=({_wc_wxy[0]:.2f},{_wc_wxy[1]:.2f}) "
                f"frozen_explore={self._frozen_explore_state}"
            )
            self._log_target_event("TARGET-WORLD-ESTABLISH",
                "trigger=weak_confirm_shortcut",
                world_xy=_wc_wxy, conf=_wc_conf, trigger="weak_confirm_direct")
            self._log_event("TARGET-WEAK-COMMIT-ENTER",
                f"conf={_wc_conf:.2f} "
                f"xy=({_wc_wxy[0]:.2f},{_wc_wxy[1]:.2f})")

        # ════════════════════════════════════════════════════════════════════
        # 任务1：目标确认后立即激活独占控制 —— 不再与探索状态机混用
        # ════════════════════════════════════════════════════════════════════
        if self.target_confirmed and not self._approach_exclusive:
            # 模块C：三级面积门控 —— 确认后还需目标足够大 + 居中才允许进入独占
            _enter_area_ok = (
                self.target_area_ratio >= self.target_approach_enter_area_ratio
            )
            _enter_center_ok = (
                self._approach_center_ok_cnt >= self.target_approach_enter_center_frames
            )
            # 累积居中帧数（仅当检测到目标时更新）
            _dx_now = float(getattr(self, 'target_offset_x', 1.0))
            if abs(_dx_now) < self.target_center_tol_x:
                self._approach_center_ok_cnt += 1
            else:
                self._approach_center_ok_cnt = max(0, self._approach_center_ok_cnt - 1)

            if not (_enter_area_ok and _enter_center_ok):
                # 已确认但面积/居中条件未满足 → 激活 locked_far 预接近
                if not getattr(self, '_locked_far_active', False):
                    self._locked_far_active = True
                    self._approach_exclusive = True
                    self._approach_inner_state = 'locked_far'
                    self._lf_centered_since = None
                    self._tgt_lost_since = None
                    self._tgt_centered_since = None
                    self._hover_confirm_enter_t = None
                    self._centered_logged = False
                    self._close_enough_logged = False
                    self._frozen_explore_state = self.explore_state
                    self.explore_state = 'TARGET_APPROACH_LOCK'
                    self.ssg_viewpoint_goal = None
                    self.astar_path = None
                    self.get_logger().warn(
                        f'[TARGET-LOCKED-FAR-ENTER] ★ confirmed+too_far → locked_far yaw-servo '
                        f'area={self.target_area_ratio:.4f}/{self.target_approach_enter_area_ratio:.4f}'
                    )
                    self._log_event('TARGET-LOCKED-FAR-ENTER',
                        f'area={self.target_area_ratio:.4f}')
                else:
                    if _plock_now - getattr(self, '_enter_gate_log_t', 0.0) > 2.0:
                        self._enter_gate_log_t = _plock_now
                        self.get_logger().info(
                            f'[TARGET-LOCKED-FAR] area={self.target_area_ratio:.4f}/{self.target_approach_enter_area_ratio:.4f} '
                            f'center_cnt={self._approach_center_ok_cnt}/{self.target_approach_enter_center_frames}'
                        )
            else:
                self._locked_far_active = False   # FIX7: clear on upgrade
                self._approach_exclusive = True
                self._approach_inner_state = 'align'   # 先对准 yaw，再前进
                self._approach_enter_t = _plock_now
                self._tgt_lost_since = None
                self._tgt_centered_since = None
                self._hover_confirm_enter_t = None
                self._centered_logged = False
                self._close_enough_logged = False
                self._approach_center_ok_cnt = 0   # 重置计数器
                # 任务6：冻结探索逻辑
                self._frozen_explore_state = self.explore_state
                self.explore_state = "TARGET_APPROACH_LOCK"
                self.ssg_viewpoint_goal = None
                self.astar_path = None
                self.get_logger().warn(
                    f"[TARGET-APPROACH-ENTER] ★ 目标确认+大小满足，独占控制激活 "
                    f"area={self.target_area_ratio:.3f} enter_gate={self.target_approach_enter_area_ratio:.3f} "
                    f"frozen_explore={self._frozen_explore_state}"
                )
                self._log_event("TARGET-APPROACH-ENTER",
                    f"area={self.target_area_ratio:.3f} dx={self.target_offset_x:.3f} "
                    f"dy={self.target_offset_y:.3f}")
                self.get_logger().warn("[TARGET-APPROACH-LOCK] 探索状态机已冻结，目标接近独占控制")
                self._log_event("TARGET-APPROACH-LOCK", "explore_frozen=True")

        # 进入 HOVER_SCAN → 告知 bridge 强制打开门控让 SLAM 获取更多扫描帧
        # 离开 HOVER_SCAN → 恢复正常门控（仅在非独占模式时触发）
        _cur_explore = self.explore_state
        if not self._approach_exclusive and _cur_explore != self._prev_explore_state:
            if _cur_explore == "HOVER_SCAN":
                self._set_slam_gate_mode("hover_strict")
            elif self._prev_explore_state == "HOVER_SCAN":
                self._set_slam_gate_mode("normal")
            self._prev_explore_state = _cur_explore

        if self._approach_exclusive:
            # ════════════════════════════════════════════════════════════════
            # 目标接近独占控制块
            # ════════════════════════════════════════════════════════════════
            import time as _ta_mt
            _ta_now = _ta_mt.monotonic()
            _seen_ago = 99.0
            if self.target_last_seen_time is not None:
                _seen_ago = (now - self.target_last_seen_time).nanoseconds * 1e-9

            # ── 任务3：分级丢失检测 ─────────────────────────────────────────
            _lost_abort = _seen_ago > self.target_track_timeout_abort
            if _lost_abort:
                # ★[A5] 终端任务锁定：即使超时也绝对不回搜索，强制降落
                if getattr(self, 'target_terminal_locked', False):
                    _post_states = ('conservative_descend', 'land', 'commit_hover',
                                    'commit_nav', 'world_establish')
                    if self._approach_inner_state not in _post_states:
                        import time as _fatl_t
                        _fatl_now = _fatl_t.monotonic()
                        # 保留已有 commit_goal，否则用世界坐标估计
                        if getattr(self, '_tgt_commit_goal', None) is None:
                            _wxy_tl = getattr(self, '_tgt_world_est_xy', None)
                            if _wxy_tl is not None:
                                self._tgt_commit_goal = _wxy_tl.copy()
                        self._approach_inner_state = 'conservative_descend'
                        self._tgt_commit_hover_enter_t = _fatl_now
                        self.get_logger().warn(
                            f"[TARGET-ABORT-TERMINAL-LAND] ★[A5] 终端锁定 seen_ago={_seen_ago:.1f}s "
                            f"inner was={self._approach_inner_state} → 强制 conservative_descend，绝不回搜索"
                        )
                        self._log_target_event("TARGET-ABORT-TERMINAL-LAND",
                            f"seen_ago={_seen_ago:.1f}s",
                            world_xy=getattr(self, '_tgt_world_est_xy', None),
                            conf=self._tgt_world_est_conf,
                            trigger="lost_abort_terminal")
                    else:
                        # 已经在后期阶段，继续执行
                        if _ta_now - getattr(self, '_abort_term_log_t', 0.0) > 5.0:
                            self._abort_term_log_t = _ta_now
                            self.get_logger().info(
                                f"[TARGET-ABORT-TERMINAL] inner={self._approach_inner_state} "
                                f"seen_ago={_seen_ago:.1f}s → 继续执行终端链"
                            )
                    # 不执行任何回搜索逻辑，直接继续下方视觉伺服块
                else:
                # ★[A5 NON-TERMINAL] 非终端锁定：超时后回搜索（原有逻辑）
                    # 超过最长超时 → 彻底中止
                    self.get_logger().warn(
                        f"[TARGET-LOST-ABORT] 目标丢失>{self.target_track_timeout_abort:.1f}s → "
                        "完全放弃接近，恢复探索"
                    )
                    self._log_event("TARGET-LOST-ABORT",
                        f"seen_ago={_seen_ago:.1f}s inner_st={self._approach_inner_state}")
                    # 重置所有目标和独占状态
                    self.target_confirmed = False
                    self.target_candidate = False
                    self.target_found = False
                    self._target_cand_frames = 0
                    self._target_conf_frames = 0
                    self._target_cand_center_hist.clear()
                    self._target_confirm_start_t = None
                    self._target_approach_phase = 1
                    self._target_xy_stable_since = None
                    self._target_found_logged = False
                    self._target_land_sent = False
                    self._approach_exclusive = False
                    self._locked_far_active = False
                    self._approach_inner_state = 'align'
                    self._tgt_lost_since = None
                # ★[G4] TARGET_LOST_ABORT（非终端）：清除世界坐标导航目标
                if getattr(self, '_visual_target_goal', None) is not None:
                    _vtg_old = self._visual_target_goal
                    self._visual_target_goal = None
                    self._tgt_world_est_conf = max(0.0, self._tgt_world_est_conf - 0.3)
                    self.get_logger().info(
                        f"[TARGET-WORLD-GOAL-CLEAR] ★[G4] reason=lost_abort "
                        f"old=({_vtg_old[0]:.2f},{_vtg_old[1]:.2f}) conf→{self._tgt_world_est_conf:.2f}"
                    )
                    self._log_event("TARGET-WORLD-GOAL-CLEAR", "reason=lost_abort")
                # 恢复探索状态（任务6：解冻）
                self.explore_state = getattr(self, '_frozen_explore_state', 'HOVER_SCAN')
                self.hover_scan_start_time = None
                self.get_logger().warn(
                    f"[TARGET-APPROACH-EXIT] reason=lost_abort → 探索状态恢复={self.explore_state}"
                )
                self._log_event("TARGET-APPROACH-EXIT", "reason=lost_abort explore_resumed=True")
                self.state = "SEARCH"
                _vz_tl = float(np.clip(
                    self.alt_kp * z_err - self.alt_kd * self.prev_vel[2],
                    -self.alt_max_climb_speed, self.alt_max_descend_speed,
                ))
                _vel_tl = np.array([0.0, 0.0, _vz_tl], dtype=float)
                self.publish_setpoint(_vel_tl, now_us)
                self.log_status(now, _vel_tl, safety_ok=True, safety_reason="ok")
                self.prev_vel = _vel_tl
                self.last_loop_time = now
                return

            # ── 任务2/4：视觉伺服状态机 ────────────────────────────────────
            # FIX8: if still locked_far but area now >= enter_area_ratio, force upgrade
            if (self._approach_inner_state == 'locked_far'
                    and self.target_area_ratio >= self.target_approach_enter_area_ratio
                    and self.target_last_seen_time is not None
                    and (now - self.target_last_seen_time).nanoseconds * 1e-9 < 1.0):
                self._approach_inner_state = 'align'
                self._locked_far_active = False
                self._tgt_centered_since = None
                self.get_logger().warn(
                    f'[TARGET-LOCKED-FAR-UPGRADE-GATE] '
                    f'area={self.target_area_ratio:.4f}>={self.target_approach_enter_area_ratio:.4f}'
                    f' -> align')
            vel_cmd, yawspeed_servo = self._target_approach_servo_cmd(now, z)

            # ── 任务5：yaw 控制切换到视觉伺服（纯偏航速率模式，不锁定航向角）─────────
            # ★[B] yaw=NaN 告知 PX4 不要位置保持航向，只用 yawspeed 做伺服控制
            # 若设 yaw=att_yaw（当前角），PX4 会忽略 yawspeed 而维持固定航向，伺服失效
            yaw = float('nan')           # 启用 PX4 纯偏航速率控制
            yawspeed = yawspeed_servo    # 视觉伺服输出的 yawrate

            # ── 内部子状态 → 外部 self.state ──────────────────────────────
            _inner = self._approach_inner_state
            if _inner == 'land':
                # 触发真正降落命令
                if not getattr(self, '_target_land_sent', False):
                    self._target_land_sent = True
                    _vc_land = VehicleCommand()
                    _vc_land.timestamp = now_us
                    _vc_land.command = VehicleCommand.VEHICLE_CMD_NAV_LAND
                    _vc_land.param1 = 0.0
                    _vc_land.param2 = 0.0
                    _vc_land.param5 = float('nan')
                    _vc_land.param6 = float('nan')
                    _vc_land.param7 = 0.0
                    _vc_land.target_system = 1
                    _vc_land.target_component = 1
                    _vc_land.source_system = 255
                    _vc_land.source_component = 0
                    _vc_land.from_external = True
                    self.vehicle_cmd_pub.publish(_vc_land)
                    self._log_target_event("TARGET-LAND",
                        f"z={z:.3f} land_z={self.target_land_z:.3f} area={self.target_area_ratio:.3f}",
                        world_xy=getattr(self, '_tgt_world_est_xy', None),
                        conf=self._tgt_world_est_conf, trigger="land_sent")
                    self._log_event("TARGET-LAND",
                        f"z={z:.3f} land_z={self.target_land_z:.3f} "
                        f"area={self.target_area_ratio:.3f}")
                    self.get_logger().warn(
                        f"[TARGET-LAND] ★★ 发送 NAV_LAND "
                        f"z={z:.3f} area={self.target_area_ratio:.3f}"
                    )
                    # ★[A6] MISSION_DONE：任务完成锁存，禁止一切搜索恢复
                    self._mission_done = True
                    self.target_terminal_locked = True  # 确保终端锁不丢失
                    self._approach_exclusive = True     # 保持独占，防止探索状态机重新接管
                    self.explore_state = 'MISSION_DONE'
                    self.get_logger().warn(
                        "[MISSION-DONE] ★★[A6] 任务完成锁存！禁止再次探索/起飞！"
                    )
                    self._log_target_event("MISSION-DONE",
                        f"z={z:.3f} area={self.target_area_ratio:.3f}",
                        world_xy=getattr(self, '_tgt_world_est_xy', None),
                        conf=self._tgt_world_est_conf, trigger="land_sent")
                    self._log_event("MISSION-DONE", f"z={z:.3f} area={self.target_area_ratio:.3f}")
                if getattr(self, '_mission_done', False):
                    self.state = "MISSION_DONE"
                else:
                    self.state = "TARGET_LAND"
                vel_cmd = np.zeros(3, dtype=float)
            elif _inner in ('lost_short',):
                self.state = "TARGET_LOST_SHORT"
            elif _inner in ('lost_sweep',):
                self.state = "TARGET_LOST_SCAN"
                if _ta_now - getattr(self, '_sweep_log_t', 0.0) > 1.0:
                    self._sweep_log_t = _ta_now
                    self.get_logger().info(
                        f"[TARGET-LOST-SCAN] 扇扫中 seen_ago={_seen_ago:.1f}s "
                        f"sweep_dir={self._sweep_dir:.0f}"
                    )
            elif _inner == 'hover_confirm':
                self.state = "TARGET_HOVER_CONFIRM"
            elif _inner == 'descend':
                self.state = "TARGET_DESCEND"
            elif _inner == 'locked_far':
                self.state = "TARGET_LOCKED_FAR"
            elif _inner in ('align', 'approach'):
                self.state = "TARGET_APPROACH_ALIGN"
            elif _inner == 'forward':
                self.state = "TARGET_APPROACH_FORWARD"
            # ★[G6] 单提交点逼近新状态映射
            elif _inner == 'world_establish':
                self.state = "TARGET_WORLD_ESTABLISH"
            elif _inner == 'commit_nav':
                self.state = "TARGET_COMMIT_POINT"
            elif _inner == 'commit_hover':
                self.state = "TARGET_HOVER_COMMIT"
            elif _inner == 'conservative_descend':
                self.state = "TARGET_DESCEND_EXEC"  # ★[D] 用 TARGET_DESCEND_EXEC 统一对外名称
            elif _inner == 'lost_nav':
                self.state = "TARGET_LOST_SCAN"
            else:
                self.state = "TARGET_APPROACH"

        else:
            # ── HOVER_SCAN 硬优先：无论 local/APF 指令是否新鲜，HOVER_SCAN 独占控制 ──
            # 移除原来 "not is_cmd_fresh" 前提，让 HOVER_SCAN 真正独占 XY
            import time as _hs_mt
            _hs_now = _hs_mt.monotonic()

            if self.explore_state == "HOVER_SCAN":
                # [CTRL-OWNER] 5Hz 节流：标记当前控制权归属
                if _hs_now - getattr(self, '_ctrl_owner_log_t', 0.0) > 5.0:
                    self._ctrl_owner_log_t = _hs_now
                    self.get_logger().info(
                        f'[CTRL-OWNER] owner=SEARCH_FSM state=HOVER_SCAN '
                        f'escape={getattr(self, "_hover_scan_escape_active", False)}'
                    )
                # ★ SAFE_HOLD suppression：ESC-FAIL 后短时间内禁止重新触发 HOVER_SCAN escape，
                # 防止"SAFE_HOLD → 同帧重进 HOVER_SCAN → 立即再触发 escape"的无限循环
                _safe_hold_until = getattr(self, '_hscan_safe_hold_until', 0.0)
                if _hs_now < _safe_hold_until:
                    _sh_remain = _safe_hold_until - _hs_now
                    if _hs_now - getattr(self, '_sh_log_t', 0.0) > 3.0:
                        self._sh_log_t = _hs_now
                        self.get_logger().warn(
                            f"[HOVER-SAFE-HOLD] ESC-FAIL suppression active, "
                            f"remain={_sh_remain:.1f}s → position hold"
                        )
                    _vz_sh = float(np.clip(
                        self.alt_kp * z_err - self.alt_kd * self.prev_vel[2],
                        -self.alt_max_climb_speed, self.alt_max_descend_speed,
                    ))
                    vel_cmd = np.array([0.0, 0.0, _vz_sh], dtype=float)
                    self.publish_setpoint(vel_cmd, now_us)
                    self.log_status(now, vel_cmd, safety_ok=True, safety_reason="ok")
                    self.prev_vel = vel_cmd
                    self.last_loop_time = now
                    self.state = "SAFE_HOLD"
                    return

                # 记录被屏蔽的外部指令（节流日志）
                _had_fresh_local = self.is_cmd_fresh(self.last_local_stamp, now)
                _apf_xy_active = (
                    self.is_cmd_fresh(self.last_apf_stamp, now)
                    and self.last_apf_cmd is not None
                    and (abs(self.last_apf_cmd.cmd.linear.x) > 0.05
                         or abs(self.last_apf_cmd.cmd.linear.y) > 0.05)
                )
                if (_had_fresh_local or _apf_xy_active) and \
                        _hs_now - getattr(self, '_hover_bypass_log_t', 0.0) > 3.0:
                    self._hover_bypass_log_t = _hs_now
                    self.get_logger().debug(
                        f"[HOVER-SCAN-BYPASS-BLOCKED] "
                        f"local_fresh={_had_fresh_local} apf_xy={_apf_xy_active} "
                        f"→ HOVER_SCAN 独占，外部 XY 指令被屏蔽"
                    )

                self.state = f"SEARCH/{self.explore_state}"
                self.explore_cmd(now)  # 推进状态机（更新旋转角度 / scan_done 等）

                # ── HOVER_SCAN_ESCAPE：贴墙时先脱离，再继续扫描 ──────────────
                _near = self._nearest_lidar_distance()
                # ★[B3] 过门/门口目标保护：若当前在door_transit_mode（几何判定正在穿门），
                # 不触发ESCAPE——近墙是门框正常现象，强制扫描→Escape→SAFE_HOLD会打断过门序列。
                # 改为：若在door_transit_mode或goal是门类型，直接恢复到NAV_TO_VIEWPOINT继续追踪门目标。
                _in_door_transit_hs = getattr(self, '_door_transit_mode', False)
                _hs_is_door_goal = getattr(self, 'ssg_goal_type', 'none') in (
                    'frontier_gap_pre', 'frontier_gap_center', 'frontier_gap_post',
                    'frontier_gap_probe', 'confirmed_door')
                if (_in_door_transit_hs or _hs_is_door_goal) and self.ssg_viewpoint_goal is not None:
                    self.explore_state = "NAV_TO_VIEWPOINT"
                    if _hs_now - getattr(self, '_hs_door_resume_log_t', 0.0) > 2.0:
                        self._hs_door_resume_log_t = _hs_now
                        _near_s_hs = f"{_near:.2f}" if _near is not None else "?"
                        self.get_logger().info(
                            f"[HOVER-SCAN-DOOR-RESUME] ★ door_transit={_in_door_transit_hs} "
                            f"goal_type={getattr(self,'ssg_goal_type','?')} "
                            f"near={_near_s_hs}m → 跳过ESCAPE直接恢复导航"
                        )
                    return  # 跳过本帧position hold，下帧explore_state=NAV恢复导航
                _escape_trigger = (_near is not None
                                   and _near < self._hover_scan_emerg_stop_dist
                                   and not self._hover_scan_escape_active)
                if _escape_trigger:
                    # ★ 进入 ESCAPE：全圆激光扇区扫描，选最宽连续空闲弧中心方向
                    # 算法：① raw 角度→body frame ② 将360°等分36扇区,取各扇最小距离
                    #        ③ 找最宽连续空闲弧(min_dist > 阈值) → 取弧中心
                    #        ④ body frame角 + att_yaw → ENU world frame方向
                    _esc_dir = None
                    _esc_log_detail = "APF-fallback"
                    _esc_free_thr = 0.40  # 认为"空闲"的最小激光距离(m)
                    _scan_for_esc = getattr(self, 'last_scan', None)
                    if _scan_for_esc is not None and len(_scan_for_esc.ranges) > 0:
                        _raw_r = np.array(_scan_for_esc.ranges, dtype=float)
                        _raw_a = (_scan_for_esc.angle_min
                                  + np.arange(len(_raw_r), dtype=float)
                                  * _scan_for_esc.angle_increment)
                        _body_a = self._lidar_raw_to_body_angles(_raw_a)
                        _min_vd = float(getattr(self, 'lidar_min_valid_dist', 0.15))
                        _valid = np.isfinite(_raw_r) & (_raw_r >= _min_vd)
                        _r_clean = np.where(_valid, _raw_r, 9999.0)
                        # 36扇区: 每扇区10°，覆盖[−π, +π)
                        _n_sec = 36
                        _step = 2.0 * np.pi / _n_sec
                        _sec_min = np.full(_n_sec, 9999.0)
                        # ★[BUG-FIX] 追踪每扇区是否有真实激光数据（非哨兵值）
                        # 修复根因：无激光数据的扇区(_sec_min=9999哨兵)曾被误判为"9999m空旷"
                        # 正确处理：无数据扇区视为"未知/不可信"，不参与空闲弧选取
                        _sec_has_data = np.zeros(_n_sec, dtype=bool)
                        for _si in range(_n_sec):
                            _a_lo = -np.pi + _si * _step
                            _a_hi = _a_lo + _step
                            _mk = (_body_a >= _a_lo) & (_body_a < _a_hi)
                            _pts = _r_clean[_mk]
                            if _pts.size > 0:
                                _sec_min[_si] = float(np.min(_pts))
                                # 有效数据 = 至少一个读数 < 9000（非哨兵）
                                _sec_has_data[_si] = bool(np.any(_pts < 9000.0))
                        # 扇区"空闲"条件：①有真实激光数据 ②测量距离>阈值（0.40m）
                        _free = _sec_has_data & (_sec_min > _esc_free_thr)
                        # 环形连续弧：展开2倍
                        _free2 = np.tile(_free, 2)
                        _best_start, _best_len = 0, 0
                        _cs2, _cl2 = 0, 0
                        for _ii in range(2 * _n_sec):
                            if _free2[_ii]:
                                if _cl2 == 0:
                                    _cs2 = _ii
                                _cl2 += 1
                                if _cl2 > _best_len:
                                    _best_len, _best_start = _cl2, _cs2
                            else:
                                _cl2 = 0
                        if _best_len > 0:
                            _arc_idx = (_best_start + _best_len // 2) % _n_sec
                            _arc_body_ang = -np.pi + (_arc_idx + 0.5) * _step
                            _drone_yaw = float(self.att_yaw) if self.att_yaw is not None else 0.0
                            _world_ang = _drone_yaw + _arc_body_ang
                            _esc_dir = np.array([np.cos(_world_ang), np.sin(_world_ang)])
                            # clr值：使用真实测量值；若扇区无数据（不应出现，防御性处理）则报-1
                            _arc_clr = _sec_min[_arc_idx] if _sec_min[_arc_idx] < 9000.0 else -1.0
                            _esc_log_detail = (
                                f"free_arc center_body={np.rad2deg(_arc_body_ang):.0f}deg "
                                f"width={_best_len * 10}deg "
                                f"clr={_arc_clr:.2f}m"
                            )
                    if _esc_dir is None:
                        # 备用：APF 推斥方向（扫描不可用时）
                        _esc_apf = self.apply_proximity_constraints(np.zeros(3, dtype=float))
                        if np.linalg.norm(_esc_apf[:2]) > 0.01:
                            _esc_dir = _esc_apf[:2] / np.linalg.norm(_esc_apf[:2])
                        else:
                            _yw_fb = float(self.att_yaw) if self.att_yaw is not None else 0.0
                            _esc_dir = np.array([-np.cos(_yw_fb), -np.sin(_yw_fb)])
                    self._hover_scan_escape_dir = _esc_dir
                    self._hover_scan_escape_active = True
                    self._hover_scan_escape_start = _hs_now
                    self._log_event(
                        "HOVER-SCAN-ESCAPE",
                        f"enter near={_near:.2f}m "
                        f"esc_dir=({self._hover_scan_escape_dir[0]:.2f},"
                        f"{self._hover_scan_escape_dir[1]:.2f}) "
                        f"method={_esc_log_detail}"
                    )
                    self.get_logger().warn(
                        f"[HOVER-SCAN-ESCAPE] enter near={_near:.2f}m "
                        f"→ 脱离模式 dir=({self._hover_scan_escape_dir[0]:.2f},"
                        f"{self._hover_scan_escape_dir[1]:.2f}) "
                        f"[{_esc_log_detail}]"
                    )

                if self._hover_scan_escape_active:
                    # 严格恢复条件：
                    # 1) 最短持续 >= escape_min_dur_s (0.4s)
                    # 2) clearance > resume_clearance_m (0.60m) 持续 resume_hold_s (0.5s)
                    # 3) 超时保底 8s 防止卡死
                    _dur = _hs_now - self._hover_scan_escape_start
                    _past_min_dur = _dur >= self._hover_scan_escape_min_dur_s
                    _clr_ok = (_near is None or _near >= self._hover_scan_resume_clearance_m)
                    if _clr_ok:
                        if self._hover_scan_escape_clear_since <= 0.0:
                            self._hover_scan_escape_clear_since = _hs_now
                    else:
                        self._hover_scan_escape_clear_since = 0.0
                    _clr_held = (
                        self._hover_scan_escape_clear_since > 0.0
                        and (_hs_now - self._hover_scan_escape_clear_since) >= self._hover_scan_resume_hold_s
                    )
                    _escape_success = _past_min_dur and _clr_held
                    # ★ 临界清空：clearance < 0.25m 且逃离 > 0.5s → 立即视为超时升级
                    # 防止机体贴墙超过 3s 才触发升级（已知问题：0.25m 反复 FAIL 不升级）
                    _critical_near = (_near is not None and _near < 0.25 and _dur > 0.5)
                    _escape_timeout = ((_dur > 2.0 or _critical_near) and not _escape_success)  # ★[FIX1] 3.0→2.0s: 缩短 escape 最大持续时间
                    # 1Hz 状态日志
                    _near_str = f"{_near:.2f}" if _near is not None else "None"
                    if _critical_near and not _escape_success:
                        if _hs_now - getattr(self, '_crit_near_log_t', 0.0) > 0.5:
                            self._crit_near_log_t = _hs_now
                            self.get_logger().error(
                                f"[HSCAN-ESC-CRITICAL] clearance={_near_str}m < 0.25m "
                                f"dur={_dur:.1f}s → 立即升级（不等待3s超时）"
                            )
                    if _hs_now - self._hover_scan_escape_log_t > 1.0:
                        self._hover_scan_escape_log_t = _hs_now
                        self.get_logger().info(
                            f"[HSCAN-ESC-ACTIVE] dur={_dur:.1f}s "
                            f"clearance={_near_str}m "
                            f"(need>{self._hover_scan_resume_clearance_m:.2f}m "
                            f"for {self._hover_scan_resume_hold_s:.1f}s) "
                            f"clr_held={_clr_held} success={_escape_success} timeout={_escape_timeout}"
                        )
                    if _escape_timeout:
                        # ★ FAIL: 超时不算成功 → 升级策略
                        # ★ 修复：普通逃逸失败【不再冻结地图、不调 _slam_pause_scan】
                        # 只有真正的位姿失效才允许暂停 SLAM 输入；否则地图会被锁死
                        self._hover_scan_escape_active = False
                        self._hover_scan_escape_clear_since = 0.0
                        self.hover_scan_start_time = None
                        self.hover_scan_accum_angle = 0.0
                        self._hover_scan_last_yaw = None
                        # ── 记录失败锚点（黑名单），避免在同一位置反复触发 ──────
                        import time as _sht
                        import math as _bm
                        if not hasattr(self, '_hs_fail_anchors'):
                            self._hs_fail_anchors = []
                        _fa_pos = (
                            float(self.pose.x) if self.pose else 0.0,
                            float(self.pose.y) if self.pose else 0.0,
                        )
                        self._hs_fail_anchors.append((_sht.monotonic(), _fa_pos))
                        # 保留最近 10 条，清理 >120s 的条目
                        self._hs_fail_anchors = [
                            (t, p) for t, p in self._hs_fail_anchors
                            if _sht.monotonic() - t < 120.0
                        ][-10:]
                        # ── ★[B] 三层升级策略：连续失败计数 ────────────────────────────
                        # Tier-1 (fail_cnt==1)  : 短暂 SAFE_HOLD 0.8s + 尝试 frontier 逃逸点
                        # Tier-2 (fail_cnt==2)  : SAFE_HOLD 1.5s   + 尝试 frontier 逃逸点
                        # Tier-3 (fail_cnt>=3)  : SAFE_HOLD 2.5s   + 强制空间冷却 2× TTL
                        # 目标：门口场景首次失败不进入长时锁定，给无人机机会恢复导航
                        self._hover_scan_escape_fail_count = getattr(
                            self, '_hover_scan_escape_fail_count', 0) + 1
                        _fail_cnt = self._hover_scan_escape_fail_count
                        if _fail_cnt >= 3:
                            # Tier-3：长时 SAFE_HOLD，加倍空间冷却惩罚
                            self._hover_scan_escape_level = 3
                            _sh_dur = 2.5
                        elif _fail_cnt == 2:
                            # Tier-2：中等 SAFE_HOLD
                            self._hover_scan_escape_level = 2
                            _sh_dur = 1.5
                        else:
                            # Tier-1（第1次失败）：最短 SAFE_HOLD，优先恢复探索
                            self._hover_scan_escape_level = 1
                            _sh_dur = 0.8
                        self._hscan_safe_hold_until = _sht.monotonic() + _sh_dur
                        # 尝试找 frontier 逃逸点（Tier-1/2 均尝试，有目标则自动切 NAV）
                        _front_esc = self._find_frontier_escape_point()
                        if _front_esc is not None:
                            self.ssg_viewpoint_goal = _front_esc
                            self.astar_path = None
                            self.astar_path_idx = 0
                        self.explore_state = "SAFE_HOLD"
                        self.get_logger().warn(
                            f"[HSCAN-ESC-FAIL] timeout={_dur:.1f}s clearance={_near_str}m "
                            f"fail_count={_fail_cnt} tier={self._hover_scan_escape_level} "
                            f"safe_hold={_sh_dur:.1f}s (地图不冻结，继续更新)"
                        )
                        self._log_event(
                            "HSCAN-ESC-FAIL",
                            f"timeout={_dur:.1f}s clearance={_near_str} "
                            f"fail_count={_fail_cnt} tier={self._hover_scan_escape_level} "
                            f"safe_hold={_sh_dur:.1f}s (map NOT frozen)"
                        )
                        # 空间冷却：Tier-3 使用 2× TTL，其余正常 TTL
                        if self.pose is not None:
                            _fa_x = float(self.pose.x)
                            _fa_y = float(self.pose.y)
                            if _fail_cnt >= 3:
                                # Tier-3：双倍 TTL 的空间冷却
                                _ttl2x = getattr(self, 'spatial_cooldown_ttl_sec', 90.0) * 2.0
                                self._spatial_cooldown_stamp(_fa_x, _fa_y, ttl_override=_ttl2x)
                                self.get_logger().warn(
                                    f"[SPATIAL-COOLDOWN] tier3 hscan_esc_fail DOUBLE stamp "
                                    f"pos=({_fa_x:.2f},{_fa_y:.2f}) "
                                    f"ttl={_ttl2x:.0f}s"
                                )
                            else:
                                self._spatial_cooldown_stamp(_fa_x, _fa_y)
                                self.get_logger().warn(
                                    f"[SPATIAL-COOLDOWN] hscan_esc_fail stamp "
                                    f"pos=({_fa_x:.2f},{_fa_y:.2f}) "
                                    f"ttl={getattr(self,'spatial_cooldown_ttl_sec',90.0):.0f}s"
                                )
                        self.get_logger().warn(
                            f"[SAFE-HOLD-ENTER] reason=hscan_esc_fail "
                            f"fail_count={_fail_cnt} tier={self._hover_scan_escape_level} "
                            f"near={_near_str}m safe_hold={_sh_dur:.1f}s"
                        )
                        # SAFE_HOLD setpoint: Z 保持高度，XY 停
                        _vz_fail = float(np.clip(
                            self.alt_kp * z_err - self.alt_kd * self.prev_vel[2],
                            -self.alt_max_climb_speed, self.alt_max_descend_speed
                        ))
                        vel_cmd = np.array([0.0, 0.0, _vz_fail], dtype=float)
                        self.publish_setpoint(vel_cmd, now_us)
                        self.log_status(now, vel_cmd, safety_ok=True, safety_reason="ok")
                        self.prev_vel = vel_cmd
                        self.last_loop_time = now
                        self.state = "SAFE_HOLD"
                        # ★ 关键：重置 hover_scan 状态，防止下一帧立即重进 escape（死循环根因）
                        self.hover_scan_start_time = None
                        self.hover_scan_accum_angle = 0.0
                        self._hover_scan_last_yaw = None
                        return
                    elif _escape_success:
                        # SUCCESS: clearance 持续足够时间 → 重置失败计数
                        self._hover_scan_escape_active = False
                        self._hover_scan_escape_clear_since = 0.0
                        self._hover_scan_last_escape_t = _hs_now
                        self._hover_scan_escape_fail_count = 0    # 成功 → 清零履历
                        self._hover_scan_escape_level = 0
                        _clr_d = _near if _near is not None else -1.0
                        self.get_logger().info(
                            f"[HSCAN-ESC-DONE] clearance={_clr_d:.2f}m dur={_dur:.1f}s → HOVER_SCAN"
                        )
                        self._log_event(
                            "HSCAN-ESC-DONE",
                            f"clearance={_clr_d:.2f}m dur={_dur:.1f}s stable={_clr_held}"
                        )
                        # ★ Fix-RCA2: 将 hold_pos 更新为逃脱后当前位置，防止 position control
                        # 把无人机重新拉回原近墙位置，导致逃脱-回拉-再逃脱的振荡→高倾斜
                        if self.pose is not None:
                            self._hover_scan_hold_pos = (
                                float(self.pose.x),
                                float(self.pose.y),
                                float(self.alt_target_z),
                            )
                            self.get_logger().info(
                                f"[HSCAN-ESC-DONE] hold_pos updated to escaped pos "
                                f"({self.pose.x:.2f},{self.pose.y:.2f}) to prevent wall return"
                            )
                        # 继续进入 HOVER_SCAN position hold（不 return）
                    else:
                        # 仍在 escape 中：输出逃脱速度（XY），Z 保持高度
                        _spd = self._hover_scan_escape_speed
                        _vz_e = float(np.clip(
                            self.alt_kp * z_err - self.alt_kd * self.prev_vel[2],
                            -self.alt_max_climb_speed, self.alt_max_descend_speed
                        ))
                        # 防御：escape_dir 可能因 HOVER-EMERG 路径未赋值而为 None
                        if self._hover_scan_escape_dir is None:
                            _yw_fb = float(self.att_yaw) if self.att_yaw is not None else 0.0
                            self._hover_scan_escape_dir = np.array(
                                [-np.cos(_yw_fb), -np.sin(_yw_fb)])
                            self.get_logger().warn(
                                "[HSCAN-ESC-DIR-FALLBACK] escape_dir was None, using yaw fallback"
                            )
                        vel_cmd = np.array([
                            float(self._hover_scan_escape_dir[0]) * _spd,
                            float(self._hover_scan_escape_dir[1]) * _spd,
                            _vz_e,
                        ], dtype=float)
                        self.publish_setpoint(vel_cmd, now_us)
                        self.log_status(now, vel_cmd, safety_ok=True, safety_reason="ok")
                        self.prev_vel = vel_cmd
                        self.last_loop_time = now
                        return

                # ── 正常 HOVER_SCAN position hold ────────────────────────────
                if self._hover_scan_hold_pos is not None:
                    hx, hy, hz = self._hover_scan_hold_pos
                else:
                    hx = float(self.pose.x) if self.pose is not None else 0.0
                    hy = float(self.pose.y) if self.pose is not None else 0.0
                    hz = float(self.alt_target_z)
                hz = float(self.alt_target_z)
                # ★ Fix-RCA3: 对Z误差限幅，防止EKF漂移时PX4全速爬升/下沉
                # position setpoint 模式下 alt_max_climb_speed 不起作用，需在此处限制目标值
                if self.pose is not None:
                    _hz_cur = float(self.pose.z)
                    _hz_err = hz - _hz_cur
                    _hz_max_step = 0.8  # 单帧最大Z偏差(m)，超过则截断
                    if abs(_hz_err) > _hz_max_step:
                        hz = _hz_cur + float(np.sign(_hz_err)) * _hz_max_step
                        if _hs_now - getattr(self, '_hover_z_clamp_log_t', 0.0) > 5.0:
                            self._hover_z_clamp_log_t = _hs_now
                            self.get_logger().warn(
                                f"[HOVER-Z-CLAMP] Z误差={_hz_err:.2f}m > {_hz_max_step}m "
                                f"(EKF漂移?) → 截断 hz={hz:.2f}"
                            )

                # 节流 [HOVER-SCAN-HOLD] 日志
                if _hs_now - getattr(self, '_hover_hold_log_t', 0.0) > 5.0:
                    self._hover_hold_log_t = _hs_now
                    self.get_logger().info(
                        f"[HOVER-SCAN-HOLD] engaged hold_pos=({hx:.2f},{hy:.2f},{hz:.2f}) "
                        f"ignore_local_xy=True ignore_apf_xy=True "
                        f"escape_active={self._hover_scan_escape_active}"
                    )
                self.publish_position_yaw_setpoint(
                    hx, hy, hz, now_us,
                    yawspeed=float(self.hover_scan_yaw_speed),
                    enable_spin=True  # 正常 HOVER_SCAN 旋转扫描
                )
                vel_cmd = np.zeros(3, dtype=float)
                self.log_status(now, vel_cmd, safety_ok=True, safety_reason="ok")
                self.prev_vel = np.zeros(3, dtype=float)
                self.last_loop_time = now
                return

            # ── ★[B] ROOM_ENTRY_SCAN：位置保持 + 偏航扇扫（与 HOVER_SCAN 采用相同 position setpoint）
            elif self.explore_state == "ROOM_ENTRY_SCAN":
                # 调用扇扫状态机更新 _servo_yawspeed（可能切换 explore_state → HOVER_SCAN）
                _re_now_sec = now.nanoseconds * 1e-9
                self._room_entry_scan_cmd(now, _re_now_sec)
                _re_hold_pos = getattr(self, '_hover_scan_hold_pos', None)
                if _re_hold_pos is not None:
                    _rex, _rey, _rez = _re_hold_pos
                else:
                    if self.pose is not None:
                        _rex = float(self.pose.x)
                        _rey = float(self.pose.y)
                    else:
                        _rex = _rey = 0.0
                    _rez = float(self.alt_target_z)
                _rez = float(self.alt_target_z)
                _re_yawspeed = float(getattr(self, '_servo_yawspeed', 0.0))
                self.publish_position_yaw_setpoint(
                    _rex, _rey, _rez, now_us,
                    yawspeed=_re_yawspeed,
                    enable_spin=abs(_re_yawspeed) > 0.01
                )
                vel_cmd = np.zeros(3, dtype=float)
                self.log_status(now, vel_cmd, safety_ok=True, safety_reason="ok")
                self.prev_vel = np.zeros(3, dtype=float)
                self.last_loop_time = now
                return

            # ── 非 HOVER_SCAN（NAV_TO_VIEWPOINT / 外部指令）：velocity 控制 ──
            _apf_xy_active = (
                self.is_cmd_fresh(self.last_apf_stamp, now)
                and self.last_apf_cmd is not None
                and (abs(self.last_apf_cmd.cmd.linear.x) > 0.05
                     or abs(self.last_apf_cmd.cmd.linear.y) > 0.05)
            )

            # ─── 2s 周期 CTRL-ARB 诊断日志（所有搜索/导航/目标状态均打印）────────────────────
            import time as _carb_t
            _carb_now = _carb_t.monotonic()
            _local_fresh   = self.is_cmd_fresh(self.last_local_stamp, now)
            _apf_fresh     = self.is_cmd_fresh(self.last_apf_stamp, now)
            _apf_clearance = (self.last_apf_cmd.clearance
                              if self.last_apf_cmd is not None else 9.9)
            # ★[A-ARCH] 扩展日志范围：不再仅限 SEARCH，NAV/TARGET 均输出
            # TAKEOFF/ARMING/IDLE/LAND阶段静默，避免噪音
            _in_search     = (self.state not in ('ARMING', 'IDLE', 'TAKEOFF',
                                                  'TAKEOFF_STABILIZE', 'TARGET_LAND')
                               and 'PLANAR' not in self.state)
            if _in_search and _carb_now - getattr(self, '_carb_log_t', 0.0) > 2.0:
                self._carb_log_t = _carb_now
                _lcxy = "None"
                if self.last_local_cmd is not None:
                    _l = self.last_local_cmd.cmd.linear
                    _lcxy = f"({_l.x:.2f},{_l.y:.2f})"
                _apfxy = "None"
                if self.last_apf_cmd is not None:
                    _a = self.last_apf_cmd.cmd.linear
                    _apfxy = f"({_a.x:.2f},{_a.y:.2f})"
                # 预期仲裁模式：若 local_fresh → LOCAL_DOMINANT (+APF additive)
                #               若 apf_fresh only → apf
                #               否则 → explore
                _exp_mode = ("LOCAL_DOMINANT" if _local_fresh
                             else ("APF" if _apf_fresh else "EXPLORE"))
                # ★[I] source 字段对齐 A1 设计规范
                _source_str = ("LOCAL" if _local_fresh
                               else ("APF" if _apf_fresh else "FALLBACK"))
                # ★[H1] CTRL-ARB：追加目标链、世界坐标状态
                _twe_conf_arb = float(getattr(self, '_tgt_world_est_conf', 0.0))
                _vtg_arb = getattr(self, '_visual_target_goal', None)
                _vtg_str = (f"({_vtg_arb[0]:.2f},{_vtg_arb[1]:.2f})"
                            if _vtg_arb is not None else "None")
                _inner_st_arb = getattr(self, '_approach_inner_state', '-')
                _approach_arb = getattr(self, '_approach_exclusive', False)
                self.get_logger().info(
                    f"[CTRL-ARB] state={self.state} source={_source_str} "
                    f"explore={getattr(self,'explore_state','?')} "
                    f"local_fresh={_local_fresh} local_xy={_lcxy} "
                    f"apf_fresh={_apf_fresh} apf_xy={_apfxy} "
                    f"apf_clr={_apf_clearance:.2f} "
                    f"expected_mode={_exp_mode} "
                    f"target_inner={_inner_st_arb} approach_excl={_approach_arb} "
                    f"world_est_conf={_twe_conf_arb:.2f} vtg={_vtg_str}"
                )
                # ★[I1] CTRL-LAYER：控制分层健康诊断（SSG=规划层, LOCAL_MPC=执行层, SAFE_HOLD=保护层）
                _in_approach = getattr(self, '_approach_exclusive', False)
                _in_safe_hold = self.state == "SAFE_HOLD"
                _planner  = "SSG"
                _executor = ("TARGET_SERVO" if _in_approach
                             else ("SAFE_HOLD" if _in_safe_hold
                                   else ("LOCAL_MPC" if _local_fresh else "FALLBACK")))
                _protector = ("SAFE_HOLD" if _in_safe_hold else
                              ("APF_STOP" if getattr(self,'last_apf_cmd',None) is not None
                               and getattr(self.last_apf_cmd,'clearance',9.9) <= 0.45
                               else "OK"))
                self.get_logger().info(
                    f"[CTRL-LAYER] planner={_planner} executor={_executor} "
                    f"protector={_protector} "
                    f"approach={_in_approach} safe_hold={_in_safe_hold}"
                )
                # ★[A-ARCH] LOCAL_MPC 健康评估
                if _local_fresh:
                    self.get_logger().info(
                        f"[LOCAL-MPC-ACTIVE] state={self.state} "
                        f"explore={getattr(self,'explore_state','?')} "
                        f"cmd=({_lcxy}) src=local_cmd"
                    )
                elif not _local_fresh and _apf_fresh:
                    self.get_logger().info(
                        "[LOCAL-MPC-PREEMPT] reason=stale_local_cmd source=APF"
                    )
                else:
                    self.get_logger().warn(
                        f"[LOCAL-MPC-PREEMPT] reason=stale_all "
                        f"state={self.state} explore={getattr(self,'explore_state','?')} "
                        f"source=EXPLORE_FALLBACK"
                    )

            vel_xy = None
            # ── 优先级: apf_stop_lock (边界紧急) > apf_emerg > local_cmd > apf > explore ──
            # APF 现在输出 ENU 坐标（x=东, y=北），无需坐标转换
            # NOTE: APF clearance 已修为 xy-only（不含 z 轴），起飞时不会误触发 stop_lock
            # 1) apf_stop_lock: clearance <= stop_radius (0.42m)，即使输出为 (0,0,0) 也必须锁车
            #    额外保护：TAKEOFF_STABILIZE 阶段不应被 APF stop_lock 锁死
            _in_takeoff = self.state in ("TAKEOFF_STABILIZE", "ARMING", "IDLE")
            _apf_stop_lock = (
                not _in_takeoff
                and self.is_cmd_fresh(self.last_apf_stamp, now)
                and self.last_apf_cmd is not None
                and self.last_apf_cmd.clearance <= 0.42
            )
            # 2) apf_emerg: 0.42 < clearance < 0.45, APF 还有适途速度幸出
            _apf_emerg_active = (
                self.is_cmd_fresh(self.last_apf_stamp, now)
                and self.last_apf_cmd is not None
                and self.last_apf_cmd.clearance < 0.45
                and not _apf_stop_lock
            )
            if _apf_stop_lock:
                # 硬停锁车：APF 已到边界并归零输出，保持 XY 位置，override_active=True
                _apf_clr_sl = self.last_apf_cmd.clearance
                vel_xy = np.zeros(3, dtype=float)
                self.get_logger().warn(
                    f"[CTRL-ARB] branch=hard_block clearance={_apf_clr_sl:.3f}m",
                    throttle_duration_sec=0.5
                )
                self.get_logger().warn(
                    f"[CTRL-PRIORITY] source=apf_stop_lock "
                    f"clearance={_apf_clr_sl:.3f}m <=0.42m → XY锁车",
                    throttle_duration_sec=0.5
                )
                self._log_event("APF-EMERGENCY",
                    f"stop_lock clearance={_apf_clr_sl:.3f}m cmd=(0,0) stop_lock=True")
            elif _apf_emerg_active:
                tw = self.last_apf_cmd.cmd
                # APF 输出 ENU (x=东, y=北)，直接使用
                vel_xy = np.array([tw.linear.x, tw.linear.y, 0.0], dtype=float)
                self.get_logger().warn(
                    f"[CTRL-ARB] branch=apf_emerg clearance={self.last_apf_cmd.clearance:.3f}m",
                    throttle_duration_sec=0.5
                )
                self.get_logger().warn(
                    f"[CTRL-PRIORITY] source=emergency "
                    f"apf_clearance={self.last_apf_cmd.clearance:.3f}m "
                    f"cmd_enu=({vel_xy[0]:.3f},{vel_xy[1]:.3f})",
                    throttle_duration_sec=0.5
                )
            elif self.is_cmd_fresh(self.last_local_stamp, now):
                # ★[A2] local_cmd 作为 PRIMARY 速度源（SEARCH/NAV_TO_VIEWPOINT 不再屏蔽）
                # APF 作为 ADDITIVE 叠加修正方向，仅在非 stop_lock/emerg 时生效
                import time as _ctrl_t
                _ctrl_now = _ctrl_t.monotonic()
                tw = self.last_local_cmd.cmd
                vel_xy = np.array([tw.linear.x, tw.linear.y, 0.0], dtype=float)
                # ★[I2] 持续零速保护：MPC fresh 但一直输出(0,0) > 2s → 绕到 explore
                # 场景：MPC 路径点全被阻断但未清路径（防御性兜底，I1 已解决根因）
                _i2_mag = float(np.linalg.norm(vel_xy[:2]))
                if _i2_mag < 0.02:
                    _i2_now = _ctrl_now
                    if getattr(self, '_local_zero_since', 0.0) < 1e-3:
                        self._local_zero_since = _i2_now
                    elif _i2_now - self._local_zero_since > 2.0:
                        _i2_dur = _i2_now - self._local_zero_since
                        self.get_logger().warn(
                            f"[CTRL-ARB-I2] MPC持续零速>{_i2_dur:.1f}s"
                            f" → explore fallback",
                            throttle_duration_sec=3.0
                        )
                        # ★[C4/C5] MPC-BLOCKED 检测：持续零速 > 5s 且正在导航
                        # → 认为当前目标 MPC 无法规划路径，黑名单该目标（BLACKLIST-GOAL/FAILED-TRANSITION）
                        _MPC_BLOCK_THR = 5.0
                        if (_i2_dur > _MPC_BLOCK_THR
                                and getattr(self, 'explore_state', '') == 'NAV_TO_VIEWPOINT'
                                and getattr(self, 'ssg_viewpoint_goal', None) is not None):
                            _blk_goal = self.ssg_viewpoint_goal
                            _blk_type = getattr(self, 'ssg_goal_type', 'unknown')
                            _blk_key  = f"{_blk_goal[0]:.1f}_{_blk_goal[1]:.1f}"
                            _blk_last_t = getattr(self, '_mpc_blocked_last_t', {})
                            _blk_prev = _blk_last_t.get(_blk_key, 0.0)
                            if _i2_now - _blk_prev > 10.0:   # 10s 防抖（防止重复加黑名单）
                                _blk_last_t[_blk_key] = _i2_now
                                self._mpc_blocked_last_t = _blk_last_t
                                # 黑名单 60s，给地图更新时间
                                self._target_blacklist[_blk_key] = _i2_now + 60.0
                                self.get_logger().warn(
                                    f"[MPC-BLOCKED] 持续零速 {_i2_dur:.1f}s > {_MPC_BLOCK_THR:.0f}s "
                                    f"goal=({_blk_goal[0]:.2f},{_blk_goal[1]:.2f}) type={_blk_type} "
                                    f"→ blacklisted 60s"
                                )
                                self._log_event("MPC-BLOCKED",
                                    f"dur={_i2_dur:.1f}s goal=({_blk_goal[0]:.2f},{_blk_goal[1]:.2f}) "
                                    f"type={_blk_type} blacklisted=60s")
                                self.get_logger().warn(
                                    f"[FAILED-TRANSITION] reason=mpc_zero_stuck "
                                    f"goal=({_blk_goal[0]:.2f},{_blk_goal[1]:.2f}) type={_blk_type}"
                                )
                                self._log_event("FAILED-TRANSITION",
                                    f"reason=mpc_zero_stuck type={_blk_type} "
                                    f"goal=({_blk_goal[0]:.2f},{_blk_goal[1]:.2f})")
                                self.get_logger().warn(
                                    f"[BLACKLIST-GOAL] key={_blk_key} ttl=60s "
                                    f"type={_blk_type} reason=mpc_blocked"
                                )
                                self._log_event("BLACKLIST-GOAL",
                                    f"key={_blk_key} ttl=60s type={_blk_type} "
                                    f"reason=mpc_blocked")
                                # 清除当前目标强制重选
                                self.ssg_viewpoint_goal = None
                                self.ssg_goal_type = "none"
                                self.astar_path = None
                                self.explore_state = "HOVER_SCAN"
                                self.hover_scan_start_time = None
                        self.state = f"SEARCH/{self.explore_state}"
                        vel_xy = self.explore_cmd(now)
                else:
                    self._local_zero_since = 0.0
                _apf_additive_ok = (
                    self.is_cmd_fresh(self.last_apf_stamp, now)
                    and self.last_apf_cmd is not None
                    and self.last_apf_cmd.clearance < 90.0
                    and not _apf_stop_lock
                    and not _apf_emerg_active
                )
                _ctrl_mode = "LOCAL_DOMINANT"
                if _apf_additive_ok:
                    _a = self.last_apf_cmd.cmd
                    _apf_add = np.array([_a.linear.x * 0.4, _a.linear.y * 0.4, 0.0])
                    vel_xy = vel_xy + _apf_add
                    _spd_limit = float(getattr(self, 'ssg_nav_speed', 0.6))
                    _spd = float(np.linalg.norm(vel_xy[:2]))
                    if _spd > _spd_limit:
                        vel_xy[:2] *= _spd_limit / _spd
                    _ctrl_mode = "LOCAL+APF"
                if _ctrl_now - getattr(self, '_ctrl_local_log_t', 0.0) > 2.0:
                    self._ctrl_local_log_t = _ctrl_now
                    self.get_logger().info(
                        f"[CTRL-ARB] branch=local_cmd mode={_ctrl_mode} state={self.state} "
                        f"explore={getattr(self,'explore_state','?')} "
                        f"cmd=({float(tw.linear.x):.2f},{float(tw.linear.y):.2f}) "
                        f"final=({vel_xy[0]:.2f},{vel_xy[1]:.2f})"
                    )
                # ★[ARRIVE-CHECK] 修复：LOCAL_DOMINANT 模式下到达检测永远不运行的根因
                # 问题根因：_nav_to_viewpoint_cmd（含 到达检测 + astar_path_idx 推进）
                #   仅在 explore_cmd 的 FALLBACK 分支调用。local_fresh=True 时 CTRL-ARB
                #   不走 FALLBACK → _nav_to_viewpoint_cmd 从不执行 → ssg_goal 永不清除
                #   → astar_path_idx 永远不推进 → _publish_ssg_path 每0.4s重新从idx=0
                #   发布完整路径 → local_mpc 每次 reset idx=0 重新回到起始点追踪
                #   → 无人机在规划点附近永久振荡来回徘徊。
                # 修复：local_fresh=True 且在 NAV_TO_VIEWPOINT 时，始终调用 explore_cmd
                #   推进状态机（到达/卡停检测 + 路径推进），忽略其返回速度（仍用 local_mpc）。
                # 条件 _i2_mag >= 0.02：‌I2-fallback 未激活（elif-I2 已调用 explore_cmd）
                #   时跳过避免重复调用；local_mpc 近零速（_i2_mag<0.02）时到达检测延迟
                #   至 I2 fallback（2s）内触发，可接受。
                if (_i2_mag >= 0.02
                        and getattr(self, 'explore_state', '') in ('NAV_TO_VIEWPOINT', 'POST_DOOR_COMMIT')):
                    self.explore_cmd(now)
            elif self.is_cmd_fresh(self.last_apf_stamp, now):
                tw = self.last_apf_cmd.cmd
                # APF 输出 ENU (x=东, y=北)，直接使用（无需 NED→ENU 交换）
                # ★ 安全检查：若 APF 正被抑制（clearance>=90 哨兵值），跳到 explore
                _apf_suppressed_sentinel = (self.last_apf_cmd.clearance >= 90.0)
                if _apf_suppressed_sentinel:
                    self.state = f"SEARCH/{self.explore_state}"
                    vel_xy = self.explore_cmd(now)
                    self.get_logger().debug("[CTRL-ARB] branch=explore(apf_suppressed)")
                else:
                    vel_xy = np.array([tw.linear.x, tw.linear.y, 0.0], dtype=float)
                    self.get_logger().debug("[CTRL-ARB] branch=apf")
                    self.get_logger().debug(
                        f"[CTRL-PRIORITY] source=apf cmd_enu=({vel_xy[0]:.3f},{vel_xy[1]:.3f})"
                    )
            else:
                self.state = f"SEARCH/{self.explore_state}"
                vel_xy = self.explore_cmd(now)
                self.get_logger().debug("[CTRL-ARB] branch=explore")
                self.get_logger().debug("[CTRL-PRIORITY] source=explore")

            if not self.scan_is_recent(now):
                vel_xy[:2] = 0.0

            vz = float(np.clip(self.alt_kp * z_err - self.alt_kd * self.prev_vel[2],
                               -self.alt_max_climb_speed, self.alt_max_descend_speed))
            vel_cmd = np.array([vel_xy[0], vel_xy[1], vz], dtype=float)
            vel_cmd = self.apply_proximity_constraints(vel_cmd)
            vel_cmd = self.clip_bounds(vel_cmd)

            yawspeed = 0.0
            # ★ [A10/PATH-YAW] NAV_TO_VIEWPOINT 阶段：机头跟随路径切线方向，防止摄像头长期对着墙
            # HOVER_SCAN 阶段继续韇偏航（旋转扫评），其他阶段用路径切线
            _path_yaw_target = None
            if (self.explore_state == 'NAV_TO_VIEWPOINT'
                    and self.astar_path
                    and self.astar_path_idx < len(self.astar_path)
                    and self.pose is not None
                    and self.att_yaw is not None):
                _wp_nav = self.astar_path[self.astar_path_idx]
                _dx_nav = float(_wp_nav[0]) - self._plan_px
                _dy_nav = float(_wp_nav[1]) - self._plan_py
                _d_nav  = float(np.hypot(_dx_nav, _dy_nav))
                if _d_nav > 0.15:   # 有意义的步长才用路径方向
                    _path_yaw_target = float(np.arctan2(_dy_nav, _dx_nav))
            if _path_yaw_target is not None:
                _cur_yaw_nav = float(self.att_yaw)
                _yaw_err_nav = (_path_yaw_target - _cur_yaw_nav + np.pi) % (2.0 * np.pi) - np.pi
                _YAW_RATE_MAX = 0.6   # rad/s，@20Hz = 0.03 rad/step
                _YAW_DT       = 0.05  # 20Hz 循环周期
                _yaw_step = float(np.clip(_yaw_err_nav,
                                          -_YAW_RATE_MAX * _YAW_DT,
                                           _YAW_RATE_MAX * _YAW_DT))
                yaw = _cur_yaw_nav + _yaw_step
                import time as _py_t
                if _py_t.monotonic() - getattr(self, '_path_yaw_log_t', 0.0) > 3.0:
                    self._path_yaw_log_t = _py_t.monotonic()
                    self.get_logger().info(
                        f"[PATH-YAW] 路径切线跟随: target={np.rad2deg(_path_yaw_target):.1f}° "
                        f"cur={np.rad2deg(_cur_yaw_nav):.1f}° "
                        f"err={np.rad2deg(_yaw_err_nav):.1f}° "
                        f"step={np.rad2deg(_yaw_step):.2f}°/frame "
                        f"wp=({float(_wp_nav[0]):.2f},{float(_wp_nav[1]):.2f})"
                    )
            else:
                yaw = float(self.locked_yaw) if self.locked_yaw is not None else (
                    float(self.att_yaw) if self.att_yaw is not None else 0.0
                )

        # ── Phase7: 进度看门狗（goal-stuck 检测与恢复）────────────────────────
        # 仅在搜索模式（非 HOVER_SCAN / TARGET_APPROACH / SAFE_HOLD）激活
        import time as _pwd_mt
        _pwd_now = _pwd_mt.monotonic()
        _is_explore_nav = (
            self.state.startswith("SEARCH")
            and not self.target_found
            and self.explore_state not in ("HOVER_SCAN", "TARGET_APPROACH_LOCK")
            and self.pose is not None
            and not getattr(self, '_approach_exclusive', False)   # 任务1：独占期间禁止进度看门狗
        )
        if _is_explore_nav:
            _cx = float(self.pose.x)
            _cy = float(self.pose.y)
            _vel_mag = float(np.hypot(vel_cmd[0], vel_cmd[1]))
            # 初始化参考位置
            if self._progress_wd_pos is None:
                self._progress_wd_pos = (_cx, _cy)
                self._progress_wd_since = _pwd_now
            else:
                _dx = _cx - self._progress_wd_pos[0]
                _dy = _cy - self._progress_wd_pos[1]
                _moved = float(np.hypot(_dx, _dy))
                _elapsed = _pwd_now - self._progress_wd_since
                if _moved >= self._progress_wd_min_move_m:
                    # 有实质进度 → 重置参考位置
                    self._progress_wd_pos = (_cx, _cy)
                    self._progress_wd_since = _pwd_now
                    self._progress_wd_backstep_frames = 0
                elif (_elapsed >= self._progress_wd_timeout_sec
                      and _vel_mag > 0.04
                      and self._progress_wd_backstep_frames == 0):
                    # 超时 + 命令速度非零 + 未在后退 → 触发看门狗
                    self._progress_wd_trig_cnt += 1
                    self.get_logger().warn(
                        f"[PROG-WD] 进度看门狗触发 #{self._progress_wd_trig_cnt}: "
                        f"pos=({_cx:.2f},{_cy:.2f}) "
                        f"位移={_moved:.2f}m < {self._progress_wd_min_move_m}m "
                        f"已卡住={_elapsed:.1f}s > {self._progress_wd_timeout_sec}s "
                        f"vel_cmd=({vel_cmd[0]:.2f},{vel_cmd[1]:.2f}) "
                        f"→ 触发 HOVER_SCAN 重新扫描"
                    )
                    # 后退方向：与当前速度反向（或随机后退）
                    if _vel_mag > 1e-3:
                        self._progress_wd_backstep_dir = -np.array(
                            [vel_cmd[0], vel_cmd[1], 0.0], dtype=float
                        ) / _vel_mag * 0.15   # 0.15m/s 后退
                    else:
                        self._progress_wd_backstep_dir = np.array(
                            [0.1, 0.1, 0.0], dtype=float
                        )
                    self._progress_wd_backstep_frames = 15  # ~0.75s @20Hz
                    # 强制进入 HOVER_SCAN 重新扫描并规划
                    self.explore_state = "HOVER_SCAN"
                    self._progress_wd_pos = None   # 重置参考位置

        # 执行后退帧（看门狗触发后的恢复后退）
        if (self._progress_wd_backstep_frames > 0
                and self._progress_wd_backstep_dir is not None
                and not getattr(self, '_approach_exclusive', False)):  # 独占期间禁止看门狗后退
            vel_cmd = self._progress_wd_backstep_dir.copy()
            vel_cmd[2] = float(np.clip(
                self.alt_kp * (self.alt_target_z - float(self.pose.z if self.pose else 0.0)),
                -self.alt_max_climb_speed, self.alt_max_descend_speed
            ))
            self._progress_wd_backstep_frames -= 1
        elif not _is_explore_nav:
            # 非搜索模式时重置参考位置，避免切换模式后立即触发
            self._progress_wd_pos = None
            self._progress_wd_since = _pwd_now

        vel_cmd = self.apply_smoothing(vel_cmd, now)
        # ★[A1] SSG 路径持续广播：NAV_TO_VIEWPOINT 期间以 2.5Hz 节流发布 A* 路径
        # 为 local_mpc 提供高优先级路径，确保 local_fresh=True （解决 global_planner 抑制问题）
        # 仅在非独占控制模式且有有效路径时广播；离开 NAV_TO_VIEWPOINT 时发布空消息
        if (self.state.startswith("SEARCH")
                and not getattr(self, '_approach_exclusive', False)):
            if getattr(self, 'explore_state', '') == 'NAV_TO_VIEWPOINT':
                self._publish_ssg_path()
            else:
                # HOVER_SCAN 等非导航阶段：发布空消息，释放 local_mpc 给 global_path 控制
                import time as _ssg_cl_t
                _ssg_cl_now = _ssg_cl_t.monotonic()
                if _ssg_cl_now - self._ssg_path_pub_t > 1.0:  # 避免频繁空发
                    self._ssg_path_pub_t = _ssg_cl_now
                    self._ssg_path_pub.publish(_SearchPath())
        # ★ Phase5: 统一安全限幅 + 贴墙检测（最终发布前调用）
        # 任务1: 独占控制期间跳过 apply_safety_clamp（安全夹已在门控里排除，视觉伺服优先）
        _now_sec_cl = now.nanoseconds * 1e-9
        if not getattr(self, '_approach_exclusive', False):
            vel_cmd = self._apply_safety_clamp(vel_cmd, now, _now_sec_cl)
        self.publish_setpoint(vel_cmd, now_us, yaw=yaw, yawspeed=yawspeed)
        self.log_status(now, vel_cmd, safety_ok=True, safety_reason="ok")

        self.prev_vel = vel_cmd
        self.last_loop_time = now

    def close_log(self) -> None:
        if self.log_file is not None:
            try:
                self.log_file.flush()
                self.log_file.close()
            except Exception:
                pass
        if self._event_log_file is not None:
            try:
                self._log_event("NODE-SHUTDOWN", "target_search_node 正常退出")
                self._event_log_file.flush()
                self._event_log_file.close()
            except Exception:
                pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetSearch()
    atexit.register(node.close_log)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_log()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
