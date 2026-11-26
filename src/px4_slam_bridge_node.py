#!/usr/bin/env python3
"""
px4_slam_bridge_node.py
========================
PX4 SITL → slam_toolbox 桥接节点

功能：
  1. 把 PX4 VehicleLocalPosition (NED: x=北, y=东, z=下)
     转换为 标准 ROS odom ENU： x=East(东)=NED_y, y=North(北)=NED_x, z=Up(上)=-NED_z
  2. 发布动态 TF:  odom → base_link  (位置 + yaw 旋转, 均在 ENU 帧下)
  3. 发布静态 TF:  base_link → laser_link  (LiDAR 安装角补偿)
  4. 将 PX4 激光话题中继到 /scan，frame_id 改为 laser_link
     → 这样 slam_toolbox 通过 TF 可以把激光变换到 odom 系下做 scan matching

坐标系约定（标准 ROS ENU）：
  odom   : x=East(东)=NED_y, y=North(北)=NED_x, z=Up(上)=-NED_z
             右手系： x×y=z ✓ | 与 REP-103、nav2、robot_localization 完全兼容
  base_link : x=机头前方, y=机体左侧, z=向上  (ROS 标准 FLU)
  laser_link: 激光传感器安装系，通过 base_link→laser_link 静态 TF 补偿安装角

yaw 转换公式（NED_heading → ENU_yaw）：
  yaw_enu = π/2 - heading_ned
  heading=0(北) → yaw=π/2(面北, ENU 90°) ✓
  heading=π/2(东, CW) → yaw=0 (面东, ENU 0°) ✓

LiDAR 安装角约定 (lidar_mount_yaw_deg, 默认 0.0°)：
  Gazebo rplidar_link orientation=(0,0,0)，激光x轴与机体x轴对齐（前向），安装偏转 yaw = 0°
  base_link→laser_link TF 旋转 = Rz(lidar_mount_yaw_deg)
"""
import math
import time as _time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

import numpy as np
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from std_msgs.msg import String
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude


def qos_sensor():
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


class Px4SlamBridgeNode(Node):
    def __init__(self):
        super().__init__("px4_slam_bridge")

        # ── 参数 ──────────────────────────────────────────────────────────────
        self.fmu_prefix = self.declare_parameter("fmu_prefix", "/fmu").value
        self.lidar_mount_yaw_deg = self.declare_parameter(
            "lidar_mount_yaw_deg", 0.0
        ).value
        self.lidar_topic_in = self.declare_parameter(
            "lidar_topic_in", "/px4/lidar_controller/out"
        ).value
        self.scan_out_topic = self.declare_parameter("scan_out_topic", "/scan").value
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.laser_frame = self.declare_parameter("laser_frame", "laser_link").value

        # ── scan 双话题参数（A1）──────────────────────────────────────────────
        # scan_raw_topic : 始终发布原始扫描（调试/可视化用）
        # scan_slam_topic: 门控后仅稳定状态时发布（slam_toolbox 输入）
        self.scan_raw_topic  = self.declare_parameter("scan_raw_topic",  "/scan_raw").value
        self.scan_slam_topic = self.declare_parameter("scan_slam_topic", "/scan").value
        # ── 稳定门控参数（A4）────────────────────────────────────────────────
        self._slam_scan_gate_enabled = bool(self.declare_parameter("slam_scan_gate_enabled", True).value)
        self._gate_roll_deg      = float(self.declare_parameter("gate_roll_deg",      5.0).value)
        self._gate_pitch_deg     = float(self.declare_parameter("gate_pitch_deg",     5.0).value)
        self._gate_xy_speed_mps  = float(self.declare_parameter("gate_xy_speed_mps",  0.15).value)
        self._gate_z_speed_mps   = float(self.declare_parameter("gate_z_speed_mps",   0.08).value)
        self._gate_yaw_rate_dps  = float(self.declare_parameter("gate_yaw_rate_dps", 10.0).value)  # ★ Phase7: 15→10dps，减少旋转时误送
        self._gate_hold_sec      = float(self.declare_parameter("gate_hold_sec",      0.3).value)
        # hover_strict 门控阈值——专为 HOVER_SCAN 设计的严格限制 (roll/pitch≤6°, yr≤8dps)
        self._hover_gate_roll_deg    = float(self.declare_parameter("hover_gate_roll_deg",    6.0).value)
        self._hover_gate_pitch_deg   = float(self.declare_parameter("hover_gate_pitch_deg",   6.0).value)
        self._hover_gate_xy_speed_mps= float(self.declare_parameter("hover_gate_xy_speed_mps",0.08).value)
        self._hover_gate_z_speed_mps = float(self.declare_parameter("hover_gate_z_speed_mps", 0.05).value)
        self._hover_gate_yaw_rate_dps= float(self.declare_parameter("hover_gate_yaw_rate_dps", 8.0).value)

        # ── 坐标系约定（标准 ROS ENU，REP-103 兼容）─────────────────────────────
        # odom frame:  x = East（东）= NED_y
        #              y = North（北）= NED_x
        #              z = Up（上）  = -NED_z
        #              右手系：x×y = z（向上）✓
        # base_link:   FLU (x=机头前, y=机头左, z=向上)
        # yaw 来源优先级：
        #   1. VehicleLocalPosition.heading（EKF融合直接输出，~50Hz，首选）
        #   2. VehicleAttitude 四元数解算（100Hz，fallback）
        # yaw 转换公式（NED heading → ENU yaw，必须应用）：
        #   yaw_enu = π/2 - heading_ned
        #   heading=0(北)  → yaw_enu=+π/2 → Rz(π/2) body_x → +y(北) ✓
        #   heading=π/2(东, CW) → yaw_enu=0  → Rz(0)    body_x → +x(东) ✓
        #   heading=π(南)  → yaw_enu=-π/2 → Rz(-π/2) body_x → -y(南) ✓
        # 注意：ENU yaw 是 CCW 正方向；NED heading 是 CW 正方向。
        #       因此必须做 yaw_enu = π/2 - heading_ned，不可省略此翻转。

        # ── 状态 ──────────────────────────────────────────────────────────────
        self._x_enu = 0.0   # ENU x = East（东）
        self._y_enu = 0.0   # ENU y = North（北）
        self._z_ned = 0.0   # 向下为正（保持 NED_z 约定，与 PX4 一致）
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._att_yaw = 0.0   # ENU odom yaw, 0=东(East), 逆时针为正（弧度）
        self._att_q  = np.array([0.0, 0.0, 0.0, 1.0])  # [x,y,z,w] scipy convention
        # ★ yaw 质量标志：True=使用 VehicleLocalPosition.heading（EKF直接输出）
        #                 False=使用 VehicleAttitude 四元数解算
        self._yaw_from_pos = False    # 是否从 pos_cb 的 heading 字段更新
        self._heading_good = False    # VehicleLocalPosition.heading_good_for_control
        # ── yaw 诊断缓存 ──
        self._att_q_yaw_deg: float = 0.0   # 四元数解算的 yaw 角（度），用于对比
        self._pos_heading_deg: float = 0.0  # heading 字段（度），用于对比
        # ── 1.2 YAW-SOURCE-FALLBACK 主动切换状态 ─────────────────────────
        # 持续比较 heading_yaw 与 quat_yaw，差值 >12° 连续 8 帧 → 强制用 quat
        # 连续 20 帧 <5° → 切回 heading（hysteresis 防抖动）
        self._yaw_src_force_quat: bool  = False   # True=强制使用 quat yaw
        self._yaw_mismatch_frames: int  = 0        # 连续超阈值帧数
        self._yaw_match_frames: int     = 0        # 连续恢复(差值<5°)帧数
        self._YAW_MISMATCH_TH_DEG: float = 12.0   # 差值阈值：超过则计数
        self._YAW_MISMATCH_FRAMES: int   = 8       # 超阈值连续帧数 → 强制切换
        self._YAW_RECOVER_FRAMES:  int   = 20      # 恢复帧数 → 切回 heading
        self._bridge_yaw_log_t: float = 0.0        # 上次 [BRIDGE-YAW] 1Hz 打印时刻
        # ── 姿态（tilt_deg = 机体倾角，用于稳定门控；roll/pitch 仅诊断）───────
        self._att_roll:      float = 0.0   # rad（诊断用，不再作为 gate 依据）
        self._att_pitch:     float = 0.0   # rad（诊断用，不再作为 gate 依据）
        self._att_tilt_deg:  float = 0.0   # 机体倾角（°），gate 核心判据
        # ── 稳定门控状态（A2）──────────────────────────────────────────────
        self._slam_scan_gate_open: bool  = not self._slam_scan_gate_enabled
        self._gate_stable_since:   float = -1.0   # monotonic time（稳定开始时刻，-1=无效）
        self._prev_yaw_for_rate:   float = 0.0
        self._prev_yaw_time:       float = 0.0
        self._curr_yaw_rate_dps:   float = 0.0
        self._scan_slam_count: int = 0    # 发送给 SLAM 的帧数
        self._scan_drop_count: int = 0    # 门控丢弃的帧数
        # ── 门控强制发布（A5）────────────────────────────────────────────
        # 若门连续关闭超过 scan_force_interval_sec，强制发布一帧防止 slam_toolbox 饥饿
        self._scan_force_interval_sec = float(self.declare_parameter(
            "scan_force_interval_sec", 3.0).value)
        self._last_slam_scan_time: float = -1.0  # 上次成功将 scan 送入 SLAM 的单调时刻
        self._scan_force_count: int = 0           # 强制发布次数
        # ── 分项拒绝计数器 ────────────────────────────────────────────────
        self._gate_rej_roll:  int = 0
        self._gate_rej_pitch: int = 0
        self._gate_rej_vxy:   int = 0
        self._gate_rej_vz:    int = 0
        self._gate_rej_yr:    int = 0
        # ── 外部门控模式（A6）──────────────────────────────────────────────
        # 由 target_search_node 通过 /slam_gate_mode 话题控制：
        #   "normal"      – 正常门控逻辑
        #   "force_open"  – 强制打开（HOVER_SCAN 期间用）
        self._gate_mode: str = "normal"
        # ── Phase7: 每秒 SCAN-TF-DIAG 节流时刻 ─────────────────────────────
        self._scan_diag_last_t: float = 0.0

        # ── TF ───────────────────────────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_laser_tf()

        # ── 发布 ──────────────────────────────────────────────────────────────
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        # scan 发布用 BEST_EFFORT，与 slam_toolbox 订阅 QoS 匹配（避免 Fast DDS QoS 兼容问题）
        from rclpy.qos import QoSProfile, ReliabilityPolicy as RP, DurabilityPolicy as DP, HistoryPolicy as HP
        qos_scan_pub = QoSProfile(
            reliability=RP.BEST_EFFORT,
            durability=DP.VOLATILE,
            history=HP.KEEP_LAST,
            depth=10,
        )
        # ★ 双话题发布器（A1）：slam=门控，raw=始终
        self._scan_pub     = self.create_publisher(LaserScan, self.scan_slam_topic, qos_scan_pub)
        self._scan_raw_pub = self.create_publisher(LaserScan, self.scan_raw_topic,  qos_scan_pub)

        # ── 订阅 ──────────────────────────────────────────────────────────────
        self.create_subscription(
            VehicleLocalPosition,
            f"{self.fmu_prefix}/out/vehicle_local_position_v1",
            self._pos_cb,
            qos_sensor(),
        )
        self.create_subscription(
            VehicleAttitude,
            f"{self.fmu_prefix}/out/vehicle_attitude",
            self._att_cb,
            qos_sensor(),
        )
        # ── 外部门控模式订阅（A6）──────────────────────────────────────
        self.create_subscription(
            String, "/slam_gate_mode",
            self._gate_mode_cb, 10
        )
        # 激光扫描中继（允许配置 fallback）
        fallback = self.declare_parameter("lidar_topic_fallback", "").value
        self.create_subscription(
            LaserScan, self.lidar_topic_in,
            lambda m: self._scan_cb(m), 10
        )
        if fallback and fallback != self.lidar_topic_in:
            self.create_subscription(
                LaserScan, fallback,
                lambda m: self._scan_cb(m), 10
            )

        # ── TF chain watchdog：10s 未收到 position/attitude 则报警 ──────────
        import time as _wt
        self._last_pos_time: float = 0.0      # monotonic time of last pos_cb
        self._last_scan_time: float = 0.0     # monotonic time of last scan_cb
        self._last_tf_time: float = 0.0       # monotonic time of last TF publish
        # ★ TF-scan 时间戳对齐缓存
        # scan_cb 优先使用最近一次 TF 发布时的 stamp，保证 slam_toolbox 在 TF 缓存中
        # 能精确命中对应的 odom→base_link 帧，消除 MessageFilter timestamp mismatch 告警
        self._last_tf_stamp_msg = None        # builtin_interfaces/Time，由 _publish_odom_tf 更新
        self._last_tf_stamp_mono: float = 0.0  # monotonic time，用于判断 TF stamp 新鲜度（<200ms）
        self._pos_recv_count: int = 0
        self._scan_recv_count: int = 0
        self._scan_hz_t0: float = _wt.monotonic()  # ★ Hz 计时起点
        self._watchdog_warned: bool = False
        # 记录节点启动的挂钟时间，用于 watchdog 冷却（避免 sim_time 追赶触发误报）
        self._node_start_wall: float = _wt.monotonic()

        # ── TF 跳变检测状态 ─────────────────────────────────────────────────
        self._prev_x_enu: float = 0.0      # 上一帧发布的 x（ENU East）
        self._prev_y_enu: float = 0.0      # 上一帧发布的 y（ENU North）
        self._prev_z_ned: float = 0.0      # 上一帧发布的 z（向下为正）
        self._tf_jump_count: int = 0        # 累计跳变次数
        self._tf_jump_th_xy: float = 1.0   # XY 跳变阈值（m）→ 超过则记录 [TF-JUMP]
        self._tf_jump_th_z:  float = 1.0   # Z 跳变阈值（m）
        self._tf_max_xy_jump: float = 0.0  # 历史最大 XY 跳变量（用于诊断）
        self._tf_first_pub: bool = True    # 首次发布标志（跳过跳变检测）
        # ── TF-FREEZE：大幅跳变时冻结 TF，防止 slam_toolbox 积分错位扫描 ────────
        # 当 Δxy > _tf_freeze_th_m 时，在 _tf_freeze_max_frames 帧内持续使用
        # 冻结位置发布 TF，同时强制关闭扫描门控（无论 gate_mode 如何）。
        # 超时后接受新位置，或者新位置回落到冻结点 1m 内时立即解冻。
        self._tf_freeze_th_m: float = float(
            self.declare_parameter("tf_freeze_th_m", 3.0).value
        )
        self._tf_freeze_max_frames: int = int(
            self.declare_parameter("tf_freeze_max_frames", 50).value
        )
        self._tf_freeze_active: bool = False        # 当前是否处于冻结状态
        self._tf_freeze_frame_cnt: int = 0          # 冻结持续帧数计数器
        self._tf_freeze_pos: tuple = (0.0, 0.0, 0.0)  # (x_enu, y_enu, z_ned) 冻结位置
        self._tf_freeze_count: int = 0              # 历史冻结次数（诊断）

        self.create_timer(10.0, self._watchdog_cb)

        self.get_logger().info(
            f"[BRIDGE] 初始化完成: "
            f"LiDAR {self.lidar_topic_in} → slam:{self.scan_slam_topic} / raw:{self.scan_raw_topic} "
            f"frame_id={self.laser_frame} mount_yaw={self.lidar_mount_yaw_deg}° "
            f"odom={self.odom_frame} base={self.base_frame}"
        )
        self.get_logger().info(
            f"[SCAN-GATE] 配置: enabled={self._slam_scan_gate_enabled} "
            f"roll<{self._gate_roll_deg}° pitch<{self._gate_pitch_deg}° "
            f"vxy<{self._gate_xy_speed_mps}m/s vz<{self._gate_z_speed_mps}m/s "
            f"yr<{self._gate_yaw_rate_dps}dps hold={self._gate_hold_sec}s"
        )
        # ── ENU 坐标系标准化启动证明 (按 REP-103) ─────────────────────────────
        # yaw_enu = π/2 - heading_ned： heading_ned=0(北) → yaw_enu=π/2 (ENU 90°)，面北时 x→东 y→北 ✓
        _test_heading_ned_0 = 0.0      # 面北 (NED heading = 0)
        _yaw_enu_0 = math.pi / 2.0 - _test_heading_ned_0
        _test_q = R.from_euler("z", _yaw_enu_0).as_quat()
        _test_body_x = R.from_quat(_test_q).apply([1.0, 0.0, 0.0])
        self.get_logger().info(
            f"[BRIDGE-COORD] 坐标系: 标准 ROS ENU (x=东, y=北, z=上)\n"
            f"  PX4 NED 映射到 odom ENU: odom.x=NED_y(东), odom.y=NED_x(北), odom.z=-NED_z(上)\n"
            f"  yaw 公式: yaw_enu = π/2 - heading_ned\n"
            f"  验证: heading_ned=0(面北) → yaw_enu=π/2 → body_x方向=({_test_body_x[0]:.3f},{_test_body_x[1]:.3f},{_test_body_x[2]:.3f})"
            f" 应为(0,1,0)=北=ENU_y ✓  | REP-103 完全合规"
        )
        self.get_logger().info(
            f"[BRIDGE-YAW] yaw 来源策略: "
            f"首选 VehicleLocalPosition.heading (EKF直接输出, heading_good_for_control=True), "
            f"降级 VehicleAttitude四元数(zyx Euler)。"
            f"验证命令: ros2 run tf2_ros tf2_echo odom base_link"
        )
        self.get_logger().info(
            f"[BRIDGE-FRAME] odom=ENU: x=East=NED_y, y=North=NED_x, z=Up=-NED_z  "
            f"yaw_formula=pi/2-heading_ned  frame={self.odom_frame}  "
            f"PX4 NED→odom ENU: odom.x=vlp.y, odom.y=vlp.x, odom.z=-vlp.z, "
            f"quat=pi/2-vlp.heading"
        )
        # ── 启动参数快照文件（新增）──────────────────────────────────
        # 与 target_search_node 的 params_*.txt 配套，方便对比两个节点的配置
        _log_dir = Path("/home/ubuntu/PX4-Autopilot/log")
        _log_dir.mkdir(parents=True, exist_ok=True)
        _ts = int(_time.time())
        _bp_path = _log_dir / f"bridge_params_{_ts}.txt"
        self._bridge_diag_path = _log_dir / "bridge_diag_latest.txt"
        try:
            _bp_lines = [
                f"# px4_slam_bridge_node 启动参数快照  ts={_ts}\n",
                f"# 生成时间: {_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                f"fmu_prefix={self.fmu_prefix}\n",
                f"lidar_topic_in={self.lidar_topic_in}\n",
                f"scan_out_topic={self.scan_out_topic}\n",
                f"lidar_mount_yaw_deg={self.lidar_mount_yaw_deg}\n",
                f"odom_frame={self.odom_frame}\n",
                f"base_frame={self.base_frame}\n",
                f"laser_frame={self.laser_frame}\n\n",
                f"[坐标系约定]\n",
                f"  odom: x=东(East=NED_y), y=北(North=NED_x), z=上(-NED_z) (标准ROS ENU)\n",
                f"  yaw_enu = pi/2 - heading_ned (0=东, CCW正, REP-103)\n\n",
                f"[诊断文件路径]\n",
                f"  参数快照: {_bp_path}\n",
                f"  实时诊断: {self._bridge_diag_path}  (每10s更新)\n",
            ]
            with _bp_path.open("w", encoding="utf-8") as _f:
                _f.writelines(_bp_lines)
            self.get_logger().info(
                f"[BRIDGE-DIAG] 参数快照: {_bp_path}\n"
                f"  实时诊断: {self._bridge_diag_path}  (tail -f 监看)"
            )
        except Exception as _e:
            self.get_logger().warn(f"[BRIDGE-DIAG] 参数快照失败: {_e}")
            self._bridge_diag_path = None

        # ── 启动身份日志（同时追加写入 log/bridge_params_*.txt，满足「日志输出到log目录」要求）──
        import os as _bof, hashlib as _bfid_hl, time as _bfid_t
        _bfid_data = open(__file__, 'rb').read()
        _bfid_sha = _bfid_hl.sha256(_bfid_data).hexdigest()[:16]
        _bfid_started = _bfid_t.strftime('%Y-%m-%dT%H:%M:%S')
        _bf_line = (
            f"[FILE-ID] ★ node=px4_slam_bridge  file={_bof.path.abspath(__file__)}  "
            f"sha256={_bfid_sha}  started={_bfid_started}  version=2026-03-01_frame_unified"
        )
        self.get_logger().info(
            _bf_line
            + f" | odom={self.odom_frame} base={self.base_frame} laser={self.laser_frame}"
            + f" | coord=ENU(x=East,y=North,z=Up) | mount_yaw={self.lidar_mount_yaw_deg}deg"
            + f" | tf_freeze_th={self._tf_freeze_th_m}m freeze_max_frames={self._tf_freeze_max_frames}"
        )
        # 同时追加到 bridge_params 文件（log/ 目录下持久化）
        try:
            with _bp_path.open("a", encoding="utf-8") as _bf:
                _bf.write(f"\n{_bf_line}\n")
        except Exception:
            pass

    # ── 静态 TF: base_link → laser_link ──────────────────────────────────────
    def _publish_static_laser_tf(self):
        """
        安装角补偿：将激光传感器坐标系与机体系对齐。
        Gazebo rplidar_link orientation=(0,0,0): 激光x轴=机体前向x轴，mount_yaw=0°即单位TF。
        若激光x轴指向机体右侧(+Y_body)，则应设置 mount_yaw=-90°。
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_frame
        t.child_frame_id  = self.laser_frame
        # 平移量：假设传感器在机体中心（实际偏置可通过参数扩展）
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        # 旋转：绕 z 轴旋转 mount_yaw
        mount_rad = math.radians(float(self.lidar_mount_yaw_deg))
        rot = R.from_euler("z", mount_rad)
        q   = rot.as_quat()   # [x, y, z, w]
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self._static_tf_broadcaster.sendTransform(t)
        self.get_logger().info(
            f"[BRIDGE] 静态 TF 已发布: {self.base_frame}→{self.laser_frame} "
            f"Rz({self.lidar_mount_yaw_deg}°)"
        )
        # ★ Phase7: MOUNT-CHECK 启动自检日志 ─────────────────────────────────
        _mount_rad2 = math.radians(float(self.lidar_mount_yaw_deg))
        _body_x_in_laser = (math.cos(-_mount_rad2), math.sin(-_mount_rad2))
        _direction_str = (
            "body.x(前) → laser.x(前)" if abs(self.lidar_mount_yaw_deg) < 5 else
            "body.x(前) → laser.y(左)" if abs(self.lidar_mount_yaw_deg - 90) < 5 else
            "body.x(前) → laser.x(后)" if abs(abs(self.lidar_mount_yaw_deg) - 180) < 5 else
            f"body.x(前) → laser({_body_x_in_laser[0]:+.2f},{_body_x_in_laser[1]:+.2f})"
            if abs(self.lidar_mount_yaw_deg + 90) > 5 else
            "body.x(前) → laser.(-y)(右)"
        )
        self.get_logger().info(
            f"[MOUNT-CHECK] TF链路: odom→{self.base_frame}→{self.laser_frame} "
            f"安装角={self.lidar_mount_yaw_deg}° "
            f"旋转方向: {_direction_str} | "
            f"门控: yaw_rate<{self._gate_yaw_rate_dps}dps "
            f"vxy<{self._gate_xy_speed_mps}m/s "
            f"roll<{self._gate_roll_deg}° pitch<{self._gate_pitch_deg}° | "
            f"TF时效窗口: 50ms (Phase7)"
        )

    # ── PX4 位置回调 ──────────────────────────────────────────────────────────
    def _pos_cb(self, msg: VehicleLocalPosition):
        import time as _wt
        if not (msg.xy_valid and msg.z_valid):
            return
        # ENU: x=东(East)=NED_y, y=北(North)=NED_x
        self._x_enu = float(msg.y)   # East = NED_y
        self._y_enu = float(msg.x)   # North = NED_x
        self._z_ned = float(msg.z)   # NED_z 保持不变（向下为正）
        self._vx    = float(msg.vx)
        self._vy    = float(msg.vy)
        self._vz    = float(msg.vz)
        self._pos_recv_count += 1
        self._last_pos_time = _wt.monotonic()
        # ★ ~1 Hz NED→ENU 诊断日志（每50帧，位置回调约50Hz）
        if self._pos_recv_count % 50 == 0:
            self.get_logger().info(
                f"[BRIDGE-NED\u2192ENU] NED=(N={msg.x:.2f},E={msg.y:.2f},D={msg.z:.2f}) "
                f"\u2192 ENU=(East={self._x_enu:.2f},North={self._y_enu:.2f},Up={-self._z_ned:.2f}) "
                f"[ENU.x=NED.y  ENU.y=NED.x  ENU.z=-NED.z]"
            )

        # ★ 优先使用 VehicleLocalPosition.heading（EKF 融合直接输出）作为 yaw 来源
        # 相比四元数解算，heading 字段：
        #   1. 直接由 EKF 输出，无需 Euler 分解（避免万向锁、符号模糊问题）
        #   2. 包含 heading_good_for_control 质量标志
        #   3. 数值约定：-π..+π，0=北，正 = 顺时针（东），与 NED heading 一致
        #
        # ENU odom(x=东, y=北, z=上)中的 yaw 转换：yaw_enu = π/2 - NED_heading
        #   heading_ned=0(北)  → yaw_enu=π/2 (面北, ENU 90°) ✓
        #   heading_ned=π/2(东, CW) → yaw_enu=0 (面东, ENU 0°) ✓
        heading_raw = float(getattr(msg, 'heading', float('nan')))
        heading_good = bool(getattr(msg, 'heading_good_for_control', False))
        if np.isfinite(heading_raw) and heading_good:
            _heading_yaw_enu = math.pi / 2.0 - heading_raw   # NED heading → ENU yaw
            # ── 1.2 主动比较 heading_yaw 与 quat_yaw，差值大时强制降级到 quat ──
            _h_deg = float(np.rad2deg(_heading_yaw_enu))
            _q_deg = float(self._att_q_yaw_deg)   # 由 _att_cb 实时更新
            _diff_deg = abs(((_h_deg - _q_deg + 180.0) % 360.0) - 180.0)
            if _diff_deg > self._YAW_MISMATCH_TH_DEG:
                self._yaw_mismatch_frames += 1
                self._yaw_match_frames = 0
                if self._yaw_mismatch_frames >= self._YAW_MISMATCH_FRAMES and not self._yaw_src_force_quat:
                    self._yaw_src_force_quat = True
                    self.get_logger().warn(
                        f"[YAW-SOURCE-FALLBACK] heading/quat 持续偏差 {_diff_deg:.1f}° "
                        f"(>{self._YAW_MISMATCH_TH_DEG}° × {self._yaw_mismatch_frames}帧) "
                        f"→ 强制切换到 att_quat yaw | "
                        f"heading_yaw={_h_deg:.1f}° quat_yaw={_q_deg:.1f}° new_source=att_quat"
                    )
                    self.get_logger().warn(
                        f"[YAW-SWITCH] reason=heading_quat_mismatch src=att_quat "
                        f"diff={_diff_deg:.1f}deg"
                    )
            else:
                self._yaw_match_frames += 1
                self._yaw_mismatch_frames = 0
                if self._yaw_src_force_quat and self._yaw_match_frames >= self._YAW_RECOVER_FRAMES:
                    self._yaw_src_force_quat = False
                    self.get_logger().info(
                        f"[YAW-SOURCE-FALLBACK] heading/quat 差值恢复 <5° × {self._yaw_match_frames}帧 "
                        f"→ 切回 heading(EKF) yaw | heading_yaw={_h_deg:.1f}° quat_yaw={_q_deg:.1f}°"
                    )
                    self.get_logger().info(
                        f"[YAW-SWITCH] reason=mismatch_recovered src=heading_enu "
                        f"match_frames={self._yaw_match_frames}"
                    )
            # 根据降级状态决定用哪个 yaw
            if self._yaw_src_force_quat:
                # _att_q_yaw_deg 已是 ENU yaw deg（_att_cb 用四元数算出）
                self._att_yaw = math.radians(_q_deg)
                # att_q 已由 _att_cb 写好，保持不变
            else:
                self._att_yaw = _heading_yaw_enu
                q_yaw = R.from_euler("z", self._att_yaw).as_quat()
                self._att_q = q_yaw
            self._yaw_from_pos = True
            self._heading_good = True
            self._pos_heading_deg = float(np.rad2deg(heading_raw))
        else:
            # heading 字段无效时降级到四元数解算（_att_cb 已更新 _att_yaw）
            self._yaw_from_pos = False
            self._heading_good = False
            self._yaw_mismatch_frames = 0

        self._publish_odom_tf()

    # ── PX4 姿态回调 ──────────────────────────────────────────────────────────
    def _gate_mode_cb(self, msg: String) -> None:
        """接收 target_search_node 发来的门控模式指令。
        'hover_strict' – HOVER_SCAN 期间宽松门控（推荐）
        'force_open'   – 强制打开（仅调试用，target_search 已自动降级为 hover_strict）
        'normal'       – 恢复正常门控逻辑
        """
        new_mode = msg.data.strip()
        if new_mode != self._gate_mode:
            self.get_logger().info(
                f"[GATE-MODE] {self._gate_mode!r} → {new_mode!r}"
            )
            self._gate_mode = new_mode
            # [GATE-THRESH] 输出新模式下的门控阈值，便于日志核查
            if new_mode == "hover_strict":
                _gt_r, _gt_p = self._hover_gate_roll_deg, self._hover_gate_pitch_deg
                _gt_vxy, _gt_vz = self._hover_gate_xy_speed_mps, self._hover_gate_z_speed_mps
                _gt_yr = self._hover_gate_yaw_rate_dps
            else:
                _gt_r, _gt_p = self._gate_roll_deg, self._gate_pitch_deg
                _gt_vxy, _gt_vz = self._gate_xy_speed_mps, self._gate_z_speed_mps
                _gt_yr = self._gate_yaw_rate_dps
            self.get_logger().info(
                f"[GATE-THRESH] mode={new_mode} "
                f"roll≤{_gt_r:.1f}° pitch≤{_gt_p:.1f}° "
                f"vxy≤{_gt_vxy:.2f}m/s vz≤{_gt_vz:.2f}m/s "
                f"yr≤{_gt_yr:.1f}dps"
            )
            if new_mode == "force_open":
                # 立即打开门（降级后不应走到这里，但兼容旧调用者）
                self._slam_scan_gate_open = True
                self._gate_stable_since = -1.0
            elif new_mode == "normal":
                # 切回 normal：重置稳定计时，让门控重新判断
                self._gate_stable_since = -1.0

    def _att_cb(self, msg: VehicleAttitude):
        import time as _wt_att
        # PX4 四元数 [w, x, y, z]；scipy 是 [x, y, z, w]
        q_px4 = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]])
        rot = R.from_quat(q_px4)

        # ── ★ 计算倾角 tilt_deg（与坐标系约定无关的稳定姿态指标）──────────
        # 机体 z 轴（FRD 下方向 = [0,0,1]）在世界 NED 的方向
        # 水平飞行时 body_z_ned = [0,0,1]（对齐世界 NED_down），tilt≈0°
        # tilt = body_z_ned 与 world_down [0,0,1] 的夹角
        # 此方法与 NED/ENU、FRD/FLU 坐标约定无关，不依赖 Euler 分解顺序
        body_z_ned = rot.apply(np.array([0.0, 0.0, 1.0]))
        _cz = float(np.clip(body_z_ned[2], -1.0, 1.0))
        _tilt_deg = float(math.degrees(math.acos(abs(_cz))))
        self._att_tilt_deg = _tilt_deg  # 保存供门控状态查询

        try:
            # 'zyx' Euler: [yaw, pitch, roll]（内在旋转）
            yaw_q, pitch, roll = rot.as_euler("zyx")
            self._att_q_yaw_deg = float(np.rad2deg(math.pi / 2.0 - yaw_q))  # ENU yaw（诊断）
            # 保留 roll/pitch 用于诊断对比，不再作为 gate 决策依据
            self._att_roll  = float(roll)
            self._att_pitch = float(pitch)
        except Exception:
            yaw_q = self._att_yaw  # fallback

        # ★ 仅在 heading 字段不可用时，用四元数 yaw 作为 TF yaw
        # 若 pos_cb 已用 heading 字段更新了 _att_yaw，这里不覆盖（heading 质量更高）
        if not self._heading_good:
            self._att_yaw = float(math.pi / 2.0 - yaw_q)  # NED→ENU yaw（fallback）
            # slam_toolbox 2D 只关心 yaw，用 yaw-only 四元数即可
            q_yaw_only = R.from_euler("z", self._att_yaw).as_quat()  # [x,y,z,w]
            self._att_q = q_yaw_only

        # ── yaw_rate 估算（差分，处理 wrap）───────────────────────────────────
        _now_att = _wt_att.monotonic()
        _dt_yaw = _now_att - self._prev_yaw_time
        if 0.002 < _dt_yaw < 1.0:
            _dyaw = self._att_yaw - self._prev_yaw_for_rate
            while _dyaw >  math.pi: _dyaw -= 2.0 * math.pi
            while _dyaw < -math.pi: _dyaw += 2.0 * math.pi
            self._curr_yaw_rate_dps = abs(math.degrees(_dyaw / _dt_yaw))
        self._prev_yaw_for_rate = float(self._att_yaw)
        self._prev_yaw_time = _now_att

        # ── 稳定门控状态机更新（★ 改用 tilt_deg 取代 raw roll/pitch）─────────
        # ★ 修复：用"机体倾角 tilt_deg"替代 raw Euler roll/pitch 做 gate 判定。
        #   原因：roll/pitch 值在 NED/FRD → ENU/FLU 转换下轴意义不稳定，
        #   Euler 分解顺序不同会导致 roll/pitch 在同等倾斜下数值完全不同。
        #   tilt_deg = body_z 与世界竖直方向夹角，与坐标系约定和分解顺序无关，
        #   始终准确反映真实机身倾斜量，用于 gate 更稳健。
        if self._slam_scan_gate_enabled and self._gate_mode not in ("force_open",):
            # 将原 roll/pitch 独立阈值折算为等效倾角阈值（RSS 合并）
            # hover_strict 或 normal 模式——选择门控阈值
            if self._gate_mode == 'hover_strict':
                # 原 roll<6° & pitch<6° → 等效 tilt < sqrt(6²+6²) ≈ 8.5°
                _tilt_thr_deg = math.sqrt(
                    self._hover_gate_roll_deg ** 2 + self._hover_gate_pitch_deg ** 2
                )
                _vxy_thr = self._hover_gate_xy_speed_mps
                _vz_thr  = self._hover_gate_z_speed_mps
                _yr_thr  = self._hover_gate_yaw_rate_dps
            else:  # normal
                _tilt_thr_deg = math.sqrt(
                    self._gate_roll_deg ** 2 + self._gate_pitch_deg ** 2
                )
                _vxy_thr = self._gate_xy_speed_mps
                _vz_thr  = self._gate_z_speed_mps
                _yr_thr  = self._gate_yaw_rate_dps

            _tilt_ok  = _tilt_deg                     < _tilt_thr_deg
            _vxy_ok   = math.hypot(self._vx, self._vy) < _vxy_thr
            _vz_ok    = abs(self._vz)                   < _vz_thr
            _yr_ok    = self._curr_yaw_rate_dps         < _yr_thr
            _all_ok   = _tilt_ok and _vxy_ok and _vz_ok and _yr_ok

            # hover_strict：每 2s 输出一次门控状态日志（tilt 格式）
            if self._gate_mode == 'hover_strict':
                if _now_att - getattr(self, '_hover_gate_log_t', 0.0) > 2.0:
                    self._hover_gate_log_t = _now_att
                    self.get_logger().info(
                        f"[SCAN-GATE-HOVER] open={self._slam_scan_gate_open} "
                        f"tilt={_tilt_deg:.1f}°/{_tilt_thr_deg:.1f}° "
                        f"vxy={math.hypot(self._vx,self._vy):.2f}/{_vxy_thr:.2f}m/s "
                        f"yr={self._curr_yaw_rate_dps:.1f}/{_yr_thr:.0f}dps "
                        f"all_ok={_all_ok} "
                        f"(diag roll={math.degrees(abs(self._att_roll)):.1f}° "
                        f"pitch={math.degrees(abs(self._att_pitch)):.1f}°)"
                    )
            if _all_ok:
                if not self._slam_scan_gate_open:
                    if self._gate_stable_since < 0.0:
                        self._gate_stable_since = _now_att
                    elif _now_att - self._gate_stable_since >= self._gate_hold_sec:
                        self._slam_scan_gate_open = True
            else:
                self._slam_scan_gate_open = False
                self._gate_stable_since = -1.0
                # ★ 分项拒绝计数（tilt 拒绝复用 roll 计数器）
                if not _tilt_ok:  self._gate_rej_roll  += 1
                if not _vxy_ok:   self._gate_rej_vxy   += 1
                if not _vz_ok:    self._gate_rej_vz    += 1
                if not _yr_ok:    self._gate_rej_yr    += 1
        else:
            # gate disabled 或 force_open 模式 → 直接打开
            self._slam_scan_gate_open = True

        self._publish_odom_tf()

    # ── 发布 odom→base_link TF 和 /odom 消息 ─────────────────────────────────
    def _publish_odom_tf(self):
        import time as _wt
        now = self.get_clock().now()

        # ── TF-FREEZE：大幅位置跳变检测 + 冻结处理 ───────────────────────────
        # 核心思路：EKF 大幅跳变时，继续发布"冻结"位置的 TF（保持 slam_toolbox 定位稳定），
        # 同时强制关闭扫描门控（不让错位扫描进入 slam_toolbox），直至跳变恢复或超时。
        _pub_x = self._x_enu
        _pub_y = self._y_enu
        _pub_z = self._z_ned

        if not self._tf_first_pub:
            _dxy = float(np.hypot(self._x_enu - self._prev_x_enu,
                                  self._y_enu - self._prev_y_enu))
            _dz  = float(abs(self._z_ned - self._prev_z_ned))

            # ── 大幅跳变：激活 TF-FREEZE ────────────────────────────────────
            if _dxy > self._tf_freeze_th_m:
                self._tf_jump_count += 1
                self._tf_max_xy_jump = max(self._tf_max_xy_jump, _dxy)
                if not self._tf_freeze_active:
                    # 首次激活：记录冻结位置（上一帧的良好位置）
                    self._tf_freeze_active = True
                    self._tf_freeze_frame_cnt = 0
                    self._tf_freeze_pos = (self._prev_x_enu, self._prev_y_enu, self._prev_z_ned)
                    self._tf_freeze_count += 1
                    self.get_logger().warn(
                        f"[TF-FREEZE-ENTER] #{self._tf_freeze_count} 大幅跳变 Δxy={_dxy:.2f}m > "
                        f"{self._tf_freeze_th_m:.1f}m → 冻结 TF+关闭扫描门控 "
                        f"(max {self._tf_freeze_max_frames} 帧) "
                        f"frozen_pos=({self._prev_x_enu:.2f},{self._prev_y_enu:.2f}) "
                        f"jumped_to=({self._x_enu:.2f},{self._y_enu:.2f})"
                    )
                    # [TF-FREEZE-ESCALATE]: 多次冻结说明位姿估计器反复不稳定
                    if self._tf_freeze_count % 5 == 0:
                        self.get_logger().error(
                            f"[TF-FREEZE-ESCALATE] 冻结次数达到 {self._tf_freeze_count} "
                            f"→ estimator 可能反复不稳定，建议检查 EKF2 模块状态"
                        )
                else:
                    # 持续跳变：继续冻结
                    pass
            elif _dxy > self._tf_jump_th_xy or _dz > self._tf_jump_th_z:
                # 普通小幅跳变（仅记录日志，不触发冻结）
                self._tf_jump_count += 1
                self._tf_max_xy_jump = max(self._tf_max_xy_jump, _dxy)
                self.get_logger().warn(
                    f"[TF-JUMP] #{self._tf_jump_count} "
                    f"Δxy={_dxy:.3f}m Δz={_dz:.3f}m "
                    f"prev=({self._prev_x_enu:.2f},{self._prev_y_enu:.2f}) "
                    f"curr=({self._x_enu:.2f},{self._y_enu:.2f}) "
                    f"max_xy_jump={self._tf_max_xy_jump:.2f}m"
                )

            # ── TF-FREEZE 状态处理 ────────────────────────────────────────
            if self._tf_freeze_active:
                # 检查是否可以提前解冻（位置回落到冻结点 1m 内 + 连续稳定 30 帧）
                _frozen_x, _frozen_y, _frozen_z = self._tf_freeze_pos
                _dist_from_freeze = float(np.hypot(
                    self._x_enu - _frozen_x, self._y_enu - _frozen_y
                ))
                self._tf_freeze_frame_cnt += 1
                # 追踪连续稳定帧数（位置变化 < 0.15m）
                _prev_freeze_chk = getattr(self, '_tf_freeze_stable_frames', 0)
                if _dist_from_freeze < 1.0:
                    self._tf_freeze_stable_frames = _prev_freeze_chk + 1
                else:
                    self._tf_freeze_stable_frames = 0   # 不稳定时重置

                _stable_enough = getattr(self, '_tf_freeze_stable_frames', 0) >= 30

                if _dist_from_freeze < 1.0 and _stable_enough:
                    # 位置回落且连续稳定 30 帧 → 解冻
                    self._tf_freeze_active = False
                    self._tf_freeze_stable_frames = 0
                    self.get_logger().info(
                        f"[TF-FREEZE-EXIT] reason=pos_recovered_stable dist={_dist_from_freeze:.2f}m < 1.0m"
                        f" stable_frames={getattr(self,'_tf_freeze_stable_frames',0)}"
                        f" frames={self._tf_freeze_frame_cnt} freeze_count={self._tf_freeze_count}"
                    )
                    _pub_x = self._x_enu
                    _pub_y = self._y_enu
                    _pub_z = self._z_ned
                elif _dist_from_freeze < 1.0 and not _stable_enough:
                    # 位置接近冻结点但稳定帧数不足，继续冻结并等待
                    _pub_x = _frozen_x
                    _pub_y = _frozen_y
                    _pub_z = _frozen_z
                    self._slam_scan_gate_open = False
                    if self._tf_freeze_frame_cnt % 10 == 1:
                        self.get_logger().debug(
                            f"[TF-FREEZE-HOLD] 等待稳定 stable_frames={getattr(self,'_tf_freeze_stable_frames',0)}/30 "
                            f"dist={_dist_from_freeze:.2f}m frames={self._tf_freeze_frame_cnt}"
                        )
                elif self._tf_freeze_frame_cnt >= self._tf_freeze_max_frames:
                    # 超时 → 接受新位置
                    self._tf_freeze_active = False
                    self._tf_freeze_stable_frames = 0
                    self.get_logger().warn(
                        f"[TF-FREEZE-EXIT] reason=timeout frames={self._tf_freeze_frame_cnt}"
                        f" new_pos=({self._x_enu:.2f},{self._y_enu:.2f})"
                        f" dist_from_freeze={_dist_from_freeze:.2f}m"
                    )
                    _pub_x = self._x_enu
                    _pub_y = self._y_enu
                    _pub_z = self._z_ned
                else:
                    # 仍在冻结中：使用冻结位置 + 强制关闭扫描门控
                    _pub_x = _frozen_x
                    _pub_y = _frozen_y
                    _pub_z = _frozen_z
                    # 强制关闭扫描门控（覆盖 force_open 模式）
                    self._slam_scan_gate_open = False
        else:
            self._tf_first_pub = False

        # 更新上一帧位置（使用实际ENU位置，而非冻结位置，以便下帧计算真实delta）
        self._prev_x_enu = self._x_enu
        self._prev_y_enu = self._y_enu
        self._prev_z_ned = self._z_ned

        # ── TF: odom → base_link（使用 _pub_x/y/z，冻结期间为冻结坐标）────────
        # 坐标约定（标准 ROS ENU）：
        #   odom frame: x = East = NED_y,  y = North = NED_x,  z = Up = -NED_z
        #   yaw_enu = π/2 - heading_ned（已在 _pos_cb / _att_cb 中换算）
        #   四元数 = R.from_euler("z", yaw_enu)，yaw=0 → body_x 朝东 ✓
        t = TransformStamped()
        t.header.stamp    = now.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id  = self.base_frame
        t.transform.translation.x = _pub_x
        t.transform.translation.y = _pub_y
        t.transform.translation.z = -_pub_z   # NED z→上
        q = self._att_q
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self._tf_broadcaster.sendTransform(t)
        self._last_tf_time = _wt.monotonic()
        self._tf_pub_count = getattr(self, '_tf_pub_count', 0) + 1
        # ★ 缓存 TF stamp 供 scan_cb 使用（精确对齐，消除 MessageFilter 时间戳偏差）
        self._last_tf_stamp_msg   = t.header.stamp
        self._last_tf_stamp_mono  = self._last_tf_time


        if self._tf_pub_count % 200 == 1:
            yaw_src = "heading(EKF)" if self._heading_good else "att_quat(fallback)"
            yaw_deg = float(np.rad2deg(self._att_yaw))
            q_yaw_deg = self._att_q_yaw_deg
            heading_deg = self._pos_heading_deg
            # 检查 yaw 一致性：heading 与四元数差值 > 5° 时报警
            yaw_diff = abs(((yaw_deg - q_yaw_deg + 180) % 360) - 180)
            yaw_warn = f"  ⚠ yaw_diff={yaw_diff:.1f}°>5° 检查姿态链路!" if yaw_diff > 5.0 and self._heading_good else ""
            _freeze_str = (
                f"  [FREEZE #{self._tf_freeze_frame_cnt}/{self._tf_freeze_max_frames}]"
                if self._tf_freeze_active else ""
            )
            self.get_logger().info(
                f"[BRIDGE-TF] #{self._tf_pub_count} "
                f"actual=({self._x_enu:.2f},{self._y_enu:.2f},{-self._z_ned:.2f}m) "
                f"pub=({_pub_x:.2f},{_pub_y:.2f},{-_pub_z:.2f}m) "
                f"yaw={yaw_deg:.1f}° src={yaw_src} "
                f"heading_field={heading_deg:.1f}° "
                f"q_yaw={q_yaw_deg:.1f}°"
                f"{yaw_warn}{_freeze_str}"
            )

        # ── /odom 消息（与 TF 保持一致：冻结期间也用冻结坐标）────────────────
        odom = Odometry()
        odom.header.stamp    = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id  = self.base_frame
        odom.pose.pose.position.x = _pub_x
        odom.pose.pose.position.y = _pub_y
        odom.pose.pose.position.z = -_pub_z   # NED z→上
        odom.pose.pose.orientation.x = float(q[0])
        odom.pose.pose.orientation.y = float(q[1])
        odom.pose.pose.orientation.z = float(q[2])
        odom.pose.pose.orientation.w = float(q[3])
        # 速度（NED→ENU: x=East=NED_vy, y=North=NED_vx, z=Up=-NED_vz）
        odom.twist.twist.linear.x = self._vy   # East = NED_vy
        odom.twist.twist.linear.y = self._vx   # North = NED_vx
        odom.twist.twist.linear.z = -self._vz  # Up = -NED_vz
        self._odom_pub.publish(odom)

    # ── 激光扫描中继 ──────────────────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan):
        """
        将 PX4 激光扫描中继到 /scan，修改 frame_id 为 laser_link。
        slam_toolbox 通过 TF (odom→base_link→laser_link) 自动做空间对齐，
        无需在这里手动旋转角度——安装角已编码在静态 TF 中。

        ★ 时间戳策略（已修复）：
          始终使用 self.get_clock().now()（ROS sim clock），
          保证 scan.header.stamp 与 odom→base_link TF.header.stamp 来自同一时基。

          原方案：保留 msg.header.stamp（PX4 仿真时间），会导致:
            slam_toolbox MessageFilter: "timestamp earlier than transform cache"
            原因：TF 以 get_clock().now() 发布，scan 以 PX4 ts 发布，两者偏差
            超出 tf2 cache 容忍值（~9ms）时 slam_toolbox 丢 scan。
        """
        import time as _wt
        # ★ TF-scan 时间戳精确对齐（Phase7: 200ms→50ms）
        # 优先使用最近一次 TF 发布时的 stamp（<50ms 内有效），
        # 确保 slam_toolbox MessageFilter 能在 TF 缓存中精确命中对应的 odom→base_link 帧。
        # <50ms: 以 pos_cb 频率 ~50Hz 更新，TF→scan 延迟 ≤20ms，理论偏差 <0.3°@10dps
        # ≥50ms: 降级为 get_clock().now()，避免旋转期间用过时 pose 对齐扫描。
        _now_mono = _wt.monotonic()
        _tf_age = _now_mono - getattr(self, '_last_tf_stamp_mono', 0.0)
        _tf_stamp_fresh = self._last_tf_stamp_msg is not None and _tf_age < 0.050
        if _tf_stamp_fresh:
            now_stamp = self._last_tf_stamp_msg
        else:
            now_stamp = self.get_clock().now().to_msg()

        out = LaserScan()
        out.header.stamp     = now_stamp
        out.header.frame_id  = self.laser_frame   # ← 关键：frame_id = laser_link
        out.angle_min        = msg.angle_min
        out.angle_max        = msg.angle_max
        out.angle_increment  = msg.angle_increment
        out.time_increment   = msg.time_increment
        out.scan_time        = msg.scan_time
        out.range_min        = msg.range_min
        out.range_max        = msg.range_max
        out.ranges           = msg.ranges
        out.intensities      = msg.intensities

        # ★ A1/A2: 双话题发布——raw 始终发布；SLAM 话题受稳定门控
        self._scan_raw_pub.publish(out)          # → /scan_raw（始终）
        self._scan_recv_count += 1
        self._last_scan_time = _wt.monotonic()
        _now_scan = _wt.monotonic()

        # ★ TF-FREEZE 守卫：冻结期间无论 gate_mode 如何都不向 slam_toolbox 发送扫描
        # 原因：TF 已冻结在旧位置，此时 slam_toolbox 接到的扫描会叠加在正确位置上，
        # 但一旦解冻，slam_toolbox将用新位置处理扫描—— 位置是错的但扫描延迟积分到旧位置
        # 总之：冻结期间没有意义的扫描，跳过更安全。
        if self._tf_freeze_active:
            # 冻结期间：禁止发送 slam 扫描（force_open 也无效）
            self._scan_drop_count += 1
            # 节流日志
            if self._tf_freeze_frame_cnt % 20 == 1:
                self.get_logger().warn(
                    f"[TF-FREEZE-HOLD] frames={self._tf_freeze_frame_cnt}/"
                    f"{self._tf_freeze_max_frames} 扫描被阻止，防止错位积分到 slam_toolbox"
                )
        elif self._slam_scan_gate_open:
            # hover_strict 近墙扫描门控：min_range < 0.45m 时丢帧防止误建图
            _min_range = min(
                (r for r in msg.ranges if msg.range_min < r < msg.range_max and math.isfinite(r)),
                default=float('inf')
            )
            if self._gate_mode == "hover_strict" and _min_range < 0.45:
                self._scan_drop_count += 1
                self.get_logger().warn(
                    f"[SCAN-GATE-DECISION] drop reason=near_wall "
                    f"min_range={_min_range:.2f}m mode={self._gate_mode}"
                )
            else:
                self._scan_pub.publish(out)          # → /scan（门开，喂给 slam_toolbox）
                self._scan_slam_count += 1
                self._last_slam_scan_time = _now_scan
        else:
            self._scan_drop_count += 1
            # ★ A5: 强制发布保底——若 SLAM 扫描间隔超过阈值则强制喂一帧
            _slam_gap = (_now_scan - self._last_slam_scan_time
                         if self._last_slam_scan_time > 0 else float('inf'))
            if _slam_gap >= self._scan_force_interval_sec:
                # hover_strict 或 TF冻结期间禁止 GATE-FORCE（地图一致性优先）
                _force_blocked = (self._gate_mode == "hover_strict" or self._tf_freeze_active)
                if _force_blocked:
                    self.get_logger().warn(
                        f"[GATE-FORCE-BLOCKED] reason="
                        f"{'hover_strict' if self._gate_mode == 'hover_strict' else 'tf_freeze'} "
                        f"gap={_slam_gap:.1f}s mode={self._gate_mode} "
                        f"tf_freeze={self._tf_freeze_active}"
                    )
                else:
                    self._scan_pub.publish(out)
                    self._scan_force_count += 1
                    self._scan_slam_count += 1
                    self._last_slam_scan_time = _now_scan
                    self.get_logger().warn(
                        f"[GATE-FORCE] SLAM 已 {_slam_gap:.1f}s 无扫描输入，"
                        f"强制发布一帧 (mode={self._gate_mode} "
                        f"force_total={self._scan_force_count})"
                    )

        # ── Phase7: 每秒一次 SCAN-TF-DIAG + BRIDGE-YAW 诊断 ──────────────
        _delta_ms = (_now_mono - getattr(self, '_last_tf_stamp_mono', 0.0)) * 1000.0
        if _now_mono - self._scan_diag_last_t >= 1.0:
            self._scan_diag_last_t = _now_mono
            _fallback_flag = 'fallback' if not _tf_stamp_fresh else 'tf-match'
            _freeze_diag = (
                f" FREEZE={self._tf_freeze_frame_cnt}/{self._tf_freeze_max_frames}"
                if self._tf_freeze_active else ""
            )
            _ned_heading_deg = float(np.rad2deg(math.pi / 2.0 - self._att_yaw))  # ENU yaw→NED heading
            self.get_logger().info(
                f"[SCAN-TF-DIAG] delta_ms={_delta_ms:.1f} "
                f"tf_age_ms={_tf_age*1000:.1f} "
                f"yaw_enu={math.degrees(self._att_yaw):.1f}\u00b0 "
                f"ned_heading={_ned_heading_deg:.1f}\u00b0 "
                f"yr={self._curr_yaw_rate_dps:.1f}dps "
                f"stamp={_fallback_flag} "
                f"gate={'open' if self._slam_scan_gate_open else 'closed'} "
                f"mode={self._gate_mode}"
                f"{_freeze_diag}"
            )
            # 1.4 [BRIDGE-YAW] 独立 1Hz 日志：heading 与 quat 对比 + 降级状态 ────
            _h_yaw = float(self._pos_heading_deg)      # NED heading deg
            _q_yaw = float(self._att_q_yaw_deg)        # ENU quat yaw deg
            _h_enu = float(math.degrees(math.pi / 2.0 - math.radians(_h_yaw))) if _h_yaw == _h_yaw else float('nan')
            _yd    = abs(((_h_enu - _q_yaw + 180.0) % 360.0) - 180.0)
            _ysrc  = ('att_quat[FORCED]' if self._yaw_src_force_quat
                      else ('heading(EKF)' if self._heading_good else 'att_quat(fallback)'))
            self.get_logger().info(
                f"[BRIDGE-YAW] src={_ysrc} "
                f"heading_ned={_h_yaw:.1f}deg heading_enu={_h_enu:.1f}deg "
                f"quat_enu={_q_yaw:.1f}deg diff={_yd:.1f}deg "
                f"mismatch_frames={self._yaw_mismatch_frames} "
                f"match_frames={self._yaw_match_frames} "
                f"force_quat={self._yaw_src_force_quat}"
            )
            # [YAW-SRC] 明确标记当前 yaw 来源
            _yaw_src_str = ('att_quat' if self._yaw_src_force_quat
                            else ('heading_enu' if self._heading_good else 'att_quat_fallback'))
            self.get_logger().info(
                f'[YAW-SRC] src={_yaw_src_str} '
                f'yaw={math.degrees(self._att_yaw):.1f}deg '
                f'heading_good={self._heading_good} force_quat={self._yaw_src_force_quat}'
            )

        # ── A3: 每 100 帧打印 SCAN-DIAG + SCAN-GATE 诊断 ─────────────────
        if self._scan_recv_count % 100 == 0:
            dt = _wt.monotonic() - self._scan_hz_t0
            hz = 100.0 / dt if dt > 0 else 0.0
            self._scan_hz_t0 = _wt.monotonic()
            _ranges_arr = np.array(msg.ranges, dtype=float)
            _total_pts  = len(_ranges_arr)
            _valid_mask = (np.isfinite(_ranges_arr) &
                           (_ranges_arr >= msg.range_min) &
                           (_ranges_arr <= msg.range_max))
            _valid_n   = int(np.sum(_valid_mask))
            _valid_r   = _valid_n / _total_pts if _total_pts > 0 else 0.0
            _rmin_obs  = float(np.min(_ranges_arr[_valid_mask])) if _valid_n > 0 else float('nan')
            _rmax_obs  = float(np.max(_ranges_arr[_valid_mask])) if _valid_n > 0 else float('nan')
            _total_gate = self._scan_slam_count + self._scan_drop_count
            _drop_r = self._scan_drop_count / _total_gate if _total_gate > 0 else 0.0
            self.get_logger().info(
                f"[SCAN-DIAG] #{self._scan_recv_count} "
                f"hz≈{hz:.1f} frame={self.laser_frame} "
                f"pts={_total_pts} valid={_valid_n}({_valid_r*100:.0f}%) "
                f"r_obs=[{_rmin_obs:.2f},{_rmax_obs:.2f}]m "
                f"range_cfg=[{msg.range_min:.2f},{msg.range_max:.2f}]m"
            )
            self.get_logger().info(
                f"[SCAN-GATE] mode={self._gate_mode} open={self._slam_scan_gate_open} "
                f"freeze={'YES('+str(self._tf_freeze_count)+')' if self._tf_freeze_active else 'no'} "
                f"tilt={self._att_tilt_deg:.1f}° "
                f"vxy={math.hypot(self._vx, self._vy):.3f}m/s "
                f"vz={abs(self._vz):.3f}m/s "
                f"yaw_rate={self._curr_yaw_rate_dps:.1f}dps | "
                f"slam={self._scan_slam_count} drop={self._scan_drop_count}({_drop_r:.0%}) "
                f"force={self._scan_force_count} | "
                f"rej_tilt={self._gate_rej_roll} "
                f"vxy={self._gate_rej_vxy} vz={self._gate_rej_vz} yr={self._gate_rej_yr} "
                f"(diag roll={abs(math.degrees(self._att_roll)):.1f}° "
                f"pitch={abs(math.degrees(self._att_pitch)):.1f}°)"
            )

    # ── TF chain Watchdog ─────────────────────────────────────────────────────
    def _watchdog_cb(self) -> None:
        """每 10s 检查一次关键话题接收状态并打印诊断摘要。

        冷却机制：节点启动后 30s 内（挂钟）不输出 '从未收到' 错误，
        因为 use_sim_time=true 下 sim_time 可能追赶导致定时器立即触发。
        """
        import time as _wt
        now = _wt.monotonic()
        uptime = now - self._node_start_wall   # 节点已运行的挂钟秒数
        _WARMUP_SEC = 30.0                      # 冷却期

        pos_age   = now - self._last_pos_time  if self._last_pos_time > 0  else float('inf')
        scan_age  = now - self._last_scan_time if self._last_scan_time > 0 else float('inf')
        tf_age    = now - self._last_tf_time   if self._last_tf_time > 0   else float('inf')

        if self._pos_recv_count == 0:
            if uptime < _WARMUP_SEC:
                self.get_logger().debug(
                    f"[BRIDGE-WDG] 等待 VehicleLocalPosition... (uptime={uptime:.1f}s < {_WARMUP_SEC}s)"
                )
            else:
                self.get_logger().error(
                    f"[BRIDGE-WDG] ❌ 从未收到 VehicleLocalPosition! "
                    f"检查 px4_ros_com_ws source 和 /fmu/out/vehicle_local_position_v1 话题"
                )
        elif pos_age > 5.0:
            self.get_logger().warn(
                f"[BRIDGE-WDG] ⚠ VehicleLocalPosition 超时 {pos_age:.1f}s (总接收={self._pos_recv_count})"
            )

        if self._scan_recv_count == 0:
            if uptime < _WARMUP_SEC:
                self.get_logger().debug(
                    f"[BRIDGE-WDG] 等待激光扫描... (uptime={uptime:.1f}s < {_WARMUP_SEC}s)"
                )
            else:
                self.get_logger().error(
                    f"[BRIDGE-WDG] ❌ 从未收到激光扫描! "
                    f"检查 {self.lidar_topic_in} 话题是否有数据 (run: ros2 topic hz {self.lidar_topic_in})"
                )
        elif scan_age > 5.0:
            self.get_logger().warn(
                f"[BRIDGE-WDG] ⚠ 激光扫描超时 {scan_age:.1f}s (总接收={self._scan_recv_count})"
            )
        else:
            self.get_logger().info(
                f"[BRIDGE-WDG] ✓ pos #{self._pos_recv_count} age={pos_age:.1f}s | "
                f"scan #{self._scan_recv_count} age={scan_age:.1f}s | "
                f"TF age={tf_age:.1f}s"
            )

        # ── [TF-DIAG] 定时诊断汇总（每次 watchdog 输出）────────────────────
        _heading_src = "heading(EKF)" if self._heading_good else "att_quat(fallback)"
        _yaw_deg = float(np.rad2deg(self._att_yaw))
        _qyaw_deg = self._att_q_yaw_deg
        _yaw_diff = abs(((_yaw_deg - _qyaw_deg + 180.0) % 360.0) - 180.0)
        self.get_logger().info(
            f"[TF-DIAG] pos=({self._x_enu:.2f},{self._y_enu:.2f},{-self._z_ned:.2f}m) "
            f"yaw={_yaw_deg:.1f}° src={_heading_src} qyaw={_qyaw_deg:.1f}° "
            f"diff={_yaw_diff:.1f}° | "
            f"jumps={self._tf_jump_count} max_Δxy={self._tf_max_xy_jump:.2f}m | "
            f"TF_pub={getattr(self,'_tf_pub_count',0)}"
        )


        _diag_path = getattr(self, '_bridge_diag_path', None)
        if _diag_path is not None:
            try:
                _lines = [
                    f"# px4_slam_bridge 实时诊断  更新: {_time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                    f"# tail -f {_diag_path}\n\n",
                    f"[话题接收状态]\n",
                    f"  VehicleLocalPosition: 总次={self._pos_recv_count} age={pos_age:.1f}s\n",
                    f"  LaserScan:            总次={self._scan_recv_count} age={scan_age:.1f}s\n",
                    f"  TF发布:              age={tf_age:.1f}s\n\n",
                    f"[yaw状态]\n",
                    f"  yaw_from_pos={self._yaw_from_pos}  heading_good={self._heading_good}\n",
                    f"  pos_heading_deg={self._pos_heading_deg:.1f}  att_q_yaw_deg={self._att_q_yaw_deg:.1f}\n\n",
                    f"[当前位姿]\n",
                    f"  x={self._x_enu:.3f}  y={self._y_enu:.3f}  z={self._z_ned:.3f}\n",
                    f"  att_yaw={self._att_yaw:.4f}rad ({math.degrees(self._att_yaw):.1f}°)\n\n",
                    f"[配置]\n",
                    f"  lidar_in={self.lidar_topic_in} scan_out={self.scan_out_topic}\n",
                    f"  mount_yaw={self.lidar_mount_yaw_deg}° frames: odom={self.odom_frame} base={self.base_frame} laser={self.laser_frame}\n\n",
                    f"[TF跳变统计]\n",
                    f"  tf_jump_count={self._tf_jump_count}  max_xy_jump={self._tf_max_xy_jump:.3f}m\n",
                    f"  tf_jump_th_xy={self._tf_jump_th_xy:.2f}m  tf_jump_th_z={self._tf_jump_th_z:.2f}m\n",
                ]
                with Path(_diag_path).open("w", encoding="utf-8") as _df:
                    _df.writelines(_lines)
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = Px4SlamBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
