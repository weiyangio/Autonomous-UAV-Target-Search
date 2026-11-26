#!/usr/bin/env python3
"""Local MPC-like velocity tracking of global path.
- Subscribes: global_path (SearchPath), current pose from /fmu/out/vehicle_local_position_v1.
- Publishes: LocalCommand cmd_vel (ENU) at 20 Hz.

Coordinate convention (MUST match all upstream/downstream nodes):
  ENU: x = East (= NED_y = VehicleLocalPosition.y)
       y = North (= NED_x = VehicleLocalPosition.x)
  global_path waypoints are also in ENU.
  Output LocalCommand.cmd.linear.x/y are ENU East/North.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from px4_msgs.msg import VehicleLocalPosition
from indoor_search.msg import SearchPath, LocalCommand
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry, OccupancyGrid


class LocalMPC(Node):
    def __init__(self):
        super().__init__('local_mpc')
        self.fmu_prefix = self.declare_parameter('fmu_prefix', '/fmu').value.rstrip('/')
        self.max_vel = float(self.declare_parameter('max_vel', 1.2).value)
        self.max_acc = float(self.declare_parameter('max_acc', 0.5).value)
        self.max_yaw_rate = float(self.declare_parameter('max_yaw_rate', 0.8).value)
        self.safety_radius = float(self.declare_parameter('safety_radius', 0.5).value)
        # ENU bounds: x=East, y=North — matches maze.world [-6,6] footprint
        _bxy = float(self.declare_parameter('bound_xy', 6.0).value)
        self.bounds_xy = _bxy  # symmetric ±bound_xy
        self.pose = None
        # ENU position cache (updated in pose_cb)
        self._x_enu: float = 0.0
        self._y_enu: float = 0.0
        self._z: float = 0.0
        # ── odom 主坐标源（bridge ENU）────────────────────────────────
        # 主：/odom（bridge ENU）；备： VehicleLocalPosition NED→ENU 转换
        self._odom_received: bool = False
        self._x_odom: float = 0.0     # ENU x = East
        self._y_odom: float = 0.0     # ENU y = North
        self._z_odom: float = 0.0
        self._mpc_src: str = 'px4_fallback'
        self.path = None
        self.idx = 0
        self.prev_vel = np.zeros(3, dtype=float)
        self.timer_period = 0.05
        self.last_progress_dist = None
        self.last_progress_time = self.get_clock().now()
        self._mpc_frame_log_t: float = 0.0
        # ── FSM 抢占状态 ──────────────────────────────────────────────────────────
        # 当 target_search 处于以下状态时，停止发布 local_cmd 以避免刷新时间戳
        # 干扰 target_search 的控制仲裁（local_cmd 应为 stale）。
        self._fsm_state: str = ''
        self._fsm_preempt_active: bool = False
        # 抢占时重置 prev_vel → 恢复后无突变加速
        self._fsm_preempt_log_t: float = 0.0
        self.create_subscription(
            VehicleLocalPosition,
            f'{self.fmu_prefix}/out/vehicle_local_position_v1',
            self.pose_cb, qos_profile_sensor_data
        )
        self.create_subscription(SearchPath, 'global_path', self.path_cb, 10)
        self.create_subscription(String, '/indoor_search/fsm_state', self._fsm_cb, 1)
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_profile_sensor_data
        )
        # ── OccupancyGrid for indoor wall hard constraints ───────────────────────
        self._occ_grid: OccupancyGrid = None
        self.create_subscription(
            OccupancyGrid, '/map', self._occ_cb, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(LocalCommand, 'local_cmd', 10)
        # ★[A1] SSG A* path (高优先级路径：target_search 的 A* 规划结果)
        # 当 local_mpc 订阅到 ssg_path 时，优先跟随该路径（local_fresh=True）
        # global_path 仍作为备选（SSG路径过期后降级）
        self.create_subscription(SearchPath, 'ssg_path', self.ssg_path_cb, 10)
        self._using_ssg_path: bool = False   # 当前是否在用 SSG 路径
        self._ssg_path_stale_t: float = 0.0  # SSG 路径有效截止时间(monotonic)
        self.timer = self.create_timer(self.timer_period, self.tick)  # 20 Hz
        import hashlib as _mfid_hl, time as _mfid_t
        _mfid_data = open(__file__, 'rb').read()
        _mfid_sha = _mfid_hl.sha256(_mfid_data).hexdigest()[:16]
        self.get_logger().info(
            f"[FILE-ID] ★ node=local_mpc  file={__file__}  "
            f"sha256={_mfid_sha}  started={_mfid_t.strftime('%Y-%m-%dT%H:%M:%S')}  "
            f"version=2026-03-09_through_fix"
        )
        import os as _msync_os
        _minstall_path = _msync_os.path.join(
            _msync_os.path.dirname(__file__).replace(
                '/ros2_ws/src/indoor_search/src', '/ros2_ws/install/lib/indoor_search'),
            'local_mpc_node.py')
        if _msync_os.path.exists(_minstall_path):
            _minst_sha = _mfid_hl.sha256(open(_minstall_path,'rb').read()).hexdigest()[:16]
            _msync_ok = (_minst_sha == _mfid_sha)
        else:
            _minst_sha, _msync_ok = 'NOT_FOUND', False
        if not _msync_ok:
            self.get_logger().error(
                f'[CODE-SYNC-MISMATCH] node=local_mpc install_sha={_minst_sha} != src_sha={_mfid_sha}'
                f' — 请重新 colcon build')
        else:
            self.get_logger().info(f'[CODE-SYNC] node=local_mpc sha={_mfid_sha} match=True')
        self.get_logger().info(
            f"[MPC-FRAME] using ENU (x=East=NED_y, y=North=NED_x) "
            f"bounds=±{_bxy:.1f}m safety_r={self.safety_radius:.2f}m"
        )

    def pose_cb(self, msg: VehicleLocalPosition):
        self.pose = msg
        # Convert PX4 NED to ENU: x_enu=East=NED_y, y_enu=North=NED_x
        # 备用：仅当 odom 未到时使用
        if not self._odom_received:
            self._x_enu = float(msg.y)
            self._y_enu = float(msg.x)
            self._z = float(msg.z)
            self._mpc_src = 'px4_fallback'

    # ── OccupancyGrid helpers ────────────────────────────────────────────
    def _occ_cb(self, msg: OccupancyGrid) -> None:
        self._occ_grid = msg

    def _occ_clearance(self, x: float, y: float,
                       check_radius: float = 0.65) -> float:
        """Return minimum distance to any occupied cell within check_radius (m).
        Returns check_radius if no occupied cell found (safe).
        """
        import math as _mc_m
        g = self._occ_grid
        if g is None:
            return check_radius
        sres = g.info.resolution
        if sres <= 0:
            return check_radius
        ox = g.info.origin.position.x
        oy = g.info.origin.position.y
        n_steps = max(1, int(_mc_m.ceil(check_radius / sres)))
        min_dist = check_radius
        for di in range(-n_steps, n_steps + 1):
            for dj in range(-n_steps, n_steps + 1):
                wx = x + di * sres
                wy = y + dj * sres
                col = int((wx - ox) / sres)
                row = int((wy - oy) / sres)
                if 0 <= col < g.info.width and 0 <= row < g.info.height:
                    if g.data[row * g.info.width + col] >= 65:
                        d = _mc_m.hypot(x - wx, y - wy)
                        if d < min_dist:
                            min_dist = d
        return min_dist

    def _is_path_blocked(self, x0: float, y0: float, x1: float, y1: float,
                         steps: int = 20) -> bool:
        """Return True if straight line from (x0,y0) to (x1,y1) passes through an occupied cell.
        ★[FIX-LOS] Threshold raised to 90 to avoid false positives from SLAM uncertainty near
        walls (values 65-85 often appear near free-but-close-to-wall cells in slam_toolbox).
        NOTE: This check is only used for legacy global_path. SSG A* paths use _is_waypoint_at_obstacle.
        """
        g = self._occ_grid
        if g is None:
            return False
        sres = g.info.resolution
        if sres <= 0:
            return False
        ox = g.info.origin.position.x
        oy = g.info.origin.position.y
        for i in range(1, steps + 1):
            t = i / steps
            wx = x0 + t * (x1 - x0)
            wy = y0 + t * (y1 - y0)
            col = int((wx - ox) / sres)
            row = int((wy - oy) / sres)
            if 0 <= col < g.info.width and 0 <= row < g.info.height:
                if g.data[row * g.info.width + col] >= 90:  # ★[FIX-LOS] raised from 65→90
                    return True
        return False

    def _is_waypoint_at_obstacle(self, x: float, y: float) -> bool:
        """★[FIX-MPC-LOS] Check if a waypoint position is solidly inside an obstacle.
        Uses a 1-cell neighbourhood and high threshold=90 to avoid false positives.
        For SSG A* paths: ONLY use this check (not full LOS).
        Why: A* already verified path is obstacle-free with inflation margin.
        MPC's straight-line LOS check incorrectly rejects paths that go AROUND walls;
        it cannot "see" through valid gap/door passages that A* found via graph search.
        """
        g = self._occ_grid
        if g is None:
            return False
        sres = g.info.resolution
        if sres <= 0:
            return False
        ox = g.info.origin.position.x
        oy = g.info.origin.position.y
        col = int((x - ox) / sres)
        row = int((y - oy) / sres)
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                c, r = col + dc, row + dr
                if 0 <= c < g.info.width and 0 <= r < g.info.height:
                    if g.data[r * g.info.width + c] >= 90:
                        return True
        return False

    def _odom_cb(self, msg: Odometry) -> None:
        """Primary ENU pose from bridge. Updates _x_enu/_y_enu (ENU)."""
        self._x_odom = float(msg.pose.pose.position.x)   # ENU East
        self._y_odom = float(msg.pose.pose.position.y)   # ENU North
        self._z_odom = float(msg.pose.pose.position.z)   # ENU Up
        # 将主坐标源切换到 odom
        self._x_enu = self._x_odom
        self._y_enu = self._y_odom
        self._z = -self._z_odom   # ENU Up → NED Down（z负表示地面之上）
        self._mpc_src = 'bridge_odom'
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info(
                f'[MPC-FRAME] src=bridge_odom first_msg '
                f'enu=({self._x_odom:.2f},{self._y_odom:.2f})'
            )

    def ssg_path_cb(self, msg) -> None:
        """★[A1] 高优先级 SSG A* 路径回调（target_search 的当前 A* 规划路径）。
        有内容 → 作为 primary 路径跟随；内容为空 → 释放控制，允许 global_path 接管。
        """
        import time as _sp_t
        if msg.poses:
            self.path = msg.poses
            self.idx = 0
            self._using_ssg_path = True
            self._ssg_path_stale_t = _sp_t.monotonic() + 2.5  # 2.5s 有效窗口
            self.last_progress_dist = None
            self.last_progress_time = self.get_clock().now()
            self.get_logger().debug(
                f'[LOCAL-MPC-ACTIVE] src=ssg_path waypoints={len(msg.poses)} '
                f'state={self._fsm_state}'
            )
        else:
            # 空路径信号：SSG 导航结束，释放控制权给 global_path
            if self._using_ssg_path:
                self._using_ssg_path = False
                self.get_logger().info(
                    f'[LOCAL-MPC-ACTIVE] ssg_path=empty → global_path fallback '
                    f'state={self._fsm_state}'
                )

    def path_cb(self, msg):
        import time as _pg_t
        # ★[A1] SSG 路径有效时优先，忽略 global_path（避免两路径冲突）
        if self._using_ssg_path and _pg_t.monotonic() < self._ssg_path_stale_t:
            return
        self._using_ssg_path = False
        self.path = msg.poses
        self.idx = 0

    def _fsm_cb(self, msg: String) -> None:
        """Track target_search FSM state for preemption decisions."""
        import time as _fc_t
        _prev = self._fsm_state
        self._fsm_state = msg.data
        # ★[A2] 抢占状态集合（精确控制）：
        #   SEARCH / NAV_TO_VIEWPOINT / TARGET_APPROACH / TARGET_LOCKED / TARGET_ALIGN
        #   均已从抢占列表移除 → local_mpc 在这些阶段持续发布，保持 prev_vel 热态；
        #   target_search 在 TARGET_APPROACH/LOCKED/ALIGN 期间独占发布 setpoint（不经 CTRL-ARB），
        #   local_mpc 的输出对无人机无实际影响，但可确保恢复 SEARCH 时无突变加速。
        #   TARGET_DESCEND/LAND: 视觉着降落，完全由 target_search 发 setpoint → 抢占防止干扰；
        #   TARGET_LOST_SHORT: 目标短时丢失期间零速悬停 + Kalman偏航 → 抢占防止 MPC 横跑
        #     例外：若 target_search 切换到 lost_nav 子状态（world_est 导航），
        #     则会重新发布 ssg_path，local_mpc 收到后将以 _using_ssg_path=True 主动恢复。
        #   TARGET_LOST_SCAN: 扫描扫描期间 yaw摆动 → 抢占防止 MPC 横跑。
        # ★[F-ARCH] 目标接近全链路抢占：local_mpc 在视觉伺服期间不得主导方向
        #   target_search 在 _approach_exclusive=True 时独占控制权（已通过 fsm_state 广播）
        #   把 TARGET_LOCKED_FAR/ALIGN/FORWARD/HOVER_CONFIRM 也加入抢占，保持控制权清晰
        _PREEMPT = ('SAFE_HOLD', 'MAP_RECOVER',
                    'HOVER_SCAN_ESCAPE', 'HOVER_STRICT', 'EMERG',
                    'TARGET_LOCKED_FAR', 'TARGET_APPROACH_ALIGN',
                    'TARGET_APPROACH_FORWARD', 'TARGET_HOVER_CONFIRM',
                    'TARGET_DESCEND', 'TARGET_LAND',
                    'TARGET_LOST_SHORT', 'TARGET_LOST_SCAN',
                    # ★[B] 终端链全部状态抢占：local_mpc 不干预目标终端任务
                    'TARGET_WORLD_ESTABLISH', 'TARGET_COMMIT_POINT',
                    'TARGET_HOVER_COMMIT', 'TARGET_CONSERVATIVE_DESCEND',
                    'TARGET_DESCEND_EXEC',
                    'MISSION_DONE')
        # ★[G4] lost_nav 例外：当 target_search 进入 lost_nav 子状态时会重新发 ssg_path，
        #   local_mpc 收到非空 ssg_path 后 _using_ssg_path=True，tick() 正常执行路径跟踪。
        #   该路径 2.5s 有效，自然超期后回落到 fallback。无需修改抢占集合。
        _now_preempt = any(s in self._fsm_state for s in _PREEMPT)
        if _now_preempt and not self._fsm_preempt_active:
            # 进入抢占：重置速度积分，恢复后无突变
            self.prev_vel = np.zeros(3, dtype=float)
            # 立即发布零速度帧：清除残余速度
            _zero_msg = LocalCommand()
            _zero_msg.cmd = Twist()   # linear/angular 全零
            self.pub.publish(_zero_msg)
            self.get_logger().info(
                f'[LOCAL-MPC-PREEMPT] reason=enter_preempt_state state={self._fsm_state} '
                f'[CTRL-ARB] source=PREEMPT'
            )
        elif not _now_preempt and self._fsm_preempt_active:
            # 离开抢占：恢复正常发布
            self.get_logger().info(
                f'[LOCAL-MPC-ACTIVE] reason=resume_from_preempt new_state={self._fsm_state} '
                f'ssg_path={self._using_ssg_path} path_wps={len(self.path) if self.path else 0} '
                f'[CTRL-ARB] source=LOCAL_MPC'
            )
        self._fsm_preempt_active = _now_preempt

    def tick(self):
        import time as _mt
        # 需要 odom 或 px4 位姿至少一个可用
        if (self.pose is None and not self._odom_received) or not self.path or self.idx >= len(self.path):
            return

        # ── 上层 FSM 抢占：HOVER_SCAN / SAFE_HOLD / MAP_RECOVER 等优先接管 ──────
        # 停止发布 local_cmd（不刷 last_local_stamp）→ target_search 视为 stale
        # prev_vel 已在 _fsm_cb 进入抢占时重置为零，恢复后无突变
        if self._fsm_preempt_active:
            _now_m = _mt.monotonic()
            if _now_m - self._fsm_preempt_log_t > 5.0:
                self._fsm_preempt_log_t = _now_m
                self.get_logger().info(
                    f'[MPC-HOLD] reason=preempt_state state={self._fsm_state} (holding, not publishing)'
                )
            return  # 不发布任何帧，last_local_stamp 自然超时 → target_search 视为 stale

        # curr in ENU
        curr = np.array([self._x_enu, self._y_enu, self._z])

        # 1Hz [MPC-FRAME] diagnostic
        _now_m = _mt.monotonic()
        if _now_m - self._mpc_frame_log_t > 1.0:
            self._mpc_frame_log_t = _now_m
            self.get_logger().debug(
                f"[MPC-FRAME] src={self._mpc_src} "
                f"enu=({self._x_enu:.2f},{self._y_enu:.2f})"
            )

        # Target in ENU (global_planner publishes ENU waypoints)
        tgt_pose = self.path[self.idx].position
        tgt = np.array([float(tgt_pose.x), float(tgt_pose.y), float(tgt_pose.z)])

        # Clamp target to safe bounds (in ENU, symmetric ±bounds_xy)
        margin = self.safety_radius
        tgt[0] = float(np.clip(tgt[0], -self.bounds_xy + margin, self.bounds_xy - margin))
        tgt[1] = float(np.clip(tgt[1], -self.bounds_xy + margin, self.bounds_xy - margin))

        # ★[FIX-MPC-LOS] 路径阻断检查：SSG A* 路径 vs 传统 global_path 策略不同
        # ★[FIX-A ARCH] SSG 路径完全跳过障碍物二次检查:
        #   根本原因：A* 用 target_search 内部网格（prob_occ_th=180）规划，
        #   local_mpc 用 ROS /map（threshold=90）检查，两个网格阈值不同：
        #   - /map 值 65-89 → target_search 视为 FREE（65/100*255=165 < 180），local_mpc 视为 BLOCKED（65<90）
        #   - 门框附近 /map 值通常 65-87 → 造成大量假阳性 MPC-WP-IN-OBS
        #   - 误清路径 → local_fresh=False → CTRL-ARB FALLBACK → 直飞目标不避障 → 撞墙
        #   修复：SSG路径完全信任A*规划器，不做任何二次检查。
        #   安全保障由 target_search 的 PROX-BRAKE / NEAR-WALL-HARDSTOP 提供。
        #   传统 global_path 保留 full LOS 检查（threshold=90）。
        import time as _mpc_fix_t
        _ssg_active = (self._using_ssg_path and hasattr(self, '_ssg_path_stale_t'))
        _ssg_fresh  = (_ssg_active and _mpc_fix_t.monotonic() < self._ssg_path_stale_t)
        if _ssg_fresh:
            # ★[FIX-A] SSG A* 路径：完全信任规划器，不做障碍物二次检查
            pass   # 无需检查，A* 已验证路径可达性
        else:
            # 传统global_path：保留full LOS检查（threshold已升至90）
            if self._is_path_blocked(curr[0], curr[1], tgt[0], tgt[1]):
                self.get_logger().warn(
                    f'[MPC-BLOCKED] waypoint=({tgt[0]:.2f},{tgt[1]:.2f}) '
                    f'wall-blocked from ({curr[0]:.2f},{curr[1]:.2f})',
                    throttle_duration_sec=1.0
                )
                if self.idx < len(self.path) - 1:
                    self.idx += 1
                    self.last_progress_dist = None
                    return
                else:
                    self.get_logger().warn(
                        f'[MPC-BLOCKED-LAST] all {len(self.path)} waypoints blocked '
                        f'from ({curr[0]:.2f},{curr[1]:.2f}) → clear path, explore fallback',
                        throttle_duration_sec=2.0
                    )
                    self.path = None
                    self.idx = 0
                    self.prev_vel = np.zeros(3, dtype=float)
                    return

        err = tgt - curr
        dist = float(np.linalg.norm(err))

        # Pose validity: if ENU position is wildly out of bounds, publish zero
        if (abs(self._x_enu) > self.bounds_xy * 2 or abs(self._y_enu) > self.bounds_xy * 2):
            self.get_logger().warn(
                f"[MPC-FRAME] pose out of bounds enu=({self._x_enu:.2f},{self._y_enu:.2f}) → zero XY"
            )
            msg = LocalCommand()
            tw = Twist()
            msg.cmd = tw
            self.pub.publish(msg)
            return

        if dist < 0.3 and self.idx < len(self.path) - 1:
            self.idx += 1
            self.last_progress_dist = None
            return

        # ★ 终点防振荡：最后路径点距离 < 阈值时制动归零，保持 pub 维持 local_fresh hot
        # ★[FIX-B] 阈值从 0.45m → 0.25m：
        #   原问题：0.45m 阈值在无人机还处于高速时就触发制动，惯性导致过冲 →
        #   在 0.40-0.44m 区间振荡超过 1.5s → stalled progress 检测清路径 →
        #   local_fresh=False → CTRL-ARB FALLBACK → 直飞目标不避障 → 撞墙
        #   修复：0.25m 阈值确保无人机在速度已降低（speed=min(1.2,0.25)=0.25m/s）时再制动，
        #   既减少过冲，又让 stalled progress 检测有足够时间正常运作。
        if self.idx == len(self.path) - 1 and dist < 0.25:
            import time as _lw_t
            _now_lw = _lw_t.monotonic()
            if _now_lw - getattr(self, '_last_wp_hover_log_t', 0.0) > 2.0:
                self._last_wp_hover_log_t = _now_lw
                self.get_logger().info(
                    f'[MPC-LAST-WP] ★ dist={dist:.2f}m < 0.45m 终点驻停 '
                    f'wp=({tgt[0]:.2f},{tgt[1]:.2f}) curr=({curr[0]:.2f},{curr[1]:.2f}) '
                    f'→ braking to zero (anti-oscillation, keep local_fresh active)'
                )
            # 制动：每帧 max_acc*dt 减速，不立即归零避免姿态突变
            _max_dv_lw = self.max_acc * self.timer_period
            _dv_brake = np.clip(-self.prev_vel, -_max_dv_lw, _max_dv_lw)
            _v_brake = self.prev_vel + _dv_brake
            msg = LocalCommand()
            tw = Twist()
            tw.linear.x = float(_v_brake[0])
            tw.linear.y = float(_v_brake[1])
            tw.linear.z = float(_v_brake[2])
            msg.cmd = tw
            self.pub.publish(msg)   # keep local_fresh alive
            self.prev_vel = _v_brake
            return

        # Detect stalled progress: if distance reduction is tiny over 1.5 s, skip
        now = self.get_clock().now()
        if self.last_progress_dist is None:
            self.last_progress_dist = dist
            self.last_progress_time = now
        else:
            delta = self.last_progress_dist - dist
            elapsed = (now - self.last_progress_time).nanoseconds * 1e-9
            if delta < 0.05 and elapsed > 1.5:
                if self.idx < len(self.path) - 1:
                    self.idx += 1
                    self.last_progress_dist = None
                    return
                else:
                    self.path = None
                    self.last_progress_dist = None
                    return
            if delta >= 0.02:
                self.last_progress_dist = dist
                self.last_progress_time = now

        dir_vec = err / (dist + 1e-6)
        speed = min(self.max_vel, dist)

        # ── 突变方向过滤：若目标方向相对上一帧变化 > 60° 且当前速度非零，平滑过渡
        # 防止路径点跳变时瞬间横跳，避免 PX4 加速度饱和或地图坐标系混乱
        import math as _mpc_math
        _prev_xy_spd = float(np.linalg.norm(self.prev_vel[:2]))
        if _prev_xy_spd > 0.05:
            _prev_dir = self.prev_vel[:2] / (_prev_xy_spd + 1e-9)
            _curr_dir = dir_vec[:2]
            _dot = float(np.clip(np.dot(_prev_dir, _curr_dir), -1.0, 1.0))
            _angle_diff = _mpc_math.acos(_dot)
            if _angle_diff > _mpc_math.radians(60.0):
                # 角度插值：将当前方向向过去方向偏移，限制最大旋转
                _t = _mpc_math.radians(60.0) / _angle_diff   # 混合比例
                _blended = _prev_dir * (1.0 - _t) + _curr_dir * _t
                _blen = float(np.linalg.norm(_blended))
                if _blen > 1e-6:
                    dir_vec[:2] = _blended / _blen
                self.get_logger().debug(
                    f"[MPC-DIR-CLAMP] angle_diff={_mpc_math.degrees(_angle_diff):.1f}° > 60° "
                    f"→ blended dir=({dir_vec[0]:.2f},{dir_vec[1]:.2f})"
                )

        v_cmd = dir_vec * speed  # ENU velocity

        # Boundary repulsion (ENU, symmetric ±bounds_xy)
        for i in range(2):  # XY only
            lo, hi = -self.bounds_xy, self.bounds_xy
            if curr[i] < lo + self.safety_radius:
                v_cmd[i] += (self.safety_radius - (curr[i] - lo)) * 0.8
            if curr[i] > hi - self.safety_radius:
                v_cmd[i] -= (self.safety_radius - (hi - curr[i])) * 0.8

        # Acceleration clamp
        dv = v_cmd - self.prev_vel
        max_dv = self.max_acc * self.timer_period
        dv = np.clip(dv, -max_dv, max_dv)
        v_cmd = self.prev_vel + dv

        # Speed clamp
        spd = float(np.linalg.norm(v_cmd))
        if spd > self.max_vel:
            v_cmd = v_cmd / spd * self.max_vel

        self.get_logger().debug(
            f"[MPC-CMD-ENU] target=({tgt[0]:.2f},{tgt[1]:.2f}) "
            f"err=({err[0]:.2f},{err[1]:.2f}) "
            f"cmd=({v_cmd[0]:.3f},{v_cmd[1]:.3f},{v_cmd[2]:.3f})"
        )

        msg = LocalCommand()
        tw = Twist()
        # Output: ENU (x=East, y=North) — target_search reads as ENU directly
        tw.linear.x = float(v_cmd[0])
        tw.linear.y = float(v_cmd[1])
        tw.linear.z = float(v_cmd[2])
        tw.angular.z = 0.0
        msg.cmd = tw

        # ★ 近墙限速：当 OccupancyGrid 显示当前位置靠近墙体时，将 XY 速度按距离比例降至安全值
        # 動机半径 0.30m => 在 0.55m 内开始减速，0.20m 内限制蜺居๵ 0.08m/s
        _MPC_BRAKE_START = 0.55   # 开始减速的距离
        _MPC_BRAKE_STOP  = 0.20   # 最小限速小于此距离 -> 强制 0.08m/s
        _MPC_MIN_SPD     = 0.08   # 近墙时最大允许速度
        _occ_clr = self._occ_clearance(self._x_enu, self._y_enu)
        if _occ_clr < _MPC_BRAKE_START:
            _ratio = max(0.0, (_occ_clr - _MPC_BRAKE_STOP) / (_MPC_BRAKE_START - _MPC_BRAKE_STOP))
            _spd_limit = _MPC_MIN_SPD + _ratio * (self.max_vel - _MPC_MIN_SPD)
            _spd_now = float(np.linalg.norm(v_cmd[:2]))
            if _spd_now > _spd_limit:
                v_cmd[:2] = v_cmd[:2] / (_spd_now + 1e-9) * _spd_limit
                tw.linear.x = float(v_cmd[0])
                tw.linear.y = float(v_cmd[1])
                self.get_logger().debug(
                    f'[MPC-NEARWALL] occ_clr={_occ_clr:.2f}m < {_MPC_BRAKE_START:.2f}m '
                    f'spd_limit={_spd_limit:.3f}m/s (was {_spd_now:.3f}m/s)'
                )

        msg.cmd = tw
        self.pub.publish(msg)
        self.prev_vel = v_cmd


def main():
    rclpy.init()
    node = LocalMPC()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
