#!/usr/bin/env python3
"""APF emergency layer with virtual safety shell.
- Subscribes: vehicle pose (/fmu/out/vehicle_local_position_v1).
- Publishes: ApfCommand with ENU velocity override near maze boundaries.

Coordinate convention:
  ENU: x = East (= NED_y = VehicleLocalPosition.y)
       y = North (= NED_x = VehicleLocalPosition.x)
  Output ApfCommand.cmd.linear.x/y are ENU East/North.
  ApfCommand.clearance = minimum boundary distance (m).

Emergency mode (active when clearance < apf_emerg_dist):
  - Only repulsion velocity, capped at apf_emerg_max_vel (≤0.09 m/s).
  - Target_search checks clearance to override local_cmd.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from indoor_search.msg import ApfCommand
from px4_msgs.msg import VehicleLocalPosition
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import String
import math


class ApfNode(Node):
    def __init__(self):
        super().__init__('apf_node')
        self.fmu_prefix = self.declare_parameter('fmu_prefix', '/fmu').value.rstrip('/')
        self.safety_radius   = float(self.declare_parameter('safety_radius', 0.80).value)
        self.stop_radius     = float(self.declare_parameter('stop_radius',   0.40).value)
        self.k_rep           = float(self.declare_parameter('k_rep',         1.5).value)
        self.max_vel         = float(self.declare_parameter('max_vel',       1.2).value)
        # Emergency velocity cap when clearance is very small
        self.emerg_dist      = float(self.declare_parameter('apf_emerg_dist',    0.45).value)
        self.emerg_max_vel   = float(self.declare_parameter('apf_emerg_max_vel', 0.09).value)

        # ENU bounds (x=East, y=North) – default matches maze.world [-6,6] footprint
        _xmin = float(self.declare_parameter('apf_bound_xmin', -6.0).value)
        _xmax = float(self.declare_parameter('apf_bound_xmax',  6.0).value)
        _ymin = float(self.declare_parameter('apf_bound_ymin', -6.0).value)
        _ymax = float(self.declare_parameter('apf_bound_ymax',  6.0).value)
        _zmin = float(self.declare_parameter('apf_bound_zmin', -2.0).value)
        _zmax = float(self.declare_parameter('apf_bound_zmax', -1.0).value)
        # bounds order: [[xmin,xmax],[ymin,ymax],[zmin,zmax]] for ENU
        self.bounds = [[_xmin, _xmax], [_ymin, _ymax], [_zmin, _zmax]]
        self.get_logger().info(
            f"[APF-FRAME] using ENU (x=East=NED_y, y=North=NED_x) "
            f"x=[{_xmin:.1f},{_xmax:.1f}] y=[{_ymin:.1f},{_ymax:.1f}] "
            f"safety_r={self.safety_radius:.2f}m stop_r={self.stop_radius:.2f}m "
            f"emerg_dist={self.emerg_dist:.2f}m emerg_max_vel={self.emerg_max_vel:.3f}m/s"
        )
        self.get_logger().info(
            f"[APF-BOUNDS] x=[{_xmin:.1f},{_xmax:.1f}] y=[{_ymin:.1f},{_ymax:.1f}] "
            f"z=[{_zmin:.1f},{_zmax:.1f}] safety_r={self.safety_radius:.2f}m stop_r={self.stop_radius:.2f}m"
        )
        self.pose = None
        # ── odom 主坐标源（bridge ENU）────────────────────────────────
        # 主：/odom（bridge 发布 ENU）；备： VehicleLocalPosition NED→ENU 转换
        self._odom_received: bool = False
        self._x_odom: float = 0.0     # ENU x = East
        self._y_odom: float = 0.0     # ENU y = North
        self._z_odom: float = 0.0     # odom.z (up positive)
        self._apf_src: str = 'px4_fallback'
        # ── FSM 抑制：MAP_RECOVER/SAFE_HOLD/PLANAR_INVALID 时 APF 不输出 ────────
        self._apf_suppress: bool = False
        self._apf_suppress_reason: str = ''
        self._apf_fsm_state: str = ''   # ★ 当前 FSM 状态（供穿门降权使用）
        self._apf_goal_type: str = 'none'   # ★ 当前导航目标类型（由 /indoor_search/goal_type 订阅）
        # ★ 门口目标类型集合（与 target_search_node 中的 _DOOR_TYPES 一致）
        self._APF_DOOR_TYPES = frozenset((
            'frontier_gap_pre', 'frontier_gap_center', 'frontier_gap_post',
            'frontier_gap_probe', 'confirmed_door'
        ))
        self.create_subscription(
            VehicleLocalPosition,
            f'{self.fmu_prefix}/out/vehicle_local_position_v1',
            self.pose_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, '/odom', self.odom_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            String, '/indoor_search/fsm_state', self._fsm_cb, 1
        )
        # ★ 订阅导航目标类型（用于精确门口降斥模式判断）
        self.create_subscription(
            String, '/indoor_search/goal_type', self._goal_type_cb, 1
        )
        # ── OccupancyGrid for indoor wall repulsion ─────────────────────────
        self._occ_grid: OccupancyGrid = None
        self.create_subscription(
            OccupancyGrid, '/map', self._occ_cb, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(ApfCommand, 'apf_cmd', 10)
        self.timer = self.create_timer(0.05, self.tick)
        import hashlib as _afid_hl, time as _afid_t
        _afid_data = open(__file__, 'rb').read()
        _afid_sha = _afid_hl.sha256(_afid_data).hexdigest()[:16]
        self.get_logger().info(
            f"[FILE-ID] ★ node=apf  file={__file__}  "
            f"sha256={_afid_sha}  started={_afid_t.strftime('%Y-%m-%dT%H:%M:%S')}  "
            f"version=2026-03-01_frame_unified"
        )
        import os as _async_os
        _ainstall_path = _async_os.path.join(
            _async_os.path.dirname(__file__).replace(
                '/ros2_ws/src/indoor_search/src', '/ros2_ws/install/lib/indoor_search'),
            'apf_node.py')
        if _async_os.path.exists(_ainstall_path):
            _ainst_sha = _afid_hl.sha256(open(_ainstall_path,'rb').read()).hexdigest()[:16]
            _async_ok = (_ainst_sha == _afid_sha)
        else:
            _ainst_sha, _async_ok = 'NOT_FOUND', False
        if not _async_ok:
            self.get_logger().error(
                f'[CODE-SYNC-MISMATCH] node=apf install_sha={_ainst_sha} != src_sha={_afid_sha}'
                f' — 请重新 colcon build')
        else:
            self.get_logger().info(f'[CODE-SYNC] node=apf sha={_afid_sha} match=True')

    def pose_cb(self, msg: VehicleLocalPosition):
        self.pose = msg

    # ── OccupancyGrid helpers ───────────────────────────────────────────────
    def _occ_cb(self, msg: OccupancyGrid) -> None:
        self._occ_grid = msg

    def _local_occ_repulsion(self, x_enu: float, y_enu: float,
                             radius: float = 1.0, k_scale: float = 1.0) -> list:
        """Compute repulsion vector from nearby SLAM OccupancyGrid occupied cells.

        Scans a square window of radius `radius` around the drone, accumulates
        unit repulsion vectors from each occupied cell weighted by 1/(dist^2).
        Returns [rx, ry] in ENU m/s (unscaled — caller clips to speed limits).
        """
        g = self._occ_grid
        if g is None:
            return [0.0, 0.0]
        sres = g.info.resolution
        if sres <= 0:
            return [0.0, 0.0]
        rx, ry = 0.0, 0.0
        n_steps = max(1, int(radius / sres))
        ox = g.info.origin.position.x
        oy = g.info.origin.position.y
        for di in range(-n_steps, n_steps + 1):
            for dj in range(-n_steps, n_steps + 1):
                wx = x_enu + di * sres
                wy = y_enu + dj * sres
                col = int((wx - ox) / sres)
                row = int((wy - oy) / sres)
                if not (0 <= col < g.info.width and 0 <= row < g.info.height):
                    continue
                if g.data[row * g.info.width + col] >= 65:  # occupied
                    dx = x_enu - wx
                    dy = y_enu - wy
                    dist = math.hypot(dx, dy)
                    if 0.05 < dist < radius:
                        # Force proportional to penetration depth, inversely to dist^2
                        f = self.k_rep * k_scale * (radius - dist) / (dist * dist + 1e-6)
                        rx += f * dx
                        ry += f * dy
        return [rx, ry]

    def odom_cb(self, msg: Odometry) -> None:
        """Primary ENU pose from bridge. Updates _x_odom/_y_odom (ENU)."""
        self._x_odom = float(msg.pose.pose.position.x)   # ENU East
        self._y_odom = float(msg.pose.pose.position.y)   # ENU North
        self._z_odom = float(msg.pose.pose.position.z)   # ENU Up
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info(
                f'[APF-FRAME] src=bridge_odom first_msg '
                f'enu=({self._x_odom:.2f},{self._y_odom:.2f})'
            )
        self._apf_src = 'bridge_odom'

    def _fsm_cb(self, msg: String) -> None:
        """FSM 状态订阅：MAP_RECOVER/SAFE_HOLD/PLANAR_INVALID/HOVER_SCAN_ESCAPE 时抑制 APF 输出。"""
        _state = msg.data
        self._apf_last_state = _state
        self._apf_fsm_state  = _state

    def _goal_type_cb(self, msg: String) -> None:
        """★ 目标类型订阅（/indoor_search/goal_type）
        用于精确判断是否进入门口 OCC 降权模式。
        只有当前目标是门类型（frontier_gap_*/confirmed_door）时才降弱局部排斥力。
        普通 coverage/frontier 导航期间保持正常避障强度。
        """
        self._apf_goal_type = str(msg.data)
        import time as _agt
        _now_agt = _agt.monotonic()
        _is_door = self._apf_goal_type in self._APF_DOOR_TYPES
        if _now_agt - getattr(self, '_apf_goaltype_log_t', 0.0) > 3.0:
            self._apf_goaltype_log_t = _now_agt
            self.get_logger().info(
                f'[APF-DOOR-MODE] goal_type={self._apf_goal_type} is_door={_is_door}'
            )

    def tick(self):
        if self.pose is None and not self._odom_received:
            return
        # FSM 抑制：在每次 tick 实时评估（不依赖 goal_type 消息驱动），防止 topic 暂停导致状态滞留
        # SEARCH 包括所有导航和穿门阶段，APF 完全沉默，让 last_apf_stamp 自然超时
        _SUPPRESS_STATES = ('MAP_RECOVER', 'SAFE_HOLD', 'PLANAR_INVALID',
                            'TAKEOFF_STABILIZE', 'HOVER_SCAN_ESCAPE',
                            'HOVER_SCAN', 'HOVER_STRICT', 'EMERG',
                            'SEARCH',
                            'TARGET_APPROACH', 'TARGET_DESCEND', 'TARGET_LAND',
                            'TARGET_LOCKED', 'TARGET_ALIGN')
        _cur_state = getattr(self, '_apf_fsm_state', '')
        _suppress_now = any(s in _cur_state for s in _SUPPRESS_STATES)
        if _suppress_now != self._apf_suppress:
            self.get_logger().info(
                f'[APF-SUPPRESS] suppress: {self._apf_suppress}→{_suppress_now} state={_cur_state}'
            )
        self._apf_suppress = _suppress_now
        if self._apf_suppress:
            import time as _sup_t
            _sn = _sup_t.monotonic()
            if _sn - getattr(self, '_apf_suppress_log_t', 0.0) > 2.0:
                self._apf_suppress_log_t = _sn
                self.get_logger().debug(
                    f'[APF-SUPPRESS] NOT publishing (stamp will expire) state={getattr(self,"_apf_last_state","?")}'
                )
            return  # ★ 不发布，让 last_apf_stamp 自然超时

        # 主坐标源：优先使用 bridge odom（ENU），备用 px4 NED→ENU 转换
        if self._odom_received:
            x_enu = self._x_odom
            y_enu = self._y_odom
            z     = -self._z_odom   # odom.z=Up → z_ned=Down（负 = 地面以上）
        else:
            x_enu = float(self.pose.y)   # NED_y = East
            y_enu = float(self.pose.x)   # NED_x = North
            z     = float(self.pose.z)
            self._apf_src = 'px4_fallback'
        p = [x_enu, y_enu, z]

        # 1 Hz [APF-FRAME] 诊断：记录当前坐标来源
        import time as _apf_ft
        _apf_now = _apf_ft.monotonic()
        if _apf_now - getattr(self, '_apf_frame_log_t', 0.0) > 1.0:
            self._apf_frame_log_t = _apf_now
            self.get_logger().debug(
                f'[APF-FRAME] src={self._apf_src} '
                f'enu=({x_enu:.2f},{y_enu:.2f})'
            )

        # Skip if wildly out of bounds (EKF large-jump guard)
        _margin = self.safety_radius * 3.0
        if (p[0] < self.bounds[0][0] - _margin or p[0] > self.bounds[0][1] + _margin or
                p[1] < self.bounds[1][0] - _margin or p[1] > self.bounds[1][1] + _margin):
            self.get_logger().warn(
                f"[APF-SKIP] pose_out_of_bounds enu=({p[0]:.2f},{p[1]:.2f})",
                throttle_duration_sec=5.0
            )
            return

        # Compute boundary distances (ENU):
        # dists order: [west, east, south, north, bottom, top]
        dists = [
            p[0] - self.bounds[0][0],   # dist to west bound  (ENU east ← )
            self.bounds[0][1] - p[0],   # dist to east bound  (ENU east → )
            p[1] - self.bounds[1][0],   # dist to south bound (ENU north ↓)
            self.bounds[1][1] - p[1],   # dist to north bound (ENU north ↑)
            p[2] - self.bounds[2][0],   # dist to z lower     (NED z)
            self.bounds[2][1] - p[2],   # dist to z upper     (NED z)
        ]
        # Use ONLY x/y distances for stop_lock / emergency / clearance reporting.
        # Z is managed by the altitude controller; including z causes false stop_lock
        # when the drone is on the ground (z=0 NED violates z_max=-1.0 NED bound).
        _min_xy_dist = float(min(dists[:4]))
        min_bound_dist = float(min(dists))  # full (used for early-exit gate only)

        if _min_xy_dist >= self.safety_radius:
            return

        # Compute repulsion forces (ENU)
        repel = [0.0, 0.0, 0.0]
        _side_names = ['west', 'east', 'south', 'north', 'bottom', 'top']
        # Axis 0: ENU x (East–West)
        if p[0] < self.bounds[0][0] + self.safety_radius:
            _clr = p[0] - self.bounds[0][0]
            repel[0] += self.k_rep * (self.safety_radius - _clr)  # push East
            self.get_logger().warn(
                f"[APF-BOUNDARY] side=west clearance={_clr:.3f}m repel_x={repel[0]:.3f}",
                throttle_duration_sec=0.5
            )
        if p[0] > self.bounds[0][1] - self.safety_radius:
            _clr = self.bounds[0][1] - p[0]
            repel[0] -= self.k_rep * (self.safety_radius - _clr)  # push West
            self.get_logger().warn(
                f"[APF-BOUNDARY] side=east clearance={_clr:.3f}m repel_x={repel[0]:.3f}",
                throttle_duration_sec=0.5
            )
        # Axis 1: ENU y (North–South)
        if p[1] < self.bounds[1][0] + self.safety_radius:
            _clr = p[1] - self.bounds[1][0]
            repel[1] += self.k_rep * (self.safety_radius - _clr)  # push North
            self.get_logger().warn(
                f"[APF-BOUNDARY] side=south clearance={_clr:.3f}m repel_y={repel[1]:.3f}",
                throttle_duration_sec=0.5
            )
        if p[1] > self.bounds[1][1] - self.safety_radius:
            _clr = self.bounds[1][1] - p[1]
            repel[1] -= self.k_rep * (self.safety_radius - _clr)  # push South
            self.get_logger().warn(
                f"[APF-BOUNDARY] side=north clearance={_clr:.3f}m repel_y={repel[1]:.3f}",
                throttle_duration_sec=0.5
            )
        # Axis 2: z (altitude)
        if p[2] < self.bounds[2][0] + self.safety_radius:
            _clr = p[2] - self.bounds[2][0]
            repel[2] += self.k_rep * (self.safety_radius - _clr)
        if p[2] > self.bounds[2][1] - self.safety_radius:
            _clr = self.bounds[2][1] - p[2]
            repel[2] -= self.k_rep * (self.safety_radius - _clr)

        # ★ OccupancyGrid 局部障碍物排斥力（加到边界排斥力之上）
        # ★ 门口模式判断：仅当当前目标类型属于门类型时降低 OCC 排斥力
        # 原来按 NAV_TO_VIEWPOINT 状态粗纯判断过于粗箙：普通 coverage 导航也会被弱化
        # 现在：只有 goal_type 属于门类型（frontier_gap_*/confirmed_door）时才降弱
        _door_mode = self._apf_goal_type in self._APF_DOOR_TYPES
        _occ_k_scale = 0.25 if _door_mode else 1.0
        if self._occ_grid is not None:
            _occ_rep = self._local_occ_repulsion(
                x_enu, y_enu, radius=self.safety_radius, k_scale=_occ_k_scale)
            repel[0] += _occ_rep[0]
            repel[1] += _occ_rep[1]
            if _door_mode:
                import time as _dm_t
                _dm_now = _dm_t.monotonic()
                if _dm_now - getattr(self, '_door_mode_log_t', 0.0) > 1.0:
                    self._door_mode_log_t = _dm_now
                    _occ_mag = math.hypot(_occ_rep[0], _occ_rep[1])
                    self.get_logger().info(
                        f'[APF-DOOR-MODE] goal_type={self._apf_goal_type} k_scale={_occ_k_scale:.2f} '
                        f'occ_rep_mag={_occ_mag:.3f}m/s (boundary unaffected)'
                    )

        # Hard stop if extremely close (ENU x/y only; z handled by alt controller)
        if _min_xy_dist < self.stop_radius:
            repel[0] = 0.0
            repel[1] = 0.0
            repel[2] = 0.0

        # Determine if this is an emergency call (xy-only; z not included)
        _is_emergency = (_min_xy_dist < self.emerg_dist)
        # Speed clamp (stricter in emergency)
        _vel_limit = self.emerg_max_vel if _is_emergency else self.max_vel
        norm = math.hypot(repel[0], repel[1])
        if norm > _vel_limit and norm > 1e-6:
            scale = _vel_limit / norm
            repel[0] *= scale
            repel[1] *= scale

        if _is_emergency:
            # stop_lock: xy clearance 在 stop_radius 以内，输出已强制为零——标记供上层识别
            _stop_lock = (_min_xy_dist < self.stop_radius)
            self.get_logger().warn(
                f"[APF-EMERGENCY] clearance={_min_xy_dist:.3f}m < {self.emerg_dist:.2f}m "
                f"stop_lock={_stop_lock} "
                f"cmd_enu=({repel[0]:.3f},{repel[1]:.3f}) limited_to={_vel_limit:.3f}m/s",
                throttle_duration_sec=0.5
            )
            self.get_logger().info(
                f"[APF-OVERRIDE] active=True reason=boundary_proximity "
                f"clearance={_min_xy_dist:.3f}m stop_lock={_stop_lock}",
                throttle_duration_sec=1.0
            )

        tw = Twist()
        tw.linear.x = float(repel[0])   # ENU East
        tw.linear.y = float(repel[1])   # ENU North
        tw.linear.z = float(repel[2])

        msg = ApfCommand()
        msg.cmd = tw
        msg.clearance = float(_min_xy_dist)  # xy-only clearance reported to target_search
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ApfNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
