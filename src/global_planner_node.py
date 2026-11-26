#!/usr/bin/env python3
"""Global planner (Octo-RRT* inspired, heuristic coverage).
- Builds a live coverage grid and samples waypoints within bounds.
- Uses heuristic weights: coverage 0.7, visibility 0.2, boundary 0.1 (2 Hz).

Coordinate convention (MUST match all nodes):
  ENU: x = East (= NED_y = VehicleLocalPosition.y)
       y = North (= NED_x = VehicleLocalPosition.x)
  Bounds: plan_bound_xmin/xmax (ENU East), plan_bound_ymin/ymax (ENU North)
  All published waypoints are in ENU.
  z uses NED altitude convention (negative = above ground).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
from geometry_msgs.msg import Pose
from px4_msgs.msg import VehicleLocalPosition
from indoor_search.msg import SearchPath
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import String


class GlobalPlanner(Node):
    def __init__(self):
        super().__init__('global_planner')
        # ENU bounds (x=East, y=North) \u2013 match maze.world [-6,6] footprint
        _pxmin = float(self.declare_parameter('plan_bound_xmin', -6.0).value)
        _pxmax = float(self.declare_parameter('plan_bound_xmax',  6.0).value)
        _pymin = float(self.declare_parameter('plan_bound_ymin', -6.0).value)
        _pymax = float(self.declare_parameter('plan_bound_ymax',  6.0).value)
        _pzmin = float(self.declare_parameter('plan_bound_zmin', -2.0).value)
        _pzmax = float(self.declare_parameter('plan_bound_zmax', -1.0).value)
        self.bounds = np.array([[_pxmin, _pxmax], [_pymin, _pymax], [_pzmin, _pzmax]])
        self.wall_clearance = float(self.declare_parameter('wall_clearance', 0.50).value)
        # No hardcoded obstacles \u2014 boundary enforcement done via clearance check
        self.obstacles = []

        self.coverage_weight  = 0.7
        self.fov_weight       = 0.2
        self.boundary_weight  = 0.1
        self.diversity_weight = 0.2

        self.get_logger().info(
            f"[PLAN-FRAME] using ENU (x=East=NED_y, y=North=NED_x) "
            f"x=[{_pxmin:.1f},{_pxmax:.1f}] y=[{_pymin:.1f},{_pymax:.1f}] "
            f"wall_clearance={self.wall_clearance:.2f}m"
        )
        self.get_logger().info(
            f"[GLOBAL-PLAN-BOUNDS] x=[{_pxmin:.1f},{_pxmax:.1f}] "
            f"y=[{_pymin:.1f},{_pymax:.1f}] z=[{_pzmin:.1f},{_pzmax:.1f}]"
        )

        self.timer = self.create_timer(0.5, self.tick)
        self.pub = self.create_publisher(SearchPath, 'global_path', 10)

        # ENU pose cache
        self._x_enu: float = 0.0
        self._y_enu: float = 0.0
        self.pose = None
        # ── odom 主坐标源（bridge ENU）────────────────────────────────
        self._odom_received: bool = False
        self._plan_src: str = 'px4_fallback'
        # ── FSM 抖制：MAP_RECOVER/SAFE_HOLD/PLANAR_INVALID 时停止规划 ──────────────
        self._gp_suppress: bool = False
        self._gp_suppress_reason: str = ''
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.pose_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, '/odom', self._odom_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            String, '/indoor_search/fsm_state', self._fsm_cb, 1
        )
        # ── OccupancyGrid for indoor wall hard constraints ─────────────────────────
        self._occ_grid: OccupancyGrid = None
        self.create_subscription(
            OccupancyGrid, '/map', self._occ_cb, qos_profile_sensor_data
        )

        # Coverage grid (ENU)
        self.grid_res = 0.5
        xs = np.arange(self.bounds[0, 0], self.bounds[0, 1] + self.grid_res, self.grid_res)
        ys = np.arange(self.bounds[1, 0], self.bounds[1, 1] + self.grid_res, self.grid_res)
        self.grid_shape = (len(xs), len(ys))
        self.visited = np.zeros(self.grid_shape, dtype=bool)
        self.last_path = None

        # Doorway anchors \u2014 ENU positions of known passable door centers in maze.world.
        # Previous incorrect values (x=\u00b15.4, y=0.5) were near walls and have been removed.
        # Only safe door-centre estimates are kept; these are used as exploration seeds.
        z_mid = 0.5 * (self.bounds[2, 0] + self.bounds[2, 1])
        self.doorway_points = np.array([
            [-3.0, -3.5, z_mid],   # west-side door centre (ENU)
            [ 3.0, -3.5, z_mid],   # east-side door centre (ENU)
        ], dtype=float)
        self._plan_log_cnt = 0
        self._plan_frame_log_t: float = 0.0  # throttle for [PLAN-FRAME]
        # ★ 地图质量门控：區域内 occ 格数过少时，企止激进规划和门洞带入
        self._min_occ_cells_plan  = int(self.declare_parameter('min_occ_cells_for_planning', 10).value)
        self._occ_cell_count:  int  = 0       # 当前可用 OccupancyGrid 的 occ 格数
        self._map_quality_ok:  bool = False   # occ 格数 >= 阈值
        self._map_thin_log_t:  float = 0.0   # MAP_THIN 日志节流
        import hashlib as _gfid_hl, time as _gfid_t
        _gfid_data = open(__file__, 'rb').read()
        _gfid_sha = _gfid_hl.sha256(_gfid_data).hexdigest()[:16]
        self.get_logger().info(
            f"[FILE-ID] ★ node=global_planner  file={__file__}  "
            f"sha256={_gfid_sha}  started={_gfid_t.strftime('%Y-%m-%dT%H:%M:%S')}  "
            f"version=2026-03-01_frame_unified  "
            f"doorway_pts={len(self.doorway_points)} coord=ENU"
        )
        import os as _gsync_os
        _ginstall_path = _gsync_os.path.join(
            _gsync_os.path.dirname(__file__).replace(
                '/ros2_ws/src/indoor_search/src', '/ros2_ws/install/lib/indoor_search'),
            'global_planner_node.py')
        if _gsync_os.path.exists(_ginstall_path):
            _ginst_sha = _gfid_hl.sha256(open(_ginstall_path,'rb').read()).hexdigest()[:16]
            _gsync_ok = (_ginst_sha == _gfid_sha)
        else:
            _ginst_sha, _gsync_ok = 'NOT_FOUND', False
        if not _gsync_ok:
            self.get_logger().error(
                f'[CODE-SYNC-MISMATCH] node=global_planner install_sha={_ginst_sha} != src_sha={_gfid_sha}'
                f' — 请重新 colcon build')
        else:
            self.get_logger().info(f'[CODE-SYNC] node=global_planner sha={_gfid_sha} match=True')

    # ── OccupancyGrid wall helpers ─────────────────────────────────────────────
    def _occ_cb(self, msg: OccupancyGrid) -> None:
        self._occ_grid = msg
        # ★ 计算 occ 格数，更新地图质量标志
        import time as _oq_t
        _occ_cnt = sum(1 for v in msg.data if v >= 65)
        _free_cnt = sum(1 for v in msg.data if 0 <= v <= 30)
        self._occ_cell_count = _occ_cnt
        _was_ok = self._map_quality_ok
        self._map_quality_ok = (_occ_cnt >= self._min_occ_cells_plan)
        _now_oq = _oq_t.monotonic()
        if not self._map_quality_ok and _now_oq - self._map_thin_log_t > 5.0:
            self._map_thin_log_t = _now_oq
            self.get_logger().warn(
                f'[PLAN-MAP-QUALITY] MAP_THIN: occ={_occ_cnt} free={_free_cnt} '
                f'< min={self._min_occ_cells_plan} → 封印激进规划，使用保守模式'
            )
        elif self._map_quality_ok and not _was_ok:
            self.get_logger().info(
                f'[PLAN-MAP-QUALITY] 地图质量恢复: occ={_occ_cnt} >= {self._min_occ_cells_plan}'
            )

    def _is_occ_inflated(self, x: float, y: float,
                         inflation_m: float = 0.30) -> bool:
        """Return True if ENU (x,y) is within inflation_m of any occupied cell.

        Used to reject candidate goals that are too close to walls even if the
        exact cell is free.  inflation_m should be >= drone body radius.
        """
        g = self._occ_grid
        if g is None:
            return False
        sres = g.info.resolution
        if sres <= 0:
            return False
        ox = g.info.origin.position.x
        oy = g.info.origin.position.y
        n_steps = max(1, int(math.ceil(inflation_m / sres)))
        for di in range(-n_steps, n_steps + 1):
            for dj in range(-n_steps, n_steps + 1):
                wx = x + di * sres
                wy = y + dj * sres
                if math.hypot(wx - x, wy - y) > inflation_m:
                    continue
                col = int((wx - ox) / sres)
                row = int((wy - oy) / sres)
                if 0 <= col < g.info.width and 0 <= row < g.info.height:
                    if g.data[row * g.info.width + col] >= 65:
                        return True
        return False

    def _is_occ_occupied(self, x: float, y: float) -> bool:
        """Return True if ENU point (x,y) falls in an occupied OccupancyGrid cell (val≥65)."""
        g = self._occ_grid
        if g is None:
            return False
        sres = g.info.resolution
        if sres <= 0:
            return False
        col = int((x - g.info.origin.position.x) / sres)
        row = int((y - g.info.origin.position.y) / sres)
        if col < 0 or col >= g.info.width or row < 0 or row >= g.info.height:
            return False  # outside map → treat as free
        return g.data[row * g.info.width + col] >= 65

    def _has_clear_path(self, x0: float, y0: float, x1: float, y1: float,
                        steps: int = 24) -> bool:
        """Bresenham-style raycast: True iff no occupied cell between (x0,y0)→(x1,y1)."""
        if self._occ_grid is None:
            return True  # no map yet → optimistic
        for i in range(1, steps + 1):
            t = i / steps
            if self._is_occ_occupied(x0 + t * (x1 - x0), y0 + t * (y1 - y0)):
                return False
        return True

    def _is_safe_clearance(self, p: np.ndarray) -> bool:
        """Return True only if the point is \u2265 wall_clearance away from every bound."""
        clr = self.wall_clearance
        if (p[0] < self.bounds[0, 0] + clr or p[0] > self.bounds[0, 1] - clr or
                p[1] < self.bounds[1, 0] + clr or p[1] > self.bounds[1, 1] - clr):
            return False
        return True

    def is_free(self, p) -> bool:
        x, y = p[0], p[1]
        for (xmin, xmax, ymin, ymax) in self.obstacles:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return False
        return True

    def tick(self):
        # ── FSM 抖制：MAP_RECOVER/SAFE_HOLD/PLANAR_INVALID 时停止规划 ──────────────────
        if getattr(self, '_gp_suppress', False):
            return  # 不发布新路径（保留上次路径，前驱 MPC 会自然停止）
        # Skip when ENU pose is out of bounds
        if self.pose is not None:
            _lo, _hi = self.bounds[:2, 0], self.bounds[:2, 1]
            if not (_lo[0] <= self._x_enu <= _hi[0] and _lo[1] <= self._y_enu <= _hi[1]):
                self.get_logger().warn(
                    f"[GLOBAL-PLAN-SKIP] pose_out_of_bounds enu=({self._x_enu:.2f},{self._y_enu:.2f})",
                    throttle_duration_sec=5.0
                )
                return
        # ★ 地图质量门控： MAP_THIN 时使用保守模式采样
        # 当 OccupancyGrid 内墙体格数过少时，地图几何不可信，不应激进导航/使用预设门洞点
        # 保守模式：减少采样数，禁止门洞种子注入，只在已知 free 区做正常匹盖
        _map_thin = not self._map_quality_ok
        n_samples = 30 if _map_thin else 80  # MAP_THIN 时减少采样，避免向未知区漫游
        samples, src_list = self.random_samples(n_samples)
        scores = self.score(samples)
        order = np.argsort(-scores)
        best_idx = order[:6]
        best = samples[best_idx]
        best_sources = [src_list[int(i)] for i in best_idx]
        # ★ 最小推远距离约束：目标点距当前位置 < 0.8m 则视为无效探索目标，过滤掉
        # 目的：避免 " dbg-plan 贴着当前点"现象
        _PLAN_MIN_DIST = 0.8   # m
        if self.pose is not None:
            _f_best = []
            _f_src  = []
            for _bi, _bs in zip(best, best_sources):
                _pd = float(np.hypot(_bi[0] - self._x_enu, _bi[1] - self._y_enu))
                if _pd >= _PLAN_MIN_DIST:
                    _f_best.append(_bi)
                    _f_src.append(_bs)
            if _f_best:
                best = np.array(_f_best)
                best_sources = _f_src
            else:
                self.get_logger().warn(
                    f"[PLAN-TOO-CLOSE] 所有候选点距当前位置 < {_PLAN_MIN_DIST:.1f}m, 保留最远点"
                )
                # 按距离降序，保留距离最远的一个点
                _dists = [float(np.hypot(bi[0]-self._x_enu, bi[1]-self._y_enu))
                          for bi in np.array(list(samples[best_idx]))]
                _farthest = int(np.argmax(_dists))
                best = np.array([best[_farthest]])
                best_sources = [best_sources[_farthest]]
        if self._occ_grid is not None:
            _safe_best = []
            _safe_src = []
            for _bi, _bs in zip(best, best_sources):
                if self._has_clear_path(self._x_enu, self._y_enu, _bi[0], _bi[1]):
                    _safe_best.append(_bi)
                    _safe_src.append(_bs)
                else:
                    self.get_logger().warn(
                        f'[PLAN-BLOCKED] goal=({_bi[0]:.2f},{_bi[1]:.2f}) '
                        f'blocked by wall, excluded from path',
                        throttle_duration_sec=3.0
                    )
            if _safe_best:
                best = np.array(_safe_best)
                best_sources = _safe_src
            else:
                # ★[I1] 所有候选均被墙体 LOS 阻断 → 不发布未过滤路径
                # 原版直接发布墙体内目标 → local_mpc [MPC-BLOCKED] 反复卡死
                # 修复：跳过本帧，重发上次有效路径（MPC 保持上次速度），等待地图更新后重规划
                self.get_logger().warn(
                    '[PLAN-BLOCKED] all candidates wall-blocked → skip publish, re-use last path',
                    throttle_duration_sec=5.0
                )
                if self.last_path is not None:
                    self.pub.publish(self.last_path)
                return
        msg = SearchPath()
        self._plan_log_cnt += 1
        for p in best:
            pose = Pose()
            pose.position.x = float(p[0])  # ENU east
            pose.position.y = float(p[1])  # ENU north
            pose.position.z = float(p[2])
            msg.poses.append(pose)
            self.mark_visited(p)
        self.pub.publish(msg)
        self.last_path = msg
        # Periodic [PLAN-GOAL] log — every 2 s (4 ticks @ 0.5 Hz)
        if self._plan_log_cnt % 4 == 1 and len(best) > 0:
            _src0 = best_sources[0] if best_sources else 'unknown'
            _clr = self._point_clearance(best[0])
            self.get_logger().info(
                f'[PLAN-GOAL] src={_src0} pose_src={self._plan_src} '
                f'goal=({best[0][0]:.2f},{best[0][1]:.2f}) '
                f'clearance={_clr:.2f}m cnt={self._plan_log_cnt} (ENU)'
            )
            # Count per-source breakdown for diagnostics
            _n_cov = sum(1 for s in best_sources if s == 'coverage_sample')
            _n_fbk = sum(1 for s in best_sources if s == 'fallback_door')
            if _n_fbk > 0:
                self.get_logger().info(
                    f'[PLAN-GOAL] top6 breakdown: coverage_sample={_n_cov} '
                    f'fallback_door={_n_fbk}'
                )

    def _point_clearance(self, p: np.ndarray) -> float:
        """Minimum clearance to any world boundary."""
        return float(min(
            p[0] - self.bounds[0, 0],
            self.bounds[0, 1] - p[0],
            p[1] - self.bounds[1, 0],
            self.bounds[1, 1] - p[1],
        ))

    def random_samples(self, n: int):
        """Return (samples: np.ndarray shape (n,3), sources: list[str]).

        source tags:
          'coverage_sample' – normal random point inside safe bounds (wall_clearance)
          'fallback_door'   – hardcoded doorway anchor used only as last resort

        Frontier-gap protection:
          frontier_gap points are NEVER injected here.  Any caller that wants to
          try frontier-gap targets must tag them 'unconfirmed_gap' and pass them
          through _is_safe_clearance independently, then emit
          [PLAN-REJECT] reason=frontier_gap_only if rejected.
        """
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        samples: list = []
        sources: list = []
        attempts = 0
        max_attempts = n * 20
        rejected_near_wall = 0
        rejected_other = 0
        while len(samples) < n and attempts < max_attempts:
            p = lo + (hi - lo) * np.random.rand(3)
            attempts += 1
            if not self.is_free(p):
                rejected_other += 1
                continue
            if not self._is_safe_clearance(p):
                rejected_near_wall += 1
                continue
            # ★ 硬约束：拒绝落在 SLAM 墙体 OccupancyGrid 占用格中的点
            if self._is_occ_occupied(p[0], p[1]):
                rejected_other += 1
                continue
            # ★ 硬约束：拒绝路径点走 inflation_radius(0.30m) 内有展膨墙体
            if self._is_occ_inflated(p[0], p[1], inflation_m=0.30):
                rejected_near_wall += 1
                continue
            samples.append(p)
            sources.append('coverage_sample')
        if rejected_near_wall > n // 2:
            self.get_logger().warn(
                f'[PLAN-REJECT] reason=near_wall '
                f'count={rejected_near_wall} other={rejected_other} '
                f'attempts={attempts}'
            )
        # ── fallback: doorway_points 仅当采样数 < n/4 且地图质量OK时才补充 ──
        # ★ MAP_THIN: 地图几乎无墙时禁用门洞种子注入（坐标未经验证，可能把无人机推入墙）
        # 只有当 _map_quality_ok=True（occ格足够多，几何可信）时才允许
        if len(samples) < n // 4 and self._map_quality_ok:
            for pt in self.doorway_points:
                if not self.is_free(pt) or not self._is_safe_clearance(pt):
                    self.get_logger().warn(
                        f'[PLAN-REJECT] reason=near_wall src=fallback_door '
                        f'pt=({pt[0]:.1f},{pt[1]:.1f}) '
                        f'clearance={self._point_clearance(pt):.2f}m '
                        f'(<{self.wall_clearance:.2f}m threshold)'
                    )
                    continue
                # ★ 硬约束：门洞点也必须通过 OCC 占用 + 膨胀检查
                if self._is_occ_occupied(pt[0], pt[1]):
                    self.get_logger().warn(
                        f'[PLAN-REJECT] reason=occ_occupied src=fallback_door '
                        f'pt=({pt[0]:.1f},{pt[1]:.1f})'
                    )
                    continue
                if self._is_occ_inflated(pt[0], pt[1], inflation_m=0.30):
                    self.get_logger().warn(
                        f'[PLAN-REJECT] reason=occ_inflated src=fallback_door '
                        f'pt=({pt[0]:.1f},{pt[1]:.1f}) inflation=0.30m'
                    )
                    continue
                samples.append(pt.copy())
                sources.append('fallback_door')
                self.get_logger().debug(
                    f'[PLAN-FALLBACK] src=fallback_door '
                    f'pt=({pt[0]:.1f},{pt[1]:.1f}) '
                    f'clearance={self._point_clearance(pt):.2f}m '
                    f'total={len(samples)}'
                )
        if not samples:
            _ctr = (lo + hi) * 0.5
            samples.append(_ctr)
            sources.append('coverage_sample')
        while len(samples) < n:
            samples.append(samples[-1])
            sources.append(sources[-1])
        return np.array(samples), sources

    def score(self, pts: np.ndarray) -> np.ndarray:
        # Coverage: unvisited cells
        idx = np.clip(
            ((pts[:, :2] - self.bounds[:2, 0]) / self.grid_res).astype(int),
            0, np.array(self.grid_shape) - 1
        )
        cov = (~self.visited[idx[:, 0], idx[:, 1]]).astype(float)
        # FOV: prefer interior (not edges)
        span = max(self.bounds[0, 1] - self.bounds[0, 0], 1.0)
        center_bias = 1.0 - 0.2 * (np.linalg.norm(pts[:, :2], axis=1) / (span * 0.5))
        center_bias = np.clip(center_bias, 0.0, 1.0)
        # Boundary score: mild pull toward edges (but safe clearance)
        dist_edge = np.minimum(pts[:, 0] - self.bounds[0, 0], self.bounds[0, 1] - pts[:, 0])
        dist_edge = np.minimum(
            dist_edge,
            np.minimum(pts[:, 1] - self.bounds[1, 0], self.bounds[1, 1] - pts[:, 1])
        )
        edge_score = 1.0 - np.clip(dist_edge / (span * 0.5), 0.0, 1.0)
        # Diversity: push to opposite side of current ENU pose
        diversity = np.zeros_like(center_bias)
        if self.pose is not None:
            sign_x = np.sign(self._x_enu)  # ENU east
            diversity = 0.5 + 0.5 * (-sign_x) * np.sign(pts[:, 0])
        return (
            self.coverage_weight  * cov
            + self.fov_weight      * center_bias
            + self.boundary_weight * edge_score
            + self.diversity_weight * diversity
        )

    def mark_visited(self, p):
        ix = int((p[0] - self.bounds[0, 0]) / self.grid_res)
        iy = int((p[1] - self.bounds[1, 0]) / self.grid_res)
        ix = int(np.clip(ix, 0, self.grid_shape[0] - 1))
        iy = int(np.clip(iy, 0, self.grid_shape[1] - 1))
        self.visited[ix, iy] = True

    def pose_cb(self, msg: VehicleLocalPosition):
        self.pose = msg
        # Convert PX4 NED to ENU: x_enu=East=NED_y, y_enu=North=NED_x
        # 备用：仅当 odom 未到时使用
        if not self._odom_received:
            self._x_enu = float(msg.y)
            self._y_enu = float(msg.x)
            self._plan_src = 'px4_fallback'
        # 1 Hz [PLAN-FRAME] diagnostic: show raw NED → ENU conversion
        import time as _pf_t
        _pf_now = _pf_t.monotonic()
        if not self._odom_received and _pf_now - self._plan_frame_log_t > 1.0:
            self._plan_frame_log_t = _pf_now
            self.get_logger().debug(
                f'[PLAN-FRAME] src=px4_fallback raw_ned=({msg.x:.2f},{msg.y:.2f}) '
                f'-> enu=({self._x_enu:.2f},{self._y_enu:.2f})'
            )

    def _odom_cb(self, msg: Odometry) -> None:
        """Primary ENU pose from bridge."""
        self._x_enu = float(msg.pose.pose.position.x)   # ENU East
        self._y_enu = float(msg.pose.pose.position.y)   # ENU North
        self._plan_src = 'bridge_odom'
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info(
                f'[PLAN-FRAME] src=bridge_odom first_msg '
                f'enu=({self._x_enu:.2f},{self._y_enu:.2f})'
            )
        # 1 Hz [PLAN-FRAME] diagnostic
        import time as _pf_t
        _pf_now = _pf_t.monotonic()
        if _pf_now - self._plan_frame_log_t > 1.0:
            self._plan_frame_log_t = _pf_now
            self.get_logger().debug(
                f'[PLAN-FRAME] src={self._plan_src} '
                f'enu=({self._x_enu:.2f},{self._y_enu:.2f})'
            )

    def _fsm_cb(self, msg: String) -> None:
        """FSM 状态订阅：MAP_RECOVER/SAFE_HOLD/PLANAR_INVALID 时暂停全局规划。"""
        _state = msg.data
        # ★ 全面扩展抑制列表 → 与 local_mpc _PREEMPT 保持一致
        # 目的：SEARCH/NAV_TO_VIEWPOINT 等所有主动导航状态下，
        #   target_search SSG 独占目标选择，global_planner 停止推路径
        _SUPPRESS_STATES = ('MAP_RECOVER', 'SAFE_HOLD', 'PLANAR_INVALID',
                            'TAKEOFF_STABILIZE', 'HOVER_SCAN_ESCAPE',
                            'HOVER_SCAN', 'HOVER_STRICT', 'EMERG',
                            'NAV_TO_VIEWPOINT', 'SEARCH',
                            'TARGET_APPROACH', 'TARGET_DESCEND', 'TARGET_LAND',
                            'TARGET_LOCKED', 'TARGET_ALIGN')
        _was = self._gp_suppress
        self._gp_suppress = any(s in _state for s in _SUPPRESS_STATES)
        if self._gp_suppress != _was:
            self.get_logger().info(
                f'[GP-SUPPRESS] suppress={self._gp_suppress} state={_state} '
                f'reason=target_search_exclusive'
            )


def main():
    rclpy.init()
    node = GlobalPlanner()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
