#!/usr/bin/env python3
import time as _time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SafetyState:
    safe: bool
    reason: str
    action: str


class SafetyMonitor:
    """独立安全监控：只做判定，不直接发控制指令。"""

    def __init__(
        self,
        min_offboard_hold_sec: float = 1.0,
        max_pose_age_sec: float = 0.5,
        max_scan_age_sec: float = 1.2,
        z_lower_bound: float = -2.8,
        z_upper_bound: float = -0.05,
        obstacle_stop_dist: float = 0.28,
        obstacle_release_dist: float = 0.34,
        obstacle_hold_hits: int = 2,
        obstacle_hold_max_sec: float = 8.0,
    ) -> None:
        self.min_offboard_hold_sec = float(min_offboard_hold_sec)
        self.max_pose_age_sec = float(max_pose_age_sec)
        self.max_scan_age_sec = float(max_scan_age_sec)
        self.z_lower_bound = float(z_lower_bound)
        self.z_upper_bound = float(z_upper_bound)
        self.obstacle_stop_dist = float(obstacle_stop_dist)
        self.obstacle_release_dist = float(obstacle_release_dist)
        self.obstacle_hold_hits = int(max(1, obstacle_hold_hits))

        self.last_safe_time = None
        self._obstacle_hits = 0
        self._obstacle_hold_start: Optional[float] = None
        self.obstacle_hold_max_sec: float = float(obstacle_hold_max_sec)  # 超过此时间强制解除障碍物死锁

    def evaluate(
        self,
        node,
        now,
        enforce_altitude_bounds: bool = True,
        pose_age_limit_sec: Optional[float] = None,
    ) -> SafetyState:
        if node.pose is None:
            return SafetyState(False, "pose_missing", "hold")

        if node.last_pose_msg_time is None:
            return SafetyState(False, "pose_stamp_missing", "hold")

        pose_age = (now - node.last_pose_msg_time).nanoseconds * 1e-9
        pose_age_limit = self.max_pose_age_sec if pose_age_limit_sec is None else float(pose_age_limit_sec)
        if pose_age > pose_age_limit:
            return SafetyState(False, f"pose_stale:{pose_age:.2f}s", "hold")

        if enforce_altitude_bounds:
            z = float(node.pose.z)
            # NED坐标：z越负=飞越高。z_lower_bound<0（如-2.6）= 高度上限。
            # z < z_lower_bound → 飞太高 → 需要下降（vz>0，NED向下）
            # z > z_upper_bound → 高度太低/在地面 → 需要上升（vz<0，NED向上）
            if z < self.z_lower_bound:
                return SafetyState(False, f"z_too_high_alt:{z:.2f}", "descend")
            if z > self.z_upper_bound:
                return SafetyState(False, f"z_too_low_alt:{z:.2f}", "ascend")

        if node.last_scan_time is not None:
            scan_age = (now - node.last_scan_time).nanoseconds * 1e-9
            if scan_age > self.max_scan_age_sec:
                return SafetyState(False, f"scan_stale:{scan_age:.2f}s", "hold")

        nearest = None
        for key in ("front", "left", "right", "back"):
            value = node.lidar_ranges.get(key) if hasattr(node, "lidar_ranges") else None
            if value is None:
                continue
            v = float(value)
            if nearest is None or v < nearest:
                nearest = v
        if nearest is not None:
            if nearest < self.obstacle_stop_dist:
                self._obstacle_hits += 1
            elif nearest > self.obstacle_release_dist:
                self._obstacle_hits = 0
                self._obstacle_hold_start = None
            if self._obstacle_hits >= self.obstacle_hold_hits:
                now_mono = _time.monotonic()
                if self._obstacle_hold_start is None:
                    self._obstacle_hold_start = now_mono
                elif now_mono - self._obstacle_hold_start > self.obstacle_hold_max_sec:
                    # 卡墙超时：强制重置，允许逃脱逻辑介入
                    self._obstacle_hits = 0
                    self._obstacle_hold_start = None
                else:
                    return SafetyState(False, f"obstacle_too_close:{nearest:.2f}m", "hold")
            else:
                self._obstacle_hold_start = None

        if node.armed and not node.offboard_mode:
            armed_since = getattr(node, "armed_since", None)
            if armed_since is not None:
                armed_time = (now - armed_since).nanoseconds * 1e-9
                if armed_time < self.min_offboard_hold_sec:
                    return SafetyState(True, "offboard_transition", "none")
            return SafetyState(False, "offboard_lost", "hold")

        self.last_safe_time = now
        return SafetyState(True, "ok", "none")

    @staticmethod
    def safety_velocity(action: str, climb_speed: float, descend_speed: float) -> Tuple[float, float, float]:
        # NED坐标系：vz<0=上升(ascend)，vz>0=下降(descend)
        if action == "ascend":
            return (0.0, 0.0, -abs(climb_speed))   # vz负值 → 高度增加（z减小）
        if action == "descend":
            return (0.0, 0.0, +abs(descend_speed))  # vz正值 → 高度减小（z增大）
        return (0.0, 0.0, 0.0)
