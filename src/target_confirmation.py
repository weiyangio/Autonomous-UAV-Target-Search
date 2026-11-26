#!/usr/bin/env python3
import time


class TargetConfirmation:
    """多帧确认器：连续命中达到阈值才确认目标。"""

    def __init__(self, required_hits: int = 5, conf_threshold: float = 0.5, timeout_sec: float = 1.0) -> None:
        self.required_hits = int(required_hits)
        self.conf_threshold = float(conf_threshold)
        self.timeout_sec = float(timeout_sec)

        self.hit_streak = 0
        self.last_hit_time = None
        self.confirmed = False

    def update(self, max_conf: float, now_sec: float | None = None) -> bool:
        if now_sec is None:
            now_sec = time.time()

        if self.last_hit_time is not None and (now_sec - self.last_hit_time) > self.timeout_sec:
            self.hit_streak = 0

        if max_conf >= self.conf_threshold:
            self.hit_streak += 1
            self.last_hit_time = now_sec
        else:
            self.hit_streak = max(0, self.hit_streak - 1)

        if self.hit_streak >= self.required_hits:
            self.confirmed = True

        return self.confirmed

    def reset(self) -> None:
        self.hit_streak = 0
        self.last_hit_time = None
        self.confirmed = False
