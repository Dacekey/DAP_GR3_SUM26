"""
Health Monitoring Module.

Tracks blink rate, session duration, and drowsiness events to provide
real-time health status feedback to the user.

All alert thresholds are imported from ``config.py``.
"""

import os
import sys
import time
import logging
from collections import deque

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BLINK_RATE_NORMAL,
    BLINK_RATE_WARNING,
    BLINK_RATE_DANGER,
    LONG_CLOSE_WARNING_SEC,
    LONG_CLOSE_DANGER_SEC,
    USAGE_WARNING_MINUTES,
    USAGE_DANGER_MINUTES,
)

logger = logging.getLogger(__name__)

# Rolling window size in seconds for blink-rate calculation
_ROLLING_WINDOW_SEC = 60.0


class HealthMonitor:
    """
    Monitor the user's eye-health metrics in real time.

    Maintains a rolling window of blink timestamps to compute blinks-per-
    minute, tracks total session time, and evaluates drowsiness based on
    prolonged eye closure duration.

    Usage
    -----
    >>> monitor = HealthMonitor()
    >>> monitor.update(blink_detected=True, closed_duration=0.0)
    >>> status = monitor.get_health_status()
    """

    def __init__(self) -> None:
        """Initialize tracking variables and start the session timer."""
        self._blink_timestamps: deque[float] = deque()
        self._session_start: float = time.time()
        self._total_blinks: int = 0
        self._current_alert: str = "normal"
        self._drowsiness_alert: bool = False
        self._last_closed_duration: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, blink_detected: bool, closed_duration: float) -> None:
        """
        Called every frame to update health metrics.

        Parameters
        ----------
        blink_detected : bool
            True if a complete blink was detected on this frame.
        closed_duration : float
            How many seconds the eyes have been continuously closed
            (0.0 when eyes are open).
        """
        now = time.time()

        # Record blink timestamp
        if blink_detected:
            self._blink_timestamps.append(now)
            self._total_blinks += 1

        # Prune timestamps outside the rolling window
        window_start = now - _ROLLING_WINDOW_SEC
        while self._blink_timestamps and self._blink_timestamps[0] < window_start:
            self._blink_timestamps.popleft()

        # Update drowsiness flag
        self._last_closed_duration = closed_duration
        self._drowsiness_alert = closed_duration >= LONG_CLOSE_DANGER_SEC

    def get_blink_rate(self) -> float:
        """
        Compute the rolling blink rate (blinks per minute).

        Uses the number of blinks recorded in the last 60-second window.

        Returns
        -------
        float
            Blinks per minute.
        """
        now = time.time()
        elapsed = now - self._session_start
        if elapsed < _ROLLING_WINDOW_SEC:
            # Session is shorter than the window – extrapolate
            if elapsed < 1.0:
                return 0.0
            return len(self._blink_timestamps) * (_ROLLING_WINDOW_SEC / elapsed)
        return float(len(self._blink_timestamps))

    def get_usage_time(self) -> float:
        """
        Get the elapsed session time in minutes.

        Returns
        -------
        float
            Minutes since session start.
        """
        return (time.time() - self._session_start) / 60.0

    def get_health_status(self) -> dict:
        """
        Compute and return a comprehensive health-status dictionary.

        Returns
        -------
        dict
            Keys:
            - ``blink_rate`` (float): blinks per minute.
            - ``blink_rate_status`` (str): ``'normal'`` / ``'warning'`` / ``'danger'``.
            - ``usage_time_min`` (float): minutes since session start.
            - ``usage_status`` (str): ``'normal'`` / ``'warning'`` / ``'danger'``.
            - ``drowsiness_alert`` (bool): True if eyes closed ≥ 2 s.
            - ``overall_status`` (str): worst of all individual statuses.
            - ``message`` (str): human-readable health message in Vietnamese.
        """
        blink_rate = self.get_blink_rate()
        usage_min = self.get_usage_time()

        # ---- Blink-rate status -----------------------------------------------
        if blink_rate >= BLINK_RATE_NORMAL:
            br_status = "normal"
        elif blink_rate >= BLINK_RATE_WARNING:
            br_status = "warning"
        else:
            br_status = "danger"

        # ---- Usage-time status -----------------------------------------------
        if usage_min < USAGE_WARNING_MINUTES:
            usage_status = "normal"
        elif usage_min < USAGE_DANGER_MINUTES:
            usage_status = "warning"
        else:
            usage_status = "danger"

        # ---- Overall status (worst wins) -------------------------------------
        status_priority = {"normal": 0, "warning": 1, "danger": 2}
        statuses = [br_status, usage_status]
        if self._drowsiness_alert:
            statuses.append("danger")
        overall = max(statuses, key=lambda s: status_priority[s])

        # ---- Human-readable message in Vietnamese ----------------------------
        message = self._build_message(
            br_status, usage_status, self._drowsiness_alert, blink_rate, usage_min
        )

        return {
            "blink_rate": round(blink_rate, 1),
            "blink_rate_status": br_status,
            "usage_time_min": round(usage_min, 1),
            "usage_status": usage_status,
            "drowsiness_alert": self._drowsiness_alert,
            "overall_status": overall,
            "message": message,
        }

    def get_total_blinks(self) -> int:
        """
        Return the cumulative number of blinks since session start.

        Returns
        -------
        int
        """
        return self._total_blinks

    def reset(self) -> None:
        """Reset the session – clears all counters and timestamps."""
        self._blink_timestamps.clear()
        self._session_start = time.time()
        self._total_blinks = 0
        self._current_alert = "normal"
        self._drowsiness_alert = False
        self._last_closed_duration = 0.0
        logger.info("HealthMonitor session has been reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_message(
        br_status: str,
        usage_status: str,
        drowsiness: bool,
        blink_rate: float,
        usage_min: float,
    ) -> str:
        """
        Build a Vietnamese health-advice message based on current statuses.

        Parameters
        ----------
        br_status : str
            Blink-rate status (``'normal'`` / ``'warning'`` / ``'danger'``).
        usage_status : str
            Usage-time status.
        drowsiness : bool
            Whether drowsiness is currently detected.
        blink_rate : float
            Current blink rate.
        usage_min : float
            Session duration in minutes.

        Returns
        -------
        str
            Advisory message in Vietnamese.
        """
        parts: list[str] = []

        # Drowsiness takes highest priority
        if drowsiness:
            parts.append(
                "⚠️ CẢNH BÁO: Bạn đang nhắm mắt quá lâu! "
                "Hãy mở mắt và nghỉ ngơi nếu buồn ngủ."
            )
            return "\n".join(parts)

        # Blink rate feedback
        if br_status == "danger":
            parts.append(
                f"🔴 Tần suất chớp mắt rất thấp ({blink_rate:.0f}/phút). "
                "Hãy chớp mắt thường xuyên hơn để tránh khô mắt!"
            )
        elif br_status == "warning":
            parts.append(
                f"🟡 Tần suất chớp mắt hơi thấp ({blink_rate:.0f}/phút). "
                "Nhớ chớp mắt đều đặn nhé."
            )
        else:
            parts.append("🟢 Tần suất chớp mắt bình thường. Tốt lắm!")

        # Usage time feedback
        if usage_status == "danger":
            parts.append(
                f"🔴 Bạn đã sử dụng máy tính {usage_min:.0f} phút. "
                "Hãy nghỉ ngơi ngay để bảo vệ mắt!"
            )
        elif usage_status == "warning":
            parts.append(
                f"🟡 Đã dùng máy {usage_min:.0f} phút. "
                "Hãy nghỉ mắt 20 giây, nhìn xa 20 feet (quy tắc 20-20-20)."
            )

        return "\n".join(parts) if parts else "🟢 Mắt bạn đang ở trạng thái tốt."
