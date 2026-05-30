"""
Health Dashboard – Tkinter Mini Dashboard.

A compact, always-on-top window that visualises real-time eye-health
metrics: blink count, blink rate, EAR, eye state, session time, and
health alerts.

The dashboard runs on the **main thread** (tkinter requirement); camera
processing is expected to run in a separate thread.
"""

import os
import sys
import tkinter as tk
from tkinter import font as tkfont

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DASHBOARD_WIDTH,
    DASHBOARD_HEIGHT,
    DASHBOARD_BG_COLOR,
    DASHBOARD_FG_COLOR,
    DASHBOARD_ACCENT,
    COLOR_NORMAL,
    COLOR_WARNING,
    COLOR_DANGER,
    USAGE_WARNING_MINUTES,
    USAGE_DANGER_MINUTES,
)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_STATUS_COLORS = {
    "normal": COLOR_NORMAL,
    "warning": COLOR_WARNING,
    "danger": COLOR_DANGER,
}

_STATUS_ICONS = {
    "normal": "🟢",
    "warning": "🟡",
    "danger": "🔴",
}


class HealthDashboard:
    """
    Tkinter mini dashboard for real-time eye-health monitoring.

    Parameters
    ----------
    root : tk.Tk
        The root Tkinter window instance.

    Notes
    -----
    Call :meth:`update` periodically (e.g. via ``root.after``) with the
    latest detection data to keep the display current.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._setup_window()
        self._create_fonts()
        self._build_ui()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """Configure the root window: size, position, always-on-top."""
        self.root.title("👁️ Eye Health Monitor")
        self.root.configure(bg=DASHBOARD_BG_COLOR)
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        # Position at the bottom-right corner of the screen
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = screen_w - DASHBOARD_WIDTH - 20
        y = screen_h - DASHBOARD_HEIGHT - 60
        self.root.geometry(f"{DASHBOARD_WIDTH}x{DASHBOARD_HEIGHT}+{x}+{y}")

    def _create_fonts(self) -> None:
        """Pre-create frequently used font objects."""
        self._font_title = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self._font_section = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self._font_value = tkfont.Font(family="Consolas", size=13, weight="bold")
        self._font_label = tkfont.Font(family="Segoe UI", size=9)
        self._font_message = tkfont.Font(family="Segoe UI", size=9, slant="italic")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble all dashboard sections."""
        # Main container with padding
        self._container = tk.Frame(self.root, bg=DASHBOARD_BG_COLOR, padx=12, pady=8)
        self._container.pack(fill="both", expand=True)

        self._build_header()
        self._add_separator()
        self._build_blink_stats()
        self._add_separator()
        self._build_eye_status()
        self._add_separator()
        self._build_health_status()
        self._add_separator()
        self._build_usage_time()
        self._add_separator()
        self._build_alert_message()

    # ---- Header ----------------------------------------------------------

    def _build_header(self) -> None:
        """App title and icon."""
        frame = tk.Frame(self._container, bg=DASHBOARD_BG_COLOR)
        frame.pack(fill="x", pady=(0, 4))

        tk.Label(
            frame,
            text="👁️  Eye Health Monitor",
            font=self._font_title,
            bg=DASHBOARD_BG_COLOR,
            fg=DASHBOARD_FG_COLOR,
        ).pack(side="left")

    # ---- Blink Stats -----------------------------------------------------

    def _build_blink_stats(self) -> None:
        """Total blinks + blink rate with coloured indicator."""
        frame = tk.Frame(self._container, bg=DASHBOARD_BG_COLOR)
        frame.pack(fill="x", pady=2)

        tk.Label(
            frame,
            text="Chớp mắt",
            font=self._font_section,
            bg=DASHBOARD_BG_COLOR,
            fg=DASHBOARD_ACCENT,
        ).pack(anchor="w")

        row = tk.Frame(frame, bg=DASHBOARD_BG_COLOR)
        row.pack(fill="x", pady=2)

        # Total blinks
        left = tk.Frame(row, bg=DASHBOARD_BG_COLOR)
        left.pack(side="left", expand=True, fill="x")
        tk.Label(left, text="Tổng", font=self._font_label,
                 bg=DASHBOARD_BG_COLOR, fg="#888").pack(anchor="w")
        self._lbl_blink_count = tk.Label(
            left, text="0", font=self._font_value,
            bg=DASHBOARD_BG_COLOR, fg=DASHBOARD_FG_COLOR,
        )
        self._lbl_blink_count.pack(anchor="w")

        # Blink rate
        right = tk.Frame(row, bg=DASHBOARD_BG_COLOR)
        right.pack(side="right", expand=True, fill="x")
        tk.Label(right, text="Tần suất (/phút)", font=self._font_label,
                 bg=DASHBOARD_BG_COLOR, fg="#888").pack(anchor="w")
        rate_row = tk.Frame(right, bg=DASHBOARD_BG_COLOR)
        rate_row.pack(anchor="w")
        self._lbl_blink_rate_icon = tk.Label(
            rate_row, text="🟢", font=self._font_label, bg=DASHBOARD_BG_COLOR,
        )
        self._lbl_blink_rate_icon.pack(side="left")
        self._lbl_blink_rate = tk.Label(
            rate_row, text="0.0", font=self._font_value,
            bg=DASHBOARD_BG_COLOR, fg=DASHBOARD_FG_COLOR,
        )
        self._lbl_blink_rate.pack(side="left", padx=(4, 0))

    # ---- Eye Status ------------------------------------------------------

    def _build_eye_status(self) -> None:
        """Current open/closed state + EAR value."""
        frame = tk.Frame(self._container, bg=DASHBOARD_BG_COLOR)
        frame.pack(fill="x", pady=2)

        tk.Label(
            frame, text="Trạng thái mắt", font=self._font_section,
            bg=DASHBOARD_BG_COLOR, fg=DASHBOARD_ACCENT,
        ).pack(anchor="w")

        row = tk.Frame(frame, bg=DASHBOARD_BG_COLOR)
        row.pack(fill="x", pady=2)

        # Eye state label
        left = tk.Frame(row, bg=DASHBOARD_BG_COLOR)
        left.pack(side="left", expand=True, fill="x")
        tk.Label(left, text="Mắt", font=self._font_label,
                 bg=DASHBOARD_BG_COLOR, fg="#888").pack(anchor="w")
        self._lbl_eye_state = tk.Label(
            left, text="MỞ", font=self._font_value,
            bg=DASHBOARD_BG_COLOR, fg=COLOR_NORMAL,
        )
        self._lbl_eye_state.pack(anchor="w")

        # EAR value
        right = tk.Frame(row, bg=DASHBOARD_BG_COLOR)
        right.pack(side="right", expand=True, fill="x")
        tk.Label(right, text="EAR", font=self._font_label,
                 bg=DASHBOARD_BG_COLOR, fg="#888").pack(anchor="w")
        self._lbl_ear = tk.Label(
            right, text="--", font=self._font_value,
            bg=DASHBOARD_BG_COLOR, fg=DASHBOARD_FG_COLOR,
        )
        self._lbl_ear.pack(anchor="w")

    # ---- Health Status ---------------------------------------------------

    def _build_health_status(self) -> None:
        """Overall health indicator with colour-coded background."""
        frame = tk.Frame(self._container, bg=DASHBOARD_BG_COLOR)
        frame.pack(fill="x", pady=2)

        tk.Label(
            frame, text="Sức khoẻ tổng thể", font=self._font_section,
            bg=DASHBOARD_BG_COLOR, fg=DASHBOARD_ACCENT,
        ).pack(anchor="w")

        self._frm_health = tk.Frame(
            frame, bg=COLOR_NORMAL, padx=10, pady=6,
            highlightbackground="#333", highlightthickness=1,
        )
        self._frm_health.pack(fill="x", pady=4)

        self._lbl_health = tk.Label(
            self._frm_health, text="🟢 Bình thường",
            font=self._font_section, bg=COLOR_NORMAL, fg="#000",
        )
        self._lbl_health.pack()

    # ---- Usage Time ------------------------------------------------------

    def _build_usage_time(self) -> None:
        """Elapsed session time with a progress-bar style indicator."""
        frame = tk.Frame(self._container, bg=DASHBOARD_BG_COLOR)
        frame.pack(fill="x", pady=2)

        tk.Label(
            frame, text="Thời gian sử dụng", font=self._font_section,
            bg=DASHBOARD_BG_COLOR, fg=DASHBOARD_ACCENT,
        ).pack(anchor="w")

        self._lbl_usage = tk.Label(
            frame, text="0 phút", font=self._font_value,
            bg=DASHBOARD_BG_COLOR, fg=DASHBOARD_FG_COLOR,
        )
        self._lbl_usage.pack(anchor="w", pady=(2, 0))

        # Canvas-based progress bar
        self._cvs_usage = tk.Canvas(
            frame, height=10, bg="#333",
            highlightthickness=0,
        )
        self._cvs_usage.pack(fill="x", pady=(4, 0))
        self._usage_bar = self._cvs_usage.create_rectangle(
            0, 0, 0, 10, fill=COLOR_NORMAL, outline="",
        )

    # ---- Alert Message ---------------------------------------------------

    def _build_alert_message(self) -> None:
        """Bottom text area for current health advice (Vietnamese)."""
        frame = tk.Frame(self._container, bg=DASHBOARD_BG_COLOR)
        frame.pack(fill="both", expand=True, pady=(4, 0))

        self._lbl_alert = tk.Label(
            frame,
            text="🟢 Mắt bạn đang ở trạng thái tốt.",
            font=self._font_message,
            bg=DASHBOARD_BG_COLOR,
            fg=DASHBOARD_FG_COLOR,
            wraplength=DASHBOARD_WIDTH - 30,
            justify="left",
            anchor="nw",
        )
        self._lbl_alert.pack(fill="both", expand=True)

    # ---- Separator helper ------------------------------------------------

    def _add_separator(self) -> None:
        """Draw a subtle horizontal rule."""
        tk.Frame(
            self._container, bg="#333", height=1,
        ).pack(fill="x", pady=4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, data: dict) -> None:
        """
        Refresh all dashboard widgets with the latest data.

        Parameters
        ----------
        data : dict
            Expected keys (flat dict from main.py shared_data):
            - ``blink_count`` (int)
            - ``blink_rate`` (float)
            - ``ear`` (float | None)
            - ``eye_state`` (str: ``'open'`` / ``'closed'``)
            - ``blink_rate_status`` (str)
            - ``usage_status`` (str)
            - ``overall_status`` (str)
            - ``drowsiness_alert`` (bool)
            - ``message`` (str)
            - ``usage_time_min`` (float – minutes)
            - ``face_detected`` (bool)
            - ``fps`` (float)
        """
        # ---- Blink stats -----------------------------------------------------
        blink_count = data.get("blink_count", 0)
        blink_rate = data.get("blink_rate", 0.0)
        self._lbl_blink_count.config(text=str(blink_count))
        self._lbl_blink_rate.config(text=f"{blink_rate:.1f}")

        # Blink rate indicator (keys are at top level, not nested)
        br_status = data.get("blink_rate_status", "normal")
        self._lbl_blink_rate_icon.config(text=_STATUS_ICONS.get(br_status, "🟢"))
        self._lbl_blink_rate.config(fg=_STATUS_COLORS.get(br_status, COLOR_NORMAL))

        # ---- Eye status ------------------------------------------------------
        eye_state = data.get("eye_state", "open")
        if eye_state == "open":
            self._lbl_eye_state.config(text="MỞ", fg=COLOR_NORMAL)
        else:
            self._lbl_eye_state.config(text="NHẮM", fg=COLOR_DANGER)

        ear = data.get("ear")
        self._lbl_ear.config(text=f"{ear:.3f}" if ear is not None else "--")

        # ---- Health status ---------------------------------------------------
        overall = data.get("overall_status", "normal")
        color = _STATUS_COLORS.get(overall, COLOR_NORMAL)
        icon = _STATUS_ICONS.get(overall, "🟢")
        status_text_map = {
            "normal": "Bình thường",
            "warning": "Cảnh báo",
            "danger": "Nguy hiểm",
        }
        self._frm_health.config(bg=color)
        self._lbl_health.config(
            text=f"{icon} {status_text_map.get(overall, 'N/A')}",
            bg=color,
            fg="#000" if overall != "danger" else "#fff",
        )

        # ---- Usage time ------------------------------------------------------
        usage_min = data.get("usage_time_min", 0.0)
        self._lbl_usage.config(text=f"{usage_min:.1f} phút")

        # Progress bar (map usage to bar width, cap at USAGE_DANGER_MINUTES)
        self._cvs_usage.update_idletasks()
        bar_max_w = self._cvs_usage.winfo_width()
        ratio = min(usage_min / max(USAGE_DANGER_MINUTES, 1), 1.0)
        bar_w = int(bar_max_w * ratio)
        usage_status = data.get("usage_status", "normal")
        bar_color = _STATUS_COLORS.get(usage_status, COLOR_NORMAL)
        self._cvs_usage.coords(self._usage_bar, 0, 0, bar_w, 10)
        self._cvs_usage.itemconfig(self._usage_bar, fill=bar_color)

        # ---- Alert message ---------------------------------------------------
        message = data.get("message", "")
        if message:
            self._lbl_alert.config(text=message)

        # Highlight alert text colour by overall status
        alert_fg = _STATUS_COLORS.get(overall, DASHBOARD_FG_COLOR)
        self._lbl_alert.config(fg=alert_fg)
