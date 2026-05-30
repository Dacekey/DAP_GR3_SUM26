"""
collect_video.py – Record webcam videos for the Eye Blink Detection dataset.

Follows the team recording rules:
    - Resolution: 640x480, FPS: 15
    - Duration: 12 seconds per video (auto-stop)
    - Format: .mp4 (H.264 / mp4v codec)
    - Naming: MemberID_ScenarioID_EnvID_GlassID_DistID.mp4

Usage:
    python data/collect_video.py

Controls during preview:
    SPACE  → Start recording (auto-stops after 12s)
    Q      → Quit the application
"""

import os
import sys
import time
import cv2

# ---------------------------------------------------------------------------
# Import project-wide constants from config.py (located at project root)
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    CAMERA_INDEX,
    CAMERA_FPS,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    VIDEO_DIR,
)

# Recording duration in seconds (per team rules)
RECORD_DURATION_SEC = 12


# ===================================================================
# Naming convention helpers
# ===================================================================

MEMBER_IDS = [f"M{str(i).zfill(2)}" for i in range(1, 7)]  # M01..M06

SCENARIOS = {
    "SC1":   "Tap trung hoc tap / Lam viec binh thuong",
    "SC1SQ": "Tap trung + Nheo mat (Squinting)",
    "SC1LD": "Tap trung + Nhin xuong day man hinh (Looking Down)",
    "SC2":   "Giai tri luot mang xa hoi",
    "SC3":   "Xem phim / Thu gian (mat lim dim)",
    "SC4":   "Met moi / Buon ngu (Micro-sleep + dui mat)",
    "SC5":   "Phan khich / Cang thang choi game",
}

ENVIRONMENTS = {
    "E01": "Den vang (phong ngu, quan cafe)",
    "E02": "Den tuyp trang (van phong, phong hoc)",
    "E03": "Gan cua so (anh sang tu nhien)",
    "E04": "Phong toi (chi anh sang man hinh)",
}

GLASSES = {
    "G0": "Khong deo kinh",
    "G1": "Co deo kinh",
}

DISTANCES = {
    "D0": "Qua sat (< 35cm, mat > 70% khung hinh)",
    "D1": "Vua phai (50-60cm, mat 35-50% khung hinh)",
    "D2": "Xa / Ngua sau (> 80cm, mat < 25% khung hinh)",
}


def choose_option(prompt, options):
    """Display a numbered menu and return the selected key."""
    print(f"\n  {prompt}")
    keys = list(options.keys())
    for i, key in enumerate(keys, 1):
        desc = options[key] if isinstance(options[key], str) else key
        print(f"    [{i}] {key}: {desc}")

    while True:
        try:
            choice = int(input(f"  Nhap lua chon (1-{len(keys)}): "))
            if 1 <= choice <= len(keys):
                selected = keys[choice - 1]
                print(f"  -> Da chon: {selected}")
                return selected
        except (ValueError, EOFError):
            pass
        print(f"  Vui long nhap so tu 1 den {len(keys)}.")


def choose_member():
    """Select member ID."""
    print("\n  Chon thanh vien (MemberID):")
    for i, mid in enumerate(MEMBER_IDS, 1):
        print(f"    [{i}] {mid}")
    while True:
        try:
            choice = int(input(f"  Nhap lua chon (1-{len(MEMBER_IDS)}): "))
            if 1 <= choice <= len(MEMBER_IDS):
                selected = MEMBER_IDS[choice - 1]
                print(f"  -> Da chon: {selected}")
                return selected
        except (ValueError, EOFError):
            pass
        print(f"  Vui long nhap so tu 1 den {len(MEMBER_IDS)}.")


def build_filename(member, scenario, env, glass, dist):
    """Build filename per naming convention: MemberID_ScenarioID_EnvID_GlassID_DistID.mp4"""
    return f"{member}_{scenario}_{env}_{glass}_{dist}.mp4"


def print_instructions():
    """Print user-friendly instructions to the console."""
    print("=" * 60)
    print("  VIDEO COLLECTOR – Eye Blink Detection Dataset")
    print("=" * 60)
    print()
    print("  Quy chuan:")
    print(f"    Do phan giai : {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"    FPS          : {CAMERA_FPS}")
    print(f"    Thoi luong   : {RECORD_DURATION_SEC} giay (tu dong dung)")
    print(f"    Dinh dang    : .mp4")
    print(f"    Thu muc luu  : {VIDEO_DIR}")
    print()
    print("  Dieu khien:")
    print("    SPACE  →  Bat dau quay (tu dong dung sau 12 giay)")
    print("    Q      →  Thoat")
    print("=" * 60)


def main():
    """Main loop: configure → open camera → preview → record → repeat."""

    # Ensure the output directory exists
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print_instructions()

    # ------------------------------------------------------------------
    # Gather video metadata from user
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  THIET LAP THONG TIN VIDEO")
    print("=" * 60)

    member = choose_member()
    scenario = choose_option("Chon kich ban (ScenarioID):", SCENARIOS)
    env = choose_option("Chon moi truong anh sang (EnvID):", ENVIRONMENTS)
    glass = choose_option("Trang thai kinh (GlassID):", GLASSES)
    dist = choose_option("Khoang cach ngoi (DistID):", DISTANCES)

    filename = build_filename(member, scenario, env, glass, dist)
    filepath = os.path.join(VIDEO_DIR, filename)

    # Check if file already exists
    if os.path.exists(filepath):
        print(f"\n  [CANH BAO] File '{filename}' da ton tai!")
        overwrite = input("  Ghi de? (y/n): ").strip().lower()
        if overwrite != "y":
            print("  Da huy. Khoi dong lai de chon thong tin khac.")
            return

    print(f"\n  Ten file: {filename}")
    print(f"  Duong dan: {filepath}")
    print("\n  San sang! Nhan SPACE de bat dau quay.")

    # ------------------------------------------------------------------
    # Open webcam (use DirectShow backend on Windows to avoid MSMF errors)
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default backend
        print("[INFO] DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Khong the mo camera. Kiem tra ket noi hoac CAMERA_INDEX.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # Let camera run at native FPS; we control output FPS via VideoWriter
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    # Warmup: discard first few frames to let camera stabilize
    for _ in range(5):
        cap.read()

    # Use mp4v codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    is_recording = False
    writer = None
    record_start_time = 0.0
    frame_count = 0
    total_videos = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Khong doc duoc frame – dang thu lai...")
                continue

            display = frame.copy()
            elapsed = 0.0

            # -----------------------------------------------------------
            # Recording logic
            # -----------------------------------------------------------
            if is_recording:
                elapsed = time.time() - record_start_time
                remaining = max(0, RECORD_DURATION_SEC - elapsed)

                # Auto-stop after RECORD_DURATION_SEC
                if elapsed >= RECORD_DURATION_SEC:
                    is_recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    total_videos += 1
                    print(
                        f"\n  [XONG] Tu dong dung sau {RECORD_DURATION_SEC}s. "
                        f"Frames: {frame_count}  |  File: {filename}"
                    )

                    # Ask if user wants to record another
                    print("\n  Tiep tuc quay video khac? (Nhan SPACE de thiet lap lai, Q de thoat)")
                else:
                    writer.write(frame)
                    frame_count += 1

                    # --- Overlay: Recording indicator ---
                    # Red pulsating dot
                    cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)

                    # Countdown timer (large, centered)
                    countdown_text = f"{remaining:.1f}s"
                    text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                    text_x = (CAMERA_WIDTH - text_size[0]) // 2
                    cv2.putText(
                        display, countdown_text, (text_x, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
                    )

                    # Progress bar at bottom
                    progress = elapsed / RECORD_DURATION_SEC
                    bar_width = int(CAMERA_WIDTH * progress)
                    cv2.rectangle(display, (0, CAMERA_HEIGHT - 8),
                                  (bar_width, CAMERA_HEIGHT), (0, 0, 255), -1)

                    # Frame counter
                    cv2.putText(
                        display, f"REC | Frame: {frame_count}/{RECORD_DURATION_SEC * CAMERA_FPS}",
                        (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )

            else:
                # --- Standby overlay ---
                cv2.putText(
                    display, "STANDBY | Nhan SPACE de quay",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            # Show file info
            cv2.putText(
                display, f"File: {filename}",
                (10, CAMERA_HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )
            cv2.putText(
                display, f"Videos saved: {total_videos}",
                (10, CAMERA_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )

            cv2.imshow("Collect Video - Eye Blink Detection", display)

            # -----------------------------------------------------------
            # Handle key presses
            # -----------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not is_recording:
                    # If previous video was saved, ask for new metadata
                    if total_videos > 0:
                        cv2.destroyAllWindows()
                        print("\n" + "=" * 60)
                        print("  THIET LAP VIDEO MOI")
                        print("=" * 60)
                        scenario = choose_option("Chon kich ban (ScenarioID):", SCENARIOS)
                        env = choose_option("Chon moi truong anh sang (EnvID):", ENVIRONMENTS)
                        glass = choose_option("Trang thai kinh (GlassID):", GLASSES)
                        dist = choose_option("Khoang cach ngoi (DistID):", DISTANCES)
                        filename = build_filename(member, scenario, env, glass, dist)
                        filepath = os.path.join(VIDEO_DIR, filename)

                        if os.path.exists(filepath):
                            print(f"\n  [CANH BAO] File '{filename}' da ton tai!")
                            overwrite = input("  Ghi de? (y/n): ").strip().lower()
                            if overwrite != "y":
                                print("  Da huy video nay. Nhan SPACE de chon lai hoac Q de thoat.")
                                continue

                        print(f"\n  Ten file: {filename}")
                        print("  Nhan SPACE de bat dau quay.")

                    # Start recording
                    writer = cv2.VideoWriter(
                        filepath, fourcc, CAMERA_FPS,
                        (CAMERA_WIDTH, CAMERA_HEIGHT),
                    )
                    if not writer.isOpened():
                        print(f"[ERROR] Khong the tao file video: {filepath}")
                        continue

                    is_recording = True
                    frame_count = 0
                    record_start_time = time.time()
                    print(f"\n  [REC] Dang quay... Tu dong dung sau {RECORD_DURATION_SEC}s")

            elif key == ord("q") or key == ord("Q"):
                if is_recording and writer is not None:
                    writer.release()
                    total_videos += 1
                    print(f"\n  [DUNG] Da dung quay som. Frames: {frame_count}")
                print(f"\n  [THOAT] Tong so video da luu: {total_videos}")
                break

    except KeyboardInterrupt:
        print("\n  [INFO] Bi gian doan boi nguoi dung.")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
