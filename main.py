"""
Eye Blink Detection System - Main Entry Point

This application monitors eye blinks in real-time using a laptop webcam
and provides health alerts through a mini dashboard (always-on-top).

Pipeline:
    Camera → Face Detection (dlib HOG) → Facial Landmarks (68-point)
    → Eye Extraction → Feature Extraction (EAR+HOG+LBP) → SVM Classifier
    → Blink Counting (State Machine) → Health Monitor → Dashboard

Usage:
    python main.py                  # Run the full system
    python main.py --no-dashboard   # Run without dashboard (console output only)
    python main.py --debug          # Run with debug video window showing detections

Author: DAP391m Group Project
Date: May 2026
"""

import os
import sys
import time
import argparse
import threading
import logging

import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    LANDMARK_MODEL_PATH, CLASSIFIER_MODEL_PATH,
    DASHBOARD_UPDATE_MS, MODEL_DIR,
)
from src.blink_detector import BlinkDetector
from src.health_monitor import HealthMonitor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check that all required model files exist before starting."""
    errors = []

    if not os.path.exists(LANDMARK_MODEL_PATH):
        errors.append(
            f"Landmark model not found: {LANDMARK_MODEL_PATH}\n"
            f"  Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
            f"  Extract and place in: {MODEL_DIR}/"
        )

    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        errors.append(
            f"Trained classifier not found: {CLASSIFIER_MODEL_PATH}\n"
            f"  Run 'python src/train_model.py' first to train the model."
        )

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        errors.append(
            f"Scaler not found: {scaler_path}\n"
            f"  Run 'python src/train_model.py' first to train the model."
        )

    if errors:
        print("\n" + "=" * 60)
        print("  PREREQUISITE CHECK FAILED")
        print("=" * 60)
        for i, err in enumerate(errors, 1):
            print(f"\n  [{i}] {err}")
        print("\n" + "=" * 60)
        return False

    return True


def camera_loop(detector, monitor, shared_data, lock, stop_event, debug=False):
    """
    Main camera processing loop. Runs in a separate thread.

    Args:
        detector: BlinkDetector instance
        monitor: HealthMonitor instance
        shared_data: dict shared with dashboard for display updates
        lock: threading.Lock for thread-safe data access
        stop_event: threading.Event to signal loop termination
        debug: if True, show a debug window with face/eye detection overlay
    """
    # Use DirectShow backend on Windows to avoid MSMF errors
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open camera (index=%d). Is it in use?", CAMERA_INDEX)
        stop_event.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Warmup: discard first few frames to let camera stabilize
    for _ in range(5):
        cap.read()

    logger.info("Camera opened successfully. Processing started.")

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera.")
                time.sleep(0.01)
                continue

            # Process frame through blink detector
            result = detector.process_frame(frame)

            # Update health monitor
            monitor.update(
                blink_detected=result["blink_detected"],
                closed_duration=result["closed_duration"],
            )

            # Get health status
            health = monitor.get_health_status()

            # Calculate FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start_time = time.time()

            # Update shared data for dashboard (thread-safe)
            with lock:
                shared_data["blink_count"] = result["blink_count"]
                shared_data["ear"] = result["ear"]
                shared_data["eye_state"] = result["eye_state"]
                shared_data["face_detected"] = result["face_detected"]
                shared_data["blink_rate"] = health["blink_rate"]
                shared_data["usage_time_min"] = health["usage_time_min"]
                shared_data["blink_rate_status"] = health["blink_rate_status"]
                shared_data["usage_status"] = health["usage_status"]
                shared_data["drowsiness_alert"] = health["drowsiness_alert"]
                shared_data["overall_status"] = health["overall_status"]
                shared_data["message"] = health["message"]
                shared_data["fps"] = current_fps
                shared_data["total_blinks"] = monitor.get_total_blinks()

            # Debug window (optional)
            if debug:
                debug_frame = frame.copy()

                # Draw face/eye landmarks
                if result["face_detected"]:
                    if result["left_eye_points"] is not None:
                        pts = result["left_eye_points"].astype(np.int32)
                        cv2.polylines(debug_frame, [pts], True, (0, 255, 0), 1)
                    if result["right_eye_points"] is not None:
                        pts = result["right_eye_points"].astype(np.int32)
                        cv2.polylines(debug_frame, [pts], True, (0, 255, 0), 1)

                # Overlay text info
                info_lines = [
                    f"EAR: {result['ear']:.3f}" if result['ear'] else "EAR: N/A",
                    f"State: {result['eye_state']}",
                    f"Blinks: {result['blink_count']}",
                    f"Rate: {health['blink_rate']:.1f}/min",
                    f"Status: {health['overall_status'].upper()}",
                    f"FPS: {current_fps:.1f}",
                ]
                y_offset = 30
                for line in info_lines:
                    color = (0, 255, 0)
                    if "danger" in line.lower():
                        color = (0, 0, 255)
                    elif "warning" in line.lower():
                        color = (0, 255, 255)
                    cv2.putText(
                        debug_frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )
                    y_offset += 25

                cv2.imshow("Eye Blink Detection - Debug", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

    except Exception as e:
        logger.error("Error in camera loop: %s", e, exc_info=True)
    finally:
        cap.release()
        if debug:
            cv2.destroyAllWindows()
        logger.info("Camera released.")


def run_console_mode(detector, monitor, stop_event):
    """Run without dashboard, printing stats to console."""
    # Use DirectShow backend on Windows to avoid MSMF errors
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # Warmup
    for _ in range(5):
        cap.read()

    logger.info("Console mode started. Press Ctrl+C to stop.")
    print("\n" + "=" * 50)
    print("  EYE BLINK DETECTION - CONSOLE MODE")
    print("=" * 50)

    last_print = time.time()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            result = detector.process_frame(frame)
            monitor.update(result["blink_detected"], result["closed_duration"])

            # Print stats every 2 seconds
            if time.time() - last_print >= 2.0:
                health = monitor.get_health_status()
                ear_str = f"{result['ear']:.3f}" if result["ear"] else "N/A"
                print(
                    f"  Blinks: {result['blink_count']:4d} | "
                    f"Rate: {health['blink_rate']:5.1f}/min | "
                    f"EAR: {ear_str} | "
                    f"Status: {health['overall_status']:8s} | "
                    f"Time: {health['usage_time_min']:.1f}min"
                )
                if health["drowsiness_alert"]:
                    print("  ⚠️  DROWSINESS ALERT! Eyes closed too long!")
                last_print = time.time()

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        cap.release()


def run_with_dashboard(detector, monitor, debug=False):
    """Run with tkinter dashboard (dashboard in main thread, camera in background)."""
    # Import dashboard here to avoid tkinter import issues in threads
    from src.dashboard import HealthDashboard
    import tkinter as tk

    # Shared data between camera thread and dashboard
    shared_data = {
        "blink_count": 0,
        "ear": 0.0,
        "eye_state": "open",
        "face_detected": False,
        "blink_rate": 0.0,
        "usage_time_min": 0.0,
        "blink_rate_status": "normal",
        "usage_status": "normal",
        "drowsiness_alert": False,
        "overall_status": "normal",
        "message": "Hệ thống đang khởi động...",
        "fps": 0.0,
        "total_blinks": 0,
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    # Start camera processing in background thread
    camera_thread = threading.Thread(
        target=camera_loop,
        args=(detector, monitor, shared_data, lock, stop_event, debug),
        daemon=True,
    )
    camera_thread.start()
    logger.info("Camera thread started.")

    # Create and run dashboard in main thread (tkinter requirement)
    root = tk.Tk()
    dashboard = HealthDashboard(root)

    def update_dashboard():
        """Periodically update dashboard with latest data from camera thread."""
        if stop_event.is_set():
            root.destroy()
            return

        with lock:
            data_copy = dict(shared_data)

        dashboard.update(data_copy)
        root.after(DASHBOARD_UPDATE_MS, update_dashboard)

    def on_closing():
        """Handle window close."""
        logger.info("Dashboard closed. Shutting down...")
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the periodic update
    root.after(1000, update_dashboard)  # First update after 1 second

    logger.info("Dashboard started. Close the dashboard window to stop.")
    root.mainloop()

    # Wait for camera thread to finish
    stop_event.set()
    camera_thread.join(timeout=3.0)
    logger.info("System shut down cleanly.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Eye Blink Detection System - Health Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Run with mini dashboard
  python main.py --no-dashboard   Console output only
  python main.py --debug          Show debug video window
  python main.py --debug --no-dashboard   Debug + console only
        """,
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run without GUI dashboard (console output only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug window with face/eye detection overlay",
    )
    args = parser.parse_args()

    print(r"""
    ╔══════════════════════════════════════════════════╗
    ║       👁️  EYE BLINK DETECTION SYSTEM  👁️        ║
    ║          Health Monitor v1.0                     ║
    ║          DAP391m - Computer Vision               ║
    ╚══════════════════════════════════════════════════╝
    """)

    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)

    logger.info("All prerequisites OK. Initializing...")

    # Initialize components
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    detector = BlinkDetector(
        model_path=CLASSIFIER_MODEL_PATH,
        scaler_path=scaler_path,
        landmark_model_path=LANDMARK_MODEL_PATH,
    )
    monitor = HealthMonitor()

    logger.info("BlinkDetector and HealthMonitor initialized.")

    # Run
    if args.no_dashboard:
        stop_event = threading.Event()
        run_console_mode(detector, monitor, stop_event)
    else:
        run_with_dashboard(detector, monitor, debug=args.debug)


if __name__ == "__main__":
    main()
