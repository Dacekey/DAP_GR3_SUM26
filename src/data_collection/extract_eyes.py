"""
extract_eyes.py – Extract eye patches from recorded videos using dlib.

Usage:
    python data/extract_eyes.py               # Normal mode (no preview)
    python data/extract_eyes.py --preview      # Show real-time preview of detections

Pipeline (per frame):
    1. Detect face with dlib frontal face detector
    2. Locate 68 facial landmarks
    3. Compute eye alignment angle from corner landmarks (p1, p4)
    4. Rotate & crop eye region so the eye is always horizontal
    5. Resize each crop to 24x24 grayscale
    6. Compute EAR and save alongside the patch

Output:
    dataset/raw_eyes/eye_000001.png, eye_000002.png, …
    dataset/raw_eyes/ear_values.csv   (index, filename, ear, eye_side)
"""

import os
import sys
import csv
import math
import argparse

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import project-wide constants from config.py
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    VIDEO_DIR,
    DATASET_DIR,
    LANDMARK_MODEL_PATH,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EYE_PATCH_SIZE,
)

# Try importing dlib – give a helpful message if missing
try:
    import dlib
except ImportError:
    print("[ERROR] dlib is not installed. Install it with:")
    print("        pip install dlib")
    sys.exit(1)

# Output directory for raw eye patches
RAW_EYES_DIR = os.path.join(DATASET_DIR, "raw_eyes")

# Padding factor around the eye bounding box (fraction of box size)
EYE_PADDING = 0.35


# ===================================================================
# Helper functions
# ===================================================================

def compute_ear(eye_points):
    """
    Compute the Eye Aspect Ratio (EAR) given 6 (x, y) landmark points.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_points: numpy array of shape (6, 2)

    Returns:
        float: EAR value
    """
    # Vertical distances
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    h = np.linalg.norm(eye_points[0] - eye_points[3])

    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def align_and_crop_eye(frame_gray, eye_points, patch_size, padding=EYE_PADDING):
    """
    Align the eye horizontally based on corner landmarks, then crop and resize.

    The eye is rotated so that the line connecting the left corner (p1, index 0)
    and the right corner (p4, index 3) is perfectly horizontal. This ensures
    consistent orientation for both open and closed eyes.

    Args:
        frame_gray: Grayscale frame (H, W)
        eye_points: numpy array of shape (6, 2) – 6 eye landmark points
        patch_size: tuple (width, height) for resized output
        padding: fractional padding around bounding box

    Returns:
        Resized aligned eye patch (grayscale, uint8) or None if crop is invalid.
    """
    h, w = frame_gray.shape[:2]

    # --- Step 1: Compute rotation angle from eye corners ---
    # p1 = left corner (index 0), p4 = right corner (index 3)
    p1 = eye_points[0].astype(np.float64)
    p4 = eye_points[3].astype(np.float64)

    # Angle in degrees between the eye corners and horizontal
    dx = p4[0] - p1[0]
    dy = p4[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))

    # --- Step 2: Compute center of the eye ---
    eye_center = eye_points.mean(axis=0).astype(np.float64)
    cx, cy = eye_center[0], eye_center[1]

    # --- Step 3: Rotate the entire frame around the eye center ---
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame_gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # --- Step 4: Transform eye points to rotated coordinates ---
    ones = np.ones((6, 1))
    pts_hom = np.hstack([eye_points.astype(np.float64), ones])  # (6, 3)
    rotated_pts = (M @ pts_hom.T).T  # (6, 2)

    # --- Step 5: Crop with padding from rotated frame ---
    x_min, y_min = rotated_pts.min(axis=0)
    x_max, y_max = rotated_pts.max(axis=0)

    box_w = x_max - x_min
    box_h = y_max - y_min

    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, patch_size, interpolation=cv2.INTER_AREA)
    return resized


def print_progress_bar(current, total, bar_len=40, prefix="Progress"):
    """Print a simple text-based progress bar (no tqdm needed)."""
    fraction = current / max(total, 1)
    filled = int(bar_len * fraction)
    bar = "█" * filled + "░" * (bar_len - filled)
    percent = fraction * 100
    print(f"\r  {prefix} |{bar}| {percent:5.1f}%  ({current}/{total})", end="", flush=True)


# ===================================================================
# Preview visualization
# ===================================================================

def build_preview(frame, landmarks, eye_pts_left, eye_pts_right,
                  patch_left, patch_right, ear_left, ear_right, frame_idx):
    """
    Build a debug/preview frame showing:
    - Original frame with landmarks drawn
    - Zoomed left/right eye patches (enlarged from 24x24 to 120x120)
    - EAR values

    Args:
        frame: original BGR frame
        landmarks: dlib full_object_detection
        eye_pts_left, eye_pts_right: numpy arrays of eye landmarks
        patch_left, patch_right: 24x24 grayscale eye patches (or None)
        ear_left, ear_right: EAR values
        frame_idx: current frame number

    Returns:
        BGR image for display
    """
    display = frame.copy()
    h_frame, w_frame = display.shape[:2]

    # Draw all 68 landmarks as small dots
    for i in range(68):
        p = landmarks.part(i)
        cv2.circle(display, (p.x, p.y), 1, (128, 128, 128), -1)

    # Draw eye landmarks with polylines (green = open, red = closed threshold)
    for eye_pts, ear, label in [(eye_pts_left, ear_left, "L"),
                                 (eye_pts_right, ear_right, "R")]:
        # Color based on EAR
        color = (0, 255, 0) if ear >= 0.21 else (0, 0, 255)

        # Draw eye contour
        cv2.polylines(display, [eye_pts], isClosed=True, color=color, thickness=2)

        # Draw each landmark point
        for pt in eye_pts:
            cv2.circle(display, tuple(pt), 3, (255, 255, 0), -1)

        # Draw corner-to-corner line (alignment reference)
        p1 = tuple(eye_pts[0])
        p4 = tuple(eye_pts[3])
        cv2.line(display, p1, p4, (255, 0, 255), 1)

        # Label with EAR
        cx = int(eye_pts[:, 0].mean())
        cy = int(eye_pts[:, 1].min()) - 10
        cv2.putText(display, f"{label} EAR:{ear:.3f}",
                    (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Create eye patch preview panel (right side)
    patch_display_size = 120
    panel_w = patch_display_size + 20
    panel_h = h_frame

    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark background

    # Title
    cv2.putText(panel, "Eye Patches", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    y_offset = 40
    for patch, label, ear in [(patch_left, "Left", ear_left),
                               (patch_right, "Right", ear_right)]:
        if patch is not None:
            # Enlarge 24x24 → 120x120 for visibility
            enlarged = cv2.resize(patch, (patch_display_size, patch_display_size),
                                  interpolation=cv2.INTER_NEAREST)
            enlarged_bgr = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)

            # Border color based on EAR
            border_color = (0, 255, 0) if ear >= 0.21 else (0, 0, 255)
            cv2.rectangle(enlarged_bgr, (0, 0),
                          (patch_display_size - 1, patch_display_size - 1),
                          border_color, 2)

            panel[y_offset:y_offset + patch_display_size,
                  10:10 + patch_display_size] = enlarged_bgr
        else:
            cv2.putText(panel, "N/A", (40, y_offset + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # Label
        state = "OPEN" if ear >= 0.21 else "CLOSED"
        color = (0, 255, 0) if ear >= 0.21 else (0, 0, 255)
        cv2.putText(panel, f"{label}: {state}",
                    (10, y_offset + patch_display_size + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(panel, f"EAR: {ear:.3f}",
                    (10, y_offset + patch_display_size + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        y_offset += patch_display_size + 55

    # Frame info
    cv2.putText(panel, f"Frame: {frame_idx}",
                (10, panel_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Combine frame + panel side by side
    combined = np.hstack([display, panel])
    return combined


# ===================================================================
# Main extraction routine
# ===================================================================

def main():
    """Load videos, extract eye patches, save results."""

    parser = argparse.ArgumentParser(
        description="Extract aligned eye patches from recorded videos."
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show real-time preview with landmarks and eye patches. "
             "Press Q to skip video, ESC to quit all."
    )
    args = parser.parse_args()

    os.makedirs(RAW_EYES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Discover input videos
    # ------------------------------------------------------------------
    # NOTE: We use os.listdir instead of glob.glob because glob treats
    # square brackets [] as character-class patterns, which breaks when
    # the path contains e.g. [SUMMER_26].
    # Support both .mp4 (new format) and .avi (legacy format).
    video_extensions = (".mp4", ".avi")
    video_files = sorted([
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(video_extensions)
    ])

    if not video_files:
        print(f"[ERROR] No video files (.mp4/.avi) found in {VIDEO_DIR}")
        print("        Run collect_video.py first to record some videos.")
        sys.exit(1)

    print("=" * 60)
    print("  EYE PATCH EXTRACTOR – Eye Blink Detection Dataset")
    print("=" * 60)
    print(f"  Found {len(video_files)} video(s) in {VIDEO_DIR}")
    print(f"  Output directory : {RAW_EYES_DIR}")
    print(f"  Patch size       : {EYE_PATCH_SIZE}")
    print(f"  Alignment        : ENABLED (rotate to horizontal)")
    if args.preview:
        print(f"  Preview mode     : ON (press Q=skip video, ESC=quit)")
    print(f"  Landmark model   : {LANDMARK_MODEL_PATH}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load dlib models
    # ------------------------------------------------------------------
    if not os.path.isfile(LANDMARK_MODEL_PATH):
        print(f"\n[ERROR] Landmark model not found at:\n  {LANDMARK_MODEL_PATH}")
        print("  Download it from:")
        print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)

    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

    # ------------------------------------------------------------------
    # Counters / bookkeeping
    # ------------------------------------------------------------------
    eye_index = 0           # global sequential index for saved patches
    total_frames = 0
    total_extracted = 0
    skipped_frames = 0
    quit_all = False

    # CSV log: (index, filename, ear, eye_side)
    ear_records = []

    # First pass: count total frames for progress bar
    print("\n[INFO] Counting total frames …")
    frame_counts = []
    for vf in video_files:
        cap = cv2.VideoCapture(vf)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(n)
        cap.release()
    grand_total = sum(frame_counts)
    print(f"  Total frames across all videos: {grand_total}\n")

    processed_so_far = 0

    # ------------------------------------------------------------------
    # Process each video
    # ------------------------------------------------------------------
    for vid_idx, video_path in enumerate(video_files):
        if quit_all:
            break

        video_name = os.path.basename(video_path)
        print(f"\n[VIDEO {vid_idx + 1}/{len(video_files)}] {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [WARNING] Cannot open {video_path} – skipping.")
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame_idx += 1
            processed_so_far += 1

            if not args.preview:
                print_progress_bar(processed_so_far, grand_total)

            # Convert to grayscale for detection and cropping
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_detector(gray, 0)
            if len(faces) == 0:
                skipped_frames += 1
                if args.preview:
                    # Show frame without detections
                    cv2.putText(frame, "No face detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Extract Eyes - Preview", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        quit_all = True
                        break
                    elif key == ord("q"):
                        break
                continue

            # Use the first (largest / most confident) face
            face = faces[0]
            landmarks = predictor(gray, face)

            # Helper to get eye points as numpy array
            def get_eye_points(indices):
                pts = []
                for i in indices:
                    p = landmarks.part(i)
                    pts.append((p.x, p.y))
                return np.array(pts, dtype=np.int32)

            patches = {}  # Store patches for preview
            ears = {}     # Store EARs for preview

            for side, indices in [("left", LEFT_EYE_INDICES),
                                  ("right", RIGHT_EYE_INDICES)]:
                eye_pts = get_eye_points(indices)
                ear = compute_ear(eye_pts.astype(np.float64))

                # Use aligned crop instead of simple bounding box
                patch = align_and_crop_eye(gray, eye_pts, EYE_PATCH_SIZE)

                patches[side] = patch
                ears[side] = ear

                if patch is None:
                    continue

                eye_index += 1
                total_extracted += 1
                filename = f"eye_{eye_index:06d}.png"
                save_path = os.path.join(RAW_EYES_DIR, filename)
                cv2.imwrite(save_path, patch)

                ear_records.append({
                    "index": eye_index,
                    "filename": filename,
                    "ear": round(ear, 4),
                    "eye_side": side,
                    "source_video": video_name,
                    "frame": frame_idx,
                })

            # --- Preview mode ---
            if args.preview:
                preview = build_preview(
                    frame, landmarks,
                    get_eye_points(LEFT_EYE_INDICES),
                    get_eye_points(RIGHT_EYE_INDICES),
                    patches.get("left"), patches.get("right"),
                    ears.get("left", 0.0), ears.get("right", 0.0),
                    frame_idx,
                )
                cv2.imshow("Extract Eyes - Preview", preview)
                key = cv2.waitKey(30) & 0xFF  # ~33ms per frame for 30fps playback
                if key == 27:  # ESC = quit all
                    quit_all = True
                    break
                elif key == ord("q"):  # Q = skip this video
                    break

        cap.release()

    if args.preview:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Save EAR CSV
    # ------------------------------------------------------------------
    ear_csv_path = os.path.join(RAW_EYES_DIR, "ear_values.csv")
    with open(ear_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "filename", "ear", "eye_side",
                           "source_video", "frame"]
        )
        writer.writeheader()
        writer.writerows(ear_records)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total frames processed : {total_frames}")
    print(f"  Skipped frames (no face): {skipped_frames}")
    print(f"  Total eye patches saved : {total_extracted}")
    print(f"  EAR log saved to        : {ear_csv_path}")
    print(f"  Output directory        : {RAW_EYES_DIR}")
    print(f"  Alignment               : ENABLED")
    print("=" * 60)


if __name__ == "__main__":
    main()
