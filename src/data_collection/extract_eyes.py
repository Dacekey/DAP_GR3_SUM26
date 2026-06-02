"""
extract_eyes.py – Extract eye patches from recorded videos using MediaPipe Face Mesh.

Usage:
    python src/data_collection/extract_eyes.py --input raw_videos --output extracted_eyes --detector mediapipe
    python src/data_collection/extract_eyes.py --preview

Pipeline (per frame):
    1. Detect face and landmarks with MediaPipe Face Mesh
    2. Compute EAR using:
       LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
       RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
    3. Crop eyes using:
       LEFT_EYE_CROP_IDX = [33, 133, 160, 158, 159, 144, 153, 145, 246, 161, 163, 7, 173, 157, 154, 155]
       RIGHT_EYE_CROP_IDX = [362, 263, 385, 387, 386, 373, 380, 374, 466, 388, 390, 249, 398, 384, 381, 382]
    4. Compute eye alignment angle from corner landmarks and rotate frame
    5. Crop eye bounding box with padding, clamp within image dimensions
    6. Convert each crop to grayscale and resize to 24x24
    7. Save eye images and write metadata CSV.
"""

import os
import sys
import csv
import math
import argparse
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Import project-wide constants from config.py 
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    VIDEO_DIR,
    DATASET_DIR,
    EYE_PATCH_SIZE,
)

# Output directory default
DEFAULT_RAW_EYES_DIR = os.path.join(DATASET_DIR, "raw_eyes")

# Try importing MediaPipe – give a helpful message if missing
try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] mediapipe is not installed. Install it with:")
    print("        pip install mediapipe")
    sys.exit(1)

# Padding factor around the eye bounding box (fraction of box size)
EYE_PADDING = 0.35

# MediaPipe landmark indices for EAR
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

# MediaPipe landmark indices for eye crop bounding box
LEFT_EYE_CROP_IDX = [33, 133, 160, 158, 159, 144, 153, 145, 246, 161, 163, 7, 173, 157, 154, 155]
RIGHT_EYE_CROP_IDX = [362, 263, 385, 387, 386, 373, 380, 374, 466, 388, 390, 249, 398, 384, 381, 382]


# ===================================================================
# Helper functions
# ===================================================================

def compute_ear(eye_points):
    """
    Compute the Eye Aspect Ratio (EAR) given 6 (x, y) landmark points.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Vertical distances
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    # Horizontal distance
    h = np.linalg.norm(eye_points[0] - eye_points[3])

    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def align_and_crop_eye(frame_gray, crop_pts, corner_left, corner_right, patch_size, padding=EYE_PADDING):
    """
    Align the eye horizontally based on corners, crop and resize.
    """
    h, w = frame_gray.shape[:2]

    # --- Step 1: Compute rotation angle from eye corners ---
    dx = corner_right[0] - corner_left[0]
    dy = corner_right[1] - corner_left[1]
    angle = math.degrees(math.atan2(dy, dx))

    # --- Step 2: Compute center of the eye ---
    eye_center = crop_pts.mean(axis=0)
    cx, cy = eye_center[0], eye_center[1]

    # --- Step 3: Rotate the entire frame around the eye center ---
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame_gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # --- Step 4: Transform crop points to rotated coordinates ---
    ones = np.ones((len(crop_pts), 1))
    pts_hom = np.hstack([crop_pts, ones])  # (N, 3)
    rotated_pts = (M @ pts_hom.T).T  # (N, 2)

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
    """Print a simple text-based progress bar."""
    fraction = current / max(total, 1)
    filled = int(bar_len * fraction)
    bar = "█" * filled + "░" * (bar_len - filled)
    percent = fraction * 100
    print(f"\r  {prefix} |{bar}| {percent:5.1f}%  ({current}/{total})", end="", flush=True)


# ===================================================================
# Preview visualization
# ===================================================================

def build_preview(frame, pts, eye_pts_left, eye_pts_right,
                  patch_left, patch_right, ear_left, ear_right, frame_idx):
    """
    Build a debug/preview frame showing dense mesh landmarks and crop previews.
    """
    display = frame.copy()
    h_frame, w_frame = display.shape[:2]

    # Draw all landmarks as small gray dots
    for pt in pts:
        cv2.circle(display, (int(pt[0]), int(pt[1])), 1, (128, 128, 128), -1)

    # Draw eye landmarks with polylines (green = open, red = closed threshold)
    for eye_pts, ear, label in [(eye_pts_left, ear_left, "L"),
                                 (eye_pts_right, ear_right, "R")]:
        # Color based on EAR
        color = (0, 255, 0) if ear >= 0.21 else (0, 0, 255)

        # Convert to int format for cv2 drawings
        eye_pts_int = np.array(eye_pts, dtype=np.int32)

        # Draw eye contour
        cv2.polylines(display, [eye_pts_int], isClosed=True, color=color, thickness=2)

        # Draw each landmark point
        for pt in eye_pts:
            cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1)

        # Draw corner-to-corner line (alignment reference)
        p1 = (int(eye_pts[0][0]), int(eye_pts[0][1]))
        p4 = (int(eye_pts[3][0]), int(eye_pts[3][1]))
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

    cv2.putText(panel, f"Frame: {frame_idx}",
                (10, panel_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    combined = np.hstack([display, panel])
    return combined


# ===================================================================
# Main extraction routine
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract aligned eye patches from recorded videos using MediaPipe Face Mesh."
    )
    parser.add_argument(
        "--input", type=str, default=VIDEO_DIR,
        help="Path to input video folder or file (default: from config.py)"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_RAW_EYES_DIR,
        help="Path to output folder for eye patches and metadata (default: dataset/raw_eyes)"
    )
    parser.add_argument(
        "--detector", type=str, default="mediapipe",
        help="Face and landmark detector to use (default: mediapipe)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show real-time preview with landmarks and eye patches."
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # Discover input videos
    # ------------------------------------------------------------------
    video_extensions = (".mp4", ".avi")
    if os.path.isdir(args.input):
        video_files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(video_extensions)
        ])
    elif os.path.isfile(args.input) and args.input.lower().endswith(video_extensions):
        video_files = [args.input]
    else:
        video_files = []

    if not video_files:
        print(f"[ERROR] No video files found in '{args.input}'")
        sys.exit(1)

    print("=" * 60)
    print("  EYE PATCH EXTRACTOR (MediaPipe Face Mesh)")
    print("=" * 60)
    print(f"  Input path       : {args.input}")
    print(f"  Found videos     : {len(video_files)}")
    print(f"  Output directory : {args.output}")
    print(f"  Patch size       : {EYE_PATCH_SIZE}")
    print(f"  Detector         : {args.detector}")
    if args.preview:
        print(f"  Preview mode     : ON (press Q=skip video, ESC=quit)")
    print("=" * 60)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # Counters
    total_frames = 0
    total_extracted = 0
    skipped_frames = 0
    quit_all = False

    # Metadata records list
    metadata_records = []

    # Count total frames for progress bar
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
        video_id = os.path.splitext(video_name)[0]
        print(f"\n[VIDEO {vid_idx + 1}/{len(video_files)}] {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [WARNING] Cannot open {video_path} – skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 15.0  # Fallback

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame_idx += 1
            processed_so_far += 1

            timestamp_sec = (frame_idx - 1) / fps

            if not args.preview:
                print_progress_bar(processed_so_far, grand_total)

            h_img, w_img = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = face_mesh.process(rgb)

            face_detected = (results.multi_face_landmarks is not None and len(results.multi_face_landmarks) > 0)

            if not face_detected:
                skipped_frames += 1
                
                # Write skipped rows to metadata (both left and right eyes)
                for side in ["left", "right"]:
                    metadata_records.append({
                        "video_id": video_id,
                        "video_path": os.path.normpath(video_path).replace('\\', '/'),
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "eye_side": side,
                        "image_path": "",
                        "face_detected": False,
                        "landmark_detected": False,
                        "detector": args.detector,
                        "ear_left": "",
                        "ear_right": "",
                        "ear_avg": "",
                        "auto_label": "",
                        "review_label": "",
                        "final_label": "",
                        "status": "skipped",
                        "notes": "No face detected"
                    })

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

            # Face detected, extract landmarks
            face_landmarks = results.multi_face_landmarks[0]
            pts = np.array([(lm.x * w_img, lm.y * h_img) for lm in face_landmarks.landmark])

            # Get EAR points
            left_ear_pts = pts[LEFT_EYE_EAR_IDX]
            right_ear_pts = pts[RIGHT_EYE_EAR_IDX]

            ear_left = compute_ear(left_ear_pts)
            ear_right = compute_ear(right_ear_pts)
            ear_avg = (ear_left + ear_right) / 2.0

            # Get Crop points
            left_crop_pts = pts[LEFT_EYE_CROP_IDX]
            right_crop_pts = pts[RIGHT_EYE_CROP_IDX]

            # Corners for alignment
            # Left eye corners: 33 (left corner of left eye), 133 (right corner of left eye)
            left_p1 = pts[33]
            left_p4 = pts[133]
            # Right eye corners: 362 (left corner of right eye), 263 (right corner of right eye)
            right_p1 = pts[362]
            right_p4 = pts[263]

            patches = {}
            ears = {"left": ear_left, "right": ear_right}

            for side, crop_pts, p1, p4 in [("left", left_crop_pts, left_p1, left_p4),
                                            ("right", right_crop_pts, right_p1, right_p4)]:
                try:
                    patch = align_and_crop_eye(gray, crop_pts, p1, p4, EYE_PATCH_SIZE)
                except Exception as e:
                    patch = None
                    notes = f"Crop error: {str(e)}"
                else:
                    notes = ""

                patches[side] = patch

                if patch is None:
                    # Crop error or invalid
                    metadata_records.append({
                        "video_id": video_id,
                        "video_path": os.path.normpath(video_path).replace('\\', '/'),
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "eye_side": side,
                        "image_path": "",
                        "face_detected": True,
                        "landmark_detected": True,
                        "detector": args.detector,
                        "ear_left": round(ear_left, 4),
                        "ear_right": round(ear_right, 4),
                        "ear_avg": round(ear_avg, 4),
                        "auto_label": "",
                        "review_label": "",
                        "final_label": "",
                        "status": "error",
                        "notes": notes or "Crop failed (invalid coordinates)"
                    })
                    continue

                # Save eye patch image
                total_extracted += 1
                img_filename = f"{video_id}_frame{frame_idx:05d}_{side}.png"
                img_save_path = os.path.join(args.output, img_filename)
                cv2.imwrite(img_save_path, patch)

                # Store relative image path
                rel_img_path = os.path.normpath(os.path.join(args.output, img_filename)).replace('\\', '/')

                metadata_records.append({
                    "video_id": video_id,
                    "video_path": os.path.normpath(video_path).replace('\\', '/'),
                    "frame_index": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "eye_side": side,
                    "image_path": rel_img_path,
                    "face_detected": True,
                    "landmark_detected": True,
                    "detector": args.detector,
                    "ear_left": round(ear_left, 4),
                    "ear_right": round(ear_right, 4),
                    "ear_avg": round(ear_avg, 4),
                    "auto_label": "",
                    "review_label": "",
                    "final_label": "",
                    "status": "success",
                    "notes": ""
                })

            # --- Preview mode ---
            if args.preview:
                preview = build_preview(
                    frame, pts,
                    left_ear_pts, right_ear_pts,
                    patches.get("left"), patches.get("right"),
                    ear_left, ear_right,
                    frame_idx,
                )
                cv2.imshow("Extract Eyes - Preview", preview)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC = quit all
                    quit_all = True
                    break
                elif key == ord("q"):  # Q = skip this video
                    break

        cap.release()

    if args.preview:
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Save Metadata CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.output, "metadata.csv")
    fieldnames = [
        "video_id", "video_path", "frame_index", "timestamp_sec",
        "eye_side", "image_path", "face_detected", "landmark_detected",
        "detector", "ear_left", "ear_right", "ear_avg",
        "auto_label", "review_label", "final_label", "status", "notes"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_records)

    # Also save as ear_values.csv for backwards compatibility with unchanged tools (if any)
    # The columns: index, filename, ear, eye_side, source_video, frame
    ear_values_path = os.path.join(args.output, "ear_values.csv")
    compat_records = []
    compat_idx = 0
    for row in metadata_records:
        if row["status"] == "success" and row["image_path"]:
            compat_idx += 1
            compat_records.append({
                "index": compat_idx,
                "filename": os.path.basename(row["image_path"]),
                "ear": row["ear_left"] if row["eye_side"] == "left" else row["ear_right"],
                "eye_side": row["eye_side"],
                "source_video": os.path.basename(row["video_path"]),
                "frame": row["frame_index"],
            })
    with open(ear_values_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "ear", "eye_side", "source_video", "frame"])
        writer.writeheader()
        writer.writerows(compat_records)

    print("\n\n" + "=" * 60)
    print("  EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total frames processed : {total_frames}")
    print(f"  Skipped frames (no face): {skipped_frames}")
    print(f"  Total eye patches saved : {total_extracted}")
    print(f"  Metadata CSV saved to   : {csv_path}")
    print(f"  Legacy CSV saved to     : {ear_values_path}")
    print(f"  Output directory        : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
