"""
label_tool.py – Semi-automatic labeling for the Eye Blink Detection dataset.

Modes (selected via --mode):
    auto    Auto-label eye patches using EAR threshold from metadata.csv.
    review  Manual review: display each image + auto-label, user confirms,
            re-labels ('o' = open, 'c' = closed), deletes ('d' or 'x'),
            accepts (SPACE), or quits ('q').
    split   Organise labelled patches into train/test directories (80/20 split).

Usage examples:
    python src/data_collection/label_tool.py --mode auto --threshold 0.20
    python src/data_collection/label_tool.py --mode review
    python src/data_collection/label_tool.py --mode split
"""

import os
import sys
import csv
import shutil
import random
import argparse
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import project-wide constants from config.py
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    DATASET_DIR,
    TRAIN_DIR,
    TEST_DIR,
    EAR_THRESHOLD_LABELING,
    TEST_SIZE,
    RANDOM_STATE,
)

# Default paths
DEFAULT_RAW_EYES_DIR = os.path.join(DATASET_DIR, "raw_eyes")


def resolve_paths(custom_dir):
    """
    Resolve the input directory containing images and metadata.csv.
    If custom_dir is provided, use it. Otherwise, look for extracted_eyes/
    or default dataset/raw_eyes/.
    """
    if custom_dir:
        return custom_dir

    # Check if 'extracted_eyes' directory exists and has metadata.csv
    extracted_dir = "extracted_eyes"
    if os.path.isdir(extracted_dir) and os.path.isfile(os.path.join(extracted_dir, "metadata.csv")):
        return extracted_dir

    # Fallback to default raw_eyes
    return DEFAULT_RAW_EYES_DIR


def load_metadata(csv_path):
    """Load metadata.csv as a list of dicts."""
    records = []
    if not os.path.isfile(csv_path):
        return records
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def save_metadata(records, csv_path):
    """Save metadata records back to metadata.csv."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_legacy_labels_csv(records, output_dir):
    """
    Save a legacy compatibility labels.csv file mapping filename -> final_label.
    This ensures compatibility with any legacy tools.
    """
    labels_csv_path = os.path.join(output_dir, "labels.csv")
    legacy_rows = []
    for row in records:
        if row["status"] == "success" and row["image_path"] and row["final_label"]:
            legacy_rows.append({
                "filename": os.path.basename(row["image_path"]),
                "label": row["final_label"]
            })
    
    if not legacy_rows:
        return

    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for r in sorted(legacy_rows, key=lambda x: x["filename"]):
            writer.writerow(r)


# ===================================================================
# Mode: auto
# ===================================================================

def mode_auto(output_dir, threshold):
    """
    Auto-label every eye patch using the EAR values in metadata.csv.
    """
    csv_path = os.path.join(output_dir, "metadata.csv")
    print("=" * 60)
    print("  AUTO-LABELING MODE")
    print("=" * 60)
    print(f"  EAR threshold : {threshold}")
    print(f"  Source dir    : {output_dir}")
    print()

    if not os.path.isfile(csv_path):
        print(f"[ERROR] metadata.csv not found at: {csv_path}")
        print("        Run extract_eyes.py first to generate metadata.")
        return

    records = load_metadata(csv_path)
    if not records:
        print("[WARNING] metadata.csv is empty.")
        return

    open_count = 0
    closed_count = 0
    skipped_count = 0

    for row in records:
        # Only label successfully cropped images
        if row["status"] != "success" or not row["image_path"]:
            row["auto_label"] = ""
            row["final_label"] = ""
            skipped_count += 1
            continue

        # Get the EAR value corresponding to this eye side
        try:
            ear = float(row["ear_left"]) if row["eye_side"] == "left" else float(row["ear_right"])
        except (ValueError, TypeError):
            row["auto_label"] = "open"
            row["final_label"] = "open"
            open_count += 1
            continue

        # Classify based on threshold
        if ear < threshold:
            label = "closed"
            closed_count += 1
        else:
            label = "open"
            open_count += 1

        row["auto_label"] = label
        row["final_label"] = label

    # Save updated metadata
    save_metadata(records, csv_path)
    save_legacy_labels_csv(records, output_dir)

    print(f"  Total records : {len(records)}")
    print(f"  Labeled success: {open_count + closed_count}")
    print(f"    Open         : {open_count}")
    print(f"    Closed       : {closed_count}")
    print(f"  Skipped/Errors : {skipped_count}")
    print(f"\n  Metadata CSV updated: {csv_path}")
    print("=" * 60)


# ===================================================================
# Mode: review
# ===================================================================

def mode_review(output_dir):
    """
    Interactive review: show each eye patch with its current label.
    User can relabel, delete, or confirm using the keyboard.
    """
    csv_path = os.path.join(output_dir, "metadata.csv")
    print("=" * 60)
    print("  MANUAL REVIEW MODE")
    print("=" * 60)
    print("  Controls:")
    print("    O         → label as 'open'")
    print("    C         → label as 'closed'")
    print("    D / X     → delete / discard image from dataset")
    print("    SPACE     → accept current label")
    print("    Q         → quit and save progress")
    print("=" * 60)

    if not os.path.isfile(csv_path):
        print(f"[ERROR] metadata.csv not found at: {csv_path}")
        print("        Run extract_eyes.py first.")
        return

    records = load_metadata(csv_path)
    if not records:
        print("[WARNING] metadata.csv is empty.")
        return

    # Check if auto-labeled first; if not, run auto-labeling
    has_labels = any(row["final_label"] != "" for row in records if row["status"] == "success")
    if not has_labels:
        print("[WARNING] No labels found. Running auto-label first…")
        mode_auto(output_dir, EAR_THRESHOLD_LABELING)
        records = load_metadata(csv_path)

    reviewed = 0
    changed = 0
    deleted = 0

    # Variables for video reading
    current_video_path = None
    cap = None

    # Filter records that were successfully extracted
    valid_records = [r for r in records if r["status"] == "success" and r["image_path"]]

    for idx, row in enumerate(valid_records):
        img_path = row["image_path"]
        
        # If absolute path doesn't exist, try relative to current directory
        if not os.path.isfile(img_path):
            img_path = os.path.join(output_dir, os.path.basename(row["image_path"]))

        if not os.path.isfile(img_path):
            print(f"  [WARNING] Image not found, skipping: {img_path}")
            row["status"] = "error"
            row["notes"] = "Image file missing during review"
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARNING] Could not read image: {img_path}")
            row["status"] = "error"
            row["notes"] = "Corrupted image file"
            continue

        current_label = row["final_label"] or "open"

        # Scale up for display (24x24 -> 240x240)
        display_size = 240
        display = cv2.resize(img, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Draw UI overlay info
        label_color = (0, 255, 0) if current_label == "open" else (0, 0, 255)
        cv2.putText(display, f"LABEL: {current_label.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        cv2.putText(display, f"Side: {row['eye_side'].upper()}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Calculate EAR display
        try:
            ear = float(row["ear_left"]) if row["eye_side"] == "left" else float(row["ear_right"])
            ear_str = f"EAR: {ear:.3f}"
        except (ValueError, TypeError):
            ear_str = "EAR: N/A"
        cv2.putText(display, ear_str, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(display, f"Progress: {idx + 1}/{len(valid_records)}", (10, display_size - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(display, os.path.basename(img_path), (10, display_size - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

        # Draw quick controls instructions
        cv2.putText(display, "[O]=Open [C]=Closed [D/X]=Delete [SPACE]=Ok [Q]=Quit", (5, display_size - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        # Retrieve the original frame from the video
        video_path = row.get("video_path", "")
        frame_idx = int(row.get("frame_index", 0))
        full_frame_display = np.zeros((display_size, 320, 3), dtype=np.uint8)

        if video_path and os.path.isfile(video_path):
            if current_video_path != video_path:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(video_path)
                current_video_path = video_path
            
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Resize the full frame to match display_size height
                    # Assuming 640x480 (4:3) -> 320x240
                    h, w = frame.shape[:2]
                    scale = display_size / float(h)
                    new_w = int(w * scale)
                    resized_frame = cv2.resize(frame, (new_w, display_size))
                    
                    # Center it in the 320x240 box
                    if new_w <= 320:
                        offset = (320 - new_w) // 2
                        full_frame_display[:, offset:offset+new_w] = resized_frame
                    else:
                        full_frame_display = resized_frame[:, :320]
                    
                    cv2.putText(full_frame_display, f"Original Frame ({frame_idx})", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                else:
                    cv2.putText(full_frame_display, "Frame Read Error", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(full_frame_display, "Video File Missing", (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Combine both images side by side
        combined_display = np.hstack((full_frame_display, display))

        cv2.imshow("Dataset Review Tool", combined_display)

        quit_review = False
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("o") or key == ord("O"):
                if current_label != "open":
                    changed += 1
                row["review_label"] = "open"
                row["final_label"] = "open"
                row["status"] = "success"
                row["notes"] = ""
                break
            elif key == ord("c") or key == ord("C"):
                if current_label != "closed":
                    changed += 1
                row["review_label"] = "closed"
                row["final_label"] = "closed"
                row["status"] = "success"
                row["notes"] = ""
                break
            elif key == ord("d") or key == ord("D") or key == ord("x") or key == ord("X"):
                # Delete image from disk and mark as discarded
                deleted += 1
                row["review_label"] = "discarded"
                row["final_label"] = "discarded"
                row["status"] = "deleted"
                row["notes"] = "Discarded by reviewer"
                
                # Perform file deletion
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        print(f"\n  [DELETE] Removed image: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"\n  [ERROR] Cannot delete file {img_path}: {e}")
                break
            elif key == ord(" "):
                # Accept current label as is
                break
            elif key == ord("q") or key == ord("Q") or key == 27:  # ESC or Q
                quit_review = True
                break

        if quit_review:
            print(f"\n[INFO] Review stopped early at item {idx + 1}/{len(valid_records)}.")
            break

        reviewed += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    # Save changes
    save_metadata(records, csv_path)
    save_legacy_labels_csv(records, output_dir)

    print(f"\n[DONE] Review completed.")
    print(f"  Images reviewed : {reviewed}")
    print(f"  Labels changed  : {changed}")
    print(f"  Images deleted  : {deleted}")
    print(f"  Metadata CSV saved to: {csv_path}")
    print("=" * 60)


# ===================================================================
# Mode: split
# ===================================================================

def mode_split(output_dir):
    """
    Split successfully labeled eye patches into train/test directories (80/20).
    """
    csv_path = os.path.join(output_dir, "metadata.csv")
    print("=" * 60)
    print("  TRAIN / TEST SPLIT MODE")
    print("=" * 60)

    if not os.path.isfile(csv_path):
        print(f"[ERROR] metadata.csv not found at: {csv_path}")
        print("        Run --mode auto first.")
        return

    records = load_metadata(csv_path)
    if not records:
        print("[WARNING] metadata.csv is empty.")
        return

    # Filter records that are valid and labeled
    valid_records = [
        r for r in records
        if r["status"] == "success" and r["final_label"] in ("open", "closed") and r["image_path"]
    ]

    if not valid_records:
        print("[ERROR] No valid labeled records to split. Run --mode auto or --mode review first.")
        return

    open_files = [r for r in valid_records if r["final_label"] == "open"]
    closed_files = [r for r in valid_records if r["final_label"] == "closed"]

    print(f"  Total labeled  : {len(valid_records)}")
    print(f"  Open           : {len(open_files)}")
    print(f"  Closed         : {len(closed_files)}")
    print(f"  Test fraction  : {TEST_SIZE}")
    print()

    # Shuffle deterministically using config seed
    random.seed(RANDOM_STATE)
    random.shuffle(open_files)
    random.shuffle(closed_files)

    # Split lists
    def split_list(lst, test_frac):
        n_test = max(1, int(len(lst) * test_frac))
        return lst[n_test:], lst[:n_test]

    train_open, test_open = split_list(open_files, TEST_SIZE)
    train_closed, test_closed = split_list(closed_files, TEST_SIZE)

    # Target directories
    dirs = {
        "train_open": os.path.join(TRAIN_DIR, "open"),
        "train_closed": os.path.join(TRAIN_DIR, "closed"),
        "test_open": os.path.join(TEST_DIR, "open"),
        "test_closed": os.path.join(TEST_DIR, "closed"),
    }
    
    # Recreate clean directories
    for name, d in dirs.items():
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Copy files helper
    def copy_records(records_list, dest_dir):
        copied = 0
        for row in records_list:
            src = row["image_path"]
            
            # If path not exists directly, check inside output_dir
            if not os.path.isfile(src):
                src = os.path.join(output_dir, os.path.basename(row["image_path"]))

            if not os.path.isfile(src):
                print(f"  [WARNING] File missing during copy: {src}")
                continue
            
            dst = os.path.join(dest_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            copied += 1
        return copied

    n1 = copy_records(train_open, dirs["train_open"])
    n2 = copy_records(train_closed, dirs["train_closed"])
    n3 = copy_records(test_open, dirs["test_open"])
    n4 = copy_records(test_closed, dirs["test_closed"])

    print("  Files copied successfully:")
    print(f"    train/open   : {n1}")
    print(f"    train/closed : {n2}")
    print(f"    test/open    : {n3}")
    print(f"    test/closed  : {n4}")
    print(f"    TOTAL        : {n1 + n2 + n3 + n4}")
    print()
    print(f"  Train directory: {TRAIN_DIR}")
    print(f"  Test directory : {TEST_DIR}")
    print("=" * 60)


# ===================================================================
# Entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semi-automatic labeling and review tool for eye blink dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["auto", "review", "split"],
        help="Labeling mode: auto | review | split",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom EAR threshold for auto-labeling (overrides EAR_THRESHOLD_LABELING in config.py)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing eye images and metadata.csv (default: auto-detected)",
    )

    args = parser.parse_args()

    # Resolve paths
    output_dir = resolve_paths(args.dir)

    # Use specified threshold, or fallback to config
    threshold = args.threshold if args.threshold is not None else EAR_THRESHOLD_LABELING

    if args.mode == "auto":
        mode_auto(output_dir, threshold)
    elif args.mode == "review":
        mode_review(output_dir)
    elif args.mode == "split":
        mode_split(output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
