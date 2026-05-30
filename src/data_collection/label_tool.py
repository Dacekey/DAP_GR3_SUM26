"""
label_tool.py – Semi-automatic labeling for the Eye Blink Detection dataset.

Modes (selected via --mode):
    auto    Auto-label eye patches using EAR threshold from ear_values.csv.
    review  Manual review: display each image + auto-label, user confirms
            or overrides with 'o' (open) / 'c' (closed) / SPACE (accept) / 'q' (quit).
    split   Organise labelled patches into train/val directories (80/20 split).

Usage examples:
    python data/label_tool.py --mode auto
    python data/label_tool.py --mode review
    python data/label_tool.py --mode split
"""

import os
import sys
import csv
import glob
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
    EYE_PATCH_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
)

# Derived paths
RAW_EYES_DIR = os.path.join(DATASET_DIR, "raw_eyes")
EAR_CSV_PATH = os.path.join(RAW_EYES_DIR, "ear_values.csv")
LABELS_CSV_PATH = os.path.join(RAW_EYES_DIR, "labels.csv")


# ===================================================================
# Helper functions
# ===================================================================

def load_ear_map(csv_path):
    """
    Load ear_values.csv produced by extract_eyes.py.

    Returns:
        dict  {filename: float(ear)}
    """
    ear_map = {}
    if not os.path.isfile(csv_path):
        return ear_map
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ear_map[row["filename"]] = float(row["ear"])
    return ear_map


def load_labels(csv_path):
    """
    Load labels.csv.

    Returns:
        dict  {filename: label_str}  where label ∈ {'open', 'closed'}
    """
    labels = {}
    if not os.path.isfile(csv_path):
        return labels
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["filename"]] = row["label"]
    return labels


def save_labels(labels_dict, csv_path):
    """
    Persist labels dict to CSV.

    Args:
        labels_dict: {filename: label}
        csv_path: output path
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for fname in sorted(labels_dict.keys()):
            writer.writerow({"filename": fname, "label": labels_dict[fname]})


def get_eye_images():
    """Return sorted list of eye patch filenames in RAW_EYES_DIR."""
    # NOTE: We use os.listdir instead of glob.glob because glob treats
    # square brackets [] as special characters, breaking paths like [SUMMER_26].
    if not os.path.isdir(RAW_EYES_DIR):
        return []
    return sorted([
        f for f in os.listdir(RAW_EYES_DIR)
        if f.startswith("eye_") and f.endswith(".png")
    ])


# ===================================================================
# Mode: auto
# ===================================================================

def mode_auto():
    """
    Auto-label every eye patch using the EAR value from ear_values.csv.

    Rule:  EAR < EAR_THRESHOLD_LABELING  →  'closed'
           EAR >= EAR_THRESHOLD_LABELING →  'open'
    """
    print("=" * 60)
    print("  AUTO-LABELING MODE")
    print("=" * 60)
    print(f"  EAR threshold : {EAR_THRESHOLD_LABELING}")
    print(f"  Source dir    : {RAW_EYES_DIR}")
    print()

    ear_map = load_ear_map(EAR_CSV_PATH)
    images = get_eye_images()

    if not images:
        print("[ERROR] No eye patches found. Run extract_eyes.py first.")
        return

    if not ear_map:
        print("[WARNING] ear_values.csv not found or empty.")
        print("          All patches will be labelled 'open' by default.")
        print("          Run extract_eyes.py to generate EAR values.")

    labels = {}
    open_count = 0
    closed_count = 0
    missing_ear = 0

    for fname in images:
        ear = ear_map.get(fname, None)
        if ear is None:
            # No EAR data – default to open (user can fix in review mode)
            labels[fname] = "open"
            open_count += 1
            missing_ear += 1
        elif ear < EAR_THRESHOLD_LABELING:
            labels[fname] = "closed"
            closed_count += 1
        else:
            labels[fname] = "open"
            open_count += 1

    save_labels(labels, LABELS_CSV_PATH)

    print(f"  Total images  : {len(images)}")
    print(f"  Open           : {open_count}")
    print(f"  Closed         : {closed_count}")
    if missing_ear:
        print(f"  Missing EAR    : {missing_ear} (defaulted to 'open')")
    print(f"\n  Labels saved to: {LABELS_CSV_PATH}")
    print("=" * 60)


# ===================================================================
# Mode: review
# ===================================================================

def mode_review():
    """
    Interactive review: show each eye patch with its current label.
    User presses:
        'o' → open
        'c' → closed
        SPACE → accept current label
        'q' → quit review early (progress is saved)
    """
    print("=" * 60)
    print("  MANUAL REVIEW MODE")
    print("=" * 60)
    print("  Controls:")
    print("    O     → label as 'open'")
    print("    C     → label as 'closed'")
    print("    SPACE → accept current label")
    print("    Q     → quit (progress is saved)")
    print("=" * 60)

    labels = load_labels(LABELS_CSV_PATH)
    images = get_eye_images()

    if not images:
        print("[ERROR] No eye patches found. Run extract_eyes.py first.")
        return

    if not labels:
        print("[WARNING] No labels found. Running auto-label first…")
        mode_auto()
        labels = load_labels(LABELS_CSV_PATH)

    reviewed = 0
    changed = 0

    for idx, fname in enumerate(images):
        img_path = os.path.join(RAW_EYES_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        current_label = labels.get(fname, "open")

        # Scale up for display (24×24 is tiny)
        display_size = 240
        display = cv2.resize(img, (display_size, display_size),
                             interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Draw label text
        label_color = (0, 255, 0) if current_label == "open" else (0, 0, 255)
        cv2.putText(display, current_label.upper(), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
        cv2.putText(display, f"{idx + 1}/{len(images)}", (10, display_size - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, fname, (10, display_size - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.imshow("Label Review – Eye Blink Detection", display)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("o"):
                if current_label != "open":
                    changed += 1
                labels[fname] = "open"
                break
            elif key == ord("c"):
                if current_label != "closed":
                    changed += 1
                labels[fname] = "closed"
                break
            elif key == ord(" "):
                # Accept current label
                break
            elif key == ord("q") or key == ord("Q"):
                print(f"\n[INFO] Review stopped at image {idx + 1}/{len(images)}.")
                save_labels(labels, LABELS_CSV_PATH)
                cv2.destroyAllWindows()
                print(f"  Reviewed : {reviewed}")
                print(f"  Changed  : {changed}")
                print(f"  Labels saved to: {LABELS_CSV_PATH}")
                return

        reviewed += 1

    cv2.destroyAllWindows()
    save_labels(labels, LABELS_CSV_PATH)

    print(f"\n[DONE] Review complete.")
    print(f"  Reviewed : {reviewed}")
    print(f"  Changed  : {changed}")
    print(f"  Labels saved to: {LABELS_CSV_PATH}")


# ===================================================================
# Mode: split
# ===================================================================

def mode_split():
    """
    Split labelled eye patches into train/test directories (80/20).

    Directory structure:
        dataset/train/open/
        dataset/train/closed/
        dataset/test/open/
        dataset/test/closed/
    """
    print("=" * 60)
    print("  TRAIN / TEST SPLIT MODE")
    print("=" * 60)

    labels = load_labels(LABELS_CSV_PATH)
    if not labels:
        print("[ERROR] No labels found. Run --mode auto first.")
        return

    # Group by label
    open_files = [f for f, lbl in labels.items() if lbl == "open"]
    closed_files = [f for f, lbl in labels.items() if lbl == "closed"]

    print(f"  Total labelled : {len(labels)}")
    print(f"  Open           : {len(open_files)}")
    print(f"  Closed         : {len(closed_files)}")
    print(f"  Test fraction  : {TEST_SIZE}")
    print()

    # Shuffle deterministically
    random.seed(RANDOM_STATE)
    random.shuffle(open_files)
    random.shuffle(closed_files)

    def split_list(file_list, test_frac):
        n_test = max(1, int(len(file_list) * test_frac))
        return file_list[n_test:], file_list[:n_test]

    train_open, test_open = split_list(open_files, TEST_SIZE)
    train_closed, test_closed = split_list(closed_files, TEST_SIZE)

    # Create directories
    dirs = {
        "train_open": os.path.join(TRAIN_DIR, "open"),
        "train_closed": os.path.join(TRAIN_DIR, "closed"),
        "test_open": os.path.join(TEST_DIR, "open"),
        "test_closed": os.path.join(TEST_DIR, "closed"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Copy files
    def copy_files(file_list, dest_dir, label):
        copied = 0
        for fname in file_list:
            src = os.path.join(RAW_EYES_DIR, fname)
            if not os.path.isfile(src):
                print(f"  [WARNING] Missing file: {src}")
                continue
            dst = os.path.join(dest_dir, fname)
            shutil.copy2(src, dst)
            copied += 1
        return copied

    n1 = copy_files(train_open, dirs["train_open"], "open")
    n2 = copy_files(train_closed, dirs["train_closed"], "closed")
    n3 = copy_files(test_open, dirs["test_open"], "open")
    n4 = copy_files(test_closed, dirs["test_closed"], "closed")

    print("  Files copied:")
    print(f"    train/open   : {n1}")
    print(f"    train/closed : {n2}")
    print(f"    test/open    : {n3}")
    print(f"    test/closed  : {n4}")
    print(f"    TOTAL        : {n1 + n2 + n3 + n4}")
    print()
    print(f"  Train dir : {TRAIN_DIR}")
    print(f"  Test dir  : {TEST_DIR}")
    print("=" * 60)


# ===================================================================
# Entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semi-automatic labeling tool for eye blink dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/label_tool.py --mode auto     # auto-label via EAR
  python data/label_tool.py --mode review   # manual review & correction
  python data/label_tool.py --mode split    # create train/val split
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["auto", "review", "split"],
        help="Labeling mode: auto | review | split",
    )

    args = parser.parse_args()

    if args.mode == "auto":
        mode_auto()
    elif args.mode == "review":
        mode_review()
    elif args.mode == "split":
        mode_split()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
