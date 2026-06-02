"""
Central Configuration for Eye Blink Data Collection.
All constants, paths, and thresholds are defined here.
"""

import os

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")

# ============================================================
# CAMERA SETTINGS
# ============================================================
CAMERA_INDEX = 0
CAMERA_FPS = 15
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ============================================================
# EYE PATCH SETTINGS
# ============================================================
EYE_PATCH_SIZE = (24, 24)  # Width x Height for cropped eye images

# ============================================================
# EAR (Eye Aspect Ratio) SETTINGS
# ============================================================
# EAR threshold for semi-auto labeling (used in label_tool.py)
EAR_THRESHOLD_LABELING = 0.21

# ============================================================
# DATASET SPLIT SETTINGS
# ============================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
