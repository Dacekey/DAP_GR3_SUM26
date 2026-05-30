"""
Central Configuration for Eye Blink Detection System.
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
TEST_VIDEO_DIR = os.path.join(DATASET_DIR, "test_videos")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw_videos")

# dlib facial landmark model path
LANDMARK_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "shape_predictor_68_face_landmarks.dat"
)

# Trained classifier model path
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "svm_blink_model.pkl")

# ============================================================
# CAMERA SETTINGS
# ============================================================
CAMERA_INDEX = 0
CAMERA_FPS = 15
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ============================================================
# FACIAL LANDMARK INDICES (dlib 68-point model)
# ============================================================
# Left eye: points 36-41, Right eye: points 42-47
LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))

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
# FEATURE EXTRACTION SETTINGS
# ============================================================
# HOG (Histogram of Oriented Gradients)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP (Local Binary Pattern)
LBP_RADIUS = 1
LBP_N_POINTS = 8
LBP_METHOD = "uniform"

# ============================================================
# BLINK DETECTION SETTINGS
# ============================================================
# Minimum consecutive closed frames to count as a blink
MIN_BLINK_FRAMES = 2
# Maximum consecutive closed frames (beyond this = drowsiness, not blink)
MAX_BLINK_FRAMES = 10

# ============================================================
# HEALTH ALERT THRESHOLDS
# ============================================================
# Blink rate (blinks per minute)
BLINK_RATE_NORMAL = 15      # >= 15 blinks/min → healthy
BLINK_RATE_WARNING = 10     # 10-14 blinks/min → warning
BLINK_RATE_DANGER = 5       # < 5 blinks/min → danger

# Long eye closure detection (seconds)
LONG_CLOSE_WARNING_SEC = 0.4
LONG_CLOSE_DANGER_SEC = 2.0

# Continuous usage time alerts (minutes)
USAGE_WARNING_MINUTES = 30
USAGE_DANGER_MINUTES = 60

# ============================================================
# DASHBOARD SETTINGS
# ============================================================
DASHBOARD_UPDATE_MS = 1000      # Update interval in milliseconds
DASHBOARD_WIDTH = 340
DASHBOARD_HEIGHT = 480
DASHBOARD_BG_COLOR = "#1a1a2e"  # Dark background
DASHBOARD_FG_COLOR = "#e0e0e0"  # Light text
DASHBOARD_ACCENT = "#0f3460"    # Accent color

# Status colors
COLOR_NORMAL = "#00c853"   # Green
COLOR_WARNING = "#ffd600"  # Yellow
COLOR_DANGER = "#ff1744"   # Red

# ============================================================
# SVM TRAINING SETTINGS
# ============================================================
SVM_KERNEL = "rbf"
SVM_GRID_SEARCH_PARAMS = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.001],
}
RANDOM_FOREST_N_ESTIMATORS = 200
CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
