"""
Feature Extraction Module for Eye Blink Detection.

Combines three types of features:
1. EAR (Eye Aspect Ratio) - geometric ratio from facial landmarks
2. HOG (Histogram of Oriented Gradients) - texture/shape descriptor
3. LBP (Local Binary Pattern) - local texture descriptor

These features are concatenated into a single vector for classification.
"""

import os
import sys
import numpy as np
from scipy.spatial import distance as dist
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize
import cv2
from glob import glob

# ---------------------------------------------------------------------------
# Add project root to sys.path so we can import config
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EYE_PATCH_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    LBP_RADIUS,
    LBP_N_POINTS,
    LBP_METHOD,
)


# ============================================================
# EAR (Eye Aspect Ratio)
# ============================================================

def compute_ear(eye_points: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    The EAR is a scalar value that describes how "open" an eye is.
    When the eye is open, EAR is relatively large; when closed, EAR
    approaches zero.

    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye_points (np.ndarray): Array of shape (6, 2) containing the
            (x, y) coordinates of the 6 eye landmarks in order:
                p1 - left corner
                p2 - upper-left
                p3 - upper-right
                p4 - right corner
                p5 - lower-right
                p6 - lower-left

    Returns:
        float: The Eye Aspect Ratio value.
    """
    # Vertical distances
    vertical_1 = dist.euclidean(eye_points[1], eye_points[5])  # ||p2 - p6||
    vertical_2 = dist.euclidean(eye_points[2], eye_points[4])  # ||p3 - p5||

    # Horizontal distance
    horizontal = dist.euclidean(eye_points[0], eye_points[3])  # ||p1 - p4||

    # Avoid division by zero
    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def compute_avg_ear(
    left_eye_points: np.ndarray, right_eye_points: np.ndarray
) -> float:
    """
    Compute the average EAR across both eyes.

    Args:
        left_eye_points (np.ndarray): Shape (6, 2) landmarks for the left eye.
        right_eye_points (np.ndarray): Shape (6, 2) landmarks for the right eye.

    Returns:
        float: Average EAR of the two eyes.
    """
    left_ear = compute_ear(left_eye_points)
    right_ear = compute_ear(right_eye_points)
    return (left_ear + right_ear) / 2.0


# ============================================================
# HOG Features
# ============================================================

def extract_hog_features(eye_patch: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from an eye patch.

    The input image is resized to the canonical patch size defined in config
    before feature extraction to ensure consistent feature vector length.

    Args:
        eye_patch (np.ndarray): Grayscale eye patch image (2D array).

    Returns:
        np.ndarray: 1-D HOG feature vector.
    """
    # Ensure the patch is the correct size
    patch = _prepare_patch(eye_patch)

    features = hog(
        patch,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


# ============================================================
# LBP Features
# ============================================================

def extract_lbp_features(eye_patch: np.ndarray) -> np.ndarray:
    """
    Extract LBP (Local Binary Pattern) features from an eye patch.

    Computes the LBP image and then returns a normalised histogram.
    For the 'uniform' method with n_points=8, there are (n_points + 2) = 10
    distinct patterns, so the histogram has 10 bins.

    Args:
        eye_patch (np.ndarray): Grayscale eye patch image (2D array).

    Returns:
        np.ndarray: Normalised LBP histogram (length = LBP_N_POINTS + 2).
    """
    patch = _prepare_patch(eye_patch)

    # Compute LBP image
    lbp_image = local_binary_pattern(
        patch, P=LBP_N_POINTS, R=LBP_RADIUS, method=LBP_METHOD
    )

    # Number of bins for uniform LBP: n_points + 2
    n_bins = LBP_N_POINTS + 2

    # Build normalized histogram
    hist, _ = np.histogram(
        lbp_image.ravel(), bins=n_bins, range=(0, n_bins), density=True
    )
    return hist


# ============================================================
# Combined Feature Vector
# ============================================================

def extract_all_features(
    eye_patch: np.ndarray, ear_value: float = None
) -> np.ndarray:
    """
    Extract and concatenate all features into a single vector.

    Feature layout: [EAR, HOG_features, LBP_features]

    Args:
        eye_patch (np.ndarray): Grayscale eye patch image (2D array).
        ear_value (float, optional): Pre-computed EAR value. If None the
            EAR component is set to 0.0 (useful when training from cropped
            patches that lack landmark information).

    Returns:
        np.ndarray: Concatenated feature vector.
    """
    # EAR feature (scalar → 1-element array)
    ear_feat = np.array([ear_value if ear_value is not None else 0.0])

    # HOG features
    hog_feat = extract_hog_features(eye_patch)

    # LBP features
    lbp_feat = extract_lbp_features(eye_patch)

    # Concatenate: [EAR | HOG | LBP]
    feature_vector = np.concatenate([ear_feat, hog_feat, lbp_feat])
    return feature_vector


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(data_dir: str):
    """
    Load all eye-patch images from a directory and extract features.

    Expected directory structure:
        data_dir/
            open/      ← images of open eyes  (label 0)
            closed/    ← images of closed eyes (label 1)

    Each image is loaded as grayscale, resized, and features are extracted
    using `extract_all_features` (with EAR set to 0 since landmark info
    is not available from static images).

    Args:
        data_dir (str): Path to the dataset split directory (e.g. train/).

    Returns:
        tuple: (X, y) where
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Label array of shape (n_samples,).
                            0 = open, 1 = closed.
    """
    features_list = []
    labels_list = []

    # Define class mapping: folder name → numeric label
    class_map = {"open": 0, "closed": 1}

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARNING] Directory not found, skipping: {class_dir}")
            continue

        # Gather image files (common extensions)
        # NOTE: We use os.listdir instead of glob because glob treats
        # square brackets [] as special characters in the path.
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
        image_paths = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(valid_exts)
        ]

        if len(image_paths) == 0:
            print(f"[WARNING] No images found in {class_dir}")
            continue

        print(f"Loading {len(image_paths)} images from '{class_name}' ...")

        for img_path in image_paths:
            try:
                # Load as grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"  [SKIP] Could not read: {img_path}")
                    continue

                # Extract combined features (EAR = None → 0)
                feat = extract_all_features(img, ear_value=None)
                features_list.append(feat)
                labels_list.append(label)
            except Exception as e:
                print(f"  [ERROR] {img_path}: {e}")

    if len(features_list) == 0:
        raise ValueError(
            f"No valid images were loaded from {data_dir}. "
            "Please check the directory structure (open/ and closed/ subfolders)."
        )

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Open: {np.sum(y == 0)}, Closed: {np.sum(y == 1)}")
    return X, y


# ============================================================
# Helper Functions
# ============================================================

def _prepare_patch(eye_patch: np.ndarray) -> np.ndarray:
    """
    Resize and normalise an eye patch to the canonical size.

    Converts to float64 in [0, 1] range if needed and resizes to
    EYE_PATCH_SIZE (width, height) defined in config.

    Args:
        eye_patch (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Resized image with values in [0, 1].
    """
    # Convert to float [0, 1] if the image is uint8
    if eye_patch.dtype == np.uint8:
        patch = eye_patch.astype(np.float64) / 255.0
    else:
        patch = eye_patch.astype(np.float64)

    # Resize to canonical size (height, width) — note: config stores (W, H)
    target_h, target_w = EYE_PATCH_SIZE[1], EYE_PATCH_SIZE[0]
    if patch.shape[0] != target_h or patch.shape[1] != target_w:
        patch = resize(patch, (target_h, target_w), anti_aliasing=True)

    return patch
