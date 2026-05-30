"""
Model Training & Evaluation Script for Eye Blink Detection.

This script:
    1. Loads the training and validation datasets (eye patches).
    2. Extracts features (EAR + HOG + LBP) via the features module.
    3. Scales features with StandardScaler.
    4. Trains an SVM (with GridSearchCV) and/or a Random Forest classifier.
    5. Evaluates on the validation set and prints detailed metrics.
    6. Plots and saves confusion matrices.
    7. Saves the best model and scaler to disk using joblib.

Usage:
    python src/train_model.py              # Train both SVM and RF
    python src/train_model.py --model svm  # Train SVM only
    python src/train_model.py --model rf   # Train Random Forest only
"""

import os
import sys
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Add project root to sys.path so we can import config and features
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_DIR,
    TEST_DIR,
    MODEL_DIR,
    CLASSIFIER_MODEL_PATH,
    SVM_KERNEL,
    SVM_GRID_SEARCH_PARAMS,
    RANDOM_FOREST_N_ESTIMATORS,
    CV_FOLDS,
    RANDOM_STATE,
)
from src.features import load_dataset


# ============================================================
# Utility Functions
# ============================================================

def _ensure_dirs():
    """Create the models directory if it does not exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def _print_banner(text: str):
    """Print a formatted section banner."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray, model_name: str):
    """
    Evaluate a trained model on the validation set.

    Prints accuracy, precision, recall, F1-score, confusion matrix,
    and a full classification report.

    Args:
        model: Trained sklearn estimator with a .predict() method.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation labels.
        model_name (str): Human-readable name for logging.

    Returns:
        dict: Dictionary of metric values.
    """
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_val, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_val, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    _print_banner(f"{model_name} — Evaluation Results")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}\n")
    print("  Classification Report:")
    print(
        classification_report(
            y_val, y_pred, target_names=["Open (0)", "Closed (1)"]
        )
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


# ============================================================
# Confusion Matrix Plotting
# ============================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str,
):
    """
    Plot and save a confusion matrix figure.

    Args:
        y_true (np.ndarray): Ground-truth labels.
        y_pred (np.ndarray): Predicted labels.
        model_name (str): Title prefix.
        save_path (str): File path to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Open", "Closed"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved to: {save_path}")


# ============================================================
# Training Functions
# ============================================================

def train_svm(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train an SVM classifier with GridSearchCV for hyperparameter tuning.

    Uses the RBF kernel and parameter grid defined in config.py.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.

    Returns:
        sklearn.svm.SVC: Best SVM estimator found by GridSearchCV.
    """
    _print_banner("Training SVM with GridSearchCV")

    svm = SVC(kernel=SVM_KERNEL, random_state=RANDOM_STATE, probability=True)

    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=SVM_GRID_SEARCH_PARAMS,
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters : {grid_search.best_params_}")
    print(f"  Best CV F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train a Random Forest classifier.

    Args:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.

    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained RF model.
    """
    _print_banner("Training Random Forest")

    rf = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    rf.fit(X_train, y_train)

    return rf


def print_feature_importance(rf_model, n_top: int = 15):
    """
    Print the top-N most important features from a Random Forest.

    Args:
        rf_model: Trained RandomForestClassifier.
        n_top (int): Number of top features to display.
    """
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    _print_banner("Random Forest — Feature Importances (Top {})".format(n_top))
    print(f"  {'Rank':<6}{'Feature Index':<16}{'Importance':<12}")
    print(f"  {'-'*6}{'-'*16}{'-'*12}")

    for rank, idx in enumerate(indices[:n_top], start=1):
        # Provide a human-readable label for known feature positions
        label = _feature_label(idx)
        print(f"  {rank:<6}{idx:<16}{importances[idx]:<12.6f}  {label}")


def _feature_label(idx: int) -> str:
    """Return a human-readable label for a feature index."""
    if idx == 0:
        return "(EAR)"
    # HOG features start at index 1; compute HOG length to find boundary
    # For a 24x24 patch with (8,8) pixels_per_cell and (2,2) cells_per_block:
    #   cells = 24/8 = 3 per axis → 3x3 grid
    #   blocks = (3-2+1)=2 per axis → 2x2=4 blocks
    #   features per block = 2*2*9 = 36
    #   total HOG = 4*36 = 144
    hog_end = 1 + 144  # index 1..144
    if idx < hog_end:
        return f"(HOG #{idx - 1})"
    return f"(LBP #{idx - hog_end})"


# ============================================================
# Main
# ============================================================

def main():
    """Entry point for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train eye-blink classifiers (SVM / Random Forest)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["svm", "rf", "both"],
        default="both",
        help="Which model to train: 'svm', 'rf', or 'both' (default: both).",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    _ensure_dirs()

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    _print_banner("Loading Training Data")
    try:
        X_train, y_train = load_dataset(TRAIN_DIR)
    except ValueError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    _print_banner("Loading Test Data")
    try:
        X_test, y_test = load_dataset(TEST_DIR)
    except ValueError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Feature scaling
    # ------------------------------------------------------------------
    _print_banner("Feature Scaling (StandardScaler)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Test shape : {X_test_scaled.shape}")

    # Keep track of results for comparison
    results = {}

    # ------------------------------------------------------------------
    # 3. Train & evaluate SVM
    # ------------------------------------------------------------------
    if args.model in ("svm", "both"):
        svm_model = train_svm(X_train_scaled, y_train)
        svm_results = evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
        results["SVM"] = svm_results

        # Save confusion matrix plot
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix_svm.png")
        plot_confusion_matrix(y_test, svm_results["y_pred"], "SVM", cm_path)

        # Save model and scaler
        joblib.dump(svm_model, CLASSIFIER_MODEL_PATH)
        print(f"  SVM model saved to: {CLASSIFIER_MODEL_PATH}")

        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved to  : {scaler_path}")

    # ------------------------------------------------------------------
    # 4. Train & evaluate Random Forest
    # ------------------------------------------------------------------
    if args.model in ("rf", "both"):
        rf_model = train_random_forest(X_train_scaled, y_train)
        rf_results = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
        results["RF"] = rf_results

        # Save confusion matrix plot
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix_rf.png")
        plot_confusion_matrix(y_test, rf_results["y_pred"], "Random Forest", cm_path)

        # Feature importance analysis
        print_feature_importance(rf_model)

        # Save RF model (separate file so it doesn't overwrite SVM)
        rf_path = os.path.join(MODEL_DIR, "rf_blink_model.pkl")
        joblib.dump(rf_model, rf_path)
        print(f"  RF model saved to: {rf_path}")

        # If only RF was trained, also save the scaler
        if args.model == "rf":
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            print(f"  Scaler saved to  : {scaler_path}")

    # ------------------------------------------------------------------
    # 5. Combined confusion matrix (when both models are trained)
    # ------------------------------------------------------------------
    if args.model == "both" and len(results) == 2:
        # Save a side-by-side confusion matrix figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, (name, res) in zip(axes, results.items()):
            cm = res["confusion_matrix"]
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Open", "Closed"]
            )
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(
                f"{name}  (F1={res['f1']:.3f})", fontsize=13
            )

        plt.suptitle("Model Comparison — Confusion Matrices", fontsize=15)
        plt.tight_layout()
        combined_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
        plt.savefig(combined_path, dpi=150)
        plt.close(fig)
        print(f"\n  Combined confusion matrix saved to: {combined_path}")

        # Print comparison table
        _print_banner("Model Comparison Summary")
        header = f"  {'Metric':<12}{'SVM':<12}{'Random Forest':<14}"
        print(header)
        print(f"  {'-'*12}{'-'*12}{'-'*14}")
        for metric in ("accuracy", "precision", "recall", "f1"):
            svm_val = results["SVM"][metric]
            rf_val = results["RF"][metric]
            winner = " ◀" if svm_val >= rf_val else ""
            winner_rf = " ◀" if rf_val > svm_val else ""
            print(
                f"  {metric.capitalize():<12}"
                f"{svm_val:<12.4f}{winner}"
                f"{rf_val:<14.4f}{winner_rf}"
            )

    _print_banner("Training Complete ✓")
    print("  All models and artifacts have been saved to:", MODEL_DIR)
    print()


if __name__ == "__main__":
    main()
