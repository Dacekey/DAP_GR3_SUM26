"""
Real-time Blink Detection Engine.

Uses a state machine to track eye states (OPEN, MAYBE_CLOSED, CLOSED, MAYBE_OPEN)
and detect complete blinks vs. prolonged eye closures (drowsiness).

Dependencies:
    - dlib (face detection + 68-point landmark predictor)
    - scikit-learn (SVM classifier)
    - OpenCV, NumPy
"""

import os
import sys
import time
import joblib
import logging
from enum import Enum, auto

import cv2
import dlib
import numpy as np

# ---------------------------------------------------------------------------
# Path setup – allow imports from the project root
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EYE_PATCH_SIZE,
    MIN_BLINK_FRAMES,
    MAX_BLINK_FRAMES,
    CAMERA_FPS,
    LANDMARK_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
)
from src.features import compute_ear, compute_avg_ear, extract_all_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine states
# ---------------------------------------------------------------------------
class EyeState(Enum):
    """States for the blink detection state machine."""
    OPEN = auto()
    MAYBE_CLOSED = auto()
    CLOSED = auto()
    MAYBE_OPEN = auto()


class BlinkDetector:
    """
    Real-time blink detection engine.

    Processes BGR video frames one-by-one and maintains an internal state
    machine that distinguishes normal blinks from prolonged eye closures
    (drowsiness).

    Parameters
    ----------
    model_path : str
        Path to the trained SVM classifier (.pkl).
    scaler_path : str
        Path to the fitted StandardScaler (.pkl).
    landmark_model_path : str
        Path to dlib's ``shape_predictor_68_face_landmarks.dat``.
    """

    def __init__(
        self,
        model_path: str = CLASSIFIER_MODEL_PATH,
        scaler_path: str = None,
        landmark_model_path: str = LANDMARK_MODEL_PATH,
    ):
        # ---- Load dlib models ------------------------------------------------
        if not os.path.isfile(landmark_model_path):
            raise FileNotFoundError(
                f"Landmark model not found: {landmark_model_path}\n"
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        self._face_detector = dlib.get_frontal_face_detector()
        self._landmark_predictor = dlib.shape_predictor(landmark_model_path)
        logger.info("dlib models loaded successfully.")

        # ---- Load SVM classifier + scaler ------------------------------------
        # Use joblib.load() because train_model.py saves with joblib.dump()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SVM model not found: {model_path}")
        self._model = joblib.load(model_path)
        logger.info("SVM classifier loaded from %s", model_path)

        self._scaler = None
        if scaler_path is None:
            # Try default scaler path next to the model
            scaler_path = os.path.join(
                os.path.dirname(model_path), "scaler.pkl"
            )
        if os.path.isfile(scaler_path):
            self._scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded from %s", scaler_path)
        else:
            logger.warning(
                "Scaler file not found at %s – features will NOT be scaled.",
                scaler_path,
            )

        # ---- Internal state --------------------------------------------------
        self._state = EyeState.OPEN
        self._blink_count: int = 0
        self._closed_frame_count: int = 0
        self._close_start_time: float = 0.0
        self._fps: int = CAMERA_FPS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single BGR video frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H×W×3) from OpenCV ``VideoCapture``.

        Returns
        -------
        dict
            Keys:
            - ``face_detected`` (bool)
            - ``ear`` (float | None)
            - ``eye_state`` (str: ``'open'`` / ``'closed'``)
            - ``blink_detected`` (bool – True the frame a complete blink ends)
            - ``blink_count`` (int – cumulative total)
            - ``closed_duration`` (float – seconds eyes continuously closed)
            - ``left_eye_points`` (np.ndarray | None)
            - ``right_eye_points`` (np.ndarray | None)
        """
        result = {
            "face_detected": False,
            "ear": None,
            "eye_state": "open",
            "blink_detected": False,
            "blink_count": self._blink_count,
            "closed_duration": 0.0,
            "left_eye_points": None,
            "right_eye_points": None,
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Face detection ---------------------------------------------------
        faces = self._face_detector(gray, 0)
        if len(faces) == 0:
            return result
        result["face_detected"] = True

        # Use the largest face
        face = max(faces, key=lambda r: r.width() * r.height())
        landmarks = self._landmark_predictor(gray, face)

        # --- Extract eye landmark points --------------------------------------
        left_eye_pts = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES],
            dtype=np.int32,
        )
        right_eye_pts = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES],
            dtype=np.int32,
        )
        result["left_eye_points"] = left_eye_pts
        result["right_eye_points"] = right_eye_pts

        # --- Compute EAR -----------------------------------------------------
        ear = compute_avg_ear(left_eye_pts, right_eye_pts)
        result["ear"] = ear

        # --- Extract eye patch for SVM classification -------------------------
        eye_patch = self._extract_eye_patch(gray, left_eye_pts, right_eye_pts)

        # --- Feature extraction + classification ------------------------------
        features = extract_all_features(eye_patch, ear_value=ear)
        if features is not None:
            features = features.reshape(1, -1)
            if self._scaler is not None:
                features = self._scaler.transform(features)
            prediction = self._model.predict(features)[0]
            is_closed = int(prediction) == 1  # 1 = closed, 0 = open
        else:
            # Fallback: cannot extract features
            is_closed = False

        # --- State machine update ---------------------------------------------
        blink_detected = self._update_state_machine(is_closed)
        result["eye_state"] = "closed" if is_closed else "open"
        result["blink_detected"] = blink_detected
        result["blink_count"] = self._blink_count

        # --- Closed duration --------------------------------------------------
        if is_closed and self._close_start_time > 0:
            result["closed_duration"] = time.time() - self._close_start_time
        else:
            result["closed_duration"] = 0.0

        return result

    def reset(self) -> None:
        """Reset all counters and state machine."""
        self._state = EyeState.OPEN
        self._blink_count = 0
        self._closed_frame_count = 0
        self._close_start_time = 0.0

    def get_stats(self) -> dict:
        """
        Return current detection statistics.

        Returns
        -------
        dict
            Keys: ``blink_count``, ``state``, ``closed_frame_count``.
        """
        return {
            "blink_count": self._blink_count,
            "state": self._state.name,
            "closed_frame_count": self._closed_frame_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_eye_patch(
        self,
        gray: np.ndarray,
        left_pts: np.ndarray,
        right_pts: np.ndarray,
    ) -> np.ndarray:
        """
        Crop, align, and resize an eye region patch from the grayscale frame.

        The eye is rotated so that the corner-to-corner line is horizontal,
        matching the alignment applied during training data extraction.
        Uses the left eye by default (consistent with training pipeline).
        """
        import math

        eye_pts = left_pts  # Use left eye for the patch
        h, w = gray.shape[:2]

        # Compute rotation angle from eye corners (p1=index 0, p4=index 3)
        p1 = eye_pts[0].astype(np.float64)
        p4 = eye_pts[3].astype(np.float64)
        dx = p4[0] - p1[0]
        dy = p4[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))

        # Center of the eye
        cx = float(eye_pts[:, 0].mean())
        cy = float(eye_pts[:, 1].mean())

        # Rotate frame around eye center
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        # Transform eye points to rotated coordinates
        ones = np.ones((6, 1))
        pts_hom = np.hstack([eye_pts.astype(np.float64), ones])
        rotated_pts = (M @ pts_hom.T).T

        # Crop with padding
        x_min, y_min = rotated_pts.min(axis=0)
        x_max, y_max = rotated_pts.max(axis=0)
        padding = 0.35
        pad_x = int((x_max - x_min) * padding)
        pad_y = int((y_max - y_min) * padding)

        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        eye_region = rotated[y1:y2, x1:x2]
        if eye_region.size == 0:
            return np.zeros(EYE_PATCH_SIZE[::-1], dtype=np.uint8)

        eye_patch = cv2.resize(eye_region, EYE_PATCH_SIZE)
        return eye_patch

    def _update_state_machine(self, is_closed: bool) -> bool:
        """
        Advance the blink-detection state machine.

        Parameters
        ----------
        is_closed : bool
            Whether the current frame's eyes are classified as closed.

        Returns
        -------
        bool
            True if a valid blink was completed on this frame.
        """
        blink_detected = False

        if self._state == EyeState.OPEN:
            if is_closed:
                self._state = EyeState.MAYBE_CLOSED
                self._closed_frame_count = 1
                self._close_start_time = time.time()

        elif self._state == EyeState.MAYBE_CLOSED:
            if is_closed:
                self._closed_frame_count += 1
                if self._closed_frame_count >= MIN_BLINK_FRAMES:
                    self._state = EyeState.CLOSED
            else:
                # Too few closed frames – not a real blink
                self._state = EyeState.OPEN
                self._closed_frame_count = 0
                self._close_start_time = 0.0

        elif self._state == EyeState.CLOSED:
            if is_closed:
                self._closed_frame_count += 1
                # Stay in CLOSED; drowsiness check is handled by
                # HealthMonitor via closed_duration.
            else:
                self._state = EyeState.MAYBE_OPEN
                # Don't reset closed_frame_count yet – need it for validation

        elif self._state == EyeState.MAYBE_OPEN:
            if not is_closed:
                # Eyes confirmed open → evaluate blink validity
                if self._closed_frame_count <= MAX_BLINK_FRAMES:
                    # Valid blink
                    self._blink_count += 1
                    blink_detected = True
                else:
                    # Too many closed frames → drowsiness, not a normal blink
                    logger.debug(
                        "Prolonged closure (%d frames) – not counted as blink.",
                        self._closed_frame_count,
                    )
                self._state = EyeState.OPEN
                self._closed_frame_count = 0
                self._close_start_time = 0.0
            else:
                # Eyes closed again – go back to CLOSED
                self._state = EyeState.CLOSED
                self._closed_frame_count += 1

        return blink_detected
