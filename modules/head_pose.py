"""Head pose estimation helpers for fraud detection."""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None  # type: ignore

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    mp = None  # type: ignore

logger = logging.getLogger(__name__)


def mediapipe_available() -> bool:
    """Return whether mediapipe imported successfully."""
    return mp is not None


@lru_cache(maxsize=1)
def _face_mesh() -> Any:
    """Create a reusable MediaPipe FaceMesh instance."""
    if mp is None:
        logger.debug("mediapipe is unavailable, head pose detection disabled.")
        return None
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame into RGB for MediaPipe."""
    if cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame[:, :, ::-1]


def detect_head_pose(frame: np.ndarray) -> dict[str, Any]:
    """Estimate head yaw and pitch from facial landmarks."""
    mesh = _face_mesh()
    if mesh is None:
        return {"yaw": 0.0, "pitch": 0.0, "flag": "ok"}

    try:
        rgb_frame = _to_rgb(frame)
        result = mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return {"yaw": 0.0, "pitch": 0.0, "flag": "ok"}

        landmarks = result.multi_face_landmarks[0].landmark
        left_eye = np.array([landmarks[33].x, landmarks[33].y], dtype=float)
        right_eye = np.array([landmarks[263].x, landmarks[263].y], dtype=float)
        nose_tip = np.array([landmarks[1].x, landmarks[1].y], dtype=float)
        eyes_center = (left_eye + right_eye) / 2.0
        eye_distance = float(np.linalg.norm(right_eye - left_eye)) or 1e-6

        yaw = math.degrees(math.atan2(nose_tip[0] - eyes_center[0], eye_distance / 2.0))
        pitch = math.degrees(math.atan2(eyes_center[1] - nose_tip[1], eye_distance / 2.0))
        flag = "looking_away" if abs(yaw) > 20.0 else "ok"
        return {"yaw": float(yaw), "pitch": float(pitch), "flag": flag}
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Head pose estimation failed: %s", exc)
        return {"yaw": 0.0, "pitch": 0.0, "flag": "ok"}
