"""Face counting helpers for fraud detection."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _face_detector() -> Any:
    """Create and cache the Haar cascade detector once per process."""
    if cv2 is None:
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    return detector


def _haar_face_locations(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces with OpenCV Haar cascades."""
    detector = _face_detector()
    if cv2 is None or detector is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    locations: list[tuple[int, int, int, int]] = []
    for x, y, w, h in faces:
        locations.append((int(y), int(x + w), int(y + h), int(x)))
    return locations


def detect_faces(frame: np.ndarray) -> dict[str, Any]:
    """Detect faces and classify the frame state."""
    locations = _haar_face_locations(frame)
    count = len(locations)
    if count == 0:
        flag = "absent"
    elif count >= 2:
        flag = "multiple"
    else:
        flag = "ok"
    return {"count": count, "locations": locations, "flag": flag}
