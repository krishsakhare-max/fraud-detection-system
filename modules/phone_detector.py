"""Cell phone detection helpers for fraud detection."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    YOLO = None  # type: ignore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> Optional[Any]:
    """Load the YOLOv8 model once per process."""
    if YOLO is None:
        logger.debug("ultralytics is unavailable, phone detection disabled.")
        return None
    try:
        return YOLO("yolov8n.pt")
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Failed to load YOLO model: %s", exc)
        return None


def detect_phone(frame: np.ndarray) -> dict[str, Any]:
    """Detect a cell phone in the frame using YOLOv8."""
    model = _load_model()
    if model is None:
        return {"detected": False, "confidence": 0.0, "flag": "ok"}

    try:
        results = model.predict(
            source=frame,
            verbose=False,
            conf=0.20,
            imgsz=640,
            classes=[67],
            max_det=5,
        )
        result = results[0] if results else None
        if result is None or result.boxes is None:
            return {"detected": False, "confidence": 0.0, "flag": "ok"}

        names = result.names
        if isinstance(names, dict):
            phone_class_ids = {
                class_id
                for class_id, name in names.items()
                if str(name).strip().lower() in {"cell phone", "mobile phone", "phone"}
            }
        else:
            phone_class_ids = {
                index for index, name in enumerate(names) if str(name).strip().lower() in {"cell phone", "mobile phone", "phone"}
            }
        if not phone_class_ids:
            phone_class_ids = {67}

        best_confidence = 0.0
        detected = False
        class_ids = result.boxes.cls.tolist() if result.boxes.cls is not None else []
        confidences = result.boxes.conf.tolist() if result.boxes.conf is not None else []
        for class_id, confidence in zip(class_ids, confidences):
            if int(class_id) in phone_class_ids:
                detected = True
                best_confidence = max(best_confidence, float(confidence))

        flag = "phone_detected" if detected else "ok"
        return {
            "detected": detected,
            "confidence": float(best_confidence),
            "flag": flag,
        }
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Phone detection failed: %s", exc)
        return {"detected": False, "confidence": 0.0, "flag": "ok"}
