"""Optional face verification compatibility layer.

This project keeps face verification as a lightweight plug-in point so the
repository stays runnable in environments where the heavier identity
recognition stack is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

face_recognition_available = False


def load_registered_faces(directory: str | Path | None = None) -> list[dict[str, Any]]:
    """Return registered-face metadata.

    The current submission does not ship a trained face-verification model.
    This helper keeps the public interface stable for future extension.
    """
    _ = directory
    return []


def verify_face(
    frame: Any,
    registered_faces: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a disabled verification result.

    The dashboard uses the returned structure only for compatibility, so the
    function intentionally reports a safe fallback instead of raising.
    """
    _ = frame, registered_faces
    return {
        "verified": False,
        "matched_name": None,
        "confidence": 0.0,
        "flag": "disabled",
        "reason": "Face verification is disabled in this lightweight submission.",
    }
