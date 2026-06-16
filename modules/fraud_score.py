"""Fraud scoring helpers for fraud detection."""

from __future__ import annotations

from typing import Any


def calculate_score(
    face_result: dict[str, Any],
    phone_result: dict[str, Any],
    pose_result: dict[str, Any],
) -> dict[str, Any]:
    """Combine module outputs into a fraud risk score."""
    score = 0
    flags: list[str] = []

    if phone_result.get("flag") == "phone_detected":
        score += 35
        flags.append("phone_detected")

    if face_result.get("flag") == "multiple":
        score += 30
        flags.append("multiple_faces")
    elif face_result.get("flag") == "absent":
        score += 25
        flags.append("face_absent")

    if pose_result.get("flag") == "looking_away":
        score += 15
        flags.append("looking_away")

    if score < 30:
        level = "LOW"
    elif score < 60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return {"score": score, "level": level, "flags": flags}
