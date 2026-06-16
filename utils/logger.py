"""CSV logging helpers for fraud detection sessions."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "fraud_log.csv"
LOG_COLUMNS = ["timestamp", "risk_level", "score", "flags"]


def _normalize_log_path(log_path: str | Path) -> Path:
    """Resolve the log path to a Path object."""
    return Path(log_path)


def log_event(
    level: str,
    flags: Iterable[str],
    score: int,
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> None:
    """Append a fraud event to the CSV log."""
    path = _normalize_log_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp": timestamp,
        "risk_level": level,
        "score": int(score),
        "flags": ",".join(flags),
    }

    try:
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Failed to write fraud log at %s: %s", path, exc)


def load_log(log_path: str | Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    """Load the fraud log as a DataFrame."""
    path = _normalize_log_path(log_path)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Failed to read fraud log at %s: %s", path, exc)
        return pd.DataFrame(columns=LOG_COLUMNS)
