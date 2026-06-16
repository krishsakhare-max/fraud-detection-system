"""Streamlit app for fraud detection sessions."""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None  # type: ignore

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from modules.face_detection import detect_faces
from modules.fraud_score import calculate_score
from modules.head_pose import detect_head_pose, mediapipe_available
from modules.phone_detector import detect_phone
from utils.logger import DEFAULT_LOG_PATH, load_log, log_event

LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LIVE_UI: dict[str, object | None] = {
    "frame_slot": None,
    "risk_slot": None,
    "alert_slot": None,
    "log_slot": None,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _init_state() -> None:
    """Initialize Streamlit session state keys once."""
    defaults = {
        "session_active": False,
        "monitoring_source": "webcam",
        "monitoring_capture": None,
        "monitoring_video_path": None,
        "monitoring_video_name": None,
        "monitoring_frame_count": 0,
        "frame_read_failures": 0,
        "latest_log_df": pd.DataFrame(columns=["timestamp", "risk_level", "score", "flags"]),
        "risk_durations": {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0},
        "current_level": "LOW",
        "last_tick": None,
        "session_events": 0,
        "session_flags": [],
        "stop_requested": False,
        "last_frame_rgb": None,
        "analysis_stride": 3,
        "analysis_counter": 0,
        "last_face_result": {"count": 0, "locations": [], "flag": "absent"},
        "last_phone_result": {"detected": False, "confidence": 0.0, "flag": "ok"},
        "last_pose_result": {"yaw": 0.0, "pitch": 0.0, "flag": "ok"},
        "last_score_result": {"score": 0, "level": "LOW", "flags": []},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _save_uploaded_video(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Persist the uploaded monitoring video so OpenCV can stream it."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    target = APP_DIR / f"_uploaded_video{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _normalize_frame(frame: np.ndarray, target_size: tuple[int, int] = (960, 540)) -> np.ndarray:
    """Resize frames into a fixed 16:9 canvas to reduce layout jumping."""
    if cv2 is None:
        return frame

    target_width, target_height = target_size
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return frame

    scale = min(target_width / float(width), target_height / float(height))
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_height, target_width, 3), dtype=resized.dtype)
    y_offset = max(0, (target_height - resized_height) // 2)
    x_offset = max(0, (target_width - resized_width) // 2)
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return canvas


def _open_capture(video_path: str | None = None, prefer_webcam: bool = True) -> Optional[cv2.VideoCapture]:
    """Open the selected video source with a safe fallback order."""
    if cv2 is None:
        logger.error("OpenCV is unavailable, so live monitoring cannot start.")
        return None

    webcam_backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        webcam_backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        webcam_backends.append(cv2.CAP_MSMF)
    webcam_backends.append(getattr(cv2, "CAP_ANY", 0))

    if prefer_webcam:
        for backend in webcam_backends:
            webcam_capture = cv2.VideoCapture(0, backend)
            if webcam_capture.isOpened():
                webcam_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                return webcam_capture
            webcam_capture.release()

    if video_path:
        for backend in [getattr(cv2, "CAP_ANY", 0)]:
            file_capture = cv2.VideoCapture(str(Path(video_path).expanduser()), backend)
            if file_capture.isOpened():
                return file_capture
            file_capture.release()

    if not prefer_webcam:
        for backend in webcam_backends:
            webcam_capture = cv2.VideoCapture(0, backend)
            if webcam_capture.isOpened():
                webcam_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                return webcam_capture
            webcam_capture.release()

    return None


def _release_capture() -> None:
    """Release and clear any active OpenCV capture object."""
    capture = st.session_state.get("monitoring_capture")
    if capture is not None:
        try:
            capture.release()
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception("Failed to release capture.")
    st.session_state["monitoring_capture"] = None


def _request_rerun() -> None:
    """Trigger a Streamlit rerun across Streamlit versions."""
    rerun = getattr(st, "rerun", None)
    if rerun is None:
        rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


def _safe_call(func: object, *args: object, **kwargs: object) -> object:
    """Call a module function without letting one failure stop the loop."""
    try:
        return func(*args, **kwargs)  # type: ignore[misc]
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Module call failed in %s: %s", getattr(func, "__name__", "unknown"), exc)
        return None


def _draw_annotation(frame: np.ndarray, face_result: dict, score_result: dict) -> np.ndarray:
    """Draw the main fraud indicator box and label on the frame."""
    if cv2 is None:
        return frame

    annotated = frame.copy()
    height, width = annotated.shape[:2]
    color_map = {"LOW": (0, 200, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}
    color = color_map.get(score_result.get("level", "LOW"), (0, 200, 0))
    label = f'{score_result.get("level", "LOW")} | Score {score_result.get("score", 0)}'

    if face_result and face_result.get("locations"):
        top, right, bottom, left = face_result["locations"][0]
        top = max(0, top)
        left = max(0, left)
        bottom = min(height - 1, bottom)
        right = min(width - 1, right)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 3)
        cv2.putText(annotated, label, (left, max(25, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.rectangle(annotated, (8, 8), (width - 8, height - 8), color, 3)
        cv2.putText(annotated, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return annotated


def _render_alerts(score_result: dict[str, object]) -> None:
    """Render active fraud flags with severity-aware alerts."""
    flags = score_result.get("flags", [])
    if not flags:
        st.success("No active fraud flags right now.")
        return

    for flag in flags:
        message = str(flag).replace("_", " ").title()
        if score_result.get("level") == "HIGH":
            st.error(message)
        elif score_result.get("level") == "MEDIUM":
            st.warning(message)
        else:
            st.info(message)


def _render_live_status(
    face_result: dict[str, object],
    phone_result: dict[str, object],
    pose_result: dict[str, object],
) -> None:
    """Show plain-English detector status for the current frame."""
    face_flag = str(face_result.get("flag", "unknown"))
    phone_flag = str(phone_result.get("flag", "unknown"))
    pose_flag = str(pose_result.get("flag", "unknown"))

    if face_flag == "absent":
        st.error("No face detected in the frame.")
    elif face_flag == "multiple":
        st.error("Multiple faces detected. Only one face should be visible.")
    else:
        st.success("Exactly one face detected.")

    if phone_flag == "phone_detected":
        st.error("Cell phone detected in the frame.")
    else:
        st.success("No cell phone detected.")
    st.caption(f"Phone confidence: {float(phone_result.get('confidence', 0.0)):.2f}")

    if pose_flag == "looking_away":
        st.warning("The person appears to be looking away.")
    else:
        st.success("Head pose looks normal.")


def _summarize_session(log_df: pd.DataFrame) -> tuple[int, dict[str, float], str]:
    """Summarize the current session for the bottom-of-page report."""
    total_events = int(st.session_state.get("session_events", 0))
    durations = st.session_state.get("risk_durations", {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0})

    flag_counter: Counter[str] = Counter(str(flag).strip() for flag in st.session_state.get("session_flags", []) if str(flag).strip())
    if not flag_counter and not log_df.empty and "flags" in log_df.columns:
        for raw_flags in log_df["flags"].fillna(""):
            for flag in str(raw_flags).split(","):
                cleaned = flag.strip()
                if cleaned:
                    flag_counter[cleaned] += 1

    most_common_flag = flag_counter.most_common(1)[0][0] if flag_counter else "none"
    return total_events, durations, most_common_flag


def _render_live_workspace() -> None:
    """Render and refresh only the live monitoring area."""
    frame_slot = LIVE_UI.get("frame_slot")
    risk_slot = LIVE_UI.get("risk_slot")
    alert_slot = LIVE_UI.get("alert_slot")
    log_slot = LIVE_UI.get("log_slot")
    if frame_slot is None or risk_slot is None or alert_slot is None or log_slot is None:
        return

    if cv2 is None:
        frame_slot.warning("OpenCV is missing, so the app can load but live monitoring is disabled.")
        return

    if not st.session_state.get("session_active"):
        last_frame_rgb = st.session_state.get("last_frame_rgb")
        frame_slot.info("Select a source in the sidebar and press Start Session.")
        risk_slot.metric("Current Risk", "IDLE", delta="Waiting")
        if last_frame_rgb is not None:
            frame_slot.image(last_frame_rgb, channels="RGB", use_container_width=True)
        with alert_slot.container():
            st.markdown('<div class="panel"><div class="panel-header"><div><div class="panel-title">Live Alerts</div><div class="panel-subtitle">Waiting for a stream.</div></div></div></div>', unsafe_allow_html=True)
            st.caption("No stream is active yet.")
        with log_slot.container():
            st.markdown('<div class="panel"><div class="panel-header"><div><div class="panel-title">Fraud Event Log</div><div class="panel-subtitle">Latest medium and high risk events.</div></div></div></div>', unsafe_allow_html=True)
            st.dataframe(st.session_state["latest_log_df"], use_container_width=True, height=320)
        return

    capture = st.session_state.get("monitoring_capture")
    if capture is None:
        st.session_state["monitoring_capture"] = _open_capture(
            video_path=str(st.session_state.get("monitoring_video_path")) if st.session_state.get("monitoring_video_path") else None,
            prefer_webcam=st.session_state.get("monitoring_source") == "webcam",
        )
        capture = st.session_state.get("monitoring_capture")

    if capture is None:
        frame_slot.error("Could not open the selected video source.")
        st.session_state["session_active"] = False
        return

    ok, frame = capture.read()
    if not ok:
        st.session_state["frame_read_failures"] += 1
        if st.session_state.get("last_frame_rgb") is not None:
            frame_slot.image(st.session_state["last_frame_rgb"], channels="RGB", use_container_width=True)
        else:
            frame_slot.info("Waiting for the camera feed...")

        if st.session_state.get("monitoring_source") == "video" and cv2 is not None:
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                st.session_state["frame_read_failures"] = 0
                return
            except Exception:
                pass

        if st.session_state.get("frame_read_failures", 0) < 15:
            risk_slot.metric(
                "Current Risk",
                st.session_state["current_level"],
                delta=f'Retrying frame {st.session_state["frame_read_failures"]}/15',
            )
            return

        _release_capture()
        st.session_state["session_active"] = False
        frame_slot.warning("Video stream ended or the camera stopped responding.")
        return

    st.session_state["frame_read_failures"] = 0
    st.session_state["analysis_counter"] += 1
    now = time.perf_counter()
    previous_level = st.session_state["current_level"]
    last_tick = st.session_state["last_tick"]
    if last_tick is not None:
        st.session_state["risk_durations"][previous_level] += now - last_tick
    st.session_state["last_tick"] = now

    display_frame = _normalize_frame(frame, target_size=(960, 540))
    should_analyze = st.session_state["analysis_counter"] % int(st.session_state.get("analysis_stride", 3)) == 1
    if should_analyze or st.session_state.get("last_score_result") is None:
        analysis_frame = _normalize_frame(frame, target_size=(720, 405))
        face_result = _safe_call(detect_faces, analysis_frame) or {"count": 0, "locations": [], "flag": "absent"}
        phone_result = _safe_call(detect_phone, analysis_frame) or {
            "detected": False,
            "confidence": 0.0,
            "flag": "ok",
        }
        pose_result = _safe_call(detect_head_pose, analysis_frame) or {
            "yaw": 0.0,
            "pitch": 0.0,
            "flag": "ok",
        }
        score_result = _safe_call(
            calculate_score,
            face_result,
            phone_result,
            pose_result,
        ) or {"score": 0, "level": "LOW", "flags": []}
        st.session_state["last_face_result"] = face_result
        st.session_state["last_phone_result"] = phone_result
        st.session_state["last_pose_result"] = pose_result
        st.session_state["last_score_result"] = score_result
    else:
        face_result = st.session_state.get("last_face_result", {"count": 0, "locations": [], "flag": "absent"})
        phone_result = st.session_state.get("last_phone_result", {"detected": False, "confidence": 0.0, "flag": "ok"})
        pose_result = st.session_state.get("last_pose_result", {"yaw": 0.0, "pitch": 0.0, "flag": "ok"})
        score_result = st.session_state.get("last_score_result", {"score": 0, "level": "LOW", "flags": []})

    st.session_state["current_level"] = str(score_result.get("level", "LOW"))
    if should_analyze and st.session_state["current_level"] in {"MEDIUM", "HIGH"}:
        log_event(
            st.session_state["current_level"],
            score_result.get("flags", []),
            int(score_result.get("score", 0)),
            log_path=DEFAULT_LOG_PATH,
        )
        st.session_state["session_events"] += 1
        st.session_state["session_flags"].extend(score_result.get("flags", []))

    annotated = _draw_annotation(display_frame, face_result, score_result)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.session_state["last_frame_rgb"] = annotated_rgb
    st.session_state["monitoring_frame_count"] += 1

    frame_slot.image(annotated_rgb, channels="RGB", use_container_width=True)
    risk_slot.metric(
        "Current Risk",
        st.session_state["current_level"],
        delta=f'Score {score_result.get("score", 0)}',
    )

    with alert_slot.container():
        st.markdown('<div class="panel"><div class="panel-header"><div><div class="panel-title">Live Alerts</div><div class="panel-subtitle">Updated on each analyzed frame.</div></div></div></div>', unsafe_allow_html=True)
        st.write("Active detectors: face detection, phone detection, head pose")
        _render_live_status(face_result, phone_result, pose_result)
        _render_alerts(score_result)

    if should_analyze and (
        st.session_state["monitoring_frame_count"] % 6 == 0 or st.session_state["current_level"] in {"MEDIUM", "HIGH"}
    ):
        st.session_state["latest_log_df"] = load_log(DEFAULT_LOG_PATH).tail(100)

    with log_slot.container():
        st.markdown('<div class="panel"><div class="panel-header"><div><div class="panel-title">Fraud Event Log</div><div class="panel-subtitle">Latest medium and high risk events.</div></div></div></div>', unsafe_allow_html=True)
        st.dataframe(st.session_state["latest_log_df"], use_container_width=True, height=320)

    if st.session_state["monitoring_frame_count"] >= 300:
        _release_capture()
        st.session_state["session_active"] = False
        frame_slot.info("Monitoring stopped after reaching the demo frame limit.")


def main() -> None:
    """Run the fraud detection Streamlit application."""
    st.set_page_config(page_title="AI Fraud Detection", layout="wide")
    _init_state()
    st.title("Fraud Watch Console")
    st.caption("A live ops view for camera or uploaded video.")

    st.markdown(
        """
        <style>
            :root {
                --bg: #070b14;
                --surface: rgba(10, 16, 28, 0.86);
                --surface-strong: rgba(13, 20, 34, 0.96);
                --text: #edf2ff;
                --muted: #a3b0c8;
                --line: rgba(149, 170, 220, 0.14);
                --accent: #6f8cff;
                --accent-soft: rgba(111, 140, 255, 0.10);
                --warning: #ffbf66;
                --danger: #ff6f6f;
                --success: #6de08f;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(111, 140, 255, 0.16), transparent 28%),
                    radial-gradient(circle at top right, rgba(109, 224, 143, 0.08), transparent 24%),
                    linear-gradient(180deg, #070b14 0%, #0b1220 52%, #070b14 100%);
                color: var(--text);
            }
            html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"] {
                background: transparent !important;
            }
            [data-testid="stSidebar"] {
                background: rgba(4, 8, 16, 0.92) !important;
                border-right: 1px solid rgba(149, 170, 220, 0.11);
            }
            .block-container {
                padding-top: 1.0rem;
                padding-bottom: 1.8rem;
                max-width: 1320px;
            }
            .monitor-shell {
                padding: 1.05rem 1.15rem;
                border-radius: 20px;
                background: var(--surface);
                border: 1px solid var(--line);
                box-shadow: 0 16px 50px rgba(0, 0, 0, 0.35);
                margin-bottom: 1rem;
            }
            .kicker {
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                color: #8da7ff;
                margin-bottom: 0.35rem;
            }
            .hero-title {
                font-size: 1.95rem;
                font-weight: 760;
                color: var(--text);
                margin: 0;
            }
            .hero-subtitle {
                color: var(--muted);
                font-size: 0.96rem;
                line-height: 1.55;
                margin-top: 0.35rem;
            }
            .status-chip {
                display: inline-block;
                padding: 0.28rem 0.68rem;
                border-radius: 999px;
                font-size: 0.78rem;
                margin-right: 0.35rem;
                margin-top: 0.25rem;
                border: 1px solid var(--line);
                background: rgba(255,255,255,0.04);
                color: var(--text);
            }
            .good { background: rgba(109, 224, 143, 0.12); color: var(--success); }
            .warn { background: rgba(255, 191, 102, 0.12); color: var(--warning); }
            .bad { background: rgba(255, 111, 111, 0.12); color: var(--danger); }
            .section-card {
                padding: 1rem 1rem 0.9rem 1rem;
                border-radius: 18px;
                background: var(--surface);
                border: 1px solid var(--line);
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.24);
                margin-bottom: 1rem;
            }
            .section-label {
                font-size: 0.76rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--muted);
                margin-bottom: 0.45rem;
            }
            .section-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.2rem;
            }
            .section-copy {
                font-size: 0.92rem;
                color: var(--muted);
                line-height: 1.45;
            }
            .panel {
                padding: 0.9rem;
                border-radius: 16px;
                background: var(--surface-strong);
                border: 1px solid var(--line);
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
            }
            .panel-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                margin-bottom: 0.75rem;
            }
            .panel-title {
                font-size: 1rem;
                font-weight: 700;
                color: var(--text);
            }
            .panel-subtitle {
                font-size: 0.86rem;
                color: var(--muted);
            }
            div[data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                padding: 0.8rem 0.95rem;
                border-radius: 16px;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
            }
            div[data-testid="stMetric"] label {
                color: var(--muted);
            }
            div[data-testid="stMetric"] div {
                color: var(--text);
            }
            .stMarkdown, .stCaption, .stText, .stAlert, .stInfo, .stSuccess, .stWarning, .stError, p, label {
                color: var(--text) !important;
            }
            button[kind="primary"] {
                background: linear-gradient(135deg, #7e97ff 0%, #546bff 100%) !important;
                border: 1px solid rgba(126, 151, 255, 0.52) !important;
                color: #fff !important;
                box-shadow: 0 10px 24px rgba(84, 107, 255, 0.28);
            }
            button[kind="secondary"] {
                background: rgba(255,255,255,0.06) !important;
                border: 1px solid var(--line) !important;
                color: var(--text) !important;
            }
            [data-testid="stFileUploaderDropzone"] {
                background: rgba(255,255,255,0.04) !important;
                border: 1px solid var(--line) !important;
                border-radius: 16px !important;
            }
            [data-testid="stRadio"] {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 14px;
                padding: 0.65rem 0.75rem;
                margin-bottom: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="monitor-shell">
            <div class="kicker">Ops desk</div>
            <div class="hero-title">Live Review Console</div>
            <div class="hero-subtitle">
                One fixed workspace for live camera review, phone alerts, and looking-away detection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Session Control")
        st.caption("Choose a source, then start the stream.")
        verification_photo = st.file_uploader(
            "Preview image (dummy)",
            type=["png", "jpg", "jpeg", "webp"],
            help="This is just a demo upload box and is not used by the live monitor.",
        )
        monitoring_source = st.radio("Monitoring source", ["Webcam", "Uploaded video"], horizontal=False)
        uploaded_video = None
        if monitoring_source == "Uploaded video":
            uploaded_video = st.file_uploader("Monitoring video", type=["mp4", "mov", "avi", "mkv"])
        start_clicked = st.button("Start Session", type="primary", use_container_width=True)
        stop_clicked = st.button("Stop Session", use_container_width=True)
        clear_clicked = st.button("Clear Log", use_container_width=True)
        st.divider()
        st.subheader("System Health")
        cv2_status = cv2 is not None
        media_status = mediapipe_available()
        st.markdown(
            f"""
            <span class="status-chip {'good' if cv2_status else 'bad'}">
                OpenCV: {'Ready' if cv2_status else 'Not installed'}
            </span>
            <span class="status-chip {'good' if media_status else 'warn'}">
                MediaPipe: {'Ready' if media_status else 'Not installed'}
            </span>
            """,
            unsafe_allow_html=True,
        )
        if not media_status:
            st.info("Head pose detection is disabled until `mediapipe` is available.")
        if verification_photo is not None:
            st.caption(f"Preview loaded: {verification_photo.name}")

    if clear_clicked:
        _release_capture()
        if DEFAULT_LOG_PATH.exists():
            DEFAULT_LOG_PATH.unlink()
        st.session_state["risk_durations"] = {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0}
        st.session_state["session_events"] = 0
        st.session_state["session_flags"] = []
        st.session_state["session_active"] = False
        st.session_state["stop_requested"] = False
        st.session_state["monitoring_frame_count"] = 0
        st.session_state["frame_read_failures"] = 0
        st.session_state["analysis_counter"] = 0
        st.session_state["last_frame_rgb"] = None
        st.session_state["last_face_result"] = {"count": 0, "locations": [], "flag": "absent"}
        st.session_state["last_phone_result"] = {"detected": False, "confidence": 0.0, "flag": "ok"}
        st.session_state["last_pose_result"] = {"yaw": 0.0, "pitch": 0.0, "flag": "ok"}
        st.session_state["last_score_result"] = {"score": 0, "level": "LOW", "flags": []}
        st.session_state["latest_log_df"] = pd.DataFrame(columns=["timestamp", "risk_level", "score", "flags"])
        st.session_state["monitoring_video_name"] = None
        st.success("Fraud log cleared.")

    if stop_clicked:
        st.session_state["stop_requested"] = True
        st.info("Stop requested. The current session will end after the next frame.")

    st.session_state["monitoring_source"] = "webcam" if monitoring_source == "Webcam" else "video"
    if uploaded_video is not None and uploaded_video.name != st.session_state.get("monitoring_video_name"):
        video_path = _save_uploaded_video(uploaded_video)
        st.session_state["monitoring_video_path"] = str(video_path)
        st.session_state["monitoring_video_name"] = uploaded_video.name

    source_video_path = st.session_state.get("monitoring_video_path")

    if start_clicked:
        if monitoring_source == "Uploaded video" and not source_video_path:
            st.warning("Please upload a monitoring video before starting a video session.")
        elif cv2 is None:
            st.error("OpenCV is not installed in this Python environment, so live monitoring cannot run.")
        else:
            _release_capture()
            st.session_state["session_active"] = True
            st.session_state["stop_requested"] = False
            st.session_state["monitoring_frame_count"] = 0
            st.session_state["frame_read_failures"] = 0
            st.session_state["analysis_counter"] = 0
            st.session_state["last_frame_rgb"] = None
            st.session_state["last_face_result"] = {"count": 0, "locations": [], "flag": "absent"}
            st.session_state["last_phone_result"] = {"detected": False, "confidence": 0.0, "flag": "ok"}
            st.session_state["last_pose_result"] = {"yaw": 0.0, "pitch": 0.0, "flag": "ok"}
            st.session_state["last_score_result"] = {"score": 0, "level": "LOW", "flags": []}
            st.session_state["current_level"] = "LOW"
            st.session_state["last_tick"] = None
            st.session_state["monitoring_capture"] = _open_capture(
                video_path=str(source_video_path) if source_video_path else None,
                prefer_webcam=monitoring_source == "Webcam",
            )
            if st.session_state["monitoring_capture"] is None:
                st.error("Could not open the selected video source.")
                st.session_state["session_active"] = False
            else:
                st.success("Monitoring started.")
    if st.session_state.get("session_active") and st.session_state.get("stop_requested"):
        _release_capture()
        st.session_state["session_active"] = False
        st.session_state["stop_requested"] = False
        st.info("Monitoring stopped.")

    left_col, right_col = st.columns([1.2, 1.0], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-label">Live feed</div>
                <div class="section-title">Monitoring window</div>
                <div class="section-copy">This panel keeps a fixed canvas so the preview does not jump around.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if LIVE_UI["frame_slot"] is None:
            LIVE_UI["frame_slot"] = st.empty()
        if LIVE_UI["risk_slot"] is None:
            LIVE_UI["risk_slot"] = st.empty()
        st.caption("If the feed stays empty, check camera permissions or verify that your video file can play in OpenCV.")

    with right_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-label">Signals</div>
                <div class="section-title">Detector output</div>
                <div class="section-copy">Alerts and logs stay pinned beside the stream so the page feels anchored.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if LIVE_UI["alert_slot"] is None:
            LIVE_UI["alert_slot"] = st.empty()
        if LIVE_UI["log_slot"] is None:
            LIVE_UI["log_slot"] = st.empty()

    _render_live_workspace()

    with st.container():
        st.markdown(
            """
            <div class="section-card">
                <div class="section-label">Summary</div>
                <div class="section-title">Session overview</div>
                <div class="section-copy">A quick readout of event volume and time spent at each risk level.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        log_df = load_log(DEFAULT_LOG_PATH)
        total_events, durations, most_common_flag = _summarize_session(log_df)
        summary_cols = st.columns(4)
        summary_cols[0].metric("Logged Events", total_events)
        summary_cols[1].metric("Low Risk", f"{durations.get('LOW', 0.0):.1f}s")
        summary_cols[2].metric("Medium Risk", f"{durations.get('MEDIUM', 0.0):.1f}s")
        summary_cols[3].metric("High Risk", f"{durations.get('HIGH', 0.0):.1f}s")
        st.info(f"Most common flag: {most_common_flag}")

    if st.session_state.get("session_active") and not st.session_state.get("stop_requested"):
        time.sleep(0.18)
        _request_rerun()


if __name__ == "__main__":
    main()
