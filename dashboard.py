import html
import sqlite3
from datetime import date
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from backend.database import get_all_events
from backend.video_monitor import VideoEventProcessor


ROOT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = ROOT_DIR / "videos" / "people-walking.mp4"
MODEL_PATH = ROOT_DIR / "yolo11n.pt"
TRACKER_PATH = ROOT_DIR / "bytetrack_custom.yaml"
SEVERITY_ORDER = ["CRITICAL", "HIGH", "LOW"]
SEVERITY_COLORS = {
    "CRITICAL": "#fb7185",
    "HIGH": "#fbbf24",
    "LOW": "#34d399",
}
EVENT_COLUMNS = [
    "id",
    "event_type",
    "location",
    "severity",
    "timestamp",
    "description",
]

st.set_page_config(
    page_title="Campus Watch | Safety Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --canvas: #0b1220;
        --sidebar: #101a2b;
        --card: #17243a;
        --card-soft: #111d30;
        --border: rgba(154, 171, 192, 0.16);
        --text: #f4f8ff;
        --muted: #9aabc0;
        --accent: #38bdf8;
        --critical: #fb7185;
        --high: #fbbf24;
        --normal: #34d399;
    }
    .stApp {
        background: var(--canvas);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(11, 18, 32, 0.9); }
    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #c4d2e3;
        padding: 0.1rem 0;
    }
    h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.025em; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        min-height: 112px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--text); }
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    div[data-testid="stAlert"] { border-radius: 12px; }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stExpander"] {
        background: var(--card-soft);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 0.55rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.1rem 0 1.4rem;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 13px;
        font-size: 1.35rem;
    }
    .brand-name {
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        line-height: 1.15;
    }
    .brand-subtitle {
        color: var(--muted);
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }
    .eyebrow {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 0 0 0.35rem;
    }
    .section-note {
        color: var(--muted);
        margin: -0.45rem 0 1.25rem;
    }
    .incident-card {
        background: var(--card-soft);
        border: 1px solid var(--border);
        border-left: 3px solid var(--incident-color, var(--accent));
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0;
    }
    .incident-top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: center;
        color: var(--muted);
        font-size: 0.78rem;
    }
    .incident-title {
        color: var(--text);
        font-size: 0.98rem;
        font-weight: 650;
        margin: 0.42rem 0 0.3rem;
    }
    .incident-detail { color: var(--muted); font-size: 0.86rem; }
    .severity-pill {
        display: inline-block;
        border: 1px solid currentColor;
        border-radius: 999px;
        color: var(--incident-color, var(--accent));
        font-size: 0.69rem;
        font-weight: 750;
        letter-spacing: 0.07em;
        padding: 0.2rem 0.55rem;
        white-space: nowrap;
    }
    .status-line {
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 1.2rem;
        padding-top: 0.9rem;
    }
    .status-dot {
        color: var(--normal);
        font-size: 1rem;
        padding-right: 0.25rem;
    }
    .info-panel {
        height: 100%;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }
    .info-label {
        color: var(--muted);
        font-size: 0.77rem;
        margin-bottom: 0.25rem;
    }
    .info-value { color: var(--text); font-weight: 650; }
    @media (max-width: 700px) {
        .incident-top { align-items: flex-start; }
        [data-testid="stMetric"] { min-height: 96px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=15, show_spinner=False)
def load_events() -> tuple[list[dict], str | None]:
    try:
        return get_all_events(), None
    except sqlite3.Error as exc:
        return [], f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner="Loading YOLO11n detection model...")
def load_detection_model() -> YOLO:
    return YOLO(str(MODEL_PATH))


@st.cache_data(show_spinner=False)
def read_video_metadata(path: str) -> dict[str, float | int | bool]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        return {"available": False, "width": 0, "height": 0, "fps": 0.0, "frames": 0}
    metadata = {
        "available": True,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return metadata


def build_event_frame(events: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    for column in EVENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    frame = frame[EVENT_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in ("event_type", "location", "severity", "description"):
        frame[column] = frame[column].fillna("").astype(str)
    frame["severity"] = frame["severity"].str.upper()
    return frame


def format_timestamp(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "Timestamp unavailable"
    return timestamp.strftime("%d %b %Y · %I:%M:%S %p")


def severity_color(severity: str) -> str:
    return SEVERITY_COLORS.get(severity.upper(), "#9aabc0")


def render_incident_card(event: pd.Series) -> None:
    severity = str(event["severity"]).upper()
    color = severity_color(severity)
    event_id = html.escape(str(event["id"]))
    event_type = html.escape(str(event["event_type"]).replace("_", " ").title())
    location = html.escape(str(event["location"]))
    timestamp = html.escape(format_timestamp(event["timestamp"]))
    description = html.escape(str(event["description"]))
    st.markdown(
        f"""
        <div class="incident-card" style="--incident-color:{color}">
            <div class="incident-top">
                <span>EVENT #{event_id} · {timestamp}</span>
                <span class="severity-pill">{html.escape(severity or "UNKNOWN")}</span>
            </div>
            <div class="incident-title">{event_type}</div>
            <div class="incident-detail">{location} &nbsp;·&nbsp; {description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(database_ok: bool) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark">🛡️</div>
                <div>
                    <div class="brand-name">CAMPUS WATCH</div>
                    <div class="brand-subtitle">SAFETY OPERATIONS CENTER</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        section = st.radio(
            "COMMAND MENU",
            [
                "Overview",
                "Live Feed",
                "Detections",
                "Sensors",
                "Alerts",
                "Statistics",
                "History",
                "Severity Analysis",
            ],
            label_visibility="visible",
        )
        db_status = "Database connected" if database_ok else "Database unavailable"
        video_status = "Demo video ready" if VIDEO_PATH.is_file() else "Demo video missing"
        status_color = "#34d399" if database_ok and VIDEO_PATH.is_file() else "#fbbf24"
        st.markdown(
            f"""
            <div class="status-line">
                <div style="color:{status_color};font-weight:700;margin-bottom:.3rem">
                    ● SYSTEM { "OPERATIONAL" if database_ok and VIDEO_PATH.is_file() else "CHECK REQUIRED" }
                </div>
                <div>{html.escape(db_status)}</div>
                <div>{html.escape(video_status)}</div>
                <div style="margin-top:.55rem">Prototype · prerecorded video</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return section


def render_page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<p class="eyebrow">{html.escape(kicker)}</p>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="section-note">{html.escape(description)}</p>', unsafe_allow_html=True)


def render_event_charts(frame: pd.DataFrame, include_trend: bool = True) -> None:
    if frame.empty:
        st.info("Charts will appear when incidents are available in SQLite.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("#### Severity distribution")
        severity_counts = (
            frame["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
            .rename_axis("Severity").reset_index(name="Incidents")
        )
        st.bar_chart(
            severity_counts,
            x="Severity",
            y="Incidents",
            color="#38bdf8",
            height=270,
        )
    with right:
        st.markdown("#### Events by location")
        locations = (
            frame["location"].replace("", "Unspecified").value_counts().head(8)
            .rename_axis("Location").reset_index(name="Incidents")
        )
        st.bar_chart(
            locations,
            x="Incidents",
            y="Location",
            color="#34d399",
            horizontal=True,
            height=270,
        )

    if include_trend:
        valid = frame.dropna(subset=["timestamp"])
        if not valid.empty:
            st.markdown("#### Daily incident trend")
            trend = valid.set_index("timestamp").resample("D").size()
            trend = trend.tail(30).rename("Incidents").rename_axis("Date").reset_index()
            st.line_chart(trend, x="Date", y="Incidents", color="#38bdf8", height=270)


def render_overview(frame: pd.DataFrame, database_error: str | None) -> None:
    render_page_header(
        "Campus security operations",
        "Operations overview",
        "A unified view of recorded campus safety incidents and monitoring readiness.",
    )
    if database_error:
        st.error(f"Could not retrieve incidents from SQLite: {database_error}")

    critical_count = int((frame["severity"] == "CRITICAL").sum())
    high_count = int((frame["severity"] == "HIGH").sum())
    affected_locations = int(frame.loc[frame["location"] != "", "location"].nunique())
    metrics = st.columns(4)
    metrics[0].metric("Total safety events", f"{len(frame):,}")
    metrics[1].metric("Critical incidents", f"{critical_count:,}")
    metrics[2].metric("High-priority incidents", f"{high_count:,}")
    metrics[3].metric("Affected locations", f"{affected_locations:,}")

    st.markdown("")
    chart_col, feed_col = st.columns([1.45, 1])
    with chart_col:
        st.markdown("### Incident picture")
        render_event_charts(frame)
    with feed_col:
        st.markdown("### Recent incidents")
        if frame.empty:
            st.info("No incidents have been recorded.")
        else:
            recent = frame.sort_values("timestamp", ascending=False, na_position="last").head(5)
            for _, event in recent.iterrows():
                render_incident_card(event)

    most_common = (
        frame.loc[frame["event_type"] != "", "event_type"].value_counts().index[0]
        if not frame.loc[frame["event_type"] != "", "event_type"].empty
        else "—"
    )
    st.markdown("### Monitoring readiness")
    status_left, status_right = st.columns(2)
    with status_left:
        st.markdown(
            f"""
            <div class="info-panel">
                <div class="info-label">Most common recorded event</div>
                <div class="info-value">{html.escape(str(most_common).replace("_", " ").title())}</div>
                <div class="info-label" style="margin-top:.8rem">
                    Calculated from {len(frame):,} SQLite event records
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_right:
        db_text = "Connected · project SQLite database" if not database_error else "Unavailable · check database error above"
        video_text = "Prerecorded demonstration source · no physical CCTV connected"
        st.markdown(
            f"""
            <div class="info-panel">
                <div class="info-label">System status</div>
                <div class="info-value"><span class="status-dot">●</span>{html.escape(db_text)}</div>
                <div class="info-label" style="margin-top:.8rem">{html.escape(video_text)}</div>
                <div class="info-label" style="margin-top:.35rem">Vision pipeline: YOLO11n + ByteTrack · model loads on demand</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_live_feed() -> None:
    render_page_header(
        "Video monitoring",
        "Demonstration video feed",
        "Review the provided prerecorded campus clip. This is not a live CCTV connection.",
    )
    st.warning("PRERECORDED VIDEO · No physical camera or live CCTV source is connected.")
    if not VIDEO_PATH.is_file():
        st.error(f"Video file not found: {VIDEO_PATH}")
        return

    metadata = read_video_metadata(str(VIDEO_PATH))
    if not metadata["available"]:
        st.error("The video file exists but OpenCV could not open it as a valid video source.")
        return

    try:
        st.video(VIDEO_PATH.read_bytes(), width="stretch")
    except OSError as exc:
        st.error(f"Unable to read the demonstration video: {exc}")
        return

    fps = float(metadata["fps"])
    duration = (
        int(metadata["frames"]) / fps
        if fps > 0 and int(metadata["frames"]) > 0
        else None
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Source", VIDEO_PATH.name)
    col2.metric("Resolution", f'{metadata["width"]} × {metadata["height"]}')
    col3.metric("Frame rate", f"{fps:.2f} FPS" if fps > 0 else "Unavailable")
    col4.metric(
        "Duration",
        f"{int(duration // 60):02d}:{int(duration % 60):02d}" if duration is not None else "Unavailable",
    )
    st.caption("Use Detections to analyze this same clip with YOLO11n and ByteTrack.")


def render_detections() -> None:
    render_page_header(
        "Computer vision pipeline",
        "Detection & tracking",
        "Analyze the prerecorded campus clip with the existing person, restricted-zone, and baggage pipeline.",
    )
    st.info("Analysis uses a prerecorded file. It does not represent live surveillance.")
    if not VIDEO_PATH.is_file():
        st.error(f"Detection video not found: {VIDEO_PATH}")
        return
    if not MODEL_PATH.is_file():
        st.error(f"YOLO11n weights not found: {MODEL_PATH}")
        return
    if not TRACKER_PATH.is_file():
        st.error(f"ByteTrack configuration not found: {TRACKER_PATH}")
        return

    metadata = read_video_metadata(str(VIDEO_PATH))
    if not metadata["available"]:
        st.error("OpenCV could not open the selected video. Check the file format and path.")
        return

    info_cols = st.columns(4)
    info_cols[0].metric("Detection model", "YOLO11n")
    info_cols[1].metric("Multi-object tracker", "ByteTrack")
    info_cols[2].metric("Confidence threshold", "0.30")
    info_cols[3].metric("Inference size", "640 px")
    st.caption(
        f"Tracker config: {TRACKER_PATH.name} · OpenCV source: {VIDEO_PATH.name} · "
        f'{metadata["width"]} × {metadata["height"]} · '
        f'{float(metadata["fps"]):.2f} FPS' if float(metadata["fps"]) > 0
        else f"Tracker config: {TRACKER_PATH.name} · OpenCV source: {VIDEO_PATH.name} · "
             f'{metadata["width"]} × {metadata["height"]} · frame rate unavailable'
    )
    st.caption(
        "Restricted zone: right-hand demonstration area · "
        "stationary-object threshold: 10 seconds · after-hours entry logging is preserved."
    )

    run_detection = st.button("▶  Run complete detection", type="primary")
    if not run_detection:
        st.markdown(
            '<div class="info-panel"><div class="info-label">Annotated preview</div>'
            '<div class="info-value">Run analysis to inspect the annotated frame stream.</div>'
            '<div class="info-label" style="margin-top:.45rem">The YOLO11n model is loaded once and reused for this session.</div></div>',
            unsafe_allow_html=True,
        )
        return

    frame_placeholder = st.empty()
    progress = st.progress(0, text="Preparing video analysis…")
    counter_placeholder = st.empty()
    model: YOLO | None = None
    processor: VideoEventProcessor | None = None
    frame_count = 0
    detection_error: Exception | None = None

    try:
        model = load_detection_model()
        st.session_state["campus_watch_model_loaded"] = True
        processor = VideoEventProcessor(
            location="Right-side demonstration zone",
            stationary_seconds=10,
        )
        fps = float(metadata["fps"])
        if fps <= 0:
            fps = 25.0
        total_frames = int(metadata["frames"])
        results = model.track(
            source=str(VIDEO_PATH),
            stream=True,
            persist=True,
            tracker=str(TRACKER_PATH),
            conf=0.3,
            iou=0.5,
            imgsz=640,
            verbose=False,
        )
        for result in results:
            frame_count += 1
            annotated = processor.process_result(result, frame_count / fps)
            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width="stretch",
            )
            counter_placeholder.metric("Frames processed", f"{frame_count:,}")
            if total_frames > 0:
                progress.progress(
                    min(frame_count / total_frames, 1.0),
                    text=f"Processing frame {frame_count:,} of {total_frames:,}",
                )
            else:
                progress.progress(0, text=f"Processing frame {frame_count:,}")
    except Exception as exc:
        detection_error = exc
    finally:
        load_events.clear()

    if detection_error is not None:
        st.error(f"Detection could not complete: {type(detection_error).__name__}: {detection_error}")
    elif frame_count == 0:
        st.warning("The video source opened, but the detection pipeline returned no frames.")
    else:
        progress.progress(1.0, text="Analysis complete")
        st.success(f"Detection complete · {frame_count:,} frames processed.")

    if processor is not None:
        summary = st.columns(5)
        summary[0].metric("Unique person tracks", f"{len(processor.tracked_people):,}")
        summary[1].metric("Restricted-zone entries", f"{processor.zone_entries:,}")
        summary[2].metric("Bag detections", f"{processor.detected_bags:,}")
        summary[3].metric("Unique bag tracks", f"{len(processor.bag_tracks_seen):,}")
        summary[4].metric("Unattended-baggage alerts", f"{processor.unattended_bags:,}")
        if processor.tracked_people:
            st.markdown("### Person tracking results")
            people = pd.DataFrame(list(processor.tracked_people.values()))
            if "Confidence" in people.columns:
                people["Confidence"] = (people["Confidence"] * 100).round(2).map(lambda value: f"{value:.2f}%")
            st.dataframe(people, hide_index=True, width="stretch")
        st.caption(
            "Restricted-entry incidents are stored only during the configured after-hours period. "
            "Unattended-baggage incidents are stored when the processor's stationary-object conditions are met."
        )
        st.caption("Stored incidents are available in Alerts and History.")


def simulated_sensors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Device ID": "CAM-SIM-01", "Device type": "Camera status", "Location": "North Gate", "Status": "Online", "Battery": 100},
            {"Device ID": "CAM-SIM-02", "Device type": "Camera status", "Location": "Library Plaza", "Status": "Online", "Battery": 88},
            {"Device ID": "MOT-SIM-01", "Device type": "Motion sensor", "Location": "East Corridor", "Status": "Offline", "Battery": 0},
            {"Device ID": "MOT-SIM-02", "Device type": "Motion sensor", "Location": "West Gate", "Status": "Online", "Battery": 95},
            {"Device ID": "ENV-SIM-01", "Device type": "Environmental sensor", "Location": "Central Lawn", "Status": "Maintenance", "Battery": 45},
        ]
    )


def render_sensors() -> None:
    render_page_header(
        "Simulated IoT network",
        "Sensor monitoring",
        "Illustrative device inventory for the prototype — values are simulated, not hardware telemetry.",
    )
    st.warning("SIMULATED VALUES · No physical IoT sensors or cameras are connected.")
    devices = simulated_sensors()
    counts = devices["Status"].value_counts()
    metrics = st.columns(4)
    metrics[0].metric("Configured devices", len(devices))
    metrics[1].metric("Online", int(counts.get("Online", 0)))
    metrics[2].metric("Offline", int(counts.get("Offline", 0)))
    metrics[3].metric("Maintenance", int(counts.get("Maintenance", 0)))
    display = devices.copy()
    display["Battery"] = display["Battery"].astype(int)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Battery": st.column_config.ProgressColumn(
                "Battery (simulated)",
                min_value=0,
                max_value=100,
                format="%d%%",
            )
        },
    )
    st.caption("Device IDs, status, locations, and battery percentages are demonstration data only.")


def render_alerts(frame: pd.DataFrame) -> None:
    render_page_header(
        "Incident response",
        "Alerts",
        "Review the most recent incidents recorded by the existing event engine in SQLite.",
    )
    if frame.empty:
        st.info("No incidents have been recorded in SQLite.")
        return
    filters = st.columns(2)
    severity_options = ["All severities"] + [
        value for value in SEVERITY_ORDER if value in frame["severity"].unique()
    ]
    location_options = ["All locations"] + sorted(
        value for value in frame["location"].unique() if value
    )
    selected_severity = filters[0].selectbox("Severity", severity_options)
    selected_location = filters[1].selectbox("Location", location_options)
    filtered = frame.copy()
    if selected_severity != "All severities":
        filtered = filtered[filtered["severity"] == selected_severity]
    if selected_location != "All locations":
        filtered = filtered[filtered["location"] == selected_location]
    filtered = filtered.sort_values("timestamp", ascending=False, na_position="last")
    st.caption(f"Showing {len(filtered):,} of {len(frame):,} recorded incidents · newest first")
    if filtered.empty:
        st.info("No incidents match the selected filters.")
        return
    for _, event in filtered.iterrows():
        render_incident_card(event)


def render_statistics(frame: pd.DataFrame) -> None:
    render_page_header(
        "Operational analytics",
        "Statistics",
        "Explore incident patterns by type, location, severity, and date using actual database records.",
    )
    if frame.empty:
        st.info("No event data is available for statistics.")
        return
    event_counts = (
        frame["event_type"].replace("", "Unspecified").value_counts()
        .rename_axis("Event type").reset_index(name="Incidents")
    )
    st.markdown("### Events by type")
    st.bar_chart(event_counts, x="Event type", y="Incidents", color="#38bdf8", height=300)
    st.markdown("### Events by location")
    location_counts = (
        frame["location"].replace("", "Unspecified").value_counts()
        .rename_axis("Location").reset_index(name="Incidents")
    )
    st.bar_chart(
        location_counts,
        x="Incidents",
        y="Location",
        color="#34d399",
        horizontal=True,
        height=320,
    )
    st.markdown("### Daily incident trend")
    valid = frame.dropna(subset=["timestamp"])
    if valid.empty:
        st.info("No valid event timestamps are available to build a trend.")
    else:
        trend = valid.set_index("timestamp").resample("D").size()
        trend = trend.rename("Incidents").rename_axis("Date").reset_index()
        st.line_chart(trend, x="Date", y="Incidents", color="#38bdf8", height=300)
    st.markdown("### Severity distribution")
    severity_counts = (
        frame["severity"].replace("", "UNKNOWN").value_counts()
        .rename_axis("Severity").reset_index(name="Incidents")
    )
    st.bar_chart(severity_counts, x="Severity", y="Incidents", color="#fbbf24", height=260)


def render_history(frame: pd.DataFrame) -> None:
    render_page_header(
        "Incident records",
        "History",
        "Search and filter the full SQLite event history, including records from previous dates.",
    )
    if frame.empty:
        st.info("No historical events are available.")
        return

    valid_dates = frame["timestamp"].dropna()
    if valid_dates.empty:
        st.error("No valid timestamps are available to filter this event history.")
        return

    date_col, location_col, severity_col = st.columns(3)
    date_range = date_col.date_input(
        "Date range",
        value=(valid_dates.min().date(), valid_dates.max().date()),
        help="Choose the start and end dates to include.",
    )
    locations = ["All locations"] + sorted(value for value in frame["location"].unique() if value)
    severities = ["All severities"] + sorted(value for value in frame["severity"].unique() if value)
    selected_location = location_col.selectbox("Location", locations)
    selected_severity = severity_col.selectbox("Severity", severities)
    search = st.text_input("Search event type or description", placeholder="e.g. restricted entry")

    filtered = frame.dropna(subset=["timestamp"]).copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        if isinstance(start_date, date) and isinstance(end_date, date):
            filtered = filtered[
                (filtered["timestamp"].dt.date >= start_date)
                & (filtered["timestamp"].dt.date <= end_date)
            ]
    elif isinstance(date_range, date):
        filtered = filtered[filtered["timestamp"].dt.date == date_range]
    if selected_location != "All locations":
        filtered = filtered[filtered["location"] == selected_location]
    if selected_severity != "All severities":
        filtered = filtered[filtered["severity"] == selected_severity]
    if search.strip():
        query = search.strip().casefold()
        searchable = (
            filtered["event_type"].str.casefold().str.contains(query, na=False, regex=False)
            | filtered["description"].str.casefold().str.contains(query, na=False, regex=False)
        )
        filtered = filtered[searchable]

    filtered = filtered.sort_values("timestamp", ascending=False)
    st.metric("Matching incidents", f"{len(filtered):,}")
    if filtered.empty:
        st.info("No events match the selected filters.")
        return
    table = filtered[EVENT_COLUMNS].copy()
    table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download filtered history as CSV",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="campus_safety_incident_history.csv",
        mime="text/csv",
        type="primary",
    )


def render_severity_analysis(frame: pd.DataFrame) -> None:
    render_page_header(
        "Risk posture",
        "Severity analysis",
        "Measure the severity mix and inspect the underlying incidents recorded in SQLite.",
    )
    counts = frame["severity"].value_counts()
    total = len(frame)
    metrics = st.columns(3)
    for column, severity in zip(metrics, SEVERITY_ORDER):
        count = int(counts.get(severity, 0))
        percentage = count / total * 100 if total else 0
        column.metric(f"{severity.title()} incidents", f"{count:,}", f"{percentage:.1f}% of events")
    if frame.empty:
        st.info("No incidents are available for severity analysis.")
        return

    distribution = (
        counts.reindex(SEVERITY_ORDER, fill_value=0)
        .rename_axis("Severity").reset_index(name="Incidents")
    )
    st.markdown("### Severity distribution")
    st.bar_chart(distribution, x="Severity", y="Incidents", color="#fb7185", height=280)
    st.markdown("### Incident details")
    details = frame.sort_values("timestamp", ascending=False, na_position="last")[
        ["id", "event_type", "location", "severity", "timestamp", "description"]
    ].copy()
    details["timestamp"] = details["timestamp"].map(format_timestamp)
    st.dataframe(details, hide_index=True, width="stretch")


def main() -> None:
    events, database_error = load_events()
    frame = build_event_frame(events)
    section = render_sidebar(database_error is None)

    pages = {
        "Overview": lambda: render_overview(frame, database_error),
        "Live Feed": render_live_feed,
        "Detections": render_detections,
        "Sensors": render_sensors,
        "Alerts": lambda: render_alerts(frame),
        "Statistics": lambda: render_statistics(frame),
        "History": lambda: render_history(frame),
        "Severity Analysis": lambda: render_severity_analysis(frame),
    }
    pages[section]()


main()
