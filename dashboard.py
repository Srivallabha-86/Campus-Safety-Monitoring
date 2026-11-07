import streamlit as st
import pandas as pd
import numpy as np
import cv2

from datetime import datetime, timedelta
from pathlib import Path
from ultralytics import YOLO

from backend.database import get_all_events

# Page Setup
st.set_page_config(
    page_title="Campus Safety Monitoring",
    page_icon="🛡️",
    layout="wide"
)

# Header
st.title("🛡️ Campus Safety Monitoring")
st.caption("Campus safety dashboard using Computer Vision and IoT")

# Sidebar Navigation (All 7 Sections Required)
st.sidebar.title("System Navigation")
section = st.sidebar.radio(
    "Select View", 
    [
        "Live Feed", 
        "Detections", 
        "Sensors", 
        "Alerts", 
        "Statistics", 
        "History", 
        "Severity Statistics"
    ]
)

# ---------------------------------------------------------
# MOCK DATA FUNCTIONS (Modular setup for future DB swapping)
# ---------------------------------------------------------
def get_mock_detections():
    return pd.DataFrame({
        "Track ID": [101, 102, 103, 104, 105],
        "Class": ["Person", "Person", "Unattended Bag", "Vehicle", "Person"],
        "Confidence": ["98.2%", "94.5%", "89.1%", "97.8%", "91.0%"],
        "Location / Zone": ["North Gate", "Library Plaza", "East Corridor", "West Gate", "Central Lawn"],
        "Timestamp": [
            (datetime.now() - timedelta(minutes=i*3)).strftime("%H:%M:%S") for i in range(5)
        ]
    })

def get_mock_sensors():
    return pd.DataFrame({
        "Sensor ID": ["CAM-01", "CAM-02", "IR-101", "IR-102", "CAM-03"],
        "Type": ["Camera", "Camera", "Motion IR", "Motion IR", "Camera"],
        "Location": ["North Gate", "Library Plaza", "East Corridor", "West Gate", "Central Lawn"],
        "Status": ["🟢 Online", "🟢 Online", "🔴 Offline", "🟢 Online", "🟡 Maintenance"],
        "Battery Level": ["100%", "88%", "0%", "95%", "45%"]
    })

def get_mock_alerts():
    return pd.DataFrame({
        "Alert ID": ["ALT-901", "ALT-902", "ALT-903"],
        "Severity": ["Critical", "High", "Medium"],
        "Rule Triggered": ["Sensor Offline", "Restricted Area Entry", "Crowd Density Exceeded"],
        "Location": ["East Corridor", "Server Room", "Library Plaza"],
        "Status": ["Active", "Active", "Acknowledged"],
        "Timestamp": ["10:15:00 AM", "10:20:12 AM", "10:30:45 AM"]
    })

# ---------------------------------------------------------
# TOP LEVEL KPI METRICS
# ---------------------------------------------------------

events = get_all_events()

total_events = len(events)

critical_events = sum(
    1 for event in events
    if event["severity"] == "CRITICAL"
)

high_events = sum(
    1 for event in events
    if event["severity"] == "HIGH"
)

total_locations = len(
    set(event["location"] for event in events)
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Safety Events", total_events)
col2.metric("Critical Alerts", critical_events)
col3.metric("High Alerts", high_events)
col4.metric("Affected Locations", total_locations)

st.markdown("---")


st.markdown("---")


# ---------------------------------------------------------
# 1. LIVE FEED
# ---------------------------------------------------------
if section == "Live Feed":
    st.subheader("📹 Campus Video Feed")
    st.caption(
        "Prerecorded campus demonstration video. "
        "Real-time CCTV integration is planned for a later stage."
    )

    video_path = Path("videos/people-walking.mp4")

    if video_path.exists():
        st.video(video_path.read_bytes())

        st.success("Demonstration video loaded successfully.")

        st.markdown("### Video Information")

        col1, col2, col3 = st.columns(3)

        col1.metric("Video Source", "people-walking.mp4")
        col2.metric("Frame Rate", "25 FPS")
        col3.metric("Resolution", "1920 × 1080")

        st.info(
            "Use the Detections section to run YOLO11n "
            "and ByteTrack on this video."
        )
    else:
        st.error(
            "Video file not found. Check videos/people-walking.mp4."
        )


#
# ---------------------------------------------------------
# 2. DETECTIONS
# ---------------------------------------------------------
elif section == "Detections":
    st.subheader("🔍 YOLO11n + ByteTrack Detection")
    st.caption(
        "Person detection and tracking using the prerecorded video dataset."
    )

    video_path = Path("videos/people-walking.mp4")

    @st.cache_resource
    def load_detection_model():
        return YOLO("yolo11n.pt")

    if not video_path.exists():
        st.error("Detection video not found.")
    else:
        st.write("Source: people-walking.mp4")
        st.write("Model: YOLO11n")
        st.write("Tracker: ByteTrack")

        if st.button("▶ Run Detection", type="primary"):
            model = load_detection_model()

            detection_placeholder = st.empty()
            progress_bar = st.progress(0)

            tracked_people = {}
            frame_count = 0

            total_frames = 341

            results = model.track(
                source=str(video_path),
                stream=True,
                persist=True,
                tracker="bytetrack_custom.yaml",
                classes=[0],
                conf=0.3,
                iou=0.5,
                verbose=False
            )

            for result in results:
                frame_count += 1

                boxes = result.boxes

                if boxes is not None and boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()

                    for track_id, confidence in zip(
                        track_ids, confidences
                    ):
                        if track_id not in tracked_people:
                            tracked_people[track_id] = {
                                "Track ID": track_id,
                                "Class": "Person",
                                "Confidence": confidence,
                                "Frames Tracked": 0
                            }

                        tracked_people[track_id][
                            "Confidence"
                        ] = confidence

                        tracked_people[track_id][
                            "Frames Tracked"
                        ] += 1

                annotated_frame = result.plot()

                frame_rgb = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )

                detection_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )

                progress_bar.progress(
                    min(frame_count / total_frames, 1.0)
                )

            progress_bar.progress(1.0)
            st.success(
                f"Detection completed! {frame_count} frames processed."
            )

            if tracked_people:
                df_detections = pd.DataFrame(
                    list(tracked_people.values())
                )

                df_detections["Confidence"] = (
                    df_detections["Confidence"] * 100
                ).round(2).astype(str) + "%"

                st.metric(
                    "Unique Person Tracks",
                    len(tracked_people)
                )

                st.markdown("### Tracking Results")

                st.dataframe(
                    df_detections,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No people were detected in the video.")



# ---------------------------------------------------------
# 3. SENSORS
# ---------------------------------------------------------
elif section == "Sensors":
    st.subheader("📡 Simulated IoT Sensor Network")

    st.caption(
        "Demonstration sensor information. "
        "Physical sensors are not connected in this prototype."
    )

    df_sens = get_mock_sensors()

    st.dataframe(
        df_sens,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Sensor statuses and battery levels are sample data. "
        "Real sensor integration can be added in a later phase."
    )



# ---------------------------------------------------------
# 4. ALERTS
# ---------------------------------------------------------
elif section == "Alerts":
    st.subheader("🚨 Incident & Safety Alerts")
    st.caption("Safety events retrieved from the SQLite database")

    events = get_all_events()

    if not events:
        st.success("No safety events recorded.")
    else:
        st.metric("Total Recorded Events", len(events))

        for event in events:
            severity = event["severity"]
            timestamp = datetime.fromisoformat(
                event["timestamp"]
            ).strftime("%d %b %Y, %I:%M:%S %p")

            alert_message = (
                f"**Event ID:** {event['id']}\n\n"
                f"**Type:** {event['event_type']}\n\n"
                f"**Location:** {event['location']}\n\n"
                f"**Time:** {timestamp}\n\n"
                f"**Description:** {event['description']}"
            )

            if severity == "CRITICAL":
                st.error(f"🔴 **{severity}**\n\n{alert_message}")
            elif severity == "HIGH":
                st.warning(f"🟠 **{severity}**\n\n{alert_message}")
            else:
                st.info(f"🔵 **{severity}**\n\n{alert_message}")



# ---------------------------------------------------------
# 5. STATISTICS
# ---------------------------------------------------------
elif section == "Statistics":
    st.subheader("📈 Safety Event Analytics")
    st.caption("Analytics calculated from stored SQLite events")

    events = get_all_events()

    if not events:
        st.info("No event data available for statistics.")
    else:
        df_stats = pd.DataFrame(events)

        df_stats["timestamp"] = pd.to_datetime(
            df_stats["timestamp"]
        )

        # Total event count.
        total_events = len(df_stats)

        # Unique locations.
        total_locations = df_stats["location"].nunique()

        # Most common event type.
        most_common_event = df_stats["event_type"].value_counts().idxmax()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Events", total_events)
        col2.metric("Locations Affected", total_locations)
        col3.metric("Most Common Event", most_common_event)

        st.markdown("---")

        # Event count by type.
        st.markdown("### Events by Type")

        event_counts = (
            df_stats["event_type"]
            .value_counts()
            .rename_axis("Event Type")
            .to_frame("Total Events")
        )

        st.bar_chart(event_counts)

        # Event count by location.
        st.markdown("### Events by Location")

        location_counts = (
            df_stats["location"]
            .value_counts()
            .rename_axis("Location")
            .to_frame("Total Events")
        )

        st.bar_chart(location_counts)

        # Daily event trend.
        st.markdown("### Daily Event Trend")

        daily_counts = (
            df_stats.set_index("timestamp")
            .resample("D")
            .size()
        )

        st.line_chart(daily_counts)


# ---------------------------------------------------------
# 6. HISTORY
# ---------------------------------------------------------
elif section == "History":
    st.subheader("📜 Historical Event Logs")
    st.caption("Historical safety events retrieved from SQLite")

    events = get_all_events()

    if not events:
        st.info("No historical events available.")
    else:
        df_history = pd.DataFrame(events)

        # Convert timestamps into datetime values.
        df_history["timestamp"] = pd.to_datetime(
            df_history["timestamp"]
        )

        # Filter controls.
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            selected_date = st.date_input(
                "Select Date",
                value=datetime.now().date()
            )

        with col_f2:
            locations = ["All Locations"] + sorted(
                df_history["location"].unique().tolist()
            )

            selected_location = st.selectbox(
                "Filter Location",
                locations
            )

        # Apply filters.
        filtered_history = df_history[
            df_history["timestamp"].dt.date == selected_date
        ]

        if selected_location != "All Locations":
            filtered_history = filtered_history[
                filtered_history["location"] == selected_location
            ]

        # Display results.
        st.metric("Matching Events", len(filtered_history))

        display_columns = [
            "id",
            "timestamp",
            "event_type",
            "location",
            "severity",
            "description"
        ]

        if filtered_history.empty:
            st.info("No events found for the selected filters.")
        else:
            st.dataframe(
                filtered_history[display_columns],
                use_container_width=True,
                hide_index=True
            )



# ---------------------------------------------------------
# 7. SEVERITY STATISTICS
# ---------------------------------------------------------
elif section == "Severity Statistics":
    st.subheader("⚠️ Incident Breakdown by Severity Level")
    st.caption("Severity distribution calculated from SQLite events")

    events = get_all_events()

    if not events:
        st.info("No safety events available for severity analysis.")
    else:
        df_severity = pd.DataFrame(events)

        severity_order = ["LOW", "HIGH", "CRITICAL"]

        severity_counts = (
            df_severity["severity"]
            .value_counts()
            .reindex(severity_order, fill_value=0)
            .to_frame("Total Incidents")
        )

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("### Severity Distribution")
            st.bar_chart(severity_counts)

        with col_chart2:
            st.markdown("### Severity Summary")

            for severity in severity_order:
                count = int(severity_counts.loc[severity, "Total Incidents"])
                st.metric(f"{severity} Incidents", count)

        st.markdown("---")

        st.markdown("### Severity Event Details")

        st.dataframe(
            df_severity[
                ["id", "event_type", "location", "severity", "timestamp"]
            ],
            use_container_width=True,
            hide_index=True
        )
