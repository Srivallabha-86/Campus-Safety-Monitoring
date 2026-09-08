import cv2

from backend.vision import (
    open_video,
    process_frame,
    create_person_detector,
    detect_person,
    detect_motion,
)

from backend.event_engine import process_observation
from datetime import datetime

# --------------------------------------------------
# Dataset video
# --------------------------------------------------

VIDEO_SOURCE = "videos/people-walking.mp4"


# --------------------------------------------------
# Simulated campus information
# --------------------------------------------------

LOCATION = "Restricted Campus Area"

# We are treating this video as a restricted-zone camera
RESTRICTED_ZONE = True

# No door sensor is connected yet
DOOR_OPEN = False

# Simulated normal temperature
TEMPERATURE = 25.0


# --------------------------------------------------
# Open video
# --------------------------------------------------

capture = open_video(VIDEO_SOURCE)


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

detector = create_person_detector()


# --------------------------------------------------
# Previous frame for motion detection
# --------------------------------------------------

previous_frame = None

last_event_type = None
last_event_time = None

EVENT_COOLDOWN_SECONDS = 15


# --------------------------------------------------
# Process video
# --------------------------------------------------

while True:

    success, frame = capture.read()

    if not success:
        print("Video finished.")
        break


    # ----------------------------------------------
    # Person detection
    # ----------------------------------------------

    person_boxes = detect_person(
        frame,
        detector,
        confidence=0.3
    )

    person_detected = len(person_boxes) > 0


    # ----------------------------------------------
    # Motion detection
    # ----------------------------------------------

    current_frame = process_frame(frame)

    motion_detected = detect_motion(
        previous_frame,
        current_frame
    )

    previous_frame = current_frame


    # ----------------------------------------------
    # Process safety observation
    # ----------------------------------------------

    events = process_observation(
        person_detected=person_detected,
        restricted_zone=RESTRICTED_ZONE,
        door_open=DOOR_OPEN,
        motion_detected=motion_detected,
        location=LOCATION,
        temperature=TEMPERATURE,
    )


    # ----------------------------------------------
    # Draw person bounding boxes
    # ----------------------------------------------

    for x1, y1, x2, y2, score in person_boxes:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"Person {score:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ----------------------------------------------
    # Display detection count
    # ----------------------------------------------

    cv2.putText(
        frame,
        f"People detected: {len(person_boxes)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )


    # ----------------------------------------------
    # Display motion status
    # ----------------------------------------------

    if motion_detected:
        motion_text = "MOTION DETECTED"
    else:
        motion_text = "NO SIGNIFICANT MOTION"

    cv2.putText(
        frame,
        motion_text,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    # ----------------------------------------------
    # Display safety events
    # ----------------------------------------------

    if events:
      event = events[0]

      current_time = datetime.now()

      should_report_event = False

     # First occurrence of an event
      if last_event_type is None:
         should_report_event = True

      # A different event type occurred
      elif event.event_type != last_event_type:

         should_report_event = True

       # Same event, but cooldown has expired
      elif last_event_time is not None:
          elapsed_time = (
             current_time - last_event_time
            ).total_seconds()

          if elapsed_time >= EVENT_COOLDOWN_SECONDS:
              should_report_event = True

       # Report the event if necessary
      if should_report_event:
         print(
             f"NEW SAFETY EVENT: "
             f"{event.event_type} - {event.severity}"
           )

         last_event_type = event.event_type
         last_event_time = current_time

      event_text = (
         f"{event.event_type} - {event.severity}"
       )

      cv2.putText(
         frame,
         event_text,
         (20, 120),
         cv2.FONT_HERSHEY_SIMPLEX,
         0.7,
         (0, 0, 255),
         2
       )
    else:
       last_event_type = None
       last_event_time = None
 

    # ----------------------------------------------
    # Show video
    # ----------------------------------------------

    cv2.imshow(
        "Campus Safety - Dataset Video",
        frame
    )


    # ----------------------------------------------
    # Quit
    # ----------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

capture.release()
cv2.destroyAllWindows()