import cv2
import numpy as np

from pupil_apriltags import Detector
from sort_tracker import Sort

# =============================
# AprilTag detector
# =============================
apriltag_detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25
)

# =============================
# SORT tracker
# =============================
tracker = Sort( 
    max_age=10,
    min_hits=3,
    iou_threshold=0.3
)


# =============================
# Helper: AprilTags → SORT format
# =============================
def apriltags_to_sort_detections(tags):
    """
    Convert AprilTag detections to SORT format:
    [x1, y1, x2, y2, score]
    """
    detections = []

    for tag in tags:
        corners = tag.corners.astype(int)

        x1 = np.min(corners[:, 0])
        y1 = np.min(corners[:, 1])
        x2 = np.max(corners[:, 0])
        y2 = np.max(corners[:, 1])

        detections.append([x1, y1, x2, y2, 1.0])

    if len(detections) == 0:
        return np.empty((0, 5))

    return np.array(detections)


# =============================
# Video source
# =============================
cap = cv2.VideoCapture(0)  # change to filename if needed


# =============================
# Main loop
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Detect AprilTags ---
    tags = apriltag_detector.detect(gray)

    # --- Convert to SORT detections ---
    detections = apriltags_to_sort_detections(tags)

    # --- Update SORT ---
    tracks = tracker.update(detections)

    # =============================
    # Draw SORT tracks
    # =============================
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Match closest AprilTag to this track
        matched_tag_id = None
        min_dist = float("inf")

        for tag in tags:
            tx, ty = map(int, tag.center)
            dist = np.hypot(cx - tx, cy - ty)

            if dist < min_dist:
                min_dist = dist
                matched_tag_id = tag.tag_id

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"SORT {track_id}"
        if matched_tag_id is not None:
            label += f" | TAG {matched_tag_id}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # =============================
    # Draw AprilTag outlines
    # =============================
    for tag in tags:
        corners = tag.corners.astype(int)

        for i in range(4):
            cv2.line(
                frame,
                tuple(corners[i]),
                tuple(corners[(i + 1) % 4]),
                (255, 0, 0),
                2
            )

        cx, cy = map(int, tag.center)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.imshow("AprilTags + SORT", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()