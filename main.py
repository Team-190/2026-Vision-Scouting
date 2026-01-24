import cv2
import numpy as np
import math

clicked_hsv = []

# =============================
# Mouse click for HSV selection
# =============================
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param
        hsv_value = hsv_frame[y, x]
        clicked_hsv.append(hsv_value)
        print(f"Clicked HSV: {hsv_value}")
        cv2.destroyAllWindows()

# =============================
# Step 1: Select color
# =============================
test_image = cv2.imread("test.png")
hsv_frame = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

cv2.namedWindow("Select Color")
cv2.setMouseCallback("Select Color", click_event, hsv_frame)

print("Click a pixel to select the ball color")
while len(clicked_hsv) < 1:
    cv2.imshow("Select Color", test_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
cv2.destroyAllWindows()

selected_color = np.array(clicked_hsv[0], dtype=np.float32)

# =============================
# Tracker parameters
# =============================
MAX_DIST = 40          # max distance to associate detections
MIN_TRAJ_LEN = 8       # frames needed before validating trajectory
COUNT_LINE_Y = 0.6     # fraction of frame height to count ball once

next_id = 0
tracks = {}  # id -> dict(center, history, counted, missed)

ball_count = 0

# =============================
# Video
# =============================
cap = cv2.VideoCapture("test.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "ball_detection_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

# =============================
# Helper functions
# =============================
def is_ball_trajectory(points):
    if len(points) < MIN_TRAJ_LEN:
        return False

    pts = np.array(points)
    x = pts[:, 0]
    y = pts[:, 1]

    try:
        coeffs = np.polyfit(x, y, 2)  # quadratic fit
        a = coeffs[0]
        return a > 0.001  # downward opening parabola
    except:
        return False

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# =============================
# Main loop
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    diff = hsv - selected_color
    distance = np.linalg.norm(diff, axis=2)

    max_distance = np.sqrt(179**2 + 255**2 + 255**2)
    norm_dist = np.clip(distance / max_distance, 0, 1)

    mask = (norm_dist < 0.25).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for c in contours:
        if cv2.contourArea(c) > 100:
            (x, y), r = cv2.minEnclosingCircle(c)
            detections.append((int(x), int(y)))

    # =============================
    # Spot Tracker Association
    # =============================
    used = set()
    for tid in list(tracks.keys()):
        track = tracks[tid]
        best_d = MAX_DIST
        best_pt = None

        for i, pt in enumerate(detections):
            if i in used:
                continue
            d = dist(track["center"], pt)
            if d < best_d:
                best_d = d
                best_pt = (i, pt)

        if best_pt is not None:
            idx, pt = best_pt
            used.add(idx)

            track["center"] = pt
            track["history"].append(pt)
            track["missed"] = 0

            if (
                not track["counted"]
                and is_ball_trajectory(track["history"])
                and pt[1] > COUNT_LINE_Y * frame_height
            ):
                ball_count += 1
                track["counted"] = True

        else:
            track["missed"] += 1
            if track["missed"] > 5:
                del tracks[tid]

    # =============================
    # Create new tracks
    # =============================
    for i, pt in enumerate(detections):
        if i not in used:
            tracks[next_id] = {
                "center": pt,
                "history": [pt],
                "counted": False,
                "missed": 0,
            }
            next_id += 1

    # =============================
    # Visualization
    # =============================
    for tid, t in tracks.items():
        for i in range(1, len(t["history"])):
            cv2.line(
                frame,
                t["history"][i - 1],
                t["history"][i],
                (255, 0, 0),
                2,
            )
        cv2.circle(frame, t["center"], 8, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {tid}",
            (t["center"][0] + 5, t["center"][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.putText(
        frame,
        f"Ball Count: {ball_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )

    out.write(frame)
    cv2.imshow("Ball Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Saved: ball_detection_output.mp4")
print("Final ball count:", ball_count)
