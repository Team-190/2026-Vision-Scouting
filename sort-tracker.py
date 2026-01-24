import cv2
import numpy as np
import math

# =============================
# Ball class
# =============================
class Ball:
    def __init__(self, ball_id, position):
        self.id = ball_id
        self.position = position
        self.history = [position]
        self.velocity = []
        self.counted = False

    def update(self, new_pos):
        vx = new_pos[0] - self.position[0]
        vy = new_pos[1] - self.position[1]
        self.velocity.append((vx, vy))
        self.position = new_pos
        self.history.append(new_pos)

    def is_ball_trajectory(self, min_len=8):
        if len(self.velocity) < min_len:
            return False
        v = np.array(self.velocity)
        vy = v[:, 1]
        accel = np.diff(vy)
        if len(accel) < 3:
            return False
        return np.mean(accel) > 0.5


# =============================
# HSV Color Selection
# =============================
clicked_hsv = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_hsv.append(param[y, x])
        cv2.destroyAllWindows()

img = cv2.imread("test.png")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("Select Color", img)
cv2.setMouseCallback("Select Color", click_event, hsv_img)

while not clicked_hsv:
    cv2.waitKey(1)

selected_color = np.array(clicked_hsv[0], dtype=np.float32)


# =============================
# Video Setup
# =============================
cap = cv2.VideoCapture("test.mp4")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "ball_detection_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h),
)

# =============================
# SORT Tracker
# =============================
tracker = Sort(
    max_age=5,
    min_hits=1,
    iou_threshold=0.2
)

balls = {}
ball_count = 0
COUNT_LINE = int(0.6 * h)


# =============================
# Main Loop
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    diff = hsv - selected_color
    dist = np.linalg.norm(diff, axis=2)
    mask = (dist / 442 < 0.25).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for c in contours:
        if cv2.contourArea(c) > 120:
            x, y, w_box, h_box = cv2.boundingRect(c)
            detections.append([x, y, x + w_box, y + h_box, 1.0])

    detections = np.array(detections)

    # =============================
    # SORT update
    # =============================
    tracks = tracker.update(detections)

    active_ids = set()

    for x1, y1, x2, y2, track_id in tracks:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        track_id = int(track_id)

        active_ids.add(track_id)

        if track_id not in balls:
            balls[track_id] = Ball(track_id, (cx, cy))
        else:
            balls[track_id].update((cx, cy))

    # Remove dead tracks
    balls = {
        tid: ball for tid, ball in balls.items()
        if tid in active_ids
    }

    # =============================
    # Counting Logic
    # =============================
    for ball in balls.values():
        if (
            not ball.counted
            and ball.is_ball_trajectory()
            and ball.position[1] > COUNT_LINE
        ):
            ball.counted = True
            ball_count += 1

    # =============================
    # Visualization
    # =============================
    for ball in balls.values():
        for i in range(1, len(ball.history)):
            cv2.line(
                frame,
                ball.history[i - 1],
                ball.history[i],
                (255, 0, 0),
                2,
            )

        cv2.circle(frame, ball.position, 6, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {ball.id}",
            (ball.position[0] + 6, ball.position[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    cv2.line(frame, (0, COUNT_LINE), (w, COUNT_LINE), (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"Count: {ball_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )

    out.write(frame)
    cv2.imshow("Ball Tracking (SORT)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Final ball count:", ball_count)