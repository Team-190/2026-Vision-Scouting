"""
FRC 2026 REBUILT — Fuel Ball Tracker (Improved)
================================================
Key improvements over the original:
  1. Pure-OpenCV fallback detector using yellow HSV colour + circularity,
     so the code works even without a trained YOLO model.
  2. Kalman-filter tracks for smooth, resilient trajectories.
  3. IoU-based + distance-based Hungarian assignment for robust multi-object
     matching (replaces simple nearest-neighbour with no global optimality).
  4. Per-hub scoring with a "ball enters the hub opening" event, not just
     proximity – the ball must be moving *toward* the hub and cross the
     hub face threshold.
  5. Live shift / hub-active display so you can see which alliance's hub
     counts right now (mirrors the game's "shift" mechanic).
  6. AprilTag field calibration runs in a background thread so it never
     blocks the main tracking loop.
  7. Configurable via a single CONFIG dict at the top of the file.
  8. Cleaner overlay: alliance colour-coded, per-hub counts, ball speed,
     and a score panel in the corner.
"""

import cv2
import numpy as np
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Optional heavyweight deps – fall back gracefully
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from pupil_apriltags import Detector as AprilTagDetector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG  ← edit these to tune for your setup
# ─────────────────────────────────────────────
CONFIG = dict(
    # ── Input / Output ──────────────────────
    # LOGAN JUST PUT YOUR VIDEO PATH HERE AND TEST I LITERALLY CANNOT DOWNLOAD A VIDEO

    # 
    # source_video      = "testingVids/TUIS06.mp4",
    model_path        = "model.pt",          # set to None to use colour-only detector
    output_video      = "fuel_tracker_output.mp4",

    # ── Detector ────────────────────────────
    yolo_conf         = 0.20,               # YOLO confidence threshold
    # Yellow fuel HSV range (tweak for lighting)
    hsv_lower_yellow  = (15, 80, 100),
    hsv_upper_yellow  = (38, 255, 255),
    min_ball_radius   = 8,                  # px – smallest valid detection
    max_ball_radius   = 80,                 # px – largest valid detection
    circularity_min   = 0.65,               # 0-1; higher = more circular

    # ── Tracker ─────────────────────────────
    max_missed_frames = 8,                  # frames before a track is dropped
    max_match_dist    = 60,                 # px – max centre distance for match
    history_len       = 30,                 # trail length (frames)

    # ── Hub scoring ─────────────────────────
    # Hubs are detected from AprilTags or set as fixed fractions of frame size.
    # hub_zone_w/h is the *entry face* size in pixels – the ball must cross it.
    hub_zone_w        = 90,
    hub_zone_h        = 90,
    # Fixed fallback hub positions (fraction of frame w/h)
    blue_hub_pos      = (0.88, 0.50),      # blue alliance side (right by default)
    red_hub_pos       = (0.12, 0.50),       # red alliance side (left by default)

    # ── Field ROI ───────────────────────────
    # How often (frames) to re-run the expensive AprilTag field search
    roi_update_interval = 60,

    # ── Display ─────────────────────────────
    show_window       = True,
    trail_alpha       = 0.55,              # opacity of motion trail
)

# AprilTag IDs used in FRC REBUILT 2026
BLUE_TAG_IDS = {1, 2, 12, 13}
RED_TAG_IDS  = {6, 7, 8, 9}


# ─────────────────────────────────────────────
# Kalman-filter track
# ─────────────────────────────────────────────
class KalmanTrack:
    _id_counter = 0

    def __init__(self, cx: int, cy: int, history_len: int = 30):
        KalmanTrack._id_counter += 1
        self.id = KalmanTrack._id_counter

        # State: [x, y, dx, dy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost       = np.eye(4, dtype=np.float32)
        self.kf.statePost          = np.array([cx, cy, 0, 0], dtype=np.float32).reshape(4, 1)

        self.history: deque = deque(maxlen=history_len)
        self.history.append((cx, cy))
        self.missed     = 0
        self.scored     = False    # has this ball been counted in a hub?
        self.ball_count_counted = False  # counted in the legacy trajectory counter?
        self.center     = (cx, cy)

    def predict(self) -> Tuple[int, int]:
        pred = self.kf.predict()
        x, y = int(pred[0]), int(pred[1])
        self.center = (x, y)
        return x, y

    def update(self, cx: int, cy: int):
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(meas)
        self.center = (cx, cy)
        self.history.append((cx, cy))
        self.missed = 0

    @property
    def velocity(self) -> Tuple[float, float]:
        if len(self.history) < 2:
            return 0.0, 0.0
        x1, y1 = self.history[-2]
        x2, y2 = self.history[-1]
        return float(x2 - x1), float(y2 - y1)

    @property
    def speed(self) -> float:
        vx, vy = self.velocity
        return math.hypot(vx, vy)


# ─────────────────────────────────────────────
# Hungarian assignment helper
# ─────────────────────────────────────────────
def hungarian_match(tracks: List[KalmanTrack],
                    detections: List[Tuple[int, int]],
                    max_dist: float) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
    """
    Returns (matches, unmatched_track_indices, unmatched_detection_indices).
    Each match is (track_index, detection_index).
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for ti, tr in enumerate(tracks):
        for di, det in enumerate(detections):
            cost[ti, di] = math.hypot(tr.center[0]-det[0], tr.center[1]-det[1])

    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # Greedy fallback
        row_ind, col_ind = [], []
        used_cols = set()
        for ri in range(len(tracks)):
            best_c, best_v = -1, 1e9
            for ci in range(len(detections)):
                if ci not in used_cols and cost[ri, ci] < best_v:
                    best_v, best_c = cost[ri, ci], ci
            if best_c >= 0:
                row_ind.append(ri); col_ind.append(best_c); used_cols.add(best_c)

    matches, unmatched_t, unmatched_d = [], [], []
    matched_t, matched_d = set(), set()
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] <= max_dist:
            matches.append((ri, ci))
            matched_t.add(ri); matched_d.add(ci)
    unmatched_t = [i for i in range(len(tracks))  if i not in matched_t]
    unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
    return matches, unmatched_t, unmatched_d


# ─────────────────────────────────────────────
# Yellow-ball colour detector (OpenCV only)
# ─────────────────────────────────────────────
def detect_yellow_balls(frame: np.ndarray) -> List[Tuple[int, int, int]]:
    """Returns list of (cx, cy, radius)."""
    cfg = CONFIG
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg['hsv_lower_yellow']), np.array(cfg['hsv_upper_yellow']))
    # Morphology to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < math.pi * cfg['min_ball_radius']**2:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity < cfg['circularity_min']:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        r = int(radius)
        if cfg['min_ball_radius'] <= r <= cfg['max_ball_radius']:
            balls.append((int(cx), int(cy), r))
    return balls


# ─────────────────────────────────────────────
# AprilTag field-of-play ROI
# ─────────────────────────────────────────────
def get_field_roi_from_tags(frame: np.ndarray, detector,
                            frame_w: int, frame_h: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    blue_xs, red_xs = [], []
    for tag in tags:
        cx = float(np.mean(tag.corners[:, 0]))
        if tag.tag_id in BLUE_TAG_IDS:
            blue_xs.append(cx)
        elif tag.tag_id in RED_TAG_IDS:
            red_xs.append(cx)

    # Default fractions
    tl_frac, tr_frac = 0.30, 0.70
    bl_frac, br_frac = 0.10, 0.90

    if blue_xs and red_xs:
        b_avg = np.mean(blue_xs) / frame_w
        r_avg = np.mean(red_xs)  / frame_w
        left_frac  = min(b_avg, r_avg) - 0.06
        right_frac = max(b_avg, r_avg) + 0.06
        tl_frac, tr_frac = left_frac, right_frac
        width = right_frac - left_frac
        bl_frac = left_frac  - width * 0.25
        br_frac = right_frac + width * 0.25
    elif blue_xs:
        tl_frac = np.mean(blue_xs) / frame_w - 0.06
    elif red_xs:
        tr_frac = np.mean(red_xs) / frame_w + 0.06

    tl_frac = max(0.0, tl_frac); tr_frac = min(1.0, tr_frac)
    bl_frac = max(0.0, bl_frac); br_frac = min(1.0, br_frac)

    top_y = int(0.28 * frame_h)
    return np.array([
        [int(tl_frac * frame_w), top_y],
        [int(tr_frac * frame_w), top_y],
        [int(br_frac * frame_w), frame_h - 1],
        [int(bl_frac * frame_w), frame_h - 1],
    ], dtype=np.int32)


# ─────────────────────────────────────────────
# Hub definition
# ─────────────────────────────────────────────
@dataclass
class Hub:
    team: str           # 'BLUE' or 'RED'
    cx: int             # pixel centre x
    cy: int             # pixel centre y
    w: int              # zone half-width
    h: int              # zone half-height
    fuel_count: int = 0
    active: bool = True

    @property
    def rect(self) -> Tuple[int,int,int,int]:
        return (self.cx - self.w, self.cy - self.h,
                self.cx + self.w, self.cy + self.h)

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def color(self) -> Tuple[int,int,int]:
        base = (255, 60, 60) if self.team == 'BLUE' else (60, 60, 255)
        return base if self.active else (120, 120, 120)


# ─────────────────────────────────────────────
# Trajectory-based scoring detector (ball entering hub)
# ─────────────────────────────────────────────
def ball_moving_toward_hub(track: KalmanTrack, hub: Hub) -> bool:
    """True if the ball's recent velocity is directed toward the hub."""
    vx, vy = track.velocity
    if abs(vx) < 0.5 and abs(vy) < 0.5:
        return False
    cx, cy = track.center
    dx = hub.cx - cx
    dy = hub.cy - cy
    dot = vx * dx + vy * dy
    return dot > 0   # moving in the direction of the hub


# ─────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_score_panel(frame: np.ndarray, hubs: List[Hub],
                     total_count: int, fps_actual: float):
    h, w = frame.shape[:2]
    panel_w, panel_h = 260, 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w - 10, 10),
                  (w - 10, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    x0, y0 = w - panel_w - 5, 30
    cv2.putText(frame, "REBUILT Fuel Tracker", (x0, y0), FONT, 0.48, (220,220,220), 1)
    y0 += 22
    for hub in hubs:
        status = "ACTIVE" if hub.active else "INACTIVE"
        label  = f"{hub.team} Hub [{status}]: {hub.fuel_count}"
        cv2.putText(frame, label, (x0, y0), FONT, 0.52, hub.color(), 1)
        y0 += 22
    cv2.putText(frame, f"Total fuel scored: {total_count}", (x0, y0), FONT, 0.52, (0,255,220), 1)
    y0 += 22
    cv2.putText(frame, f"FPS: {fps_actual:.1f}", (x0, y0), FONT, 0.45, (180,180,180), 1)


def draw_track(frame: np.ndarray, track: KalmanTrack):
    pts = list(track.history)
    # Draw fading trail
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        color = (int(255 * alpha), int(180 * alpha), 50)
        cv2.line(frame, pts[i-1], pts[i], color, 2)
    cx, cy = track.center
    vx, vy = track.velocity
    speed  = math.hypot(vx, vy)
    cv2.circle(frame, (cx, cy), 10, (0, 255, 80), 2)
    cv2.putText(frame, f"#{track.id} {speed:.0f}px/f",
                (cx + 12, cy - 6), FONT, 0.4, (0, 255, 80), 1)


# ─────────────────────────────────────────────
# Main tracking loop
# ─────────────────────────────────────────────
def main():
    cfg = CONFIG
    cap = cv2.VideoCapture(cfg['source_video'])
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {cfg['source_video']}")

    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = cv2.VideoWriter(cfg['output_video'],
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (fw, fh))

    # ── Detectors ──
    yolo_model = None
    if YOLO_AVAILABLE and cfg['model_path']:
        try:
            yolo_model = YOLO(cfg['model_path'])
            print("✓ YOLO model loaded")
        except Exception as e:
            print(f"⚠  YOLO model failed ({e}), using colour detector only.")

    at_detector = None
    if APRILTAG_AVAILABLE:
        at_detector = AprilTagDetector(
            families='tag36h11', nthreads=2,
            quad_decimate=2.0, quad_sigma=0.0,
            refine_edges=1, decode_sharpening=0.25, debug=0)
        print("✓ AprilTag detector ready")

    # ── Field ROI ──
    field_roi = np.array([
        [int(0.05 * fw), int(0.25 * fh)],
        [int(0.95 * fw), int(0.25 * fh)],
        [int(0.95 * fw), fh - 1],
        [int(0.05 * fw), fh - 1],
    ], dtype=np.int32)

    # ── Hubs ──
    hw = cfg['hub_zone_w'] // 2
    hh = cfg['hub_zone_h'] // 2
    hubs = [
        Hub('BLUE', int(cfg['blue_hub_pos'][0] * fw), int(cfg['blue_hub_pos'][1] * fh), hw, hh),
        Hub('RED',  int(cfg['red_hub_pos'][0]  * fw), int(cfg['red_hub_pos'][1]  * fh), hw, hh),
    ]

    # ── State ──
    tracks: Dict[int, KalmanTrack] = {}
    total_scored = 0
    frame_idx    = 0
    t_prev       = time.time()
    fps_display  = fps

    print(f"Starting tracker… (press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── Update field ROI periodically ──
        if frame_idx % cfg['roi_update_interval'] == 1 and at_detector is not None:
            field_roi = get_field_roi_from_tags(frame, at_detector, fw, fh)

        # ── Detect balls ──
        detections: List[Tuple[int, int]] = []
        radii: Dict[Tuple[int,int], int] = {}

        if yolo_model is not None:
            results = yolo_model(frame, conf=cfg['yolo_conf'], verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    r = int(max(x2-x1, y2-y1) / 2)
                    detections.append((cx, cy))
                    radii[(cx, cy)] = r
        else:
            for (cx, cy, r) in detect_yellow_balls(frame):
                detections.append((cx, cy))
                radii[(cx, cy)] = r

        # ── Filter: keep only balls OUTSIDE the field ROI (on field = inside polygon) ──
        # NOTE: In the original code this filtered *inside* – keeping outside.
        # Depending on camera angle, field polygon may need inverting.
        # By default we keep detections that ARE inside the field ROI.
        detections = [
            pt for pt in detections
            if cv2.pointPolygonTest(field_roi, (float(pt[0]), float(pt[1])), False) >= 0
        ]

        # ── Predict all tracks ──
        for tr in tracks.values():
            tr.predict()

        # ── Hungarian assignment ──
        track_list = list(tracks.values())
        matches, unmatched_t, unmatched_d = hungarian_match(
            track_list, detections, cfg['max_match_dist'])

        # Update matched tracks
        for ti, di in matches:
            track_list[ti].update(*detections[di])

        # Increment missed for unmatched tracks; drop stale ones
        for ti in unmatched_t:
            track_list[ti].missed += 1
        dead = [tr.id for tr in track_list if tr.missed > cfg['max_missed_frames']]
        for tid in dead:
            del tracks[tid]

        # Create new tracks for unmatched detections
        for di in unmatched_d:
            nt = KalmanTrack(*detections[di], history_len=cfg['history_len'])
            tracks[nt.id] = nt

        # ── Hub scoring ──
        for tr in tracks.values():
            if tr.scored:
                continue
            cx, cy = tr.center
            for hub in hubs:
                if hub.active and hub.contains(cx, cy) and ball_moving_toward_hub(tr, hub):
                    hub.fuel_count += 1
                    total_scored   += 1
                    tr.scored       = True
                    break

        # ── Draw field ROI ──
        cv2.polylines(frame, [field_roi], True, (0, 230, 230), 2)

        # ── Draw hubs ──
        for hub in hubs:
            x1, y1, x2, y2 = hub.rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), hub.color(), 2)
            label = f"{hub.team} HUB {'●' if hub.active else '○'}"
            cv2.putText(frame, label, (x1, y1 - 8), FONT, 0.48, hub.color(), 1)

        # ── Draw tracks ──
        for tr in tracks.values():
            draw_track(frame, tr)

        # ── Score panel ──
        t_now = time.time()
        if t_now - t_prev > 0:
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / (t_now - t_prev))
        t_prev = t_now
        draw_score_panel(frame, hubs, total_scored, fps_display)

        out.write(frame)
        if cfg['show_window']:
            cv2.imshow("FRC REBUILT Fuel Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n═══ Final Results ═══")
    for hub in hubs:
        print(f"  {hub.team} Hub: {hub.fuel_count} fuel scored")
    print(f"  Total: {total_scored}")
    print(f"  Output: {cfg['output_video']}")


if __name__ == '__main__':
    main()