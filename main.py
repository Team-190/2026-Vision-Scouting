import numpy as np
import math
from ultralytics import YOLO
import multiprocessing as mp
from queue import Empty
import time
import cv2
from pupil_apriltags import Detector        


def get_field_trapezoid(frame, frame_width, frame_height):
    # Convert to HSV and filter for white lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Use canny edge detection and hough lines
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    if lines is None:
        # Fallback to a default trapezoid if no lines are found
        return np.array([
            [0.3 * frame_width, 0.4 * frame_height], [0.7 * frame_width, 0.4 * frame_height],
            [0.9 * frame_width, 0.9 * frame_height], [0.1 * frame_width, 0.9 * frame_height]
        ], dtype=np.int32)

    # Find the main field lines (usually near horizontal)
    bottom_line_y = 0
    top_line_y = frame_height
    left_line_x_at_bottom = 0
    right_line_x_at_bottom = frame_width

    # A simple approach: find min/max extents of near-horizontal and angled lines
    # This can be improved with more robust line clustering and fitting
    horizontal_lines = []
    side_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle < 20 or angle > 160: # Near horizontal
            horizontal_lines.append(line)
        elif 45 < angle < 135: # Angled side lines
            side_lines.append(line)

    if horizontal_lines:
        all_y = [line[0][1] for line in horizontal_lines] + [line[0][3] for line in horizontal_lines]
        top_line_y = min(all_y)
        bottom_line_y = max(all_y)

    # Fallback for line positions
    if not side_lines:
        # If no side lines, create a wide trapezoid based on horizontal lines
        tl = [0.1 * frame_width, top_line_y]
        tr = [0.9 * frame_width, top_line_y]
        br = [frame_width, bottom_line_y]
        bl = [0, bottom_line_y]
    else:
        # Find left and right-most points of side lines to form the trapezoid
        all_x = [line[0][0] for line in side_lines] + [line[0][2] for line in side_lines]
        min_x = min(all_x)
        max_x = max(all_x)
        
        # Estimate trapezoid points
        tl = [min_x, top_line_y]
        tr = [max_x, top_line_y]
        br = [max_x + (bottom_line_y - top_line_y)*0.4, bottom_line_y] # Project outwards
        bl = [min_x - (bottom_line_y - top_line_y)*0.4, bottom_line_y]

    # Clamp values to be within frame boundaries
    poly = np.array([tl, tr, br, bl], dtype=np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, frame_width)
    poly[:, 1] = np.clip(poly[:, 1], 0, frame_height)
    
    return poly

# =============================
# AprilTag Logic
# =============================
# FRC 2026/2025 Standard: Tags 1, 2, 12, 13 (typical blue side) vs 6, 7, 8, 9 (typical red side)
# Adjust based on specific event configuration if necessary.
BLUE_TAGS = {12, 13, 1, 2} 
RED_TAGS = {6, 7, 8, 9}

def calculate_perspective_skew(tags, frame_width, frame_height):
    """Calculate camera perspective angle based on AprilTag positions and sizes."""
    if len(tags) < 2:
        return 0.0  # No perspective adjustment possible
    
    # Get tag sizes and positions
    tag_data = []
    for tag in tags:
        corners = tag.corners
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        # Calculate tag width (distance between left and right corners)
        tag_width = np.mean([
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[3])
        ])
        tag_data.append((center_y, tag_width, center_x))
    
    # Tags closer to bottom of frame (higher y) should appear smaller due to perspective
    # Calculate how perspective changes tag size
    tag_data.sort(key=lambda x: x[0])
    
    if len(tag_data) >= 2:
        top_tag_y, top_tag_width, top_tag_x = tag_data[0]
        bot_tag_y, bot_tag_width, bot_tag_x = tag_data[-1]
        
        # Calculate perspective ratio
        if bot_tag_width > 0:
            perspective_ratio = top_tag_width / bot_tag_width
            # Store this for trapezoid adjustment
            return perspective_ratio
    
    return 1.0

def get_hubs_from_color(frame, frame_width, frame_height):
    """Detect scoring hubs as tower structures next to the neutral zone."""
    hubs = []
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect yellow fuel (neutral zone) to find hub positions relative to it
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find yellow contours
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hub_size = 100
    
    if contours:
        # Find the largest yellow region (neutral zone)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Neutral zone bounds
        neutral_left = x
        neutral_right = x + w
        neutral_top = y
        neutral_bottom = y + h
        neutral_cx = (neutral_left + neutral_right) // 2
        neutral_cy = (neutral_top + neutral_bottom) // 2
        
        # Hubs are positioned just outside the neutral zone (left and right)
        # Offset from the center by a fixed distance
        hub_offset_x = int(w * 0.35)  # Position hubs away from center
        
        # Hub Y position is roughly at the middle height of the field
        hub_y = int(neutral_cy)
        
        # Create hubs: one left, one right of neutral zone
        blue_hub_x = neutral_right + hub_offset_x
        red_hub_x = neutral_left - hub_offset_x
        
        # Ensure hubs are within frame bounds
        blue_hub_x = min(blue_hub_x, frame_width - 50)
        red_hub_x = max(red_hub_x, 50)
        
        # Create hub objects
        hubs = [
            {
                'center': (blue_hub_x, hub_y),
                'team': 'BLUE',
                'fuel_count': 0
            },
            {
                'center': (red_hub_x, hub_y),
                'team': 'RED',
                'fuel_count': 0
            }
        ]
    else:
        # Fallback to fixed positions if yellow zone not found
        hub_y = int(frame_height * 0.5)
        hubs = [
            {
                'center': (int(frame_width * 0.65), hub_y),
                'team': 'BLUE',
                'fuel_count': 0
            },
            {
                'center': (int(frame_width * 0.35), hub_y),
                'team': 'RED',
                'fuel_count': 0
            }
        ]
    
    return hubs, hub_size



def get_field_from_apriltags(frame, detector, frame_width, frame_height):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)
    
    blue_tags = []
    red_tags = []
    all_tag_x = []
    
    for tag in tags:
        tag_center_x = np.mean(tag.corners[:, 0])
        tag_center_y = np.mean(tag.corners[:, 1])
        
        if tag.tag_id in BLUE_TAGS:
            blue_tags.append((tag_center_x, tag_center_y, tag))
            all_tag_x.append(tag_center_x)
        elif tag.tag_id in RED_TAGS:
            red_tags.append((tag_center_x, tag_center_y, tag))
            all_tag_x.append(tag_center_x)
    
    # Calculate perspective skew from AprilTags
    perspective_ratio = calculate_perspective_skew(tags, frame_width, frame_height)
    
    # Default fallback (trapezoid ratios)
    top_l, top_r = 0.35, 0.65
    bot_l, bot_r = 0.15, 0.85  # Wider bottom by default
    
    if blue_tags and red_tags:
        # Get average x positions for each side
        blue_avg_x = np.mean([x for x, y, t in blue_tags])
        red_avg_x = np.mean([x for x, y, t in red_tags])
        
        # Determine which side is which and set top boundaries
        if blue_avg_x < red_avg_x:  # Blue is Left, Red is Right
            top_l = (blue_avg_x / frame_width) - 0.05
            top_r = (red_avg_x / frame_width) + 0.05
        else:  # Red is Left, Blue is Right
            top_l = (red_avg_x / frame_width) - 0.05
            top_r = (blue_avg_x / frame_width) + 0.05
        
        # Calculate perspective-aware bottom boundary
        # The bottom should be wider than the top due to perspective
        center = (top_l + top_r) / 2
        top_width = top_r - top_l
        
        # Use perspective ratio to expand the bottom
        expansion_factor = max(1.3, perspective_ratio * 0.8)  # At least 30% wider
        bot_width = top_width * expansion_factor
        
        bot_l = center - (bot_width / 2)
        bot_r = center + (bot_width / 2)
        
    elif blue_tags:
        # Only blue visible
        blue_avg_x = np.mean([x for x, y, t in blue_tags])
        blue_avg_y = np.mean([y for x, y, t in blue_tags])
        
        if blue_avg_x < frame_width / 2:  # Blue on left
            top_l = (blue_avg_x / frame_width) - 0.05
            top_r = 0.85
        else:  # Blue on right
            top_l = 0.15
            top_r = (blue_avg_x / frame_width) + 0.05
        
        # Apply perspective expansion
        center = (top_l + top_r) / 2
        top_width = top_r - top_l
        expansion_factor = max(1.3, perspective_ratio * 0.8)
        bot_width = top_width * expansion_factor
        bot_l = center - (bot_width / 2)
        bot_r = center + (bot_width / 2)
        
    elif red_tags:
        # Only red visible
        red_avg_x = np.mean([x for x, y, t in red_tags])
        
        if red_avg_x < frame_width / 2:  # Red on left
            top_l = (red_avg_x / frame_width) - 0.05
            top_r = 0.85
        else:  # Red on right
            top_l = 0.15
            top_r = (red_avg_x / frame_width) + 0.05
        
        # Apply perspective expansion
        center = (top_l + top_r) / 2
        top_width = top_r - top_l
        expansion_factor = max(1.3, perspective_ratio * 0.8)
        bot_width = top_width * expansion_factor
        bot_l = center - (bot_width / 2)
        bot_r = center + (bot_width / 2)

    # Clamp values to frame boundaries
    top_l, top_r = max(0, top_l), min(1, top_r)
    bot_l, bot_r = max(0, bot_l), min(1, bot_r)
    
    # Ensure bottom is wider than top
    bot_width = bot_r - bot_l
    top_width = top_r - top_l
    if bot_width < top_width:
        center = (bot_l + bot_r) / 2
        bot_l = center - (top_width / 2) * 1.2
        bot_r = center + (top_width / 2) * 1.2
        bot_l, bot_r = max(0, bot_l), min(1, bot_r)

    poly = np.array([
        [top_l * frame_width, 0.3 * frame_height], [top_r * frame_width, 0.3 * frame_height],
        [bot_r * frame_width, frame_height], [bot_l * frame_width, frame_height]
    ], dtype=np.int32)
    return poly

# =============================
# Inference Process
# =============================
def inference_worker(model_path, frame_queue, results_queue):
    model = YOLO(model_path)
    while True:
        try:
            data = frame_queue.get(timeout=1)
            if data is None: break
            frame_idx, frame = data
            results = model(frame, conf=0.15, verbose=False)
            results_queue.put((frame_idx, frame, results))
        except Empty:
            continue
        except Exception as e:
            print(f"Error in inference worker: {e}")
            break

# =============================
# Helper functions
# =============================
def is_ball_trajectory(points, min_traj_len):
    if len(points) < min_traj_len: return False
    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]
    try:
        coeffs = np.polyfit(x, y, 2)
        return coeffs[0] > 0.001
    except: return False

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    model_path = 'model.pt'
    source_video = "testingVids/TUIS06.mp4"
    max_dist, min_traj_len, count_line_y_ratio = 40, 8, 0.6
    
    cap = cv2.VideoCapture(source_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

    out = cv2.VideoWriter("testingVids/ball_detection_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    frame_queue, results_queue = mp.Queue(maxsize=10), mp.Queue(maxsize=10)
    num_processes = mp.cpu_count() if mp.cpu_count() > 2 else 1
    inf_processes = []
    for _ in range(num_processes):
        p = mp.Process(target=inference_worker, args=(model_path, frame_queue, results_queue))
        p.start()
        inf_processes.append(p)

    next_id, tracks, ball_count = 0, {}, 0
    next_frame_to_process, results_buffer, frame_idx_producer, stopped = 0, {}, 0, False
    field_roi = None
    hubs = []
    hub_size = 80

    print(f"Starting tracking with {num_processes} inference processes and dynamic field detection...")

    try:
        while True:
            if not stopped and frame_queue.qsize() < 5:
                ret, frame = cap.read()
                if not ret:
                    stopped = True
                    for _ in range(num_processes): frame_queue.put(None)
                else:
                    # Periodically update ignore zone using AprilTags on the producer side to avoid lag
                    if frame_idx_producer % 90 == 0 or field_roi is None:
                        field_roi = get_field_from_apriltags(frame, at_detector, frame_width, frame_height)
                        hubs, hub_size = get_hubs_from_color(frame, frame_width, frame_height)
                    frame_queue.put((frame_idx_producer, frame))
                    frame_idx_producer += 1

            while not results_queue.empty():
                try:
                    f_idx, f_img, f_results = results_queue.get_nowait()
                    results_buffer[f_idx] = (f_img, f_results)
                except Empty: break

            if next_frame_to_process in results_buffer:
                frame, results = results_buffer.pop(next_frame_to_process)
                detections = []
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        if cv2.pointPolygonTest(field_roi, (float(cx), float(cy)), False) >= 0:
                            continue

                        detections.append((cx, cy))
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"Fuel {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                used = set()
                for tid in list(tracks.keys()):
                    track = tracks[tid]
                    best_d, best_pt = max_dist, None
                    for i, pt in enumerate(detections):
                        if i in used: continue
                        d = dist(track["center"], pt)
                        if d < best_d: best_d, best_pt = d, (i, pt)

                    if best_pt is not None:
                        idx, pt = best_pt
                        used.add(idx)
                        track.update({"center": pt, "missed": 0})
                        track["history"].append(pt)
                        
                        # Check if fuel scored in a hub
                        if not track["scored"]:
                            for hub in hubs:
                                hub_x, hub_y = hub['center']
                                hub_box_size = hub_size
                                if (hub_x - hub_box_size < pt[0] < hub_x + hub_box_size and 
                                    hub_y - hub_box_size < pt[1] < hub_y + hub_box_size):
                                    hub['fuel_count'] += 1
                                    track["scored"] = True
                                    break
                        
                        # Original counting logic for trajectory-based detection
                        if (not track["counted"] and is_ball_trajectory(track["history"], min_traj_len) and 
                            pt[1] > count_line_y_ratio * frame_height):
                            ball_count += 1
                            track["counted"] = True
                    else:
                        track["missed"] += 1
                        if track["missed"] > 5: del tracks[tid]

                for i, pt in enumerate(detections):
                    if i not in used:
                        tracks[next_id] = {"center": pt, "history": [pt], "counted": False, "missed": 0, "scored": False}
                        next_id += 1

                cv2.polylines(frame, [field_roi], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.putText(frame, "FIELD OF PLAY", (field_roi[0][0] + 5, field_roi[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Draw hub bounding boxes
                if hubs:
                    for hub in hubs:
                        hub_x, hub_y = hub['center']
                        hub_box_size = hub_size
                        x1, y1 = max(0, hub_x - hub_box_size), max(0, hub_y - hub_box_size)
                        x2, y2 = min(frame_width, hub_x + hub_box_size), min(frame_height, hub_y + hub_box_size)
                        
                        # Color based on team
                        color = (255, 0, 0) if hub['team'] == 'BLUE' else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{hub['team']} Hub: {hub['fuel_count']}", (x1 + 5, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                for tid, t in tracks.items():
                    for i in range(1, len(t["history"])):
                        cv2.line(frame, t["history"][i - 1], t["history"][i], (255, 0, 0), 2)
                    cv2.circle(frame, t["center"], 8, (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {tid}", (t["center"][0] + 5, t["center"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(frame, f"Ball Count: {ball_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                out.write(frame)
                cv2.imshow("Tracking", frame)
                next_frame_to_process += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped = True
                    break
            else:
                if stopped and next_frame_to_process >= frame_idx_producer: break
                time.sleep(0.001)

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        for p in inf_processes: p.terminate(); p.join()
        print(f"Final ball count: {ball_count}")
