import cv2
import numpy as np
import math
from ultralytics import YOLO
import multiprocessing as mp
from queue import Empty
import time
from pupil_apriltags import Detector

# =============================
# AprilTag Logic
# =============================
# FRC 2026/2025 Standard: Tags 1, 2, 12, 13 (typical blue side) vs 6, 7, 8, 9 (typical red side)
# Adjust based on specific event configuration if necessary.
BLUE_TAGS = {12, 13, 1, 2} 
RED_TAGS = {6, 7, 8, 9}

def get_dynamic_ignore_polygon(frame, detector, frame_width, frame_height):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)
    
    blue_x = []
    red_x = []
    
    for tag in tags:
        if tag.tag_id in BLUE_TAGS:
            blue_x.append(np.mean(tag.corners[:, 0]))
        elif tag.tag_id in RED_TAGS:
            red_x.append(np.mean(tag.corners[:, 0]))
            
    # Default fallback (trapezoid ratios)
    top_l, top_r = 0.35, 0.65
    bot_l, bot_r = 0.25, 0.75
    
    if blue_x and red_x:
        # Sort so we know which is left/right
        b_avg = np.mean(blue_x)
        r_avg = np.mean(red_x)
        if b_avg < r_avg: # Blue is Left, Red is Right
            top_l, top_r = b_avg/frame_width + 0.12, r_avg/frame_width - 0.12
            bot_l, bot_r = top_l - 0.05, top_r + 0.05
        else: # Red is Left, Blue is Right
            top_l, top_r = r_avg/frame_width + 0.12, b_avg/frame_width - 0.12
            bot_l, bot_r = top_l - 0.05, top_r + 0.05
    elif blue_x:
        # Only blue visible, guess red is off-screen
        b_avg = np.mean(blue_x)
        if b_avg < frame_width / 2: # Blue on left
            top_l, top_r = b_avg/frame_width + 0.12, 0.6
        else: # Blue on right
            top_l, top_r = 0.4, b_avg/frame_width - 0.12
        bot_l, bot_r = top_l - 0.05, top_r + 0.05
    elif red_x:
        # Only red visible
        r_avg = np.mean(red_x)
        if r_avg < frame_width / 2: # Red on left
            top_l, top_r = r_avg/frame_width + 0.12, 0.6
        else: # Red on right
            top_l, top_r = 0.4, r_avg/frame_width - 0.12
        bot_l, bot_r = top_l - 0.05, top_r + 0.05

    # Clamp values
    top_l, top_r = max(0, top_l), min(1, top_r)
    bot_l, bot_r = max(0, bot_l), min(1, bot_r)

    poly = np.array([
        [top_l * frame_width, 0], [top_r * frame_width, 0],
        [bot_r * frame_width, 0.65 * frame_height], [bot_l * frame_width, 0.65 * frame_height]
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
    source_video = "testingVids/citrusTest.mp4"
    max_dist, min_traj_len, count_line_y_ratio = 40, 8, 0.6
    
    cap = cv2.VideoCapture(source_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

    out = cv2.VideoWriter("testingVids/ball_detection_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    frame_queue, results_queue = mp.Queue(maxsize=10), mp.Queue(maxsize=10)
    num_processes = mp.cpu_count() // 2 if mp.cpu_count() > 2 else 1
    inf_processes = []
    for _ in range(num_processes):
        p = mp.Process(target=inference_worker, args=(model_path, frame_queue, results_queue))
        p.start()
        inf_processes.append(p)

    next_id, tracks, ball_count = 0, {}, 0
    next_frame_to_process, results_buffer, frame_idx_producer, stopped = 0, {}, 0, False
    ignore_polygon = None

    print(f"Starting tracking with {num_processes} inference processes and AprilTag detection...")

    try:
        while True:
            if not stopped and frame_queue.qsize() < 5:
                ret, frame = cap.read()
                if not ret:
                    stopped = True
                    for _ in range(num_processes): frame_queue.put(None)
                else:
                    # Periodically update ignore zone using AprilTags on the producer side to avoid lag
                    if frame_idx_producer % 30 == 0 or ignore_polygon is None:
                        ignore_polygon = get_dynamic_ignore_polygon(frame, at_detector, frame_width, frame_height)
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

                        if cv2.pointPolygonTest(ignore_polygon, (float(cx), float(cy)), False) >= 0:
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
                        if (not track["counted"] and is_ball_trajectory(track["history"], min_traj_len) and 
                            pt[1] > count_line_y_ratio * frame_height):
                            ball_count += 1
                            track["counted"] = True
                    else:
                        track["missed"] += 1
                        if track["missed"] > 5: del tracks[tid]

                for i, pt in enumerate(detections):
                    if i not in used:
                        tracks[next_id] = {"center": pt, "history": [pt], "counted": False, "missed": 0}
                        next_id += 1

                cv2.polylines(frame, [ignore_polygon], isClosed=True, color=(0, 0, 0), thickness=2)
                cv2.putText(frame, "DYNAMIC IGNORE ZONE", (ignore_polygon[0][0] + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

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
