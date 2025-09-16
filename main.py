# queue_analysis_stable_enhanced.py
"""
Stable queue analysis using YOLOv8 (track) + ByteTrack (streaming).
Features:
- Draw two queue areas (Queue1 = waiting, Queue2 = processing) by mouse.
- Uses model.track(source=..., stream=True, tracker='bytetrack.yaml') so IDs are stable.
- Logs per-person: entry_time (first seen), exit_time (when disappeared),
  waiting_time_sec (total in Queue1), processing_time_sec (total in Queue2),
  total_journey_sec, queue entries, and a timeline of enter/exit events.
- Saves detailed CSV next to the video file.
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_PATH = r"D:\queue_analysis\WhatsApp Video 2025-09-11 at 16.31.52_204c0680.mp4"
MODEL_PATH = "yolov8m.pt"   # yolov8m / yolov8l for better accuracy than yolov8n
CONFIDENCE = 0.45           # detection confidence threshold (tweak if needed)
MIN_AREA = 1500             # minimal bbox area to ignore tiny false positives (tweak)
TRACKER_NAME = "bytetrack.yaml"  # use ByteTrack tracker provided by ultralytics

# -----------------------------
# 1) Prepare first frame and let user draw two rectangles (Queue1 and Queue2)
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

ret, first_frame = cap.read()
if not ret or first_frame is None:
    cap.release()
    raise RuntimeError("Could not read first frame from video.")

frame_for_drawing = first_frame.copy()
drawing = False
ix = iy = -1
queue_areas = []  # list of (x1,y1,x2,y2)
temp_frame = frame_for_drawing.copy()

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, temp_frame, frame_for_drawing, queue_areas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_frame = frame_for_drawing.copy()
        cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1, x2, y2 = min(ix, x), min(iy, y), max(ix, x), max(iy, y)
        queue_areas.append((x1, y1, x2, y2))
        temp_frame = frame_for_drawing.copy()
        # draw all saved areas
        for (ax1, ay1, ax2, ay2) in queue_areas:
            cv2.rectangle(temp_frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
        print(f"> Queue Area {len(queue_areas)} saved: {(x1, y1, x2, y2)}")

win_name = "Draw Queues: left-click-drag rectangles (draw 2), press 'q' when done"
cv2.namedWindow(win_name)
cv2.setMouseCallback(win_name, draw_rectangle)

print("\nDraw Queue 1 and Queue 2 as rectangles on the first frame.")
print(" - Left click + drag to draw a rectangle.")
print(" - Draw two rectangles (Queue1 then Queue2) and press 'q' when done.\n")

while True:
    cv2.imshow(win_name, temp_frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or len(queue_areas) >= 2:
        break

cv2.destroyWindow(win_name)
cap.release()

if len(queue_areas) < 2:
    raise SystemExit("You must draw 2 queue areas. Run again.")

QUEUE1_AREA, QUEUE2_AREA = queue_areas[:2]
print(f"Final Queue1 area: {QUEUE1_AREA}")
print(f"Final Queue2 area: {QUEUE2_AREA}")

# -----------------------------
# 2) Helper utilities and person state structure
# -----------------------------
def point_in_rect(point, rect):
    x, y = point
    x1, y1, x2, y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)

# person_times keyed by track id (int)
# structure:
# {
#   pid: {
#       "first_seen_sec": float,
#       "first_seen_ts": str,
#       "exit_sec": float or None,
#       "exit_ts": str or None,
#       "queue1_total": float,
#       "queue2_total": float,
#       "queue1_start": float or None,
#       "queue2_start": float or None,
#       "in_q1": bool,
#       "in_q2": bool,
#       "q1_entries": int,
#       "q2_entries": int,
#       "timeline": [ (event_str, sec) , ... ]
#   }
# }
person_times = {}

def ensure_person(pid, video_time):
    if pid not in person_times:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        person_times[pid] = {
            "first_seen_sec": video_time,
            "first_seen_ts": ts,
            "exit_sec": None,
            "exit_ts": None,
            "queue1_total": 0.0,
            "queue2_total": 0.0,
            "queue1_start": None,
            "queue2_start": None,
            "in_q1": False,
            "in_q2": False,
            "q1_entries": 0,
            "q2_entries": 0,
            "timeline": [("first_seen", video_time)]
        }

def update_queue_state(pid, in_q1, in_q2, video_time):
    """
    Update the person's queue states based on whether they are currently in queue1/queue2.
    Adds timeline events and accumulates totals only when transitions happen.
    """
    ensure_person(pid, video_time)
    p = person_times[pid]

    # Queue1 transitions
    if in_q1 and not p["in_q1"]:
        p["queue1_start"] = video_time
        p["in_q1"] = True
        p["q1_entries"] += 1
        p["timeline"].append(("Queue1_enter", video_time))
        print(f"[{video_time:.2f}s] Person {pid} ENTERED Queue1")
    elif (not in_q1) and p["in_q1"]:
        # exiting queue1
        if p["queue1_start"] is not None:
            delta = max(0.0, video_time - p["queue1_start"])
            p["queue1_total"] += delta
            p["timeline"].append(("Queue1_exit", video_time))
            print(f"[{video_time:.2f}s] Person {pid} LEFT Queue1 (waited {delta:.2f}s)")
        p["queue1_start"] = None
        p["in_q1"] = False

    # Queue2 transitions
    if in_q2 and not p["in_q2"]:
        p["queue2_start"] = video_time
        p["in_q2"] = True
        p["q2_entries"] += 1
        p["timeline"].append(("Queue2_enter", video_time))
        print(f"[{video_time:.2f}s] Person {pid} ENTERED Queue2")
    elif (not in_q2) and p["in_q2"]:
        # exiting queue2
        if p["queue2_start"] is not None:
            delta = max(0.0, video_time - p["queue2_start"])
            p["queue2_total"] += delta
            p["timeline"].append(("Queue2_exit", video_time))
            print(f"[{video_time:.2f}s] Person {pid} LEFT Queue2 (processed {delta:.2f}s)")
        p["queue2_start"] = None
        p["in_q2"] = False

# -----------------------------
# 3) Run tracker in streaming mode (single tracker instance)
# -----------------------------
print("\nStarting tracking (streaming)... this maintains stable IDs.")
model = YOLO(MODEL_PATH)

frame_idx = 0
prev_seen_ids = set()

# Use model.track in streaming mode so tracker state persists across frames.
stream = model.track(source=VIDEO_PATH,
                     conf=CONFIDENCE,
                     iou=0.45,
                     classes=[0],              # person only
                     tracker=TRACKER_NAME,
                     stream=True)

for result in stream:
    # result is a single-frame result (with tracking applied)
    frame_idx += 1
    video_time = frame_idx / fps  # seconds since start (approx)
    frame = result.orig_img  # original BGR image for display

    seen_ids = set()

    # boxes exist?
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        # get numpy arrays safely
        try:
            xyxy = boxes.xyxy.cpu().numpy()
        except Exception:
            xyxy = np.array(boxes.xyxy)

        # ids, confs may or may not exist; handle safely
        ids = None
        confs = None
        try:
            ids = boxes.id.cpu().numpy()
        except Exception:
            try:
                ids = np.array([int(x) for x in boxes.id])
            except Exception:
                ids = None
        try:
            confs = boxes.conf.cpu().numpy()
        except Exception:
            confs = None

        # iterate detections
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            area = (x2 - x1) * (y2 - y1)
            conf = float(confs[i]) if confs is not None else 1.0
            pid = int(ids[i]) if ids is not None else -1

            # basic filters: confidence and area
            if conf < CONFIDENCE or area < MIN_AREA:
                continue

            seen_ids.add(pid)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_q1 = point_in_rect((cx, cy), QUEUE1_AREA)
            in_q2 = point_in_rect((cx, cy), QUEUE2_AREA)

            # update person state
            update_queue_state(pid, in_q1, in_q2, video_time)

            # draw bbox and id, and small overlay with times
            color = (0, 200, 200)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{pid}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # optional: show waiting/processing time current so far on frame
            p = person_times.get(pid)
            if p is not None:
                waiting_now = p["queue1_total"]
                if p["queue1_start"] is not None:
                    waiting_now += max(0.0, video_time - p["queue1_start"])
                proc_now = p["queue2_total"]
                if p["queue2_start"] is not None:
                    proc_now += max(0.0, video_time - p["queue2_start"])

                info = f"W:{waiting_now:.1f}s P:{proc_now:.1f}s"
                cv2.putText(frame, info, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # detect exits: ids seen in prev frame but not in current -> finalize them
    gone_ids = prev_seen_ids - seen_ids
    if gone_ids:
        for gone in gone_ids:
            if gone in person_times and person_times[gone]["exit_sec"] is None:
                # finalize any ongoing queue durations using current video_time
                p = person_times[gone]
                # finalize queue1
                if p["in_q1"] and p["queue1_start"] is not None:
                    delta = max(0.0, video_time - p["queue1_start"])
                    p["queue1_total"] += delta
                    p["timeline"].append(("Queue1_exit", video_time))
                    p["queue1_start"] = None
                    p["in_q1"] = False
                    print(f"[{video_time:.2f}s] Person {gone} (gone) FINALIZED Queue1 +{delta:.2f}s")
                # finalize queue2
                if p["in_q2"] and p["queue2_start"] is not None:
                    delta = max(0.0, video_time - p["queue2_start"])
                    p["queue2_total"] += delta
                    p["timeline"].append(("Queue2_exit", video_time))
                    p["queue2_start"] = None
                    p["in_q2"] = False
                    print(f"[{video_time:.2f}s] Person {gone} (gone) FINALIZED Queue2 +{delta:.2f}s")

                # mark exit
                p["exit_sec"] = video_time
                p["exit_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                p["timeline"].append(("final_exit", video_time))
                print(f"[{video_time:.2f}s] Person {gone} EXITED (not seen)")

    prev_seen_ids = seen_ids.copy()

    # draw queue rectangles & live counts
    live_q1 = sum(1 for pid in person_times if person_times[pid]["in_q1"])
    live_q2 = sum(1 for pid in person_times if person_times[pid]["in_q2"])
    cv2.rectangle(frame, (QUEUE1_AREA[0], QUEUE1_AREA[1]), (QUEUE1_AREA[2], QUEUE1_AREA[3]), (0, 255, 0), 2)
    cv2.putText(frame, f"Queue1 ({live_q1})", (QUEUE1_AREA[0], max(0, QUEUE1_AREA[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.rectangle(frame, (QUEUE2_AREA[0], QUEUE2_AREA[1]), (QUEUE2_AREA[2], QUEUE2_AREA[3]), (0, 0, 255), 2)
    cv2.putText(frame, f"Queue2 ({live_q2})", (QUEUE2_AREA[0], max(0, QUEUE2_AREA[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show frame
    cv2.imshow("Queue Analysis (press q to stop)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Early stop requested by user.")
        break

# -----------------------------
# 4) Finalize any remaining persons still present at end of video
# -----------------------------
final_video_time = frame_idx / fps
for pid, p in person_times.items():
    # finalize any open queue starts
    if p["in_q1"] and p["queue1_start"] is not None:
        delta = max(0.0, final_video_time - p["queue1_start"])
        p["queue1_total"] += delta
        p["timeline"].append(("Queue1_exit_end", final_video_time))
        p["queue1_start"] = None
        p["in_q1"] = False
    if p["in_q2"] and p["queue2_start"] is not None:
        delta = max(0.0, final_video_time - p["queue2_start"])
        p["queue2_total"] += delta
        p["timeline"].append(("Queue2_exit_end", final_video_time))
        p["queue2_start"] = None
        p["in_q2"] = False
    if p["exit_sec"] is None:
        p["exit_sec"] = final_video_time
        p["exit_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        p["timeline"].append(("final_exit_end", final_video_time))

cv2.destroyAllWindows()

# -----------------------------
# 5) Build DataFrame and save CSV
# -----------------------------
rows = []
for pid, p in sorted(person_times.items(), key=lambda x: x[0]):
    entry = p.get("first_seen_sec")
    exit_ = p.get("exit_sec")
    total_journey = None
    if entry is not None and exit_ is not None:
        total_journey = max(0.0, exit_ - entry)
    rows.append({
        "person_id": pid,
        "entry_time_sec": round(entry, 2) if entry is not None else None,
        "exit_time_sec": round(exit_, 2) if exit_ is not None else None,
        "waiting_time_sec": round(p.get("queue1_total", 0.0), 2),
        "processing_time_sec": round(p.get("queue2_total", 0.0), 2),
        "total_journey_sec": round(total_journey, 2) if total_journey is not None else None,
        "queue1_entries": p.get("q1_entries", 0),
        "queue2_entries": p.get("q2_entries", 0),
        "timeline": p.get("timeline", []),
        "first_seen_ts": p.get("first_seen_ts"),
        "exit_ts": p.get("exit_ts")
    })

df = pd.DataFrame(rows)
if not df.empty:
    avg_wait = df["waiting_time_sec"].mean()
    avg_proc = df["processing_time_sec"].mean()
    avg_total = df["total_journey_sec"].mean()
else:
    avg_wait = avg_proc = avg_total = 0.0

print(f"\nTracked persons: {len(df)}")
print(f"Avg waiting (Queue1): {avg_wait:.2f}s")
print(f"Avg processing (Queue2): {avg_proc:.2f}s")
print(f"Avg total journey: {avg_total:.2f}s")

video_dir = os.path.dirname(os.path.abspath(VIDEO_PATH)) or "."
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(video_dir, f"queue_analysis_results_{timestamp}.csv")
df.to_csv(output_file, index=False)
print(f"Saved detailed results to: {output_file}")
