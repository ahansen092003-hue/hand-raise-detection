from ultralytics import YOLO
import sys, os, cv2, json
from collections import defaultdict, deque

model = YOLO("yolo11n-pose.pt")

vid_in  = "inputs/input.mp4"
vid_out = "outputs/output.mp4"
os.makedirs("outputs", exist_ok=True)

cap = cv2.VideoCapture(vid_in)
if not cap.isOpened():
    print(f"Error opening the video file: {vid_in}")
    sys.exit(1)
print(f"Video opened successfully: {vid_in}")

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"FPS: {fps:.2f}, Resolution: {w}x{h}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(vid_out, fourcc, fps, (w, h))
if not writer.isOpened():
    print("Error creating output writer.")
    sys.exit(1)
print(f"Output video writer ready: {vid_out}")

recent = defaultdict(lambda: deque(maxlen=3))
prev_state = defaultdict(lambda: False)
raise_count = defaultdict(int)

seen_ids = set()

events = []  

def hand_raised_simple(kp_xy):

    L_SH, R_SH = 5, 6
    L_EL, R_EL = 7, 8
    L_WR, R_WR = 9, 10

    def is_vertical(elbow, wrist, ratio_threshold=0.5):
        dx = abs(wrist[0] - elbow[0])
        dy = abs(wrist[1] - elbow[1])
        return dy > 0 and (dx / dy) < ratio_threshold

    left_wrist_above = kp_xy[L_WR, 1] < kp_xy[L_SH, 1]
    left_vertical = is_vertical(kp_xy[L_EL], kp_xy[L_WR])

    right_wrist_above = kp_xy[R_WR, 1] < kp_xy[R_SH, 1]
    right_vertical = is_vertical(kp_xy[R_EL], kp_xy[R_WR])


    return bool((left_wrist_above and left_vertical) or (right_wrist_above and right_vertical))


results = model.track(
    source=vid_in,
    stream=True,
    tracker="bytetrack.yaml",
    persist=True,
    verbose=False
)

frame_idx = 0
max_frames = int(fps * 60)  

for r in results:
    annotated = r.plot()
    writer.write(annotated)

    if (r.keypoints is not None) and (r.keypoints.xy is not None):
        kps = r.keypoints.xy  
        ids = r.boxes.id
        if ids is not None:
            ids = ids.int().cpu().numpy()
            kps_np = kps.cpu().numpy()

            for sid, kp in zip(ids, kps_np):
                seen_ids.add(int(sid))

                up_now = hand_raised_simple(kp)

                recent[sid].append(up_now)
                stable_up = len(recent[sid]) == 3 and all(recent[sid])

                if stable_up and not prev_state[sid]:
                    timestamp = frame_idx / float(fps)
                    events.append({
                        "type": "hand_raise_start",
                        "frame": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "student_id": int(sid)
                    })
                    raise_count[sid] += 1
                    prev_state[sid] = True

                if not stable_up and prev_state[sid]:
                    prev_state[sid] = False

    frame_idx += 1
    if frame_idx % int(max(1, fps)) == 0:
        print(f"Processed {frame_idx // int(fps)} s")

    if frame_idx >= max_frames:
        print("Reached 60 s of video, stopping early.")
        break

writer.release()
cap.release()
print(f"Output saved to: {vid_out}")

events_path = "outputs/events.ndjson"
with open(events_path, "w") as f:
    for ev in events:
        f.write(json.dumps(ev) + "\n")
print(f"Events written to: {events_path} (count={len(events)})")

summary = {
    "duration_seconds": round(frame_idx / float(fps), 3),
    "total_students": len(seen_ids),
    "students_with_hands_raised": sorted(int(s) for s, c in raise_count.items() if c > 0),
    "total_hand_raises": int(sum(raise_count.values())),
    "per_student_raise_counts": {str(int(s)): int(c) for s, c in raise_count.items() if c > 0}
}
summary_path = "outputs/summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to: {summary_path}")
