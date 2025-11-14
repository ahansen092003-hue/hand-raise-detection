#  Hand-Raise Detector (CoGuide Mini-Assessment)

Detect when students raise their hands (no facial recognition) using pose landmarks from a classroom video.

The program outputs:
- **Annotated video:** `outputs/output.mp4`
- **Per-frame events log:** `outputs/events.ndjson`
- **Summary report:** `outputs/summary.json`

## Tools Used

| Tool | Purpose |
|------|----------|
| **Ultralytics YOLOv11 Pose (`yolo11n-pose.pt`)** | Pose estimation (17 body keypoints per person). |
| **ByteTrack** | Tracks individuals across frames → stable `student_id`s. |
| **OpenCV** | Handles video input/output and annotation writing. |

## How It Works
1. **Pose + Tracking** — YOLO pose runs on each frame while ByteTrack maintains consistent IDs for each student.  
2. **Hand-Raise Rule** — a hand is considered raised when:
   - The **wrist** is above the same-side **shoulder**.
   - The **forearm** (elbow → wrist) is roughly vertical (wrist mostly above elbow).
   - This condition is true for **≥3 consecutive frames** to reduce jitter.  
3. **Events** — every *rising edge* (hand goes from down → up) generates one event line in `events.ndjson`.  
4. **Summary** — after processing, the script aggregates total students, total hand raises, and per-student counts.

### Example `summary.json`

{
  "duration_seconds": 60.03,
  "total_students": 12,
  "students_with_hands_raised": [2, 5, 7],
  "total_hand_raises": 5,
  "per_student_raise_counts": {"2": 2, "5": 1, "7": 2}
}


### Run Instructions

pip install -r requirements.txt   
python run.py  
By default, it processes the first 60 seconds of inputs/input.mp4.
To process the full video, remove or change the max_frames limit in run.py
