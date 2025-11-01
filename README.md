README.md — Hand-Raise Detector (CoGuide Mini-Assessment)

Detect when students raise their hands (no facial recognition) using pose landmarks from a classroom video.
The program outputs:
An annotated video (outputs/output.mp4)
A per-frame events log (outputs/events.ndjson)
A summary report (outputs/summary.json)

Tool	Purpose
Ultralytics YOLOv11 Pose (yolo11n-pose.pt)	Pose estimation (17 body keypoints per person).
ByteTrack	Tracks individuals across frames → stable student_ids.
OpenCV	Video I/O and annotation writing.

How It Works
Pose + Tracking — runs YOLO pose on each frame and maintains consistent IDs for each student.
Hand-Raise Rule — a hand counts as raised when:
The wrist is above the same-side shoulder.
The forearm (elbow→wrist) is roughly vertical (wrist mostly above elbow).
This condition is true for ≥3 consecutive frames (reduces jitter).
Events — each “rising edge” (hand goes from down → up) generates one event line in events.ndjson.
Summary — at the end, the script compiles total students, number of hand raises, and per-student counts.

Example summary:
{
  "duration_seconds": 60.03,
  "total_students": 12,
  "students_with_hands_raised": [2, 5, 7],
  "total_hand_raises": 5,
  "per_student_raise_counts": {"2": 2, "5": 1, "7": 2}
}

Run Instructions

pip install ultralytics opencv-python
python run.py
By default, it processes the first 60 seconds of inputs/input.mp4.
To process the full video, remove or change the max_frames limit in run.py